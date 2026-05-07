"""
Génération en lot (batched) de mots de passe avec le Transformer Decoder-Only.
Produit les trois fichiers requis pour la Phase 3 :
    outputs/generated/10k.txt
    outputs/generated/100k.txt
    outputs/generated/1M.txt
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "../outputs/models/transformer_decoder_best.pt"
OUTPUT_DIR = "../outputs/generated"
MAX_GEN_LEN = 32
TEMPERATURE = 0.9
TOP_K       = 0        # 0 = désactivé (nucleus pur, meilleure couverture)
TOP_P       = 0.92
BATCH_SIZE  = 512      # GPU : augmenter si VRAM disponible
MIN_LEN     = 4        # ignorer les mots de passe trop courts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Architecture (identique au script d'entraînement) ─────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, pad_idx, max_len):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        try:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True
            )
        except TypeError:
            # PyTorch < 1.11 n'a pas norm_first
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x):
        mask = self._causal_mask(x.size(1), x.device)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        return self.fc(x)


# ── Filtrage Top-K / Nucleus (Top-P) ─────────────────────────────────────────

def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Applique le filtrage top-k puis top-p sur un tenseur [B, V]."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        threshold = topk_vals[:, -1].unsqueeze(1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs    = torch.cumsum(sorted_probs, dim=-1)
        # Conserver le premier token qui fait dépasser le seuil
        remove_mask = (cum_probs - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

    return logits


# ── Génération en lot ──────────────────────────────────────────────────────────

def generate_batch(model, char2idx, idx2char, batch_size):
    sos_idx = char2idx["<SOS>"]
    eos_idx = char2idx["<EOS>"]
    pad_idx = char2idx["<PAD>"]

    sequences = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=DEVICE)
    finished  = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    with torch.inference_mode():
        for _ in range(MAX_GEN_LEN):
            logits = model(sequences)[:, -1, :]          # [B, V]
            logits = logits / TEMPERATURE
            logits = top_k_top_p_filter(logits, TOP_K, TOP_P)
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)          # [B, 1]
            next_t[finished.unsqueeze(1)] = pad_idx
            sequences = torch.cat([sequences, next_t], dim=1)
            finished  = finished | (next_t.squeeze(1) == eos_idx)
            if finished.all():
                break

    # Décodage
    eos_idx_val = eos_idx
    pad_idx_val = pad_idx
    specials    = {"<SOS>", "<EOS>", "<PAD>", "<UNK>"}
    passwords   = []
    for i in range(batch_size):
        chars = []
        for j in range(1, sequences.size(1)):
            idx = sequences[i, j].item()
            if idx == eos_idx_val or idx == pad_idx_val:
                break
            ch = idx2char.get(idx, "")
            if ch and ch not in specials:
                chars.append(ch)
        pw = "".join(chars)
        if len(pw) >= MIN_LEN:
            passwords.append(pw)
    return passwords


def generate_n(model, char2idx, idx2char, n, deduplicate=True):
    """Génère n mots de passe (uniques si deduplicate=True)."""
    collected = []
    seen      = set() if deduplicate else None
    max_tries = n * 5
    attempts  = 0

    while len(collected) < n and attempts < max_tries:
        batch = generate_batch(model, char2idx, idx2char, BATCH_SIZE)
        attempts += BATCH_SIZE
        for pw in batch:
            if len(collected) >= n:
                break
            if deduplicate:
                if pw not in seen:
                    seen.add(pw)
                    collected.append(pw)
            else:
                collected.append(pw)

    # Complément si on n'atteint pas n (rare, modèle peu diversifié)
    while len(collected) < n:
        collected.append(collected[len(collected) % max(1, len(collected) - 1)])

    return collected[:n]


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main():
    print("Chargement du checkpoint :", MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    vocab    = ckpt["vocab"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["config"]
    pad_idx  = char2idx["<PAD>"]

    model = TransformerDecoderOnly(
        vocab_size=len(vocab),
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"], pad_idx=pad_idx, max_len=cfg["max_len"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Modèle chargé sur :", DEVICE)
    print(f"Config : d_model={cfg['d_model']}, layers={cfg['num_layers']}, "
          f"nhead={cfg['nhead']}, ff={cfg['dim_feedforward']}")
    print(f"Sampling : T={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, "
          f"batch={BATCH_SIZE}\n")

    targets = [
        ("transformer_10k.txt",  10_000,    True),
        ("transformer_100k.txt", 100_000,   True),
        ("transformer_1M.txt",   1_000_000, True),
    ]

    for filename, n, dedup in targets:
        out_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Génération de {n:,} mots de passe → {out_path}")
        passwords = generate_n(model, char2idx, idx2char, n, deduplicate=dedup)
        with open(out_path, "w", encoding="utf-8") as f:
            for pw in passwords:
                f.write(pw + "\n")
        unique_count = len(set(passwords))
        print(f"  Terminé : {len(passwords):,} écrits, "
              f"{unique_count:,} uniques ({100*unique_count/len(passwords):.1f}%)\n")

    print("Tous les fichiers générés.")


if __name__ == "__main__":
    main()
