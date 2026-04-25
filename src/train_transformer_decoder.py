import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import load_passwords, build_vocab, PasswordDataset, collate_fn

TRAIN_PATH = "../data/train.txt"
EVAL_PATH  = "../data/eval.txt"
MODEL_DIR  = "../outputs/models"
LOG_DIR    = "../logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── Hyperparamètres optimisés pour la couverture / intelligibilité ───────────
BATCH_SIZE      = 256
D_MODEL         = 384
NHEAD           = 6
NUM_LAYERS      = 6
DIM_FEEDFORWARD = 1024
DROPOUT         = 0.1
EPOCHS          = 40
LR              = 5e-4
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.0     # 0 = matche mieux la distribution réelle (meilleure couverture)
WARMUP_STEPS    = 1000
PATIENCE        = 5       # early stopping : stop si pas d'amélioration eval
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement des données...")
train_pw = load_passwords(TRAIN_PATH)
eval_pw  = load_passwords(EVAL_PATH)
print("Train size:", len(train_pw))
print("Eval size :", len(eval_pw))

vocab, char2idx, idx2char = build_vocab(train_pw)
pad_idx = char2idx["<PAD>"]
MAX_LEN = max(max(len(p) for p in train_pw), max(len(p) for p in eval_pw)) + 5
print("Vocab size:", len(vocab))
print("MAX_LEN   :", MAX_LEN)
print("Device    :", DEVICE)

train_dataset = PasswordDataset(train_pw, char2idx)
eval_dataset  = PasswordDataset(eval_pw,  char2idx)

_nw = 4 if DEVICE.type == "cuda" else 0
_pm = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_idx),
    num_workers=_nw, pin_memory=_pm
)
eval_loader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_idx),
    num_workers=_nw, pin_memory=_pm
)


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

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,         # pre-norm : entraînement plus stable
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.fc.weight,        std=0.02)
        nn.init.zeros_(self.fc.bias)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x):
        mask = self._causal_mask(x.size(1), x.device)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        return self.fc(x)


model = TransformerDecoderOnly(
    vocab_size=len(vocab), d_model=D_MODEL, nhead=NHEAD,
    num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT, pad_idx=pad_idx, max_len=MAX_LEN
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {n_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                              betas=(0.9, 0.95))

TOTAL_STEPS = EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# AMP API moderne (compatible avec versions récentes de PyTorch)
try:
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    _autocast = lambda: torch.amp.autocast(device_type="cuda")
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
    _autocast = lambda: torch.cuda.amp.autocast()


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad()
            if scaler is not None:
                with _autocast():
                    out  = model(x_batch)
                    loss = criterion(out.reshape(-1, out.size(-1)), y_batch.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = model(x_batch)
                loss = criterion(out.reshape(-1, out.size(-1)), y_batch.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                out  = model(x_batch)
                loss = criterion(out.reshape(-1, out.size(-1)), y_batch.reshape(-1))

        total_loss += loss.item()

    return total_loss / len(loader)


best_eval_loss = float("inf")
epochs_no_improve = 0
history = []

print("Début de l'entraînement Transformer Decoder-Only (version optimisée)...")

def make_checkpoint():
    return {
        "model_state_dict": model.state_dict(),
        "vocab": vocab, "char2idx": char2idx, "idx2char": idx2char,
        "config": {
            "d_model": D_MODEL, "nhead": NHEAD, "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD, "dropout": DROPOUT, "max_len": MAX_LEN
        },
        "best_eval_loss": best_eval_loss
    }

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = run_epoch(train_loader, train=True)
    eval_loss  = run_epoch(eval_loader,  train=False)
    elapsed    = time.time() - t0
    cur_lr     = optimizer.param_groups[0]["lr"]
    ppl        = math.exp(min(eval_loss, 20))

    history.append((epoch, train_loss, eval_loss, elapsed, cur_lr))
    print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.4f} | "
          f"eval={eval_loss:.4f} | ppl={ppl:.2f} | lr={cur_lr:.6f} | {elapsed:.1f}s")

    if eval_loss < best_eval_loss - 1e-4:
        best_eval_loss = eval_loss
        epochs_no_improve = 0
        torch.save(make_checkpoint(),
                   os.path.join(MODEL_DIR, "transformer_decoder_best.pt"))
        print(f"  → Meilleur modèle sauvegardé (eval={best_eval_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  (pas d'amélioration : {epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping déclenché.")
            break

torch.save(make_checkpoint(), os.path.join(MODEL_DIR, "transformer_decoder_last.pt"))
print("Dernier checkpoint sauvegardé.")

log_path = os.path.join(LOG_DIR, "transformer_decoder_training_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("epoch\ttrain_loss\teval_loss\ttime_sec\tlr\n")
    for ep, tr, ev, t, lr in history:
        f.write(f"{ep}\t{tr:.6f}\t{ev:.6f}\t{t:.2f}\t{lr:.8f}\n")

print("Log sauvegardé :", log_path)
print("Entraînement terminé.")
