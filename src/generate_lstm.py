import os
import torch
import torch.nn as nn
from data_utils import decode_indices

MODEL_PATH = "../outputs/models/lstm_best.pt"
OUTPUT_DIR = "../outputs/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

vocab = checkpoint["vocab"]
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]
config = checkpoint["config"]

pad_idx = char2idx["<PAD>"]
sos_idx = char2idx["<SOS>"]
eos_idx = char2idx["<EOS>"]

class LSTMPasswordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

model = LSTMPasswordModel(
    vocab_size=len(vocab),
    embed_dim=config["embed_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    pad_idx=pad_idx,
    dropout=config.get("dropout", 0.0)
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def sample_next_token(logits, temperature=0.8, top_k=20):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_k is not None and top_k < probs.size(0):
        top_probs, top_idx = torch.topk(probs, top_k)
        top_probs = top_probs / top_probs.sum()
        next_token = top_idx[torch.multinomial(top_probs, 1)].item()
    else:
        next_token = torch.multinomial(probs, 1).item()

    return next_token

def generate_password(max_len=32, temperature=0.8, top_k=20):
    input_ids = [sos_idx]
    hidden = None

    for _ in range(max_len):
        x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            outputs, hidden = model(x, hidden=None)

        logits = outputs[0, -1]
        next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k)

        if next_idx == eos_idx:
            break

        input_ids.append(next_idx)

    pw = decode_indices(input_ids, idx2char)
    return pw

def generate_file(filename, n, max_len=32, temperature=0.8, top_k=20):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            pw = generate_password(max_len=max_len, temperature=temperature, top_k=top_k)
            f.write(pw + "\n")
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{n} générés")
    print("Fichier créé :", path)

print("Exemples générés :")
for i in range(10):
    print(f"{i+1:02d}:", generate_password())

generate_file("lstm_100k.txt", 100000)
