import os
import time
import torch
from torch.utils.data import DataLoader
from data_utils import load_passwords, build_vocab, PasswordDataset, collate_fn

TRAIN_PATH = "../data/train.txt"
EVAL_PATH = "../data/eval.txt"
MODEL_DIR = "../outputs/models"
LOG_DIR = "../logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 128
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 20
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Chargement des données...")
train_pw = load_passwords(TRAIN_PATH)
eval_pw = load_passwords(EVAL_PATH)

print("Train size:", len(train_pw))
print("Eval size :", len(eval_pw))

vocab, char2idx, idx2char = build_vocab(train_pw)
pad_idx = char2idx["<PAD>"]

print("Vocab size:", len(vocab))
print("Device:", DEVICE)

train_dataset = PasswordDataset(train_pw, char2idx)
eval_dataset = PasswordDataset(eval_pw, char2idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_idx)
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, pad_idx)
)

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

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = LSTMPasswordModel(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    pad_idx=pad_idx,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            y_batch.reshape(-1)
        )

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

best_eval_loss = float("inf")
history = []

print("Début de l'entraînement...")

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer)
    eval_loss = run_epoch(model, eval_loader, criterion)

    elapsed = time.time() - start_time
    history.append((epoch, train_loss, eval_loss, elapsed))

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | time={elapsed:.1f}s")

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "char2idx": char2idx,
            "idx2char": idx2char,
            "config": {
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT
            },
            "best_eval_loss": best_eval_loss
        }
        torch.save(best_checkpoint, os.path.join(MODEL_DIR, "lstm_best.pt"))
        print("Nouveau meilleur modèle sauvegardé : lstm_best.pt")

last_checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "char2idx": char2idx,
    "idx2char": idx2char,
    "config": {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT
    },
    "best_eval_loss": best_eval_loss
}

torch.save(last_checkpoint, os.path.join(MODEL_DIR, "lstm_last.pt"))
print("Dernier modèle sauvegardé : lstm_last.pt")

log_path = os.path.join(LOG_DIR, "lstm_training_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("epoch\ttrain_loss\teval_loss\ttime_sec\n")
    for epoch, tr, ev, t in history:
        f.write(f"{epoch}\t{tr:.6f}\t{ev:.6f}\t{t:.2f}\n")

print("Log sauvegardé :", log_path)
print("Entraînement terminé")
