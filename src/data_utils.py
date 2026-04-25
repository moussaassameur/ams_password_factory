import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

def load_passwords(path):
    passwords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pw = line.strip()
            if pw:
                passwords.append(pw)
    return passwords

def build_vocab(passwords):
    chars = sorted(set("".join(passwords)))
    vocab = SPECIAL_TOKENS + chars
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return vocab, char2idx, idx2char

def encode_password(password, char2idx):
    unk_idx = char2idx["<UNK>"]
    return [char2idx["<SOS>"]] + [char2idx.get(c, unk_idx) for c in password] + [char2idx["<EOS>"]]

def decode_indices(indices, idx2char):
    chars = []
    for idx in indices:
        token = idx2char[int(idx)]
        if token == "<EOS>":
            break
        if token not in ["<SOS>", "<PAD>", "<UNK>"]:
            chars.append(token)
    return "".join(chars)

class PasswordDataset(Dataset):
    def __init__(self, passwords, char2idx):
        self.sequences = [torch.tensor(encode_password(p, char2idx), dtype=torch.long) for p in passwords]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

def collate_fn(batch, pad_idx):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_idx)
    return xs_padded, ys_padded
