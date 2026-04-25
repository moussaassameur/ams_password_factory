from data_utils import load_passwords, build_vocab, encode_password, decode_indices

train_path = "../data/train.txt"

train_pw = load_passwords(train_path)
print("Train size:", len(train_pw))

vocab, char2idx, idx2char = build_vocab(train_pw)
print("Vocab size:", len(vocab))
print("First 20 vocab tokens:", vocab[:20])

example = train_pw[0]
encoded = encode_password(example, char2idx)
decoded = decode_indices(encoded, idx2char)

print("Example password :", example)
print("Encoded          :", encoded)
print("Decoded          :", decoded)
