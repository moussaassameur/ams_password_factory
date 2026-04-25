from collections import Counter

def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def extract_ngrams(words, n):
    ngrams = set()
    for w in words:
        for i in range(len(w)-n+1):
            ngrams.add(w[i:i+n])
    return ngrams

def compute_ngram_coverage(train, generated, n=3):

    train_ngrams = extract_ngrams(train, n)
    gen_ngrams = extract_ngrams(generated, n)

    overlap = train_ngrams & gen_ngrams

    coverage = len(overlap) / len(train_ngrams)

    print(f"{n}-gram coverage: {coverage*100:.2f}%")
    print(f"train ngrams: {len(train_ngrams)}")
    print(f"generated ngrams: {len(gen_ngrams)}")
    print(f"overlap: {len(overlap)}")


train = load_words("../data/train.txt")

print("\n=== LSTM ===")
lstm = load_words("../outputs/generated/lstm_10k.txt")
compute_ngram_coverage(train, lstm, 3)

print("\n=== Transformer Decoder ===")
trans = load_words("../outputs/generated/transformer_10k.txt")
compute_ngram_coverage(train, trans, 3)
