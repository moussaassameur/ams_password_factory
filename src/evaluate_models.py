import os

TRAIN_FILE = "../data/train.txt"
EVAL_FILE = "../data/eval.txt"

MODEL_FILES = {
    "LSTM": "../outputs/generated/lstm_10k.txt",
    "Transformer": "../outputs/generated/transformer_10k.txt",
    # "GRU": "../outputs/generated/gru_10k.txt",
}

def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_set(path):
    return set(load_list(path))

train_set = load_set(TRAIN_FILE)
eval_set = load_set(EVAL_FILE)

print(f"Taille train : {len(train_set)}")
print(f"Taille eval  : {len(eval_set)}")
print()

header = (
    f"{'Model':<15}"
    f"{'Generated':>12}"
    f"{'Unique':>12}"
    f"{'MatchesEval':>15}"
    f"{'Coverage%':>12}"
    f"{'Precision%':>12}"
    f"{'Novelty%':>12}"
    f"{'AvgLen':>10}"
)
print(header)
print("-" * len(header))

for model_name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"{model_name:<15} fichier introuvable : {path}")
        continue

    generated_list = load_list(path)
    generated_set = set(generated_list)

    total_generated = len(generated_list)
    unique_generated = len(generated_set)

    matches_eval = generated_set.intersection(clean_eval_set)
    coverage = (len(matches_eval) / len(clean_eval_set) * 100) if clean_eval_set else 0.0
    precision = (len(matches_eval) / total_generated * 100) if total_generated else 0.0
    uniqueness = (unique_generated / total_generated * 100) if total_generated else 0.0

    novel = [pw for pw in generated_set if pw not in train_set and pw not in clean_eval_set]
    novelty = (len(novel) / unique_generated * 100) if unique_generated else 0.0

    avg_len = (sum(len(pw) for pw in generated_list) / total_generated) if total_generated else 0.0

    print(
        f"{model_name:<15}"
        f"{total_generated:>12}"
        f"{unique_generated:>12}"
        f"{len(matches_eval):>15}"
        f"{coverage:>12.2f}"
        f"{precision:>12.2f}"
        f"{novelty:>12.2f}"
        f"{avg_len:>10.2f}"
    )
