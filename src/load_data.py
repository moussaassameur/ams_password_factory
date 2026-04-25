def load_passwords(path):
    passwords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pw = line.strip()
            if pw:
                passwords.append(pw)
    return passwords


train_path = "../data/train.txt"
eval_path = "../data/eval.txt"

train_pw = load_passwords(train_path)
eval_pw = load_passwords(eval_path)

print("Train size:", len(train_pw))
print("Eval size :", len(eval_pw))

print("Example passwords:")
for i in range(10):
    print(train_pw[i])
