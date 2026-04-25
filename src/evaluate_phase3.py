"""
Phase 3 – Évaluation du système de génération de mots de passe
===============================================================
Protocole 1 : Couverture & Diversité
    Mesure dans quelle mesure les mots de passe générés couvrent le spectre
    des vrais mots de passe (variabilité + intelligibilité séquentielle).

Protocole 2 : Simulation d'attaque par devinette (Guessing Attack)
    Simule une attaque réelle : les mots de passe générés sont classés par
    fréquence (les plus courants d'abord) et on mesure combien de mots de
    passe réels sont devinés en fonction du nombre de tentatives.
"""

import os
import sys
import math
import collections

TRAIN_PATH = "../data/train.txt"
EVAL_PATH  = "../data/eval.txt"
OUTPUT_DIR = "../outputs/generated"
REPORT_DIR = "../outputs/evaluation"

os.makedirs(REPORT_DIR, exist_ok=True)

GEN_FILES = {
    "10k":  os.path.join(OUTPUT_DIR, "10k.txt"),
    "100k": os.path.join(OUTPUT_DIR, "100k.txt"),
    "1M":   os.path.join(OUTPUT_DIR, "1M.txt"),
}


# ── Utilitaires ───────────────────────────────────────────────────────────────

def load_list(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]

def load_set(path):
    return set(load_list(path))

def get_ngrams(passwords, n):
    """Retourne l'ensemble des n-grammes de caractères du corpus."""
    ngrams = set()
    for pw in passwords:
        for i in range(len(pw) - n + 1):
            ngrams.add(pw[i:i+n])
    return ngrams

def char_freq(passwords):
    """Fréquence relative de chaque caractère."""
    counter = collections.Counter()
    total   = 0
    for pw in passwords:
        for c in pw:
            counter[c] += 1
            total += 1
    if total == 0:
        return {}
    return {c: v / total for c, v in counter.items()}

def kl_divergence(p_freq, q_freq):
    """KL(P || Q) – divergence de Kullback-Leibler (nats)."""
    chars = set(p_freq) | set(q_freq)
    eps   = 1e-9
    kl    = 0.0
    for c in chars:
        p = p_freq.get(c, eps)
        q = q_freq.get(c, eps)
        if p > 0:
            kl += p * math.log(p / q)
    return kl

def length_distribution(passwords):
    counter = collections.Counter(len(pw) for pw in passwords)
    return dict(sorted(counter.items()))

def entropy_bits(passwords):
    """Entropie empirique en bits (sur les mots de passe distincts)."""
    counter = collections.Counter(passwords)
    total   = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


# ── Protocole 1 : Couverture & Diversité ─────────────────────────────────────

def protocol1(label, gen_list, train_set, eval_set, log_lines):
    gen_set = set(gen_list)
    total   = len(gen_list)
    unique  = len(gen_set)
    if total == 0:
        log_lines.append(f"[{label}] Fichier vide ou introuvable.\n")
        return

    hits_eval = gen_set & eval_set
    coverage  = len(hits_eval) / len(eval_set) * 100 if eval_set else 0.0
    uniqueness= unique / total * 100
    novel     = gen_set - train_set - eval_set
    novelty   = len(novel) / unique * 100 if unique else 0.0
    avg_len   = sum(len(pw) for pw in gen_list) / total

    train_2g  = get_ngrams(train_set, 2)
    gen_2g    = get_ngrams(gen_set,   2)
    train_3g  = get_ngrams(train_set, 3)
    gen_3g    = get_ngrams(gen_set,   3)
    cov_2g    = len(gen_2g & train_2g) / len(train_2g) * 100 if train_2g else 0.0
    cov_3g    = len(gen_3g & train_3g) / len(train_3g) * 100 if train_3g else 0.0

    p_freq    = char_freq(list(train_set))
    q_freq    = char_freq(gen_list)
    kl_div    = kl_divergence(p_freq, q_freq)
    ent_bits  = entropy_bits(gen_list)

    lines = [
        f"\n{'='*60}",
        f"Protocole 1 – Couverture & Diversité  [{label}]",
        f"{'='*60}",
        f"  Passwords générés  : {total:>12,}",
        f"  Uniques            : {unique:>12,}  ({uniqueness:.1f}%)",
        f"  Nouveaux (≠ train) : {len(novel):>12,}  ({novelty:.1f}%)",
        f"  Longueur moyenne   : {avg_len:>12.2f}",
        f"  Entropie (bits)    : {ent_bits:>12.2f}",
        f"",
        f"  ── Couverture sur l'ensemble d'évaluation ──",
        f"  Mots trouvés       : {len(hits_eval):>12,}  / {len(eval_set):,}",
        f"  Coverage (%)       : {coverage:>12.2f}",
        f"",
        f"  ── Couverture n-grammes de caractères ──",
        f"  Couv. bigrammes    : {cov_2g:>12.2f}%  ({len(gen_2g & train_2g):,}/{len(train_2g):,})",
        f"  Couv. trigrammes   : {cov_3g:>12.2f}%  ({len(gen_3g & train_3g):,}/{len(train_3g):,})",
        f"",
        f"  ── Distribution des longueurs (top 10) ──",
    ]

    len_dist = length_distribution(gen_list)
    for length, cnt in sorted(len_dist.items(), key=lambda x: -x[1])[:10]:
        pct = cnt / total * 100
        lines.append(f"    len={length:>2} : {cnt:>8,}  ({pct:.1f}%)")

    lines += [
        f"",
        f"  ── Distribution des caractères (KL vs train) ──",
        f"  KL divergence (nats) : {kl_div:.4f}  "
        f"(0 = identique à l'entraînement)",
        f"  Top 10 caractères générés :",
    ]
    for ch, freq in sorted(q_freq.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"    '{ch}' : {freq*100:.2f}%")

    for l in lines:
        print(l)
    log_lines.extend(l + "\n" for l in lines)


# ── Protocole 2 : Simulation d'attaque par devinette ─────────────────────────

def protocol2(label, gen_list, eval_set, log_lines):
    if not gen_list:
        log_lines.append(f"[{label}] Fichier vide ou introuvable.\n")
        return

    # Classement par fréquence décroissante (les mots courants = essayés en 1er)
    freq_counter = collections.Counter(gen_list)
    ranked       = [pw for pw, _ in freq_counter.most_common()]

    # Couverture cumulative
    cumulative_hits = []
    found           = set()
    total_hits      = 0
    for pw in ranked:
        if pw in eval_set:
            total_hits += 1
            found.add(pw)
        cumulative_hits.append(total_hits)

    # Seuils d'analyse
    eval_size = len(eval_set)
    thresholds = {}
    for target_pct in [1, 5, 10, 25, 50, 75, 100]:
        target_hits = math.ceil(eval_size * target_pct / 100)
        g_needed    = next((i + 1 for i, h in enumerate(cumulative_hits)
                            if h >= target_hits), None)
        thresholds[target_pct] = g_needed

    # Points de rapport (1 K, 10 K, 100 K, 1 M)
    checkpoints = [1_000, 10_000, 100_000, 1_000_000]

    lines = [
        f"\n{'='*60}",
        f"Protocole 2 – Simulation d'attaque par devinette  [{label}]",
        f"{'='*60}",
        f"  Passwords uniques classés : {len(ranked):,}",
        f"  Eval passwords cibles     : {eval_size:,}",
        f"",
        f"  ── Couverture cumulative par nombre de tentatives ──",
    ]
    for cp in checkpoints:
        if cp <= len(cumulative_hits):
            hits = cumulative_hits[cp - 1]
            pct  = hits / eval_size * 100 if eval_size else 0.0
            lines.append(f"    {cp:>10,} tentatives : {hits:>6,} hits  ({pct:.2f}%)")
        else:
            hits = cumulative_hits[-1] if cumulative_hits else 0
            pct  = hits / eval_size * 100 if eval_size else 0.0
            lines.append(f"    {cp:>10,} tentatives : {hits:>6,} hits  ({pct:.2f}%)  [limite atteinte]")

    lines += [
        f"",
        f"  ── Tentatives nécessaires pour couvrir X% de l'eval ──",
    ]
    for pct, needed in thresholds.items():
        n_str = f"{needed:,}" if needed else "impossible"
        lines.append(f"    {pct:>3}% : {n_str} tentatives")

    lines += [
        f"",
        f"  ── Position des hits dans la liste classée ──",
    ]
    hit_positions = [i + 1 for i, pw in enumerate(ranked) if pw in eval_set]
    if hit_positions:
        lines += [
            f"    Premier hit    : tentative {hit_positions[0]:,}",
            f"    Dernier hit    : tentative {hit_positions[-1]:,}",
            f"    Médiane hits   : tentative {sorted(hit_positions)[len(hit_positions)//2]:,}",
            f"    Total trouvés  : {len(hit_positions):,} / {eval_size:,}  "
            f"({100*len(hit_positions)/eval_size:.2f}%)",
        ]
    else:
        lines.append("    Aucun hit trouvé.")

    # Enregistrer les données brutes pour un graphique éventuel
    curve_path = os.path.join(REPORT_DIR, f"attack_curve_{label}.tsv")
    sample_pts = list(range(0, len(cumulative_hits), max(1, len(cumulative_hits)//500)))
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("guesses\thits\tcoverage_pct\n")
        for idx in sample_pts:
            h   = cumulative_hits[idx]
            pct = h / eval_size * 100 if eval_size else 0.0
            f.write(f"{idx+1}\t{h}\t{pct:.4f}\n")
    lines.append(f"\n  Courbe d'attaque sauvegardée → {curve_path}")

    for l in lines:
        print(l)
    log_lines.extend(l + "\n" for l in lines)


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main():
    print("Chargement des données de référence...")
    train_set = load_set(TRAIN_PATH)
    eval_set  = load_set(EVAL_PATH)
    print(f"  Train : {len(train_set):,}  |  Eval : {len(eval_set):,}")

    log_lines = [
        "Phase 3 – Rapport d'évaluation du Transformer Decoder-Only\n",
        f"Train size : {len(train_set):,}\n",
        f"Eval  size : {len(eval_set):,}\n",
    ]

    for label, path in GEN_FILES.items():
        if not os.path.exists(path):
            print(f"\n[ATTENTION] Fichier introuvable : {path}")
            print("  → Lancez d'abord generate_transformer_decoder.py")
            log_lines.append(f"\n[{label}] Fichier introuvable : {path}\n")
            continue

        print(f"\nChargement de {path} ...")
        gen_list = load_list(path)
        print(f"  {len(gen_list):,} mots de passe chargés.")

        protocol1(label, gen_list, train_set, eval_set, log_lines)
        protocol2(label, gen_list, eval_set, log_lines)

    report_path = os.path.join(REPORT_DIR, "phase3_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"\nRapport complet sauvegardé → {report_path}")


if __name__ == "__main__":
    main()
