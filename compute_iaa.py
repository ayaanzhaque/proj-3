"""
Compute inter-annotator agreement on iaa_labels.csv for rows with answer2.
Metrics: token-level F1, Cohen's Kappa, Krippendorff's Alpha.
"""

import csv
import re
import string
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between two answers (SQuAD-style)."""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(a: str, b: str) -> int:
    """1 if normalized answers match exactly, 0 otherwise."""
    return int(normalize(a) == normalize(b))


def main():
    rows = []
    with open("iaa_labels.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a2 = row["answer2"].strip()
            if a2:  # keep only rows with a second annotation
                rows.append(row)

    print(f"Rows with two annotations: {len(rows)}\n")

    # --- Token-level F1 ---
    f1_scores = [token_f1(r["answer"], r["answer2"]) for r in rows]
    avg_f1 = np.mean(f1_scores)

    # --- Exact match ---
    em_flags = [exact_match(r["answer"], r["answer2"]) for r in rows]
    em_rate = np.mean(em_flags)

    # --- Cohen's Kappa (on exact match per question) ---
    # Encode each annotator's normalized answer; kappa measures how often they
    # land on the same label beyond chance.
    labels_a1 = [normalize(r["answer"]) for r in rows]
    labels_a2 = [normalize(r["answer2"]) for r in rows]
    # Build a shared label set so sklearn can handle it
    all_labels = sorted(set(labels_a1 + labels_a2))
    kappa = cohen_kappa_score(labels_a1, labels_a2, labels=all_labels)

    # --- Krippendorff's Alpha (nominal) ---
    # Represent each unique normalized answer as an integer code
    label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}
    coder1 = [label_to_id[l] for l in labels_a1]
    coder2 = [label_to_id[l] for l in labels_a2]
    reliability_data = np.array([coder1, coder2])
    alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement="nominal"
    )

    # --- Report ---
    print(f"{'Metric':<30} {'Value':>8}")
    print("-" * 40)
    print(f"{'Avg Token-level F1':<30} {avg_f1:>8.4f}")
    print(f"{'Exact Match Rate':<30} {em_rate:>8.4f}")
    print(f"{'Cohen\'s Kappa (nominal)':<30} {kappa:>8.4f}")
    print(f"{'Krippendorff\'s Alpha (nominal)':<30} {alpha:>8.4f}")

    # Per-question breakdown for F1
    print(f"\n{'='*70}")
    print("Per-question Token F1:")
    print(f"{'='*70}")
    for r, f1 in zip(rows, f1_scores):
        q = r["question"][:55].ljust(55)
        print(f"  {q}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
