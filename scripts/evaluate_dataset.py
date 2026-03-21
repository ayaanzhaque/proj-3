#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from rag import RAGModel
from rag.io_utils import read_jsonl
from rag.text_utils import best_f1, exact_match


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local dataset with the RAG model.")
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset_path)
    questions = [row["question"] for row in rows]
    answers = [row["answers"].split("|") for row in rows]
    urls = [row["evidence_url"] for row in rows]
    model = RAGModel()
    predictions = model.predict(questions)

    em_scores = [1.0 if exact_match(pred, gold) else 0.0 for pred, gold in zip(predictions, answers)]
    f1_scores = [best_f1(pred, gold) for pred, gold in zip(predictions, answers)]

    print(f"examples={len(rows)}")
    print(f"exact_match={statistics.mean(em_scores):.4f}")
    print(f"token_f1={statistics.mean(f1_scores):.4f}")
    for idx, row in enumerate(rows[:10]):
        print("---")
        print("Q:", row["question"])
        print("Pred:", predictions[idx])
        print("Gold:", row["answers"])
        print("URL:", urls[idx])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
