#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rag import RAGModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark submission latency.")
    parser.add_argument("questions_path", type=str)
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to a corpus JSONL file (default: rag/corpus/official/corpus.jsonl).",
    )
    args = parser.parse_args()

    with open(args.questions_path, "r", encoding="utf-8") as handle:
        questions = [line.strip() for line in handle if line.strip()]
    model = RAGModel(corpus_path=args.corpus)
    start = time.perf_counter()
    answers = model.predict(questions)
    elapsed = time.perf_counter() - start
    print(f"questions={len(questions)}")
    print(f"elapsed_sec={elapsed:.4f}")
    print(f"sec_per_question={elapsed / max(len(questions), 1):.4f}")
    print(f"answers={len(answers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
