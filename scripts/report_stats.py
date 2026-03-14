#!/usr/bin/env python3
from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.constants import DATASET_PATH, HOLDOUT_PATH, IAA_TEMPLATE_PATH, PAGES_PATH, REPORT_GENERATED_DIR
from project.io_utils import read_jsonl, write_json


def main() -> int:
    pages = read_jsonl(PAGES_PATH) if PAGES_PATH.exists() else []
    dev = read_jsonl(DATASET_PATH) if DATASET_PATH.exists() else []
    holdout = read_jsonl(HOLDOUT_PATH) if HOLDOUT_PATH.exists() else []
    iaa = read_jsonl(IAA_TEMPLATE_PATH) if IAA_TEMPLATE_PATH.exists() else []

    REPORT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    corpus_stats = {
        "page_count": len(pages),
        "avg_page_text_chars": statistics.mean([len(page["text"]) for page in pages]) if pages else 0.0,
        "source_type_counts": Counter(page["source_type"] for page in pages),
    }
    dataset_stats = {
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "iaa_template_count": len(iaa),
        "category_counts": Counter(row["category"] for row in dev + holdout),
        "answer_type_counts": Counter(row["answer_type"] for row in dev + holdout),
        "sample_questions": [row["question"] for row in dev[:5]],
    }
    write_json(REPORT_GENERATED_DIR / "corpus_stats.json", corpus_stats)
    write_json(REPORT_GENERATED_DIR / "dataset_stats.json", dataset_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
