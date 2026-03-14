#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.constants import (
    ARTIFACT_DIR,
    CHUNKS_PATH,
    DENSE_ENCODER_DIR,
    DENSE_ENCODER_NAME,
    PAGE_EMBEDDINGS_PATH,
    PAGES_PATH,
    QA_MODEL_DIR,
    QA_MODEL_NAME,
    RUNTIME_CONFIG_PATH,
)
from project.io_utils import read_jsonl, write_json
from project.modeling import DenseEncoder
from project.text_utils import squash_ws, url_tokens


def materialize_model(model_name: str, model_dir, *, qa: bool = False) -> None:
    if (model_dir / "config.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    if qa:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(model_dir)


def build_embeddings() -> np.ndarray:
    pages = read_jsonl(PAGES_PATH)
    encoder = DenseEncoder(DENSE_ENCODER_DIR)
    texts = [
        squash_ws(
            " ".join(
                [
                    page["title"],
                    " ".join(page.get("headings", [])),
                    url_tokens(page["url"]),
                    page["text"],
                    " ".join(page.get("table_rows", [])),
                ]
            )
        )
        for page in pages
    ]
    embeddings = encoder.encode(texts, batch_size=64, max_length=256)
    np.save(PAGE_EMBEDDINGS_PATH, embeddings.astype(np.float32))
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local model and retrieval artifacts.")
    parser.add_argument("--null-margin", type=float, default=2.0)
    parser.add_argument("--min-page-score", type=float, default=0.04)
    parser.add_argument("--min-chunk-score", type=float, default=0.15)
    args = parser.parse_args()

    if not PAGES_PATH.exists() or not CHUNKS_PATH.exists():
        raise SystemExit("Corpus not found. Run scripts/build_corpus.py first.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    materialize_model(DENSE_ENCODER_NAME, DENSE_ENCODER_DIR, qa=False)
    materialize_model(QA_MODEL_NAME, QA_MODEL_DIR, qa=True)
    build_embeddings()
    write_json(
        RUNTIME_CONFIG_PATH,
        {
            "null_margin": args.null_margin,
            "min_page_score": args.min_page_score,
            "min_chunk_score": args.min_chunk_score,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
