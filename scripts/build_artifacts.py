#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.constants import (
    ARTIFACT_DIR,
    CORPUS_PATH,
    DENSE_ENCODER_DIR,
    DENSE_ENCODER_NAME,
    PAGE_EMBEDDINGS_PATH,
    RUNTIME_CONFIG_PATH,
)
from project.io_utils import read_jsonl, write_json
from project.modeling import DenseEncoder
from project.text_utils import squash_ws, url_tokens


def materialize_model(model_name: str, model_dir) -> None:
    if (model_dir / "config.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(model_dir)


def build_embeddings() -> np.ndarray:
    raw_docs = read_jsonl(CORPUS_PATH)
    docs = [doc for doc in raw_docs if (doc.get("text") or "").strip()]
    encoder = DenseEncoder(DENSE_ENCODER_DIR)
    texts = [
        squash_ws(url_tokens(doc["url"]) + " " + doc["text"])
        for doc in docs
    ]
    embeddings = encoder.encode(texts, batch_size=64, max_length=256)
    np.save(PAGE_EMBEDDINGS_PATH, embeddings.astype(np.float32))
    print(f"Built embeddings for {len(docs)} documents -> {PAGE_EMBEDDINGS_PATH}")
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local model and retrieval artifacts.")
    parser.add_argument("--null-margin", type=float, default=2.0)
    parser.add_argument("--min-page-score", type=float, default=0.04)
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(f"Corpus not found at {CORPUS_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    materialize_model(DENSE_ENCODER_NAME, DENSE_ENCODER_DIR)
    build_embeddings()
    write_json(
        RUNTIME_CONFIG_PATH,
        {
            "null_margin": args.null_margin,
            "min_page_score": args.min_page_score,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
