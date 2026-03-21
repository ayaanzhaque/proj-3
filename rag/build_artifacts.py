#!/usr/bin/env python3
"""Build page embeddings for a given corpus.

The embeddings file is saved alongside the corpus JSONL with a ``.npy``
extension (e.g. ``corpus.jsonl`` → ``corpus.npy``).

The dense-encoder model is shared across all corpora and lives at
``<project_root>/artifacts/models/dense_encoder``.

Usage:
    python -m rag.build_artifacts                          # default corpus
    python -m rag.build_artifacts --corpus rag/corpus/ours/corpus.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.constants import CORPUS_PATH, DENSE_ENCODER_DIR, DENSE_ENCODER_NAME
from rag.io_utils import read_jsonl
from rag.encoder import DenseEncoder
from rag.text_utils import squash_ws, url_tokens


def materialize_model(model_name: str, model_dir: Path) -> None:
    """Download model from HuggingFace Hub and save locally (skip if already present)."""
    if (model_dir / "config.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))


def build_embeddings(corpus_path: Path) -> np.ndarray:
    raw_docs = read_jsonl(corpus_path)
    docs = [doc for doc in raw_docs if (doc.get("text") or "").strip()]
    encoder = DenseEncoder(DENSE_ENCODER_DIR)
    texts = [
        squash_ws(url_tokens(doc["url"]) + " " + doc["text"])
        for doc in docs
    ]
    embeddings = encoder.encode(texts, batch_size=64, max_length=256)
    out_path = corpus_path.with_suffix(".npy")
    np.save(out_path, embeddings.astype(np.float32))
    print(f"Built embeddings for {len(docs)} documents -> {out_path}")
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build page embeddings for a corpus.")
    parser.add_argument(
        "--corpus", type=str, default=None,
        help=f"Path to the corpus JSONL (default: {CORPUS_PATH}).",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else CORPUS_PATH
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found at {corpus_path}")

    materialize_model(DENSE_ENCODER_NAME, DENSE_ENCODER_DIR)
    build_embeddings(corpus_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
