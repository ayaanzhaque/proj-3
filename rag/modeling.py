from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEncoder:
    """Thin wrapper around SentenceTransformer that keeps the same interface
    used by runtime.py and rag/corpus/build_artifacts.py."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model = SentenceTransformer(str(model_dir), device=device)

    def encode(self, texts: list[str], batch_size: int = 64, max_length: int = 256) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension() or 384
            return np.zeros((0, dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
