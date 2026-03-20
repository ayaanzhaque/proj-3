from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from project.constants import (
    ARTIFACT_DIR,
    CORPUS_PATH,
    DEFAULT_MIN_PAGE_SCORE,
    DEFAULT_NULL_MARGIN,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_ANSWER_WORDS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    PAGE_TOP_K,
)
from project.io_utils import read_json, read_jsonl
from project.modeling import DenseEncoder
from project.text_utils import (
    reciprocal_rank_fusion,
    squash_ws,
    tokenize,
    truncate_answer_words,
    url_tokens,
)

log = logging.getLogger(__name__)


def _page_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


@dataclass
class RuntimeConfig:
    null_margin: float = DEFAULT_NULL_MARGIN
    min_page_score: float = DEFAULT_MIN_PAGE_SCORE
    page_top_k: int = PAGE_TOP_K


class RetrievalRuntime:
    """Single-stage page-level retrieval over the rewritten corpus."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ARTIFACT_DIR.parent
        corpus_path = self.root / CORPUS_PATH.name
        raw_docs = read_jsonl(corpus_path)

        # Filter out empty documents and assign stable page IDs
        self.docs: list[dict] = []
        for doc in raw_docs:
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            doc["page_id"] = _page_id_from_url(doc["url"])
            self.docs.append(doc)

        self.page_embeddings = np.load(self.root / "artifacts" / "page_embeddings.npy")
        runtime_config_path = self.root / "artifacts" / "runtime_config.json"
        raw_config = read_json(runtime_config_path) if runtime_config_path.exists() else {}
        raw_config.pop("min_chunk_score", None)
        self.config = RuntimeConfig(**raw_config)

        self.doc_ids = [doc["page_id"] for doc in self.docs]
        self.id_to_idx = {doc["page_id"]: i for i, doc in enumerate(self.docs)}

        # BM25 index over page texts (prepend URL tokens for domain signal)
        doc_corpus = [
            squash_ws(url_tokens(doc["url"]) + " " + doc["text"])
            for doc in self.docs
        ]
        self.doc_tokens = [tokenize(text) for text in doc_corpus]
        self.doc_bm25 = BM25Okapi(self.doc_tokens)

        self.dense_encoder = DenseEncoder(self.root / "artifacts" / "models" / "dense_encoder")

    def _retrieve_for_question(
        self,
        *,
        question: str,
        question_tokens: list[str],
        query_embedding: np.ndarray,
    ) -> list[dict]:
        # Sparse (BM25) scores
        scores_sparse = self.doc_bm25.get_scores(question_tokens)
        # Dense (embedding) scores
        scores_dense = self.page_embeddings @ query_embedding

        n_candidates = max(self.config.page_top_k * 4, 20)
        sparse_ranking = [
            self.doc_ids[idx]
            for idx in np.argsort(scores_sparse)[::-1][:n_candidates]
        ]
        dense_ranking = [
            self.doc_ids[idx]
            for idx in np.argsort(scores_dense)[::-1][:n_candidates]
        ]
        fused_scores = reciprocal_rank_fusion([sparse_ranking, dense_ranking])

        top_ids = [
            page_id
            for page_id, _ in sorted(
                fused_scores.items(), key=lambda item: item[1], reverse=True
            )[: self.config.page_top_k]
        ]
        if not top_ids:
            return []

        best_score = max(fused_scores.values(), default=0.0)
        max_sparse = float(np.max(scores_sparse)) if len(scores_sparse) else 0.0
        if best_score < self.config.min_page_score and max_sparse <= 0.0:
            return []

        results: list[dict] = []
        for page_id in top_ids:
            idx = self.id_to_idx.get(page_id)
            if idx is None:
                continue
            doc = dict(self.docs[idx])
            doc["retrieval_score"] = fused_scores.get(page_id, 0.0)
            results.append(doc)
        return results

    def retrieve_chunks(self, question: str) -> list[dict]:
        question_tokens = tokenize(question)
        query_embedding = self.dense_encoder.encode([question], batch_size=1)[0]
        return self._retrieve_for_question(
            question=question,
            question_tokens=question_tokens,
            query_embedding=query_embedding,
        )

    def retrieve_many(self, questions: list[str]) -> list[list[dict]]:
        if not questions:
            return []
        query_embeddings = self.dense_encoder.encode(questions, batch_size=64)
        question_tokens_list = [tokenize(q) for q in questions]
        return [
            self._retrieve_for_question(
                question=question,
                question_tokens=question_tokens,
                query_embedding=query_embedding,
            )
            for question, question_tokens, query_embedding in zip(
                questions, question_tokens_list, query_embeddings
            )
        ]

    def answer_many(self, questions: list[str], retrieved_docs: list[list[dict]]) -> list[str]:
        from llm import call_llm

        answers: list[str] = []
        for question, docs in zip(questions, retrieved_docs):
            answers.append(self._llm_answer(call_llm, question, docs))
        return answers

    def _llm_answer(self, call_llm, question: str, docs: list[dict]) -> str:
        context = build_context(docs, MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHARS)
        if not context:
            return "UNKNOWN"

        query = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            raw = call_llm(
                query=query,
                system_prompt=LLM_SYSTEM_PROMPT,
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT,
            )
        except Exception as exc:
            log.warning("LLM call failed for question %r: %s", question, exc)
            return "UNKNOWN"

        return clean_llm_answer(question, raw)


def build_context(docs: list[dict], max_docs: int, max_chars: int) -> str:
    """Build LLM context from retrieved pages, truncating each to fit budget."""
    parts: list[str] = []
    total = 0
    for doc in docs[:max_docs]:
        text = squash_ws(doc.get("text", ""))
        if total + len(text) > max_chars and parts:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def clean_llm_answer(question: str, raw: str) -> str:
    answer = raw.strip().strip('"\'').strip()
    for prefix in ("Answer:", "The answer is", "Based on the context,", "According to the context,"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip().strip(":").strip()

    answer = answer.split("\n")[0].strip()
    answer = truncate_answer_words(answer, MAX_ANSWER_WORDS).strip(" ,;:.")

    if not answer or answer.upper() == "UNKNOWN":
        return "UNKNOWN"
    return answer
