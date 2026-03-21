from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag.constants import (
    CORPUS_PATH,
    DEFAULT_MIN_PAGE_SCORE,
    DEFAULT_NULL_MARGIN,
    DENSE_ENCODER_DIR,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_ANSWER_WORDS,
    MAX_CONTEXT_CHARS,
    PAGE_TOP_K,
    RRF_CANDIDATES,
)
from rag.io_utils import read_jsonl
from rag.encoder import DenseEncoder
from rag.text_utils import (
    STOPWORDS,
    reciprocal_rank_fusion,
    split_into_sections,
    squash_ws,
    tokenize,
    truncate_answer_words,
    url_tokens,
)
from llm import call_llm

log = logging.getLogger(__name__)


def _page_id_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


@dataclass
class RuntimeConfig:
    null_margin: float = DEFAULT_NULL_MARGIN
    min_page_score: float = DEFAULT_MIN_PAGE_SCORE
    page_top_k: int = PAGE_TOP_K


@dataclass
class AnswerDiag:
    """Diagnostic info from a single LLM answer call."""
    answer: str
    llm_query: str
    system_prompt: str
    raw_response: str
    error: str | None = None


class RAGModel:
    """Single-stage page-level retrieval over the rewritten corpus."""

    def __init__(self, corpus_path: Path | str | None = None) -> None:
        corpus_path = Path(corpus_path) if corpus_path else CORPUS_PATH
        raw_docs = read_jsonl(corpus_path)

        # Filter out empty documents and assign stable page IDs
        self.docs: list[dict] = []
        for doc in raw_docs:
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            doc["page_id"] = _page_id_from_url(doc["url"])
            self.docs.append(doc)

        embeddings_path = corpus_path.with_suffix(".npy")
        self.page_embeddings = np.load(embeddings_path)
        self.config = RuntimeConfig()

        self.doc_ids = [doc["page_id"] for doc in self.docs]
        self.id_to_idx = {doc["page_id"]: i for i, doc in enumerate(self.docs)}

        # BM25 index over page texts (prepend URL tokens for domain signal)
        doc_corpus = [
            squash_ws(url_tokens(doc["url"]) + " " + doc["text"])
            for doc in self.docs
        ]
        self.doc_tokens = [tokenize(text) for text in doc_corpus]
        self.doc_bm25 = BM25Okapi(self.doc_tokens)

        self.dense_encoder = DenseEncoder(DENSE_ENCODER_DIR)

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

        sparse_ranking = [
            self.doc_ids[idx]
            for idx in np.argsort(scores_sparse)[::-1][:RRF_CANDIDATES]
        ]
        dense_ranking = [
            self.doc_ids[idx]
            for idx in np.argsort(scores_dense)[::-1][:RRF_CANDIDATES]
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

    def predict(self, questions: list[str]) -> list[str]:
        if not questions:
            return []
        retrieved = self.retrieve_many(questions)
        return self.answer_many(questions, retrieved)

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
        answers: list[str] = []
        for question, docs in zip(questions, retrieved_docs):
            answers.append(self._llm_answer(call_llm, question, docs))
        return answers

    def answer_one_debug(self, question: str, docs: list[dict]) -> AnswerDiag:
        """Like _llm_answer but returns full diagnostics for debugging."""
        context = build_context(docs, question, MAX_CONTEXT_CHARS)
        if not context:
            return AnswerDiag(
                answer="UNKNOWN",
                llm_query="",
                system_prompt=LLM_SYSTEM_PROMPT,
                raw_response="",
                error="no context (all retrieved docs were empty)",
            )

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
            return AnswerDiag(
                answer="UNKNOWN",
                llm_query=query,
                system_prompt=LLM_SYSTEM_PROMPT,
                raw_response="",
                error=f"{type(exc).__name__}: {exc}",
            )

        return AnswerDiag(
            answer=clean_llm_answer(question, raw),
            llm_query=query,
            system_prompt=LLM_SYSTEM_PROMPT,
            raw_response=raw,
        )

    def _llm_answer(self, call_llm, question: str, docs: list[dict]) -> str:
        context = build_context(docs, question, MAX_CONTEXT_CHARS)
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


def _score_section(query_tokens: set[str], section: str) -> float:
    """Fraction of non-stopword query tokens that appear in the section."""
    if not query_tokens:
        return 0.0
    sec_tokens = set(tokenize(section))
    return len(query_tokens & sec_tokens) / len(query_tokens)


def build_context(docs: list[dict], question: str, max_chars: int) -> str:
    """Build LLM context by selecting the most query-relevant sections.

    Each retrieved page is split on markdown headers.  Sections are scored by
    lexical overlap with the question, then greedily packed into the character
    budget so the LLM sees the most relevant passages across *all* retrieved
    pages.
    """
    query_tokens = set(tokenize(question)) - STOPWORDS

    # Score every section across all pages
    page_sections: list[list[tuple[float, str]]] = []
    for doc in docs:
        text = doc.get("text", "")
        sections = split_into_sections(text)
        page_sections.append([
            (_score_section(query_tokens, sec), sec) for sec in sections
        ])

    # Reserve part of the budget for the top-ranked page so its full context
    # is preserved, then fill the rest with the best cross-page sections.
    top_page_budget = max_chars * 2 // 5

    separator = "\n\n"
    sep_len = len(separator)

    def _pack(candidates: list[tuple[float, str]], budget: int,
              parts: list[str], total: int, seen: set[str]) -> int:
        """Greedily pack highest-scoring sections into parts up to budget."""
        for _score, section in candidates:
            if section in seen:
                continue
            added_len = len(section) + (sep_len if parts else 0)
            if total + added_len > budget:
                remaining = budget - total - (sep_len if parts else 0)
                if remaining > 100:
                    parts.append(section[:remaining])
                    total += remaining + (sep_len if len(parts) > 1 else 0)
                    seen.add(section[:remaining])
                break
            parts.append(section)
            total += added_len
            seen.add(section)
        return total

    parts: list[str] = []
    seen: set[str] = set()
    total = 0

    # Pass 1: top-ranked page
    if page_sections:
        top_secs = sorted(page_sections[0], key=lambda t: t[0], reverse=True)
        total = _pack(top_secs, top_page_budget, parts, total, seen)

    # Pass 2: best remaining sections from all pages
    all_secs = sorted(
        [item for page in page_sections for item in page],
        key=lambda t: t[0], reverse=True,
    )
    total = _pack(all_secs, max_chars, parts, total, seen)

    return separator.join(parts)


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
