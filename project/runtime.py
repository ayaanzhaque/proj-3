from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from project.constants import (
    ARTIFACT_DIR,
    CHUNK_TOP_K,
    DEFAULT_MIN_CHUNK_SCORE,
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
    lexical_overlap,
    normalize_answer,
    reciprocal_rank_fusion,
    squash_ws,
    tokenize,
    truncate_answer_words,
    url_tokens,
)

log = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    null_margin: float = DEFAULT_NULL_MARGIN
    min_page_score: float = DEFAULT_MIN_PAGE_SCORE
    min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE
    page_top_k: int = PAGE_TOP_K
    chunk_top_k: int = CHUNK_TOP_K


class RetrievalRuntime:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ARTIFACT_DIR.parent
        self.pages = read_jsonl(self.root / "data" / "corpus" / "pages.jsonl")
        self.chunks = read_jsonl(self.root / "data" / "corpus" / "chunks.jsonl")
        self.page_embeddings = np.load(self.root / "artifacts" / "page_embeddings.npy")
        runtime_config_path = self.root / "artifacts" / "runtime_config.json"
        raw_config = read_json(runtime_config_path) if runtime_config_path.exists() else {}
        self.config = RuntimeConfig(**raw_config)
        self.page_lookup = {page["page_id"]: page for page in self.pages}
        self.chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.chunk_ids = [chunk["chunk_id"] for chunk in self.chunks]
        self.chunk_page_ids = np.array([chunk["page_id"] for chunk in self.chunks], dtype=object)
        self.page_ids = [page["page_id"] for page in self.pages]

        page_corpus = [
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
            for page in self.pages
        ]
        self.page_tokens = [tokenize(text) for text in page_corpus]
        self.page_bm25 = BM25Okapi(self.page_tokens)

        chunk_corpus = [
            squash_ws(" ".join([chunk["title"], chunk["heading"], chunk["text"]])) for chunk in self.chunks
        ]
        self.chunk_tokens = [tokenize(text) for text in chunk_corpus]
        self.chunk_bm25 = BM25Okapi(self.chunk_tokens)

        self.dense_encoder = DenseEncoder(self.root / "artifacts" / "models" / "dense_encoder")

    def _retrieve_for_question(
        self,
        *,
        question: str,
        question_tokens: list[str],
        query_embedding: np.ndarray,
    ) -> list[dict]:
        page_scores_sparse = self.page_bm25.get_scores(question_tokens)
        page_scores_dense = self.page_embeddings @ query_embedding

        sparse_ranking = [
            self.page_ids[idx]
            for idx in np.argsort(page_scores_sparse)[::-1][: max(self.config.page_top_k * 4, 20)]
        ]
        dense_ranking = [
            self.page_ids[idx]
            for idx in np.argsort(page_scores_dense)[::-1][: max(self.config.page_top_k * 4, 20)]
        ]
        fused_scores = reciprocal_rank_fusion([sparse_ranking, dense_ranking])
        top_pages = [
            page_id
            for page_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[
                : self.config.page_top_k
            ]
        ]
        if not top_pages:
            return []

        max_sparse = float(np.max(page_scores_sparse)) if len(page_scores_sparse) else 0.0
        best_page_score = max(fused_scores.values(), default=0.0)
        if best_page_score < self.config.min_page_score and max_sparse <= 0.0:
            return []

        mask = np.isin(self.chunk_page_ids, np.array(top_pages, dtype=object))
        chunk_scores_sparse = self.chunk_bm25.get_scores(question_tokens)
        candidate_indices = np.where(mask)[0]
        scored_chunks: list[tuple[float, int]] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            score = float(chunk_scores_sparse[idx])
            score += 0.25 * lexical_overlap(question, chunk["heading"])
            score += 0.15 * lexical_overlap(question, chunk["title"])
            scored_chunks.append((score, idx))
        scored_chunks.sort(reverse=True)
        top_chunks: list[dict] = []
        for score, idx in scored_chunks[: self.config.chunk_top_k]:
            if score < self.config.min_chunk_score and top_chunks:
                continue
            chunk = dict(self.chunks[idx])
            chunk["retrieval_score"] = score
            top_chunks.append(chunk)
        return top_chunks

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
        question_tokens_list = [tokenize(question) for question in questions]
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

    def answer_many(self, questions: list[str], retrieved_chunks: list[list[dict]]) -> list[str]:
        from llm import call_llm

        answers: list[str] = []
        for question, chunks in zip(questions, retrieved_chunks):
            rule_answer = try_rule_based_answer(question, chunks)
            if rule_answer is not None:
                answers.append(rule_answer)
                continue
            answers.append(self._llm_answer(call_llm, question, chunks))
        return answers

    def _llm_answer(self, call_llm, question: str, chunks: list[dict]) -> str:
        context = build_context(chunks, MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHARS)
        if not context:
            breakpoint()
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
            breakpoint()
            log.warning("LLM call failed for question %r: %s", question, exc)
            return "UNKNOWN"

        return clean_llm_answer(question, raw)


def build_context(chunks: list[dict], max_chunks: int, max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks[:max_chunks]:
        header = chunk.get("title", "")
        heading = chunk.get("heading", "")
        text = chunk.get("text", "")
        snippet = squash_ws(" | ".join(filter(None, [header, heading, text])))
        if total + len(snippet) > max_chars and parts:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)


def clean_llm_answer(question: str, raw: str) -> str:
    answer = raw.strip().strip('"\'').strip()
    # Remove common LLM preamble patterns
    for prefix in ("Answer:", "The answer is", "Based on the context,", "According to the context,"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip().strip(":").strip()

    answer = answer.split("\n")[0].strip()
    answer = truncate_answer_words(answer, MAX_ANSWER_WORDS).strip(" ,;:.")

    if not answer or answer.upper() == "UNKNOWN":
        return "UNKNOWN"
    return answer


def parse_kv_row(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split("|"):
        part = squash_ws(part)
        if ": " in part:
            key, value = part.split(": ", 1)
            fields[key.lower()] = value
    return fields


def pipe_parts(text: str) -> list[str]:
    return [squash_ws(part) for part in text.split("|") if squash_ws(part)]


def matches_target(text: str, target: str) -> bool:
    return normalize_answer(target) in normalize_answer(text)


def try_rule_based_answer(question: str, chunks: list[dict]) -> str | None:
    question_norm = normalize_answer(question)
    course_title_match = re.match(r"what is the title of (.+)", question_norm)
    reverse_course_match = re.match(r"which (.+) course is titled (.+)", question_norm)
    capacity_match = re.match(r"what is the capacity of (.+)", question_norm)
    grad_match = re.match(r"when is (.+) expected to graduate", question_norm)
    breadth_match = re.match(r"what is the breadth area of (.+)", question_norm)
    date_match = re.match(r"when is (.+)", question_norm)
    happens_match = re.match(r"what happens on (.+) in", question_norm)
    stat_match = re.match(r"what number is listed for (.+) on the", question_norm)
    email_match = re.match(r"which email is mentioned for (.+)", question_norm)

    for chunk in chunks:
        text = chunk["text"]
        fields = parse_kv_row(text)
        parts = pipe_parts(text)

        if course_title_match:
            target = course_title_match.group(1)
            if fields and fields.get("course number") and (
                matches_target(fields["course number"], target) or matches_target(target, fields["course number"])
            ):
                answer = fields.get("course title") or fields.get("course")
                if answer:
                    return answer
            if len(parts) >= 2 and matches_target(parts[0], target):
                return parts[1]

        if reverse_course_match:
            target_title = reverse_course_match.group(2)
            if fields:
                answer = fields.get("course number") or fields.get("number")
                title = fields.get("course title") or fields.get("course")
                if answer and title and matches_target(title, target_title):
                    return answer
            if len(parts) >= 2 and matches_target(parts[1], target_title):
                return parts[0]

        if capacity_match:
            target = capacity_match.group(1)
            if fields.get("room name/number") and matches_target(fields["room name/number"], target):
                answer = fields.get("cap.")
                if answer:
                    return answer

        if grad_match:
            target = grad_match.group(1)
            if fields.get("full name") and matches_target(fields["full name"], target):
                answer = fields.get("semester of graduation")
                if answer:
                    return answer

        if breadth_match:
            target = breadth_match.group(1)
            if fields.get("full name") and matches_target(fields["full name"], target):
                answer = fields.get("breadth area")
                if answer:
                    return answer

        if date_match and fields.get("proceeding") and fields.get("date"):
            target = date_match.group(1)
            if matches_target(fields["proceeding"], target):
                return fields["date"]

        if happens_match and fields.get("proceeding") and fields.get("date"):
            target = happens_match.group(1)
            if matches_target(fields["date"], target):
                return fields["proceeding"]

        if stat_match and len(parts) == 2:
            target = stat_match.group(1)
            if matches_target(parts[0], target):
                return parts[1]

        if email_match:
            target = email_match.group(1)
            if fields.get("room name/number") and matches_target(fields["room name/number"], target):
                match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+)", text)
                if match:
                    return match.group(1)

        if "retroactive change in class schedule" in question_norm and fields.get("form(s)"):
            if "retroactive change in class schedule" in normalize_answer(fields.get("description", "")):
                return fields["form(s)"]

        if "request to take the qualifying exam" in question_norm and fields.get("form(s)"):
            if "request to take the qualifying exam" in normalize_answer(fields.get("description", "")):
                return fields["form(s)"]

    return None
