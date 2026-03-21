from __future__ import annotations

import html
import re
import string
import unicodedata
from collections import Counter
from typing import Iterable


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@/&+-]*")

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "were",
    "are", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "which", "who", "whom", "whose", "what", "where",
    "when", "how", "why", "all", "each", "every", "both", "few", "more",
    "most", "some", "any", "such", "only", "same", "other", "into",
    "through", "about", "above", "below", "between", "under", "over",
    "after", "before", "during", "up", "down", "out", "off", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "one", "also", "just", "very",
})


def strip_diacritics(text: str) -> str:
    """Fold accented/special Unicode chars to ASCII equivalents (ö→o, é→e, etc.)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def squash_ws(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(text or "")).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(strip_diacritics(text or ""))]


def normalize_answer(text: str) -> str:
    text = squash_ws(text).lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def answer_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truths: Iterable[str]) -> bool:
    norm_pred = normalize_answer(prediction)
    return any(norm_pred == normalize_answer(answer) for answer in ground_truths)


def best_f1(prediction: str, ground_truths: Iterable[str]) -> float:
    return max((answer_f1(prediction, answer) for answer in ground_truths), default=0.0)


YES_NO_RE = re.compile(
    r"^(is|are|am|was|were|do|does|did|can|could|should|would|will|has|have|had|may)\b",
    re.IGNORECASE,
)


def is_yes_no_question(question: str) -> bool:
    return YES_NO_RE.match(question.strip()) is not None


def truncate_answer_words(answer: str, limit: int) -> str:
    words = squash_ws(answer).split()
    return " ".join(words[:limit])


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def url_tokens(url: str) -> str:
    cleaned = url.replace("https://", " ").replace("http://", " ")
    cleaned = cleaned.replace("/", " ").replace("-", " ").replace("_", " ").replace(".", " ")
    return squash_ws(cleaned)


_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_MIN_SECTION_CHARS = 80
_MAX_SECTION_CHARS = 800


def _merge_short(pieces: list[str], sep: str = "\n") -> list[str]:
    """Merge consecutive short pieces so every result >= _MIN_SECTION_CHARS."""
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        combined = (carry + sep + piece).strip() if carry else piece
        if len(combined) < _MIN_SECTION_CHARS:
            carry = combined
        else:
            merged.append(combined)
            carry = ""
    if carry:
        if merged:
            merged[-1] = merged[-1] + sep + carry
        else:
            merged.append(carry)
    return merged


def _split_long_section(text: str) -> list[str]:
    """Break an oversized section on paragraph / newline boundaries.

    Tries double-newline splits first; falls back to single-newline if the
    section has no double-newlines.
    """
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(parts) <= 1:
        return [text]
    return _merge_short(parts)


def split_into_sections(text: str) -> list[str]:
    """Split page text on markdown headers into sections.

    Very short sections (<80 chars) are merged with neighbors.  Oversized
    sections (>800 chars) are further split on paragraph boundaries so the
    scorer can select the most relevant passages.  Pages with no headers
    are split by paragraphs directly.
    """
    splits = _HEADER_RE.split(text)
    if len(splits) <= 1:
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) > _MAX_SECTION_CHARS:
            return _split_long_section(stripped)
        return [stripped]

    # re.split drops the matched header markers; recover them so each section
    # keeps its heading line for context.
    headers = _HEADER_RE.findall(text)
    sections: list[str] = []
    if splits[0].strip():
        sections.append(splits[0].strip())
    for hdr, body in zip(headers, splits[1:]):
        sections.append((hdr + body).strip())

    merged = _merge_short(sections)

    # Break up oversized sections on paragraph / newline boundaries
    result: list[str] = []
    for sec in merged:
        if len(sec) > _MAX_SECTION_CHARS:
            result.extend(_split_long_section(sec))
        else:
            result.append(sec)

    return result

