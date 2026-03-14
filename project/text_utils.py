from __future__ import annotations

import html
import re
import string
from collections import Counter
from typing import Iterable


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@/&+-]*")


def squash_ws(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(text or "")).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


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

