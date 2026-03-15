#!/usr/bin/env python3
"""Run the RAG model on a CSV test set and evaluate with EM / token-F1.

Usage:
    python scripts/evaluate_test_set.py test_set.csv
    python scripts/evaluate_test_set.py test_set.csv -v
    python scripts/evaluate_test_set.py test_set.csv --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag import RAGModel
from project.text_utils import best_f1, exact_match, normalize_answer


def load_csv(path: str) -> list[dict[str, str]]:
    """Read a CSV with columns: question, answer, url, labeler."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def normalize_url(url: str) -> str:
    """Strip scheme, trailing slash, and fragment for URL comparison."""
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}".lower()


def url_hit(gold_url: str, retrieved_urls: list[str]) -> bool:
    """Check if the gold URL matches any retrieved URL (after normalization)."""
    gold_norm = normalize_url(gold_url)
    return any(normalize_url(u) == gold_norm for u in retrieved_urls)


def answer_in_chunks(gold_answers: list[str], chunks: list[dict]) -> bool:
    """Check if any gold answer string appears in any retrieved chunk text."""
    for ans in gold_answers:
        norm_ans = normalize_answer(ans)
        if not norm_ans:
            continue
        for chunk in chunks:
            if norm_ans in normalize_answer(chunk.get("text", "")):
                return True
    return False


def truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG model on a CSV test set (question,answer,url,labeler)."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV test set.")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-question retrieval details (retrieved URLs, top chunks, scores).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional path to write per-question results CSV.",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        return 1

    questions = [r["question"] for r in rows]
    gold_answers = [r["answer"].split("|") for r in rows]
    urls = [r.get("url", "") for r in rows]

    print(f"Loaded {len(rows)} questions from {args.csv_path}")
    print("Loading RAG model …")
    model = RAGModel()
    runtime = model.runtime

    # Build a set of all URLs in the corpus for coverage checks
    corpus_urls = {normalize_url(p["url"]) for p in runtime.pages}

    print("Retrieving …")
    t0 = time.perf_counter()
    all_retrieved = runtime.retrieve_many(questions)
    t_retrieve = time.perf_counter() - t0

    print("Answering …")
    t1 = time.perf_counter()
    predictions = runtime.answer_many(questions, all_retrieved)
    t_answer = time.perf_counter() - t1

    elapsed = t_retrieve + t_answer
    print(
        f"Done in {elapsed:.2f}s  "
        f"(retrieve {t_retrieve:.2f}s + answer {t_answer:.2f}s, "
        f"{elapsed / len(rows):.3f}s/question)\n"
    )

    # --- per-question metrics ---
    em_scores: list[float] = []
    f1_scores: list[float] = []
    url_recalls: list[float] = []
    answer_recalls: list[float] = []
    in_corpus_flags: list[float] = []
    result_rows: list[dict[str, str]] = []

    for idx, (pred, gold, chunks) in enumerate(
        zip(predictions, gold_answers, all_retrieved)
    ):
        em = 1.0 if exact_match(pred, gold) else 0.0
        f1 = best_f1(pred, gold)
        retrieved_urls = list({c["url"] for c in chunks})
        u_hit = 1.0 if url_hit(urls[idx], retrieved_urls) else 0.0
        a_hit = 1.0 if answer_in_chunks(gold, chunks) else 0.0
        in_corpus = 1.0 if normalize_url(urls[idx]) in corpus_urls else 0.0

        em_scores.append(em)
        f1_scores.append(f1)
        url_recalls.append(u_hit)
        answer_recalls.append(a_hit)
        in_corpus_flags.append(in_corpus)

        result_rows.append(
            {
                "idx": str(idx + 1),
                "question": questions[idx],
                "gold": "|".join(gold),
                "predicted": pred,
                "em": f"{em:.0f}",
                "f1": f"{f1:.4f}",
                "url": urls[idx],
                "in_corpus": f"{in_corpus:.0f}",
                "url_recall": f"{u_hit:.0f}",
                "answer_recall": f"{a_hit:.0f}",
            }
        )

        if args.verbose:
            marker = "CORRECT" if em else (f"PARTIAL (F1={f1:.2f})" if f1 > 0 else "MISS")
            print(f"  [{idx + 1}] {marker}")
            print(f"       Q    : {questions[idx]}")
            print(f"       Gold : {' | '.join(gold)}")
            print(f"       Pred : {pred}")
            print(f"       Gold URL : {urls[idx]}")
            print(f"       In corpus: {'YES' if in_corpus else 'NO'}    "
                  f"URL retrieved: {'YES' if u_hit else 'NO'}    "
                  f"Answer in chunks: {'YES' if a_hit else 'NO'}")
            if chunks:
                print(f"       Retrieved {len(chunks)} chunks:")
                for ci, c in enumerate(chunks):
                    print(
                        f"         #{ci + 1}  score={c['retrieval_score']:.3f}  "
                        f"url={c['url']}"
                    )
                    print(f"              title: {c['title']}")
                    print(f"              text : {truncate(c['text'])}")
            else:
                print("       (no chunks retrieved)")
            print()

    # --- aggregate metrics ---
    mean_em = statistics.mean(em_scores)
    mean_f1 = statistics.mean(f1_scores)
    mean_url_recall = statistics.mean(url_recalls)
    mean_answer_recall = statistics.mean(answer_recalls)
    mean_in_corpus = statistics.mean(in_corpus_flags)

    print("=" * 72)
    print(f"  Examples        : {len(rows)}")
    print(f"  Exact Match     : {mean_em:.4f}  ({sum(em_scores):.0f}/{len(em_scores)})")
    print(f"  Token F1        : {mean_f1:.4f}")
    print(f"  Corpus Coverage : {mean_in_corpus:.4f}  "
          f"({sum(in_corpus_flags):.0f}/{len(in_corpus_flags)})")
    print(f"  URL Recall      : {mean_url_recall:.4f}  "
          f"({sum(url_recalls):.0f}/{len(url_recalls)})")
    print(f"  Answer Recall   : {mean_answer_recall:.4f}  "
          f"({sum(answer_recalls):.0f}/{len(answer_recalls)})")
    print("=" * 72)

    # --- list URLs missing from corpus ---
    missing_urls = [r for r in result_rows if r["in_corpus"] == "0"]
    if missing_urls:
        seen = set()
        print(f"\nURLs not in corpus ({len(missing_urls)} questions):\n")
        for r in missing_urls:
            if r["url"] not in seen:
                seen.add(r["url"])
                count = sum(1 for x in missing_urls if x["url"] == r["url"])
                print(f"  {r['url']}  ({count} question{'s' if count > 1 else ''})")
        print()

    # --- print misses (only in non-verbose mode, since verbose already shows everything) ---
    if not args.verbose:
        misses = [r for r in result_rows if r["em"] == "0"]
        if misses:
            print(f"\nMissed questions ({len(misses)}):\n")
            for r in misses:
                print(f"  [{r['idx']}] Q: {r['question']}")
                print(f"       Gold : {r['gold']}")
                print(f"       Pred : {r['predicted']}")
                print(f"       F1   : {r['f1']}   URL: {r['url']}")
                print(f"       In corpus: {('YES' if r['in_corpus'] == '1' else 'NO')}   "
                      f"URL retrieved: {('YES' if r['url_recall'] == '1' else 'NO')}   "
                      f"Answer in chunks: {('YES' if r['answer_recall'] == '1' else 'NO')}")
                print()

    # --- optional CSV output ---
    if args.out:
        fieldnames = [
            "idx", "question", "gold", "predicted", "em", "f1",
            "url", "in_corpus", "url_recall", "answer_recall",
        ]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"Per-question results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
