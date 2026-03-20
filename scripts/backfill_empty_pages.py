#!/usr/bin/env python3
"""Backfill empty pages in eecs_text_bs_rewritten.jsonl.

Finds all entries with empty ``text``, re-fetches each URL, linearizes the
HTML using the same pipeline as build_corpus.py, and writes the updated
corpus back to disk.

Usage:
    python scripts/backfill_empty_pages.py
    python scripts/backfill_empty_pages.py --dry-run
    python scripts/backfill_empty_pages.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.constants import CORPUS_PATH
from scripts.build_corpus import build_session, clean_page, _fetch_with_backoff


def load_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_corpus(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def fetch_and_linearize(
    url: str, session: requests.Session
) -> str | None:
    """Fetch a URL and return linearized text, or None on failure."""
    resp = _fetch_with_backoff(session, url)
    if resp is None:
        return None
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None
    page = clean_page(
        url=url,
        html_text=resp.text,
        source_type="backfill",
        updated_at=None,
    )
    if page is None or not page.text.strip():
        return None
    # Match the existing corpus format: "# {title}\n\n{body}"
    return f"# {page.title}\n\n{page.text}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill empty pages in eecs_text_bs_rewritten.jsonl."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be updated without writing changes.",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel download threads (default: 16).",
    )
    args = parser.parse_args()

    corpus = load_corpus(CORPUS_PATH)
    empty_indices = [
        i for i, row in enumerate(corpus)
        if not (row.get("text") or "").strip()
    ]
    print(f"Corpus has {len(corpus)} entries, {len(empty_indices)} with empty text.")
    if not empty_indices:
        print("Nothing to backfill.")
        return 0

    session = build_session()
    filled = 0
    failed_urls: list[str] = []

    def _process(idx: int) -> tuple[int, str | None]:
        url = corpus[idx]["url"]
        text = fetch_and_linearize(url, session)
        return idx, text

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process, i): i for i in empty_indices}
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="backfilling", unit="page",
        ):
            idx, text = future.result()
            url = corpus[idx]["url"]
            if text:
                if args.dry_run:
                    preview = text[:120].replace("\n", "\\n")
                    print(f"  [FILL] {url}  →  {preview}…")
                else:
                    corpus[idx]["text"] = text
                filled += 1
            else:
                failed_urls.append(url)

    print(f"\nFilled: {filled}/{len(empty_indices)}")
    if failed_urls:
        print(f"Failed to fetch or extract text ({len(failed_urls)}):")
        for u in sorted(failed_urls)[:30]:
            print(f"  {u}")
        if len(failed_urls) > 30:
            print(f"  … and {len(failed_urls) - 30} more")

    if not args.dry_run and filled > 0:
        save_corpus(CORPUS_PATH, corpus)
        print(f"\nUpdated {CORPUS_PATH}")
    elif args.dry_run:
        print("\n(dry-run mode — no files were written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
