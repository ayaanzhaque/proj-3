#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from rag.constants import (
    BLOCK_TAGS,
    DROP_TAGS,
    MAIN_DOMAIN,
    USER_AGENT,
    WWW2_SEEDS,
)
from rag.io_utils import write_jsonl
from rag.text_utils import squash_ws


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return session


def canonicalize_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname not in {MAIN_DOMAIN, "www2.eecs.berkeley.edu"}:
        return None
    path = parsed.path or "/"
    lowered = path.lower()
    if any(lowered.endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx")):
        return None
    normalized = parsed._replace(
        scheme="https",
        netloc=hostname,
        query="",
        fragment="",
        path=path.rstrip("/") + ("/" if path.endswith("/") else ""),
    )
    candidate = urlunparse(normalized)
    return candidate.rstrip("#")


def spider_domain(
    session: requests.Session,
    seeds: list[str],
    source_type: str = "spider",
    depth: int = 3,
    max_urls: int = 3000,
) -> list[dict[str, str]]:
    """General-purpose spider that follows all links within allowed domains."""
    seen: set[str] = set()
    discovered: list[dict[str, str]] = []
    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
    pbar = tqdm(desc=source_type, unit="url")
    while queue:
        if len(discovered) >= max_urls:
            break
        url, level = queue.popleft()
        url = canonicalize_url(url) or url
        if url in seen:
            continue
        seen.add(url)
        discovered.append({"url": url, "source_type": source_type, "updated_at": None})
        pbar.update(1)
        if level >= depth:
            continue
        response = fetch_with_backoff(session, url)
        if response is None:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            child = canonicalize_url(urljoin(url, anchor["href"]))
            if not child or child in seen:
                continue
            if "Protected" in child:
                continue
            if len(seen) + len(queue) >= max_urls:
                continue
            queue.append((child, level + 1))
    pbar.close()
    return discovered


def select_root(soup: BeautifulSoup) -> Tag:
    candidates = []
    for selector in ("main", "#site-main", "article", ".entry-content", ".post-content", "body"):
        node = soup.select_one(selector)
        if node is None:
            continue
        text_len = len(squash_ws(node.get_text(" ", strip=True)))
        candidates.append((text_len, node))
    if not candidates:
        raise ValueError("No content root found")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def has_block_ancestor(node: Tag, root: Tag) -> bool:
    parent = node.parent
    while parent and parent is not root:
        if getattr(parent, "name", None) in BLOCK_TAGS:
            return True
        parent = parent.parent
    return False


def parse_table(table: Tag) -> list[str]:
    headers: list[str] = []
    rows: list[str] = []
    for row_idx, tr in enumerate(table.find_all("tr")):
        cells = tr.find_all(["th", "td"])
        values = [squash_ws(cell.get_text(" ", strip=True)) for cell in cells]
        values = [value for value in values if value]
        if not values:
            continue
        if row_idx == 0 and tr.find_all("th"):
            headers = values
            continue
        if headers and len(headers) == len(values):
            row_text = " | ".join(f"{header}: {value}" for header, value in zip(headers, values))
        else:
            row_text = " | ".join(values)
        rows.append(row_text)
    return rows


def clean_page(url: str, html_text: str) -> dict[str, str] | None:
    """Extract linearized text from an HTML page. Returns {"url", "text"} or None."""
    soup = BeautifulSoup(html_text, "html.parser")
    title = squash_ws(soup.title.get_text(" ", strip=True) if soup.title else url)
    root = select_root(soup)
    root = deepcopy(root)

    for tag in root.find_all(DROP_TAGS):
        tag.decompose()
    for element in root.find_all(attrs={"aria-hidden": "true"}):
        element.decompose()

    table_rows: list[str] = []
    for table in root.find_all("table"):
        table_rows.extend(parse_table(table))
        table.decompose()

    section_lines: list[str] = [f"# {title}"]
    for tag in root.find_all(list(BLOCK_TAGS)):
        if has_block_ancestor(tag, root):
            continue
        text = squash_ws(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            section_lines.append(f"## {text}")
        else:
            if section_lines and section_lines[-1] == text:
                continue
            section_lines.append(text)

    if table_rows:
        section_lines.append("## Table Data")
        section_lines.extend(table_rows)

    cleaned_text = "\n".join(section_lines).strip()
    if len(cleaned_text) < 60 and not table_rows:
        return None
    return {"url": url, "text": cleaned_text}


def fetch_with_backoff(
    session: requests.Session, url: str, max_retries: int = 5
) -> requests.Response | None:
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 429:
                wait = 2 ** attempt
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                print(f"  [429] {url} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError:
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    print(f"  [warn] gave up on {url} after {max_retries} retries", flush=True)
    return None


def fetch_and_clean(row: dict[str, str], session: requests.Session) -> dict[str, str] | None:
    url = row["url"]
    response = fetch_with_backoff(session, url)
    if response is None:
        return None
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None
    return clean_page(url=url, html_text=response.text)


def crawl_pages(
    url_rows: list[dict[str, str]], workers: int = 16
) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in url_rows:
        if row["url"] not in seen:
            seen.add(row["url"])
            unique_rows.append(row)

    pages: list[dict[str, str]] = []
    session = build_session()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_and_clean, row, session): row for row in unique_rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="crawling", unit="page"):
            page = future.result()
            if page is not None:
                pages.append(page)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EECS retrieval corpus.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads.")
    parser.add_argument("--depth", type=int, default=3, help="Max link-follow depth for the spider.")
    parser.add_argument("--max-urls", type=int, default=6000, help="Max URLs to discover.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of URLs to crawl (0 = no limit).")
    parser.add_argument("--save-path", type=str, required=True, help="Save path of corpus")
    args = parser.parse_args()

    session = build_session()
    seeds = [f"https://{MAIN_DOMAIN}/"] + list(WWW2_SEEDS)
    url_rows = spider_domain(session, seeds, depth=args.depth, max_urls=args.max_urls)
    url_rows.sort(key=lambda row: row["url"])
    if args.limit > 0:
        url_rows = url_rows[: args.limit]
    print(f"Crawling {len(url_rows)} URLs", flush=True)

    pages = crawl_pages(url_rows, workers=args.workers)
    print(f"Built {len(pages)} pages", flush=True)
    write_jsonl(args.save_path, pages)
    print(f"Wrote corpus to {args.save_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
