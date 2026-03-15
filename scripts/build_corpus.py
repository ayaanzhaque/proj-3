#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.constants import (
    ALLOWED_DOMAINS,
    BLOCK_TAGS,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    CORPUS_DIR,
    DROP_TAGS,
    MAIN_DOMAIN,
    PAGES_PATH,
    CHUNKS_PATH,
    USER_AGENT,
    WORDPRESS_TYPES,
    WWW2_SEEDS,
)
from project.io_utils import write_json, write_jsonl
from project.schema import ChunkRecord, PageRecord
from project.text_utils import squash_ws


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


def enumerate_wordpress_urls(session: requests.Session) -> list[dict[str, str]]:
    discovered: list[dict[str, str]] = []
    pbar = tqdm(desc="wordpress", unit="url")
    for endpoint in WORDPRESS_TYPES:
        page = 1
        while True:
            url = (
                f"https://{MAIN_DOMAIN}/wp-json/wp/v2/{endpoint}"
                f"?per_page=100&page={page}&_fields=link,modified_gmt,modified,type"
            )
            response = session.get(url, timeout=30)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            for row in rows:
                link = canonicalize_url(row.get("link", ""))
                if not link:
                    continue
                discovered.append(
                    {
                        "url": link,
                        "source_type": row.get("type", endpoint),
                        "updated_at": row.get("modified_gmt") or row.get("modified"),
                    }
                )
                pbar.update(1)
            total_pages = int(response.headers.get("X-WP-TotalPages", "1"))
            if page >= total_pages:
                break
            page += 1
    pbar.close()
    return discovered


def spider_domain(
    session: requests.Session,
    seeds: list[str],
    source_type: str = "spider",
    depth: int = 3,
    max_urls: int = 3000,
) -> list[dict[str, str]]:
    """General-purpose spider that follows all links within ALLOWED_DOMAINS."""
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
        response = _fetch_with_backoff(session, url)
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


def enumerate_www2_urls(
    session: requests.Session, depth: int = 3, max_urls: int = 3000
) -> list[dict[str, str]]:
    seen: set[str] = set()
    discovered: list[dict[str, str]] = []
    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in WWW2_SEEDS)
    pbar = tqdm(desc="www2", unit="url")
    while queue:
        if len(discovered) >= max_urls:
            break
        url, level = queue.popleft()
        url = canonicalize_url(url) or url
        if url in seen:
            continue
        seen.add(url)
        discovered.append({"url": url, "source_type": "legacy_html", "updated_at": None})
        pbar.update(1)
        if level >= depth:
            continue
        response = _fetch_with_backoff(session, url)
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


def clean_page(url: str, html_text: str, source_type: str, updated_at: str | None) -> PageRecord | None:
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

    headings: list[str] = []
    section_lines: list[str] = []
    for tag in root.find_all(list(BLOCK_TAGS)):
        if has_block_ancestor(tag, root):
            continue
        text = squash_ws(tag.get_text(" ", strip=True))
        if not text:
            continue
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            headings.append(text)
            section_lines.append(f"## {text}")
        else:
            if section_lines and section_lines[-1] == text:
                continue
            section_lines.append(text)

    if table_rows:
        section_lines.append("## Table Data")
        section_lines.extend(table_rows)

    cleaned_text = "\n".join(section_lines).strip()
    raw_length = len(html_text)
    if len(cleaned_text) < 60 and not table_rows:
        return None
    page_id = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    language = soup.html.get("lang") if soup.html else None
    return PageRecord(
        page_id=page_id,
        url=url,
        title=title,
        source_type=source_type,
        updated_at=updated_at,
        headings=headings,
        text=cleaned_text,
        table_rows=table_rows,
        raw_length=raw_length,
        language=language,
    )


def split_sections(text: str, fallback_heading: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = fallback_heading
    current_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, " ".join(current_lines)))
                current_lines = []
            current_heading = line[3:].strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, " ".join(current_lines)))
    return sections


def chunk_words(words: list[str]) -> Iterable[str]:
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + CHUNK_WORDS)
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_chunks(page: PageRecord) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    sections = split_sections(page.text, page.title)
    chunk_idx = 0
    for section_idx, (heading, body) in enumerate(sections):
        words = body.split()
        for local_idx, chunk_text in enumerate(chunk_words(words)):
            chunk_idx += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{page.page_id}-s{section_idx}-c{local_idx}",
                    page_id=page.page_id,
                    url=page.url,
                    title=page.title,
                    heading=heading,
                    text=chunk_text,
                    is_table_row=False,
                )
            )
    for row_idx, row in enumerate(page.table_rows):
        chunks.append(
            ChunkRecord(
                chunk_id=f"{page.page_id}-t{row_idx}",
                page_id=page.page_id,
                url=page.url,
                title=page.title,
                heading="Table Data",
                text=row,
                is_table_row=True,
            )
        )
    return chunks


def _fetch_with_backoff(
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


def _fetch_and_clean(row: dict[str, str], session: requests.Session) -> tuple[PageRecord | None, list[ChunkRecord]]:
    url = row["url"]
    response = _fetch_with_backoff(session, url)
    if response is None:
        return None, []
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None, []
    page = clean_page(
        url=url,
        html_text=response.text,
        source_type=row["source_type"],
        updated_at=row["updated_at"],
    )
    if page is None:
        return None, []
    return page, build_chunks(page)


def crawl_pages(
    url_rows: list[dict[str, str]], workers: int = 16
) -> tuple[list[PageRecord], list[ChunkRecord]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in url_rows:
        if row["url"] not in seen:
            seen.add(row["url"])
            unique_rows.append(row)

    pages: list[PageRecord] = []
    chunks: list[ChunkRecord] = []
    session = build_session()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_and_clean, row, session): row for row in unique_rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="crawling", unit="page"):
            page, page_chunks = future.result()
            if page is not None:
                pages.append(page)
                chunks.extend(page_chunks)
    return pages, chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EECS retrieval corpus.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel download threads.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    wordpress_rows = enumerate_wordpress_urls(session)
    main_spider_seeds = [f"https://{MAIN_DOMAIN}/"]
    main_spider_rows = spider_domain(session, main_spider_seeds, source_type="main_spider")
    legacy_rows = enumerate_www2_urls(session)
    merged: dict[str, dict[str, str]] = {}
    for row in wordpress_rows + main_spider_rows + legacy_rows:
        merged.setdefault(row["url"], row)
    url_rows = list(merged.values())
    url_rows.sort(key=lambda row: row["url"])
    if args.limit > 0:
        url_rows = url_rows[: args.limit]
    print(f"Crawling {len(url_rows)} URLs", flush=True)

    pages, chunks = crawl_pages(url_rows, workers=args.workers)
    print(f"Built {len(pages)} pages and {len(chunks)} chunks", flush=True)
    write_jsonl(PAGES_PATH, (page.to_dict() for page in pages))
    write_jsonl(CHUNKS_PATH, (chunk.to_dict() for chunk in chunks))
    write_json(
        CORPUS_DIR / "manifest.json",
        {
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "url_count": len(url_rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
