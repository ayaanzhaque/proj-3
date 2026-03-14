from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageRecord:
    page_id: str
    url: str
    title: str
    source_type: str
    updated_at: str | None
    headings: list[str]
    text: str
    table_rows: list[str] = field(default_factory=list)
    raw_length: int = 0
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    page_id: str
    url: str
    title: str
    heading: str
    text: str
    is_table_row: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAExample:
    question_id: str
    question: str
    answers: str
    evidence_url: str
    evidence_text: str
    answer_type: str
    category: str
    annotator_id: str
    valid_as_of: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

