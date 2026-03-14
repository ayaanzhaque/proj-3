#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from rag import RAGModel


def read_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_answers(path: Path, answers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write((answer or "").replace("\n", " ").strip() + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("Usage: python3 predict.py <questions_txt_path> <predictions_out_path>")
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    model = RAGModel()
    answers = model.predict(read_questions(input_path))
    write_answers(output_path, answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

