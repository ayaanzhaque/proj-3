#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

from rag.constants import DATASET_PATH, HOLDOUT_PATH, IAA_SUBSET_PATH, IAA_TEMPLATE_PATH, PAGES_PATH
from rag.io_utils import read_jsonl, write_jsonl
from rag.schema import QAExample
from rag.text_utils import squash_ws


def parse_kv_row(row: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in row.split("|"):
        part = squash_ws(part)
        if ": " in part:
            key, value = part.split(": ", 1)
            fields[squash_ws(key).lower()] = squash_ws(value)
    return fields


def question_id(question: str, evidence_url: str) -> str:
    return hashlib.md5(f"{question}|{evidence_url}".encode("utf-8")).hexdigest()[:12]


def make_example(
    *,
    question: str,
    answers: str,
    evidence_url: str,
    evidence_text: str,
    answer_type: str,
    category: str,
    annotator_id: str = "assistant_v1",
    valid_as_of: str = "2026-03-14",
) -> QAExample:
    return QAExample(
        question_id=question_id(question, evidence_url),
        question=question,
        answers=answers,
        evidence_url=evidence_url,
        evidence_text=squash_ws(evidence_text),
        answer_type=answer_type,
        category=category,
        annotator_id=annotator_id,
        valid_as_of=valid_as_of,
    )


def generate_from_tables(pages: list[dict]) -> list[QAExample]:
    examples: list[QAExample] = []
    for page in pages:
        url = page["url"]
        title = page["title"]
        for row in page.get("table_rows", []):
            fields = parse_kv_row(row)
            name = fields.get("name")
            email = fields.get("email")
            office = fields.get("office") or fields.get("location") or fields.get("room")
            phone = fields.get("phone")
            role = fields.get("title") or fields.get("position")
            full_name = fields.get("full name")
            graduation = fields.get("semester of graduation")
            breadth = fields.get("breadth area")
            course_num = fields.get("course number") or fields.get("number")
            course_title = fields.get("course title") or fields.get("course")
            subject = fields.get("subject")
            date = fields.get("date")
            proceeding = fields.get("proceeding")
            room = fields.get("room name/number")
            capacity = fields.get("cap.")
            forms = fields.get("form(s)")
            description = fields.get("description")

            if name and email and "@" in email:
                examples.append(
                    make_example(
                        question=f"What is the email address of {name}?",
                        answers=email,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="directory",
                    )
                )
            if name and office:
                examples.append(
                    make_example(
                        question=f"What is the office location of {name}?",
                        answers=office,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="directory",
                    )
                )
            if name and phone:
                examples.append(
                    make_example(
                        question=f"What is the phone number of {name}?",
                        answers=phone,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="directory",
                    )
                )
            if name and role:
                examples.append(
                    make_example(
                        question=f"What is the title of {name}?",
                        answers=role,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="directory",
                    )
                )
            if full_name and graduation:
                examples.append(
                    make_example(
                        question=f"When is {full_name} expected to graduate?",
                        answers=graduation,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="students",
                    )
                )
            if full_name and breadth:
                examples.append(
                    make_example(
                        question=f"What is the breadth area of {full_name}?",
                        answers=breadth,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="students",
                    )
                )
            if course_num and course_title and any(ch.isdigit() for ch in course_num):
                examples.append(
                    make_example(
                        question=f"What is the title of {course_num}?",
                        answers=course_title,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="courses",
                    )
                )
            if subject and course_title:
                examples.append(
                    make_example(
                        question=f"Which {subject} course is titled {course_title}?",
                        answers=course_num or subject,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="courses",
                    )
                )
            if date and proceeding:
                examples.append(
                    make_example(
                        question=f"When is {proceeding}?",
                        answers=date,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="deadlines",
                    )
                )
                examples.append(
                    make_example(
                        question=f"What happens on {date} in {title.replace(' - EECS at Berkeley', '')}?",
                        answers=proceeding,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="deadlines",
                    )
                )
            if room and capacity:
                examples.append(
                    make_example(
                        question=f"What is the capacity of {room}?",
                        answers=capacity,
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="facilities",
                    )
                )
            if room:
                email_match = re.search(r"([A-Za-z0-9._%+-]+@(?:[A-Za-z0-9.-]+))", row)
                if email_match:
                    examples.append(
                        make_example(
                            question=f"Which email is mentioned for {room}?",
                            answers=email_match.group(1),
                            evidence_url=url,
                            evidence_text=row,
                            answer_type="extractive",
                            category="facilities",
                        )
                    )
            if forms and description:
                description_lower = description.lower()
                if "retroactive change in class schedule" in description_lower:
                    examples.append(
                        make_example(
                            question="Which form is used to request a retroactive change in class schedule?",
                            answers=forms,
                            evidence_url=url,
                            evidence_text=row,
                            answer_type="extractive",
                            category="forms",
                        )
                    )
                if "qualifying exam" in description_lower:
                    examples.append(
                        make_example(
                            question="Which form should be completed to request to take the qualifying exam?",
                            answers=forms,
                            evidence_url=url,
                            evidence_text=row,
                            answer_type="extractive",
                            category="forms",
                        )
                    )
            if fields.get("winner") and fields.get("year"):
                examples.append(
                    make_example(
                        question=f"Who is the winner listed for {fields['year']}?",
                        answers=fields["winner"],
                        evidence_url=url,
                        evidence_text=row,
                        answer_type="extractive",
                        category="awards",
                    )
                )
            if not fields and " | " in row:
                parts = [squash_ws(part) for part in row.split("|") if squash_ws(part)]
                if len(parts) == 2 and any(ch.isdigit() for ch in parts[0]) and len(parts[1].split()) <= 12:
                    examples.append(
                        make_example(
                            question=f"What is the title of {parts[0]}?",
                            answers=parts[1],
                            evidence_url=url,
                            evidence_text=row,
                            answer_type="extractive",
                            category="courses",
                        )
                    )
                elif (
                    len(parts) == 2
                    and len(parts[1].split()) <= 6
                    and ("By the Numbers" in title or "Milestones" in title)
                ):
                    examples.append(
                        make_example(
                            question=f"What number is listed for {parts[0]} on the {title.replace(' - EECS at Berkeley', '')} page?",
                            answers=parts[1],
                            evidence_url=url,
                            evidence_text=row,
                            answer_type="extractive",
                            category="stats",
                        )
                    )
    return examples


CURATED_EXAMPLES = [
    {
        "question": "How many recommendation letters are suggested for a graduate application to UC Berkeley?",
        "answers": "three",
        "evidence_url": "https://eecs.berkeley.edu/academics/graduate/faq-3/",
        "evidence_text": "Contact three individuals, preferably professors, to write letters of recommendation.",
        "answer_type": "extractive",
        "category": "admissions",
    },
    {
        "question": "What should be included to aid recommenders writing graduate application letters?",
        "answers": "transcripts",
        "evidence_url": "https://eecs.berkeley.edu/academics/graduate/faq-3/",
        "evidence_text": "You may want to prepare packages for them that include copies of work you have done with them, transcripts, and a resume/CV.",
        "answer_type": "extractive",
        "category": "admissions",
    },
    {
        "question": "Which award was Eric Paulos elected to in March 2026?",
        "answers": "ACM CHI Academy",
        "evidence_url": "https://eecs.berkeley.edu/news/eric-paulos-elected-to-the-acm-chi-academy/",
        "evidence_text": "Eric Paulos, Professor of Electrical Engineering and Computer Sciences, to the ACM CHI Academy.",
        "answer_type": "extractive",
        "category": "news",
    },
    {
        "question": "How many individuals were inducted to the CHI Academy that year according to the Eric Paulos news post?",
        "answers": "eleven|11",
        "evidence_url": "https://eecs.berkeley.edu/news/eric-paulos-elected-to-the-acm-chi-academy/",
        "evidence_text": "Paulos is one of eleven individuals inducted to the Academy this year.",
        "answer_type": "abstractive",
        "category": "news",
    },
    {
        "question": "Which page contains the EECS Faculty and Staff Directory?",
        "answers": "directory-nostudents.html",
        "evidence_url": "https://www2.eecs.berkeley.edu/Directories/directory-nostudents.html",
        "evidence_text": "EECS Faculty and Staff Directory",
        "answer_type": "extractive",
        "category": "directory",
    },
    {
        "question": "Which role does Diane Greene hold at Google in the 2016 alumni spotlight?",
        "answers": "Senior Vice President",
        "evidence_url": "https://eecs.berkeley.edu/2016/08/alumni-spotlight-diane-greene/",
        "evidence_text": "Diane was a founder and CEO of VMware in 1998 and is currently the Senior Vice President of (cloud) Enterprise Business at Google.",
        "answer_type": "extractive",
        "category": "alumni",
    },
    {
        "question": "Which field did Virginia Smith study as an undergraduate?",
        "answers": "math",
        "evidence_url": "https://eecs.berkeley.edu/2016/09/top-women-in-tech-virginia-smith/",
        "evidence_text": "As an undergraduate, she studied math, a field where there are almost equal numbers of men and women.",
        "answer_type": "extractive",
        "category": "people",
    },
]


def deduplicate(examples: list[QAExample]) -> list[QAExample]:
    seen: set[tuple[str, str]] = set()
    deduped: list[QAExample] = []
    for example in examples:
        key = (example.question, example.evidence_url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a local QA dataset from the corpus.")
    parser.add_argument("--target-size", type=int, default=150)
    parser.add_argument("--holdout-size", type=int, default=25)
    parser.add_argument("--iaa-size", type=int, default=45)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    pages = read_jsonl(PAGES_PATH)
    if not pages:
        raise SystemExit("Corpus not found or empty. Run rag/corpus/build_corpus.py first.")
    generated_examples = deduplicate(generate_from_tables(pages))
    curated_examples = [make_example(**row) for row in CURATED_EXAMPLES]
    generated_examples = [
        example
        for example in generated_examples
        if (example.question, example.evidence_url)
        not in {(item.question, item.evidence_url) for item in curated_examples}
    ]
    random.Random(args.seed).shuffle(generated_examples)
    examples = deduplicate(curated_examples + generated_examples)
    examples = examples[: args.target_size]

    if len(examples) <= args.holdout_size:
        raise SystemExit("Not enough examples generated. Build the corpus first or add more structured pages.")

    holdout = examples[: args.holdout_size]
    dev = examples[args.holdout_size :]
    iaa_subset = dev[: min(args.iaa_size, len(dev))]

    write_jsonl(DATASET_PATH, (example.to_dict() for example in dev))
    write_jsonl(HOLDOUT_PATH, (example.to_dict() for example in holdout))
    write_jsonl(IAA_TEMPLATE_PATH, (example.to_dict() for example in iaa_subset))
    if not IAA_SUBSET_PATH.exists():
        write_jsonl(IAA_SUBSET_PATH, (example.to_dict() for example in iaa_subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
