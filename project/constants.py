from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QA_DIR = DATA_DIR / "qa"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_GENERATED_DIR = REPORT_DIR / "generated"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

PAGES_PATH = CORPUS_DIR / "pages.jsonl"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
CORPUS_PATH = ROOT_DIR / "eecs_text_bs_rewritten.jsonl"
DATASET_PATH = QA_DIR / "local_dev.jsonl"
HOLDOUT_PATH = QA_DIR / "local_holdout.jsonl"
IAA_SUBSET_PATH = QA_DIR / "iaa_subset.jsonl"
IAA_TEMPLATE_PATH = QA_DIR / "iaa_subset_template.jsonl"

PAGE_EMBEDDINGS_PATH = ARTIFACT_DIR / "page_embeddings.npy"
RUNTIME_CONFIG_PATH = ARTIFACT_DIR / "runtime_config.json"

DENSE_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_ENCODER_DIR = MODEL_DIR / "dense_encoder"

LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
LLM_MAX_TOKENS = 48
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT = 20
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARS = 12000

LLM_SYSTEM_PROMPT = """You are a concise question-answering assistant.
Answer the question using ONLY the provided context.
Give the shortest possible answer — a name, number, date, or short phrase (under 10 words).
Do NOT repeat the question or add explanation.
If the answer is a yes/no question, respond with only 'Yes' or 'No'.
If the answer cannot be found in the context, respond with exactly 'UNKNOWN'.
Format the time of day like '7:00pm' and expand any day of the week abbreviations, like 'M' should be 'Monday'."""

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
MAIN_DOMAIN = "eecs.berkeley.edu"
LEGACY_DOMAIN = "www2.eecs.berkeley.edu"
ALLOWED_DOMAINS = {MAIN_DOMAIN, LEGACY_DOMAIN}

WORDPRESS_TYPES = [
    "pages",
    "posts",
    "news_post",
    "media_mention",
    "book",
    "wall_display",
]

WWW2_SEEDS = [
    "https://www2.eecs.berkeley.edu/Directories/directory-nostudents.html",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Research/Areas/",
    "https://www2.eecs.berkeley.edu/Research/Areas/Centers/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "https://www2.eecs.berkeley.edu/Pubs/Dissertations/",
    "https://www2.eecs.berkeley.edu/Faculty/Awards/",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/faculty.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/CS/faculty.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/EE/faculty.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/list.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/new.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/teaching.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/visiting.html",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/women.html",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule.html",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule.html",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
]

BLOCK_TAGS = {
    "p",
    "li",
    "dd",
    "dt",
    "blockquote",
    "pre",
    "code",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "button",
    "input",
    "select",
    "textarea",
    "form",
    "nav",
    "footer",
    "aside",
}

CHUNK_WORDS = 130
CHUNK_OVERLAP = 40

PAGE_TOP_K = 10

MAX_ANSWER_WORDS = 10

DEFAULT_NULL_MARGIN = 0.0
DEFAULT_MIN_PAGE_SCORE = 0.04
