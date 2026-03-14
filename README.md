# EECS Berkeley RAG Project

This repository contains a fully local retrieval-augmented question answering system for the UC Berkeley EECS website. It is designed to match the assignment interface:

```bash
bash run.sh <questions_txt_path> <predictions_out_path>
```

At runtime, the system:

1. Loads a prebuilt EECS corpus and chunk index.
2. Retrieves relevant pages/chunks for each question.
3. Answers with a combination of:
   - rule-based extraction for structured table-like rows
   - a lightweight local extractive QA model for free-text passages
4. Writes one final answer per input line.

The code is built to run locally without network calls during inference.

## 1. Current Project Status

This section is a snapshot of what is already built in this repo.

- Corpus build status:
  - `1958` cleaned EECS pages
  - `9684` text chunks
- Local models already downloaded and materialized:
  - dense encoder: `sentence-transformers/all-MiniLM-L6-v2`
  - QA model: `deepset/minilm-uncased-squad2`
- Local QA dataset already generated:
  - `125` dev examples
  - `25` holdout examples
  - `45` IAA subset examples
- Runtime benchmark already measured:
  - `100` questions in about `12.44s`
  - about `0.124s/question`
  - peak RSS about `1.17 GB`

Important caveats:

- The code supports crawling legacy `www2.eecs.berkeley.edu` pages, but during the full crawl that host was not reachable from this machine. The current built corpus is therefore effectively main-domain only.
- The QA dataset is currently a bootstrapped local dataset generated from structured pages plus a small curated seed. It is useful for local iteration, but it still needs manual cleanup and real second-human IAA work before final submission quality is reached.
- The local holdout score is only a sanity check. It is not a reliable estimate of hidden dev/test performance.

## 2. High-Level Architecture

The pipeline is split into an **offline build phase** and an **online inference phase**.

### Offline build phase

The offline phase is where all expensive or network-dependent work happens.

1. Discover EECS URLs.
   - Use the WordPress REST API on `eecs.berkeley.edu`.
   - Seed a small crawl for legacy `www2.eecs.berkeley.edu` paths.
2. Download HTML pages.
3. Clean each page.
   - Remove navigation, scripts, forms, repeated chrome.
   - Keep headings, prose, and tables.
4. Convert cleaned pages into chunks.
5. Download local transformer models.
6. Build dense page embeddings.
7. Generate a local QA dataset scaffold.
8. Generate simple corpus/dataset stats for report support.

### Online inference phase

This is what happens when the autograder runs `bash run.sh ...`.

1. Load `RAGModel`.
2. Load:
   - cleaned pages
   - cleaned chunks
   - dense page embeddings
   - dense encoder
   - extractive QA model
3. For each batch of questions:
   - retrieve top candidate pages using BM25 + dense similarity + reciprocal-rank fusion
   - retrieve top chunks inside those pages
   - answer structured questions using deterministic rules when possible
   - otherwise run extractive QA over retrieved chunks
4. Return one short text answer per question.

## 3. Assignment Task Mapping

This maps the assignment requirements to the code and artifacts in the repo.

### Task A: Build a retrieval corpus

Implemented in:

- [`scripts/build_corpus.py`](./scripts/build_corpus.py)
- [`project/constants.py`](./project/constants.py)
- [`project/schema.py`](./project/schema.py)
- [`project/text_utils.py`](./project/text_utils.py)

Outputs:

- [`data/corpus/pages.jsonl`](./data/corpus/pages.jsonl)
- [`data/corpus/chunks.jsonl`](./data/corpus/chunks.jsonl)
- [`data/corpus/manifest.json`](./data/corpus/manifest.json)

### Task B: Create a QA dataset

Implemented in:

- [`scripts/bootstrap_dataset.py`](./scripts/bootstrap_dataset.py)

Outputs:

- [`data/qa/local_dev.jsonl`](./data/qa/local_dev.jsonl)
- [`data/qa/local_holdout.jsonl`](./data/qa/local_holdout.jsonl)
- [`data/qa/iaa_subset.jsonl`](./data/qa/iaa_subset.jsonl)
- [`data/qa/iaa_subset_template.jsonl`](./data/qa/iaa_subset_template.jsonl)

### Task C: Build the RAG system

Implemented in:

- [`rag.py`](./rag.py)
- [`project/runtime.py`](./project/runtime.py)
- [`project/modeling.py`](./project/modeling.py)

Artifacts used by runtime:

- [`artifacts/page_embeddings.npy`](./artifacts/page_embeddings.npy)
- [`artifacts/runtime_config.json`](./artifacts/runtime_config.json)
- [`artifacts/models/dense_encoder/`](./artifacts/models/dense_encoder)
- [`artifacts/models/qa_model/`](./artifacts/models/qa_model)

### Task D: Submission entrypoint

Implemented in:

- [`run.sh`](./run.sh)
- [`predict.py`](./predict.py)
- [`rag.py`](./rag.py)

### Task E: Evaluation, benchmarking, and report helpers

Implemented in:

- [`scripts/evaluate_dataset.py`](./scripts/evaluate_dataset.py)
- [`scripts/report_stats.py`](./scripts/report_stats.py)
- [`scripts/benchmark_submission.py`](./scripts/benchmark_submission.py)

Generated report helpers:

- [`reports/generated/corpus_stats.json`](./reports/generated/corpus_stats.json)
- [`reports/generated/dataset_stats.json`](./reports/generated/dataset_stats.json)

## 4. Repository Layout

### Top level

- [`run.sh`](./run.sh)
  - required submission entrypoint
  - accepts exactly two positional args
- [`predict.py`](./predict.py)
  - thin runner that loads `RAGModel`, reads questions, writes answers
- [`rag.py`](./rag.py)
  - assignment-facing model class
- [`requirements.txt`](./requirements.txt)
  - runtime dependency list
- [`requirements-offline.txt`](./requirements-offline.txt)
  - optional offline preprocessing helpers

### `project/`

- [`project/constants.py`](./project/constants.py)
  - shared paths
  - model names
  - retrieval hyperparameters
  - crawl seeds
- [`project/schema.py`](./project/schema.py)
  - dataclasses for pages, chunks, QA examples
- [`project/io_utils.py`](./project/io_utils.py)
  - JSON / JSONL helpers
- [`project/text_utils.py`](./project/text_utils.py)
  - normalization
  - tokenization
  - EM/F1 scoring helpers
  - retrieval helpers like RRF
- [`project/modeling.py`](./project/modeling.py)
  - dense encoder wrapper
  - QA model wrapper
- [`project/runtime.py`](./project/runtime.py)
  - retrieval logic
  - rule-based structured answering
  - extractive QA inference

### `scripts/`

- [`scripts/build_corpus.py`](./scripts/build_corpus.py)
  - discovers URLs
  - fetches pages
  - cleans HTML
  - extracts tables
  - builds chunks
- [`scripts/build_artifacts.py`](./scripts/build_artifacts.py)
  - downloads transformer models
  - builds dense page embeddings
  - writes runtime config
- [`scripts/bootstrap_dataset.py`](./scripts/bootstrap_dataset.py)
  - mines QA examples from structured rows
  - adds a small curated seed set
  - writes dev / holdout / IAA files
- [`scripts/evaluate_dataset.py`](./scripts/evaluate_dataset.py)
  - runs local EM and token F1 over a JSONL QA file
- [`scripts/report_stats.py`](./scripts/report_stats.py)
  - writes simple corpus and dataset stats to JSON
- [`scripts/benchmark_submission.py`](./scripts/benchmark_submission.py)
  - measures inference latency on a text file of questions

### `data/`

- [`data/corpus/pages.jsonl`](./data/corpus/pages.jsonl)
  - one cleaned page record per line
- [`data/corpus/chunks.jsonl`](./data/corpus/chunks.jsonl)
  - one chunk record per line
- [`data/qa/local_dev.jsonl`](./data/qa/local_dev.jsonl)
  - local dev set
- [`data/qa/local_holdout.jsonl`](./data/qa/local_holdout.jsonl)
  - local holdout set
- [`data/qa/iaa_subset_template.jsonl`](./data/qa/iaa_subset_template.jsonl)
  - partner-facing template to re-annotate
- [`data/qa/iaa_subset.jsonl`](./data/qa/iaa_subset.jsonl)
  - current copy of the IAA subset
- [`data/sample_questions.txt`](./data/sample_questions.txt)
  - sample input used for benchmarking
- [`data/sample_answers.txt`](./data/sample_answers.txt)
  - sample output from a prior run

### `artifacts/`

- [`artifacts/page_embeddings.npy`](./artifacts/page_embeddings.npy)
  - precomputed dense embeddings for every page
- [`artifacts/runtime_config.json`](./artifacts/runtime_config.json)
  - runtime thresholds
- [`artifacts/models/dense_encoder/`](./artifacts/models/dense_encoder)
  - local dense encoder files
- [`artifacts/models/qa_model/`](./artifacts/models/qa_model)
  - local QA model files

### `reports/generated/`

- [`reports/generated/corpus_stats.json`](./reports/generated/corpus_stats.json)
- [`reports/generated/dataset_stats.json`](./reports/generated/dataset_stats.json)

## 5. Detailed Pipeline Walkthrough

### 5.1 Corpus discovery

The main domain is discovered using the WordPress REST API:

- `pages`
- `posts`
- `news_post`
- `media_mention`
- `book`
- `wall_display`

This discovery is handled in [`scripts/build_corpus.py`](./scripts/build_corpus.py).

The legacy site is seeded from a fixed list of high-value paths:

- directory pages
- course pages
- research areas
- publications
- faculty lists

The crawler normalizes URLs and filters out obvious non-HTML assets like:

- PDFs
- images
- zip files
- Word docs

### 5.2 HTML cleaning

Each page is parsed with BeautifulSoup.

The cleaner:

- prefers `main`, `#site-main`, `article`, `.entry-content`, `.post-content`, then `body`
- removes:
  - `script`
  - `style`
  - `noscript`
  - `svg`
  - `iframe`
  - `form`
  - `nav`
  - `footer`
  - other chrome-like tags
- extracts top-level headings and text blocks
- extracts HTML tables into row strings

Table rows are intentionally preserved because many factual questions live in tables.

Example row format:

```text
Course Number: CS 194-16 | Course Title: Introduction to Data Science | Units: Variable units
```

### 5.3 Chunking

After cleaning, each page is split into sections by heading.

Each section is chunked using:

- `130` words per chunk
- `40` words overlap

In addition:

- each table row becomes its own chunk

This makes factual row-based questions much easier to answer.

### 5.4 Retrieval

Retrieval has two stages.

#### Stage 1: page retrieval

For every page, the retrieval text includes:

- title
- headings
- URL tokens
- cleaned page text
- table rows

Two scorers are used:

- BM25 sparse retrieval
- dense cosine similarity using `all-MiniLM-L6-v2`

The two rankings are fused with reciprocal-rank fusion.

Default runtime fanout:

- top `8` pages

#### Stage 2: chunk retrieval

Chunk retrieval only runs inside the selected top pages.

Each chunk gets:

- BM25 score
- heading lexical-overlap bonus
- title lexical-overlap bonus

Default runtime fanout:

- top `8` chunks

### 5.5 Answering

There are two answer paths.

#### Path A: rule-based structured answering

Before calling the QA model, the runtime tries to answer common structured question types directly from parsed rows.

Handled patterns include:

- `What is the title of <course>?`
- `Which <subject> course is titled <title>?`
- `What is the capacity of <room>?`
- `When is <student> expected to graduate?`
- `What is the breadth area of <student>?`
- `When is <proceeding>?`
- `What happens on <date> in <page>?`
- simple numeric stat questions
- selected form/email questions

This logic lives in [`project/runtime.py`](./project/runtime.py).

#### Path B: extractive QA

If no direct structured answer is found:

- the top chunks are paired with the question
- the QA model runs over `(question, chunk)` pairs
- the best context-only span is selected
- the answer is trimmed to a short final span

The QA model is:

- `deepset/minilm-uncased-squad2`

If confidence is weak, the system returns:

- `UNKNOWN`

### 5.6 Output formatting

The final output is always:

- one line per answer
- same order as the input questions
- no embedded newlines

This is handled by [`predict.py`](./predict.py).

## 6. Models Used

### Dense encoder

- model: `sentence-transformers/all-MiniLM-L6-v2`
- role: page-level dense retrieval
- implementation path:
  - model files stored locally
  - loaded using `transformers` in [`project/modeling.py`](./project/modeling.py)

### QA model

- model: `deepset/minilm-uncased-squad2`
- role: extractive short-answer selection from retrieved chunks

Why these choices were made:

- small enough for CPU-only runtime
- easy to package locally
- better fit than a generative 8B-style model under a 3 GB RAM constraint

## 7. Data File Formats

### `pages.jsonl`

Each line is a JSON object with fields such as:

- `page_id`
- `url`
- `title`
- `source_type`
- `updated_at`
- `headings`
- `text`
- `table_rows`
- `raw_length`
- `language`

### `chunks.jsonl`

Each line contains:

- `chunk_id`
- `page_id`
- `url`
- `title`
- `heading`
- `text`
- `is_table_row`

### QA JSONL files

Each example contains:

- `question_id`
- `question`
- `answers`
- `evidence_url`
- `evidence_text`
- `answer_type`
- `category`
- `annotator_id`
- `valid_as_of`

Multiple valid answers are pipe-separated, for example:

```text
eleven|11
```

## 8. How to Rebuild Everything

This is the clean rebuild sequence.

### Environment

Recommended local environment:

```bash
conda activate cs188
```

If needed, install packages:

```bash
pip install -r requirements.txt
pip install -r requirements-offline.txt
```

### Step 1: rebuild the corpus

```bash
python scripts/build_corpus.py
```

Useful options:

```bash
python scripts/build_corpus.py --limit 50
python scripts/build_corpus.py --delay-seconds 1.0
```

### Step 2: rebuild runtime artifacts

```bash
python scripts/build_artifacts.py
```

This will:

- ensure local model files exist
- build dense page embeddings
- write runtime thresholds

### Step 3: regenerate the local QA dataset scaffold

```bash
python scripts/bootstrap_dataset.py
```

### Step 4: regenerate report helper stats

```bash
python scripts/report_stats.py
```

### Step 5: run a local evaluation

```bash
python scripts/evaluate_dataset.py data/qa/local_holdout.jsonl
```

### Step 6: benchmark runtime

```bash
python scripts/benchmark_submission.py data/sample_questions.txt
```

### Step 7: run the submission interface exactly

```bash
bash run.sh data/sample_questions.txt data/sample_answers.txt
```

## 9. Submission Interface Details

The autograder-facing interface is:

```bash
bash run.sh <questions_txt_path> <predictions_out_path>
```

The code path is:

1. [`run.sh`](./run.sh)
2. [`predict.py`](./predict.py)
3. [`rag.py`](./rag.py)
4. [`project/runtime.py`](./project/runtime.py)

`run.sh` behavior:

- first tries plain `python3` if required packages are already visible
- otherwise falls back to `conda run -n cs188 python3 ...` for local convenience

The conda fallback is a local convenience feature. The important part for submission is still that the script accepts exactly two positional arguments and ultimately runs the Python predictor.

## 10. Local Testing That Has Already Been Done

The following checks were already run successfully:

- full corpus build
- artifact/model build
- dataset generation
- `python -m py_compile` over repo Python files
- end-to-end run through `bash run.sh ...`
- 100-question runtime benchmark

Measured runtime:

- about `12.44s` for `100` questions
- about `0.124s/question`

Measured memory:

- peak RSS about `1.17 GB`

Local holdout evaluation also ran successfully.

What has **not** been fully validated:

- hidden dev/test performance
- final report content
- full legacy `www2` coverage in the currently built corpus
- a true second-human IAA pass

## 11. Known Limitations

### Legacy `www2` site coverage

The code supports it, but the host was unreachable during the full crawl from this machine.

Consequence:

- current artifacts do not include legacy directory/course/dissertation pages
- questions whose evidence only lives on `www2` may underperform

### Dataset quality

The current local dataset is useful for engineering iteration but not yet polished enough to treat as the final report-quality QA set.

It still needs:

- manual spot-checking
- removal of weak or awkwardly phrased questions
- more deliberate category balancing
- real second-human annotation for IAA

### Structured questions dominate the current local set

Because the generator mines tables well, the current dataset has a bias toward:

- course titles
- room capacities
- deadlines
- honors-student rows

This is acceptable for a scaffold, but the final dataset should be more intentionally balanced.

## 12. Recommended Next Steps For Partner Handoff

If someone else picks this up, the most valuable next steps are:

1. Manually review and improve [`data/qa/local_dev.jsonl`](./data/qa/local_dev.jsonl) and [`data/qa/local_holdout.jsonl`](./data/qa/local_holdout.jsonl).
2. Re-annotate [`data/qa/iaa_subset_template.jsonl`](./data/qa/iaa_subset_template.jsonl) with a second human annotator.
3. If `www2.eecs.berkeley.edu` becomes reachable, rerun [`scripts/build_corpus.py`](./scripts/build_corpus.py) and then rebuild artifacts.
4. Run ablations by adjusting:
   - page/chunk top-k in [`project/constants.py`](./project/constants.py)
   - runtime thresholds in [`scripts/build_artifacts.py`](./scripts/build_artifacts.py)
   - chunking parameters in [`project/constants.py`](./project/constants.py)
5. Draft the report using:
   - [`reports/generated/corpus_stats.json`](./reports/generated/corpus_stats.json)
   - [`reports/generated/dataset_stats.json`](./reports/generated/dataset_stats.json)

## 13. Minimal Command Cheat Sheet

Rebuild everything:

```bash
conda activate cs188
python scripts/build_corpus.py
python scripts/build_artifacts.py
python scripts/bootstrap_dataset.py
python scripts/report_stats.py
```

Evaluate locally:

```bash
python scripts/evaluate_dataset.py data/qa/local_holdout.jsonl
```

Benchmark inference:

```bash
python scripts/benchmark_submission.py data/sample_questions.txt
```

Run the submission path:

```bash
bash run.sh data/sample_questions.txt data/sample_answers.txt
```

## 14. Bottom Line

This repo already contains a working local RAG system with:

- a full main-domain EECS corpus
- chunked retrieval data
- local transformer models
- a packaged submission interface
- dataset scaffolding
- evaluation and benchmark scripts

The biggest remaining work is not infrastructure. It is:

- improving the QA dataset manually
- adding true IAA
- recovering legacy `www2` coverage if possible
- writing the final report
