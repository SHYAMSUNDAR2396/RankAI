# Candidate Ranking System

Ranks a set of candidates against a single job description using a locally hosted LLM (Ollama), local sentence-transformers embeddings, and an on-disk ChromaDB vector store. It produces a ranked CSV of every candidate alongside a counterfactual bias audit report that flags scoring sensitivity to demographic signals. After the initial model downloads it runs entirely free and offline, with no API keys and no paid services.

## Prerequisites

- **Python 3.11+** (the package targets `requires-python >= 3.11`).
- **Ollama** installed and runnable locally.
- All required **Ollama models** pulled (see Installation).

On first use, `sentence-transformers` downloads the embedding model **`BAAI/bge-large-en-v1.5`** (~1.3GB). This is the only network access the pipeline needs, and only once.

## Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull all required models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b      # Skip if on low-RAM machine, set LOW_MEMORY_MODE=True in config.py
ollama pull deepseek-r1:7b

# On Intel/Apple 16GB machines: set LOW_MEMORY_MODE = True in config.py
# This automatically substitutes lighter models for the heavy ones.

# 3. (Recommended) create and activate a virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 4. Install the Python dependencies
pip install -r requirements.txt

# 5. Download the spaCy model used for the name/email fallback
python -m spacy download en_core_web_sm
```

## Running the pipeline

Start the Ollama server in a separate terminal if it is not already running:

```bash
ollama serve
```

Run with the bundled sample data (defaults shown):

```bash
# Equivalent to:
#   --candidates-dir ./data/candidates/
#   --job-description ./data/sample_job_description.json
#   --output-dir ./output/
python main.py
```

Run with your own candidates directory and job description:

```bash
python main.py --candidates-dir ./my_resumes --job-description ./my_jd.txt --output-dir ./out
```

Run faster by skipping the counterfactual fairness audit:

```bash
python main.py --skip-audit
```

Enable verbose/DEBUG logging (LLM request/response detail, per-phase timing):

```bash
python main.py --verbose
```

### Example output

When scoring finishes, the CLI prints a rich table of the top 10 candidates (or all of them when fewer than 10 were scored) followed by a summary panel. The values below are illustrative:

```
                         Top 10 Candidates
┏━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Rank ┃ Name             ┃ Composite ┃ Verdict    ┃ Human Review ┃ Bias Flag ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│    1 │ Sofia Ramirez    │      7.92 │ strong_yes │ No           │ Clean     │
│    2 │ David Okoye      │      7.41 │ yes        │ No           │ Clean     │
│    3 │ Aisha Khan       │      6.88 │ yes        │ Yes          │ ⚠ FLAG    │
│    4 │ Mei Tanaka       │      6.10 │ maybe      │ No           │ Clean     │
│    5 │ Raj Patel        │      5.55 │ maybe      │ No           │ Clean     │
│  ... │ ...              │       ... │ ...        │ ...          │ ...       │
└──────┴──────────────────┴───────────┴────────────┴──────────────┴───────────┘
╭───────────────────────────── Summary ──────────────────────────────╮
│ Run complete                                                        │
│ Candidates scored: 15                                               │
│ Elapsed: 312.84s                                                    │
│ Ranked CSV: output/ranked_candidates.csv                            │
│ Bias audit report: output/bias_audit_report.json                    │
│ Output directory: output                                            │
╰─────────────────────────────────────────────────────────────────────╯
```

## Interactive Web Dashboard

In addition to the CLI, RankAI includes a fully interactive web dashboard built with a **FastAPI backend** and a **Vite + React + Tailwind CSS v4 frontend** matching the design guidelines.

The web app allows you to:
- **Explore Rankings**: Review the candidate list with live search, filtering, and composite/persona score bars.
- **Analyze Candidate Details**: View individual candidate profile breakdowns, LLM-generated narratives, strengths/concerns, and consensus verdict actions.
- **Audit Fairness**: Inspect the counterfactual bias audit logs with delta comparison charts.
- **Run New Pipelines**: Use the interactive 3-step setup wizard to upload job descriptions and resumes (via drag-and-drop), and trigger ranking runs.

### Running the Dashboard

Ensure your virtual environment is active.

#### 1. Start the Backend API Server
From the project root directory, run:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
The interactive API documentation is available at `http://localhost:8000/docs`.

#### 2. Start the Frontend Dev Server
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
The web dashboard is served at `http://localhost:5173/`. Requests to `/api/*` are proxied to the backend API automatically.

## Running tests

```bash
pytest
```

The suite mocks every Ollama call (via `unittest.mock.patch` on `ollama.chat`) and runs fully offline and deterministically, so no Ollama server or network access is required. Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).

## Output files

Both artifacts are written to the `--output-dir` (default `./output/`).

### `ranked_candidates.csv`

One header row plus exactly one row per scored candidate, with columns in this order:

| Column | Description |
| --- | --- |
| `rank` | Consecutive 1-based rank in descending composite-score order (ties broken by `candidate_id` ascending). |
| `candidate_id` | Unique candidate identifier (`uuid4`, or `cf_<id>` for a counterfactual twin). |
| `name` | Candidate name (defaults to `Unknown Candidate` when unresolved). |
| `composite_score` | Weighted panel score on a 0–10 scale, rounded to 2 decimals. |
| `trajectory_score` | The LLM-derived seniority score (0–10) from the trajectory vector. |
| `hiring_manager_score` | Hiring-manager persona score (0–10). |
| `peer_interviewer_score` | Peer-interviewer persona score (0–10). |
| `devils_advocate_score` | Devil's-advocate persona score (0–10); subtracted in the composite. |
| `panel_variance` | Population variance across the three persona scores. |
| `requires_human_review` | `True` when `panel_variance` exceeds 2.5. |
| `verdict_consensus` | The verdict held by at least two personas, else the hiring-manager's verdict. |
| `strengths` | Pipe-separated list of de-duplicated strengths (empty list → empty string). |
| `concerns` | Pipe-separated list of de-duplicated concerns (empty list → empty string). |
| `narrative` | A three-sentence summary of the evaluation. |
| `bias_flag` | `True` when the counterfactual delta exceeds 0.75. |
| `counterfactual_delta` | Absolute difference between the candidate's and twin's composite scores (rounded to 2 decimals); empty when the audit was skipped or failed. |

### `bias_audit_report.json`

| Field | Description |
| --- | --- |
| `total_candidates_audited` | Number of candidates whose twin was successfully re-scored. |
| `flagged_count` | Number of audited candidates with `bias_flag = true`. |
| `flag_rate` | `flagged_count / total_candidates_audited` (0 when nothing was audited). |
| `bias_flag_threshold` | The delta threshold above which a candidate is flagged (`0.75`). |
| `flagged_candidates` | Details for each flagged candidate (id, name, delta, original/cf scores). |
| `clean_candidates_count` | Audited candidates that were not flagged. |
| `methodology_note` | How twins are built and the caveat that a clean result is not proof of no bias. |
| `audit_failures` | Entries for candidates whose twin could not be re-parsed/re-enriched/re-scored. |
| `audit_skipped` | Present and `true` only when run with `--skip-audit`. |

## Using your own data

### Candidates (`--candidates-dir`)

Point `--candidates-dir` at a directory of resume files. Supported extensions:

- **`.json`** — loaded directly as a `CandidateProfile` (no LLM call).
- **`.pdf` / `.docx`** — text is extracted (PyMuPDF / python-docx) and parsed into a `CandidateProfile` by the LLM.

If the directory is missing or contains no candidate files, the pipeline falls back to `data/sample_candidates.json`.

A candidate `.json` file matches the `CandidateProfile` schema:

```json
{
  "candidate_id": "7f3e9a2c-1b4d-4c8e-9f6a-2d5b8e1c3a47",
  "name": "Sofia Ramirez",
  "email": "sofia.ramirez@example.com",
  "years_experience": 8.0,
  "roles": [
    {
      "title": "Senior Software Engineer",
      "company": "Cadence Payments",
      "company_size_estimate": "scaleup 50-500",
      "start_date": "2019-07",
      "end_date": "2022-06",
      "duration_months": 35,
      "scope_keywords": ["distributed systems", "system design", "lead", "mentor"]
    }
  ],
  "skills_claimed": ["Python", "distributed systems", "SQL", "REST APIs"],
  "education": [
    {"institution": "Stanford University", "degree": "B.S. Computer Science", "year": 2016}
  ],
  "raw_text": "Sofia Ramirez is a senior software engineer with eight years...",
  "source_file": "sofia_ramirez.json"
}
```

Only `candidate_id` is strictly required; every other field has a sensible default, so partial profiles still validate.

### Job description (`--job-description`)

- **`.json`** — loaded directly as a `JobDescription` (no LLM call).
- **`.txt`** — classified into requirement buckets and dimensions by the LLM.

A job description `.json` file matches the `JobDescription` schema:

```json
{
  "job_id": "6f9619ff-8b86-5d11-b42d-00cf4fc964ff",
  "title": "Senior Software Engineer",
  "company": "Northwind Systems",
  "raw_text": "Northwind Systems is a Series B SaaS company hiring a Senior Software Engineer...",
  "requirements": [
    {"text": "4+ years of hands-on Python development", "bucket": "must_have", "dimension": "technical"},
    {"text": "Experience with Kubernetes", "bucket": "nice_to_have", "dimension": "technical"},
    {"text": "Thrives in an async-first environment", "bucket": "culture_signal", "dimension": "soft_skill"},
    {"text": "Experience mentoring junior engineers", "bucket": "seniority_marker", "dimension": "soft_skill"}
  ]
}
```

Each requirement `bucket` is one of `must_have`, `nice_to_have`, `culture_signal`, `seniority_marker`; each `dimension` is one of `technical`, `soft_skill`, `domain`, `experience_level`.

## Performance note

Everything runs on CPU with `llama3.2:3b`, so runtime is dominated by the number of LLM calls. For the bundled 15-candidate dataset:

- **Scoring** makes ~4 LLM calls per candidate (3 evaluator personas + 1 narrative), plus 1 seniority call per candidate during enrichment.
- **The audit** roughly doubles the LLM work, because it re-scores a counterfactual twin for each candidate.

On a CPU-only machine this lands on the order of several minutes for 15 candidates — a realistic ballpark is **~3–10 minutes** depending on hardware. Running with `--skip-audit` cuts the LLM work roughly in half. Remember that the embedding model (`BAAI/bge-large-en-v1.5`, ~1.3GB) downloads on the first run.

## Methodology

See [methodology.md](./methodology.md) for the detailed methodology behind ingestion, trajectory enrichment, embedding and RAG, multi-persona scoring, the composite/variance math, the counterfactual audit, and reproducibility.

## Alternative backend: Groq free tier

The system is designed so the only swap seam for the LLM is the `OllamaClient` call implementation. Per Requirement 1.11, you can switch to the [Groq](https://groq.com/) free tier with `llama-3.1-70b` by changing **only** the `OllamaClient` LLM call implementation and the dependency manifest (`requirements.txt`). The Resume Parser, JD Parser, TrajectoryEnricher, VectorStoreManager, CandidateScoringPipeline, CounterfactualAuditor, and CLI all stay unchanged.
