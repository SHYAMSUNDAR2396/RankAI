# RankAI — Architecture & Implementation Guide

> **Local-first, bias-audited AI candidate ranking system.**
> Uses Ollama (Llama 3.2), sentence-transformers, ChromaDB, and a React + FastAPI dashboard.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [File Structure](#file-structure)
3. [Backend Pipeline (Python)](#backend-pipeline-python)
4. [Data Models](#data-models)
5. [Pipeline Phases](#pipeline-phases)
6. [Web Dashboard (React + FastAPI)](#web-dashboard-react--fastapi)
7. [Technology Stack](#technology-stack)
8. [Data Flow](#data-flow)

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React Dashboard (Vite)                   │
│  ┌──────────┐ ┌───────────────┐ ┌──────────┐ ┌───────────┐  │
│  │Dashboard │ │Candidate      │ │Audit     │ │Pipeline   │  │
│  │Page      │ │Detail Page    │ │Report    │ │Setup      │  │
│  └──────────┘ └───────────────┘ └──────────┘ └───────────┘  │
│                    ▼ Vite Proxy /api → :8080                 │
├──────────────────────────────────────────────────────────────┤
│                   FastAPI Backend (server.py)                │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │GET /api/│ │POST /api/│ │POST /api/│ │GET /api/export/ │  │
│  │candidates│ │run       │ │upload/*  │ │csv              │  │
│  └─────────┘ └──────────┘ └──────────┘ └─────────────────┘  │
│                    ▼ Subprocess                              │
├──────────────────────────────────────────────────────────────┤
│              Python ML Pipeline (main.py)                    │
│  ┌────────┐ ┌────────┐ ┌───────┐ ┌───────┐ ┌─────────────┐  │
│  │INGEST  │→│ENRICH  │→│EMBED &│→│SCORE  │→│AUDIT &      │  │
│  │        │ │        │ │STORE  │ │       │ │OUTPUT       │  │
│  └────────┘ └────────┘ └───────┘ └───────┘ └─────────────┘  │
│       ▼          ▼          ▼          ▼                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
│  │Ollama  │ │Ollama  │ │ChromaDB│ │Ollama  │                │
│  │LLM     │ │LLM     │ │Vector  │ │LLM     │                │
│  │        │ │        │ │Store   │ │(3 pers)│                │
│  └────────┘ └────────┘ └────────┘ └────────┘                │
└──────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
RankAI/
├── main.py                        # Pipeline orchestrator & CLI entry point
├── config.py                      # Centralized configuration (single source of truth)
├── server.py                      # FastAPI backend serving the React dashboard
├── pyproject.toml                 # Python project metadata & dependencies
├── requirements.txt               # Pinned dependency manifest
├── methodology.md                 # Scoring methodology documentation
├── README.md                      # Project overview & run instructions
├── ARCHITECTURE.md                # ← You are here
│
├── models/                        # Pydantic v2 data models
│   ├── __init__.py
│   ├── candidate.py               # CandidateProfile, CandidateRole, TrajectoryVector
│   └── job.py                     # JobDescription, JobRequirement
│
├── pipeline/                      # Core pipeline phases
│   ├── __init__.py
│   ├── ingest.py                  # ResumeParser (PDF/DOCX/JSON) & JdParser (JSON/TXT)
│   ├── enrich.py                  # TrajectoryEnricher (career metrics)
│   ├── embed.py                   # VectorStoreManager (ChromaDB) & embedding helpers
│   └── score.py                   # CandidateScoringPipeline (multi-persona scoring)
│
├── audit/                         # Fairness auditing
│   ├── __init__.py
│   └── counterfactual.py          # CounterfactualAuditor (bias detection via twin swaps)
│
├── output/                        # Python output module
│   ├── __init__.py
│   └── writer.py                  # rank_candidates(), write_ranked_csv()
│
├── personas/                      # LLM persona system prompts
│   ├── __init__.py
│   ├── hiring_manager.txt         # Strategic fit & trajectory evaluator
│   ├── peer_interviewer.txt       # Technical depth & collaboration evaluator
│   └── devils_advocate.txt        # Risk, red flags & overqualification evaluator
│
├── utils/                         # Cross-cutting utilities
│   ├── __init__.py
│   └── ollama_client.py           # OllamaClient: retry, logging, JSON parsing
│
├── data/                          # Input data
│   ├── sample_candidates.json     # 15 sample candidate profiles
│   └── sample_job_description.json # Sample job description
│
├── results/                       # Generated pipeline output (gitignored; actual folder name in codebase is 'output/')
│   ├── ranked_candidates.csv      # Final ranked results
│   └── bias_audit_report.json     # Fairness audit report
│
├── chroma_store/                  # ChromaDB on-disk vector store (auto-generated)
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_ingest.py             # INGEST phase tests
│   ├── test_embed.py              # EMBED & STORE phase tests
│   └── test_score.py              # SCORE phase tests
│
└── frontend/                      # React + Vite dashboard
    ├── package.json
    ├── vite.config.ts             # Dev proxy /api → localhost:8080
    ├── tailwind.config.ts         # Tailwind v4 design tokens
    ├── tsconfig.json
    └── src/
        ├── main.tsx               # React entry point
        ├── App.tsx                # Router (BrowserRouter, 7 routes)
        ├── api.ts                 # API client (fetch wrapper for /api/*)
        ├── types.ts               # TypeScript interfaces
        ├── index.css              # Design system (tokens, animations, utilities)
        ├── components/
        │   ├── SideNavBar.tsx      # Fixed sidebar navigation
        │   └── TopAppBar.tsx       # Page-level header bar
        └── pages/
            ├── DashboardPage.tsx          # Ranked candidates table + metrics
            ├── CandidateDetailPage.tsx     # Individual candidate deep-dive
            ├── AuditReportPage.tsx         # Fairness audit visualization
            └── SetupPipelinePage.tsx       # 3-step pipeline wizard + progress UI
```

---

## Backend Pipeline (Python)

### Entry Point: `main.py`

The orchestrator that ties all phases together. It:

1. **Parses CLI arguments** (`--candidates-dir`, `--jd-path`, `--output-dir`, `--skip-audit`).
2. **Startup verification**: Confirms Ollama is reachable and the configured model (`llama3.2:3b`) is installed.
3. **Runs the 5-phase pipeline** sequentially, with per-candidate progress logging.

```
main.py
  ├── check_ollama_reachable()    → Verifies Ollama server at localhost:11434
  ├── check_models_present()     → Confirms llama3.2:3b is pulled
  └── run_pipeline()             → Orchestrates INGEST → ENRICH → EMBED → SCORE → AUDIT → OUTPUT
```

### Configuration: `config.py`

The **single source of truth** for all tunable constants. No other module redefines these values.

| Group | Key Constants |
|-------|--------------|
| **Model Selection** | `OLLAMA_MODEL = "llama3.2:3b"`, `EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"` |
| **Token Limits** | `MAX_TOKENS_EXTRACTION = 800`, `MAX_TOKENS_SCORING = 400`, `MAX_TOKENS_NARRATIVE = 150` |
| **Scoring Weights** | `hiring_manager: 0.45`, `peer_interviewer: 0.35`, `devils_advocate: -0.20` |
| **Thresholds** | `BIAS_FLAG_THRESHOLD = 0.75`, `HUMAN_REVIEW_VARIANCE_THRESHOLD = 2.5` |
| **Lookup Tables** | `TITLE_LEVELS` (seniority ordinals), `COUNTERFACTUAL_NAME_PAIRS`, `INSTITUTION_SWAPS` |
| **Calibration** | 10 labeled reference candidates (5 strong_hire, 5 no_hire) for RAG anchoring |

---

## Data Models

### `models/candidate.py`

```
CandidateRole
  ├── title: str                    (required)
  ├── company: str                  (required)
  ├── start_date: str
  ├── end_date: str | None
  ├── duration_months: int = 0
  ├── company_size_estimate: str | None
  └── scope_keywords: list[str]

CandidateProfile
  ├── candidate_id: str             (required, uuid4)
  ├── name: str = "Unknown Candidate"
  ├── email: str | None
  ├── years_experience: float = 0.0
  ├── roles: list[CandidateRole]
  ├── skills_claimed: list[str]
  ├── education: list[dict]
  ├── trajectory_vector: dict | None   (populated by ENRICH phase)
  ├── raw_text: str                    (preserved for audit re-parsing)
  ├── source_file: str
  └── is_complete: bool = False
```

### `models/job.py`

```
JobRequirement
  ├── text: str
  ├── bucket: "must_have" | "nice_to_have" | "culture_signal" | "seniority_marker"
  └── dimension: "technical" | "soft_skill" | "domain" | "experience_level"

JobDescription
  ├── job_id: str                   (uuid5 from file path)
  ├── title: str = "Untitled Role"
  ├── company: str
  ├── requirements: list[JobRequirement]
  ├── raw_text: str
  ├── by_bucket(bucket) → list[JobRequirement]
  └── context_strings() → list[str]   (for embedding/RAG)
```

---

## Pipeline Phases

### Phase 1 — INGEST (`pipeline/ingest.py`)

**Purpose**: Convert raw resume/JD files into validated Pydantic models.

| Component | Description |
|-----------|-------------|
| `ResumeParser` | Parses `.pdf` (PyMuPDF), `.docx` (python-docx), `.json` (direct load). LLM extraction for unstructured formats. Retry-once on validation failure. spaCy fallback for missing name/email. |
| `JdParser` | Parses `.json` (direct) or `.txt` (LLM classification into requirement buckets). Fail-fast for invalid JD. |

### Phase 2 — ENRICH (`pipeline/enrich.py`)

**Purpose**: Compute a 5-metric career `Trajectory_Vector` for each candidate.

| Metric | Computation | Range |
|--------|-------------|-------|
| `growth_rate` | Seniority levels crossed per year (from `TITLE_LEVELS`) | [0.0, 1.0] |
| `complexity_arc` | Company-size trend (startup → scaleup → enterprise) | "ascending" / "descending" / "stable" / "mixed" |
| `leadership_progression` | Fraction of roles with leadership keywords | [0.0, 1.0] |
| `tenure_consistency` | 1 − (std_dev / mean) of role durations | [0.0, 1.0] |
| `seniority_score` | **LLM-derived** (single Ollama call, clamped) | [0.0, 10.0] |

### Phase 3 — EMBED & STORE (`pipeline/embed.py`)

**Purpose**: Embed text into vectors and store in ChromaDB for RAG retrieval.

| Collection | Contents | ID Scheme |
|-----------|----------|-----------|
| `jd_requirements` | Each JD requirement text + bucket/dimension metadata | `{job_id}_{i}` |
| `candidate_profiles` | Two chunks per candidate: profile summary + skills | `{candidate_id}_summary`, `{candidate_id}_skills` |
| `calibration_examples` | 10 labeled reference candidates for scoring anchors | `calib_{i}` |

- **Embedding model**: `BAAI/bge-large-en-v1.5` (sentence-transformers, loaded lazily)
- **All embeddings normalized** (`normalize_embeddings=True`) for consistent cosine similarity
- **Atomic stores**: All embeddings computed before any collection mutation

### Phase 4 — SCORE (`pipeline/score.py`)

**Purpose**: Evaluate each candidate through a 3-persona AI panel.

```
For each candidate:
  1. Retrieve top-5 JD context via vector similarity
  2. Retrieve top-3 calibration examples via vector similarity
  3. Score through 3 personas sequentially:
     ├── Hiring Manager   (weight: +0.45)  → strategic fit, trajectory
     ├── Peer Interviewer  (weight: +0.35)  → technical depth, collaboration
     └── Devil's Advocate  (weight: −0.20)  → risks, red flags (subtracted)
  4. Compute composite_score = Σ(weight × persona_score), clamped [0, 10]
  5. Compute panel_variance = population std dev of 3 persona scores
  6. Flag for human review if panel_variance > 2.5
  7. Generate 3-sentence narrative summary
```

Each persona returns: `{ score, confidence, strengths[], concerns[], verdict }`

### Phase 5 — AUDIT (`audit/counterfactual.py`)

**Purpose**: Detect demographic bias via counterfactual fairness testing.

```
For each candidate:
  1. Build a "twin" by swapping:
     ├── Names       (bidirectional pairs from config)
     ├── Pronouns    (he↔she, him↔her, his↔hers)
     └── Institutions (prestigious → lower-prestige)
  2. Re-parse → Re-enrich → Re-score the twin
  3. Compute delta = |original_score − twin_score|
  4. Flag if delta > 0.75 (potential bias detected)
```

- Swaps are single-pass regex (no cascading)
- Failures are contained per-candidate (never crash the run)
- Output: `bias_audit_report.json` with flag rates and methodology notes

### Phase 6 — OUTPUT (`output/writer.py`)

**Purpose**: Rank candidates and write the final CSV.

- **Ranking**: Descending `composite_score`, `candidate_id` tie-break, consecutive 1-based ranks
- **Verdict consensus**: Majority of 3 personas; `hiring_manager` breaks ties
- **Atomic write**: Temp file + `os.replace()` — no partial CSV ever left behind
- **CSV schema**: 16 columns (rank, scores, verdicts, bias flags, narrative)

---

## Web Dashboard (React + FastAPI)

### FastAPI Backend (`server.py`)

Serves data to the React frontend via REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/candidates` | GET | All ranked candidates sorted by score |
| `/api/candidates/{id}` | GET | Single candidate detail |
| `/api/audit` | GET | Bias audit report |
| `/api/job-description` | GET | Current job description |
| `/api/pipeline/status` | GET | Pipeline running/completed status |
| `/api/run` | POST | Trigger pipeline (background task) |
| `/api/upload/resumes` | POST | Upload candidate files (clears old data) |
| `/api/upload/job-description` | POST | Upload JD file |
| `/api/export/csv` | GET | Download `ranked_candidates.csv` |

### React Frontend (`frontend/`)

Built with **Vite + React + TypeScript + Tailwind CSS v4**.

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | Ranked candidates table, metric cards, score distribution |
| **Candidate Detail** | `/candidates/:id` | Deep-dive: scores, trajectory, strengths/concerns, narrative |
| **Audit Report** | `/audit` | Fairness audit results, flag rates, methodology |
| **Pipeline Setup** | `/pipeline` | 3-step wizard: JD → Upload Resumes → Configure & Run |

**Design System**: Material Design 3 tokens with a teal/navy palette, Inter typography, flat design aesthetic with no shadows.

**Key UI Features**:
- Circular progress loader during pipeline execution (matching dashboard-2 mockup)
- Step-by-step status indicators (parsing → scoring → auditing → signals)
- Drag-and-drop file upload for resumes
- Real-time pipeline status polling

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Ollama + Llama 3.2 (3B) | Local inference for extraction, scoring, enrichment |
| **Embeddings** | sentence-transformers (`bge-large-en-v1.5`) | Semantic text embeddings |
| **Vector Store** | ChromaDB (PersistentClient) | RAG retrieval for JD context & calibration |
| **Data Models** | Pydantic v2 | Schema validation for candidates & job descriptions |
| **Resume Parsing** | PyMuPDF, python-docx | PDF/DOCX text extraction |
| **NLP Fallback** | spaCy (`en_core_web_sm`) | Name/email recovery from unstructured text |
| **Output** | pandas | CSV serialization |
| **Backend API** | FastAPI + Uvicorn | REST API serving dashboard data |
| **Frontend** | React + Vite + TypeScript | Interactive dashboard UI |
| **Styling** | Tailwind CSS v4 | Material Design 3 design system |
| **Testing** | pytest + Hypothesis | Unit + property-based testing |

---

## Data Flow

```
INPUT                    PIPELINE                         OUTPUT
─────                    ────────                         ──────

sample_candidates.json ─┐
  (or PDF/DOCX files)   ├→ INGEST → CandidateProfile[] ─┐
                        │                                │
sample_job_description  │                                │
  .json (or .txt)      ─┘→ INGEST → JobDescription      │
                                                         │
                              ENRICH ← OllamaClient      │
                                │                        │
                                ▼                        │
                         CandidateProfile[]              │
                         (with trajectory_vector)        │
                                │                        │
                                ▼                        │
                          EMBED & STORE                  │
                                │                        │
                         ┌──────┴───────┐                │
                         ▼              ▼                │
                    ChromaDB        ChromaDB             │
                  (JD reqs)     (Calibration)            │
                         │              │                │
                         └──────┬───────┘                │
                                ▼                        │
                             SCORE ← 3 Personas          │
                                │                        │
                                ▼                        │
                         Scored Results[]                │
                                │                        │
                         ┌──────┴───────┐                │
                         ▼              ▼                │
                      AUDIT          OUTPUT              │
                         │              │                │
                         ▼              ▼                │
              bias_audit_report   ranked_candidates      │
                    .json              .csv               │
                         │              │                │
                         └──────┬───────┘                │
                                ▼                        │
                        FastAPI (server.py)               │
                                │                        │
                                ▼                        │
                     React Dashboard (Vite)              │
```

---

## Running the Project

### Prerequisites

- Python 3.11+
- Node.js 18+
- Ollama installed and running (`ollama serve`)
- Llama 3.2 model pulled (`ollama pull llama3.2:3b`)

### Backend

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline (CLI)
python main.py

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → Dashboard available at http://localhost:5173
```

### Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
