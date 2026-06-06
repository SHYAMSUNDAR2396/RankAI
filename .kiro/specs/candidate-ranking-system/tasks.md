# Implementation Plan: Candidate Ranking System

## Overview

This plan converts the Candidate Ranking System design into a sequence of discrete,
incremental Python (3.11+) coding tasks. Tasks are ordered to respect the design's
dependency direction: config and Pydantic models first, then the cross-cutting Ollama
and embedding utilities, then the five pipeline phases (INGEST → ENRICH → EMBED & STORE
→ SCORE → AUDIT), then output, then the CLI that wires everything together, and finally
sample data, tests, and documentation. Each task builds on prior tasks and ends with
integration so there is no orphaned code.

Testing follows the design's dual approach. Example/integration/smoke tests live in
`tests/test_ingest.py`, `tests/test_embed.py`, and `tests/test_score.py` (per the design
layout and the user's milestone). Property-based tests use Hypothesis with a minimum of
100 examples per property, are co-located in those same three files near the phase they
validate, and each is tagged `# Feature: candidate-ranking-system, Property N: ...`. All
`ollama.chat` calls are mocked via `unittest.mock.patch`; the suite runs fully offline and
deterministically (Requirements 12.5–12.8).

All test-related sub-tasks are marked optional with `*` and may be skipped for a faster MVP;
core implementation sub-tasks are never optional.

## Tasks

- [x] 1. Project scaffolding and centralized configuration
  - [x] 1.1 Create dependency manifest and repository structure
    - Create `requirements.txt` pinning exact versions of: ollama, sentence-transformers, chromadb, PyMuPDF, python-docx, spacy, pandas, pydantic (v2), rich, pytest, pytest-mock, hypothesis
    - Create the package directory skeleton with `__init__.py` files: `pipeline/`, `audit/`, `personas/`, `models/`, `output/`, `data/`, `tests/`
    - Target Python 3.11+ in packaging metadata
    - _Requirements: 1.9, 1.10_

  - [x] 1.2 Implement `config.py` as the single source of truth for all constants
    - Define `OLLAMA_MODEL` (`llama3.2:3b`), `OLLAMA_HOST`, `EMBEDDING_MODEL` (`BAAI/bge-large-en-v1.5`), `CHROMA_PERSIST_DIR`
    - Define token limits `extraction`, `scoring`, `narrative` (positive ints) and `SCORING_TEMPERATURE` in [0.0, 1.0]
    - Define `PERSONA_WEIGHTS` {hiring_manager: 0.45, peer_interviewer: 0.35, devils_advocate: -0.20}, `BIAS_FLAG_THRESHOLD` = 0.75, `HUMAN_REVIEW_VARIANCE_THRESHOLD` = 2.5
    - Define `TITLE_LEVELS`, `COUNTERFACTUAL_NAME_PAIRS`, `INSTITUTION_SWAPS` (each ≥ 1 entry)
    - Define `CALIBRATION_EXAMPLES` with exactly 5 `strong_hire` + 5 `no_hire` entries
    - Add a module docstring; ensure no other module will redefine these constants
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 2. Define Pydantic v2 data models
  - [x] 2.1 Implement `models/candidate.py`
    - Define `CandidateRole`, `TrajectoryVector`, and `CandidateProfile` as Pydantic v2 models
    - Give every optional field an explicit default (e.g. `name="Unknown Candidate"`, `email=""`, `is_complete=False`, `trajectory=None`, list fields via `default_factory`)
    - Encode `TrajectoryVector` defaults: `growth_rate=0.0`, `complexity_arc="stable"`, `leadership_progression=0.0`, `tenure_consistency=1.0`, `seniority_score=5.0`
    - _Requirements: 2.7, 4.1, 11.4_

  - [x] 2.2 Implement `models/job.py`
    - Define `JobRequirement` (text, bucket, dimension) and `JobDescription` (job_id, title, company, raw_text, requirements) as Pydantic v2 models with defaults
    - Add helper methods `by_bucket(bucket)` and `context_strings()`
    - _Requirements: 3.8, 11.2_

- [x] 3. Cross-cutting utilities (LLM wrapper and embedding module)
  - [x] 3.1 Implement `OllamaClient.chat` with retry/backoff in `utils/ollama_client.py`
    - Wrap all `ollama.chat` calls against `OLLAMA_HOST`/`OLLAMA_MODEL`; this is the single swap seam for an alternative backend
    - Retry up to 2 additional times on exception/no response, with `time.sleep` backoff starting at 1s then doubling (1s, 2s)
    - Raise `OllamaCallError` when the initial call plus both retries are exhausted
    - Emit request/response detail at DEBUG level
    - _Requirements: 1.1, 1.11, 13.6, 13.7, 13.8_

  - [x] 3.2 Implement `OllamaClient.chat_json` JSON helper
    - Strip markdown code fences (```json ... ```), parse JSON, and on parse failure only return the caller-supplied fallback; propagate all non-parse exceptions
    - _Requirements: 13.9, 13.10_

  - [-]* 3.3 Write property test for `chat_json` fallback (in `tests/test_score.py`)
    - **Property 24: chat_json falls back on non-JSON without raising**
    - **Validates: Requirements 13.9**

  - [-]* 3.4 Write example tests for `OllamaClient` retry/backoff (in `tests/test_score.py`)
    - Assert 3 attempts with `time.sleep` patched and called with 1s then 2s, then raises; assert non-parse error propagates; assert DEBUG request/response logging
    - Mock `ollama.chat` via `unittest.mock.patch`; no network calls
    - _Requirements: 13.6, 13.7, 13.8, 13.10, 12.5, 12.6_

  - [x] 3.5 Implement the embedding module in `pipeline/embed.py`
    - Lazy-loaded process-wide `SentenceTransformer` singleton via `get_embedding_model()` (never loaded at import time)
    - `embed_text(text)` and `embed_texts(texts)` using `normalize_embeddings=True` with `EMBEDDING_MODEL` from config
    - _Requirements: 1.2, 5.2_

- [x] 4. INGEST: Resume_Parser
  - [x] 4.1 Implement `ResumeParser` in `pipeline/ingest.py`
    - Dispatch by extension: `.pdf` → PyMuPDF text, `.docx` → python-docx text, `.json` → direct structured load (no LLM) with completion flag set, other → log warning + skip + continue
    - For `.pdf`/`.docx`: call `OllamaClient` to extract structured profile JSON; assign `candidate_id` via `uuid4`
    - spaCy `en_core_web_sm` fallback for missing name/email; fill remaining optional fields with model defaults
    - Validate against the Pydantic v2 model; on failure retry extraction once with a correction prompt; on second failure log + skip + continue
    - Skip + continue on empty/whitespace `.pdf`/`.docx` text or structurally invalid `.json`
    - Expose `parse_text(raw_text, source_path)` for audit re-parse
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12_

  - [-]* 4.2 Write property test for parsed profile defaults (in `tests/test_ingest.py`)
    - **Property 1: Parsed profiles validate with defaults populated**
    - **Validates: Requirements 2.7, 2.9, 12.1**

  - [-]* 4.3 Write example tests for Resume_Parser dispatch and resilience (in `tests/test_ingest.py`)
    - `.json` load asserts `ollama.chat` not called and `is_complete` true; `.pdf`/`.docx` dispatch to extractors then mocked LLM; validation retry-once-then-skip; unsupported extension and empty/invalid input skip-and-continue
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.8, 2.10, 2.11, 2.12, 12.1_

- [x] 5. INGEST: JD_Parser
  - [x] 5.1 Implement `JdParser` in `pipeline/ingest.py`
    - `.txt` → call `OllamaClient` to classify into buckets/dimensions; `.json` → direct structured load (no LLM)
    - Unsupported extension → report error, produce no `JobDescription`; corrupt/invalid `.json` → report error, produce no model, do not invoke the LLM
    - Classify into buckets `must_have`/`nice_to_have`/`culture_signal`/`seniority_marker` and dimensions `technical`/`soft_skill`/`domain`/`experience_level`
    - Assign `job_id` via `uuid5` on the file path; validate against the model; apply defined fallback values that still validate when classification JSON is malformed after retries
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [-]* 5.2 Write property test for JD bucket coverage and dimensions (in `tests/test_ingest.py`)
    - **Property 2: JD classification covers all buckets and uses valid dimensions**
    - **Validates: Requirements 3.5, 3.6, 12.2**

  - [-]* 5.3 Write property test for JD always validating including fallback (in `tests/test_ingest.py`)
    - **Property 3: JobDescription always validates, including on malformed classification**
    - **Validates: Requirements 3.8, 3.9**

  - [-]* 5.4 Write property test for deterministic uuid5 job_id (in `tests/test_ingest.py`)
    - **Property 4: job_id is a deterministic uuid5 of the file path**
    - **Validates: Requirements 3.7**

  - [-]* 5.5 Write example tests for JD_Parser dispatch and error paths (in `tests/test_ingest.py`)
    - `.txt` classification (mocked LLM), `.json` direct load (no LLM), unsupported-extension error, corrupt `.json` error with no LLM call
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6. ENRICH: TrajectoryEnricher
  - [x] 6.1 Implement pure `compute_*` trajectory functions in `pipeline/enrich.py`
    - `compute_growth_rate`, `compute_complexity_arc`, `compute_leadership_progression`, `compute_tenure_consistency` as pure functions (no I/O)
    - Encode degenerate defaults: zero roles → tenure_consistency 1.0 / growth_rate 0.0 / leadership_progression 0.0 / complexity_arc `stable`; years_experience 0 → growth_rate 0.0; < 2 distinct company_size_estimate → complexity_arc `stable`
    - _Requirements: 4.1, 4.5_

  - [x] 6.2 Implement `TrajectoryEnricher.enrich` with LLM seniority
    - Call `OllamaClient` for `seniority_score` ∈ [0, 10], clamp to range, default 5.0 if unparseable without raising
    - Attach the computed `TrajectoryVector` to the profile and return it
    - _Requirements: 4.2, 4.3, 4.4, 4.6_

  - [-]* 6.3 Write property test for trajectory range, enum, and attachment (in `tests/test_ingest.py`)
    - **Property 5: Trajectory metrics stay in range and are attached**
    - **Validates: Requirements 4.1, 4.6**

  - [-]* 6.4 Write property test for degenerate trajectory defaults (in `tests/test_ingest.py`)
    - **Property 6: Degenerate trajectory inputs map to defined defaults**
    - **Validates: Requirements 4.5**

  - [-]* 6.5 Write property test for seniority score clamping (in `tests/test_ingest.py`)
    - **Property 7: Seniority score is clamped to [0, 10]**
    - **Validates: Requirements 4.2, 4.3**

  - [-]* 6.6 Write example test for unparseable seniority default (in `tests/test_ingest.py`)
    - Mocked unparseable seniority response → `seniority_score` 5.0 without raising
    - _Requirements: 4.4_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. EMBED & STORE: VectorStoreManager
  - [x] 8.1 Implement `VectorStoreManager` initialization and storage in `pipeline/embed.py`
    - On init create three ChromaDB collections (`jd_requirements`, `candidate_profiles`, `calibration_examples`) each with `embedding_function=None`
    - Compute embeddings with the embedding module and manually inject vectors (`embeddings=[...]`)
    - Store each candidate as exactly two chunks (`profile_summary`, `skills`) with metadata `candidate_id` and `chunk_type`
    - Store exactly 10 calibration examples (5 `strong_hire` + 5 `no_hire`) from `config.CALIBRATION_EXAMPLES`
    - On embedding-computation failure during a store, abort the store, leave the target collection unchanged, and raise an embedding-failure error
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7, 1.3_

  - [x] 8.2 Implement retrieval methods on `VectorStoreManager`
    - `query_jd_context(query, n=5)` embeds via `query_embeddings`, returns ≤ 5 JD context strings descending by similarity
    - `query_calibration(query, n=3)` embeds via `query_embeddings`, returns ≤ 3 calibration metadata dicts (`outcome`, `reason`) descending by similarity
    - _Requirements: 5.4, 5.5_

  - [x]* 8.3 Write property test for two labeled candidate chunks (in `tests/test_embed.py`)
    - **Property 8: A candidate is stored as exactly two labeled chunks**
    - **Validates: Requirements 5.3, 12.3**

  - [x]* 8.4 Write property test for balanced calibration store (in `tests/test_embed.py`)
    - **Property 9: Calibration store holds exactly ten balanced examples**
    - **Validates: Requirements 5.6, 12.3**

  - [x]* 8.5 Write property test for JD retrieval bound and ordering (in `tests/test_embed.py`)
    - **Property 10: JD retrieval returns at most five items ordered by similarity**
    - **Validates: Requirements 5.4**

  - [x]* 8.6 Write property test for calibration retrieval bound and ordering (in `tests/test_embed.py`)
    - **Property 11: Calibration retrieval returns at most three well-formed items ordered by similarity**
    - **Validates: Requirements 5.5**

  - [x]* 8.7 Write integration tests for collections and embedding failure (in tests/test_embed.py)
    - Three collections created with `embedding_function=None`; candidate stores exactly two chunks; retrieval non-empty; calibration count == 10; embedding failure mid-store aborts and leaves the collection unchanged
    - _Requirements: 5.1, 5.7, 12.3_

- [x] 9. SCORE: CandidateScoringPipeline
  - [x] 9.1 Author persona system prompt files
    - Create `personas/hiring_manager.txt`, `personas/peer_interviewer.txt`, `personas/devils_advocate.txt` (text files, not config)
    - _Requirements: 6.1_

  - [x] 9.2 Implement pure `composite_score` and `panel_variance` static methods in `pipeline/score.py`
    - `composite_score` = `round(0.45*hm + 0.35*peer - 0.20*devil, 2)` then clamp to [0, 10]
    - `panel_variance` = population variance (mean of squared deviations from the mean)
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 9.3 Implement `CandidateScoringPipeline.score` orchestration
    - Load persona prompts; build query, retrieve 5 JD + 3 calibration items from `VectorStoreManager`
    - Run three persona Ollama calls sequentially, each producing a score ∈ [0, 10] and verdict; retry a persona once on non-JSON; substitute default verdict + default persona score on second failure and continue
    - Set `requires_human_review = (panel_variance > 2.5)`; generate a narrative of exactly three sentences via Ollama
    - Return the full result dict with de-duplicated `strengths`/`concerns`
    - _Requirements: 6.1, 6.2, 6.8, 6.9, 6.10, 6.11_

  - [-]* 9.4 Write property test for composite score clamping (in `tests/test_score.py`)
    - **Property 13: Composite score is the rounded weighted sum clamped to [0, 10]**
    - **Validates: Requirements 6.3, 6.4, 12.9**

  - [-]* 9.5 Write property test for panel variance and human-review trigger (in `tests/test_score.py`)
    - **Property 14: Panel variance is population variance and drives human review**
    - **Validates: Requirements 6.5, 6.6, 6.7, 12.10**

  - [-]* 9.6 Write property test for persona score range (in `tests/test_score.py`)
    - **Property 12: Each persona score is in [0, 10]**
    - **Validates: Requirements 6.1**

  - [-]* 9.7 Write property test for schema-complete deduplicated result (in `tests/test_score.py`)
    - **Property 15: Score result is schema-complete with deduplicated lists**
    - **Validates: Requirements 6.11, 12.4**

  - [-]* 9.8 Write example tests for scoring resilience and RAG assembly (in `tests/test_score.py`)
    - Persona JSON retry-once then default-verdict substitution; narrative reduced to exactly 3 sentences; RAG passes 5 JD + 3 calibration items to persona prompts; result fields present and `counterfactual_delta` float ≥ 0
    - Mock all `ollama.chat`; deterministic and offline
    - _Requirements: 6.2, 6.8, 6.9, 6.10, 12.4, 12.5, 12.8_

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. AUDIT: CounterfactualAuditor
  - [x] 11.1 Implement pure swap helpers and twin construction in `audit/counterfactual.py`
    - `swap_pronouns` (whole-word, case-insensitive he/him/his ↔ she/her/hers) and `swap_tokens` (whole-word name/institution swaps) as pure functions
    - `build_twin(profile)` applies `COUNTERFACTUAL_NAME_PAIRS`, pronoun swaps, and `INSTITUTION_SWAPS`, leaving unconfigured tokens unchanged
    - _Requirements: 7.1_

  - [x] 11.2 Implement `CounterfactualAuditor.audit`
    - Re-parse and re-enrich the twin, re-score through the same `CandidateScoringPipeline` and `VectorStoreManager` with `candidate_id = "cf_" + original_id`
    - Compute `counterfactual_delta = round(abs(original - twin), 2)`; set `bias_flag = (delta > 0.75)`
    - Update the original result in place with `bias_flag`/`counterfactual_delta` and append to `AUDIT_LOG`
    - On twin re-parse/re-enrich/re-score failure, record an audit-failure entry, do not set `bias_flag` true, and continue
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.9_

  - [x] 11.3 Implement `write_report` for the audit JSON
    - Write `bias_audit_report.json` with total audited, flagged count, flag rate (flagged/total, 0 when total is 0), threshold, flagged candidates, clean count, methodology note, and audit failures
    - When skipped, write a report documenting that no bias analysis was performed
    - _Requirements: 7.7, 7.8_

  - [-]* 11.4 Write property test for swap correctness (in `tests/test_score.py`)
    - **Property 16: Counterfactual swaps are whole-word, case-insensitive, and leave unconfigured tokens unchanged**
    - **Validates: Requirements 7.1**

  - [-]* 11.5 Write property test for twin id and shared store (in `tests/test_score.py`)
    - **Property 17: The twin reuses the same store with a cf_-prefixed id**
    - **Validates: Requirements 7.2**

  - [-]* 11.6 Write property test for counterfactual delta and bias flag (in `tests/test_score.py`)
    - **Property 18: Counterfactual delta is a non-negative rounded difference that drives the bias flag**
    - **Validates: Requirements 7.3, 7.4, 7.5, 12.4**

  - [-]* 11.7 Write property test for flag rate computation (in `tests/test_score.py`)
    - **Property 19: Flag rate equals flagged over total**
    - **Validates: Requirements 7.7**

  - [-]* 11.8 Write example tests for audit lifecycle (in `tests/test_score.py`)
    - In-place result update + `AUDIT_LOG` append; skip-audit report content; twin-failure handling (no bias_flag true, continue)
    - _Requirements: 7.6, 7.8, 7.9_

- [x] 12. OUTPUT: Ranked CSV and writer
  - [x] 12.1 Implement pure `rank_candidates` and `verdict_consensus` in `output/writer.py`
    - `rank_candidates`: sort by composite_score descending, tie-break by candidate_id ascending, assign consecutive ranks from 1
    - `verdict_consensus`: verdict held by ≥ 2 of 3 personas, else the hiring_manager verdict
    - _Requirements: 8.1, 8.2, 8.5, 8.6_

  - [x] 12.2 Implement atomic `write_ranked_csv`
    - Write fixed column order with pandas; pipe-separate `strengths`/`concerns` (empty list → empty string)
    - Write to a temp path then atomically rename so no partial `ranked_candidates.csv` remains on failure
    - _Requirements: 8.3, 8.4, 8.7_

  - [-]* 12.3 Write property test for ranking and consecutive ranks (in `tests/test_score.py`)
    - **Property 20: Ranking is correctly ordered with consecutive ranks**
    - **Validates: Requirements 8.1, 8.2**

  - [-]* 12.4 Write property test for verdict consensus (in `tests/test_score.py`)
    - **Property 22: Verdict consensus is the majority, else the hiring manager's verdict**
    - **Validates: Requirements 8.5, 8.6**

  - [-]* 12.5 Write property test for CSV columns and list serialization (in `tests/test_score.py`)
    - **Property 21: CSV encoding has fixed columns and correct list serialization**
    - **Validates: Requirements 8.4**

- [x] 13. CLI: orchestration entry point
  - [x] 13.1 Implement argparse and startup verification in `main.py`
    - Flags `--candidates-dir`, `--job-description`, `--output-dir`, `--skip-audit`, `--verbose`
    - Verify Ollama reachable within a 10s timeout via `ollama.list()`; on failure report setup instructions and exit
    - Verify the LLM and Embedding_Model are present locally; if missing, name each and exit without running any phase or writing artifacts
    - _Requirements: 9.1, 9.2, 9.3, 1.7, 1.8_

  - [x] 13.2 Implement pipeline orchestration and logging configuration
    - Load + parse the JD and candidate files (`.json`/`.pdf`/`.docx`); fall back to `data/sample_candidates.json` when the dir is missing or empty
    - Embed JD + calibration examples before scoring; run scoring then audit (unless `--skip-audit`); wire writer + auditor report
    - Configure logging: `--verbose` → DEBUG else INFO+; route all output through `logging` (no `print`); INFO logs at phase begin/end and per-candidate completion
    - _Requirements: 9.4, 9.5, 9.6, 9.10, 9.11, 9.12, 13.4, 13.5_

  - [x] 13.3 Implement rich progress, top-N table, and summary
    - Rich progress bar over candidate count during scoring
    - Rich table of top 10 (or all if fewer) by composite_score showing rank, name, composite_score, verdict_consensus, requires_human_review, bias_flag
    - Summary with total candidates, elapsed seconds, and output paths
    - _Requirements: 9.7, 9.8, 9.9_

  - [-]* 13.4 Write example tests for the CLI (in `tests/test_score.py`)
    - Arg parsing; reachability-failure exit; missing-model exit (no artifacts); sample-data fallback; top-10 table and summary content; `--skip-audit`; INFO/DEBUG log-level switching
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 1.7, 1.8_

- [x] 14. Sample data and dataset validation
  - [x] 14.1 Create `data/sample_candidates.json`
    - 15 profiles: 3 strong, 3 partial, 3 overqualified, 3 underqualified, 3 non-traditional (each in exactly one category)
    - Each profile validates against `CandidateProfile`; each role has `start_date <= end_date` and `duration_months` equal to the whole months between the dates
    - _Requirements: 11.1, 11.4, 11.5_

  - [x] 14.2 Create `data/sample_job_description.json`
    - Senior Software Engineer at a Series B SaaS company; validates against `JobDescription`; at least one requirement in each bucket (`must_have`, `nice_to_have`, `culture_signal`, `seniority_marker`)
    - _Requirements: 11.2, 11.3_

  - [-]* 14.3 Write property test for sample role date consistency (in `tests/test_ingest.py`)
    - **Property 23: Sample role dates are consistent with duration**
    - **Validates: Requirements 11.5**

  - [-]* 14.4 Write smoke tests for sample data, config, and code quality (in `tests/test_ingest.py`)
    - 15 profiles across the 5 categories; sample JD validates with ≥ 1 per bucket; all config constants present with valid ranges and `CALIBRATION_EXAMPLES` is 5+5; constants sourced only from `config.py`; module docstrings/type hints present and no `print`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 10.1, 10.2, 10.3, 10.4, 10.5, 13.1, 13.2, 13.3, 13.4_

- [x] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Documentation
  - [x] 16.1 Write `methodology.md`
    - ~1200 words covering the nine specified sections (overview, ingest/enrich, embedding & RAG, multi-persona scoring, composite + variance math, counterfactual audit, ranking/output, limitations, reproducibility)
    - _Requirements: 6.3, 6.5, 7.1, 7.3_

  - [x] 16.2 Write `README.md`
    - Setup and usage (Python 3.11+, dependency install, Ollama/model downloads, running the CLI), plus the documented Groq `llama-3.1-70b` alternative backend that swaps only the Ollama_Client LLM call and the manifest
    - _Requirements: 1.9, 1.10, 1.11, 9.1_

- [x] 17. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (unit, property, integration, and smoke tests) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each property test maps to exactly one Correctness Property, uses Hypothesis with ≥ 100 examples, and carries the tag `# Feature: candidate-ranking-system, Property N: ...`.
- Properties 13 and 14 (composite clamping and the panel-variance human-review trigger) are the two explicitly required by Requirements 12.9 and 12.10.
- All `ollama.chat` calls are mocked via `unittest.mock.patch`; the suite runs offline and deterministically (Requirements 12.5–12.8); `time.sleep` is patched to assert backoff without real delays.
- The embedding module and `VectorStoreManager` share `pipeline/embed.py`; the embedding helpers are implemented first (Task 3.5) and the store is layered on top (Task 8).
- Checkpoints provide incremental validation; each task references the specific requirements and design properties it satisfies.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "2.2", "3.1", "3.5", "9.1"] },
    { "id": 2, "tasks": ["3.2", "6.1", "9.2", "11.1", "12.1", "8.1", "14.1", "14.2"] },
    { "id": 3, "tasks": ["4.1", "6.2", "8.2", "12.2"] },
    { "id": 4, "tasks": ["5.1", "9.3", "13.1"] },
    { "id": 5, "tasks": ["11.2"] },
    { "id": 6, "tasks": ["11.3"] },
    { "id": 7, "tasks": ["13.2"] },
    { "id": 8, "tasks": ["13.3"] },
    { "id": 9, "tasks": ["3.3", "4.2", "8.3", "16.1", "16.2"] },
    { "id": 10, "tasks": ["3.4", "4.3", "8.4"] },
    { "id": 11, "tasks": ["9.4", "5.2", "8.5"] },
    { "id": 12, "tasks": ["9.5", "5.3", "8.6"] },
    { "id": 13, "tasks": ["9.6", "5.4", "8.7"] },
    { "id": 14, "tasks": ["9.7", "5.5"] },
    { "id": 15, "tasks": ["9.8", "6.3"] },
    { "id": 16, "tasks": ["11.4", "6.4"] },
    { "id": 17, "tasks": ["11.5", "6.5"] },
    { "id": 18, "tasks": ["11.6", "6.6"] },
    { "id": 19, "tasks": ["11.7", "14.3"] },
    { "id": 20, "tasks": ["11.8", "14.4"] },
    { "id": 21, "tasks": ["12.3"] },
    { "id": 22, "tasks": ["12.4"] },
    { "id": 23, "tasks": ["12.5"] },
    { "id": 24, "tasks": ["13.4"] }
  ]
}
```
