# Requirements Document

## Introduction

The Candidate Ranking System is a production-quality, AI-powered command-line pipeline that ingests resumes and a job description, enriches candidate career trajectories, embeds and stores them in a local vector database, scores candidates through a multi-persona evaluation panel, audits scoring for demographic bias via counterfactual analysis, and produces a ranked candidate report.

The entire system runs on a free, local, open-source stack. All Large Language Model inference is performed by a locally hosted Ollama server, all embeddings are computed by a local sentence-transformers model, and all vector storage uses a local on-disk database. After the initial model downloads, the system requires no internet access, no API keys, and no paid services.

The pipeline executes five phases in sequence: INGEST, ENRICH, EMBED & STORE, SCORE, and COUNTERFACTUAL FAIRNESS AUDIT. The system emits a ranked candidate CSV and a bias audit JSON report.

## Glossary

- **System**: The Candidate Ranking System, the complete command-line pipeline described in this document.
- **Ollama_Client**: The component that wraps all calls to the locally hosted Ollama server for LLM inference.
- **Ollama_Server**: The locally running Ollama process that hosts the LLM, reachable at the configured Ollama host URL.
- **LLM**: The local large language model served by Ollama (default `llama3.2:3b`).
- **Embedding_Model**: The local sentence-transformers model that converts text into dense vectors (default `BAAI/bge-large-en-v1.5`).
- **Resume_Parser**: The component that converts a resume file into structured data.
- **JD_Parser**: The component that converts a job description into a structured JobDescription model.
- **CandidateProfile**: A Pydantic v2 data model representing one candidate's structured profile.
- **JobDescription**: A Pydantic v2 data model representing the parsed job description with requirement buckets and dimensions.
- **TrajectoryEnricher**: The component that computes career trajectory metrics for a candidate.
- **Trajectory_Vector**: The set of computed trajectory metrics (growth_rate, complexity_arc, leadership_progression, tenure_consistency) plus the seniority_score.
- **VectorStoreManager**: The component that manages embeddings and the three vector database collections.
- **Vector_Store**: The local on-disk ChromaDB persistent client and its collections.
- **Calibration_Example**: A labeled reference candidate (strong_hire or no_hire) used to anchor scoring.
- **CandidateScoringPipeline**: The component that scores a candidate using the three-persona panel and RAG context.
- **Persona**: One of the three evaluator viewpoints: hiring_manager, peer_interviewer, devils_advocate.
- **Composite_Score**: The weighted aggregate score for a candidate on a 0 to 10 scale.
- **Panel_Variance**: The statistical variance across the three persona scores for a candidate.
- **CounterfactualAuditor**: The component that performs counterfactual fairness auditing.
- **Counterfactual_Twin**: A copy of a candidate profile with name, gendered pronouns, and institutions swapped.
- **Counterfactual_Delta**: The absolute difference between a candidate's composite score and the counterfactual twin's composite score.
- **Bias_Flag**: A boolean indicating that a candidate's counterfactual delta exceeded the bias threshold.
- **Audit_Log**: The in-memory and persisted record of counterfactual audit results.
- **CLI**: The command-line interface entry point (`main.py`).
- **Config**: The configuration module (`config.py`) holding all constants.
- **Ranked_Output**: The `ranked_candidates.csv` file containing ranked candidate results.
- **Audit_Report**: The `bias_audit_report.json` file containing the fairness audit results.

## Requirements

### Requirement 1: Free and Local Open-Source Stack Constraint

**User Story:** As an operator with no budget for paid APIs, I want the entire system to run on a free local open-source stack, so that I can run candidate ranking without API keys, paid services, or internet access after setup.

#### Acceptance Criteria

1. THE System SHALL perform all LLM inference through the Ollama_Client connected to a locally hosted Ollama_Server.
2. THE System SHALL compute all embeddings using the local Embedding_Model `BAAI/bge-large-en-v1.5`.
3. THE System SHALL store all vectors using a local on-disk ChromaDB PersistentClient configured with `embedding_function=None`.
4. THE System SHALL operate without any OpenAI, Anthropic, Cohere, or other paid third-party API.
5. THE System SHALL operate without requiring any API key.
6. WHILE all required model downloads have completed, THE System SHALL execute the full pipeline without opening an outbound network connection to any host other than the configured local Ollama_Server host.
7. WHEN the System starts, THE System SHALL verify that the LLM and the Embedding_Model are present locally before executing any pipeline phase.
8. IF the LLM or the Embedding_Model is not present locally at startup verification, THEN THE System SHALL report an error naming each missing model and SHALL exit without executing any pipeline phase and without producing the Ranked_Output or the Audit_Report.
9. THE System SHALL target Python version 3.11 or higher.
10. THE System SHALL use the dependency set: Ollama Python SDK, sentence-transformers, ChromaDB, PyMuPDF, python-docx, spaCy with `en_core_web_sm`, pandas, Pydantic version 2, rich, pytest, and pytest-mock.
11. WHERE an alternative LLM backend is documented, THE System SHALL support substituting the Ollama_Client LLM call implementation and the dependency manifest to use the Groq free tier with `llama-3.1-70b` while leaving the Resume_Parser, JD_Parser, TrajectoryEnricher, VectorStoreManager, CandidateScoringPipeline, CounterfactualAuditor, and CLI components unchanged.

### Requirement 2: Ingest Resume Files into Candidate Profiles

**User Story:** As a recruiter, I want resumes in multiple formats parsed into structured candidate profiles, so that the pipeline can evaluate every candidate consistently.

#### Acceptance Criteria

1. WHEN a resume file with a `.pdf` extension is provided, THE Resume_Parser SHALL extract raw text using PyMuPDF.
2. WHEN a resume file with a `.docx` extension is provided, THE Resume_Parser SHALL extract raw text using python-docx.
3. WHEN raw resume text is extracted, THE Resume_Parser SHALL invoke the Ollama_Client to extract the text into a structured CandidateProfile in JSON form.
4. WHEN a resume file with a `.json` extension is provided, THE Resume_Parser SHALL load the structured CandidateProfile directly without invoking the LLM and SHALL set a completion flag indicating the profile was successfully created.
5. WHEN a CandidateProfile is created, THE Resume_Parser SHALL assign a `candidate_id` generated with `uuid4`.
6. IF the LLM extraction does not yield a candidate name or email, THEN THE Resume_Parser SHALL apply a spaCy `en_core_web_sm` fallback to extract the missing candidate name and email, and SHALL assign the defined default value for any of those fields that remain unresolved after the fallback.
7. IF an optional CandidateProfile field is missing from the source, THEN THE Resume_Parser SHALL produce a valid CandidateProfile using the defined default value for that field.
8. IF a resume file has an unsupported extension, THEN THE Resume_Parser SHALL log a warning, skip the file, and continue processing the remaining resume files.
9. THE Resume_Parser SHALL validate each CandidateProfile against the Pydantic version 2 model before returning it.
10. IF a CandidateProfile fails validation against the Pydantic version 2 model, THEN THE Resume_Parser SHALL retry the extraction once with a correction prompt.
11. IF a CandidateProfile fails validation against the Pydantic version 2 model after the single retry, THEN THE Resume_Parser SHALL log a warning, skip the resume file, and continue processing the remaining resume files.
12. IF text extraction from a `.pdf` or `.docx` resume file yields empty or whitespace-only content, or a `.json` resume file is structurally invalid, THEN THE Resume_Parser SHALL log a warning, skip the resume file, and continue processing the remaining resume files.

### Requirement 3: Ingest Job Description into Structured Model

**User Story:** As a hiring manager, I want the job description parsed into structured requirement buckets, so that scoring can reason about specific role expectations.

#### Acceptance Criteria

1. WHEN a job description file with a `.txt` extension is provided, THE JD_Parser SHALL invoke the Ollama_Client to classify the description content into the requirement buckets and dimensions.
2. WHEN a job description file with a `.json` extension is provided, THE JD_Parser SHALL load the structured JobDescription directly without invoking the LLM.
3. IF a job description file with an extension other than `.txt` or `.json` is provided, THEN THE JD_Parser SHALL report an error indicating the unsupported extension and SHALL NOT produce a JobDescription.
4. IF a `.json` job description file is corrupted or structurally invalid, THEN THE JD_Parser SHALL report an error indicating the file is invalid, SHALL NOT produce a JobDescription, and SHALL halt without invoking the Ollama_Client.
5. THE JD_Parser SHALL classify job description content into the requirement buckets `must_have`, `nice_to_have`, `culture_signal`, and `seniority_marker`.
6. THE JD_Parser SHALL classify job description content into the dimensions `technical`, `soft_skill`, `domain`, and `experience_level`.
7. WHEN a JobDescription is created, THE JD_Parser SHALL assign a `job_id` generated with `uuid5` on the file path.
8. THE JD_Parser SHALL validate the JobDescription against the Pydantic version 2 model before returning it.
9. IF the LLM classification returns malformed JSON after the Ollama_Client retry attempts are exhausted, THEN THE JD_Parser SHALL apply the defined fallback values and produce a JobDescription that validates against the Pydantic version 2 model.

### Requirement 4: Enrich Candidate Career Trajectory

**User Story:** As an evaluator, I want each candidate's career trajectory quantified, so that growth and seniority are reflected in scoring beyond a static snapshot.

#### Acceptance Criteria

1. WHEN a CandidateProfile is enriched, THE TrajectoryEnricher SHALL compute a Trajectory_Vector containing `growth_rate`, `leadership_progression`, and `tenure_consistency` each as a value in the inclusive range 0.0 to 1.0, and `complexity_arc` as one of the values `ascending`, `descending`, `stable`, or `mixed`.
2. WHEN a CandidateProfile is enriched, THE TrajectoryEnricher SHALL invoke the Ollama_Client to derive a `seniority_score` on a 0 to 10 scale.
3. THE TrajectoryEnricher SHALL clamp the `seniority_score` to the inclusive range 0 to 10.
4. IF the `seniority_score` cannot be parsed from the Ollama_Client response, THEN THE TrajectoryEnricher SHALL assign the default `seniority_score` of 5.0 and continue without raising an unhandled exception.
5. IF a candidate has years_experience of 0 for `growth_rate`, fewer than 2 distinct company_size_estimate values for `complexity_arc`, or zero roles for the remaining metrics, THEN THE TrajectoryEnricher SHALL assign the defined default value for the affected metric, where `tenure_consistency` defaults to 1.0, `growth_rate` defaults to 0.0, `leadership_progression` defaults to 0.0, and `complexity_arc` defaults to `stable`.
6. WHEN the Trajectory_Vector computation completes, THE TrajectoryEnricher SHALL attach the computed Trajectory_Vector to the CandidateProfile.

### Requirement 5: Embed and Store Vectors

**User Story:** As a system integrator, I want job requirements, candidate profiles, and calibration examples embedded and stored locally, so that scoring can retrieve relevant context through similarity search.

#### Acceptance Criteria

1. WHEN the VectorStoreManager is initialized, THE VectorStoreManager SHALL create three ChromaDB collections named `jd_requirements`, `candidate_profiles`, and `calibration_examples`, each configured with `embedding_function=None`.
2. WHEN content is submitted for storage, THE VectorStoreManager SHALL compute embeddings using the Embedding_Model and store the content by manually injecting the computed vectors into ChromaDB.
3. WHEN a candidate is embedded, THE VectorStoreManager SHALL store the candidate as two chunks in the `candidate_profiles` collection: a profile_summary chunk and a skills chunk, each carrying metadata `candidate_id` and `chunk_type`.
4. WHEN a query is submitted to the JobDescription retrieval method, THE VectorStoreManager SHALL embed the query using `query_embeddings` and return up to the 5 most similar JobDescription context document strings ranked in descending order of embedding similarity.
5. WHEN a query is submitted to the Calibration_Example retrieval method, THE VectorStoreManager SHALL embed the query using `query_embeddings` and return up to the 3 most similar Calibration_Example metadata dicts containing `outcome` and `reason` ranked in descending order of embedding similarity.
6. THE VectorStoreManager SHALL store exactly 10 Calibration_Examples comprising 5 labeled `strong_hire` and 5 labeled `no_hire`.
7. IF embedding computation fails while storing content, THEN THE VectorStoreManager SHALL abort the store operation, leave the target collection unchanged, and report an embedding-failure error.

### Requirement 6: Score Candidates with Multi-Persona Panel

**User Story:** As a hiring manager, I want each candidate scored by a panel of distinct evaluator personas with retrieved context, so that I receive a balanced, defensible composite score and narrative.

#### Acceptance Criteria

1. WHEN a candidate is scored, THE CandidateScoringPipeline SHALL evaluate the candidate through three personas: `hiring_manager`, `peer_interviewer`, and `devils_advocate`, running the three Ollama calls sequentially and producing one persona score in the inclusive range 0 to 10 for each persona.
2. WHEN a persona evaluates a candidate, THE CandidateScoringPipeline SHALL provide the persona with the 5 most similar retrieved JobDescription context items and the 3 most similar retrieved Calibration_Example context items as RAG input.
3. THE CandidateScoringPipeline SHALL compute the Composite_Score as the weighted sum of the three persona scores using the weights 0.45 for hiring_manager, 0.35 for peer_interviewer, and -0.20 for devils_advocate, rounded to 2 decimal places.
4. THE CandidateScoringPipeline SHALL clamp the Composite_Score to the inclusive range 0 to 10.
5. THE CandidateScoringPipeline SHALL compute the Panel_Variance as the population variance of the three persona scores, equal to the mean of the squared deviations of the three persona scores from their mean.
6. IF the Panel_Variance exceeds 2.5, THEN THE CandidateScoringPipeline SHALL set `requires_human_review` to true.
7. WHILE the Panel_Variance is 2.5 or below, THE CandidateScoringPipeline SHALL set `requires_human_review` to false.
8. IF a persona response cannot be parsed as JSON, THEN THE CandidateScoringPipeline SHALL retry that persona evaluation exactly once.
9. IF a persona response cannot be parsed as JSON after the retry, THEN THE CandidateScoringPipeline SHALL substitute the defined default verdict result, including the defined default persona score, for that persona and SHALL continue scoring the remaining personas without raising an exception.
10. THE CandidateScoringPipeline SHALL generate a narrative of exactly three sentences summarizing the evaluation via an Ollama call.
11. THE CandidateScoringPipeline SHALL return a result containing the candidate identity as candidate_id and name, the trajectory_score, the three persona scores, the Composite_Score, the Panel_Variance, `requires_human_review`, the persona verdicts, a strengths list with duplicate entries removed, a concerns list with duplicate entries removed, the narrative, the bias_flag, and the counterfactual_delta.

### Requirement 7: Counterfactual Fairness Audit

**User Story:** As a compliance reviewer, I want each candidate re-scored after swapping demographic signals, so that I can detect and report potential bias in the scoring.

#### Acceptance Criteria

1. WHEN a candidate is audited, THE CounterfactualAuditor SHALL create a Counterfactual_Twin by swapping the candidate name using the configured name pairs, swapping gendered pronouns (he/him/his ↔ she/her/hers, whole-word and case-insensitive), and swapping institutions using the configured institution swaps, leaving any name, pronoun, or institution token that has no configured swap unchanged.
2. WHEN a Counterfactual_Twin is created, THE CounterfactualAuditor SHALL re-parse and re-enrich the twin and re-score it through the CandidateScoringPipeline using the same Vector_Store, assigning the twin a `candidate_id` equal to the original candidate_id prefixed with `cf_`.
3. THE CounterfactualAuditor SHALL compute the Counterfactual_Delta as the absolute difference between the original Composite_Score and the twin Composite_Score, rounded to 2 decimal places.
4. IF the Counterfactual_Delta exceeds 0.75, THEN THE CounterfactualAuditor SHALL set the Bias_Flag to true.
5. WHILE the Counterfactual_Delta is 0.75 or below, THE CounterfactualAuditor SHALL set the Bias_Flag to false.
6. WHEN a candidate is audited, THE CounterfactualAuditor SHALL update the original result in place with `bias_flag` and `counterfactual_delta`, and SHALL append the audit result to the Audit_Log.
7. WHEN the audit completes, THE CounterfactualAuditor SHALL write the Audit_Report to `bias_audit_report.json` including the total audited count, the flagged count, the flag rate computed as flagged count divided by total audited (0 when total audited is 0), the bias threshold, the flagged candidates, the clean candidate count, and a methodology note.
8. WHERE the audit is skipped by operator request, THE System SHALL complete the pipeline without performing the counterfactual fairness audit and SHALL write an Audit_Report to `bias_audit_report.json` documenting that no bias analysis was performed.
9. IF re-parsing, re-enriching, or re-scoring a Counterfactual_Twin fails, THEN THE CounterfactualAuditor SHALL record an audit-failure entry for that candidate, SHALL NOT set the Bias_Flag to true, and SHALL continue auditing the remaining candidates.

### Requirement 8: Produce Ranked Candidate Output

**User Story:** As a recruiter, I want a ranked CSV of all candidates with their scores and audit results, so that I can review and act on the rankings.

#### Acceptance Criteria

1. WHEN scoring completes for all candidates, THE System SHALL sort candidates by Composite_Score in descending order, breaking ties by candidate_id in ascending order.
2. WHEN candidates are sorted, THE System SHALL assign each candidate a unique rank that is a consecutive integer starting at 1 and incrementing by 1 in sorted order.
3. THE System SHALL write the Ranked_Output to `ranked_candidates.csv` using pandas.
4. THE Ranked_Output SHALL contain a header row plus exactly one row per scored candidate, with the columns in this order: rank, candidate_id, name, composite_score, trajectory_score, hiring_manager_score, peer_interviewer_score, devils_advocate_score, panel_variance, requires_human_review, verdict_consensus, strengths, concerns, narrative, bias_flag, counterfactual_delta, where the strengths and concerns columns are pipe-separated and an empty list is written as an empty string.
5. THE verdict_consensus SHALL be the verdict assigned by at least two of the three personas.
6. IF no verdict is assigned by at least two of the three personas, THEN THE verdict_consensus SHALL be the hiring_manager persona's verdict.
7. IF writing the Ranked_Output file fails, THEN THE System SHALL report an error and exit without leaving a partial `ranked_candidates.csv` file.

### Requirement 9: Command-Line Interface

**User Story:** As an operator, I want a single command-line entry point with clear options and progress feedback, so that I can run the full pipeline and see results in the terminal.

#### Acceptance Criteria

1. THE CLI SHALL accept the arguments `--candidates-dir`, `--job-description`, `--output-dir`, `--skip-audit`, and `--verbose` via argparse.
2. WHEN the CLI starts, THE CLI SHALL verify within a 10-second connection timeout that the Ollama_Server is running and reachable.
3. IF the Ollama_Server is not reachable within the 10-second connection timeout, THEN THE CLI SHALL report the failure with setup instructions and exit without running the pipeline.
4. WHEN the CLI runs, THE CLI SHALL load and parse the JobDescription and the candidate files with `.json`, `.pdf`, and `.docx` extensions from the candidates directory.
5. IF the candidates directory does not exist or contains no candidate files, THEN THE CLI SHALL fall back to `data/sample_candidates.json`.
6. WHEN the pipeline runs, THE CLI SHALL embed the JobDescription and the Calibration_Examples before scoring.
7. WHILE the pipeline is processing candidates, THE CLI SHALL display a rich progress bar indicating the number of candidates processed out of the total candidate count.
8. WHEN scoring completes, THE CLI SHALL display a rich table of the top 10 candidates ranked by composite_score, or all candidates ranked by composite_score when fewer than 10 candidates were scored, showing rank, name, composite_score, verdict_consensus, requires_human_review, and bias_flag.
9. WHEN the pipeline completes, THE CLI SHALL display a summary containing the total candidate count, the elapsed time in seconds, and the output file paths.
10. WHERE `--skip-audit` is provided, THE CLI SHALL run the pipeline without the counterfactual fairness audit.
11. WHERE `--verbose` is provided, THE CLI SHALL emit DEBUG level log output.
12. WHILE `--verbose` is not provided, THE CLI SHALL emit log output at the INFO level and above.

### Requirement 10: Centralized Configuration

**User Story:** As a maintainer, I want all tunable constants in a single configuration module, so that behavior can be adjusted without editing pipeline logic.

#### Acceptance Criteria

1. THE Config SHALL define `OLLAMA_MODEL`, `OLLAMA_HOST`, `EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`, the token limits `extraction`, `scoring`, and `narrative` each as a positive integer greater than 0, and `SCORING_TEMPERATURE` as a numeric value within the inclusive range 0.0 to 1.0.
2. THE Config SHALL define `PERSONA_WEIGHTS` as 0.45 for `hiring_manager`, 0.35 for `peer_interviewer`, and -0.20 for `devils_advocate`, `BIAS_FLAG_THRESHOLD` set to 0.75, and `HUMAN_REVIEW_VARIANCE_THRESHOLD` set to 2.5.
3. THE Config SHALL define `TITLE_LEVELS`, `COUNTERFACTUAL_NAME_PAIRS`, and `INSTITUTION_SWAPS`, each containing at least one entry.
4. THE Config SHALL define `CALIBRATION_EXAMPLES` containing exactly 5 `strong_hire` and 5 `no_hire` entries for a total of 10 entries.
5. THE System SHALL read every tunable constant enumerated in acceptance criteria 1 through 4 exclusively from the Config and SHALL contain no literal redefinition of those constants in any module outside the Config, excluding prompt template text and persona files.

### Requirement 11: Sample Data

**User Story:** As a new user, I want bundled sample data, so that I can run and evaluate the system without supplying my own resumes.

#### Acceptance Criteria

1. THE System SHALL include exactly 15 sample candidate profiles distributed as 3 strong matches, 3 partial matches, 3 overqualified candidates, 3 underqualified candidates, and 3 non-traditional career paths, with each profile assigned to exactly one of these five categories.
2. THE System SHALL include a sample job description for a Senior Software Engineer role at a Series B SaaS company that validates against the JobDescription Pydantic version 2 model.
3. THE sample job description SHALL include at least one requirement entry in each of the buckets `must_have`, `nice_to_have`, `culture_signal`, and `seniority_marker`.
4. THE sample candidate profiles SHALL each validate against the CandidateProfile Pydantic version 2 model.
5. THE sample candidate profiles SHALL contain, for each employment entry, a start date on or before its end date and a `duration_months` value equal to the whole number of months between that start date and end date.

### Requirement 12: Automated Testing

**User Story:** As a developer, I want a deterministic test suite that mocks all LLM calls, so that I can verify pipeline correctness without a running Ollama server or network access.

#### Acceptance Criteria

1. THE System SHALL include a test that parses a `.json` resume with optional fields omitted and asserts the resulting CandidateProfile validates against the Pydantic version 2 model with the defined default values populated for the omitted fields.
2. THE System SHALL include a test that asserts JobDescription classification produces at least one requirement entry in each of the buckets `must_have`, `nice_to_have`, `culture_signal`, and `seniority_marker`.
3. THE System SHALL include tests that assert candidate embedding stores exactly two chunks in the `candidate_profiles` collection, that retrieval returns non-empty results, and that the Calibration_Example count equals 10.
4. THE System SHALL include a test that asserts the score result contains all required schema fields and that the Counterfactual_Delta is a float greater than or equal to 0.
5. THE System SHALL mock all Ollama_Client calls in tests via `unittest.mock.patch` on `ollama.chat`.
6. WHILE the test suite runs, THE System SHALL make no real Ollama server calls and no network calls.
7. WHILE no network access is available, THE System SHALL run the full test suite to completion.
8. WHEN the test suite is run more than once on unchanged inputs, THE System SHALL produce identical test results across runs.
9. FOR ALL Composite_Score test inputs, scoring then clamping SHALL produce a value within the inclusive range 0 to 10 (property).
10. FOR ALL Panel_Variance values strictly greater than 2.5, the score result SHALL set `requires_human_review` to true (property).

### Requirement 13: Code Quality Standards

**User Story:** As a maintainer, I want consistent documentation, typing, logging, and resilient external calls, so that the codebase is reliable and maintainable.

#### Acceptance Criteria

1. THE System SHALL include a non-empty module docstring in every module.
2. THE System SHALL document every public function with a docstring containing an Args section that describes each parameter (or states that the function takes no parameters) and a Returns section that describes the return value (or states that the function returns None).
3. THE System SHALL include type hints for every parameter and for the return value on all function signatures.
4. THE System SHALL route all progress and diagnostic output through the Python logging library and SHALL NOT use print statements for such output.
5. WHEN the pipeline begins or completes a pipeline phase, or completes processing a candidate, THE System SHALL emit progress information at the INFO log level.
6. WHEN the Ollama_Client makes an LLM call, THE Ollama_Client SHALL emit the request and response detail of that call at the DEBUG log level.
7. WHEN an Ollama_Client call fails by raising an exception or by not returning a response, THE Ollama_Client SHALL retry the call up to 2 additional times, waiting before each retry a backoff delay implemented with `time.sleep` that starts at 1 second and doubles before each subsequent retry.
8. IF an Ollama_Client call still fails after the initial attempt and the 2 retry attempts, THEN THE Ollama_Client SHALL raise an exception indicating that all attempts failed.
9. IF JSON parsing of an LLM response fails after the retry attempts are exhausted, THEN THE System SHALL substitute the defined fallback value for that response and SHALL continue processing without raising an exception.
10. IF an error other than a JSON parsing failure occurs while handling an LLM response, THEN THE System SHALL propagate that exception rather than substituting a fallback value.
