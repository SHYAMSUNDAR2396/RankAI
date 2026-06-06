# Methodology

This document explains how the Candidate Ranking System decides who rises to the
top of a stack of resumes, and why it is built the way it is. Every design choice
trades against a single goal: produce a ranking a hiring manager can trust *and
interrogate*, on a stack that costs nothing to run and keeps candidate data on the
operator's own machine.

## 1. Problem framing

Keyword filters fail because resumes are prose, not databases. A boolean search for
"Kubernetes" rejects the engineer who wrote "operated our container orchestration
platform" and accepts the one who listed Kubernetes once in a skills footer they
never touched. Keyword matching has no notion of depth, ownership, or trajectory,
and it offers no explanation beyond "matched" or "did not match."

The deeper problem is the explainability gap. Most automated screening produces a
score with no defensible reasoning, which means a recruiter cannot challenge it and
a candidate cannot appeal it. "The right candidate" is not the one with the most
keyword hits; it is the one whose demonstrated experience, growth, and scope best
fit the specific role, judged the way a thoughtful hiring panel would judge them.
This system models that panel explicitly and shows its work.

## 2. Architecture overview

The pipeline runs five phases in strict sequence: **INGEST → ENRICH → EMBED & STORE
→ SCORE → COUNTERFACTUAL FAIRNESS AUDIT**.

- **INGEST** parses each resume (PDF, DOCX, or JSON) into a validated
  `CandidateProfile` and the job description into a `JobDescription` with classified
  requirement buckets.
- **ENRICH** computes a career `Trajectory_Vector` for each candidate.
- **EMBED & STORE** writes job requirements, candidate chunks, and calibration
  examples into three local ChromaDB collections.
- **SCORE** evaluates each candidate through a three-persona panel using retrieved
  context, producing a composite score, panel variance, verdicts, and a narrative.
- **COUNTERFACTUAL FAIRNESS AUDIT** rebuilds a demographically swapped twin of each
  candidate, re-scores it, and flags large scoring deltas.

The sequence matters because each phase depends on the structured output of the one
before it: you cannot score what you have not enriched and embedded, and you cannot
audit a delta you have not yet scored. The system emits two artifacts:
`ranked_candidates.csv` and `bias_audit_report.json`.

## 3. Free and local stack rationale & model selection strategy

All inference runs locally: **Ollama** serves local LLMs, **sentence-transformers** runs `BAAI/bge-large-en-v1.5` for embeddings, and **ChromaDB** persists vectors on disk. We chose this over cloud APIs deliberately. Local inference means reproducibility (the same model weights every run), privacy (candidate resumes never leave the machine), zero marginal cost, and no rate limits.

Rather than a single model for all tasks, RankAI assigns each agent the model best suited to its cognitive load:

| Agent | Model | Rationale |
|---|---|---|
| Resume + JD parser | qwen2.5:7b | Best open-source JSON adherence |
| Orchestrator | llama3.1:8b | Strong multi-step reasoning |
| Skills match | llama3.2:3b | RAG does heavy lifting; list comparison only |
| Trajectory scorer | deepseek-r1:7b | Reasoning model; infers growth from sparse data |
| Hiring manager | llama3.1:8b | Highest weight; needs nuanced judgment |
| Peer interviewer | qwen2.5:14b | Technical depth requires larger knowledge base |
| Devil's advocate | deepseek-r1:7b | Adversarial reasoning; chain-of-thought finds gaps |
| Narrative | llama3.2:3b | Short text generation; smallest capable model |

A hardware-aware `LOW_MEMORY_MODE` flag substitutes lighter models on machines with less than 20GB RAM (e.g., apple/intel 16GB machines), ensuring the pipeline runs on a wider range of hardware without changing the scoring architecture.

## 4. RAG design

Three ChromaDB collections separate concerns: `jd_requirements` (what the role
needs), `candidate_profiles` (each candidate stored as a `profile_summary` chunk and
a `skills` chunk), and `calibration_examples` (labeled reference candidates). At
scoring time, the system retrieves the **5** most similar JD context items and the
**3** most similar calibration items for the candidate being judged.

Retrieving JD chunks grounds scoring better than pasting a full job description into
the prompt: it surfaces the requirements most relevant to *this* candidate, keeps the
prompt short enough for a small model to attend to, and avoids diluting the signal
with boilerplate. The calibration examples act as few-shot anchors. The store holds
exactly ten — five `strong_hire` and five `no_hire` — so every scoring call sees
concrete, labeled reference points ("this profile was a strong hire because…") that
calibrate the model's notion of the bar rather than letting it drift.

## 5. Multi-persona scoring panel

Each candidate is evaluated by three personas with distinct lenses, run sequentially
as separate LLM calls. The **hiring_manager** judges strategic fit, trajectory, and
seniority alignment ("would I advance them to a final round?"). The
**peer_interviewer** judges technical credibility and collaboration ("could I work
alongside and learn from them?"). The **devils_advocate** builds the strongest case
*against* hiring, citing gaps, short tenures, and overstated claims.

The composite combines them as a weighted sum:

```
composite = 0.45 * hiring_manager + 0.35 * peer_interviewer - 0.20 * devils_advocate
```

The weights live in `config.PERSONA_WEIGHTS`, with the devil's advocate carrying a
**negative** weight (-0.20) so its score is subtracted as a penalty. The result is
rounded to two decimals and clamped to `[0, 10]`. The hiring manager carries the most
weight because they own the decision; the peer interviewer validates technical
reality; the devil's advocate guards against hype.

Panel disagreement is itself a signal. The system computes `panel_variance` as the
*population variance* of the three persona scores and sets `requires_human_review`
when it exceeds **2.5** (`HUMAN_REVIEW_VARIANCE_THRESHOLD`). High variance means the
personas saw the candidate very differently — exactly the ambiguous case a human
should adjudicate. The hiring manager's narrative explanation matters because a score
alone is not trustworthy; the prose tells the manager *why* the panel landed where it
did.

## 6. Trajectory scoring

The `Trajectory_Vector` captures career shape beyond a static snapshot. `growth_rate`
measures seniority levels crossed per year, treating velocity as a predictor of
future performance. `complexity_arc` classifies the trend in company size
(ascending, descending, stable, or mixed) as a context signal. `leadership_progression`
is the fraction of roles showing leadership scope, `tenure_consistency` rewards stable
tenure patterns, and an LLM-derived `seniority_score` (clamped to `[0, 10]`, defaulting
to 5.0 when unparseable) provides a holistic read. The limitations are real: title
inflation can inflate `growth_rate`, and company-name ambiguity weakens the complexity
arc, so trajectory informs the score rather than dictating it.

## 7. Counterfactual fairness audit

For each candidate the auditor builds a twin by swapping first names (via
`COUNTERFACTUAL_NAME_PAIRS`), gendered pronouns (he/him/his ↔ she/her/hers,
whole-word and case-insensitive), and institutions (via `INSTITUTION_SWAPS`), then
re-parses, re-enriches, and re-scores it through the *same* pipeline with a `cf_`
prefixed id. The `counterfactual_delta` is `abs(original - twin)` composite, rounded
to two decimals, and a `bias_flag` is raised when it exceeds **0.75**
(`BIAS_FLAG_THRESHOLD`). The threshold is set low enough to catch meaningful
sensitivity while tolerating the small jitter of a non-deterministic model.

A flag means the score moved when demographic proxies changed — *sensitivity*, not
confirmed discrimination, and the methodology note in the report says so explicitly.
A clean result does **not** guarantee fairness; it only shows robustness to the
specific swaps tested, which are not exhaustive.

## 8. Limitations and future work

The most significant limitation is the `llama3.2:3b` quality ceiling: a larger model
would reason more reliably about nuanced resumes. Scoring quality also depends on the
calibration examples — weak or unrepresentative anchors propagate into every judgment.
A natural future enhancement is OSINT enrichment, but it must be consent-gated and
paired with a demographic correlation audit before use. Low-risk signals (public
GitHub activity, Stack Overflow contributions, published papers) are defensible;
high-risk signals (social media, inferred demographics) should stay out.

## 9. Explainability design

Strengths and concerns surface as the deduplicated union of all three personas'
outputs, so the report reflects the full debate rather than one viewpoint. The
narrative is generated by a final LLM call constrained to exactly three sentences —
one for the primary strength, one for the main concern, one for what evidence would
raise confidence — short enough to read at a glance yet structured enough to defend a
ranking. Finally, `requires_human_review` explicitly routes high-variance, ambiguous
candidates to human judgment instead of pretending the machine resolved them.
