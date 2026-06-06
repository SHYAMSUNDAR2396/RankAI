"""Candidate-side Pydantic v2 data models for the Candidate Ranking System.

This module defines the three structured schemas that represent a candidate as
the pipeline ingests, enriches, scores, and audits them:

* :class:`CandidateRole` -- one employment entry in a candidate's history.
* :class:`TrajectoryVector` -- the computed career-trajectory metrics attached
  during the ENRICH phase.
* :class:`CandidateProfile` -- a candidate's full structured profile.

The field names mirror the canonical requirement schema (``roles``,
``skills_claimed``, ``education``, ``trajectory_vector``, ``raw_text``,
``source_file``) so the INGEST, ENRICH, EMBED & STORE, SCORE, and AUDIT phases
all integrate against a single shared vocabulary.

Every optional field carries an explicit default. This is deliberate: a partial
source (a sparse ``.json`` resume or an LLM extraction that omits fields) must
still produce a model that validates, with the omitted fields populated by their
defined defaults (Requirements 2.7, 11.4). The only required fields are the ones
a profile is meaningless without: a role's ``title``/``company`` and a profile's
``candidate_id``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateRole(BaseModel):
    """One employment entry in a candidate's work history.

    Dates are stored as plain strings (simple ISO-like values such as
    ``"2021-03"`` or ``"2021-03-01"``) per the requirement schema, keeping
    ingestion tolerant of the loosely formatted dates that appear in resumes.

    Attributes:
        title: The role/job title. Required.
        company: The employing organization. Required.
        start_date: Role start date as an ISO-like string.
        end_date: Role end date as an ISO-like string, or ``None`` for the
            candidate's current (ongoing) role. Defaults to ``None``.
        duration_months: Whole number of months the role spanned. Defaults
            to ``0``.
        company_size_estimate: Coarse company-size bucket such as
            ``"startup <50"``, ``"scaleup 50-500"``, or ``"enterprise 500+"``.
            Consumed by ``TrajectoryEnricher.compute_complexity_arc``. Defaults
            to ``None`` when unknown.
        scope_keywords: Free-form keywords describing the scope/responsibilities
            of the role. Consumed by ``TrajectoryEnricher`` when computing
            leadership progression. Defaults to an empty list.
    """

    title: str
    company: str
    start_date: str
    end_date: str | None = None
    duration_months: int = 0
    company_size_estimate: str | None = None
    scope_keywords: list[str] = Field(default_factory=list)


class TrajectoryVector(BaseModel):
    """Computed career-trajectory metrics attached during the ENRICH phase.

    The deterministic metrics are bounded scores; ``seniority_score`` is the
    single LLM-derived value. All fields default to the neutral values the
    ``TrajectoryEnricher`` falls back to for degenerate inputs (zero roles,
    zero years of experience, or fewer than two distinct company sizes), so an
    un-enriched vector is still valid (Requirements 4.1, 4.5).

    Attributes:
        growth_rate: Career growth signal in the inclusive range [0.0, 1.0].
            Defaults to ``0.0``.
        complexity_arc: One of ``"ascending"``, ``"descending"``, ``"stable"``,
            or ``"mixed"``. Defaults to ``"stable"``.
        leadership_progression: Leadership growth signal in the inclusive range
            [0.0, 1.0]. Defaults to ``0.0``.
        tenure_consistency: Tenure-stability signal in the inclusive range
            [0.0, 1.0]. Defaults to ``1.0``.
        seniority_score: LLM-derived seniority on a 0-to-10 scale. Defaults to
            ``5.0`` (the neutral value used when the score cannot be parsed).
    """

    growth_rate: float = 0.0
    complexity_arc: str = "stable"
    leadership_progression: float = 0.0
    tenure_consistency: float = 1.0
    seniority_score: float = 5.0


class CandidateProfile(BaseModel):
    """A candidate's full structured profile.

    Built during INGEST and progressively populated by later phases. Only
    ``candidate_id`` is required; supplying just that id yields a fully valid
    profile with every other field defaulted, which is what lets the
    ``Resume_Parser`` produce a valid model from a partial source
    (Requirements 2.7, 11.4).

    Attributes:
        candidate_id: Unique identifier. A ``uuid4`` for real candidates, or
            ``"cf_<id>"`` for a counterfactual twin. Required.
        name: Candidate name. Defaults to ``"Unknown Candidate"``.
        email: Candidate email, or ``None`` when unresolved. Defaults to
            ``None``.
        years_experience: Total years of professional experience. Defaults to
            ``0.0``.
        roles: Employment history as :class:`CandidateRole` entries. Defaults
            to an empty list.
        skills_claimed: Skills claimed by the candidate. Defaults to an empty
            list.
        education: Education entries as dicts. Defaults to an empty list.
        trajectory_vector: The enriched :class:`TrajectoryVector` data stored as
            a dict (e.g. ``trajectory_vector["seniority_score"]`` feeds the
            CSV's ``trajectory_score``), or ``None`` before enrichment. Defaults
            to ``None``.
        raw_text: The original resume text, retained so the audit phase can
            re-parse a swapped twin. Defaults to an empty string.
        source_file: Path/name of the source resume file. Defaults to an empty
            string.
        is_complete: Completion flag set ``True`` once the profile is
            successfully built (Requirement 2.4). Defaults to ``False``.
    """

    candidate_id: str
    name: str = "Unknown Candidate"
    email: str | None = None
    years_experience: float = 0.0
    roles: list[CandidateRole] = Field(default_factory=list)
    skills_claimed: list[str] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)
    trajectory_vector: dict | None = None
    raw_text: str = ""
    source_file: str = ""
    is_complete: bool = False
