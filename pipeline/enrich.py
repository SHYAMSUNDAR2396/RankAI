"""Career-trajectory enrichment for the Candidate Ranking System.

This module owns the ENRICH phase. It hosts :class:`TrajectoryEnricher`, which
turns a candidate's employment history into a ``Trajectory_Vector`` made up of
four deterministic metrics plus a single LLM-derived ``seniority_score``.

This file implements the four *pure* deterministic metrics as static methods on
:class:`TrajectoryEnricher`:

* :meth:`TrajectoryEnricher.compute_growth_rate`
* :meth:`TrajectoryEnricher.compute_complexity_arc`
* :meth:`TrajectoryEnricher.compute_leadership_progression`
* :meth:`TrajectoryEnricher.compute_tenure_consistency`

Each is a pure function of its inputs (no I/O, deterministic), which keeps the
trajectory defaults directly unit- and property-testable without a running
Ollama server (Requirements 4.1, 4.5, 12.5). The LLM-backed
:meth:`TrajectoryEnricher.enrich` method derives the ``seniority_score`` via the
injected ``Ollama_Client``, assembles the four deterministic metrics plus the
seniority score into a ``Trajectory_Vector``, and attaches it to the
``CandidateProfile`` (Requirements 4.2, 4.3, 4.4, 4.6).

Seniority levels and their ordinal values are read exclusively from
``config.TITLE_LEVELS`` so this module redefines no tunable constants
(Requirement 10.5).
"""

from __future__ import annotations

import logging
import re
import statistics

import config
from models.candidate import CandidateProfile, CandidateRole
from utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

#: Captures the first signed integer or decimal number in a string, used to
#: parse the LLM's free-form seniority response (e.g. ``"7"``, ``"7.5"``, or
#: ``"around 8 out of 10"``) into a float before clamping (Requirement 4.3).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: Neutral seniority fallback used when the LLM response cannot be parsed or the
#: chat call fails, so enrichment never raises on the seniority step
#: (Requirement 4.4).
_DEFAULT_SENIORITY_SCORE = 5.0

#: Inclusive bounds the seniority score is clamped to (Requirement 4.3).
_SENIORITY_MIN = 0.0
_SENIORITY_MAX = 10.0


#: Keywords that signal leadership scope within a role. This is a behavioral
#: keyword list specific to the leadership-progression heuristic (not a global
#: tunable constant), so it lives here rather than in ``config.py``: it encodes
#: *how* leadership is detected from free-form scope text and titles, which is
#: enrichment logic rather than a cross-cutting setting (Requirement 10.5). The
#: match is a case-insensitive substring test, so stems like ``"manage"`` also
#: catch ``"manager"``/``"managed"`` and ``"lead"`` catches ``"leader"``.
LEADERSHIP_KEYWORDS: list[str] = [
    "lead",
    "manage",
    "mentor",
    "architect",
    "principal",
    "director",
    "head",
    "team",
]

#: Ordered company-size tiers used by ``compute_complexity_arc``. A
#: ``company_size_estimate`` string is mapped to a tier by checking whether one
#: of these substrings appears in its lower-cased value; the associated integer
#: defines the ordering ``startup`` < ``scaleup`` < ``enterprise``.
_COMPANY_SIZE_TIERS: list[tuple[str, int]] = [
    ("startup", 0),
    ("scaleup", 1),
    ("enterprise", 2),
]


def _title_level(title: str) -> int | None:
    """Infer an ordinal seniority level from a role title.

    The title is lower-cased and tested against every key in
    ``config.TITLE_LEVELS`` using a substring match. When several keywords are
    present (for example ``"Senior Engineering Manager"`` matches both
    ``"senior"`` and ``"manager"``), the highest matched level is returned, on
    the assumption that the most senior signal in a title dominates.

    Args:
        title: The role/job title to classify.

    Returns:
        int | None: The highest ordinal level among the matched keywords, or
        ``None`` when no configured keyword appears in the title.
    """
    title_lower = title.lower()
    matched = [
        level for keyword, level in config.TITLE_LEVELS.items() if keyword in title_lower
    ]
    if not matched:
        return None
    return max(matched)


def _company_tier(company_size_estimate: str | None) -> int | None:
    """Map a coarse company-size estimate to an ordered tier.

    The estimate is lower-cased and tested for the substrings ``"startup"``,
    ``"scaleup"``, and ``"enterprise"`` (in that order of increasing tier), so
    free-form values such as ``"startup <50"`` or ``"enterprise 500+"`` are
    recognized.

    Args:
        company_size_estimate: The role's company-size estimate string, or
            ``None`` when unknown.

    Returns:
        int | None: ``0`` for startup, ``1`` for scaleup, ``2`` for enterprise,
        or ``None`` when the value is missing or unrecognized.
    """
    if not company_size_estimate:
        return None
    value = company_size_estimate.lower()
    for keyword, tier in _COMPANY_SIZE_TIERS:
        if keyword in value:
            return tier
    return None


class TrajectoryEnricher:
    """Computes a candidate's ``Trajectory_Vector``.

    The four ``compute_*`` methods are pure, deterministic functions of their
    inputs with no I/O, so the trajectory metrics and their degenerate defaults
    are directly testable (Requirements 4.1, 4.5). :meth:`enrich` orchestrates
    them, derives the LLM-backed ``seniority_score`` through the injected
    :class:`~utils.ollama_client.OllamaClient`, and attaches the assembled
    ``Trajectory_Vector`` to the profile (Requirements 4.2, 4.3, 4.4, 4.6).
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """Initialize the enricher with the LLM wrapper used for seniority.

        Args:
            ollama_client: The :class:`~utils.ollama_client.OllamaClient` used
                to derive the ``seniority_score``. It is the single seam through
                which this component performs LLM inference (Requirement 4.2).

        Returns:
            None.
        """
        self.ollama_client = ollama_client

    def enrich(self, profile: CandidateProfile) -> CandidateProfile:
        """Compute the ``Trajectory_Vector`` and attach it to ``profile``.

        The four deterministic metrics are computed from the candidate's roles
        and years of experience via the pure ``compute_*`` helpers, and the
        ``seniority_score`` is derived from a single Ollama chat call
        (Requirement 4.2). The seniority value is parsed from the response, then
        clamped to the inclusive range ``[0.0, 10.0]`` (Requirement 4.3). If the
        chat call fails or the response cannot be parsed into a number, the
        score falls back to ``5.0`` and enrichment continues without raising
        (Requirement 4.4). The assembled metrics are stored as a dict on
        ``profile.trajectory_vector`` and the mutated profile is returned
        (Requirement 4.6).

        Args:
            profile: The candidate profile to enrich. Mutated in place; its
                ``trajectory_vector`` field is populated with the computed
                metrics.

        Returns:
            CandidateProfile: The same ``profile`` instance, with its
            ``trajectory_vector`` attribute set to the computed trajectory dict.
        """
        growth_rate = self.compute_growth_rate(
            profile.roles, profile.years_experience
        )
        complexity_arc = self.compute_complexity_arc(profile.roles)
        leadership_progression = self.compute_leadership_progression(profile.roles)
        tenure_consistency = self.compute_tenure_consistency(profile.roles)

        seniority_score = self._derive_seniority_score(profile)

        trajectory = {
            "growth_rate": growth_rate,
            "complexity_arc": complexity_arc,
            "leadership_progression": leadership_progression,
            "tenure_consistency": tenure_consistency,
            "seniority_score": seniority_score,
        }
        profile.trajectory_vector = trajectory

        logger.debug(
            "Computed trajectory for candidate %s: %s",
            profile.candidate_id,
            trajectory,
        )
        logger.info(
            "Enriched candidate %s with trajectory vector", profile.candidate_id
        )
        return profile

    def _derive_seniority_score(self, profile: CandidateProfile) -> float:
        """Derive the LLM-backed seniority score for ``profile``.

        Builds the seniority prompt from the candidate's name, years of
        experience, and role titles/companies, then issues a single Ollama chat
        call (Requirement 4.2). The first number in the response is extracted and
        clamped to the inclusive range ``[0.0, 10.0]`` (Requirement 4.3). Any
        failure of the chat call or parsing results in the neutral default of
        ``5.0`` being returned without raising (Requirement 4.4).

        Args:
            profile: The candidate profile whose seniority is being scored.

        Returns:
            float: The parsed, clamped seniority score, or ``5.0`` on any
            failure.
        """
        role_titles_and_companies = ", ".join(
            f"{role.title} at {role.company}" for role in profile.roles
        )
        user_prompt = (
            "Rate this candidate's seniority from 0 to 10. Return ONLY a number. "
            "0 = student/intern, 5 = mid-level, 10 = executive/distinguished "
            f"engineer. Candidate: {profile.name}, {profile.years_experience} "
            f"years, roles: {role_titles_and_companies}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise evaluator. Return only a single number "
                    "from 0 to 10."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.ollama_client.chat(
                messages,
                max_tokens=config.MAX_TOKENS_NARRATIVE,
                model=config.get_model("trajectory"),
            )
            match = _NUMBER_RE.search(response)
            if match is None:
                raise ValueError(
                    f"no number found in seniority response: {response!r}"
                )
            seniority_score = float(match.group())
        except Exception as exc:  # noqa: BLE001 - default to 5.0 on any failure
            logger.warning(
                "Failed to derive seniority score for candidate %s; defaulting "
                "to %.1f (error=%s)",
                profile.candidate_id,
                _DEFAULT_SENIORITY_SCORE,
                exc,
            )
            return _DEFAULT_SENIORITY_SCORE

        return max(_SENIORITY_MIN, min(seniority_score, _SENIORITY_MAX))

    @staticmethod
    def compute_growth_rate(roles: list[CandidateRole], years: float) -> float:
        """Estimate career growth as seniority levels crossed per year.

        Each role's title is mapped to an ordinal seniority level via
        ``config.TITLE_LEVELS`` (see :func:`_title_level`). The number of levels
        crossed is taken as ``max_level - min_level`` across all roles whose
        titles matched a configured keyword, and the growth rate is that count
        divided by ``years``, capped at ``1.0``.

        Returns the default of ``0.0`` when ``years`` is ``0`` (Requirement
        4.5), when ``roles`` is empty, or when no role title matches a
        configured seniority keyword (so no levels can be measured).

        Args:
            roles: The candidate's employment history.
            years: The candidate's total years of professional experience.

        Returns:
            float: The growth rate in the inclusive range [0.0, 1.0].
        """
        if not roles or years <= 0:
            return 0.0

        levels = [
            level
            for role in roles
            if (level := _title_level(role.title)) is not None
        ]
        if not levels:
            return 0.0

        levels_crossed = max(levels) - min(levels)
        growth_rate = levels_crossed / years
        return min(growth_rate, 1.0)

    @staticmethod
    def compute_complexity_arc(roles: list[CandidateRole]) -> str:
        """Classify the trend in company size across a candidate's roles.

        Each role's ``company_size_estimate`` is mapped to an ordered tier
        (``startup`` < ``scaleup`` < ``enterprise``; see :func:`_company_tier`),
        and the recognized tiers are read in the given role order. The sequence
        is then classified:

        * ``"ascending"`` -- non-decreasing with at least one strict increase.
        * ``"descending"`` -- non-increasing with at least one strict decrease.
        * ``"mixed"`` -- any sequence that both rises and falls.

        Returns the default of ``"stable"`` when fewer than 2 *distinct*
        recognizable tiers are present (including zero or one recognizable
        company size), per Requirement 4.5.

        Args:
            roles: The candidate's employment history, in chronological order as
                provided.

        Returns:
            str: One of ``"ascending"``, ``"descending"``, ``"stable"``, or
            ``"mixed"``.
        """
        tiers = [
            tier
            for role in roles
            if (tier := _company_tier(role.company_size_estimate)) is not None
        ]
        if len(set(tiers)) < 2:
            return "stable"

        increases = any(b > a for a, b in zip(tiers, tiers[1:]))
        decreases = any(b < a for a, b in zip(tiers, tiers[1:]))

        if increases and not decreases:
            return "ascending"
        if decreases and not increases:
            return "descending"
        return "mixed"

    @staticmethod
    def compute_leadership_progression(roles: list[CandidateRole]) -> float:
        """Compute the fraction of roles that show leadership scope.

        A role counts as a leadership role when any keyword in
        :data:`LEADERSHIP_KEYWORDS` appears (case-insensitive substring match)
        in its ``scope_keywords`` (joined) or its ``title``.

        Returns the default of ``0.0`` when there are zero roles (Requirement
        4.5).

        Args:
            roles: The candidate's employment history.

        Returns:
            float: The leadership fraction in the inclusive range [0.0, 1.0].
        """
        if not roles:
            return 0.0

        leadership_count = 0
        for role in roles:
            haystack = " ".join([*role.scope_keywords, role.title]).lower()
            if any(keyword in haystack for keyword in LEADERSHIP_KEYWORDS):
                leadership_count += 1

        return leadership_count / len(roles)

    @staticmethod
    def compute_tenure_consistency(roles: list[CandidateRole]) -> float:
        """Compute how consistent a candidate's role tenures are.

        Consistency is ``1 - (population_std_dev / mean)`` over the roles'
        ``duration_months`` (the coefficient of variation subtracted from one),
        clamped to the inclusive range [0.0, 1.0]. Population standard deviation
        (:func:`statistics.pstdev`) is used to match the population-variance
        convention used elsewhere in the design.

        Returns the default of ``1.0`` for zero roles (Requirement 4.5) or a
        single role (a lone tenure is trivially consistent), and also ``1.0``
        when the mean duration is ``0`` (to avoid division by zero).

        Args:
            roles: The candidate's employment history.

        Returns:
            float: The tenure-consistency score in the inclusive range
            [0.0, 1.0].
        """
        if len(roles) <= 1:
            return 1.0

        durations = [role.duration_months for role in roles]
        mean_duration = statistics.mean(durations)
        if mean_duration == 0:
            return 1.0

        coefficient_of_variation = statistics.pstdev(durations) / mean_duration
        consistency = 1.0 - coefficient_of_variation
        return max(0.0, min(consistency, 1.0))
