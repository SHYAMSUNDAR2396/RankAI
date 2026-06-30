"""Deterministic scoring of Redrob behavioral and engagement signals.

This module computes a 0–1 score for a candidate from the 25 Redrob signal
fields provided in the competition JSONL.  No LLM inference is performed —
all scoring is pure arithmetic.

The score is broken into five sub-dimensions that map onto the competition
scoring weights defined in :mod:`config`:

* ``engagement`` — profile completeness, activity recency, presence.
* ``responsiveness`` — response rate, speed, interview completion.
* ``openness`` — open-to-work flag, notice period, work-mode flexibility.
* ``social_proof`` — endorsements, connections, certifications, awards.
* ``technical_signals`` — GitHub activity, skill assessments.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import config
from models.candidate import CandidateProfile, RedrobSignals

logger = logging.getLogger(__name__)


def normalize(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [0, 1] after linear rescale from [lo, hi].

    Values outside [lo, hi] are clamped, not clipped.
    """
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _days_since(iso_date: str | None) -> int | None:
    """Return days elapsed since *iso_date*, or ``None`` on parse failure."""
    if not iso_date:
        return None
    try:
        dt = datetime.fromisoformat(iso_date[:10]).date()
        return (date.today() - dt).days
    except (ValueError, TypeError):
        return None


def _compute_engagement(sig: RedrobSignals) -> float:
    """Profile completeness + activity recency + online presence."""
    completeness = normalize(sig.profile_completeness_score, 0, 100)
    days = _days_since(sig.last_active_date)
    recency = normalize(365 - (days or 365), 0, 365) if days is not None else 0.0
    presence = normalize(sig.online_presence_score, 0, 100)
    views = normalize(sig.profile_views_received_30d, 0, 500)
    return 0.35 * completeness + 0.30 * recency + 0.20 * presence + 0.15 * views


def _compute_responsiveness(sig: RedrobSignals) -> float:
    """Response rate + speed + interview completion."""
    resp_rate = normalize(sig.recruiter_response_rate, 0, 1)
    speed = normalize(72 - sig.avg_response_time_hours, 0, 72)
    interview = normalize(sig.interview_completion_rate, 0, 1)
    return 0.40 * resp_rate + 0.30 * speed + 0.30 * interview


def _compute_openness(sig: RedrobSignals) -> float:
    """Open-to-work flag + notice period + work-mode flexibility."""
    otw = 1.0 if sig.open_to_work_flag else 0.0
    notice = normalize(90 - sig.notice_period_days, 0, 90)
    flexible = 1.0 if sig.preferred_work_mode in ("remote", "flexible") else 0.5
    return 0.50 * otw + 0.25 * notice + 0.25 * flexible


def _compute_social_proof(sig: RedrobSignals) -> float:
    """Endorsements + connections + certifications + awards + publications."""
    endorsements = normalize(sig.endorsements_received, 0, 50)
    connections = normalize(sig.connection_count, 0, 1000)
    certs = normalize(sig.certifications_count, 0, 5)
    awards = normalize(sig.awards_count, 0, 3)
    pubs = normalize(sig.publications_count, 0, 5)
    return 0.30 * endorsements + 0.25 * connections + 0.20 * certs + 0.15 * awards + 0.10 * pubs


def _compute_technical(sig: RedrobSignals) -> float:
    """GitHub activity + skill assessments + repos."""
    gh_score = normalize(sig.github_activity_score, -1, 100)
    repos = normalize(sig.github_public_repos, 0, 50)
    commits = normalize(sig.github_recent_commits_6m, 0, 200)
    skills = sig.skill_assessment_scores
    if skills:
        avg_skill = sum(skills.values()) / len(skills) / 100.0
    else:
        avg_skill = 0.0
    return 0.35 * gh_score + 0.25 * repos + 0.20 * commits + 0.20 * avg_skill


def score_signals(profile: CandidateProfile) -> float:
    """Compute a deterministic 0–1 score from Redrob signals.

    Returns ``0.5`` (neutral) when no signals are available so that
    signal-less candidates are neither penalised nor boosted.

    Args:
        profile: A candidate profile, ideally with ``redrob_signals`` populated.

    Returns:
        Score in [0, 1].
    """
    sig = profile.redrob_signals
    if sig is None:
        return 0.5

    engagement = _compute_engagement(sig)
    responsiveness = _compute_responsiveness(sig)
    openness = _compute_openness(sig)
    social = _compute_social_proof(sig)
    technical = _compute_technical(sig)

    combined = (
        0.25 * engagement
        + 0.20 * responsiveness
        + 0.15 * openness
        + 0.20 * social
        + 0.20 * technical
    )
    return max(0.0, min(1.0, combined))


def score_signals_breakdown(profile: CandidateProfile) -> dict[str, float]:
    """Like :func:`score_signals` but returns per-dimension breakdown.

    Useful for reasoning generation and debugging.
    """
    sig = profile.redrob_signals
    if sig is None:
        return {"engagement": 0.5, "responsiveness": 0.5, "openness": 0.5,
                "social_proof": 0.5, "technical": 0.5, "combined": 0.5}

    engagement = _compute_engagement(sig)
    responsiveness = _compute_responsiveness(sig)
    openness = _compute_openness(sig)
    social = _compute_social_proof(sig)
    technical = _compute_technical(sig)
    combined = (
        0.25 * engagement + 0.20 * responsiveness + 0.15 * openness
        + 0.20 * social + 0.20 * technical
    )
    return {
        "engagement": round(engagement, 4),
        "responsiveness": round(responsiveness, 4),
        "openness": round(openness, 4),
        "social_proof": round(social, 4),
        "technical": round(technical, 4),
        "combined": round(max(0.0, min(1.0, combined)), 4),
    }
