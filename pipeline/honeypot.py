"""Honeypot detection for competition-mode candidate ranking.

Identifies fake or low-quality candidates ("honeypots") using structural
anomalies that do not require LLM inference:

* **Career timeline impossibility** — overlapping roles that imply >24h/day
  of work, or career spans exceeding ``years_of_experience``.
* **Skill inconsistency** — claiming "expert" level with few endorsements,
  or listing skills with no related career history.
* **YOE–career mismatch** — ``years_of_experience`` wildly inconsistent with
  the actual career_history span.
* **Statistical outliers** — extreme Redrob signal values that fall outside
  plausible human ranges.

A honeypot flag is binary: a candidate either is or is not a honeypot.  The
severity (0.0–1.0) indicates how strongly the evidence points to a fake
profile and is used downstream as a soft penalty on the composite score.
"""

from __future__ import annotations

import logging
from datetime import datetime

import config
from models.candidate import CandidateProfile

logger = logging.getLogger(__name__)


def _parse_month(date_str: str) -> tuple[int, int] | None:
    """Parse an ISO-ish date string to ``(year, month)``.

    Accepts ``"2021-03"``, ``"2021-03-01"``, ``"2021"``.
    Returns ``None`` on failure.
    """
    if not date_str:
        return None
    try:
        parts = date_str[:10].split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        return (year, max(1, min(12, month)))
    except (ValueError, IndexError):
        return None


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    """Absolute month difference between two ``(year, month)`` tuples."""
    return abs((end[0] - start[0]) * 12 + (end[1] - start[1]))


def detect_timeline_impossibility(profile: CandidateProfile) -> tuple[bool, float]:
    """Detect overlapping roles that cannot coexist in calendar time.

    Returns:
        ``(flagged, severity)`` where *flagged* is ``True`` when any pair of
        roles overlaps by more than the configured threshold, and *severity*
        is in [0, 1] scaled by the maximum overlap ratio.
    """
    roles = profile.roles
    if len(roles) < 2:
        return False, 0.0

    parsed: list[tuple[str, tuple[int, int] | None, tuple[int, int] | None]] = []
    for r in roles:
        s = _parse_month(r.start_date)
        e = _parse_month(r.end_date) if r.end_date else None
        parsed.append((r.title, s, e))

    max_overlap_months = 0
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            _, s1, e1 = parsed[i]
            _, s2, e2 = parsed[j]
            if not s1 or not s2:
                continue
            end1 = e1 or (2026, 6)
            end2 = e2 or (2026, 6)
            overlap_start = max(s1, s2)
            overlap_end = min(end1, end2)
            if overlap_start < overlap_end:
                overlap = _months_between(overlap_start, overlap_end)
                max_overlap_months = max(max_overlap_months, overlap)

    threshold = config.HONEYPOT_CAREER_IMPOSSIBILITY_THRESHOLD * 12
    if max_overlap_months > threshold:
        severity = min(1.0, max_overlap_months / (threshold * 2))
        return True, severity

    return False, 0.0


def detect_skill_inconsistency(profile: CandidateProfile) -> tuple[bool, float]:
    """Detect skill claims unsupported by endorsements or career history.

    Flags candidates who list many "expert" skills with near-zero
    endorsements, or who claim skills entirely unrelated to their roles.
    """
    skills = profile.skills_claimed
    if not skills:
        return False, 0.0

    signals = profile.redrob_signals
    if signals is None:
        return False, 0.0

    avg_endorsements = (
        signals.endorsements_received / len(skills) if skills else 0
    )
    low_endorsement_ratio = sum(
        1 for s in skills
        if signals.skill_assessment_scores.get(s.lower(), 0) > 70
        and avg_endorsements < config.HONEYPOT_SKILL_CONSISTENCY_MIN
    ) / len(skills)

    role_text = " ".join(
        f"{r.title} {r.company} {' '.join(r.scope_keywords)}"
        for r in profile.roles
    ).lower()

    orphan_skills = sum(
        1 for s in skills
        if s.lower() not in role_text and len(s) > 3
    ) / len(skills)

    severity = 0.6 * low_endorsement_ratio + 0.4 * orphan_skills
    flagged = severity > 0.4
    return flagged, min(1.0, severity)


def detect_yoe_career_mismatch(profile: CandidateProfile) -> tuple[bool, float]:
    """Detect ``years_of_experience`` inconsistent with career_history span.

    If the earliest role start and latest role end span N years but the
    profile claims Y years of experience, and |N - Y| exceeds the threshold,
    the profile is flagged.
    """
    if not profile.roles:
        return False, 0.0

    starts = [_parse_month(r.start_date) for r in profile.roles]
    ends = [_parse_month(r.end_date) for r in profile.roles if r.end_date]

    starts = [s for s in starts if s]
    ends = [e for e in ends if e]

    if not starts:
        return False, 0.0

    earliest = min(starts)
    latest = max(ends) if ends else (2026, 6)
    span_years = _months_between(earliest, latest) / 12.0

    claimed = profile.years_experience
    if claimed <= 0:
        return False, 0.0

    diff = abs(span_years - claimed)
    if diff > config.HONEYPOT_YOE_CAREER_MISMATCH_YEARS:
        severity = min(1.0, diff / (config.HONEYPOT_YOE_CAREER_MISMATCH_YEARS * 3))
        return True, severity

    return False, 0.0


def detect_statistical_outliers(profile: CandidateProfile) -> tuple[bool, float]:
    """Detect Redrob signal values outside plausible human ranges.

    Extreme values (e.g. ``github_activity_score > 95`` combined with
    ``publications_count > 20``) suggest fabricated data.
    """
    sig = profile.redrob_signals
    if sig is None:
        return False, 0.0

    flags = 0
    total = 0

    if sig.github_activity_score > 95 and sig.github_public_repos > 40:
        flags += 1
    total += 1

    if sig.publications_count > 15 and profile.years_experience < 3:
        flags += 1
    total += 1

    if sig.recruiter_outreach_count_6m > 100 and sig.avg_response_time_hours < 0.5:
        flags += 1
    total += 1

    if sig.interview_completion_rate > 0.95 and sig.applications_submitted_30d > 50:
        flags += 1
    total += 1

    severity = flags / total if total else 0.0
    return flags >= 2, severity


def compute_honeypot_score(profile: CandidateProfile) -> tuple[bool, float]:
    """Run all honeypot detectors and return aggregate flag + severity.

    Returns:
        ``(is_honeypot, severity)`` where *severity* is the weighted average
        of individual detector severities.  ``is_honeypot`` is ``True`` when
        any single detector flags with severity > 0.5, or when two or more
        detectors fire regardless of severity.
    """
    detectors = [
        detect_timeline_impossibility,
        detect_skill_inconsistency,
        detect_yoe_career_mismatch,
        detect_statistical_outliers,
    ]

    fired = 0
    total_severity = 0.0
    max_single_severity = 0.0

    for detector in detectors:
        flagged, severity = detector(profile)
        total_severity += severity
        max_single_severity = max(max_single_severity, severity)
        if flagged:
            fired += 1

    avg_severity = total_severity / len(detectors)
    is_honeypot = (max_single_severity > 0.5) or (fired >= 2)

    return is_honeypot, min(1.0, avg_severity)
