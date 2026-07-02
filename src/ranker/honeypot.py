"""Honeypot and quality-flags detector for the Redrob candidate-ranking pipeline.

Honeypots in the dataset are candidates with subtly impossible profiles:
* "expert" in 10 AI skills with 0 endorsements and 0 career evidence
* 8 years of experience at a company founded 3 years ago
* Career spans wildly inconsistent with stated ``years_of_experience``
* Senior Data Scientist title with 1 year of experience
* Many advanced AI skills but every role is in a non-technical field

Honeypot detection is structural (no LLM) and runs in O(1) per candidate.

Output: ``HoneypotResult`` with a binary ``flag`` plus a 0..1 ``severity`` and
a list of human-readable reasons.  A ``flag=True`` candidate is automatically
demoted out of the top-100 in :mod:`score`; partial-severity candidates get a
soft penalty.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .features import (
    ADVANCED_AI_SKILLS,
    DISQUALIFIER_CONSULTING_FIRMS,
    NON_TECH_TITLE_TOKENS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

#: Hard thresholds (used to *flag* — soft severity is computed below)
IMPOSSIBLE_TIMELINE_OVERLAP_MONTHS = 24  # >2yr of dual-employment overlap
HONEYPOT_YOE_SPAN_TOLERANCE_YEARS = 1.0  # career may exceed yoe by this much
HONEYPOT_EXPERT_SKILL_ENDORSEMENT_THRESHOLD = 2  # "expert" with <=2 endorsements = suspicious
HONEYPOT_EXPERT_SKILL_COUNT_THRESHOLD = 4  # 4+ such skills = strong honeypot
HONEYPOT_TITLE_YOE_GAP_YEARS = 5.0  # "Senior" with <3y exp = suspicious

#: Proficiency ordering (for "expert" detection)
PROFICIENCY_LEVELS = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class HoneypotResult:
    """Result of honeypot detection on a single candidate."""

    flag: bool = False          # binary — should we hard-demote?
    severity: float = 0.0       # 0..1 — how confident we are
    reasons: List[str] = field(default_factory=list)
    # Detailed sub-results (for debugging / reasoning)
    timeline_impossible: bool = False
    yoe_span_mismatch: bool = False
    expert_without_endorsements: bool = False
    title_seniority_inflation: bool = False
    skill_stuffing_count: int = 0


# =============================================================================
# Helpers
# =============================================================================

def _parse_date(d: Optional[str]) -> Optional[date]:
    """Parse ISO-ish date string to a date object. Returns None on failure."""
    if not d:
        return None
    try:
        return datetime.fromisoformat(str(d)[:10]).date()
    except (ValueError, TypeError):
        return None


def _months_between(d1: date, d2: date) -> int:
    """Signed month difference d1 - d2 (positive when d1 is later)."""
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# =============================================================================
# Individual detectors — each returns (flag, severity, reason)
# =============================================================================

def _detect_timeline_impossibility(cand: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Detect overlapping roles that cannot coexist in calendar time.

    The dataset orders career_history reverse-chronologically (current first).
    We check every pair for overlap; >24 months of overlap is a strong
    impossible-profile signal.
    """
    career = cand.get("career_history", [])
    if len(career) < 2:
        return False, 0.0, ""

    parsed: List[Tuple[str, Optional[date], Optional[date]]] = []
    for r in career:
        s = _parse_date(r.get("start_date"))
        e = _parse_date(r.get("end_date"))
        parsed.append((r.get("title", ""), s, e))

    # For each pair, compute overlap in months
    max_overlap = 0
    worst_pair = ("", "")
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            ti, si, ei = parsed[i]
            tj, sj, ej = parsed[j]
            if not si or not sj:
                continue
            end_i = ei or date(2026, 7, 2)
            end_j = ej or date(2026, 7, 2)
            overlap_start = max(si, sj)
            overlap_end = min(end_i, end_j)
            if overlap_start < overlap_end:
                overlap_months = _months_between(overlap_end, overlap_start)
                if overlap_months > max_overlap:
                    max_overlap = overlap_months
                    worst_pair = (ti, tj)

    if max_overlap > IMPOSSIBLE_TIMELINE_OVERLAP_MONTHS:
        severity = min(1.0, max_overlap / 60.0)  # 5yr overlap = max severity
        reason = f"career_overlap_{max_overlap}mo"
        return True, severity, reason
    return False, 0.0, ""


def _detect_yoe_span_mismatch(cand: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Detect YOE wildly inconsistent with career span.

    The dataset provides both ``profile.years_of_experience`` and a list of
    career_history roles. Summing the durations of declared roles should be
    close to (or slightly less than, with gaps) the stated YOE.

    * If the *sum of role durations* far exceeds the stated YOE, the candidate
      is over-claiming (impossible).
    * If the *stated YOE* is far greater than the *earliest-to-latest career
      span*, that's also impossible.
    """
    yoe = float(cand.get("years_of_experience", 0.0) or 0.0)
    career = cand.get("career_history", [])
    if yoe <= 0 or not career:
        return False, 0.0, ""

    # Sum of role durations
    total_role_months = sum(int(r.get("duration_months", 0) or 0) for r in career)
    total_role_years = total_role_months / 12.0

    # Career span (earliest start to latest end)
    starts = [_parse_date(r.get("start_date")) for r in career]
    ends = [_parse_date(r.get("end_date")) for r in career]
    valid_starts = [s for s in starts if s is not None]
    valid_ends = [e for e in ends if e is not None]
    if not valid_starts:
        return False, 0.0, ""
    earliest = min(valid_starts)
    latest_end = max(valid_ends) if valid_ends else date(2026, 7, 2)
    span_months = _months_between(latest_end, earliest)
    span_years = max(0.0, span_months / 12.0)

    # Check 1: role-sums > yoe by a lot (impossible — they worked more years than claimed)
    if total_role_years > yoe + HONEYPOT_YOE_SPAN_TOLERANCE_YEARS * 2:
        severity = min(1.0, (total_role_years - yoe) / 5.0)
        reason = f"role_sum_{total_role_years:.1f}y_exceeds_yoe_{yoe:.1f}y"
        return True, severity, reason

    # Check 2: span < yoe by a HUGE margin (>3 years gap is suspicious).
    # Normal candidates often have gaps in career_history (unreported early
    # roles, education period, etc.) so a 1-2 year gap is expected. Only
    # flag when the mismatch is impossible-level extreme.
    if span_years + 3.0 < yoe:
        gap = yoe - span_years
        severity = min(1.0, gap / 8.0)
        reason = f"yoe_{yoe:.1f}y_exceeds_career_span_{span_years:.1f}y"
        return True, severity, reason

    return False, 0.0, ""


def _detect_expert_without_endorsements(cand: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Detect candidates claiming expert-level skills with no endorsements
    or career history to back it up.

    Strong honeypot signal: HR Manager with "expert" in TensorFlow, PyTorch,
    LangChain, RAG, Fine-tuning, etc. — but no engineering role.
    """
    skills = cand.get("skills", [])
    expert_no_endorsement_count = 0
    expert_skills: List[str] = []
    for s in skills:
        proficiency = s.get("proficiency", "").lower()
        endorsements = int(s.get("endorsements", 0) or 0)
        name = (s.get("name") or "").lower()
        if proficiency == "expert" and endorsements <= HONEYPOT_EXPERT_SKILL_ENDORSEMENT_THRESHOLD:
            # AND the skill is a real AI/ML skill
            if name in ADVANCED_AI_SKILLS:
                expert_no_endorsement_count += 1
                expert_skills.append(name)

    if expert_no_endorsement_count >= HONEYPOT_EXPERT_SKILL_COUNT_THRESHOLD:
        severity = min(1.0, expert_no_endorsement_count / 6.0)
        reason = f"expert_in_{expert_no_endorsement_count}_ai_skills_no_endorsements"
        return True, severity, reason
    return False, 0.0, ""


def _detect_title_seniority_inflation(cand: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Detect 'Senior' / 'Staff' / 'Principal' titles claimed with very low YOE.

    Per the JD: senior engineers who "haven't written production code in the last
    18 months because they've moved into architecture or tech lead roles" are
    flagged; same logic applies to a Senior ML Engineer with 2 years of
    experience. JD explicitly says the role "writes code".
    """
    title = _normalize(cand.get("current_title", ""))
    yoe = float(cand.get("years_of_experience", 0.0) or 0.0)
    headline = _normalize(cand.get("headline", ""))

    # Seniority tokens
    senior_tokens = ["senior", "staff", "principal", "lead", "head", "director", "chief", "vp"]
    is_senior = any(t in title for t in senior_tokens) or any(t in headline for t in senior_tokens)

    if is_senior and yoe < 3.0:
        severity = min(1.0, (3.0 - yoe) / 3.0)
        reason = f"senior_title_with_{yoe:.1f}y_experience"
        return True, severity, reason
    return False, 0.0, ""


def _detect_skill_stuffing(cand: Dict[str, Any]) -> Tuple[int, str]:
    """Count how many advanced AI skills a non-technical title claims.

    This is a *soft* signal — not a binary flag, but a count for the
    downstream scorer to weigh.  HR Manager with 7 advanced AI skills is
    much more likely a honeypot than HR Manager with 1.
    """
    title = _normalize(cand.get("current_title", ""))
    is_non_tech = any(tok in title for tok in NON_TECH_TITLE_TOKENS)
    if not is_non_tech:
        return 0, ""

    skills = cand.get("_all_skill_names", [])
    advanced_count = sum(1 for s in skills if s in ADVANCED_AI_SKILLS)
    return advanced_count, f"non_tech_title_with_{advanced_count}_advanced_ai_skills"


def _detect_unbacked_skill_endorsements(cand: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Detect skills with very high endorsements but no related career evidence.

    If a candidate claims 50+ endorsements on a skill like 'PyTorch' but their
    career history is entirely non-technical (sales, marketing, etc.), that's
    suspicious — endorsements are typically from peers, and the network should
    match the career arc.
    """
    career = cand.get("career_history", [])
    if not career:
        return False, 0.0, ""

    title = _normalize(cand.get("current_title", ""))
    is_non_tech = any(tok in title for tok in NON_TECH_TITLE_TOKENS)
    if not is_non_tech:
        return False, 0.0, ""

    endorsements = cand.get("_all_skill_endorsements", {}) or {}
    high_endorsement_skills = [
        skill for skill, count in endorsements.items()
        if count >= 30 and skill in ADVANCED_AI_SKILLS
    ]
    if len(high_endorsement_skills) >= 2:
        severity = min(1.0, len(high_endorsement_skills) / 4.0)
        reason = f"high_endorsements_{len(high_endorsement_skills)}_ai_skills_non_tech"
        return True, severity, reason
    return False, 0.0, ""


# =============================================================================
# Public API
# =============================================================================

def detect_honeypot(cand: Dict[str, Any]) -> HoneypotResult:
    """Run all honeypot detectors and return an aggregated result.

    A candidate is *flagged* if:
    * timeline_impossibility OR
    * yoe_span_mismatch (with severity >= 0.5) OR
    * expert_without_endorsements OR
    * title_seniority_inflation (with severity >= 0.5) OR
    * unbacked_skill_endorsements OR
    * skill_stuffing_count >= 6

    Args:
        cand: A normalized candidate dict (from :mod:`io._normalize_candidate`).

    Returns:
        :class:`HoneypotResult` with binary flag, severity, and reasons.
    """
    result = HoneypotResult()

    # Detector 1: impossible timeline
    flag1, sev1, reason1 = _detect_timeline_impossibility(cand)
    result.timeline_impossible = flag1
    if flag1:
        result.reasons.append(reason1)
        result.severity = max(result.severity, sev1)

    # Detector 2: YOE / span mismatch
    flag2, sev2, reason2 = _detect_yoe_span_mismatch(cand)
    result.yoe_span_mismatch = flag2
    if flag2:
        result.reasons.append(reason2)
        result.severity = max(result.severity, sev2)

    # Detector 3: expert without endorsements
    flag3, sev3, reason3 = _detect_expert_without_endorsements(cand)
    result.expert_without_endorsements = flag3
    if flag3:
        result.reasons.append(reason3)
        result.severity = max(result.severity, sev3)

    # Detector 4: title seniority inflation
    flag4, sev4, reason4 = _detect_title_seniority_inflation(cand)
    result.title_seniority_inflation = flag4
    if flag4:
        result.reasons.append(reason4)
        result.severity = max(result.severity, sev4)

    # Detector 5: skill stuffing (soft count)
    stuff_count, stuff_reason = _detect_skill_stuffing(cand)
    result.skill_stuffing_count = stuff_count
    if stuff_count >= 6:
        result.reasons.append(stuff_reason)
        result.severity = max(result.severity, min(1.0, stuff_count / 8.0))

    # Detector 6: unbacked high endorsements on AI skills (non-tech)
    flag6, sev6, reason6 = _detect_unbacked_skill_endorsements(cand)
    if flag6:
        result.reasons.append(reason6)
        result.severity = max(result.severity, sev6)

    # Final flag: only high-confidence, rare signals.
    # yoe_span_mismatch alone does NOT flag — career gaps are normal.
    # It only boosts severity when combined with another signal.
    if (
        result.timeline_impossible
        or result.expert_without_endorsements
        or (result.title_seniority_inflation and result.severity >= 0.5)
        or flag6
        or stuff_count >= 6
    ):
        result.flag = True

    return result


def is_safe_for_top100(honeypot_result: HoneypotResult) -> bool:
    """Return True if the candidate is safe to include in the top-100.

    A flagged candidate is auto-demoted; a high-severity non-flagged candidate
    is also demoted (e.g. 5+ skill stuffing signals but not quite 6).
    """
    if honeypot_result.flag:
        return False
    if honeypot_result.severity >= 0.7:
        return False
    return True
