"""Streaming JSONL loader for the competition candidate pool.

Reads candidates.jsonl line by line, yielding lightweight dicts that match the
real competition schema exactly. No Pydantic validation overhead — pure Python
structural mapping for speed and memory efficiency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---- Schema keys for quick access ----
REQUIRED_TOP_KEYS = {
    "candidate_id", "profile", "career_history", "education",
    "skills", "certifications", "languages", "redrob_signals"
}

REQUIRED_PROFILE_KEYS = {
    "anonymized_name", "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_company_size", "current_industry"
}

REQUIRED_CAREER_KEYS = {
    "company", "title", "start_date", "end_date", "duration_months",
    "is_current", "industry", "company_size", "description"
}

REQUIRED_SKILLS_KEYS = {"name", "proficiency", "endorsements", "duration_months"}

REQUIRED_SIGNALS_KEYS = {
    "profile_completeness_score", "signup_date", "last_active_date",
    "open_to_work_flag", "profile_views_received_30d", "applications_submitted_30d",
    "recruiter_response_rate", "avg_response_time_hours", "skill_assessment_scores",
    "connection_count", "endorsements_received", "notice_period_days",
    "expected_salary_range_inr_lpa", "preferred_work_mode", "willing_to_relocate",
    "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d",
    "interview_completion_rate", "offer_acceptance_rate", "verified_email",
    "verified_phone", "linkedin_connected"
}


def load_candidates_jsonl(
    path: str | Path,
    *,
    max_candidates: int | None = None,
    validate: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Stream candidates from a competition-format JSONL file.

    Each line is a JSON object matching the competition candidate_schema.json.
    This function performs NO LLM inference — pure structural mapping.

    Args:
        path: Path to the .jsonl file (or .jsonl.gz if you add gzip.open).
        max_candidates: Optional cap; stop after yielding this many profiles.
        validate: If True, skip lines that fail structural validation.

    Yields:
        Validated candidate dicts. Lines that fail validation are logged and skipped.
    """
    path = Path(path)
    logger.info("Loading candidates from JSONL: %s", path)

    count = 0
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if max_candidates is not None and count >= max_candidates:
                logger.info(
                    "Reached max_candidates=%d — stopping JSONL load",
                    max_candidates,
                )
                break

            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                skipped += 1
                continue

            if validate and not _validate_structure(row):
                logger.warning(
                    "Skipping candidate on line %d: structural validation failed", line_no
                )
                skipped += 1
                continue

            # Map to lightweight internal structure for downstream scoring
            yield _normalize_candidate(row)
            count += 1

    logger.info("Loaded %d candidates, skipped %d", count, skipped)


def _validate_structure(row: Dict[str, Any]) -> bool:
    """Quick structural validation — fail fast on missing required top-level keys."""
    if not isinstance(row, dict):
        return False
    if not REQUIRED_TOP_KEYS.issubset(row.keys()):
        return False
    if not isinstance(row.get("candidate_id"), str) or not row["candidate_id"].startswith("CAND_"):
        return False
    # profile sub-structure
    profile = row.get("profile", {})
    if not isinstance(profile, dict) or not REQUIRED_PROFILE_KEYS.issubset(profile.keys()):
        return False
    # career_history is array
    career = row.get("career_history", [])
    if not isinstance(career, list) or len(career) == 0:
        return False
    # redrob_signals
    signals = row.get("redrob_signals", {})
    if not isinstance(signals, dict) or not REQUIRED_SIGNALS_KEYS.issubset(signals.keys()):
        return False
    return True


def _normalize_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten/normalize the competition row to a scoring-friendly dict."""
    profile = row["profile"]
    career = row["career_history"]
    education = row.get("education", [])
    skills = row.get("skills", [])
    certs = row.get("certifications", [])
    languages = row.get("languages", [])
    signals = row["redrob_signals"]

    return {
        "candidate_id": row["candidate_id"],
        # profile
        "name": profile.get("anonymized_name", "Unknown"),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "years_of_experience": float(profile.get("years_of_experience", 0.0)),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_company_size": profile.get("current_company_size", ""),
        "current_industry": profile.get("current_industry", ""),
        # career_history (list of roles)
        "career_history": career,
        # education
        "education": education,
        # skills (list of {name, proficiency, endorsements, duration_months})
        "skills": skills,
        # certifications
        "certifications": certs,
        # languages
        "languages": languages,
        # redrob_signals (23 fields)
        "signals": signals,
        # derived helpers for fast access
        "_most_recent_role": career[0] if career else None,
        "_all_skill_names": [s.get("name", "").lower() for s in skills if s.get("name")],
        "_all_skill_proficiencies": {
            s.get("name", "").lower(): s.get("proficiency", "beginner") for s in skills if s.get("name")
        },
        "_all_skill_endorsements": {
            s.get("name", "").lower(): s.get("endorsements", 0) for s in skills if s.get("name")
        },
        "_all_skill_durations": {
            s.get("name", "").lower(): s.get("duration_months", 0) for s in skills if s.get("name")
        },
    }


def load_sample_candidates(
    path: str | Path,
    n: int = 1000,
) -> List[Dict[str, Any]]:
    """Load first N candidates for sandbox / testing."""
    return list(load_candidates_jsonl(path, max_candidates=n, validate=True))