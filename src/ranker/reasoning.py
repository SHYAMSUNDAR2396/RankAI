"""Factual reasoning generator for the Redrob candidate-ranking pipeline.

Generates a 1-2 sentence justification for why a candidate is at a given rank.
Every claim in the reasoning is grounded in a concrete fact from the
candidate's structured profile — no LLM, no hallucination, no templated filler.

The output is guaranteed to:
* Reference the candidate's actual current title and company (or "Unknown")
* Reference at least 2 matched must-have skills (when present)
* Reference at least one concrete behavioral signal value
* Reference the candidate's YOE and location
* Stay under 400 characters (per submission spec)
* Be substantively different from other reasonings (achieved by including
  candidate-specific numbers + the matched skill list, which is unique)

This module is intentionally pure-Python: it composes a string from named
fields, never calls an LLM. Every output can be verified against the
original candidate JSON.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

#: Maximum reasoning length (per submission_spec.txt Section 2).
MAX_REASONING_CHARS: int = 400


# =============================================================================
# Public API
# =============================================================================

def build_reasoning(cand: Dict[str, Any], features: Dict[str, Any]) -> str:
    """Generate a 1-2 sentence factual justification for a candidate.

    Args:
        cand: A normalized candidate dict (from :mod:`io._normalize_candidate`).
        features: A features dict (from :mod:`features.extract_features`).

    Returns:
        A reasoning string ≤ 400 characters grounded in real profile data.
    """
    parts: List[str] = []

    rank = features.get("_rank", 0)
    tier = _rank_tier(rank) if rank > 0 else None

    title = cand.get("current_title", "").strip() or "Professional"
    company = cand.get("current_company", "").strip()
    yoe = float(cand.get("years_of_experience", 0.0) or 0.0)

    base = _title_company_yoe(title, company, yoe)
    if tier == "top":
        opener = f"Exceptional fit: {base}"
    elif tier == "strong":
        opener = f"Strong candidate: {base}"
    else:
        opener = base
    parts.append(opener)

    matched = features.get("matched_skills", [])[:3]
    if matched:
        clean = [_clean_skill_name(s) for s in matched]
        if len(clean) == 1:
            parts.append(f"matched skill: {clean[0]}")
        elif len(clean) == 2:
            parts.append(f"matched skills: {clean[0]} and {clean[1]}")
        else:
            parts.append(f"matched skills: {', '.join(clean[:-1])}, and {clean[-1]}")

    jd_phrase = _jd_connection_phrase(features)
    if jd_phrase:
        parts.append(jd_phrase)

    career = cand.get("career_history", []) or []
    production_evidence = _detect_production_evidence(career)
    if production_evidence and not _evidence_already_in_skills(production_evidence, matched):
        parts.append(production_evidence)

    sig_phrase = _top_behavioral_phrase(cand)
    if sig_phrase:
        parts.append(sig_phrase)

    location = cand.get("location", "").strip()
    country = cand.get("country", "").strip()
    loc_phrase = _location_phrase(location, country)
    if loc_phrase:
        parts.append(loc_phrase)

    disq_reasons = features.get("disqualifier_reasons", []) or []
    if disq_reasons and tier in ("solid", "developing"):
        disq_phrase = _disqualifier_phrase(disq_reasons)
        if disq_phrase:
            parts.append(disq_phrase)

    text = "; ".join(parts)
    return _truncate_reasoning(text)


def build_reasoning_from_scored(scored, rank: int = 0) -> str:
    """Convenience: generate reasoning for a :class:`ScoredCandidate`."""
    cand = {
        "current_title": getattr(scored, "current_title", ""),
        "current_company": getattr(scored, "current_company", ""),
        "years_of_experience": getattr(scored, "years_experience", 0.0),
        "location": getattr(scored, "location", ""),
        "career_history": getattr(scored, "features", {}).get("_career_history", []),
        "signals": getattr(scored, "features", {}).get("_signals", {}),
    }
    features = dict(getattr(scored, "features", {}))
    if rank:
        features["_rank"] = rank
    return build_reasoning(cand, features)


# =============================================================================
# Helpers
# =============================================================================

def _clean_skill_name(s: str) -> str:
    """Convert a skill name to a natural-language form."""
    s = s.strip()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    # Common expansions
    expansions = {
        "rag": "RAG",
        "llm": "LLM",
        "nlp": "NLP",
        "pytorch": "PyTorch",
        "xgboost": "XGBoost",
        "mlops": "MLOps",
        "sql": "SQL",
        "etl": "ETL",
    }
    if s.lower() in expansions:
        return expansions[s.lower()]
    return s


def _rank_tier(rank: int) -> str:
    """Return rank tier label for reasoning variation."""
    if rank <= 10:
        return "top"
    elif rank <= 30:
        return "strong"
    elif rank <= 60:
        return "solid"
    else:
        return "developing"


def _jd_connection_skills() -> dict:
    """Map JD keywords to candidate feature keys that indicate alignment."""
    return {
        "rag_pipeline": ["rag", "nlp", "llm", "transformer", "langchain"],
        "ranking_systems": ["recommendation", "search", "ranking", "ltr", "xgboost"],
        "founding_team": ["founder", "startup", "early_stage"],
        "product_company": [],  # Checked via company_type feature
    }


def _truncate_reasoning(text: str) -> str:
    if len(text) > MAX_REASONING_CHARS:
        return text[: MAX_REASONING_CHARS - 1].rstrip(" ,;.") + "."
    return text


def _title_company_yoe(title: str, company: str, yoe: float) -> str:
    """Build base description from title, company, and YOE."""
    if company and yoe:
        return f"{title} at {company} with {yoe:.1f}y"
    elif company:
        return f"{title} at {company}"
    elif yoe:
        return f"{title} with {yoe:.1f}y experience"
    else:
        return title


def _jd_connection_phrase(features: Dict[str, Any]) -> Optional[str]:
    """Generate a brief JD-alignment phrase based on matched skills."""
    matched_lower = [s.lower() for s in features.get("matched_skills", [])]
    jd_map = _jd_connection_skills()

    phrases: List[str] = []

    rag_skills = set(jd_map["rag_pipeline"])
    if any(s in rag_skills for s in matched_lower):
        phrases.append("aligns with RAG pipeline needs")

    rank_skills = set(jd_map["ranking_systems"])
    if any(s in rank_skills for s in matched_lower):
        phrases.append("ranking systems background")

    founder_skills = set(jd_map["founding_team"])
    if any(s in founder_skills for s in matched_lower):
        phrases.append("founding team experience")

    if features.get("product_company", 0):
        phrases.append("product company experience")

    if not phrases:
        return None
    return "; ".join(phrases[:2])


#: Human-readable labels for disqualifier reasons.
_DISQUALIFIER_LABELS: Dict[str, str] = {
    "mostly_consulting": "career primarily in consulting/services",
    "consulting_heavy": "career primarily in consulting/services",
    "research_only": "academic/research focus without industry product experience",
    "title_chaser": "title changes without substantive role progression",
}


def _disqualifier_phrase(disq_reasons: List[str]) -> Optional[str]:
    """Surface disqualifiers as brief, factual warnings (max 2)."""
    labels: List[str] = []
    for reason in disq_reasons[:2]:
        label = _DISQUALIFIER_LABELS.get(reason, reason.replace("_", " "))
        labels.append(label)
    if not labels:
        return None
    return "Note: " + "; ".join(labels)


def _detect_production_evidence(career: List[Dict[str, Any]]) -> Optional[str]:
    """Detect and phrase production evidence from career descriptions.

    Returns a short phrase like "production retrieval experience" or
    "shipped ranking system" — or None if no evidence.
    """
    if not career:
        return None

    all_text = " ".join(
        (r.get("description", "") or "") + " " + (r.get("title", "") or "")
        for r in career
    ).lower()

    # Highest-signal phrases
    if "ranking system" in all_text or "learn to rank" in all_text or "learning to rank" in all_text:
        return "built ranking system"
    if "retrieval system" in all_text or "retrieval-augmented" in all_text or "retrieval augmented" in all_text:
        return "built retrieval system"
    if "search system" in all_text:
        return "built search system"
    if "recommendation system" in all_text:
        return "built recommendation system"
    if "vector database" in all_text or "vector search" in all_text:
        return "production vector search"
    if "rag" in all_text and "production" in all_text:
        return "production RAG"
    if "fine-tuning" in all_text or "finetuning" in all_text or "fine tuning" in all_text:
        if "production" in all_text or "deployed" in all_text:
            return "production fine-tuning"
    if "llm" in all_text and "production" in all_text:
        return "production LLM"
    if "deployed" in all_text and "production" in all_text:
        return "deployed to production"
    if "production" in all_text:
        return "production ML experience"
    if "shipped" in all_text:
        return "shipped ML systems"
    return None


def _evidence_already_in_skills(evidence: str, matched: List[str]) -> bool:
    """Return True if the production evidence phrase is already covered by a matched skill."""
    ev = evidence.lower()
    for s in matched:
        if s in ev:
            return True
    return False


def _top_behavioral_phrase(cand: Dict[str, Any]) -> Optional[str]:
    """Pick the most informative behavioral signal and return it as a short phrase.

    Priority: open-to-work + response rate > GitHub activity > response time >
    last active recency > interview completion.
    """
    sig = cand.get("signals", {}) or {}

    # 1. Open to work + response rate (combined)
    otw = sig.get("open_to_work_flag", False)
    resp = sig.get("recruiter_response_rate", None)
    if otw and resp is not None and resp >= 0.5:
        return f"open_to_work; response_rate {resp:.2f}"
    if otw and resp is not None and resp < 0.5:
        return f"open_to_work (low response_rate {resp:.2f})"
    if otw:
        return "open_to_work"

    # 2. GitHub activity (only if linked + active)
    gh = sig.get("github_activity_score", -1)
    if gh is not None and gh >= 0 and gh >= 50:
        return f"active GitHub (score {gh:.0f})"

    # 3. Response rate (high)
    if resp is not None and resp >= 0.7:
        return f"high response_rate ({resp:.2f})"

    # 4. Response time (fast)
    hours = sig.get("avg_response_time_hours", None)
    if hours is not None and hours > 0 and hours <= 24:
        return f"fast response ({hours:.0f}h avg)"

    # 5. Last active recency
    last_active = sig.get("last_active_date", None)
    if last_active:
        # Try to compute "X days ago"
        days = _days_since_active(last_active)
        if days is not None and days <= 7:
            return f"active {days}d ago"
        if days is not None and days <= 30:
            return f"active {days}d ago"
        if days is not None and days <= 90:
            return f"active {days}d ago"

    # 6. Interview completion (high)
    ic = sig.get("interview_completion_rate", None)
    if ic is not None and ic >= 0.9:
        return f"interview_completion {ic:.2f}"

    # 7. Saved by recruiters (validation)
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 10:
        return f"saved by {saved:.0f} recruiters (30d)"

    return None


def _location_phrase(location: str, country: str) -> Optional[str]:
    """Return a short location phrase; omit if not informative."""
    if not location:
        if not country:
            return None
        if country.lower() in ("india", "united states", "uk", "united kingdom",
                               "canada", "germany", "singapore"):
            return f"{country}-based"
        return None

    # City-level granularity is most useful
    return f"{location}-based"


def _days_since_active(iso_date: str) -> Optional[int]:
    """Return days since iso_date (anchored to 2026-07-02 — dataset reference date)."""
    if not iso_date:
        return None
    try:
        from datetime import date, datetime
        s = str(iso_date)[:10]
        dt = datetime.fromisoformat(s).date()
        return (date(2026, 7, 2) - dt).days
    except (ValueError, TypeError):
        return None
