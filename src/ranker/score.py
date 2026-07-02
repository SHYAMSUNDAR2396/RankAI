"""Composite scoring for the Redrob candidate-ranking pipeline.

Combines the six feature dimensions extracted by :mod:`features` with the
honeypot penalty from :mod:`honeypot` into a single 0..1 final score.

Composite formula:

    raw  = 0.40 * must_have_score
         + 0.20 * title_score
         + 0.15 * career_score
         + 0.10 * experience_score
         + 0.10 * behavioral_score
         + 0.05 * logistics_score

    final = max(0, raw) * (1 - honeypot_severity * 0.8) * (1 - disqualifier_penalty)

The must_have dimension is heavily weighted (40%) because the JD is explicit
about the must-have skills; candidates without them are auto-low even if
their title says "ML Engineer".  Honeypot severity is applied as a strong
multiplicative penalty (up to 80% reduction) so detected honeypots naturally
fall out of the top-100 without explicit exclusion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .features import extract_features
from .honeypot import HoneypotResult, detect_honeypot, is_safe_for_top100

logger = logging.getLogger(__name__)


# =============================================================================
# Weights — derived from JD analysis
# =============================================================================

#: Composite dimension weights (must sum to 1.0).
WEIGHTS: Dict[str, float] = {
    "must_have": 0.40,   # JD explicit must-haves
    "title": 0.20,       # current title alignment
    "career": 0.15,      # career trajectory + production evidence
    "experience": 0.10,  # YOE proximity to 5-9 range
    "behavioral": 0.10,  # Redrob 23-signal multipliers
    "logistics": 0.05,   # location + salary
}

#: Multiplicative penalty factor for honeypot severity (max 80% reduction).
HONEYPOT_PENALTY_FACTOR: float = 0.80

#: Multiplicative penalty factor for disqualifier penalty (max 100% reduction).
DISQUALIFIER_PENALTY_FACTOR: float = 1.0  # already a 0..1 penalty in features


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class ScoredCandidate:
    """Result of scoring a single candidate."""

    candidate_id: str
    score: float                                # 0..1, final composite
    features: Dict[str, Any]                    # from extract_features
    honeypot: HoneypotResult                    # from detect_honeypot
    is_honeypot: bool = False                   # convenience: honeypot.flag
    is_safe_for_top100: bool = True             # convenience: is_safe_for_top100
    # Carried for downstream reasoning generation
    name: str = ""
    current_title: str = ""
    current_company: str = ""
    location: str = ""
    years_experience: float = 0.0
    matched_skills: List[str] = field(default_factory=list)
    title_bucket: str = "unknown_technical"
    disqualifier_reasons: List[str] = field(default_factory=list)


# =============================================================================
# Public API
# =============================================================================

def score_candidate(cand: Dict[str, Any]) -> ScoredCandidate:
    """Score a single candidate.

    Args:
        cand: A normalized candidate dict (from :func:`ranker.io._normalize_candidate`).

    Returns:
        A :class:`ScoredCandidate` with the final 0..1 score, intermediate
        features, and honeypot result.
    """
    # 1. Extract features
    features = extract_features(cand)

    # 2. Detect honeypot
    honeypot = detect_honeypot(cand)

    # 3. Compute weighted composite
    raw = (
        WEIGHTS["must_have"] * features["must_have_score"]
        + WEIGHTS["title"] * features["title_score"]
        + WEIGHTS["career"] * features["career_score"]
        + WEIGHTS["experience"] * features["experience_score"]
        + WEIGHTS["behavioral"] * features["behavioral_score"]
        + WEIGHTS["logistics"] * features["logistics_score"]
    )

    # 4. Apply honeypot penalty (multiplicative, max 80% reduction)
    honeypot_multiplier = max(0.0, 1.0 - honeypot.severity * HONEYPOT_PENALTY_FACTOR)

    # 5. Apply disqualifier penalty (multiplicative, max 100% reduction)
    disq_penalty = features.get("disqualifier_penalty", 0.0)
    disq_multiplier = max(0.0, 1.0 - disq_penalty * DISQUALIFIER_PENALTY_FACTOR)

    # 6. Final score
    final_score = raw * honeypot_multiplier * disq_multiplier
    final_score = max(0.0, min(1.0, final_score))

    safe = is_safe_for_top100(honeypot)

    return ScoredCandidate(
        candidate_id=cand.get("candidate_id", ""),
        score=final_score,
        features=features,
        honeypot=honeypot,
        is_honeypot=honeypot.flag,
        is_safe_for_top100=safe,
        name=cand.get("name", ""),
        current_title=cand.get("current_title", ""),
        current_company=cand.get("current_company", ""),
        location=cand.get("location", ""),
        years_experience=float(cand.get("years_of_experience", 0.0) or 0.0),
        matched_skills=features.get("matched_skills", []),
        title_bucket=features.get("title_bucket", "unknown_technical"),
        disqualifier_reasons=features.get("disqualifier_reasons", []),
    )


def score_all(candidates) -> List[ScoredCandidate]:
    """Score a stream (or list) of candidates.

    Args:
        candidates: An iterable of normalized candidate dicts.

    Returns:
        A list of :class:`ScoredCandidate` (one per input candidate), in input order.
    """
    results: List[ScoredCandidate] = []
    for cand in candidates:
        results.append(score_candidate(cand))
    return results


def select_top_n(
    scored: List[ScoredCandidate],
    n: int = 100,
) -> List[ScoredCandidate]:
    """Pick the top N safest + highest-scoring candidates.

    Strategy:
    1. Drop all flagged honeypots.
    2. Sort remaining by score descending.
    3. If fewer than N remain, fill the rest with the highest-scoring
       non-safe candidates (so the submission still has 100 rows).
    4. Tie-break by ``candidate_id`` ascending (per submission spec).

    Args:
        scored: List of scored candidates.
        n: Number to select (default 100).

    Returns:
        List of exactly N :class:`ScoredCandidate`, sorted by score desc.
    """
    safe = [s for s in scored if s.is_safe_for_top100]
    unsafe = [s for s in scored if not s.is_safe_for_top100]

    # Sort both by score desc, then candidate_id asc
    safe.sort(key=lambda s: (-round(s.score, 4), s.candidate_id))
    unsafe.sort(key=lambda s: (-round(s.score, 4), s.candidate_id))

    selected = safe[:n]
    if len(selected) < n:
        # Fill with the highest-scoring "unsafe" (least-worst) candidates
        shortfall = n - len(selected)
        selected.extend(unsafe[:shortfall])

    return selected
