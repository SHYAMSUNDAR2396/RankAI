"""Competition-mode ranking engine.

Zero-LLM, deterministic scoring pipeline that reads 100K candidates from
JSONL, scores each on four feature dimensions, detects honeypots, and
produces a top-100 ranked submission CSV — all within the 5-minute / 16GB
CPU-only compute budget.

Usage::

    from pipeline.competition import CompetitionRanker

    ranker = CompetitionRanker()
    ranked = ranker.rank("candidates.jsonl", top_n=100)
    ranker.write_csv(ranked, "submission.csv")

Scoring dimensions (see :mod:`config.COMPETITION_WEIGHTS`):

* **skill_match** (35%) — overlap between candidate skills and JD keywords.
* **experience_fit** (25%) — YOE proximity to optimal range + trajectory signals.
* **behavioral_signals** (20%) — Redrob engagement / responsiveness / social proof.
* **trajectory_education** (20%) — career progression + institution tier.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import config
from models.candidate import CandidateProfile
from output.writer import write_competition_csv
from pipeline.honeypot import compute_honeypot_score
from pipeline.ingest import load_candidates_jsonl
from pipeline.signals import score_signals

logger = logging.getLogger(__name__)

_TIER1_SCHOOLS = {
    "stanford", "mit", "harvard", "caltech", "berkeley", "cmu",
    "oxford", "cambridge", "eth zurich", "princeton", "yale",
    "columbia", "cornell", "ucl", "imperial college", "tsinghua",
    "peking university", "iit", "iisc", "bits pilani",
}


def _skill_match_score(profile: CandidateProfile) -> float:
    candidate_skills = {s.lower() for s in profile.skills_claimed}
    role_text = " ".join(
        f"{r.title} {' '.join(r.scope_keywords)}"
        for r in profile.roles
    ).lower()
    candidate_skills.update(w for w in role_text.split() if len(w) > 2)

    must = config.COMPETITION_MUST_HAVE_SKILLS
    nice = config.COMPETITION_NICE_TO_HAVE_SKILLS

    if not must and not nice:
        return 0.0

    must_hits = sum(1 for kw in must if kw in candidate_skills or kw in role_text)
    nice_hits = sum(1 for kw in nice if kw in candidate_skills or kw in role_text)

    must_ratio = must_hits / len(must) if must else 1.0
    nice_ratio = nice_hits / len(nice) if nice else 1.0

    return 0.7 * must_ratio + 0.3 * nice_ratio


def _title_relevance_score(profile: CandidateProfile) -> float:
    title_scores = config.COMPETITION_TITLE_SCORES
    most_recent_title = profile.roles[0].title.lower().strip() if profile.roles else ""

    if not most_recent_title:
        return 0.3

    if most_recent_title in title_scores:
        return title_scores[most_recent_title]

    best_match = 0.3
    for known_title, score in title_scores.items():
        if known_title in most_recent_title or most_recent_title in known_title:
            best_match = max(best_match, score)
    return best_match


def _company_type_score(profile: CandidateProfile) -> float:
    """Penalize candidates whose entire career is at consulting firms."""
    consulting = config.COMPETITION_CONSULTING_FIRMS
    total_roles = len(profile.roles)
    if total_roles == 0:
        return 0.5

    consulting_roles = sum(
        1 for r in profile.roles
        if any(cf in r.company.lower() for cf in consulting)
    )
    consulting_ratio = consulting_roles / total_roles

    if consulting_ratio >= 1.0:
        return 0.2
    if consulting_ratio >= 0.75:
        return 0.35
    if consulting_ratio >= 0.5:
        return 0.5
    if consulting_ratio <= 0.25:
        return 0.8
    return 0.65


def _education_field_score(profile: CandidateProfile) -> float:
    """Score alignment of education field with AI/ML engineering."""
    field_scores = config.COMPETITION_EDUCATION_FIELD_SCORES
    if not profile.education:
        return 0.3

    best = 0.0
    for edu in profile.education:
        degree = edu.get("degree", "").lower()
        for field, score in field_scores.items():
            if field in degree:
                best = max(best, score)
    return best if best > 0 else 0.3


def _career_description_score(profile: CandidateProfile) -> float:
    desc_keywords = config.COMPETITION_CAREER_DESCRIPTION_KEYWORDS
    role_text = " ".join(
        " ".join(r.scope_keywords)
        for r in profile.roles
    ).lower()

    if not role_text:
        return 0.0

    positive = 0.0
    negative = 0.0
    for keyword, weight in desc_keywords.items():
        if keyword in role_text:
            if weight > 0:
                positive += weight
            else:
                negative += abs(weight)

    raw = min(1.0, positive / 5.0) - min(0.5, negative / 3.0)
    return max(0.0, min(1.0, raw))


def _location_score(profile: CandidateProfile) -> float:
    india = config.COMPETITION_INDIA_LOCATIONS
    company_text = " ".join(r.company.lower() for r in profile.roles)

    if any(loc in company_text for loc in india):
        return 0.8
    return 0.5


def _anti_pattern_score(profile: CandidateProfile) -> float:
    """Detect keyword stuffing — AI keywords + non-technical title = penalty."""
    stuffing_titles = config.COMPETITION_STUFFING_TITLE_KEYWORDS
    most_recent_title = profile.roles[0].title.lower().strip() if profile.roles else ""

    if most_recent_title not in stuffing_titles:
        return 0.0

    must_skills = set(config.COMPETITION_MUST_HAVE_SKILLS)
    candidate_skills = {s.lower() for s in profile.skills_claimed}
    match_count = len(must_skills & candidate_skills)

    if match_count >= 3:
        return 0.4
    if match_count >= 2:
        return 0.2
    return 0.0


def _experience_fit_score(profile: CandidateProfile) -> float:
    yoe = profile.years_experience
    opt_min = config.COMPETITION_OPTIMAL_YOE_MIN
    opt_max = config.COMPETITION_OPTIMAL_YOE_MAX

    if opt_min <= yoe <= opt_max:
        yoe_score = 1.0
    elif yoe < opt_min:
        yoe_score = max(0.0, 1.0 - (opt_min - yoe) / opt_min)
    else:
        overshoot = yoe - opt_max
        yoe_score = max(0.0, 1.0 - overshoot / 10.0)

    tv = profile.trajectory_vector or {}
    growth = tv.get("growth_rate", 0.0) if isinstance(tv, dict) else 0.0
    leadership = tv.get("leadership_progression", 0.0) if isinstance(tv, dict) else 0.0
    trajectory_bonus = 0.3 * growth + 0.2 * leadership

    return min(1.0, 0.6 * yoe_score + 0.4 * trajectory_bonus)


def _trajectory_education_score(profile: CandidateProfile) -> float:
    all_text = " ".join(
        f"{r.title} {' '.join(r.scope_keywords)}"
        for r in profile.roles
    ).lower()

    traj_hits = sum(
        1 for kw in config.COMPETITION_TRAJECTORY_KEYWORDS
        if kw in all_text
    )
    traj_score = min(1.0, traj_hits / 4.0)

    edu_text = " ".join(
        e.get("institution", "") + " " + e.get("degree", "")
        for e in profile.education
    ).lower()

    tier1 = any(school in edu_text for school in _TIER1_SCHOOLS)
    edu_score = 0.9 if tier1 else 0.5

    return 0.6 * traj_score + 0.4 * edu_score


class CompetitionRanker:
    """Deterministic ranking engine for the India Runs competition.

    Orchestrates signal scoring, honeypot detection, feature fusion, and
    top-N selection.  No LLM calls are made at any point.
    """

    def rank(
        self,
        jsonl_path: str | Path,
        *,
        top_n: int = 100,
        max_candidates: int | None = None,
    ) -> list[dict]:
        """Load, score, and rank candidates from a JSONL file.

        Args:
            jsonl_path: Path to the competition ``candidates.jsonl``.
            top_n: Number of top candidates to return (default 100).
            max_candidates: Optional cap on total candidates to process.

        Returns:
            List of dicts with keys ``candidate_id``, ``score``,
            ``reasoning``, sorted descending by score.
        """
        t0 = time.time()
        profiles: list[CandidateProfile] = list(
            load_candidates_jsonl(jsonl_path, max_candidates=max_candidates)
        )
        t_load = time.time() - t0
        logger.info("Loaded %d candidates in %.1fs", len(profiles), t_load)

        scored = self._score_all(profiles)
        scored.sort(key=lambda x: -x["score"])
        top = scored[:top_n]

        t_total = time.time() - t0
        logger.info(
            "Scored %d candidates, selected top %d in %.1fs total",
            len(scored), len(top), t_total,
        )
        return top

    def _score_all(self, profiles: list[CandidateProfile]) -> list[dict]:
        w = config.COMPETITION_WEIGHTS
        results: list[dict] = []

        for profile in profiles:
            skill = _skill_match_score(profile)
            experience = _experience_fit_score(profile)
            behavioral = score_signals(profile)
            trajectory = _trajectory_education_score(profile)

            title_rel = _title_relevance_score(profile)
            company_type = _company_type_score(profile)
            education_field = _education_field_score(profile)
            career_desc = _career_description_score(profile)
            location = _location_score(profile)
            anti_pattern = _anti_pattern_score(profile)

            skill_adjusted = skill * title_rel
            experience_adjusted = experience * company_type
            trajectory_adjusted = (
                0.6 * trajectory
                + 0.2 * education_field
                + 0.15 * career_desc
                + 0.05 * location
            )

            raw_score = (
                w["skill_match"] * skill_adjusted
                + w["experience_fit"] * experience_adjusted
                + w["behavioral_signals"] * behavioral
                + w["trajectory_education"] * trajectory_adjusted
            )

            raw_score -= anti_pattern

            is_honey, honey_sev = compute_honeypot_score(profile)
            honeypot_penalty = honey_sev * 0.5
            final_score = max(0.0, min(1.0, raw_score - honeypot_penalty))

            reasoning = self._build_reasoning(
                profile, skill, experience, behavioral, trajectory,
                title_rel, company_type, education_field, career_desc,
                is_honey, honey_sev,
            )

            results.append({
                "candidate_id": profile.candidate_id,
                "score": round(final_score, 6),
                "reasoning": reasoning,
                "_is_honeypot": is_honey,
            })

        return results

    def _build_reasoning(
        self,
        profile: CandidateProfile,
        skill: float,
        experience: float,
        behavioral: float,
        trajectory: float,
        title_rel: float,
        company_type: float,
        education_field: float,
        career_desc: float,
        is_honey: bool,
        honey_sev: float,
    ) -> str:
        parts: list[str] = []
        yoe = profile.years_experience

        if skill > 0.7:
            parts.append(f"Strong skill alignment ({skill:.0%})")
        elif skill > 0.4:
            parts.append(f"Moderate skill match ({skill:.0%})")
        else:
            parts.append(f"Limited skill overlap ({skill:.0%})")

        if title_rel >= 0.7:
            parts.append(f"title aligns well ({title_rel:.0%})")
        elif title_rel <= 0.3:
            parts.append(f"title mismatch ({title_rel:.0%})")

        if company_type <= 0.3:
            parts.append("consulting-only background")
        elif company_type >= 0.7:
            parts.append("product company experience")

        if education_field >= 0.7:
            parts.append("CS/AI degree")
        elif education_field <= 0.2:
            parts.append("unrelated degree field")

        if career_desc >= 0.5:
            parts.append("relevant project evidence")
        elif career_desc <= 0.1:
            parts.append("no ML/ranking project evidence")

        if experience > 0.8:
            parts.append(f"excellent YOE fit ({yoe:.0f}y)")
        elif experience > 0.5:
            parts.append(f"good experience level ({yoe:.0f}y)")
        else:
            parts.append(f"experience outside optimal range ({yoe:.0f}y)")

        if behavioral > 0.6:
            parts.append("high engagement signals")
        elif behavioral < 0.3:
            parts.append("low platform engagement")

        if is_honey:
            parts.append(f"⚠ honeypot risk ({honey_sev:.0%})")

        return "; ".join(parts)[:400]

    def write_csv(self, ranked: list[dict], output_path: str | Path) -> Path:
        """Write ranked results to competition submission CSV."""
        return write_competition_csv(ranked, Path(output_path))
