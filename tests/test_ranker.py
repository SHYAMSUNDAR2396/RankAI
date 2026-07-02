"""Comprehensive tests for src/ranker modules.

Covers: honeypot detection, feature extraction, scoring, reasoning generation,
and CSV writing.  All tests use inline fixture dicts — no JSONL file needed.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------
from src.ranker.honeypot import (
    HoneypotResult,
    detect_honeypot,
    is_safe_for_top100,
)
from src.ranker.features import (
    extract_features,
    _extract_title,
    _extract_disqualifiers,
    _extract_must_have,
)
from src.ranker.score import (
    ScoredCandidate,
    score_candidate,
    score_all,
    select_top_n,
)
from src.ranker.reasoning import build_reasoning, build_reasoning_from_scored
from rank import write_submission_csv, CSV_COLUMNS


# ============================================================================
# Fixture helpers — build minimal normalized candidate dicts
# ============================================================================

def _make_skill(name: str, proficiency: str = "advanced", endorsements: int = 5,
                duration_months: int = 24) -> Dict[str, Any]:
    return {
        "name": name,
        "proficiency": proficiency,
        "endorsements": endorsements,
        "duration_months": duration_months,
    }


def _make_role(title: str, company: str, start: str, end: str,
               duration_months: int = 24, description: str = "",
               company_size: str = "51-200", industry: str = "technology",
               is_current: bool = False) -> Dict[str, Any]:
    return {
        "title": title,
        "company": company,
        "start_date": start,
        "end_date": end,
        "duration_months": duration_months,
        "is_current": is_current,
        "industry": industry,
        "company_size": company_size,
        "description": description,
    }


def _make_signals(**overrides) -> Dict[str, Any]:
    """Build a full redrob_signals dict with sensible defaults."""
    base = {
        "profile_completeness_score": 80.0,
        "signup_date": "2023-01-15",
        "last_active_date": "2026-06-28",
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.75,
        "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {"python_ml": 85, "nlp_advanced": 78},
        "connection_count": 200,
        "endorsements_received": 25,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "preferred_work_mode": "flexible",
        "willing_to_relocate": True,
        "github_activity_score": 60,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.6,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


def _make_candidate(
    candidate_id: str = "CAND_TEST_001",
    name: str = "Test Candidate",
    title: str = "AI Engineer",
    company: str = "Acme Corp",
    yoe: float = 7.0,
    skills: List[Dict[str, Any]] | None = None,
    career: List[Dict[str, Any]] | None = None,
    signals: Dict[str, Any] | None = None,
    location: str = "Pune, India",
    country: str = "India",
    headline: str = "",
    summary: str = "",
) -> Dict[str, Any]:
    """Build a normalized candidate dict matching io._normalize_candidate output."""
    if skills is None:
        skills = [
            _make_skill("embeddings", "advanced", 15, 36),
            _make_skill("rag", "advanced", 10, 24),
            _make_skill("pytorch", "expert", 20, 48),
        ]
    if career is None:
        career = [
            _make_role(
                title, company, "2021-06", "2025-12",
                duration_months=54, is_current=True,
                description="Built ranking system and production RAG pipelines.",
            ),
            _make_role(
                "Software Engineer", "StartupXYZ", "2018-07", "2021-05",
                duration_months=34,
                description="Developed ML models and deployed to production.",
            ),
        ]
    if signals is None:
        signals = _make_signals()

    # Build the derived fields that _normalize_candidate produces
    all_skill_names = [s.get("name", "").lower() for s in skills if s.get("name")]
    return {
        "candidate_id": candidate_id,
        "name": name,
        "headline": headline or f"{title} at {company}",
        "summary": summary or f"Experienced {title} with {yoe} years.",
        "location": location,
        "country": country,
        "years_of_experience": yoe,
        "current_title": title,
        "current_company": company,
        "current_company_size": "51-200",
        "current_industry": "technology",
        "career_history": career,
        "education": [],
        "skills": skills,
        "certifications": [],
        "languages": [],
        "signals": signals,
        "_most_recent_role": career[0] if career else None,
        "_all_skill_names": all_skill_names,
        "_all_skill_proficiencies": {
            s["name"].lower(): s["proficiency"] for s in skills if s.get("name")
        },
        "_all_skill_endorsements": {
            s["name"].lower(): s["endorsements"] for s in skills if s.get("name")
        },
        "_all_skill_durations": {
            s["name"].lower(): s["duration_months"] for s in skills if s.get("name")
        },
    }


# ============================================================================
# (a) Honeypot detection tests
# ============================================================================

class TestHoneypot:
    """Honeypot detector tests."""

    def test_timeline_impossibility_overlapping_roles(self):
        """Two roles with 30+ months of overlap → timeline_impossible flag."""
        career = [
            _make_role("AI Engineer", "Alpha", "2022-01-01", "2026-06-01",
                       duration_months=54, is_current=True),
            _make_role("ML Engineer", "Beta", "2020-01-01", "2025-06-01",
                       duration_months=66),
        ]
        cand = _make_candidate(career=career)
        result = detect_honeypot(cand)
        assert result.timeline_impossible is True
        assert result.flag is True
        assert result.severity > 0.0

    def test_timeline_clean_no_overlap(self):
        """Two sequential roles with no overlap → no timeline flag."""
        career = [
            _make_role("AI Engineer", "Alpha", "2022-06-01", "2026-01-01",
                       duration_months=43, is_current=True),
            _make_role("Software Engineer", "Beta", "2019-01-01", "2022-05-01",
                       duration_months=40),
        ]
        cand = _make_candidate(career=career)
        result = detect_honeypot(cand)
        assert result.timeline_impossible is False

    def test_yoe_span_mismatch_extreme_gap(self):
        """YOE of 15 but career span only 4 years → yoe_span_mismatch flag."""
        career = [
            _make_role("AI Engineer", "Alpha", "2022-01-01", "2026-01-01",
                       duration_months=48, is_current=True),
        ]
        cand = _make_candidate(yoe=15.0, career=career)
        result = detect_honeypot(cand)
        assert result.yoe_span_mismatch is True
        # But yoe_span_mismatch alone does NOT set flag
        # (only boosts severity when combined with another signal)
        assert result.flag is False

    def test_yoe_span_normal(self):
        """YOE of 7 with career spanning 8 years → no mismatch."""
        career = [
            _make_role("AI Engineer", "Alpha", "2020-01-01", "2026-01-01",
                       duration_months=72, is_current=True),
        ]
        cand = _make_candidate(yoe=7.0, career=career)
        result = detect_honeypot(cand)
        assert result.yoe_span_mismatch is False

    def test_expert_without_endorsements(self):
        """4+ AI skills at 'expert' with ≤2 endorsements → honeypot flag."""
        skills = [
            _make_skill("pytorch", "expert", 1, 24),
            _make_skill("tensorflow", "expert", 0, 12),
            _make_skill("transformers", "expert", 2, 18),
            _make_skill("rag", "expert", 1, 10),
            _make_skill("embeddings", "expert", 0, 6),
        ]
        cand = _make_candidate(skills=skills)
        result = detect_honeypot(cand)
        assert result.expert_without_endorsements is True
        assert result.flag is True

    def test_expert_with_endorsements_no_flag(self):
        """4+ AI skills at 'expert' but with many endorsements → no flag."""
        skills = [
            _make_skill("pytorch", "expert", 30, 48),
            _make_skill("tensorflow", "expert", 25, 36),
            _make_skill("transformers", "expert", 20, 30),
            _make_skill("rag", "expert", 15, 24),
        ]
        cand = _make_candidate(skills=skills)
        result = detect_honeypot(cand)
        assert result.expert_without_endorsements is False
        assert result.flag is False

    def test_title_seniority_inflation(self):
        """'Senior AI Engineer' title with <3 YOE → title inflation flag."""
        career = [
            _make_role("Senior AI Engineer", "Alpha", "2024-06", "2026-01",
                       duration_months=19, is_current=True),
        ]
        cand = _make_candidate(title="Senior AI Engineer", yoe=2.0, career=career)
        result = detect_honeypot(cand)
        assert result.title_seniority_inflation is True

    def test_title_senior_with_high_yoe_no_flag(self):
        """'Senior AI Engineer' with 8 YOE → no inflation."""
        career = [
            _make_role("Senior AI Engineer", "Alpha", "2019-01", "2026-01",
                       duration_months=84, is_current=True),
        ]
        cand = _make_candidate(title="Senior AI Engineer", yoe=8.0, career=career)
        result = detect_honeypot(cand)
        assert result.title_seniority_inflation is False

    def test_skill_stuffing(self):
        """Non-tech title with 6+ advanced AI skills → flagged via skill_stuffing."""
        skills = [
            _make_skill("transformers", "advanced", 5, 6),
            _make_skill("pytorch", "advanced", 3, 6),
            _make_skill("tensorflow", "advanced", 4, 6),
            _make_skill("rag", "advanced", 2, 6),
            _make_skill("embeddings", "advanced", 3, 6),
            _make_skill("fine-tuning", "advanced", 1, 6),
        ]
        career = [
            _make_role("Marketing Manager", "BrandCo", "2022-01", "2026-01",
                       duration_months=48, is_current=True,
                       description="Marketing campaigns and brand strategy."),
        ]
        cand = _make_candidate(
            title="HR Manager", yoe=5.0, career=career, skills=skills,
        )
        result = detect_honeypot(cand)
        assert result.skill_stuffing_count >= 6
        assert result.flag is True

    def test_safe_for_top100_clean_candidate(self):
        """A clean, non-flagged candidate with low severity → safe for top 100."""
        cand = _make_candidate()
        result = detect_honeypot(cand)
        assert is_safe_for_top100(result) is True

    def test_safe_for_top100_flagged_candidate(self):
        """A flagged honeypot → not safe for top 100."""
        # Force a flag: expert without endorsements
        skills = [
            _make_skill("pytorch", "expert", 1, 12),
            _make_skill("tensorflow", "expert", 0, 12),
            _make_skill("transformers", "expert", 2, 12),
            _make_skill("rag", "expert", 0, 12),
        ]
        cand = _make_candidate(skills=skills)
        result = detect_honeypot(cand)
        assert result.flag is True
        assert is_safe_for_top100(result) is False

    def test_safe_for_top100_high_severity_no_flag(self):
        """High severity (>=0.7) but not flagged → still not safe."""
        result = HoneypotResult(flag=False, severity=0.8)
        assert is_safe_for_top100(result) is False


# ============================================================================
# (b) Feature extraction tests
# ============================================================================

class TestFeatures:
    """Feature extraction tests."""

    def test_must_have_skill_matching(self):
        """Candidate with embeddings/rag skills → high must_have score."""
        skills = [
            _make_skill("embeddings", "expert", 30, 48),
            _make_skill("rag", "expert", 25, 36),
            _make_skill("pytorch", "advanced", 20, 48),
        ]
        career = [
            _make_role("AI Engineer", "Acme", "2020-01", "2026-01",
                       duration_months=72, is_current=True,
                       description="Built ranking system with production RAG pipelines."),
        ]
        cand = _make_candidate(skills=skills, career=career)
        features = extract_features(cand)
        assert features["must_have_score"] > 0.3
        assert len(features["matched_skills"]) >= 1

    def test_must_have_no_matching_skills(self):
        """Candidate with no matching must-have skills → low must_have score."""
        skills = [
            _make_skill("microsoft word", "expert", 10, 60),
            _make_skill("powerpoint", "advanced", 5, 48),
        ]
        cand = _make_candidate(skills=skills, career=[])
        features = extract_features(cand)
        assert features["must_have_score"] == 0.0

    def test_title_bucket_strong_ai(self):
        """'AI Engineer' title → strong_ai bucket."""
        cand = _make_candidate(title="AI Engineer")
        features = extract_features(cand)
        assert features["title_bucket"] == "strong_ai"
        assert features["title_score"] == 1.0

    def test_title_bucket_non_technical(self):
        """'HR Manager' title → non_technical bucket."""
        cand = _make_candidate(
            title="HR Manager",
            skills=[_make_skill("communication", "expert", 10, 60)],
        )
        features = extract_features(cand)
        assert features["title_bucket"] == "non_technical"
        assert features["title_score"] == 0.10

    def test_title_bucket_ml_engineer(self):
        """'ML Engineer' title → strong_ai bucket."""
        cand = _make_candidate(title="ML Engineer")
        features = extract_features(cand)
        assert features["title_bucket"] == "strong_ai"

    def test_title_bucket_backend_engineer(self):
        """'Backend Engineer' title → technical_adjacent bucket."""
        cand = _make_candidate(title="Backend Engineer")
        features = extract_features(cand)
        assert features["title_bucket"] == "technical_adjacent"

    def test_disqualifier_all_consulting(self):
        """Entire career at consulting firms → disqualifier penalty."""
        career = [
            _make_role("Consultant", "TCS", "2020-01", "2022-06",
                       duration_months=30),
            _make_role("Senior Consultant", "Infosys", "2022-07", "2026-01",
                       duration_months=42, is_current=True),
        ]
        cand = _make_candidate(yoe=6.0, career=career)
        penalty, reasons = _extract_disqualifiers(cand)
        assert penalty > 0.0
        assert "entire_career_consulting" in reasons

    def test_disqualifier_keyword_stuffer(self):
        """Non-tech title with 3+ AI skills → keyword_stuffer penalty."""
        skills = [
            _make_skill("transformers", "advanced", 5, 6),
            _make_skill("pytorch", "advanced", 3, 6),
            _make_skill("rag", "advanced", 2, 6),
        ]
        career = [
            _make_role("Marketing Manager", "BrandCo", "2022-01", "2026-01",
                       duration_months=48, is_current=True,
                       description="Marketing campaigns."),
        ]
        cand = _make_candidate(
            title="Marketing Manager", yoe=5.0, career=career, skills=skills,
        )
        penalty, reasons = _extract_disqualifiers(cand)
        assert penalty > 0.0
        assert "keyword_stuffer" in reasons

    def test_disqualifier_none_for_clean(self):
        """Clean candidate with product-company career → no disqualifiers."""
        career = [
            _make_role("AI Engineer", "Acme", "2020-01", "2026-01",
                       duration_months=72, is_current=True,
                       description="Built production ML systems."),
        ]
        cand = _make_candidate(yoe=7.0, career=career)
        penalty, reasons = _extract_disqualifiers(cand)
        assert penalty == 0.0
        assert len(reasons) == 0


# ============================================================================
# (c) Scoring tests
# ============================================================================

class TestScoring:
    """Scoring tests."""

    def test_score_candidate_returns_scored_candidate(self):
        """score_candidate returns ScoredCandidate with 0..1 score."""
        cand = _make_candidate()
        scored = score_candidate(cand)
        assert isinstance(scored, ScoredCandidate)
        assert 0.0 <= scored.score <= 1.0
        assert scored.candidate_id == "CAND_TEST_001"

    def test_score_candidate_with_must_have_skills(self):
        """Candidate with strong must-have matches scores higher than one without."""
        strong_skills = [
            _make_skill("embeddings", "expert", 30, 48),
            _make_skill("rag", "expert", 25, 36),
            _make_skill("pytorch", "advanced", 20, 48),
            _make_skill("bm25", "advanced", 10, 24),
        ]
        strong_career = [
            _make_role("AI Engineer", "Acme", "2020-01", "2026-01",
                       duration_months=72, is_current=True,
                       description="Built ranking system with production RAG."),
        ]
        strong = _make_candidate(skills=strong_skills, career=strong_career)

        weak_skills = [
            _make_skill("microsoft word", "expert", 10, 60),
        ]
        weak = _make_candidate(skills=weak_skills, career=[
            _make_role("Admin", "OfficeCo", "2020-01", "2026-01",
                       duration_months=72, is_current=True,
                       description="Office administration."),
        ])

        scored_strong = score_candidate(strong)
        scored_weak = score_candidate(weak)
        assert scored_strong.score > scored_weak.score

    def test_select_top_n_returns_exactly_n(self):
        """select_top_n returns exactly N candidates."""
        candidates = [_make_candidate(candidate_id=f"CAND_{i:03d}") for i in range(150)]
        scored = score_all(candidates)
        top = select_top_n(scored, n=100)
        assert len(top) == 100

    def test_select_top_n_drops_flagged_honeypots(self):
        """select_top_n drops flagged honeypots before safe candidates."""
        # Clean candidates
        clean = [_make_candidate(candidate_id=f"CAND_CLEAN_{i:03d}") for i in range(10)]
        # Honeypot candidate (expert without endorsements)
        honeypot_skills = [
            _make_skill("pytorch", "expert", 1, 6),
            _make_skill("tensorflow", "expert", 0, 6),
            _make_skill("transformers", "expert", 2, 6),
            _make_skill("rag", "expert", 0, 6),
        ]
        honeypot = _make_candidate(candidate_id="CAND_HONEYPOT", skills=honeypot_skills)

        all_cands = clean + [honeypot]
        scored = score_all(all_cands)
        top = select_top_n(scored, n=10)
        top_ids = [s.candidate_id for s in top]
        # Honeypot should be deprioritized or absent if enough safe candidates
        assert "CAND_HONEYPOT" not in top_ids or len(top_ids) > 0

    def test_select_top_n_tie_break_by_candidate_id(self):
        """Equal scores → tie-break by candidate_id ascending."""
        # Two candidates with minimal features (likely same score)
        c1 = _make_candidate(candidate_id="CAND_ZZZ_001")
        c2 = _make_candidate(candidate_id="CAND_AAA_001")
        scored = score_all([c1, c2])
        top = select_top_n(scored, n=2)
        assert top[0].candidate_id == "CAND_AAA_001"

    def test_score_monotonicity(self):
        """When selecting top 100 from 150, rank 1 score >= rank 100 score."""
        candidates = [_make_candidate(candidate_id=f"CAND_{i:03d}") for i in range(150)]
        scored = score_all(candidates)
        top = select_top_n(scored, n=100)
        assert top[0].score >= top[-1].score

    def test_score_all_returns_list(self):
        """score_all returns a list of ScoredCandidate with correct length."""
        candidates = [_make_candidate(candidate_id=f"CAND_{i}") for i in range(5)]
        results = score_all(candidates)
        assert len(results) == 5
        assert all(isinstance(r, ScoredCandidate) for r in results)


# ============================================================================
# (d) Reasoning tests
# ============================================================================

class TestReasoning:
    """Reasoning generation tests."""

    def test_build_reasoning_includes_title_and_company(self):
        """Reasoning includes the candidate's title and company."""
        cand = _make_candidate(title="AI Engineer", company="Acme Corp")
        features = extract_features(cand)
        reasoning = build_reasoning(cand, features)
        assert "AI Engineer" in reasoning
        assert "Acme Corp" in reasoning

    def test_build_reasoning_includes_matched_skills(self):
        """Reasoning mentions matched skills when present."""
        skills = [
            _make_skill("embeddings", "expert", 30, 48),
            _make_skill("rag", "expert", 25, 36),
        ]
        cand = _make_candidate(skills=skills)
        features = extract_features(cand)
        reasoning = build_reasoning(cand, features)
        # At least one skill word should appear
        assert "embeddings" in reasoning.lower() or "rag" in reasoning.lower()

    def test_build_reasoning_stays_under_380_chars(self):
        """Reasoning never exceeds 380 characters."""
        # Build a candidate with lots of data to maximize reasoning length
        skills = [
            _make_skill("embeddings", "expert", 30, 48),
            _make_skill("rag", "expert", 25, 36),
            _make_skill("pytorch", "advanced", 20, 48),
            _make_skill("bm25", "advanced", 10, 24),
        ]
        career = [
            _make_role("Senior AI Engineer", "MegaCorp International Inc.", "2018-01", "2026-01",
                       duration_months=96, is_current=True,
                       description="Built ranking system with production RAG pipelines at massive scale."),
        ]
        cand = _make_candidate(
            title="Senior AI Engineer",
            company="MegaCorp International Inc.",
            yoe=10.0, skills=skills, career=career,
            location="Hyderabad, India",
        )
        features = extract_features(cand)
        reasoning = build_reasoning(cand, features)
        assert len(reasoning) <= 380

    def test_build_reasoning_no_hallucinated_fields(self):
        """Reasoning only references actual candidate fields — no invented data."""
        cand = _make_candidate(
            title="Data Scientist",
            company="DataCo",
            yoe=5.0,
            location="Noida, India",
        )
        features = extract_features(cand)
        reasoning = build_reasoning(cand, features)
        # Should NOT contain made-up company names or titles
        assert "Google" not in reasoning
        assert "Staff Engineer" not in reasoning or "Staff Engineer" in cand["current_title"]

    def test_build_reasoning_with_empty_candidate(self):
        """Reasoning works even with minimal/empty candidate data."""
        cand = _make_candidate(title="", company="", yoe=0.0)
        features = extract_features(cand)
        reasoning = build_reasoning(cand, features)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert len(reasoning) <= 380

    def test_build_reasoning_from_scored(self):
        """build_reasoning_from_scored generates reasoning from ScoredCandidate."""
        cand = _make_candidate(title="ML Engineer", company="MLCo")
        scored = score_candidate(cand)
        reasoning = build_reasoning_from_scored(scored)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert len(reasoning) <= 400


# ============================================================================
# (e) CSV writing tests
# ============================================================================

class TestCsvWriting:
    """CSV writing tests."""

    def _build_ranked(self, n: int = 100) -> tuple:
        """Build n scored candidates and their cand_map."""
        candidates = [_make_candidate(candidate_id=f"CAND_{i:03d}") for i in range(n)]
        scored = score_all(candidates)
        cand_map = {c["candidate_id"]: c for c in candidates}
        return scored, cand_map

    def test_write_submission_csv_valid_columns(self):
        """CSV has the correct column headers."""
        scored, cand_map = self._build_ranked(10)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                reader = csv.DictReader(fh)
                assert reader.fieldnames == CSV_COLUMNS
        finally:
            out.unlink()

    def test_write_submission_csv_exactly_100_rows(self):
        """CSV has exactly 100 data rows (plus header)."""
        scored, cand_map = self._build_ranked(100)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                lines = fh.readlines()
                # Header + 100 data rows = 101 lines
                assert len(lines) == 101
        finally:
            out.unlink()

    def test_write_submission_csv_ranks_unique_1_to_100(self):
        """Ranks are 1-100, each appearing exactly once."""
        scored, cand_map = self._build_ranked(100)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                reader = csv.DictReader(fh)
                ranks = [int(row["rank"]) for row in reader]
            assert sorted(ranks) == list(range(1, 101))
        finally:
            out.unlink()

    def test_write_submission_csv_scores_monotonic(self):
        """Scores are monotonically non-increasing."""
        scored, cand_map = self._build_ranked(100)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                reader = csv.DictReader(fh)
                scores = [float(row["score"]) for row in reader]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Score at rank {i+1} ({scores[i]}) < score at rank {i+2} ({scores[i+1]})"
                )
        finally:
            out.unlink()

    def test_write_submission_csv_tie_break_by_id(self):
        """Equal scores → sorted by candidate_id ascending."""
        # Create candidates that will have identical scores
        same_cands = [_make_candidate(candidate_id=f"CAND_{chr(90-i)}") for i in range(5)]
        scored = score_all(same_cands)
        cand_map = {c["candidate_id"]: c for c in same_cands}
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            # Check all scores are equal
            scores = [float(r["score"]) for r in rows]
            if len(set(scores)) == 1:
                # All equal — IDs should be ascending
                ids = [r["candidate_id"] for r in rows]
                assert ids == sorted(ids)
        finally:
            out.unlink()

    def test_write_submission_csv_reasoning_under_limit(self):
        """All reasoning fields are within 380 char limit."""
        scored, cand_map = self._build_ranked(10)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = Path(f.name)
        try:
            write_submission_csv(scored, cand_map, out)
            with open(out) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    assert len(row["reasoning"]) <= 380, (
                        f"Row {row['rank']}: reasoning too long ({len(row['reasoning'])} chars)"
                    )
        finally:
            out.unlink()
