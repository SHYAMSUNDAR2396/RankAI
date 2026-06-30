"""Tests for the competition-mode ranking pipeline.

Covers models, signals, honeypot detection, JSONL ingestion, competition
scoring, CSV output, and CLI entry point. All tests are fully offline and
deterministic — no LLM calls, no network, no external data files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import config
from models.candidate import CandidateProfile, CandidateRole, RedrobSignals
from output.writer import write_competition_csv
from pipeline.competition import (
    CompetitionRanker,
    _anti_pattern_score,
    _career_description_score,
    _company_type_score,
    _education_field_score,
    _experience_fit_score,
    _location_score,
    _skill_match_score,
    _title_relevance_score,
    _trajectory_education_score,
)
from pipeline.honeypot import (
    _months_between,
    _parse_month,
    compute_honeypot_score,
    detect_skill_inconsistency,
    detect_statistical_outliers,
    detect_timeline_impossibility,
    detect_yoe_career_mismatch,
)
from pipeline.ingest import _map_competition_row, load_candidates_jsonl
from pipeline.signals import (
    _compute_engagement,
    _compute_openness,
    _compute_responsiveness,
    _compute_social_proof,
    _compute_technical,
    normalize,
    score_signals,
    score_signals_breakdown,
)
from rank import build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signals(**overrides: Any) -> RedrobSignals:
    defaults: dict[str, Any] = {
        "profile_completeness_score": 80.0,
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.7,
        "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {"python": 85.0, "pytorch": 70.0},
        "interview_completion_rate": 0.8,
        "notice_period_days": 30,
        "preferred_work_mode": "remote",
        "last_active_date": "2026-06-01",
        "github_activity_score": 60.0,
        "github_public_repos": 10,
        "github_recent_commits_6m": 50,
        "blog_posts_count": 3,
        "online_presence_score": 55.0,
        "connection_count": 200,
        "endorsements_received": 25,
        "certifications_count": 2,
        "patent_count": 0,
        "publications_count": 1,
        "speaking_engagements": 0,
        "awards_count": 1,
        "profile_views_received_30d": 100,
        "applications_submitted_30d": 5,
        "active_job_titles_count": 3,
        "recruiter_outreach_count_6m": 15,
    }
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _make_profile(**overrides: Any) -> CandidateProfile:
    defaults: dict[str, Any] = {
        "candidate_id": "test-001",
        "name": "Test Candidate",
        "years_experience": 7.0,
        "roles": [
            CandidateRole(
                title="Software Engineer",
                company="Corp A",
                start_date="2019-06",
                end_date="2022-01",
                duration_months=31,
                scope_keywords=["python", "ml"],
            ),
            CandidateRole(
                title="Senior Engineer",
                company="Corp B",
                start_date="2022-02",
                end_date=None,
                duration_months=0,
                scope_keywords=["llm", "embeddings"],
            ),
        ],
        "skills_claimed": ["python", "pytorch", "llm", "embeddings", "fastapi"],
        "education": [
            {"institution": "Stanford University", "degree": "MS CS", "year": 2019}
        ],
        "redrob_signals": _make_signals(),
    }
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _make_competition_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": "row-001",
        "profile": {
            "name": "Row Candidate",
            "years_of_experience": 6.0,
            "headline": "Engineer",
        },
        "career_history": [
            {
                "title": "Engineer",
                "company": "Corp A",
                "start_date": "2020-01",
                "end_date": "2023-06",
                "duration_months": 42,
            }
        ],
        "skills": [{"name": "python"}, {"name": "pytorch"}],
        "education": [{"institution": "MIT", "degree": "BS CS"}],
        "redrob_signals": {
            "profile_completeness_score": 75.0,
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.6,
        },
    }
    row.update(overrides)
    return row


def _write_jsonl(candidates: list[dict], path: Path | None = None) -> Path:
    if path is None:
        path = Path(tempfile.mktemp(suffix=".jsonl"))
    with open(path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return path


# ---------------------------------------------------------------------------
# models/candidate.py — RedrobSignals
# ---------------------------------------------------------------------------


class TestRedrobSignals:
    def test_defaults_are_valid(self):
        sig = RedrobSignals()
        assert sig.profile_completeness_score == 0.0
        assert sig.open_to_work_flag is False
        assert sig.github_activity_score == 0.0
        assert sig.skill_assessment_scores == {}
        assert sig.preferred_work_mode == "flexible"

    def test_custom_values(self):
        sig = _make_signals(
            profile_completeness_score=95.0,
            open_to_work_flag=False,
            github_activity_score=88.0,
        )
        assert sig.profile_completeness_score == 95.0
        assert sig.open_to_work_flag is False
        assert sig.github_activity_score == 88.0

    def test_skill_assessment_scores_dict(self):
        sig = _make_signals(skill_assessment_scores={"rust": 90.0, "go": 75.0})
        assert sig.skill_assessment_scores == {"rust": 90.0, "go": 75.0}


class TestCandidateProfileRedrobSignals:
    def test_profile_with_signals(self):
        sig = _make_signals()
        profile = _make_profile(redrob_signals=sig)
        assert profile.redrob_signals is not None
        assert profile.redrob_signals.profile_completeness_score == 80.0

    def test_profile_without_signals(self):
        profile = _make_profile(redrob_signals=None)
        assert profile.redrob_signals is None

    def test_profile_validates_with_none_signals(self):
        profile = CandidateProfile(candidate_id="x", redrob_signals=None)
        assert profile.redrob_signals is None


# ---------------------------------------------------------------------------
# pipeline/signals.py
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_midpoint(self):
        assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_at_lower_bound(self):
        assert normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_upper_bound(self):
        assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_below_lower_clamps(self):
        assert normalize(-5.0, 0.0, 10.0) == 0.0

    def test_above_upper_clamps(self):
        assert normalize(15.0, 0.0, 10.0) == 1.0

    def test_equal_bounds_returns_zero(self):
        assert normalize(5.0, 5.0, 5.0) == 0.0


class TestScoreSignals:
    def test_no_signals_returns_neutral(self):
        profile = _make_profile(redrob_signals=None)
        assert score_signals(profile) == 0.5

    def test_score_in_range(self):
        profile = _make_profile()
        score = score_signals(profile)
        assert 0.0 <= score <= 1.0

    def test_high_signals_score_high(self):
        high = _make_profile(
            redrob_signals=_make_signals(
                profile_completeness_score=95.0,
                open_to_work_flag=True,
                recruiter_response_rate=0.95,
                avg_response_time_hours=1.0,
                interview_completion_rate=0.9,
                github_activity_score=90.0,
                github_public_repos=30,
                github_recent_commits_6m=100,
                endorsements_received=40,
                connection_count=500,
                certifications_count=4,
                awards_count=3,
                publications_count=4,
                online_presence_score=90.0,
                profile_views_received_30d=400,
            )
        )
        score = score_signals(high)
        assert score > 0.6

    def test_low_signals_score_low(self):
        low = _make_profile(
            redrob_signals=_make_signals(
                profile_completeness_score=5.0,
                open_to_work_flag=False,
                recruiter_response_rate=0.05,
                avg_response_time_hours=72.0,
                interview_completion_rate=0.1,
                github_activity_score=-1,
                github_public_repos=0,
                github_recent_commits_6m=0,
                endorsements_received=0,
                connection_count=0,
                certifications_count=0,
                awards_count=0,
                publications_count=0,
                online_presence_score=0.0,
                profile_views_received_30d=0,
            )
        )
        score = score_signals(low)
        assert score < 0.3


class TestScoreSignalsBreakdown:
    def test_returns_all_dimensions(self):
        profile = _make_profile()
        bd = score_signals_breakdown(profile)
        assert set(bd.keys()) == {
            "engagement", "responsiveness", "openness",
            "social_proof", "technical", "combined",
        }

    def test_no_signals_all_neutral(self):
        profile = _make_profile(redrob_signals=None)
        bd = score_signals_breakdown(profile)
        for k in ("engagement", "responsiveness", "openness", "social_proof", "technical"):
            assert bd[k] == 0.5
        assert bd["combined"] == 0.5

    def test_all_in_range(self):
        profile = _make_profile()
        bd = score_signals_breakdown(profile)
        for v in bd.values():
            assert 0.0 <= v <= 1.0


class TestSubDimensionScores:
    def test_engagement_in_range(self):
        sig = _make_signals()
        assert 0.0 <= _compute_engagement(sig) <= 1.0

    def test_responsiveness_in_range(self):
        sig = _make_signals()
        assert 0.0 <= _compute_responsiveness(sig) <= 1.0

    def test_openness_in_range(self):
        sig = _make_signals()
        assert 0.0 <= _compute_openness(sig) <= 1.0

    def test_social_proof_in_range(self):
        sig = _make_signals()
        assert 0.0 <= _compute_social_proof(sig) <= 1.0

    def test_technical_in_range(self):
        sig = _make_signals()
        assert 0.0 <= _compute_technical(sig) <= 1.0

    def test_empty_skills_technical_zero(self):
        sig = _make_signals(skill_assessment_scores={})
        score = _compute_technical(sig)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# pipeline/honeypot.py
# ---------------------------------------------------------------------------


class TestParseMonth:
    def test_valid_yyyy_mm(self):
        assert _parse_month("2021-03") == (2021, 3)

    def test_valid_yyyy_mm_dd(self):
        assert _parse_month("2021-03-15") == (2021, 3)

    def test_valid_yyyy_only(self):
        assert _parse_month("2021") == (2021, 1)

    def test_empty_string(self):
        assert _parse_month("") is None

    def test_none(self):
        assert _parse_month(None) is None

    def test_garbage(self):
        assert _parse_month("not-a-date") is None


class TestMonthsBetween:
    def test_same_month(self):
        assert _months_between((2021, 3), (2021, 3)) == 0

    def test_one_year(self):
        assert _months_between((2021, 1), (2022, 1)) == 12

    def test_partial_year(self):
        assert _months_between((2021, 3), (2021, 9)) == 6

    def test_reverse_order(self):
        assert _months_between((2022, 1), (2021, 1)) == 12


class TestDetectTimelineImpossibility:
    def test_no_overlap(self):
        profile = _make_profile(
            roles=[
                CandidateRole(title="A", company="X", start_date="2019-01", end_date="2021-01", duration_months=24),
                CandidateRole(title="B", company="Y", start_date="2021-02", end_date="2023-01", duration_months=23),
            ]
        )
        flagged, severity = detect_timeline_impossibility(profile)
        assert flagged is False
        assert severity == 0.0

    def test_overlap_detected(self):
        profile = _make_profile(
            roles=[
                CandidateRole(title="A", company="X", start_date="2019-01", end_date="2024-01", duration_months=60),
                CandidateRole(title="B", company="Y", start_date="2020-01", end_date="2025-01", duration_months=60),
            ]
        )
        flagged, severity = detect_timeline_impossibility(profile)
        assert flagged is True
        assert severity > 0.0

    def test_single_role_no_flag(self):
        profile = _make_profile(
            roles=[CandidateRole(title="A", company="X", start_date="2020-01", end_date="2023-01", duration_months=36)]
        )
        flagged, _ = detect_timeline_impossibility(profile)
        assert flagged is False


class TestDetectSkillInconsistency:
    def test_no_inconsistency(self):
        profile = _make_profile(
            skills_claimed=["python"],
            redrob_signals=_make_signals(
                endorsements_received=50,
                skill_assessment_scores={"python": 80.0},
            ),
        )
        flagged, severity = detect_skill_inconsistency(profile)
        assert flagged is False

    def test_orphan_skills(self):
        profile = _make_profile(
            skills_claimed=["quantum_computing", "blockchain", "rust"],
            roles=[CandidateRole(title="Data Analyst", company="Corp", start_date="2020-01", end_date="2023-01", duration_months=36, scope_keywords=["excel", "sql"])],
            redrob_signals=_make_signals(endorsements_received=0, skill_assessment_scores={}),
        )
        flagged, severity = detect_skill_inconsistency(profile)
        assert severity > 0.0


class TestDetectYoeCareerMismatch:
    def test_no_mismatch(self):
        profile = _make_profile(years_experience=4.0)
        flagged, severity = detect_yoe_career_mismatch(profile)
        assert flagged is False

    def test_large_mismatch(self):
        profile = _make_profile(
            years_experience=2.0,
            roles=[
                CandidateRole(title="A", company="X", start_date="2010-01", end_date="2024-01", duration_months=168),
            ],
        )
        flagged, severity = detect_yoe_career_mismatch(profile)
        assert flagged is True
        assert severity > 0.0

    def test_no_roles_no_flag(self):
        profile = _make_profile(roles=[], years_experience=5.0)
        flagged, _ = detect_yoe_career_mismatch(profile)
        assert flagged is False


class TestDetectStatisticalOutliers:
    def test_no_outliers(self):
        sig = _make_signals(github_activity_score=50, github_public_repos=10, publications_count=1)
        profile = _make_profile(redrob_signals=sig)
        flagged, _ = detect_statistical_outliers(profile)
        assert flagged is False

    def test_extreme_github(self):
        sig = _make_signals(
            github_activity_score=98, github_public_repos=45,
            publications_count=20,
        )
        profile = _make_profile(redrob_signals=sig, years_experience=2.0)
        flagged, severity = detect_statistical_outliers(profile)
        assert flagged is True

    def test_no_signals_no_flag(self):
        profile = _make_profile(redrob_signals=None)
        flagged, _ = detect_statistical_outliers(profile)
        assert flagged is False


class TestComputeHoneypotScore:
    def test_clean_profile(self):
        profile = _make_profile()
        is_honey, severity = compute_honeypot_score(profile)
        assert isinstance(is_honey, bool)
        assert 0.0 <= severity <= 1.0

    def test_multiple_flags_increases_severity(self):
        profile = _make_profile(
            years_experience=1.0,
            roles=[
                CandidateRole(title="A", company="X", start_date="2010-01", end_date="2024-01", duration_months=168),
                CandidateRole(title="B", company="Y", start_date="2015-01", end_date="2024-01", duration_months=108),
            ],
            skills_claimed=["quantum", "blockchain", "rust", "haskell"],
            redrob_signals=_make_signals(
                github_activity_score=99,
                github_public_repos=50,
                publications_count=20,
                endorsements_received=0,
            ),
        )
        is_honey, severity = compute_honeypot_score(profile)
        assert severity > 0.1


# ---------------------------------------------------------------------------
# pipeline/ingest.py — JSONL loading
# ---------------------------------------------------------------------------


class TestMapCompetitionRow:
    def test_basic_mapping(self):
        row = _make_competition_row()
        profile = _map_competition_row(row, 1)
        assert profile.candidate_id == "row-001"
        assert profile.name == "Row Candidate"
        assert profile.years_experience == 6.0
        assert len(profile.roles) == 1
        assert len(profile.skills_claimed) == 2

    def test_missing_candidate_id_raises(self):
        row = {"profile": {"name": "X"}}
        with pytest.raises(KeyError):
            _map_competition_row(row, 1)

    def test_skills_as_strings(self):
        row = _make_competition_row(skills=["python", "pytorch"])
        profile = _map_competition_row(row, 1)
        assert profile.skills_claimed == ["python", "pytorch"]

    def test_skills_as_dicts(self):
        row = _make_competition_row(skills=[{"name": "python"}, {"name": "pytorch"}])
        profile = _map_competition_row(row, 1)
        assert profile.skills_claimed == ["python", "pytorch"]

    def test_redrob_signals_parsed(self):
        row = _make_competition_row()
        profile = _map_competition_row(row, 1)
        assert profile.redrob_signals is not None
        assert profile.redrob_signals.profile_completeness_score == 75.0

    def test_no_redrob_signals(self):
        row = _make_competition_row()
        del row["redrob_signals"]
        profile = _map_competition_row(row, 1)
        assert profile.redrob_signals is None

    def test_career_history_end_date_computes_duration(self):
        row = _make_competition_row(
            career_history=[
                {
                    "title": "Engineer",
                    "company": "Corp",
                    "start_date": "2020-01",
                    "end_date": "2022-06",
                }
            ]
        )
        profile = _map_competition_row(row, 1)
        assert profile.roles[0].duration_months == 29

    def test_top_level_fallback(self):
        row = {
            "candidate_id": "top-001",
            "name": "Top Level",
            "years_of_experience": 5.0,
        }
        profile = _map_competition_row(row, 1)
        assert profile.candidate_id == "top-001"


class TestLoadCandidatesJsonl:
    def test_loads_all(self):
        rows = [_make_competition_row(candidate_id=f"c-{i}") for i in range(5)]
        path = _write_jsonl(rows)
        profiles = list(load_candidates_jsonl(path))
        assert len(profiles) == 5
        path.unlink()

    def test_max_candidates_cap(self):
        rows = [_make_competition_row(candidate_id=f"c-{i}") for i in range(10)]
        path = _write_jsonl(rows)
        profiles = list(load_candidates_jsonl(path, max_candidates=3))
        assert len(profiles) == 3
        path.unlink()

    def test_skips_malformed_json(self):
        rows = [_make_competition_row(candidate_id="good")]
        path = _write_jsonl(rows)
        with open(path, "a") as f:
            f.write("NOT VALID JSON\n")
        profiles = list(load_candidates_jsonl(path))
        assert len(profiles) == 1
        path.unlink()

    def test_skips_empty_lines(self):
        rows = [_make_competition_row(candidate_id="a")]
        path = _write_jsonl(rows)
        with open(path, "a") as f:
            f.write("\n\n\n")
        profiles = list(load_candidates_jsonl(path))
        assert len(profiles) == 1
        path.unlink()

    def test_empty_file(self):
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        path.touch()
        profiles = list(load_candidates_jsonl(path))
        assert len(profiles) == 0
        path.unlink()

    def test_skips_invalid_candidates(self):
        rows = [
            _make_competition_row(candidate_id="good"),
            {"no_candidate_id": True, "profile": {}},
            _make_competition_row(candidate_id="also-good"),
        ]
        path = _write_jsonl(rows)
        profiles = list(load_candidates_jsonl(path))
        assert len(profiles) == 2
        path.unlink()


# ---------------------------------------------------------------------------
# pipeline/competition.py — scoring functions
# ---------------------------------------------------------------------------


class TestSkillMatchScore:
    def test_perfect_match(self):
        profile = _make_profile(
            skills_claimed=["embeddings", "retrieval", "ranking", "llm", "fine-tuning",
                            "pytorch", "transformers", "rag", "vector", "fastapi"],
            roles=[CandidateRole(title="AI Engineer", company="X", start_date="2020-01",
                                 end_date="2024-01", duration_months=48,
                                 scope_keywords=["ranking", "retrieval", "production"])],
        )
        score = _skill_match_score(profile)
        assert score > 0.6

    def test_no_match(self):
        profile = _make_profile(
            skills_claimed=["cooking", "painting"],
            roles=[CandidateRole(title="Chef", company="Restaurant", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=["cooking"])],
        )
        score = _skill_match_score(profile)
        assert score < 0.3


class TestExperienceFitScore:
    def test_optimal_range(self):
        profile = _make_profile(years_experience=7.0)
        score = _experience_fit_score(profile)
        assert score > 0.5

    def test_below_optimal(self):
        profile = _make_profile(years_experience=1.0)
        score = _experience_fit_score(profile)
        assert score < 0.5

    def test_above_optimal(self):
        profile = _make_profile(years_experience=20.0)
        score = _experience_fit_score(profile)
        assert score < 0.8


class TestTrajectoryEducationScore:
    def test_tier1_school(self):
        profile = _make_profile(
            education=[{"institution": "Stanford University", "degree": "MS"}],
            roles=[CandidateRole(title="Senior Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=["lead", "built", "scaled", "shipped"])],
        )
        score = _trajectory_education_score(profile)
        assert score > 0.5

    def test_no_tier1_school(self):
        profile = _make_profile(
            education=[{"institution": "State University", "degree": "BS"}],
            roles=[CandidateRole(title="Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=["maintained"])],
        )
        score = _trajectory_education_score(profile)
        assert score < 0.6


class TestCompetitionRanker:
    def test_rank_returns_top_n(self):
        rows = [_make_competition_row(candidate_id=f"c-{i:03d}") for i in range(20)]
        path = _write_jsonl(rows)
        ranker = CompetitionRanker()
        ranked = ranker.rank(path, top_n=5)
        assert len(ranked) == 5
        path.unlink()

    def test_scores_are_monotonic(self):
        rows = [_make_competition_row(candidate_id=f"c-{i:03d}") for i in range(10)]
        path = _write_jsonl(rows)
        ranker = CompetitionRanker()
        ranked = ranker.rank(path, top_n=10)
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)
        path.unlink()

    def test_all_have_reasoning(self):
        rows = [_make_competition_row(candidate_id=f"c-{i:03d}") for i in range(5)]
        path = _write_jsonl(rows)
        ranker = CompetitionRanker()
        ranked = ranker.rank(path)
        for r in ranked:
            assert "reasoning" in r
            assert len(r["reasoning"]) <= 400
        path.unlink()

    def test_candidate_ids_preserved(self):
        rows = [_make_competition_row(candidate_id=f"unique-{i}") for i in range(5)]
        path = _write_jsonl(rows)
        ranker = CompetitionRanker()
        ranked = ranker.rank(path)
        ids = {r["candidate_id"] for r in ranked}
        assert ids == {f"unique-{i}" for i in range(5)}
        path.unlink()

    def test_max_candidates_cap(self):
        rows = [_make_competition_row(candidate_id=f"c-{i:03d}") for i in range(50)]
        path = _write_jsonl(rows)
        ranker = CompetitionRanker()
        ranked = ranker.rank(path, top_n=100, max_candidates=10)
        assert len(ranked) == 10
        path.unlink()

    def test_write_csv(self):
        rows = [_make_competition_row(candidate_id=f"c-{i:03d}") for i in range(5)]
        path = _write_jsonl(rows)
        out = Path(tempfile.mktemp(suffix=".csv"))
        ranker = CompetitionRanker()
        ranked = ranker.rank(path, top_n=5)
        ranker.write_csv(ranked, out)
        df = pd.read_csv(out)
        assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
        assert len(df) == 5
        path.unlink()
        out.unlink()


# ---------------------------------------------------------------------------
# output/writer.py — write_competition_csv
# ---------------------------------------------------------------------------


class TestWriteCompetitionCsv:
    def test_writes_correct_columns(self):
        ranked = [
            {"candidate_id": "a", "score": 0.9, "reasoning": "Strong"},
            {"candidate_id": "b", "score": 0.7, "reasoning": "Good"},
        ]
        out = Path(tempfile.mktemp(suffix=".csv"))
        write_competition_csv(ranked, out)
        df = pd.read_csv(out)
        assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
        out.unlink()

    def test_ranks_are_sequential(self):
        ranked = [
            {"candidate_id": "a", "score": 0.9, "reasoning": "x"},
            {"candidate_id": "b", "score": 0.8, "reasoning": "y"},
            {"candidate_id": "c", "score": 0.7, "reasoning": "z"},
        ]
        out = Path(tempfile.mktemp(suffix=".csv"))
        write_competition_csv(ranked, out)
        df = pd.read_csv(out)
        assert df["rank"].tolist() == [1, 2, 3]
        out.unlink()

    def test_scores_rounded_to_6_places(self):
        ranked = [{"candidate_id": "a", "score": 0.123456789, "reasoning": "x"}]
        out = Path(tempfile.mktemp(suffix=".csv"))
        write_competition_csv(ranked, out)
        df = pd.read_csv(out)
        assert df["score"].iloc[0] == pytest.approx(0.123457, abs=1e-5)
        out.unlink()

    def test_reasoning_truncated_to_400(self):
        ranked = [{"candidate_id": "a", "score": 0.5, "reasoning": "x" * 500}]
        out = Path(tempfile.mktemp(suffix=".csv"))
        write_competition_csv(ranked, out)
        df = pd.read_csv(out)
        assert len(df["reasoning"].iloc[0]) <= 400
        out.unlink()

    def test_empty_list(self):
        out = Path(tempfile.mktemp(suffix=".csv"))
        write_competition_csv([], out)
        df = pd.read_csv(out)
        assert len(df) == 0
        assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
        out.unlink()

    def test_creates_parent_dirs(self):
        out = Path(tempfile.mktemp(suffix="")) / "sub" / "submission.csv"
        ranked = [{"candidate_id": "a", "score": 0.5, "reasoning": "x"}]
        write_competition_csv(ranked, out)
        assert out.exists()
        out.unlink()
        out.parent.rmdir()


# ---------------------------------------------------------------------------
# rank.py — CLI
# ---------------------------------------------------------------------------


class TestRankCli:
    def test_parser_required_args(self):
        parser = build_parser()
        args = parser.parse_args(["--candidates", "./data.jsonl"])
        assert args.candidates == Path("./data.jsonl")
        assert args.out == Path("./submission.csv")
        assert args.top_n == 100
        assert args.max_candidates is None
        assert args.verbose is False

    def test_parser_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--candidates", "./in.jsonl",
            "--out", "./out.csv",
            "--top-n", "50",
            "--max-candidates", "1000",
            "-v",
        ])
        assert args.top_n == 50
        assert args.max_candidates == 1000
        assert args.verbose is True

    def test_missing_candidates_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# config.py — competition constants
# ---------------------------------------------------------------------------


class TestCompetitionConfig:
    def test_weights_sum_to_one(self):
        assert sum(config.COMPETITION_WEIGHTS.values()) == pytest.approx(1.0)

    def test_weights_are_positive(self):
        for v in config.COMPETITION_WEIGHTS.values():
            assert v > 0.0

    def test_yoe_range_valid(self):
        assert config.COMPETITION_OPTIMAL_YOE_MIN < config.COMPETITION_OPTIMAL_YOE_MAX

    def test_honeypot_threshold_in_range(self):
        assert 0.0 < config.HONEYPOT_HONEYPOT_RATE_DISQUALIFY < 1.0

    def test_must_have_skills_non_empty(self):
        assert len(config.COMPETITION_MUST_HAVE_SKILLS) > 0

    def test_nice_to_have_skills_non_empty(self):
        assert len(config.COMPETITION_NICE_TO_HAVE_SKILLS) > 0

    def test_trajectory_keywords_non_empty(self):
        assert len(config.COMPETITION_TRAJECTORY_KEYWORDS) > 0

    def test_title_scores_non_empty(self):
        assert len(config.COMPETITION_TITLE_SCORES) > 0

    def test_consulting_firms_non_empty(self):
        assert len(config.COMPETITION_CONSULTING_FIRMS) > 0

    def test_education_field_scores_non_empty(self):
        assert len(config.COMPETITION_EDUCATION_FIELD_SCORES) > 0

    def test_india_locations_non_empty(self):
        assert len(config.COMPETITION_INDIA_LOCATIONS) > 0

    def test_career_description_keywords_non_empty(self):
        assert len(config.COMPETITION_CAREER_DESCRIPTION_KEYWORDS) > 0

    def test_stuffing_title_keywords_non_empty(self):
        assert len(config.COMPETITION_STUFFING_TITLE_KEYWORDS) > 0


class TestTitleRelevanceScore:
    def test_ai_engineer_title(self):
        profile = _make_profile(
            roles=[CandidateRole(title="AI Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _title_relevance_score(profile)
        assert score >= 0.9

    def test_marketing_manager_title(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Marketing Manager", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _title_relevance_score(profile)
        assert score <= 0.2

    def test_unknown_title_returns_neutral(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Random Job Title", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _title_relevance_score(profile)
        assert score == 0.3

    def test_no_roles_returns_neutral(self):
        profile = _make_profile(roles=[])
        score = _title_relevance_score(profile)
        assert score == 0.3

    def test_partial_match(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Senior Software Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _title_relevance_score(profile)
        assert score >= 0.6


class TestCompanyTypeScore:
    def test_all_consulting(self):
        profile = _make_profile(
            roles=[
                CandidateRole(title="Engineer", company="TCS", start_date="2018-01", end_date="2020-01", duration_months=24, scope_keywords=[]),
                CandidateRole(title="Engineer", company="Infosys", start_date="2020-02", end_date="2024-01", duration_months=47, scope_keywords=[]),
            ]
        )
        score = _company_type_score(profile)
        assert score <= 0.25

    def test_no_consulting(self):
        profile = _make_profile(
            roles=[
                CandidateRole(title="Engineer", company="Google", start_date="2018-01", end_date="2020-01", duration_months=24, scope_keywords=[]),
                CandidateRole(title="Engineer", company="Meta", start_date="2020-02", end_date="2024-01", duration_months=47, scope_keywords=[]),
            ]
        )
        score = _company_type_score(profile)
        assert score >= 0.7

    def test_mixed_background(self):
        profile = _make_profile(
            roles=[
                CandidateRole(title="Engineer", company="TCS", start_date="2018-01", end_date="2020-01", duration_months=24, scope_keywords=[]),
                CandidateRole(title="Engineer", company="Google", start_date="2020-02", end_date="2024-01", duration_months=47, scope_keywords=[]),
            ]
        )
        score = _company_type_score(profile)
        assert 0.4 <= score <= 0.8

    def test_no_roles(self):
        profile = _make_profile(roles=[])
        score = _company_type_score(profile)
        assert score == 0.5


class TestEducationFieldScore:
    def test_cs_degree(self):
        profile = _make_profile(
            education=[{"institution": "MIT", "degree": "BS Computer Science", "year": 2018}]
        )
        score = _education_field_score(profile)
        assert score >= 0.9

    def test_mechanical_engineering(self):
        profile = _make_profile(
            education=[{"institution": "State U", "degree": "BS Mechanical Engineering", "year": 2018}]
        )
        score = _education_field_score(profile)
        assert score <= 0.3

    def test_no_education(self):
        profile = _make_profile(education=[])
        score = _education_field_score(profile)
        assert score == 0.3


class TestCareerDescriptionScore:
    def test_rich_ml_keywords(self):
        profile = _make_profile(
            roles=[CandidateRole(
                title="ML Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48,
                scope_keywords=["ranking system", "embeddings", "vector search", "rag", "production"],
            )]
        )
        score = _career_description_score(profile)
        assert score >= 0.5

    def test_no_keywords(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _career_description_score(profile)
        assert score == 0.0

    def test_research_only_keywords(self):
        profile = _make_profile(
            roles=[CandidateRole(
                title="Researcher", company="X", start_date="2020-01", end_date="2024-01", duration_months=48,
                scope_keywords=["research paper", "academic", "thesis"],
            )]
        )
        score = _career_description_score(profile)
        assert score <= 0.3


class TestLocationScore:
    def test_india_company(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Engineer", company="Infosys Pune", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _location_score(profile)
        assert score >= 0.7

    def test_non_india_company(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Engineer", company="Google SF", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])]
        )
        score = _location_score(profile)
        assert score <= 0.6


class TestAntiPatternScore:
    def test_non_technical_title_with_skills(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Marketing Manager", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])],
            skills_claimed=["embeddings", "retrieval", "ranking", "llm"],
        )
        score = _anti_pattern_score(profile)
        assert score >= 0.3

    def test_technical_title_no_penalty(self):
        profile = _make_profile(
            roles=[CandidateRole(title="AI Engineer", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])],
            skills_claimed=["embeddings", "retrieval", "ranking", "llm"],
        )
        score = _anti_pattern_score(profile)
        assert score == 0.0

    def test_non_technical_title_few_skills(self):
        profile = _make_profile(
            roles=[CandidateRole(title="Accountant", company="X", start_date="2020-01", end_date="2024-01", duration_months=48, scope_keywords=[])],
            skills_claimed=["python"],
        )
        score = _anti_pattern_score(profile)
        assert score == 0.0
