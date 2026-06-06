"""Tests for the INGEST and ENRICH phases plus sample-data / code-quality smoke checks.

This module covers the following tasks of the candidate-ranking-system spec, all
written into one cohesive file:

* 4.2 -> Property 1  -- parsed ``.json`` profiles validate with defaults populated.
* 4.3              -- ``ResumeParser`` dispatch and resilience (example tests).
* 5.2 -> Property 2  -- JD classification covers all buckets with valid dimensions.
* 5.3 -> Property 3  -- ``JobDescription`` always validates (incl. malformed input).
* 5.4 -> Property 4  -- ``job_id`` is a deterministic ``uuid5`` of the file path.
* 5.5              -- ``JdParser`` dispatch and error paths (example tests).
* 6.3 -> Property 5  -- trajectory metrics stay in range and are attached.
* 6.4 -> Property 6  -- degenerate trajectory inputs map to defined defaults.
* 6.5 -> Property 7  -- the seniority score is clamped to ``[0, 10]``.
* 6.6              -- an unparseable seniority response defaults to ``5.0``.
* 14.3 -> Property 23 -- sample role dates are consistent with their durations.
* 14.4             -- sample data, config, and code-quality smoke tests.

Testability strategy
--------------------
Every test runs fully offline and deterministically. All LLM inference flows
through ``OllamaClient``, which calls the module-level ``ollama.chat`` function;
the tests construct a real :class:`~utils.ollama_client.OllamaClient` and patch
``ollama.chat`` (via :func:`unittest.mock.patch`) with a fake response shaped like
``{"message": {"content": "<string>"}}`` (Requirement 12.5). The
``SentenceTransformer`` embedding model is never loaded -- these ingest/enrich
checks do not touch the vector store. The spaCy ``en_core_web_sm`` model is never
required: the ``.json`` resume path performs no NLP, and the parser's spaCy
fallback already degrades gracefully when the model is absent.
"""

from __future__ import annotations

import ast
import json
import string
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import config
from models.candidate import CandidateProfile, CandidateRole
from models.job import JobDescription, JobRequirement
from pipeline.enrich import TrajectoryEnricher
from pipeline.ingest import JdParser, ResumeParser
from utils.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Shared helpers, constants, and Hypothesis strategies
# ---------------------------------------------------------------------------

#: Repository root, derived from this file's location so the data/source-file
#: lookups work regardless of the pytest invocation directory.
_ROOT = Path(__file__).resolve().parent.parent

#: The four requirement buckets and dimensions the JD parser must honour
#: (Requirements 3.5, 3.6).
_BUCKETS = ["must_have", "nice_to_have", "culture_signal", "seniority_marker"]
_DIMENSIONS = ["technical", "soft_skill", "domain", "experience_level"]
_COMPLEXITY_ARCS = {"ascending", "descending", "stable", "mixed"}

#: Printable-ASCII text keeps generated values JSON- and filesystem-safe while
#: still exercising punctuation and whitespace.
_SAFE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    max_size=40,
)

#: Filesystem-safe identifier/stem alphabet (no path separators or dots).
_ID_ALPHABET = string.ascii_letters + string.digits + "_-"

#: Optional ``CandidateProfile`` fields and the model defaults that must appear
#: when they are omitted from a ``.json`` resume (Requirements 2.7, 11.4).
_PROFILE_DEFAULTS: dict[str, Any] = {
    "name": "Unknown Candidate",
    "email": None,
    "years_experience": 0.0,
    "roles": [],
    "skills_claimed": [],
    "education": [],
    "trajectory_vector": None,
    "raw_text": "",
}

#: Shared property-test settings: 100 examples per property, deadlines disabled
#: (some examples write temp files), and the function-scoped-fixture / too-slow
#: health checks suppressed so the ``tmp_path`` fixture can be used inside
#: ``@given`` tests (mirroring ``tests/test_embed.py``).
_PROP_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)


def _chat_response(content: str) -> dict[str, dict[str, str]]:
    """Build a fake ``ollama.chat`` response wrapping ``content``.

    Args:
        content: The message content the mocked LLM should return. For
            ``chat_json`` callers this is a (optionally fenced) JSON string; for
            the plain seniority ``chat`` call it is a free-form/number string.

    Returns:
        A dict shaped like a real ``ollama.chat`` response:
        ``{"message": {"content": content}}``.
    """
    return {"message": {"content": content}}


@st.composite
def _role_json(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a minimal valid resume-role dict (title/company/start_date)."""
    return {
        "title": draw(_SAFE_TEXT),
        "company": draw(_SAFE_TEXT),
        "start_date": "2020-01",
    }


@st.composite
def _partial_resume_json(draw: st.DrawFn) -> tuple[dict[str, Any], set[str], bool]:
    """Generate a ``.json`` resume dict with an arbitrary subset of fields omitted.

    Each optional content field is independently included (with a generated
    value) or omitted. ``candidate_id`` is independently included or left out so
    both the "id supplied" and "parser assigns uuid4" paths are exercised.

    Returns:
        A ``(data, omitted, include_id)`` tuple where ``data`` is the dict to
        serialise, ``omitted`` is the set of optional content fields left out
        (each of which must appear at its model default after parsing), and
        ``include_id`` records whether ``candidate_id`` was supplied.
    """
    field_strategies: dict[str, st.SearchStrategy[Any]] = {
        "name": _SAFE_TEXT,
        "email": _SAFE_TEXT,
        "years_experience": st.floats(
            min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
        ),
        "roles": st.lists(_role_json(), max_size=3),
        "skills_claimed": st.lists(_SAFE_TEXT, max_size=5),
        "education": st.lists(
            st.fixed_dictionaries({"institution": _SAFE_TEXT}), max_size=3
        ),
        "trajectory_vector": st.dictionaries(
            st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=8),
            st.integers(min_value=0, max_value=10),
            max_size=3,
        ),
        "raw_text": _SAFE_TEXT,
    }

    data: dict[str, Any] = {}
    omitted: set[str] = set()
    for field, strategy in field_strategies.items():
        if draw(st.booleans()):
            data[field] = draw(strategy)
        else:
            omitted.add(field)

    include_id = draw(st.booleans())
    if include_id:
        data["candidate_id"] = draw(
            st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=12)
        )
    return data, omitted, include_id


@st.composite
def _valid_requirement_dicts(draw: st.DrawFn) -> list[dict[str, str]]:
    """Generate a classification array covering all four buckets.

    Exactly one requirement per bucket is guaranteed (so bucket coverage holds),
    each with a randomly chosen valid dimension, plus up to four extra
    requirements with arbitrary valid bucket/dimension pairs.
    """
    requirements = [
        {
            "text": draw(_SAFE_TEXT),
            "bucket": bucket,
            "dimension": draw(st.sampled_from(_DIMENSIONS)),
        }
        for bucket in _BUCKETS
    ]
    extra = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "text": _SAFE_TEXT,
                    "bucket": st.sampled_from(_BUCKETS),
                    "dimension": st.sampled_from(_DIMENSIONS),
                }
            ),
            max_size=4,
        )
    )
    requirements.extend(extra)
    return requirements


@st.composite
def _enrich_role(draw: st.DrawFn) -> CandidateRole:
    """Generate a varied ``CandidateRole`` for enrichment property tests."""
    return CandidateRole(
        title=draw(
            st.sampled_from(
                [
                    "Intern",
                    "Junior Engineer",
                    "Software Engineer",
                    "Senior Engineer",
                    "Staff Engineer",
                    "Principal Engineer",
                    "Engineering Manager",
                    "Director of Engineering",
                    "VP of Engineering",
                    "Lead Developer",
                ]
            )
        ),
        company=draw(_SAFE_TEXT),
        start_date="2020-01",
        duration_months=draw(st.integers(min_value=0, max_value=120)),
        company_size_estimate=draw(
            st.sampled_from(
                [None, "startup <50", "scaleup 50-500", "enterprise 500+"]
            )
        ),
        scope_keywords=draw(
            st.lists(
                st.sampled_from(
                    ["lead", "mentor", "architect", "team", "backend", "design"]
                ),
                max_size=4,
            )
        ),
    )


@st.composite
def _enrich_profile(draw: st.DrawFn) -> CandidateProfile:
    """Generate a ``CandidateProfile`` with varied roles for enrichment tests."""
    return CandidateProfile(
        candidate_id=draw(st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=12)),
        name=draw(_SAFE_TEXT),
        years_experience=draw(
            st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False)
        ),
        roles=draw(st.lists(_enrich_role(), min_size=0, max_size=5)),
    )


# ===========================================================================
# Task 4.2 -> Property 1
# ===========================================================================


@_PROP_SETTINGS
@given(payload=_partial_resume_json())
def test_property_1_parsed_profiles_validate_with_defaults(payload, tmp_path):
    # Feature: candidate-ranking-system, Property 1: Parsed profiles validate with defaults populated
    """A ``.json`` resume with optional fields omitted parses into a valid
    ``CandidateProfile`` whose omitted fields hold their model defaults.

    Writes the (partial) dict to a temp ``.json`` file, parses it, and asserts
    the result validates, that ``candidate_id`` is populated (supplied or a fresh
    ``uuid4``), that ``is_complete`` is true, and that every omitted optional
    field equals its model default. The ``.json`` path uses no LLM, so
    ``ollama.chat`` must never be called (Requirements 2.7, 2.9, 12.1)."""
    data, omitted, _include_id = payload
    resume_file = tmp_path / "resume.json"
    resume_file.write_text(json.dumps(data), encoding="utf-8")

    with mock.patch("ollama.chat") as mock_chat:
        parser = ResumeParser(OllamaClient())
        profile = parser.parse_file(resume_file)

    assert isinstance(profile, CandidateProfile)
    assert isinstance(profile.candidate_id, str) and profile.candidate_id
    assert profile.is_complete is True
    for field in omitted:
        assert getattr(profile, field) == _PROFILE_DEFAULTS[field], (
            f"omitted field {field!r} did not fall back to its model default"
        )
    mock_chat.assert_not_called()


# ===========================================================================
# Task 4.3 -> ResumeParser dispatch and resilience (example tests)
# ===========================================================================


def test_json_resume_skips_llm_and_marks_complete(tmp_path):
    """A ``.json`` resume loads directly: no LLM call and ``is_complete`` true
    (Requirements 2.4, 2.5)."""
    resume_file = tmp_path / "candidate.json"
    resume_file.write_text(
        json.dumps({"name": "Ada Lovelace", "skills_claimed": ["math"]}),
        encoding="utf-8",
    )

    with mock.patch("ollama.chat") as mock_chat:
        profile = ResumeParser(OllamaClient()).parse_file(resume_file)

    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Ada Lovelace"
    assert profile.is_complete is True
    assert profile.candidate_id  # assigned a uuid4 when absent
    mock_chat.assert_not_called()


def test_pdf_dispatch_extracts_then_calls_llm(tmp_path, monkeypatch):
    """A ``.pdf`` resume extracts text then builds a profile from the mocked LLM
    extraction (Requirements 2.1, 2.3)."""
    pdf_file = tmp_path / "resume.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    parser = ResumeParser(OllamaClient())
    monkeypatch.setattr(parser, "_extract_pdf_text", lambda path: "Resume body text")

    valid_profile = json.dumps(
        {"name": "Grace Hopper", "years_experience": 12.0, "skills_claimed": ["cobol"]}
    )
    with mock.patch("ollama.chat", return_value=_chat_response(valid_profile)) as mock_chat:
        profile = parser.parse_file(pdf_file)

    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Grace Hopper"
    assert profile.is_complete is True
    assert mock_chat.call_count == 1


def test_docx_dispatch_extracts_then_calls_llm(tmp_path, monkeypatch):
    """A ``.docx`` resume extracts text then builds a profile from the mocked LLM
    extraction (Requirements 2.2, 2.3)."""
    docx_file = tmp_path / "resume.docx"
    docx_file.write_bytes(b"PK fake docx")
    parser = ResumeParser(OllamaClient())
    monkeypatch.setattr(parser, "_extract_docx_text", lambda path: "Resume body text")

    valid_profile = json.dumps({"name": "Alan Turing", "skills_claimed": ["logic"]})
    with mock.patch("ollama.chat", return_value=_chat_response(valid_profile)) as mock_chat:
        profile = parser.parse_file(docx_file)

    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Alan Turing"
    assert mock_chat.call_count == 1


def test_unsupported_extension_is_skipped(tmp_path):
    """An unsupported extension is skipped (``None``) without an LLM call
    (Requirement 2.8)."""
    other_file = tmp_path / "resume.rtf"
    other_file.write_text("not supported", encoding="utf-8")

    with mock.patch("ollama.chat") as mock_chat:
        result = ResumeParser(OllamaClient()).parse_file(other_file)

    assert result is None
    mock_chat.assert_not_called()


def test_empty_extracted_text_is_skipped(tmp_path, monkeypatch):
    """Empty/whitespace-only extracted text is skipped without an LLM call
    (Requirement 2.12)."""
    pdf_file = tmp_path / "blank.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    parser = ResumeParser(OllamaClient())
    monkeypatch.setattr(parser, "_extract_pdf_text", lambda path: "   \n\t  ")

    with mock.patch("ollama.chat") as mock_chat:
        result = parser.parse_file(pdf_file)

    assert result is None
    mock_chat.assert_not_called()


def test_validation_retry_once_then_succeeds(tmp_path):
    """An extraction that fails validation triggers exactly one correction retry
    that then succeeds, for two total LLM calls (Requirements 2.9, 2.10)."""
    parser = ResumeParser(OllamaClient())
    invalid = _chat_response(json.dumps({"years_experience": "not-a-number"}))
    valid = _chat_response(json.dumps({"name": "Katherine Johnson", "years_experience": 9}))

    with mock.patch("ollama.chat", side_effect=[invalid, valid]) as mock_chat:
        profile = parser.parse_text("raw resume text", tmp_path / "r.pdf")

    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Katherine Johnson"
    assert mock_chat.call_count == 2


def test_validation_fails_twice_then_skips(tmp_path):
    """Two consecutive validation failures yield ``None`` after the single retry
    (two total LLM calls) (Requirement 2.11)."""
    parser = ResumeParser(OllamaClient())
    invalid = _chat_response(json.dumps({"years_experience": "still-not-a-number"}))

    with mock.patch("ollama.chat", side_effect=[invalid, invalid]) as mock_chat:
        profile = parser.parse_text("raw resume text", tmp_path / "r.pdf")

    assert profile is None
    assert mock_chat.call_count == 2


# ===========================================================================
# Task 5.2 -> Property 2
# ===========================================================================


@_PROP_SETTINGS
@given(requirements=_valid_requirement_dicts())
def test_property_2_jd_covers_all_buckets_valid_dimensions(requirements, tmp_path):
    # Feature: candidate-ranking-system, Property 2: JD classification covers all buckets and uses valid dimensions
    """When the LLM classifies a ``.txt`` JD into requirements that cover every
    bucket, the parsed ``JobDescription`` contains at least one requirement per
    bucket and only valid dimensions (Requirements 3.5, 3.6, 12.2)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("Job description body", encoding="utf-8")

    content = _chat_response(json.dumps(requirements))
    with mock.patch("ollama.chat", return_value=content):
        jd = JdParser(OllamaClient()).parse_file(jd_file)

    assert isinstance(jd, JobDescription)
    present_buckets = {req.bucket for req in jd.requirements}
    assert set(_BUCKETS) <= present_buckets
    for req in jd.requirements:
        assert req.dimension in set(_DIMENSIONS)


# ===========================================================================
# Task 5.3 -> Property 3
# ===========================================================================

#: Content strategy spanning well-formed arrays and assorted malformed/non-JSON
#: payloads (a non-empty min size avoids the empty-response retry path in
#: ``OllamaClient.chat``, which is not part of this property).
_jd_content_strategy = st.one_of(
    _valid_requirement_dicts().map(json.dumps),
    st.text(min_size=1),
    st.sampled_from(
        [
            "{}",
            "[]",
            "null",
            "not json at all",
            "[{",
            '[{"text": "x", "bucket": "bogus", "dimension": "technical"}]',
            "```json\n[]\n```",
        ]
    ),
)


@_PROP_SETTINGS
@given(content=_jd_content_strategy)
def test_property_3_jobdescription_always_validates(content, tmp_path):
    # Feature: candidate-ranking-system, Property 3: JobDescription always validates, including on malformed classification
    """For arbitrary classification content -- valid arrays through outright
    garbage -- ``JdParser.parse_file`` on a ``.txt`` JD returns a valid
    ``JobDescription`` (requirements may be empty) (Requirements 3.8, 3.9)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("Job description body", encoding="utf-8")

    with mock.patch("ollama.chat", return_value=_chat_response(content)):
        jd = JdParser(OllamaClient()).parse_file(jd_file)

    assert isinstance(jd, JobDescription)
    assert isinstance(jd.requirements, list)


# ===========================================================================
# Task 5.4 -> Property 4
# ===========================================================================


@_PROP_SETTINGS
@given(stem=st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=24))
def test_property_4_job_id_is_deterministic_uuid5(stem, tmp_path):
    # Feature: candidate-ranking-system, Property 4: job_id is a deterministic uuid5 of the file path
    """The assigned ``job_id`` equals ``uuid5(NAMESPACE_URL, str(path))`` and is
    identical across repeated parses of the same path (Requirement 3.7)."""
    jd_file = tmp_path / f"{stem}.txt"
    jd_file.write_text("Job description body", encoding="utf-8")
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, str(jd_file)))

    with mock.patch("ollama.chat", return_value=_chat_response("[]")):
        parser = JdParser(OllamaClient())
        first = parser.parse_file(jd_file)
        second = parser.parse_file(jd_file)

    assert first.job_id == expected
    assert second.job_id == expected


# ===========================================================================
# Task 5.5 -> JdParser dispatch and error paths (example tests)
# ===========================================================================


def test_json_jd_direct_load_skips_llm(tmp_path):
    """A ``.json`` JD loads directly with no LLM call (Requirement 3.2)."""
    jd_file = tmp_path / "jd.json"
    jd_file.write_text(
        json.dumps(
            {
                "title": "Backend Engineer",
                "requirements": [
                    {"text": "Python", "bucket": "must_have", "dimension": "technical"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with mock.patch("ollama.chat") as mock_chat:
        jd = JdParser(OllamaClient()).parse_file(jd_file)

    assert isinstance(jd, JobDescription)
    assert jd.title == "Backend Engineer"
    assert len(jd.requirements) == 1
    mock_chat.assert_not_called()


def test_corrupt_json_jd_raises_without_llm(tmp_path):
    """A corrupt ``.json`` JD raises and never invokes the LLM (Requirement 3.4)."""
    jd_file = tmp_path / "broken.json"
    jd_file.write_text("{ this is not valid json", encoding="utf-8")

    with mock.patch("ollama.chat") as mock_chat:
        with pytest.raises(ValueError):
            JdParser(OllamaClient()).parse_file(jd_file)

    mock_chat.assert_not_called()


def test_unsupported_jd_extension_raises(tmp_path):
    """An unsupported JD extension raises, producing no model (Requirement 3.3)."""
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("# job description", encoding="utf-8")

    with mock.patch("ollama.chat") as mock_chat:
        with pytest.raises(ValueError):
            JdParser(OllamaClient()).parse_file(jd_file)

    mock_chat.assert_not_called()


def test_txt_jd_classification_builds_requirements(tmp_path):
    """A ``.txt`` JD is classified by the LLM into requirements (Requirement 3.1)."""
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text("We need a senior engineer.", encoding="utf-8")
    classified = [
        {"text": "5+ years Python", "bucket": "must_have", "dimension": "technical"},
        {"text": "Team player", "bucket": "culture_signal", "dimension": "soft_skill"},
    ]

    with mock.patch("ollama.chat", return_value=_chat_response(json.dumps(classified))):
        jd = JdParser(OllamaClient()).parse_file(jd_file)

    assert isinstance(jd, JobDescription)
    assert len(jd.requirements) == 2
    assert {req.bucket for req in jd.requirements} == {"must_have", "culture_signal"}


# ===========================================================================
# Task 6.3 -> Property 5
# ===========================================================================


@_PROP_SETTINGS
@given(profile=_enrich_profile())
def test_property_5_trajectory_metrics_in_range_and_attached(profile):
    # Feature: candidate-ranking-system, Property 5: Trajectory metrics stay in range and are attached
    """After ``enrich``, the profile carries a non-``None`` trajectory dict whose
    bounded metrics lie in ``[0, 1]`` and whose ``complexity_arc`` is one of the
    four defined values (Requirements 4.1, 4.6)."""
    with mock.patch("ollama.chat", return_value=_chat_response("5")):
        enriched = TrajectoryEnricher(OllamaClient()).enrich(profile)

    trajectory = enriched.trajectory_vector
    assert isinstance(trajectory, dict)
    assert 0.0 <= trajectory["growth_rate"] <= 1.0
    assert 0.0 <= trajectory["leadership_progression"] <= 1.0
    assert 0.0 <= trajectory["tenure_consistency"] <= 1.0
    assert trajectory["complexity_arc"] in _COMPLEXITY_ARCS


# ===========================================================================
# Task 6.4 -> Property 6
# ===========================================================================


def test_property_6_zero_roles_map_to_defaults():
    # Feature: candidate-ranking-system, Property 6: Degenerate trajectory inputs map to defined defaults
    """With zero roles, the pure ``compute_*`` methods return their defined
    defaults: tenure 1.0, growth 0.0, leadership 0.0, arc ``stable``
    (Requirement 4.5)."""
    assert TrajectoryEnricher.compute_tenure_consistency([]) == 1.0
    assert TrajectoryEnricher.compute_growth_rate([], 10.0) == 0.0
    assert TrajectoryEnricher.compute_leadership_progression([]) == 0.0
    assert TrajectoryEnricher.compute_complexity_arc([]) == "stable"


@_PROP_SETTINGS
@given(roles=st.lists(_enrich_role(), max_size=5))
def test_property_6_zero_years_gives_zero_growth(roles):
    # Feature: candidate-ranking-system, Property 6: Degenerate trajectory inputs map to defined defaults
    """``years_experience == 0`` forces ``growth_rate`` to ``0.0`` for any roles
    (Requirement 4.5)."""
    assert TrajectoryEnricher.compute_growth_rate(roles, 0.0) == 0.0


@_PROP_SETTINGS
@given(
    size=st.sampled_from([None, "startup <50", "scaleup 50-500", "enterprise 500+"]),
    count=st.integers(min_value=1, max_value=5),
)
def test_property_6_fewer_than_two_distinct_sizes_is_stable(size, count):
    # Feature: candidate-ranking-system, Property 6: Degenerate trajectory inputs map to defined defaults
    """Fewer than two distinct ``company_size_estimate`` values yield a
    ``stable`` complexity arc (Requirement 4.5)."""
    roles = [
        CandidateRole(
            title="Engineer",
            company="Acme",
            start_date="2020-01",
            company_size_estimate=size,
        )
        for _ in range(count)
    ]
    assert TrajectoryEnricher.compute_complexity_arc(roles) == "stable"


# ===========================================================================
# Task 6.5 -> Property 7
# ===========================================================================

#: Numeric-ish seniority responses, including out-of-range and embedded numbers.
_seniority_content_strategy = st.one_of(
    st.integers(min_value=-100, max_value=100).map(str),
    st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ).map(lambda value: f"{value:.2f}"),
    st.sampled_from(
        ["42", "-5", "7.5", "0", "10", "11", "-0.1", "  8 ", "score is 9 of 10"]
    ),
)


@_PROP_SETTINGS
@given(content=_seniority_content_strategy)
def test_property_7_seniority_score_clamped(content):
    # Feature: candidate-ranking-system, Property 7: Seniority score is clamped to [0, 10]
    """An arbitrary numeric seniority response is clamped into ``[0, 10]``
    (Requirements 4.2, 4.3)."""
    profile = CandidateProfile(
        candidate_id="c1",
        name="Test Candidate",
        years_experience=5.0,
        roles=[CandidateRole(title="Engineer", company="Acme", start_date="2020-01")],
    )

    with mock.patch("ollama.chat", return_value=_chat_response(content)):
        enriched = TrajectoryEnricher(OllamaClient()).enrich(profile)

    score = enriched.trajectory_vector["seniority_score"]
    assert 0.0 <= score <= 10.0


# ===========================================================================
# Task 6.6 -> unparseable seniority default (example test)
# ===========================================================================


def test_unparseable_seniority_defaults_to_five():
    """A non-numeric seniority response defaults to ``5.0`` without raising
    (Requirement 4.4)."""
    profile = CandidateProfile(
        candidate_id="c2",
        name="Test Candidate",
        years_experience=3.0,
        roles=[CandidateRole(title="Engineer", company="Acme", start_date="2020-01")],
    )

    with mock.patch("ollama.chat", return_value=_chat_response("no idea, hard to say")):
        enriched = TrajectoryEnricher(OllamaClient()).enrich(profile)

    assert enriched.trajectory_vector["seniority_score"] == 5.0


# ===========================================================================
# Task 14.3 -> Property 23
# ===========================================================================

#: Reference "current" month used by the sample data for roles with no end date.
_CURRENT_REFERENCE = (2024, 6)


def _parse_year_month(value: str) -> tuple[int, int]:
    """Parse a ``YYYY-MM`` (or ``YYYY-MM-DD``) date string to a ``(year, month)``.

    Args:
        value: An ISO-like date string; only the year and month are significant.

    Returns:
        A ``(year, month)`` tuple.
    """
    parts = value.split("-")
    return int(parts[0]), int(parts[1])


def _whole_months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    """Return the whole number of months between two ``(year, month)`` pairs.

    Args:
        start: The earlier ``(year, month)`` pair.
        end: The later ``(year, month)`` pair.

    Returns:
        ``(end_year - start_year) * 12 + (end_month - start_month)``.
    """
    return (end[0] - start[0]) * 12 + (end[1] - start[1])


def test_property_23_sample_role_dates_consistent_with_duration():
    # Feature: candidate-ranking-system, Property 23: Sample role dates are consistent with duration
    """For every role in every sample profile, ``start_date <= end_date`` (a null
    end date is treated as the ``2024-06`` reference) and ``duration_months``
    equals the whole months between the dates (Requirement 11.5)."""
    profiles = json.loads(
        (_ROOT / "data" / "sample_candidates.json").read_text(encoding="utf-8")
    )

    checked = 0
    for profile in profiles:
        for role in profile["roles"]:
            start = _parse_year_month(role["start_date"])
            end = (
                _parse_year_month(role["end_date"])
                if role["end_date"]
                else _CURRENT_REFERENCE
            )
            assert start <= end, (
                f"{profile['name']} role {role['title']!r}: start {start} after end {end}"
            )
            assert role["duration_months"] == _whole_months_between(start, end), (
                f"{profile['name']} role {role['title']!r}: duration_months "
                f"{role['duration_months']} != months between {start} and {end}"
            )
            checked += 1

    assert checked > 0  # the dataset actually exercised the property


# ===========================================================================
# Task 14.4 -> sample data, config, and code-quality smoke tests
# ===========================================================================


def test_sample_candidates_count_and_validate():
    """``sample_candidates.json`` has exactly 15 entries, each a valid
    ``CandidateProfile`` (Requirements 11.1, 11.4)."""
    data = json.loads(
        (_ROOT / "data" / "sample_candidates.json").read_text(encoding="utf-8")
    )
    assert isinstance(data, list)
    assert len(data) == 15
    for obj in data:
        profile = CandidateProfile(**obj)
        assert isinstance(profile, CandidateProfile)


def test_sample_job_description_validates_with_bucket_coverage():
    """``sample_job_description.json`` validates as a ``JobDescription`` with at
    least one requirement per bucket (Requirements 11.2, 11.3)."""
    data = json.loads(
        (_ROOT / "data" / "sample_job_description.json").read_text(encoding="utf-8")
    )
    jd = JobDescription(**data)
    present_buckets = {req.bucket for req in jd.requirements}
    assert set(_BUCKETS) <= present_buckets


def test_config_constants_present_and_valid():
    """All tunable config constants are present with valid ranges and the
    calibration set is balanced 5+5 (Requirements 10.1, 10.2, 10.3, 10.4)."""
    outcomes = [example["outcome"] for example in config.CALIBRATION_EXAMPLES]
    assert len(config.CALIBRATION_EXAMPLES) == 10
    assert outcomes.count("strong_hire") == 5
    assert outcomes.count("no_hire") == 5

    assert config.PERSONA_WEIGHTS == {
        "hiring_manager": 0.45,
        "peer_interviewer": 0.35,
        "devils_advocate": -0.20,
    }
    assert config.BIAS_FLAG_THRESHOLD == 0.75
    assert config.HUMAN_REVIEW_VARIANCE_THRESHOLD == 2.5

    for token_limit in (
        config.MAX_TOKENS_EXTRACTION,
        config.MAX_TOKENS_SCORING,
        config.MAX_TOKENS_NARRATIVE,
    ):
        assert isinstance(token_limit, int) and token_limit > 0

    assert 0.0 <= config.SCORING_TEMPERATURE <= 1.0

    assert config.TITLE_LEVELS
    assert config.COUNTERFACTUAL_NAME_PAIRS
    assert config.INSTITUTION_SWAPS


#: First-party package directories scanned for module docstrings / print usage.
_PACKAGE_DIRS = ["pipeline", "audit", "models", "output", "utils"]


def _first_party_modules() -> list[Path]:
    """Return the first-party module files (``config``, ``main``, and packages).

    Returns:
        Sorted ``.py`` paths for ``config.py``, ``main.py``, and every module
        under the first-party package directories.
    """
    modules = [_ROOT / "config.py", _ROOT / "main.py"]
    for package in _PACKAGE_DIRS:
        modules.extend(sorted((_ROOT / package).glob("*.py")))
    return modules


def _uses_print_call(source: str) -> bool:
    """Return ``True`` when ``source`` contains a bare ``print(...)`` call.

    Uses the AST (not a text scan) so attribute calls like ``console.print(...)``
    and the substring ``print(`` inside docstrings/comments do not register.

    Args:
        source: Python source text to analyse.

    Returns:
        ``True`` if a ``print`` builtin call node is present, else ``False``.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            return True
    return False


def test_every_first_party_module_has_module_docstring():
    """Every first-party module carries a non-empty module docstring
    (Requirement 13.1)."""
    for module_path in _first_party_modules():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        assert docstring and docstring.strip(), f"{module_path} lacks a module docstring"


def test_non_cli_modules_avoid_print_for_diagnostics():
    """No non-CLI first-party module uses ``print(...)`` for diagnostics; only
    ``main.py`` (the CLI, which writes setup messages to stderr) is exempt
    (Requirement 13.4)."""
    for module_path in _first_party_modules():
        if module_path.name == "main.py":
            continue
        assert not _uses_print_call(
            module_path.read_text(encoding="utf-8")
        ), f"{module_path} uses print() for diagnostics"
