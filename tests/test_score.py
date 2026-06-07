"""Tests for the SCORE, AUDIT, OUTPUT, and CLI surfaces of the pipeline.

This module covers the following tasks of the candidate-ranking-system spec, all
written into this single file:

* ``3.3`` / ``3.4`` -- ``utils.ollama_client.OllamaClient`` JSON fallback
  (Property 24) and retry/backoff example tests.
* ``9.4``-``9.8`` -- ``pipeline.score.CandidateScoringPipeline`` composite score
  (Property 13), panel variance + human review (Property 14), per-persona score
  range (Property 12), schema-complete deduplicated result (Property 15), plus
  scoring-resilience and RAG-assembly example tests.
* ``11.4``-``11.8`` -- ``audit.counterfactual.CounterfactualAuditor`` swap
  correctness (Property 16), shared-store twin id (Property 17), delta/bias-flag
  (Property 18), flag-rate (Property 19), plus audit-lifecycle example tests.
* ``12.3``-``12.5`` -- ``output.writer`` ranking (Property 20), verdict consensus
  (Property 22), and CSV encoding (Property 21).
* ``13.4`` -- ``main`` CLI argument parsing, logging configuration, and startup
  verification example tests.

Testability strategy
--------------------
The whole suite runs fully offline and deterministically. Every LLM call flows
through ``ollama.chat`` (the single seam, Requirement 12.5) and is mocked with
``unittest.mock.patch("ollama.chat")``; the CLI's reachability/model checks call
the module-level ``ollama.list`` and are mocked the same way. ``time.sleep`` is
patched so retry/backoff is asserted without real delays. No
``SentenceTransformer`` model is ever loaded: scoring/audit tests inject a small
``FakeStore`` stub (exposing ``query_jd_context``/``query_calibration``) into the
``CandidateScoringPipeline`` and use stub parser/enricher/scorer collaborators
for the auditor, so no embeddings are needed.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from typing import Any, Iterable
from unittest import mock

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import config
import main
from audit.counterfactual import (
    AUDIT_LOG,
    CounterfactualAuditor,
    reset_audit_log,
)
from models.candidate import CandidateProfile, CandidateRole
from output.writer import rank_candidates, verdict_consensus, write_ranked_csv
from pipeline.score import CandidateScoringPipeline
from utils.ollama_client import (
    OllamaCallError,
    OllamaClient,
    _strip_code_fences,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: The complete set of keys a score result dict must contain (Requirement 6.11,
#: Property 15). Kept independent of the source so the test asserts the schema
#: rather than trusting the implementation.
EXPECTED_RESULT_KEYS = frozenset(
    {
        "candidate_id",
        "name",
        "trajectory_score",
        "hiring_manager_score",
        "peer_interviewer_score",
        "devils_advocate_score",
        "composite_score",
        "panel_variance",
        "requires_human_review",
        "persona_verdicts",
        "strengths",
        "concerns",
        "narrative",
        "bias_flag",
        "counterfactual_delta",
    }
)

#: The exact 16 CSV columns, in order, the ranked output must use
#: (Requirement 8.4, Property 21). Hard-coded so the test does not depend on the
#: writer's private constant.
EXPECTED_CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "name",
    "composite_score",
    "trajectory_score",
    "hiring_manager_score",
    "peer_interviewer_score",
    "devils_advocate_score",
    "panel_variance",
    "requires_human_review",
    "verdict_consensus",
    "strengths",
    "concerns",
    "narrative",
    "bias_flag",
    "counterfactual_delta",
]

#: The four allowed persona verdicts (Requirement 6.11 / 8.5).
VERDICTS = ["strong_yes", "yes", "maybe", "no"]

#: Bidirectional gendered-pronoun map mirrored from the design (he/him/his <->
#: she/her/hers). Defined here so Property 16 validates against an independent
#: expectation rather than the implementation's private table.
PRONOUN_MAP = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "hers",
    "hers": "his",
}

#: Shared Hypothesis settings: the spec mandates >= 100 examples per property.
#: ``deadline=None`` and the health-check suppressions keep the property tests
#: stable when they lean on per-function fixtures (``tmp_path``) or do a little
#: more work per example.
PBT_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_response(content: str) -> dict:
    """Build a fake ``ollama.chat`` response dict.

    Args:
        content: The message content the mocked chat call should return.

    Returns:
        dict: A response shaped like the Ollama SDK's
        ``{"message": {"content": ...}}``.
    """
    return {"message": {"content": content}}


def _persona_payload(
    *,
    score: Any = 7.0,
    confidence: Any = 0.8,
    strengths: list | None = None,
    concerns: list | None = None,
    verdict: str = "yes",
) -> dict:
    """Build a persona response payload dict (pre-serialization).

    Args:
        score: The persona score value (any type, to exercise coercion).
        confidence: The persona confidence value.
        strengths: The persona strengths list; defaults to a single item.
        concerns: The persona concerns list; defaults to a single item.
        verdict: The persona verdict string.

    Returns:
        dict: The persona payload ready to be JSON-encoded.
    """
    return {
        "score": score,
        "confidence": confidence,
        "strengths": strengths if strengths is not None else ["strength"],
        "concerns": concerns if concerns is not None else ["concern"],
        "verdict": verdict,
    }


def _persona_json(**kwargs: Any) -> str:
    """Return a JSON-encoded persona payload (see :func:`_persona_payload`)."""
    return json.dumps(_persona_payload(**kwargs))


def make_profile(
    *,
    candidate_id: str = "cand-1",
    name: str = "Test Candidate",
    raw_text: str = "Test Candidate resume body.",
    trajectory: dict | None = None,
) -> CandidateProfile:
    """Build a minimal enriched ``CandidateProfile`` for scoring/audit tests.

    The profile carries a populated ``trajectory_vector`` (so ``score`` can read
    ``seniority_score``) and a single role, which is enough to render the RAG
    query and persona prompt blocks without any embeddings.

    Args:
        candidate_id: The candidate id.
        name: The candidate name.
        raw_text: The resume body (used by counterfactual twin construction).
        trajectory: Optional trajectory-vector dict; a sensible default is used
            when omitted.

    Returns:
        CandidateProfile: The constructed profile.
    """
    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        years_experience=5.0,
        skills_claimed=["python", "go"],
        roles=[
            CandidateRole(title="Engineer", company="Acme", start_date="2019-01-01")
        ],
        education=[{"degree": "BS", "institution": "State University"}],
        trajectory_vector=trajectory
        or {
            "seniority_score": 6.0,
            "growth_rate": 0.5,
            "complexity_arc": "ascending",
            "leadership_progression": 0.4,
            "tenure_consistency": 0.8,
        },
        raw_text=raw_text,
    )


class FakeStore:
    """A stub vector store exposing only the two methods the scorer calls.

    Records the ``n`` argument passed to each retrieval method so the RAG
    assembly test can assert the scorer requested 5 JD items and 3 calibration
    items, and returns fixed, inspectable context so the items can be located in
    the persona prompts.
    """

    def __init__(
        self,
        jd_items: list[str] | None = None,
        calibration_items: list[dict] | None = None,
    ) -> None:
        self.jd_items = (
            jd_items
            if jd_items is not None
            else [f"JD requirement {i}" for i in range(5)]
        )
        self.calibration_items = (
            calibration_items
            if calibration_items is not None
            else [
                {"outcome": "strong_hire", "reason": f"calibration reason {i}"}
                for i in range(3)
            ]
        )
        self.jd_calls: list[tuple[str, int]] = []
        self.calibration_calls: list[tuple[str, int]] = []

    def query_jd_context(self, query: str, n: int) -> list[str]:
        self.jd_calls.append((query, n))
        return list(self.jd_items)

    def query_calibration(self, query: str, n: int) -> list[dict]:
        self.calibration_calls.append((query, n))
        return list(self.calibration_items)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer_no_store() -> CandidateScoringPipeline:
    """A scoring pipeline with no vector store (empty RAG context).

    Module-scoped so it does not trip Hypothesis' function-scoped-fixture health
    check when used by property tests. The pipeline loads the three persona
    prompt files once; it is otherwise stateless across examples (the LLM seam is
    re-patched per example).
    """
    return CandidateScoringPipeline(OllamaClient(), None)


@pytest.fixture(autouse=True)
def _isolate_audit_log() -> Iterable[None]:
    """Clear the shared ``AUDIT_LOG`` before and after every test.

    Keeps audit-lifecycle tests independent. Property tests that append many
    entries across examples additionally reset within their own body.
    """
    reset_audit_log()
    yield
    reset_audit_log()


# ===========================================================================
# Task 3.3 -> Property 24: chat_json falls back on non-JSON without raising
# ===========================================================================


def _is_invalid_json(text: str) -> bool:
    """Return True when ``text`` (after fence stripping) is NOT valid JSON."""
    try:
        json.loads(_strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        return True
    return False


@PBT_SETTINGS
@given(
    content=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=120,
    ).filter(lambda s: bool(s.strip()) and _is_invalid_json(s))
)
def test_chat_json_falls_back_on_non_json(content: str) -> None:
    # Feature: candidate-ranking-system, Property 24: chat_json falls back on non-JSON without raising
    """For any non-JSON content, ``chat_json`` returns the supplied fallback and
    never raises (Requirement 13.9).

    ``ollama.chat`` is patched to return the generated (invalid-JSON) content;
    the JSON parse must fail, triggering the fallback substitution rather than an
    exception.
    """
    fallback = {"fallback": True, "value": 42}
    client = OllamaClient()

    with mock.patch("ollama.chat", return_value=_chat_response(content)):
        result = client.chat_json([{"role": "user", "content": "x"}], fallback=fallback)

    assert result == fallback


def test_chat_json_returns_parsed_value_on_valid_json() -> None:
    """A valid JSON body (optionally fenced) is parsed and returned, not the
    fallback (Requirement 13.9 control case)."""
    client = OllamaClient()
    fenced = "```json\n{\"score\": 9, \"verdict\": \"yes\"}\n```"

    with mock.patch("ollama.chat", return_value=_chat_response(fenced)):
        result = client.chat_json(
            [{"role": "user", "content": "x"}], fallback={"fallback": True}
        )

    assert result == {"score": 9, "verdict": "yes"}


# ===========================================================================
# Task 3.4 -> OllamaClient retry/backoff example tests
# (Requirements 13.6, 13.7, 13.8, 13.10)
# ===========================================================================


def test_chat_retries_three_times_then_raises() -> None:
    """When every attempt fails, ``chat`` makes exactly 3 attempts, backs off
    with ``time.sleep(1)`` then ``time.sleep(2)``, and finally raises
    ``OllamaCallError`` (Requirements 13.7, 13.8)."""
    client = OllamaClient()

    with mock.patch("ollama.chat", side_effect=RuntimeError("boom")) as chat_mock, \
        mock.patch("utils.ollama_client.time.sleep") as sleep_mock:
        with pytest.raises(OllamaCallError):
            client.chat([{"role": "user", "content": "x"}])

    assert chat_mock.call_count == 3
    # Exponential backoff: 1s before the first retry, 2s before the second.
    assert [call.args[0] for call in sleep_mock.call_args_list] == [1, 2]


def test_chat_succeeds_on_third_attempt() -> None:
    """``chat`` returns the content when the third attempt succeeds after two
    failures, sleeping twice in between (Requirement 13.7)."""
    client = OllamaClient()
    side_effects = [
        RuntimeError("fail-1"),
        RuntimeError("fail-2"),
        _chat_response("recovered content"),
    ]

    with mock.patch("ollama.chat", side_effect=side_effects) as chat_mock, \
        mock.patch("utils.ollama_client.time.sleep") as sleep_mock:
        result = client.chat([{"role": "user", "content": "x"}])

    assert result == "recovered content"
    assert chat_mock.call_count == 3
    assert sleep_mock.call_count == 2


def test_chat_json_propagates_ollama_call_error() -> None:
    """A non-JSON-parse failure (``OllamaCallError`` from exhausted retries)
    propagates through ``chat_json`` rather than being swallowed into the
    fallback (Requirement 13.10)."""
    client = OllamaClient()

    with mock.patch("ollama.chat", side_effect=RuntimeError("down")), \
        mock.patch("utils.ollama_client.time.sleep"):
        with pytest.raises(OllamaCallError):
            client.chat_json(
                [{"role": "user", "content": "x"}], fallback={"fallback": True}
            )


def test_chat_emits_debug_request_and_response_logging(caplog) -> None:
    """``chat`` logs request and response detail at DEBUG level
    (Requirement 13.6)."""
    client = OllamaClient()

    with mock.patch("ollama.chat", return_value=_chat_response("hello world")), \
        caplog.at_level(logging.DEBUG, logger="utils.ollama_client"):
        client.chat([{"role": "user", "content": "ping"}])

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "ollama.chat request" in messages
    assert "ollama.chat response" in messages


# ===========================================================================
# Scoring helpers: a persona-dispatching ollama.chat mock
# ===========================================================================


def make_chat_dispatch(
    *,
    hiring_manager: str | None = None,
    peer_interviewer: str | None = None,
    devils_advocate: str | None = None,
    narrative: str = "Strong engineer. One concern remains. More evidence would help.",
    raise_on_narrative: bool = False,
):
    """Build an ``ollama.chat`` side-effect that routes by the system prompt.

    The scorer issues four chat calls per candidate -- one per persona then one
    for the narrative -- all through ``ollama.chat``. This dispatcher inspects the
    system message to decide which persona is calling and returns that persona's
    configured content, so a single ``patch("ollama.chat")`` can drive the whole
    panel deterministically.

    Args:
        hiring_manager: Raw content for the hiring-manager persona call.
        peer_interviewer: Raw content for the peer-interviewer persona call.
        devils_advocate: Raw content for the devil's-advocate persona call.
        narrative: Content returned for the narrative call.
        raise_on_narrative: When True, the narrative call raises (so retries are
            exhausted and ``_generate_narrative`` falls back).

    Returns:
        Callable: A side-effect suitable for ``mock.patch("ollama.chat", side_effect=...)``.
    """
    hm = hiring_manager if hiring_manager is not None else _persona_json(score=7.0, verdict="yes")
    peer = peer_interviewer if peer_interviewer is not None else _persona_json(score=6.0, verdict="yes")
    da = devils_advocate if devils_advocate is not None else _persona_json(score=3.0, verdict="no")

    def _dispatch(*_args: Any, **kwargs: Any) -> dict:
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system.startswith("You write"):
            if raise_on_narrative:
                raise RuntimeError("narrative backend down")
            return _chat_response(narrative)
        if "senior hiring manager" in system:
            return _chat_response(hm)
        if "work directly alongside" in system:
            return _chat_response(peer)
        if "case AGAINST" in system:
            return _chat_response(da)
        raise AssertionError(f"unexpected system prompt: {system!r}")

    return _dispatch


# ===========================================================================
# Task 9.4 -> Property 13: Composite score is the rounded weighted sum clamped
# ===========================================================================


def _expected_composite(hm: float, peer: float, da: float) -> float:
    """Reference composite: weighted sum, rounded to 2dp, clamped to [0, 10]."""
    scores = {
        "hiring_manager": hm,
        "peer_interviewer": peer,
        "devils_advocate": da,
    }
    weighted = sum(
        weight * scores[persona]
        for persona, weight in config.PERSONA_WEIGHTS.items()
    )
    return float(max(0.0, min(10.0, round(weighted, 2))))


@PBT_SETTINGS
@given(
    hm=st.floats(min_value=0.0, max_value=10.0),
    peer=st.floats(min_value=0.0, max_value=10.0),
    da=st.floats(min_value=0.0, max_value=10.0),
)
def test_composite_score_valid_range(hm: float, peer: float, da: float) -> None:
    # Feature: candidate-ranking-system, Property 13: Composite score is the rounded weighted sum clamped to [0, 10]
    """For in-range persona scores, the composite equals the rounded weighted sum
    clamped to [0, 10] and always lies in [0, 10] (Requirements 6.3, 6.4)."""
    scores = {
        "hiring_manager": hm,
        "peer_interviewer": peer,
        "devils_advocate": da,
    }
    composite = CandidateScoringPipeline.composite_score(scores)
    assert composite == _expected_composite(hm, peer, da)
    assert 0.0 <= composite <= 10.0


@PBT_SETTINGS
@given(
    hm=st.floats(min_value=-50.0, max_value=50.0),
    peer=st.floats(min_value=-50.0, max_value=50.0),
    da=st.floats(min_value=-50.0, max_value=50.0),
)
def test_composite_score_clamps_out_of_range(hm: float, peer: float, da: float) -> None:
    # Feature: candidate-ranking-system, Property 13: Composite score is the rounded weighted sum clamped to [0, 10]
    """Even for out-of-range persona inputs, the composite stays clamped to
    [0, 10] and matches the clamped reference computation (Requirement 6.4)."""
    scores = {
        "hiring_manager": hm,
        "peer_interviewer": peer,
        "devils_advocate": da,
    }
    composite = CandidateScoringPipeline.composite_score(scores)
    assert composite == _expected_composite(hm, peer, da)
    assert 0.0 <= composite <= 10.0


# ===========================================================================
# Task 9.5 -> Property 14: Panel variance is population variance + human review
# ===========================================================================


@PBT_SETTINGS
@given(
    hm=st.floats(min_value=0.0, max_value=10.0),
    peer=st.floats(min_value=0.0, max_value=10.0),
    da=st.floats(min_value=0.0, max_value=10.0),
)
def test_panel_variance_is_population_variance(hm: float, peer: float, da: float) -> None:
    # Feature: candidate-ranking-system, Property 14: Panel variance is population variance and drives human review
    """``panel_variance`` equals the population variance of the three scores
    (mean of squared deviations) and the review threshold is exactly 2.5
    (Requirements 6.5, 6.6, 6.7)."""
    scores = {
        "hiring_manager": hm,
        "peer_interviewer": peer,
        "devils_advocate": da,
    }
    variance = CandidateScoringPipeline.panel_variance(scores)
    assert variance == pytest.approx(statistics.pvariance([hm, peer, da]))
    assert variance >= 0.0
    # The human-review rule is "strictly greater than 2.5".
    assert config.HUMAN_REVIEW_VARIANCE_THRESHOLD == 2.5


@pytest.mark.parametrize(
    "triple",
    [
        (5.0, 5.0, 5.0),  # variance 0.0 -> below
        (4.0, 5.0, 6.0),  # variance ~0.667 -> below
        (3.2, 5.0, 6.8),  # variance 2.16 -> below
        (3.0, 5.0, 7.0),  # variance ~2.667 -> above
        (2.0, 5.0, 8.0),  # variance 6.0 -> above
        (0.0, 5.0, 10.0),  # variance ~16.67 -> above
    ],
)
def test_requires_human_review_follows_variance_threshold(
    scorer_no_store, triple
) -> None:
    # Feature: candidate-ranking-system, Property 14: Panel variance is population variance and drives human review
    """``score`` sets ``requires_human_review`` iff panel variance > 2.5, for
    triples that straddle the threshold (Requirements 6.6, 6.7)."""
    hm, peer, da = triple
    dispatch = make_chat_dispatch(
        hiring_manager=_persona_json(score=hm, verdict="yes"),
        peer_interviewer=_persona_json(score=peer, verdict="yes"),
        devils_advocate=_persona_json(score=da, verdict="no"),
    )
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    expected_variance = statistics.pvariance([hm, peer, da])
    assert result["panel_variance"] == pytest.approx(expected_variance)
    assert result["requires_human_review"] == (expected_variance > 2.5)


# ===========================================================================
# Task 9.6 -> Property 12: Each persona score is in [0, 10]
# ===========================================================================

#: Arbitrary, possibly-garbage score values: wide finite floats, out-of-range
#: ints, non-numeric junk, and missing-ish values. NaN/inf are excluded since
#: the property concerns range, not IEEE special values.
_GARBAGE_SCORE = st.one_of(
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=8),
    st.none(),
    st.booleans(),
)


@PBT_SETTINGS
@given(hm=_GARBAGE_SCORE, peer=_GARBAGE_SCORE, da=_GARBAGE_SCORE)
def test_each_persona_score_in_range(scorer_no_store, hm, peer, da) -> None:
    # Feature: candidate-ranking-system, Property 12: Each persona score is in [0, 10]
    """For any (even garbage/out-of-range) persona response, each of the three
    persona scores in the result is coerced/clamped to [0, 10] (Requirement 6.1)."""
    dispatch = make_chat_dispatch(
        hiring_manager=_persona_json(score=hm, verdict="yes"),
        peer_interviewer=_persona_json(score=peer, verdict="maybe"),
        devils_advocate=_persona_json(score=da, verdict="no"),
    )
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    for key in (
        "hiring_manager_score",
        "peer_interviewer_score",
        "devils_advocate_score",
    ):
        assert 0.0 <= result[key] <= 10.0


# ===========================================================================
# Task 9.7 -> Property 15: Score result is schema-complete with dedup lists
# ===========================================================================

#: A small alphabet of strength/concern strings so overlap (and thus dedup) is
#: exercised across the three personas.
_ITEM_POOL = ["a", "b", "c", "d", "e"]


@PBT_SETTINGS
@given(
    hm_strengths=st.lists(st.sampled_from(_ITEM_POOL), max_size=4),
    peer_strengths=st.lists(st.sampled_from(_ITEM_POOL), max_size=4),
    da_concerns=st.lists(st.sampled_from(_ITEM_POOL), max_size=4),
    hm_concerns=st.lists(st.sampled_from(_ITEM_POOL), max_size=4),
)
def test_result_schema_complete_and_deduplicated(
    scorer_no_store, hm_strengths, peer_strengths, da_concerns, hm_concerns
) -> None:
    # Feature: candidate-ranking-system, Property 15: Score result is schema-complete with deduplicated lists
    """The result dict has every required key and its ``strengths``/``concerns``
    lists contain no duplicates, even when personas report overlapping items
    (Requirement 6.11)."""
    dispatch = make_chat_dispatch(
        hiring_manager=_persona_json(
            score=7.0, strengths=hm_strengths, concerns=hm_concerns, verdict="yes"
        ),
        peer_interviewer=_persona_json(
            score=6.0, strengths=peer_strengths, concerns=hm_concerns, verdict="yes"
        ),
        devils_advocate=_persona_json(
            score=3.0, strengths=peer_strengths, concerns=da_concerns, verdict="no"
        ),
    )
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    assert set(result.keys()) == set(EXPECTED_RESULT_KEYS)
    assert len(result["strengths"]) == len(set(result["strengths"]))
    assert len(result["concerns"]) == len(set(result["concerns"]))


# ===========================================================================
# Task 9.8 -> scoring resilience + RAG assembly example tests
# (Requirements 6.2, 6.8, 6.9, 6.10)
# ===========================================================================


def test_unparseable_persona_substitutes_default_and_continues(scorer_no_store) -> None:
    """A persona whose response is not valid JSON contributes the default score
    5.0 and verdict ``maybe`` while the remaining personas score normally and
    scoring completes (Requirement 6.9)."""
    dispatch = make_chat_dispatch(
        hiring_manager="not json at all {{{",
        peer_interviewer=_persona_json(score=8.0, verdict="strong_yes"),
        devils_advocate=_persona_json(score=2.0, verdict="no"),
    )
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    # Default substitution for the unparseable hiring-manager persona.
    assert result["hiring_manager_score"] == 5.0
    assert result["persona_verdicts"]["hiring_manager"] == "maybe"
    # The other personas scored normally; scoring continued to completion.
    assert result["peer_interviewer_score"] == 8.0
    assert result["devils_advocate_score"] == 2.0
    assert result["persona_verdicts"]["peer_interviewer"] == "strong_yes"


def test_narrative_is_used_when_generated(scorer_no_store) -> None:
    """The generated narrative string is carried through into the result
    (Requirement 6.10)."""
    narrative = "Primary strength noted. Main concern flagged. More evidence requested."
    dispatch = make_chat_dispatch(narrative=narrative)
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    assert result["narrative"] == narrative


def test_narrative_failure_falls_back(scorer_no_store) -> None:
    """When the narrative call raises ``OllamaCallError`` (retries exhausted), a
    fallback narrative is used and scoring still succeeds (Requirement 6.10)."""
    dispatch = make_chat_dispatch(
        hiring_manager=_persona_json(score=7.0, concerns=[], verdict="yes"),
        peer_interviewer=_persona_json(score=6.0, concerns=[], verdict="yes"),
        devils_advocate=_persona_json(
            score=3.0, concerns=["limited scale experience"], verdict="no"
        ),
        raise_on_narrative=True,
    )
    with mock.patch("ollama.chat", side_effect=dispatch), \
        mock.patch("utils.ollama_client.time.sleep"):
        result = scorer_no_store.score(make_profile())

    # Fallback narrative is the first concern when the LLM narrative call fails.
    assert result["narrative"] == "limited scale experience"


def test_rag_context_assembly_passes_five_jd_and_three_calibration() -> None:
    """The scorer queries the store for 5 JD items and 3 calibration items, and
    those retrieved items appear in every persona's user prompt (Requirement 6.2)."""
    jd_items = [f"UNIQUE_JD_{i}" for i in range(5)]
    calibration_items = [
        {"outcome": "strong_hire", "reason": f"UNIQUE_CAL_{i}"} for i in range(3)
    ]
    store = FakeStore(jd_items=jd_items, calibration_items=calibration_items)
    scorer = CandidateScoringPipeline(OllamaClient(), store)

    dispatch = make_chat_dispatch()
    with mock.patch("ollama.chat", side_effect=dispatch) as chat_mock:
        scorer.score(make_profile())

    # The store was queried with the documented retrieval sizes.
    assert store.jd_calls and all(n == 5 for _, n in store.jd_calls)
    assert store.calibration_calls and all(n == 3 for _, n in store.calibration_calls)

    # Collect the user prompts from the three persona calls (skip the narrative).
    persona_prompts = [
        call.kwargs["messages"][1]["content"]
        for call in chat_mock.call_args_list
        if not call.kwargs["messages"][0]["content"].startswith("You write")
    ]
    assert len(persona_prompts) == 3
    for prompt in persona_prompts:
        for jd in jd_items:
            assert jd in prompt
        for cal in calibration_items:
            assert cal["reason"] in prompt


def test_counterfactual_delta_default_is_nonnegative_float(scorer_no_store) -> None:
    """A freshly scored candidate has ``counterfactual_delta`` defaulting to 0.0
    (a non-negative float) before any audit runs (Requirement 6.11)."""
    dispatch = make_chat_dispatch()
    with mock.patch("ollama.chat", side_effect=dispatch):
        result = scorer_no_store.score(make_profile())

    assert isinstance(result["counterfactual_delta"], float)
    assert result["counterfactual_delta"] >= 0.0
    assert result["counterfactual_delta"] == 0.0
    assert result["bias_flag"] is False


# ===========================================================================
# Audit stubs (no LLM, no embeddings)
# ===========================================================================


class StubParser:
    """A ResumeParser stand-in whose ``parse_text`` returns a fixed profile.

    Records every ``parse_text`` call. When ``return_none`` is set it returns
    ``None`` to simulate a twin re-extraction that failed validation twice.
    """

    def __init__(self, return_none: bool = False) -> None:
        self.return_none = return_none
        self.calls: list[tuple[str, Any]] = []

    def parse_text(self, raw_text: str, source_path: Any) -> CandidateProfile | None:
        self.calls.append((raw_text, source_path))
        if self.return_none:
            return None
        return make_profile(candidate_id="reparsed", raw_text=raw_text)


class StubEnricher:
    """A TrajectoryEnricher stand-in whose ``enrich`` returns the profile as-is."""

    def __init__(self) -> None:
        self.calls: list[CandidateProfile] = []

    def enrich(self, profile: CandidateProfile) -> CandidateProfile:
        self.calls.append(profile)
        if profile.trajectory_vector is None:
            profile.trajectory_vector = {"seniority_score": 5.0}
        return profile


class RecordingScorer:
    """A CandidateScoringPipeline stand-in returning a fixed twin composite.

    Holds a ``store`` attribute so a test can assert the twin is scored against
    the same store instance, and records the profile passed to ``score``.
    """

    def __init__(self, store: Any, cf_composite: float = 5.0) -> None:
        self.store = store
        self.cf_composite = cf_composite
        self.scored: list[CandidateProfile] = []

    def score(self, profile: CandidateProfile) -> dict:
        self.scored.append(profile)
        return {
            "candidate_id": profile.candidate_id,
            "name": profile.name,
            "composite_score": self.cf_composite,
            "persona_verdicts": {
                "hiring_manager": "yes",
                "peer_interviewer": "yes",
                "devils_advocate": "no",
            },
        }


def _base_result(composite: float, candidate_id: str = "cand-1") -> dict:
    """A minimal original score result for the auditor to update in place."""
    return {
        "candidate_id": candidate_id,
        "name": "Test Candidate",
        "composite_score": composite,
        "bias_flag": False,
        "counterfactual_delta": 0.0,
    }


# ===========================================================================
# Task 11.4 -> Property 16: swaps are whole-word, case-insensitive, unconfigured
#                            tokens unchanged
# ===========================================================================

#: Bidirectional name map mirrored from config for independent expectations.
_NAME_MAP = {}
for _orig, _swapped in config.COUNTERFACTUAL_NAME_PAIRS:
    _NAME_MAP[_orig] = _swapped
    _NAME_MAP[_swapped] = _orig
_NAME_LOOKUP = {key.lower(): value for key, value in _NAME_MAP.items()}

#: Single-word institution keys only (multi-word phrases excluded so the
#: space-joined token stream cannot accidentally form a phrase key).
_SINGLE_WORD_INSTITUTIONS = {
    key: value for key, value in config.INSTITUTION_SWAPS.items() if " " not in key
}
_INSTITUTION_LOOKUP = {key.lower(): value for key, value in _SINGLE_WORD_INSTITUTIONS.items()}

#: Unconfigured filler words guaranteed not to be any configured name, pronoun,
#: or institution. Includes "the"/"here" which embed pronoun substrings but must
#: NOT be swapped (whole-word boundary guarantee).
_FILLERS = ["the", "here", "engineer", "worked", "led", "team", "project", "and"]


def _recase(token: str, how: str) -> str:
    """Apply a casing transform identified by ``how`` to ``token``."""
    if how == "upper":
        return token.upper()
    if how == "lower":
        return token.lower()
    if how == "capitalize":
        return token.capitalize()
    return token


def _match_case(original: str, replacement: str) -> str:
    """Replicate the implementation's pronoun case-matching for expectations."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


@st.composite
def _token_plan(draw):
    """Generate a list of (input_token, expected_token) pairs across categories.

    Each entry is a configured name, a single-word institution, a gendered
    pronoun (with randomized casing), or an unconfigured filler. The expected
    output token mirrors ``build_twin``'s rules: names/institutions are replaced
    with their mapped value verbatim, pronouns are case-matched, and fillers are
    left unchanged.
    """
    count = draw(st.integers(min_value=1, max_value=12))
    pairs: list[tuple[str, str]] = []
    for _ in range(count):
        category = draw(st.sampled_from(["name", "institution", "pronoun", "filler"]))
        casing = draw(st.sampled_from(["identity", "upper", "lower", "capitalize"]))
        if category == "name":
            base = draw(st.sampled_from(list(_NAME_MAP)))
            token = _recase(base, casing)
            pairs.append((token, _NAME_LOOKUP[token.lower()]))
        elif category == "institution":
            base = draw(st.sampled_from(list(_SINGLE_WORD_INSTITUTIONS)))
            token = _recase(base, casing)
            pairs.append((token, _INSTITUTION_LOOKUP[token.lower()]))
        elif category == "pronoun":
            base = draw(st.sampled_from(list(PRONOUN_MAP)))
            token = _recase(base, casing)
            pairs.append((token, _match_case(token, PRONOUN_MAP[token.lower()])))
        else:
            base = draw(st.sampled_from(_FILLERS))
            token = _recase(base, casing)
            pairs.append((token, token))
    return pairs


@PBT_SETTINGS
@given(plan=_token_plan())
def test_build_twin_swaps_whole_word_case_insensitive(plan) -> None:
    # Feature: candidate-ranking-system, Property 16: Counterfactual swaps are whole-word, case-insensitive, and leave unconfigured tokens unchanged
    """``build_twin`` swaps every configured name, pronoun, and institution on
    whole-word, case-insensitive boundaries while leaving unconfigured tokens
    unchanged (Requirement 7.1)."""
    inputs = [pair[0] for pair in plan]
    expected_tokens = [pair[1] for pair in plan]

    text = " ".join(inputs)
    expected = " ".join(expected_tokens)

    auditor = CounterfactualAuditor(None, None, None)
    twin = auditor.build_twin(make_profile(raw_text=text))

    assert twin == expected


def test_pronoun_swap_leaves_the_and_here_unchanged() -> None:
    # Feature: candidate-ranking-system, Property 16: Counterfactual swaps are whole-word, case-insensitive, and leave unconfigured tokens unchanged
    """``swap_pronouns`` must not touch substrings: "the" and "here" embed "he"
    but are not whole-word pronouns (Requirement 7.1)."""
    text = "the engineer worked here and he led her team"
    swapped = CounterfactualAuditor.swap_pronouns(text)
    assert swapped == "the engineer worked here and she led him team"


def test_unconfigured_name_left_unchanged() -> None:
    # Feature: candidate-ranking-system, Property 16: Counterfactual swaps are whole-word, case-insensitive, and leave unconfigured tokens unchanged
    """A name not present in ``COUNTERFACTUAL_NAME_PAIRS`` is left unchanged by
    twin construction (Requirement 7.1)."""
    assert "Alexander" not in {n for pair in config.COUNTERFACTUAL_NAME_PAIRS for n in pair}
    auditor = CounterfactualAuditor(None, None, None)
    twin = auditor.build_twin(make_profile(raw_text="Alexander is an engineer"))
    assert "Alexander" in twin


# ===========================================================================
# Task 11.5 -> Property 17: twin reuses the same store with a cf_-prefixed id
# ===========================================================================


@PBT_SETTINGS
@given(
    candidate_id=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=12,
    )
)
def test_twin_uses_cf_prefixed_id_and_same_store(candidate_id: str) -> None:
    # Feature: candidate-ranking-system, Property 17: The twin reuses the same store with a cf_-prefixed id
    """``audit`` re-scores the twin through the same scorer/store, assigning the
    twin ``candidate_id = "cf_" + original`` (Requirement 7.2)."""
    store = FakeStore()
    parser = StubParser()
    enricher = StubEnricher()
    scorer = RecordingScorer(store, cf_composite=5.0)
    auditor = CounterfactualAuditor(parser, enricher, scorer)

    profile = make_profile(candidate_id=candidate_id)
    result = _base_result(composite=5.0, candidate_id=candidate_id)
    auditor.audit(profile, result)

    assert len(scorer.scored) == 1
    twin = scorer.scored[0]
    assert twin.candidate_id == f"cf_{candidate_id}"
    # The twin is scored against the very same store instance as the original.
    assert scorer.store is store


# ===========================================================================
# Task 11.6 -> Property 18: delta is a non-negative rounded diff driving the flag
# ===========================================================================


@PBT_SETTINGS
@given(
    original=st.floats(min_value=0.0, max_value=10.0),
    twin=st.floats(min_value=0.0, max_value=10.0),
)
def test_counterfactual_delta_and_bias_flag(original: float, twin: float) -> None:
    # Feature: candidate-ranking-system, Property 18: Counterfactual delta is a non-negative rounded difference that drives the bias flag
    """After ``audit``, ``counterfactual_delta == round(abs(orig - twin), 2) >= 0``
    and ``bias_flag`` is True iff the delta exceeds 0.75 (Requirements 7.3-7.5)."""
    store = FakeStore()
    scorer = RecordingScorer(store, cf_composite=twin)
    auditor = CounterfactualAuditor(StubParser(), StubEnricher(), scorer)

    result = _base_result(composite=original)
    auditor.audit(make_profile(), result)

    expected_delta = round(abs(original - twin), 2)
    assert result["counterfactual_delta"] == expected_delta
    assert result["counterfactual_delta"] >= 0.0
    assert result["bias_flag"] == (expected_delta > config.BIAS_FLAG_THRESHOLD)


# ===========================================================================
# Task 11.7 -> Property 19: flag rate equals flagged over total
# ===========================================================================


@PBT_SETTINGS
@given(flags=st.lists(st.booleans(), max_size=20))
def test_flag_rate_equals_flagged_over_total(tmp_path, flags) -> None:
    # Feature: candidate-ranking-system, Property 19: Flag rate equals flagged over total
    """The report's ``flag_rate`` equals flagged/total over successful audit
    entries, and is 0 when nothing was audited (Requirement 7.7)."""
    reset_audit_log()
    for index, flagged in enumerate(flags):
        AUDIT_LOG.append(
            {
                "candidate_id": f"cand-{index}",
                "name": f"Candidate {index}",
                "original_score": 5.0,
                "cf_score": 6.0 if flagged else 5.0,
                "delta": 1.0 if flagged else 0.0,
                "bias_flag": flagged,
            }
        )

    auditor = CounterfactualAuditor(StubParser(), StubEnricher(), RecordingScorer(None))
    report_path = auditor.write_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    total = len(flags)
    flagged_count = sum(1 for flag in flags if flag)
    expected_rate = flagged_count / total if total > 0 else 0

    assert report["total_candidates_audited"] == total
    assert report["flagged_count"] == flagged_count
    assert report["flag_rate"] == pytest.approx(expected_rate)
    if total == 0:
        assert report["flag_rate"] == 0


# ===========================================================================
# Task 11.8 -> audit lifecycle example tests (Requirements 7.6, 7.8, 7.9)
# ===========================================================================


def test_audit_updates_result_in_place_and_appends_log() -> None:
    """``audit`` mutates the original result in place (bias_flag +
    counterfactual_delta) and appends a success entry to ``AUDIT_LOG``
    (Requirement 7.6)."""
    reset_audit_log()
    store = FakeStore()
    scorer = RecordingScorer(store, cf_composite=8.0)
    auditor = CounterfactualAuditor(StubParser(), StubEnricher(), scorer)

    result = _base_result(composite=5.0, candidate_id="cand-7")
    entry = auditor.audit(make_profile(candidate_id="cand-7"), result)

    # delta = |5.0 - 8.0| = 3.0 > 0.75 -> flagged.
    assert result["counterfactual_delta"] == 3.0
    assert result["bias_flag"] is True
    assert len(AUDIT_LOG) == 1
    assert AUDIT_LOG[0] is entry
    assert entry["candidate_id"] == "cand-7"
    assert entry["bias_flag"] is True


def test_write_report_skipped_zeroes_counts(tmp_path) -> None:
    """``write_report(skipped=True)`` writes an ``audit_skipped`` report with all
    counts zeroed (Requirement 7.8)."""
    reset_audit_log()
    # Even with entries present, the skipped report ignores the log.
    AUDIT_LOG.append({"candidate_id": "x", "name": "X", "bias_flag": True, "delta": 9.0})

    auditor = CounterfactualAuditor(StubParser(), StubEnricher(), RecordingScorer(None))
    report_path = auditor.write_report(tmp_path, skipped=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["audit_skipped"] is True
    assert report["total_candidates_audited"] == 0
    assert report["flagged_count"] == 0
    assert report["flag_rate"] == 0
    assert report["flagged_candidates"] == []


def test_audit_twin_failure_records_failure_and_does_not_raise() -> None:
    """When twin re-parsing returns ``None``, ``audit`` records an audit-failure
    entry, leaves ``bias_flag`` False, and does not raise (Requirement 7.9)."""
    reset_audit_log()
    parser = StubParser(return_none=True)
    scorer = RecordingScorer(FakeStore(), cf_composite=9.0)
    auditor = CounterfactualAuditor(parser, StubEnricher(), scorer)

    result = _base_result(composite=5.0, candidate_id="cand-9")
    entry = auditor.audit(make_profile(candidate_id="cand-9"), result)  # must not raise

    assert result["bias_flag"] is False
    assert entry["audit_failure"] is True
    assert entry["cf_score"] is None
    # The failing twin was never scored.
    assert scorer.scored == []
    assert len(AUDIT_LOG) == 1
    assert AUDIT_LOG[0]["audit_failure"] is True


# ===========================================================================
# Output result-dict strategy
# ===========================================================================

#: Simple tokens free of CSV-significant characters (no pipe/comma/quote/newline)
#: so pipe-join round-tripping is unambiguous.
_SAFE_WORD = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=6,
)


@st.composite
def _result_dicts(draw, min_size: int = 0, max_size: int = 8) -> list[dict]:
    """Generate a list of full score-result dicts with unique candidate ids.

    Each dict carries every key consumed by the writer (Requirement 8.4),
    including a ``persona_verdicts`` mapping for the three personas. Candidate
    ids are made unique by index so ranking is fully deterministic.
    """
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    results: list[dict] = []
    for index in range(count):
        results.append(
            {
                "candidate_id": f"cand-{index:03d}",
                "name": draw(_SAFE_WORD),
                "composite_score": draw(
                    st.floats(
                        min_value=0.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
                "trajectory_score": draw(
                    st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
                ),
                "hiring_manager_score": draw(
                    st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
                ),
                "peer_interviewer_score": draw(
                    st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
                ),
                "devils_advocate_score": draw(
                    st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
                ),
                "panel_variance": draw(
                    st.floats(min_value=0.0, max_value=25.0, allow_nan=False)
                ),
                "requires_human_review": draw(st.booleans()),
                "persona_verdicts": {
                    "hiring_manager": draw(st.sampled_from(VERDICTS)),
                    "peer_interviewer": draw(st.sampled_from(VERDICTS)),
                    "devils_advocate": draw(st.sampled_from(VERDICTS)),
                },
                "strengths": draw(st.lists(_SAFE_WORD, max_size=4)),
                "concerns": draw(st.lists(_SAFE_WORD, max_size=4)),
                "narrative": draw(_SAFE_WORD),
                "bias_flag": draw(st.booleans()),
                "counterfactual_delta": draw(
                    st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
                ),
            }
        )
    return results


# ===========================================================================
# Task 12.3 -> Property 20: ranking is correctly ordered with consecutive ranks
# ===========================================================================


@PBT_SETTINGS
@given(results=_result_dicts(min_size=0, max_size=10))
def test_ranking_ordered_with_consecutive_ranks(results: list[dict]) -> None:
    # Feature: candidate-ranking-system, Property 20: Ranking is correctly ordered with consecutive ranks
    """``rank_candidates`` orders by composite descending, breaks ties by
    candidate_id ascending, and assigns ranks 1..N with no gaps
    (Requirements 8.1, 8.2)."""
    ranked = rank_candidates(results)

    # Same population, no rows lost or added.
    assert len(ranked) == len(results)

    # Ranks are exactly the consecutive integers 1..N in order.
    assert [row["rank"] for row in ranked] == list(range(1, len(results) + 1))

    # Order matches the reference sort key (-composite, candidate_id).
    expected_order = sorted(
        results, key=lambda r: (-r["composite_score"], r["candidate_id"])
    )
    assert [row["candidate_id"] for row in ranked] == [
        r["candidate_id"] for r in expected_order
    ]

    # Pairwise: each row's composite is >= the next, and ties break by id.
    for first, second in zip(ranked, ranked[1:]):
        assert (first["composite_score"], second["candidate_id"]) >= (
            second["composite_score"],
            first["candidate_id"],
        ) or first["composite_score"] > second["composite_score"]


# ===========================================================================
# Task 12.4 -> Property 22: verdict consensus is the majority else hiring mgr's
# ===========================================================================


@PBT_SETTINGS
@given(
    hm=st.sampled_from(VERDICTS),
    peer=st.sampled_from(VERDICTS),
    da=st.sampled_from(VERDICTS),
)
def test_verdict_consensus_majority_else_hiring_manager(hm, peer, da) -> None:
    # Feature: candidate-ranking-system, Property 22: Verdict consensus is the majority, else the hiring manager's verdict
    """``verdict_consensus`` returns the verdict held by >= 2 personas, else the
    hiring manager's verdict (Requirements 8.5, 8.6)."""
    verdicts = {
        "hiring_manager": hm,
        "peer_interviewer": peer,
        "devils_advocate": da,
    }
    consensus = verdict_consensus(verdicts)

    counts = Counter([hm, peer, da])
    majority = [value for value, count in counts.items() if count >= 2]
    if majority:
        # With three personas at most one verdict can reach a count of >= 2.
        assert consensus == majority[0]
    else:
        # All three disagree -> fall back to the hiring manager's verdict.
        assert consensus == hm


# ===========================================================================
# Task 12.5 -> Property 21: CSV columns fixed + correct list serialization
# ===========================================================================


@PBT_SETTINGS
@given(results=_result_dicts(min_size=0, max_size=6))
def test_csv_columns_and_list_serialization(tmp_path, results: list[dict]) -> None:
    # Feature: candidate-ranking-system, Property 21: CSV encoding has fixed columns and correct list serialization
    """``write_ranked_csv`` writes the 16 fixed columns in order, one data row per
    candidate, with ``strengths``/``concerns`` pipe-joined and an empty list
    serialized as the empty string (Requirement 8.4)."""
    import pandas as pd

    csv_path = write_ranked_csv(results, tmp_path)

    # keep_default_na=False so an empty-string cell is read back as "" not NaN.
    frame = pd.read_csv(csv_path, keep_default_na=False, dtype=str)

    # Exactly the 16 columns, in the required order.
    assert list(frame.columns) == EXPECTED_CSV_COLUMNS

    # One data row per candidate.
    assert len(frame) == len(results)

    # strengths/concerns are pipe-joined; empty list -> empty string. Compare per
    # candidate by id (the file is ranked, so look each row up by candidate_id).
    by_id = {row["candidate_id"]: row for row in results}
    for _, csv_row in frame.iterrows():
        source = by_id[csv_row["candidate_id"]]
        assert csv_row["strengths"] == "|".join(source["strengths"])
        assert csv_row["concerns"] == "|".join(source["concerns"])
        if not source["strengths"]:
            assert csv_row["strengths"] == ""
        if not source["concerns"]:
            assert csv_row["concerns"] == ""


# ===========================================================================
# Task 13.4 -> CLI example tests (Requirements 9.1, 9.2, 9.3, 9.11, 9.12, 1.7, 1.8)
# ===========================================================================


def test_parse_args_defaults() -> None:
    """``parse_args([])`` yields the documented defaults for all flags
    (Requirement 9.1)."""
    args = main.parse_args([])
    assert args.candidates_dir == "./data/candidates/"
    assert args.job_description == "./data/sample_job_description.json"
    assert args.output_dir == "./output/"
    assert args.skip_audit is False
    assert args.verbose is False


def test_parse_args_flags_and_paths() -> None:
    """``--skip-audit`` and ``--verbose`` are store_true flags and the three path
    options are captured (Requirement 9.1)."""
    args = main.parse_args(
        [
            "--candidates-dir",
            "/tmp/cands",
            "--job-description",
            "/tmp/jd.json",
            "--output-dir",
            "/tmp/out",
            "--skip-audit",
            "--verbose",
        ]
    )
    assert args.candidates_dir == "/tmp/cands"
    assert args.job_description == "/tmp/jd.json"
    assert args.output_dir == "/tmp/out"
    assert args.skip_audit is True
    assert args.verbose is True


def test_configure_logging_sets_debug_when_verbose() -> None:
    """``configure_logging(True)`` sets the root level to DEBUG (Requirement 9.11)."""
    main.configure_logging(True)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_sets_info_when_not_verbose() -> None:
    """``configure_logging(False)`` sets the root level to INFO (Requirement 9.12)."""
    main.configure_logging(False)
    assert logging.getLogger().level == logging.INFO


def test_check_ollama_reachable_returns_when_list_succeeds() -> None:
    """``check_ollama_reachable`` returns normally when ``ollama.list`` succeeds
    (Requirement 9.2)."""
    with mock.patch("main.ollama.list", return_value={"models": []}):
        # Must not raise / exit.
        main.check_ollama_reachable(timeout=5.0)


def test_check_ollama_reachable_exits_on_failure() -> None:
    """``check_ollama_reachable`` exits with status 1 when ``ollama.list`` raises
    (Requirements 9.2, 9.3)."""
    with mock.patch("main.ollama.list", side_effect=ConnectionError("refused")):
        with pytest.raises(SystemExit) as exc_info:
            main.check_ollama_reachable(timeout=5.0)
    assert exc_info.value.code == 1


def test_check_models_present_ok_when_model_listed() -> None:
    """``check_models_present`` returns normally when the configured LLMs are among
    the installed models (Requirements 1.7, 1.8)."""
    required = set(
        (config.OLLAMA_MODELS_LOW_MEM if config.LOW_MEMORY_MODE else config.OLLAMA_MODELS).values()
    )
    response = {"models": [{"model": m} for m in required]}
    with mock.patch("main.ollama.list", return_value=response):
        main.check_models_present()


def test_check_models_present_exits_when_model_missing() -> None:
    """``check_models_present`` exits with status 1 when the configured LLM is not
    installed locally (Requirements 1.7, 1.8)."""
    response = {"models": [{"model": "some-other-model:1b"}]}
    with mock.patch("main.ollama.list", return_value=response):
        with pytest.raises(SystemExit) as exc_info:
            main.check_models_present()
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Groq Backend Tests
# ---------------------------------------------------------------------------


def test_verify_startup_groq_missing_key() -> None:
    """verify_startup exits with status 1 when LLM_BACKEND is 'groq' but GROQ_API_KEY is missing."""
    with mock.patch("config.LLM_BACKEND", "groq"), \
         mock.patch("config.GROQ_API_KEY", ""), \
         mock.patch.dict("os.environ", {"GROQ_API_KEY": ""}), \
         pytest.raises(SystemExit) as exc_info:
        main.verify_startup()
    assert exc_info.value.code == 1


def test_verify_startup_groq_valid_key() -> None:
    """verify_startup succeeds when LLM_BACKEND is 'groq' and GROQ_API_KEY is provided."""
    with mock.patch("config.LLM_BACKEND", "groq"), \
         mock.patch("config.GROQ_API_KEY", "mock-key"), \
         mock.patch("importlib.util.find_spec", return_value=mock.MagicMock()):
        main.verify_startup()  # Should not raise


def test_ollama_client_uses_groq_completions() -> None:
    """OllamaClient.chat calls groq client completions when LLM_BACKEND is 'groq'."""
    mock_response = mock.MagicMock()
    mock_response.choices[0].message.content = "mocked groq response"

    mock_groq_instance = mock.MagicMock()
    mock_groq_instance.chat.completions.create.return_value = mock_response

    with mock.patch("config.LLM_BACKEND", "groq"), \
         mock.patch("config.GROQ_API_KEY", "mock-key"), \
         mock.patch("utils.ollama_client.Groq", return_value=mock_groq_instance):
        client = OllamaClient()
        response = client.chat([{"role": "user", "content": "hello"}])
        assert response == "mocked groq response"
        mock_groq_instance.chat.completions.create.assert_called_once()

