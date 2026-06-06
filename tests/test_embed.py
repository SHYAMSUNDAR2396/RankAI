"""Tests for the EMBED & STORE phase (``pipeline.embed.VectorStoreManager``).

This module covers tasks 8.3-8.7 of the candidate-ranking-system spec, validating
correctness Properties 8-11 plus the integration behaviors of Requirement 5 (Embed
and Store Vectors).

Testability strategy
--------------------
These tests must run fully offline and fast, so the heavy ``SentenceTransformer``
model is never loaded. Instead the module-level embedding helpers
``pipeline.embed.embed_text`` / ``pipeline.embed.embed_texts`` are patched (module
autouse fixture :func:`_patch_embeddings`) with small, deterministic fake vectors
derived from a SHA-256 hash of the input text. ChromaDB accepts arbitrary-but-
consistent vector dimensions, so a fixed 8-dim fake keeps the store happy while
avoiding any model download or network access.

ChromaDB itself is exercised for real via a local ``PersistentClient`` pointed at a
``pytest`` ``tmp_path`` (on-disk, no server). Because the embeddings are faked, exact
nearest-neighbour ordering is not deterministic; the retrieval property tests
therefore assert the strong, deterministic invariants the spec guarantees -- result
counts (``min(n, stored)``), membership/subset relationships, and well-formed
metadata -- rather than a specific similarity ranking.
"""

from __future__ import annotations

import hashlib
import string
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import config
import pipeline.embed as embed_module
from models.candidate import CandidateProfile, CandidateRole
from models.job import JobDescription, JobRequirement
from pipeline.embed import EmbeddingError, VectorStoreManager

# ---------------------------------------------------------------------------
# Fake embeddings (deterministic, tiny, offline)
# ---------------------------------------------------------------------------

#: Fixed fake-embedding dimension. Small and constant so every vector stored in a
#: given ChromaDB collection has a consistent shape.
_FAKE_DIM = 8


def _fake_vector(text: str) -> list[float]:
    """Return a deterministic fake embedding for ``text`` (no model needed)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:_FAKE_DIM]]


def _fake_embed_text(text: str) -> list[float]:
    """Stand-in for :func:`pipeline.embed.embed_text`."""
    return _fake_vector(text)


def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
    """Stand-in for :func:`pipeline.embed.embed_texts`."""
    return [_fake_vector(text) for text in texts]


@pytest.fixture(scope="module", autouse=True)
def _patch_embeddings():
    """Patch the module-level embedding helpers with fast deterministic fakes.

    Module-scoped (rather than function-scoped) so it does not trip Hypothesis'
    ``function_scoped_fixture`` health check, and stays active across every
    generated example.
    """
    patches = [
        mock.patch.object(embed_module, "embed_text", _fake_embed_text),
        mock.patch.object(embed_module, "embed_texts", _fake_embed_texts),
    ]
    for patch in patches:
        patch.start()
    yield
    for patch in patches:
        patch.stop()


@pytest.fixture
def store(tmp_path) -> VectorStoreManager:
    """A VectorStoreManager backed by a fresh on-disk ChromaDB at ``tmp_path``."""
    return VectorStoreManager(persist_dir=str(tmp_path))


def _clear_collection(collection) -> None:
    """Remove every entry from a ChromaDB collection (reset between examples)."""
    existing = collection.get()
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


# ---------------------------------------------------------------------------
# Shared Hypothesis strategies
# ---------------------------------------------------------------------------

#: Printable-ASCII text avoids lone-surrogate encode errors and control chars
#: while still exercising punctuation/spaces.
_SAFE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=40,
)

_ID_ALPHABET = string.ascii_letters + string.digits + "_-"

_BUCKETS = ["must_have", "nice_to_have", "culture_signal", "seniority_marker"]
_DIMENSIONS = ["technical", "soft_skill", "domain", "experience_level"]


@st.composite
def _role_strategy(draw) -> CandidateRole:
    """Generate a minimal valid CandidateRole."""
    return CandidateRole(
        title=draw(_SAFE_TEXT),
        company=draw(_SAFE_TEXT),
        start_date="2020-01-01",
    )


@st.composite
def _candidate_strategy(draw) -> CandidateProfile:
    """Generate a valid CandidateProfile with a safe id, skills, and a few roles."""
    return CandidateProfile(
        candidate_id=draw(
            st.text(alphabet=_ID_ALPHABET, min_size=1, max_size=16)
        ),
        name=draw(_SAFE_TEXT),
        years_experience=draw(
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)
        ),
        skills_claimed=draw(st.lists(_SAFE_TEXT, min_size=1, max_size=5)),
        roles=draw(st.lists(_role_strategy(), min_size=0, max_size=3)),
    )


@st.composite
def _requirement_strategy(draw) -> JobRequirement:
    """Generate a valid JobRequirement with a valid bucket and dimension."""
    return JobRequirement(
        text=draw(_SAFE_TEXT),
        bucket=draw(st.sampled_from(_BUCKETS)),
        dimension=draw(st.sampled_from(_DIMENSIONS)),
    )


#: Shared settings for the property tests in this module. ``deadline=None`` because
#: real (local) ChromaDB I/O per example can exceed Hypothesis' default deadline;
#: the health-check suppressions allow the function-scoped ``store`` fixture and
#: the I/O-bound (slower) examples.
_PROP_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)


# ---------------------------------------------------------------------------
# Task 8.3 -> Property 8
# ---------------------------------------------------------------------------


@_PROP_SETTINGS
@given(profile=_candidate_strategy())
def test_candidate_stored_as_two_labeled_chunks(store, profile):
    # Feature: candidate-ranking-system, Property 8: A candidate is stored as exactly two labeled chunks
    """embed_candidate adds exactly two chunks: a ``profile_summary`` chunk and a
    ``skills`` chunk, each id'd ``{cid}_summary`` / ``{cid}_skills`` and each
    carrying metadata ``candidate_id`` and ``chunk_type`` (Requirement 5.3)."""
    _clear_collection(store.candidate_collection)

    store.embed_candidate(profile)

    assert store.candidate_collection.count() == 2

    cid = profile.candidate_id
    summary_id = f"{cid}_summary"
    skills_id = f"{cid}_skills"

    fetched = store.candidate_collection.get(ids=[summary_id, skills_id])
    assert set(fetched["ids"]) == {summary_id, skills_id}

    meta_by_id = dict(zip(fetched["ids"], fetched["metadatas"]))
    assert meta_by_id[summary_id]["candidate_id"] == cid
    assert meta_by_id[summary_id]["chunk_type"] == "profile_summary"
    assert meta_by_id[skills_id]["candidate_id"] == cid
    assert meta_by_id[skills_id]["chunk_type"] == "skills"


# ---------------------------------------------------------------------------
# Task 8.4 -> Property 9
# ---------------------------------------------------------------------------


@_PROP_SETTINGS
@given(examples=st.permutations(config.CALIBRATION_EXAMPLES))
def test_calibration_store_holds_ten_balanced(store, examples):
    # Feature: candidate-ranking-system, Property 9: Calibration store holds exactly ten balanced examples
    """Regardless of input ordering, embedding ``config.CALIBRATION_EXAMPLES``
    leaves the calibration collection with exactly 10 entries split 5
    ``strong_hire`` + 5 ``no_hire`` (Requirement 5.6)."""
    _clear_collection(store.calibration_collection)

    store.embed_calibration_examples(list(examples))

    assert store.calibration_collection.count() == 10

    metadatas = store.calibration_collection.get()["metadatas"]
    outcomes = [meta["outcome"] for meta in metadatas]
    assert outcomes.count("strong_hire") == 5
    assert outcomes.count("no_hire") == 5


# ---------------------------------------------------------------------------
# Task 8.5 -> Property 10
# ---------------------------------------------------------------------------


@_PROP_SETTINGS
@given(
    reqs=st.lists(_requirement_strategy(), min_size=0, max_size=12),
    n=st.integers(min_value=1, max_value=10),
    query=_SAFE_TEXT,
)
def test_jd_retrieval_bounded_and_subset(store, reqs, n, query):
    # Feature: candidate-ranking-system, Property 10: JD retrieval returns at most five items ordered by similarity
    """query_jd_context returns exactly ``min(n, M)`` requirement document
    strings, never more than ``n`` (and never more than 5 by default), and every
    returned string was one of the embedded requirements (Requirement 5.4).

    Ordering-by-similarity is not asserted here because the embeddings are faked;
    the deterministic guarantees (count bound + subset) are validated instead."""
    _clear_collection(store.jd_collection)

    jd = JobDescription(job_id="job", requirements=list(reqs))
    store.embed_job_description(jd)

    result = store.query_jd_context(query, n=n)
    m = len(reqs)

    assert len(result) == min(n, m)
    assert len(result) <= n

    stored_texts = {req.text for req in reqs}
    assert set(result).issubset(stored_texts)

    # The default cap is five context items (Requirement 5.4).
    assert len(store.query_jd_context(query)) <= 5


# ---------------------------------------------------------------------------
# Task 8.6 -> Property 11
# ---------------------------------------------------------------------------


@_PROP_SETTINGS
@given(k=st.integers(min_value=0, max_value=10), query=_SAFE_TEXT)
def test_calibration_retrieval_bounded_well_formed(store, k, query):
    # Feature: candidate-ranking-system, Property 11: Calibration retrieval returns at most three well-formed items ordered by similarity
    """query_calibration returns ``min(3, stored)`` metadata dicts, never more than
    3, and every returned dict carries the ``outcome`` and ``reason`` keys
    (Requirement 5.5).

    Ordering-by-similarity is not asserted (faked embeddings); the count bound and
    well-formedness are the deterministic guarantees validated here."""
    _clear_collection(store.calibration_collection)

    store.embed_calibration_examples(config.CALIBRATION_EXAMPLES[:k])

    result = store.query_calibration(query)

    assert len(result) == min(3, k)
    assert len(result) <= 3
    for meta in result:
        assert "outcome" in meta
        assert "reason" in meta


# ---------------------------------------------------------------------------
# Task 8.7 -> integration / example tests
# ---------------------------------------------------------------------------


def test_init_creates_three_collections(store):
    """(a) Construction creates the three expected collections, each created with
    ``embedding_function=None`` and queryable from the start (Requirement 5.1)."""
    names = {getattr(coll, "name", coll) for coll in store.client.list_collections()}
    assert {"jd_requirements", "candidate_profiles", "calibration_examples"}.issubset(names)

    # Created with embedding_function=None and initially empty / queryable.
    assert store.jd_collection.count() == 0
    assert store.candidate_collection.count() == 0
    assert store.calibration_collection.count() == 0


def test_embed_stores_two_chunks_calibration_ten_and_retrieves(store):
    """(b) A concrete end-to-end example: a candidate yields exactly two chunks, the
    calibration set yields exactly 10 entries, and retrieval is non-empty once
    content has been stored (Requirements 5.3, 5.4, 5.5, 5.6)."""
    profile = CandidateProfile(
        candidate_id="cand-1",
        name="Sample Candidate",
        years_experience=5.0,
        skills_claimed=["python", "go"],
        roles=[CandidateRole(title="Engineer", company="Acme", start_date="2019-01-01")],
    )
    store.embed_candidate(profile)
    assert store.candidate_collection.count() == 2

    store.embed_calibration_examples(config.CALIBRATION_EXAMPLES)
    assert store.calibration_collection.count() == 10

    jd = JobDescription(
        job_id="job-1",
        requirements=[
            JobRequirement(text="5+ years Python", bucket="must_have", dimension="technical"),
            JobRequirement(text="Strong team player", bucket="culture_signal", dimension="soft_skill"),
        ],
    )
    store.embed_job_description(jd)

    assert store.query_jd_context("python experience") != []
    assert store.query_calibration("strong backend engineer") != []


def test_embedding_failure_leaves_collections_unchanged(store, monkeypatch):
    """(c) When embedding computation fails mid-store, embed_candidate and
    embed_job_description raise EmbeddingError and leave the target collection
    unchanged (Requirement 5.7)."""

    def _boom_text(_text):
        raise RuntimeError("embedding backend unavailable")

    def _boom_texts(_texts):
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(embed_module, "embed_text", _boom_text)
    monkeypatch.setattr(embed_module, "embed_texts", _boom_texts)

    # embed_candidate uses embed_texts -> failure must abort with no mutation.
    profile = CandidateProfile(
        candidate_id="cand-x",
        name="Doomed Candidate",
        skills_claimed=["python"],
    )
    candidate_before = store.candidate_collection.count()
    with pytest.raises(EmbeddingError):
        store.embed_candidate(profile)
    assert store.candidate_collection.count() == candidate_before

    # embed_job_description uses embed_text -> failure must abort with no mutation.
    jd = JobDescription(
        job_id="job-x",
        requirements=[
            JobRequirement(text="a requirement", bucket="must_have", dimension="technical"),
        ],
    )
    jd_before = store.jd_collection.count()
    with pytest.raises(EmbeddingError):
        store.embed_job_description(jd)
    assert store.jd_collection.count() == jd_before
