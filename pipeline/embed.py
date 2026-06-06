"""Embedding helpers and vector storage for the Candidate Ranking System.

This module owns the EMBED & STORE phase. It exposes two concerns:

* A lazy-loaded, process-wide ``SentenceTransformer`` singleton plus thin
  ``embed_text`` / ``embed_texts`` helpers. The (heavy) model is loaded at most
  once per process and *never* at import time, so merely importing
  ``pipeline.embed`` does not pull in ``sentence_transformers`` or trigger a
  model download (Requirement 1.2).
* ``VectorStoreManager`` which manages the three on-disk ChromaDB collections
  and manually injects the vectors produced by the helpers below. ``chromadb``
  is imported lazily inside its constructor so importing this module for the
  helpers alone never requires ``chromadb``.

All embeddings are computed with ``normalize_embeddings=True`` so that cosine
similarity in ChromaDB behaves consistently across stored and queried vectors
(Requirement 5.2). The embedding model is ``config.EMBEDDING_MODEL``
(``BAAI/bge-large-en-v1.5``) (Requirement 1.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    # Imported only for type checking so that importing this module at runtime
    # does not require ``sentence_transformers`` / ``chromadb`` to be installed
    # or load the (heavy) model weights.
    from sentence_transformers import SentenceTransformer

    from models.candidate import CandidateProfile
    from models.job import JobDescription

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding computation fails during a vector-store operation.

    The :class:`VectorStoreManager` computes every embedding for a store
    operation *before* mutating the target collection, so raising this error
    guarantees the collection is left unchanged (Requirement 5.7).
    """

#: Process-wide ``SentenceTransformer`` singleton. ``None`` until the first call
#: to :func:`get_embedding_model`, which loads it lazily. Loading the model is
#: expensive (and may download weights), so it must never happen at import time.
_model: "SentenceTransformer | None" = None


def get_embedding_model() -> "SentenceTransformer":
    """Return the process-wide ``SentenceTransformer`` singleton.

    The model is loaded lazily on first use from ``config.EMBEDDING_MODEL`` and
    cached in the module-level ``_model`` global for every subsequent call. The
    ``sentence_transformers`` import is performed inside this function so that
    importing ``pipeline.embed`` neither imports the heavy library nor triggers
    a model download (Requirement 1.2).

    Args:
        None.

    Returns:
        SentenceTransformer: The cached embedding model instance, loaded on the
        first invocation and reused thereafter.
    """
    global _model
    if _model is None:
        # Lazy import: keep the heavy dependency out of module import.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string into a normalized dense vector.

    Args:
        text: The string to embed.

    Returns:
        list[float]: The dense embedding vector for ``text``, computed with
        ``normalize_embeddings=True`` (Requirement 5.2).
    """
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into normalized dense vectors.

    Args:
        texts: The list of strings to embed.

    Returns:
        list[list[float]]: One dense embedding vector per input string, in the
        same order as ``texts``, each computed with ``normalize_embeddings=True``
        (Requirement 5.2).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [embedding.tolist() for embedding in embeddings]


class VectorStoreManager:
    """Own the three on-disk ChromaDB collections and all vector injection.

    The manager backs the EMBED & STORE phase. On construction it opens a
    ``chromadb.PersistentClient`` at ``persist_dir`` and gets-or-creates three
    collections -- ``jd_requirements``, ``candidate_profiles``, and
    ``calibration_examples`` -- each configured with ``embedding_function=None``
    so ChromaDB never embeds on its own; this class always supplies vectors it
    computed with the module-level :func:`embed_text` / :func:`embed_texts`
    helpers and injects them manually via ``collection.add(embeddings=[...])``
    (Requirements 5.1, 5.2, 1.3).

    ``chromadb`` is imported lazily inside :meth:`__init__` so that merely
    importing ``pipeline.embed`` (for the embedding helpers) does not require
    ``chromadb`` to be installed.

    Every store method computes *all* of its embeddings before calling
    ``collection.add``. If any embedding computation raises, the method raises
    :class:`EmbeddingError` and the target collection is left unchanged
    (Requirement 5.7).

    Attributes:
        client: The underlying ``chromadb.PersistentClient``.
        jd_collection: The ``jd_requirements`` collection.
        candidate_collection: The ``candidate_profiles`` collection.
        calibration_collection: The ``calibration_examples`` collection.
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        """Open the persistent client and get-or-create the three collections.

        Args:
            persist_dir: On-disk directory for the ChromaDB ``PersistentClient``.
                When ``None`` (the default), falls back to
                ``config.CHROMA_PERSIST_DIR`` (Requirement 1.3).

        Returns:
            None.
        """
        # Lazy import: keep ``chromadb`` out of module import so that importing
        # ``pipeline.embed`` for the embedding helpers stays light.
        import chromadb

        resolved_dir = persist_dir if persist_dir is not None else config.CHROMA_PERSIST_DIR
        logger.info("Opening ChromaDB persistent client at: %s", resolved_dir)
        self.client = chromadb.PersistentClient(path=resolved_dir)

        # Each collection disables ChromaDB's own embedding function; this class
        # always injects vectors it computed itself (Requirement 5.1).
        self.jd_collection = self.client.get_or_create_collection(
            name="jd_requirements", embedding_function=None
        )
        self.candidate_collection = self.client.get_or_create_collection(
            name="candidate_profiles", embedding_function=None
        )
        self.calibration_collection = self.client.get_or_create_collection(
            name="calibration_examples", embedding_function=None
        )
        logger.info(
            "Initialized ChromaDB collections: jd_requirements, "
            "candidate_profiles, calibration_examples"
        )

    def embed_job_description(self, jd: JobDescription) -> None:
        """Embed and store each job-description requirement.

        Each requirement in ``jd.requirements`` is stored in the
        ``jd_requirements`` collection with id ``"{job_id}_{i}"``, the
        requirement text as its document, an embedding computed from that text,
        and metadata carrying the requirement's ``bucket`` and ``dimension`` and
        the owning ``job_id`` (Requirement 5.2). All embeddings are computed
        before any mutation, so an embedding failure leaves the collection
        unchanged (Requirement 5.7).

        Args:
            jd: The parsed job description whose requirements are stored.

        Returns:
            None.

        Raises:
            EmbeddingError: If embedding computation fails; the collection is
                left unchanged.
        """
        if not jd.requirements:
            logger.info("Job description %s has no requirements to embed", jd.job_id)
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, req in enumerate(jd.requirements):
            ids.append(f"{jd.job_id}_{i}")
            documents.append(req.text)
            metadatas.append(
                {
                    "bucket": req.bucket,
                    "dimension": req.dimension,
                    "job_id": jd.job_id,
                }
            )

        # Compute every embedding BEFORE touching the collection so a failure
        # aborts the store without partial mutation (Requirement 5.7).
        try:
            embeddings = [embed_text(text) for text in documents]
        except Exception as exc:  # noqa: BLE001 - re-raised as EmbeddingError
            raise EmbeddingError(
                f"Failed to embed job description requirements for job {jd.job_id}"
            ) from exc

        self.jd_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Stored %d job-description requirements for job %s",
            len(ids),
            jd.job_id,
        )

    def embed_calibration_examples(self, examples: list[dict]) -> None:
        """Embed and store the provided calibration examples.

        Each example is stored in the ``calibration_examples`` collection with
        id ``"calib_{i}"``, the example's ``profile_summary`` as both the
        embedded text and the stored document, and metadata carrying the
        example's ``outcome`` and ``reason`` (Requirement 5.6). Callers pass
        ``config.CALIBRATION_EXAMPLES`` (exactly 5 ``strong_hire`` + 5
        ``no_hire``). All embeddings are computed before any mutation, so an
        embedding failure leaves the collection unchanged (Requirement 5.7).

        Args:
            examples: The calibration examples to store; each a dict with
                ``profile_summary``, ``outcome``, and ``reason`` keys.

        Returns:
            None.

        Raises:
            EmbeddingError: If embedding computation fails; the collection is
                left unchanged.
        """
        if not examples:
            logger.info("No calibration examples to embed")
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, example in enumerate(examples):
            ids.append(f"calib_{i}")
            documents.append(example["profile_summary"])
            metadatas.append(
                {
                    "outcome": example["outcome"],
                    "reason": example["reason"],
                }
            )

        try:
            embeddings = [embed_text(text) for text in documents]
        except Exception as exc:  # noqa: BLE001 - re-raised as EmbeddingError
            raise EmbeddingError("Failed to embed calibration examples") from exc

        self.calibration_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Stored %d calibration examples", len(ids))

    def embed_candidate(self, profile: CandidateProfile) -> None:
        """Embed and store a candidate as exactly two labeled chunks.

        The candidate is stored in the ``candidate_profiles`` collection as two
        chunks (Requirement 5.3):

        * ``profile_summary`` -- ``"{name} | {years} years | Skills: {skills} |
          Roles: {roles}"`` where the roles segment joins ``"{title} at
          {company}"`` for each role, with id ``"{candidate_id}_summary"``.
        * ``skills`` -- the comma-joined claimed skills, with id
          ``"{candidate_id}_skills"``.

        Each chunk carries metadata ``candidate_id`` and ``chunk_type``. Both
        embeddings are computed before any mutation, so an embedding failure
        leaves the collection unchanged (Requirement 5.7).

        Args:
            profile: The candidate profile to store.

        Returns:
            None.

        Raises:
            EmbeddingError: If embedding computation fails; the collection is
                left unchanged.
        """
        role_titles_string = ", ".join(
            f"{role.title} at {role.company}" for role in profile.roles
        )
        skills_string = ", ".join(profile.skills_claimed)
        summary_text = (
            f"{profile.name} | {profile.years_experience} years | "
            f"Skills: {skills_string} | Roles: {role_titles_string}"
        )

        ids = [
            f"{profile.candidate_id}_summary",
            f"{profile.candidate_id}_skills",
        ]
        documents = [summary_text, skills_string]
        metadatas = [
            {"candidate_id": profile.candidate_id, "chunk_type": "profile_summary"},
            {"candidate_id": profile.candidate_id, "chunk_type": "skills"},
        ]

        # Compute both embeddings BEFORE mutating the collection so a failure
        # aborts the store without partial mutation (Requirement 5.7).
        try:
            embeddings = embed_texts(documents)
        except Exception as exc:  # noqa: BLE001 - re-raised as EmbeddingError
            raise EmbeddingError(
                f"Failed to embed candidate {profile.candidate_id}"
            ) from exc

        self.candidate_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Stored candidate %s as profile_summary + skills chunks",
            profile.candidate_id,
        )

    def query_jd_context(self, query: str, n: int = 5) -> list[str]:
        """Retrieve the most similar job-description context document strings.

        The ``query`` text is embedded with the module-level :func:`embed_text`
        helper and passed to ChromaDB via ``query_embeddings`` (the
        ``jd_requirements`` collection was created with
        ``embedding_function=None``, so ChromaDB never embeds the query itself).
        The matching requirement document strings are returned nearest-first,
        i.e. in descending order of embedding similarity (Requirement 5.4).

        ChromaDB returns results keyed by each submitted query, so
        ``results["documents"]`` has shape ``[[...]]`` for the single query; the
        inner list is unwrapped. When no documents are returned (a missing,
        empty, or ``None`` result) an empty list is returned.

        Args:
            query: The free-text query to match against stored JD requirements.
            n: The maximum number of context strings to return (default 5).

        Returns:
            list[str]: Up to ``n`` requirement document strings, ordered
            nearest-first by embedding similarity. Empty when there are no
            matches.
        """
        logger.debug("Querying JD context (n=%d) for query: %s", n, query)
        embedding = embed_text(query)
        results = self.jd_collection.query(
            query_embeddings=[embedding],
            n_results=n,
        )

        documents = results.get("documents")
        if not documents or not documents[0]:
            logger.debug("JD context query returned 0 results")
            return []

        matched = documents[0]
        logger.debug("JD context query returned %d results", len(matched))
        return matched

    def query_calibration(self, query: str, n: int = 3) -> list[dict]:
        """Retrieve the most similar calibration-example metadata dicts.

        The ``query`` text is embedded with the module-level :func:`embed_text`
        helper and passed to ChromaDB via ``query_embeddings`` (the
        ``calibration_examples`` collection was created with
        ``embedding_function=None``, so ChromaDB never embeds the query itself).
        The matching calibration metadata dicts -- each carrying ``outcome`` and
        ``reason`` -- are returned nearest-first, i.e. in descending order of
        embedding similarity (Requirement 5.5).

        ChromaDB returns results keyed by each submitted query, so
        ``results["metadatas"]`` has shape ``[[...]]`` for the single query; the
        inner list is unwrapped. When no metadatas are returned (a missing,
        empty, or ``None`` result) an empty list is returned.

        Args:
            query: The free-text query to match against stored calibration
                examples.
            n: The maximum number of metadata dicts to return (default 3).

        Returns:
            list[dict]: Up to ``n`` calibration metadata dicts (each with
            ``outcome`` and ``reason``), ordered nearest-first by embedding
            similarity. Empty when there are no matches.
        """
        logger.debug("Querying calibration (n=%d) for query: %s", n, query)
        embedding = embed_text(query)
        results = self.calibration_collection.query(
            query_embeddings=[embedding],
            n_results=n,
        )

        metadatas = results.get("metadatas")
        if not metadatas or not metadatas[0]:
            logger.debug("Calibration query returned 0 results")
            return []

        matched = metadatas[0]
        logger.debug("Calibration query returned %d results", len(matched))
        return matched
