"""Tests package: offline, deterministic test suite.

Example/integration/smoke tests and Hypothesis property-based tests live in
``test_ingest.py``, ``test_embed.py``, and ``test_score.py``. All ``ollama.chat``
calls are mocked via ``unittest.mock.patch`` so the suite runs fully offline and
deterministically, with no real Ollama server or network access.
"""
