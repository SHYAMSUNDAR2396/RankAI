"""Utils package: cross-cutting wrappers for external dependencies.

Contains ``ollama_client.py`` with ``OllamaClient``, the single seam through
which every ``ollama.chat`` call flows. Centralizing the LLM call here provides
one place for retry/backoff, DEBUG-level request/response logging, and markdown
fence-stripping JSON parsing, and one swap point for an alternative LLM backend.
"""
