"""Pipeline package: domain logic for the five candidate-ranking phases.

This package holds the per-phase components of the Candidate Ranking System:

- ``ingest.py``  -- ``ResumeParser`` and ``JdParser`` (INGEST).
- ``enrich.py``  -- ``TrajectoryEnricher`` (ENRICH).
- ``embed.py``   -- embedding helpers and ``VectorStoreManager`` (EMBED & STORE).
- ``score.py``   -- ``CandidateScoringPipeline`` (SCORE).

These modules depend downward on ``models`` and ``config`` and on the
cross-cutting utilities in ``utils`` (LLM wrapper); they are kept independent of
orchestration so their pure logic can be property-tested offline.
"""
