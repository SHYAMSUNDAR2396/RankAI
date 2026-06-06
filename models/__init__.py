"""Models package: Pydantic v2 data schemas.

Defines the structured data models the rest of the system depends on:

- ``candidate.py`` -- ``CandidateRole``, ``TrajectoryVector``, ``CandidateProfile``.
- ``job.py``       -- ``JobRequirement``, ``JobDescription``.

Every optional field carries an explicit default so a partial source still
produces a valid model.
"""
