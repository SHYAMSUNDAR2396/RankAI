"""Pydantic v2 data models for the parsed job description.

Defines the two structured models the JD_Parser produces and the rest of the
pipeline consumes:

- :class:`JobRequirement` -- one classified requirement (text + bucket + dimension).
- :class:`JobDescription` -- the whole job description with its classified
  requirements and helper methods for bucket filtering and RAG rendering.

Both models target Pydantic v2. Every optional field carries an explicit default
so a partial source (for example, a ``.json`` job description that only supplies a
``job_id``) still produces a valid model (Requirements 3.8, 11.2). The ``bucket``
and ``dimension`` fields are constrained with :data:`typing.Literal`, so Pydantic
rejects any value outside the allowed requirement buckets and dimensions
(Requirements 3.5, 3.6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Allowed requirement buckets (Requirement 3.5).
RequirementBucket = Literal[
    "must_have",
    "nice_to_have",
    "culture_signal",
    "seniority_marker",
]

#: Allowed requirement dimensions (Requirement 3.6).
RequirementDimension = Literal[
    "technical",
    "soft_skill",
    "domain",
    "experience_level",
]


class JobRequirement(BaseModel):
    """One classified requirement extracted from a job description.

    Attributes:
        text: The human-readable requirement statement.
        bucket: The requirement bucket; one of ``must_have``, ``nice_to_have``,
            ``culture_signal``, or ``seniority_marker``. Any other value fails
            validation (Requirement 3.5).
        dimension: The requirement dimension; one of ``technical``,
            ``soft_skill``, ``domain``, or ``experience_level``. Any other value
            fails validation (Requirement 3.6).
    """

    text: str
    bucket: RequirementBucket
    dimension: RequirementDimension


class JobDescription(BaseModel):
    """A parsed job description with its classified requirements.

    Optional fields carry explicit defaults so a job description that supplies
    only a ``job_id`` still validates against the model (Requirements 3.8, 11.2).

    Attributes:
        job_id: Stable identifier for the job description, assigned by the
            JD_Parser via ``uuid5`` on the file path (Requirement 3.7).
        title: The role title; defaults to ``"Untitled Role"``.
        company: The hiring company; defaults to the empty string.
        requirements: The classified requirements; defaults to an empty list.
        raw_text: The original job description text; defaults to the empty
            string.
    """

    job_id: str
    title: str = "Untitled Role"
    company: str = ""
    requirements: list[JobRequirement] = Field(default_factory=list)
    raw_text: str = ""

    def by_bucket(self, bucket: str) -> list[JobRequirement]:
        """Return the requirements that belong to the given bucket.

        Args:
            bucket: The requirement bucket to filter on (for example,
                ``"must_have"``). Buckets with no matching requirements yield an
                empty list.

        Returns:
            The requirements whose ``bucket`` equals ``bucket``, in their
            original order.
        """
        return [req for req in self.requirements if req.bucket == bucket]

    def context_strings(self) -> list[str]:
        """Render each requirement to a string suitable for embedding/RAG.

        Each requirement is rendered as ``"[<bucket>/<dimension>] <text>"`` so
        the bucket and dimension travel with the text into the vector store and
        persona prompts.

        Args:
            None.

        Returns:
            One rendered string per requirement, in their original order. An
            empty list when there are no requirements.
        """
        return [
            f"[{req.bucket}/{req.dimension}] {req.text}" for req in self.requirements
        ]
