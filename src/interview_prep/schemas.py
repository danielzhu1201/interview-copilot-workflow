"""Structured contracts for the extraction slice."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

RequirementCategory = Literal[
    "technical",
    "product",
    "analytics",
    "communication",
    "leadership",
    "domain",
    "experience",
    "education",
]


class JobRequirement(BaseModel):
    """One requirement grounded in exact text from the job description."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirement_id: str = Field(
        pattern=r"^REQ-\d{2}$",
        description="Stable sequential ID such as REQ-01.",
    )
    category: RequirementCategory
    requirement: str = Field(
        min_length=8,
        description="Concise normalized statement of the requirement.",
    )
    importance: int = Field(
        ge=1,
        le=5,
        description="Priority from 1 (low) to 5 (critical).",
    )
    requirement_type: Literal["must_have", "preferred"]
    source_quote: str = Field(
        min_length=8,
        description="Exact supporting quote copied from the JD.",
    )


class RequirementExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    requirements: list[JobRequirement]


class WorkflowInput(TypedDict):
    """Values supplied when the graph starts."""

    job_description: str
    resume_text: str


class WorkflowState(TypedDict):
    """Business state shared by the extraction and validation nodes."""

    job_description: str
    resume_text: str
    role_title: str
    company: str
    requirements: list[JobRequirement]
    validation_errors: list[str]
    requirements_valid: bool
    status: str
