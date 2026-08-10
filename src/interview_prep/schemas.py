"""Structured business-state contracts for Interview Prep Workflow V1."""

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
Coverage = Literal["FULL", "PARTIAL", "GAP"]


class WorkflowModel(BaseModel):
    """Strict base model shared by values stored in workflow state."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CandidateEvidence(WorkflowModel):
    """One candidate claim supplied by the resume or another source."""

    evidence_id: str = Field(pattern=r"^EXP-\d{2}$")
    claim: str = Field(min_length=8)
    source: str = Field(min_length=2)


class JobRequirement(WorkflowModel):
    """One requirement grounded in exact text from the job description."""

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


class RequirementExtraction(WorkflowModel):
    requirements: list[JobRequirement]


class EvidenceMatch(WorkflowModel):
    """Candidate evidence supporting one requirement, or an explicit gap."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    evidence_ids: list[str]
    coverage: Coverage
    explanation: str = Field(min_length=8)
    confidence: float = Field(ge=0, le=1)


class EvidenceMatchList(WorkflowModel):
    evidence_matches: list[EvidenceMatch]


class FocusArea(WorkflowModel):
    """A deterministic recommendation for allocating preparation time."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    coverage: Coverage
    priority: int = Field(ge=1, le=15)
    preparation_action: str = Field(min_length=8)
    reason: str = Field(min_length=8)


class StrategyItem(WorkflowModel):
    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    evidence_ids: list[str]
    preparation_theme: str = Field(min_length=8)
    rationale: str = Field(min_length=8)


class StoryPlan(WorkflowModel):
    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    evidence_ids: list[str]
    story_to_prepare: str = Field(min_length=8)


class RiskItem(WorkflowModel):
    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    risk: str = Field(min_length=8)
    mitigation: str = Field(min_length=8)


class InterviewStrategy(WorkflowModel):
    top_priorities: list[StrategyItem]
    positioning_statement: str = Field(min_length=12)
    stories_to_prepare: list[StoryPlan]
    risks_to_address: list[RiskItem]


class MockQuestion(WorkflowModel):
    question: str = Field(min_length=8)
    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    capability_tested: str = Field(min_length=4)
    evidence_ids: list[str]
    follow_up_probe: str = Field(min_length=8)
    answer_outline: list[str] = Field(min_length=2)


class MockQuestionList(WorkflowModel):
    mock_questions: list[MockQuestion] = Field(min_length=8)


class PrepPackage(WorkflowModel):
    """Validated, candidate-facing preparation product."""

    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]
    focus_areas: list[FocusArea]
    interview_strategy: InterviewStrategy
    mock_questions: list[MockQuestion]


class WorkflowInput(TypedDict):
    """Untouched source documents supplied when the graph starts."""

    job_description: str
    resume_text: str


class WorkflowState(TypedDict):
    """Business state retaining raw inputs alongside derived objects."""

    # Source inputs
    job_description: str
    resume_text: str
    # Derived source evidence
    candidate_evidence: list[CandidateEvidence]
    # Grounded intelligence
    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]
    focus_areas: list[FocusArea]
    # Preparation outputs
    interview_strategy: InterviewStrategy
    mock_questions: list[MockQuestion]
    prep_package: PrepPackage | None
    # Reliability
    validation_errors: list[str]
    package_valid: bool
