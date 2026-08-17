"""Structured contracts for the interview workflow and governed agent."""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

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


class InterviewRound(WorkflowModel):
    """Structured context parsed from optional user-supplied freeform text."""

    round_type: str | None = None
    format: str | None = None
    interviewer_roles: list[str] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)
    notes: str | None = None


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

    interview_round: InterviewRound | None
    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]
    focus_areas: list[FocusArea]
    interview_strategy: InterviewStrategy
    mock_questions: list[MockQuestion]


# =============================================================================
# LESSON 6 ROUND-GUIDED AGENT V2 CONTRACTS
# Context, progress, admitted evidence, and audit state stay separate.
# =============================================================================


class CandidateClarification(WorkflowModel):
    """One answer admitted by code as candidate evidence."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    question: str = Field(min_length=8)
    answer: str = Field(min_length=8)
    accepted_claim: str = Field(min_length=8)


class ClarificationAssessment(WorkflowModel):
    """Gemini's structured advice about one resumed answer."""

    target_requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    is_valid: bool
    relevance_reason: str = Field(min_length=8)
    specificity_reason: str = Field(min_length=8)
    accepted_claim: str | None = None


class ClarificationRecord(WorkflowModel):
    """Auditable accepted or rejected result for every processed GAP."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    question: str = Field(min_length=8)
    answer: str
    assessment: ClarificationAssessment
    accepted: bool
    decision_reason: str = Field(min_length=8)
    accepted_claim: str | None = None


class AgentInput(TypedDict):
    """Source documents and optional next-round context for Agent V2."""

    job_description: str
    resume_text: str
    interview_round: NotRequired[str]


class AgentState(TypedDict, total=False):
    """Workflow state plus deterministic queue and evidence-admission state."""

    # Source inputs and round context
    job_description: str
    resume_text: str
    interview_round: str
    interview_round_context: InterviewRound | None
    # Progress, evidence admission, and audit
    processed_requirement_ids: Annotated[list[str], operator.add]
    accepted_clarifications: Annotated[list[CandidateClarification], operator.add]
    clarification_records: Annotated[list[ClarificationRecord], operator.add]
    current_gap: JobRequirement | None
    current_question: str | None
    pending_answer: str | None
    candidate_evidence: list[CandidateEvidence]
    # Workflow V1 derived state
    requirements: list[JobRequirement]
    evidence_matches: list[EvidenceMatch]
    focus_areas: list[FocusArea]
    interview_strategy: InterviewStrategy
    mock_questions: list[MockQuestion]
    prep_package: PrepPackage | None
    validation_errors: list[str]
    package_valid: bool
    # Agent control state and terminal result
    initial_package_generated: bool
    final_package_generated: bool
    stop_reason: str | None
    agent_error: str | None


# =============================================================================
# WORKFLOW V1 STATE (with two narrow Agent V1 compatibility hooks)
# =============================================================================


class WorkflowInput(TypedDict):
    """Untouched source documents supplied when the graph starts."""

    job_description: str
    resume_text: str
    interview_round: NotRequired[str]
    interview_round_context: NotRequired[InterviewRound | None]
    persist_package: NotRequired[bool]
    # Agent hook: pass only admitted evidence into the fixed workflow.
    candidate_clarifications: NotRequired[list[CandidateClarification]]


class WorkflowState(TypedDict):
    """Business state retaining raw inputs alongside derived objects."""

    # Source inputs
    job_description: str
    resume_text: str
    interview_round: NotRequired[str]
    interview_round_context: NotRequired[InterviewRound | None]
    persist_package: NotRequired[bool]
    # Derived source evidence
    candidate_evidence: list[CandidateEvidence]
    # Agent hook: retain admitted evidence while Workflow V1 runs.
    candidate_clarifications: NotRequired[list[CandidateClarification]]
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
