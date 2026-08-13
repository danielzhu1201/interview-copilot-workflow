"""Structured contracts for Interview Prep Workflow V1 and Agent V1."""

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


# =============================================================================
# LESSON 5 AGENT V1 CONTRACTS
# Everything specific to the agent is intentionally grouped in this section.
# =============================================================================

AgentAction = Literal["ASK_USER", "GENERATE_PREP_PACKAGE", "FINISH"]


class CandidateClarification(WorkflowModel):
    """One factual answer supplied after a resumable agent interrupt."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    question: str = Field(min_length=8)
    answer: str = Field(min_length=8)


class HighPriorityGap(WorkflowModel):
    """Decision context for one code-eligible, unresolved evidence gap."""

    requirement_id: str = Field(pattern=r"^REQ-\d{2}$")
    requirement: str = Field(min_length=8)
    importance: int = Field(ge=4, le=5)
    explanation: str = Field(min_length=8)


class AgentObservation(WorkflowModel):
    """Deterministic decision facts derived from the current business state."""

    package_generated: bool
    package_valid: bool
    high_priority_gap_ids: list[str]
    high_priority_gaps: list[HighPriorityGap]
    asked_requirement_ids: list[str]
    allowed_actions: list[AgentAction]
    latest_clarification: str | None = None
    last_action: AgentAction | None = None
    steps_remaining: int = Field(ge=0, le=4)


class AgentDecision(WorkflowModel):
    """Exactly one model-proposed action for the agent runtime to authorize."""

    next_action: AgentAction
    target_requirement_id: str | None = Field(
        default=None,
        pattern=r"^REQ-\d{2}$",
    )
    question: str | None = None
    reason_summary: str = Field(min_length=8)


class AgentInput(TypedDict):
    """Raw source documents supplied when the Lesson 5 agent starts."""

    job_description: str
    resume_text: str


class AgentState(TypedDict, total=False):
    """Workflow V1 business state plus bounded Lesson 5 agent control state."""

    # Source inputs and candidate-supplied evidence
    job_description: str
    resume_text: str
    candidate_clarifications: Annotated[list[CandidateClarification], operator.add]
    asked_requirement_ids: Annotated[list[str], operator.add]
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
    # Agent control state
    goal: str
    observation: AgentObservation
    next_decision: AgentDecision
    package_generated: bool
    last_action: AgentAction
    action_count: int
    agent_error: str | None
    stop_reason: str | None
    decision_retry_count: int
    authorized_route: Literal[
        "generate",
        "ask_user",
        "finish",
        "retry",
        "invalid",
    ]


# =============================================================================
# WORKFLOW V1 STATE (with two narrow Agent V1 compatibility hooks)
# =============================================================================


class WorkflowInput(TypedDict):
    """Untouched source documents supplied when the graph starts."""

    job_description: str
    resume_text: str
    # Agent V1 hook: pass resumed evidence into an otherwise unchanged workflow.
    candidate_clarifications: NotRequired[list[CandidateClarification]]


class WorkflowState(TypedDict):
    """Business state retaining raw inputs alongside derived objects."""

    # Source inputs
    job_description: str
    resume_text: str
    # Derived source evidence
    candidate_evidence: list[CandidateEvidence]
    # Agent V1 hook: retain resumed evidence while Workflow V1 runs.
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
