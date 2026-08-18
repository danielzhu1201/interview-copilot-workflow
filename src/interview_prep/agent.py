"""Round-guided, evidence-gated human-in-the-loop agent for Lesson 6."""

from __future__ import annotations

from typing import Any, Literal

from google.genai import types
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .graph import graph as workflow_graph
from .llm import get_gemini_client, get_model_name
from .prompts import (
    build_clarification_assessment_prompt,
    build_interview_round_parsing_prompt,
)
from .schemas import (
    AgentInput,
    AgentState,
    CandidateClarification,
    ClarificationAssessment,
    ClarificationRecord,
    InterviewRound,
    JobRequirement,
)

# =============================================================================
# LESSON 6 AGENT V2 CONFIGURATION
# =============================================================================

MIN_CLARIFICATION_LENGTH = 24
WORKFLOW_RESULT_FIELDS = (
    "interview_round_context",
    "candidate_evidence",
    "requirements",
    "evidence_matches",
    "focus_areas",
    "interview_strategy",
    "mock_questions",
    "prep_package",
    "validation_errors",
    "package_valid",
)


def parse_round_context(state: AgentState) -> dict[str, Any]:
    """Parse optional freeform round text once before either workflow run."""

    raw_interview_round = state.get("interview_round", "")
    if raw_interview_round is None:
        raw_interview_round = ""
    if not isinstance(raw_interview_round, str):
        raise ValueError("Interview round must be freeform text when supplied.")
    interview_round_text = raw_interview_round.strip()
    if not interview_round_text:
        return {"interview_round_context": None}

    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_interview_round_parsing_prompt(interview_round_text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=InterviewRound.model_json_schema(),
        ),
    )
    return {"interview_round_context": InterviewRound.model_validate(response.parsed)}


def _workflow_input(state: AgentState, *, persist_package: bool) -> dict[str, Any]:
    """Build the canonical full-context workflow input."""

    return {
        "job_description": state["job_description"],
        "resume_text": state["resume_text"],
        "interview_round": state.get("interview_round", ""),
        "interview_round_context": state.get("interview_round_context"),
        "candidate_clarifications": state.get("accepted_clarifications", []),
        "persist_package": persist_package,
    }


# =============================================================================
# INITIAL PACKAGE AND DETERMINISTIC GAP QUEUE
# =============================================================================


def generate_initial_package(state: AgentState) -> dict[str, Any]:
    """Run the full workflow once without writing the preliminary package."""

    result = workflow_graph.invoke(_workflow_input(state, persist_package=False))
    return {
        **{field: result[field] for field in WORKFLOW_RESULT_FIELDS},
        "initial_package_generated": True,
        "final_package_generated": False,
        "agent_error": None,
        "stop_reason": None,
    }


def select_next_gap(state: AgentState) -> JobRequirement | None:
    """Return the highest-impact GAP that has not already been processed."""

    requirements_by_id = {
        requirement.requirement_id: requirement
        for requirement in state.get("requirements", [])
    }
    processed_requirement_ids = set(state.get("processed_requirement_ids", []))
    gaps = [
        requirements_by_id[match.requirement_id]
        for match in state.get("evidence_matches", [])
        if match.coverage == "GAP"
        and match.requirement_id in requirements_by_id
        and match.requirement_id not in processed_requirement_ids
    ]
    gaps.sort(
        key=lambda requirement: (-requirement.importance, requirement.requirement_id)
    )
    return gaps[0] if gaps else None


def observe_gaps(state: AgentState) -> dict[str, Any]:
    """Expose only the next deterministic queue item to the interaction loop."""

    return {"current_gap": select_next_gap(state)}


def route_after_observation(
    state: AgentState,
) -> Literal["ask_user", "generate_final", "invalid"]:
    """Route from code-owned package validity and queue state."""

    if not state.get("package_valid", False):
        return "invalid"
    return "ask_user" if state.get("current_gap") is not None else "generate_final"


def _question_for(gap: JobRequirement) -> str:
    return (
        "Please share one specific example from your experience that demonstrates "
        f"this requirement: {gap.requirement} Include what you did, the methods or "
        "tools you used, and the result."
    )


def interrupt_for_gap(state: AgentState) -> dict[str, Any]:
    """Pause for one answer to the current GAP and resume in the same thread."""

    gap = state.get("current_gap")
    if gap is None:
        raise ValueError("ASK_USER requires a current GAP.")
    question = _question_for(gap)
    answer = interrupt(
        {
            "type": "candidate_evidence_request",
            "requirement_id": gap.requirement_id,
            "question": question,
        }
    )
    if not isinstance(answer, str):
        raise ValueError("Candidate clarification answer must be a string.")
    return {"current_question": question, "pending_answer": answer.strip()}


# =============================================================================
# SHORT-CONTEXT ASSESSMENT AND CODE-OWNED EVIDENCE GATE
# =============================================================================


def _rejection_reason(
    answer: str,
    assessment: ClarificationAssessment,
    target_requirement_id: str,
) -> str | None:
    if len(answer.strip()) < MIN_CLARIFICATION_LENGTH:
        return f"Answer must contain at least {MIN_CLARIFICATION_LENGTH} characters."
    if assessment.target_requirement_id != target_requirement_id:
        return "Assessment targeted a different requirement ID."
    if not assessment.is_valid:
        return (
            "Assessment rejected the answer: "
            f"{assessment.relevance_reason} {assessment.specificity_reason}"
        )
    if not assessment.accepted_claim or not assessment.accepted_claim.strip():
        return "A valid assessment must provide a non-empty accepted claim."
    return None


def should_accept_clarification(
    answer: str,
    assessment: ClarificationAssessment,
    target_requirement_id: str,
) -> bool:
    """Return whether every code-owned evidence-admission gate passes."""

    return (
        len(answer.strip()) >= MIN_CLARIFICATION_LENGTH
        and assessment.target_requirement_id == target_requirement_id
        and assessment.is_valid
        and bool(assessment.accepted_claim and assessment.accepted_claim.strip())
    )


def assess_and_record_clarification(state: AgentState) -> dict[str, Any]:
    """Request short-context advice, apply gates, and append one audit record."""

    gap = state.get("current_gap")
    question = state.get("current_question")
    answer = state.get("pending_answer")
    if gap is None or question is None or answer is None:
        raise ValueError(
            "Clarification assessment requires a GAP, question, and answer."
        )

    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_clarification_assessment_prompt(
            requirement=gap,
            question=question,
            answer=answer,
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ClarificationAssessment.model_json_schema(),
        ),
    )
    assessment = ClarificationAssessment.model_validate(response.parsed)
    accepted = should_accept_clarification(
        answer,
        assessment,
        gap.requirement_id,
    )
    rejection_reason = _rejection_reason(answer, assessment, gap.requirement_id)
    decision_reason = rejection_reason or (
        "The model assessment and every code-owned admission gate passed."
    )
    accepted_claim = assessment.accepted_claim if accepted else None
    record = ClarificationRecord(
        requirement_id=gap.requirement_id,
        question=question,
        answer=answer,
        assessment=assessment,
        accepted=accepted,
        decision_reason=decision_reason,
        accepted_claim=accepted_claim,
    )
    update: dict[str, Any] = {
        "processed_requirement_ids": [gap.requirement_id],
        "clarification_records": [record],
        "pending_answer": None,
    }
    if accepted and accepted_claim is not None:
        update["accepted_clarifications"] = [
            CandidateClarification(
                requirement_id=gap.requirement_id,
                question=question,
                answer=answer,
                accepted_claim=accepted_claim,
            )
        ]
    return update


# =============================================================================
# FINAL FULL-CONTEXT REGENERATION
# =============================================================================


def generate_final_package(state: AgentState) -> dict[str, Any]:
    """Run one final workflow after the GAP queue closes and write the package."""

    result = workflow_graph.invoke(_workflow_input(state, persist_package=True))
    package_valid = result["package_valid"]
    errors = result["validation_errors"]
    return {
        **{field: result[field] for field in WORKFLOW_RESULT_FIELDS},
        "final_package_generated": True,
        "stop_reason": (
            "valid_package_complete" if package_valid else "invalid_final_package"
        ),
        "agent_error": None if package_valid else " ".join(errors),
    }


def stop_invalid(state: AgentState) -> dict[str, Any]:
    """Stop safely when the initial package cannot pass deterministic validation."""

    errors = state.get("validation_errors", [])
    return {
        "stop_reason": "invalid_initial_package",
        "agent_error": " ".join(errors) or "The initial package is invalid.",
    }


# =============================================================================
# GRAPH ASSEMBLY
# =============================================================================


def build_agent_graph(checkpointer: Any = None):
    """Compile the round-guided, all-GAP, evidence-gated Agent V2 graph."""

    builder = StateGraph(AgentState, input_schema=AgentInput)
    builder.add_node("parse_interview_round", parse_round_context)
    builder.add_node("generate_initial_package", generate_initial_package)
    builder.add_node("observe_gaps", observe_gaps)
    builder.add_node("ask_user", interrupt_for_gap)
    builder.add_node("assess_clarification", assess_and_record_clarification)
    builder.add_node("generate_final_package", generate_final_package)
    builder.add_node("invalid", stop_invalid)

    builder.add_edge(START, "parse_interview_round")
    builder.add_edge("parse_interview_round", "generate_initial_package")
    builder.add_edge("generate_initial_package", "observe_gaps")
    builder.add_conditional_edges(
        "observe_gaps",
        route_after_observation,
        {
            "ask_user": "ask_user",
            "generate_final": "generate_final_package",
            "invalid": "invalid",
        },
    )
    builder.add_edge("ask_user", "assess_clarification")
    builder.add_edge("assess_clarification", "observe_gaps")
    builder.add_edge("generate_final_package", END)
    builder.add_edge("invalid", END)
    return builder.compile(checkpointer=checkpointer)


agent_graph = build_agent_graph()
