"""Bounded human-in-the-loop agent wrapped around Workflow V1."""

from __future__ import annotations

from typing import Any, Literal

from google.genai import types  # noqa: F401 - used in Lesson 5 Live Build 1
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt  # noqa: F401 - used in Lesson 5 Live Build 2

from .graph import graph as workflow_graph
from .llm import get_gemini_client  # noqa: F401 - used in Lesson 5 Live Build 1
from .prompts import build_agent_prompt  # noqa: F401 - used in Lesson 5 Live Build 1
from .schemas import (
    AgentAction,
    AgentDecision,  # noqa: F401 - used in Lesson 5 Live Build 1
    AgentInput,
    AgentObservation,
    AgentState,
    CandidateClarification,  # noqa: F401 - used in Lesson 5 Live Build 2
    HighPriorityGap,
)

# =============================================================================
# LESSON 5 AGENT V1 CONFIGURATION
# =============================================================================

DEFAULT_GOAL = "Produce a grounded, validated prep package without inventing evidence."
DECISION_MODEL = "gemini-3.7-flash"
MAX_AGENT_ACTIONS = 4
MAX_CLASSROOM_QUESTIONS = 1
MAX_DECISION_RETRIES = 1
WORKFLOW_RESULT_FIELDS = (
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


# =============================================================================
# OBSERVE AND DECIDE
# =============================================================================


def observe_state(state: AgentState) -> dict[str, Any]:
    """Compress business state into factual, deterministic decision inputs."""

    requirements = {
        requirement.requirement_id: requirement
        for requirement in state.get("requirements", [])
    }
    high_priority_gaps: list[HighPriorityGap] = []
    matches = state.get("evidence_matches", [])
    for match in matches:
        requirement = requirements.get(match.requirement_id)
        if (
            match.coverage == "GAP"
            and requirement is not None
            and requirement.importance >= 4
        ):
            high_priority_gaps.append(
                HighPriorityGap(
                    requirement_id=requirement.requirement_id,
                    requirement=requirement.requirement,
                    importance=requirement.importance,
                    explanation=match.explanation,
                )
            )
    high_priority_gaps.sort(key=lambda gap: (-gap.importance, gap.requirement_id))
    high_priority_gap_ids = [gap.requirement_id for gap in high_priority_gaps]

    clarifications = state.get("candidate_clarifications", [])
    asked_requirement_ids = state.get("asked_requirement_ids", [])
    unasked_gap_ids = [
        requirement_id
        for requirement_id in high_priority_gap_ids
        if requirement_id not in asked_requirement_ids
    ]
    steps_remaining = max(
        0,
        MAX_AGENT_ACTIONS - state.get("action_count", 0),
    )
    package_generated = state.get("package_generated", False)
    package_valid = state.get("package_valid", False)
    last_action = state.get("last_action")

    allowed_actions: list[AgentAction]
    if steps_remaining == 0:
        allowed_actions = []
    elif not package_generated:
        allowed_actions = ["GENERATE_PREP_PACKAGE"]
    elif last_action == "ASK_USER" and clarifications:
        allowed_actions = ["GENERATE_PREP_PACKAGE"]
    elif unasked_gap_ids and len(asked_requirement_ids) < MAX_CLASSROOM_QUESTIONS:
        allowed_actions = ["ASK_USER"]
    elif package_valid:
        allowed_actions = ["FINISH"]
    else:
        allowed_actions = ["GENERATE_PREP_PACKAGE"]

    observation = AgentObservation(
        package_generated=package_generated,
        package_valid=package_valid,
        high_priority_gap_ids=high_priority_gap_ids,
        high_priority_gaps=high_priority_gaps,
        asked_requirement_ids=asked_requirement_ids,
        allowed_actions=allowed_actions,
        latest_clarification=(clarifications[-1].answer if clarifications else None),
        last_action=last_action,
        steps_remaining=steps_remaining,
    )
    return {
        "goal": state.get("goal", DEFAULT_GOAL),
        "observation": observation,
        "decision_retry_count": 0,
        "agent_error": None,
    }


def decide_next_action(state: AgentState) -> dict[str, Any]:
    """Ask Gemini for exactly one schema-validated next action."""

    # === LESSON 5 LIVE BUILD 1: START ===
    # CLASSROOM RESET:
    # Delete only the LIVE IMPLEMENTATION block below and temporarily replace it:
    #     raise NotImplementedError("Complete Live Build 1 in class")
    #
    # Implement this logic:
    # 1. Read goal and observation from state.
    # 2. Build the bounded prompt, including any prior authorization error.
    # 3. Request Gemini JSON using AgentDecision.model_json_schema().
    # 4. Validate response.parsed as AgentDecision.
    # 5. Return only {"next_decision": decision}; never store the raw response.
    # --- LIVE IMPLEMENTATION: START ---
    raise NotImplementedError("Complete Live Build 1 in class")
    # --- LIVE IMPLEMENTATION: END ---
    # === LESSON 5 LIVE BUILD 1: END ===


# =============================================================================
# CODE-OWNED AUTHORIZATION AND ROUTING
# =============================================================================


def _invalid(message: str) -> tuple[Literal["invalid"], dict[str, Any]]:
    return "invalid", {"agent_error": message, "stop_reason": "invalid_decision"}


def authorize_decision(
    state: AgentState,
) -> tuple[Literal["generate", "ask_user", "finish", "invalid"], dict[str, Any]]:
    """Apply every code-owned gate before mapping a decision to a capability."""

    decision = state["next_decision"]
    observation = state["observation"]
    action = decision.next_action

    if observation.steps_remaining < 1:
        return _invalid("The four-action agent budget is exhausted.")

    if action not in observation.allowed_actions:
        allowed = ", ".join(observation.allowed_actions) or "no action"
        return _invalid(f"{action} is not allowed now; choose {allowed}.")

    update: dict[str, Any] = {
        "last_action": action,
        "action_count": state.get("action_count", 0) + 1,
        "agent_error": None,
    }

    if action == "GENERATE_PREP_PACKAGE":
        return "generate", update

    if action == "FINISH":
        if not observation.package_valid:
            return _invalid("FINISH requires a valid prep package.")
        unasked_gap_ids = set(observation.high_priority_gap_ids) - set(
            observation.asked_requirement_ids
        )
        if (
            unasked_gap_ids
            and len(observation.asked_requirement_ids) < MAX_CLASSROOM_QUESTIONS
        ):
            return _invalid("FINISH requires no eligible unasked high-priority gap.")
        update["stop_reason"] = "valid_package_complete"
        return "finish", update

    requirement_id = decision.target_requirement_id
    if not requirement_id or not decision.question:
        return _invalid("ASK_USER requires a question and requirement ID.")
    if requirement_id not in observation.high_priority_gap_ids:
        return _invalid("ASK_USER must target an eligible high-priority gap.")
    if requirement_id in observation.asked_requirement_ids:
        return _invalid("A requirement cannot be asked twice.")
    if len(observation.asked_requirement_ids) >= MAX_CLASSROOM_QUESTIONS:
        return _invalid("At most one classroom question is allowed.")
    return "ask_user", update


def validate_and_route(state: AgentState) -> dict[str, Any]:
    """Authorize the proposed action and store the deterministic route."""

    route, update = authorize_decision(state)
    if (
        route == "invalid"
        and state.get("decision_retry_count", 0) < MAX_DECISION_RETRIES
    ):
        return {
            **update,
            "authorized_route": "retry",
            "decision_retry_count": state.get("decision_retry_count", 0) + 1,
            "stop_reason": None,
        }
    return {**update, "authorized_route": route}


def route_authorized_action(
    state: AgentState,
) -> Literal["generate", "ask_user", "finish", "retry", "invalid"]:
    """Return only the route previously selected by code-owned validation."""

    return state["authorized_route"]


# =============================================================================
# AUTHORIZED CAPABILITIES
# =============================================================================


def generate_prep_package(state: AgentState) -> dict[str, Any]:
    """Run the unchanged Workflow V1 and return only its derived business state."""

    workflow_input = {
        "job_description": state["job_description"],
        "resume_text": state["resume_text"],
        "candidate_clarifications": state.get("candidate_clarifications", []),
    }
    result = workflow_graph.invoke(workflow_input)
    return {
        **{field: result[field] for field in WORKFLOW_RESULT_FIELDS},
        "package_generated": True,
    }


def interrupt_and_record(state: AgentState) -> dict[str, Any]:
    """Pause for one factual answer and record it after same-thread resume."""

    # === LESSON 5 LIVE BUILD 2: START ===
    # CLASSROOM RESET:
    # Delete only the LIVE IMPLEMENTATION block below and temporarily replace it:
    #     raise NotImplementedError("Complete Live Build 2 in class")
    #
    # Implement this logic:
    # 1. Read the authorized decision and its requirement ID.
    # 2. Call interrupt() with type, requirement_id, and question.
    # 3. On same-thread resume, validate the returned factual answer.
    # 4. Create one CandidateClarification from the question and answer.
    # 5. Return additive clarification and asked-ID updates.
    # Keep everything before interrupt() idempotent because the node restarts.
    # --- LIVE IMPLEMENTATION: START ---
    raise NotImplementedError("Complete Live Build 2 in class")
    # --- LIVE IMPLEMENTATION: END ---
    # === LESSON 5 LIVE BUILD 2: END ===


def finish_agent(_state: AgentState) -> dict[str, Any]:
    """End after code has authorized FINISH."""

    return {}


def stop_invalid(_state: AgentState) -> dict[str, Any]:
    """End without executing a capability after an invalid decision."""

    return {}


# =============================================================================
# GRAPH ASSEMBLY
# =============================================================================


def build_agent_graph(checkpointer: Any = None):
    """Compile the Lesson 5 observe-decide-authorize resumable loop."""

    builder = StateGraph(AgentState, input_schema=AgentInput)
    builder.add_node("observe", observe_state)
    builder.add_node("decide", decide_next_action)
    builder.add_node("validate_and_route", validate_and_route)
    builder.add_node("generate_prep_package", generate_prep_package)
    builder.add_node("ask_user", interrupt_and_record)
    builder.add_node("finish", finish_agent)
    builder.add_node("invalid", stop_invalid)

    builder.add_edge(START, "observe")
    builder.add_edge("observe", "decide")
    builder.add_edge("decide", "validate_and_route")
    builder.add_conditional_edges(
        "validate_and_route",
        route_authorized_action,
        {
            "generate": "generate_prep_package",
            "ask_user": "ask_user",
            "finish": "finish",
            "retry": "decide",
            "invalid": "invalid",
        },
    )
    builder.add_edge("generate_prep_package", "observe")
    builder.add_edge("ask_user", "observe")
    builder.add_edge("finish", END)
    builder.add_edge("invalid", END)
    return builder.compile(checkpointer=checkpointer)


agent_graph = build_agent_graph()
