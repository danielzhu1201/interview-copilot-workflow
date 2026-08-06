"""LangGraph nodes for the extraction live build."""

from __future__ import annotations

from typing import Any

from .schemas import WorkflowState
from .validation import validate_requirement_set


def extract_requirements(state: WorkflowState) -> dict[str, Any]:
    """TODO: call Gemini and return a validated partial state update.

    Complete this node in class. The intended implementation uses:

    - build_extraction_prompt(state["job_description"])
    - get_gemini_client()
    - get_model_name()
    - google.genai.types.GenerateContentConfig
    - RequirementExtraction as the response schema

    Return only these business-state fields:

    {
        "role_title": parsed.role_title,
        "company": parsed.company,
        "requirements": parsed.requirements,
    }

    Do not put the raw Gemini response in WorkflowState.
    """

    # The empty update keeps the graph runnable before the live build.
    return {"requirements": []}


def validate_requirements(state: WorkflowState) -> dict[str, Any]:
    """Apply deterministic grounding and completeness checks."""

    errors = validate_requirement_set(
        state["job_description"],
        state.get("requirements", []),
    )
    return {
        "validation_errors": errors,
        "requirements_valid": not errors,
    }


def mark_extraction_ready(_state: WorkflowState) -> dict[str, Any]:
    """Mark the first slice ready for the next workflow node."""

    return {"status": "ready"}


def report_validation_errors(_state: WorkflowState) -> dict[str, Any]:
    """Mark the slice invalid without hiding the validation errors."""

    return {"status": "invalid"}


def route_after_validation(state: WorkflowState) -> str:
    """Choose one of two predefined branches from deterministic state."""

    return "ready" if state["requirements_valid"] else "invalid"
