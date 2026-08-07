"""LangGraph nodes for the extraction live build."""

from __future__ import annotations

from typing import Any

from google.genai import types

from .llm import get_gemini_client, get_model_name
from .prompts import build_extraction_prompt
from .schemas import RequirementExtraction, WorkflowState
from .validation import validate_requirement_set


def extract_requirements(state: WorkflowState) -> dict[str, Any]:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_extraction_prompt(state["job_description"]),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RequirementExtraction.model_json_schema(),
        ),
    )
    extraction = RequirementExtraction.model_validate(response.parsed)

    return {"requirements": extraction.requirements}


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
