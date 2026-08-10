"""Bounded transformations used by Interview Prep Workflow V1."""

from __future__ import annotations

from typing import Any

from google.genai import types

from .inputs import resume_markdown_to_evidence
from .llm import get_gemini_client, get_model_name
from .prompts import (
    build_extraction_prompt,
    build_questions_prompt,
    build_strategy_prompt,
)
from .schemas import (
    EvidenceMatch,
    FocusArea,
    InterviewStrategy,
    MockQuestionList,
    PrepPackage,
    RequirementExtraction,
    WorkflowState,
)


def validate_inputs(state: WorkflowState) -> dict[str, Any]:
    """Validate raw documents and derive stable candidate evidence."""

    # TODO(class): simplify this node to validate the two raw documents, call
    # resume_markdown_to_evidence(), and return only candidate_evidence. The
    # parser already owns sequential IDs, so the duplicate-ID check is
    # redundant; package fields should be initialized by package validation.
    job_description = state.get("job_description", "")
    resume_text = state.get("resume_text", "")
    evidence = resume_markdown_to_evidence(resume_text) if resume_text.strip() else []

    errors: list[str] = []
    if not job_description.strip():
        errors.append("Job description must not be empty.")
    if not resume_text.strip():
        errors.append("Resume must not be empty.")
    elif not evidence:
        errors.append("Resume must contain at least one evidence item.")

    evidence_ids = [item.evidence_id for item in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("Candidate evidence IDs must be unique.")

    if errors:
        raise ValueError("Invalid workflow input: " + " ".join(errors))

    return {
        "candidate_evidence": evidence,
        "validation_errors": [],
        "package_valid": False,
        "prep_package": None,
    }


def extract_requirements(state: WorkflowState) -> dict[str, Any]:
    """Extract schema-validated and source-grounded job requirements."""

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


def match_evidence(state: WorkflowState) -> dict[str, Any]:
    """Return safe GAPs until the Lesson 4 matching live build is completed.

    TODO(lesson-4-live-build-1): replace this fail-closed placeholder with the
    grounded Gemini structured-output call and deterministic ID guards.
    """

    matches = [
        EvidenceMatch(
            requirement_id=requirement.requirement_id,
            evidence_ids=[],
            coverage="GAP",
            explanation=(
                "Evidence matching is pending; no candidate support is asserted."
            ),
            confidence=0,
        )
        for requirement in state["requirements"]
    ]
    return {"evidence_matches": matches}


def assess_gaps(state: WorkflowState) -> dict[str, Any]:
    """Convert coverage into deterministic, importance-weighted focus areas."""

    requirements = {item.requirement_id: item for item in state["requirements"]}
    coverage_weight = {"FULL": 1, "PARTIAL": 2, "GAP": 3}
    action_for = {
        "FULL": "Prepare a concise story that proves this strength.",
        "PARTIAL": "Strengthen the story and address the unsupported dimension.",
        "GAP": "Prepare an honest gap response and a concrete learning plan.",
    }

    focus_areas: list[FocusArea] = []
    for match in state["evidence_matches"]:
        requirement = requirements[match.requirement_id]
        focus_areas.append(
            FocusArea(
                requirement_id=match.requirement_id,
                coverage=match.coverage,
                priority=requirement.importance * coverage_weight[match.coverage],
                preparation_action=action_for[match.coverage],
                reason=match.explanation,
            )
        )

    focus_areas.sort(key=lambda item: item.priority, reverse=True)
    return {"focus_areas": focus_areas}


def build_strategy(state: WorkflowState) -> dict[str, Any]:
    """Use Gemini to build an ID-linked preparation strategy."""

    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_strategy_prompt(
            state["requirements"],
            state["evidence_matches"],
            state["focus_areas"],
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=InterviewStrategy.model_json_schema(),
        ),
    )
    strategy = InterviewStrategy.model_validate(response.parsed)
    return {"interview_strategy": strategy}


def generate_questions(state: WorkflowState) -> dict[str, Any]:
    """Use Gemini to turn the grounded strategy into mock interview practice."""

    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=build_questions_prompt(
            state["requirements"],
            state["evidence_matches"],
            state["interview_strategy"],
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=MockQuestionList.model_json_schema(),
        ),
    )
    question_list = MockQuestionList.model_validate(response.parsed)
    return {"mock_questions": question_list.mock_questions}


def validate_package(state: WorkflowState) -> dict[str, Any]:
    """Temporarily check package shape until the validation live build.

    TODO(lesson-4-live-build-2): replace these minimal readiness checks with
    the complete deterministic reference, coverage, and section invariants.
    """

    errors: list[str] = []
    if not state.get("requirements"):
        errors.append("The package has no requirements.")
    if not state.get("evidence_matches"):
        errors.append("The package has no evidence matches.")
    if not state.get("focus_areas"):
        errors.append("The package has no focus areas.")
    if not state.get("interview_strategy"):
        errors.append("The package has no interview strategy.")
    if len(state.get("mock_questions", [])) < 8:
        errors.append("The package must contain at least eight mock questions.")

    return {"validation_errors": errors, "package_valid": not errors}


def route_after_validation(state: WorkflowState) -> str:
    """Select one of the two predefined output branches."""

    return "valid" if state["package_valid"] else "invalid"


def assemble_package(state: WorkflowState) -> dict[str, Any]:
    """Assemble candidate-facing output only after validation succeeds."""

    package = PrepPackage(
        requirements=state["requirements"],
        evidence_matches=state["evidence_matches"],
        focus_areas=state["focus_areas"],
        interview_strategy=state["interview_strategy"],
        mock_questions=state["mock_questions"],
    )
    return {"prep_package": package}


def report_errors(_state: WorkflowState) -> dict[str, Any]:
    """Keep invalid output unavailable while preserving validation errors."""

    return {"prep_package": None}
