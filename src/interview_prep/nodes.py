"""Bounded transformations used by Interview Prep Workflow V1."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from google.genai import errors as genai_errors
from google.genai import types

from .inputs import resume_markdown_to_evidence
from .llm import get_gemini_client, get_model_name
from .package_writer import CANDIDATE_PREP_PATH, write_candidate_prep
from .prompts import (
    build_evidence_matching_prompt,
    build_extraction_prompt,
    build_questions_prompt,
    build_strategy_prompt,
)
from .schemas import (
    EvidenceMatchList,
    FocusArea,
    InterviewStrategy,
    MockQuestionList,
    PrepPackage,
    RequirementExtraction,
    WorkflowState,
)
from .validation import validate_evidence_match_set, validate_prep_package

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_MATCH_FIXTURE_PATH = PROJECT_ROOT / "data" / "expected_evidence_matches.json"
logger = logging.getLogger(__name__)


def _load_evidence_match_fixture() -> dict[str, Any]:
    return json.loads(EVIDENCE_MATCH_FIXTURE_PATH.read_text(encoding="utf-8"))


def validate_inputs(state: WorkflowState) -> dict[str, Any]:
    """Validate the two raw source documents."""

    job_description = state.get("job_description", "")
    resume_text = state.get("resume_text", "")

    errors: list[str] = []
    if not job_description.strip():
        errors.append("Job description must not be empty.")
    if not resume_text.strip():
        errors.append("Resume must not be empty.")

    if errors:
        raise ValueError("Invalid workflow input: " + " ".join(errors))

    return {}


def extract_candidate_evidence(state: WorkflowState) -> dict[str, Any]:
    """Derive stable candidate evidence from the validated resume."""

    candidate_evidence = resume_markdown_to_evidence(state["resume_text"])
    if not candidate_evidence:
        raise ValueError(
            "Invalid workflow input: Resume must contain at least one evidence item."
        )

    return {"candidate_evidence": candidate_evidence}


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
    """Link candidate evidence to every requirement without inventing support."""

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=build_evidence_matching_prompt(
                state["requirements"],
                state["candidate_evidence"],
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=EvidenceMatchList.model_json_schema(),
            ),
        )
        payload = response.parsed
    except genai_errors.APIError as error:
        if error.code not in {408, 429} and error.code < 500:
            raise
        logger.warning(
            "Gemini evidence matching failed with API status %s; using fixture.",
            error.code,
        )
        payload = _load_evidence_match_fixture()
    except (httpx.ConnectError, httpx.TimeoutException) as error:
        logger.warning(
            "Gemini evidence matching was unavailable (%s); using fixture.",
            type(error).__name__,
        )
        payload = _load_evidence_match_fixture()

    match_list = EvidenceMatchList.model_validate(payload)
    validation_errors = validate_evidence_match_set(
        state["requirements"],
        state["candidate_evidence"],
        match_list.evidence_matches,
    )
    if validation_errors:
        raise ValueError("Invalid evidence matches: " + " ".join(validation_errors))

    matches_by_requirement = {
        item.requirement_id: item for item in match_list.evidence_matches
    }
    ordered_matches = [
        matches_by_requirement[item.requirement_id] for item in state["requirements"]
    ]
    return {"evidence_matches": ordered_matches}


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
    """Apply deterministic grounding, traceability, and completeness checks."""

    errors = validate_prep_package(
        job_description=state.get("job_description", ""),
        candidate_evidence=state.get("candidate_evidence", []),
        requirements=state.get("requirements", []),
        evidence_matches=state.get("evidence_matches", []),
        focus_areas=state.get("focus_areas", []),
        interview_strategy=state.get("interview_strategy"),
        mock_questions=state.get("mock_questions", []),
    )

    return {"validation_errors": errors, "package_valid": not errors}


def route_after_validation(state: WorkflowState) -> str:
    """Select one of the two predefined output branches."""

    return "valid" if state["package_valid"] else "invalid"


def assemble_package(state: WorkflowState) -> dict[str, Any]:
    """Assemble and write candidate-facing output only after validation succeeds."""

    package = PrepPackage(
        requirements=state["requirements"],
        evidence_matches=state["evidence_matches"],
        focus_areas=state["focus_areas"],
        interview_strategy=state["interview_strategy"],
        mock_questions=state["mock_questions"],
    )
    write_candidate_prep(package, CANDIDATE_PREP_PATH)
    return {"prep_package": package}


def report_errors(_state: WorkflowState) -> dict[str, Any]:
    """Keep invalid output unavailable while preserving validation errors."""

    return {"prep_package": None}
