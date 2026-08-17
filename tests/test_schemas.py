import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from interview_prep.schemas import (
    ClarificationAssessment,
    InterviewRound,
    JobRequirement,
    RequirementExtraction,
    WorkflowState,
)

ROOT = Path(__file__).resolve().parents[1]


def test_expected_fixture_matches_the_pydantic_contract() -> None:
    payload = json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )

    result = RequirementExtraction.model_validate(
        {"requirements": payload["requirements"]}
    )

    assert set(RequirementExtraction.model_fields) == {"requirements"}
    assert len(result.requirements) == 8


def test_requirement_rejects_an_invalid_id() -> None:
    with pytest.raises(ValidationError):
        JobRequirement(
            requirement_id="requirement-1",
            category="technical",
            requirement="Use advanced SQL for product analysis.",
            importance=5,
            requirement_type="must_have",
            source_quote="Advanced SQL skills are required.",
        )


def test_lesson6_round_and_assessment_contracts_are_strict() -> None:
    empty_round = InterviewRound()
    interview_round = InterviewRound.model_validate(
        {
            "round_type": "analytics case",
            "interviewer_roles": ["Hiring Manager"],
            "focus": ["hypothesis testing"],
        }
    )
    assessment = ClarificationAssessment.model_validate(
        {
            "target_requirement_id": "REQ-04",
            "is_valid": True,
            "relevance_reason": "The answer directly addresses experiments.",
            "specificity_reason": "It includes a concrete action and result.",
            "accepted_claim": "I designed and analyzed seven experiments.",
        }
    )

    assert empty_round.model_dump() == {
        "round_type": None,
        "format": None,
        "interviewer_roles": [],
        "focus": [],
        "notes": None,
    }
    assert interview_round.format is None
    assert assessment.target_requirement_id == "REQ-04"
    with pytest.raises(ValidationError):
        InterviewRound.model_validate(
            {"round_type": "case", "unexpected": "not allowed"}
        )


def test_workflow_state_retains_both_raw_source_documents() -> None:
    assert WorkflowState.__required_keys__ == {
        "job_description",
        "resume_text",
        "candidate_evidence",
        "requirements",
        "evidence_matches",
        "focus_areas",
        "interview_strategy",
        "mock_questions",
        "prep_package",
        "validation_errors",
        "package_valid",
    }
