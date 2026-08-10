import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from interview_prep.schemas import JobRequirement, RequirementExtraction, WorkflowState

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
