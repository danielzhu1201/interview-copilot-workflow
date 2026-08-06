import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from interview_prep.schemas import JobRequirement, RequirementExtraction

ROOT = Path(__file__).resolve().parents[1]


def test_expected_fixture_matches_the_pydantic_contract() -> None:
    payload = json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )

    result = RequirementExtraction.model_validate(payload)

    assert result.role_title == "Senior Product Data Analyst"
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
