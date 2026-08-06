import json
from pathlib import Path

from interview_prep.schemas import RequirementExtraction
from interview_prep.validation import validate_requirement_set

ROOT = Path(__file__).resolve().parents[1]


def _load_fixture() -> tuple[str, RequirementExtraction]:
    jd = (ROOT / "data" / "mock_jd.txt").read_text(encoding="utf-8")
    payload = json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )
    return jd, RequirementExtraction.model_validate(payload)


def test_expected_fixture_passes_all_grounding_checks() -> None:
    jd, extraction = _load_fixture()

    errors = validate_requirement_set(jd, extraction.requirements)

    assert errors == []


def test_validator_rejects_an_ungrounded_source_quote() -> None:
    jd, extraction = _load_fixture()
    bad_requirements = list(extraction.requirements)
    bad_requirements[0] = bad_requirements[0].model_copy(
        update={"source_quote": "Must have ten years of robotics experience."}
    )

    errors = validate_requirement_set(jd, bad_requirements)

    assert "REQ-01 source_quote is not grounded in the JD." in errors
