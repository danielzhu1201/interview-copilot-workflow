import json
from pathlib import Path

from interview_prep.inputs import resume_markdown_to_evidence
from interview_prep.schemas import EvidenceMatchList, RequirementExtraction
from interview_prep.validation import (
    validate_evidence_match_set,
    validate_requirement_set,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_fixture() -> tuple[str, RequirementExtraction]:
    jd = (ROOT / "data" / "mock_jd.txt").read_text(encoding="utf-8")
    payload = json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )
    return jd, RequirementExtraction.model_validate(
        {"requirements": payload["requirements"]}
    )


def _load_matching_fixture() -> tuple[RequirementExtraction, EvidenceMatchList]:
    _, requirements = _load_fixture()
    payload = json.loads(
        (ROOT / "data" / "expected_evidence_matches.json").read_text(encoding="utf-8")
    )
    return requirements, EvidenceMatchList.model_validate(payload)


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


def test_evidence_match_fixture_passes_all_reference_checks() -> None:
    requirements, match_list = _load_matching_fixture()
    evidence = resume_markdown_to_evidence(
        (ROOT / "data" / "mock_resume.md").read_text(encoding="utf-8")
    )

    errors = validate_evidence_match_set(
        requirements.requirements,
        evidence,
        match_list.evidence_matches,
    )

    assert errors == []


def test_evidence_match_validator_rejects_missing_and_inconsistent_coverage() -> None:
    requirements, match_list = _load_matching_fixture()
    evidence = resume_markdown_to_evidence(
        (ROOT / "data" / "mock_resume.md").read_text(encoding="utf-8")
    )
    bad_matches = list(match_list.evidence_matches[:-1])
    bad_matches[0] = bad_matches[0].model_copy(update={"coverage": "GAP"})
    bad_matches[1] = bad_matches[1].model_copy(update={"evidence_ids": []})

    errors = validate_evidence_match_set(
        requirements.requirements,
        evidence,
        bad_matches,
    )

    assert "Evidence matches are missing requirement IDs: REQ-08." in errors
    assert "REQ-01 is GAP and must not reference evidence." in errors
    assert "REQ-02 is PARTIAL and must reference at least one evidence item." in errors
