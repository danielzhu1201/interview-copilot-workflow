import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from interview_prep.graph import graph
from interview_prep.nodes import (
    assess_gaps,
    extract_candidate_evidence,
    extract_requirements,
    match_evidence,
    report_errors,
    route_after_validation,
    validate_inputs,
    validate_package,
)
from interview_prep.run import load_inputs
from interview_prep.schemas import JobRequirement

ROOT = Path(__file__).resolve().parents[1]


def _requirements_payload() -> dict:
    return json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )


def _strategy_payload() -> dict:
    return {
        "top_priorities": [
            {
                "requirement_id": "REQ-01",
                "evidence_ids": [],
                "preparation_theme": "Address the experience requirement honestly.",
                "rationale": "The placeholder matcher currently exposes a gap.",
            }
        ],
        "positioning_statement": (
            "Lead with transferable analytics strengths while addressing gaps honestly."
        ),
        "stories_to_prepare": [
            {
                "requirement_id": "REQ-01",
                "evidence_ids": [],
                "story_to_prepare": "Prepare an honest adjacent-experience example.",
            }
        ],
        "risks_to_address": [
            {
                "requirement_id": "REQ-01",
                "risk": "The required experience is not yet linked to evidence.",
                "mitigation": (
                    "Explain transferable experience without inventing claims."
                ),
            }
        ],
    }


def _questions_payload() -> dict:
    return {
        "mock_questions": [
            {
                "question": f"How would you approach requirement REQ-{index:02d}?",
                "requirement_id": f"REQ-{index:02d}",
                "capability_tested": "Grounded communication",
                "evidence_ids": [],
                "follow_up_probe": "What would you learn or verify next?",
                "answer_outline": [
                    "State the relevant transferable strength.",
                    "Acknowledge the evidence gap honestly.",
                ],
            }
            for index in range(1, 9)
        ]
    }


def _fake_client(captured: list[dict] | None = None):
    def generate_content(**kwargs):
        if captured is not None:
            captured.append(kwargs)
        properties = set(kwargs["config"].response_json_schema["properties"])
        if properties == {"requirements"}:
            parsed = {"requirements": _requirements_payload()["requirements"]}
        elif "top_priorities" in properties:
            parsed = _strategy_payload()
        elif properties == {"mock_questions"}:
            parsed = _questions_payload()
        else:  # pragma: no cover - protects this fixture from schema drift
            raise AssertionError(f"Unexpected response schema: {properties}")
        return SimpleNamespace(parsed=parsed)

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_extract_requirements_requests_a_json_schema_requirement_list(
    monkeypatch,
) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        "interview_prep.nodes.get_gemini_client", lambda: _fake_client(captured)
    )
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")

    result = extract_requirements({"job_description": "Example JD"})

    config = captured[0]["config"]
    assert result["requirements"][0].requirement_id == "REQ-01"
    assert config.response_schema is None
    assert set(config.response_json_schema["properties"]) == {"requirements"}


def test_validate_inputs_returns_no_derived_state() -> None:
    result = validate_inputs(
        {
            "job_description": "Example job description",
            "resume_text": "## Experience\n- Built a reliable analytics pipeline.",
        }
    )

    assert result == {}


def test_extract_candidate_evidence_rejects_resume_without_evidence() -> None:
    with pytest.raises(
        ValueError, match="Resume must contain at least one evidence item"
    ):
        extract_candidate_evidence(
            {
                "resume_text": "## Experience",
            }
        )


def test_extract_candidate_evidence_returns_parsed_evidence() -> None:
    result = extract_candidate_evidence(
        {"resume_text": "## Experience\n- Built a reliable analytics pipeline."}
    )

    assert set(result) == {"candidate_evidence"}
    assert result["candidate_evidence"][0].evidence_id == "EXP-01"


def test_placeholder_matching_fails_closed_and_assessment_prioritizes_gaps() -> None:
    requirement = JobRequirement.model_validate(
        _requirements_payload()["requirements"][0]
    )

    match_update = match_evidence({"requirements": [requirement]})
    focus_update = assess_gaps(
        {
            "requirements": [requirement],
            "evidence_matches": match_update["evidence_matches"],
        }
    )

    assert match_update["evidence_matches"][0].coverage == "GAP"
    assert match_update["evidence_matches"][0].evidence_ids == []
    assert focus_update["focus_areas"][0].priority == requirement.importance * 3


def test_complete_graph_reaches_an_assembled_package_without_live_gemini(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    fake_client = _fake_client()
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")

    inputs = load_inputs()
    result = graph.invoke(inputs)

    assert result["job_description"] == inputs["job_description"]
    assert result["resume_text"] == inputs["resume_text"]
    assert len(result["candidate_evidence"]) == 17
    assert result["package_valid"] is True
    assert result["validation_errors"] == []
    assert result["prep_package"] is not None
    assert len(result["prep_package"].mock_questions) == 8


def test_incomplete_package_uses_the_invalid_output_branch() -> None:
    validation_update = validate_package({})

    assert validation_update["package_valid"] is False
    assert validation_update["validation_errors"]
    assert route_after_validation(validation_update) == "invalid"
    assert report_errors(validation_update) == {"prep_package": None}
