import json
from pathlib import Path
from types import SimpleNamespace

import httpx
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
from interview_prep.run import _display_event, load_inputs
from interview_prep.schemas import JobRequirement

ROOT = Path(__file__).resolve().parents[1]


def _requirements_payload() -> dict:
    return json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )


def _evidence_matches_payload() -> dict:
    return json.loads(
        (ROOT / "data" / "expected_evidence_matches.json").read_text(encoding="utf-8")
    )


def _strategy_payload() -> dict:
    return {
        "top_priorities": [
            {
                "requirement_id": "REQ-01",
                "evidence_ids": ["EXP-01"],
                "preparation_theme": "Address the experience requirement honestly.",
                "rationale": "The supplied evidence supports digital product work.",
            }
        ],
        "positioning_statement": (
            "Lead with transferable analytics strengths while addressing gaps honestly."
        ),
        "stories_to_prepare": [
            {
                "requirement_id": "REQ-01",
                "evidence_ids": ["EXP-01"],
                "story_to_prepare": "Prepare the product analytics ownership story.",
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
    evidence_by_requirement = {
        1: ["EXP-01"],
        2: ["EXP-02"],
        3: ["EXP-04"],
        4: ["EXP-03"],
        5: ["EXP-06"],
        6: ["EXP-05"],
        7: ["EXP-04"],
        8: ["EXP-13"],
    }
    return {
        "mock_questions": [
            {
                "question": f"How would you approach requirement REQ-{index:02d}?",
                "requirement_id": f"REQ-{index:02d}",
                "capability_tested": "Grounded communication",
                "evidence_ids": evidence_by_requirement[index],
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
        elif properties == {"evidence_matches"}:
            parsed = _evidence_matches_payload()
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


def test_display_event_handles_noop_node(capsys) -> None:
    final_state = {"existing": "value"}

    _display_event({"validate_inputs": None}, final_state)

    assert final_state == {"existing": "value"}
    assert "NODE: validate_inputs" in capsys.readouterr().out


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


def _matching_state() -> dict:
    requirements = [
        JobRequirement.model_validate(item)
        for item in _requirements_payload()["requirements"]
    ]
    inputs = load_inputs()
    candidate_evidence = extract_candidate_evidence(inputs)["candidate_evidence"]
    return {
        "requirements": requirements,
        "candidate_evidence": candidate_evidence,
    }


def test_match_evidence_uses_schema_and_returns_requirement_order(monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        "interview_prep.nodes.get_gemini_client", lambda: _fake_client(captured)
    )
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")

    state = _matching_state()
    result = match_evidence(state)
    focus_update = assess_gaps(
        {
            "requirements": state["requirements"],
            "evidence_matches": result["evidence_matches"],
        }
    )

    config = captured[0]["config"]
    assert set(config.response_json_schema["properties"]) == {"evidence_matches"}
    assert [item.requirement_id for item in result["evidence_matches"]] == [
        item.requirement_id for item in state["requirements"]
    ]
    assert result["evidence_matches"][0].coverage == "PARTIAL"
    assert result["evidence_matches"][0].evidence_ids == ["EXP-01"]
    assert focus_update["focus_areas"][0].priority == 10


def test_match_evidence_rejects_unknown_evidence_id(monkeypatch) -> None:
    payload = _evidence_matches_payload()
    payload["evidence_matches"][0]["evidence_ids"] = ["EXP-99"]

    def generate_content(**_kwargs):
        return SimpleNamespace(parsed=payload)

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)

    with pytest.raises(ValueError, match="unknown evidence IDs: EXP-99"):
        match_evidence(_matching_state())


def test_match_evidence_uses_validated_fixture_on_network_failure(
    monkeypatch,
) -> None:
    def generate_content(**_kwargs):
        raise httpx.ConnectError("offline")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)

    result = match_evidence(_matching_state())

    assert len(result["evidence_matches"]) == 8
    assert result["evidence_matches"][2].coverage == "FULL"
    assert result["evidence_matches"][2].evidence_ids == ["EXP-04", "EXP-09"]


def test_complete_graph_reaches_an_assembled_package_without_live_gemini(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    fake_client = _fake_client()
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")
    candidate_prep_path = tmp_path / "candidate-prep.md"
    monkeypatch.setattr("interview_prep.nodes.CANDIDATE_PREP_PATH", candidate_prep_path)

    inputs = load_inputs()
    result = graph.invoke(inputs)

    assert result["job_description"] == inputs["job_description"]
    assert result["resume_text"] == inputs["resume_text"]
    assert len(result["candidate_evidence"]) == 17
    assert result["package_valid"] is True
    assert result["validation_errors"] == []
    assert result["prep_package"] is not None
    assert len(result["prep_package"].mock_questions) == 8
    markdown = candidate_prep_path.read_text(encoding="utf-8")
    assert "# Next-Round Interview Prep" in markdown
    assert "## Positioning" in markdown
    assert "## Practice questions" in markdown
    assert "EXP-01" not in markdown
    assert "source_quote" not in markdown


def test_validate_package_rejects_ungrounded_requirements_and_gap_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    fake_client = _fake_client()
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")
    monkeypatch.setattr(
        "interview_prep.nodes.CANDIDATE_PREP_PATH", tmp_path / "candidate-prep.md"
    )
    state = graph.invoke(load_inputs())

    state["requirements"][0] = state["requirements"][0].model_copy(
        update={"source_quote": "Unsupported robotics experience is required."}
    )
    state["evidence_matches"][0] = state["evidence_matches"][0].model_copy(
        update={"coverage": "GAP", "evidence_ids": []}
    )
    state["focus_areas"] = [
        item.model_copy(update={"coverage": "GAP"})
        if item.requirement_id == "REQ-01"
        else item
        for item in state["focus_areas"]
    ]

    validation_update = validate_package(state)

    assert validation_update["package_valid"] is False
    assert (
        "REQ-01 source_quote is not grounded in the JD."
        in validation_update["validation_errors"]
    )
    assert (
        "Strategy item for REQ-01 must not reference evidence for a GAP requirement."
        in validation_update["validation_errors"]
    )
    assert (
        "Mock question item for REQ-01 must not reference evidence for a GAP "
        "requirement."
    ) in validation_update["validation_errors"]


def test_incomplete_package_uses_the_invalid_output_branch() -> None:
    validation_update = validate_package({})

    assert validation_update["package_valid"] is False
    assert validation_update["validation_errors"]
    assert route_after_validation(validation_update) == "invalid"
    assert report_errors(validation_update) == {"prep_package": None}
