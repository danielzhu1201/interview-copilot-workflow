from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from interview_prep import agent
from interview_prep.agent import (
    build_agent_graph,
    select_next_gap,
    should_accept_clarification,
)
from interview_prep.schemas import (
    CandidateEvidence,
    ClarificationAssessment,
    EvidenceMatch,
    JobRequirement,
)

ROOT = Path(__file__).resolve().parents[1]


def _requirement(
    requirement_id: str,
    requirement: str,
    *,
    importance: int = 5,
) -> JobRequirement:
    return JobRequirement(
        requirement_id=requirement_id,
        category="technical",
        requirement=requirement,
        importance=importance,
        requirement_type="must_have",
        source_quote=f"Required qualification: {requirement}",
    )


REQUIREMENTS = [
    _requirement("REQ-02", "Use advanced SQL on a cloud data warehouse."),
    _requirement("REQ-03", "Use Python for analysis and automation."),
    _requirement("REQ-04", "Design experiments for product decisions."),
]


def _match(requirement_id: str, coverage: str = "GAP") -> EvidenceMatch:
    return EvidenceMatch(
        requirement_id=requirement_id,
        evidence_ids=[] if coverage == "GAP" else ["EXP-01"],
        coverage=coverage,
        explanation="The supplied evidence determines this coverage result.",
        confidence=0.95,
    )


def _assessment(
    *,
    target_requirement_id: str = "REQ-02",
    is_valid: bool = True,
    accepted_claim: str | None = "I used advanced SQL with complex joins.",
) -> ClarificationAssessment:
    return ClarificationAssessment(
        target_requirement_id=target_requirement_id,
        is_valid=is_valid,
        relevance_reason="The answer directly addresses the target requirement.",
        specificity_reason="The answer includes a concrete tool and activity.",
        accepted_claim=accepted_claim,
    )


def test_select_next_gap_processes_every_gap_in_stable_priority_order() -> None:
    requirements = [
        REQUIREMENTS[0].model_copy(update={"importance": 4}),
        REQUIREMENTS[1],
        REQUIREMENTS[2],
    ]
    state = {
        "requirements": requirements,
        "evidence_matches": [_match(item.requirement_id) for item in requirements],
        "processed_requirement_ids": ["REQ-03"],
    }

    assert select_next_gap(state).requirement_id == "REQ-04"
    state["processed_requirement_ids"].append("REQ-04")
    assert select_next_gap(state).requirement_id == "REQ-02"
    state["processed_requirement_ids"].append("REQ-02")
    assert select_next_gap(state) is None


def test_select_next_gap_ignores_non_gap_matches() -> None:
    state = {
        "requirements": REQUIREMENTS,
        "evidence_matches": [
            _match("REQ-02", "FULL"),
            _match("REQ-03", "PARTIAL"),
            _match("REQ-04", "GAP"),
        ],
    }

    assert select_next_gap(state).requirement_id == "REQ-04"


@pytest.mark.parametrize(
    ("answer", "assessment", "target_requirement_id", "expected"),
    [
        (
            "I wrote SQL.",
            _assessment(),
            "REQ-02",
            False,
        ),
        (
            "I regularly wrote complex SQL queries in BigQuery.",
            _assessment(target_requirement_id="REQ-03"),
            "REQ-02",
            False,
        ),
        (
            "I regularly wrote complex SQL queries in BigQuery.",
            _assessment(is_valid=False, accepted_claim=None),
            "REQ-02",
            False,
        ),
        (
            "I regularly wrote complex SQL queries in BigQuery.",
            _assessment(accepted_claim=None),
            "REQ-02",
            False,
        ),
        (
            "I regularly wrote complex SQL queries in BigQuery.",
            _assessment(),
            "REQ-02",
            True,
        ),
    ],
)
def test_should_accept_clarification_applies_every_code_gate(
    answer: str,
    assessment: ClarificationAssessment,
    target_requirement_id: str,
    expected: bool,
) -> None:
    assert (
        should_accept_clarification(answer, assessment, target_requirement_id)
        is expected
    )


class _FakeWorkflow:
    def __init__(self, *, has_gaps: bool = True):
        self.has_gaps = has_gaps
        self.calls: list[dict] = []

    def invoke(self, workflow_input: dict) -> dict:
        self.calls.append(workflow_input)
        accepted = workflow_input["candidate_clarifications"]
        coverage = "GAP" if self.has_gaps else "FULL"
        matches = [_match(item.requirement_id, coverage) for item in REQUIREMENTS]
        return {
            "interview_round_context": workflow_input["interview_round_context"],
            "candidate_evidence": [
                CandidateEvidence(
                    evidence_id=f"EXP-{index:02d}",
                    claim=item.accepted_claim,
                    source="Candidate clarification",
                )
                for index, item in enumerate(accepted, start=1)
            ],
            "requirements": REQUIREMENTS,
            "evidence_matches": matches,
            "focus_areas": [],
            "interview_strategy": None,
            "mock_questions": [],
            "prep_package": {"valid": True},
            "validation_errors": [],
            "package_valid": True,
        }


def _assessment_client(payloads: list[dict]):
    pending = list(payloads)
    captured: list[dict] = []

    def generate_content(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(parsed=pending.pop(0))

    return (
        SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        captured,
    )


def _valid_payload(requirement_id: str, claim: str) -> dict:
    return {
        "target_requirement_id": requirement_id,
        "is_valid": True,
        "relevance_reason": "The answer directly addresses the requirement.",
        "specificity_reason": "The answer names concrete work and methods.",
        "accepted_claim": claim,
    }


def _invalid_payload(requirement_id: str) -> dict:
    return {
        "target_requirement_id": requirement_id,
        "is_valid": False,
        "relevance_reason": "The answer mentions the skill but gives no experience.",
        "specificity_reason": "The answer has no concrete action or result.",
        "accepted_claim": None,
    }


def test_agent_processes_all_gaps_then_runs_one_final_workflow(monkeypatch) -> None:
    fake_workflow = _FakeWorkflow()
    fake_client, captured = _assessment_client(
        [
            {
                "round_type": "analytics case",
                "format": "60-minute live case",
                "interviewer_roles": ["Hiring Manager"],
                "focus": ["SQL", "experimentation"],
                "notes": None,
            },
            _valid_payload(
                "REQ-02",
                "I used advanced SQL with complex joins and window functions.",
            ),
            _invalid_payload("REQ-03"),
            _valid_payload(
                "REQ-04",
                "I designed and analyzed seven controlled product experiments.",
            ),
        ]
    )
    monkeypatch.setattr(agent, "workflow_graph", fake_workflow)
    monkeypatch.setattr(agent, "get_gemini_client", lambda: fake_client)
    monkeypatch.setattr(agent, "get_model_name", lambda: "test-model")

    compiled = build_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lesson-6-all-gaps"}}
    inputs = {
        "job_description": "Example JD",
        "resume_text": "Example resume",
        "interview_round": (
            "A 60-minute live analytics case with the hiring manager, focused "
            "on SQL and experimentation."
        ),
    }

    paused = compiled.invoke(inputs, config=config)
    assert paused["__interrupt__"][0].value["requirement_id"] == "REQ-02"

    paused = compiled.invoke(
        Command(
            resume=(
                "I wrote complex SQL in BigQuery using joins, window functions, "
                "and query-plan optimization."
            )
        ),
        config=config,
    )
    assert paused["__interrupt__"][0].value["requirement_id"] == "REQ-03"

    paused = compiled.invoke(
        Command(resume="I have used Python and would like to use it more."),
        config=config,
    )
    assert paused["__interrupt__"][0].value["requirement_id"] == "REQ-04"

    result = compiled.invoke(
        Command(
            resume=(
                "I designed seven controlled onboarding experiments and used "
                "the results to prioritize the product roadmap."
            )
        ),
        config=config,
    )

    assert result["stop_reason"] == "valid_package_complete"
    assert result["processed_requirement_ids"] == ["REQ-02", "REQ-03", "REQ-04"]
    assert [record.accepted for record in result["clarification_records"]] == [
        True,
        False,
        True,
    ]
    assert [item.requirement_id for item in result["accepted_clarifications"]] == [
        "REQ-02",
        "REQ-04",
    ]
    assert len(fake_workflow.calls) == 2
    assert (
        fake_workflow.calls[0]["interview_round_context"]
        == (fake_workflow.calls[1]["interview_round_context"])
    )
    assert fake_workflow.calls[0]["persist_package"] is False
    assert fake_workflow.calls[0]["candidate_clarifications"] == []
    assert fake_workflow.calls[1]["persist_package"] is True
    assert [
        item.requirement_id
        for item in fake_workflow.calls[1]["candidate_clarifications"]
    ] == ["REQ-02", "REQ-04"]
    assert len(captured) == 4
    assert set(captured[0]["config"].response_json_schema["properties"]) == {
        "round_type",
        "format",
        "interviewer_roles",
        "focus",
        "notes",
    }
    assessment_calls = captured[1:]
    assert "Example JD" not in assessment_calls[0]["contents"]
    assert "Example resume" not in assessment_calls[0]["contents"]
    assert set(assessment_calls[0]["config"].response_json_schema["properties"]) == {
        "target_requirement_id",
        "is_valid",
        "relevance_reason",
        "specificity_reason",
        "accepted_claim",
    }


def test_agent_without_gaps_still_builds_one_canonical_final_package(
    monkeypatch,
) -> None:
    fake_workflow = _FakeWorkflow(has_gaps=False)
    fake_client, captured = _assessment_client([])
    monkeypatch.setattr(agent, "workflow_graph", fake_workflow)
    monkeypatch.setattr(agent, "get_gemini_client", lambda: fake_client)

    compiled = build_agent_graph()
    result = compiled.invoke(
        {"job_description": "Example JD", "resume_text": "Example resume"}
    )

    assert result["stop_reason"] == "valid_package_complete"
    assert result["interview_round_context"] is None
    assert result.get("processed_requirement_ids", []) == []
    assert result.get("clarification_records", []) == []
    assert len(fake_workflow.calls) == 2
    assert captured == []


def test_lesson6_studio_fixture_separates_round_and_gap_scenarios() -> None:
    fixture = json.loads(
        (ROOT / "data" / "lesson6_studio_inputs.json").read_text(encoding="utf-8")
    )
    case = fixture["perfect_resume_analytics_case"]
    panel = fixture["perfect_resume_cross_functional_panel"]
    imperfect = fixture["imperfect_profile_with_gaps"]

    assert case["job_description"] == panel["job_description"]
    assert case["resume_text"] == panel["resume_text"]
    assert case["resume_text"] == (ROOT / "data" / "mock_resume.md").read_text(
        encoding="utf-8"
    )
    assert "Strong in SQL, Python" in case["resume_text"]
    assert "controlled experiments" in case["resume_text"]
    assert "analytics case" in case["interview_round"]
    assert "cross-functional panel" in panel["interview_round"]
    assert imperfect["job_description"] == case["job_description"]
    assert "difference-in-differences" in imperfect["resume_text"]
    assert "complex joins" not in imperfect["resume_text"]
    assert "controlled experiments" not in imperfect["resume_text"]
    assert fixture["expected_gap_flow"] == {
        "expected_queue": ["REQ-02", "REQ-03", "REQ-04"],
        "expected_processed_requirement_ids": ["REQ-02", "REQ-03", "REQ-04"],
        "expected_accepted_requirement_ids": ["REQ-02", "REQ-04"],
        "expected_rejected_requirement_ids": ["REQ-03"],
    }
    responses = fixture["gap_responses"]
    assert "complex joins" in responses["REQ-02"]["answer"]
    assert responses["REQ-02"]["expected_result"] == "accepted"
    assert responses["REQ-03"] == {
        "answer": "I have used Python and would like to use it more.",
        "expected_result": "rejected",
    }
    assert "seven controlled" in responses["REQ-04"]["answer"]
    assert responses["REQ-04"]["expected_result"] == "accepted"
