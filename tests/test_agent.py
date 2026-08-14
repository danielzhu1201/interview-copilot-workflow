from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from interview_prep import agent
from interview_prep.agent import (
    authorize_decision,
    build_agent_graph,
    decide_next_action,
    observe_state,
)
from interview_prep.nodes import extract_candidate_evidence
from interview_prep.schemas import (
    AgentDecision,
    AgentObservation,
    CandidateClarification,
    EvidenceMatch,
    HighPriorityGap,
    JobRequirement,
)

ROOT = Path(__file__).resolve().parents[1]


def _requirement(*, importance: int = 5) -> JobRequirement:
    return JobRequirement(
        requirement_id="REQ-04",
        category="analytics",
        requirement="Design experiments for product decisions.",
        importance=importance,
        requirement_type="must_have",
        source_quote="Hands-on experience designing experiments for product decisions.",
    )


def _match(coverage: str) -> EvidenceMatch:
    return EvidenceMatch(
        requirement_id="REQ-04",
        evidence_ids=[] if coverage == "GAP" else ["EXP-18"],
        coverage=coverage,
        explanation="The supplied evidence determines this coverage result.",
        confidence=0.95,
    )


def _observation(**updates) -> AgentObservation:
    values = {
        "package_generated": True,
        "package_valid": True,
        "high_priority_gap_ids": ["REQ-04"],
        "high_priority_gaps": [
            HighPriorityGap(
                requirement_id="REQ-04",
                requirement="Design experiments for product decisions.",
                importance=5,
                explanation="No direct experiment evidence was supplied.",
            )
        ],
        "asked_requirement_ids": [],
        "allowed_actions": ["ASK_USER"],
        "latest_clarification": None,
        "last_action": "GENERATE_PREP_PACKAGE",
        "steps_remaining": 3,
    }
    values.update(updates)
    return AgentObservation.model_validate(values)


def _decision(action: str, **updates) -> AgentDecision:
    values = {
        "next_action": action,
        "reason_summary": "This is the next bounded action.",
    }
    values.update(updates)
    return AgentDecision.model_validate(values)


def test_observation_derives_threshold_clarification_and_budget() -> None:
    result = observe_state(
        {
            "requirements": [_requirement(importance=4)],
            "evidence_matches": [_match("GAP")],
            "candidate_clarifications": [
                CandidateClarification(
                    requirement_id="REQ-04",
                    question="What experiment did you design?",
                    answer="I designed a controlled onboarding experiment.",
                )
            ],
            "asked_requirement_ids": ["REQ-04"],
            "action_count": 2,
            "last_action": "ASK_USER",
            "package_generated": True,
            "package_valid": True,
        }
    )

    observation = result["observation"]
    assert observation.high_priority_gap_ids == ["REQ-04"]
    assert observation.high_priority_gaps[0].requirement.startswith("Design")
    assert observation.allowed_actions == ["GENERATE_PREP_PACKAGE"]
    assert observation.latest_clarification.startswith("I designed")
    assert observation.steps_remaining == 2


def test_observation_excludes_importance_three_gap() -> None:
    result = observe_state(
        {
            "requirements": [_requirement(importance=3)],
            "evidence_matches": [_match("GAP")],
        }
    )

    assert result["observation"].high_priority_gap_ids == []
    assert result["observation"].high_priority_gaps == []
    assert result["observation"].allowed_actions == ["GENERATE_PREP_PACKAGE"]


def test_observation_requires_ask_before_finish_for_unasked_gap() -> None:
    result = observe_state(
        {
            "requirements": [_requirement()],
            "evidence_matches": [_match("GAP")],
            "package_generated": True,
            "package_valid": True,
            "last_action": "GENERATE_PREP_PACKAGE",
            "action_count": 1,
        }
    )

    observation = result["observation"]
    assert observation.high_priority_gap_ids == ["REQ-04"]
    assert observation.allowed_actions == ["ASK_USER"]


def test_observation_exposes_three_high_priority_gaps_deterministically() -> None:
    requirement_specs = [
        ("REQ-02", "Use advanced SQL on a cloud data warehouse."),
        ("REQ-03", "Use Python for analysis and automation."),
        ("REQ-04", "Design experiments for product decisions."),
    ]
    requirements = [
        JobRequirement(
            requirement_id=requirement_id,
            category="technical" if requirement_id != "REQ-04" else "analytics",
            requirement=requirement,
            importance=5,
            requirement_type="must_have",
            source_quote=f"Required qualification: {requirement}",
        )
        for requirement_id, requirement in requirement_specs
    ]
    matches = [
        EvidenceMatch(
            requirement_id=requirement_id,
            evidence_ids=[],
            coverage="GAP",
            explanation="No direct supporting candidate evidence was supplied.",
            confidence=1,
        )
        for requirement_id, _ in requirement_specs
    ]

    observation = observe_state(
        {
            "requirements": requirements,
            "evidence_matches": matches,
            "package_generated": True,
            "package_valid": True,
            "last_action": "GENERATE_PREP_PACKAGE",
            "action_count": 1,
        }
    )["observation"]

    assert observation.high_priority_gap_ids == ["REQ-02", "REQ-03", "REQ-04"]
    assert observation.allowed_actions == ["ASK_USER"]


def test_decide_next_action_requests_agent_decision_schema(monkeypatch) -> None:
    captured = []

    def generate_content(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            parsed={
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "No package has been generated yet.",
            }
        )

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(agent, "get_gemini_client", lambda: fake_client)
    monkeypatch.setattr(agent, "get_model_name", lambda: "test-agent-model")

    result = decide_next_action(
        {
            "goal": agent.DEFAULT_GOAL,
            "observation": _observation(
                package_generated=False,
                allowed_actions=["GENERATE_PREP_PACKAGE"],
            ),
            "agent_error": "The prior action was not allowed.",
        }
    )

    assert result["next_decision"].next_action == "GENERATE_PREP_PACKAGE"
    assert captured[0]["model"] == "test-agent-model"
    assert set(captured[0]["config"].response_json_schema["properties"]) == {
        "next_action",
        "target_requirement_id",
        "question",
        "reason_summary",
    }
    assert '"allowed_actions": [' in captured[0]["contents"]
    assert "Design experiments for product decisions" in captured[0]["contents"]
    assert "PRIOR DECISION REJECTED BY CODE" in captured[0]["contents"]


@pytest.mark.parametrize(
    ("decision", "observation", "error"),
    [
        (
            _decision("ASK_USER"),
            _observation(),
            "requires a question and requirement ID",
        ),
        (
            _decision(
                "ASK_USER",
                target_requirement_id="REQ-04",
                question="What experiment did you design?",
            ),
            _observation(high_priority_gap_ids=[]),
            "eligible high-priority gap",
        ),
        (
            _decision(
                "ASK_USER",
                target_requirement_id="REQ-04",
                question="What experiment did you design?",
            ),
            _observation(asked_requirement_ids=["REQ-04"]),
            "cannot be asked twice",
        ),
        (
            _decision(
                "ASK_USER",
                target_requirement_id="REQ-04",
                question="What experiment did you design?",
            ),
            _observation(asked_requirement_ids=["REQ-03"]),
            "At most one classroom question",
        ),
        (
            _decision("FINISH"),
            _observation(package_valid=False, allowed_actions=["FINISH"]),
            "requires a valid prep package",
        ),
        (
            _decision("FINISH"),
            _observation(allowed_actions=["FINISH"]),
            "no eligible unasked high-priority gap",
        ),
        (
            _decision("GENERATE_PREP_PACKAGE"),
            _observation(steps_remaining=0),
            "budget is exhausted",
        ),
    ],
)
def test_authorization_rejects_invalid_decisions(
    decision: AgentDecision,
    observation: AgentObservation,
    error: str,
) -> None:
    route, update = authorize_decision(
        {
            "next_decision": decision,
            "observation": observation,
            "action_count": 1,
        }
    )

    assert route == "invalid"
    assert error in update["agent_error"]
    assert update["stop_reason"] == "invalid_decision"


def _decision_client(payloads: list[dict]):
    pending = list(payloads)

    def generate_content(**_kwargs):
        return SimpleNamespace(parsed=pending.pop(0))

    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


class _FakeWorkflow:
    def __init__(self, *, starts_with_gap: bool):
        self.starts_with_gap = starts_with_gap
        self.calls: list[dict] = []

    def invoke(self, workflow_input: dict) -> dict:
        self.calls.append(workflow_input)
        clarified = bool(workflow_input["candidate_clarifications"])
        coverage = "GAP" if self.starts_with_gap and not clarified else "FULL"
        return {
            "candidate_evidence": [],
            "requirements": [_requirement()],
            "evidence_matches": [_match(coverage)],
            "focus_areas": [],
            "interview_strategy": None,
            "mock_questions": [],
            "prep_package": {"valid": True},
            "validation_errors": [],
            "package_valid": True,
        }


def _run_agent(monkeypatch, decisions: list[dict], *, starts_with_gap: bool):
    fake_workflow = _FakeWorkflow(starts_with_gap=starts_with_gap)
    fake_client = _decision_client(decisions)
    monkeypatch.setattr(agent, "workflow_graph", fake_workflow)
    monkeypatch.setattr(
        agent,
        "get_gemini_client",
        lambda: fake_client,
    )
    compiled = build_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lesson-5-test"}}
    return compiled, config, fake_workflow


def test_enough_evidence_trajectory_finishes_without_interrupt(monkeypatch) -> None:
    compiled, config, fake_workflow = _run_agent(
        monkeypatch,
        [
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Generate the initial validated package.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "The package is valid and complete.",
            },
        ],
        starts_with_gap=False,
    )

    result = compiled.invoke(
        {"job_description": "Example JD", "resume_text": "Example resume"},
        config=config,
    )

    assert result["stop_reason"] == "valid_package_complete"
    assert result["action_count"] == 2
    assert result.get("asked_requirement_ids", []) == []
    assert len(fake_workflow.calls) == 1


def test_gap_trajectory_interrupts_resumes_and_regenerates(monkeypatch) -> None:
    compiled, config, fake_workflow = _run_agent(
        monkeypatch,
        [
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Generate the initial validated package.",
            },
            {
                "next_action": "ASK_USER",
                "target_requirement_id": "REQ-04",
                "question": "What experiment did you design and analyze?",
                "reason_summary": "One factual answer could resolve the gap.",
            },
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Regenerate with the candidate clarification.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "The regenerated package is valid.",
            },
        ],
        starts_with_gap=True,
    )

    paused = compiled.invoke(
        {"job_description": "Example JD", "resume_text": "Example resume"},
        config=config,
    )
    request = paused["__interrupt__"][0].value

    assert request == {
        "type": "candidate_evidence_request",
        "requirement_id": "REQ-04",
        "question": "What experiment did you design and analyze?",
    }

    answer = (
        "I designed and analyzed seven controlled product experiments at Meridian "
        "Works. One onboarding experiment increased team activation by 8.4% and "
        "informed the next-quarter roadmap."
    )
    result = compiled.invoke(Command(resume=answer), config=config)

    assert result["stop_reason"] == "valid_package_complete"
    assert result["action_count"] == 4
    assert result["asked_requirement_ids"] == ["REQ-04"]
    assert len(result["candidate_clarifications"]) == 1
    assert len(fake_workflow.calls) == 2
    clarification = fake_workflow.calls[1]["candidate_clarifications"][0]
    assert clarification.answer == answer


def test_invalid_finish_is_retried_once_as_ask_user(monkeypatch) -> None:
    compiled, config, fake_workflow = _run_agent(
        monkeypatch,
        [
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Generate the initial validated package.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "Incorrectly try to finish despite the gap.",
            },
            {
                "next_action": "ASK_USER",
                "target_requirement_id": "REQ-04",
                "question": "What experiment did you design and analyze?",
                "reason_summary": "Ask about the eligible experiment gap.",
            },
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Regenerate with the clarification.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "The regenerated package is valid.",
            },
        ],
        starts_with_gap=True,
    )

    paused = compiled.invoke(
        {"job_description": "Example JD", "resume_text": "Example resume"},
        config=config,
    )

    assert paused["__interrupt__"][0].value["requirement_id"] == "REQ-04"
    result = compiled.invoke(
        Command(resume="I designed and analyzed a controlled product experiment."),
        config=config,
    )

    assert result["stop_reason"] == "valid_package_complete"
    assert result["action_count"] == 4
    assert len(fake_workflow.calls) == 2


def test_second_invalid_decision_stops_without_spending_action(monkeypatch) -> None:
    compiled, config, fake_workflow = _run_agent(
        monkeypatch,
        [
            {
                "next_action": "GENERATE_PREP_PACKAGE",
                "reason_summary": "Generate the initial validated package.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "Incorrectly finish despite the gap.",
            },
            {
                "next_action": "FINISH",
                "reason_summary": "Repeat the unauthorized finish decision.",
            },
        ],
        starts_with_gap=True,
    )

    result = compiled.invoke(
        {"job_description": "Example JD", "resume_text": "Example resume"},
        config=config,
    )

    assert result["stop_reason"] == "invalid_decision"
    assert result["authorized_route"] == "invalid"
    assert result["decision_retry_count"] == 1
    assert result["action_count"] == 1
    assert len(fake_workflow.calls) == 1


def test_studio_inputs_are_copy_ready_and_create_the_two_conditions() -> None:
    fixtures = json.loads(
        (ROOT / "data" / "lesson5_studio_inputs.json").read_text(encoding="utf-8")
    )
    enough = fixtures["enough_evidence"]
    gap = fixtures["high_priority_gap"]

    assert enough["job_description"] == (ROOT / "data" / "mock_jd.txt").read_text(
        encoding="utf-8"
    )
    assert enough["resume_text"] == (ROOT / "data" / "mock_resume.md").read_text(
        encoding="utf-8"
    )
    assert gap["job_description"] == enough["job_description"]
    gap_evidence = extract_candidate_evidence(gap)["candidate_evidence"]
    assert any(
        "Six years of professional experience" in item.claim
        and "four years directly supporting digital product teams" in item.claim
        for item in gap_evidence
    )
    assert "controlled experiments" not in gap["resume_text"]
    assert "matched-market comparisons" not in gap["resume_text"]
    assert "A/B testing" not in gap["resume_text"]
    assert "Strong in SQL, Python" not in gap["resume_text"]
    assert "BigQuery" not in gap["resume_text"]
    assert "Used Python and SQL" not in gap["resume_text"]
    assert "Developed SQL models" not in gap["resume_text"]
    assert "Built Python notebooks" not in gap["resume_text"]
    assert "measurable hypotheses, event-tracking" not in gap["resume_text"]
    assert "pause a proposed notification feature" not in gap["resume_text"]
    answers = fixtures["clarification_answers"]
    assert set(answers) == {"REQ-02", "REQ-03", "REQ-04"}
    assert "complex joins and window functions" in answers["REQ-02"]
    assert "Python and pandas" in answers["REQ-03"]
    assert "seven controlled product experiments" in answers["REQ-04"]
