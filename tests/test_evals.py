from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from langsmith.evaluation import run_evaluator
from langsmith.schemas import Example, Run

from evals.evaluators import evaluate_locally, model_backed
from evals.run import _example_id, _sync_dataset
from evals.runtime import CountingGeminiClient
from evals.scenarios import dataset_example, scenario_by_id, scenarios_for_suite
from evals.target import run_scenario
from interview_prep import agent


def _run_baseline(scenario_id: str) -> tuple[dict, dict]:
    scenario = scenario_by_id(scenario_id)
    example = dataset_example(scenario)
    outputs = run_scenario(example["inputs"], suite="baseline")
    return outputs, example["outputs"]


def test_dataset_defines_five_baseline_and_four_live_scenarios() -> None:
    baseline_ids = [item["scenario_id"] for item in scenarios_for_suite("baseline")]
    live_ids = [item["scenario_id"] for item in scenarios_for_suite("baseline-live")]

    assert baseline_ids == [
        "complete-profile-no-round",
        "round-guidance-comparison",
        "mixed-clarifications",
        "all-clarifications-rejected",
        "wrong-target-assessment",
    ]
    assert live_ids == baseline_ids[:-1]


def test_live_model_counter_requires_a_delegated_request() -> None:
    class Models:
        @staticmethod
        def generate_content(**kwargs):
            return kwargs

    delegate = type("Delegate", (), {"models": Models()})()
    client = CountingGeminiClient(delegate)

    assert client.call_count == 0
    assert client.models.generate_content(contents="test") == {"contents": "test"}
    assert client.call_count == 1


def test_model_backed_accepts_langsmith_run_and_example() -> None:
    run_id = uuid4()
    run = Run(
        id=run_id,
        trace_id=run_id,
        name="baseline-live-target",
        run_type="chain",
        start_time=datetime.now(UTC),
        outputs={
            "model_backed": True,
            "runs": {
                "live": {"model_backed": True, "model_call_count": 2},
            },
        },
    )
    example = Example(id=uuid4(), outputs={"runs": {}})

    result = run_evaluator(model_backed).evaluate_run(run, example)

    assert result.key == "model_backed"
    assert result.score is True


def test_live_examples_use_distinct_ids_and_reruns_update_them() -> None:
    dataset_name = "interview-copilot-lesson-7"
    scenario_id = "complete-profile-no-round"
    baseline_id = _example_id(dataset_name, "baseline", scenario_id)
    live_id = _example_id(dataset_name, "baseline-live", scenario_id)

    assert baseline_id == uuid5(NAMESPACE_URL, f"{dataset_name}:{scenario_id}")
    assert live_id != baseline_id

    class FakeClient:
        def __init__(self) -> None:
            self.existing_ids = set()
            self.created = []
            self.updated = []

        @staticmethod
        def has_dataset(**_kwargs):
            return True

        def list_examples(self, **_kwargs):
            return [SimpleNamespace(id=example_id) for example_id in self.existing_ids]

        def create_examples(self, *, examples, **_kwargs):
            self.created.extend(examples)
            self.existing_ids.update(item.id for item in examples)

        def update_examples(self, *, updates, **_kwargs):
            self.updated.extend(updates)

    client = FakeClient()
    _sync_dataset(client, suite="baseline-live")
    assert len(client.created) == 4
    assert client.updated == []
    assert baseline_id not in client.existing_ids

    _sync_dataset(client, suite="baseline-live")
    assert len(client.created) == 4
    assert len(client.updated) == 4


@pytest.mark.parametrize(
    "scenario_id",
    [item["scenario_id"] for item in scenarios_for_suite("baseline")],
)
def test_every_baseline_scenario_is_green(scenario_id: str) -> None:
    outputs, reference = _run_baseline(scenario_id)

    assert all(evaluate_locally(outputs, reference, suite="baseline").values())


def test_round_comparison_changes_guidance_but_not_evidence() -> None:
    outputs, _reference = _run_baseline("round-guidance-comparison")
    case = outputs["runs"]["analytics-case"]
    panel = outputs["runs"]["cross-functional-panel"]

    assert case["interrupt_count"] == panel["interrupt_count"] == 0
    assert case["evidence_signature"] == panel["evidence_signature"]
    assert case["coverage_signature"] == panel["coverage_signature"]
    assert case["guidance_signature"] != panel["guidance_signature"]


def test_wrong_target_assessment_is_rejected_and_audited() -> None:
    outputs, _reference = _run_baseline("wrong-target-assessment")
    result = outputs["runs"]["wrong-target"]

    assert result["accepted_ids"] == ["REQ-03", "REQ-04"]
    assert result["rejected_ids"] == ["REQ-02"]
    assert result["admitted_requirement_ids"] == ["REQ-03", "REQ-04"]
    assert result["audit_records"][0] == {
        "requirement_id": "REQ-02",
        "assessment_target_id": "REQ-03",
        "accepted": False,
    }


def test_one_question_regression_turns_path_and_state_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_select_next_gap = agent.select_next_gap

    def select_at_most_one_gap(state):
        if state.get("processed_requirement_ids"):
            return None
        return original_select_next_gap(state)

    monkeypatch.setattr(agent, "select_next_gap", select_at_most_one_gap)
    outputs, reference = _run_baseline("mixed-clarifications")
    metrics = evaluate_locally(outputs, reference, suite="baseline")
    result = outputs["runs"]["mixed"]

    assert metrics["graph_trajectory_strict_match"] is False
    assert metrics["all_gaps_processed"] is False
    assert metrics["admission_set_correct"] is False
    assert metrics["terminal_state_valid"] is True
    assert result["package_valid"] is True
    assert result["stop_reason"] == "valid_package_complete"
    assert result["processed_ids"] == ["REQ-02"]
