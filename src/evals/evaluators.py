"""Deterministic behavioral evaluators for Lesson 7 experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _expected_runs(reference_outputs: dict[str, Any]) -> dict[str, Any]:
    return reference_outputs["runs"]


def trajectory_match(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Require every interrupt/resume turn to follow the frozen node order."""

    scores = []
    for run_id, expected in _expected_runs(reference_outputs).items():
        result = graph_trajectory_strict_match(
            outputs={"steps": outputs["runs"][run_id]["steps"], "results": []},
            reference_outputs={
                "steps": expected["expected_steps"],
                "results": [],
            },
        )
        scores.append(bool(result["score"]))
    return {"key": "graph_trajectory_strict_match", "score": all(scores)}


def all_gaps_processed(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Check processed IDs and interrupt counts for every scenario run."""

    score = all(
        outputs["runs"][run_id]["processed_ids"] == expected["expected_processed_ids"]
        and outputs["runs"][run_id]["interrupt_count"]
        == expected["expected_interrupt_count"]
        for run_id, expected in _expected_runs(reference_outputs).items()
    )
    return {"key": "all_gaps_processed", "score": score}


def admission_set_correct(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Check accepted/rejected sets, audit coverage, and evidence isolation."""

    scores = []
    for run_id, expected in _expected_runs(reference_outputs).items():
        actual = outputs["runs"][run_id]
        audited_ids = [record["requirement_id"] for record in actual["audit_records"]]
        rejected_are_isolated = not (
            set(actual["rejected_ids"]) & set(actual["admitted_requirement_ids"])
        )
        scores.append(
            actual["accepted_ids"] == expected["expected_accepted_ids"]
            and actual["rejected_ids"] == expected["expected_rejected_ids"]
            and actual["remaining_gap_ids"] == expected["expected_remaining_gap_ids"]
            and audited_ids == expected["expected_processed_ids"]
            and rejected_are_isolated
        )
    return {"key": "admission_set_correct", "score": all(scores)}


def terminal_state_valid(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Check the final package gate and terminal stop reason."""

    score = all(
        outputs["runs"][run_id]["package_valid"] == expected["expected_package_valid"]
        and outputs["runs"][run_id]["stop_reason"] == expected["expected_stop_reason"]
        for run_id, expected in _expected_runs(reference_outputs).items()
    )
    return {"key": "terminal_state_valid", "score": score}


def round_guidance_changed(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Compare round guidance while holding evidence and coverage constant."""

    expected = reference_outputs.get("round_guidance_changed")
    if expected is None:
        return {
            "key": "round_guidance_changed",
            "score": True,
            "comment": "Not applicable to this scenario.",
        }
    runs = list(outputs["runs"].values())
    score = (
        len(runs) == 2
        and runs[0]["evidence_signature"] == runs[1]["evidence_signature"]
        and runs[0]["coverage_signature"] == runs[1]["coverage_signature"]
        and runs[0]["guidance_signature"] != runs[1]["guidance_signature"]
    )
    return {"key": "round_guidance_changed", "score": score == expected}


def model_backed(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """Confirm that the live suite did not use deterministic dependencies."""

    del reference_outputs
    score = outputs.get("model_backed") is True and all(
        run["model_backed"] is True and run["model_call_count"] > 0
        for run in outputs["runs"].values()
    )
    return {"key": "model_backed", "score": score}


def evaluators_for_suite(suite: str) -> list[Evaluator]:
    """Return the evaluator set for a named suite."""

    evaluators: list[Evaluator] = [
        trajectory_match,
        all_gaps_processed,
        admission_set_correct,
        terminal_state_valid,
        round_guidance_changed,
    ]
    if suite == "baseline-live":
        evaluators.append(model_backed)
    return evaluators


def evaluate_locally(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
    *,
    suite: str,
) -> dict[str, bool]:
    """Apply the same evaluators without uploading a LangSmith experiment."""

    return {
        result["key"]: bool(result["score"])
        for evaluator in evaluators_for_suite(suite)
        for result in [evaluator(outputs, reference_outputs)]
    }
