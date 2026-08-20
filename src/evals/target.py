"""Run Lesson 7 scenarios through real LangGraph threads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from interview_prep import agent
from interview_prep.agent import build_agent_graph

from .runtime import fixture_runtime, live_runtime
from .scenarios import scenario_by_id, source_inputs

CHECKPOINT_ALLOWED_TYPES = [
    ("interview_prep.schemas", type_name)
    for type_name in (
        "CandidateClarification",
        "CandidateEvidence",
        "ClarificationRecord",
        "EvidenceMatch",
        "FocusArea",
        "InterviewRound",
        "InterviewStrategy",
        "JobRequirement",
        "MockQuestion",
        "PrepPackage",
    )
]


def _serialize_models(items: list[Any]) -> list[dict[str, Any]]:
    return [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in items
    ]


def _run_thread(
    *,
    profile: str,
    run: dict[str, Any],
    model_backed: bool,
) -> dict[str, Any]:
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES,
    )
    compiled = build_agent_graph(checkpointer=InMemorySaver(serde=serializer))
    config = {"configurable": {"thread_id": f"lesson7-{uuid4()}"}}
    inputs = {
        **source_inputs(profile),
        "interview_round": run["interview_round"],
    }
    state = compiled.invoke(inputs, config=config)
    interrupt_count = 0
    while state.get("__interrupt__"):
        interrupt_value = state["__interrupt__"][0].value
        requirement_id = interrupt_value["requirement_id"]
        answers = run["answers_by_requirement"]
        if requirement_id not in answers:
            raise ValueError(
                f"{run['run_id']} has no answer for interrupt {requirement_id}."
            )
        interrupt_count += 1
        state = compiled.invoke(Command(resume=answers[requirement_id]), config=config)

    extracted = extract_langgraph_trajectory_from_thread(compiled, config)
    records = state.get("clarification_records", [])
    accepted = state.get("accepted_clarifications", [])
    evidence_matches = state.get("evidence_matches", [])
    strategy = state.get("interview_strategy")
    mock_questions = state.get("mock_questions", [])
    return {
        "steps": extracted["outputs"]["steps"],
        "interrupt_count": interrupt_count,
        "processed_ids": list(state.get("processed_requirement_ids", [])),
        "accepted_ids": [item.requirement_id for item in accepted],
        "rejected_ids": [
            record.requirement_id for record in records if not record.accepted
        ],
        "admitted_requirement_ids": [item.requirement_id for item in accepted],
        "remaining_gap_ids": [
            match.requirement_id
            for match in evidence_matches
            if match.coverage == "GAP"
        ],
        "package_valid": state.get("package_valid", False),
        "stop_reason": state.get("stop_reason"),
        "audit_records": [
            {
                "requirement_id": record.requirement_id,
                "assessment_target_id": record.assessment.target_requirement_id,
                "accepted": record.accepted,
            }
            for record in records
        ],
        "evidence_signature": _serialize_models(state.get("candidate_evidence", [])),
        "coverage_signature": [
            {
                "requirement_id": match.requirement_id,
                "coverage": match.coverage,
                "evidence_ids": match.evidence_ids,
            }
            for match in evidence_matches
        ],
        "guidance_signature": {
            "positioning_statement": (
                strategy.positioning_statement if strategy is not None else None
            ),
            "questions": [question.question for question in mock_questions],
        },
        "model_backed": model_backed,
        "model_call_count": 0,
    }


def run_scenario(inputs: dict[str, Any], *, suite: str) -> dict[str, Any]:
    """Run one dataset scenario with fixture or live model dependencies."""

    profile = inputs["profile"]
    fixture_runs = {
        run["run_id"]: run for run in scenario_by_id(inputs["scenario_id"])["runs"]
    }
    results: dict[str, Any] = {}
    for run in inputs["runs"]:
        if suite == "baseline":
            with fixture_runtime(
                profile=profile,
                assessments_by_requirement=fixture_runs[run["run_id"]][
                    "assessments_by_requirement"
                ],
            ):
                results[run["run_id"]] = _run_thread(
                    profile=profile,
                    run=run,
                    model_backed=False,
                )
        elif suite == "baseline-live":
            with live_runtime() as client:
                result = _run_thread(
                    profile=profile,
                    run=run,
                    model_backed=False,
                )
                result["model_call_count"] = client.call_count
                result["model_backed"] = client.call_count > 0
                results[run["run_id"]] = result
        else:
            raise ValueError(f"Unknown Lesson 7 suite: {suite}")

    return {
        "scenario_id": inputs["scenario_id"],
        "suite": suite,
        "model": agent.get_model_name() if suite == "baseline-live" else None,
        "model_backed": suite == "baseline-live"
        and all(result["model_backed"] for result in results.values()),
        "runs": results,
    }


def make_target(suite: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the target callable passed to LangSmith evaluate()."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_scenario(inputs, suite=suite)

    target.__name__ = suite.replace("-", "_")
    return target
