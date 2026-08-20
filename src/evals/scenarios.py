"""Load and validate the local Lesson 7 evaluation dataset."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "lesson7_eval_dataset.json"
LESSON6_INPUTS_PATH = PROJECT_ROOT / "data" / "lesson6_studio_inputs.json"
MOCK_JD_PATH = PROJECT_ROOT / "data" / "mock_jd.txt"
MOCK_RESUME_PATH = PROJECT_ROOT / "data" / "mock_resume.md"
SUITES = {"baseline", "baseline-live"}


def load_dataset() -> dict[str, Any]:
    """Return the validated local dataset document."""

    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Lesson 7 dataset must contain scenarios.")

    scenario_ids = [scenario.get("scenario_id") for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Lesson 7 scenario IDs must be unique.")
    for scenario in scenarios:
        scenario_suites = set(scenario.get("suites", []))
        if not scenario_suites or not scenario_suites <= SUITES:
            raise ValueError(
                f"{scenario.get('scenario_id')} declares unsupported suites."
            )
        runs = scenario.get("runs", [])
        reference_runs = scenario.get("reference", {}).get("runs", {})
        run_ids = [run.get("run_id") for run in runs]
        if len(run_ids) != len(set(run_ids)) or set(run_ids) != set(reference_runs):
            raise ValueError(
                f"{scenario.get('scenario_id')} run references are inconsistent."
            )
        for expected in reference_runs.values():
            if expected.get("expected_trajectory") not in payload.get(
                "trajectory_templates", {}
            ):
                raise ValueError(
                    f"{scenario.get('scenario_id')} references an unknown trajectory."
                )
    return payload


def scenarios_for_suite(suite: str) -> list[dict[str, Any]]:
    """Select dataset scenarios belonging to one named suite."""

    if suite not in SUITES:
        raise ValueError(f"Unknown Lesson 7 suite: {suite}")
    return [
        scenario
        for scenario in load_dataset()["scenarios"]
        if suite in scenario["suites"]
    ]


def scenario_by_id(scenario_id: str) -> dict[str, Any]:
    """Resolve one scenario by its stable ID."""

    for scenario in load_dataset()["scenarios"]:
        if scenario["scenario_id"] == scenario_id:
            return scenario
    raise KeyError(f"Unknown Lesson 7 scenario: {scenario_id}")


def source_inputs(profile: str) -> dict[str, str]:
    """Resolve the complete or intentionally incomplete teaching profile."""

    job_description = MOCK_JD_PATH.read_text(encoding="utf-8")
    if profile == "complete":
        resume_text = MOCK_RESUME_PATH.read_text(encoding="utf-8")
    elif profile == "imperfect":
        lesson6 = json.loads(LESSON6_INPUTS_PATH.read_text(encoding="utf-8"))
        resume_text = lesson6["imperfect_profile_with_gaps"]["resume_text"]
    else:
        raise ValueError(f"Unknown Lesson 7 profile: {profile}")
    return {"job_description": job_description, "resume_text": resume_text}


def dataset_example(scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert a local scenario into LangSmith example inputs and outputs."""

    inputs = {key: scenario[key] for key in ("scenario_id", "description", "profile")}
    inputs["runs"] = [
        {
            key: run[key]
            for key in ("run_id", "interview_round", "answers_by_requirement")
        }
        for run in scenario["runs"]
    ]
    reference = deepcopy(scenario["reference"])
    templates = load_dataset()["trajectory_templates"]
    for expected in reference["runs"].values():
        trajectory_name = expected.pop("expected_trajectory")
        expected["expected_steps"] = templates[trajectory_name]
    return {
        "inputs": inputs,
        "outputs": reference,
        "metadata": {"lesson": 7, "suites": scenario["suites"]},
    }
