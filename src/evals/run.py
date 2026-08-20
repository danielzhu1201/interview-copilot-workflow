"""CLI for the Lesson 7 baseline and baseline-live suites."""

from __future__ import annotations

import argparse
import os
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from langsmith import Client
from langsmith.schemas import ExampleCreate, ExampleUpdate

from interview_prep.config import load_environment

from .evaluators import evaluate_locally, evaluators_for_suite
from .scenarios import dataset_example, load_dataset, scenarios_for_suite
from .target import make_target, run_scenario

BASELINE_METRICS = (
    "graph_trajectory_strict_match",
    "all_gaps_processed",
    "admission_set_correct",
    "terminal_state_valid",
    "round_guidance_changed",
)


def _example_id(dataset_name: str, suite: str, scenario_id: str) -> UUID:
    if suite == "baseline":
        key = f"{dataset_name}:{scenario_id}"
    else:
        key = f"{dataset_name}:{suite}:{scenario_id}"
    return uuid5(NAMESPACE_URL, key)


def _sync_dataset(client: Client, *, suite: str) -> tuple[str, dict[str, UUID]]:
    dataset_payload = load_dataset()
    dataset_name = dataset_payload["dataset_name"]
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name,
            description=(
                "Lesson 7 behavioral contracts for the Interview Copilot agent."
            ),
            metadata={"lesson": 7, "reference_kind": "behavior-not-prose"},
        )

    example_ids: dict[str, UUID] = {}
    examples_by_id: dict[UUID, dict[str, Any]] = {}
    for scenario in scenarios_for_suite(suite):
        example = dataset_example(scenario)
        scenario_id = scenario["scenario_id"]
        stable_id = _example_id(dataset_name, suite, scenario_id)
        example_ids[scenario_id] = stable_id
        examples_by_id[stable_id] = example

    desired_ids = set(examples_by_id)
    existing_ids = {
        item.id
        for item in client.list_examples(dataset_name=dataset_name)
        if item.id in desired_ids
    }
    creates = []
    updates = []
    for stable_id, example in examples_by_id.items():
        example_type = ExampleUpdate if stable_id in existing_ids else ExampleCreate
        item = example_type(
            id=stable_id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
        )
        if stable_id in existing_ids:
            updates.append(item)
        else:
            creates.append(item)

    if creates:
        client.create_examples(dataset_name=dataset_name, examples=creates)
    if updates:
        client.update_examples(dataset_name=dataset_name, updates=updates)
    return dataset_name, example_ids


def _print_matrix(rows: list[tuple[str, dict[str, bool]]]) -> bool:
    metric_names = list(rows[0][1]) if rows else []
    scenario_width = max([len("scenario"), *(len(name) for name, _ in rows)], default=8)
    metric_widths = {name: max(len(name), len("PASS")) for name in metric_names}
    print(
        "scenario".ljust(scenario_width),
        *(name.ljust(metric_widths[name]) for name in metric_names),
        sep="  ",
    )
    all_green = True
    for scenario_id, metrics in rows:
        values = []
        for metric_name in metric_names:
            passed = metrics[metric_name]
            all_green = all_green and passed
            label = "PASS" if passed else "FAIL"
            values.append(label.ljust(metric_widths[metric_name]))
        print(scenario_id.ljust(scenario_width), *values, sep="  ")
    return all_green


def _run_local(suite: str) -> bool:
    rows = []
    for scenario in scenarios_for_suite(suite):
        example = dataset_example(scenario)
        outputs = run_scenario(example["inputs"], suite=suite)
        metrics = evaluate_locally(outputs, example["outputs"], suite=suite)
        rows.append((scenario["scenario_id"], metrics))
    return _print_matrix(rows)


def _require_environment(suite: str) -> None:
    load_environment()
    missing = []
    if not os.getenv("LANGSMITH_API_KEY"):
        missing.append("LANGSMITH_API_KEY")
    if suite == "baseline-live" and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY")
    if missing:
        raise RuntimeError("Set " + " and ".join(missing) + " before running.")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "interview-copilot-workflow")


def _remote_scores(
    results: Any, *, suite: str
) -> tuple[list[tuple[str, dict[str, bool]]], bool]:
    rows = []
    targets_succeeded = True
    expected_metrics = (
        (*BASELINE_METRICS, "model_backed")
        if suite == "baseline-live"
        else BASELINE_METRICS
    )
    for row in results:
        scenario_id = row["example"].inputs["scenario_id"]
        observed = {
            result.key: bool(result.score)
            for result in row["evaluation_results"]["results"]
        }
        metrics = {key: observed.get(key, False) for key in expected_metrics}
        targets_succeeded = targets_succeeded and row["run"].error is None
        rows.append((scenario_id, metrics))
    return sorted(rows), targets_succeeded


def _run_remote(suite: str, experiment: str) -> bool:
    _require_environment(suite)
    client = Client()
    dataset_name, example_ids = _sync_dataset(client, suite=suite)
    selected = scenarios_for_suite(suite)
    selected_examples = list(
        client.list_examples(
            dataset_name=dataset_name,
            example_ids=[example_ids[item["scenario_id"]] for item in selected],
        )
    )
    results = client.evaluate(
        make_target(suite),
        data=selected_examples,
        evaluators=evaluators_for_suite(suite),
        experiment_prefix=experiment,
        max_concurrency=1,
        metadata={
            "lesson": 7,
            "suite": suite,
            "runtime": "gemini" if suite == "baseline-live" else "fixture",
        },
        error_handling="log",
    )
    rows, targets_succeeded = _remote_scores(results, suite=suite)
    matrix_green = _print_matrix(rows)
    all_green = len(rows) == len(selected) and targets_succeeded and matrix_green
    if results.url:
        print(f"LangSmith experiment: {results.url}")
    return all_green


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("baseline", "baseline-live"),
        required=True,
        help="Named Lesson 7 evaluation suite.",
    )
    parser.add_argument(
        "--experiment",
        help="LangSmith experiment prefix; defaults to the suite name.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run baseline locally without creating a LangSmith experiment.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.local and args.suite != "baseline":
        raise SystemExit("--local is supported only by the baseline suite.")
    experiment = args.experiment or args.suite
    all_green = (
        _run_local(args.suite) if args.local else _run_remote(args.suite, experiment)
    )
    if not all_green:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
