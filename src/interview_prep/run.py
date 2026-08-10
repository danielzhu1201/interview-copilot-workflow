"""Run the live-build graph against the supplied fictional inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_environment
from .graph import graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUN_CONFIG = {"configurable": {"enable_interrupts": False}}


def load_inputs() -> dict[str, str]:
    """Load the mock JD and resume as the two source inputs."""

    return {
        "job_description": (DATA_DIR / "mock_jd.txt").read_text(encoding="utf-8"),
        "resume_text": (DATA_DIR / "mock_resume.md").read_text(encoding="utf-8"),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _display_event(event: dict[str, Any], final_state: dict[str, Any]) -> None:
    """Print one normal state update from the non-interactive CLI run."""

    for node_name, update in event.items():
        final_state.update(update)
        print(f"NODE: {node_name}")
        print(json.dumps(update, indent=2, default=_json_default))
        print()


def main() -> None:
    load_environment()
    print("Running: START → extract → validate → ready/invalid → END\n")

    final_state: dict[str, Any] = {}
    for event in graph.stream(
        load_inputs(),
        config=RUN_CONFIG,
        stream_mode="updates",
    ):
        _display_event(event, final_state)

    print("RESULT")
    print(f"status: {final_state.get('status', 'unknown')}")
    print(f"requirements: {len(final_state.get('requirements', []))}")
    print(f"valid: {final_state.get('requirements_valid', False)}")
    if final_state.get("validation_errors"):
        print("validation errors:")
        for error in final_state["validation_errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
