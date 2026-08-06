"""Validate the supplied offline requirement fixture."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import RequirementExtraction
from .validation import validate_requirement_set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    jd = (DATA_DIR / "mock_jd.txt").read_text(encoding="utf-8")
    payload = json.loads(
        (DATA_DIR / "expected_requirements.json").read_text(encoding="utf-8")
    )
    extraction = RequirementExtraction.model_validate(payload)
    errors = validate_requirement_set(jd, extraction.requirements)

    if errors:
        print("Fixture validation: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Fixture validation: PASSED")
    print(f"role: {extraction.role_title}")
    print(f"company: {extraction.company}")
    print(f"requirements: {len(extraction.requirements)}")


if __name__ == "__main__":
    main()
