"""Validate the supplied offline requirement fixture."""

from __future__ import annotations

import json
from pathlib import Path

from .inputs import resume_markdown_to_evidence
from .schemas import EvidenceMatchList, RequirementExtraction
from .validation import validate_evidence_match_set, validate_requirement_set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    jd = (DATA_DIR / "mock_jd.txt").read_text(encoding="utf-8")
    payload = json.loads(
        (DATA_DIR / "expected_requirements.json").read_text(encoding="utf-8")
    )
    extraction = RequirementExtraction.model_validate(
        {"requirements": payload["requirements"]}
    )
    errors = validate_requirement_set(jd, extraction.requirements)
    evidence = resume_markdown_to_evidence(
        (DATA_DIR / "mock_resume.md").read_text(encoding="utf-8")
    )
    match_payload = json.loads(
        (DATA_DIR / "expected_evidence_matches.json").read_text(encoding="utf-8")
    )
    matches = EvidenceMatchList.model_validate(match_payload)
    errors.extend(
        validate_evidence_match_set(
            extraction.requirements,
            evidence,
            matches.evidence_matches,
        )
    )

    if errors:
        print("Fixture validation: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Fixture validation: PASSED")
    print(f"requirements: {len(extraction.requirements)}")
    print(f"evidence matches: {len(matches.evidence_matches)}")


if __name__ == "__main__":
    main()
