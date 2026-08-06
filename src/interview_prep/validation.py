"""Deterministic validation for model-produced requirements."""

from __future__ import annotations

import re

from .schemas import JobRequirement


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_requirement_set(
    job_description: str,
    requirements: list[JobRequirement],
) -> list[str]:
    """Return actionable validation errors; an empty list means valid."""

    errors: list[str] = []

    if len(requirements) < 5:
        errors.append(
            f"Expected at least 5 requirements; received {len(requirements)}."
        )
    if len(requirements) > 12:
        errors.append(
            f"Expected at most 12 requirements; received {len(requirements)}."
        )

    ids = [item.requirement_id for item in requirements]
    if len(ids) != len(set(ids)):
        errors.append("Requirement IDs must be unique.")

    expected_ids = [f"REQ-{index:02d}" for index in range(1, len(ids) + 1)]
    if ids and ids != expected_ids:
        errors.append("Requirement IDs must be sequential and ordered from REQ-01.")

    normalized_jd = _normalize_whitespace(job_description)
    for item in requirements:
        if _normalize_whitespace(item.source_quote) not in normalized_jd:
            errors.append(
                f"{item.requirement_id} source_quote is not grounded in the JD."
            )

    statements = [_normalize_whitespace(item.requirement) for item in requirements]
    if len(statements) != len(set(statements)):
        errors.append("Requirement statements must be unique.")

    return errors
