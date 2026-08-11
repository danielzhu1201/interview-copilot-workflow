"""Deterministic validation for model-produced requirements."""

from __future__ import annotations

import re
from collections import Counter

from .schemas import (
    CandidateEvidence,
    EvidenceMatch,
    FocusArea,
    InterviewStrategy,
    JobRequirement,
    MockQuestion,
    StoryPlan,
    StrategyItem,
)


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


def validate_evidence_match_set(
    requirements: list[JobRequirement],
    candidate_evidence: list[CandidateEvidence],
    matches: list[EvidenceMatch],
) -> list[str]:
    """Return deterministic reference and coverage errors for matching output."""

    errors: list[str] = []
    requirement_ids = [item.requirement_id for item in requirements]
    requirement_id_set = set(requirement_ids)
    evidence_id_set = {item.evidence_id for item in candidate_evidence}
    matched_requirement_ids = [item.requirement_id for item in matches]
    requirement_match_counts = Counter(matched_requirement_ids)

    duplicate_requirement_ids = sorted(
        requirement_id
        for requirement_id, count in requirement_match_counts.items()
        if count > 1
    )
    if duplicate_requirement_ids:
        errors.append(
            "Evidence matches contain duplicate requirement IDs: "
            + ", ".join(duplicate_requirement_ids)
            + "."
        )

    missing_requirement_ids = sorted(requirement_id_set - set(matched_requirement_ids))
    if missing_requirement_ids:
        errors.append(
            "Evidence matches are missing requirement IDs: "
            + ", ".join(missing_requirement_ids)
            + "."
        )

    unknown_requirement_ids = sorted(set(matched_requirement_ids) - requirement_id_set)
    if unknown_requirement_ids:
        errors.append(
            "Evidence matches reference unknown requirement IDs: "
            + ", ".join(unknown_requirement_ids)
            + "."
        )

    for match in matches:
        evidence_counts = Counter(match.evidence_ids)
        duplicate_evidence_ids = sorted(
            evidence_id for evidence_id, count in evidence_counts.items() if count > 1
        )
        if duplicate_evidence_ids:
            errors.append(
                f"{match.requirement_id} repeats evidence IDs: "
                + ", ".join(duplicate_evidence_ids)
                + "."
            )

        unknown_evidence_ids = sorted(set(match.evidence_ids) - evidence_id_set)
        if unknown_evidence_ids:
            errors.append(
                f"{match.requirement_id} references unknown evidence IDs: "
                + ", ".join(unknown_evidence_ids)
                + "."
            )

        if match.coverage == "GAP" and match.evidence_ids:
            errors.append(
                f"{match.requirement_id} is GAP and must not reference evidence."
            )
        if match.coverage in {"FULL", "PARTIAL"} and not match.evidence_ids:
            errors.append(
                f"{match.requirement_id} is {match.coverage} and must reference "
                "at least one evidence item."
            )

    return errors


def _validate_requirement_references(
    items: list[FocusArea | StrategyItem | StoryPlan | MockQuestion],
    section: str,
    requirement_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    for item in items:
        if item.requirement_id not in requirement_ids:
            errors.append(
                f"{section} references unknown requirement ID: {item.requirement_id}."
            )
    return errors


def _validate_evidence_links(
    items: list[StrategyItem | StoryPlan | MockQuestion],
    section: str,
    matches_by_requirement: dict[str, EvidenceMatch],
) -> list[str]:
    """Ensure downstream evidence links remain within the matched evidence."""

    errors: list[str] = []
    for item in items:
        match = matches_by_requirement.get(item.requirement_id)
        if match is None:
            continue

        item_name = f"{section} item for {item.requirement_id}"
        if match.coverage == "GAP":
            if item.evidence_ids:
                errors.append(
                    f"{item_name} must not reference evidence for a GAP requirement."
                )
            continue

        if not item.evidence_ids:
            errors.append(f"{item_name} must retain at least one matched evidence ID.")
            continue

        unsupported_evidence_ids = sorted(
            set(item.evidence_ids) - set(match.evidence_ids)
        )
        if unsupported_evidence_ids:
            errors.append(
                f"{item_name} references evidence not matched to the requirement: "
                + ", ".join(unsupported_evidence_ids)
                + "."
            )

    return errors


def validate_prep_package(
    job_description: str,
    candidate_evidence: list[CandidateEvidence],
    requirements: list[JobRequirement],
    evidence_matches: list[EvidenceMatch],
    focus_areas: list[FocusArea],
    interview_strategy: InterviewStrategy | None,
    mock_questions: list[MockQuestion],
) -> list[str]:
    """Validate the deterministic release criteria for a prep package."""

    errors: list[str] = []
    if not job_description.strip():
        errors.append("The package has no job description for grounding checks.")
    if not candidate_evidence:
        errors.append("The package has no candidate evidence.")
    if not requirements:
        errors.append("The package has no requirements.")
    if not evidence_matches:
        errors.append("The package has no evidence matches.")
    if not focus_areas:
        errors.append("The package has no focus areas.")
    if interview_strategy is None:
        errors.append("The package has no interview strategy.")
    if len(mock_questions) < 8:
        errors.append("The package must contain at least eight mock questions.")

    evidence_ids = [item.evidence_id for item in candidate_evidence]
    duplicate_evidence_ids = sorted(
        evidence_id for evidence_id, count in Counter(evidence_ids).items() if count > 1
    )
    if duplicate_evidence_ids:
        errors.append(
            "Candidate evidence IDs must be unique: "
            + ", ".join(duplicate_evidence_ids)
            + "."
        )

    if requirements and job_description.strip():
        errors.extend(validate_requirement_set(job_description, requirements))

    if requirements and candidate_evidence and evidence_matches:
        errors.extend(
            validate_evidence_match_set(
                requirements,
                candidate_evidence,
                evidence_matches,
            )
        )

    requirement_ids = {item.requirement_id for item in requirements}
    matches_by_requirement = {item.requirement_id: item for item in evidence_matches}

    if focus_areas:
        errors.extend(
            _validate_requirement_references(
                focus_areas,
                "Focus area",
                requirement_ids,
            )
        )
        focus_counts = Counter(item.requirement_id for item in focus_areas)
        for requirement_id in sorted(requirement_ids - set(focus_counts)):
            errors.append(f"Focus areas are missing requirement ID: {requirement_id}.")
        for requirement_id, count in sorted(focus_counts.items()):
            if count > 1:
                errors.append(
                    f"Focus areas contain duplicate requirement ID: {requirement_id}."
                )
        for focus_area in focus_areas:
            match = matches_by_requirement.get(focus_area.requirement_id)
            if match is not None and focus_area.coverage != match.coverage:
                errors.append(
                    f"Focus area {focus_area.requirement_id} coverage does not match "
                    "the evidence match."
                )

    if interview_strategy is not None:
        strategy_items = interview_strategy.top_priorities
        story_plans = interview_strategy.stories_to_prepare
        risks = interview_strategy.risks_to_address
        errors.extend(
            _validate_requirement_references(
                strategy_items,
                "Strategy",
                requirement_ids,
            )
        )
        errors.extend(
            _validate_requirement_references(
                story_plans,
                "Story plan",
                requirement_ids,
            )
        )
        for risk in risks:
            if risk.requirement_id not in requirement_ids:
                errors.append(
                    "Strategy risk references unknown requirement ID: "
                    f"{risk.requirement_id}."
                )
        errors.extend(
            _validate_evidence_links(
                strategy_items,
                "Strategy",
                matches_by_requirement,
            )
        )
        errors.extend(
            _validate_evidence_links(
                story_plans,
                "Story plan",
                matches_by_requirement,
            )
        )

    if mock_questions:
        errors.extend(
            _validate_requirement_references(
                mock_questions,
                "Mock question",
                requirement_ids,
            )
        )
        errors.extend(
            _validate_evidence_links(
                mock_questions,
                "Mock question",
                matches_by_requirement,
            )
        )

    return errors
