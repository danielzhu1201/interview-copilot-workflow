"""Candidate-facing Markdown rendering for validated preparation packages."""

from __future__ import annotations

from pathlib import Path

from .schemas import PrepPackage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PREP_PATH = PROJECT_ROOT / "output" / "interview-prep-package.md"


def render_candidate_prep(package: PrepPackage) -> str:
    """Render only information the candidate needs for interview preparation."""

    requirements_by_id = {
        item.requirement_id: item.requirement for item in package.requirements
    }
    lines = [
        "# Next-Round Interview Prep",
        "",
    ]
    interview_round = package.interview_round
    if interview_round is not None:
        lines.extend(["## Target interview round", ""])
        if interview_round.round_type:
            lines.append(f"- Round type: {interview_round.round_type}")
        if interview_round.format:
            lines.append(f"- Format: {interview_round.format}")
        if interview_round.interviewer_roles:
            lines.append(
                "- Interviewer roles: " + ", ".join(interview_round.interviewer_roles)
            )
        if interview_round.focus:
            lines.append("- Focus: " + ", ".join(interview_round.focus))
        if interview_round.notes:
            lines.append(f"- Notes: {interview_round.notes}")
        lines.append("")

    lines.extend(
        [
            "## Positioning",
            "",
            package.interview_strategy.positioning_statement,
            "",
            "## What to prioritize",
            "",
        ]
    )

    for focus_area in package.focus_areas:
        requirement = requirements_by_id[focus_area.requirement_id]
        lines.extend(
            [
                f"### {requirement}",
                "",
                f"- Current coverage: {focus_area.coverage.title()}",
                f"- Preparation action: {focus_area.preparation_action}",
                f"- Why: {focus_area.reason}",
                "",
            ]
        )

    lines.extend(["## Themes to lead with", ""])
    for item in package.interview_strategy.top_priorities:
        lines.extend(
            [
                f"- {item.preparation_theme} - {item.rationale}",
            ]
        )

    lines.extend(["", "## Stories to prepare", ""])
    for story in package.interview_strategy.stories_to_prepare:
        lines.extend([f"- {story.story_to_prepare}"])

    lines.extend(["", "## Risks to address", ""])
    for risk in package.interview_strategy.risks_to_address:
        lines.extend(
            [
                f"- Risk: {risk.risk}",
                f"  Plan: {risk.mitigation}",
            ]
        )

    lines.extend(["", "## Practice questions", ""])
    for index, question in enumerate(package.mock_questions, start=1):
        lines.extend(
            [
                f"### {index}. {question.question}",
                "",
                f"Capability to show: {question.capability_tested}",
                "",
                f"Follow-up to prepare for: {question.follow_up_probe}",
                "",
                "Answer outline:",
                *[f"- {point}" for point in question.answer_outline],
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_candidate_prep(
    package: PrepPackage,
    destination: Path = CANDIDATE_PREP_PATH,
) -> Path:
    """Write the candidate-facing preparation brief after package validation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_candidate_prep(package), encoding="utf-8")
    return destination
