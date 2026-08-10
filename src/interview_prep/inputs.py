"""Adapters that turn local teaching fixtures into workflow source inputs."""

from __future__ import annotations

from .schemas import CandidateEvidence


def resume_markdown_to_evidence(resume_text: str) -> list[CandidateEvidence]:
    """Convert resume bullets into stable, source-labelled evidence items."""

    evidence: list[CandidateEvidence] = []
    section = "Resume"
    subsection = ""
    bullet_parts: list[str] = []
    bullet_source = section

    def flush_bullet() -> None:
        if not bullet_parts:
            return
        evidence.append(
            CandidateEvidence(
                evidence_id=f"EXP-{len(evidence) + 1:02d}",
                claim=" ".join(bullet_parts),
                source=bullet_source,
            )
        )
        bullet_parts.clear()

    for raw_line in [*resume_text.splitlines(), ""]:
        stripped = raw_line.strip()
        if raw_line.startswith("## "):
            flush_bullet()
            section = stripped.removeprefix("## ")
            subsection = ""
        elif raw_line.startswith("### "):
            flush_bullet()
            subsection = stripped.removeprefix("### ")
        elif raw_line.startswith("- "):
            flush_bullet()
            bullet_source = " / ".join(part for part in (section, subsection) if part)
            bullet_parts.append(raw_line.removeprefix("- ").strip())
        elif bullet_parts and (raw_line.startswith("  ") or stripped):
            bullet_parts.append(stripped)
        else:
            flush_bullet()

    return evidence
