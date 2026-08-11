"""Grounded prompts for the Gemini-backed workflow nodes."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

EXTRACTION_INSTRUCTIONS = """You extract explicit job requirements for an
interview-preparation workflow.

Rules:
- Extract 6 to 10 distinct requirements that materially affect interview prep.
- Include both required and preferred qualifications when present.
- Assign sequential stable IDs beginning with REQ-01.
- Copy source_quote exactly from the supplied job description.
- Do not invent skills, experience, credentials, or company facts.
- Use importance 5 for explicit core requirements and lower values for
  secondary or preferred qualifications.
- Return only data that conforms to the supplied response schema.
"""


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _dump(value: Any) -> str:
    return json.dumps(_json_value(value), indent=2)


def build_extraction_prompt(job_description: str) -> str:
    """Place the source document after the extraction instructions."""

    return f"""{EXTRACTION_INSTRUCTIONS}

JOB DESCRIPTION
---------------
{job_description}
"""


def build_evidence_matching_prompt(
    requirements: list[Any],
    candidate_evidence: list[Any],
) -> str:
    """Ask Gemini to link supplied evidence without inventing support."""

    return f"""Match the supplied candidate evidence to the job requirements.
Return exactly one EvidenceMatch for every requirement_id, in requirement order.

Coverage rules:
- FULL: supplied evidence directly supports all important parts of the requirement.
- PARTIAL: related supplied evidence exists but misses an important dimension.
- GAP: no supplied evidence supports the requirement.

Use only requirement_id and evidence_id values present below. Every FULL or
PARTIAL match must contain at least one supporting evidence_id. Every GAP must
have an empty evidence_ids list. Judge only the supplied claims; never infer or
invent candidate experience. Keep each explanation concise and evidence-based.
Return only data conforming to the response schema.

REQUIREMENTS
{_dump(requirements)}

CANDIDATE EVIDENCE
{_dump(candidate_evidence)}
"""


def build_strategy_prompt(
    requirements: list[Any],
    evidence_matches: list[Any],
    focus_areas: list[Any],
) -> str:
    """Build a strategy using only validated, ID-linked workflow state."""

    return f"""Create a concise interview-preparation strategy from the supplied
workflow state. Prioritize high-importance PARTIAL and GAP areas while using
FULL matches as candidate stories. Never invent candidate experience. Every
strategy and story item must retain a valid requirement_id and only the
evidence_ids supplied for that requirement. GAP items must use no evidence IDs.
Return only data conforming to the response schema.

REQUIREMENTS
{_dump(requirements)}

EVIDENCE MATCHES
{_dump(evidence_matches)}

FOCUS AREAS
{_dump(focus_areas)}
"""


def build_questions_prompt(
    requirements: list[Any],
    evidence_matches: list[Any],
    strategy: Any,
) -> str:
    """Turn the grounded strategy into structured interview practice."""

    return f"""Generate 8 to 12 realistic mock interview questions grounded in
the supplied strategy. Cover the top priorities and risks. Each question must
retain a valid requirement_id and may use only evidence_ids already linked to
that requirement. GAP questions must use an empty evidence_ids list and coach
an honest answer rather than inventing experience. Provide a useful follow-up
probe and at least two concise answer-outline bullets. Return only data
conforming to the response schema.

REQUIREMENTS
{_dump(requirements)}

EVIDENCE MATCHES
{_dump(evidence_matches)}

INTERVIEW STRATEGY
{_dump(strategy)}
"""
