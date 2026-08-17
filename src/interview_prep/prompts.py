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


def build_interview_round_parsing_prompt(interview_round_text: str) -> str:
    """Parse optional freeform round notes without inventing missing details."""

    return f"""Parse the user's freeform description of their next interview round.
Extract only details explicitly present in the text. Do not infer missing facts.
Leave optional strings null and optional lists empty when the user did not
provide them. Return only data conforming to the response schema.

INTERVIEW ROUND DESCRIPTION
{interview_round_text}
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

Experimentation safeguard:
- A requirement for hands-on experiment design and statistical interpretation
  needs direct evidence of designing or analyzing an experiment,
  quasi-experiment, causal study, or statistical test.
- Forming hypotheses, defining event tracking, or making a recommendation from
  observational analysis does not by itself support that experimentation
  requirement. Use GAP when those proxy activities are the only related claims.

Technical-skill safeguards:
- Advanced SQL requires a direct claim of using SQL; dashboards, metrics, a
  cloud warehouse, or BI tools alone do not support it.
- Python proficiency requires a direct claim of using Python; automation,
  Airflow, forecasting, or basic automated checks alone do not support it.
- Use GAP when only these adjacent tools or activities are supplied.

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
    interview_round: Any,
) -> str:
    """Build a strategy using only validated, ID-linked workflow state."""

    return f"""Create a concise interview-preparation strategy from the supplied
workflow state. Prioritize high-importance PARTIAL and GAP areas while using
FULL matches as candidate stories. Never invent candidate experience. Every
strategy and story item must retain a valid requirement_id and only the
evidence_ids supplied for that requirement. GAP items must use no evidence IDs.
When target interview round context is supplied, tailor emphasis, communication
style, and preparation advice to it. When it is null, create general interview
preparation without inventing round details. Round context changes preparation,
never candidate evidence.
Return only data conforming to the response schema.

TARGET INTERVIEW ROUND
{_dump(interview_round)}

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
    interview_round: Any,
) -> str:
    """Turn the grounded strategy into structured interview practice."""

    return f"""Generate 8 to 12 realistic mock interview questions grounded in
the supplied strategy. Cover the top priorities and risks. Each question must
retain a valid requirement_id and may use only evidence_ids already linked to
that requirement. GAP questions must use an empty evidence_ids list and coach
an honest answer rather than inventing experience. Provide a useful follow-up
probe and at least two concise answer-outline bullets. Return only data
conforming to the response schema. When target interview round context is
supplied, tailor question format, emphasis, and follow-up probes to it. When it
is null, create general practice questions without inventing round details.
Round context changes preparation, never candidate evidence.

TARGET INTERVIEW ROUND
{_dump(interview_round)}

REQUIREMENTS
{_dump(requirements)}

EVIDENCE MATCHES
{_dump(evidence_matches)}

INTERVIEW STRATEGY
{_dump(strategy)}
"""


# =============================================================================
# LESSON 6 SHORT-CONTEXT EVIDENCE ASSESSMENT
# =============================================================================


def build_clarification_assessment_prompt(
    *,
    requirement: Any,
    question: str,
    answer: str,
) -> str:
    """Assess one resumed answer without exposing the full workflow context."""

    return f"""Assess whether the candidate's answer may be admitted as evidence
for exactly one job requirement. Return only data conforming to the schema.

Validation rubric:
- target_requirement_id must exactly match the supplied requirement ID.
- is_valid may be true only when the answer directly addresses the requirement
  with a concrete first-person claim about what the candidate actually did.
- Reject vague interest, plans to learn, unsupported self-ratings, unrelated
  experience, and answers that merely repeat the requirement.
- accepted_claim must be a concise, faithful restatement of facts explicitly in
  the answer. Never strengthen, infer, or invent a claim.
- When is_valid is false, accepted_claim must be null.

TARGET REQUIREMENT
{_dump(requirement)}

QUESTION
{question}

CANDIDATE ANSWER
{answer}
"""
