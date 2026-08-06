"""Prompt used by the in-class extraction node."""

from __future__ import annotations

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


def build_extraction_prompt(job_description: str) -> str:
    """Place the source document after the extraction instructions."""

    return f"""{EXTRACTION_INSTRUCTIONS}

JOB DESCRIPTION
---------------
{job_description}
"""
