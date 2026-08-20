"""Runtime adapters for deterministic and live Lesson 7 experiments."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from interview_prep import agent, nodes

from .scenarios import PROJECT_ROOT

REQUIREMENTS_PATH = PROJECT_ROOT / "data" / "expected_requirements.json"
COMPLETE_MATCHES_PATH = PROJECT_ROOT / "data" / "expected_evidence_matches.json"


def _response_properties(kwargs: dict[str, Any]) -> set[str]:
    return set(kwargs["config"].response_json_schema["properties"])


def _json_between(text: str, start: str, end: str | None = None) -> Any:
    fragment = text.split(start, 1)[1]
    if end is not None:
        fragment = fragment.split(end, 1)[0]
    return json.loads(fragment.strip())


def _round_payload(contents: str) -> dict[str, Any]:
    lowered = contents.casefold()
    if "cross-functional panel" in lowered:
        return {
            "round_type": "cross-functional panel",
            "format": None,
            "interviewer_roles": ["Product Manager", "Engineering Lead"],
            "focus": ["stakeholder alignment", "concise recommendations"],
            "notes": None,
        }
    if "analytics case" in lowered:
        return {
            "round_type": "analytics case",
            "format": "60-minute live case",
            "interviewer_roles": ["Hiring Manager"],
            "focus": ["hypothesis testing", "trade-offs"],
            "notes": None,
        }
    return {
        "round_type": "hiring-manager discussion",
        "format": None,
        "interviewer_roles": ["Hiring Manager"],
        "focus": ["technical depth", "honest evidence gaps"],
        "notes": None,
    }


def _imperfect_matches(candidate_evidence: list[dict[str, Any]]) -> dict[str, Any]:
    clarification_ids = {
        item["source"].rsplit("/", 1)[-1].strip(): item["evidence_id"]
        for item in candidate_evidence
        if item["source"].startswith("Candidate clarification /")
    }
    base = {
        "REQ-01": ("FULL", ["EXP-01"]),
        "REQ-02": ("GAP", []),
        "REQ-03": ("GAP", []),
        "REQ-04": ("GAP", []),
        "REQ-05": ("FULL", ["EXP-05"]),
        "REQ-06": ("PARTIAL", ["EXP-05"]),
        "REQ-07": ("FULL", ["EXP-10"]),
        "REQ-08": ("PARTIAL", ["EXP-06"]),
    }
    for requirement_id, evidence_id in clarification_ids.items():
        if requirement_id in {"REQ-02", "REQ-03", "REQ-04"}:
            base[requirement_id] = ("FULL", [evidence_id])

    return {
        "evidence_matches": [
            {
                "requirement_id": requirement_id,
                "evidence_ids": evidence_ids,
                "coverage": coverage,
                "explanation": (
                    "The supplied evidence deterministically establishes this "
                    "coverage result for the Lesson 7 fixture."
                ),
                "confidence": 0.99,
            }
            for requirement_id, (coverage, evidence_ids) in base.items()
        ]
    }


def _strategy_payload(contents: str) -> dict[str, Any]:
    matches = _json_between(
        contents,
        "EVIDENCE MATCHES\n",
        "\n\nFOCUS AREAS",
    )
    round_context = _json_between(
        contents,
        "TARGET INTERVIEW ROUND\n",
        "\n\nREQUIREMENTS",
    )
    label = round_context.get("round_type") if round_context else "general interview"
    items = []
    stories = []
    risks = []
    for match in matches:
        requirement_id = match["requirement_id"]
        evidence_ids = match["evidence_ids"]
        items.append(
            {
                "requirement_id": requirement_id,
                "evidence_ids": evidence_ids,
                "preparation_theme": f"Prepare {requirement_id} for the {label}.",
                "rationale": "Use the validated coverage and evidence links.",
            }
        )
        stories.append(
            {
                "requirement_id": requirement_id,
                "evidence_ids": evidence_ids,
                "story_to_prepare": (
                    f"Prepare an honest, evidence-linked response for {requirement_id}."
                ),
            }
        )
        risks.append(
            {
                "requirement_id": requirement_id,
                "risk": f"Coverage for {requirement_id} may require explanation.",
                "mitigation": (
                    "Use only admitted evidence and acknowledge remaining gaps."
                ),
            }
        )
    return {
        "top_priorities": items,
        "positioning_statement": (
            f"Lead with grounded evidence tailored to the {label} while treating "
            "remaining gaps honestly."
        ),
        "stories_to_prepare": stories,
        "risks_to_address": risks,
    }


def _questions_payload(contents: str) -> dict[str, Any]:
    matches = _json_between(
        contents,
        "EVIDENCE MATCHES\n",
        "\n\nINTERVIEW STRATEGY",
    )
    round_context = _json_between(
        contents,
        "TARGET INTERVIEW ROUND\n",
        "\n\nREQUIREMENTS",
    )
    label = round_context.get("round_type") if round_context else "general interview"
    return {
        "mock_questions": [
            {
                "question": (
                    f"In the {label}, how would you address {match['requirement_id']}?"
                ),
                "requirement_id": match["requirement_id"],
                "capability_tested": "Grounded evidence",
                "evidence_ids": match["evidence_ids"],
                "follow_up_probe": (
                    "What concrete evidence or honest gap supports that?"
                ),
                "answer_outline": [
                    "State the relevant evidence or gap.",
                    "Explain the action and result without inventing details.",
                ],
            }
            for match in matches
        ]
    }


class FixtureGeminiClient:
    """Schema-aware deterministic substitute used by the baseline suite."""

    def __init__(
        self,
        *,
        profile: str,
        assessments_by_requirement: dict[str, dict[str, Any]],
    ) -> None:
        self.profile = profile
        self.assessments_by_requirement = assessments_by_requirement
        self.calls: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        properties = _response_properties(kwargs)
        contents = kwargs["contents"]
        if properties == {"requirements"}:
            requirement_fixture = json.loads(
                REQUIREMENTS_PATH.read_text(encoding="utf-8")
            )
            parsed = {"requirements": requirement_fixture["requirements"]}
        elif properties == {"evidence_matches"}:
            if self.profile == "complete":
                parsed = json.loads(COMPLETE_MATCHES_PATH.read_text(encoding="utf-8"))
            else:
                candidate_evidence = _json_between(contents, "CANDIDATE EVIDENCE\n")
                parsed = _imperfect_matches(candidate_evidence)
        elif properties == {
            "round_type",
            "format",
            "interviewer_roles",
            "focus",
            "notes",
        }:
            parsed = _round_payload(contents)
        elif "top_priorities" in properties:
            parsed = _strategy_payload(contents)
        elif properties == {"mock_questions"}:
            parsed = _questions_payload(contents)
        elif properties == {
            "target_requirement_id",
            "is_valid",
            "relevance_reason",
            "specificity_reason",
            "accepted_claim",
        }:
            match = re.search(r'"requirement_id": "(REQ-\d{2})"', contents)
            if match is None:
                raise AssertionError("Assessment prompt omitted a requirement ID.")
            parsed = self.assessments_by_requirement[match.group(1)]
        else:  # pragma: no cover - protects fixtures from schema drift
            raise AssertionError(f"Unexpected Gemini response schema: {properties}")
        return SimpleNamespace(parsed=parsed)


class CountingGeminiClient:
    """Delegate to Gemini while recording every real model request."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.call_count = 0
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.client.models.generate_content(**kwargs)


@contextmanager
def fixture_runtime(
    *,
    profile: str,
    assessments_by_requirement: dict[str, dict[str, Any]],
) -> Iterator[FixtureGeminiClient]:
    """Patch model dependencies while leaving the production graph unchanged."""

    client = FixtureGeminiClient(
        profile=profile,
        assessments_by_requirement=assessments_by_requirement,
    )
    originals = {
        "agent_client": agent.get_gemini_client,
        "agent_model": agent.get_model_name,
        "nodes_client": nodes.get_gemini_client,
        "nodes_model": nodes.get_model_name,
        "candidate_path": nodes.CANDIDATE_PREP_PATH,
    }
    with TemporaryDirectory(prefix="lesson7-baseline-") as temporary_directory:
        agent.get_gemini_client = lambda: client
        agent.get_model_name = lambda: "lesson7-fixture-model"
        nodes.get_gemini_client = lambda: client
        nodes.get_model_name = lambda: "lesson7-fixture-model"
        nodes.CANDIDATE_PREP_PATH = Path(temporary_directory) / "prep-package.md"
        try:
            yield client
        finally:
            agent.get_gemini_client = originals["agent_client"]
            agent.get_model_name = originals["agent_model"]
            nodes.get_gemini_client = originals["nodes_client"]
            nodes.get_model_name = originals["nodes_model"]
            nodes.CANDIDATE_PREP_PATH = originals["candidate_path"]


@contextmanager
def live_runtime() -> Iterator[CountingGeminiClient]:
    """Prevent fallback and package writes during a real-Gemini experiment."""

    original_agent_client = agent.get_gemini_client
    original_nodes_client = nodes.get_gemini_client
    original_fallback = nodes._can_use_evidence_match_fixture
    original_path = nodes.CANDIDATE_PREP_PATH
    client = CountingGeminiClient(original_agent_client())
    with TemporaryDirectory(prefix="lesson7-live-") as temporary_directory:
        agent.get_gemini_client = lambda: client
        nodes.get_gemini_client = lambda: client
        nodes._can_use_evidence_match_fixture = lambda _state: False
        nodes.CANDIDATE_PREP_PATH = Path(temporary_directory) / "prep-package.md"
        try:
            yield client
        finally:
            agent.get_gemini_client = original_agent_client
            nodes.get_gemini_client = original_nodes_client
            nodes._can_use_evidence_match_fixture = original_fallback
            nodes.CANDIDATE_PREP_PATH = original_path
