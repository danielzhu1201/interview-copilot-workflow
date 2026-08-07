import json
from pathlib import Path
from types import SimpleNamespace

from interview_prep.nodes import extract_requirements

ROOT = Path(__file__).resolve().parents[1]


def test_extract_requirements_requests_a_json_schema_requirement_list(
    monkeypatch,
) -> None:
    payload = json.loads(
        (ROOT / "data" / "expected_requirements.json").read_text(encoding="utf-8")
    )
    captured: dict[str, object] = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed={"requirements": payload["requirements"]})

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr("interview_prep.nodes.get_gemini_client", lambda: fake_client)
    monkeypatch.setattr("interview_prep.nodes.get_model_name", lambda: "test-model")

    result = extract_requirements({"job_description": "Example JD", "resume_text": ""})

    config = captured["config"]
    assert result["requirements"][0].requirement_id == "REQ-01"
    assert config.response_schema is None
    assert set(config.response_json_schema["properties"]) == {"requirements"}
