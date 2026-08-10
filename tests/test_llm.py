import os

from interview_prep import config, llm


def test_load_environment_reads_dotenv_without_overriding_shell(
    monkeypatch, tmp_path
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "GEMINI_API_KEY=dotenv-key\nLANGSMITH_TRACING=true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("GEMINI_API_KEY", "shell-key")
    config.load_environment.cache_clear()

    try:
        config.load_environment()
    finally:
        config.load_environment.cache_clear()

    assert os.getenv("GEMINI_API_KEY") == "shell-key"
    assert os.getenv("LANGSMITH_TRACING") == "true"


def test_gemini_client_is_initialized_once(monkeypatch) -> None:
    created_clients = []
    fake_client = object()

    def create_client(*, api_key):
        created_clients.append(api_key)
        return fake_client

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.genai, "Client", create_client)
    monkeypatch.setattr(
        llm.wrappers,
        "wrap_gemini",
        lambda client, **_kwargs: client,
    )
    llm.get_gemini_client.cache_clear()

    try:
        first = llm.get_gemini_client()
        second = llm.get_gemini_client()
    finally:
        llm.get_gemini_client.cache_clear()

    assert first is fake_client
    assert second is fake_client
    assert created_clients == ["test-key"]
