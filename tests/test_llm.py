from interview_prep import llm


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
