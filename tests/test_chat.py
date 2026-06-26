from unittest.mock import patch, MagicMock
from conftest import create_user, login_user, auth_header


def _mock_openai_response(content="Mocked LLM response"):
    """Create a mock that mimics OpenAI chat completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _setup_user(client):
    """Create an admin user and return auth headers."""
    create_user(client, "chatuser", "pass")
    token = login_user(client, "chatuser", "pass")
    return auth_header(token)


def test_chat_requires_auth(client):
    res = client.post("/chat", json={"message": "test"})
    assert res.status_code in (401, 403)


@patch("chat.router.OpenAI")
@patch("chat.service.OpenAI")
def test_chat_basic(mock_openai_cls, mock_router_openai_cls, client):
    headers = _setup_user(client)
    # Mock the router's OpenAI to return a valid classification
    mock_router_client = MagicMock()
    mock_router_client.chat.completions.create.return_value = _mock_openai_response(
        '{"skill": "general_stats", "search_queries": ["binary outcome regression"]}'
    )
    mock_router_openai_cls.return_value = mock_router_client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response("Use logistic regression.")
    # Mock Responses API (web search path for llm_only)
    mock_responses_resp = MagicMock()
    mock_responses_resp.output_text = "Use logistic regression."
    mock_client.responses.create.return_value = mock_responses_resp
    mock_openai_cls.return_value = mock_client

    res = client.post(
        "/chat",
        json={"message": "I have binary outcome"},
        headers={**headers, "x-session-id": "test-session-1"},
    )

    assert res.status_code == 200
    assert "Use logistic regression." in res.json()["response"]

@patch("chat.service._rewrite_query", side_effect=lambda msg, *a, **kw: msg)
@patch("chat.router.OpenAI")
@patch("chat.service.OpenAI")
def test_chat_sends_history(mock_openai_cls, mock_router_openai_cls, mock_rewrite, client):
    """Second message in the same session should include history."""
    headers = _setup_user(client)
    # Mock the router's OpenAI to return a valid classification
    mock_router_client = MagicMock()
    mock_router_client.chat.completions.create.return_value = _mock_openai_response(
        '{"skill": "general_stats", "search_queries": ["binary outcome regression"]}'
    )
    mock_router_openai_cls.return_value = mock_router_client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response("Answer 1")
    mock_responses_resp = MagicMock()
    mock_responses_resp.output_text = "Answer 1"
    mock_client.responses.create.return_value = mock_responses_resp
    mock_openai_cls.return_value = mock_client

    # First message
    client.post(
        "/chat",
        json={"message": "Hello"},
        headers={**headers, "x-session-id": "sess-history"},
    )

    mock_client.chat.completions.create.return_value = _mock_openai_response("Answer 2")
    mock_responses_resp2 = MagicMock()
    mock_responses_resp2.output_text = "Answer 2"
    mock_client.responses.create.return_value = mock_responses_resp2

    # Second message — should carry history
    res = client.post(
        "/chat",
        json={"message": "Follow-up"},
        headers={**headers, "x-session-id": "sess-history"},
    )

    assert res.status_code == 200
    assert "Answer 2" in res.json()["response"]

    # Verify history was passed to OpenAI (via responses.create for web-enhanced path)
    call_args = mock_client.responses.create.call_args
    messages = call_args.kwargs.get("input") or call_args[1].get("input")
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert "user" in roles[1:-1]
    assert roles[-1] == "user"
    assert "Follow-up" in messages[-1]["content"]
