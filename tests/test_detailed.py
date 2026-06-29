from unittest.mock import patch, MagicMock


def _mock_completion(content):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


@patch("chat.service.OpenAI")
def test_deep_analyze_papers_single(mock_openai_cls):
    from chat.service import _deep_analyze_papers

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_completion(
        "## Detailed Analysis\nIPCW uses $$w_i = 1/P(...)$$"
    )

    papers_sections = {
        "Paper A (2023)": [
            {"section_type": "methodology", "content": "We propose IPCW weights..."},
        ],
    }

    result = _deep_analyze_papers(
        question="How does IPCW work?",
        papers_sections=papers_sections,
        api_key="fake-key",
    )

    assert len(result) == 1
    assert "Paper A (2023)" in result
    assert "IPCW" in result["Paper A (2023)"]


@patch("chat.service.OpenAI")
def test_deep_analyze_papers_empty(mock_openai_cls):
    from chat.service import _deep_analyze_papers

    result = _deep_analyze_papers(
        question="test",
        papers_sections={},
        api_key="fake-key",
    )

    assert result == {}
    assert not mock_openai_cls.called
