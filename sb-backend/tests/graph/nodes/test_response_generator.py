"""
Unit tests for Response Generator Node.

Tests mock Ollama and GPT calls so no real API or model is used.
"""

import pytest
from unittest.mock import patch, AsyncMock

from graph.state import ConversationState
from graph.nodes.agentic_nodes.response_generator import (
    response_generator_node,
    generate_response_ollama,
    generate_response_gpt,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_state():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I've been feeling really overwhelmed lately.",
        intent="venting",
        situation="GENERAL_OVERWHELM",
        severity="medium",
    )


@pytest.fixture
def crisis_state():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I want to end my life.",
        intent="crisis_disclosure",
        situation="SUICIDAL",
        severity="high",
        is_crisis_detected=True,
    )


@pytest.fixture
def greeting_state():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="student",
        user_message="Hello!",
        intent="greeting",
        is_greeting=True,
    )


@pytest.fixture
def state_without_message():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="",
    )


# ============================================================================
# response_generator_node
# ============================================================================

class TestResponseGeneratorNodeUnit:
    """Unit tests for response_generator_node with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_missing_user_message_returns_error(self, state_without_message):
        result = await response_generator_node(state_without_message)
        assert "error" in result
        assert "Missing user message" in result["error"]

    @pytest.mark.asyncio
    async def test_crisis_state_returns_template_without_calling_llm(self, crisis_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
        ) as mock_ollama, patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
        ) as mock_gpt:
            result = await response_generator_node(crisis_state)
        # LLMs must not be called when a template applies
        mock_ollama.assert_not_called()
        mock_gpt.assert_not_called()
        assert "error" not in result
        assert len(result["response_draft"]) > 0

    @pytest.mark.asyncio
    async def test_greeting_state_returns_template_without_calling_llm(self, greeting_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
        ) as mock_ollama, patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
        ) as mock_gpt:
            result = await response_generator_node(greeting_state)
        mock_ollama.assert_not_called()
        mock_gpt.assert_not_called()
        assert "error" not in result
        assert len(result["response_draft"]) > 0

    @pytest.mark.asyncio
    async def test_returns_response_draft_and_api_response(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
            return_value="Ollama compassionate reply.",
        ), patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
            return_value="GPT compassionate reply.",
        ):
            result = await response_generator_node(sample_state)
        assert "error" not in result
        assert result["response_draft"] in ("GPT compassionate reply.", "Ollama compassionate reply.")
        assert "api_response" in result
        assert result["api_response"]["ollama"] == "Ollama compassionate reply."
        assert result["api_response"]["gpt"] == "GPT compassionate reply."

    @pytest.mark.asyncio
    async def test_api_response_includes_score_fields(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
            return_value="Ollama reply.",
        ), patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
            return_value="GPT reply.",
        ):
            result = await response_generator_node(sample_state)
        api = result["api_response"]
        assert "selected_source" in api
        assert api["selected_source"] in ("ollama", "gpt")
        assert "ollama_score" in api
        assert "gpt_score" in api
        assert isinstance(api["ollama_score"], float)
        assert isinstance(api["gpt_score"], float)

    @pytest.mark.asyncio
    async def test_falls_back_to_ollama_when_gpt_empty(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
            return_value="Ollama fallback",
        ), patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await response_generator_node(sample_state)
        assert result["response_draft"] == "Ollama fallback"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ), patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await response_generator_node(sample_state)
        assert "error" in result
        assert "Error generating response" in result["error"]


# ============================================================================
# generate_response_ollama (mocked aiohttp)
# ============================================================================

class TestGenerateResponseOllamaUnit:
    """Unit tests for generate_response_ollama with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_successful_call_returns_response(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"response": "Here is support."})
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.post.return_value = mock_resp
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await generate_response_ollama("I need help")
        assert result == "Here is support."

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Server error")
        mock_resp.json = AsyncMock(side_effect=Exception("not json"))
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.post.return_value = mock_resp
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await generate_response_ollama("I need help")
        assert result == ""


# ============================================================================
# generate_response_gpt (mocked aiohttp, no API key)
# ============================================================================

class TestGenerateResponseGptUnit:
    """Unit tests for generate_response_gpt."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        with patch("graph.nodes.agentic_nodes.response_generator.OPENAI_API_KEY", ""):
            result = await generate_response_gpt("Hello")
        assert result == ""
