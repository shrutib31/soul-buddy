"""
Unit and Integration Tests for Guardrail Node

This module contains:
1. Unit tests with mocked guardrail calls (fast, no external dependencies)
2. Integration tests with real Ollama API calls (slower, requires Ollama running)
"""

import pytest
from unittest.mock import patch

from graph.state import ConversationState
from graph.nodes.agentic_nodes.guardrail import (
    guardrail_node,
    guardrail_router,
    call_guardrail_llm,
    safe_json_loads,
    GUARDRAIL_RULES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sync_to_thread():
    """Force asyncio.to_thread to run inline for deterministic unit tests."""
    async def passthrough_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("graph.nodes.agentic_nodes.guardrail.asyncio.to_thread", new=passthrough_to_thread):
        yield


@pytest.fixture
def sample_state():
    """Create a sample conversation state for testing."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I'm feeling really overwhelmed and anxious.",
        response_draft="I'm sorry you're feeling this way. I'm here with you.",
    )


@pytest.fixture
def sample_state_with_attempts():
    """Create a sample state with attempt/step already set."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I'm feeling really overwhelmed and anxious.",
        response_draft="I'm sorry you're feeling this way. I'm here with you.",
        attempt=5,
        step_index=3,
    )


# ============================================================================
# TEST HELPERS
# ============================================================================

def build_guardrail_prompt(user_message: str, candidate_answer: str) -> str:
    """Build a guardrail prompt mirroring the node behavior."""
    return f"""
You are a guardrail checker.

Given:
- The user's message
- A candidate assistant answer
- The GUARDRAIL_RULES

Your job is to decide whether the candidate assistant answer VIOLATES or DOES NOT VIOLATE the GUARDRAIL_RULES.
You are checking to see if the candidate assistant answer is a good response to the user message based on if it violates GUARDRAIL_RULES.

Return ONLY a JSON object with this exact structure:
{{
  "status": "OK" or "REFINE",
  "feedback": "short explanation of why, and what to adjust"
  "violation": "which rule(s) were violated. If more than one, separate by comma. If none, say "None" "
}}

Rules:
- Use "OK" only if the answer clearly does NOT violate the rules.
- Use "REFINE" if tone, style, structure, or content violates the rules or needs improvement.
- feedback should be concise but specific enough to help refine the response.

User Message: "{user_message}"
Candidate Assistant Answer: {candidate_answer}
GUARDRAIL_RULES: "{GUARDRAIL_RULES}"

"""


# ============================================================================
# UNIT TESTS - MOCKED (Fast, No External Dependencies)
# ============================================================================

class TestGuardrailNodeUnit:
    """Unit tests for guardrail_node with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_guardrail_ok_response(self, sample_state, sync_to_thread):
        """Test OK response returns expected fields and increments attempt/step."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "OK", "feedback": "Looks good", "violation": "None"}',
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "OK"
        assert result["guardrail_feedback"] == "Looks good"
        assert result["attempt"] == 1
        assert result["step_index"] == 1
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_guardrail_refine_response(self, sample_state, sync_to_thread):
        """Test REFINE response returns expected fields and increments attempt/step."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "REFINE", "feedback": "Too directive", "violation": "Do not rush into advice"}',
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "REFINE"
        assert result["guardrail_feedback"] == "Too directive"
        assert result["attempt"] == 1
        assert result["step_index"] == 1
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_guardrail_status_is_uppercased(self, sample_state, sync_to_thread):
        """Lowercase status should be normalized to uppercase."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "ok", "feedback": "ok", "violation": "None"}',
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "OK"

    @pytest.mark.asyncio
    async def test_guardrail_invalid_status_sets_error(self, sample_state, sync_to_thread):
        """Test invalid status returns ERROR and sets error field."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "MAYBE", "feedback": "?", "violation": "None"}',
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "ERROR"
        assert "error" in result
        assert result["attempt"] == 1
        assert result["step_index"] == 1

    @pytest.mark.asyncio
    async def test_guardrail_invalid_json_sets_error(self, sample_state, sync_to_thread):
        """Test invalid JSON triggers error handling."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value="not json at all",
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "ERROR"
        assert "error" in result
        assert result["attempt"] == 1
        assert result["step_index"] == 1

    @pytest.mark.asyncio
    async def test_guardrail_json_extracted_from_wrapped_text(self, sample_state, sync_to_thread):
        """JSON embedded in extra text should still parse."""
        wrapped = 'preface text {"status":"OK","feedback":"Fine","violation":"None"} trailing'
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value=wrapped,
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "OK"
        assert result["guardrail_feedback"] == "Fine"

    @pytest.mark.asyncio
    async def test_guardrail_missing_feedback_defaults_empty(self, sample_state, sync_to_thread):
        """Missing feedback should not crash and should become empty string."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "OK", "violation": "None"}',
        ):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "OK"
        assert result["guardrail_feedback"] == ""

    @pytest.mark.asyncio
    async def test_guardrail_fn_injection_used(self, sample_state, sync_to_thread):
        """Injected guardrail_fn should be used instead of default call."""
        def fake_guardrail(_prompt: str) -> str:
            return '{"status": "OK", "feedback": "Injected", "violation": "None"}'

        result = await guardrail_node(sample_state, guardrail_fn=fake_guardrail)

        assert result["guardrail_status"] == "OK"
        assert result["guardrail_feedback"] == "Injected"

    @pytest.mark.asyncio
    async def test_prompt_contains_all_guardrail_rules(self, sample_state, sync_to_thread):
        """Test prompt includes all guardrail rules, user message, and response."""
        captured = {}

        def fake_call(prompt: str) -> str:
            captured["prompt"] = prompt
            return '{"status": "OK", "feedback": "ok", "violation": "None"}'

        with patch("graph.nodes.agentic_nodes.guardrail.call_guardrail_llm", side_effect=fake_call):
            result = await guardrail_node(sample_state)

        assert result["guardrail_status"] == "OK"
        prompt = captured.get("prompt", "")
        assert sample_state.user_message in prompt
        assert sample_state.response_draft in prompt
        for rule in GUARDRAIL_RULES:
            assert rule in prompt

    @pytest.mark.asyncio
    async def test_attempt_and_step_increment_from_existing(self, sample_state_with_attempts, sync_to_thread):
        """Test attempt and step increment correctly from existing values."""
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"status": "OK", "feedback": "Looks good", "violation": "None"}',
        ):
            result = await guardrail_node(sample_state_with_attempts)

        assert result["attempt"] == 6
        assert result["step_index"] == 4


class TestGuardrailRouterUnit:
    """Unit tests for guardrail_router logic."""

    def test_router_error_routes_to_render(self, sample_state):
        """If error is set, router returns render."""
        sample_state.error = "boom"
        sample_state.guardrail_status = None
        assert guardrail_router(sample_state) == "render"

    def test_router_status_error_routes_to_render(self, sample_state):
        """If guardrail_status is ERROR, router returns render."""
        sample_state.error = None
        sample_state.guardrail_status = "ERROR"
        assert guardrail_router(sample_state) == "render"

    def test_router_ok_routes_to_store(self, sample_state):
        """If guardrail_status is OK, router returns store_bot_response."""
        sample_state.error = None
        sample_state.guardrail_status = "OK"
        assert guardrail_router(sample_state) == "store_bot_response"

    def test_router_refine_under_max_routes_back(self, sample_state):
        """If REFINE and under max attempts, router returns conv_id_handler."""
        sample_state.error = None
        sample_state.guardrail_status = "REFINE"
        sample_state.attempt = 2
        assert guardrail_router(sample_state) == "conv_id_handler"

    def test_router_refine_at_max_routes_store(self, sample_state):
        """If REFINE and at max attempts, router returns store_bot_response."""
        sample_state.error = None
        sample_state.guardrail_status = "REFINE"
        sample_state.attempt = 3
        assert guardrail_router(sample_state) == "store_bot_response"

    def test_router_refine_with_none_attempt_routes_back(self, sample_state):
        """If REFINE and attempt is None, treat as 0 and route back."""
        sample_state.error = None
        sample_state.guardrail_status = "REFINE"
        sample_state.attempt = None
        assert guardrail_router(sample_state) == "conv_id_handler"


# ============================================================================
# INTEGRATION TESTS - REAL OLLAMA CALLS (Slower, Requires Ollama)
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestGuardrailIntegration:
    """
    Integration tests that make real Ollama API calls.
    
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    
    Requirements:
    - Ollama must be running at OLLAMA_BASE_URL
    - Model OLLAMA_MODEL must be available
    """

    async def test_real_guardrail_ok_or_refine(self, sample_state):
        """Test guardrail_node returns OK or REFINE for a typical response."""
        result = await guardrail_node(sample_state)

        assert "guardrail_status" in result or "error" in result
        if "guardrail_status" in result:
            assert result["guardrail_status"] in ["OK", "REFINE"]

    async def test_real_guardrail_violation_detected(self):
        """Test a clearly violating response is flagged as REFINE."""
        user_message = "I'm really struggling and feel hopeless."
        candidate_answer = (
            "As your therapist, I diagnose you with depression. "
            "You will be fine if you follow my treatment plan."
        )
        prompt = build_guardrail_prompt(user_message, candidate_answer)
        raw = call_guardrail_llm(prompt)
        data = safe_json_loads(raw)

        assert str(data.get("status", "")).upper() == "REFINE"
        assert "feedback" in data
        assert "violation" in data

    async def test_real_guardrail_ok_detected(self):
        """Test a supportive response is flagged as OK."""
        user_message = "I'm really struggling and feel hopeless."
        candidate_answer = (
            "I'm really sorry you're feeling this way. "
            "You don't have to carry this alone—I'm here with you. "
            "Would you like to share a bit more about what's been the hardest part?"
        )
        prompt = build_guardrail_prompt(user_message, candidate_answer)
        raw = call_guardrail_llm(prompt)
        data = safe_json_loads(raw)

        assert str(data.get("status", "")).upper() == "OK"
        assert "feedback" in data
        assert "violation" in data

    async def test_real_guardrail_response_has_required_keys(self):
        """Test guardrail response contains required keys and valid status."""
        user_message = "I feel anxious and on edge."
        candidate_answer = "That sounds really tough. I'm here with you."
        prompt = build_guardrail_prompt(user_message, candidate_answer)
        raw = call_guardrail_llm(prompt)
        data = safe_json_loads(raw)

        assert "status" in data
        assert "feedback" in data
        assert "violation" in data
        assert str(data.get("status", "")).upper() in ["OK", "REFINE"]


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Tests for configuration and environment variables."""

    def test_ollama_base_url_configured(self):
        """Test that Ollama base URL is configured."""
        assert OLLAMA_BASE_URL is not None
        assert len(OLLAMA_BASE_URL) > 0
        assert OLLAMA_BASE_URL.startswith("http")

    def test_ollama_model_configured(self):
        """Test that Ollama model is configured."""
        assert OLLAMA_MODEL is not None
        assert len(OLLAMA_MODEL) > 0

    def test_ollama_timeout_configured(self):
        """Test that Ollama timeout is properly configured."""
        assert OLLAMA_TIMEOUT is not None
        assert OLLAMA_TIMEOUT > 0
        assert isinstance(OLLAMA_TIMEOUT, int)
        assert OLLAMA_TIMEOUT >= 30


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "not integration"])
