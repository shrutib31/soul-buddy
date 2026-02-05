"""
Unit and Integration Tests for Intent Detection Node

This module contains:
1. Unit tests with mocked Ollama calls (fast, no external dependencies)
2. Integration tests with real Ollama API calls (slower, requires Ollama running)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from graph.state import ConversationState
from graph.nodes.agentic_nodes.intent_detection import (
    intent_detection_node,
    detect_intent_with_ollama,
    get_intent_description,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_state():
    """Create a sample conversation state for testing."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I'm feeling really stressed about my exams"
    )


@pytest.fixture
def empty_message_state():
    """State with empty user message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message=""
    )


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "model": "phi3:latest",
        "created_at": "2026-02-05T10:00:00.000000Z",
        "response": "seek_support",
        "done": True
    }


@pytest.fixture
def mock_ollama_json_response():
    """Mock Ollama response with JSON structure."""
    return {
        "model": "phi3:latest",
        "created_at": "2026-02-05T10:00:00.000000Z",
        "response": '{"intent": "seek_support"}',
        "done": True
    }


# ============================================================================
# UNIT TESTS - MOCKED (Fast, No External Dependencies)
# ============================================================================

class TestIntentDetectionNodeUnit:
    """Unit tests for intent_detection_node with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_empty_message_returns_error(self, empty_message_state):
        """Test that empty message returns an error."""
        result = await intent_detection_node(empty_message_state)
        
        assert "error" in result
        assert "Empty user message" in result["error"]
    
    
    @pytest.mark.asyncio
    async def test_successful_intent_detection(self, sample_state, mock_ollama_response):
        """Test successful intent detection with mocked Ollama."""
        with patch('graph.nodes.agentic_nodes.intent_detection.detect_intent_with_ollama') as mock_detect:
            mock_detect.return_value = "seek_support"
            
            result = await intent_detection_node(sample_state)
            
            assert "intent" in result
            assert result["intent"] == "seek_support"
            assert "error" not in result
            mock_detect.assert_called_once_with(sample_state.user_message)
    
    
    @pytest.mark.asyncio
    async def test_intent_detection_with_exception(self, sample_state):
        """Test that exceptions are handled gracefully."""
        with patch('graph.nodes.agentic_nodes.intent_detection.detect_intent_with_ollama') as mock_detect:
            mock_detect.side_effect = Exception("Connection failed")
            
            result = await intent_detection_node(sample_state)
            
            assert "error" in result
            assert "Error detecting intent" in result["error"]
    
    
    @pytest.mark.asyncio
    async def test_valid_intents_returned(self, sample_state):
        """Test that all valid intents are properly returned."""
        valid_intents = [
            "greeting", "venting", "seek_information", "seek_understanding",
            "open_to_solution", "try_tool", "seek_support", "unclear"
        ]
        
        for intent in valid_intents:
            with patch('graph.nodes.agentic_nodes.intent_detection.detect_intent_with_ollama') as mock_detect:
                mock_detect.return_value = intent
                
                result = await intent_detection_node(sample_state)
                
                assert result["intent"] == intent


class TestDetectIntentWithOllamaUnit:
    """Unit tests for detect_intent_with_ollama with mocked HTTP calls."""
    
    @pytest.mark.asyncio
    async def test_successful_ollama_call(self, mock_ollama_response):
        """Test successful Ollama API call with valid intent."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"response": "seek_support"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await detect_intent_with_ollama("I need help")
            
            assert result in [
                "greeting", "venting", "seek_information", "seek_understanding",
                "open_to_solution", "try_tool", "seek_support", "unclear"
            ]
    
    
    @pytest.mark.asyncio
    async def test_non_200_response_returns_unclear(self):
        """Test that non-200 HTTP response returns 'unclear'."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='{"error": "Internal server error"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await detect_intent_with_ollama("Test message")
            
            assert result == "unclear"
    
    
    @pytest.mark.asyncio
    async def test_invalid_intent_returns_unclear(self):
        """Test that invalid intent from LLM returns 'unclear'."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"response": "invalid_intent_name"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await detect_intent_with_ollama("Test message")
            
            assert result == "unclear"
    
    
    @pytest.mark.asyncio
    async def test_timeout_exception_returns_unclear(self):
        """Test that timeout exception returns 'unclear'."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await detect_intent_with_ollama("Test message")
            
            assert result == "unclear"
    
    
    @pytest.mark.asyncio
    async def test_multiline_response_takes_first_line(self):
        """Test that multiline response extracts first line as intent."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"response": "greeting\\nSome additional text\\nMore text"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await detect_intent_with_ollama("Hello!")
            
            assert result == "greeting"


class TestGetIntentDescription:
    """Unit tests for get_intent_description utility."""
    
    @pytest.mark.asyncio
    async def test_all_intents_have_descriptions(self):
        """Test that all valid intents have descriptions."""
        valid_intents = [
            "greeting", "venting", "seek_information", "seek_understanding",
            "open_to_solution", "try_tool", "seek_support", "unclear"
        ]
        
        for intent in valid_intents:
            description = await get_intent_description(intent)
            assert description is not None
            assert len(description) > 0
            assert isinstance(description, str)
    
    
    @pytest.mark.asyncio
    async def test_unknown_intent_returns_default(self):
        """Test that unknown intent returns default message."""
        description = await get_intent_description("unknown_intent")
        assert description == "Unknown intent"


# ============================================================================
# INTEGRATION TESTS - REAL OLLAMA CALLS (Slower, Requires Ollama)
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestIntentDetectionIntegration:
    """
    Integration tests that make real Ollama API calls.
    
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    
    Requirements:
    - Ollama must be running at OLLAMA_BASE_URL
    - Model OLLAMA_MODEL must be available
    """
    
    async def test_real_ollama_greeting_detection(self):
        """Test real Ollama call with greeting message."""
        message = "Hello! How are you doing today?"
        
        result = await detect_intent_with_ollama(message)
        
        # Should detect as greeting or unclear (acceptable for real LLM)
        assert result in ["greeting", "unclear"]
    
    
    async def test_real_ollama_venting_detection(self):
        """Test real Ollama call with venting message."""
        message = "I'm so frustrated with everything going wrong at work!"
        
        result = await detect_intent_with_ollama(message)
        
        # Should detect as venting, seek_support, or unclear
        assert result in ["venting", "seek_support", "unclear"]
    
    
    async def test_real_ollama_seek_information_detection(self):
        """Test real Ollama call with information-seeking message."""
        message = "What are some good coping strategies for anxiety?"
        
        result = await detect_intent_with_ollama(message)
        
        # Should detect as seek_information or related intents
        assert result in ["seek_information", "seek_understanding", "unclear"]
    
    
    async def test_real_ollama_seek_support_detection(self):
        """Test real Ollama call with support-seeking message."""
        message = "I'm really struggling and don't know what to do"
        
        result = await detect_intent_with_ollama(message)
        
        # Should detect as seek_support or related intents
        assert result in ["seek_support", "venting", "unclear"]
    
    
    async def test_real_ollama_with_full_node(self):
        """Test full intent_detection_node with real Ollama."""
        state = ConversationState(
            conversation_id="test-integration-123",
            mode="incognito",
            domain="general",
            user_message="I need help managing my stress levels"
        )
        
        result = await intent_detection_node(state)
        
        # Should return intent without error
        assert "intent" in result
        assert "error" not in result
        assert result["intent"] in [
            "greeting", "venting", "seek_information", "seek_understanding",
            "open_to_solution", "try_tool", "seek_support", "unclear"
        ]
    
    
    async def test_real_ollama_timeout_handling(self):
        """Test that timeout is properly configured (should not timeout under normal conditions)."""
        import time
        
        message = "This is a test message for timeout handling"
        start_time = time.time()
        
        result = await detect_intent_with_ollama(message)
        
        elapsed_time = time.time() - start_time
        
        # Should complete within timeout (120s) and return valid result
        assert elapsed_time < OLLAMA_TIMEOUT
        assert result in [
            "greeting", "venting", "seek_information", "seek_understanding",
            "open_to_solution", "try_tool", "seek_support", "unclear"
        ]


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("message,expected_intents", [
    ("Hello", ["greeting", "unclear"]),
    ("I'm so angry right now!", ["venting", "seek_support", "unclear"]),
    ("How do I deal with stress?", ["seek_information", "seek_understanding", "unclear"]),
    ("I need someone to talk to", ["seek_support", "unclear"]),
    ("What are my options?", ["seek_information", "open_to_solution", "unclear"]),
])
@pytest.mark.asyncio
async def test_various_messages_unit(message, expected_intents):
    """Parametrized test for various message types with mocked Ollama."""
    with patch('graph.nodes.agentic_nodes.intent_detection.detect_intent_with_ollama') as mock_detect:
        # Mock to return one of the expected intents
        mock_detect.return_value = expected_intents[0]
        
        state = ConversationState(
            conversation_id="test-123",
            mode="incognito",
            domain="general",
            user_message=message
        )
        
        result = await intent_detection_node(state)
        
        assert "intent" in result
        assert result["intent"] in expected_intents


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
        # Should be at least 30 seconds for LLM inference
        assert OLLAMA_TIMEOUT >= 30


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "not integration"])
