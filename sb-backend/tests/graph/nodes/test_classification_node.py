"""
Unit tests for Classification Node (classification_node, get_classifications, detect_crisis, detect_greeting).

Tests use mocks to avoid loading the real transformer model. Crisis detection and greeting
logic are tested directly; get_classifications/classification_node are tested with get_classifications mocked.
"""

import pytest
from unittest.mock import patch, MagicMock

from graph.state import ConversationState
from graph.nodes.agentic_nodes.classification_node import (
    classification_node,
    get_classifications,
    detect_crisis,
    detect_greeting,
    is_true_negation,
    INTENT_LABELS,
    SITUATION_LABELS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_state():
    """Conversation state with a non-empty user message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I've been feeling really stressed lately",
    )


@pytest.fixture
def empty_message_state():
    """State with empty user message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="",
    )


@pytest.fixture
def greeting_state():
    """State with a greeting message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="Hi there!",
    )


# ============================================================================
# detect_greeting
# ============================================================================

class TestDetectGreeting:
    """Unit tests for detect_greeting."""

    def test_simple_greeting_hi(self):
        assert detect_greeting("hi") is True
        assert detect_greeting("Hi") is True
        assert detect_greeting("HI") is True

    def test_simple_greeting_hello_hey(self):
        assert detect_greeting("hello") is True
        assert detect_greeting("hey") is True
        assert detect_greeting("Hello!") is True

    def test_time_greetings(self):
        assert detect_greeting("good morning") is True
        assert detect_greeting("good afternoon") is True
        assert detect_greeting("good evening") is True

    def test_short_greeting_with_word(self):
        assert detect_greeting("hi there") is True
        assert detect_greeting("hello world") is True

    def test_not_greeting_long_message(self):
        assert detect_greeting("I said hi to my friend") is False
        assert detect_greeting("hello i need help with something serious") is False

    def test_empty_or_whitespace(self):
        assert detect_greeting("") is False
        assert detect_greeting("   ") is False


# ============================================================================
# is_true_negation
# ============================================================================

class TestIsTrueNegation:
    """Unit tests for is_true_negation."""

    def test_negation_not_going_to_kill_myself(self):
        assert is_true_negation("i'm not going to kill myself") is True

    def test_negation_not_suicidal(self):
        assert is_true_negation("i am not suicidal") is True
        assert is_true_negation("i'm not suicidal") is True

    def test_no_negation_suicidal_ideation(self):
        assert is_true_negation("i want to kill myself") is False
        assert is_true_negation("i feel suicidal") is False

    def test_empty(self):
        assert is_true_negation("") is False


# ============================================================================
# detect_crisis
# ============================================================================

class TestDetectCrisis:
    """Unit tests for detect_crisis (no model, pattern-based)."""

    def test_invalid_message_returns_not_crisis(self):
        out = detect_crisis("")
        assert out["is_crisis"] is False
        assert out["intent"] == "unclear"
        assert out["situation"] == "NO_SITUATION"

        out = detect_crisis(None)
        assert out["is_crisis"] is False

    def test_non_string_returns_not_crisis(self):
        out = detect_crisis(123)
        assert out["is_crisis"] is False

    def test_true_negation_returns_not_crisis(self):
        out = detect_crisis("I am not suicidal at all")
        assert out["is_crisis"] is False

    def test_suicidal_ideation_detected(self):
        out = detect_crisis("I want to kill myself")
        assert out["is_crisis"] is True
        assert out["intent"] == "crisis_disclosure"
        assert out["situation"] == "SUICIDAL"
        assert out["severity"] == "high"

    def test_self_harm_detected(self):
        out = detect_crisis("I've been cutting myself to cope")
        assert out["is_crisis"] is True
        assert out["situation"] == "SELF_HARM"

    def test_normal_message_not_crisis(self):
        out = detect_crisis("I'm a bit stressed about exams")
        assert out["is_crisis"] is False
        assert out["situation"] == "NO_SITUATION" or out["situation"] == "NO_SITUATION"


# ============================================================================
# get_classifications (mocked model)
# ============================================================================

class TestGetClassificationsUnit:
    """Unit tests for get_classifications with mocked model load."""

    def test_empty_message_returns_unclear(self):
        out = get_classifications("")
        assert out["intent"] == "unclear"
        assert out["situation"] == "unclear"
        assert out["severity"] == "low"
        assert out["risk_score"] == 0.0
        assert out["risk_level"] == "low"

    def test_whitespace_only_returns_unclear(self):
        out = get_classifications("   \n\t  ")
        assert out["intent"] == "unclear"

    def test_greeting_returns_greeting_intent(self):
        out = get_classifications("Hello!")
        assert out["intent"] == "greeting"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"

    def test_crisis_message_returns_crisis_classification(self):
        out = get_classifications("I want to kill myself")
        assert out["intent"] == "crisis_disclosure"
        assert out["situation"] == "SUICIDAL"
        assert out["severity"] == "high"
        assert out["risk_level"] in ("high", "critical")

    def test_model_path_requires_loaded_model(self):
        """Without model loaded, get_classifications for non-greeting non-crisis raises RuntimeError."""
        import graph.nodes.agentic_nodes.classification_node as mod
        orig_loaded, orig_model, orig_tok = mod._model_loaded, mod._model, mod._tokenizer
        mod._model_loaded = False
        mod._model = None
        mod._tokenizer = None
        try:
            with patch.object(mod, "load_model"):  # no-op so _model stays None
                with patch.object(mod, "detect_greeting", return_value=False):
                    with patch.object(mod, "detect_crisis", return_value={"is_crisis": False}):
                        with pytest.raises(RuntimeError, match="Classification model failed to load"):
                            get_classifications("Some random message that is not greeting or crisis")
        finally:
            mod._model_loaded = orig_loaded
            mod._model = orig_model
            mod._tokenizer = orig_tok


# ============================================================================
# classification_node (get_classifications mocked)
# ============================================================================

class TestClassificationNodeUnit:
    """Unit tests for classification_node with get_classifications mocked."""

    def test_empty_message_returns_error(self, empty_message_state):
        result = classification_node(empty_message_state)
        assert "error" in result
        assert "No user message" in result["error"]

    def test_successful_classification_returns_state_updates(self, sample_state):
        mock_classifications = {
            "intent": "venting",
            "situation": "GENERAL_OVERWHELM",
            "severity": "medium",
            "risk_score": 0.4,
            "risk_level": "medium",
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(sample_state)
        assert "error" not in result or result.get("error") is None
        assert result["intent"] == "venting"
        assert result["situation"] == "GENERAL_OVERWHELM"
        assert result["severity"] == "medium"
        assert result["risk_level"] == "medium"
        assert result["is_greeting"] is False
        assert result["classification_details"] == mock_classifications

    def test_high_risk_sets_is_high_crisis(self, sample_state):
        mock_classifications = {
            "intent": "crisis_disclosure",
            "situation": "SUICIDAL",
            "severity": "high",
            "risk_score": 0.95,
            "risk_level": "high",
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(sample_state)
        assert result["is_high_crisis"] is True
        assert result["risk_level"] == "high"

    def test_greeting_sets_is_greeting(self, greeting_state):
        mock_classifications = {
            "intent": "greeting",
            "situation": "no situation",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(greeting_state)
        assert result["is_greeting"] is True

    def test_get_classifications_exception_returns_error(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            side_effect=RuntimeError("Model not loaded"),
        ):
            result = classification_node(sample_state)
        assert "error" in result
        assert "Classification failed" in result["error"]


# ============================================================================
# Label constants
# ============================================================================

class TestClassificationLabels:
    """Sanity check for label dicts used by the model."""

    def test_intent_labels_contain_expected_keys(self):
        assert "greeting" in INTENT_LABELS.values()
        assert "crisis_disclosure" in INTENT_LABELS.values()
        assert "unclear" in INTENT_LABELS.values()

    def test_situation_labels_contain_crisis_types(self):
        assert "SUICIDAL" in SITUATION_LABELS.values()
        assert "SELF_HARM" in SITUATION_LABELS.values()
        assert "NO_SITUATION" in SITUATION_LABELS.values()
