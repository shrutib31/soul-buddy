"""
Unit tests for Classification Node (classification_node, get_classifications, detect_crisis, detect_greeting).

Tests use mocks to avoid loading the real transformer model. Crisis detection and greeting
logic are tested directly; get_classifications/classification_node are tested with get_classifications mocked.
"""

import pytest
from unittest.mock import patch

from graph.state import ConversationState
from graph.nodes.agentic_nodes.classification_node import (
    classification_node,
    get_classifications,
    detect_crisis,
    detect_greeting,
    is_true_negation,
    classify_intent,
    classify_situation,
    classify_severity,
    classify_out_of_scope,
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
        chat_preference="general",
    )


@pytest.fixture
def empty_message_state():
    """State with empty user message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="",
        chat_preference="general",
    )


@pytest.fixture
def greeting_state():
    """State with a greeting message."""
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="Hi there!",
        chat_preference="general",
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
        assert out["situation"] == "NO_SITUATION"
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

    def test_what_is_up_baby_returns_greeting_intent(self):
        out = get_classifications("what is up baby?")
        assert out["intent"] == "greeting"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"
        assert out["is_greeting"] is True

    def test_crisis_message_returns_crisis_classification(self):
        out = get_classifications("I want to kill myself")
        assert out["intent"] == "crisis_disclosure"
        assert out["situation"] == "SUICIDAL"
        assert out["severity"] == "high"
        assert out["risk_level"] in ("high", "critical")

    def test_out_of_scope_message_returns_out_of_scope_classification(self):
        out = get_classifications("What is the capital of France?")
        assert out["intent"] == "out_of_scope"
        assert out["situation"] == "NO_SITUATION"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"
        assert out["is_out_of_scope"] is True
        assert out["out_of_scope_reason"] == "general_knowledge"

    def test_mixed_alphanumeric_gibberish_returns_out_of_scope_classification(self):
        out = get_classifications("infwbu94f873ucn39uq8f sad jfn9c2893fh83fh")
        assert out["intent"] == "out_of_scope"
        assert out["situation"] == "NO_SITUATION"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"
        assert out["is_out_of_scope"] is True
        assert out["out_of_scope_reason"] == "nonsense"

    def test_symbol_split_random_letters_return_out_of_scope_classification(self):
        out = get_classifications("fhowijvnaiewlnaces'da")
        assert out["intent"] == "out_of_scope"
        assert out["situation"] == "NO_SITUATION"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"
        assert out["is_out_of_scope"] is True
        assert out["out_of_scope_reason"] == "nonsense"

    def test_single_mixed_alphanumeric_token_returns_out_of_scope_classification(self):
        out = get_classifications("f9qu3hvleiurbvierowfeca")
        assert out["intent"] == "out_of_scope"
        assert out["situation"] == "NO_SITUATION"
        assert out["severity"] == "low"
        assert out["risk_level"] == "low"
        assert out["is_out_of_scope"] is True
        assert out["out_of_scope_reason"] == "nonsense"

    def test_out_of_scope_detection_disables_llm_fallback(self):
        import graph.nodes.agentic_nodes.classification_node as mod

        # Use a message long enough (>4 words) to bypass the short-message guard
        # so execution reaches the ML model check and raises RuntimeError.
        test_message = "I am feeling completely overwhelmed with everything in my life right now"

        with patch.object(mod, "detect_greeting", return_value=False):
            with patch.object(mod, "detect_crisis", return_value={"is_crisis": False}):
                with patch.object(
                    mod,
                    "detect_out_of_scope",
                    return_value={"is_out_of_scope": False},
                ) as mock_detect_out_of_scope:
                    with patch.object(mod, "detect_positive_message", return_value=False):
                        with patch.object(mod, "load_model"):
                            orig_loaded, orig_model, orig_tok = mod._model_loaded, mod._model, mod._tokenizer
                            mod._model_loaded = False
                            mod._model = None
                            mod._tokenizer = None
                            try:
                                with pytest.raises(RuntimeError, match="Classification model failed to load"):
                                    get_classifications(test_message)
                            finally:
                                mod._model_loaded = orig_loaded
                                mod._model = orig_model
                                mod._tokenizer = orig_tok

        mock_detect_out_of_scope.assert_called_once_with(
            test_message,
            domain="general",
            allow_llm_fallback=False,
        )



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

    def test_high_risk_sets_is_crisis_detected(self, sample_state):
        mock_classifications = {
            "intent": "crisis_disclosure",
            "situation": "SUICIDAL",
            "severity": "high",
            "risk_score": 0.95,
            "risk_level": "high",
            "is_crisis_detected": True,
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(sample_state)
        assert result["is_crisis_detected"] is True
        assert result["risk_level"] == "high"

    def test_greeting_sets_is_greeting(self, greeting_state):
        mock_classifications = {
            "intent": "greeting",
            "situation": "no situation",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_greeting": True,
            "is_crisis_detected": False,
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(greeting_state)
        assert result["is_greeting"] is True

    def test_out_of_scope_sets_is_out_of_scope(self, sample_state):
        mock_classifications = {
            "intent": "out_of_scope",
            "situation": "NO_SITUATION",
            "severity": "low",
            "risk_score": 0.0,
            "risk_level": "low",
            "is_out_of_scope": True,
            "out_of_scope_reason": "general_knowledge",
            "raw_scores": {},
        }
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            return_value=mock_classifications,
        ):
            result = classification_node(sample_state)
        assert result["intent"] == "out_of_scope"
        assert result["is_out_of_scope"] is True
        assert result["out_of_scope_reason"] == "general_knowledge"
        assert result["risk_level"] == "low"

    def test_get_classifications_exception_returns_error(self, sample_state):
        with patch(
            "graph.nodes.agentic_nodes.classification_node.get_classifications",
            side_effect=RuntimeError("Model not loaded"),
        ):
            result = classification_node(sample_state)
        assert "error" in result
        assert "Classification failed" in result["error"]


# ============================================================================
# Rule-based classify_* functions
# ============================================================================

class TestClassificationLabels:
    """Sanity checks for the rule-based classify_* functions."""

    def test_intent_venting(self):
        assert classify_intent("I'm so stressed and overwhelmed") == "venting"

    def test_intent_seek_information(self):
        assert classify_intent("What causes anxiety?") == "seek_information"

    def test_intent_open_to_solution(self):
        assert classify_intent("What should I do about this situation?") == "open_to_solution"

    def test_intent_fallback_unclear(self):
        assert classify_intent("okay") == "unclear"

    def test_situation_exam_anxiety(self):
        assert classify_situation("I have a big exam tomorrow") == "EXAM_ANXIETY"

    def test_situation_relationship(self):
        assert classify_situation("My girlfriend and I broke up") == "RELATIONSHIP_ISSUES"

    def test_situation_financial(self):
        assert classify_situation("I can't afford rent this month") == "FINANCIAL_STRESS"

    def test_situation_fallback_no_situation(self):
        assert classify_situation("I just feel off today") == "NO_SITUATION"

    def test_severity_high(self):
        assert classify_severity("I feel completely hopeless and desperate") == "high"

    def test_severity_medium(self):
        assert classify_severity("I've been feeling stressed and anxious") == "medium"

    def test_severity_low(self):
        assert classify_severity("I'm a bit worried sometimes") == "low"


# ============================================================================
# detect_crisis — new patterns added for passive suicidal ideation
# ============================================================================

class TestDetectCrisisNewPatterns:
    """Tests for newly added crisis detection patterns."""

    # --- SUICIDAL_IDEATION: direct death/dying phrases ---

    def test_want_to_die_detected(self):
        out = detect_crisis("I want to die")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_feel_like_dying_detected(self):
        out = detect_crisis("I feel like dying")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_feel_like_i_should_die_detected(self):
        out = detect_crisis("Sometimes I feel like I should die.")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"
        assert out["severity"] == "high"

    def test_i_should_die_detected(self):
        out = detect_crisis("I should die")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_i_want_to_die_detected(self):
        out = detect_crisis("I just feel like I want to die.")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_wish_i_could_die_detected(self):
        out = detect_crisis("I wish I could die in my sleep")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_sometimes_i_feel_like_dying_detected(self):
        out = detect_crisis("Sometimes I feel like dying is the only way out")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    def test_want_to_be_dead_detected(self):
        out = detect_crisis("I want to be dead")
        assert out["is_crisis"] is True
        assert out["situation"] == "SUICIDAL"

    # --- PASSIVE_DEATH_WISH patterns ---

    def test_tired_of_living_detected(self):
        out = detect_crisis("I'm so tired of living like this")
        assert out["is_crisis"] is True
        assert out["situation"] == "PASSIVE_DEATH_WISH"

    def test_dont_want_to_be_here_detected(self):
        out = detect_crisis("I just don't want to be here anymore")
        assert out["is_crisis"] is True
        assert out["situation"] == "PASSIVE_DEATH_WISH"

    def test_wish_i_could_disappear_detected(self):
        out = detect_crisis("I wish I could disappear from everything")
        assert out["is_crisis"] is True
        assert out["situation"] == "PASSIVE_DEATH_WISH"

    def test_wouldnt_mind_dying_detected(self):
        out = detect_crisis("I wouldn't mind dying honestly")
        assert out["is_crisis"] is True
        assert out["situation"] == "PASSIVE_DEATH_WISH"

    # --- True negation should NOT fire even with "die" in message ---

    def test_negated_i_should_die_not_crisis(self):
        out = detect_crisis("I'm not suicidal, I don't want to die")
        # Negation check should suppress the pattern match
        assert out["is_crisis"] is False

    # --- Fallback SEVERE_DISTRESS path ---

    def test_fallback_multiple_high_risk_words_triggers_crisis(self):
        # "suicide" (0.6) alone should now hit threshold
        out = detect_crisis("I've been thinking about suicide a lot")
        assert out["is_crisis"] is True

    def test_fallback_single_low_risk_word_not_crisis(self):
        out = detect_crisis("There was so much pain in the movie")
        # "pain" = 0.2 only — should not trigger crisis
        assert out["is_crisis"] is False


# ============================================================================
# detect_greeting — new patterns
# ============================================================================

class TestDetectGreetingNewPatterns:
    """Tests for extended greeting patterns."""

    def test_namaste(self):
        assert detect_greeting("namaste") is True

    def test_namaskar(self):
        assert detect_greeting("namaskar") is True

    def test_gm(self):
        assert detect_greeting("gm") is True

    def test_gn(self):
        assert detect_greeting("gn") is True

    def test_hola(self):
        assert detect_greeting("hola") is True

    def test_yo(self):
        assert detect_greeting("yo") is True

    def test_sup(self):
        assert detect_greeting("sup") is True

    def test_hiya(self):
        assert detect_greeting("hiya") is True

    def test_how_are_you(self):
        assert detect_greeting("how are you") is True

    def test_how_are_you_doing(self):
        assert detect_greeting("how are you doing") is True

    def test_how_r_u(self):
        assert detect_greeting("how r u") is True

    def test_what_is_up_baby(self):
        assert detect_greeting("what is up baby?") is True

    def test_what_is_up_with_me_is_not_greeting(self):
        assert detect_greeting("what is up with me?") is False

    def test_nice_to_meet_you(self):
        assert detect_greeting("nice to meet you") is True

    def test_enthusiastic_hiii(self):
        # Repeated chars normalised: hiii → hii (in EXACT_GREETINGS)
        assert detect_greeting("hiii") is True

    def test_enthusiastic_heyyy(self):
        assert detect_greeting("heyyy") is True

    def test_good_night(self):
        assert detect_greeting("good night") is True


# ============================================================================
# classify_out_of_scope
# ============================================================================

class TestClassifyOutOfScope:
    """Tests for classify_out_of_scope — must flag explicit bot requests for
    off-domain tasks while leaving personal narratives untouched."""

    # ── Should be flagged (direct request to the bot) ────────────────────────

    def test_recipe_request_flagged(self):
        assert classify_out_of_scope("Can you give me a recipe for pasta?") is True

    def test_code_write_flagged(self):
        assert classify_out_of_scope("Write me a Python function to sort a list") is True

    def test_code_debug_flagged(self):
        assert classify_out_of_scope("Help me debug this script") is True

    def test_legal_advice_flagged(self):
        assert classify_out_of_scope("Give me legal advice on my contract") is True

    def test_investment_advice_flagged(self):
        assert classify_out_of_scope("What stocks should I buy?") is True

    def test_movie_recommendation_flagged(self):
        assert classify_out_of_scope("Recommend a good movie to watch") is True

    def test_trip_planning_flagged(self):
        assert classify_out_of_scope("Plan a trip to Goa for me") is True

    def test_homework_flagged(self):
        assert classify_out_of_scope("Solve this math problem for me") is True

    def test_weather_flagged(self):
        assert classify_out_of_scope("What's the weather today?") is True

    def test_tax_filing_flagged(self):
        assert classify_out_of_scope("Help me file my taxes") is True

    # ── Should NOT be flagged (personal narrative / wellness context) ─────────

    def test_personal_recipe_mention_not_flagged(self):
        """Mentioning a recipe in personal context is NOT out-of-scope."""
        assert classify_out_of_scope("I spoke to my friend about a recipe today") is False

    def test_cooking_narrative_not_flagged(self):
        """Planning to cook shared as personal update is NOT out-of-scope."""
        assert classify_out_of_scope("Today I am going to prepare food for my family") is False

    def test_coding_stress_not_flagged(self):
        """Mentioning coding as a source of stress is NOT out-of-scope."""
        assert classify_out_of_scope("I was coding all night and I'm exhausted") is False

    def test_money_stress_not_flagged(self):
        """Talking about financial stress is a wellness topic, not investment advice."""
        assert classify_out_of_scope("I'm really stressed about money") is False

    def test_exam_mention_not_flagged(self):
        """Exam anxiety is squarely within the wellness domain."""
        assert classify_out_of_scope("I have a big exam tomorrow and I'm anxious") is False

    def test_wellness_message_not_flagged(self):
        assert classify_out_of_scope("I've been feeling really overwhelmed lately") is False

    def test_empty_not_flagged(self):
        assert classify_out_of_scope("") is False

    def test_none_not_flagged(self):
        assert classify_out_of_scope(None) is False

    # ── Math / Calculations ───────────────────────────────────────────────
    def test_average_calculation_flagged(self):
        assert classify_out_of_scope("What is the average of 4,5,7,9?") is True

    def test_mean_calculation_flagged(self):
        assert classify_out_of_scope("Calculate the mean of these numbers: 10, 20, 30") is True

    def test_arithmetic_flagged(self):
        assert classify_out_of_scope("What is 15 + 27?") is True

    def test_percentage_flagged(self):
        assert classify_out_of_scope("What is 20 percent of 500?") is True

    def test_unit_conversion_flagged(self):
        assert classify_out_of_scope("Convert 100 km to miles") is True

    def test_gpa_calculation_flagged(self):
        assert classify_out_of_scope("Calculate my GPA from these grades") is True

    def test_feeling_average_not_flagged(self):
        """'average' as emotional adjective must not be flagged"""
        assert classify_out_of_scope("I feel average today, not great not terrible") is False

    def test_math_stress_not_flagged(self):
        """Mentioning math in a personal context must not be flagged"""
        assert classify_out_of_scope("I stayed up all night studying for my math exam") is False
