"""
Unit tests for response_evaluator.py

Covers all 6 scoring dimensions and select_best_response logic.
No external dependencies — pure unit tests.
"""

import pytest
from graph.nodes.agentic_nodes.response_evaluator import (
    score_response,
    select_best_response,
    _empathy_score,
    _engagement_score,
    _length_score,
    _completeness_score,
    _repetition_penalty,
    _robotic_penalty,
)


# ============================================================================
# score_response — empty / None guard
# ============================================================================

class TestScoreResponseGuards:
    def test_empty_string_returns_zero(self):
        assert score_response("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert score_response("   ") == 0.0

    def test_none_returns_zero(self):
        assert score_response(None) == 0.0

    def test_valid_response_returns_nonzero(self):
        text = "I hear you. That sounds really tough. Would you like to tell me more about what's going on?"
        assert score_response(text) > 0.0


# ============================================================================
# _empathy_score
# ============================================================================

class TestEmpathyScore:
    def test_no_empathy_phrases_scores_zero(self):
        assert _empathy_score("The weather is nice today.") == 0.0

    def test_single_phrase_scores_positively(self):
        assert _empathy_score("I hear you, that must be hard.") > 0.0

    def test_multiple_phrases_score_higher(self):
        single = _empathy_score("I hear you.")
        multiple = _empathy_score("I hear you. I understand. You're not alone. I'm here.")
        assert multiple > single

    def test_score_is_capped_at_seven(self):
        # Stuff many empathy phrases in
        text = " ".join([
            "i hear you", "i understand", "that sounds", "i'm sorry",
            "you're not alone", "i care", "i'm here", "that makes sense",
            "your feelings are valid", "i see you", "you matter",
        ])
        assert _empathy_score(text) <= 7.0

    def test_case_insensitive(self):
        lower = _empathy_score("i hear you")
        upper = _empathy_score("I HEAR YOU")
        assert lower == upper


# ============================================================================
# _engagement_score
# ============================================================================

class TestEngagementScore:
    def test_no_question_scores_zero(self):
        assert _engagement_score("I'm here to help you.") == 0.0

    def test_one_question_scores_positively(self):
        assert _engagement_score("How are you feeling today?") > 0.0

    def test_two_questions_scores_higher(self):
        one = _engagement_score("How are you?")
        two = _engagement_score("How are you? What's on your mind?")
        assert two > one

    def test_more_than_two_questions_capped(self):
        two = _engagement_score("How are you? What's on your mind?")
        many = _engagement_score("How? What? Why? When? Where?")
        assert many == two  # capped at 2 questions * 1.5


# ============================================================================
# _length_score
# ============================================================================

class TestLengthScore:
    def test_too_short_penalised(self):
        short = " ".join(["word"] * 10)  # 10 words < _MIN_WORDS=20
        assert _length_score(short) == -2.0

    def test_too_long_penalised(self):
        long_text = " ".join(["word"] * 250)  # 250 words > _MAX_WORDS=200
        assert _length_score(long_text) == -1.0

    def test_ideal_range_scores_two(self):
        ideal = " ".join(["word"] * 60)  # 60 words in ideal 30-120
        assert _length_score(ideal) == 2.0

    def test_acceptable_but_outside_ideal_scores_one(self):
        acceptable = " ".join(["word"] * 25)  # 25 words — >= MIN but < IDEAL_MIN
        assert _length_score(acceptable) == 1.0


# ============================================================================
# _completeness_score
# ============================================================================

class TestCompletenessScore:
    def test_empty_string_penalised(self):
        assert _completeness_score("") == -5.0

    def test_ends_with_period_scores_positive(self):
        assert _completeness_score("That sounds really hard.") == 1.0

    def test_ends_with_question_mark_scores_positive(self):
        assert _completeness_score("Would you like to share more?") == 1.0

    def test_ends_with_exclamation_scores_positive(self):
        assert _completeness_score("You are not alone!") == 1.0

    def test_ends_with_word_no_punctuation_penalised(self):
        assert _completeness_score("I hear you and I understand") == -0.5


# ============================================================================
# _repetition_penalty
# ============================================================================

class TestRepetitionPenalty:
    def test_no_repetition_no_penalty(self):
        text = "I hear you. That sounds really hard. Would you like to share more about what's going on?"
        assert _repetition_penalty(text) == 0.0

    def test_short_text_no_penalty(self):
        assert _repetition_penalty("hi") == 0.0

    def test_repeated_ngram_penalised(self):
        # Repeat a 4-gram 3 times
        repeated = "I am here for you. I am here for you. I am here for you."
        assert _repetition_penalty(repeated) < 0.0

    def test_penalty_scales_with_repetitions(self):
        one_repeat = "I am here for you. I am here for you. I am here for you."
        two_repeats = (
            "I am here for you. I am here for you. I am here for you. "
            "you are not alone. you are not alone. you are not alone."
        )
        assert _repetition_penalty(two_repeats) < _repetition_penalty(one_repeat)


# ============================================================================
# _robotic_penalty
# ============================================================================

class TestRoboticPenalty:
    def test_no_robotic_phrase_no_penalty(self):
        assert _robotic_penalty("I hear you. That sounds really hard.") == 0.0

    def test_as_an_ai_penalised(self):
        assert _robotic_penalty("As an AI, I cannot provide therapy.") < 0.0

    def test_i_dont_have_feelings_penalised(self):
        assert _robotic_penalty("I don't have feelings so I can't empathize.") < 0.0

    def test_multiple_robotic_phrases_scale_penalty(self):
        one = _robotic_penalty("As an AI, I understand.")
        two = _robotic_penalty("As an AI, I understand. I don't have feelings.")
        assert two < one


# ============================================================================
# select_best_response
# ============================================================================

class TestSelectBestResponse:
    def test_returns_tuple_of_four(self):
        result = select_best_response("Ollama reply here.", "GPT reply here.")
        assert len(result) == 4

    def test_fallback_to_gpt_when_ollama_empty(self):
        text, source, _, _ = select_best_response("", "GPT reply here.")
        assert text == "GPT reply here."
        assert source == "gpt"

    def test_fallback_to_ollama_when_gpt_empty(self):
        text, source, _, _ = select_best_response("Ollama reply here.", "")
        assert text == "Ollama reply here."
        assert source == "ollama"

    def test_scores_returned_as_floats(self):
        _, _, ollama_score, gpt_score = select_best_response("Ollama reply.", "GPT reply.")
        assert isinstance(ollama_score, float)
        assert isinstance(gpt_score, float)

    def test_higher_scoring_response_selected(self):
        # Ollama has empathy phrases and a question — should score higher than a terse GPT reply
        ollama = (
            "I hear you, and I'm really glad you reached out. "
            "That sounds incredibly hard. "
            "Would you like to share more about what's been going on?"
        )
        gpt = "ok"
        text, source, ollama_score, gpt_score = select_best_response(ollama, gpt)
        assert source == "ollama"
        assert ollama_score > gpt_score

    def test_both_empty_returns_gpt_fallback(self):
        text, source, _, _ = select_best_response("", "")
        # Both empty: gpt branch triggered first (ollama is empty)
        assert source == "gpt"
        assert text == ""
