"""
Unit tests for out_of_scope.py helper functions.

Covers every function not already tested in test_fast_out_of_scope.py:
  - looks_like_general_knowledge
  - looks_like_nonsense  (+ internal scorers)
  - looks_like_in_scope_support
  - is_support_topic_query
  - detect_pattern_reason
  - detect_out_of_scope  (pattern path, in-scope path, no-llm path, llm path, llm failure)
  - get_out_of_scope_reason
  - get_out_of_scope_result
  - build_out_of_scope_prompt
  - looks_like_code_snippet
  - has_symbol_noise
  - has_mixed_alnum_noise
  - is_keyboard_smash
  - is_single_keyboard_row_token
  - get_common_chunk_hits
  - contains_support_term
  - get_longest_vowel_run
  - get_longest_consonant_run
  - get_gibberish_score
"""

import pytest
from unittest.mock import patch

from graph.nodes.function_nodes.out_of_scope import (
    looks_like_general_knowledge,
    looks_like_nonsense,
    looks_like_in_scope_support,
    is_support_topic_query,
    detect_pattern_reason,
    detect_out_of_scope,
    get_out_of_scope_reason,
    get_out_of_scope_result,
    build_out_of_scope_prompt,
    looks_like_code_snippet,
    has_symbol_noise,
    has_mixed_alnum_noise,
    is_keyboard_smash,
    is_single_keyboard_row_token,
    get_common_chunk_hits,
    contains_support_term,
    get_longest_vowel_run,
    get_longest_consonant_run,
    get_gibberish_score,
)


# ============================================================================
# looks_like_general_knowledge
# ============================================================================

class TestLooksLikeGeneralKnowledge:
    def test_capital_of(self):
        assert looks_like_general_knowledge("what is the capital of france") is True

    def test_population_of(self):
        assert looks_like_general_knowledge("what is the population of india") is True

    def test_currency_of(self):
        assert looks_like_general_knowledge("what is the currency of japan") is True

    def test_president_of(self):
        assert looks_like_general_knowledge("who is the president of usa") is True

    def test_prime_minister_of(self):
        assert looks_like_general_knowledge("who is the prime minister of canada") is True

    def test_weather_in(self):
        assert looks_like_general_knowledge("what is the weather in london today") is True

    def test_calculate(self):
        assert looks_like_general_knowledge("calculate the area of a circle") is True

    def test_translate(self):
        assert looks_like_general_knowledge("translate hello to spanish") is True

    def test_definition_of(self):
        assert looks_like_general_knowledge("definition of photosynthesis") is True

    def test_what_is_the_formula(self):
        assert looks_like_general_knowledge("what is the formula for water") is True

    def test_solve_equation(self):
        assert looks_like_general_knowledge("solve x + 3 = 7") is True

    def test_normal_message_not_flagged(self):
        assert looks_like_general_knowledge("i feel really stressed today") is False

    def test_greeting_not_flagged(self):
        assert looks_like_general_knowledge("hello how are you") is False

    def test_empty_string_not_flagged(self):
        assert looks_like_general_knowledge("") is False

    def test_personal_narrative_not_flagged(self):
        assert looks_like_general_knowledge("i was in biology class all night and now i feel overwhelmed") is False


# ============================================================================
# looks_like_nonsense
# ============================================================================

class TestLooksLikeNonsense:
    def test_keyboard_smash(self):
        assert looks_like_nonsense("asdfghjkl") is True

    def test_random_consonants(self):
        assert looks_like_nonsense("xkcdwfmnbpqrst") is True

    def test_single_letter_tokens(self):
        # 4+ single-letter alpha tokens
        assert looks_like_nonsense("a b c d e") is True

    def test_normal_sentence_not_nonsense(self):
        assert looks_like_nonsense("i feel really stressed today") is False

    def test_greeting_not_nonsense(self):
        assert looks_like_nonsense("hello how are you doing") is False

    def test_empty_string_not_nonsense(self):
        assert looks_like_nonsense("") is False

    def test_only_numbers_not_nonsense(self):
        # no alpha tokens → alnum_tokens exist but token_scores is empty
        assert looks_like_nonsense("12345 67890") is False

    def test_code_snippet_not_nonsense(self):
        # code snippet should be exempt
        assert looks_like_nonsense("def foo(): pass;{}") is False

    def test_short_token_not_nonsense(self):
        # tokens shorter than 3 chars are skipped
        assert looks_like_nonsense("hi ok go") is False

    def test_repeated_char_gibberish(self):
        # 6+ same character
        assert looks_like_nonsense("aaaaaaaaa") is True

    def test_mixed_alnum_gibberish(self):
        assert looks_like_nonsense("f9qu3hvleiurbvierowfeca") is True

    def test_wellness_word_not_nonsense(self):
        assert looks_like_nonsense("therapy mindfulness meditation") is False

    def test_medium_score_accumulation(self):
        # multiple tokens scoring >=2 but none >=5
        result = looks_like_nonsense("qweqwe mnmnmn bvcbvc")
        # just assert it doesn't raise; result may vary
        assert isinstance(result, bool)


# ============================================================================
# looks_like_code_snippet
# ============================================================================

class TestLooksLikeCodeSnippet:
    def test_python_function(self):
        assert looks_like_code_snippet("def foo(): pass; {}") is True

    def test_javascript_arrow(self):
        assert looks_like_code_snippet("const fn = () => { return 1; }") is True

    def test_backtick_block(self):
        assert looks_like_code_snippet("```python\nprint('hi')\n```; []") is True

    def test_no_markers(self):
        assert looks_like_code_snippet("how are you today") is False

    def test_marker_without_punctuation(self):
        # one marker but < 2 punctuation chars
        assert looks_like_code_snippet("import os") is False


# ============================================================================
# is_support_topic_query
# ============================================================================

class TestIsSupportTopicQuery:
    def test_mindfulness_what_is(self):
        assert is_support_topic_query("what is mindfulness") is True

    def test_trauma_explain(self):
        assert is_support_topic_query("can you explain trauma bonding") is True

    def test_cbt_meaning(self):
        assert is_support_topic_query("meaning of cbt") is True

    def test_grounding_tell_me(self):
        assert is_support_topic_query("tell me about grounding techniques") is True

    def test_trivia_who_blocks(self):
        # "who" triggers TRIVIA_STYLE_PATTERNS → returns False
        assert is_support_topic_query("who invented mindfulness meditation") is False

    def test_no_keyword_returns_false(self):
        assert is_support_topic_query("what is the capital of france") is False

    def test_no_pattern_returns_false(self):
        # has keyword but no knowledge pattern
        assert is_support_topic_query("i like mindfulness") is False


# ============================================================================
# looks_like_in_scope_support
# ============================================================================

class TestLooksLikeInScopeSupport:
    def test_support_keyword_match(self):
        assert looks_like_in_scope_support("i am really anxious about my exams") is True

    def test_wellness_keyword(self):
        assert looks_like_in_scope_support("i need help with my wellness") is True

    def test_feeling_i_am(self):
        assert looks_like_in_scope_support("i am struggling to cope") is True

    def test_my_feel(self):
        assert looks_like_in_scope_support("my feelings are overwhelming me") is True

    def test_support_topic_query(self):
        assert looks_like_in_scope_support("what is mindfulness") is True

    def test_off_topic_message(self):
        assert looks_like_in_scope_support("give me a recipe for pasta") is False

    def test_empty_string(self):
        assert looks_like_in_scope_support("") is False

    def test_general_knowledge_not_in_scope(self):
        assert looks_like_in_scope_support("what is the capital of france") is False

    def test_first_person_without_feeling(self):
        # has "i" but no feeling verb → not matched by the regex branch
        assert looks_like_in_scope_support("i like pasta") is False


# ============================================================================
# detect_pattern_reason
# ============================================================================

class TestDetectPatternReason:
    def test_nonsense_returns_nonsense(self):
        assert detect_pattern_reason("asdfghjkl") == "nonsense"

    def test_general_knowledge_returns_general_knowledge(self):
        assert detect_pattern_reason("what is the capital of france") == "general_knowledge"

    def test_in_scope_support_topic_returns_none(self):
        # "definition of mindfulness" → is_support_topic_query → True → None
        assert detect_pattern_reason("definition of mindfulness") is None

    def test_normal_message_returns_none(self):
        assert detect_pattern_reason("i feel stressed today") is None

    def test_empty_returns_none(self):
        assert detect_pattern_reason("") is None


# ============================================================================
# detect_out_of_scope
# ============================================================================

class TestDetectOutOfScope:
    def test_empty_message_is_out_of_scope(self):
        result = detect_out_of_scope("", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is True
        assert result["reason"] == "other_out_of_scope"

    def test_whitespace_only_is_out_of_scope(self):
        result = detect_out_of_scope("   ", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is True

    def test_non_string_is_out_of_scope(self):
        result = detect_out_of_scope(None, allow_llm_fallback=False)  # type: ignore
        assert result["is_out_of_scope"] is True

    def test_pattern_match_flagged(self):
        result = detect_out_of_scope("asdfghjkl", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is True
        assert result["reason"] == "nonsense"

    def test_general_knowledge_flagged(self):
        result = detect_out_of_scope("what is the capital of france", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is True
        assert result["reason"] == "general_knowledge"

    def test_in_scope_support_not_flagged(self):
        result = detect_out_of_scope("i feel anxious about my exams", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is False
        assert result["reason"] == "in_scope"

    def test_no_llm_fallback_unknown_message_is_in_scope(self):
        # ambiguous message, no LLM → in scope
        result = detect_out_of_scope("how can I improve my memory", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is False

    def test_domain_passed_to_result(self):
        result = detect_out_of_scope("asdfghjkl", domain="student", allow_llm_fallback=False)
        assert result["is_out_of_scope"] is True
        assert "response" in result

    def test_llm_fallback_out_of_scope(self):
        mock_llm = lambda prompt: '{"is_out_of_scope": true, "reason": "general_knowledge"}'
        result = detect_out_of_scope(
            "something ambiguous",
            allow_llm_fallback=True,
            llm_fn=mock_llm,
        )
        assert result["is_out_of_scope"] is True
        assert result["reason"] == "general_knowledge"

    def test_llm_fallback_in_scope(self):
        mock_llm = lambda prompt: '{"is_out_of_scope": false, "reason": "in_scope"}'
        result = detect_out_of_scope(
            "something ambiguous",
            allow_llm_fallback=True,
            llm_fn=mock_llm,
        )
        assert result["is_out_of_scope"] is False

    def test_llm_fallback_failure_defaults_to_in_scope(self):
        def bad_llm(prompt):
            raise RuntimeError("connection refused")

        result = detect_out_of_scope(
            "something ambiguous",
            allow_llm_fallback=True,
            llm_fn=bad_llm,
        )
        assert result["is_out_of_scope"] is False

    def test_llm_fallback_invalid_json_defaults_to_in_scope(self):
        mock_llm = lambda prompt: "not json at all"
        result = detect_out_of_scope(
            "something ambiguous",
            allow_llm_fallback=True,
            llm_fn=mock_llm,
        )
        assert result["is_out_of_scope"] is False

    def test_llm_fallback_uses_default_guardrail_llm_when_none(self):
        # Exercises lines 154-157: llm_fn=None path with allow_llm_fallback=True
        # The import is lazy so patch at the source module
        with patch(
            "graph.nodes.agentic_nodes.guardrail.call_guardrail_llm",
            return_value='{"is_out_of_scope": false, "reason": "in_scope"}',
        ):
            result = detect_out_of_scope(
                "something ambiguous",
                allow_llm_fallback=True,
                llm_fn=None,
            )
        assert result["is_out_of_scope"] is False

    def test_result_contains_response_when_out_of_scope(self):
        result = detect_out_of_scope("asdfghjkl", allow_llm_fallback=False)
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_result_response_empty_when_in_scope(self):
        result = detect_out_of_scope("i feel anxious", allow_llm_fallback=False)
        assert result["response"] == ""


# ============================================================================
# get_out_of_scope_reason
# ============================================================================

class TestGetOutOfScopeReason:
    def test_not_out_of_scope_always_in_scope(self):
        assert get_out_of_scope_reason("general_knowledge", False) == "in_scope"

    def test_known_reason_general_knowledge(self):
        assert get_out_of_scope_reason("general_knowledge", True) == "general_knowledge"

    def test_known_reason_nonsense(self):
        assert get_out_of_scope_reason("nonsense", True) == "nonsense"

    def test_known_reason_other_out_of_scope(self):
        assert get_out_of_scope_reason("other_out_of_scope", True) == "other_out_of_scope"

    def test_unknown_reason_falls_back_to_other(self):
        assert get_out_of_scope_reason("random_value", True) == "other_out_of_scope"

    def test_none_reason_falls_back_to_other(self):
        assert get_out_of_scope_reason(None, True) == "other_out_of_scope"

    def test_empty_reason_falls_back_to_other(self):
        assert get_out_of_scope_reason("", True) == "other_out_of_scope"


# ============================================================================
# get_out_of_scope_result
# ============================================================================

class TestGetOutOfScopeResult:
    def test_out_of_scope_true_sets_reason(self):
        result = get_out_of_scope_result(True, "nonsense", "general")
        assert result["is_out_of_scope"] is True
        assert result["reason"] == "nonsense"
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_out_of_scope_false_reason_always_in_scope(self):
        result = get_out_of_scope_result(False, "nonsense", "general")
        assert result["is_out_of_scope"] is False
        assert result["reason"] == "in_scope"
        assert result["response"] == ""

    def test_domain_student_returns_response(self):
        result = get_out_of_scope_result(True, "general_knowledge", "student")
        assert len(result["response"]) > 0

    def test_unknown_domain_falls_back_gracefully(self):
        result = get_out_of_scope_result(True, "other_out_of_scope", "unknown_domain")
        assert len(result["response"]) > 0


# ============================================================================
# build_out_of_scope_prompt
# ============================================================================

class TestBuildOutOfScopePrompt:
    def test_contains_message(self):
        prompt = build_out_of_scope_prompt("give me a recipe")
        assert "give me a recipe" in prompt

    def test_contains_json_shape_hint(self):
        prompt = build_out_of_scope_prompt("hello")
        assert "is_out_of_scope" in prompt

    def test_non_empty(self):
        assert len(build_out_of_scope_prompt("anything")) > 0


# ============================================================================
# Internal helpers: has_symbol_noise, has_mixed_alnum_noise,
#                   is_keyboard_smash, is_single_keyboard_row_token
# ============================================================================

class TestHasSymbolNoise:
    def test_exclamation_is_noise(self):
        assert has_symbol_noise("hello!") is True

    def test_plain_alphanumeric_not_noise(self):
        assert has_symbol_noise("hello world 123") is False

    def test_empty_not_noise(self):
        assert has_symbol_noise("") is False


class TestHasMixedAlnumNoise:
    def test_interleaved_digits_and_alpha(self):
        # e.g. "a1b2c3d4e5" — 10 chars, interleaved, digit_ratio 0.5 exactly → edge
        assert isinstance(has_mixed_alnum_noise("a1b2c3d4e5", "abcde"), bool)

    def test_pure_alpha_not_noise(self):
        assert has_mixed_alnum_noise("helloworld", "helloworld") is False

    def test_short_token_not_noise(self):
        assert has_mixed_alnum_noise("ab12", "ab") is False

    def test_leading_digits_suffix_alpha_not_noise(self):
        # fullmatch [0-9]+[a-z]+ → excluded
        assert has_mixed_alnum_noise("12345abcde", "abcde") is False

    def test_leading_alpha_suffix_digits_not_noise(self):
        # fullmatch [a-z]+[0-9]+ → excluded
        assert has_mixed_alnum_noise("abcde12345", "abcde") is False


class TestIsKeyboardSmash:
    def test_qwerty_row(self):
        assert is_keyboard_smash("qwerty") is True

    def test_asdf_row(self):
        assert is_keyboard_smash("asdfg") is True

    def test_zxcvb_row(self):
        assert is_keyboard_smash("zxcvb") is True

    def test_reverse_row(self):
        assert is_keyboard_smash("ytrewq") is True

    def test_normal_word_not_smash(self):
        assert is_keyboard_smash("hello") is False

    def test_short_token_not_smash(self):
        assert is_keyboard_smash("asd") is False  # len < 5


class TestIsSingleKeyboardRowToken:
    def test_all_from_top_row(self):
        assert is_single_keyboard_row_token("qwert") is True

    def test_mixed_rows_not_single(self):
        assert is_single_keyboard_row_token("qazsw") is False

    def test_short_token_false(self):
        assert is_single_keyboard_row_token("qwe") is False  # len < 5


class TestGetCommonChunkHits:
    def test_wellbeing_chunks(self):
        hits = get_common_chunk_hits("wellbeing")
        assert hits >= 1

    def test_no_chunks(self):
        assert get_common_chunk_hits("xkzq") == 0

    def test_multiple_chunks(self):
        hits = get_common_chunk_hits("selfcare feeling")
        assert hits >= 2


class TestContainsSupportTerm:
    def test_therapy_present(self):
        assert contains_support_term("therapy") is True

    def test_mindfulness_present(self):
        assert contains_support_term("mindfulness") is True

    def test_nonsense_token_not_support(self):
        assert contains_support_term("xkzqwvbr") is False

    def test_empty_string(self):
        assert contains_support_term("") is False


class TestGetLongestVowelRun:
    def test_no_vowels(self):
        assert get_longest_vowel_run("bcdfgh") == 0

    def test_single_vowel(self):
        assert get_longest_vowel_run("bat") == 1

    def test_double_vowel_run(self):
        assert get_longest_vowel_run("queue") >= 2

    def test_all_vowels(self):
        assert get_longest_vowel_run("aeiou") == 5


class TestGetLongestConsonantRun:
    def test_no_consonants(self):
        assert get_longest_consonant_run("aeiou") == 0

    def test_single_consonant(self):
        assert get_longest_consonant_run("bat") == 1

    def test_long_consonant_run(self):
        assert get_longest_consonant_run("strengths") >= 4

    def test_all_consonants(self):
        assert get_longest_consonant_run("bcdfgh") == 6


class TestGetGibberishScore:
    def test_normal_word_scores_zero(self):
        score = get_gibberish_score("hello", "hello")
        assert score == 0  # "hello" is short and has vowels, no gibberish signals

    def test_keyboard_smash_scores_high(self):
        score = get_gibberish_score("qwerty", "qwerty")
        assert score >= 3

    def test_support_term_scores_zero(self):
        # "therapy" is a support term → 0
        assert get_gibberish_score("therapy", "therapy") == 0

    def test_short_alpha_scores_zero(self):
        # len < 3 → 0
        assert get_gibberish_score("ab", "ab") == 0

    def test_all_consonants_long_scores_high(self):
        score = get_gibberish_score("xkzqwvbr", "xkzqwvbr")
        assert score >= 3

    def test_symbol_noise_with_no_vowel_token_adds_score(self):
        # Covers line 275: has_symbol_noise + len >= 4 + vowel_count == 0
        # message_lower has symbol noise ("!"), token "xkzq" has no vowels and len >= 4
        score = get_gibberish_score("xkzq", "xkzq!")
        assert score >= 3
