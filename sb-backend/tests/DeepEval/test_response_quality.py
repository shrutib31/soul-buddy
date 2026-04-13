import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("deepeval")

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from graph.nodes.agentic_nodes import response_generator as response_generator_module
from graph.state import ConversationState


CASES_DIR = Path(__file__).parent / "test_cases"


def load_test_cases(filename: str):
    with (CASES_DIR / filename).open() as f:
        return json.load(f)


EMPATHY_CASES = load_test_cases("empathy_cases.json")
APPROPRIATENESS_CASES = load_test_cases("appropriateness_cases.json")


def context_to_state_fields(context: list[str]) -> dict:
    fields = {}
    for item in context:
        key, _, value = item.partition(":")
        if key and value:
            fields[key.strip()] = value.strip()
    return fields


def response_generation_is_configured() -> bool:
    return (
        response_generator_module.COMPARE_RESULTS
        or response_generator_module.OLLAMA_FLAG
        or response_generator_module.OPENAI_FLAG
    )


def skip_if_eval_is_not_configured():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("DeepEval GEval requires OPENAI_API_KEY for evaluator model access")
    if not response_generation_is_configured():
        pytest.skip("No response-generation provider configured")


async def call_response_generator(case: dict) -> str:
    context = context_to_state_fields(case.get("context", []))
    state = ConversationState(
        conversation_id="eval-response-quality",
        mode="incognito",
        domain="general",
        user_message=case["input"],
        chat_preference="general",
        intent=context.get("intent"),
        situation=context.get("situation"),
        severity=context.get("severity"),
    )

    result = await response_generator_module.response_generator_node(state)
    if "error" in result:
        if "No LLM provider enabled" in result["error"]:
            pytest.skip(result["error"])
        pytest.fail(result["error"])

    actual_response = result.get("response_draft", "")
    if not actual_response.strip():
        pytest.fail("response_generator_node returned an empty response")

    return actual_response


def evaluate_case(case: dict, metric):
    actual_response = asyncio.run(call_response_generator(case))

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=actual_response,
        retrieval_context=case.get("context", []),
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[metric],
    )

    if not result.test_results:
        pytest.fail("DeepEval returned no test results")

    failed_metrics = []
    for test_result in result.test_results:
        metrics_data = test_result.metrics_data or []
        if not metrics_data and not test_result.success:
            failed_metrics.append(f"{test_result.name} failed without metric data")
            continue

        for metric_data in metrics_data:
            if metric_data.success:
                continue

            score = "None" if metric_data.score is None else f"{metric_data.score:.3f}"
            threshold = f"{metric_data.threshold:.3f}"
            reason = metric_data.error or metric_data.reason or "No reason provided"
            failed_metrics.append(
                f"{metric_data.name} failed "
                f"(score={score}, threshold={threshold}): {reason}"
            )

    assert not failed_metrics, "\n\n".join(failed_metrics)


def test_empathy_case_fixture_has_required_coverage():
    assert len(EMPATHY_CASES) >= 20

    contexts = [" | ".join(case.get("context", [])) for case in EMPATHY_CASES]
    assert any("severity: low" in context for context in contexts)
    assert any("severity: medium" in context for context in contexts)
    assert any("intent: venting" in context for context in contexts)
    assert any("intent: seek_support" in context for context in contexts)


def test_appropriateness_case_fixture_has_required_coverage():
    assert len(APPROPRIATENESS_CASES) >= 20

    contexts = [" | ".join(case.get("context", [])) for case in APPROPRIATENESS_CASES]
    assert any("severity: low" in context for context in contexts)
    assert any("severity: medium" in context for context in contexts)
    assert any("severity: high" in context for context in contexts)
    assert any("intent: venting" in context for context in contexts)
    assert any("intent: seek_support" in context for context in contexts)
    assert any("intent: seek_understanding" in context for context in contexts)

    inputs = [case["input"].lower() for case in APPROPRIATENESS_CASES]
    assert any("advice" in user_input for user_input in inputs)
    assert any("talk to me like a friend" in user_input for user_input in inputs)
    assert any("think positive" in user_input for user_input in inputs)


@pytest.mark.parametrize("case", EMPATHY_CASES, ids=lambda case: case["input"][:60])
def test_response_quality_empathy(case):
    skip_if_eval_is_not_configured()
    from tests.DeepEval.metrics import empathy_metric

    evaluate_case(case, empathy_metric)


@pytest.mark.parametrize("case", APPROPRIATENESS_CASES, ids=lambda case: case["input"][:60])
def test_response_quality_appropriateness(case):
    skip_if_eval_is_not_configured()
    from tests.DeepEval.metrics import appropriateness_metric

    evaluate_case(case, appropriateness_metric)
