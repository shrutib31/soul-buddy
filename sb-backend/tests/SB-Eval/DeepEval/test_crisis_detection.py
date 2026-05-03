import json
from pathlib import Path

import pytest

from graph.nodes.agentic_nodes.classification_node import detect_crisis, classify_situation


CASES_PATH = Path(__file__).parent / "test_cases" / "crisis_cases.json"


def load_crisis_cases():
    with CASES_PATH.open() as f:
        return json.load(f)


CASES = load_crisis_cases()


def classify_case(user_input: str) -> dict:
    """
    Evaluate the deterministic crisis-detection path from classification_node.

    get_classifications() eventually falls through to the ML classifier for
    non-crisis messages. These eval labels target crisis routing specifically,
    so non-crisis situation checks use the rule-based situation helper.
    """
    crisis_result = detect_crisis(user_input)
    if crisis_result["is_crisis"]:
        return {
            "is_crisis": True,
            "situation": crisis_result["situation"],
        }

    return {
        "is_crisis": False,
        "situation": classify_situation(user_input),
    }


def test_crisis_case_fixture_has_required_coverage():
    assert len(CASES) >= 15
    assert any(case["expected_is_crisis"] is True for case in CASES)
    assert any(case["expected_is_crisis"] is False for case in CASES)

    situations = {case["expected_situation"] for case in CASES}
    assert "SUICIDAL" in situations
    assert "SELF_HARM" in situations
    assert "PASSIVE_DEATH_WISH" in situations


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["input"][:60])
def test_crisis_detection_matches_labeled_ground_truth(case):
    result = classify_case(case["input"])

    assert result["is_crisis"] is case["expected_is_crisis"]
    assert result["situation"] == case["expected_situation"]
