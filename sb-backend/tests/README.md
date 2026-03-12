# SoulBuddy Backend — Testing Guide

## Overview

Tests run without any live DB, Redis, LLM, or GCP connection — all external dependencies are mocked or stubbed via `conftest.py`.

CI runs on every push via GitHub Actions (`.github/workflows/backend-tests.yml`).

---

## Test Structure

```
tests/
├── conftest.py                        # Root stubs (google.cloud.kms, etc.)
├── api/
│   └── test_chat.py                   # Chat endpoint tests
├── config/
│   └── test_settings.py
├── graph/
│   └── nodes/
│       ├── test_classification_node.py  # 93 tests — greeting/crisis/out-of-scope/intent/situation/severity/math
│       └── test_auth_node.py
└── services/
    └── test_cache_service.py
```

---

## Running Tests

```bash
# All unit tests (recommended)
pytest tests/ -m "not integration" -v

# Specific file
pytest tests/graph/nodes/test_classification_node.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Integration tests (require live external services)
pytest tests/ -m integration -v
```

---

## Classification Node Tests (`test_classification_node.py`)

The most comprehensive test file — 85 tests covering the fully rule-based classification pipeline.

### `TestDetectGreeting` / `TestDetectGreetingNewPatterns`
- Common words: hi, hello, hey, namaste, hola, yo, sup, hiya
- Enthusiastic variants: hiii, heyyy (repeated char normalisation)
- Time-based: good morning/afternoon/evening/night, gm, gn
- Check-in openers: how are you, how r u, nice to meet you
- Negative cases: long messages containing greeting words

### `TestIsTrueNegation`
- Explicit denials: "I am not suicidal", "not going to kill myself"
- Negative cases: genuine crisis phrases

### `TestDetectCrisis` / `TestDetectCrisisNewPatterns`
- Suicidal plan patterns (highest priority)
- Suicidal ideation: want to die, feel like dying, wish I could die
- Self-harm patterns
- Passive death wish: tired of living, don't want to be here, wish I could disappear
- True negation suppression: "I'm not suicidal, I don't want to die" → not crisis
- Fallback scoring: accumulated high-risk word weights

### `TestGetClassificationsUnit`
- Empty / whitespace → `unclear` / `NO_SITUATION` / `low`
- Greeting fast-path
- Crisis fast-path
- Rule-based fallback for normal messages

### `TestClassificationNodeUnit`
- Empty message → error dict
- Successful classification → correct state field updates
- High risk → `is_crisis_detected = True`
- Greeting → `is_greeting = True`
- Exception handling → error dict

### `TestClassificationLabels`
- `classify_intent()`: venting, seek_information, open_to_solution, fallback
- `classify_situation()`: EXAM_ANXIETY, RELATIONSHIP_ISSUES, FINANCIAL_STRESS, fallback
- `classify_severity()`: high, medium, low

### `TestClassifyOutOfScope`
Out-of-scope detection must distinguish explicit bot requests from personal narratives.

| Message | Expected |
|---------|----------|
| "Give me a recipe for pasta" | `True` |
| "Write me a Python function" | `True` |
| "Help me debug this script" | `True` |
| "What stocks should I buy?" | `True` |
| "Plan a trip to Goa" | `True` |
| "Solve this math problem" | `True` |
| "What is the average of 4,5,7,9?" | `True` |
| "Calculate the mean of these numbers" | `True` |
| "What is 15 + 27?" | `True` |
| "What is 20 percent of 500?" | `True` |
| "Convert 100 km to miles" | `True` |
| "Calculate my GPA from these grades" | `True` |
| "I spoke to my friend about a recipe" | `False` |
| "I was coding all night and I'm exhausted" | `False` |
| "I'm really stressed about money" | `False` |
| "I have a big exam tomorrow" | `False` |
| "I feel average today" | `False` |
| "I stayed up studying for my math exam" | `False` |

---

## Writing New Tests

### Unit test pattern

```python
from unittest.mock import patch
from graph.state import ConversationState

def test_my_node():
    state = ConversationState(
        conversation_id="test-123",
        mode="incognito",
        domain="general",
        user_message="I'm feeling overwhelmed",
    )
    mock_result = {"intent": "venting", ...}
    with patch("graph.nodes.agentic_nodes.my_node.some_fn", return_value=mock_result):
        result = my_node(state)
    assert result["intent"] == "venting"
```

### Integration test pattern

```python
import pytest

@pytest.mark.integration
async def test_real_api_call():
    result = await real_api_function("input")
    assert result in ["valid", "values"]
```

---

## CI Environment Variables

The following dummy values are injected in CI so imports don't fail at startup:

```
OPENAI_API_KEY=dummy-key-ci
OLLAMA_BASE_URL=http://localhost:11434
SUPABASE_URL=https://placeholder.supabase.co
SUPABASE_SERVICE_ROLE_KEY=placeholder-service-key
SUPABASE_ANON_KEY=placeholder-anon-key
DATA_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/soulbuddy
ENCRYPTION_ENABLED=false
```

---

## Best Practices

1. **Mock at the boundary** — mock the function that calls the external service, not the service itself
2. **Use `ConversationState` fixtures** — keeps test setup consistent
3. **Test both the happy path and error/fallback paths**
4. **Out-of-scope patterns** — when adding new patterns, add both a positive (should flag) and a negative (personal narrative) test case
5. **Crisis patterns** — always pair a new crisis pattern with a negation test ("I'm not suicidal")
6. **No live services in unit tests** — if a test requires Redis, DB, or an LLM, mark it `@pytest.mark.integration`
