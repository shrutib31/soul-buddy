# SoulBuddy Backend - Testing Guide

## Test Structure

```
tests/
├── __init__.py
├── graph/
│   ├── __init__.py
│   └── nodes/
│       ├── __init__.py
│       └── test_intent_detection.py
```

## Test Types

### 1. Unit Tests (Fast, Mocked)
Tests with mocked external dependencies (Ollama API calls). Run frequently during development.

**Run unit tests:**
```bash
pytest tests/ -m "not integration" -v
```

Or use the test runner:
```bash
./run_tests.sh unit
```

### 2. Integration Tests (Slower, Real API)
Tests that make actual Ollama API calls. Requires Ollama running.

**Run integration tests:**
```bash
pytest tests/ -m integration -v
```

Or use the test runner:
```bash
./run_tests.sh integration
```

## Running Tests

### Quick Start

**Run all unit tests (recommended for development):**
```bash
pytest tests/ -v -m "not integration"
```

**Run all tests including integration:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/graph/nodes/test_intent_detection.py -v
```

**Run specific test function:**
```bash
pytest tests/graph/nodes/test_intent_detection.py::TestIntentDetectionNodeUnit::test_empty_message_returns_error -v
```

### Using Test Runner Script

```bash
# Make script executable (first time only)
chmod +x run_tests.sh

# Run unit tests only (fast)
./run_tests.sh unit

# Run all tests including integration
./run_tests.sh integration

# Run with coverage report
./run_tests.sh coverage

# Run specific test file
./run_tests.sh specific tests/graph/nodes/test_intent_detection.py
```

## Test Coverage

**Generate coverage report:**
```bash
pytest tests/ --cov=graph --cov=api --cov=config --cov-report=term-missing --cov-report=html
```

View HTML report:
```bash
open htmlcov/index.html
```

## Intent Detection Tests

### Unit Tests (`test_intent_detection.py`)

**TestIntentDetectionNodeUnit:**
- `test_empty_message_returns_error` - Validates error handling for empty messages
- `test_successful_intent_detection` - Tests successful intent detection with mocked Ollama
- `test_intent_detection_with_exception` - Tests exception handling
- `test_valid_intents_returned` - Validates all 8 intent categories

**TestDetectIntentWithOllamaUnit:**
- `test_successful_ollama_call` - Mocked successful API call
- `test_non_200_response_returns_unclear` - HTTP error handling
- `test_invalid_intent_returns_unclear` - Invalid intent validation
- `test_timeout_exception_returns_unclear` - Timeout handling
- `test_multiline_response_takes_first_line` - Response parsing

**TestGetIntentDescription:**
- `test_all_intents_have_descriptions` - Validates utility function
- `test_unknown_intent_returns_default` - Default case handling

### Integration Tests

**TestIntentDetectionIntegration:**
- `test_real_ollama_greeting_detection` - Real API call with greeting
- `test_real_ollama_venting_detection` - Real API call with venting
- `test_real_ollama_seek_information_detection` - Real API call for info seeking
- `test_real_ollama_seek_support_detection` - Real API call for support seeking
- `test_real_ollama_with_full_node` - End-to-end node test
- `test_real_ollama_timeout_handling` - Timeout behavior validation

**Requirements for Integration Tests:**
- Ollama must be running at `http://194.164.151.158:11434`
- Model `phi3:latest` must be available
- Network connectivity to Ollama server

## Test Configuration

### pytest.ini
Configuration for pytest behavior, markers, and coverage options.

### Environment Variables
```bash
# Override Ollama configuration for testing
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="phi3:latest"
export OLLAMA_TIMEOUT="120"
```

## Writing New Tests

### Unit Test Template
```python
@pytest.mark.asyncio
async def test_my_feature(sample_state):
    """Test description."""
    with patch('module.function') as mock_func:
        mock_func.return_value = "expected_value"
        
        result = await my_node(sample_state)
        
        assert result["key"] == "expected_value"
```

### Integration Test Template
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_my_real_api_call():
    """Test description."""
    result = await real_api_function("input")
    
    assert result in ["valid", "values"]
```

## Continuous Integration

**GitHub Actions example:**
```yaml
- name: Run unit tests
  run: pytest tests/ -v -m "not integration"

- name: Run integration tests
  run: pytest tests/ -v -m integration
  if: github.event_name == 'push'
```

## Troubleshooting

**ImportError: No module named 'graph'**
```bash
# Make sure to run from project root
cd /path/to/sb-backend
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Integration tests fail:**
- Verify Ollama is running: `curl http://194.164.151.158:11434/api/tags`
- Check model is available: `ollama list | grep phi3`
- Verify network connectivity

**Async warnings:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

## Best Practices

1. **Write unit tests first** - Fast feedback loop during development
2. **Mock external dependencies** - Keep unit tests isolated and fast
3. **Use fixtures** - Share common test data and setup
4. **Parametrize tests** - Test multiple cases efficiently
5. **Run integration tests before deployment** - Validate real API behavior
6. **Aim for >80% coverage** - Ensure code quality
7. **Test edge cases** - Empty inputs, timeouts, errors
8. **Keep tests independent** - Tests should not depend on each other

## Next Steps

- Add tests for `situation_severity_detection.py`
- Add tests for `response_generator.py`
- Add integration tests for full graph execution
- Add performance/load tests
- Set up CI/CD pipeline with automated testing
