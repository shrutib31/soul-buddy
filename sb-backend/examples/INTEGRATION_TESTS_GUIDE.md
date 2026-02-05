# Running Integration Tests

## Prerequisites

Before running integration tests, ensure:

1. **Ollama is running** at the configured URL
2. **Model is available** (phi3:latest)
3. **Network connectivity** to Ollama server

## Verify Prerequisites

```bash
# Check if Ollama is accessible
curl http://194.164.151.158:11434/api/tags

# Expected response: JSON with available models including phi3:latest
```

## Run Integration Tests

### Option 1: Using Test Runner Script
```bash
./run_tests.sh integration
```

### Option 2: Direct pytest Command
```bash
# Run all integration tests for intent_detection
pytest tests/graph/nodes/test_intent_detection.py -v -m integration

# Run specific integration test
pytest tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_greeting_detection -v

# Run with detailed output
pytest tests/graph/nodes/test_intent_detection.py -v -m integration -s
```

## Expected Output

```
============================================== test session starts ==============================================
platform darwin -- Python 3.11.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/shrutibasu/workspace/vscode-ws/soul-buddy/sb-backend
configfile: pytest.ini
plugins: anyio-4.12.1, langsmith-0.6.8, asyncio-1.3.0
collected 25 items / 19 deselected / 6 selected

tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_greeting_detection PASSED [ 16%]
tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_venting_detection PASSED [ 33%]
tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_seek_information_detection PASSED [ 50%]
tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_seek_support_detection PASSED [ 66%]
tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_with_full_node PASSED [ 83%]
tests/graph/nodes/test_intent_detection.py::TestIntentDetectionIntegration::test_real_ollama_timeout_handling PASSED [100%]

======================================= 6 passed, 19 deselected in 45.23s ========================================
```

## Integration Test Details

### Test Cases

1. **test_real_ollama_greeting_detection**
   - Message: "Hello"
   - Expected: "greeting" or "unclear"
   - Validates: Ollama responds correctly to greetings

2. **test_real_ollama_venting_detection**
   - Message: "I'm so frustrated with everything right now"
   - Expected: "venting" or "unclear"
   - Validates: Ollama detects emotional venting

3. **test_real_ollama_seek_information_detection**
   - Message: "How do I deal with stress?"
   - Expected: "seek_information", "seek_understanding", or "unclear"
   - Validates: Ollama detects information-seeking

4. **test_real_ollama_seek_support_detection**
   - Message: "I need someone to talk to"
   - Expected: "seek_support" or "unclear"
   - Validates: Ollama detects support-seeking

5. **test_real_ollama_with_full_node**
   - Message: "I'm feeling really anxious today"
   - Expected: Valid intent in state
   - Validates: Full node execution with real API

6. **test_real_ollama_timeout_handling**
   - Timeout: 0.001 seconds (artificially low)
   - Expected: Returns "unclear" without crashing
   - Validates: Graceful timeout handling

## Troubleshooting

### Ollama Not Accessible
```bash
# Error: Connection refused
# Solution: Verify Ollama is running
curl http://194.164.151.158:11434/api/tags

# If not running, start Ollama service
# (depends on your deployment)
```

### Model Not Available
```bash
# Error: Model phi3:latest not found
# Solution: Check available models
curl http://194.164.151.158:11434/api/tags

# If phi3:latest is missing, pull it
ollama pull phi3:latest
```

### Timeout Errors
```bash
# Error: Tests timing out
# Solution: Increase timeout in environment
export OLLAMA_TIMEOUT=300  # 5 minutes

# Then run tests again
pytest tests/graph/nodes/test_intent_detection.py -v -m integration
```

### Network Issues
```bash
# Error: Network unreachable
# Solution: Check network connectivity
ping 194.164.151.158

# Test HTTP access
curl -v http://194.164.151.158:11434/api/tags
```

## Configuration

Override configuration for integration tests:

```bash
# Use different Ollama instance
export OLLAMA_BASE_URL="http://localhost:11434"

# Use different model
export OLLAMA_MODEL="llama2:latest"

# Increase timeout
export OLLAMA_TIMEOUT=180

# Run integration tests with custom config
pytest tests/graph/nodes/test_intent_detection.py -v -m integration
```

## Performance Expectations

- **Single test:** 5-10 seconds
- **All 6 integration tests:** 30-60 seconds
- **With slow network:** Up to 120 seconds

Time breakdown:
- Network latency: 0.5-2s per request
- Ollama inference: 3-8s per request
- Test overhead: <1s total

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Check Ollama availability
        run: |
          curl -f http://194.164.151.158:11434/api/tags || exit 0
      
      - name: Run integration tests
        run: |
          pytest tests/ -v -m integration
        env:
          OLLAMA_BASE_URL: http://194.164.151.158:11434
          OLLAMA_MODEL: phi3:latest
          OLLAMA_TIMEOUT: 120
```

## Best Practices

1. **Run unit tests first** - Fast feedback (0.31s)
2. **Run integration tests before deployment** - Validate real behavior
3. **Use CI/CD for integration tests** - Automated validation
4. **Monitor integration test performance** - Detect API degradation
5. **Keep integration tests separate** - Don't slow down development cycle

## Viewing Logs

Enable detailed logging for debugging:

```bash
# Run with verbose logging
pytest tests/graph/nodes/test_intent_detection.py -v -m integration -s --log-cli-level=DEBUG

# Output will show:
# - HTTP requests to Ollama
# - Response status codes
# - Response bodies
# - Timeout settings
# - Error messages
```

## Skip Integration Tests

If Ollama is not available, skip integration tests:

```bash
# Run only unit tests
pytest tests/ -v -m "not integration"

# Or use the test runner
./run_tests.sh unit
```

## Next Steps

After successful integration tests:

1. ✅ Verify all 6 integration tests pass
2. 📊 Review test execution times
3. 🔍 Check for any flaky tests
4. 📝 Document any unexpected behavior
5. 🚀 Proceed with deployment
