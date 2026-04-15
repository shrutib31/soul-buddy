# DeepEval tests

Run these commands from sb-backend

## Install

`deepeval` is listed in `requirements.txt`, so install the normal backend
requirements in your active environment:

```bash
uv pip install -r requirements.txt
```

## Quick checks

Run everything in this folder:

```bash
uv run pytest tests/DeepEval -q
```

Run only the deterministic crisis-routing checks:

```bash
uv run pytest tests/DeepEval/test_crisis_detection.py -q
```

If `deepeval` or `OPENAI_API_KEY` is not configured, the response-quality tests
skip the live LLM judging path. The fixture and crisis checks still run.

## Live response-quality evals

DeepEval `GEval` needs `OPENAI_API_KEY` for the evaluator model. The SoulBuddy
response generator also needs at least one backend enabled.

OpenAI-only generation:

```bash
export OPENAI_API_KEY="..."
export OPENAI_FLAG="true"
uv run pytest tests/DeepEval/test_response_quality.py -q
```

Ollama generation with OpenAI as the DeepEval judge:

```bash
export OPENAI_API_KEY="..."
export OLLAMA_FLAG="true"
export OLLAMA_BASE_URL="http://localhost:11434"
uv run pytest tests/DeepEval/test_response_quality.py -q
```

Use `COMPARE_RESULTS="true"` instead when you want the response generator to
call both Ollama and OpenAI and choose the best draft.

## Files

- `metrics.py`: DeepEval `GEval` rubrics and thresholds.
- `test_response_quality.py`: empathy and appropriateness response evals.
- `test_crisis_detection.py`: deterministic crisis-detection regression checks.
- `test_cases/*.json`: labeled eval cases.
