# SoulBuddy Backend – Test Gaps

This document summarizes **what is tested** vs **what is not** in `sb-backend`, and recommends priorities for closing gaps.

---

## Current test suite

| Area | Test file | What’s covered |
|------|-----------|----------------|
| **Graph nodes** | `tests/graph/nodes/test_guardrail.py` | `guardrail_node`, `guardrail_router`, `call_guardrail_llm`, `safe_json_loads`, config constants |
| **Graph nodes** | `tests/graph/nodes/test_classification_node.py` | `detect_greeting`, `is_true_negation`, `detect_crisis`, `get_classifications` (empty/greeting/crisis), `classification_node` (mocked get_classifications) |
| **Graph nodes** | `tests/graph/nodes/test_response_generator.py` | `response_generator_node`, `generate_response_ollama`, `generate_response_gpt` (mocked LLM; template short-circuit for crisis/greeting) |
| **Graph nodes** | `tests/graph/nodes/test_response_templates.py` | `get_template_response`, `_HIGH_RISK_TEMPLATES`, `_GREETING_TEMPLATES` (crisis priority, domain routing, None fallback, pool completeness) |
| **Graph nodes** | `tests/graph/nodes/test_response_evaluator.py` | `score_response`, `select_best_response`, all scoring sub-functions (`_empathy_score`, `_engagement_score`, `_length_score`, `_completeness_score`, `_repetition_penalty`, `_robotic_penalty`) |
| **Graph nodes** | `tests/graph/nodes/test_function_nodes.py` | `conv_id_handler_node`, `store_message_node`, `store_bot_response_node`, `render_node` (mocked DB) |
| **Graph** | `tests/graph/test_graph_builder.py` | `get_compiled_flow()`: nodes, edges, entry point |
| **Graph** | `tests/graph/test_graph_integration.py` | Full graph invoke (conv_id → … → render) with mocked DB and LLM (`@pytest.mark.integration`) |
| **API** | `tests/api/test_chat.py` | `create_initial_state`, POST `/api/v1/chat/incognito` (mocked flow) |
| **API** | `tests/api/test_classify.py` | POST `/api/v1/classify` (mocked get_classifications), validation, error handling |

- **Unit tests**: run with `pytest tests/ -m "not integration"` (count grows as new tests are added; see file list above for current coverage).
- **Integration tests**: run with `pytest tests/ -m integration` (e.g. full graph with mocked I/O, and existing Ollama-based intent/guardrail tests if Ollama is up).
- **pytest-cov** is in dev deps; use `pytest tests/ --cov=graph --cov=api --cov-report=term-missing` to track coverage.

---

## Critical gaps (used in production flow but untested)

The compiled LangGraph uses **classification_node**, not **intent_detection_node**. Classification node and the rest of the main path are now covered by unit tests (see Current test suite above). Remaining gaps:

| Component | File(s) | Gap |
|-----------|---------|-----|
| **Classification node (model path)** | `graph/nodes/agentic_nodes/classification_node.py` | When message is not greeting and not crisis, `get_classifications` loads the real model; unit tests mock this or use greeting/crisis/empty. No test with real model without mocking. |
| **Response generator (real LLM)** | `graph/nodes/agentic_nodes/response_generator.py` | Real Ollama/GPT calls are not tested; only mocked. |
| **Conv ID / store message / store bot (real DB)** | function nodes | Real DB writes are not tested; only mocked sessions. |
| **Streaming** | `graph/streaming.py` | `stream_response_words`, `stream_as_sse` – no tests. |
| **Config / ORM / seed** | various | No unit tests (would need mocks or test containers). |

---

## Other gaps (supporting code, not yet tested)

| Area | Files / modules | Notes |
|------|------------------|--------|
| **Situation / severity** | `graph/nodes/agentic_nodes/situation_severity_detection.py` | Node is commented out in graph but code exists; README already lists it as a gap. |
| **Streaming** | `graph/streaming.py` | `stream_response_words`, `stream_as_sse` – no tests. |
| **State** | `graph/state.py` | `ConversationState` – only used as fixture input in existing tests; no dedicated state tests. |
| **Config** | `config/database.py`, `config/auth_database.py`, `config/sqlalchemy_db.py`, `config/supabase.py`, `config/logging_config.py` | Connection, pool, auth, Supabase – no unit tests (would need mocks or test containers). |
| **ORM** | `orm/models.py`, `orm/base.py`, and other `orm/*.py` | Models and DB mapping – no tests. |
| **Seed** | `seed/seed_config.py` | Seed functions – no tests. |
| **Server** | `server.py` | App lifespan, router mounting – no tests. |
| **Scripts** | `scripts/init_db.py`, `scripts/cleanup_db.py` | DB init/cleanup – no tests. |

---

## Intent detection vs classification (important distinction)

- **`intent_detection_node`** (in `intent_detection.py`) is **not tested** and **not used** in the current graph; `test_intent_detection.py` was removed when the flow switched to classification.
- **`classification_node`** (in `classification_node.py`) **is used** in the graph and calls `get_classifications` (model + `detect_crisis` / `detect_greeting`). It is now **covered by unit tests** (`test_classification_node.py`).

The main remaining gap is the **classification_node model path** (real model without mocking) and real LLM/DB calls.

---

## Recommended priorities

1. **High** *(previously open, now resolved)*
   - ✅ Unit tests for **classification_node**: `classification_node`, `get_classifications`, `detect_crisis`, `detect_greeting` (`test_classification_node.py`).
   - ✅ Unit tests for **response_generator_node** with mocked LLM (`test_response_generator.py`).
   - ✅ Unit tests for **response_templates** and **response_evaluator** (`test_response_templates.py`, `test_response_evaluator.py`).
   - ✅ Tests for **conv_id_handler_node**, **store_message_node**, **store_bot_response_node**, **render_node** (`test_function_nodes.py`).
   - ✅ **Graph structure test** for nodes/edges (`test_graph_builder.py`).

2. **Remaining high-priority gaps**
   - Add tests for the **classification_node real-model path** (no mocking of `get_classifications`).
   - Add tests for real Ollama/GPT calls in `response_generator` (integration, requires live service).
   - Add real-DB integration tests for function nodes.

3. **Medium**
   - ✅ Tests for **api/chat.py** and **api/classify.py** (`test_chat.py`, `test_classify.py`).
   - Add **pytest-cov** to dev dependencies and run coverage (e.g. `--cov=graph --cov=api`) to track progress.

4. **Lower**
   - Situation/severity node if/when re-enabled.
   - Config/ORM/seed/scripts (can stay as integration/manual or later coverage).

---

## Quick reference: graph flow (current)

```
conv_id_handler → store_message ─┐
conv_id_handler → classification_node ─→ response_generator → store_bot_response → render → END
```

- **Tested in this path:** `conv_id_handler_node`, `store_message_node`, `classification_node`, `response_generator_node` (with template logic via `response_templates.py` and scoring via `response_evaluator.py`), `store_bot_response_node`, `render_node` — all covered by unit tests with mocked DB/LLM.
- **Guardrail** is in the codebase and tested but not wired in the current graph (edges commented out).
