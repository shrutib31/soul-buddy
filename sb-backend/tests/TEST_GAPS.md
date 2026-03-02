# SoulBuddy Backend – Test Gaps

This document summarizes **what is tested** vs **what is not** in `sb-backend`, and recommends priorities for closing gaps.

---

## Current test suite

| Area | Test file | What’s covered |
|------|-----------|----------------|
| **Graph nodes** | `tests/graph/nodes/test_classification_node.py` | `classification_node`, `get_classifications`, `detect_greeting`, `detect_crisis`, `is_true_negation`, intent/situation/severity labels |
| **Graph nodes** | `tests/graph/nodes/test_guardrail.py` | `guardrail_node`, `guardrail_router`, `call_guardrail_llm`, `safe_json_loads`, config constants |
| **Graph nodes** | `tests/graph/nodes/test_response_generator.py` | `response_generator_node`, `generate_response_ollama`, `generate_response_gpt` (mocked LLM) |
| **Graph nodes** | `tests/graph/nodes/test_response_evaluator.py` | `score_response`, `select_best_response`, scoring helpers |
| **Graph nodes** | `tests/graph/nodes/test_response_templates.py` | `get_template_response`, greeting/crisis template logic |
| **Graph nodes** | `tests/graph/nodes/test_function_nodes.py` | `conv_id_handler_node`, `store_message_node`, `store_bot_response_node`, `render_node` (mocked DB) |
| **Graph** | `tests/graph/test_graph_builder.py` | `get_compiled_flow()`: nodes, edges, entry point |
| **Graph** | `tests/graph/test_graph_integration.py` | Full graph invoke (conv_id → … → render) with mocked DB and LLM (`@pytest.mark.integration`) |
| **API** | `tests/api/test_chat.py` | `create_initial_state`, POST `/api/v1/chat/incognito` (mocked flow) |
| **API** | `tests/api/test_classify.py` | POST `/api/v1/classify` (mocked get_classifications), validation, error handling |

- **Unit tests**: run with `pytest tests/ -m "not integration"` (109 unit tests as of last update).
- **Integration tests**: run with `pytest tests/ -m integration` (e.g. full graph with mocked I/O, and existing Ollama-based intent/guardrail tests if Ollama is up).
- **pytest-cov** is in dev deps; use `pytest tests/ --cov=graph --cov=api --cov-report=term-missing` to track coverage.

---

## Critical gaps (used in production flow but untested)

The compiled LangGraph now uses **classification_node** as the main entry point (not intent_detection_node). The current test suite covers the classification-based flow, but gaps remain:

| Component | File(s) | Gap |
|-----------|---------|-----|
| **Classification node (model path)** | `graph/nodes/agentic_nodes/classification_node.py` | Real model loading (non-mocked) is not tested; only greeting/crisis/empty cases are covered. |
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

**Intent detection vs classification (important distinction):**

- **`intent_detection_node`** (in `intent_detection.py`) is no longer used or imported in the graph; its test file has been removed.
- **`classification_node`** (in `classification_node.py`) is now the main entry point in the graph and is covered by unit tests, along with its helpers (`get_classifications`, `detect_greeting`, `detect_crisis`, etc.).

The main gap is no longer classification node coverage, but rather real model/LLM/DB integration and streaming, as detailed above.

So the main gap is **classification_node** and its helpers, not intent_detection.

---

## Recommended priorities

1. **High**
   - Add unit tests for **classification_node**: `classification_node`, `get_classifications`, `detect_crisis`, `detect_greeting` (mock model and tokenizer where needed).
   - Add unit tests for **response_generator_node** (mock LLM/OpenAI).
   - Add tests for **conv_id_handler_node**, **store_message_node**, **store_bot_response_node**, **render_node** (mock DB/SQLAlchemy).
   - Add a **graph structure test** that builds the graph and asserts nodes/edges (e.g. entry point, classification → response_generator → store_bot_response → render → END).

2. **Medium**
   - Add tests for **api/chat.py** (e.g. `create_initial_state`, and optionally chat/stream endpoints with TestClient).
   - Add tests for **api/classify.py** (e.g. request/response with mocked `get_classifications`).
   - Add **pytest-cov** to dev dependencies and run coverage (e.g. `--cov=graph --cov=api`) to track progress.

3. **Lower**
   - Situation/severity node if/when re-enabled.
   - Config/ORM/seed/scripts (can stay as integration/manual or later coverage).

---

## Quick reference: graph flow (current)

```
conv_id_handler → store_message ─┐
conv_id_handler → classification_node ─→ response_generator → store_bot_response → render → END
```

- **Tested in this path:** none of these nodes (only guardrail and intent_detection, which are off or not in this path).
- **Guardrail** is in the codebase and tested but not wired in the current graph (edges commented out).
