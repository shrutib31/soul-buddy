# SoulBuddy Backend — Developer Design & Architecture

> This document is the technical deep-dive for developers working on or extending the SoulBuddy backend. For setup and operations, see [README.md](../README.md) and [README_DOCKER.md](../README_DOCKER.md).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Map](#2-module-map)
3. [Request Lifecycle](#3-request-lifecycle)
4. [LangGraph Pipeline](#4-langgraph-pipeline)
5. [ConversationState — The Data Contract](#5-conversationstate--the-data-contract)
6. [Node Reference](#6-node-reference)
7. [Classification System](#7-classification-system)
8. [Response Generation](#8-response-generation)
9. [Data Layer](#9-data-layer)
10. [Cache Architecture](#10-cache-architecture)
11. [Encryption](#11-encryption)
12. [Authentication](#12-authentication)
13. [Configuration System](#13-configuration-system)
14. [Streaming (SSE)](#14-streaming-sse)
15. [Key Design Decisions](#15-key-design-decisions)
16. [Extension Guide](#16-extension-guide)

---

## 1. System Overview

SoulBuddy is an **emotional support chatbot API**. The backend is a FastAPI service that runs every user message through a LangGraph state-machine pipeline. The pipeline classifies the message (greeting, crisis, intent, situation, severity) and generates a contextual response via one or two LLMs (Ollama and/or GPT-4o-mini).

```
┌─────────────────────────────────────────────────────────────────────┐
│  Client (React frontend / API consumer)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────────────┐
│  FastAPI  (server.py)                                                │
│  ┌─────────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  api/chat.py        │  │ api/classify │  │ api/guardrail       │ │
│  └──────────┬──────────┘  └──────────────┘  └─────────────────────┘ │
└─────────────┼───────────────────────────────────────────────────────┘
              │ ConversationState
┌─────────────▼───────────────────────────────────────────────────────┐
│  LangGraph Pipeline  (graph/graph_builder.py)                        │
│                                                                      │
│  conv_id_handler ──► load_user_context ──► [parallel: store_message │
│                                             + out_of_scope_node]     │
│                             │                                        │
│                    out_of_scope_router                               │
│                    /                \                                │
│               render            classification_node                  │
│                                       │                             │
│                                 response_generator                   │
│                                       │                             │
│                                 store_bot_response                   │
│                                       │                             │
│                                    render ──► END                    │
└─────────────────────────────────────────────────────────────────────┘
              │                   │                    │
┌─────────────▼───┐  ┌────────────▼──────────┐  ┌────▼──────────────┐
│  Data DB        │  │  Auth DB              │  │  Redis Cache       │
│  (soulbuddy)    │  │  (souloxy-db)         │  │                    │
│  - conversations│  │  - users              │  │  - profiles        │
│  - turns        │  │  - personality profiles│  │  - conv history    │
│  - summaries    │  │  - detailed profiles  │  │  - summaries       │
└─────────────────┘  └───────────────────────┘  └────────────────────┘
```

**Tech Stack:**

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI 0.115+ |
| Async server | Uvicorn (1 worker, concurrency 20) |
| Pipeline orchestration | LangGraph 0.2+ |
| Primary LLM | Ollama (llama3.2, self-hosted) |
| Secondary LLM | OpenAI GPT-4o-mini |
| ML classification | SoulBuddyClassifier (BERT-base fine-tuned, PyTorch) |
| Data DB driver | asyncpg + SQLAlchemy 2 async |
| Auth DB driver | asyncpg + SQLAlchemy 2 async |
| Cache | Redis via aioredis |
| Auth | Supabase JWT |
| Encryption | AES-256-GCM (HKDF key derivation, optional GCP KMS) |

---

## 2. Module Map

```
sb-backend/
├── server.py                    # App entry point, lifespan, router registration
│
├── api/                         # HTTP boundary — thin handlers, Pydantic validation
│   ├── chat.py                  # Chat, streaming, conversation history endpoints
│   ├── classify.py              # Standalone classification endpoint
│   ├── guardrail.py             # Standalone out-of-scope detection endpoint
│   └── supabase_auth.py         # FastAPI dependencies: optional_user / require_user
│
├── graph/                       # LangGraph pipeline
│   ├── graph_builder.py         # Assembles and compiles the graph (singleton `flow`)
│   ├── state.py                 # ConversationState Pydantic model (the data contract)
│   ├── streaming.py             # SSE event emitter (wraps graph.astream)
│   └── nodes/
│       ├── function_nodes/      # Deterministic: DB I/O, state mutations, rendering
│       │   ├── conv_id_handler.py      # Conversation ID creation / validation
│       │   ├── load_user_context.py    # Cache-aside context loader
│       │   ├── store_message.py        # Persist user turn (encrypted if enabled)
│       │   ├── store_bot_response.py   # Persist bot turn + upsert summary
│       │   ├── get_messages.py         # Retrieve + decrypt conversation history
│       │   ├── out_of_scope.py         # Fast pattern-based OOS check
│       │   └── render.py               # Build final api_response dict
│       └── agentic_nodes/       # LLM/ML-driven decisions
│           ├── classification_node.py  # 7-level classification hierarchy
│           ├── response_generator.py   # Template or LLM response
│           ├── response_evaluator.py   # Score + select best response
│           ├── guardrail.py            # Post-generation safety check
│           └── response_templates.py   # Hardcoded templates (crisis/greeting/OOS)
│
├── orm/                         # SQLAlchemy 2 ORM models
│   ├── base.py                  # DeclarativeBase for data DB
│   ├── auth_base.py             # DeclarativeBase for auth DB
│   ├── models.py                # Data DB: SbConversation, ConversationTurn, UserConversationSummary
│   └── auth_models.py           # Auth DB: AuthUser, UserPersonalityProfile, UserDetailedProfile
│
├── services/                    # Shared business logic (no graph awareness)
│   ├── cache_service.py         # Redis wrapper: cache_service singleton, fail-silent
│   └── key_manager.py           # AES-256-GCM encryption: key_manager singleton
│
├── config/                      # Configuration and connection singletons
│   ├── settings.py              # Central AppSettings dataclass — the only settings source
│   ├── sqlalchemy_db.py         # SQLAlchemyDataDB / SQLAlchemyAuthDB singletons
│   ├── database.py              # asyncpg pool singleton (data DB)
│   ├── auth_database.py         # asyncpg pool singleton (auth DB)
│   ├── redis.py                 # RedisConfig singleton + reconnect loop
│   ├── supabase.py              # Supabase client singleton
│   └── logging_config.py        # YAML-based logging setup
│
├── transformer_models/          # Custom PyTorch model definition
│   └── SoulBuddyClassifier.py   # Multi-task BERT classifier (4 heads)
│
├── scripts/                     # One-off admin scripts (not imported by app)
│   ├── init_db.py               # Create tables, seed domain_config rows
│   └── cleanup_db.py            # Drop all tables
│
├── migrations/                  # Alembic migration scripts
├── tests/                       # Unit tests (no live external dependencies)
└── conftest.py                  # Pytest root config, CI stubs (google.cloud.kms)
```

---

## 3. Request Lifecycle

### 3.1 Non-streaming chat (`POST /api/v1/chat`)

```
1. HTTP request arrives at FastAPI

2. api/chat.py: chat()
   a. Pydantic validates ChatRequest
   b. If is_incognito=false: supabase_auth.get_optional_user() extracts supabase_uid from JWT
   c. Builds initial ConversationState (conversation_id="", mode=incognito/cognito, ...)
   d. Calls: final_state = await flow.ainvoke(state.model_dump())
   e. Returns: final_state["api_response"]

3. LangGraph graph execution (see §4)

4. HTTP response: the api_response dict
```

### 3.2 Streaming chat (`POST /api/v1/chat/stream`)

```
1–3: Same as above up to graph invocation

4. api/chat.py: chat_stream()
   - Returns StreamingResponse(content=stream_as_sse(...), media_type="text/event-stream")

5. graph/streaming.py: stream_as_sse()
   - Calls flow.astream(state, stream_mode="updates")
   - After each node update yields: data: {"event": "node_complete", "node": "<name>", ...}
   - When render node is reached, yields final response event
   - On error yields: data: {"event": "error", "detail": "..."}
```

### 3.3 Conversation history (`GET /api/v1/chat/conversations/{id}/messages`)

```
1. Requires Authorization: Bearer <supabase-token>
2. supabase_auth.require_user() validates token → extracts supabase_uid
3. api/chat.py:
   a. Queries SbConversation by id, verifies supabase_user_id matches caller
   b. Fetches ConversationTurn rows ordered by turn_index
   c. Calls key_manager.decrypt(conversation_id, message) on each turn
   d. Returns decrypted messages list
```

---

## 4. LangGraph Pipeline

### 4.1 Graph topology

```python
# graph/graph_builder.py (abbreviated)

builder = StateGraph(ConversationState)

# Nodes
builder.add_node("conv_id_handler",       conv_id_handler_node)
builder.add_node("load_user_context",     load_user_context_node)
builder.add_node("store_message",         store_message_node)
builder.add_node("out_of_scope",          out_of_scope_node)
builder.add_node("classification_node",   classification_node)
builder.add_node("response_generator",    response_generator_node)
builder.add_node("store_bot_response",    store_bot_response_node)
builder.add_node("render",                render_node)

# Sequential backbone
builder.set_entry_point("conv_id_handler")
builder.add_edge("conv_id_handler", "load_user_context")

# Parallel fan-out after context load
builder.add_edge("load_user_context", "store_message")
builder.add_edge("load_user_context", "out_of_scope")

# Conditional routing after OOS check
builder.add_conditional_edges(
    "out_of_scope",
    out_of_scope_router,           # returns "render" or "classification_node"
    {"render": "render", "classification_node": "classification_node"}
)

# Main pipeline
builder.add_edge("classification_node",   "response_generator")
builder.add_edge("response_generator",    "store_bot_response")
builder.add_edge("store_bot_response",    "render")
builder.add_edge("render",                END)

flow = builder.compile()          # compiled once at module import
```

### 4.2 Node return contract

Every node is an async function with signature:

```python
async def some_node(state: ConversationState) -> dict[str, Any]:
    # ... do work ...
    return {"field_to_update": new_value, ...}
```

LangGraph merges the returned dict into the existing state. Nodes must only return keys they intend to modify — unset keys are preserved.

The `error` field uses a custom merge reducer (`_keep_last_error`) so errors from parallel nodes don't overwrite each other silently.

### 4.3 The `out_of_scope_router`

```python
def out_of_scope_router(state: ConversationState) -> str:
    if state.is_out_of_scope:
        return "render"          # skip classification + generation entirely
    return "classification_node"
```

This is a **performance optimisation**: out-of-scope messages bypass the ML model and LLM calls entirely.

---

## 5. ConversationState — The Data Contract

`graph/state.py` — `ConversationState(BaseModel)`

All state is passed through every node. Understanding the fields is essential for tracing bugs.

```
┌─────────────────────────────────────────────────────────┐
│  ConversationState                                       │
│                                                         │
│  Identification                                         │
│    conversation_id: str       ← set by conv_id_handler  │
│    mode: str                  ← "incognito" | "cognito" │
│    supabase_uid: str | None   ← from JWT (cognito only) │
│    user_id: str | None        ← internal DB user id     │
│                                                         │
│  User Input (set at API layer, never mutated)           │
│    user_message: str                                    │
│    domain: str                ← student|employee|general│
│    chat_preference: str       ← general|focused|deep    │
│    chat_mode: str             ← default|reflection|...  │
│                                                         │
│  Context (set by load_user_context)                     │
│    conversation_history: List[Dict]  ← last N turns     │
│    conversation_summary: str | None  ← rolling summary  │
│    user_personality_profile: Dict                       │
│    user_preferences: Dict                               │
│    page_context: Dict                                   │
│    domain_config: Dict                                  │
│                                                         │
│  Classification (set by classification_node)            │
│    intent: str | None         ← venting|seek_support|...│
│    situation: str | None      ← EXAM_ANXIETY|SUICIDAL|..│
│    severity: str | None       ← low|medium|high         │
│    risk_level: str            ← low|medium|high|critical│
│    is_crisis_detected: bool                             │
│    is_greeting: bool                                    │
│    is_out_of_scope: bool                                │
│    out_of_scope_reason: str | None                      │
│                                                         │
│  Generation (set by response_generator)                 │
│    response_draft: str        ← final bot text          │
│                                                         │
│  Output (set by render)                                 │
│    api_response: Dict | None  ← returned to caller      │
│                                                         │
│  Internal / Bookkeeping                                 │
│    error: str | None          ← last error (merged)     │
│    guardrail_status: str | None                         │
│    guardrail_feedback: str | None                       │
│    attempt: int               ← retry counter           │
│    step_index: int            ← streaming step counter  │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Node Reference

### 6.1 `conv_id_handler` (function node)

**File:** `graph/nodes/function_nodes/conv_id_handler.py`

**Responsibility:** Ensure a valid `conversation_id` exists before any other node runs.

| Scenario | Behaviour |
|----------|-----------|
| `conversation_id` is empty | Generate new UUID4 |
| Incognito, existing ID | Check `started_at` — expire if >24 h |
| Cognito, existing ID | Query `SbConversation` table, verify `supabase_user_id` matches |
| Cognito, new ID | Insert new `SbConversation` row |

**Returns:** `{"conversation_id": "..."}`

---

### 6.2 `load_user_context` (function node)

**File:** `graph/nodes/function_nodes/load_user_context.py`

**Responsibility:** Populate all context fields in state from Redis (or DB on cache miss).

**Cache-aside pattern per field:**

```
1. cache_service.get_X(user_id)
2. If None → query Auth DB or Data DB
3. cache_service.set_X(user_id, result)
4. Return result (or empty default)
```

**Fields loaded:**

| State field | Cache key | DB source |
|-------------|-----------|-----------|
| `user_personality_profile` | `user:<id>:personality_profile` | `user_personality_profiles` (auth DB) |
| `user_preferences` | `user:<id>:profile` | `user_detailed_profiles` (auth DB) |
| `conversation_summary` | `user:<id>:conversation_summary` | `user_conversation_summaries` (data DB) |
| `conversation_history` | `conv:<id>:history` | `conversation_turns` (data DB), decrypted |
| `page_context` / `ui_state` | `user:<id>:ui_state` | `page_context` field from state |
| `domain_config` | `config:<domain>` | `domain_config` table (data DB) — stub |

**Incognito mode:** skips all user-specific loads (no `supabase_uid`). Loads only `domain_config`.

**Returns:** All loaded context fields as a dict.

---

### 6.3 `store_message` (function node)

**File:** `graph/nodes/function_nodes/store_message.py`

Inserts a `ConversationTurn` row with `speaker="user"`. Encrypts the message text if `ENCRYPTION_ENABLED=true` using `key_manager.encrypt(conversation_id, text)`.

After insert: calls `cache_service.invalidate_conversation_history(conversation_id)` to ensure the next fetch reflects the new turn.

**Returns:** `{}` (side-effect only node)

---

### 6.4 `out_of_scope` (function node)

**File:** `graph/nodes/function_nodes/out_of_scope.py`

Fast pattern-based detection. No LLM call. Detects explicit requests for off-domain tasks (cooking, coding, legal, financial, travel, entertainment, homework, math, weather/news/sports).

**Critical nuance:** Only flags explicit bot requests like "give me a recipe" or "write code for X". Personal narratives that *mention* off-domain topics ("I was coding all night and I'm anxious") are **in-scope** — they are valid wellness context.

Detection algorithm:
1. Check for REQUEST_VERBS (give, explain, help me with, how do I, write, create…)
2. Check for OFF_DOMAIN_TOPIC patterns
3. If both match → out-of-scope

**Returns:** `{"is_out_of_scope": bool, "out_of_scope_reason": str | None}`

---

### 6.5 `classification_node` (agentic node)

**File:** `graph/nodes/agentic_nodes/classification_node.py`

See [§7 Classification System](#7-classification-system) for full detail.

**Returns:**
```python
{
    "intent": "venting",
    "situation": "EXAM_ANXIETY",
    "severity": "medium",
    "risk_level": "low",
    "is_crisis_detected": False,
    "is_greeting": False,
}
```

---

### 6.6 `response_generator` (agentic node)

**File:** `graph/nodes/agentic_nodes/response_generator.py`

See [§8 Response Generation](#8-response-generation) for full detail.

**Returns:** `{"response_draft": "..."}`

---

### 6.7 `store_bot_response` (function node)

**File:** `graph/nodes/function_nodes/store_bot_response.py`

1. Inserts `ConversationTurn(speaker="bot")`. Encrypts if enabled.
2. In cognito mode: builds and upserts `UserConversationSummary`.

**Summary format:**
```
[2026-04-03] Domain: student. Situation: EXAM_ANXIETY (medium severity).
Intent: venting. Risk: low. Turns: 6.
```

3. Invalidates `conv:<id>:history` and `user:<id>:conversation_summary` caches.

**Returns:** `{"conversation_summary": "..."}` (cognito) or `{}` (incognito)

---

### 6.8 `render` (function node)

**File:** `graph/nodes/function_nodes/render.py`

Builds the `api_response` dict returned to the caller.

```json
{
  "success": true,
  "conversation_id": "<uuid>",
  "mode": "incognito",
  "domain": "student",
  "response": "That sounds really tough...",
  "metadata": {
    "intent": "venting",
    "situation": "EXAM_ANXIETY",
    "severity": "medium",
    "risk_level": "low",
    "timestamp": "2026-04-03T10:00:00Z"
  }
}
```

**Returns:** `{"api_response": {...}}`

---

## 7. Classification System

The classification node runs a **7-level hierarchy** in descending priority. The first level that matches terminates further classification.

```
Level 1: Greeting detection     (rule-based regex)
Level 2: Crisis detection        (rule-based patterns + confidence scoring)
Level 3: Out-of-scope detection  (function node — runs in parallel, result fed in)
Level 4: Intent classification   (rule-based, first-match)
Level 5: Situation classification(rule-based, first-match)
Level 6: Severity classification (rule-based keyword scoring)
Level 7: ML model inference      (SoulBuddyClassifier, BERT-based)
```

### 7.1 Level 1 — Greeting Detection

Rule-based exact-match and pattern-matching against ~40 greeting phrases. Sets `is_greeting=True` which routes to greeting template (no LLM).

### 7.2 Level 2 — Crisis Detection

Four crisis categories in descending severity:

| Category | Example patterns | Confidence multiplier |
|----------|------------------|-----------------------|
| `SUICIDAL_PLAN` | "plan to kill myself", "have a rope", "written a goodbye letter" | 0.95 |
| `SUICIDAL_IDEATION` | "want to kill myself", "want to die", "no reason to live" | 0.90 |
| `SELF_HARM` | "cut myself", "hurt myself", "pulling my hair out" | 0.90 |
| `PASSIVE_DEATH_WISH` | "hope I don't wake up", "wish I could disappear" | 0.85 |
| `SEVERE_DISTRESS` | Risk word scoring fallback | variable |

**Negation guard:** True negations ("NOT going to kill myself", "would never hurt myself") suppress the match.

**Risk score calculation:**
```
risk_score = sum(pattern_match_score × confidence_multiplier for all matched patterns)
```

If `risk_score >= threshold` → `is_crisis_detected=True`, routes to crisis template.

### 7.3 Level 4 — Intent Classification

Evaluated in order (first match wins):

| Intent | Key signals |
|--------|------------|
| `greeting` | (from level 1) |
| `open_to_solution` | "what should I do", "any advice", "how can I fix" |
| `try_tool` | "breathing exercise", "help me calm down", "show me a technique" |
| `seek_information` | "how do/can/should I", "what is", "tell me about" |
| `seek_understanding` | "why do I feel", "help me understand", "what does it mean" |
| `seek_support` | "I need help", "I'm lost", "I'm scared", "I feel alone" |
| `venting` | "I'm stressed", "I'm overwhelmed", "I can't cope", "I'm so tired" |
| `crisis_disclosure` | (from level 2) |
| `unclear` | No match |

### 7.4 Level 5 — Situation Classification

| Situation | Key signals |
|-----------|------------|
| `EXAM_ANXIETY` | exam, test, finals, quiz, studying, failing class |
| `ACADEMIC_COMPARISON` | grades/gpa vs others, top of class, falling behind |
| `RELATIONSHIP_ISSUES` | boyfriend/girlfriend, breakup, friend conflicts, family |
| `FINANCIAL_STRESS` | money, debt, can't afford, bills, tuition |
| `HEALTH_CONCERNS` | sick, illness, doctor, medication, panic attack |
| `BELONGING_DOUBT` | don't fit in, outcast, lonely, no friends, invisible |
| `LOW_MOTIVATION` | can't start, procrastinating, no motivation, lazy |
| `FUTURE_UNCERTAINTY` | career, college, graduation, unsure about future |
| `GENERAL_OVERWHELM` | overwhelmed, burned out, too much, falling apart |
| `SUICIDAL` | (from crisis detection) |
| `NO_SITUATION` | No match |

### 7.5 Level 6 — Severity Classification

| Severity | Key signals |
|----------|------------|
| `high` | "can't cope anymore", "breaking down", "hopeless", "nothing will change", "completely lost" |
| `medium` | "stressed", "anxious", "worried", "struggling", "difficult", "overwhelmed" |
| `low` | "a bit", "slightly", "sometimes", "not too serious", "just wanted to vent" |

### 7.6 Level 7 — ML Model (SoulBuddyClassifier)

**Architecture:** Fine-tuned `bhadresh-savani/bert-base-uncased-emotion` (BERT-base) with four classification heads.

**Definition:** `transformer_models/SoulBuddyClassifier.py`

```
BERT encoder (768-dim hidden)
    ↓ [CLS] token
    ├── situation_head   → logits[21 classes]
    ├── severity_head    → logits[3 classes]
    ├── intent_head      → logits[10 classes]
    └── risk_head        → sigmoid → float [0.0, 1.0]
```

**Inference flow:**
```python
inputs = tokenizer(message, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    sit_logits, sev_logits, int_logits, risk_score = model(**inputs)

# Confidence filtering: if max logit < 0.5 → "unclear"
situation = label_map[sit_logits.argmax()] if sit_logits.max() >= 0.5 else "unclear"
severity  = label_map[sev_logits.argmax()] if sev_logits.max() >= 0.5 else "unclear"
intent    = label_map[int_logits.argmax()] if int_logits.max() >= 0.5 else "unclear"
risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "high"
```

**Model weights:** loaded from `model_weights.pt` at startup. Without it, the model initialises with random weights (meaningless output). Place the file at `sb-backend/model_weights.pt` or set `MODEL_WEIGHTS_PATH`.

**Note:** Rule-based levels (1–6) run first. The ML model is only invoked if rules don't yield a confident classification — or if the rule path is configured to call it explicitly for a second opinion.

---

## 8. Response Generation

**File:** `graph/nodes/agentic_nodes/response_generator.py`

### 8.1 Decision tree

```
is_crisis_detected? ──yes──► crisis template (response_templates.py)
        │
       no
        │
is_greeting? ──────yes──► greeting template
        │
       no
        │
is_out_of_scope? ──yes──► out-of-scope template (reason-specific)
        │
       no
        │
COMPARE_RESULTS=true? ──yes──► call Ollama + GPT in parallel → evaluator → best
        │
       no
        │
OLLAMA_FLAG=true? ──yes──► Ollama only
        │
       no
        │
OPENAI_FLAG=true? ──yes──► OpenAI only
        │
       no
        └──► error: no LLM configured
```

### 8.2 Template responses (`response_templates.py`)

Template responses bypass all LLM calls. Used for:

- **Crisis**: Hardcoded empathetic response + hotline numbers (iCall: 9152987821, Vandrevala: 1860-2662-345, AASRA: 9820466627). Template **cannot** be overridden by config — safety first.
- **Greeting**: Warm welcome message introducing SoulBuddy.
- **Out-of-scope**: Redirects to wellness focus, reason-aware (e.g., different message for coding vs. cooking requests).

### 8.3 LLM prompt construction

System message components (assembled from ConversationState):

```
1. Persona definition: "You are SoulBuddy, an empathetic emotional support companion..."
2. Mode instructions:
   - default: balanced support
   - reflection: help user explore feelings
   - venting: active listening, no advice unless asked
   - therapist: structured cognitive-behavioural framing
3. Situation context: "User situation: EXAM_ANXIETY, severity: medium"
4. Intent context: "User intent: venting"
5. Chat preference: general|focused|deep (controls response depth)
6. Conversation history: last N turns formatted as "User: ... / Bot: ..."
7. Summary: compact prior context if available
8. User profile: personality traits, demographic hints (if available)
```

### 8.4 Ollama integration

```
POST {OLLAMA_BASE_URL}/api/generate
Body: {
  "model": "llama3.2",           # configurable
  "prompt": "<system>\n...\n<human>\n{message}",
  "temperature": 0.7,
  "stream": false
}
Timeout: 120s
```

Response parsed from `response.json()["response"]`, with surrounding quotes stripped.

### 8.5 OpenAI integration

```
POST https://api.openai.com/v1/chat/completions
Headers: Authorization: Bearer {OPENAI_API_KEY}
Body: {
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "{message}"}
  ],
  "temperature": 0.7,
  "max_tokens": 200
}
```

### 8.6 Dual LLM comparison (`COMPARE_RESULTS=true`)

When enabled:
1. Both LLMs called in parallel via `asyncio.gather()`
2. Both responses scored by `response_evaluator.select_best_response()`
3. Scoring criteria: length appropriateness, empathy keywords, specificity, absence of canned phrases
4. Highest-scoring response becomes `response_draft`
5. Both scores logged for monitoring

---

## 9. Data Layer

### 9.1 Database separation rationale

Two separate PostgreSQL databases — not schemas, not tables.

| Database | Name | Purpose |
|----------|------|---------|
| Data DB | `soulbuddy` | SoulBuddy-specific data: conversations, turns, summaries |
| Auth DB | `souloxy-db` | Identity platform: users, profiles, personality data |

**Why separate?** Auth DB is shared infrastructure across multiple Souloxy services. SoulBuddy only reads from it (never writes). This boundary prevents SoulBuddy from accidentally corrupting identity data and allows the auth service to evolve its schema independently.

### 9.2 Data DB schema

**`sb_conversations`**
```
id              UUID     PK, default gen_random_uuid()
supabase_user_id UUID    NULL (incognito) or FK-by-convention to auth.users
mode            TEXT     'incognito' | 'cognito'
started_at      TIMESTAMP WITH TIME ZONE  DEFAULT now()
ended_at        TIMESTAMP WITH TIME ZONE  NULL
```

**`conversation_turns`**
```
id          UUID       PK
session_id  UUID       FK → sb_conversations.id
turn_index  INTEGER    sequential (0, 1, 2, …)
speaker     TEXT       'user' | 'bot'
message     TEXT       plaintext or 'ENC:v1:<base64>'
created_at  TIMESTAMP WITH TIME ZONE  DEFAULT now()
```

**`user_conversation_summaries`**
```
user_id             UUID    PK, FK → auth_users.id (cross-DB reference by convention)
summary             TEXT    compact structured summary string
last_conversation_id UUID   NULL
updated_at          TIMESTAMP WITH TIME ZONE  DEFAULT now()
```

### 9.3 Auth DB schema (read-only from SoulBuddy)

**`users`** (AuthUser)
```
id           INTEGER  PK
supabase_uid TEXT     UNIQUE  (primary lookup key from JWT)
email        TEXT
full_name    TEXT     NULL
role         TEXT     NULL
```

**`user_personality_profiles`** (UserPersonalityProfile)
```
id                       INTEGER  PK
supabase_uid             TEXT     UNIQUE
personality_profile_data JSONB    flexible schema
```

**`user_detailed_profiles`** (UserDetailedProfile)
```
id                      INTEGER  PK
user_id                 INTEGER  UNIQUE, FK → users.id
first_name / last_name  TEXT
age                     INTEGER
age_group               TEXT
gender                  TEXT
pronouns                TEXT
country / timezone      TEXT
languages               TEXT[]
communication_language  TEXT
education_level         TEXT
occupation              TEXT
marital_status          TEXT
hobbies                 JSONB
interests               TEXT[]
mental_health_history   JSONB
physical_health_history JSONB
```

### 9.4 SQLAlchemy session pattern

**Always use:**
```python
from config.sqlalchemy_db import data_db_sqlalchemy, auth_db_sqlalchemy

# For queries
async with data_db_sqlalchemy.get_session() as session:
    result = await session.execute(select(ConversationTurn).where(...))
    turns = result.scalars().all()

# For writes with explicit transaction
async with data_db_sqlalchemy.get_transaction() as session:
    session.add(new_row)
    # auto-commit on exit, auto-rollback on exception
```

**Never use** the asyncpg pools directly from graph nodes — those are for admin scripts only.

---

## 10. Cache Architecture

### 10.1 CacheService design

`services/cache_service.py` — singleton `cache_service` imported everywhere.

**Principle:** The application must function correctly even when Redis is completely unavailable. Caching is a performance optimisation, not a correctness requirement.

**Internal structure:**
```
CacheService
├── _get(key: str) → Any | None          # fetch + JSON decode
├── _set(key: str, value, ttl: int)      # JSON encode + SET EX
├── _delete(key: str)                    # DEL
└── _delete_pattern(pattern: str)        # SCAN + DEL (for bulk invalidation)
```

All public methods call these three internals. Any exception is caught, logged as a warning, and `None` returned.

**To add L1 in-memory cache:** subclass `CacheService` and override `_get`, `_set`, `_delete`. No other changes needed.

### 10.2 Cache key schema

| Key pattern | TTL | Description |
|-------------|-----|-------------|
| `user:<uid>:personality_profile` | 2h | Personality JSONB from auth DB |
| `user:<uid>:profile` | 2h | Detailed profile from auth DB |
| `user:<uid>:conversation_summary` | 30m | Latest rolling summary text |
| `user:<uid>:ui_state` | 30m | `page_context` dict from last request |
| `user:<uid>:config:<key>` | 24h | Per-user config values |
| `conv:<cid>:history` | 30m | Last N turns as List[Dict] |
| `config:<key>` | 24h | Global config values |

### 10.3 Invalidation points

| Invalidation trigger | Keys invalidated | Location |
|---------------------|-----------------|---------|
| User message stored | `conv:<cid>:history` | `store_message_node` |
| Bot response stored | `conv:<cid>:history`, `user:<uid>:conversation_summary` | `store_bot_response_node` |
| Profile updated | `user:<uid>:personality_profile`, `user:<uid>:profile` | `cache_service.invalidate_user_profile()` (call from profile write path) |
| All user data | All `user:<uid>:*` keys | `cache_service.invalidate_all_user_data(uid)` |

### 10.4 Redis reconnect loop

`config/redis.py` — `RedisConfig`

On startup failure or mid-operation connection loss:
1. `redis_config.mark_unavailable()` sets `is_available=False`
2. `CacheService` methods early-return `None` (skipping all Redis calls)
3. Background coroutine (`start_reconnect_loop()`) retries PING every `RECONNECT_INTERVAL` seconds (default 30s)
4. On successful reconnect: `is_available=True` restored, cache resumes

---

## 11. Encryption

**File:** `services/key_manager.py` — singleton `key_manager`

**Opt-in:** Disabled by default (`ENCRYPTION_ENABLED=false`). No GCP credentials needed unless enabled.

### 11.1 Key derivation

```
GCP KMS master key (remote, hardware-backed)
    │
    └─► HKDF-SHA256(
          key=master_key,
          salt="souloxy-conversation-salt",
          info=f"conversation:{conversation_id}",
          length=32
        )
        │
        └─► per-conversation AES-256 key (never stored, derived on demand)
```

Without GCP KMS (dev/test fallback):
```python
master_key = SHA-256("souloxy-master-key-seed-v1")  # deterministic, NOT secure for prod
```

### 11.2 Encryption scheme

Each message encrypted independently:
```
ciphertext = AES-256-GCM(
    key=per_conversation_key,
    iv=os.urandom(12),         # 96-bit random nonce
    data=message_bytes,
    auth_tag=16 bytes
)

stored = "ENC:v1:" + base64(iv + ciphertext + auth_tag)
```

Detection: `key_manager.is_data_encrypted(text)` checks for `"ENC:v1:"` prefix.

### 11.3 Transparent encryption/decryption

```
Write path:  store_message_node / store_bot_response_node
             → key_manager.encrypt(conversation_id, plaintext) → store "ENC:v1:..."

Read path:   load_user_context / get_messages API endpoint
             → key_manager.decrypt(conversation_id, value)
             → if not "ENC:v1:..." prefix → return as-is (graceful for unencrypted legacy data)
```

---

## 12. Authentication

**File:** `api/supabase_auth.py`

Two FastAPI dependency functions:

```python
# Optional — for chat endpoint (supports both incognito and cognito)
async def get_optional_user(authorization: str | None = Header(None)) -> str | None:
    if authorization is None:
        return None  # incognito
    token = authorization.removeprefix("Bearer ")
    user = supabase_client.auth.get_user(token)
    return user.user.id  # supabase_uid

# Required — for conversation history endpoints
async def require_user(authorization: str = Header(...)) -> str:
    # same as above but raises 401 if missing or invalid
```

**Supabase UID** (`supabase_uid`) is the primary cross-system user identifier:
- Stored in `sb_conversations.supabase_user_id`
- Used as cache key prefix (`user:<supabase_uid>:*`)
- Used to look up `AuthUser` by `supabase_uid` field

---

## 13. Configuration System

**File:** `config/settings.py`

Single `AppSettings` instance (`settings`) created at module import. All code imports from it:

```python
from config.settings import settings

settings.llm.ollama_base_url
settings.redis.ttl_conversation
settings.encryption.enabled
```

**Structure:**

```python
@dataclass
class AppSettings:
    logging: LoggingSettings
    data_db: DataDBSettings
    auth_db: AuthDBSettings
    supabase: SupabaseSettings
    redis: RedisSettings
    ollama: OllamaSettings
    openai: OpenAISettings
    llm: LLMSettings           # compare_results, ollama_flag, openai_flag
    encryption: EncryptionConfig
```

All values read from environment variables at dataclass instantiation. No runtime `.env` reloading — restart required for config changes.

**Environment variable resolution order:**
1. Process environment (set by Docker / shell)
2. `.env` file (loaded by `python-dotenv` at startup)
3. Hardcoded defaults in dataclass field definitions

---

## 14. Streaming (SSE)

**File:** `graph/streaming.py`

Uses LangGraph's `astream()` with `stream_mode="updates"` which yields state diffs after each node.

```python
async def stream_as_sse(state: dict) -> AsyncGenerator[str, None]:
    async for update in flow.astream(state, stream_mode="updates"):
        node_name = list(update.keys())[0]
        yield f"data: {json.dumps({'event': 'node_complete', 'node': node_name})}\n\n"

        if node_name == "render":
            response = update["render"].get("api_response", {})
            yield f"data: {json.dumps({'event': 'complete', 'response': response})}\n\n"
            return
```

**SSE event types:**

| event | When | Payload |
|-------|------|---------|
| `node_complete` | After each node | `{"event": "node_complete", "node": "<name>"}` |
| `complete` | After render | `{"event": "complete", "response": <api_response>}` |
| `error` | On exception | `{"event": "error", "detail": "<message>"}` |

---

## 15. Key Design Decisions

### 15.1 Why a single async worker?

`uvicorn --workers 1 --limit-concurrency 20`

The BERT model requires ~650-700 MB RSS on load. Two workers = ~1.4 GB minimum. The async worker handles 20 concurrent requests efficiently for I/O-bound workloads (DB, Redis, LLM calls). CPU-bound work (ML inference) is brief enough not to block.

### 15.2 Why LangGraph?

LangGraph's `StateGraph` provides:
- Explicit parallelism (`store_message` + `out_of_scope` run concurrently)
- Type-safe state updates (Pydantic model + dict merge)
- Built-in streaming (`astream`)
- Easy node-level unit testing (each node is just an async function)

Alternative: vanilla async orchestration with `asyncio.gather()` — more control but loses the streaming and state-merge guarantees.

### 15.3 Why dual databases?

Auth DB is managed by the Souloxy auth service. Shared PostgreSQL, separate databases (not schemas) enforces the boundary at the connection level: SoulBuddy cannot accidentally write auth tables even with a programming error, because the `auth_db_sqlalchemy` engine's credentials are read-only.

### 15.4 Why fail-silent Redis?

The failure mode of Redis being unavailable should never surface to end users as an error. Caching is latency optimization, not data storage. Every piece of data cached is also in PostgreSQL. The reconnect loop means temporary Redis outages are self-healing.

### 15.5 Why template responses for crisis?

Crisis responses contain specific hotline numbers and phrasing vetted by mental health guidelines. LLM output is non-deterministic and could deviate in harmful ways. Hardcoded templates guarantee safety for the highest-risk scenarios.

### 15.6 Rule-based classification before ML

Rule-based checks are:
- Faster (no model inference overhead for greetings)
- More predictable (testable with exact inputs)
- Safer for crisis detection (explicit patterns, not probabilistic)

ML model handles the long tail of nuanced cases that rules can't enumerate.

---

## 16. Extension Guide

### 16.1 Adding a new graph node

1. Create `graph/nodes/function_nodes/my_node.py` (or `agentic_nodes/`)
2. Implement: `async def my_node(state: ConversationState) -> dict[str, Any]`
3. If the node needs new state fields, add them to `ConversationState` in `graph/state.py`
4. Register in `graph/graph_builder.py`:
   ```python
   builder.add_node("my_node", my_node)
   builder.add_edge("previous_node", "my_node")
   builder.add_edge("my_node", "next_node")
   ```

### 16.2 Adding a new API endpoint

1. Create `api/my_module.py` with a FastAPI `APIRouter`
2. Register in `server.py`:
   ```python
   from api.my_module import router as my_router
   app.include_router(my_router, prefix="/api/v1", tags=["MyTag"])
   ```

### 16.3 Adding a new cache key

1. Add get/set/invalidate methods to `CacheService` in `services/cache_service.py`
2. Follow the naming pattern: `user:<uid>:<type>` or `conv:<cid>:<type>` or `config:<key>`
3. Choose TTL from existing constants or add to `RedisSettings` in `config/settings.py`
4. Call invalidation at the appropriate write point in a node

### 16.4 Adding a new LLM provider

1. Implement `async def call_my_llm(prompt: str, state: ConversationState) -> str` in `response_generator.py`
2. Add a flag to `LLMSettings` in `config/settings.py`
3. Wire into the decision tree in `response_generator.py` (§8.1)
4. Optionally integrate with `response_evaluator.py` for dual-LLM comparison

### 16.5 Adding a new situation or intent

Classification is rule-based: add new patterns to the ordered lists in `classification_node.py`. Each rule set is a list of `(pattern_regex, label)` tuples evaluated in order — first match wins.

For ML model coverage: fine-tune `SoulBuddyClassifier` with labeled examples for the new class, update label maps in `transformer_models/SoulBuddyClassifier.py`, and replace `model_weights.pt`.

### 16.6 Running tests

```bash
# All unit tests (no external deps required)
pytest

# With coverage
pytest --cov=. --cov-report=term-missing

# Specific module
pytest tests/graph/test_classification_node.py -v
```

Tests mock: databases (`asyncpg`, SQLAlchemy sessions), Redis (`aioredis`), LLM APIs (httpx), Supabase, and GCP KMS (via `conftest.py` stub).

---

*Last updated: 2026-04-03*
