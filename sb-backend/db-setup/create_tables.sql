-- =========================
-- CORE
-- =========================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===============================
-- SOULBUDDY CONVERSATION TABLES
-- ===============================
CREATE TABLE IF NOT EXISTS sb_conversations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  supabase_user_id UUID NULL ,
  mode TEXT CHECK (mode IN ('cognito','incognito')),
  started_at TIMESTAMP DEFAULT NOW(),
  ended_at TIMESTAMP
);

-- =========================
-- CONFIGURATION TABLES
-- =========================

CREATE TABLE IF NOT EXISTS situations (
  situation_id TEXT PRIMARY KEY,
  category TEXT,
  description TEXT,
  is_crisis BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS severity_levels (
  severity TEXT PRIMARY KEY,
  description TEXT
);

CREATE TABLE IF NOT EXISTS intents (
  intent_id TEXT PRIMARY KEY,
  description TEXT
);

CREATE TABLE IF NOT EXISTS conversation_phases (
  phase_id TEXT PRIMARY KEY,
  description TEXT
);

CREATE TABLE IF NOT EXISTS conversation_steps (
  step_id TEXT PRIMARY KEY,
  phase_id TEXT REFERENCES conversation_phases(phase_id),
  description TEXT
);

CREATE TABLE IF NOT EXISTS flows (
  flow_id TEXT PRIMARY KEY,
  flow_mode TEXT,
  description TEXT,
  is_crisis BOOLEAN DEFAULT FALSE,
  allow_tools BOOLEAN DEFAULT FALSE,
  allow_psychoeducation BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS flow_steps (
  flow_id TEXT REFERENCES flows(flow_id),
  step_order INT,
  step_id TEXT REFERENCES conversation_steps(step_id),
  PRIMARY KEY (flow_id, step_order)
);

CREATE TABLE IF NOT EXISTS flow_situation_mapping (
  situation_id TEXT REFERENCES situations(situation_id),
  severity TEXT CHECK (severity IN ('low','medium','high')),
  flow_id TEXT REFERENCES flows(flow_id),
  confidence_min FLOAT DEFAULT 0.0,
  PRIMARY KEY (situation_id, severity, flow_id)
);

-- =========================
-- RUNTIME TABLES
-- =========================

CREATE TABLE IF NOT EXISTS conversation_turns (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID REFERENCES sb_conversations(id),
  turn_index INT,
  speaker TEXT CHECK (speaker IN ('user','bot')),
  message TEXT,
  language VARCHAR(10) DEFAULT 'en-IN',
  romanised_content TEXT,
  canonical_content TEXT,
  mixed_content TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS situation_assessments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  turn_id UUID REFERENCES conversation_turns(id),
  situation_id TEXT,
  severity TEXT,
  risk_flag BOOLEAN,
  confidence FLOAT
);

CREATE TABLE IF NOT EXISTS risk_assessments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  turn_id UUID REFERENCES conversation_turns(id),
  risk_level TEXT,
  confidence FLOAT
);

CREATE TABLE IF NOT EXISTS flow_state (
  session_id UUID PRIMARY KEY REFERENCES sb_conversations(id),
  flow_id TEXT,
  current_step TEXT,
  readiness_signal BOOLEAN DEFAULT FALSE,
  user_intent TEXT,
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS step_outputs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  turn_id UUID REFERENCES conversation_turns(id),
  emotion_label TEXT,
  paraphrase TEXT,
  validation TEXT,
  gentle_summary TEXT,
  tool_offer TEXT
);

CREATE TABLE IF NOT EXISTS crisis_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID REFERENCES sb_conversations(id),
  triggered_at TIMESTAMP DEFAULT NOW(),
  risk_level TEXT,
  resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_conversation_summaries (
  user_id UUID PRIMARY KEY,
  summary TEXT NOT NULL,
  last_conversation_id UUID REFERENCES sb_conversations(id),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS encryption_audit_log (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL,
    accessed_by_type VARCHAR(20) NOT NULL,
    accessed_by_id VARCHAR(255) NOT NULL,
    accessed_reason TEXT,
    vault_key_name VARCHAR(100),
    accessed_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- INTELLIGENCE LAYER TABLES
-- =========================

-- Tracks mode, style, detected emotion and intensity for every turn.
-- Enables mode-aware summarization and emotional trajectory analysis.
CREATE TABLE IF NOT EXISTS conversation_context (
  turn_id       UUID PRIMARY KEY REFERENCES conversation_turns(id) ON DELETE CASCADE,
  mode          TEXT CHECK (mode IN ('default','reflection','venting','therapist')),
  style         TEXT CHECK (style IN ('gentle','balanced','practical')),
  detected_emotion TEXT,
  intensity     FLOAT CHECK (intensity >= 0 AND intensity <= 1),
  created_at    TIMESTAMP DEFAULT NOW()
);

-- Records mode transition segments within a session.
-- Enables "User started venting → switched to coaching" summaries.
CREATE TABLE IF NOT EXISTS session_mode_segments (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id    UUID REFERENCES sb_conversations(id) ON DELETE CASCADE,
  mode          TEXT CHECK (mode IN ('default','reflection','venting','therapist')),
  style         TEXT CHECK (style IN ('gentle','balanced','practical')),
  start_turn    INT NOT NULL,
  end_turn      INT,
  created_at    TIMESTAMP DEFAULT NOW()
);

-- Rich per-session summary generated by LLM (mode-aware, structured JSON).
-- Generated incrementally every 5 turns (async) and finalised at session end.
-- Replaces user_conversation_summaries.
CREATE TABLE IF NOT EXISTS session_summaries (
  session_id          UUID PRIMARY KEY REFERENCES sb_conversations(id) ON DELETE CASCADE,
  user_id             UUID NOT NULL,
  -- Incremental snapshot (updated every 5 turns, lightweight)
  incremental_summary JSONB,
  -- Full holistic summary (generated once at session end)
  final_summary       JSONB,
  -- Aggregated fields for fast querying without parsing JSON
  emotional_start     TEXT,
  emotional_end       TEXT,
  dominant_mode       TEXT,
  risk_level          TEXT CHECK (risk_level IN ('low','medium','high')) DEFAULT 'low',
  turn_count          INT DEFAULT 0,
  is_finalised        BOOLEAN DEFAULT FALSE,
  created_at          TIMESTAMP DEFAULT NOW(),
  updated_at          TIMESTAMP DEFAULT NOW()
);

-- Single evolving row per user — the long-term user narrative / mental model.
-- Updated weekly by cron or lazily on session end.
-- This is the primary cross-session context injected into the LLM.
CREATE TABLE IF NOT EXISTS user_memory (
  user_id             UUID PRIMARY KEY,
  recurring_themes    TEXT[],
  behavioral_patterns TEXT[],
  emotional_baseline  TEXT,
  preferred_modes     TEXT[],
  preferred_styles    TEXT[],
  triggers            TEXT[],
  growth_summary      TEXT,     -- compact narrative injected into LLM (~100 tokens)
  risk_signals        JSONB,    -- {level, notes, last_crisis_date}
  last_session_id     UUID REFERENCES sb_conversations(id),
  last_updated        TIMESTAMP DEFAULT NOW()
);

-- Time-stamped insight metrics per user — powers team dashboard.
-- Written by rule-based InsightScoringService (no LLM cost).
CREATE TABLE IF NOT EXISTS user_insights (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL,
  session_id      UUID REFERENCES sb_conversations(id) ON DELETE SET NULL,
  metric_type     TEXT CHECK (metric_type IN ('emotional','behavioral','engagement','progress','safety')),
  metric_name     TEXT NOT NULL,
  metric_value    FLOAT,
  metadata        JSONB,
  computed_at     TIMESTAMP DEFAULT NOW()
);

-- Tracks whether specific bot interventions led to positive user responses.
-- Enables mode_effectiveness and intervention_success scoring.
CREATE TABLE IF NOT EXISTS intervention_outcomes (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id          UUID REFERENCES sb_conversations(id) ON DELETE CASCADE,
  turn_id             UUID REFERENCES conversation_turns(id) ON DELETE CASCADE,
  intervention_type   TEXT,   -- e.g. 'breathing_exercise', 'reframe', 'validation'
  user_response_score FLOAT,  -- sentiment delta after intervention
  success_flag        BOOLEAN,
  created_at          TIMESTAMP DEFAULT NOW()
);

-- =========================
-- INDEXES
-- =========================
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_situation_assessments_turn ON situation_assessments(turn_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_turn ON risk_assessments(turn_id);
CREATE INDEX IF NOT EXISTS idx_flow_state_session ON flow_state(session_id);
CREATE INDEX IF NOT EXISTS idx_crisis_events_session ON crisis_events(session_id);

-- Intelligence layer indexes
CREATE INDEX IF NOT EXISTS idx_session_mode_segments_session ON session_mode_segments(session_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_user ON session_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_finalised ON session_summaries(is_finalised);
CREATE INDEX IF NOT EXISTS idx_user_insights_user ON user_insights(user_id);
CREATE INDEX IF NOT EXISTS idx_user_insights_session ON user_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_user_insights_computed ON user_insights(computed_at);
CREATE INDEX IF NOT EXISTS idx_intervention_outcomes_session ON intervention_outcomes(session_id);