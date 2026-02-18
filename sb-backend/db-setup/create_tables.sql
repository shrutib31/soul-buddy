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
  created_at TIMESTAMP DEFAULT NOW()
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

CREATE TABLE IF NOT EXISTS situation_assessments (
  turn_id UUID REFERENCES conversation_turns(id),
  situation_id TEXT,
  severity TEXT,
  risk_flag BOOLEAN,
  confidence FLOAT
);

CREATE TABLE IF NOT EXISTS risk_assessments (
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
  turn_id UUID REFERENCES conversation_turns(id),
  emotion_label TEXT,
  paraphrase TEXT,
  validation TEXT,
  gentle_summary TEXT,
  tool_offer TEXT
);

CREATE TABLE IF NOT EXISTS crisis_events (
  session_id UUID REFERENCES sb_conversations(id),
  triggered_at TIMESTAMP DEFAULT NOW(),
  risk_level TEXT,
  resolved BOOLEAN DEFAULT FALSE
);

-- =========================
-- INDEXES
-- =========================
CREATE INDEX IF NOT EXISTS idx_conversation_turns_session ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_situation_assessments_turn ON situation_assessments(turn_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_turn ON risk_assessments(turn_id);
CREATE INDEX IF NOT EXISTS idx_flow_state_session ON flow_state(session_id);
CREATE INDEX IF NOT EXISTS idx_crisis_events_session ON crisis_events(session_id);