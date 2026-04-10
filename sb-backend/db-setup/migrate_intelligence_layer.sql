-- =============================================================================
-- Migration: Intelligence Layer Tables
-- Run this against an existing database that already has the base schema.
-- Safe to re-run (all statements use IF NOT EXISTS / IF EXISTS guards).
-- =============================================================================

-- ── Step 1: Drop old summary table (no longer used) ──────────────────────────
-- Data loss warning: user_conversation_summaries will be dropped.
-- Ensure you have exported any data you need before running this.
DROP TABLE IF EXISTS user_conversation_summaries;

-- ── Step 2: conversation_turns column additions (from main branch) ────────────
-- Safe to re-run: ADD COLUMN IF NOT EXISTS is idempotent.
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS language          VARCHAR(10) DEFAULT 'en-IN';
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS romanised_content TEXT;
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS canonical_content TEXT;
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS mixed_content     TEXT;

-- ── Step 3: New intelligence layer tables ────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_context (
  turn_id       UUID PRIMARY KEY REFERENCES conversation_turns(id) ON DELETE CASCADE,
  mode          TEXT CHECK (mode IN ('default','reflection','venting','therapist')),
  style         TEXT CHECK (style IN ('gentle','balanced','practical')),
  detected_emotion TEXT,
  intensity     FLOAT CHECK (intensity >= 0 AND intensity <= 1),
  created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_mode_segments (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id    UUID REFERENCES sb_conversations(id) ON DELETE CASCADE,
  mode          TEXT CHECK (mode IN ('default','reflection','venting','therapist')),
  style         TEXT CHECK (style IN ('gentle','balanced','practical')),
  start_turn    INT NOT NULL,
  end_turn      INT,
  created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_summaries (
  session_id          UUID PRIMARY KEY REFERENCES sb_conversations(id) ON DELETE CASCADE,
  user_id             UUID NOT NULL,
  incremental_summary JSONB,
  final_summary       JSONB,
  emotional_start     TEXT,
  emotional_end       TEXT,
  dominant_mode       TEXT,
  risk_level          TEXT CHECK (risk_level IN ('low','medium','high')) DEFAULT 'low',
  turn_count          INT DEFAULT 0,
  is_finalised        BOOLEAN DEFAULT FALSE,
  created_at          TIMESTAMP DEFAULT NOW(),
  updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_memory (
  user_id             UUID PRIMARY KEY,
  recurring_themes    TEXT[],
  behavioral_patterns TEXT[],
  emotional_baseline  TEXT,
  preferred_modes     TEXT[],
  preferred_styles    TEXT[],
  triggers            TEXT[],
  growth_summary      TEXT,
  risk_signals        JSONB,
  last_session_id     UUID REFERENCES sb_conversations(id),
  last_updated        TIMESTAMP DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS intervention_outcomes (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id          UUID REFERENCES sb_conversations(id) ON DELETE CASCADE,
  turn_id             UUID REFERENCES conversation_turns(id) ON DELETE CASCADE,
  intervention_type   TEXT,
  user_response_score FLOAT,
  success_flag        BOOLEAN,
  created_at          TIMESTAMP DEFAULT NOW()
);

-- ── Step 3: Indexes ───────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_session_mode_segments_session ON session_mode_segments(session_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_user        ON session_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_session_summaries_finalised   ON session_summaries(is_finalised);
CREATE INDEX IF NOT EXISTS idx_user_insights_user            ON user_insights(user_id);
CREATE INDEX IF NOT EXISTS idx_user_insights_session         ON user_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_user_insights_computed        ON user_insights(computed_at);
CREATE INDEX IF NOT EXISTS idx_intervention_outcomes_session ON intervention_outcomes(session_id);
