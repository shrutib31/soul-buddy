-- Migration: Add conversation_insights table
-- This table stores LLM-generated insights at session, daily, weekly, and monthly levels.

CREATE TABLE IF NOT EXISTS conversation_insights (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL,
  session_id UUID REFERENCES sb_conversations(id) ON DELETE SET NULL,
  insight_type TEXT NOT NULL CHECK (insight_type IN ('session','daily','weekly','monthly')),
  source_type TEXT NOT NULL DEFAULT 'chat' CHECK (source_type IN ('chat','journal')),
  insight_data JSONB,
  period_start TIMESTAMPTZ,
  period_end TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_insights_user ON conversation_insights(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_insights_type ON conversation_insights(user_id, insight_type);
CREATE INDEX IF NOT EXISTS idx_conversation_insights_period ON conversation_insights(user_id, insight_type, period_start);
CREATE INDEX IF NOT EXISTS idx_conversation_insights_session ON conversation_insights(session_id);
