from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, PrimaryKeyConstraint, String, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func, text
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional

from .base import Base


class SbConversation(Base):
    __tablename__ = "sb_conversations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    supabase_user_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())
    ended_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("mode IN ('cognito','incognito')", name="sb_conversations_mode_check"),
    )


class Situation(Base):
    __tablename__ = "situations"

    situation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_crisis: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))


class SeverityLevel(Base):
    __tablename__ = "severity_levels"

    severity: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class Intent(Base):
    __tablename__ = "intents"

    intent_id: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConversationPhase(Base):
    __tablename__ = "conversation_phases"

    phase_id: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConversationStep(Base):
    __tablename__ = "conversation_steps"

    step_id: Mapped[str] = mapped_column(Text, primary_key=True)
    phase_id: Mapped[str | None] = mapped_column(Text, ForeignKey("conversation_phases.phase_id"), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class Flow(Base):
    __tablename__ = "flows"

    flow_id: Mapped[str] = mapped_column(Text, primary_key=True)
    flow_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_crisis: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))
    allow_tools: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))
    allow_psychoeducation: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))


class FlowStep(Base):
    __tablename__ = "flow_steps"

    flow_id: Mapped[str] = mapped_column(Text, ForeignKey("flows.flow_id"))
    step_order: Mapped[int] = mapped_column(Integer)
    step_id: Mapped[str | None] = mapped_column(Text, ForeignKey("conversation_steps.step_id"), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("flow_id", "step_order", name="flow_steps_pk"),
    )


class FlowSituationMapping(Base):
    __tablename__ = "flow_situation_mapping"

    situation_id: Mapped[str] = mapped_column(Text, ForeignKey("situations.situation_id"))
    severity: Mapped[str] = mapped_column(Text)
    flow_id: Mapped[str] = mapped_column(Text, ForeignKey("flows.flow_id"))
    confidence_min: Mapped[float | None] = mapped_column(Float, server_default=text("0.0"))

    __table_args__ = (
        PrimaryKeyConstraint("situation_id", "severity", "flow_id", name="flow_situation_mapping_pk"),
        CheckConstraint("severity IN ('low','medium','high')", name="flow_situation_mapping_severity_check"),
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sb_conversations.id"), nullable=True)
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    speaker: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True, server_default=text("'en-IN'"))
    romanised_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    canonical_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    mixed_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("speaker IN ('user','bot')", name="conversation_turns_speaker_check"),
    )


class SituationAssessment(Base):
    __tablename__ = "situation_assessments"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    turn_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversation_turns.id"), nullable=True)
    situation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    turn_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversation_turns.id"), nullable=True)
    risk_level: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class FlowState(Base):
    __tablename__ = "flow_state"

    session_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sb_conversations.id"), primary_key=True)
    flow_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_step: Mapped[str | None] = mapped_column(Text, nullable=True)
    readiness_signal: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))
    user_intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


class StepOutput(Base):
    __tablename__ = "step_outputs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    turn_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("conversation_turns.id"), nullable=True)
    emotion_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    paraphrase: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation: Mapped[str | None] = mapped_column(Text, nullable=True)
    gentle_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_offer: Mapped[str | None] = mapped_column(Text, nullable=True)


class CrisisEvent(Base):
    __tablename__ = "crisis_events"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id: Mapped[UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sb_conversations.id"), nullable=True)
    triggered_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())
    risk_level: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))


class ConversationContext(Base):
    """
    Mode, style, detected emotion and intensity for every conversation turn.
    Enables mode-aware summarization and emotional trajectory analysis.
    One row per turn (1:1 with conversation_turns).
    """

    __tablename__ = "conversation_context"

    turn_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversation_turns.id", ondelete="CASCADE"), primary_key=True
    )
    mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    style: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_emotion: Mapped[str | None] = mapped_column(Text, nullable=True)
    intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("mode IN ('default','reflection','venting','therapist')", name="conv_context_mode_check"),
        CheckConstraint("style IN ('gentle','balanced','practical')", name="conv_context_style_check"),
        CheckConstraint("intensity >= 0 AND intensity <= 1", name="conv_context_intensity_check"),
    )


class SessionModeSegment(Base):
    """
    Records mode transition segments within a session.
    Each row represents a continuous block where the user stayed in one mode/style.
    Enables "User started venting → switched to therapist" summaries.
    end_turn is NULL for the currently active segment.
    """

    __tablename__ = "session_mode_segments"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id", ondelete="CASCADE"), nullable=True
    )
    mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    style: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_turn: Mapped[int] = mapped_column(Integer, nullable=False)
    end_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("mode IN ('default','reflection','venting','therapist')", name="mode_segment_mode_check"),
        CheckConstraint("style IN ('gentle','balanced','practical')", name="mode_segment_style_check"),
    )


class SessionSummary(Base):
    """
    Rich per-session summary generated by LLM (mode-aware, structured JSON output).

    incremental_summary: updated async every 5 turns (cheap, partial)
    final_summary:       full holistic summary generated once at session end
    is_finalised:        True once the full summary has been written

    Replaces the old user_conversation_summaries table.
    Redis cache key: conv:<conversationId>:session_summary  (TTL 2h)
    """

    __tablename__ = "session_summaries"

    session_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    incremental_summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    final_summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    emotional_start: Mapped[str | None] = mapped_column(Text, nullable=True)
    emotional_end: Mapped[str | None] = mapped_column(Text, nullable=True)
    dominant_mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(Text, nullable=True, server_default=text("'low'"))
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    is_finalised: Mapped[bool] = mapped_column(Boolean, server_default=text("FALSE"))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("risk_level IN ('low','medium','high')", name="session_summary_risk_check"),
    )


class UserMemory(Base):
    """
    Single evolving row per user — the long-term user narrative / mental model.

    Updated weekly by cron or lazily when a session is finalised.
    growth_summary is the compact narrative injected into the LLM system prompt
    (~100 tokens) to give cross-session continuity without token explosion.

    Redis cache key: user:<userId>:user_memory  (TTL 24h)
    """

    __tablename__ = "user_memory"

    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    recurring_themes: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    behavioral_patterns: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    emotional_baseline: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_modes: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    preferred_styles: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    triggers: Mapped[Optional[list]] = mapped_column(ARRAY(Text), nullable=True)
    growth_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_signals: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    last_session_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id"), nullable=True
    )
    last_updated: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


class UserInsight(Base):
    """
    Time-stamped insight metrics per user — powers the team dashboard.
    Written by rule-based InsightScoringService (no LLM cost).
    Multiple rows per user per session (one per metric computed).

    metric_type: emotional | behavioral | engagement | progress | safety
    metric_name: e.g. 'emotional_stability_score', 'engagement_score'
    metric_value: float score
    metadata: any additional context (e.g. mode breakdown, top triggers)
    """

    __tablename__ = "user_insights"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id", ondelete="SET NULL"), nullable=True
    )
    metric_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    computed_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "metric_type IN ('emotional','behavioral','engagement','progress','safety')",
            name="user_insight_type_check"
        ),
    )


class InterventionOutcome(Base):
    """
    Tracks whether specific bot interventions led to positive user responses.
    user_response_score: sentiment delta measured after the intervention turn.
    Enables mode_effectiveness and intervention_success scoring in InsightScoringService.
    """

    __tablename__ = "intervention_outcomes"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id", ondelete="CASCADE"), nullable=True
    )
    turn_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversation_turns.id", ondelete="CASCADE"), nullable=True
    )
    intervention_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_response_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    success_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

class EncryptionAuditLog(Base):
    """
    Stores audit logs for encryption operations.

    Each row represents a single encryption or decryption operation,
    including details about the entity involved, the operation type,
    the user or system that accessed it, and the reason for access.
    """
    __tablename__ = "encryption_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    operation: Mapped[str] = mapped_column(String(20), nullable=False)
    accessed_by_type: Mapped[str] = mapped_column(String(20), nullable=False)
    accessed_by_id: Mapped[str] = mapped_column(String(255), nullable=False)
    accessed_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    vault_key_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    accessed_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationInsight(Base):
    """
    Stores LLM-generated insights at multiple granularity levels:

      session  — generated at end of each chat/journaling session
      daily    — rollup of all sessions that day (generated by midnight cron)
      weekly   — rollup of 7 daily insights (daily rows deleted after)
      monthly  — rollup of ~4 weekly insights

    For cognito users, insights are persisted and retrievable via API.
    For incognito users, insights are generated on-the-fly and returned
    but never stored.

    insight_data JSONB schema:
      {
        "summary": "...",
        "emotional_patterns": ["..."],
        "tips": ["..."],
        "knowledge": ["..."],
        "mood_start": "...",
        "mood_end": "...",
        "progress_note": "..."
      }
    """

    __tablename__ = "conversation_insights"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()")
    )
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    session_id: Mapped[UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sb_conversations.id", ondelete="SET NULL"), nullable=True
    )
    insight_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'chat'"))
    insight_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    period_start: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    period_end: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "insight_type IN ('session','daily','weekly','monthly')",
            name="conversation_insight_type_check",
        ),
        CheckConstraint(
            "source_type IN ('chat','journal')",
            name="conversation_insight_source_check",
        ),
    )
