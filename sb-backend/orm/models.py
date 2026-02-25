from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, PrimaryKeyConstraint, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func, text
from sqlalchemy.orm import Mapped, mapped_column

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

class EncryptionAuditLog(Base):
    __tablename__ = "encryption_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    operation: Mapped[str] = mapped_column(String(20), nullable=False)
    accessed_by_type: Mapped[str] = mapped_column(String(20), nullable=False)
    accessed_by_id: Mapped[str] = mapped_column(String(255), nullable=False)
    accessed_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    vault_key_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    accessed_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())