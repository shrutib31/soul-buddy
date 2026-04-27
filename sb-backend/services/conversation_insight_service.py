"""
ConversationInsightService

Generates rich, LLM-powered insights at the end of each chat or journaling
session. Insights include:
  - Summary of what was discussed
  - Key emotional patterns detected
  - Actionable tips/suggestions
  - Relevant knowledge/psychoeducation
  - Mood tracking (emotional start → end)
  - Progress note compared to previous sessions

For cognito users the insight is persisted to `conversation_insights` and
returned. For incognito users the insight is generated and returned but
never stored.

All public methods are fire-and-forget safe (catch-all exception handling).
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import settings
from config.sqlalchemy_db import SQLAlchemyDataDB
from orm.models import (
    ConversationInsight,
    ConversationTurn,
    ConversationContext,
    SessionSummary,
    UserMemory,
)
from services.cache_service import cache_service

logger = logging.getLogger(__name__)

OPENAI_API_KEY = settings.openai.api_key
OLLAMA_BASE_URL = settings.ollama.base_url
OLLAMA_MODEL = settings.ollama.model
OLLAMA_TIMEOUT = settings.ollama.timeout
OLLAMA_FLAG = settings.llm.ollama_flag
_data_db = SQLAlchemyDataDB()

# ============================================================================
# INSIGHT JSON SCHEMA
# ============================================================================

_SESSION_INSIGHT_SCHEMA = {
    "summary": "2-4 sentence summary of the conversation",
    "emotional_patterns": ["list of emotional patterns observed"],
    "tips": ["1-3 actionable tips based on the conversation"],
    "knowledge": ["1-2 relevant psychoeducation points"],
    "mood_start": "emotional state at the beginning",
    "mood_end": "emotional state at the end",
    "progress_note": "brief note on progress compared to context provided",
}


# ============================================================================
# INTERNAL LLM CALLER
# ============================================================================

async def _call_gpt(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Optional[Dict]:
    if not OPENAI_API_KEY:
        logger.debug("conversation_insight_service: OpenAI API key not configured")
        return None
    try:
        import ssl
        import certifi
        import aiohttp

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as http:
            async with http.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    return json.loads(content)
                else:
                    err = await resp.text()
                    logger.warning("conversation_insight_service: GPT error status=%d err=%s", resp.status, err[:200])
                    return None
    except Exception as exc:
        logger.warning("conversation_insight_service: GPT call failed error=%s", exc)
        return None


async def _call_ollama(system_prompt: str, user_prompt: str) -> Optional[Dict]:
    """Call local Ollama with a JSON-mode prompt."""
    if not OLLAMA_BASE_URL:
        logger.debug("conversation_insight_service: Ollama base URL not configured")
        return None
    try:
        import aiohttp

        combined = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond with ONLY valid JSON — no markdown, no code fences, no extra text.\n\n"
            f"{user_prompt}"
        )
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": combined, "stream": False, "format": "json"},
                timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data.get("response", "").strip()
                    # Strip any accidental markdown fences
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    raw = raw.strip()
                    # Try direct parse first
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                    # Fallback: extract substring between first { and last }
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        try:
                            return json.loads(raw[start:end + 1])
                        except json.JSONDecodeError:
                            pass
                    # Last resort: strip trailing commas before } or ] and retry
                    import re as _re
                    cleaned = _re.sub(r",\s*([}\]])", r"\1", raw)
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1:
                        try:
                            return json.loads(cleaned[start:end + 1])
                        except json.JSONDecodeError as exc2:
                            logger.warning(
                                "conversation_insight_service: Ollama JSON parse failed after cleanup error=%s raw=%s",
                                exc2, raw[:300],
                            )
                    return None
                else:
                    err = await resp.text()
                    logger.warning("conversation_insight_service: Ollama error status=%d err=%s", resp.status, err[:200])
                    return None
    except Exception as exc:
        logger.warning("conversation_insight_service: Ollama call failed error=%s", exc)
        return None


async def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Optional[Dict]:
    """Route to OpenAI or Ollama based on config."""
    if OPENAI_API_KEY and not OLLAMA_FLAG:
        return await _call_gpt(system_prompt, user_prompt, max_tokens)
    if OLLAMA_FLAG:
        return await _call_ollama(system_prompt, user_prompt)
    # Both unavailable — try OpenAI if key is set, otherwise Ollama
    if OPENAI_API_KEY:
        return await _call_gpt(system_prompt, user_prompt, max_tokens)
    return await _call_ollama(system_prompt, user_prompt)

async def _fetch_session_turns(conversation_id: str) -> List[Dict[str, Any]]:
    try:
        from services.key_manager import get_key_manager
        km = get_key_manager()
        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationTurn, ConversationContext)
                .outerjoin(ConversationContext, ConversationContext.turn_id == ConversationTurn.id)
                .where(ConversationTurn.session_id == uuid.UUID(conversation_id))
                .order_by(ConversationTurn.turn_index.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()
            logger.info("conversation_insight_service: _fetch_session_turns raw row count=%d conv=%s", len(rows), conversation_id)

            turns = []
            for turn, ctx in rows:
                if turn.canonical_content:
                    message = turn.canonical_content
                elif turn.romanised_content:
                    message = turn.romanised_content
                elif turn.mixed_content:
                    message = turn.mixed_content
                else:
                    try:
                        message = await km.decrypt(conversation_id, turn.message)
                    except Exception:
                        message = turn.message or ""
                turns.append({
                    "speaker": turn.speaker,
                    "message": message,
                    "turn_index": turn.turn_index,
                    "intensity": ctx.intensity if ctx else None,
                    "mode": ctx.mode if ctx else None,
                    "detected_emotion": ctx.detected_emotion if ctx else None,
                })
            return turns
    except Exception as exc:
        logger.warning("conversation_insight_service: turn fetch failed error=%s", exc, exc_info=True)
        return []


async def _fetch_user_growth_summary(user_id: str) -> Optional[str]:
    try:
        async with _data_db.get_session() as session:
            stmt = select(UserMemory.growth_summary).where(
                UserMemory.user_id == uuid.UUID(user_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    except Exception:
        return None


# ============================================================================
# PUBLIC API — SESSION INSIGHT
# ============================================================================

async def generate_session_insight(
    conversation_id: str,
    user_id: Optional[str] = None,
    source_type: str = "chat",
) -> Optional[Dict[str, Any]]:
    """
    Generate an insight for a completed session.

    Args:
        conversation_id: the session to summarise
        user_id: if provided (cognito), persist the insight
        source_type: 'chat' or 'journal'

    Returns:
        The insight_data dict, or None on failure.
    """
    try:
        turns = await _fetch_session_turns(conversation_id)
        if not turns or len(turns) < 2:
            logger.info("conversation_insight_service: too few turns for insight | conv=%s", conversation_id)
            return None

        # Build transcript (cap at last 40 turns)
        transcript = "\n".join(
            f"{'User' if t['speaker'] == 'user' else 'SoulBuddy'} (turn {t['turn_index']}): {t['message']}"
            for t in turns[-40:]
        )

        # Mood tracking from intensity
        start_intensities = [t["intensity"] for t in turns[:3] if t.get("intensity") is not None]
        end_intensities = [t["intensity"] for t in turns[-3:] if t.get("intensity") is not None]
        mood_hint_start = _intensity_to_mood(start_intensities)
        mood_hint_end = _intensity_to_mood(end_intensities)

        # Fetch growth summary for progress context
        growth_context = ""
        if user_id:
            growth = await _fetch_user_growth_summary(user_id)
            if growth:
                growth_context = f"\nUser's journey so far: {growth}\n"

        schema_str = json.dumps(_SESSION_INSIGHT_SCHEMA, indent=2)

        system_prompt = (
            "You are generating end-of-session insights for a mental wellness conversation.\n"
            "Produce a warm, helpful insight summary the user will see after their session.\n\n"
            "Output ONLY valid JSON matching this schema (no extra keys, no markdown):\n"
            f"{schema_str}\n\n"
            "Guidelines:\n"
            "- summary: 2-4 sentences capturing the essence of the session\n"
            "- emotional_patterns: emotions/patterns you noticed (max 4)\n"
            "- tips: 1-3 specific, actionable suggestions tailored to what was discussed\n"
            "- knowledge: 1-2 brief psychoeducation points relevant to their topics "
            "(e.g. 'Anxiety often manifests as...')\n"
            "- mood_start/mood_end: describe emotional state at start and end\n"
            "- progress_note: if user history is provided, note progress; otherwise say "
            "'This is your first tracked session — great start!'\n"
            "- Keep the tone warm, supportive, and encouraging\n"
            "- DO NOT use clinical jargon\n"
            "- Respond in the same language the user used in the conversation"
        )

        user_prompt = (
            f"Session type: {source_type}\n"
            f"{growth_context}"
            f"Mood intensity hint — start: {mood_hint_start}, end: {mood_hint_end}\n\n"
            f"Conversation transcript:\n{transcript}\n\n"
            "Generate the session insight JSON."
        )

        insight_data = await _call_llm(system_prompt, user_prompt, max_tokens=500)
        if not insight_data:
            return None

        # ----------------------------------------------------------------
        # Crisis escalation: if the session contains dark/suicidal themes,
        # ensure a professional-help tip is always present in the insight.
        # ----------------------------------------------------------------
        _CRISIS_KEYWORDS = {"dark thoughts", "suicidal", "self-harm", "death wish", "crisis"}
        patterns_lower = {p.lower() for p in (insight_data.get("emotional_patterns") or [])}
        any_crisis_turn = any(t.get("detected_emotion") in {"crisis_disclosure"} for t in turns)
        if patterns_lower & _CRISIS_KEYWORDS or any_crisis_turn:
            crisis_tip = (
                "If these feelings become overwhelming, please reach out to a mental health "
                "professional or a crisis helpline (e.g. iCall: 9152987821). You deserve support."
            )
            tips = insight_data.get("tips") or []
            if crisis_tip not in tips:
                insight_data["tips"] = [crisis_tip] + tips

        # Persist for cognito users
        if user_id:
            await _persist_session_insight(
                user_id=user_id,
                conversation_id=conversation_id,
                source_type=source_type,
                insight_data=insight_data,
            )

        logger.info(
            "conversation_insight_service: session insight generated | conv=%s user=%s source=%s",
            conversation_id, user_id or "incognito", source_type,
        )
        return insight_data

    except Exception as exc:
        logger.warning("conversation_insight_service: generate_session_insight failed error=%s", exc)
        return None


# ============================================================================
# PERSISTENCE
# ============================================================================

async def _persist_session_insight(
    user_id: str,
    conversation_id: str,
    source_type: str,
    insight_data: Dict[str, Any],
) -> None:
    try:
        now = datetime.now(timezone.utc)
        async with _data_db.get_session() as session:
            row = ConversationInsight(
                user_id=uuid.UUID(user_id),
                session_id=uuid.UUID(conversation_id),
                insight_type="session",
                source_type=source_type,
                insight_data=insight_data,
                period_start=now,
                period_end=now,
                created_at=now,
            )
            session.add(row)
            await session.commit()
    except Exception as exc:
        logger.warning("conversation_insight_service: persist failed error=%s", exc)


# ============================================================================
# RETRIEVAL
# ============================================================================

async def get_user_insights(
    user_id: str,
    insight_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve stored insights for a user, optionally filtered by type."""
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationInsight)
                .where(ConversationInsight.user_id == uuid.UUID(user_id))
            )
            if insight_type:
                stmt = stmt.where(ConversationInsight.insight_type == insight_type)
            stmt = stmt.order_by(ConversationInsight.created_at.desc()).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "insight_type": r.insight_type,
                    "source_type": r.source_type,
                    "insight_data": r.insight_data,
                    "period_start": r.period_start.isoformat() if r.period_start else None,
                    "period_end": r.period_end.isoformat() if r.period_end else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "session_id": str(r.session_id) if r.session_id else None,
                }
                for r in rows
            ]
    except Exception as exc:
        logger.warning("conversation_insight_service: get_user_insights failed error=%s", exc)
        return []


async def get_session_insight(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the insight for a specific session."""
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationInsight)
                .where(
                    ConversationInsight.session_id == uuid.UUID(conversation_id),
                    ConversationInsight.insight_type == "session",
                )
                .limit(1)
            )
            result = await session.execute(stmt)
            r = result.scalar_one_or_none()
            if not r:
                return None
            return {
                "id": str(r.id),
                "insight_type": r.insight_type,
                "source_type": r.source_type,
                "insight_data": r.insight_data,
                "period_start": r.period_start.isoformat() if r.period_start else None,
                "period_end": r.period_end.isoformat() if r.period_end else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
    except Exception as exc:
        logger.warning("conversation_insight_service: get_session_insight failed error=%s", exc)
        return None


# ============================================================================
# UTILITY
# ============================================================================

def _intensity_to_mood(intensities: List[Optional[float]]) -> str:
    vals = [i for i in intensities if i is not None]
    if not vals:
        return "unknown"
    avg = sum(vals) / len(vals)
    if avg >= 0.7:
        return "high distress"
    elif avg >= 0.4:
        return "moderate concern"
    else:
        return "calm/positive"
