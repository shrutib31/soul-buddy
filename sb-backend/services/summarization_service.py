"""
SummarizationService

Generates LLM-based summaries at two levels:

  1. Incremental session summary (called every 5 turns, fire-and-forget)
     — lightweight, mode-aware JSON snapshot of the conversation so far.

  2. Final session summary (called once at session end, fire-and-forget)
     — holistic narrative: session_story, mode_journey, emotional_arc,
       key_takeaways, recommended_next_step.

  3. User memory update (called after final session summary)
     — merges the new session's insights into the user's long-term
       user_memory row (recurring_themes, behavioral_patterns, growth_summary, etc.)

Design principles
-----------------
* All public methods are fire-and-forget friendly — they catch all exceptions
  and never raise, so callers can safely asyncio.create_task() them.
* LLM calls use GPT-4o-mini with strict max_tokens budgets to cap cost.
* Output is always structured JSON — parsed and stored in JSONB columns.
* Mode-specific prompts produce richer, more useful summaries than a single
  generic prompt.
"""

import asyncio
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
    ConversationTurn,
    ConversationContext,
    SessionModeSegment,
    SessionSummary,
    UserMemory,
)
from services.cache_service import cache_service

logger = logging.getLogger(__name__)

OPENAI_API_KEY = settings.openai.api_key

_data_db = SQLAlchemyDataDB()


# ============================================================================
# MODE-SPECIFIC SUMMARIZATION PROMPTS
# ============================================================================

_MODE_PROMPTS: Dict[str, Dict[str, Any]] = {
    "venting": {
        "focus": (
            "Focus on: what the user is feeling (not solving), emotional intensity and shifts, "
            "key triggers, and validation moments."
        ),
        "schema": {
            "emotions": ["list of emotions expressed"],
            "triggers": ["list of key triggers mentioned"],
            "emotional_shift": "start emotion → end emotion",
            "validation_needed": True,
            "key_moments": ["notable moments of intensity or relief"],
            "summary": "2-3 sentence narrative",
        },
    },
    "therapist": {
        "focus": (
            "Focus on: goals discussed, obstacles identified, actionable steps, "
            "readiness to change, and any cognitive patterns observed."
        ),
        "schema": {
            "goal": "primary goal or concern",
            "blockers": ["list of obstacles"],
            "actions": ["suggested or agreed actions"],
            "readiness_score": 0.0,
            "progress_signal": "low/medium/high",
            "cognitive_patterns": ["any patterns observed"],
            "summary": "2-3 sentence narrative",
        },
    },
    "reflection": {
        "focus": (
            "Focus on: insights gained, thought patterns surfaced, reframes achieved, "
            "and depth of self-awareness demonstrated."
        ),
        "schema": {
            "key_insights": ["list of insights"],
            "thought_patterns": ["patterns identified"],
            "reframes": ["perspective shifts achieved"],
            "depth_score": 0.0,
            "summary": "2-3 sentence narrative",
        },
    },
    "default": {
        "focus": (
            "Focus on: general topics discussed, overall emotional tone, "
            "engagement level, and any notable positive or negative moments."
        ),
        "schema": {
            "topics": ["list of topics discussed"],
            "tone": "positive/neutral/negative",
            "engagement_score": 0.0,
            "notable_moments": ["any significant moments"],
            "summary": "2-3 sentence narrative",
        },
    },
}

_HOLISTIC_SUMMARY_SCHEMA = {
    "session_story": "2-3 sentence narrative of the full session",
    "mode_journey": ["venting (turns 0-4)", "therapist (turns 5-12)"],
    "emotional_arc": "overwhelmed → slightly hopeful",
    "key_takeaways": ["list of key takeaways"],
    "recommended_next_step": "suggestion for the next session",
    "dominant_emotion": "primary emotion of the session",
    "risk_level": "low/medium/high",
}

_USER_MEMORY_UPDATE_SCHEMA = {
    "recurring_themes": ["updated list of recurring themes"],
    "behavioral_patterns": ["updated behavioral patterns"],
    "emotional_baseline": "updated emotional baseline description",
    "preferred_modes": ["modes the user engages with most"],
    "preferred_styles": ["styles that work best"],
    "triggers": ["known trigger points"],
    "growth_summary": "compact 2-3 sentence narrative of the user's journey and growth (max 100 words)",
    "risk_signals": {
        "level": "low/medium/high",
        "notes": "any safety-relevant observations",
        "last_crisis_date": None,
    },
}


# ============================================================================
# INTERNAL LLM CALLER
# ============================================================================

async def _call_gpt(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> Optional[Dict]:
    """
    Call GPT-4o-mini and parse the JSON response.
    Returns None on any failure — callers must handle None gracefully.
    """
    if not OPENAI_API_KEY:
        logger.debug("summarization_service: OpenAI API key not configured")
        return None
    try:
        import ssl
        import certifi
        import aiohttp

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.3,   # low temp for consistent structured output
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
                    logger.warning("summarization_service: GPT error status=%d err=%s", resp.status, err[:200])
                    return None
    except Exception as exc:
        logger.warning("summarization_service: GPT call failed error=%s", exc)
        return None


# ============================================================================
# TURN FETCHING HELPERS
# ============================================================================

async def _fetch_turns_with_context(
    conversation_id: str,
) -> List[Dict[str, Any]]:
    """
    Fetch all turns for a conversation joined with their conversation_context rows.
    Returns list of dicts: {speaker, message, turn_index, mode, style, intensity}
    """
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationTurn, ConversationContext)
                .outerjoin(ConversationContext, ConversationContext.turn_id == ConversationTurn.id)
                .where(ConversationTurn.session_id == uuid.UUID(conversation_id))
                .order_by(ConversationTurn.turn_index.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()

            turns = []
            from services.key_manager import get_key_manager
            km = get_key_manager()
            for turn, ctx in rows:
                # Content column priority for summarization:
                #   canonical_content — pure native script (Devanagari) or English:
                #                       use directly (native script will be summarised as-is
                #                       by the LLM which handles Indic scripts)
                #   romanised_content — Hinglish / romanised Indian language in latin script:
                #                       use directly (LLM handles romanised Indic well)
                #   mixed_content     — blend of native + latin: use directly
                #   message           — encrypted fallback: decrypt first
                # All three plaintext columns are only populated when encryption is disabled,
                # so if any is set we can use it without decryption.
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
                    "language": turn.language,
                    "mode": ctx.mode if ctx else None,
                    "style": ctx.style if ctx else None,
                    "intensity": ctx.intensity if ctx else None,
                })
            return turns
    except Exception as exc:
        logger.warning("summarization_service: turn fetch failed error=%s", exc)
        return []


async def _fetch_mode_segments(conversation_id: str) -> List[Dict[str, Any]]:
    """Fetch all mode segments for a session, ordered by start_turn."""
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(SessionModeSegment)
                .where(SessionModeSegment.session_id == uuid.UUID(conversation_id))
                .order_by(SessionModeSegment.start_turn.asc())
            )
            result = await session.execute(stmt)
            segments = result.scalars().all()
            return [
                {"mode": s.mode, "style": s.style, "start_turn": s.start_turn, "end_turn": s.end_turn}
                for s in segments
            ]
    except Exception as exc:
        logger.warning("summarization_service: segment fetch failed error=%s", exc)
        return []


# ============================================================================
# PUBLIC API
# ============================================================================

async def summarize_session_incremental(
    conversation_id: str,
    user_id: str,
    dominant_mode: str,
    turn_count: int,
) -> None:
    """
    Generate a lightweight incremental session summary and upsert it.
    Called every 5 turns, fire-and-forget.
    Only summarizes user turns to keep token cost low.
    """
    try:
        turns = await _fetch_turns_with_context(conversation_id)
        if not turns:
            return

        # Build a condensed conversation transcript (user turns only for brevity)
        user_turns = [t for t in turns if t["speaker"] == "user"]
        if len(user_turns) < 2:
            return  # not enough content yet

        mode_cfg = _MODE_PROMPTS.get(dominant_mode, _MODE_PROMPTS["default"])
        transcript = "\n".join(
            f"User (turn {t['turn_index']}): {t['message']}" for t in user_turns[-10:]
        )
        schema_str = json.dumps(mode_cfg["schema"], indent=2)

        system_prompt = (
            f"You are summarizing a mental wellness conversation in '{dominant_mode}' mode.\n"
            f"{mode_cfg['focus']}\n\n"
            "Output ONLY valid JSON matching this schema (no extra keys, no markdown):\n"
            f"{schema_str}\n\n"
            "Keep the summary field under 60 words. Be concise and factual."
        )
        user_prompt = f"Conversation transcript:\n{transcript}\n\nGenerate the summary JSON."

        result = await _call_gpt(system_prompt, user_prompt, max_tokens=300)
        if not result:
            return

        now = datetime.now(timezone.utc)
        session_uuid = uuid.UUID(conversation_id)
        user_uuid = uuid.UUID(user_id)

        async with _data_db.get_session() as db_session:
            stmt = (
                pg_insert(SessionSummary)
                .values(
                    session_id=session_uuid,
                    user_id=user_uuid,
                    incremental_summary=result,
                    dominant_mode=dominant_mode,
                    turn_count=turn_count,
                    is_finalised=False,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["session_id"],
                    set_={
                        "incremental_summary": result,
                        "dominant_mode": dominant_mode,
                        "turn_count": turn_count,
                        "updated_at": now,
                    },
                )
            )
            await db_session.execute(stmt)
            await db_session.commit()

        # Update cache
        existing = await cache_service.get_session_summary(conversation_id) or {}
        existing["incremental_summary"] = result
        existing["dominant_mode"] = dominant_mode
        await cache_service.set_session_summary(conversation_id, existing)

        logger.info(
            "summarization_service: incremental summary updated | conversation_id=%s turns=%d mode=%s",
            conversation_id, turn_count, dominant_mode,
        )

    except Exception as exc:
        logger.warning("summarization_service: incremental summary failed error=%s", exc)


async def summarize_session_final(
    conversation_id: str,
    user_id: str,
    dominant_mode: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate the full holistic session summary at session end.
    Stores in session_summaries.final_summary and marks is_finalised=True.
    Returns the summary dict (or None on failure) so callers can chain update_user_memory.
    Fire-and-forget safe.
    """
    try:
        turns = await _fetch_turns_with_context(conversation_id)
        segments = await _fetch_mode_segments(conversation_id)

        if not turns:
            return None

        turn_count = len(turns)

        # Full transcript (both speakers) capped at last 30 turns
        transcript = "\n".join(
            f"{'User' if t['speaker'] == 'user' else 'SoulBuddy'} (turn {t['turn_index']}): {t['message']}"
            for t in turns[-30:]
        )
        mode_journey_str = ", ".join(
            f"{s['mode']} (turns {s['start_turn']}–{s['end_turn'] or 'end'})"
            for s in segments
        ) or dominant_mode

        schema_str = json.dumps(_HOLISTIC_SUMMARY_SCHEMA, indent=2)

        system_prompt = (
            "You are generating a holistic end-of-session summary for a mental wellness conversation.\n"
            "Input includes the full conversation transcript and mode journey.\n\n"
            "Output ONLY valid JSON matching this schema (no extra keys, no markdown):\n"
            f"{schema_str}\n\n"
            "Keep session_story under 60 words. Keep key_takeaways to max 4 items. "
            "recommended_next_step should be one concrete, warm suggestion."
        )
        user_prompt = (
            f"Mode journey: {mode_journey_str}\n\n"
            f"Conversation transcript:\n{transcript}\n\n"
            "Generate the holistic session summary JSON."
        )

        result = await _call_gpt(system_prompt, user_prompt, max_tokens=400)
        if not result:
            return None

        # Determine emotional start/end from turns
        emotional_start = _extract_emotion(turns[:3])
        emotional_end = _extract_emotion(turns[-3:])
        risk_level = result.get("risk_level", "low")
        now = datetime.now(timezone.utc)
        session_uuid = uuid.UUID(conversation_id)
        user_uuid = uuid.UUID(user_id)

        async with _data_db.get_session() as db_session:
            stmt = (
                pg_insert(SessionSummary)
                .values(
                    session_id=session_uuid,
                    user_id=user_uuid,
                    final_summary=result,
                    emotional_start=emotional_start,
                    emotional_end=emotional_end,
                    dominant_mode=dominant_mode,
                    risk_level=risk_level if risk_level in ("low", "medium", "high") else "low",
                    turn_count=turn_count,
                    is_finalised=True,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["session_id"],
                    set_={
                        "final_summary": result,
                        "emotional_start": emotional_start,
                        "emotional_end": emotional_end,
                        "dominant_mode": dominant_mode,
                        "risk_level": risk_level if risk_level in ("low", "medium", "high") else "low",
                        "turn_count": turn_count,
                        "is_finalised": True,
                        "updated_at": now,
                    },
                )
            )
            await db_session.execute(stmt)
            await db_session.commit()

        # Refresh cache with full summary
        await cache_service.set_session_summary(conversation_id, {
            "final_summary": result,
            "emotional_start": emotional_start,
            "emotional_end": emotional_end,
            "dominant_mode": dominant_mode,
            "risk_level": risk_level,
            "turn_count": turn_count,
            "is_finalised": True,
        })

        logger.info(
            "summarization_service: final summary written | conversation_id=%s turns=%d mode=%s",
            conversation_id, turn_count, dominant_mode,
        )
        return result

    except Exception as exc:
        logger.warning("summarization_service: final summary failed error=%s", exc)
        return None


async def update_user_memory(
    user_id: str,
    new_session_summary: Dict[str, Any],
    existing_memory: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Merge the new session's insights into the user's long-term user_memory row.
    Called after summarize_session_final, fire-and-forget.

    existing_memory: the current user_memory dict (if any) — passed in to avoid
    an extra DB read when the caller already has it.
    """
    try:
        existing_str = json.dumps(existing_memory, indent=2) if existing_memory else "No prior memory."
        new_session_str = json.dumps(new_session_summary, indent=2)
        schema_str = json.dumps(_USER_MEMORY_UPDATE_SCHEMA, indent=2)

        system_prompt = (
            "You are updating a user's long-term mental wellness memory profile.\n"
            "You will receive their existing memory and a new session summary.\n"
            "Merge them intelligently: update recurring themes, behavioral patterns, "
            "emotional baseline, and growth_summary based on new evidence.\n\n"
            "Output ONLY valid JSON matching this schema (no extra keys, no markdown):\n"
            f"{schema_str}\n\n"
            "growth_summary MUST be under 100 words — this is injected into every LLM prompt "
            "so brevity is critical. Preserve important historical context while adding new insights."
        )
        user_prompt = (
            f"Existing user memory:\n{existing_str}\n\n"
            f"New session summary:\n{new_session_str}\n\n"
            "Generate the updated user memory JSON."
        )

        result = await _call_gpt(system_prompt, user_prompt, max_tokens=400)
        if not result:
            return

        now = datetime.now(timezone.utc)
        user_uuid = uuid.UUID(user_id)

        async with _data_db.get_session() as db_session:
            stmt = (
                pg_insert(UserMemory)
                .values(
                    user_id=user_uuid,
                    recurring_themes=result.get("recurring_themes"),
                    behavioral_patterns=result.get("behavioral_patterns"),
                    emotional_baseline=result.get("emotional_baseline"),
                    preferred_modes=result.get("preferred_modes"),
                    preferred_styles=result.get("preferred_styles"),
                    triggers=result.get("triggers"),
                    growth_summary=result.get("growth_summary"),
                    risk_signals=result.get("risk_signals"),
                    last_updated=now,
                )
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={
                        "recurring_themes": result.get("recurring_themes"),
                        "behavioral_patterns": result.get("behavioral_patterns"),
                        "emotional_baseline": result.get("emotional_baseline"),
                        "preferred_modes": result.get("preferred_modes"),
                        "preferred_styles": result.get("preferred_styles"),
                        "triggers": result.get("triggers"),
                        "growth_summary": result.get("growth_summary"),
                        "risk_signals": result.get("risk_signals"),
                        "last_updated": now,
                    },
                )
            )
            await db_session.execute(stmt)
            await db_session.commit()

        # Refresh cache
        await cache_service.set_user_memory(user_id, result)

        logger.info("summarization_service: user_memory updated | user_id=%s", user_id)

    except Exception as exc:
        logger.warning("summarization_service: user_memory update failed | user_id=%s error=%s", user_id, exc)


# ============================================================================
# UTILITY
# ============================================================================

def _extract_emotion(turns: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract a rough emotional label from the intensity values of a set of turns.
    Falls back to None if no intensity data is available.
    """
    intensities = [t.get("intensity") for t in turns if t.get("intensity") is not None]
    if not intensities:
        return None
    avg = sum(intensities) / len(intensities)
    if avg >= 0.7:
        return "high distress"
    elif avg >= 0.4:
        return "moderate distress"
    else:
        return "calm/positive"


