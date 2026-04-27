"""
InsightRollupService

Aggregates conversation insights across time windows:

  Daily   — rolls up all session insights for a given day into one daily insight
  Weekly  — rolls up 7 daily insights into one weekly insight, then deletes the dailies
  Monthly — rolls up ~4 weekly insights into one monthly insight

Called by the insight_scheduler cron jobs. All methods are fire-and-forget safe.

Rollup hierarchy:
  session → daily → weekly (dailies deleted) → monthly
"""

import json
import logging
import uuid
from datetime import datetime, date, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete, and_

from config.settings import settings
from config.sqlalchemy_db import SQLAlchemyDataDB
from orm.models import ConversationInsight

logger = logging.getLogger(__name__)

OPENAI_API_KEY = settings.openai.api_key
OLLAMA_BASE_URL = settings.ollama.base_url
OLLAMA_MODEL = settings.ollama.model
OLLAMA_TIMEOUT = settings.ollama.timeout
OLLAMA_FLAG = settings.llm.ollama_flag
_data_db = SQLAlchemyDataDB()


# ============================================================================
# LLM CALLER (shared pattern)
# ============================================================================

async def _call_gpt(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Optional[Dict]:
    if not OPENAI_API_KEY:
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
            "temperature": 0.3,
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
                err = await resp.text()
                logger.warning("insight_rollup: GPT error status=%d err=%s", resp.status, err[:200])
                return None
    except Exception as exc:
        logger.warning("insight_rollup: GPT call failed error=%s", exc)
        return None


async def _call_ollama(system_prompt: str, user_prompt: str) -> Optional[Dict]:
    if not OLLAMA_BASE_URL:
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
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    raw = raw.strip()
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        try:
                            return json.loads(raw[start:end + 1])
                        except json.JSONDecodeError:
                            pass
                    import re as _re
                    cleaned = _re.sub(r",\s*([}\]])", r"\1", raw)
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1:
                        try:
                            return json.loads(cleaned[start:end + 1])
                        except json.JSONDecodeError as exc2:
                            logger.warning("insight_rollup: Ollama JSON parse failed after cleanup error=%s raw=%s", exc2, raw[:300])
                    return None
                err = await resp.text()
                logger.warning("insight_rollup: Ollama error status=%d err=%s", resp.status, err[:200])
                return None
    except Exception as exc:
        logger.warning("insight_rollup: Ollama call failed error=%s", exc)
        return None


async def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Optional[Dict]:
    if OPENAI_API_KEY and not OLLAMA_FLAG:
        return await _call_gpt(system_prompt, user_prompt, max_tokens)
    if OLLAMA_FLAG:
        return await _call_ollama(system_prompt, user_prompt)
    if OPENAI_API_KEY:
        return await _call_gpt(system_prompt, user_prompt, max_tokens)
    return await _call_ollama(system_prompt, user_prompt)
# ROLLUP SCHEMAS
# ============================================================================

_DAILY_INSIGHT_SCHEMA = {
    "summary": "2-3 sentence summary of the day's sessions",
    "emotional_patterns": ["patterns across all sessions today"],
    "key_moments": ["notable moments from the day"],
    "tips": ["1-3 tips for tomorrow based on today"],
    "mood_trajectory": "how mood changed across sessions today",
    "session_count": 0,
}

_WEEKLY_INSIGHT_SCHEMA = {
    "summary": "3-4 sentence summary of the week",
    "recurring_themes": ["themes that appeared across the week"],
    "emotional_trends": "how emotional state evolved over the week",
    "growth_areas": ["areas of growth observed"],
    "tips_for_next_week": ["2-3 actionable tips for next week"],
    "highlight": "most positive moment of the week",
    "concern": "area that may need attention",
}

_MONTHLY_INSIGHT_SCHEMA = {
    "summary": "3-5 sentence summary of the month",
    "major_themes": ["dominant themes of the month"],
    "emotional_journey": "narrative of emotional progression",
    "growth_summary": "overall growth and changes observed",
    "strengths": ["strengths that emerged"],
    "areas_for_focus": ["areas to work on next month"],
    "milestones": ["any milestones or breakthroughs"],
}


# ============================================================================
# DAILY ROLLUP
# ============================================================================

async def generate_daily_rollup(user_id: str, target_date: date) -> Optional[Dict[str, Any]]:
    """
    Roll up all session insights for a given day into a single daily insight.
    """
    try:
        user_uuid = uuid.UUID(user_id)
        day_start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationInsight)
                .where(
                    ConversationInsight.user_id == user_uuid,
                    ConversationInsight.insight_type == "session",
                    ConversationInsight.created_at >= day_start,
                    ConversationInsight.created_at < day_end,
                )
                .order_by(ConversationInsight.created_at.asc())
            )
            result = await session.execute(stmt)
            session_insights = result.scalars().all()

        if not session_insights:
            return None

        # Combine all session insight_data into input for LLM
        sessions_text = []
        for i, si in enumerate(session_insights, 1):
            data = si.insight_data or {}
            source = si.source_type or "chat"
            sessions_text.append(
                f"Session {i} ({source}):\n"
                f"  Summary: {data.get('summary', 'N/A')}\n"
                f"  Patterns: {', '.join(data.get('emotional_patterns', []))}\n"
                f"  Tips given: {', '.join(data.get('tips', []))}\n"
                f"  Mood: {data.get('mood_start', '?')} → {data.get('mood_end', '?')}"
            )

        schema_str = json.dumps(_DAILY_INSIGHT_SCHEMA, indent=2)
        system_prompt = (
            "You are summarizing a user's mental wellness day from their session insights.\n"
            "Produce a warm, helpful daily summary.\n\n"
            "Output ONLY valid JSON matching this schema:\n"
            f"{schema_str}\n\n"
            "- Identify patterns ACROSS sessions, not just repeat each one\n"
            "- Tips should be forward-looking (for tomorrow)\n"
            "- Set session_count to the actual number of sessions\n"
            "- Keep tone warm and encouraging"
        )
        user_prompt = (
            f"Date: {target_date.isoformat()}\n"
            f"Number of sessions: {len(session_insights)}\n\n"
            + "\n\n".join(sessions_text)
            + "\n\nGenerate the daily insight JSON."
        )

        insight_data = await _call_llm(system_prompt, user_prompt, max_tokens=400)
        if not insight_data:
            return None

        insight_data["session_count"] = len(session_insights)

        # Persist
        async with _data_db.get_session() as session:
            row = ConversationInsight(
                user_id=user_uuid,
                session_id=None,
                insight_type="daily",
                source_type="chat",
                insight_data=insight_data,
                period_start=day_start,
                period_end=day_end,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            await session.commit()

        logger.info("insight_rollup: daily rollup generated | user=%s date=%s sessions=%d",
                     user_id, target_date, len(session_insights))
        return insight_data

    except Exception as exc:
        logger.warning("insight_rollup: daily rollup failed | user=%s error=%s", user_id, exc)
        return None


# ============================================================================
# WEEKLY ROLLUP
# ============================================================================

async def generate_weekly_rollup(user_id: str, week_end_date: date) -> Optional[Dict[str, Any]]:
    """
    Roll up the last 7 daily insights into a weekly insight.
    Deletes the daily insights after successful weekly creation.
    """
    try:
        user_uuid = uuid.UUID(user_id)
        week_start = week_end_date - timedelta(days=6)
        period_start = datetime.combine(week_start, time.min, tzinfo=timezone.utc)
        period_end = datetime.combine(week_end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationInsight)
                .where(
                    ConversationInsight.user_id == user_uuid,
                    ConversationInsight.insight_type == "daily",
                    ConversationInsight.period_start >= period_start,
                    ConversationInsight.period_start < period_end,
                )
                .order_by(ConversationInsight.period_start.asc())
            )
            result = await session.execute(stmt)
            daily_insights = result.scalars().all()

        if not daily_insights:
            return None

        dailies_text = []
        for di in daily_insights:
            data = di.insight_data or {}
            day_str = di.period_start.strftime("%A %Y-%m-%d") if di.period_start else "?"
            dailies_text.append(
                f"{day_str}:\n"
                f"  Summary: {data.get('summary', 'N/A')}\n"
                f"  Patterns: {', '.join(data.get('emotional_patterns', []))}\n"
                f"  Mood trajectory: {data.get('mood_trajectory', '?')}\n"
                f"  Sessions: {data.get('session_count', '?')}"
            )

        schema_str = json.dumps(_WEEKLY_INSIGHT_SCHEMA, indent=2)
        system_prompt = (
            "You are summarizing a user's mental wellness week from their daily insights.\n"
            "Produce a warm, insightful weekly summary.\n\n"
            "Output ONLY valid JSON matching this schema:\n"
            f"{schema_str}\n\n"
            "- Identify TRENDS across the week, not just list each day\n"
            "- Note growth areas — what improved?\n"
            "- Highlight one positive moment and one concern\n"
            "- Tips should be specific and achievable for next week\n"
            "- Keep tone warm and constructive"
        )
        user_prompt = (
            f"Week: {week_start.isoformat()} to {week_end_date.isoformat()}\n"
            f"Days with data: {len(daily_insights)}\n\n"
            + "\n\n".join(dailies_text)
            + "\n\nGenerate the weekly insight JSON."
        )

        insight_data = await _call_llm(system_prompt, user_prompt, max_tokens=500)
        if not insight_data:
            return None

        # Persist weekly and delete dailies in one transaction
        daily_ids = [di.id for di in daily_insights]
        async with _data_db.get_session() as session:
            row = ConversationInsight(
                user_id=user_uuid,
                session_id=None,
                insight_type="weekly",
                source_type="chat",
                insight_data=insight_data,
                period_start=period_start,
                period_end=period_end,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)

            # Delete daily insights for this week
            del_stmt = delete(ConversationInsight).where(
                ConversationInsight.id.in_(daily_ids)
            )
            await session.execute(del_stmt)
            await session.commit()

        logger.info("insight_rollup: weekly rollup generated | user=%s week=%s..%s dailies=%d (deleted)",
                     user_id, week_start, week_end_date, len(daily_insights))
        return insight_data

    except Exception as exc:
        logger.warning("insight_rollup: weekly rollup failed | user=%s error=%s", user_id, exc)
        return None


# ============================================================================
# MONTHLY ROLLUP
# ============================================================================

async def generate_monthly_rollup(user_id: str, year: int, month: int) -> Optional[Dict[str, Any]]:
    """
    Roll up weekly insights for a month into a monthly insight.
    """
    try:
        user_uuid = uuid.UUID(user_id)
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationInsight)
                .where(
                    ConversationInsight.user_id == user_uuid,
                    ConversationInsight.insight_type == "weekly",
                    ConversationInsight.period_start >= month_start,
                    ConversationInsight.period_start < month_end,
                )
                .order_by(ConversationInsight.period_start.asc())
            )
            result = await session.execute(stmt)
            weekly_insights = result.scalars().all()

        if not weekly_insights:
            return None

        weeklies_text = []
        for wi in weekly_insights:
            data = wi.insight_data or {}
            start = wi.period_start.strftime("%Y-%m-%d") if wi.period_start else "?"
            end = wi.period_end.strftime("%Y-%m-%d") if wi.period_end else "?"
            weeklies_text.append(
                f"Week {start} to {end}:\n"
                f"  Summary: {data.get('summary', 'N/A')}\n"
                f"  Themes: {', '.join(data.get('recurring_themes', []))}\n"
                f"  Trends: {data.get('emotional_trends', '?')}\n"
                f"  Growth: {', '.join(data.get('growth_areas', []))}"
            )

        schema_str = json.dumps(_MONTHLY_INSIGHT_SCHEMA, indent=2)
        system_prompt = (
            "You are summarizing a user's mental wellness month from their weekly insights.\n"
            "Produce a warm, comprehensive monthly summary.\n\n"
            "Output ONLY valid JSON matching this schema:\n"
            f"{schema_str}\n\n"
            "- Focus on the big picture — overall trajectory\n"
            "- Celebrate milestones and growth\n"
            "- Be honest but gentle about areas needing attention\n"
            "- Keep tone warm, supportive, and forward-looking"
        )
        user_prompt = (
            f"Month: {year}-{month:02d}\n"
            f"Weeks with data: {len(weekly_insights)}\n\n"
            + "\n\n".join(weeklies_text)
            + "\n\nGenerate the monthly insight JSON."
        )

        insight_data = await _call_llm(system_prompt, user_prompt, max_tokens=500)
        if not insight_data:
            return None

        async with _data_db.get_session() as session:
            row = ConversationInsight(
                user_id=user_uuid,
                session_id=None,
                insight_type="monthly",
                source_type="chat",
                insight_data=insight_data,
                period_start=month_start,
                period_end=month_end,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            await session.commit()

        logger.info("insight_rollup: monthly rollup generated | user=%s month=%d-%02d weeklies=%d",
                     user_id, year, month, len(weekly_insights))
        return insight_data

    except Exception as exc:
        logger.warning("insight_rollup: monthly rollup failed | user=%s error=%s", user_id, exc)
        return None


# ============================================================================
# BATCH ROLLUP RUNNERS (called by scheduler)
# ============================================================================

async def run_daily_rollups() -> None:
    """Generate daily rollups for all users who had sessions yesterday."""
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        day_start = datetime.combine(yesterday, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(yesterday + timedelta(days=1), time.min, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            from sqlalchemy import distinct
            stmt = (
                select(distinct(ConversationInsight.user_id))
                .where(
                    ConversationInsight.insight_type == "session",
                    ConversationInsight.created_at >= day_start,
                    ConversationInsight.created_at < day_end,
                )
            )
            result = await session.execute(stmt)
            user_ids = [str(uid) for uid in result.scalars().all()]

        logger.info("insight_rollup: daily rollup batch | users=%d date=%s", len(user_ids), yesterday)
        for uid in user_ids:
            await generate_daily_rollup(uid, yesterday)

    except Exception as exc:
        logger.warning("insight_rollup: run_daily_rollups failed error=%s", exc)


async def run_weekly_rollups() -> None:
    """Generate weekly rollups for all users who have 7 days of daily insights."""
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        # Weekly rollup covers the last 7 days ending yesterday
        week_start = yesterday - timedelta(days=6)
        period_start = datetime.combine(week_start, time.min, tzinfo=timezone.utc)
        period_end = datetime.combine(yesterday + timedelta(days=1), time.min, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            from sqlalchemy import distinct
            stmt = (
                select(distinct(ConversationInsight.user_id))
                .where(
                    ConversationInsight.insight_type == "daily",
                    ConversationInsight.period_start >= period_start,
                    ConversationInsight.period_start < period_end,
                )
            )
            result = await session.execute(stmt)
            user_ids = [str(uid) for uid in result.scalars().all()]

        logger.info("insight_rollup: weekly rollup batch | users=%d week=%s..%s", len(user_ids), week_start, yesterday)
        for uid in user_ids:
            await generate_weekly_rollup(uid, yesterday)

    except Exception as exc:
        logger.warning("insight_rollup: run_weekly_rollups failed error=%s", exc)


async def run_monthly_rollups() -> None:
    """Generate monthly rollups for the previous month."""
    try:
        today = datetime.now(timezone.utc).date()
        # Previous month
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        year = last_month_end.year
        month = last_month_end.month

        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        async with _data_db.get_session() as session:
            from sqlalchemy import distinct
            stmt = (
                select(distinct(ConversationInsight.user_id))
                .where(
                    ConversationInsight.insight_type == "weekly",
                    ConversationInsight.period_start >= month_start,
                    ConversationInsight.period_start < month_end,
                )
            )
            result = await session.execute(stmt)
            user_ids = [str(uid) for uid in result.scalars().all()]

        logger.info("insight_rollup: monthly rollup batch | users=%d month=%d-%02d", len(user_ids), year, month)
        for uid in user_ids:
            await generate_monthly_rollup(uid, year, month)

    except Exception as exc:
        logger.warning("insight_rollup: run_monthly_rollups failed error=%s", exc)
