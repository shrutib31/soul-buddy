"""
Insights API

Exposes intelligence-layer data for the team dashboard and user profile:

  GET /api/v1/insights/session/{conversation_id}
      Per-session metrics (emotional_stability, engagement, progress, risk, mode_preference).
      Requires authentication — user must own the session.

  GET /api/v1/insights/session/{conversation_id}/summary
      The incremental or final session summary (JSONB) for a session.

  GET /api/v1/insights/user/weekly
      Weekly growth score trend for the authenticated user (last 7 sessions).

  GET /api/v1/insights/user/memory
      User's long-term memory: growth_summary, recurring_themes, behavioral_patterns,
      risk_signals, emotional_baseline.  Used to display progress on the profile page.
"""

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from typing import Optional

from api.supabase_auth import optional_supabase_token, verify_supabase_token
from config.sqlalchemy_db import SQLAlchemyDataDB
from orm.models import SbConversation, UserInsight, UserMemory, SessionSummary
from services.cache_service import cache_service
from services.conversation_insight_service import (
    generate_session_insight,
    get_user_insights as get_conversation_insights,
    get_session_insight as get_conversation_session_insight,
)
from services.summarization_service import summarize_session_final, update_user_memory

router = APIRouter(prefix="/insights")
logger = logging.getLogger(__name__)

_data_db = SQLAlchemyDataDB()


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/session/{conversation_id}")
async def get_session_insights(
    conversation_id: str,
    user=Depends(verify_supabase_token),
):
    """
    Return all insight metrics computed for a session.

    Verifies that the conversation belongs to the authenticated user before
    returning any data (returns 404 otherwise to avoid information disclosure).
    """
    if not _is_valid_uuid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id — must be a UUID")

    supabase_uid: str = user["id"]

    try:
        async with _data_db.get_session() as session:
            # Ownership check: confirm the conversation belongs to this user
            conv_stmt = select(SbConversation).where(
                SbConversation.id == uuid.UUID(conversation_id),
                SbConversation.supabase_user_id == uuid.UUID(supabase_uid),
            )
            conv_result = await session.execute(conv_stmt)
            if conv_result.scalar_one_or_none() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Fetch all insight rows for this session
            stmt = (
                select(UserInsight)
                .where(UserInsight.session_id == uuid.UUID(conversation_id))
                .order_by(UserInsight.computed_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        metrics = [
            {
                "metric_type": r.metric_type,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "metadata": r.metric_metadata,
                "computed_at": r.computed_at.isoformat() if r.computed_at else None,
            }
            for r in rows
        ]

        return {"conversation_id": conversation_id, "metrics": metrics}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "get_session_insights failed | conversation_id=%s error=%s", conversation_id, exc, exc_info=True
        )
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve session insights"})


@router.get("/session/{conversation_id}/summary")
async def get_session_summary(
    conversation_id: str,
    user=Depends(verify_supabase_token),
):
    """
    Return the session summary for a conversation.

    Returns the final holistic summary if the session is finalised,
    otherwise returns the latest incremental summary.
    Checks the Redis cache first (TTL 2 h); falls back to DB on miss.
    """
    if not _is_valid_uuid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id — must be a UUID")

    supabase_uid: str = user["id"]

    try:
        # Cache-aside — check before opening a DB session
        cached = await cache_service.get_session_summary(conversation_id)

        async with _data_db.get_session() as session:
            # Ownership check: confirm the conversation belongs to this user
            conv_stmt = select(SbConversation).where(
                SbConversation.id == uuid.UUID(conversation_id),
                SbConversation.supabase_user_id == uuid.UUID(supabase_uid),
            )
            conv_result = await session.execute(conv_stmt)
            if conv_result.scalar_one_or_none() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")

            if cached:
                return {"conversation_id": conversation_id, "summary": cached, "source": "cache"}

            stmt = select(SessionSummary).where(
                SessionSummary.session_id == uuid.UUID(conversation_id)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

        if row is None:
            return {"conversation_id": conversation_id, "summary": None,
                    "message": "No summary yet — summaries are generated every 5 turns"}

        summary = row.final_summary or row.incremental_summary
        is_finalised = row.is_finalised

        # Populate cache
        if summary:
            await cache_service.set_session_summary(conversation_id, summary)

        return {
            "conversation_id": conversation_id,
            "summary": summary,
            "is_finalised": is_finalised,
            "source": "db",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "get_session_summary failed | conversation_id=%s error=%s", conversation_id, exc, exc_info=True
        )
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve session summary"})


@router.get("/user/weekly")
async def get_weekly_growth(user=Depends(verify_supabase_token)):
    """
    Return the most recent weekly_growth_score for the authenticated user.

    The weekly_growth_score is a linear regression slope over the last 7
    session progress_scores — positive = improving, negative = worsening.
    """
    supabase_uid: str = user["id"]

    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(UserInsight)
                .where(
                    UserInsight.user_id == uuid.UUID(supabase_uid),
                    UserInsight.metric_name == "weekly_growth_score",
                )
                .order_by(UserInsight.computed_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

        if row is None:
            return {"weekly_growth_score": None, "trend": None, "message": "Not enough data yet"}

        return {
            "weekly_growth_score": row.metric_value,
            "trend": (row.metric_metadata or {}).get("trend"),
            "sessions_analysed": (row.metric_metadata or {}).get("sessions_analysed"),
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        }

    except Exception as exc:
        logger.error("get_weekly_growth failed | user=%s error=%s", supabase_uid, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve weekly growth"})


@router.get("/user/memory")
async def get_user_memory(user=Depends(verify_supabase_token)):
    """
    Return the authenticated user's long-term memory record.

    Checks the Redis cache first (TTL 24 h); falls back to DB on miss.
    Used by the frontend profile page to display the user's growth journey.
    """
    supabase_uid: str = user["id"]

    try:
        # Cache-aside: try Redis first
        cached = await cache_service.get_user_memory(supabase_uid)
        if cached:
            return {"memory": cached, "source": "cache"}

        async with _data_db.get_session() as session:
            stmt = select(UserMemory).where(
                UserMemory.user_id == uuid.UUID(supabase_uid)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

        if row is None:
            return {"memory": None, "message": "No memory record yet — start chatting to build your profile"}

        memory = {
            "growth_summary": row.growth_summary,
            "recurring_themes": row.recurring_themes,
            "behavioral_patterns": row.behavioral_patterns,
            "emotional_baseline": row.emotional_baseline,
            "preferred_modes": row.preferred_modes,
            "risk_signals": row.risk_signals,
            "last_updated": row.last_updated.isoformat() if row.last_updated else None,
        }

        # Populate cache for next request
        await cache_service.set_user_memory(supabase_uid, memory)

        return {"memory": memory, "source": "db"}

    except Exception as exc:
        logger.error("get_user_memory failed | user=%s error=%s", supabase_uid, exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve user memory"})


# ============================================================================
# END SESSION + CONVERSATION INSIGHTS
# ============================================================================

class EndSessionRequest(BaseModel):
    conversation_id: str = Field(..., description="The session to end and generate insights for")
    source_type: str = Field("chat", description="'chat' or 'journal'")
    is_incognito: bool = Field(True, description="True for anonymous sessions")
    chat_mode: str = Field("default", description="Chat mode: default | reflection | venting | therapist")


@router.post("/end-session")
async def end_session(req: EndSessionRequest, user=Depends(optional_supabase_token)):
    """
    End a chat/journal session and generate insights.

    For cognito users: generates insight, persists it, and returns it.
    For incognito users: generates insight and returns it (not stored).

    The frontend should call this when the user clicks the close/end button.
    """
    if not req.is_incognito and user is None:
        raise HTTPException(status_code=401, detail="Authorization header required for cognito mode")

    user_id: Optional[str] = None if req.is_incognito else user["id"]

    try:
        insight_data = await generate_session_insight(
            conversation_id=req.conversation_id,
            user_id=user_id,
            source_type=req.source_type,
        )

        # For cognito users, also finalise session summary + update user memory
        if user_id:
            try:
                final_summary = await summarize_session_final(
                    conversation_id=req.conversation_id,
                    user_id=user_id,
                    dominant_mode=req.chat_mode,
                )
                if final_summary:
                    await update_user_memory(
                        user_id=user_id,
                        new_session_summary=final_summary,
                    )
            except Exception as exc:
                logger.warning("end_session: summary/memory update failed (non-fatal) | error=%s", exc)

        if insight_data:
            return {
                "success": True,
                "conversation_id": req.conversation_id,
                "insight": insight_data,
            }
        else:
            return {
                "success": True,
                "conversation_id": req.conversation_id,
                "insight": None,
                "message": "Session ended. Not enough conversation data to generate insights.",
            }

    except Exception as e:
        logger.error("end_session failed | conv=%s error=%s", req.conversation_id, e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to end session: {str(e)}"},
        )


@router.get("/conversation-insights")
async def list_conversation_insights(
    user=Depends(verify_supabase_token),
    insight_type: Optional[str] = Query(None, description="Filter: session, daily, weekly, monthly"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """
    List conversation insights for the authenticated user.
    Supports filtering by type: session, daily, weekly, monthly.
    """
    user_id = user["id"]
    try:
        insights = await get_conversation_insights(
            user_id=user_id,
            insight_type=insight_type,
            limit=limit,
        )
        return {"success": True, "insights": insights, "count": len(insights)}
    except Exception as e:
        logger.error("list_conversation_insights failed | user=%s error=%s", user_id, e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to retrieve insights: {str(e)}"},
        )


@router.get("/conversation-insights/{conversation_id}")
async def get_insight_for_session(
    conversation_id: str,
    _user=Depends(verify_supabase_token),
):
    """Get the conversation insight generated for a specific session."""
    if not _is_valid_uuid(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id — must be a UUID")
    try:
        insight = await get_conversation_session_insight(conversation_id)
        if not insight:
            return {"success": True, "insight": None, "message": "No insight found for this session."}
        return {"success": True, "insight": insight}
    except Exception as e:
        logger.error("get_insight_for_session failed | conv=%s error=%s", conversation_id, e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to retrieve insight: {str(e)}"},
        )
