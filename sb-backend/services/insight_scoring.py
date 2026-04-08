"""
InsightScoringService

Rule-based, zero-LLM insight scoring for the team dashboard.
Computes quantifiable signals from raw turn/context data and writes
them to the user_insights table after every bot response.

Scores computed per session:
  emotional   → emotional_stability_score
  engagement  → engagement_score
  progress    → progress_score
  behavioral  → mode_preference (dominant mode this session)
  safety      → risk_score (derived from risk_level in state)

Scores computed weekly (called from insight_scheduler.py):
  progress    → weekly_growth_score (trend across sessions)

All methods are fail-silent — exceptions are logged and swallowed
so insight scoring never blocks the main conversation flow.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func

from config.sqlalchemy_db import SQLAlchemyDataDB
from orm.models import ConversationContext, ConversationTurn, UserInsight

logger = logging.getLogger(__name__)

_data_db = SQLAlchemyDataDB()


# ============================================================================
# SESSION-LEVEL SCORING  (called from store_bot_response_node)
# ============================================================================

async def score_session(
    conversation_id: str,
    user_id: str,
    turn_count: int,
    session_duration_seconds: Optional[float],
    risk_level: str,
    dominant_mode: str,
) -> None:
    """
    Compute and persist all session-level insight metrics.
    Fire-and-forget safe — catches all exceptions.
    """
    try:
        contexts = await _fetch_contexts(conversation_id)

        metrics: List[Dict[str, Any]] = []

        # 1. Emotional stability score
        stability = _emotional_stability(contexts)
        if stability is not None:
            metrics.append({
                "metric_type": "emotional",
                "metric_name": "emotional_stability_score",
                "metric_value": round(stability, 3),
                "metadata": {"turn_count": turn_count},
            })

        # 2. Progress score (emotional improvement + engagement)
        progress = _progress_score(contexts)
        if progress is not None:
            metrics.append({
                "metric_type": "progress",
                "metric_name": "progress_score",
                "metric_value": round(progress, 3),
                "metadata": {"dominant_mode": dominant_mode},
            })

        # 3. Engagement score
        engagement = _engagement_score(turn_count, session_duration_seconds)
        metrics.append({
            "metric_type": "engagement",
            "metric_name": "engagement_score",
            "metric_value": round(engagement, 3),
            "metadata": {
                "turn_count": turn_count,
                "duration_seconds": session_duration_seconds,
            },
        })

        # 4. Mode preference (dominant mode this session)
        mode_dist = _mode_distribution(contexts)
        metrics.append({
            "metric_type": "behavioral",
            "metric_name": "mode_preference",
            "metric_value": None,
            "metadata": {"dominant_mode": dominant_mode, "mode_distribution": mode_dist},
        })

        # 5. Safety / risk score
        risk_value = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(risk_level, 0.1)
        metrics.append({
            "metric_type": "safety",
            "metric_name": "risk_score",
            "metric_value": risk_value,
            "metadata": {"risk_level": risk_level},
        })

        await _persist_metrics(user_id, conversation_id, metrics)

        logger.info(
            "insight_scoring: session metrics written | conversation_id=%s metrics=%d",
            conversation_id, len(metrics),
        )

    except Exception as exc:
        logger.warning("insight_scoring: score_session failed | conversation_id=%s error=%s", conversation_id, exc)


# ============================================================================
# WEEKLY GROWTH SCORE  (called from insight_scheduler.py)
# ============================================================================

async def score_weekly_growth(user_id: str) -> None:
    """
    Compute the weekly growth score from the last 7 sessions' progress_scores.
    Persists as a user_insight row (no session_id — it's a user-level metric).
    Fire-and-forget safe.
    """
    try:
        async with _data_db.get_session() as session:
            stmt = (
                select(UserInsight.metric_value, UserInsight.computed_at)
                .where(
                    UserInsight.user_id == uuid.UUID(user_id),
                    UserInsight.metric_name == "progress_score",
                )
                .order_by(UserInsight.computed_at.desc())
                .limit(7)
            )
            result = await session.execute(stmt)
            rows = result.all()

        if len(rows) < 2:
            return  # not enough data for a trend

        scores = [r.metric_value for r in reversed(rows) if r.metric_value is not None]
        if len(scores) < 2:
            return

        # Linear trend: positive = improving, negative = worsening
        growth = _linear_trend(scores)
        trend_label = "improving" if growth > 0.05 else ("worsening" if growth < -0.05 else "stable")

        await _persist_metrics(user_id, None, [{
            "metric_type": "progress",
            "metric_name": "weekly_growth_score",
            "metric_value": round(growth, 3),
            "metadata": {
                "trend": trend_label,
                "sessions_analysed": len(scores),
                "scores": scores,
            },
        }])

        logger.info("insight_scoring: weekly growth score written | user_id=%s trend=%s", user_id, trend_label)

    except Exception as exc:
        logger.warning("insight_scoring: score_weekly_growth failed | user_id=%s error=%s", user_id, exc)


# ============================================================================
# SCORING ALGORITHMS
# ============================================================================

def _emotional_stability(contexts: List[Dict]) -> Optional[float]:
    """
    stability = 1 - variance(intensity over turns)
    High variance → low stability (user's emotional state fluctuated a lot).
    Returns None if not enough intensity data.
    """
    intensities = [c["intensity"] for c in contexts if c.get("intensity") is not None]
    if len(intensities) < 2:
        return None
    mean = sum(intensities) / len(intensities)
    variance = sum((x - mean) ** 2 for x in intensities) / len(intensities)
    # variance is in [0, 1] since intensity ∈ [0, 1]
    return max(0.0, 1.0 - variance)


def _progress_score(contexts: List[Dict]) -> Optional[float]:
    """
    progress = (avg_intensity_last_quarter - avg_intensity_first_quarter) * -1
              + engagement_weight

    Negative emotional shift (intensity going down) = positive progress.
    Returns None if not enough data.
    """
    intensities = [c["intensity"] for c in contexts if c.get("intensity") is not None]
    if len(intensities) < 4:
        return None

    quarter = max(1, len(intensities) // 4)
    start_avg = sum(intensities[:quarter]) / quarter
    end_avg = sum(intensities[-quarter:]) / quarter
    # Improvement = intensity decreased (user became calmer)
    emotional_improvement = (start_avg - end_avg)  # positive if calmer at end
    # Normalize to [-1, 1] and shift to [0, 1]
    score = (emotional_improvement + 1.0) / 2.0
    return max(0.0, min(1.0, score))


def _engagement_score(turn_count: int, duration_seconds: Optional[float]) -> float:
    """
    engagement = weighted average of turn count and session duration signals.
    Both signals are normalized then averaged.
    - turn_count: 10+ turns = full engagement (1.0), 0 turns = 0.0
    - duration: 15+ minutes = full engagement (1.0)
    """
    turn_score = min(1.0, turn_count / 10.0)
    if duration_seconds is not None:
        duration_score = min(1.0, duration_seconds / 900.0)  # 900s = 15 min
        return round((turn_score + duration_score) / 2.0, 3)
    return round(turn_score, 3)


def _mode_distribution(contexts: List[Dict]) -> Dict[str, float]:
    """
    Returns percentage of turns spent in each mode.
    e.g. {"therapist": 0.6, "venting": 0.4}
    """
    if not contexts:
        return {}
    counts: Dict[str, int] = {}
    for c in contexts:
        mode = c.get("mode") or "default"
        counts[mode] = counts.get(mode, 0) + 1
    total = len(contexts)
    return {mode: round(count / total, 2) for mode, count in counts.items()}


def _linear_trend(values: List[float]) -> float:
    """
    Returns the slope of the linear regression through the values.
    Positive slope = improving trend, negative = worsening.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ============================================================================
# DB HELPERS
# ============================================================================

async def _fetch_contexts(conversation_id: str) -> List[Dict[str, Any]]:
    """Fetch all conversation_context rows for a session."""
    try:
        session_uuid = uuid.UUID(conversation_id)
        async with _data_db.get_session() as session:
            stmt = (
                select(ConversationContext)
                .join(ConversationTurn, ConversationTurn.id == ConversationContext.turn_id)
                .where(ConversationTurn.session_id == session_uuid)
                .order_by(ConversationTurn.turn_index.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {"mode": r.mode, "style": r.style, "intensity": r.intensity}
                for r in rows
            ]
    except Exception as exc:
        logger.warning("insight_scoring: context fetch failed error=%s", exc)
        return []


async def _persist_metrics(
    user_id: str,
    conversation_id: Optional[str],
    metrics: List[Dict[str, Any]],
) -> None:
    """Bulk-insert UserInsight rows."""
    try:
        now = datetime.now(timezone.utc)
        user_uuid = uuid.UUID(user_id)
        conv_uuid = uuid.UUID(conversation_id) if conversation_id else None

        async with _data_db.get_session() as session:
            for m in metrics:
                row = UserInsight(
                    user_id=user_uuid,
                    session_id=conv_uuid,
                    metric_type=m["metric_type"],
                    metric_name=m["metric_name"],
                    metric_value=m.get("metric_value"),
                    metric_metadata=m.get("metadata"),
                    computed_at=now,
                )
                session.add(row)
            await session.commit()
    except Exception as exc:
        logger.warning("insight_scoring: persist_metrics failed error=%s", exc)
