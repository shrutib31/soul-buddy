"""
InsightScheduler — background cron tasks for the intelligence layer.

Runs two periodic jobs inside the FastAPI process (no external worker needed):

  Daily (every 24 h):
    aggregate_daily_insights()
    ─ Rule-based, zero LLM cost.
    ─ Reads the last 24 h of user_insights rows, computes per-user
      average engagement / progress / emotional_stability, and logs a
      summary.  (Future: write aggregated rows to a daily_insights table.)

  Weekly (every 7 days):
    score_weekly_growth_for_all_users()
    ─ Calls InsightScoringService.score_weekly_growth() for every user
      who has had at least one session in the past 7 days.
    ─ Uses linear regression on the last 7 progress_scores to compute
      a weekly_growth_score row (still rule-based, no LLM).

Usage
-----
  Call start_scheduler() from the FastAPI lifespan on startup.
  Call stop_scheduler() from the lifespan on shutdown.

Both jobs are fire-and-forget inside asyncio — a failure in one run
is logged and swallowed so the scheduler never crashes the server.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select, distinct

from config.sqlalchemy_db import SQLAlchemyDataDB
from orm.models import UserInsight
from services.insight_scoring import score_weekly_growth
from services.insight_rollup_service import (
    run_daily_rollups,
    run_weekly_rollups,
    run_monthly_rollups,
)

logger = logging.getLogger(__name__)

_data_db = SQLAlchemyDataDB()

_DAILY_INTERVAL_SECONDS = 24 * 60 * 60   # 24 hours
_WEEKLY_INTERVAL_SECONDS = 7 * 24 * 60 * 60  # 7 days
_MONTHLY_INTERVAL_SECONDS = 30 * 24 * 60 * 60  # ~30 days

# Task handles — kept so stop_scheduler() can cancel them
_daily_task: Optional[asyncio.Task] = None
_weekly_task: Optional[asyncio.Task] = None
_monthly_task: Optional[asyncio.Task] = None


# ============================================================================
# PUBLIC API
# ============================================================================

def start_scheduler() -> None:
    """
    Schedule background cron tasks.  Call once from the FastAPI lifespan
    after all DB connections are ready.
    """
    global _daily_task, _weekly_task, _monthly_task
    _daily_task = asyncio.create_task(_run_daily_loop(), name="insight_daily_loop")
    _weekly_task = asyncio.create_task(_run_weekly_loop(), name="insight_weekly_loop")
    _monthly_task = asyncio.create_task(_run_monthly_loop(), name="insight_monthly_loop")
    logger.info("insight_scheduler: started (daily + weekly + monthly loops)")


async def stop_scheduler() -> None:
    """Cancel scheduled tasks gracefully on shutdown."""
    for task in (_daily_task, _weekly_task, _monthly_task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("insight_scheduler: stopped")


# ============================================================================
# LOOP RUNNERS
# ============================================================================

async def _run_daily_loop() -> None:
    """Sleep 24 h, run daily jobs (aggregate + conversation insight rollup), repeat."""
    while True:
        await asyncio.sleep(_DAILY_INTERVAL_SECONDS)
        try:
            await aggregate_daily_insights()
        except Exception as exc:
            logger.warning("insight_scheduler: daily aggregation failed error=%s", exc)
        try:
            await run_daily_rollups()
        except Exception as exc:
            logger.warning("insight_scheduler: daily insight rollup failed error=%s", exc)


async def _run_weekly_loop() -> None:
    """Sleep 7 days, run weekly jobs (growth scoring + insight rollup), repeat."""
    while True:
        await asyncio.sleep(_WEEKLY_INTERVAL_SECONDS)
        try:
            await score_weekly_growth_for_all_users()
        except Exception as exc:
            logger.warning("insight_scheduler: weekly growth scoring failed error=%s", exc)
        try:
            await run_weekly_rollups()
        except Exception as exc:
            logger.warning("insight_scheduler: weekly job failed error=%s", exc)


async def _run_monthly_loop() -> None:
    """Sleep ~30 days, run monthly insight rollup, repeat."""
    while True:
        await asyncio.sleep(_MONTHLY_INTERVAL_SECONDS)
        try:
            await run_monthly_rollups()
        except Exception as exc:
            logger.warning("insight_scheduler: monthly insight rollup failed error=%s", exc)


# ============================================================================
# JOB IMPLEMENTATIONS
# ============================================================================

async def aggregate_daily_insights() -> None:
    """
    Rule-based daily aggregation (zero LLM cost).

    Reads all user_insights rows created in the past 24 h and logs
    per-user averages for engagement, progress, and emotional_stability.

    This is intentionally lightweight — a starting point that can be
    extended to write rows into a daily_insights table as needed.
    """
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

        async with _data_db.get_session() as session:
            stmt = (
                select(UserInsight)
                .where(UserInsight.computed_at >= since)
                .order_by(UserInsight.user_id, UserInsight.metric_name)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        if not rows:
            logger.info("insight_scheduler: daily job — no new insights in last 24 h")
            return

        # Aggregate per user per metric
        aggregated: dict = {}
        for row in rows:
            key = (str(row.user_id), row.metric_name)
            if row.metric_value is not None:
                aggregated.setdefault(key, []).append(row.metric_value)

        user_count = len({k[0] for k in aggregated})
        metric_count = len(aggregated)
        logger.info(
            "insight_scheduler: daily aggregation complete | users=%d metric_series=%d",
            user_count, metric_count,
        )

        for (user_id, metric_name), values in aggregated.items():
            avg = sum(values) / len(values)
            logger.debug(
                "insight_scheduler: daily avg | user=%s metric=%s avg=%.3f n=%d",
                user_id, metric_name, avg, len(values),
            )

    except Exception as exc:
        logger.warning("insight_scheduler: aggregate_daily_insights failed error=%s", exc)


async def score_weekly_growth_for_all_users() -> None:
    """
    Run score_weekly_growth() for every user who had a session in the past 7 days.

    Uses a linear regression over the last 7 progress_scores to produce a
    weekly_growth_score row in user_insights (rule-based, no LLM).
    """
    try:
        since = datetime.now(timezone.utc) - timedelta(days=7)

        async with _data_db.get_session() as session:
            stmt = (
                select(distinct(UserInsight.user_id))
                .where(
                    UserInsight.computed_at >= since,
                    UserInsight.metric_name == "progress_score",
                )
            )
            result = await session.execute(stmt)
            user_ids: List[str] = [str(row) for row in result.scalars().all()]

        if not user_ids:
            logger.info("insight_scheduler: weekly job — no active users in last 7 days")
            return

        logger.info(
            "insight_scheduler: weekly growth scoring | users=%d", len(user_ids)
        )

        for user_id in user_ids:
            try:
                await score_weekly_growth(user_id)
            except Exception as exc:
                logger.warning(
                    "insight_scheduler: score_weekly_growth failed | user_id=%s error=%s",
                    user_id, exc,
                )

        logger.info("insight_scheduler: weekly growth scoring complete")

    except Exception as exc:
        logger.warning("insight_scheduler: score_weekly_growth_for_all_users failed error=%s", exc)
