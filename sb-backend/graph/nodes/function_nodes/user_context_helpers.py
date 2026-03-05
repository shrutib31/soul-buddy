"""
Helpers for loading user context for cognito flows.

This module resolves user identity values and fetches profile context
from auth database tables.
"""

from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text

from config.supabase import get_user_by_id, verify_token
from config.sqlalchemy_db import SQLAlchemyAuthDB


auth_db = SQLAlchemyAuthDB()

DETAILED_PROFILE_FIELDS = [
    "first_name",
    "last_name",
    "date_of_birth",
    "age",
    "age_group",
    "gender",
    "pronouns",
    "phone",
    "preferred_contact_method",
    "country",
    "state",
    "city",
    "pincode",
    "timezone",
    "geo_coordinates",
    "education_level",
    "occupation",
    "industry",
    "income_range",
    "marital_status",
    "languages",
    "religion",
    "ethnicity",
    "hobbies",
    "interests",
    "communication_language",
    "subscription_status",
]


async def resolve_supabase_uid_from_app_user_id(app_user_id: int) -> str:
    """
    Resolve Supabase UID from internal app user_id.

    Primary source: auth-db public.users.id -> users.supabase_uid
    Fallback: config.supabase.get_user_by_id flow
    Raises ValueError when resolution fails.
    """
    # 1) Preferred path: souloxy-db users table maps app user id -> supabase uid.
    stmt = text(
        """
        SELECT supabase_uid
        FROM public.users
        WHERE id = :user_id
        LIMIT 1
        """
    )
    async with auth_db.get_session() as session:
        result = await session.execute(stmt, {"user_id": app_user_id})
        row = result.mappings().first()
        if row:
            resolved_supabase_uid = row.get("supabase_uid")
            if resolved_supabase_uid:
                return str(resolved_supabase_uid)

    # 2) Fallback to existing Supabase helper flow.
    try:
        user = await get_user_by_id(str(app_user_id))
    except Exception as e:
        raise ValueError(f"Failed to resolve supabase uid from user_id {app_user_id}: {str(e)}") from e

    resolved_supabase_uid = getattr(user, "id", None)
    if not resolved_supabase_uid:
        raise ValueError(f"Failed to resolve supabase uid from user_id {app_user_id}: missing id")

    return str(resolved_supabase_uid)


async def resolve_supabase_uid_from_payload_user_id(app_user_id: int) -> str:
    """
    Backward-compatible wrapper.
    The input is internal app user_id even though legacy variable names used
    "supabase_user_id" in older code.
    """
    return await resolve_supabase_uid_from_app_user_id(app_user_id)


async def resolve_app_user_id_from_supabase_uid(supabase_uid: str) -> int:
    """
    Resolve internal app user_id from Supabase UID using auth-db public.users.
    Raises ValueError when resolution fails.
    """
    stmt = text(
        """
        SELECT id
        FROM public.users
        WHERE supabase_uid = :supabase_uid
        LIMIT 1
        """
    )

    async with auth_db.get_session() as session:
        result = await session.execute(stmt, {"supabase_uid": supabase_uid})
        row = result.mappings().first()
        if not row or row.get("id") is None:
            raise ValueError(f"No app user_id found for supabase_uid {supabase_uid}")

        try:
            return int(row["id"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid app user_id for supabase_uid {supabase_uid}") from e


async def resolve_cognito_identity_from_access_token(access_token: str) -> Tuple[str, int]:
    """
    Resolve both IDs needed by cognito mode from access token:
    - supabase_uid (string)
    - app user_id (integer)
    """
    token = (access_token or "").strip()
    if not token:
        raise ValueError("Missing access_token header for cognito mode")
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        raise ValueError("Invalid access_token header for cognito mode")

    try:
        user = await verify_token(token)
    except Exception as e:
        raise ValueError(f"Failed to verify access_token: {str(e)}") from e

    supabase_uid = getattr(user, "id", None)
    if not supabase_uid and isinstance(user, dict):
        supabase_uid = user.get("id")
    if not supabase_uid:
        raise ValueError("Failed to resolve supabase uid from access_token: missing id")

    app_user_id = await resolve_app_user_id_from_supabase_uid(str(supabase_uid))
    return str(supabase_uid), app_user_id


async def fetch_user_detailed_profile(app_user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed profile row by integer app user_id.
    Returns only approved fields, or None if not found.
    """
    columns = ", ".join(DETAILED_PROFILE_FIELDS)
    stmt = text(
        f"""
        SELECT {columns}
        FROM public.user_detailed_profiles
        WHERE user_id = :user_id
        LIMIT 1
        """
    )

    async with auth_db.get_session() as session:
        result = await session.execute(stmt, {"user_id": app_user_id})
        row = result.mappings().first()
        return dict(row) if row else None


async def fetch_user_personality_profile(supabase_uid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch full personality profile json by Supabase UID.
    Returns personality_profile_data JSON dict or None if not found.
    """
    stmt = text(
        """
        SELECT personality_profile_data
        FROM public.user_personality_profiles
        WHERE supabase_uid = :supabase_uid
        LIMIT 1
        """
    )

    async with auth_db.get_session() as session:
        result = await session.execute(stmt, {"supabase_uid": supabase_uid})
        row = result.mappings().first()
        if not row:
            return None
        data = row.get("personality_profile_data")
        return data if isinstance(data, dict) else None
