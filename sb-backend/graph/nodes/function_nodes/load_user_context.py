"""
User Context Loader (utility)

Fetches user profile, personality, and domain configuration from Supabase
for cognito mode conversations. Intended to be called from agentic nodes
that need enriched context (e.g., response generation).
"""

from typing import Dict, Any

from graph.state import ConversationState
from config.supabase import get_user_by_id


def _extract_user_metadata(user: Any) -> Dict[str, Any]:
    """
    Safely pull user metadata regardless of Supabase client attribute names.
    """
    if hasattr(user, "user_metadata") and isinstance(user.user_metadata, dict):
        return user.user_metadata
    if hasattr(user, "raw_user_meta_data") and isinstance(user.raw_user_meta_data, dict):
        return user.raw_user_meta_data
    return {}


async def load_user_context(state: ConversationState) -> Dict[str, Any]:
    """
    Hydrate cognito sessions with stored user context from Supabase.
    Returns partial state updates. If mode is not cognito, returns {}.
    """
    try:
        if state.mode != "cognito":
            return {}

        if not state.user_id or not str(state.user_id).strip():
            return {"error": "Missing user_id for cognito mode"}

        user = await get_user_by_id(str(state.user_id))
        verified_user_id = getattr(user, "id", None)
        if not verified_user_id:
            return {"error": "User verification failed: missing id"}

        metadata = _extract_user_metadata(user)

        updates: Dict[str, Any] = {"user_id": str(verified_user_id)}

        profile = metadata.get("user_profile") or metadata.get("profile")
        if isinstance(profile, dict):
            updates["user_profile"] = profile

        personalities = metadata.get("user_personality_profiles") or metadata.get("personality_profiles")
        if isinstance(personalities, dict):
            updates["user_personality_profile"] = personalities

        domain_config = metadata.get("domain_config")
        if isinstance(domain_config, dict):
            updates["domain_config"] = domain_config

        domain = metadata.get("domain")
        if isinstance(domain, str) and domain.strip():
            updates["domain"] = domain

        return updates

    except Exception as e:
        return {"error": f"Error loading user context: {str(e)}"}
