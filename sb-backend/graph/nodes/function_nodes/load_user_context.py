"""
User Context Loader (utility)

Fetches user profile and personality data from PostgreSQL tables
for cognito mode conversations. Intended to be called from agentic nodes
that need enriched context (e.g., response generation).
"""

from typing import Dict, Any

from graph.state import ConversationState
from graph.nodes.function_nodes.user_context_helpers import (
    fetch_user_detailed_profile,
    fetch_user_personality_profile,
)


async def load_user_context(state: ConversationState) -> Dict[str, Any]:
    """
    Hydrate cognito sessions with user context from Postgres profile tables.
    Returns partial state updates. If mode is not cognito, returns {}.
    """
    try:
        if state.mode != "cognito":
            return {}

        if not state.user_id or not str(state.user_id).strip():
            return {"error": "Missing supabase user_id for cognito mode"}
        if state.supabase_user_id is None:
            return {"error": "Missing supabase_user_id for cognito mode"}

        updates: Dict[str, Any] = {}

        detailed_profile = await fetch_user_detailed_profile(state.supabase_user_id)
        if isinstance(detailed_profile, dict):
            updates["user_profile"] = detailed_profile

        personality_profile = await fetch_user_personality_profile(str(state.user_id))
        if isinstance(personality_profile, dict):
            updates["user_personality_profile"] = personality_profile

        return updates

    except Exception as e:
        return {"error": f"Error loading user context: {str(e)}"}
