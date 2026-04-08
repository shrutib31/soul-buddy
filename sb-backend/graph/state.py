from typing import Annotated, Optional, Dict, Any, List
from pydantic import BaseModel


def _keep_last_error(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Reducer for the error field: keep the most recent non-None error."""
    return b if b is not None else a


class ConversationState(BaseModel):
    conversation_id: str
    mode: str                    # incognito / cognito
    domain: str                  # student / employee / general
    user_message: str
    language: str = "en-IN"             # BCP-47 language tag sent by Sarvam STT / frontend
    supabase_uid: Optional[str] = None  # supabase user ID (cognito only)
    chat_preference: str = "general"
    chat_mode: str = "default"          # default / reflection / venting / therapist

    # intent detection
    intent: Optional[str] = None
    is_greeting: bool = False
    is_out_of_scope: bool = False
    out_of_scope_reason: Optional[str] = None

    # safety
    risk_level: str = "low"
    is_crisis_detected: bool = False

    # situation
    situation: Optional[str] = None
    severity: Optional[str] = None
    flow_id: Optional[str] = None

    # flow execution
    step_index: int = 0
    response_draft: str = ""

    # readiness & solutions
    readiness_score: int = 0
    tool: Optional[Dict[str, Any]] = None

    # context — all fields below are populated by load_user_context_node (cache-aside: Redis → DB)
    page_context: Dict[str, Any] = {}           # UI page the request originates from (e.g., dashboard, profile)
    domain_config: Dict[str, Any] = {}          # domain-specific config (student / employee / corporate)
    user_personality_profile: Dict[str, Any] = {} # personality traits; DB schema pending, cached when available
    user_preferences: Dict[str, Any] = {}        # user preferences; DB schema pending, cached when available
    conversation_history: List[Dict[str, Any]] = []  # last N turns [{speaker, message, turn_index}]
    conversation_summary: Optional[str] = None       # legacy field — kept for backward compat during transition

    # Intelligence layer context (populated by load_user_context_node)
    session_summary: Optional[Dict[str, Any]] = None   # current session's incremental/final summary (JSONB)
    user_memory: Optional[Dict[str, Any]] = None        # user's long-term memory row (growth_summary, themes, etc.)
    emotion_intensity: Optional[float] = None           # emotion intensity for current turn (0.0–1.0), set by classifier
    is_new_session: bool = False                        # True on first turn of a brand-new conversation

    # metadata
    error: Annotated[Optional[str], _keep_last_error] = None
    
    #Guardrail Helpers
    guardrail_status: Optional[str] = None
    guardrail_feedback: Optional[str] = None
    attempt: Optional[int] = 0
    
    # This allows the render node to save the final JSON response here
    api_response: Optional[Dict[str, Any]] = None
