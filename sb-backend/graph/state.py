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
    supabase_uid: Optional[str] = None  # supabase user ID (cognito only)
    chat_preference: str
    
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
    conversation_summary: Optional[str] = None       # latest summarised context (written by summarisation job)

    # metadata
    error: Annotated[Optional[str], _keep_last_error] = None
    
    #Guardrail Helpers
    guardrail_status: Optional[str] = None
    guardrail_feedback: Optional[str] = None
    attempt: Optional[int] = 0
    
    # This allows the render node to save the final JSON response here
    api_response: Optional[Dict[str, Any]] = None
