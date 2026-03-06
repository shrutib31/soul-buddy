from typing import Optional, Dict, Any, List
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ConversationState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str
    mode: str                    # incognito / cognito
    domain: str                  # student / employee / general
    user_message: str
    supabase_uid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("supabase_uid", "user_id"),
    )  # Supabase auth uid
    app_user_id: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("app_user_id", "supabase_user_id"),
    )  # Internal app user id (public.users.id)

    # intent detection
    intent: Optional[str] = None
    is_greeting: bool = False

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
    #user_preferences: Dict[str, Any] = {}        # user preferences; DB schema pending, cached when available
    user_profile: Dict[str, Any] = {} # user profile info to be populated frmo db # SHOUDL CHANGE TO user_preferences
    conversation_history: List[Dict[str, Any]] = []  # last N turns [{speaker, message, turn_index}]
    conversation_summary: Optional[str] = None       # latest summarised context (written by summarisation job)

    # metadata
    error: Optional[str] = None
    
    #Guardrail Helpers
    guardrail_status: Optional[str] = None
    guardrail_feedback: Optional[str] = None
    attempt: Optional[int] = 0
    
    # This allows the render node to save the final JSON response here
    api_response: Optional[Dict[str, Any]] = None
