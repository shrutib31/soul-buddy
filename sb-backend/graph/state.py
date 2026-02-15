from typing import Optional, Dict, Any
from pydantic import BaseModel


class ConversationState(BaseModel):
    conversation_id: str
    mode: str                    # incognito / cognito
    domain: str                  # student / employee / general
    user_message: str

    # intent detection
    intent: Optional[str] = None

    # safety
    risk_level: str = "low"

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

    # context
    page_context: Dict[str, Any] = {} # from which UI page it is coming from (e.g., dashboard, profile, settings)
    domain_config: Dict[str, Any] = {} # domain-specific configurations
    user_personality_profile: Dict[str, Any] = {} # personality traits of the user to be populated from the db
    user_preferences: Dict[str, Any] = {} # user preferences to be populated from the db

    # metadata
    error: Optional[str] = None
    
    #Guardrail Helpers
    guardrail_status: Optional[str] = None
    guardrail_feedback: Optional[str] = None
    attempt: Optional[int] = 0
    
    # This allows the render node to save the final JSON response here
    api_response: Optional[Dict[str, Any]] = None
