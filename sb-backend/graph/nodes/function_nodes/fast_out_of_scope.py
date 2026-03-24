"""
Fast out-of-scope detection node.

Runs after conversation setup/context loading and before the slower
classification/generation path.
"""

from typing import Dict, Any
import logging
import uuid

from graph.state import ConversationState
from graph.nodes.agentic_nodes.guardrail import detect_pattern_reason
from graph.nodes.agentic_nodes.response_templates import get_out_of_scope_response

logger = logging.getLogger(__name__)


async def out_of_scope_node(state: ConversationState) -> Dict[str, Any]:
    """
    Detect cheap, high-confidence out-of-scope cases before expensive graph work.

    Returns:
        Empty dict when the message should continue through the normal graph.
        Otherwise returns a render-ready out-of-scope state update.
    """
    out_of_scope_reason = detect_pattern_reason(state.user_message)
    if out_of_scope_reason is None:
        return {}

    logger.info(
        "out_of_scope: matched pattern",
        extra={"reason": out_of_scope_reason, "domain": state.domain},
    )
    return {
        "conversation_id": state.conversation_id or str(uuid.uuid4()),
        "intent": "out_of_scope",
        "situation": "NO_SITUATION",
        "severity": "low",
        "risk_level": "low",
        "is_out_of_scope": True,
        "out_of_scope_reason": out_of_scope_reason,
        "response_draft": get_out_of_scope_response(state.domain, reason=out_of_scope_reason),
    }


def out_of_scope_router(state: ConversationState) -> str:
    if getattr(state, "is_out_of_scope", False):
        return "render"
    return "classification_node"
