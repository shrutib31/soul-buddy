"""
SSE streaming for the SoulBuddy conversation graph.

Uses flow.astream() which yields {node_name: state_updates} after each node
completes. Only the nodes that produce client-visible output emit SSE events;
internal nodes (store_message, conv_id_handler, etc.) emit a lightweight
node_complete event so the client can show progress if desired.
"""

import json
import logging
from typing import AsyncGenerator

from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow

logger = logging.getLogger(__name__)


async def stream_as_sse(state: ConversationState) -> AsyncGenerator[str, None]:
    """
    Run the graph and yield SSE-formatted events as each node completes.

    Event types emitted:
      {"type": "node_complete",    "node": "<name>"}
      {"type": "response_chunk",   "content": "<draft text>"}
      {"type": "complete",         "data": {<api_response dict>}}
      {"type": "done"}
      {"type": "error",            "error": "<message>"}

    Each event is formatted as:   data: <json>\n\n
    """
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        flow = get_compiled_flow()

        async for chunk in flow.astream(state.model_dump()):
            for node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue

                if node_name == "render":
                    api_response = node_output.get("api_response")
                    if api_response:
                        yield _sse({"type": "complete", "data": api_response})
                    else:
                        yield _sse({"type": "node_complete", "node": node_name})

                elif node_name == "response_generator":
                    draft = node_output.get("response_draft")
                    if draft:
                        yield _sse({"type": "response_chunk", "content": draft})
                    else:
                        yield _sse({"type": "node_complete", "node": node_name})

                else:
                    yield _sse({"type": "node_complete", "node": node_name})

        yield _sse({"type": "done"})

    except Exception as e:
        logger.exception("stream_as_sse failed")
        yield _sse({"type": "error", "error": str(e)})
