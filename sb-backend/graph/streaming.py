"""
Streaming-enabled LangGraph for Soul Buddy

This module provides a streaming version of the conversation graph that yields
state updates as each node completes, enabling real-time streaming responses to clients.
"""

from typing import AsyncGenerator, Dict, Any, Optional
from graph.state import ConversationState
from graph.graph_builder import get_compiled_flow


async def stream_graph(
    state: ConversationState,
    include_node_info: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the graph execution, yielding state updates after each node completes.
    
    This enables real-time feedback as the conversation flows through different nodes:
    - conv_id_handler
    - store_message, intent_detection, situation_severity_detection (parallel)
    - response_generator
    - store_bot_response
    - render
    
    Args:
        state: Initial conversation state
        include_node_info: If True, include node execution info in stream
    
    Yields:
        Dict with node_name and updated state fields
    """
    try:
        flow = get_compiled_flow()
        
        # Stream with node names via astream_log (requires config)
        async for event in flow.astream_log(state.model_dump(), include_keys=["response_draft", "intent", "situation", "severity"]):
            # event structure: {'type': 'task_start', 'name': '...', ...} or {'type': 'task_end', 'data': {...}}
            if event.get("type") == "task_end":
                # Yield node completion event
                node_name = event.get("name", "unknown")
                
                if include_node_info:
                    yield {
                        "type": "node_end",
                        "node": node_name,
                        "timestamp": event.get("timestamp"),
                    }
                
                # Emit intermediate state updates for key fields
                data = event.get("data", {})
                if "output" in data:
                    output = data["output"]
                    
                    # Yield updates from specific nodes
                    if node_name == "render":
                        # Final API response from render node
                        if isinstance(output, dict) and "api_response" in output:
                            yield {
                                "type": "final_response",
                                "data": output["api_response"],
                            }
                    
                    elif node_name == "response_generator":
                        # Partial response from generator
                        if isinstance(output, dict) and "response_draft" in output:
                            yield {
                                "type": "response_chunk",
                                "response_draft": output["response_draft"],
                            }
                    
                    elif node_name in ["intent_detection", "situation_severity_detection"]:
                        # Analysis results
                        if isinstance(output, dict):
                            yield {
                                "type": "analysis_update",
                                "node": node_name,
                                "data": output,
                            }
        
    except Exception as e:
        yield {
            "type": "error",
            "error": str(e),
        }


async def stream_graph_responses(
    state: ConversationState
) -> AsyncGenerator[str, None]:
    """
    Stream the graph and yield only the final response text word-by-word.
    
    This is the simplest streaming interface - just returns response words
    as the graph completes.
    
    Args:
        state: Initial conversation state
    
    Yields:
        Response words as they become available
    """
    final_response = ""
    
    async for event in stream_graph(state, include_node_info=False):
        if event.get("type") == "final_response":
            # Extract response from final API response
            api_response = event.get("data", {})
            final_response = api_response.get("response", "")
            
            # Yield each word
            for word in final_response.split():
                yield f"{word} "
        
        elif event.get("type") == "error":
            yield f"[Error: {event.get('error')}]"


async def stream_graph_with_metadata(
    state: ConversationState
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the graph yielding both response and metadata updates.
    
    Emits JSON chunks suitable for SSE (Server-Sent Events).
    
    Args:
        state: Initial conversation state
    
    Yields:
        Dict with type, content, and metadata
    """
    response_buffer = ""
    
    async for event in stream_graph(state, include_node_info=True):
        event_type = event.get("type")
        
        if event_type == "node_end":
            # Emit node completion
            yield {
                "type": "node_complete",
                "node": event.get("node"),
            }
        
        elif event_type == "response_chunk":
            # Accumulate response
            response_buffer += event.get("response_draft", "")
            yield {
                "type": "response_update",
                "content": response_buffer,
            }
        
        elif event_type == "analysis_update":
            # Emit analysis results
            yield {
                "type": "analysis",
                "node": event.get("node"),
                "data": event.get("data"),
            }
        
        elif event_type == "final_response":
            # Emit complete API response
            api_response = event.get("data", {})
            yield {
                "type": "complete",
                "data": api_response,
            }
        
        elif event_type == "error":
            # Emit error
            yield {
                "type": "error",
                "error": event.get("error"),
            }


# ============================================================================
# Convenience Functions
# ============================================================================

async def stream_response_words(
    state: ConversationState
) -> AsyncGenerator[str, None]:
    """
    Simple word-by-word streaming of the response.
    Perfect for UI consumption where each word is displayed as it arrives.
    
    Args:
        state: Initial conversation state
    
    Yields:
        Individual words with spaces
    """
    async for word in stream_graph_responses(state):
        yield word


async def stream_as_sse(
    state: ConversationState
) -> AsyncGenerator[str, None]:
    """
    Stream formatted as Server-Sent Events (SSE).
    Each event is JSON-encoded and prefixed with "data: ".
    
    Args:
        state: Initial conversation state
    
    Yields:
        SSE-formatted strings
    """
    import json
    
    async for event in stream_graph_with_metadata(state):
        yield f"data: {json.dumps(event)}\n\n"
