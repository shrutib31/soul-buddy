from langgraph.graph import StateGraph, END
from graph.state import ConversationState

from graph.nodes.function_nodes.out_of_scope import (
    out_of_scope_node,
    out_of_scope_router,
)
from graph.nodes.function_nodes.conv_id_handler import conv_id_handler_node
from graph.nodes.function_nodes.load_user_context import load_user_context_node
from graph.nodes.function_nodes.store_message import store_message_node
from graph.nodes.function_nodes.render import render_node
from graph.nodes.function_nodes.store_bot_response import store_bot_response_node
from graph.nodes.agentic_nodes.response_generator import response_generator_node
from graph.nodes.agentic_nodes.classification_node import classification_node


def get_compiled_flow():
    graph = StateGraph(ConversationState)

    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("conv_id_handler", conv_id_handler_node)
    graph.add_node("load_user_context", load_user_context_node)
    graph.add_node("store_message", store_message_node)
    graph.add_node("classification_node", classification_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("store_bot_response", store_bot_response_node)
    graph.add_node("render", render_node)

    graph.set_entry_point("conv_id_handler")

    # Ensure a valid conversation_id exists before loading any user data
    graph.add_edge("conv_id_handler", "load_user_context")

    # Run persistence and fast out-of-scope detection in parallel after context load.
    graph.add_edge("load_user_context", "store_message")
    graph.add_edge("load_user_context", "out_of_scope")

    # Route to render for cheap out-of-scope cases; otherwise continue normally.
    graph.add_conditional_edges("out_of_scope", out_of_scope_router)

    graph.add_edge("classification_node", "response_generator")
    graph.add_edge("response_generator", "store_bot_response")
    graph.add_edge("store_bot_response", "render")
    graph.add_edge("render", END)

    return graph.compile()
