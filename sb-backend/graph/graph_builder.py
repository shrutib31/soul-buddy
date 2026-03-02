from langgraph.graph import StateGraph, END
from graph.state import ConversationState

from graph.nodes.function_nodes.conv_id_handler import conv_id_handler_node
from graph.nodes.function_nodes.store_message import store_message_node
from graph.nodes.function_nodes.render import render_node
from graph.nodes.function_nodes.store_bot_response import store_bot_response_node
from graph.nodes.agentic_nodes.response_generator import response_generator_node
from graph.nodes.agentic_nodes.classification_node import classification_node
from graph.nodes.function_nodes.privacy_masking import new_masking_node as privacy_masking_node

def get_compiled_flow():
    graph = StateGraph(ConversationState)

    graph.add_node("conv_id_handler", conv_id_handler_node)
    graph.add_node("store_message", store_message_node)
    graph.add_node("classification_node", classification_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("store_bot_response", store_bot_response_node)
    graph.add_node("render", render_node)

    graph.set_entry_point("conv_id_handler")

    # store_message and classification_node run in parallel after conv_id_handler
    graph.add_edge("conv_id_handler", "store_message")
    graph.add_edge("conv_id_handler", "classification_node")

    graph.add_edge("classification_node", "response_generator")
    graph.add_edge("response_generator", "store_bot_response")
    graph.add_edge("store_bot_response", "render")
    graph.add_edge("render", END)

    return graph.compile()
