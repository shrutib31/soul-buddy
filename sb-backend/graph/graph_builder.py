from langgraph.graph import StateGraph, END
from graph.state import ConversationState

from graph.nodes.function_nodes.conv_id_handler import conv_id_handler_node
from graph.nodes.function_nodes.store_message import store_message_node
from graph.nodes.function_nodes.render import render_node
from graph.nodes.function_nodes.store_bot_response import store_bot_response_node
from graph.nodes.agentic_nodes.intent_detection import intent_detection_node
from graph.nodes.agentic_nodes.situation_severity_detection import situation_severity_detection_node
from graph.nodes.agentic_nodes.response_generator import response_generator_node
from graph.nodes.agentic_nodes.guardrail import guardrail_node, guardrail_router


def get_compiled_flow():
    graph = StateGraph(ConversationState)

    # nodes
    graph.add_node("conv_id_handler", conv_id_handler_node)
    
    # Parallel execution nodes
    # graph.add_node("store_message", store_message_node)
    graph.add_node("intent_detection", intent_detection_node)
    # graph.add_node("situation_severity_detection", situation_severity_detection_node)
    
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("store_bot_response", store_bot_response_node)
    graph.add_node("render", render_node)

    graph.add_node("guardrail", guardrail_node)

    # edges
    graph.set_entry_point("conv_id_handler")
    
    # After conv_id_handler, run store_message, intent_detection, and situation/severity detection in parallel
    # graph.add_edge("conv_id_handler", "store_message")
    graph.add_edge("conv_id_handler", "intent_detection")
    # graph.add_edge("conv_id_handler", "situation_severity_detection")

    # Parallel nodes converge to response_generator
    # graph.add_edge("store_message", "response_generator")
    graph.add_edge("intent_detection", "response_generator")
    # graph.add_edge("situation_severity_detection", "response_generator")
    
    #Response generator to Guardrail check
    graph.add_edge("response_generator", "guardrail")

    #Links GUARDRAIL back to starting node OR continue to "store_bot_response"
    graph.add_conditional_edges("guardrail", guardrail_router)

    # Response generator → store bot response → render → end
    graph.add_edge("store_bot_response", "render")
    graph.add_edge("render", END)

    return graph.compile()
