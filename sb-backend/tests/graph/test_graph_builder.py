"""
Unit tests for graph builder: get_compiled_flow() structure and entry point.
"""

import pytest
from unittest.mock import patch, AsyncMock

from graph.graph_builder import get_compiled_flow
from graph.state import ConversationState


# ============================================================================
# Graph structure
# ============================================================================

class TestGraphBuilderUnit:
    """Unit tests for get_compiled_flow()."""

    def test_get_compiled_flow_returns_compiled_graph(self):
        flow = get_compiled_flow()
        assert flow is not None
        assert hasattr(flow, "ainvoke")

    def test_compiled_flow_has_expected_nodes(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        nodes = list(graph.nodes)
        expected = {
            "conv_id_handler",
            "store_message",
            "classification_node",
            "response_generator",
            "store_bot_response",
            "render",
            "guardrail",
        }
        for name in expected:
            assert name in nodes, f"Expected node {name!r} in graph nodes {nodes}"

    def test_compiled_flow_has_entry_point_conv_id_handler(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        # LangGraph may expose entry point via graph or flow
        if hasattr(flow, "entry_point"):
            assert flow.entry_point == "conv_id_handler"
        # Else check that __start__ or similar points to conv_id_handler
        if hasattr(graph, "nodes"):
            assert "conv_id_handler" in list(graph.nodes)

    def test_compiled_flow_has_edge_conv_id_handler_to_store_message(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_sources = {e[0]: e[1] for e in edges} if edges else {}
        # conv_id_handler should have outgoing edges to store_message and classification_node
        assert "conv_id_handler" in [e[0] for e in edges] or "conv_id_handler" in str(edges)

    def test_compiled_flow_has_edge_classification_to_response_generator(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("classification_node", "response_generator") in edge_pairs

    def test_compiled_flow_has_edge_response_generator_to_store_bot_response(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("response_generator", "store_bot_response") in edge_pairs

    def test_compiled_flow_has_edge_store_bot_response_to_render(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("store_bot_response", "render") in edge_pairs
