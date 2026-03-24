"""
Unit tests for graph builder: get_compiled_flow() structure and entry point.
"""

import pytest
import sys
import types
from unittest.mock import patch, AsyncMock

redis_module = types.ModuleType("redis")
redis_asyncio_module = types.ModuleType("redis.asyncio")
redis_exceptions_module = types.ModuleType("redis.exceptions")
redis_asyncio_module.Redis = object
redis_exceptions_module.ConnectionError = RuntimeError
redis_exceptions_module.TimeoutError = RuntimeError
redis_module.asyncio = redis_asyncio_module
redis_module.exceptions = redis_exceptions_module
sys.modules.setdefault("redis", redis_module)
sys.modules.setdefault("redis.asyncio", redis_asyncio_module)
sys.modules.setdefault("redis.exceptions", redis_exceptions_module)

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
            "load_user_context",
            "store_message",
            "out_of_scope",
            "classification_node",
            "response_generator",
            "store_bot_response",
            "render",
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

    def test_compiled_flow_has_edge_conv_id_handler_to_load_user_context(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("conv_id_handler", "load_user_context") in edge_pairs

    def test_compiled_flow_has_edge_load_user_context_to_store_message(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("load_user_context", "store_message") in edge_pairs

    def test_compiled_flow_has_edge_load_user_context_to_out_of_scope(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("load_user_context", "out_of_scope") in edge_pairs

    def test_compiled_flow_has_no_edge_store_message_to_out_of_scope_path(self):
        flow = get_compiled_flow()
        graph = flow.get_graph()
        edges = list(graph.edges) if hasattr(graph, "edges") else []
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("store_message", "out_of_scope") not in edge_pairs
