"""
Unit tests for stream_as_sse (graph/streaming.py).

The compiled graph is fully mocked — no DB, LLM, or Redis required.
Each test drives a specific node output and asserts the correct SSE event type
and payload are emitted.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from graph.state import ConversationState
from graph.streaming import stream_as_sse


# ============================================================================
# HELPERS
# ============================================================================

def parse_sse(raw: str) -> dict:
    """Parse a single SSE line ('data: {...}\\n\\n') into a dict."""
    line = raw.strip()
    assert line.startswith("data: "), f"Unexpected SSE format: {line!r}"
    return json.loads(line[len("data: "):])


async def collect_events(state: ConversationState, mock_flow) -> list[dict]:
    """Run stream_as_sse with a mocked flow and return parsed events."""
    with patch("graph.streaming.get_compiled_flow", return_value=mock_flow):
        events = []
        async for raw in stream_as_sse(state):
            events.append(parse_sse(raw))
    return events


def make_mock_flow(chunks: list[dict]) -> MagicMock:
    """Build a mock flow whose astream() yields the given chunks."""
    async def _astream(_state_dict):
        for chunk in chunks:
            yield chunk

    mock = MagicMock()
    mock.astream = _astream
    return mock


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_state():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="I need some help.",
    )


# ============================================================================
# TESTS
# ============================================================================

class TestStreamAsSse:

    @pytest.mark.asyncio
    async def test_non_output_nodes_emit_node_complete(self, sample_state):
        """Internal nodes (conv_id_handler, store_message, etc.) emit node_complete."""
        mock_flow = make_mock_flow([
            {"conv_id_handler": {"conversation_id": "abc-123"}},
            {"store_message": {}},
        ])
        events = await collect_events(sample_state, mock_flow)

        types = [e["type"] for e in events]
        assert types.count("node_complete") == 2
        assert "done" in types

        node_names = [e["node"] for e in events if e["type"] == "node_complete"]
        assert "conv_id_handler" in node_names
        assert "store_message" in node_names

    @pytest.mark.asyncio
    async def test_response_generator_with_draft_emits_response_chunk(self, sample_state):
        """response_generator node with a response_draft emits a response_chunk event."""
        mock_flow = make_mock_flow([
            {"response_generator": {"response_draft": "Here is some support."}},
        ])
        events = await collect_events(sample_state, mock_flow)

        chunk_events = [e for e in events if e["type"] == "response_chunk"]
        assert len(chunk_events) == 1
        assert chunk_events[0]["content"] == "Here is some support."

    @pytest.mark.asyncio
    async def test_response_generator_without_draft_emits_node_complete(self, sample_state):
        """response_generator node with no draft falls back to node_complete."""
        mock_flow = make_mock_flow([
            {"response_generator": {}},
        ])
        events = await collect_events(sample_state, mock_flow)

        node_complete = [e for e in events if e["type"] == "node_complete"]
        assert any(e["node"] == "response_generator" for e in node_complete)

    @pytest.mark.asyncio
    async def test_render_with_api_response_emits_complete(self, sample_state):
        """render node with api_response emits a complete event with the full payload."""
        api_response = {
            "success": True,
            "conversation_id": "test-conv-123",
            "response": "Hello! How can I help?",
            "metadata": {"intent": "greeting"},
        }
        mock_flow = make_mock_flow([
            {"render": {"api_response": api_response}},
        ])
        events = await collect_events(sample_state, mock_flow)

        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["data"] == api_response

    @pytest.mark.asyncio
    async def test_render_without_api_response_emits_node_complete(self, sample_state):
        """render node with no api_response falls back to node_complete."""
        mock_flow = make_mock_flow([
            {"render": {}},
        ])
        events = await collect_events(sample_state, mock_flow)

        node_complete = [e for e in events if e["type"] == "node_complete"]
        assert any(e["node"] == "render" for e in node_complete)

    @pytest.mark.asyncio
    async def test_always_ends_with_done_event(self, sample_state):
        """The last event must always be {type: done}."""
        mock_flow = make_mock_flow([
            {"conv_id_handler": {}},
            {"render": {"api_response": {"success": True}}},
        ])
        events = await collect_events(sample_state, mock_flow)

        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_exception_in_graph_emits_error_event(self, sample_state):
        """If get_compiled_flow() raises, stream_as_sse yields an error event."""
        with patch("graph.streaming.get_compiled_flow", side_effect=RuntimeError("Graph exploded")):
            events = []
            async for raw in stream_as_sse(sample_state):
                events.append(parse_sse(raw))

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "Graph exploded" in events[0]["error"]

    @pytest.mark.asyncio
    async def test_non_dict_node_output_is_skipped(self, sample_state):
        """Node outputs that aren't dicts must be silently skipped."""
        mock_flow = make_mock_flow([
            {"some_node": "plain string output"},
            {"render": {"api_response": {"success": True}}},
        ])
        events = await collect_events(sample_state, mock_flow)

        # "some_node" must not produce any event; render should still fire
        node_names = [e.get("node") for e in events if e["type"] == "node_complete"]
        assert "some_node" not in node_names
        assert any(e["type"] == "complete" for e in events)

    @pytest.mark.asyncio
    async def test_full_graph_sequence_emits_all_expected_events(self, sample_state):
        """End-to-end sequence with all major nodes produces events in order."""
        api_response = {"success": True, "response": "I'm here for you."}
        mock_flow = make_mock_flow([
            {"conv_id_handler": {"conversation_id": "abc"}},
            {"store_message": {}},
            {"classification_node": {"intent": "venting"}},
            {"response_generator": {"response_draft": "I'm here for you."}},
            {"guardrail": {"guardrail_status": "OK"}},
            {"store_bot_response": {}},
            {"render": {"api_response": api_response}},
        ])
        events = await collect_events(sample_state, mock_flow)

        types = [e["type"] for e in events]
        assert "node_complete" in types
        assert "response_chunk" in types
        assert "complete" in types
        assert types[-1] == "done"
