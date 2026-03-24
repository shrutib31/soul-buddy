import pytest

from graph.state import ConversationState
from graph.nodes.function_nodes.fast_out_of_scope import (
    out_of_scope_node,
    out_of_scope_router,
)


@pytest.fixture
def sample_state():
    return ConversationState(
        conversation_id="test-conv-123",
        mode="incognito",
        domain="general",
        user_message="Hello there",
        chat_preference="general",
    )


class TestFastOutOfScopeNode:
    @pytest.mark.asyncio
    async def test_in_scope_message_returns_no_updates(self, sample_state):
        result = await out_of_scope_node(sample_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_nonsense_message_returns_render_ready_updates(self, sample_state):
        result = await out_of_scope_node(
            sample_state.model_copy(update={"user_message": "asdfghjkl"})
        )

        assert result["intent"] == "out_of_scope"
        assert result["out_of_scope_reason"] == "nonsense"
        assert result["is_out_of_scope"] is True
        assert "SoulGym" in result["response_draft"]

    @pytest.mark.asyncio
    async def test_psychology_definition_query_does_not_fast_path(self, sample_state):
        result = await out_of_scope_node(
            sample_state.model_copy(update={"user_message": "definition of mindfulness"})
        )

        assert result == {}


class TestFastOutOfScopeRouter:
    def test_out_of_scope_routes_to_render(self, sample_state):
        routed = out_of_scope_router(
            sample_state.model_copy(update={"is_out_of_scope": True})
        )
        assert routed == "render"

    def test_in_scope_routes_to_classification(self, sample_state):
        assert out_of_scope_router(sample_state) == "classification_node"
