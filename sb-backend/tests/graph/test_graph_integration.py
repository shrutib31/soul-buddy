"""
Integration-style test: full graph execution with mocked DB and LLM.

Marks: @pytest.mark.integration
Run with: pytest tests/ -v -m integration

This test invokes the compiled flow end-to-end with all external I/O mocked,
so no real database or Ollama/OpenAI is required.
"""

import pytest
import sys
import types
from unittest.mock import patch, AsyncMock, MagicMock

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


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_graph_invoke_with_mocked_io(mock_session):
    """
    Invoke the compiled graph from conv_id_handler through render with all I/O mocked.
    Uses a greeting message so classification returns greeting without loading the model.
    """
    # Mock DB sessions for conv_id_handler, store_message, store_bot_response
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalar.return_value = 0
    mock_session.execute.return_value = mock_result

    with patch(
        "graph.nodes.function_nodes.conv_id_handler.data_db"
    ) as mock_conv_db, patch(
        "graph.nodes.function_nodes.store_message.data_db"
    ) as mock_store_db, patch(
        "graph.nodes.function_nodes.store_bot_response.data_db"
    ) as mock_bot_db:
        mock_conv_db.get_session.return_value = mock_session
        mock_store_db.get_session.return_value = mock_session
        mock_bot_db.get_session.return_value = mock_session

        with patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_ollama",
            new_callable=AsyncMock,
            return_value="I'm here for you. How can I support you today?",
        ), patch(
            "graph.nodes.agentic_nodes.response_generator.generate_response_gpt",
            new_callable=AsyncMock,
            return_value="I'm here for you. How can I support you today?",
        ):
            flow = get_compiled_flow()
            state = ConversationState(
                conversation_id="",
                mode="incognito",
                domain="general",
                user_message="Hi!",  # Greeting so no model load in classification
                chat_preference="general",
            )
            result = await flow.ainvoke(state.model_dump())

    assert result is not None
    assert "api_response" in result
    api_resp = result["api_response"]
    assert api_resp.get("success") is True
    assert "response" in api_resp
    assert len(api_resp["response"]) > 0
    assert "conversation_id" in api_resp


@pytest.mark.asyncio
async def test_fast_out_of_scope_happens_before_classification():
    async def fake_conv_id_handler(_state):
        return {"conversation_id": "test-conv-123"}

    async def fake_load_user_context(_state):
        return {}

    async def fake_store_message(_state):
        return {}

    def unexpected_classification(_state):
        raise AssertionError("classification_node should not run")

    with patch(
        "graph.graph_builder.conv_id_handler_node",
        new=fake_conv_id_handler,
    ), patch(
        "graph.graph_builder.load_user_context_node",
        new=fake_load_user_context,
    ), patch(
        "graph.graph_builder.store_message_node",
        new=fake_store_message,
    ), patch(
        "graph.graph_builder.classification_node",
        new=unexpected_classification,
    ):
        flow = get_compiled_flow()
        state = ConversationState(
            conversation_id="test-conv-123",
            mode="incognito",
            domain="general",
            user_message="asdfghjkl",
            chat_preference="general",
        )
        result = await flow.ainvoke(state.model_dump())

    assert result["api_response"]["success"] is True
    assert result["api_response"]["metadata"]["intent"] == "out_of_scope"
    assert result["api_response"]["metadata"]["out_of_scope_reason"] == "nonsense"
