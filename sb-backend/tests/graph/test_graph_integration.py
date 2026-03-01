"""
Integration-style test: full graph execution with mocked DB and LLM.

Marks: @pytest.mark.integration
Run with: pytest tests/ -v -m integration

This test invokes the compiled flow end-to-end with all external I/O mocked,
so no real database or Ollama/OpenAI is required.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

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
            )
            result = await flow.ainvoke(state.model_dump())

    assert result is not None
    assert "api_response" in result
    api_resp = result["api_response"]
    assert api_resp.get("success") is True
    assert "response" in api_resp
    assert len(api_resp["response"]) > 0
    assert "conversation_id" in api_resp
