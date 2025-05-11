"""
Tests for the chat memory functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from elastic_cloud_agent.main import chat_loop


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-api-key",
            "ELASTIC_CLOUD_BASE_URL": "https://test.es.example.com",
            "ELASTIC_CLOUD_API_KEY": "test-elastic-key",
        },
    ):
        yield


@patch("elastic_cloud_agent.main.create_agent")
@patch("builtins.input")
@patch("builtins.print")
def test_chat_loop_maintains_history(mock_print, mock_input, mock_create_agent, mock_env_vars):
    """Test that the chat loop maintains conversation history across turns."""
    # Mock the agent's invoke method to return a simple response
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = lambda x: {"output": f"Response to: {x['input']}"}
    mock_create_agent.return_value = mock_agent

    # Set up input sequence: two messages followed by exit
    mock_input.side_effect = ["First message", "Second message", "exit"]

    # Run the chat loop
    chat_loop()

    # Check that history was maintained by examining the calls to invoke
    assert mock_agent.invoke.call_count == 2

    # First call should have empty history
    first_call_args = mock_agent.invoke.call_args_list[0][0][0]
    assert first_call_args["input"] == "First message"
    assert first_call_args["chat_history"] == []

    # Second call should include history from first interaction
    second_call_args = mock_agent.invoke.call_args_list[1][0][0]
    assert second_call_args["input"] == "Second message"
    assert len(second_call_args["chat_history"]) == 2  # User message + AI response

    # Check message content
    assert isinstance(second_call_args["chat_history"][0], HumanMessage)
    assert second_call_args["chat_history"][0].content == "First message"
    assert isinstance(second_call_args["chat_history"][1], AIMessage)


@patch("elastic_cloud_agent.main.create_agent")
@patch("builtins.input")
@patch("builtins.print")
def test_chat_loop_clears_history(mock_print, mock_input, mock_create_agent, mock_env_vars):
    """Test that the 'clear' command resets the conversation history."""
    # Mock the agent's invoke method to return a simple response
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = lambda x: {"output": f"Response to: {x['input']}"}
    mock_create_agent.return_value = mock_agent

    # Set up input sequence: message, clear command, another message, then exit
    mock_input.side_effect = ["First message", "clear", "New message after clear", "exit"]

    # Run the chat loop
    chat_loop()

    # We should have two calls to invoke (for the two messages, not counting the clear command)
    assert mock_agent.invoke.call_count == 2

    # The second real message should have empty history (after clear)
    third_call_args = mock_agent.invoke.call_args_list[1][0][0]
    assert third_call_args["input"] == "New message after clear"
    assert third_call_args["chat_history"] == []


@patch("elastic_cloud_agent.main.create_agent")
@patch("builtins.input")
@patch("builtins.print")
def test_chat_loop_handles_errors(mock_print, mock_input, mock_create_agent, mock_env_vars):
    """Test that the chat loop properly handles errors during invocation."""
    # Mock the agent to raise an exception
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = Exception("Test error")
    mock_create_agent.return_value = mock_agent

    # Set up input sequence: one message that will cause an error, then exit
    mock_input.side_effect = ["Message causing error", "exit"]

    # Run the chat loop
    chat_loop()

    # Verify the agent was called
    mock_agent.invoke.assert_called_once()

    # Check that an error message was printed
    error_messages = [
        call_args[0][0]
        for call_args in mock_print.call_args_list
        if isinstance(call_args[0][0], str) and "Error:" in call_args[0][0]
    ]
    assert any(error_messages)
