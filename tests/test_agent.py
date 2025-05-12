"""
Tests for the agent module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import MessagesPlaceholder

from elastic_cloud_agent.agent import create_agent, create_agent_prompt, create_llm


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-api-key",
            "ELASTIC_CLOUD_BASE_URL": "https://test.es.example.com",
            "ELASTIC_CLOUD_API_KEY": "test-elastic-key",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
            "AZURE_OPENAI_API_VERSION": "test-version",
            "AZURE_OPENAI_ENDPOINT": "https://test.azure.example.com",
        },
    ):
        yield


def test_create_agent_prompt():
    """Test that the agent prompt is created correctly."""
    prompt = create_agent_prompt()
    assert prompt is not None
    assert isinstance(prompt.messages, list)
    assert len(prompt.messages) == 4  # system, chat_history, human input, agent_scratchpad

    # Check for chat_history placeholder specifically
    placeholder_variables = [
        msg.variable_name for msg in prompt.messages if isinstance(msg, MessagesPlaceholder)
    ]
    assert "chat_history" in placeholder_variables


@patch("elastic_cloud_agent.agent.AzureChatOpenAI")
def test_create_llm(mock_chat_openai, mock_env_vars):
    """Test that the language model is created correctly."""
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    llm = create_llm()

    mock_chat_openai.assert_called_once()
    assert llm == mock_instance


@patch("elastic_cloud_agent.agent.create_llm")
@patch("elastic_cloud_agent.agent.create_search_tool")
@patch("elastic_cloud_agent.agent.create_openapi_toolkit")
@patch("elastic_cloud_agent.agent.create_openai_functions_agent")
@patch("elastic_cloud_agent.agent.AgentExecutor")
def test_create_agent(
    mock_agent_executor,
    mock_create_openai_functions_agent,
    mock_create_openapi_toolkit,
    mock_create_search_tool,
    mock_create_llm,
    mock_env_vars,
):
    """Test that the agent is created correctly."""
    # Mock the return values
    mock_llm = MagicMock(spec=BaseLanguageModel)
    mock_create_llm.return_value = mock_llm

    mock_search_tool = MagicMock()
    mock_create_search_tool.return_value = mock_search_tool

    mock_openapi_tools = [MagicMock(), MagicMock()]
    mock_create_openapi_toolkit.return_value = mock_openapi_tools

    mock_agent = MagicMock()
    mock_create_openai_functions_agent.return_value = mock_agent

    mock_executor = MagicMock()
    mock_agent_executor.return_value = mock_executor

    # Call the function being tested
    result = create_agent()

    # Verify the correct calls were made
    mock_create_llm.assert_called_once()
    mock_create_search_tool.assert_called_once()
    mock_create_openapi_toolkit.assert_called_once_with(llm=mock_llm)

    # Check that the tools are combined correctly
    __tools = [mock_search_tool] + mock_openapi_tools
    mock_create_openai_functions_agent.assert_called_once()

    # Check that the agent executor is created correctly
    mock_agent_executor.assert_called_once()

    # Check that the result is the mocked executor
    assert result == mock_executor


@patch("elastic_cloud_agent.agent.create_llm")
def test_create_agent_with_custom_llm(mock_create_llm, mock_env_vars):
    """Test that the agent is created with a custom LLM if provided."""
    custom_llm = MagicMock(spec=BaseLanguageModel)

    with (
        patch("elastic_cloud_agent.agent.create_search_tool"),
        patch("elastic_cloud_agent.agent.create_openapi_toolkit"),
        patch("elastic_cloud_agent.agent.create_openai_functions_agent"),
        patch("elastic_cloud_agent.agent.AgentExecutor"),
    ):
        create_agent(llm=custom_llm)

        # Verify that create_llm was not called since we provided a custom LLM
        mock_create_llm.assert_not_called()


def test_agent_prompt_has_chat_history_placeholder():
    """Test that the agent prompt includes a placeholder for chat history."""
    prompt = create_agent_prompt()

    # Find all MessagesPlaceholder instances and their variable names
    placeholders = [msg for msg in prompt.messages if isinstance(msg, MessagesPlaceholder)]
    placeholder_variables = [p.variable_name for p in placeholders]

    # Check that chat_history exists as a placeholder variable
    assert "chat_history" in placeholder_variables
