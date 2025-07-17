"""
Tests for the SmartJsonExplorerOpenAPIToolkit.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel

from elastic_cloud_agent.tools.smart_openapi_toolkit import (
    SmartJsonExplorerOpenAPIToolkit,
    SmartJsonExplorerTool,
    create_smart_openapi_toolkit,
)


@pytest.fixture
def mock_llm():
    """Create a mock language model for testing."""
    return Mock(spec=BaseLanguageModel)


@pytest.fixture
def sample_api_spec():
    """Create a sample OpenAPI specification for testing."""
    return {
        "swagger": "2.0",
        "info": {"version": "1", "title": "Test API"},
        "host": "api.example.com",
        "basePath": "/api/v1",
        "tags": [
            {"name": "Accounts"},
            {"name": "Deployments"},
            {"name": "BillingCostsAnalysis"},
        ],
        "paths": {
            "/account": {
                "get": {
                    "tags": ["Accounts"],
                    "summary": "Get account info",
                    "operationId": "get-account",
                }
            },
            "/deployments": {
                "get": {
                    "tags": ["Deployments"],
                    "summary": "List deployments",
                    "operationId": "list-deployments",
                },
                "post": {
                    "tags": ["Deployments"],
                    "summary": "Create deployment",
                    "operationId": "create-deployment",
                },
            },
            "/billing/costs": {
                "get": {
                    "tags": ["BillingCostsAnalysis"],
                    "summary": "Get billing costs",
                    "operationId": "get-billing-costs",
                }
            },
        },
    }


@pytest.fixture
def mock_requests_wrapper():
    """Create a mock requests wrapper."""
    from langchain_community.utilities.requests import TextRequestsWrapper

    # Create a real wrapper with mock headers to avoid Pydantic validation issues
    return TextRequestsWrapper(headers={})


class TestSmartJsonExplorerTool:
    """Test cases for SmartJsonExplorerTool."""

    def test_tool_initialisation(self, mock_llm, sample_api_spec):
        """Test that the tool initialises correctly."""
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)

        assert tool.name == "json_explorer"
        assert "openapi spec" in tool.description.lower()
        assert tool.llm == mock_llm
        assert tool.api_spec == sample_api_spec
        assert len(tool.available_tags) == 3
        assert "Accounts" in tool.available_tags
        assert "Deployments" in tool.available_tags
        assert "BillingCostsAnalysis" in tool.available_tags

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_json_agent")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.classify_intent")
    def test_run_with_intent_classification(
        self, mock_classify_intent, mock_create_json_agent, mock_llm, sample_api_spec
    ):
        """Test the _run method with intent classification."""
        # Setup mocks
        mock_classify_intent.return_value = ["Deployments"]

        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Mock agent response"}
        mock_create_json_agent.return_value = mock_agent

        # Create tool and run
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)
        result = tool._run("How do I list deployments?")

        # Verify
        mock_classify_intent.assert_called_once()
        mock_create_json_agent.assert_called_once()
        assert result == "Mock agent response"

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_json_agent")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.classify_intent")
    def test_run_handles_exceptions(
        self, mock_classify_intent, mock_create_json_agent, mock_llm, sample_api_spec
    ):
        """Test that _run handles exceptions gracefully."""
        # Make intent classification fail
        mock_classify_intent.side_effect = Exception("Classification error")

        # Create tool and run
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)
        result = tool._run("Some query")

        # Should return error message
        assert "Error in smart JSON exploration" in result
        assert "Classification error" in result

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_json_agent")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.classify_intent")
    def test_agent_caching_cache_hit(
        self, mock_classify_intent, mock_create_json_agent, mock_llm, sample_api_spec
    ):
        """Test that agents are cached and reused for the same tag set."""
        # Setup mocks
        mock_classify_intent.return_value = ["Deployments"]

        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Mock agent response"}
        mock_create_json_agent.return_value = mock_agent

        # Create tool
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)

        # First call - should create and cache agent
        result1 = tool._run("How do I list deployments?")
        
        # Second call with same tags - should use cached agent
        result2 = tool._run("How do I create deployments?")

        # Verify
        assert mock_classify_intent.call_count == 2
        mock_create_json_agent.assert_called_once()  # Should only create agent once
        assert result1 == "Mock agent response"
        assert result2 == "Mock agent response"

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_json_agent")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.classify_intent")
    def test_agent_caching_cache_miss(
        self, mock_classify_intent, mock_create_json_agent, mock_llm, sample_api_spec
    ):
        """Test that different tag sets create different cached agents."""
        # Setup mocks for different tag sets
        mock_classify_intent.side_effect = [["Deployments"], ["Accounts"]]

        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Mock agent response"}
        mock_create_json_agent.return_value = mock_agent

        # Create tool
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)

        # First call with Deployments tags
        result1 = tool._run("How do I list deployments?")
        
        # Second call with different Accounts tags - should create new agent
        result2 = tool._run("How do I manage accounts?")

        # Verify
        assert mock_classify_intent.call_count == 2
        assert mock_create_json_agent.call_count == 2  # Should create two different agents
        assert result1 == "Mock agent response"
        assert result2 == "Mock agent response"

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_json_agent")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.classify_intent")
    def test_agent_caching_frozenset_ordering(
        self, mock_classify_intent, mock_create_json_agent, mock_llm, sample_api_spec
    ):
        """Test that different tag orderings result in cache hits."""
        # Setup mocks with different tag orderings
        mock_classify_intent.side_effect = [["Deployments", "Accounts"], ["Accounts", "Deployments"]]

        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Mock agent response"}
        mock_create_json_agent.return_value = mock_agent

        # Create tool
        tool = SmartJsonExplorerTool(llm=mock_llm, api_spec=sample_api_spec)

        # First call with tags in one order
        result1 = tool._run("Query 1")
        
        # Second call with same tags in different order - should use cached agent
        result2 = tool._run("Query 2")

        # Verify
        assert mock_classify_intent.call_count == 2
        mock_create_json_agent.assert_called_once()  # Should only create agent once due to frozenset
        assert result1 == "Mock agent response"
        assert result2 == "Mock agent response"


class TestSmartJsonExplorerOpenAPIToolkit:
    """Test cases for SmartJsonExplorerOpenAPIToolkit."""

    def test_toolkit_initialisation(self, mock_llm, sample_api_spec, mock_requests_wrapper):
        """Test that the toolkit initialises correctly."""
        toolkit = SmartJsonExplorerOpenAPIToolkit(
            llm=mock_llm,
            api_spec=sample_api_spec,
            requests_wrapper=mock_requests_wrapper,
            allow_dangerous_requests=True,
        )

        assert toolkit.llm == mock_llm
        assert toolkit.api_spec == sample_api_spec
        assert toolkit.requests_wrapper == mock_requests_wrapper
        assert toolkit.allow_dangerous_requests is True

    def test_get_tools_returns_correct_tools(
        self, mock_llm, sample_api_spec, mock_requests_wrapper
    ):
        """Test that get_tools returns the expected tools."""
        toolkit = SmartJsonExplorerOpenAPIToolkit(
            llm=mock_llm,
            api_spec=sample_api_spec,
            requests_wrapper=mock_requests_wrapper,
            allow_dangerous_requests=True,
        )

        tools = toolkit.get_tools()

        # Should return 6 tools: 5 HTTP tools + 1 smart JSON explorer
        assert len(tools) == 6

        tool_names = [tool.name for tool in tools]
        assert "requests_get" in tool_names
        assert "requests_post" in tool_names
        assert "requests_patch" in tool_names
        assert "requests_put" in tool_names
        assert "requests_delete" in tool_names
        assert "json_explorer" in tool_names

        # Find the json_explorer tool and verify it exists
        json_explorer = next(tool for tool in tools if tool.name == "json_explorer")
        assert json_explorer.name == "json_explorer"
        assert "openapi spec" in json_explorer.description.lower()

    def test_from_llm_class_method(self, mock_llm, sample_api_spec, mock_requests_wrapper):
        """Test the from_llm class method."""
        toolkit = SmartJsonExplorerOpenAPIToolkit.from_llm(
            llm=mock_llm,
            api_spec=sample_api_spec,
            requests_wrapper=mock_requests_wrapper,
            allow_dangerous_requests=True,
        )

        assert isinstance(toolkit, SmartJsonExplorerOpenAPIToolkit)
        assert toolkit.llm == mock_llm
        assert toolkit.api_spec == sample_api_spec
        assert toolkit.requests_wrapper == mock_requests_wrapper
        assert toolkit.allow_dangerous_requests is True


class TestCreateSmartOpenAPIToolkit:
    """Test cases for the create_smart_openapi_toolkit function."""

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.load_api_spec")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_requests_wrapper")
    def test_create_smart_openapi_toolkit(
        self, mock_create_requests_wrapper, mock_load_api_spec, mock_llm, sample_api_spec
    ):
        """Test the create_smart_openapi_toolkit function."""
        from langchain_community.utilities.requests import TextRequestsWrapper

        # Setup mocks
        mock_load_api_spec.return_value = sample_api_spec
        mock_create_requests_wrapper.return_value = TextRequestsWrapper(headers={})

        # Call the function
        tools = create_smart_openapi_toolkit(llm=mock_llm)

        # Verify
        mock_load_api_spec.assert_called_once_with(None)
        mock_create_requests_wrapper.assert_called_once()

        # Should return 6 tools
        assert len(tools) == 6
        tool_names = [tool.name for tool in tools]
        assert "requests_get" in tool_names
        assert "json_explorer" in tool_names

    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.load_api_spec")
    @patch("elastic_cloud_agent.tools.smart_openapi_toolkit.create_requests_wrapper")
    def test_create_smart_openapi_toolkit_with_spec_path(
        self, mock_create_requests_wrapper, mock_load_api_spec, mock_llm, sample_api_spec
    ):
        """Test the create_smart_openapi_toolkit function with custom spec path."""
        from pathlib import Path

        from langchain_community.utilities.requests import TextRequestsWrapper

        # Setup mocks
        mock_load_api_spec.return_value = sample_api_spec
        mock_create_requests_wrapper.return_value = TextRequestsWrapper(headers={})

        spec_path = Path("/custom/path/to/spec.json")

        # Call the function
        tools = create_smart_openapi_toolkit(llm=mock_llm, spec_path=spec_path)

        # Verify spec path was passed
        mock_load_api_spec.assert_called_once_with(spec_path)

        # Should still return 6 tools
        assert len(tools) == 6
