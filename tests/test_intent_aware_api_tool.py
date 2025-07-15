"""
Tests for the intent-aware API tool.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from elastic_cloud_agent.tools.intent_aware_api_tool import (
    IntentAwareApiTool,
    create_intent_aware_api_tool,
)


class TestIntentAwareApiTool:
    """Tests for the IntentAwareApiTool class."""

    @pytest.fixture
    def sample_spec(self):
        """Sample OpenAPI spec for testing."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0"},
            "paths": {
                "/deployments": {
                    "get": {
                        "operationId": "list-deployments",
                        "tags": ["Deployments"],
                        "summary": "List deployments",
                        "description": "Get a list of all deployments",
                    },
                    "post": {
                        "operationId": "create-deployment",
                        "tags": ["Deployments"],
                        "summary": "Create deployment",
                        "description": "Create a new deployment",
                    },
                },
                "/deployments/{id}": {
                    "get": {
                        "operationId": "get-deployment",
                        "tags": ["Deployments"],
                        "summary": "Get deployment",
                        "description": "Get deployment details",
                    },
                    "delete": {
                        "operationId": "delete-deployment",
                        "tags": ["Deployments"],
                        "summary": "Delete deployment",
                        "description": "Delete a deployment",
                    },
                },
                "/api-keys": {
                    "get": {
                        "operationId": "list-api-keys",
                        "tags": ["Authentication"],
                        "summary": "List API keys",
                        "description": "Get all API keys",
                    }
                },
            },
        }

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_requests_wrapper(self):
        """Create a mock requests wrapper."""
        mock_wrapper = Mock()
        mock_wrapper.get.return_value = '{"result": "success"}'
        mock_wrapper.post.return_value = '{"created": true}'
        mock_wrapper.put.return_value = '{"updated": true}'
        mock_wrapper.delete.return_value = '{"deleted": true}'
        return mock_wrapper

    @pytest.fixture
    def api_tool(self, mock_llm, sample_spec, temp_cache_dir, mock_requests_wrapper):
        """Create an IntentAwareApiTool instance for testing."""
        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.load_api_spec"
        ) as mock_load_spec:
            mock_load_spec.return_value = sample_spec

            with patch(
                "elastic_cloud_agent.tools.intent_aware_api_tool.create_requests_wrapper"
            ) as mock_create_wrapper:
                mock_create_wrapper.return_value = mock_requests_wrapper

                tool = IntentAwareApiTool(llm=mock_llm, spec_path=None, cache_dir=temp_cache_dir)
                return tool

    def test_tool_initialization(self, api_tool, mock_llm):
        """Test that the tool initializes correctly."""
        assert api_tool.name == "elastic_cloud_api"
        assert api_tool.llm == mock_llm
        assert api_tool.spec_registry is not None
        assert api_tool.cache_manager is not None

    def test_is_exploratory_query(self, api_tool):
        """Test detection of exploratory queries using LLM."""
        # Mock LLM to return "exploratory" for all queries
        mock_response = Mock()
        mock_response.content = "exploratory"
        api_tool.llm.invoke.return_value = mock_response

        exploratory_queries = [
            "What APIs are available?",
            "How do I create a deployment?",
            "Which endpoints can I use?",
            "Show me available operations",
            "Can I list deployments?",
            "Help me with the API",
        ]

        for query in exploratory_queries:
            assert api_tool._is_exploratory_query(query), f"Should detect '{query}' as exploratory"
        
        # Test empty query defaults to exploratory
        assert api_tool._is_exploratory_query(""), "Empty query should default to exploratory"
        assert api_tool._is_exploratory_query(None), "None query should default to exploratory"

    def test_is_execution_query(self, api_tool):
        """Test detection of execution queries using LLM."""
        # Test with explicit parameters (should return True immediately)
        assert api_tool._is_execution_query("Any query", {"method": "GET"})
        assert api_tool._is_execution_query("Any query", {"endpoint": "/deployments"})
        
        # Test empty query defaults to False
        assert not api_tool._is_execution_query("", {}), "Empty query should default to exploratory"
        assert not api_tool._is_execution_query(None, {}), "None query should default to exploratory"

        # Mock LLM to return "execution" for action queries
        mock_response = Mock()
        mock_response.content = "execution"
        api_tool.llm.invoke.return_value = mock_response

        # Test with action words
        execution_queries = [
            "Create a new deployment",
            "Delete the deployment",
            "Update deployment settings",
            "Start the cluster",
            "Stop the service",
            "Restart the deployment",
        ]

        for query in execution_queries:
            assert api_tool._is_execution_query(query, {}), f"Should detect '{query}' as execution"

    def test_provide_api_guidance(self, api_tool):
        """Test API guidance generation."""
        query = "How do I manage deployments?"
        intent = "management"
        optimised_spec = {
            "paths": {
                "/deployments": {
                    "get": {"summary": "List deployments"},
                    "post": {"summary": "Create deployment"},
                },
                "/deployments/{id}": {
                    "get": {"summary": "Get deployment"},
                    "delete": {"summary": "Delete deployment"},
                },
            }
        }

        guidance = api_tool._provide_api_guidance(query, intent, optimised_spec)

        assert "management" in guidance
        assert "/deployments" in guidance
        assert "List deployments" in guidance
        assert "Create deployment" in guidance
        assert "**Intent Classification:** management" in guidance
        assert "2 endpoints loaded" in guidance

    def test_provide_api_guidance_empty_spec(self, api_tool):
        """Test API guidance when no paths are available."""
        query = "Help me with unknown operations"
        intent = "unknown"
        optimised_spec = {"paths": {}}

        guidance = api_tool._provide_api_guidance(query, intent, optimised_spec)

        assert "No specific API endpoints found" in guidance
        assert "unknown" in guidance

    def test_execute_api_call_get(self, api_tool, mock_requests_wrapper):
        """Test GET API call execution."""
        query = "Get deployment list"
        intent = "basic"
        optimised_spec = {}
        kwargs = {"method": "GET", "endpoint": "/deployments"}

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            response = api_tool._execute_api_call(query, intent, optimised_spec, kwargs)

        mock_requests_wrapper.get.assert_called_once_with("https://test.elastic.com/deployments")
        assert "API Call Executed Successfully" in response
        assert "GET" in response
        assert "basic" in response

    def test_execute_api_call_post(self, api_tool, mock_requests_wrapper):
        """Test POST API call execution."""
        query = "Create a deployment"
        intent = "management"
        optimised_spec = {}
        kwargs = {"method": "POST", "endpoint": "/deployments", "data": {"name": "test-deployment"}}

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            response = api_tool._execute_api_call(query, intent, optimised_spec, kwargs)

        mock_requests_wrapper.post.assert_called_once_with(
            "https://test.elastic.com/deployments", data={"name": "test-deployment"}
        )
        assert "API Call Executed Successfully" in response
        assert "POST" in response
        assert "management" in response

    def test_execute_api_call_missing_endpoint(self, api_tool):
        """Test API call execution with missing endpoint."""
        query = "Do something"
        intent = "management"
        optimised_spec = {}
        kwargs = {"method": "GET"}

        response = api_tool._execute_api_call(query, intent, optimised_spec, kwargs)

        assert "specify the endpoint path" in response

    def test_execute_api_call_unsupported_method(self, api_tool):
        """Test API call execution with unsupported method."""
        query = "Patch something"
        intent = "management"
        optimised_spec = {}
        kwargs = {"method": "PATCH", "endpoint": "/deployments"}

        response = api_tool._execute_api_call(query, intent, optimised_spec, kwargs)

        assert "Unsupported HTTP method: PATCH" in response

    def test_format_api_response_json(self, api_tool):
        """Test formatting JSON API response."""
        method = "GET"
        url = "https://test.elastic.com/deployments"
        response = '{"deployments": [{"id": "1", "name": "test"}]}'
        intent = "basic"

        formatted = api_tool._format_api_response(method, url, response, intent)

        assert "API Call Executed Successfully" in formatted
        assert "GET" in formatted
        assert "basic" in formatted
        assert "https://test.elastic.com/deployments" in formatted
        assert '"deployments"' in formatted

    def test_format_api_response_non_json(self, api_tool):
        """Test formatting non-JSON API response."""
        method = "GET"
        url = "https://test.elastic.com/deployments"
        response = "Plain text response"
        intent = "basic"

        formatted = api_tool._format_api_response(method, url, response, intent)

        assert "API Call Executed Successfully" in formatted
        assert "Plain text response" in formatted

    def test_run_exploratory_query(self, api_tool):
        """Test running an exploratory query."""
        query = "What APIs are available for deployments?"

        # Mock the intent analysis
        api_tool.spec_registry.analyse_query_intent = Mock(return_value={"intent": "management"})
        api_tool.cache_manager.get_spec_lazy = Mock(
            return_value={
                "paths": {
                    "/deployments": {
                        "get": {"summary": "List deployments"},
                        "post": {"summary": "Create deployment"},
                    }
                }
            }
        )

        result = api_tool._run(query)

        assert "management" in result
        assert "/deployments" in result
        assert "List deployments" in result

    def test_run_execution_query(self, api_tool, mock_requests_wrapper):
        """Test running an execution query."""
        query = "Create a deployment"

        # Mock the intent analysis
        api_tool.spec_registry.analyse_query_intent = Mock(return_value={"intent": "management"})
        api_tool.cache_manager.get_spec_lazy = Mock(return_value={"paths": {}})

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            result = api_tool._run(
                query, method="POST", endpoint="/deployments", data={"name": "test"}
            )

        mock_requests_wrapper.post.assert_called_once()
        assert "API Call Executed Successfully" in result

    def test_run_error_handling(self, api_tool):
        """Test error handling in _run method."""
        query = "Test query"

        # Mock an error in intent analysis
        api_tool.spec_registry.analyse_query_intent = Mock(side_effect=Exception("Test error"))

        result = api_tool._run(query)

        assert "Error processing query: Test error" in result

    def test_create_intent_aware_api_tool(self, mock_llm, temp_cache_dir):
        """Test the convenience function for creating the tool."""
        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.load_api_spec"
        ) as mock_load_spec:
            mock_load_spec.return_value = {"swagger": "2.0", "paths": {}}

            with patch("elastic_cloud_agent.tools.intent_aware_api_tool.create_requests_wrapper"):
                tool = create_intent_aware_api_tool(
                    llm=mock_llm, spec_path=None, cache_dir=temp_cache_dir
                )

                assert isinstance(tool, IntentAwareApiTool)
                assert tool.llm == mock_llm

    def test_async_run(self, api_tool):
        """Test async version of _run method."""
        query = "Test query"

        # Mock the sync _run method
        api_tool._run = Mock(return_value="Test result")

        # Test async version
        import asyncio

        result = asyncio.run(api_tool._arun(query))

        assert result == "Test result"
        api_tool._run.assert_called_once_with(query)

    def test_runtime_validation_with_empty_query(self, api_tool, mock_requests_wrapper):
        """Test runtime validation when agent calls with only method/endpoint (no query)."""
        # This simulates the scenario where the agent makes a follow-up call
        # with only method and endpoint parameters, without a query
        
        # Mock the intent analysis to handle empty query
        api_tool.spec_registry.analyse_query_intent = Mock(return_value={"intent": "management"})
        api_tool.cache_manager.get_spec_lazy = Mock(return_value={"paths": {}})
        
        # Mock LLM response for _is_execution_query
        mock_response = Mock()
        mock_response.content = "execution"
        api_tool.llm.invoke.return_value = mock_response

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            # This should not raise a validation error
            result = api_tool._run("", method="GET", endpoint="/deployments")

        # Verify the API call was executed
        mock_requests_wrapper.get.assert_called_once_with("https://test.elastic.com/deployments")
        assert "API Call Executed Successfully" in result
        assert "GET" in result

    def test_agent_follow_up_call_pattern(self, api_tool, mock_requests_wrapper):
        """Test the exact agent follow-up call pattern with only kwargs."""
        # This simulates the agent calling with only method/endpoint as kwargs
        # without any positional arguments
        
        # Mock the intent analysis to handle empty query
        api_tool.spec_registry.analyse_query_intent = Mock(return_value={"intent": "management"})
        api_tool.cache_manager.get_spec_lazy = Mock(return_value={"paths": {}})
        
        # Mock LLM response for _is_execution_query
        mock_response = Mock()
        mock_response.content = "execution"
        api_tool.llm.invoke.return_value = mock_response

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            # This mimics the exact agent call pattern: {'method': 'GET', 'endpoint': '/deployments'}
            result = api_tool._run(method="GET", endpoint="/deployments")

        # Verify the API call was executed
        mock_requests_wrapper.get.assert_called_once_with("https://test.elastic.com/deployments")
        assert "API Call Executed Successfully" in result
        assert "GET" in result

    def test_fallback_guidance(self, api_tool):
        """Test fallback guidance when no optimised spec is available."""
        query = "How do I manage unknown resources?"
        intent = "management"
        
        # Mock LLM response for fallback guidance
        mock_response = Mock()
        mock_response.content = "**Suggested endpoints:**\n- GET /deployments\n- GET /account"
        api_tool.llm.invoke.return_value = mock_response
        
        result = api_tool._provide_fallback_guidance(query, intent)
        
        assert "No specific API endpoints found" in result
        assert "Alternative approaches:" in result
        assert "management" in result
        assert query in result

    def test_query_expansion(self, api_tool):
        """Test query expansion for better API matching."""
        query = "I want to create a new deployment for my application"
        
        # Mock LLM response for query expansion
        mock_response = Mock()
        mock_response.content = "create deployment provision new instance application cluster"
        api_tool.llm.invoke.return_value = mock_response
        
        expanded = api_tool._expand_query_for_api_matching(query)
        
        assert "create deployment provision new instance application cluster" in expanded
        
        # Test short query doesn't get expanded
        short_query = "list deployments"
        expanded_short = api_tool._expand_query_for_api_matching(short_query)
        assert expanded_short == short_query

    def test_response_caching(self, api_tool, mock_requests_wrapper):
        """Test API response caching for GET requests."""
        # Mock the intent analysis and LLM
        api_tool.spec_registry.analyse_query_intent = Mock(return_value={"intent": "management"})
        api_tool.cache_manager.get_spec_lazy = Mock(return_value={"paths": {}})
        
        mock_response = Mock()
        mock_response.content = "execution"
        api_tool.llm.invoke.return_value = mock_response

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.Config.ELASTIC_CLOUD_BASE_URL",
            "https://test.elastic.com",
        ):
            # First call - should hit the API
            result1 = api_tool._run(method="GET", endpoint="/deployments")
            mock_requests_wrapper.get.assert_called_once_with("https://test.elastic.com/deployments")
            
            # Reset mock call count
            mock_requests_wrapper.get.reset_mock()
            
            # Second call - should use cache
            result2 = api_tool._run(method="GET", endpoint="/deployments")
            mock_requests_wrapper.get.assert_not_called()  # Should not call API again
            
            # Results should be the same
            assert result1 == result2

    def test_response_validation(self, api_tool):
        """Test API response validation."""
        # Test valid response
        valid_response = {"deployments": [{"id": "123", "name": "test"}]}
        validation = api_tool._validate_api_response(valid_response, "GET", "/deployments")
        assert validation["is_valid"] is True
        
        # Test error response
        error_response = {"error": "Not found"}
        validation = api_tool._validate_api_response(error_response, "GET", "/deployments")
        assert validation["is_valid"] is False
        assert "API Error" in validation["error"]
        
        # Test HTTP error status
        http_error = {"status": 404, "message": "Not found"}
        validation = api_tool._validate_api_response(http_error, "GET", "/deployments")
        assert validation["is_valid"] is False
        assert "HTTP 404" in validation["error"]

    def test_intelligent_fallback_generation(self, api_tool):
        """Test intelligent fallback suggestions using LLM."""
        query = "How do I scale my cluster?"
        intent = "management"
        
        # Mock LLM response for intelligent fallback
        mock_response = Mock()
        mock_response.content = """
## Suggested Endpoints

1. **GET /deployments/{id}** - Retrieve current deployment details
2. **PUT /deployments/{id}** - Update deployment configuration for scaling
3. **GET /deployments/{id}/elasticsearch** - Check current Elasticsearch configuration
"""
        api_tool.llm.invoke.return_value = mock_response
        
        result = api_tool._generate_intelligent_fallback(query, intent)
        
        assert "Suggested Endpoints" in result
        assert "GET /deployments/{id}" in result
        assert "PUT /deployments/{id}" in result

    def test_response_summary_generation(self, api_tool):
        """Test intelligent response summary generation."""
        # Test deployment response
        deployment_response = {"deployments": [{"id": "1"}, {"id": "2"}]}
        summary = api_tool._generate_response_summary(deployment_response, "management", "list deployments")
        assert "Found 2 deployment(s)" in summary
        
        # Test account response
        account_response = {"account": {"id": "123", "name": "test"}}
        summary = api_tool._generate_response_summary(account_response, "basic", "get account")
        assert "Account information retrieved" in summary
        
        # Test single resource response
        resource_response = {"id": "abc123", "name": "test-resource"}
        summary = api_tool._generate_response_summary(resource_response, "management", "get resource")
        assert "Resource retrieved (ID: abc123)" in summary
        
        # Test text response
        text_response = "Simple text response"
        summary = api_tool._generate_response_summary(text_response, "basic", "test")
        assert "Simple text response" in summary

    def test_cache_key_generation(self, api_tool):
        """Test cache key generation for API responses."""
        # Test with different parameters
        key1 = api_tool._generate_cache_key("GET", "/deployments", None)
        key2 = api_tool._generate_cache_key("GET", "/deployments", {"param": "value"})
        key3 = api_tool._generate_cache_key("POST", "/deployments", None)
        
        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
        
        # Same parameters should generate same key
        key4 = api_tool._generate_cache_key("GET", "/deployments", None)
        assert key1 == key4

    def test_cache_expiration(self, api_tool):
        """Test cache expiration logic."""
        cache_key = "test_key"
        response = "test_response"
        
        # Cache a response
        api_tool._cache_response(cache_key, response)
        
        # Should be retrievable immediately
        cached = api_tool._get_cached_response(cache_key)
        assert cached == response
        
        # Mock time to simulate expiration
        with patch('time.time', return_value=time.time() + 400):  # 400 seconds later
            cached_expired = api_tool._get_cached_response(cache_key)
            assert cached_expired is None

    def test_cache_cleanup(self, api_tool):
        """Test cache cleanup when size limit exceeded."""
        # Fill cache beyond limit
        for i in range(105):  # Exceed the 100 entry limit
            api_tool._cache_response(f"key_{i}", f"response_{i}")
        
        # Should trigger cleanup
        assert len(api_tool._response_cache) <= 100

    def test_error_response_formatting(self, api_tool):
        """Test error response formatting with suggestions."""
        method = "GET"
        url = "https://test.elastic.com/deployments"
        error_data = {"error": "Unauthorized"}
        intent = "management"
        error_msg = "API Error: Unauthorized"
        
        result = api_tool._format_error_response(method, url, error_data, intent, error_msg)
        
        assert "API Call Failed" in result
        assert "management" in result
        assert "GET" in result
        assert "Unauthorized" in result
        assert "Suggestions:" in result
        assert "Check if the endpoint path is correct" in result

    def test_api_error_handling(self, api_tool):
        """Test API error handling with troubleshooting steps."""
        method = "POST"
        endpoint = "/deployments"
        error = "Connection timeout"
        intent = "management"
        
        result = api_tool._handle_api_error(method, endpoint, error, intent)
        
        assert "API Call Error" in result
        assert "management" in result
        assert "POST" in result
        assert "Connection timeout" in result
        assert "Troubleshooting Steps:" in result
        assert "Check your network connection" in result


class TestIntegration:
    """Integration tests for the intent-aware API tool."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        with patch("elastic_cloud_agent.tools.intent_aware_api_tool.Config") as mock_config:
            mock_config.ELASTIC_CLOUD_BASE_URL = "https://test.elastic.com"
            yield mock_config

    def test_full_workflow_exploratory(self, mock_config):
        """Test full workflow for exploratory query."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        sample_spec = {
            "swagger": "2.0",
            "paths": {
                "/deployments": {
                    "get": {"summary": "List deployments"},
                    "post": {"summary": "Create deployment"},
                }
            },
        }

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.load_api_spec"
        ) as mock_load_spec:
            mock_load_spec.return_value = sample_spec

            with patch("elastic_cloud_agent.tools.intent_aware_api_tool.create_requests_wrapper"):
                tool = create_intent_aware_api_tool(llm=mock_llm)

                result = tool._run("What APIs are available for deployments?")

                assert "management" in result
                assert "/deployments" in result
                assert "List deployments" in result
                assert "Create deployment" in result

    def test_full_workflow_execution(self, mock_config):
        """Test full workflow for execution query."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        sample_spec = {"swagger": "2.0", "paths": {}}

        mock_requests_wrapper = Mock()
        mock_requests_wrapper.post.return_value = '{"created": true}'

        with patch(
            "elastic_cloud_agent.tools.intent_aware_api_tool.load_api_spec"
        ) as mock_load_spec:
            mock_load_spec.return_value = sample_spec

            with patch(
                "elastic_cloud_agent.tools.intent_aware_api_tool.create_requests_wrapper"
            ) as mock_create_wrapper:
                mock_create_wrapper.return_value = mock_requests_wrapper

                tool = create_intent_aware_api_tool(llm=mock_llm)

                result = tool._run(
                    "Create a deployment",
                    method="POST",
                    endpoint="/deployments",
                    data={"name": "test-deployment"},
                )

                mock_requests_wrapper.post.assert_called_once()
                assert "API Call Executed Successfully" in result
                assert "POST" in result
                assert "management" in result
