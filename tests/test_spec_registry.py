"""
Tests for the spec registry module.
"""

from unittest.mock import Mock

import pytest

from elastic_cloud_agent.spec_optimization.spec_registry import (
    SpecRegistry,
    analyse_query_intent,
    create_spec_registry,
    get_spec_for_query,
)


class TestSpecRegistry:
    """Tests for the SpecRegistry class."""

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
                        "description": "Get a list of all deployments with detailed information",
                        "example": {"deployments": []},
                    },
                    "post": {
                        "operationId": "create-deployment",
                        "tags": ["Deployments"],
                        "summary": "Create deployment",
                        "description": "Create a new deployment with specified configuration",
                    },
                },
                "/deployments/{id}": {
                    "get": {
                        "operationId": "get-deployment",
                        "tags": ["Deployments"],
                        "summary": "Get deployment",
                    },
                    "delete": {
                        "operationId": "delete-deployment",
                        "tags": ["Deployments"],
                        "summary": "Delete deployment",
                    },
                },
                "/costs": {
                    "get": {
                        "operationId": "get-costs-overview",
                        "tags": ["BillingCostsAnalysis"],
                        "summary": "Get billing costs",
                    }
                },
                "/api-keys": {
                    "get": {
                        "operationId": "get-api-keys",
                        "tags": ["Authentication"],
                        "summary": "List API keys",
                    }
                },
            },
            "definitions": {
                "Deployment": {
                    "type": "object",
                    "description": "A deployment object with all configuration details",
                    "properties": {"id": {"type": "string"}},
                    "example": {"id": "123"},
                }
            },
        }

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @pytest.fixture
    def registry(self, sample_spec, mock_llm):
        """Create a registry instance for testing."""
        return SpecRegistry(sample_spec, mock_llm)

    def test_classify_intent_basic(self, sample_spec):
        """Test classification of basic intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("get deployment status") == "basic"
        assert registry.classify_intent("list all deployments") == "basic"
        assert registry.classify_intent("show me deployment info") == "basic"
        assert registry.classify_intent("view cluster health") == "basic"

    def test_classify_intent_management(self, sample_spec):
        """Test classification of management intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("create a new deployment") == "management"
        assert registry.classify_intent("restart the cluster") == "management"
        assert registry.classify_intent("scale up my deployment") == "management"
        assert registry.classify_intent("update cluster configuration") == "management"

    def test_classify_intent_advanced(self, sample_spec):
        """Test classification of advanced intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "advanced"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("migrate deployment to new template") == "advanced"
        assert registry.classify_intent("force restart with upgrade") == "advanced"
        assert registry.classify_intent("bulk export configurations") == "advanced"
        assert registry.classify_intent("admin organization settings") == "advanced"

    def test_classify_intent_monitoring(self, sample_spec):
        """Test classification of monitoring intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "monitoring"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("check cluster health and performance") == "monitoring"
        assert registry.classify_intent("view logs and metrics") == "monitoring"
        assert registry.classify_intent("monitor deployment status") == "monitoring"

    def test_classify_intent_security(self, sample_spec):
        """Test classification of security intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "security"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("manage API keys and authentication") == "security"
        assert registry.classify_intent("setup user authentication") == "security"
        assert registry.classify_intent("configure role permissions") == "security"

    def test_classify_intent_billing(self, sample_spec):
        """Test classification of billing intents."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "billing"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("check my billing costs") == "billing"
        assert registry.classify_intent("view usage and pricing") == "billing"
        assert registry.classify_intent("get invoice information") == "billing"

    def test_classify_intent_default(self, sample_spec):
        """Test default classification for unclear queries."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "invalid_intent"  # Invalid intent should fallback to management
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry(sample_spec, mock_llm)

        assert registry.classify_intent("hello world") == "management"
        assert registry.classify_intent("") == "management"
        assert registry.classify_intent("random text here") == "management"

    def test_get_optimised_spec_basic(self, registry):
        """Test getting optimised spec for basic intent."""
        spec = registry.get_optimised_spec("basic")

        # Should have core operations
        assert "/deployments" in spec["paths"]
        assert "/deployments/{id}" in spec["paths"]

        # Should not have advanced operations
        assert "/costs" not in spec["paths"]

        # Should be minimized (no examples)
        get_op = spec["paths"]["/deployments"]["get"]
        assert "example" not in get_op

    def test_get_optimised_spec_security(self, registry):
        """Test getting optimised spec for security intent."""
        # Mock the LLM response for tag selection
        registry.llm.invoke.return_value.content = '["Authentication"]'

        spec = registry.get_optimised_spec("security")

        # Should have security-related operations
        assert "/api-keys" in spec["paths"]

        # Should not have unrelated operations
        assert "/costs" not in spec["paths"]

    def test_get_optimised_spec_billing(self, registry):
        """Test getting optimised spec for billing intent."""
        spec = registry.get_optimised_spec("billing")

        # Should have billing operations
        assert "/costs" in spec["paths"]

    def test_get_optimised_spec_caching(self, registry):
        """Test that spec caching works correctly."""
        # First call should populate cache
        spec1 = registry.get_optimised_spec("basic", use_cache=True)

        # Second call should return cached version
        spec2 = registry.get_optimised_spec("basic", use_cache=True)

        # Should be the same object (cached)
        assert spec1 is spec2

        # Call without cache should return different object
        spec3 = registry.get_optimised_spec("basic", use_cache=False)
        assert spec1 is not spec3

    def test_get_spec_for_query(self, registry):
        """Test getting spec directly from query."""
        spec = registry.get_spec_for_query("list all deployments")

        # Should classify as basic and return appropriate spec
        assert "/deployments" in spec["paths"]

        # Should be minimized
        get_op = spec["paths"]["/deployments"]["get"]
        assert "example" not in get_op

    def test_get_available_intents(self, registry):
        """Test getting available intent classifications."""
        intents = registry.get_available_intents()

        assert "basic" in intents
        assert "management" in intents
        assert "advanced" in intents
        assert "monitoring" in intents
        assert "security" in intents
        assert "billing" in intents

        # Should have descriptions
        assert isinstance(intents["basic"], str)
        assert len(intents["basic"]) > 0

    def test_get_intent_keywords(self, registry):
        """Test getting keywords for specific intents."""
        # This method doesn't exist in the current implementation
        # Removing this test as it's not part of the LLM-based implementation
        pass

    def test_analyse_query_intent(self, registry):
        """Test detailed query intent analysis."""
        # Mock the LLM response for intent classification
        registry.llm.invoke.return_value.content = "management"

        result = registry.analyse_query_intent("create a new deployment")

        assert result["intent"] == "management"
        assert result["confidence"] == 1.0  # LLM-based classification has confidence 1.0
        assert "query_words" in result

    def test_analyse_query_intent_multiple_matches(self, registry):
        """Test intent analysis with multiple keyword matches."""
        # Mock the LLM response for intent classification
        registry.llm.invoke.return_value.content = "monitoring"

        result = registry.analyse_query_intent("get deployment health status")

        # Should pick the intent based on LLM classification
        assert result["intent"] == "monitoring"
        assert result["confidence"] == 1.0

    def test_cache_management(self, registry):
        """Test cache management functions."""
        # Initially empty
        stats = registry.get_cache_stats()
        assert stats["cached_specs"] == 0

        # Load some specs
        registry.get_optimised_spec("basic")
        registry.get_optimised_spec("management")

        stats = registry.get_cache_stats()
        assert stats["cached_specs"] == 2
        assert "intent_basic" in stats["cache_keys"]
        assert "intent_management" in stats["cache_keys"]

        # Clear cache
        registry.clear_cache()
        stats = registry.get_cache_stats()
        assert stats["cached_specs"] == 0

    def test_preload_common_specs(self, registry):
        """Test preloading common specs."""
        # Initially empty
        assert registry.get_cache_stats()["cached_specs"] == 0

        # Preload
        registry.preload_common_specs()

        # Should have cached common intents
        stats = registry.get_cache_stats()
        assert stats["cached_specs"] == 3  # basic, management, advanced
        assert "intent_basic" in stats["cache_keys"]
        assert "intent_management" in stats["cache_keys"]
        assert "intent_advanced" in stats["cache_keys"]

    def test_invalid_intent_fallback(self, registry):
        """Test fallback for invalid intents."""
        spec = registry.get_optimised_spec("invalid_intent")

        # Should fall back to management intent
        assert "paths" in spec
        assert isinstance(spec["paths"], dict)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_spec(self):
        """Sample spec for convenience function tests."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API"},
            "paths": {
                "/deployments": {
                    "get": {
                        "operationId": "list-deployments",
                        "tags": ["Deployments"],
                        "description": "Long description with verbose details",
                        "example": {"data": "example"},
                    }
                }
            },
        }

    def test_create_spec_registry(self, sample_spec):
        """Test spec registry creation."""
        mock_llm = Mock()
        registry = create_spec_registry(sample_spec, mock_llm)

        assert isinstance(registry, SpecRegistry)
        assert registry.base_spec == sample_spec
        assert registry.llm == mock_llm

    def test_get_spec_for_query_convenience(self, sample_spec):
        """Test convenience function for getting spec by query."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        spec = get_spec_for_query("list deployments", sample_spec, mock_llm)

        # Should return optimised spec
        assert "paths" in spec
        assert "/deployments" in spec["paths"]

        # Should be minimized
        get_op = spec["paths"]["/deployments"]["get"]
        assert "example" not in get_op

    def test_analyse_query_intent_convenience(self):
        """Test convenience function for intent analysis."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        result = analyse_query_intent("create new deployment", mock_llm)

        assert result["intent"] == "management"
        assert "confidence" in result


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_spec(self):
        """Test registry with empty spec."""
        mock_llm = Mock()
        registry = SpecRegistry({}, mock_llm)

        # Should handle gracefully
        spec = registry.get_optimised_spec("basic")
        assert isinstance(spec, dict)

    def test_malformed_spec(self):
        """Test registry with malformed spec."""
        malformed_spec = {"swagger": "2.0", "paths": "invalid_paths_value"}  # Should be dict
        mock_llm = Mock()

        registry = SpecRegistry(malformed_spec, mock_llm)

        # Should handle gracefully
        spec = registry.get_optimised_spec("basic")
        assert isinstance(spec, dict)

    def test_query_with_special_characters(self):
        """Test intent classification with special characters."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "basic"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry({}, mock_llm)

        # Should handle special characters gracefully
        assert registry.classify_intent("get @deployment #status!") == "basic"

        # Test management intent
        mock_response.content = "management"
        assert registry.classify_intent("create/update deployment") == "management"

    def test_very_long_query(self):
        """Test intent classification with very long query."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry({}, mock_llm)

        long_query = "create " + "deployment " * 100 + "with advanced configuration"
        intent = registry.classify_intent(long_query)

        # Should still classify correctly despite length
        assert intent == "management"

    def test_empty_query(self):
        """Test intent classification with empty query."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "management"
        mock_llm.invoke.return_value = mock_response

        registry = SpecRegistry({}, mock_llm)

        assert registry.classify_intent("") == "management"
        assert registry.classify_intent("   ") == "management"
