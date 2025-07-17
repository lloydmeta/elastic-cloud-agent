"""
Tests for the openapi_utils module.
"""

from unittest.mock import Mock

import pytest

from elastic_cloud_agent.utils.openapi_utils import (
    classify_intent,
    extract_tags_from_spec,
    filter_spec_by_tags,
)


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
            "/mixed-endpoint": {
                "get": {
                    "tags": ["Accounts", "Deployments"],
                    "summary": "Mixed tags endpoint",
                    "operationId": "mixed-endpoint",
                }
            },
        },
    }


class TestExtractTagsFromSpec:
    """Test cases for extract_tags_from_spec function."""

    def test_extract_tags_with_tags_section(self, sample_api_spec):
        """Test extracting tags from the tags section of the spec."""
        tags = extract_tags_from_spec(sample_api_spec)

        # Should extract from the sample spec's tags section
        expected_tags = ["Accounts", "BillingCostsAnalysis", "Deployments"]
        assert sorted(tags) == sorted(expected_tags)

    def test_extract_tags_from_operations_fallback(self, sample_api_spec):
        """Test extracting tags from operations when tags section is missing."""
        # Remove tags section from spec
        spec_without_tags = sample_api_spec.copy()
        del spec_without_tags["tags"]

        tags = extract_tags_from_spec(spec_without_tags)

        # Should extract from operations
        expected_tags = ["Accounts", "BillingCostsAnalysis", "Deployments"]
        assert sorted(tags) == sorted(expected_tags)

    def test_extract_tags_handles_empty_spec(self):
        """Test tag extraction with empty spec."""
        tags = extract_tags_from_spec({})

        # Should return empty list
        assert tags == []

    def test_extract_tags_handles_string_tags(self):
        """Test extraction with string tags instead of dict tags."""
        spec_with_string_tags = {
            "tags": ["StringTag1", "StringTag2"],
            "paths": {
                "/test": {
                    "get": {
                        "tags": ["StringTag3"],
                        "summary": "Test endpoint",
                    }
                }
            },
        }

        tags = extract_tags_from_spec(spec_with_string_tags)

        expected_tags = ["StringTag1", "StringTag2", "StringTag3"]
        assert sorted(tags) == sorted(expected_tags)


class TestClassifyIntent:
    """Test cases for classify_intent function."""

    def test_classify_intent_deployment_query(self):
        """Test intent classification for deployment-related queries."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Deployments"
        mock_llm.invoke.return_value = mock_response

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent("How do I create a new deployment?", available_tags, mock_llm)

        assert "Deployments" in result
        mock_llm.invoke.assert_called_once()

    def test_classify_intent_multiple_categories(self):
        """Test intent classification returning multiple categories."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Deployments, Accounts, BillingCostsAnalysis"
        mock_llm.invoke.return_value = mock_response

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent(
            "Show me deployment costs for my account", available_tags, mock_llm
        )

        assert len(result) <= 3  # Should limit to 3 categories max
        assert "Deployments" in result
        assert "Accounts" in result
        assert "BillingCostsAnalysis" in result

    def test_classify_intent_invalid_categories(self):
        """Test intent classification with invalid category names."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "InvalidCategory, Deployments"
        mock_llm.invoke.return_value = mock_response

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent("Some query", available_tags, mock_llm)

        assert "InvalidCategory" not in result
        assert "Deployments" in result

    def test_classify_intent_fallback(self):
        """Test intent classification fallback when LLM fails."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM error")

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent("Some query", available_tags, mock_llm)

        assert result == ["Deployments", "Accounts"]

    def test_classify_intent_empty_response(self):
        """Test intent classification with empty LLM response."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent("Some query", available_tags, mock_llm)

        assert result == ["Deployments", "Accounts"]

    def test_classify_intent_no_content_attribute(self):
        """Test intent classification when response has no content attribute."""
        mock_llm = Mock()
        mock_response = "Deployments"  # String response instead of object with content
        mock_llm.invoke.return_value = mock_response

        available_tags = ["Accounts", "Deployments", "BillingCostsAnalysis"]
        result = classify_intent("Some query", available_tags, mock_llm)

        assert "Deployments" in result


class TestFilterSpecByTags:
    """Test cases for filter_spec_by_tags function."""

    def test_filter_spec_by_tags_single_tag(self, sample_api_spec):
        """Test filtering OpenAPI spec by a single tag."""
        filtered_spec = filter_spec_by_tags(sample_api_spec, ["Accounts"])

        paths = filtered_spec["paths"]

        # Should include account endpoint
        assert "/account" in paths

        # Should include mixed endpoint (has Accounts tag)
        assert "/mixed-endpoint" in paths

        # Should not include deployments-only endpoint
        assert "/deployments" not in paths

        # Should not include billing endpoint
        assert "/billing/costs" not in paths

    def test_filter_spec_by_tags_multiple_tags(self, sample_api_spec):
        """Test filtering OpenAPI spec by multiple tags."""
        filtered_spec = filter_spec_by_tags(sample_api_spec, ["Accounts", "Deployments"])

        paths = filtered_spec["paths"]

        # Should include all Accounts and Deployments endpoints
        assert "/account" in paths
        assert "/deployments" in paths
        assert "/mixed-endpoint" in paths

        # Should not include billing endpoint
        assert "/billing/costs" not in paths

    def test_filter_spec_by_tags_nonexistent_tag(self, sample_api_spec):
        """Test filtering OpenAPI spec by non-existent tags."""
        filtered_spec = filter_spec_by_tags(sample_api_spec, ["NonExistentTag"])

        paths = filtered_spec["paths"]

        # Should have no paths
        assert len(paths) == 0

    def test_filter_spec_preserves_structure(self, sample_api_spec):
        """Test that filtering preserves the overall OpenAPI spec structure."""
        filtered_spec = filter_spec_by_tags(sample_api_spec, ["Accounts"])

        # Should preserve top-level structure
        assert "swagger" in filtered_spec
        assert "info" in filtered_spec
        assert "host" in filtered_spec
        assert "basePath" in filtered_spec
        assert "tags" in filtered_spec
        assert "paths" in filtered_spec

    def test_filter_handles_empty_paths(self):
        """Test filtering with empty paths section."""
        spec = {"paths": {}}
        result = filter_spec_by_tags(spec, ["Deployments"])
        assert result["paths"] == {}

    def test_filter_handles_missing_tags(self):
        """Test filtering operations that have no tags."""
        spec = {
            "paths": {
                "/no-tags": {
                    "get": {"summary": "Operation without tags", "operationId": "no-tags-op"}
                }
            }
        }

        result = filter_spec_by_tags(spec, ["Deployments"])

        # Should not include operations without tags
        assert "/no-tags" not in result["paths"]

    def test_filter_handles_non_dict_operations(self):
        """Test filtering with non-dictionary operation values."""
        spec = {
            "paths": {
                "/invalid": {
                    "get": "not a dict",
                    "post": {"tags": ["Deployments"], "summary": "Valid operation"},
                }
            }
        }

        result = filter_spec_by_tags(spec, ["Deployments"])

        # Should include the path but only the valid operation
        assert "/invalid" in result["paths"]
        assert "post" in result["paths"]["/invalid"]
        assert "get" not in result["paths"]["/invalid"]
