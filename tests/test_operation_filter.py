"""
Tests for the operation filter module.
"""

import pytest

from elastic_cloud_agent.spec_optimization.operation_filter import (
    analyse_operation_distribution,
    classify_operation_complexity,
    filter_operations_by_complexity,
    get_advanced_operations_spec,
    get_core_operations_spec,
    get_moderate_operations_spec,
    get_operations_by_complexity,
)


class TestClassifyOperationComplexity:
    """Tests for operation complexity classification."""

    def test_core_operation_patterns(self):
        """Test classification of explicit core operations."""
        # Core deployment operations
        assert (
            classify_operation_complexity(
                "get-deployment", ["Deployments"], "GET", "/deployments/{id}"
            )
            == "core"
        )

        assert (
            classify_operation_complexity(
                "create-deployment", ["Deployments"], "POST", "/deployments"
            )
            == "core"
        )

        # Core account operations
        assert (
            classify_operation_complexity("get-current-account", ["Accounts"], "GET", "/account")
            == "core"
        )

    def test_advanced_operation_patterns(self):
        """Test classification of explicit advanced operations."""
        # Advanced deployment operations
        assert (
            classify_operation_complexity(
                "deploy-deployment-template", ["Deployments"], "POST", "/deployments/_template"
            )
            == "advanced"
        )

        # Traffic filtering
        assert (
            classify_operation_complexity(
                "create-traffic-filter-ruleset",
                ["DeploymentsTrafficFilter"],
                "POST",
                "/traffic-filter",
            )
            == "advanced"
        )

        # Extensions
        assert (
            classify_operation_complexity("upload-extension", ["Extensions"], "POST", "/extensions")
            == "advanced"
        )

    def test_tag_based_classification(self):
        """Test classification based on operation tags."""
        # Core tags should default to core (unless other factors override)
        assert (
            classify_operation_complexity("some-operation", ["Deployments"], "GET", "/some/path")
            == "core"
        )

        assert (
            classify_operation_complexity(
                "another-operation", ["Accounts"], "GET", "/account/something"
            )
            == "core"
        )

        # Advanced tags should be advanced
        assert (
            classify_operation_complexity(
                "some-operation", ["BillingCostsAnalysis"], "GET", "/costs"
            )
            == "advanced"
        )

        assert (
            classify_operation_complexity(
                "org-operation", ["Organizations"], "POST", "/organizations"
            )
            == "advanced"
        )

    def test_method_based_classification(self):
        """Test classification based on HTTP method."""
        # GET is simple for unknown operations
        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "GET", "/some/path") == "core"
        )

        # POST/PUT are moderate for unknown operations
        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "POST", "/some/path")
            == "moderate"
        )

        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "PUT", "/some/path")
            == "moderate"
        )

        # DELETE is complex
        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "DELETE", "/some/path")
            == "advanced"
        )

    def test_path_complexity_classification(self):
        """Test classification based on path complexity."""
        # Deep nesting suggests complexity
        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "GET", "/a/b/c/d/e/f")
            == "advanced"
        )

        # Admin paths are advanced
        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "GET", "/admin/config")
            == "advanced"
        )

        assert (
            classify_operation_complexity("unknown-op", ["SomeTag"], "GET", "/management/settings")
            == "advanced"
        )

    def test_summary_based_classification(self):
        """Test classification based on operation summary."""
        # Advanced keywords in summary
        assert (
            classify_operation_complexity(
                "unknown-op", ["SomeTag"], "GET", "/path", "Advanced configuration"
            )
            == "advanced"
        )

        assert (
            classify_operation_complexity(
                "unknown-op", ["SomeTag"], "GET", "/path", "Force restart cluster"
            )
            == "advanced"
        )

        assert (
            classify_operation_complexity(
                "unknown-op", ["SomeTag"], "GET", "/path", "Internal API endpoint"
            )
            == "advanced"
        )

        # Normal summary stays with method-based classification
        assert (
            classify_operation_complexity(
                "unknown-op", ["SomeTag"], "GET", "/path", "Get user information"
            )
            == "core"
        )

    def test_core_tag_with_complex_method(self):
        """Test that core tags with complex methods become moderate."""
        # Core tag with DELETE method
        assert (
            classify_operation_complexity(
                "unknown-op", ["Deployments"], "DELETE", "/deployments/{id}"
            )
            == "moderate"
        )


class TestFilterOperationsByComplexity:
    """Tests for filtering operations by complexity."""

    @pytest.fixture
    def sample_spec(self):
        """Sample OpenAPI spec with operations of different complexity."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API", "version": "1.0"},
            "paths": {
                "/deployments": {
                    "get": {
                        "operationId": "list-deployments",
                        "tags": ["Deployments"],
                        "summary": "List deployments",
                    },
                    "post": {
                        "operationId": "create-deployment",
                        "tags": ["Deployments"],
                        "summary": "Create deployment",
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
                "/admin/config": {
                    "put": {
                        "operationId": "update-admin-config",
                        "tags": ["Config"],
                        "summary": "Update advanced configuration",
                    }
                },
            },
        }

    def test_filter_core_operations(self, sample_spec):
        """Test filtering for core operations only."""
        result = filter_operations_by_complexity(sample_spec, {"core"})

        paths = result["paths"]

        # Should include core operations
        assert "/deployments" in paths
        assert "get" in paths["/deployments"]  # list-deployments is core
        assert "post" in paths["/deployments"]  # create-deployment is core

        assert "/deployments/{id}" in paths
        assert "get" in paths["/deployments/{id}"]  # get-deployment is core

        # Should not include advanced operations
        assert "/costs" not in paths
        assert "/admin/config" not in paths

        # delete-deployment is explicitly in CORE_OPERATION_PATTERNS, so it should be included
        assert "delete" in paths["/deployments/{id}"]

    def test_filter_core_and_moderate_operations(self, sample_spec):
        """Test filtering for core and moderate operations."""
        result = filter_operations_by_complexity(sample_spec, {"core", "moderate"})

        paths = result["paths"]

        # Should include core operations
        assert "/deployments" in paths
        assert "get" in paths["/deployments"]
        assert "post" in paths["/deployments"]

        # Should include moderate operations (DELETE in core tag)
        assert "/deployments/{id}" in paths
        assert "delete" in paths["/deployments/{id}"]

        # Should not include advanced operations
        assert "/costs" not in paths
        assert "/admin/config" not in paths

    def test_filter_all_operations(self, sample_spec):
        """Test filtering for all operation types."""
        result = filter_operations_by_complexity(sample_spec, {"core", "moderate", "advanced"})

        paths = result["paths"]

        # Should include all operations
        assert "/deployments" in paths
        assert "/deployments/{id}" in paths
        assert "/costs" in paths
        assert "/admin/config" in paths

    def test_empty_complexity_set(self, sample_spec):
        """Test filtering with empty complexity set."""
        result = filter_operations_by_complexity(sample_spec, set())

        # Should have no operations
        assert result["paths"] == {}

    def test_invalid_spec_structure(self):
        """Test handling of specs without paths."""
        spec = {"swagger": "2.0", "info": {"title": "Test"}}
        result = filter_operations_by_complexity(spec, {"core"})

        # Should return original spec unchanged
        assert result == spec


class TestConvenienceFunctions:
    """Tests for convenience filtering functions."""

    @pytest.fixture
    def sample_spec(self):
        """Sample spec for testing convenience functions."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API"},
            "paths": {
                "/deployments": {
                    "get": {"operationId": "list-deployments", "tags": ["Deployments"]}
                },
                "/deployments/{id}": {
                    "delete": {"operationId": "delete-deployment", "tags": ["Deployments"]}
                },
                "/costs": {
                    "get": {"operationId": "get-costs-overview", "tags": ["BillingCostsAnalysis"]}
                },
            },
        }

    def test_get_core_operations_spec(self, sample_spec):
        """Test getting core operations only."""
        result = get_core_operations_spec(sample_spec)

        # Should only have core operations
        assert "/deployments" in result["paths"]
        assert "/deployments/{id}" in result["paths"]  # delete-deployment is core
        assert "/costs" not in result["paths"]  # Advanced

    def test_get_moderate_operations_spec(self, sample_spec):
        """Test getting core and moderate operations."""
        result = get_moderate_operations_spec(sample_spec)

        # Should have core and moderate
        assert "/deployments" in result["paths"]
        assert "/deployments/{id}" in result["paths"]  # delete-deployment is core
        assert "/costs" not in result["paths"]  # Advanced only

    def test_get_advanced_operations_spec(self, sample_spec):
        """Test getting all operations."""
        result = get_advanced_operations_spec(sample_spec)

        # Should have all operations
        assert "/deployments" in result["paths"]
        assert "/deployments/{id}" in result["paths"]
        assert "/costs" in result["paths"]


class TestAnalysisFunctions:
    """Tests for operation analysis functions."""

    @pytest.fixture
    def sample_spec(self):
        """Sample spec for analysis testing."""
        return {
            "swagger": "2.0",
            "info": {"title": "Test API"},
            "paths": {
                "/deployments": {
                    "get": {"operationId": "list-deployments", "tags": ["Deployments"]},
                    "post": {"operationId": "create-deployment", "tags": ["Deployments"]},
                },
                "/deployments/{id}": {
                    "delete": {"operationId": "delete-deployment", "tags": ["Deployments"]}
                },
                "/costs": {
                    "get": {"operationId": "get-costs-overview", "tags": ["BillingCostsAnalysis"]}
                },
                "/admin/config": {"put": {"operationId": "update-config", "tags": ["Config"]}},
            },
        }

    def test_analyse_operation_distribution(self, sample_spec):
        """Test analysis of operation distribution."""
        result = analyse_operation_distribution(sample_spec)

        # Should count operations by complexity
        assert result["core"] >= 2  # At least the core deployment operations
        assert result["moderate"] >= 0  # May not have moderate operations in this test
        assert result["advanced"] >= 2  # At least costs and admin operations

        # Total should match number of operations
        total = sum(result.values())
        assert total == 5

    def test_get_operations_by_complexity(self, sample_spec):
        """Test getting operations grouped by complexity."""
        result = get_operations_by_complexity(sample_spec)

        # Should have all complexity levels
        assert "core" in result
        assert "moderate" in result
        assert "advanced" in result

        # Core should include deployment GET/POST
        core_ops = result["core"]
        core_operation_ids = [op["operation_id"] for op in core_ops]
        assert "list-deployments" in core_operation_ids
        assert "create-deployment" in core_operation_ids

        # Advanced should include costs
        advanced_ops = result["advanced"]
        advanced_operation_ids = [op["operation_id"] for op in advanced_ops]
        assert "get-costs-overview" in advanced_operation_ids

    def test_empty_spec_analysis(self):
        """Test analysis with empty or invalid spec."""
        empty_spec = {"swagger": "2.0", "info": {"title": "Empty"}}

        distribution = analyse_operation_distribution(empty_spec)
        assert distribution == {"core": 0, "moderate": 0, "advanced": 0}

        operations = get_operations_by_complexity(empty_spec)
        assert operations == {"core": [], "moderate": [], "advanced": []}

    def test_malformed_spec_handling(self):
        """Test handling of malformed spec data."""
        malformed_spec = {
            "paths": {
                "/test": {
                    "get": "invalid_operation_data",  # Should be dict
                    "post": {"operationId": "valid-op", "tags": ["Test"]},
                }
            }
        }

        # Should handle gracefully and count only valid operations
        distribution = analyse_operation_distribution(malformed_spec)
        assert sum(distribution.values()) == 1  # Only the valid POST operation
