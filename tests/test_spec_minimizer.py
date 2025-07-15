"""
Tests for the spec minimizer module.
"""

import pytest

from elastic_cloud_agent.spec_optimization.spec_minimizer import (
    calculate_size_reduction,
    minimize_description,
    minimize_operation,
    minimize_schema_object,
    minimize_spec,
    remove_verbose_fields,
)


class TestMinimizeDescription:
    """Tests for the minimize_description function."""

    def test_short_description_unchanged(self):
        """Test that short descriptions are unchanged."""
        description = "Short description"
        result = minimize_description(description)
        assert result == "Short description"

    def test_long_description_truncated(self):
        """Test that long descriptions are truncated."""
        description = "This is a very long description that exceeds the maximum length and should be truncated properly."
        result = minimize_description(description, max_length=50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_sentence_boundary_truncation(self):
        """Test that truncation respects sentence boundaries."""
        description = "First sentence. Second sentence. Third sentence."
        result = minimize_description(description, max_length=30)
        # With max_length=30, only "First sentence." (15 chars) fits
        assert result == "First sentence."

    def test_whitespace_cleanup(self):
        """Test that extra whitespace is cleaned up."""
        description = "  Multiple   spaces\n\nand   newlines  "
        result = minimize_description(description)
        assert result == "Multiple spaces and newlines"

    def test_empty_description(self):
        """Test handling of empty descriptions."""
        assert minimize_description("") == ""
        assert minimize_description(None) == ""
        assert minimize_description(123) == ""

    def test_single_long_sentence(self):
        """Test truncation of single long sentence."""
        description = (
            "This is one very long sentence without periods that should be truncated with ellipsis"
        )
        result = minimize_description(description, max_length=50)
        assert len(result) <= 50
        assert result.endswith("...")


class TestRemoveVerboseFields:
    """Tests for the remove_verbose_fields function."""

    def test_remove_default_verbose_fields(self):
        """Test removal of default verbose fields."""
        obj = {
            "type": "object",
            "description": "Keep this",
            "example": "Remove this",
            "x-doc": "Remove this",
            "properties": {"field": {"type": "string", "examples": ["Remove this"]}},
        }

        result = remove_verbose_fields(obj)

        assert "type" in result
        assert "description" in result
        assert "example" not in result
        assert "x-doc" not in result
        assert "examples" not in result["properties"]["field"]

    def test_remove_custom_fields(self):
        """Test removal of custom specified fields."""
        obj = {"keep": "this", "remove": "this", "nested": {"remove": "this too", "keep": "this"}}

        result = remove_verbose_fields(obj, fields_to_remove={"remove"})

        assert "keep" in result
        assert "remove" not in result
        assert "keep" in result["nested"]
        assert "remove" not in result["nested"]

    def test_handle_lists(self):
        """Test handling of lists in objects."""
        obj = {
            "items": [
                {"keep": "this", "example": "remove"},
                {"keep": "this too", "x-doc": "remove"},
            ]
        }

        result = remove_verbose_fields(obj)

        assert len(result["items"]) == 2
        assert "keep" in result["items"][0]
        assert "example" not in result["items"][0]
        assert "x-doc" not in result["items"][1]

    def test_handle_primitives(self):
        """Test handling of primitive values."""
        assert remove_verbose_fields("string") == "string"
        assert remove_verbose_fields(123) == 123
        assert remove_verbose_fields(True) is True


class TestMinimizeSchemaObject:
    """Tests for minimize_schema_object function."""

    def test_keep_essential_fields(self):
        """Test that essential fields are kept."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "description": "A test schema",
            "example": {"id": "123"},  # Should be removed by other processing
        }

        result = minimize_schema_object(schema)

        assert result["type"] == "object"
        assert "properties" in result
        assert "required" in result
        assert "description" in result

    def test_minimize_nested_properties(self):
        """Test minimization of nested properties."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "description": "Very long description that should be shortened",
                    "properties": {"name": {"type": "string", "description": "User name"}},
                }
            },
        }

        result = minimize_schema_object(schema)

        assert "properties" in result
        assert "user" in result["properties"]
        # Nested properties should also be minimized
        assert "properties" in result["properties"]["user"]

    def test_handle_array_items(self):
        """Test handling of array item schemas."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "description": "Array item description",
            },
        }

        result = minimize_schema_object(schema)

        assert result["type"] == "array"
        assert "items" in result
        assert result["items"]["type"] == "object"


class TestMinimizeOperation:
    """Tests for minimize_operation function."""

    def test_keep_essential_operation_fields(self):
        """Test that essential operation fields are kept."""
        operation = {
            "tags": ["Users"],
            "operationId": "getUser",
            "summary": "Get a user",
            "description": "Retrieve user information by ID",
            "parameters": [],
            "responses": {"200": {"description": "Success"}},
            "deprecated": True,  # Should be removed
            "x-custom": "remove",  # Should be removed
        }

        result = minimize_operation(operation)

        assert result["tags"] == ["Users"]
        assert result["operationId"] == "getUser"
        assert "summary" in result
        assert "description" in result
        assert "parameters" in result
        assert "responses" in result

    def test_minimize_parameters(self):
        """Test minimization of operation parameters."""
        operation = {
            "operationId": "test",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "type": "string",
                    "description": "The user ID parameter with a very long description",
                }
            ],
        }

        result = minimize_operation(operation)

        assert len(result["parameters"]) == 1
        param = result["parameters"][0]
        assert param["name"] == "id"
        assert param["in"] == "path"
        assert param["required"] is True
        assert param["type"] == "string"

    def test_minimize_responses(self):
        """Test minimization of operation responses."""
        operation = {
            "operationId": "test",
            "responses": {
                "200": {
                    "description": "Success response with detailed explanation",
                    "schema": {"type": "object", "properties": {"id": {"type": "string"}}},
                }
            },
        }

        result = minimize_operation(operation)

        assert "200" in result["responses"]
        response = result["responses"]["200"]
        assert "description" in response
        assert "schema" in response


class TestMinimizeSpec:
    """Tests for the full spec minimization."""

    @pytest.fixture
    def sample_spec(self):
        """Sample OpenAPI spec for testing."""
        return {
            "swagger": "2.0",
            "info": {
                "title": "Test API",
                "version": "1.0",
                "description": "This is a very detailed description of the test API that explains everything in great detail",
            },
            "host": "api.example.com",
            "tags": [
                {
                    "name": "Users",
                    "description": "Everything about users including CRUD operations and authentication",
                }
            ],
            "paths": {
                "/users": {
                    "get": {
                        "tags": ["Users"],
                        "operationId": "listUsers",
                        "summary": "List all users in the system",
                        "description": "This endpoint returns a paginated list of all users with detailed information",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "type": "integer",
                                "description": "The maximum number of users to return in a single page",
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successfully retrieved the list of users",
                                "schema": {"$ref": "#/definitions/UserList"},
                            }
                        },
                        "x-doc": "Additional documentation",
                    }
                }
            },
            "definitions": {
                "User": {
                    "type": "object",
                    "description": "A user object containing all user information including personal details",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The unique identifier for the user",
                        },
                        "name": {"type": "string", "description": "The full name of the user"},
                    },
                    "example": {"id": "123", "name": "John Doe"},
                },
                "UserList": {
                    "type": "object",
                    "properties": {
                        "users": {"type": "array", "items": {"$ref": "#/definitions/User"}}
                    },
                },
            },
        }

    def test_minimize_full_spec(self, sample_spec):
        """Test minimization of a full spec."""
        result = minimize_spec(sample_spec)

        # Basic structure should be preserved
        assert result["swagger"] == "2.0"
        assert result["info"]["title"] == "Test API"
        assert result["host"] == "api.example.com"

        # Paths should be minimized
        assert "/users" in result["paths"]
        get_op = result["paths"]["/users"]["get"]
        assert get_op["operationId"] == "listUsers"
        assert "x-doc" not in get_op  # Verbose field removed

        # Definitions should be minimized
        assert "User" in result["definitions"]
        user_def = result["definitions"]["User"]
        assert user_def["type"] == "object"
        assert "required" in user_def
        assert "example" not in user_def  # Verbose field removed

    def test_preserve_descriptions_when_disabled(self, sample_spec):
        """Test that descriptions are preserved when minimization is disabled."""
        result = minimize_spec(sample_spec, minimize_descriptions=False)

        # Original descriptions should be preserved
        original_info_desc = sample_spec["info"]["description"]
        assert result["info"]["description"] == original_info_desc

    def test_keep_verbose_fields_when_disabled(self, sample_spec):
        """Test that verbose fields are kept when removal is disabled."""
        result = minimize_spec(
            sample_spec, remove_verbose_fields_flag=False, minimize_descriptions=False
        )

        # Verbose fields should still be present when both flags are disabled
        get_op = result["paths"]["/users"]["get"]
        assert "x-doc" in get_op  # Should be preserved
        user_def = result["definitions"]["User"]
        assert "example" in user_def  # Should be preserved


class TestCalculateSizeReduction:
    """Tests for size reduction calculation."""

    def test_calculate_size_reduction(self):
        """Test size reduction calculation."""
        original = {
            "field1": "value1",
            "field2": "value2",
            "verbose": "This is a very long verbose description that takes up space",
        }

        minimized = {"field1": "value1", "field2": "value2"}

        metrics = calculate_size_reduction(original, minimized)

        assert metrics["original_size"] > metrics["minimized_size"]
        assert metrics["reduction_bytes"] > 0
        assert metrics["reduction_percent"] > 0
        assert metrics["reduction_percent"] < 100

    def test_no_reduction(self):
        """Test when there's no size reduction."""
        spec = {"field": "value"}
        metrics = calculate_size_reduction(spec, spec)

        assert metrics["original_size"] == metrics["minimized_size"]
        assert metrics["reduction_bytes"] == 0
        assert metrics["reduction_percent"] == 0

    def test_empty_specs(self):
        """Test with empty specs."""
        metrics = calculate_size_reduction({}, {})

        assert metrics["original_size"] == metrics["minimized_size"]
        assert metrics["reduction_bytes"] == 0
        assert metrics["reduction_percent"] == 0
