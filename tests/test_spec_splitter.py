"""
Tests for the spec splitter module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from elastic_cloud_agent.spec_optimization.spec_splitter import (
    create_tag_spec,
    extract_tags_from_spec,
    get_operations_by_tag,
    get_referenced_definitions,
    load_spec_from_file,
    split_spec_by_tags,
)


@pytest.fixture
def sample_spec():
    """Create a sample OpenAPI spec for testing."""
    return {
        "swagger": "2.0",
        "info": {"version": "1", "title": "Test API"},
        "host": "api.example.com",
        "basePath": "/api/v1",
        "schemes": ["https"],
        "security": [{"apiKey": []}],
        "tags": [
            {"name": "Users", "description": "User operations"},
            {"name": "Orders", "description": "Order operations"},
        ],
        "paths": {
            "/users": {
                "get": {
                    "tags": ["Users"],
                    "summary": "List users",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "schema": {"$ref": "#/definitions/UserList"},
                        }
                    },
                }
            },
            "/users/{id}": {
                "get": {
                    "tags": ["Users"],
                    "summary": "Get user",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "schema": {"$ref": "#/definitions/User"},
                        }
                    },
                }
            },
            "/orders": {
                "get": {
                    "tags": ["Orders"],
                    "summary": "List orders",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "schema": {"$ref": "#/definitions/OrderList"},
                        }
                    },
                },
                "post": {
                    "tags": ["Orders"],
                    "summary": "Create order",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "schema": {"$ref": "#/definitions/CreateOrderRequest"},
                        }
                    ],
                    "responses": {
                        "201": {
                            "description": "Created",
                            "schema": {"$ref": "#/definitions/Order"},
                        }
                    },
                },
            },
        },
        "definitions": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "profile": {"$ref": "#/definitions/UserProfile"},
                },
            },
            "UserProfile": {
                "type": "object",
                "properties": {"bio": {"type": "string"}},
            },
            "UserList": {
                "type": "object",
                "properties": {"users": {"type": "array", "items": {"$ref": "#/definitions/User"}}},
            },
            "Order": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "items": {"type": "array", "items": {"$ref": "#/definitions/OrderItem"}},
                },
            },
            "OrderItem": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
            },
            "OrderList": {
                "type": "object",
                "properties": {
                    "orders": {"type": "array", "items": {"$ref": "#/definitions/Order"}}
                },
            },
            "CreateOrderRequest": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"$ref": "#/definitions/OrderItem"}}
                },
            },
        },
    }


def test_extract_tags_from_spec(sample_spec):
    """Test extracting tags from spec."""
    tags = extract_tags_from_spec(sample_spec)
    assert tags == {"Users", "Orders"}


def test_extract_tags_from_spec_no_tags():
    """Test extracting tags when none exist."""
    spec = {"paths": {}}
    tags = extract_tags_from_spec(spec)
    assert tags == set()


def test_extract_tags_from_spec_operations_only():
    """Test extracting tags from operations when no global tags."""
    spec = {
        "paths": {
            "/test": {
                "get": {"tags": ["TestTag"]},
                "post": {"tags": ["TestTag", "AnotherTag"]},
            }
        }
    }
    tags = extract_tags_from_spec(spec)
    assert tags == {"TestTag", "AnotherTag"}


def test_get_operations_by_tag(sample_spec):
    """Test filtering operations by tag."""
    user_operations = get_operations_by_tag(sample_spec, "Users")

    assert "/users" in user_operations
    assert "/users/{id}" in user_operations
    assert "/orders" not in user_operations

    assert "get" in user_operations["/users"]
    assert "get" in user_operations["/users/{id}"]


def test_get_operations_by_tag_nonexistent():
    """Test filtering operations by non-existent tag."""
    sample_spec = {"paths": {"/test": {"get": {"tags": ["Other"]}}}}
    operations = get_operations_by_tag(sample_spec, "NonExistent")
    assert operations == {}


def test_get_referenced_definitions(sample_spec):
    """Test extracting referenced definitions."""
    user_operations = get_operations_by_tag(sample_spec, "Users")
    referenced_defs = get_referenced_definitions(sample_spec, user_operations)

    # Should include User, UserList, and transitively UserProfile
    expected_defs = {"User", "UserList", "UserProfile"}
    assert referenced_defs == expected_defs


def test_get_referenced_definitions_transitive(sample_spec):
    """Test that transitive references are included."""
    order_operations = get_operations_by_tag(sample_spec, "Orders")
    referenced_defs = get_referenced_definitions(sample_spec, order_operations)

    # Should include Order, OrderList, CreateOrderRequest, and transitively OrderItem
    expected_defs = {"Order", "OrderList", "CreateOrderRequest", "OrderItem"}
    assert referenced_defs == expected_defs


def test_create_tag_spec(sample_spec):
    """Test creating a spec for a specific tag."""
    user_spec = create_tag_spec(sample_spec, "Users")

    # Check basic structure is preserved
    assert user_spec["swagger"] == "2.0"
    assert user_spec["info"]["title"] == "Test API"
    assert user_spec["host"] == "api.example.com"

    # Check only Users tag is included
    assert len(user_spec["tags"]) == 1
    assert user_spec["tags"][0]["name"] == "Users"

    # Check only user operations are included
    assert "/users" in user_spec["paths"]
    assert "/users/{id}" in user_spec["paths"]
    assert "/orders" not in user_spec["paths"]

    # Check only referenced definitions are included
    assert "User" in user_spec["definitions"]
    assert "UserList" in user_spec["definitions"]
    assert "UserProfile" in user_spec["definitions"]
    assert "Order" not in user_spec["definitions"]
    assert "OrderList" not in user_spec["definitions"]


def test_split_spec_by_tags(sample_spec):
    """Test splitting spec by all tags."""
    tag_specs = split_spec_by_tags(sample_spec)

    assert "Users" in tag_specs
    assert "Orders" in tag_specs

    # Verify Users spec
    user_spec = tag_specs["Users"]
    assert "/users" in user_spec["paths"]
    assert "/orders" not in user_spec["paths"]
    assert "User" in user_spec["definitions"]
    assert "Order" not in user_spec["definitions"]

    # Verify Orders spec
    order_spec = tag_specs["Orders"]
    assert "/orders" in order_spec["paths"]
    assert "/users" not in order_spec["paths"]
    assert "Order" in order_spec["definitions"]
    assert "User" not in order_spec["definitions"]


def test_split_spec_by_tags_with_filter(sample_spec):
    """Test splitting spec with tag filter."""
    tag_specs = split_spec_by_tags(sample_spec, tag_filter=["Users"])

    assert "Users" in tag_specs
    assert "Orders" not in tag_specs


def test_split_spec_by_tags_with_output_dir(sample_spec):
    """Test splitting spec and writing to files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        split_spec_by_tags(sample_spec, output_dir=output_dir)

        # Check files were created
        users_file = output_dir / "users_spec.json"
        orders_file = output_dir / "orders_spec.json"

        assert users_file.exists()
        assert orders_file.exists()

        # Verify file contents
        with open(users_file) as f:
            users_spec = json.load(f)
        assert "/users" in users_spec["paths"]
        assert "/orders" not in users_spec["paths"]


def test_load_spec_from_file():
    """Test loading spec from JSON file."""
    sample_spec = {"swagger": "2.0", "info": {"title": "Test"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_spec, f)
        temp_path = Path(f.name)

    try:
        loaded_spec = load_spec_from_file(temp_path)
        assert loaded_spec == sample_spec
    finally:
        temp_path.unlink()


def test_load_spec_from_file_not_found():
    """Test loading spec from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_spec_from_file(Path("/nonexistent/file.json"))


def test_load_spec_from_file_unsupported_format():
    """Test loading spec from unsupported file format."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        temp_path = Path(f.name)
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_spec_from_file(temp_path)


@patch("yaml.safe_load")
def test_load_spec_from_yaml_file(mock_safe_load):
    """Test loading spec from YAML file."""
    sample_spec = {"swagger": "2.0", "info": {"title": "Test"}}
    mock_safe_load.return_value = sample_spec

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("dummy yaml content")
        temp_path = Path(f.name)

    try:
        loaded_spec = load_spec_from_file(temp_path)
        assert loaded_spec == sample_spec
        mock_safe_load.assert_called_once()
    finally:
        temp_path.unlink()


def test_empty_spec_handling():
    """Test handling of empty or minimal specs."""
    empty_spec = {"swagger": "2.0", "info": {"title": "Empty"}}

    tags = extract_tags_from_spec(empty_spec)
    assert tags == set()

    operations = get_operations_by_tag(empty_spec, "NonExistent")
    assert operations == {}

    tag_specs = split_spec_by_tags(empty_spec)
    assert tag_specs == {}


def test_malformed_spec_handling():
    """Test handling of malformed spec data."""
    malformed_spec = {
        "swagger": "2.0",
        "paths": {
            "/test": {
                "get": "invalid_operation_data",  # Should be dict
                "post": {"tags": ["Valid"]},
            }
        },
    }

    # Should handle malformed data gracefully
    tags = extract_tags_from_spec(malformed_spec)
    assert tags == {"Valid"}

    operations = get_operations_by_tag(malformed_spec, "Valid")
    assert "/test" in operations
    assert "post" in operations["/test"]
    assert "get" not in operations["/test"]
