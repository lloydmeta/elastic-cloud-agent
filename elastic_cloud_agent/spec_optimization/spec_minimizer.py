"""
Schema minimizer for reducing OpenAPI spec size by removing verbose content.
"""

from typing import Any, Dict, List, Optional, Set, Union

# Type aliases for OpenAPI schema objects
SchemaValue = Union[str, int, float, bool, None]
SchemaObject = Union[Dict[str, Any], SchemaValue, List[Any]]


def minimize_description(description: str, max_length: int = 100) -> str:
    """
    Minimize a description by truncating and cleaning it.

    Args:
        description: The original description
        max_length: Maximum length to keep

    Returns:
        Minimized description
    """
    if not description or not isinstance(description, str):
        return ""

    # Remove extra whitespace and newlines
    cleaned = " ".join(description.strip().split())

    # Truncate if too long
    if len(cleaned) > max_length:
        # Try to truncate at sentence boundary
        sentences = cleaned.split(".")
        result = ""
        for i, sentence in enumerate(sentences):
            if i == len(sentences) - 1 and not sentence.strip():
                # Skip empty last element from split
                continue
            test_addition = sentence + "." if sentence.strip() else ""
            if len(result + test_addition) <= max_length:
                result += test_addition
            else:
                break

        # If no sentence fits, just truncate
        if not result and len(cleaned) > max_length:
            result = cleaned[: max_length - 3] + "..."

        return result.strip()

    return cleaned


def remove_verbose_fields(obj: Any, fields_to_remove: Optional[Set[str]] = None) -> Any:
    """
    Remove verbose fields from OpenAPI spec objects.

    Args:
        obj: The object to process (dict, list, or primitive)
        fields_to_remove: Set of field names to remove

    Returns:
        Object with verbose fields removed
    """
    if fields_to_remove is None:
        fields_to_remove = {
            "example",
            "examples",
            "externalDocs",
            "x-doc",
            "x-examples",
            "x-description",
            "deprecationDate",
            "deprecated",
        }

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key not in fields_to_remove:
                result[key] = remove_verbose_fields(value, fields_to_remove)
        return result
    elif isinstance(obj, list):
        return [remove_verbose_fields(item, fields_to_remove) for item in obj]
    else:
        return obj


def minimize_schema_definitions(
    definitions: Dict[str, Any], keep_required: bool = True
) -> Dict[str, Any]:
    """
    Minimize schema definitions by removing verbose content.

    Args:
        definitions: The definitions section of an OpenAPI spec
        keep_required: Whether to keep required field lists

    Returns:
        Minimized definitions
    """
    minimized = {}

    for name, definition in definitions.items():
        if not isinstance(definition, dict):
            minimized[name] = definition
            continue

        mini_def = {}

        # Always keep essential structure fields
        for key in ["type", "properties", "$ref", "allOf", "oneOf", "anyOf", "items"]:
            if key in definition:
                mini_def[key] = definition[key]

        # Keep required if specified
        if keep_required and "required" in definition:
            mini_def["required"] = definition["required"]

        # Minimize description
        if "description" in definition:
            minimized_desc = minimize_description(definition["description"])
            if minimized_desc:
                mini_def["description"] = minimized_desc

        # Recursively process properties
        if "properties" in mini_def and isinstance(mini_def["properties"], dict):
            mini_def["properties"] = minimize_properties(mini_def["properties"])

        # Recursively process items for arrays
        if "items" in mini_def:
            mini_def["items"] = minimize_schema_object(mini_def["items"])

        minimized[name] = mini_def

    return minimized


def minimize_properties(properties: Dict[str, Any]) -> Dict[str, SchemaObject]:
    """
    Minimize schema properties by removing verbose content.

    Args:
        properties: Properties dict from a schema

    Returns:
        Minimized properties
    """
    minimized: Dict[str, SchemaObject] = {}

    for prop_name, prop_def in properties.items():
        minimized[prop_name] = minimize_schema_object(prop_def)

    return minimized


def minimize_schema_object(schema_obj: SchemaObject) -> SchemaObject:
    """
    Minimize a single schema object.

    Args:
        schema_obj: Schema object to minimize

    Returns:
        Minimized schema object
    """
    if not isinstance(schema_obj, dict):
        return schema_obj

    minimized: Dict[str, Any] = {}

    # Keep essential fields
    essential_fields = {
        "type",
        "$ref",
        "format",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "items",
        "properties",
        "allOf",
        "oneOf",
        "anyOf",
        "required",
        "additionalProperties",
    }

    for key, value in schema_obj.items():
        if key in essential_fields:
            if key == "properties" and isinstance(value, dict):
                minimized[key] = minimize_properties(value)
            elif key == "items":
                minimized[key] = minimize_schema_object(value)
            else:
                minimized[key] = value
        elif key == "description" and isinstance(value, str):
            minimized_desc = minimize_description(value)
            if minimized_desc:
                minimized[key] = minimized_desc  # type: ignore[assignment]

    return minimized


def minimize_operation(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimize an OpenAPI operation by removing verbose content.

    Args:
        operation: The operation object to minimize

    Returns:
        Minimized operation
    """
    minimized = {}

    # Keep essential operation fields
    essential_fields = {
        "tags",
        "operationId",
        "parameters",
        "responses",
        "security",
        "consumes",
        "produces",
        "schemes",
    }

    for key, value in operation.items():
        if key in essential_fields:
            if key == "parameters" and isinstance(value, list):
                minimized[key] = [minimize_parameter(param) for param in value]
            elif key == "responses" and isinstance(value, dict):
                minimized[key] = minimize_responses(value)  # type: ignore[assignment]
            else:
                minimized[key] = value
        elif key in ["summary", "description"] and isinstance(value, str):
            minimized_desc = minimize_description(value, max_length=50 if key == "summary" else 100)
            if minimized_desc:
                minimized[key] = minimized_desc  # type: ignore[assignment]

    return minimized


def minimize_parameter(parameter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimize a parameter object.

    Args:
        parameter: Parameter object to minimize

    Returns:
        Minimized parameter
    """
    minimized = {}

    # Keep essential parameter fields
    essential_fields = {
        "name",
        "in",
        "required",
        "type",
        "format",
        "schema",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
    }

    for key, value in parameter.items():
        if key in essential_fields:
            if key == "schema":
                minimized[key] = minimize_schema_object(value)
            else:
                minimized[key] = value
        elif key == "description" and isinstance(value, str):
            minimized_desc = minimize_description(value, max_length=50)
            if minimized_desc:
                minimized[key] = minimized_desc

    return minimized


def minimize_responses(responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimize responses object.

    Args:
        responses: Responses dict from operation

    Returns:
        Minimized responses
    """
    minimized = {}

    for status_code, response in responses.items():
        if isinstance(response, dict):
            mini_response = {}

            # Keep essential response fields
            for key in ["schema", "headers"]:
                if key in response:
                    if key == "schema":
                        mini_response[key] = minimize_schema_object(response[key])
                    else:
                        mini_response[key] = response[key]

            # Minimize description
            if "description" in response and isinstance(response["description"], str):
                minimized_desc = minimize_description(response["description"], max_length=50)
                if minimized_desc:
                    mini_response["description"] = minimized_desc

            minimized[status_code] = mini_response
        else:
            minimized[status_code] = response

    return minimized


def minimize_spec(
    spec: Dict[str, Any],
    remove_verbose_fields_flag: bool = True,
    minimize_descriptions: bool = True,
) -> Dict[str, Any]:
    """
    Minimize an entire OpenAPI specification.

    Args:
        spec: The OpenAPI specification to minimize
        remove_verbose_fields_flag: Whether to remove verbose fields like examples
        minimize_descriptions: Whether to minimize descriptions

    Returns:
        Minimized OpenAPI specification
    """
    minimized = {}

    # Copy basic metadata
    for key in ["swagger", "info", "host", "basePath", "schemes", "security", "tags"]:
        if key in spec:
            if key == "info" and isinstance(spec[key], dict):
                # Minimize info descriptions
                info = dict(spec[key])
                if (
                    minimize_descriptions
                    and "description" in info
                    and isinstance(info["description"], str)
                ):
                    info["description"] = minimize_description(info["description"], max_length=100)
                minimized[key] = info
            elif key == "tags" and isinstance(spec[key], list):
                # Minimize tag descriptions
                tags = []
                for tag in spec[key]:
                    if isinstance(tag, dict):
                        mini_tag = dict(tag)
                        if (
                            minimize_descriptions
                            and "description" in mini_tag
                            and isinstance(mini_tag["description"], str)
                        ):
                            mini_tag["description"] = minimize_description(
                                mini_tag["description"], max_length=50
                            )
                        tags.append(mini_tag)
                    else:
                        tags.append(tag)
                minimized[key] = tags  # type: ignore[assignment]
            else:
                minimized[key] = spec[key]

    # Minimize paths
    if "paths" in spec and isinstance(spec["paths"], dict):
        minimized_paths = {}
        for path, path_obj in spec["paths"].items():
            if isinstance(path_obj, dict):
                minimized_path = {}
                for method, operation in path_obj.items():
                    if isinstance(operation, dict):
                        if minimize_descriptions:
                            minimized_path[method] = minimize_operation(operation)
                        else:
                            # Just copy operation without description minimization
                            minimized_path[method] = dict(operation)
                    else:
                        minimized_path[method] = operation
                minimized_paths[path] = minimized_path
            else:
                minimized_paths[path] = path_obj
        minimized["paths"] = minimized_paths

    # Minimize definitions
    if "definitions" in spec and isinstance(spec["definitions"], dict):
        if minimize_descriptions:
            minimized["definitions"] = minimize_schema_definitions(spec["definitions"])
        else:
            # Just copy definitions without minimization
            minimized["definitions"] = dict(spec["definitions"])

    # Remove verbose fields if requested
    if remove_verbose_fields_flag:
        minimized = remove_verbose_fields(minimized)

    return minimized


def calculate_size_reduction(
    original: Dict[str, Any], minimized: Dict[str, Any]
) -> Dict[str, Union[int, float]]:
    """
    Calculate the size reduction metrics.

    Args:
        original: Original spec
        minimized: Minimized spec

    Returns:
        Dictionary with size metrics
    """
    import json

    original_str = json.dumps(original, separators=(",", ":"))
    minimized_str = json.dumps(minimized, separators=(",", ":"))

    original_size = len(original_str)
    minimized_size = len(minimized_str)
    reduction_bytes = original_size - minimized_size
    reduction_percent = (reduction_bytes / original_size) * 100 if original_size > 0 else 0

    return {
        "original_size": original_size,
        "minimized_size": minimized_size,
        "reduction_bytes": reduction_bytes,
        "reduction_percent": reduction_percent,
    }
