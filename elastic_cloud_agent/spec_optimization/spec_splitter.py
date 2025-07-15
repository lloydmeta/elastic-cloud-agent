"""
Spec splitter for breaking down large OpenAPI specs by tags.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def extract_tags_from_spec(spec: Dict[str, Any]) -> Set[str]:
    """
    Extract all unique tags from an OpenAPI specification.

    Args:
        spec: The OpenAPI specification dictionary

    Returns:
        Set of unique tag names found in the spec
    """
    tags = set()

    # Extract from global tags definition
    if "tags" in spec:
        for tag in spec["tags"]:
            if isinstance(tag, dict) and "name" in tag:
                tags.add(tag["name"])

    # Extract from operation tags
    if "paths" in spec:
        for path_methods in spec["paths"].values():
            if isinstance(path_methods, dict):
                for method_data in path_methods.values():
                    if isinstance(method_data, dict) and "tags" in method_data:
                        tags.update(method_data["tags"])

    return tags


def get_operations_by_tag(spec: Dict[str, Any], target_tag: str) -> Dict[str, Any]:
    """
    Extract all operations for a specific tag from the spec.

    Args:
        spec: The OpenAPI specification dictionary
        target_tag: The tag to filter operations by

    Returns:
        Dictionary containing paths with operations matching the target tag
    """
    filtered_paths: Dict[str, Any] = {}

    if "paths" not in spec:
        return filtered_paths

    for path, path_methods in spec["paths"].items():
        if not isinstance(path_methods, dict):
            continue

        filtered_methods = {}
        for method, method_data in path_methods.items():
            if not isinstance(method_data, dict):
                continue

            # Check if this operation has the target tag
            operation_tags = method_data.get("tags", [])
            if target_tag in operation_tags:
                filtered_methods[method] = method_data

        # Only include the path if it has matching operations
        if filtered_methods:
            filtered_paths[path] = filtered_methods

    return filtered_paths


def get_referenced_definitions(spec: Dict[str, Any], operations: Dict[str, Any]) -> Set[str]:
    """
    Extract all schema definitions referenced by the given operations.

    Args:
        spec: The full OpenAPI specification
        operations: The filtered operations to analyse

    Returns:
        Set of definition names that are referenced
    """
    referenced_defs = set()

    def extract_refs_from_obj(obj: Any) -> None:
        """Recursively extract $ref references from an object."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/definitions/"):
                    def_name = ref.replace("#/definitions/", "")
                    referenced_defs.add(def_name)
            else:
                for value in obj.values():
                    extract_refs_from_obj(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_refs_from_obj(item)

    # Extract references from the operations
    extract_refs_from_obj(operations)

    # Recursively find referenced definitions (definitions can reference other definitions)
    def find_transitive_refs(def_names: Set[str]) -> Set[str]:
        """Find all transitively referenced definitions."""
        all_refs = set(def_names)
        definitions = spec.get("definitions", {})

        # Keep adding references until we find no new ones
        while True:
            for def_name in all_refs:
                if def_name in definitions:
                    extract_refs_from_obj(definitions[def_name])

            # Check if we found any new references
            current_new = referenced_defs - all_refs
            if not current_new:
                break
            all_refs.update(current_new)

        return all_refs

    return find_transitive_refs(referenced_defs)


def create_tag_spec(spec: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """
    Create a new OpenAPI spec containing only operations for a specific tag.

    Args:
        spec: The original OpenAPI specification
        tag: The tag to filter by

    Returns:
        A new OpenAPI spec containing only the specified tag's operations
    """
    # Start with the base spec structure
    tag_spec = {
        "swagger": spec.get("swagger"),
        "info": spec.get("info", {}),
        "host": spec.get("host"),
        "basePath": spec.get("basePath"),
        "schemes": spec.get("schemes", []),
        "security": spec.get("security", []),
    }

    # Add only the target tag to the tags list
    if "tags" in spec:
        tag_spec["tags"] = [t for t in spec["tags"] if t.get("name") == tag]

    # Get operations for this tag
    operations = get_operations_by_tag(spec, tag)
    tag_spec["paths"] = operations

    # Get referenced definitions
    referenced_defs = get_referenced_definitions(spec, operations)

    # Add only the referenced definitions
    if referenced_defs and "definitions" in spec:
        tag_spec["definitions"] = {
            name: definition
            for name, definition in spec["definitions"].items()
            if name in referenced_defs
        }

    return tag_spec


def split_spec_by_tags(
    spec: Dict[str, Any], output_dir: Optional[Path] = None, tag_filter: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Split an OpenAPI specification into separate specs by tags.

    Args:
        spec: The OpenAPI specification to split
        output_dir: Optional directory to write split specs to
        tag_filter: Optional list of tags to include (if None, includes all tags)

    Returns:
        Dictionary mapping tag names to their respective specs
    """
    available_tags = extract_tags_from_spec(spec)

    # Filter tags if specified
    if tag_filter:
        tags_to_process = set(tag_filter) & available_tags
    else:
        tags_to_process = available_tags

    tag_specs = {}

    for tag in tags_to_process:
        tag_spec = create_tag_spec(spec, tag)
        tag_specs[tag] = tag_spec

        # Write to file if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{tag.lower()}_spec.json"
            with open(output_file, "w") as f:
                json.dump(tag_spec, f, indent=2)

    return tag_specs


def load_spec_from_file(spec_path: Path) -> Dict[str, Any]:
    """
    Load an OpenAPI specification from a file.

    Args:
        spec_path: Path to the OpenAPI specification file

    Returns:
        The loaded OpenAPI specification

    Raises:
        FileNotFoundError: If the spec file doesn't exist
        ValueError: If the file format is not supported
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with open(spec_path, "r") as f:
        if spec_path.suffix == ".json":
            return json.load(f)
        elif spec_path.suffix in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {spec_path.suffix}")
