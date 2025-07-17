"""
Utilities for OpenAPI specification handling and intent classification.
"""

import json
from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel


def extract_tags_from_spec(api_spec: Dict[str, Any]) -> List[str]:
    """
    Extract all available tags from the OpenAPI specification.

    Args:
        api_spec: The OpenAPI specification dictionary

    Returns:
        List of all available tags/categories in the API spec
    """
    tag_set = set()

    # Try to get tags from the top-level 'tags' section first
    if "tags" in api_spec:
        for tag_info in api_spec["tags"]:
            if isinstance(tag_info, dict) and "name" in tag_info:
                tag_set.add(tag_info["name"])
            elif isinstance(tag_info, str):
                tag_set.add(tag_info)

    # If no tags found or we want additional tags, extract from operations
    if not tag_set or True:  # Always extract from operations for completeness
        for path_ops in api_spec.get("paths", {}).values():
            for operation in path_ops.values():
                if isinstance(operation, dict) and "tags" in operation:
                    tag_set.update(operation["tags"])

    return sorted(list(tag_set))


def classify_intent(query: str, available_tags: List[str], llm: BaseLanguageModel) -> List[str]:
    """
    Classify user intent to determine relevant API categories.

    Args:
        query: User's question or request
        available_tags: List of available OpenAPI tags
        llm: Language model for intent classification

    Returns:
        List of relevant OpenAPI tags/categories
    """
    # Create a prompt for intent classification
    classification_prompt = f"""
Analyse the following user query and determine which Elastic Cloud API categories are most relevant.

Available categories:
{', '.join(available_tags)}

User query: "{query}"

Respond with 1-3 most relevant categories from the list above, comma-separated.
If the query is general or unclear, respond with: Deployments, Accounts

Categories:"""

    try:
        # Use the LLM to classify intent
        response = llm.invoke(classification_prompt)

        # Extract categories from response
        if hasattr(response, "content"):
            categories_text = str(response.content).strip()
        else:
            categories_text = str(response).strip()

        # Parse the categories
        suggested_categories = [cat.strip() for cat in categories_text.split(",")]

        # Filter to only valid categories
        relevant_categories = [cat for cat in suggested_categories if cat in available_tags]

        # Default fallback if no valid categories found
        if not relevant_categories:
            relevant_categories = ["Deployments", "Accounts"]

        return relevant_categories[:3]  # Limit to 3 categories max

    except Exception:
        # Fallback to default categories if classification fails
        return ["Deployments", "Accounts"]


def filter_spec_by_tags(api_spec: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
    """
    Filter the OpenAPI specification to include only specified tags.

    Args:
        api_spec: The full OpenAPI specification
        tags: List of OpenAPI tags to include

    Returns:
        Filtered OpenAPI specification
    """
    # Create a copy of the original spec
    filtered_spec = json.loads(json.dumps(api_spec))

    # Filter paths to only include operations with specified tags
    filtered_paths = {}

    for path, operations in api_spec.get("paths", {}).items():
        filtered_operations = {}

        for method, operation in operations.items():
            if isinstance(operation, dict):
                operation_tags = operation.get("tags", [])

                # Include operation if it has any of the specified tags
                if any(tag in operation_tags for tag in tags):
                    filtered_operations[method] = operation

        # Include path if it has any relevant operations
        if filtered_operations:
            filtered_paths[path] = filtered_operations

    # Update the filtered spec
    filtered_spec["paths"] = filtered_paths

    return filtered_spec
