"""
Operation filter for categorizing OpenAPI operations by complexity/usage.
"""

from typing import Dict, List, Optional, Set, Union

# Core operations that are commonly used
CORE_OPERATION_PATTERNS = {
    # Account/User operations
    "get-current-account",
    "update-current-account",
    # Deployment operations - most common
    "get-deployment",
    "create-deployment",
    "update-deployment",
    "delete-deployment",
    "list-deployments",
    "restart-deployment",
    "start-deployment",
    "stop-deployment",
    # Elasticsearch cluster operations
    "get-elasticsearch-cluster",
    "update-elasticsearch-cluster",
    "restart-elasticsearch-cluster",
    "start-elasticsearch-cluster",
    "stop-elasticsearch-cluster",
    # Kibana operations
    "get-kibana-cluster",
    "update-kibana-cluster",
    "restart-kibana-cluster",
    # Basic monitoring
    "get-deployment-health",
    "get-deployment-logs",
    # Authentication
    "get-api-keys",
    "create-api-key",
    "delete-api-key",
}

# Tags that typically contain core operations
CORE_TAGS = {
    "Deployments",
    "Accounts",
    "Authentication",
}

# HTTP methods by complexity
SIMPLE_METHODS = {"GET"}
MODERATE_METHODS = {"POST", "PUT", "PATCH"}
COMPLEX_METHODS = {"DELETE"}

# Operations that are typically for advanced users
ADVANCED_OPERATION_PATTERNS = {
    # Advanced deployment configuration
    "deploy-deployment-template",
    "validate-deployment-template",
    "export-deployment-template",
    # Advanced cluster management
    "upgrade-elasticsearch-cluster",
    "cancel-elasticsearch-cluster-pending-plan",
    "force-restart-elasticsearch-cluster",
    # Traffic filtering (advanced networking)
    "create-traffic-filter-ruleset",
    "update-traffic-filter-ruleset",
    "delete-traffic-filter-ruleset",
    "associate-traffic-filter-ruleset",
    # Extensions and plugins
    "upload-extension",
    "delete-extension",
    "update-extension",
    # Organization management
    "create-organization",
    "delete-organization",
    "update-organization",
    "get-organization-invitations",
    "create-organization-invitation",
    # Advanced IAM
    "create-role-mapping",
    "update-role-mapping",
    "delete-role-mapping",
    # Billing and costs
    "get-costs-overview",
    "get-itemized-costs",
    "get-deployment-costs",
    # Platform management
    "get-platform-info",
    "get-platform-configuration",
}

# Tags that typically contain advanced operations
ADVANCED_TAGS = {
    "BillingCostsAnalysis",
    "DeploymentsTrafficFilter",
    "Extensions",
    "Organizations",
    "IamService",
    "TrustedEnvironments",
    "UserRoleAssignments",
    "Stack",
}


def classify_operation_complexity(
    operation_id: str, tags: List[str], method: str, path: str, summary: Optional[str] = None
) -> str:
    """
    Classify an operation as 'core', 'moderate', or 'advanced'.

    Args:
        operation_id: The operationId from the OpenAPI spec
        tags: List of tags associated with the operation
        method: HTTP method (GET, POST, etc.)
        path: API path
        summary: Optional operation summary

    Returns:
        Classification: 'core', 'moderate', or 'advanced'
    """
    # Check explicit core patterns first
    if operation_id in CORE_OPERATION_PATTERNS:
        return "core"

    # Check explicit advanced patterns
    if operation_id in ADVANCED_OPERATION_PATTERNS:
        return "advanced"

    # Check tags
    tag_set = set(tags) if tags else set()
    if tag_set & CORE_TAGS:
        # Core tag, but check for complexity indicators
        if method in COMPLEX_METHODS:
            return "moderate"
        return "core"

    if tag_set & ADVANCED_TAGS:
        return "advanced"

    # Check path complexity
    path_segments = path.strip("/").split("/")
    if len(path_segments) > 4:  # Deep nesting suggests complexity
        return "advanced"

    # Check for admin/management paths
    if any(segment in ["admin", "management", "internal"] for segment in path_segments):
        return "advanced"

    # Check summary for complexity indicators
    if summary:
        summary_lower = summary.lower()
        if any(
            word in summary_lower
            for word in [
                "advanced",
                "internal",
                "admin",
                "force",
                "dangerous",
                "migrate",
                "export",
                "import",
                "bulk",
                "batch",
            ]
        ):
            return "advanced"

    # Method-based classification
    if method in SIMPLE_METHODS:
        return "core"
    elif method in MODERATE_METHODS:
        return "moderate"
    else:
        return "advanced"


def filter_operations_by_complexity(
    spec: Dict[str, Union[str, Dict]], complexity_levels: Set[str]
) -> Dict[str, Union[str, Dict]]:
    """
    Filter OpenAPI spec to include only operations of specified complexity levels.

    Args:
        spec: OpenAPI specification
        complexity_levels: Set of complexity levels to include ('core', 'moderate', 'advanced')

    Returns:
        Filtered spec with only matching operations
    """
    if not isinstance(spec.get("paths"), dict):
        return spec

    filtered_paths = {}

    for path, path_methods in spec["paths"].items():  # type: ignore[union-attr]
        if not isinstance(path_methods, dict):
            continue

        filtered_methods = {}

        for method, operation in path_methods.items():
            if not isinstance(operation, dict):
                continue

            operation_id = operation.get("operationId", "")
            tags = operation.get("tags", [])
            summary = operation.get("summary")

            complexity = classify_operation_complexity(
                operation_id, tags, method.upper(), path, summary
            )

            if complexity in complexity_levels:
                filtered_methods[method] = operation

        if filtered_methods:
            filtered_paths[path] = filtered_methods

    # Create new spec with filtered paths
    filtered_spec = dict(spec)
    filtered_spec["paths"] = filtered_paths

    return filtered_spec


def get_core_operations_spec(spec: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:
    """
    Get a spec containing only core operations for basic usage.

    Args:
        spec: Original OpenAPI specification

    Returns:
        Spec with only core operations
    """
    return filter_operations_by_complexity(spec, {"core"})


def get_moderate_operations_spec(spec: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:
    """
    Get a spec containing core and moderate operations for intermediate usage.

    Args:
        spec: Original OpenAPI specification

    Returns:
        Spec with core and moderate operations
    """
    return filter_operations_by_complexity(spec, {"core", "moderate"})


def get_advanced_operations_spec(spec: Dict[str, Union[str, Dict]]) -> Dict[str, Union[str, Dict]]:
    """
    Get a spec containing all operations including advanced ones.

    Args:
        spec: Original OpenAPI specification

    Returns:
        Complete spec (same as input)
    """
    return filter_operations_by_complexity(spec, {"core", "moderate", "advanced"})


def analyse_operation_distribution(spec: Dict[str, Union[str, Dict]]) -> Dict[str, int]:
    """
    Analyze the distribution of operations by complexity level.

    Args:
        spec: OpenAPI specification

    Returns:
        Dictionary with counts for each complexity level
    """
    if not isinstance(spec.get("paths"), dict):
        return {"core": 0, "moderate": 0, "advanced": 0}

    counts = {"core": 0, "moderate": 0, "advanced": 0}

    for path, path_methods in spec["paths"].items():  # type: ignore[union-attr]
        if not isinstance(path_methods, dict):
            continue

        for method, operation in path_methods.items():
            if not isinstance(operation, dict):
                continue

            operation_id = operation.get("operationId", "")
            tags = operation.get("tags", [])
            summary = operation.get("summary")

            complexity = classify_operation_complexity(
                operation_id, tags, method.upper(), path, summary
            )

            counts[complexity] += 1

    return counts


def get_operations_by_complexity(
    spec: Dict[str, Union[str, Dict]],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Get all operations grouped by complexity level.

    Args:
        spec: OpenAPI specification

    Returns:
        Dictionary mapping complexity levels to lists of operation info
    """
    if not isinstance(spec.get("paths"), dict):
        return {"core": [], "moderate": [], "advanced": []}

    operations: Dict[str, List[Dict[str, str]]] = {"core": [], "moderate": [], "advanced": []}

    for path, path_methods in spec["paths"].items():  # type: ignore[union-attr]
        if not isinstance(path_methods, dict):
            continue

        for method, operation in path_methods.items():
            if not isinstance(operation, dict):
                continue

            operation_id = operation.get("operationId", "")
            tags = operation.get("tags", [])
            summary = operation.get("summary", "")

            complexity = classify_operation_complexity(
                operation_id, tags, method.upper(), path, summary
            )

            operations[complexity].append(
                {
                    "operation_id": operation_id,
                    "method": method.upper(),
                    "path": path,
                    "tags": tags,
                    "summary": summary,
                }
            )

    return operations
