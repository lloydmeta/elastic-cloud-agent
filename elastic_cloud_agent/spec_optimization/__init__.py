"""Spec optimisation utilities for the Elastic Cloud Agent."""

from elastic_cloud_agent.spec_optimization.operation_filter import (
    get_core_operations_spec,
    get_moderate_operations_spec,
)
from elastic_cloud_agent.spec_optimization.spec_minimizer import minimize_spec
from elastic_cloud_agent.spec_optimization.spec_registry import (
    create_spec_registry,
    get_spec_for_query,
)
from elastic_cloud_agent.spec_optimization.spec_splitter import split_spec_by_tags

__all__ = [
    "split_spec_by_tags",
    "minimize_spec",
    "get_core_operations_spec",
    "get_moderate_operations_spec",
    "create_spec_registry",
    "get_spec_for_query",
]
