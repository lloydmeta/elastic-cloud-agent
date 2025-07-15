"""
Spec registry for mapping user intents to optimised OpenAPI spec subsets.
"""

from typing import Any, Dict, List, Union

from langchain_core.language_models import BaseLanguageModel

from elastic_cloud_agent.spec_optimization.operation_filter import (
    get_core_operations_spec,
    get_moderate_operations_spec,
)
from elastic_cloud_agent.spec_optimization.spec_minimizer import minimize_spec
from elastic_cloud_agent.spec_optimization.spec_splitter import split_spec_by_tags

# Intent patterns that map to different spec optimisation strategies
INTENT_PATTERNS = {
    "basic": {
        "description": "Basic read operations and information retrieval",
        "optimisation": "core_minimal",
    },
    "management": {
        "description": "Standard deployment and cluster management",
        "optimisation": "moderate_minimal",
    },
    "advanced": {
        "description": "Advanced operations and administrative tasks",
        "optimisation": "full_minimal",
    },
    "monitoring": {
        "description": "Monitoring and observability operations",
        "optimisation": "monitoring_tags",
    },
    "security": {
        "description": "Security and authentication operations",
        "optimisation": "security_tags",
    },
    "billing": {
        "description": "Billing and cost management operations",
        "optimisation": "billing_tags",
    },
}


class SpecRegistry:
    """Registry for managing OpenAPI spec optimisations based on user intents."""

    def __init__(self, base_spec: Dict[str, Union[str, Dict]], llm: BaseLanguageModel):
        """
        Initialize the spec registry with a base OpenAPI specification.

        Args:
            base_spec: The complete OpenAPI specification to optimise
            llm: The language model to use for intent classification
        """
        self.base_spec = base_spec
        self.llm = llm
        self._spec_cache: Dict[str, Dict[str, Union[str, Dict]]] = {}

    def _get_relevant_tags_for_intent(self, intent: str) -> List[str]:
        """
        Use LLM to determine relevant OpenAPI tags for a given intent.

        Args:
            intent: The intent classification

        Returns:
            List of relevant OpenAPI tags
        """
        # Extract all available tags from the base spec
        available_tags = set()
        if "paths" in self.base_spec and isinstance(self.base_spec["paths"], dict):
            for path_data in self.base_spec["paths"].values():
                if isinstance(path_data, dict):
                    for operation_data in path_data.values():
                        if isinstance(operation_data, dict) and "tags" in operation_data:
                            if isinstance(operation_data["tags"], list):
                                available_tags.update(operation_data["tags"])

        if not available_tags:
            return []

        available_tags_list = sorted(list(available_tags))

        intent_description = INTENT_PATTERNS.get(intent, {}).get(
            "description", "General operations"
        )

        prompt = f"""
You are an API tag selector for an Elastic Cloud management system. Given an intent and available OpenAPI tags, select the most relevant tags.

Intent: {intent}
Intent Description: {intent_description}

Available tags:
{chr(10).join([f"- {tag}" for tag in available_tags_list])}

Select the tags most relevant to the "{intent}" intent. Consider:
- Security intent: Tags related to authentication, authorization, users, roles, API keys
- Billing intent: Tags related to costs, billing, usage, pricing
- Monitoring intent: Tags related to health, metrics, logging, monitoring
- Management intent: Tags related to deployments, clusters, basic operations
- Advanced intent: Tags related to advanced features, administration, extensions
- Basic intent: Tags for simple read operations and basic information

Respond with a JSON array of tag names only, e.g., ["Tag1", "Tag2"]. If no tags are relevant, respond with an empty array [].

Tags:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]) if content else "[]"
            else:
                content = str(content)

            # Try to parse JSON response
            import json

            try:
                selected_tags = json.loads(content.strip())
                if isinstance(selected_tags, list):
                    # Validate that all tags are in the available tags
                    valid_tags = [tag for tag in selected_tags if tag in available_tags_list]
                    return valid_tags
            except json.JSONDecodeError:
                pass

            # Fallback: try to extract tags from response text
            selected_tags = []
            for tag in available_tags_list:
                if tag.lower() in content.lower():
                    selected_tags.append(tag)

            return selected_tags

        except Exception:
            # Fallback to reasonable defaults based on intent
            fallback_mapping = {
                "security": [
                    tag
                    for tag in available_tags_list
                    if any(
                        keyword in tag.lower()
                        for keyword in ["auth", "security", "iam", "user", "role"]
                    )
                ],
                "billing": [
                    tag
                    for tag in available_tags_list
                    if any(keyword in tag.lower() for keyword in ["billing", "cost", "analysis"])
                ],
                "monitoring": [
                    tag
                    for tag in available_tags_list
                    if any(
                        keyword in tag.lower() for keyword in ["deployment", "health", "monitor"]
                    )
                ],
                "management": [
                    tag
                    for tag in available_tags_list
                    if any(keyword in tag.lower() for keyword in ["deployment", "account"])
                ],
                "advanced": [
                    tag
                    for tag in available_tags_list
                    if any(
                        keyword in tag.lower()
                        for keyword in ["organization", "extension", "stack", "traffic"]
                    )
                ],
                "basic": [
                    tag
                    for tag in available_tags_list
                    if any(keyword in tag.lower() for keyword in ["deployment", "account"])
                ],
            }
            return fallback_mapping.get(intent, [])

    def classify_intent(self, query: str) -> str:
        """
        Classify user intent based on query text using LLM.

        Args:
            query: User query text

        Returns:
            Intent classification ('basic', 'management', 'advanced', etc.)
        """
        available_intents = list(INTENT_PATTERNS.keys())
        intent_descriptions = {
            intent: config["description"] for intent, config in INTENT_PATTERNS.items()
        }

        prompt = f"""
You are an intent classifier for an Elastic Cloud management system. Classify the following user query into one of these intents:

Available intents:
{chr(10).join([f"- {intent}: {desc}" for intent, desc in intent_descriptions.items()])}

User query: "{query}"

Respond with ONLY the intent name (one word) from the list above. Consider:
- Basic intents: Simple read operations, viewing information
- Management intents: Creating, updating, deploying, controlling resources
- Advanced intents: Complex operations, migrations, bulk operations, administrative tasks
- Monitoring intents: Health checks, performance metrics, diagnostics
- Security intents: Authentication, authorisation, API keys, permissions
- Billing intents: Costs, invoices, pricing, usage information

Intent:"""

        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            # Handle case where content is a list
            predicted_intent = str(content[0]).strip().lower() if content else ""
        else:
            predicted_intent = str(content).strip().lower()

        # Validate the predicted intent
        if predicted_intent in available_intents:
            return predicted_intent
        else:
            # Fallback to management if invalid
            return "management"

    def get_optimised_spec(
        self, intent: str, use_cache: bool = True
    ) -> Dict[str, Union[str, Dict]]:
        """
        Get an optimised spec for a specific intent.

        Args:
            intent: The intent classification
            use_cache: Whether to use cached specs

        Returns:
            Optimized OpenAPI specification
        """
        cache_key = f"intent_{intent}"

        if use_cache and cache_key in self._spec_cache:
            return self._spec_cache[cache_key]

        # Get the optimisation strategy for this intent
        if intent not in INTENT_PATTERNS:
            intent = "management"  # Default fallback

        optimisation = INTENT_PATTERNS[intent]["optimisation"]
        optimised_spec = self._apply_optimisation(str(optimisation))

        if use_cache:
            self._spec_cache[cache_key] = optimised_spec

        return optimised_spec

    def get_spec_for_query(self, query: str, use_cache: bool = True) -> Dict[str, Union[str, Dict]]:
        """
        Get an optimised spec based on a user query.

        Args:
            query: User query text
            use_cache: Whether to use cached specs

        Returns:
            Optimized OpenAPI specification
        """
        intent = self.classify_intent(query)
        return self.get_optimised_spec(intent, use_cache)

    def _apply_optimisation(self, optimisation: str) -> Dict[str, Union[str, Dict]]:
        """
        Apply a specific optimisation strategy to the base spec.

        Args:
            optimisation: The optimisation strategy to apply

        Returns:
            Optimized specification
        """
        if optimisation == "core_minimal":
            # Basic operations with minimal descriptions
            spec = get_core_operations_spec(self.base_spec)
            return minimize_spec(spec, remove_verbose_fields_flag=True, minimize_descriptions=True)

        elif optimisation == "moderate_minimal":
            # Core + moderate operations with minimal descriptions
            spec = get_moderate_operations_spec(self.base_spec)
            return minimize_spec(spec, remove_verbose_fields_flag=True, minimize_descriptions=True)

        elif optimisation == "full_minimal":
            # All operations but minimized
            return minimize_spec(
                self.base_spec, remove_verbose_fields_flag=True, minimize_descriptions=True
            )

        elif optimisation.endswith("_tags"):
            # LLM-based tag filtering
            tag_prefix = optimisation.replace("_tags", "")
            tags = self._get_relevant_tags_for_intent(tag_prefix)

            if tags:
                tag_specs = split_spec_by_tags(self.base_spec, tag_filter=tags)

                # Combine specs from multiple tags
                combined_spec = dict(self.base_spec)
                combined_paths = {}

                for tag in tags:
                    if tag in tag_specs:
                        tag_spec = tag_specs[tag]
                        if "paths" in tag_spec:
                            combined_paths.update(tag_spec["paths"])

                combined_spec["paths"] = combined_paths
                return minimize_spec(
                    combined_spec, remove_verbose_fields_flag=True, minimize_descriptions=True
                )
            else:
                # Fallback to full spec if no relevant tags found
                return minimize_spec(
                    self.base_spec, remove_verbose_fields_flag=True, minimize_descriptions=True
                )

        # Default: return minimized full spec
        return minimize_spec(
            self.base_spec, remove_verbose_fields_flag=True, minimize_descriptions=True
        )

    def get_available_intents(self) -> Dict[str, str]:
        """
        Get available intent classifications and their descriptions.

        Returns:
            Dictionary mapping intent names to descriptions
        """
        return {intent: str(config["description"]) for intent, config in INTENT_PATTERNS.items()}

    def analyse_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyse a query and provide detailed intent classification results using LLM.

        Args:
            query: User query text

        Returns:
            Dictionary with intent analysis results
        """
        # Use LLM for classification
        best_intent = self.classify_intent(query)

        return {
            "intent": best_intent,
            "confidence": 1.0,  # LLM-based classification doesn't have numeric confidence
            "query_words": len(query.split()),
        }

    def clear_cache(self) -> None:
        """Clear the spec cache."""
        self._spec_cache.clear()

    def get_cache_stats(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_specs": len(self._spec_cache),
            "cache_keys": list(self._spec_cache.keys()),
        }

    def preload_common_specs(self) -> None:
        """Preload commonly used spec optimisations into cache."""
        common_intents = ["basic", "management", "advanced"]
        for intent in common_intents:
            self.get_optimised_spec(intent, use_cache=True)


def create_spec_registry(
    base_spec: Dict[str, Union[str, Dict]], llm: BaseLanguageModel
) -> SpecRegistry:
    """
    Create and return a spec registry instance.

    Args:
        base_spec: The complete OpenAPI specification
        llm: The language model to use for intent classification

    Returns:
        Initialized SpecRegistry instance
    """
    return SpecRegistry(base_spec, llm)


def get_spec_for_query(
    query: str, base_spec: Dict[str, Union[str, Dict]], llm: BaseLanguageModel
) -> Dict[str, Union[str, Dict]]:
    """
    Convenience function to get optimised spec for a query.

    Args:
        query: User query text
        base_spec: Complete OpenAPI specification
        llm: The language model to use for intent classification

    Returns:
        Optimised OpenAPI specification
    """
    registry = create_spec_registry(base_spec, llm)
    return registry.get_spec_for_query(query)


def analyse_query_intent(query: str, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Convenience function to analyse query intent.

    Args:
        query: User query text
        llm: The language model to use for intent classification

    Returns:
        Intent analysis results
    """
    # Create a dummy registry for intent analysis
    registry = SpecRegistry({}, llm)
    return registry.analyse_query_intent(query)
