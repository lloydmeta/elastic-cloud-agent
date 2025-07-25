"""
Smart OpenAPI Toolkit with intent-aware JSON exploration.

This toolkit enhances the standard OpenAPIToolkit by making the json_explorer
tool smarter - it classifies user intent and filters the OpenAPI specification
to only show relevant endpoints, reducing token usage and improving accuracy.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents.agent import AgentExecutor
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.openapi.prompt import DESCRIPTION
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, Tool

from elastic_cloud_agent.utils.config import Config, get_api_spec_path
from elastic_cloud_agent.utils.openapi_utils import (
    classify_intent,
    extract_tags_from_spec,
    filter_spec_by_tags,
)


def load_api_spec(spec_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and parse the OpenAPI specification.

    Args:
        spec_path: Path to the OpenAPI specification file. If None, use the default path.

    Returns:
        Dict[str, Any]: The parsed OpenAPI specification
    """
    if spec_path is None:
        spec_path = get_api_spec_path()

    with open(spec_path, "r") as f:
        if spec_path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        elif spec_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {spec_path.suffix}")


def create_requests_wrapper() -> TextRequestsWrapper:
    """
    Create a requests wrapper for the Elastic Cloud API.

    Returns:
        TextRequestsWrapper: A configured requests wrapper
    """
    # Configure headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {Config.ELASTIC_CLOUD_API_KEY}",
    }

    return TextRequestsWrapper(headers=headers)


class SmartJsonExplorerTool:
    """
    A smart JSON explorer tool that filters API specs based on user intent.

    This tool wraps the standard JsonAgent but enhances it by:
    1. Extracting intent from the user's query
    2. Filtering the OpenAPI spec to relevant categories
    3. Creating a focused JsonAgent with reduced scope
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        api_spec: Dict[str, Any],
        **kwargs: Any,
    ):
        """
        Initialise the smart JSON explorer tool.

        Args:
            llm: Language model for intent classification and JSON exploration
            api_spec: Full OpenAPI specification
            **kwargs: Additional arguments passed to Tool
        """
        self.llm = llm
        self.api_spec = api_spec
        self.available_tags = extract_tags_from_spec(api_spec)
        self.name = "json_explorer"
        self.description = DESCRIPTION
        self._agent_cache: Dict[frozenset, AgentExecutor] = {}

    def _run(self, query: str) -> str:
        """
        Execute the smart JSON exploration.

        Args:
            query: User's query about the API

        Returns:
            Response from the focused JSON agent
        """
        try:
            # Step 1: Classify intent to determine relevant categories
            relevant_tags = classify_intent(query, self.available_tags, self.llm)
            tags_key = frozenset(relevant_tags)

            print(f"Smart JSON Explorer - Classified intent: {relevant_tags}")

            # Step 2: Check cache for existing agent
            if tags_key in self._agent_cache:
                json_agent = self._agent_cache[tags_key]
            else:
                # Step 3: Filter the API spec to relevant categories
                filtered_spec = filter_spec_by_tags(self.api_spec, relevant_tags)

                paths = list(filtered_spec.get("paths", {}).keys())
                preview = paths[:5]
                if len(paths) > 5:
                    preview.append("...")
                print(f"Smart JSON Explorer - Filtered to paths: {preview}")

                # Step 4: Create a focused JsonAgent with the filtered spec
                json_spec = JsonSpec(dict_=filtered_spec)
                json_toolkit = JsonToolkit(spec=json_spec)
                json_agent = create_json_agent(
                    self.llm,
                    json_toolkit,
                    verbose=False,
                    agent_executor_kwargs={"handle_parsing_errors": True, "max_iterations": 30},
                )

                # Step 5: Cache the agent
                self._agent_cache[tags_key] = json_agent

            # Step 6: Run the focused JsonAgent
            response = json_agent.invoke({"input": query})
            result = response["output"]

            return str(result)

        except Exception as e:
            return f"Error in smart JSON exploration: {str(e)}"


class SmartJsonExplorerOpenAPIToolkit:
    """
    Smart OpenAPI Toolkit that combines RequestsToolkit with intent-aware JSON exploration.

    This toolkit provides:
    1. All standard HTTP request tools (GET, POST, PATCH, PUT, DELETE) - unchanged
    2. A smart json_explorer that filters the API spec based on user intent

    This reduces token usage while maintaining full functionality.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        api_spec: Dict[str, Any],
        requests_wrapper: TextRequestsWrapper,
        allow_dangerous_requests: bool = False,
        **kwargs: Any,
    ):
        """
        Initialise the Smart OpenAPI Toolkit.

        Args:
            llm: Language model for intent classification and JSON exploration
            api_spec: Full OpenAPI specification
            requests_wrapper: Configured requests wrapper for HTTP calls
            allow_dangerous_requests: Whether to allow dangerous requests
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.api_spec = api_spec
        self.requests_wrapper = requests_wrapper
        self.allow_dangerous_requests = allow_dangerous_requests

    def get_tools(self) -> List[BaseTool]:
        """
        Get all tools in the toolkit.

        Returns:
            List of tools including HTTP request tools and smart JSON explorer
        """
        # Get all the standard HTTP request tools (unchanged)
        request_toolkit = RequestsToolkit(
            requests_wrapper=self.requests_wrapper,
            allow_dangerous_requests=self.allow_dangerous_requests,
        )
        http_tools = request_toolkit.get_tools()

        # Create the smart JSON explorer tool
        smart_json_explorer_helper = SmartJsonExplorerTool(
            llm=self.llm,
            api_spec=self.api_spec,
        )

        # Create a Tool object from the helper
        smart_json_explorer = Tool(
            name=smart_json_explorer_helper.name,
            description=smart_json_explorer_helper.description,
            func=smart_json_explorer_helper._run,
        )

        # Return all tools: HTTP tools + smart JSON explorer
        return [*http_tools, smart_json_explorer]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        api_spec: Dict[str, Any],
        requests_wrapper: TextRequestsWrapper,
        allow_dangerous_requests: bool = False,
        **kwargs: Any,
    ) -> "SmartJsonExplorerOpenAPIToolkit":
        """
        Create a SmartJsonExplorerOpenAPIToolkit from components.

        This method provides the same interface as OpenAPIToolkit.from_llm()
        for easy drop-in replacement.

        Args:
            llm: Language model for intent classification and JSON exploration
            api_spec: Full OpenAPI specification (as dict, not JsonSpec)
            requests_wrapper: Configured requests wrapper for HTTP calls
            allow_dangerous_requests: Whether to allow dangerous requests
            **kwargs: Additional arguments

        Returns:
            Configured SmartJsonExplorerOpenAPIToolkit instance
        """
        return cls(
            llm=llm,
            api_spec=api_spec,
            requests_wrapper=requests_wrapper,
            allow_dangerous_requests=allow_dangerous_requests,
            **kwargs,
        )


def create_smart_openapi_toolkit(
    llm: BaseLanguageModel, spec_path: Optional[Path] = None
) -> List[BaseTool]:
    """
    Create a smart OpenAPI toolkit for the Elastic Cloud API.

    This function creates a SmartJsonExplorerOpenAPIToolkit that:
    1. Provides all standard HTTP request tools (GET, POST, PATCH, PUT, DELETE)
    2. Replaces the naive json_explorer with an intent-aware one that filters
       the OpenAPI spec based on user queries

    This is a drop-in replacement for create_openapi_toolkit() with enhanced
    JSON exploration capabilities.

    Args:
        llm: Language model to use for intent classification and JSON exploration
        spec_path: Path to the OpenAPI specification file. If None, use the default path.

    Returns:
        List[BaseTool]: A list of tools for interacting with the Elastic Cloud API
    """
    # Load the API specification
    api_spec = load_api_spec(spec_path)

    # Create a requests wrapper for making API calls
    requests_wrapper = create_requests_wrapper()

    # Create the smart OpenAPI toolkit
    toolkit = SmartJsonExplorerOpenAPIToolkit.from_llm(
        llm=llm,
        api_spec=api_spec,
        requests_wrapper=requests_wrapper,
        allow_dangerous_requests=True,
    )

    # Get all tools from the toolkit
    tools = toolkit.get_tools()

    return tools
