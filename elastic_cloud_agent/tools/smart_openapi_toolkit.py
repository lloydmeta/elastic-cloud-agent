"""
Smart OpenAPI Toolkit with intent-aware JSON exploration.

This toolkit enhances the standard OpenAPIToolkit by making the json_explorer
tool smarter - it classifies user intent and filters the OpenAPI specification
to only show relevant endpoints, reducing token usage and improving accuracy.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.openapi.prompt import DESCRIPTION
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, Tool

from elastic_cloud_agent.tools.openapi_tool import create_requests_wrapper, load_api_spec
from elastic_cloud_agent.utils.openapi_utils import (
    classify_intent,
    extract_tags_from_spec,
    filter_spec_by_tags,
)


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

            print(f"Smart JSON Explorer - Classified intent: {relevant_tags}")

            # Step 2: Filter the API spec to relevant categories
            filtered_spec = filter_spec_by_tags(self.api_spec, relevant_tags)

            paths = list(filtered_spec.get("paths", {}).keys())
            preview = paths[:5]
            if len(paths) > 5:
                preview.append("...")
            print(f"Smart JSON Explorer - Filtered to paths: [{preview}]")

            # Step 3: Create a focused JsonAgent with the filtered spec
            json_spec = JsonSpec(dict_=filtered_spec)
            json_toolkit = JsonToolkit(spec=json_spec)
            json_agent = create_json_agent(self.llm, json_toolkit, verbose=False)

            # Step 4: Run the focused JsonAgent
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
