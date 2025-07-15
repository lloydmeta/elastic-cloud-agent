"""
Intent-aware OpenAPI tool for the Elastic Cloud Agent.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from elastic_cloud_agent.spec_optimization.lazy_loader import SpecCacheManager
from elastic_cloud_agent.spec_optimization.spec_registry import SpecRegistry
from elastic_cloud_agent.tools.openapi_tool import create_requests_wrapper, load_api_spec
from elastic_cloud_agent.utils.config import Config


class ApiRequest(BaseModel):
    """Schema for API request inputs."""

    query: str = Field(description="The user's query describing what they want to do")
    method: Optional[str] = Field(default=None, description="HTTP method (GET, POST, etc.)")
    endpoint: Optional[str] = Field(default=None, description="API endpoint path")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request body data")


class IntentAwareApiTool(BaseTool):
    """
    Intent-aware OpenAPI tool that optimises API spec based on user intent.
    """

    name: str = "elastic_cloud_api"
    description: str = """
    Execute Elastic Cloud API operations with intelligent intent-based optimisation.
    
    This tool automatically:
    1. Analyses your query to understand your intent
    2. Loads the most relevant API endpoints for your needs
    3. Provides guidance on the best endpoints to use
    4. Executes API calls when you provide specific parameters
    
    Usage modes:
    - Exploratory: "What APIs are available for managing deployments?"
    - Guidance: "How do I create a new deployment?"
    - Execution: "Create a deployment with name 'my-cluster'"
    
    Input should be a natural language query describing what you want to do.
    """

    args_schema: type[BaseModel] = ApiRequest

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(
        self,
        llm: BaseLanguageModel,
        spec_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize the intent-aware API tool.

        Args:
            llm: Language model for intent classification
            spec_path: Path to OpenAPI specification file
            cache_dir: Directory for caching optimised specs
            **kwargs: Additional tool arguments
        """
        super().__init__(**kwargs)

        self.llm = llm
        self.requests_wrapper = create_requests_wrapper()

        # Load the base API specification
        self.base_spec = load_api_spec(spec_path)

        # Create spec registry for intent classification
        self.spec_registry = SpecRegistry(self.base_spec, llm)

        # Create cache manager for performance
        if cache_dir is None:
            cache_dir = Path.cwd() / ".spec_cache"
        self.cache_manager = SpecCacheManager(
            self.base_spec,
            llm,
            cache_dir=cache_dir,
            max_cache_size=50,
            cache_ttl_seconds=1800,  # 30 minutes
        )

        # Preload common specs in background
        self.cache_manager.preload_common_specs()

    def _run(self, query: str, **kwargs) -> str:
        """
        Execute the API tool with intent-based optimisation.

        Args:
            query: User query describing what they want to do
            **kwargs: Additional parameters (method, endpoint, data)

        Returns:
            API response or guidance
        """
        try:
            # Classify user intent
            intent_analysis = self.spec_registry.analyse_query_intent(query)
            intent = intent_analysis["intent"]

            # Get optimised spec for this intent
            optimised_spec = self.cache_manager.get_spec_lazy(query)

            # If user is asking for guidance or exploration
            if self._is_exploratory_query(query):
                return self._provide_api_guidance(query, intent, optimised_spec)

            # If user wants to execute an API call
            if self._is_execution_query(query, kwargs):
                return self._execute_api_call(query, intent, optimised_spec, kwargs)

            # Default: provide guidance with examples
            return self._provide_api_guidance(query, intent, optimised_spec)

        except Exception as e:
            # Fallback to basic guidance
            return f"Error processing query: {str(e)}. Please try rephrasing your request or specify more details."

    def _is_exploratory_query(self, query: str) -> bool:
        """Check if query is asking for exploration/guidance."""
        exploratory_keywords = [
            "what",
            "how",
            "which",
            "available",
            "can i",
            "help",
            "show",
            "list",
            "explain",
        ]
        return any(keyword in query.lower() for keyword in exploratory_keywords)

    def _is_execution_query(self, query: str, kwargs: Dict[str, Any]) -> bool:
        """Check if query is asking for execution."""
        execution_indicators = [
            kwargs.get("method") is not None,
            kwargs.get("endpoint") is not None,
            any(
                action in query.lower()
                for action in ["create", "delete", "update", "start", "stop", "restart"]
            ),
        ]
        return any(execution_indicators)

    def _provide_api_guidance(self, query: str, intent: str, optimised_spec: Dict[str, Any]) -> str:
        """Provide guidance on available APIs for the user's intent."""
        try:
            # Extract relevant paths from optimised spec
            paths = optimised_spec.get("paths", {})

            if not paths:
                return f"No specific API endpoints found for '{intent}' intent. Please try a different query or check the API specification."

            # Generate guidance
            guidance = [
                f"Based on your query about '{intent}' operations, here are relevant API endpoints:",
                "",
            ]

            # List available endpoints with descriptions
            for path, methods in paths.items():
                guidance.append(f"**{path}**")
                for method, details in methods.items():
                    if isinstance(details, dict) and "summary" in details:
                        guidance.append(f"  - {method.upper()}: {details['summary']}")
                guidance.append("")

            guidance.extend(
                [
                    f"**Intent Classification:** {intent}",
                    f"**Optimised Spec:** {len(paths)} endpoints loaded (vs ~{len(self.base_spec.get('paths', {}))} in full spec)",
                    "",
                    "To execute an API call, please specify:",
                    "- HTTP method (GET, POST, PUT, DELETE)",
                    "- Endpoint path",
                    "- Any required parameters or data",
                    "",
                    f"Base URL: {Config.ELASTIC_CLOUD_BASE_URL}",
                ]
            )

            return "\n".join(guidance)

        except Exception as e:
            return f"Error generating guidance: {str(e)}"

    def _execute_api_call(
        self, query: str, intent: str, optimised_spec: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> str:
        """Execute an API call based on the parameters."""
        try:
            method = kwargs.get("method", "GET").upper()
            endpoint = kwargs.get("endpoint")
            data = kwargs.get("data")

            if not endpoint:
                return "To execute an API call, please specify the endpoint path."

            # Construct full URL
            base_url = Config.ELASTIC_CLOUD_BASE_URL.rstrip("/")
            full_url = f"{base_url}{endpoint}"

            # Execute the API call
            if method == "GET":
                response = self.requests_wrapper.get(full_url)
            elif method == "POST":
                post_data = data if data else {}
                response = self.requests_wrapper.post(full_url, data=post_data)
            elif method == "PUT":
                put_data = data if data else {}
                response = self.requests_wrapper.put(full_url, data=put_data)
            elif method == "DELETE":
                response = self.requests_wrapper.delete(full_url)
            else:
                return f"Unsupported HTTP method: {method}"

            # Format response
            return self._format_api_response(method, full_url, response, intent)

        except Exception as e:
            return f"Error executing API call: {str(e)}"

    def _format_api_response(
        self, method: str, url: str, response: Union[str, Dict[str, Any]], intent: str
    ) -> str:
        """Format the API response for the user."""
        try:
            # Try to parse as JSON for better formatting
            if isinstance(response, str):
                response_data = json.loads(response)
                formatted_response = json.dumps(response_data, indent=2)
            else:
                formatted_response = json.dumps(response, indent=2)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, return as-is
            formatted_response = str(response)

        return f"""
**API Call Executed Successfully**

**Intent:** {intent}
**Method:** {method}
**URL:** {url}

**Response:**
```json
{formatted_response}
```
        """.strip()

    async def _arun(self, query: str, **kwargs) -> str:
        """Async version of _run (delegates to sync version)."""
        return self._run(query, **kwargs)


def create_intent_aware_api_tool(
    llm: BaseLanguageModel,
    spec_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> IntentAwareApiTool:
    """
    Create an intent-aware API tool.

    Args:
        llm: Language model for intent classification
        spec_path: Path to OpenAPI specification file
        cache_dir: Directory for caching optimised specs

    Returns:
        IntentAwareApiTool instance
    """
    return IntentAwareApiTool(llm=llm, spec_path=spec_path, cache_dir=cache_dir)
