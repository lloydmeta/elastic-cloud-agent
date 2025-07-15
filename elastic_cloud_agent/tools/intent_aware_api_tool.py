"""
Intent-aware OpenAPI tool for the Elastic Cloud Agent.
"""

import json
import hashlib
import time
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

    query: Optional[str] = Field(default="", description="The user's query describing what they want to do")
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
        
        # Initialize response cache for caching API responses
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes TTL for API responses

    def _run(self, query: str = "", **kwargs) -> str:
        """
        Execute the API tool with intent-based optimisation.

        Args:
            query: User query describing what they want to do (optional)
            **kwargs: Additional parameters (method, endpoint, data)

        Returns:
            API response or guidance
        """
        try:
            # Expand query for better API matching
            expanded_query = self._expand_query_for_api_matching(query)
            
            # Classify user intent
            intent_analysis = self.spec_registry.analyse_query_intent(expanded_query)
            intent = intent_analysis["intent"]

            # Get optimised spec for this intent
            optimised_spec = self.cache_manager.get_spec_lazy(expanded_query)

            # If user wants to execute an API call (check this first)
            if self._is_execution_query(query, kwargs):
                return self._execute_api_call(query, intent, optimised_spec, kwargs)

            # If user is asking for guidance or exploration
            if self._is_exploratory_query(query):
                return self._provide_api_guidance(query, intent, optimised_spec)

            # Default: provide guidance with examples
            return self._provide_api_guidance(query, intent, optimised_spec)

        except Exception as e:
            # Fallback to basic guidance
            return f"Error processing query: {str(e)}. Please try rephrasing your request or specify more details."

    def _is_exploratory_query(self, query: str) -> bool:
        """Check if query is asking for exploration/guidance using LLM."""
        if not query or not query.strip():
            return True  # Default to exploratory for empty queries
        
        prompt = f"""
You are a query classifier for an API tool. Determine if the following query is asking for exploration/guidance about available APIs, or if it's asking for execution of a specific API operation.

Query: "{query}"

Exploratory queries are those that:
- Ask about what APIs are available
- Ask for help or guidance
- Ask how to do something
- Ask which endpoints to use
- Request explanations or information
- Are general questions about capabilities

Execution queries are those that:
- Request specific API operations to be performed
- Give specific parameters or data
- Use action verbs like "create", "delete", "update", "get specific item"
- Provide concrete details for an operation

Respond with only "exploratory" or "execution".

Classification:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]).strip().lower() if content else ""
            else:
                content = str(content).strip().lower()
            
            return "exploratory" in content
        except Exception:
            # Fallback to exploratory if LLM fails
            return True

    def _is_execution_query(self, query: str, kwargs: Dict[str, Any]) -> bool:
        """Check if query is asking for execution using LLM and parameters."""
        # If explicit method/endpoint provided, it's definitely execution
        if kwargs.get("method") is not None or kwargs.get("endpoint") is not None:
            return True
        
        if not query or not query.strip():
            return False  # Empty query defaults to exploratory
        
        prompt = f"""
You are a query classifier for an API tool. Determine if the following query is asking for execution of a specific API operation, or if it's asking for exploration/guidance about available APIs.

Query: "{query}"

Execution queries are those that:
- Request specific API operations to be performed
- Give specific parameters or data
- Use action verbs like "create", "delete", "update", "get specific item"
- Provide concrete details for an operation
- Ask to perform a specific task

Exploratory queries are those that:
- Ask about what APIs are available
- Ask for help or guidance
- Ask how to do something
- Ask which endpoints to use
- Request explanations or information
- Are general questions about capabilities

Respond with only "execution" or "exploratory".

Classification:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]).strip().lower() if content else ""
            else:
                content = str(content).strip().lower()
            
            return "execution" in content
        except Exception:
            # Fallback to exploratory if LLM fails
            return False

    def _provide_api_guidance(self, query: str, intent: str, optimised_spec: Dict[str, Any]) -> str:
        """Provide guidance on available APIs for the user's intent."""
        try:
            # Extract relevant paths from optimised spec
            paths = optimised_spec.get("paths", {})

            if not paths:
                return self._provide_fallback_guidance(query, intent)

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

    def _provide_fallback_guidance(self, query: str, intent: str) -> str:
        """Provide fallback guidance when no optimised spec is available."""
        try:
            # Use LLM to suggest potential API endpoints based on query
            fallback_guidance = self._generate_intelligent_fallback(query, intent)
            
            # Add general guidance
            general_guidance = [
                f"**Intent Classification:** {intent}",
                f"**Query:** {query}",
                "",
                "No specific API endpoints found for this intent, but here are some suggestions:",
                "",
                fallback_guidance,
                "",
                "**Alternative approaches:**",
                "- Try rephrasing your query with more specific terms",
                "- Use broader intent categories (e.g., 'list deployments' instead of specific deployment names)",
                "- Check if the API specification includes the functionality you need",
                "",
                f"Base URL: {Config.ELASTIC_CLOUD_BASE_URL}",
            ]
            
            return "\n".join(general_guidance)
            
        except Exception as e:
            return f"Error generating fallback guidance: {str(e)}"

    def _generate_intelligent_fallback(self, query: str, intent: str) -> str:
        """Generate intelligent fallback suggestions using LLM."""
        # Extract all available paths from base spec for context
        available_paths = list(self.base_spec.get("paths", {}).keys())
        
        if not available_paths:
            return "No API endpoints available in the current specification."
        
        # Sample a subset of paths to avoid token limits
        sample_paths = available_paths[:20] if len(available_paths) > 20 else available_paths
        
        prompt = f"""
You are an API guidance assistant for Elastic Cloud operations. Given a user query and intent, suggest the most relevant API endpoints from the available paths.

User Query: "{query}"
Intent: {intent}

Available API Endpoints:
{chr(10).join([f"- {path}" for path in sample_paths])}

Based on the query and intent, suggest 2-3 most relevant API endpoints that might help the user. For each suggestion, provide:
1. The endpoint path
2. Likely HTTP method (GET, POST, PUT, DELETE)
3. Brief explanation of what it does
4. Why it's relevant to the query

Format your response as markdown with clear structure.

Suggestions:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = str(content[0]) if content else "No suggestions available."
            else:
                content = str(content)
            
            return content.strip()
            
        except Exception:
            # Final fallback - return basic suggestions
            return "Consider trying common endpoints like:\n- GET /deployments (list deployments)\n- GET /account (account information)\n- GET /deployments/{id} (specific deployment details)"

    def _expand_query_for_api_matching(self, query: str) -> str:
        """Expand query with API-relevant synonyms and context for better matching."""
        if not query or not query.strip():
            return query
        
        # For simple queries, return as-is to avoid over-processing
        if len(query.split()) <= 3:
            return query
        
        prompt = f"""
You are a query expansion assistant for Elastic Cloud API operations. Given a user query, expand it with relevant API terminology and synonyms to improve API endpoint matching.

Original Query: "{query}"

Expand the query by:
1. Adding relevant API terminology (e.g., "list" → "list, get, retrieve, fetch")
2. Including common Elastic Cloud terms (e.g., "cluster" → "cluster, deployment, instance")
3. Adding operation synonyms (e.g., "create" → "create, add, provision, deploy")
4. Keeping the original intent clear

Return only the expanded query text, not explanations. Keep it concise and focused.

Expanded Query:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                expanded = str(content[0]).strip() if content else query
            else:
                expanded = str(content).strip()
            
            # Validate expansion isn't too long or nonsensical
            if len(expanded) > len(query) * 3 or len(expanded) > 500:
                return query  # Return original if expansion is too aggressive
            
            return expanded if expanded else query
            
        except Exception:
            # Return original query if expansion fails
            return query

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

            # Check cache for GET requests (safe to cache)
            if method == "GET":
                cache_key = self._generate_cache_key(method, endpoint, data)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    return cached_response

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

            # Validate and format response
            formatted_response = self._validate_and_format_response(method, full_url, response, intent, query)
            
            # Cache GET responses for future use
            if method == "GET":
                cache_key = self._generate_cache_key(method, endpoint, data)
                self._cache_response(cache_key, formatted_response)
            
            return formatted_response

        except Exception as e:
            return self._handle_api_error(method, endpoint or "", str(e), intent)

    def _validate_and_format_response(
        self, method: str, url: str, response: Union[str, Dict[str, Any]], intent: str, query: str
    ) -> str:
        """Validate and format the API response for the user."""
        try:
            # Parse response if it's a string
            if isinstance(response, str):
                try:
                    response_data = json.loads(response)
                except json.JSONDecodeError:
                    # Not JSON, treat as plain text
                    response_data = response
            else:
                response_data = response
            
            # Validate response structure
            validation_result = self._validate_api_response(response_data, method, url)
            
            if validation_result["is_valid"]:
                return self._format_successful_response(method, url, response_data, intent, query)
            else:
                return self._format_error_response(method, url, response_data, intent, validation_result["error"])
        
        except Exception as e:
            return self._handle_api_error(method, url, str(e), intent)

    def _validate_api_response(self, response_data: Union[str, Dict[str, Any]], method: str, url: str) -> Dict[str, Any]:
        """Validate API response structure and content."""
        try:
            # Check for common error patterns
            if isinstance(response_data, dict):
                # Check for standard error fields
                if "error" in response_data or "errors" in response_data:
                    error_msg = response_data.get("error", response_data.get("errors", "Unknown error"))
                    return {"is_valid": False, "error": f"API Error: {error_msg}"}
                
                # Check for HTTP error status
                if "status" in response_data and isinstance(response_data["status"], int):
                    if response_data["status"] >= 400:
                        return {"is_valid": False, "error": f"HTTP {response_data['status']}: {response_data.get('message', 'Error')}"}
                
                # Check for empty response on operations that should return data
                if method == "GET" and not response_data:
                    return {"is_valid": False, "error": "Empty response received for GET request"}
            
            elif isinstance(response_data, str):
                # Check for error keywords in string responses
                error_keywords = ["error", "failed", "invalid", "not found", "unauthorized", "forbidden"]
                if any(keyword in response_data.lower() for keyword in error_keywords):
                    return {"is_valid": False, "error": f"Potential error in response: {response_data[:200]}..."}
            
            return {"is_valid": True, "error": None}
            
        except Exception as e:
            return {"is_valid": False, "error": f"Validation error: {str(e)}"}

    def _format_successful_response(
        self, method: str, url: str, response_data: Union[str, Dict[str, Any]], intent: str, query: str
    ) -> str:
        """Format a successful API response."""
        try:
            # Format response data
            if isinstance(response_data, str):
                formatted_response = response_data
            else:
                formatted_response = json.dumps(response_data, indent=2)
        except (json.JSONDecodeError, TypeError):
            formatted_response = str(response_data)

        # Generate intelligent summary
        summary = self._generate_response_summary(response_data, intent, query)

        return f"""
**API Call Executed Successfully**

**Intent:** {intent}
**Method:** {method}
**URL:** {url}

**Summary:** {summary}

**Response:**
```json
{formatted_response}
```
        """.strip()

    def _format_error_response(
        self, method: str, url: str, response_data: Union[str, Dict[str, Any]], intent: str, error: str
    ) -> str:
        """Format an error response with helpful suggestions."""
        try:
            if isinstance(response_data, str):
                formatted_response = response_data
            else:
                formatted_response = json.dumps(response_data, indent=2)
        except (json.JSONDecodeError, TypeError):
            formatted_response = str(response_data)

        return f"""
**API Call Failed**

**Intent:** {intent}
**Method:** {method}
**URL:** {url}
**Error:** {error}

**Response:**
```json
{formatted_response}
```

**Suggestions:**
- Check if the endpoint path is correct
- Verify required parameters are provided
- Ensure you have proper authentication
- Try a different HTTP method if appropriate
        """.strip()

    def _handle_api_error(self, method: str, endpoint: str, error: str, intent: str) -> str:
        """Handle API execution errors with helpful suggestions."""
        return f"""
**API Call Error**

**Intent:** {intent}
**Method:** {method}
**Endpoint:** {endpoint}
**Error:** {error}

**Troubleshooting Steps:**
1. Check your network connection
2. Verify the API endpoint URL is correct
3. Ensure authentication credentials are valid
4. Check if the API service is available
5. Review the request parameters for correctness

**Next Steps:**
- Try a simpler API call first (e.g., GET /account)
- Check the API documentation for this endpoint
- Verify the base URL configuration: {Config.ELASTIC_CLOUD_BASE_URL}
        """.strip()

    def _generate_response_summary(self, response_data: Union[str, Dict[str, Any]], intent: str, query: str) -> str:
        """Generate an intelligent summary of the API response."""
        try:
            if isinstance(response_data, str):
                if len(response_data) < 100:
                    return response_data
                else:
                    return f"Text response ({len(response_data)} characters)"
            
            elif isinstance(response_data, dict):
                if "deployments" in response_data:
                    count = len(response_data["deployments"]) if isinstance(response_data["deployments"], list) else 1
                    return f"Found {count} deployment(s)"
                elif "account" in response_data:
                    return "Account information retrieved"
                elif "id" in response_data:
                    return f"Resource retrieved (ID: {response_data['id']})"
                elif len(response_data) == 1:
                    key = list(response_data.keys())[0]
                    return f"Retrieved {key} information"
                else:
                    return f"Retrieved {len(response_data)} data fields"
            
            return "API response received"
            
        except Exception:
            return "Response summary unavailable"

    def _generate_cache_key(self, method: str, endpoint: str, data: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for API responses."""
        # Create a unique key based on method, endpoint, and data
        key_data = f"{method}:{endpoint}:{json.dumps(data, sort_keys=True) if data else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        if cache_key in self._response_cache:
            cached_entry = self._response_cache[cache_key]
            if time.time() - cached_entry["timestamp"] < self._cache_ttl:
                return cached_entry["response"]
            else:
                # Remove expired entry
                del self._response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache an API response with timestamp."""
        self._response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self._response_cache) > 100:
            # Remove the oldest 20 entries
            oldest_keys = sorted(
                self._response_cache.keys(),
                key=lambda k: self._response_cache[k]["timestamp"]
            )[:20]
            for key in oldest_keys:
                del self._response_cache[key]

    def _format_api_response(
        self, method: str, url: str, response: Union[str, Dict[str, Any]], intent: str
    ) -> str:
        """Format the API response for the user (legacy method for backward compatibility)."""
        return self._validate_and_format_response(method, url, response, intent, "")

    async def _arun(self, query: str = "", **kwargs) -> str:
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
