"""
OpenAPI tool for the Elastic Cloud Agent.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.language_models import BaseLanguageModel

from elastic_cloud_agent.utils.config import Config, get_api_spec_path


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


def create_requests_wrapper() -> RequestsWrapper:
    """
    Create a requests wrapper for the Elastic Cloud API.

    Returns:
        RequestsWrapper: A configured requests wrapper
    """
    # Configure headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"ApiKey {Config.ELASTIC_CLOUD_API_KEY}",
    }

    return RequestsWrapper(headers=headers)


def create_openapi_toolkit(
    llm: BaseLanguageModel, spec_path: Optional[Path] = None
) -> List[BaseTool]:
    """
    Create an OpenAPI toolkit for the Elastic Cloud API.

    Args:
        llm: Language model to use for the JSON agent
        spec_path: Path to the OpenAPI specification file. If None, use the default path.

    Returns:
        List[BaseTool]: A list of tools for interacting with the Elastic Cloud API
    """
    # Load the API specification
    api_spec = load_api_spec(spec_path)

    # Create a JSON specification for the API
    json_spec = JsonSpec(dict_=api_spec)

    # Create a requests wrapper for making API calls
    requests_wrapper = create_requests_wrapper()

    # Create the OpenAPI toolkit
    toolkit = OpenAPIToolkit.from_llm(
        json_spec=json_spec,
        llm=llm,
        requests_wrapper=requests_wrapper,
        allow_dangerous_requests=True,
        handle_parsing_errors=True,  # Handle parsing errors gracefully
    )

    # Get all tools from the toolkit
    tools = toolkit.get_tools()

    return tools
