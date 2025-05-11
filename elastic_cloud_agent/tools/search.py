"""
Search tool for the Elastic Cloud Agent.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import BaseTool


def create_search_tool(
    region: str = "wt-wt",
    safe_search: str = "moderate",
    time: str | None = None,
    max_results: int = 3,
) -> BaseTool:
    """
    Create a search tool using DuckDuckGo.

    Args:
        region: Region for search results (e.g., "wt-wt", "us-en")
        safe_search: SafeSearch setting ("on", "moderate", "off")
        time: Time filter for search results ("d" - day, "w" - week, "m" - month)
        max_results: Maximum number of search results to return

    Returns:
        BaseTool: A search tool that can be used by the agent
    """
    # Create a customized DuckDuckGo search wrapper with the specified settings
    wrapper = DuckDuckGoSearchAPIWrapper(
        region=region,
        safesearch=safe_search,
        time=time,
        max_results=max_results,
    )

    # Create and return the search tool
    search_tool = DuckDuckGoSearchRun(
        api_wrapper=wrapper,
        name="web_search",
        description=(
            "Useful for searching the internet for information about Elastic Cloud, "
            "Elasticsearch, and related topics. Input should be a search query."
        ),
    )

    return search_tool
