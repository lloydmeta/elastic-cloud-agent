"""
Agent configuration for the Elastic Cloud Agent.
"""

from typing import Optional

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from elastic_cloud_agent.tools import create_search_tool, create_smart_openapi_toolkit
from elastic_cloud_agent.utils import Config


def create_agent_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for the agent.

    Returns:
        ChatPromptTemplate: The prompt template
    """
    # Define the system message that instructs the agent on its role and capabilities
    system_message = f"""
    You are an expert Elastic Cloud administrator. You help users manage and understand
    their Elastic Cloud deployments using both your knowledge and the available tools.

    You have access to two types of tools:
    1. Search tools that can find information about Elastic Cloud, Elasticsearch, and related topics
    2. API tools that can interact with the Elastic Cloud API

    When answering questions:
    - For general information about Elastic Cloud concepts, use your knowledge or search
      for information
    - For specific information about the user's deployments or to perform actions, use the API tools
    - Always verify information from searches and explain your reasoning
    - Be concise but thorough in your explanations
    - If you need to use an API endpoint, first check its documentation using the JSON explorer tool
    - Always format code, configuration, and JSON examples in code blocks

    Remember previous messages in this conversation and maintain context.
    If the user refers to something mentioned earlier, use that context to provide better responses.

    Remember that your actions can affect real Elastic Cloud deployments, so be
    careful when making changes.

    Do not try to figure out what authentication is required for calling the
    API; you have been programmed in a way that that is taken care of
    automatically when you try to use the tool.

    When making API calls to elastic cloud, you must
    use {Config.ELASTIC_CLOUD_BASE_URL} as the base path.
    """

    # Create and return the prompt template with chat history
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),  # For tracking conversation history
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def create_llm() -> BaseChatOpenAI:
    """
    Create and configure the language model.
    Uses Azure OpenAI if Azure configuration is provided, otherwise uses standard OpenAI.

    Returns:
        BaseChatOpenAI: The configured language model
    """
    if Config.is_azure_config():
        return AzureChatOpenAI(
            model="gpt-4o",  # Use GPT-4 for better reasoning capabilities
            temperature=0,  # Use deterministic responses for consistency
            api_key=Config.OPENAI_API_KEY,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
    else:
        return ChatOpenAI(
            model="gpt-4o",  # Use GPT-4 for better reasoning capabilities
            temperature=0,  # Use deterministic responses for consistency
            api_key=Config.OPENAI_API_KEY,
        )


def create_agent(llm: Optional[BaseChatOpenAI] = None) -> AgentExecutor:
    """
    Create an agent executor with the appropriate tools and configuration.

    Args:
        llm: Optional language model to use. If None, a new one will be created.

    Returns:
        AgentExecutor: The configured agent executor
    """
    # Create the language model if not provided
    if llm is None:
        llm = create_llm()

    # Create the search tool
    search_tool = create_search_tool()

    # Create the smart OpenAPI toolkit with all HTTP tools and intent-aware JSON explorer
    # This gives us requests_get, requests_post, etc. + smart json_explorer
    api_tools = create_smart_openapi_toolkit(llm=llm)

    # Combine all tools
    tools = [search_tool] + api_tools

    # Create the agent prompt
    prompt = create_agent_prompt()

    # Create the agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    # Create and return the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to True to see the agent's thoughts, False for production
        max_iterations=10,  # Limit the maximum number of steps the agent can take
        memory=None,  # We'll manage memory in the main.py file for simplicity
        handle_parsing_errors=True,  # Handle parsing errors gracefully
    )
