"""
Main entry point for the Elastic Cloud chat agent.
"""

import os
import sys
from pathlib import Path
from typing import List

from colorama import Fore, Style, init
from langchain_core.messages import AIMessage, HumanMessage

from elastic_cloud_agent.agent import create_agent
from elastic_cloud_agent.utils import Config

# Initialize colorama for colored output
init(autoreset=True)


def print_welcome_message() -> None:
    """Print a welcome message to the console."""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}{'Elastic Cloud Chat Agent':^80}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(
        "\nWelcome to the Elastic Cloud Chat Agent! Ask questions "
        "or request actions related to "
        "your Elastic Cloud deployments.\n"
    )
    print(f"{Fore.YELLOW}Type 'exit', 'quit', or press Ctrl+C to exit the chat.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'clear' to start a new conversation.{Style.RESET_ALL}\n")


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.

    Returns:
        bool: True if all required environment variables are set, False otherwise
    """
    is_valid, errors = Config.validate()

    if not is_valid:
        print(f"{Fore.RED}Error: Unable to start the chat agent.{Style.RESET_ALL}")
        print("The following configuration issues were found:")
        for error in errors:
            print(f"  - {error}")

        print("\nPlease check your .env file or environment variables.")
        return False

    return True


def create_data_directory() -> None:
    """
    Create the data directory if it doesn't exist.
    This is where the API specification file should be placed.
    """
    data_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / "data"
    data_dir.mkdir(exist_ok=True)

    spec_path = Path(Config.API_SPEC_PATH)
    if not spec_path.exists():
        print(
            f"{Fore.YELLOW}Note: API specification file not found at {spec_path}{Style.RESET_ALL}"
        )
        print("Please place your Elastic Cloud OpenAPI specification at this location.")


def chat_loop() -> None:
    """Run the main chat loop."""
    # Create the agent
    agent_executor = create_agent()

    # Initialise an in-memory message history
    chat_history: List = []

    # Loop for continuous conversation
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print(f"\n{Fore.CYAN}Thanks for using the Elastic Cloud Agent.{Style.RESET_ALL}")
                break

            # Check for clear conversation command
            if user_input.lower() == "clear":
                chat_history = []
                print(f"{Fore.YELLOW}Conversation history has been cleared.{Style.RESET_ALL}")
                continue

            # Process the user's query
            if user_input.strip():
                # Add the user message to chat history
                chat_history.append(HumanMessage(content=user_input))

                print(f"\n{Fore.BLUE}Agent: {Style.RESET_ALL}", end="")

                # Invoke the agent with chat history
                response = agent_executor.invoke(
                    {
                        "input": user_input,
                        "chat_history": chat_history[
                            :-1
                        ],  # All previous messages except the current one
                    }
                )

                output = response.get("output", "I'm sorry, I couldn't process that request.")
                print(output)

                # Add the agent's response to chat history
                chat_history.append(AIMessage(content=output))

                print()  # Add a blank line for readability

        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Interrupted. Goodbye!{Style.RESET_ALL}")
            break

        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            print("Please try again with a different query.")


def main() -> None:
    """Main entry point for the application."""
    try:
        # Validate the environment
        if not validate_environment():
            sys.exit(1)

        # Create the data directory
        create_data_directory()

        # Print the welcome message
        print_welcome_message()

        # Start the chat loop
        chat_loop()

    except KeyboardInterrupt:
        print(f"\n\n{Fore.CYAN}Interrupted. Goodbye!{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
