"""
Configuration utilities for the Elastic Cloud Agent.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the application."""

    # OpenAI API configuration
    OPENAI_API_KEY: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))

    # Azure OpenAI configuration
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")

    # Elastic Cloud configuration
    ELASTIC_CLOUD_BASE_URL: str = os.getenv("ELASTIC_CLOUD_BASE_URL", "")
    ELASTIC_CLOUD_API_KEY: str = os.getenv("ELASTIC_CLOUD_API_KEY", "")

    # Path to OpenAPI specification
    API_SPEC_PATH: str = os.getenv(
        "API_SPEC_PATH", str(Path(__file__).parents[2] / "data" / "elastic_cloud_api.json")
    )

    @classmethod
    def is_azure_config(cls) -> bool:
        """
        Check if Azure OpenAI configuration is provided.

        Returns:
            bool: True if all Azure OpenAI environment variables are set
        """
        return bool(
            cls.AZURE_OPENAI_DEPLOYMENT
            and cls.AZURE_OPENAI_API_VERSION
            and cls.AZURE_OPENAI_ENDPOINT
        )

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate that all required configuration is present.

        Returns:
            tuple[bool, list[str]]: A tuple containing:
                - Boolean indicating if the configuration is valid
                - List of error messages if any
        """
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set in the environment")

        # Check Azure configuration if any Azure env vars are partially set
        has_azure_vars = any(
            [cls.AZURE_OPENAI_DEPLOYMENT, cls.AZURE_OPENAI_API_VERSION, cls.AZURE_OPENAI_ENDPOINT]
        )

        if has_azure_vars and not cls.is_azure_config():
            if not cls.AZURE_OPENAI_DEPLOYMENT:
                errors.append(
                    "AZURE_OPENAI_DEPLOYMENT is not set (required when using Azure OpenAI)"
                )
            if not cls.AZURE_OPENAI_API_VERSION:
                errors.append(
                    "AZURE_OPENAI_API_VERSION is not set (required when using Azure OpenAI)"
                )
            if not cls.AZURE_OPENAI_ENDPOINT:
                errors.append("AZURE_OPENAI_ENDPOINT is not set (required when using Azure OpenAI)")

        if not cls.ELASTIC_CLOUD_BASE_URL:
            errors.append("ELASTIC_CLOUD_BASE_URL is not set in the environment")

        if not cls.ELASTIC_CLOUD_API_KEY:
            errors.append("ELASTIC_CLOUD_API_KEY is not set in the environment")

        if not Path(cls.API_SPEC_PATH).exists():
            errors.append(f"API specification file not found at {cls.API_SPEC_PATH}")

        return len(errors) == 0, errors


def get_api_spec_path() -> Path:
    """
    Get the path to the OpenAPI specification file.

    Returns:
        Path: The path to the OpenAPI specification file.
    """
    path = Path(Config.API_SPEC_PATH)
    if not path.exists():
        raise FileNotFoundError(f"API specification file not found at {path}")
    return path
