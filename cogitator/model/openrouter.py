"""Provides an LLM provider implementation for interacting with OpenRouter API."""

import logging
from typing import Any, List, Optional

from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI

from .openai import OpenAILLM

logger = logging.getLogger(__name__)


class OpenRouterLLM(OpenAILLM):
    """LLM provider implementation for OpenRouter API.

    OpenRouter provides access to 300+ models through an OpenAI-compatible API.
    This class extends OpenAILLM and configures it to use the OpenRouter endpoint.

    Model names follow the format: provider/model (e.g., "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct").

    Reference:
        OpenRouter API Documentation: https://openrouter.ai/docs
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> None:
        """Initializes the OpenRouterLLM provider.

        Args:
            api_key: Your OpenRouter API key.
            model: The model identifier in provider/model format
                   (e.g., "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet").
            temperature: The sampling temperature for generation.
            max_tokens: The maximum number of tokens to generate.
            stop: A list of sequences where the API will stop generation.
            seed: The random seed for reproducibility (if supported by the model).
            retry_attempts: Number of retries upon API call failure.
            retry_backoff: Initial backoff factor for retries (exponential).
            site_url: Optional URL of your site for OpenRouter tracking/rankings.
            site_name: Optional name of your site for OpenRouter tracking.
        """
        # Store OpenRouter-specific settings before calling parent init
        self._site_url = site_url
        self._site_name = site_name

        # Initialize parent class attributes without calling its __init__
        # We need to set up clients with custom base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        # Build default headers for OpenRouter
        default_headers: dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        # Create clients with OpenRouter base URL
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.OPENROUTER_BASE_URL,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self.client = SyncOpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        # Initialize base LLM attributes (caching, token counts)
        self._cache: dict[str, str] = {}
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None

        # For OpenRouter, we use a generic encoding since models vary widely
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}")
            self.encoding = None  # type: ignore[assignment]

        logger.info(f"Initialized OpenRouterLLM with model: {self.model}")

    def _reset_token_counts(self) -> None:
        """Resets the last recorded token counts."""
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

    def _create_cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Creates a hashable cache key from prompt and keyword arguments."""
        import hashlib
        import json

        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
