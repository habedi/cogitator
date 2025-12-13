"""Provides an LLM provider implementation for interacting with Ollama models."""

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Type

from ollama import AsyncClient, Client
from pydantic import BaseModel

from .base import BaseLLM
from ..utils import approx_token_length  # Add import

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """LLM provider implementation for Ollama (https://ollama.com/).

    Allows interaction with LLMs served locally or remotely via the Ollama API.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",  # Changed default
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        ollama_host: Optional[str] = None,
    ) -> None:
        """Initializes the OllamaLLM provider.

        Args:
            model: The name of the Ollama model to use (e.g., "gemma3:4b", "llama3").
            temperature: The sampling temperature for generation.
            max_tokens: The maximum number of tokens to generate (`num_predict` in Ollama).
            stop: A list of stop sequences.
            seed: The random seed for generation.
            ollama_host: The host address of the Ollama server (e.g., "http://localhost:11434").
                         If None, the default host used by the `ollama` library is used.
        """
        super().__init__()  # Call BaseLLM init
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.host = ollama_host
        try:
            self._client = Client(host=self.host)
            self._async_client = AsyncClient(host=self.host)
            logger.debug(f"Ollama client initialized for host: {self.host}")
        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama client (host: {self.host}): {e}", exc_info=True
            )
            logger.warning(
                f"Could not establish initial connection to Ollama host: {self.host}. Client created, but connection may fail later."
            )

    def _strip_content(self, resp: Any) -> str:
        """Extracts the 'content' string from an Ollama chat response chunk/object.

        Args:
            resp: The response object or dictionary from the Ollama client.

        Returns:
            The extracted content string, or an empty string if not found.
        """
        content = ""
        try:
            if isinstance(resp, dict):
                message = resp.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
            elif hasattr(resp, "message") and hasattr(resp.message, "content"):
                content = getattr(resp.message, "content", "")
        except AttributeError as e:
            logger.warning(f"Could not extract content from Ollama response object: {e}")

        return str(content).strip() if isinstance(content, (str, int, float)) else ""

    def _prepare_options(self, **kwargs: Any) -> dict[str, Any]:
        """Prepares the 'options' dictionary for the Ollama API call.

        Merges default parameters with provided kwargs.

        Args:
            **kwargs: Overrides for temperature, max_tokens, seed, stop, etc.

        Returns:
            A dictionary of Ollama options.
        """
        opts = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_predict": kwargs.pop("max_tokens", self.max_tokens),
            "seed": kwargs.pop("seed", self.seed),
        }
        stop_list = kwargs.pop("stop", self.stop)
        if stop_list:
            opts["stop"] = stop_list

        opts.update(kwargs)

        if opts.get("seed") is not None:
            try:
                opts["seed"] = int(opts["seed"])
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert seed value {opts['seed']} to int. Removing seed from options."
                )
                # Don't set seed to None, remove it if invalid
                opts.pop("seed", None)

        return {k: v for k, v in opts.items() if v is not None}

    def _update_token_counts(self, prompt: str, resp: Any, completion_text: str) -> None:
        """Updates token counts using API response or approximation."""
        # Ollama response structure for non-streaming chat usually contains usage keys
        if isinstance(resp, dict):
            prompt_tokens = resp.get("prompt_eval_count")  # Input tokens
            completion_tokens = resp.get("eval_count")  # Output tokens
            if prompt_tokens is not None and completion_tokens is not None:
                try:
                    self._last_prompt_tokens = int(prompt_tokens)
                    self._last_completion_tokens = int(completion_tokens)
                    logger.debug(
                        f"Got token usage from Ollama API: P={self._last_prompt_tokens}, C={self._last_completion_tokens}"
                    )
                    return  # Use API values if available
                except (ValueError, TypeError):
                    logger.warning("Could not parse token counts from Ollama API response.")
        # Fallback to approximation if API counts are not available or parsing failed
        self._last_prompt_tokens = approx_token_length(prompt)
        self._last_completion_tokens = approx_token_length(completion_text)
        logger.debug(
            f"Estimated token usage via approx_token_length: P={self._last_prompt_tokens}, C={self._last_completion_tokens}"
        )

    def generate(self, prompt: str, use_cache: bool = False, **kwargs: Any) -> str:
        """Generates a single text completion using the configured Ollama model.

        Args:
            prompt: The input text prompt.
            use_cache: If True, enables caching for the request.
            **kwargs: Additional Ollama options (overrides defaults like temperature,
                max_tokens, seed, stop).

        Returns:
            The generated text completion.

        Raises:
            RuntimeError: If the Ollama API call fails.
        """
        if use_cache:
            cache_key = self._create_cache_key(prompt=prompt, **kwargs)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                self._reset_token_counts()
                return self._cache[cache_key]

        self._reset_token_counts()
        opts = self._prepare_options(**kwargs)
        try:
            resp = self._client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            content = self._strip_content(resp)
            self._update_token_counts(prompt, resp, content)  # Update counts
            if use_cache:
                cache_key = self._create_cache_key(prompt=prompt, **kwargs)
                self._cache[cache_key] = content
                logger.debug(f"Cached result for key: {cache_key}")
            return content
        except Exception as e:
            self._reset_token_counts()  # Reset on error
            logger.error(f"Ollama generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama generate failed: {e}") from e

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generates a single text completion using Ollama.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional Ollama options.

        Returns:
            The generated text completion.

        Raises:
            RuntimeError: If the asynchronous Ollama API call fails.
        """
        self._reset_token_counts()
        opts = self._prepare_options(**kwargs)
        try:
            resp = await self._async_client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            content = self._strip_content(resp)
            self._update_token_counts(prompt, resp, content)  # Update counts
            return content
        except Exception as e:
            self._reset_token_counts()  # Reset on error
            logger.error(f"Ollama async generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama async generate failed: {e}") from e

    def _make_response_options_and_schema(
        self, kwargs: Any, response_model: Type[BaseModel]
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Prepares options and JSON schema for structured output requests.

        Args:
            kwargs: Keyword arguments passed to the generation function.
            response_model: The Pydantic model defining the desired JSON structure.

        Returns:
            A tuple containing:
                - The Ollama options dictionary.
                - The JSON schema derived from the response_model.
        """
        # Note: Ollama's `format="json"` doesn't use the schema directly like OpenAI's structured output,
        # but we calculate it anyway in case future versions do or for consistency.
        schema = response_model.model_json_schema()
        opts = {
            "temperature": kwargs.pop("temperature", 0.1),
            "num_predict": kwargs.pop("max_tokens", self.max_tokens),
            "seed": kwargs.pop("seed", self.seed),
        }
        stop_list = kwargs.pop("stop", self.stop)
        if stop_list:
            opts["stop"] = stop_list

        opts.update(kwargs)
        # Ensure seed is int or removed
        if opts.get("seed") is not None:
            try:
                opts["seed"] = int(opts["seed"])
            except (ValueError, TypeError):
                opts.pop("seed", None)

        return {k: v for k, v in opts.items() if v is not None}, schema

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Internal method for Ollama JSON generation using the 'format' parameter.

        Args:
            prompt: The input prompt.
            response_model: The Pydantic model for the expected response.
            **kwargs: Additional Ollama options.

        Returns:
            A tuple containing:
                - The raw JSON string response from Ollama.
                - The string "ollama_schema_format" indicating the mode used.

        Raises:
            RuntimeError: If the Ollama API call fails.
        """
        self._reset_token_counts()
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        mode_used = "ollama_schema_format"
        logger.debug(f"Using Ollama structured output with schema for model: {self.model}")
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",  # Request JSON format
                options=opts,
            )
            # Ollama client library handles the JSON parsing when format='json'
            # The response structure contains the parsed content directly.
            raw_content = ""
            if isinstance(resp, dict) and resp.get("message"):
                raw_content = resp["message"].get("content", "")
            elif hasattr(resp, "message") and hasattr(resp.message, "content"):
                raw_content = getattr(resp.message, "content", "")

            # Update counts after successful call
            self._update_token_counts(prompt, resp, raw_content)
            return str(raw_content), mode_used
        except Exception as e:
            self._reset_token_counts()  # Reset on error
            logger.error(
                f"Ollama JSON generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama JSON generation failed: {e}") from e

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Asynchronous internal method for Ollama JSON generation.

        Args:
            prompt: The input prompt.
            response_model: The Pydantic model for the expected response.
            **kwargs: Additional Ollama options.

        Returns:
            A tuple containing:
                - The raw JSON string response from Ollama.
                - The string "ollama_schema_format" indicating the mode used.

        Raises:
            RuntimeError: If the asynchronous Ollama API call fails.
        """
        self._reset_token_counts()
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        mode_used = "ollama_schema_format"
        logger.debug(f"Using Ollama async structured output with schema for model: {self.model}")
        try:
            resp = await self._async_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",  # Request JSON format
                options=opts,
            )
            raw_content = ""
            if isinstance(resp, dict) and resp.get("message"):
                raw_content = resp["message"].get("content", "")
            elif hasattr(resp, "message") and hasattr(resp.message, "content"):
                raw_content = getattr(resp.message, "content", "")

            # Update counts after successful call
            self._update_token_counts(prompt, resp, raw_content)
            return str(raw_content), mode_used
        except Exception as e:
            self._reset_token_counts()  # Reset on error
            logger.error(
                f"Ollama async JSON generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama async JSON generation failed: {e}") from e

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Generates a stream of text chunks using the configured Ollama model.

        Note: Token counts are reset but not reliably updated during streaming.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional Ollama options.

        Yields:
            Strings representing chunks of the generated text.

        Raises:
            RuntimeError: If starting the stream generation fails.
        """
        self._reset_token_counts()  # Reset counts for stream
        opts = self._prepare_options(**kwargs)
        try:
            stream = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options=opts,
            )
            for chunk in stream:
                content = self._strip_content(chunk)
                if content:
                    yield content
            # Final usage might be in the last chunk's dict, but accessing it reliably
            # across library versions/models is tricky. Not implemented here.
        except Exception as e:
            logger.error(
                f"Ollama stream generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama stream generation failed: {e}") from e

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously generates a stream of text chunks using Ollama.

        Note: Token counts are reset but not reliably updated during streaming.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional Ollama options.

        Yields:
            Strings representing chunks of the generated text asynchronously.

        Raises:
            RuntimeError: If starting the asynchronous stream generation fails.
        """
        self._reset_token_counts()  # Reset counts for stream
        opts = self._prepare_options(**kwargs)
        try:
            stream = await self._async_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options=opts,
            )
            async for chunk in stream:
                content = self._strip_content(chunk)
                if content:
                    yield content
            # Final usage might be in the last chunk's dict
        except Exception as e:
            logger.error(
                f"Ollama async stream generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama async stream generation failed: {e}") from e
