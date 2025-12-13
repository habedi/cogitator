"""Provides an LLM provider implementation for interacting with OpenAI models."""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type

import openai
import tiktoken  # Add import
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from pydantic import BaseModel

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """LLM provider implementation for OpenAI API models.

    Handles interactions with models like GPT-4, GPT-4o, etc., supporting
    standard generation, streaming, JSON mode, and structured outputs where available.
    Includes retry logic for common API errors.

    Model capability detection can be configured via constructor parameters or
    will be auto-detected based on known model prefixes.
    """

    # Known model prefixes for capability auto-detection (used as fallback)
    _KNOWN_STRUCTURED_OUTPUT_PREFIXES = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gpt-5-mini",
    )

    _KNOWN_JSON_MODE_PREFIXES = (
        "gpt-4",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
        supports_structured_output: Optional[bool] = None,
        supports_json_mode: Optional[bool] = None,
    ) -> None:
        """Initializes the OpenAILLM provider.

        Args:
            api_key: Your OpenAI API key.
            model: The OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo").
            temperature: The sampling temperature for generation.
            max_tokens: The maximum number of tokens to generate.
            stop: A list of sequences where the API will stop generation.
            seed: The random seed for reproducibility (if supported by the model).
            retry_attempts: Number of retries upon API call failure.
            retry_backoff: Initial backoff factor for retries (exponential).
            supports_structured_output: Override for structured output capability.
                If None, auto-detects based on known model prefixes.
                Set True to force to enable, False to force to disable.
            supports_json_mode: Override for JSON mode capability.
                If None, auto-detects based on known model prefixes.
                Set True to force enable, False to force disable.
        """
        super().__init__()  # Call BaseLLM init
        self.client = SyncOpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        # Configure model capabilities (auto-detect or use explicit overrides)
        if supports_structured_output is not None:
            self._supports_structured_output = supports_structured_output
        else:
            self._supports_structured_output = any(
                model.startswith(prefix) for prefix in self._KNOWN_STRUCTURED_OUTPUT_PREFIXES
            )

        if supports_json_mode is not None:
            self._supports_json_mode = supports_json_mode
        else:
            # JSON mode is supported by structured output models + known JSON mode models
            self._supports_json_mode = self._supports_structured_output or any(
                model.startswith(prefix) for prefix in self._KNOWN_JSON_MODE_PREFIXES
            )

        # Load tiktoken encoding
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            logger.warning(f"No tiktoken encoding found for model {self.model}. Using cl100k_base.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(
            f"Initialized OpenAILLM with model: {self.model} (structured_output={self._supports_structured_output}, json_mode={self._supports_json_mode})"
        )

    def _update_token_counts(
        self, prompt: str, response: Any, completion_text: Optional[str]
    ) -> None:
        """Updates token counts using API response or tiktoken."""
        prompt_tokens = None
        completion_tokens = None
        source = "unknown"

        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            source = "api"

        # Fallback or verification using tiktoken
        if prompt_tokens is None or completion_tokens is None:
            source = "tiktoken"
            try:
                if prompt:
                    prompt_tokens = len(self.encoding.encode(prompt))
                if completion_text:
                    completion_tokens = len(self.encoding.encode(completion_text))
                else:
                    completion_tokens = 0  # Set completion to 0 if text is None or empty
            except Exception as e:
                logger.warning(f"tiktoken encoding failed during fallback: {e}", exc_info=False)
                # Keep existing API values if only fallback failed, otherwise reset
                if not (hasattr(response, "usage") and response.usage):
                    self._reset_token_counts()
                    return  # Exit if fallback fails and no API data exists

        # Store the determined values
        self._last_prompt_tokens = prompt_tokens
        self._last_completion_tokens = completion_tokens
        logger.debug(
            f"Token usage ({source}): P={self._last_prompt_tokens}, C={self._last_completion_tokens}"
        )

    def _prepare_api_params(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Prepares the parameters dictionary for the OpenAI API call.

        Determines the appropriate 'response_format' based on whether JSON mode
        or structured output is requested, the provided schema, and model support.

        Args:
            is_json_mode: Flag indicating if JSON output is requested.
            response_schema: The Pydantic model if structured output is desired.
            **kwargs: Additional parameters to pass to the API call, overriding defaults.

        Returns:
            A tuple containing:
                - The dictionary of parameters ready for the API call.
                - A string indicating the JSON mode used ('json_schema', 'json_object', None),
                  used for downstream processing logic.
        """
        params = kwargs.copy()
        mode_used: Optional[str] = None

        # Use instance-level capability flags
        supports_structured = self._supports_structured_output
        supports_json_object = self._supports_json_mode

        if is_json_mode:
            if response_schema:
                if supports_structured:
                    try:
                        schema_dict = response_schema.model_json_schema()
                        # Ensure additionalProperties is false for strictness if it's an object
                        if schema_dict.get("type") == "object":
                            schema_dict["additionalProperties"] = False
                        params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_schema.__name__,
                                "description": response_schema.__doc__
                                or f"Schema for {response_schema.__name__}",
                                "strict": True,  # Enable strict schema validation
                                "schema": schema_dict,
                            },
                        }
                        mode_used = "json_schema"
                        logger.debug(
                            f"Using OpenAI Structured Outputs (json_schema) for model: {self.model}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate/set JSON schema for {response_schema.__name__}: {e}. Falling back."
                        )
                        # Fallback to json_object if schema fails but model supports it
                        if supports_json_object:
                            params["response_format"] = {"type": "json_object"}
                            mode_used = "json_object"
                            logger.debug(
                                f"Fell back to OpenAI JSON mode (json_object) after schema failure for model: {self.model}"
                            )
                        else:
                            mode_used = None  # Cannot use JSON mode
                            logger.debug(
                                "Fallback failed, JSON mode not supported. Relying on extraction."
                            )

                elif supports_json_object:
                    # Model supports json_object but not full structured output, use json_object
                    params["response_format"] = {"type": "json_object"}
                    mode_used = "json_object"
                    logger.debug(
                        f"Model {self.model} supports only json_object, using that despite schema being provided."
                    )

                else:
                    # Model doesn't officially support either, but attempt structured output anyway if schema provided
                    logger.warning(
                        f"Model {self.model} not known to support JSON modes. Attempting json_schema anyway as schema was provided..."
                    )
                    try:
                        schema_dict = response_schema.model_json_schema()
                        # Ensure additionalProperties is false for strictness if it's an object
                        if schema_dict.get("type") == "object":
                            schema_dict["additionalProperties"] = False
                        params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_schema.__name__,
                                "description": response_schema.__doc__
                                or f"Schema for {response_schema.__name__}",
                                "strict": True,
                                "schema": schema_dict,
                            },
                        }
                        mode_used = "json_schema"
                        logger.debug(
                            "Attempting OpenAI Structured Outputs (json_schema) on potentially unsupported model..."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate/set JSON schema for unsupported model attempt: {e}. Relying on extraction."
                        )
                        mode_used = None
            else:  # is_json_mode is True, but no response_schema provided
                if supports_json_object:
                    params["response_format"] = {"type": "json_object"}
                    mode_used = "json_object"
                    logger.debug("Using OpenAI JSON mode (json_object) as no schema provided.")
                else:
                    mode_used = None  # Cannot use JSON mode
                    logger.debug(
                        "JSON requested, no schema, model doesn't support json_object. Relying on extraction."
                    )
        else:  # is_json_mode is False
            mode_used = None

        # Add seed if not present and set in instance
        if "seed" not in params and self.seed is not None:
            params["seed"] = self.seed

        # Ensure seed is an integer
        if params.get("seed") is not None:
            try:
                params["seed"] = int(params["seed"])
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert seed value {params['seed']} to int. Setting seed to None."
                )
                if "seed" in params:
                    del params["seed"]

        return params, mode_used

    def _call_api(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[str]]:
        """Makes a synchronous call to the OpenAI chat completions API with retries."""
        attempts = 0
        api_params, mode_used = self._prepare_api_params(
            is_json_mode=is_json_mode, response_schema=response_schema, **kwargs
        )
        # Get prompt text for token counting before potential modification/removal
        prompt_for_count = (
            api_params["messages"][-1]["content"] if api_params.get("messages") else ""
        )
        self._reset_token_counts()  # Reset before attempting call

        while True:
            try:
                completion = self.client.chat.completions.create(**api_params)
                # Extract completion text for token counting
                completion_text = ""
                if completion.choices:
                    msg = getattr(completion.choices[0], "message", None)
                    if msg:
                        completion_text = getattr(msg, "content", "") or ""
                # Update token counts using API response or tiktoken
                self._update_token_counts(prompt_for_count, completion, completion_text)
                return completion, mode_used
            except openai.OpenAIError as e:
                self._reset_token_counts()  # Reset counts on error
                attempts += 1
                if attempts > self.retry_attempts:
                    logger.error(f"OpenAI API call failed after {attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"OpenAI API error (attempt {attempts}/{self.retry_attempts + 1}): {e}. Retrying..."
                )
                time.sleep(self.retry_backoff * (2 ** (attempts - 1)))
            except Exception as e:
                self._reset_token_counts()  # Reset counts on unexpected error
                logger.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
                raise

    async def _call_api_async(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[str]]:
        """Makes an asynchronous call to the OpenAI chat completions API with retries."""
        attempts = 0
        api_params, mode_used = self._prepare_api_params(
            is_json_mode=is_json_mode, response_schema=response_schema, **kwargs
        )
        prompt_for_count = (
            api_params["messages"][-1]["content"] if api_params.get("messages") else ""
        )
        self._reset_token_counts()  # Reset before attempting call

        while True:
            try:
                completion = await self.async_client.chat.completions.create(**api_params)
                # Extract completion text for token counting
                completion_text = ""
                if completion.choices:
                    msg = getattr(completion.choices[0], "message", None)
                    if msg:
                        completion_text = getattr(msg, "content", "") or ""
                # Update token counts using API response or tiktoken
                self._update_token_counts(prompt_for_count, completion, completion_text)
                return completion, mode_used
            except openai.OpenAIError as e:
                self._reset_token_counts()  # Reset counts on error
                attempts += 1
                if attempts > self.retry_attempts:
                    logger.error(f"Async OpenAI API call failed after {attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"Async OpenAI API error (attempt {attempts}/{self.retry_attempts + 1}): {e}. Retrying..."
                )
                await asyncio.sleep(self.retry_backoff * (2 ** (attempts - 1)))
            except Exception as e:
                self._reset_token_counts()  # Reset counts on unexpected error
                logger.error(f"Unexpected error during async OpenAI API call: {e}", exc_info=True)
                raise

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generates a single text completion using the configured OpenAI model."""
        if use_cache:
            cache_key = self._create_cache_key(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                self._reset_token_counts()
                return self._cache[cache_key]

        # _reset_token_counts is handled by _call_api
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            **kwargs,
        }
        resp, _ = self._call_api(is_json_mode=False, **call_kwargs)
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"OpenAI response missing choices or content for prompt: {prompt[:100]}..."
            )
            # Reset counts if response is invalid, as _update might not have run correctly
            self._reset_token_counts()
            raise RuntimeError("OpenAI returned empty choices or content")
        text = choices[0].message.content.strip()
        if use_cache:
            cache_key = self._create_cache_key(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
            self._cache[cache_key] = text
            logger.debug(f"Cached result for key: {cache_key}")

        # Note: _update_token_counts was already called in _call_api
        return text

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generates a single text completion using OpenAI."""
        # _reset_token_counts is handled by _call_api_async
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            **kwargs,
        }
        resp, _ = await self._call_api_async(is_json_mode=False, **call_kwargs)
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"Async OpenAI response missing choices or content for prompt: {prompt[:100]}..."
            )
            # Reset counts if response is invalid
            self._reset_token_counts()
            raise RuntimeError("Async OpenAI returned empty choices or content")
        text = choices[0].message.content
        # Note: _update_token_counts was already called in _call_api_async
        return text.strip()

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Internal method for OpenAI JSON generation."""
        # _reset_token_counts is handled by _call_api
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),  # Default lower temp for JSON
            **kwargs,
        }
        resp, mode_used = self._call_api(
            is_json_mode=True, response_schema=response_model, **call_kwargs
        )
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"OpenAI JSON response missing choices or content for prompt: {prompt[:100]}..."
            )
            # Reset counts if response is invalid
            self._reset_token_counts()
            raise RuntimeError("OpenAI returned empty choices or content for JSON request")
        # Token counts updated within _call_api
        return choices[0].message.content, mode_used

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Asynchronous internal method for OpenAI JSON generation."""
        # _reset_token_counts is handled by _call_api_async
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),  # Default lower temp for JSON
            **kwargs,
        }
        resp, mode_used = await self._call_api_async(
            is_json_mode=True, response_schema=response_model, **call_kwargs
        )
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"Async OpenAI JSON response missing choices or content for prompt: {prompt[:100]}..."
            )
            # Reset counts if response is invalid
            self._reset_token_counts()
            raise RuntimeError("Async OpenAI returned empty choices or content for JSON request")
        # Token counts updated within _call_api_async
        return choices[0].message.content, mode_used

    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Generates a stream of text chunks using the configured OpenAI model.

        Note: Token counts are reset but not reliably updated during streaming.
        Use `get_last...` methods after streaming for potential approximations
        based on the prompt if needed, but completion counts will be unreliable.
        """
        self._reset_token_counts()  # Reset counts for stream start
        # Estimate prompt tokens before starting stream
        try:
            self._last_prompt_tokens = len(self.encoding.encode(prompt))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed for stream prompt: {e}")
            self._last_prompt_tokens = None

        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            "stream": True,
            **kwargs,
        }
        try:
            resp_stream = self.client.chat.completions.create(**call_kwargs)
            for chunk in resp_stream:
                if chunk.choices:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta and delta.content:
                        yield delta.content
            # Final usage stats are not typically available in the stream response object
        except openai.OpenAIError as e:
            logger.error(f"OpenAI stream API call failed: {e}")
            self._reset_token_counts()
            raise RuntimeError(f"OpenAI stream failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI stream call: {e}", exc_info=True)
            self._reset_token_counts()
            raise

    async def generate_stream_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously generates a stream of text chunks using OpenAI.

        Note: Token counts are reset but not reliably updated during streaming.
        """
        self._reset_token_counts()  # Reset counts for stream start
        # Estimate prompt tokens before starting stream
        try:
            self._last_prompt_tokens = len(self.encoding.encode(prompt))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed for async stream prompt: {e}")
            self._last_prompt_tokens = None

        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            "stream": True,
            **kwargs,
        }
        try:
            resp_stream = await self.async_client.chat.completions.create(**call_kwargs)
            async for chunk in resp_stream:
                if chunk.choices:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta and delta.content:
                        yield delta.content
            # Final usage stats are not typically available in the stream response object
        except openai.OpenAIError as e:
            logger.error(f"Async OpenAI stream API call failed: {e}")
            self._reset_token_counts()
            raise RuntimeError(f"Async OpenAI stream failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during async OpenAI stream call: {e}", exc_info=True)
            self._reset_token_counts()
            raise
