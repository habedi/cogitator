"""Defines the abstract base class for LLM providers."""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class defining the interface for LLM providers."""

    def __init__(self) -> None:
        """Initializes token count storage and caching."""
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        self._cache: Dict[str, Any] = {}

    def get_last_prompt_tokens(self) -> Optional[int]:
        """Returns the token count for the last prompt, if available."""
        return self._last_prompt_tokens

    def get_last_completion_tokens(self) -> Optional[int]:
        """Returns the token count for the last completion, if available."""
        return self._last_completion_tokens

    def _reset_token_counts(self) -> None:
        """Resets the stored token counts."""
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

    def _create_cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Creates a cache key from the prompt and critical generation parameters."""
        # Filter out non-critical or mutable parameters
        critical_params = {
            "model",
            "seed",
            "stop",
            "stop_sequences",
            "temperature",
            "top_p",
            "max_tokens",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in critical_params}
        # Sort for consistency
        sorted_params = sorted(filtered_kwargs.items())

        # Combine prompt and params into a single string
        key_str = json.dumps({"prompt": prompt, "params": sorted_params})
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generates a single text completion for the given prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional provider-specific parameters (e.g., temperature,
                max_tokens, stop sequences, seed).

        Returns:
            The generated text completion as a string.

        Raises:
            RuntimeError: If the generation fails after retries or due to API errors.
        """
        ...

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generates a single text completion for the given prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text completion as a string.

        Raises:
            RuntimeError: If the asynchronous generation fails.
        """
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Generates a stream of text chunks for the given prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Strings representing chunks of the generated text.

        Raises:
            RuntimeError: If starting the stream generation fails.
        """
        ...

    @abstractmethod
    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously generates a stream of text chunks for the given prompt.

        Args:
            prompt: The input text prompt.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Strings representing chunks of the generated text asynchronously.

        Raises:
            RuntimeError: If starting the asynchronous stream generation fails.
        """
        ...

    @abstractmethod
    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Internal method to generate raw JSON output string from the LLM.

        This method handles the actual API call for JSON generation, potentially
        using provider-specific features like JSON mode or schema enforcement.
        It should also handle updating the internal token counts.

        Args:
            prompt: The input prompt, potentially instructing JSON format.
            response_model: The Pydantic model class for the expected response structure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A tuple containing:
                - The raw string response from the LLM, expected to be JSON.
                - An optional string indicating the JSON generation mode used (e.g.,
                  'json_schema', 'json_object', 'heuristic'), or None if extraction
                  is needed.

        Raises:
            RuntimeError: If the underlying LLM call fails.
        """
        ...

    @abstractmethod
    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        """Asynchronous internal method to generate raw JSON output string from the LLM.

        It should also handle updating the internal token counts.

        Args:
            prompt: The input prompt, potentially instructing JSON format.
            response_model: The Pydantic model class for the expected response structure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A tuple containing:
                - The raw string response from the LLM, expected to be JSON.
                - An optional string indicating the JSON generation mode used.

        Raises:
            RuntimeError: If the underlying asynchronous LLM call fails.
        """
        ...

    def _extract_json_block(self, text: str) -> str:
        """Extracts the first JSON object or array from a string.

        Handles JSON enclosed in markdown code fences (```json ... ``` or ``` ... ```)
        or finds the first substring starting with '{' and ending with '}' or
        starting with '[' and ending with ']'.

        Args:
            text: The string possibly containing a JSON block.

        Returns:
            The extracted JSON string, or the original text if no block is found.
        """
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            return fence_match.group(1)

        # Find the first standalone JSON object or array
        first_obj_start = text.find("{")
        first_arr_start = text.find("[")

        if first_obj_start == -1 and first_arr_start == -1:
            return text  # No JSON start found

        start_index = -1
        if first_obj_start != -1 and first_arr_start != -1:
            start_index = min(first_obj_start, first_arr_start)
        elif first_obj_start != -1:
            start_index = first_obj_start
        else:  # first_arr_start != -1
            start_index = first_arr_start

        # Attempt to find the matching end brace/bracket
        # This is a simplified approach and might fail for complex nested structures
        # if they appear outside the main intended JSON block.
        json_str = text[start_index:]
        try:
            # Try parsing to find the end implicitly
            parsed_obj, end_index = json.JSONDecoder().raw_decode(json_str)
            return json_str[:end_index]
        except json.JSONDecodeError:
            # Fallback: Search for the last brace/bracket if raw_decode fails
            # This is less reliable.
            last_brace = text.rfind("}")
            last_bracket = text.rfind("]")
            end_index = max(last_brace, last_bracket)
            if end_index > start_index:
                potential_json = text[start_index : end_index + 1]
                # Final check if this substring is valid JSON
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass  # Fall through if this substring isn't valid

        # If parsing/fallback fails, return the original text
        return text

    def generate_json(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        retries: int = 2,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> BaseModel:
        """Generates a response and parses it into a Pydantic model instance.

        Uses `_generate_json_internal` and attempts to parse the result.
        Retries on validation or decoding errors. Also updates internal token counts.

        Args:
            prompt: The input prompt, often instructing the LLM to respond in JSON.
            response_model: The Pydantic model class to validate the response against.
            retries: The number of times to retry on parsing/validation failure.
            use_cache: If True, enables caching for the request.
            **kwargs: Additional provider-specific parameters for generation.

        Returns:
            An instance of the `response_model` populated with data from the LLM response.

        Raises:
            RuntimeError: If parsing fails after all retries.
            ValidationError: If the final response does not match the `response_model`.
            json.JSONDecodeError: If the final response is not valid JSON.
        """
        if use_cache:
            cache_key = self._create_cache_key(
                prompt, response_model=response_model.model_json_schema(), **kwargs
            )
            if cache_key in self._cache:
                logger.debug("Cache hit for key: %s", cache_key)
                cached_data = self._cache[cache_key]
                # Assuming token counts are not essential for cached responses
                self._reset_token_counts()
                return response_model.model_validate(cached_data)

        last_error = None
        temp = kwargs.pop("temperature", 0.1)
        json_kwargs = {**kwargs, "temperature": temp}
        self._reset_token_counts()  # Reset before attempts

        for attempt in range(retries + 1):
            raw = ""
            block = ""
            mode_used = None
            try:
                # _generate_json_internal is responsible for updating token counts
                raw, mode_used = self._generate_json_internal(prompt, response_model, **json_kwargs)

                if mode_used in ["json_schema", "json_object", "ollama_schema_format"]:
                    # Assume the provider handled JSON enforcement
                    block = raw
                else:
                    # Fallback to extracting JSON block heuristically
                    block = self._extract_json_block(raw)

                validated_model = response_model.model_validate_json(block.strip())
                if use_cache:
                    # Cache the successful result
                    cache_key = self._create_cache_key(
                        prompt, response_model=response_model.model_json_schema(), **kwargs
                    )
                    self._cache[cache_key] = validated_model.model_dump()
                    logger.debug("Cached result for key: %s", cache_key)
                # Token counts should have been set by _generate_json_internal
                return validated_model
            except (json.JSONDecodeError, ValidationError) as ve:
                last_error = ve
                logger.warning(
                    "JSON validation/decode error %d/%d (mode: %s): %s\nBlock: %.200s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    mode_used,
                    ve,
                    block,
                    raw,
                )
                self._reset_token_counts()  # Reset counts on error
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating JSON %d/%d (mode: %s): %s",
                    attempt + 1,
                    retries + 1,
                    mode_used,
                    e,
                    exc_info=True,
                )
                self._reset_token_counts()  # Reset counts on error

            if attempt < retries:
                sleep_time = 2**attempt
                logger.info(f"Retrying JSON generation in {sleep_time} seconds...")
                time.sleep(sleep_time)
                self._reset_token_counts()  # Reset before retry

        # If loop finishes without success
        raise RuntimeError(
            f"generate_json failed after {retries + 1} attempts. Last error: {type(last_error).__name__}: {last_error}"
        )

    async def generate_json_async(
        self, prompt: str, response_model: Type[BaseModel], retries: int = 2, **kwargs: Any
    ) -> BaseModel:
        """Asynchronously generates a response and parses it into a Pydantic model instance.

        Uses `_generate_json_internal_async` and attempts to parse the result.
        Retries on validation or decoding errors. Also updates internal token counts.

        Args:
            prompt: The input prompt, often instructing the LLM to respond in JSON.
            response_model: The Pydantic model class to validate the response against.
            retries: The number of times to retry on parsing/validation failure.
            **kwargs: Additional provider-specific parameters for generation.

        Returns:
            An instance of the `response_model` populated with data from the LLM response.

        Raises:
            RuntimeError: If parsing fails after all retries.
            ValidationError: If the final response does not match the `response_model`.
            json.JSONDecodeError: If the final response is not valid JSON.
        """
        last_error = None
        temp = kwargs.pop("temperature", 0.1)
        json_kwargs = {**kwargs, "temperature": temp}
        self._reset_token_counts()  # Reset before attempts

        for attempt in range(retries + 1):
            raw = ""
            block = ""
            mode_used = None
            try:
                # _generate_json_internal_async is responsible for updating token counts
                raw, mode_used = await self._generate_json_internal_async(
                    prompt, response_model, **json_kwargs
                )

                if mode_used in ["json_schema", "json_object", "ollama_schema_format"]:
                    block = raw
                else:
                    block = self._extract_json_block(raw)

                validated_model = response_model.model_validate_json(block.strip())
                # Token counts should have been set by _generate_json_internal_async
                return validated_model
            except (json.JSONDecodeError, ValidationError) as ve:
                last_error = ve
                logger.warning(
                    "Async JSON validation/decode error %d/%d (mode: %s): %s\nBlock: %.200s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    mode_used,
                    ve,
                    block,
                    raw,
                )
                self._reset_token_counts()  # Reset counts on error
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating JSON async %d/%d (mode: %s): %s",
                    attempt + 1,
                    retries + 1,
                    mode_used,
                    e,
                    exc_info=True,
                )
                self._reset_token_counts()  # Reset counts on error

            if attempt < retries:
                sleep_time = 2**attempt
                logger.info(f"Retrying async JSON generation in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
                self._reset_token_counts()  # Reset before retry

        # If loop finishes without success
        raise RuntimeError(
            f"generate_json_async failed after {retries + 1} attempts. Last error: {type(last_error).__name__}: {last_error}"
        )
