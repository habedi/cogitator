import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type

import openai
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from pydantic import BaseModel

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    _STRUCTURED_OUTPUT_SUPPORTING_MODELS = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
    }

    _JSON_MODE_SUPPORTING_MODELS = {
                                       "gpt-4",
                                       "gpt-4-turbo",
                                       "gpt-4-turbo-preview",
                                       "gpt-3.5-turbo-1106",
                                       "gpt-3.5-turbo-0125",
                                   } | _STRUCTURED_OUTPUT_SUPPORTING_MODELS

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.client = SyncOpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        logger.info(f"Initialized OpenAILLM with model: {self.model}")

    def _prepare_api_params(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        params = kwargs.copy()
        mode_used: Optional[str] = None

        supports_structured = any(
            self.model.startswith(known) for known in self._STRUCTURED_OUTPUT_SUPPORTING_MODELS
        )
        supports_json_object = any(
            self.model.startswith(known) for known in self._JSON_MODE_SUPPORTING_MODELS
        )

        if is_json_mode:
            if response_schema:
                if supports_structured:
                    try:
                        schema_dict = response_schema.model_json_schema()
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
                            f"Using OpenAI Structured Outputs (json_schema) for model: {self.model}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate/set JSON schema for {response_schema.__name__}: {e}. Falling back."
                        )
                        if supports_json_object:
                            params["response_format"] = {"type": "json_object"}
                            mode_used = "json_object"
                            logger.debug(
                                f"Fell back to OpenAI JSON mode (json_object) after schema failure for model: {self.model}"
                            )
                        else:
                            mode_used = None
                            logger.debug(
                                "Fallback failed, JSON mode not supported. Relying on extraction."
                            )

                elif supports_json_object:
                    params["response_format"] = {"type": "json_object"}
                    mode_used = "json_object"
                    logger.debug(
                        f"Model {self.model} supports only json_object, using that despite schema being provided."
                    )

                else:
                    logger.warning(
                        f"Model {self.model} not known to support JSON modes. Attempting json_schema anyway as schema was provided..."
                    )
                    try:
                        schema_dict = response_schema.model_json_schema()
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
            else:
                if supports_json_object:
                    params["response_format"] = {"type": "json_object"}
                    mode_used = "json_object"
                    logger.debug("Using OpenAI JSON mode (json_object) as no schema provided.")
                else:
                    mode_used = None
                    logger.debug(
                        "JSON requested, no schema, model doesn't support json_object. Relying on extraction."
                    )
        else:
            mode_used = None

        if "seed" not in params and self.seed is not None:
            params["seed"] = self.seed

        return params, mode_used

    def _call_api(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[str]]:
        attempts = 0
        api_params, mode_used = self._prepare_api_params(
            is_json_mode=is_json_mode, response_schema=response_schema, **kwargs
        )
        while True:
            try:
                completion = self.client.chat.completions.create(**api_params)
                return completion, mode_used
            except openai.OpenAIError as e:
                attempts += 1
                if attempts > self.retry_attempts:
                    logger.error(f"OpenAI API call failed after {attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"OpenAI API error (attempt {attempts}/{self.retry_attempts + 1}): {e}. Retrying..."
                )
                time.sleep(self.retry_backoff * (2 ** (attempts - 1)))
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
                raise

    async def _call_api_async(
        self,
        is_json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[str]]:
        attempts = 0
        api_params, mode_used = self._prepare_api_params(
            is_json_mode=is_json_mode, response_schema=response_schema, **kwargs
        )
        while True:
            try:
                completion = await self.async_client.chat.completions.create(**api_params)
                return completion, mode_used
            except openai.OpenAIError as e:
                attempts += 1
                if attempts > self.retry_attempts:
                    logger.error(f"Async OpenAI API call failed after {attempts} attempts: {e}")
                    raise
                logger.warning(
                    f"Async OpenAI API error (attempt {attempts}/{self.retry_attempts + 1}): {e}. Retrying..."
                )
                await asyncio.sleep(self.retry_backoff * (2 ** (attempts - 1)))
            except Exception as e:
                logger.error(f"Unexpected error during async OpenAI API call: {e}", exc_info=True)
                raise

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
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
            raise RuntimeError("OpenAI returned empty choices or content")
        text = choices[0].message.content
        return text.strip()

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
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
            raise RuntimeError("Async OpenAI returned empty choices or content")
        text = choices[0].message.content
        return text.strip()

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),
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
            raise RuntimeError("OpenAI returned empty choices or content for JSON request")
        return choices[0].message.content, mode_used

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),
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
            raise RuntimeError("Async OpenAI returned empty choices or content for JSON request")
        return choices[0].message.content, mode_used

    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            "stream": True,
            **kwargs,
        }
        resp_stream, _ = self._call_api(is_json_mode=False, **call_kwargs)
        for chunk in resp_stream:
            if chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and delta.content:
                    yield delta.content

    async def generate_stream_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": stop or self.stop,
            "stream": True,
            **kwargs,
        }
        resp_stream, _ = await self._call_api_async(is_json_mode=False, **call_kwargs)
        async for chunk in resp_stream:
            if chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and delta.content:
                    yield delta.content
