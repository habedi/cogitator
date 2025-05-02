import abc
import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Iterator, List, Optional, Type

import openai
from ollama import AsyncClient, Client
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...

    @abc.abstractmethod
    async def generate_async(self, prompt: str, **kwargs: Any) -> str: ...

    @abc.abstractmethod
    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]: ...

    @abc.abstractmethod
    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]: ...

    @abc.abstractmethod
    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str: ...

    @abc.abstractmethod
    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str: ...

    def _extract_json_block(self, text: str) -> str:
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            return fence_match.group(1)
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
        if brace_match and bracket_match:
            return (
                brace_match.group(1)
                if brace_match.start() < bracket_match.start()
                else bracket_match.group(1)
            )
        if brace_match:
            return brace_match.group(1)
        if bracket_match:
            return bracket_match.group(1)
        return text

    def generate_json(
        self, prompt: str, response_model: Type[BaseModel], retries: int = 2, **kwargs: Any
    ) -> BaseModel:
        last_error = None
        temp = kwargs.pop("temperature", 0.1)
        json_kwargs = {**kwargs, "temperature": temp}

        for attempt in range(retries + 1):
            raw = ""
            block = ""
            try:
                raw = self._generate_json_internal(prompt, response_model, **json_kwargs)
                block = self._extract_json_block(raw)
                return response_model.model_validate_json(block.strip())
            except (json.JSONDecodeError, ValidationError) as ve:
                last_error = ve
                logger.warning(
                    "JSON validation/decode error %d/%d: %s\nBlock: %.200s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    ve,
                    block,
                    raw,
                )
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating JSON %d/%d: %s",
                    attempt + 1,
                    retries + 1,
                    e,
                    exc_info=True,
                )
            time.sleep(2**attempt)
        raise RuntimeError(
            f"generate_json failed after {retries + 1} attempts. Last error: {type(last_error).__name__}: {last_error}"
        )

    async def generate_json_async(
        self, prompt: str, response_model: Type[BaseModel], retries: int = 2, **kwargs: Any
    ) -> BaseModel:
        last_error = None
        temp = kwargs.pop("temperature", 0.1)
        json_kwargs = {**kwargs, "temperature": temp}

        for attempt in range(retries + 1):
            raw = ""
            block = ""
            try:
                raw = await self._generate_json_internal_async(
                    prompt, response_model, **json_kwargs
                )
                block = self._extract_json_block(raw)
                return response_model.model_validate_json(block.strip())
            except (json.JSONDecodeError, ValidationError) as ve:
                last_error = ve
                logger.warning(
                    "Async JSON validation/decode error %d/%d: %s\nBlock: %.200s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    ve,
                    block,
                    raw,
                )
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating JSON async %d/%d: %s",
                    attempt + 1,
                    retries + 1,
                    e,
                    exc_info=True,
                )
            await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"generate_json_async failed after {retries + 1} attempts. Last error: {type(last_error).__name__}: {last_error}"
        )


class OpenAILLM(BaseLLM):
    _JSON_MODE_SUPPORTING_MODELS = {
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
    }

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
    ):
        self.client = SyncOpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        is_known_json_model = any(
            known in self.model for known in self._JSON_MODE_SUPPORTING_MODELS
        )
        if not is_known_json_model:
            logger.warning(
                f"Model '{self.model}' may not support OpenAI JSON mode directly. "
                "JSON generation might rely on prompting and extraction."
            )

    def _prepare_api_params(self, is_json_mode: bool = False, **kwargs: Any) -> dict[str, Any]:
        params = kwargs.copy()
        supports_json = any(
            self.model.startswith(known) for known in self._JSON_MODE_SUPPORTING_MODELS
        )
        if is_json_mode and supports_json:
            params["response_format"] = {"type": "json_object"}
            logger.debug(f"Enabling OpenAI JSON mode for model: {self.model}")
        elif is_json_mode:
            logger.debug(
                f"OpenAI JSON mode not explicitly enabled for model: {self.model}. Relying on extraction."
            )

        if "seed" not in params and self.seed is not None:
            params["seed"] = self.seed
        return params

    def _call_api(self, is_json_mode: bool = False, **kwargs: Any) -> Any:
        attempts = 0
        api_params = self._prepare_api_params(is_json_mode=is_json_mode, **kwargs)
        while True:
            try:
                return self.client.chat.completions.create(**api_params)
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

    async def _call_api_async(self, is_json_mode: bool = False, **kwargs: Any) -> Any:
        attempts = 0
        api_params = self._prepare_api_params(is_json_mode=is_json_mode, **kwargs)
        while True:
            try:
                return await self.async_client.chat.completions.create(**api_params)
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
        resp = self._call_api(is_json_mode=False, **call_kwargs)
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"OpenAI response missing choices or content for prompt: {prompt[:100]}..."
            )
            # Consider raising an error or returning empty string based on desired behavior
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
        resp = await self._call_api_async(is_json_mode=False, **call_kwargs)
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
    ) -> str:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),
            **kwargs,
        }
        resp = self._call_api(is_json_mode=True, **call_kwargs)
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"OpenAI JSON response missing choices or content for prompt: {prompt[:100]}..."
            )
            raise RuntimeError("OpenAI returned empty choices or content for JSON request")
        return choices[0].message.content

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        call_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 0.1),
            **kwargs,
        }
        resp = await self._call_api_async(is_json_mode=True, **call_kwargs)
        choices = resp.choices or []
        if not choices or not choices[0].message or choices[0].message.content is None:
            logger.warning(
                f"Async OpenAI JSON response missing choices or content for prompt: {prompt[:100]}..."
            )
            raise RuntimeError("Async OpenAI returned empty choices or content for JSON request")
        return choices[0].message.content

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
        resp = self._call_api(is_json_mode=False, **call_kwargs)
        for chunk in resp:
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
        resp = await self._call_api_async(is_json_mode=False, **call_kwargs)
        async for chunk in resp:
            if chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and delta.content:
                    yield delta.content


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        ollama_host: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.host = ollama_host
        try:
            self._client = Client(host=self.host)
            self._async_client = AsyncClient(host=self.host)
            logger.debug(f"Checking available models on Ollama host: {self._client.list()}",
                         exc_info=True)
        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama client (host: {self.host}): {e}", exc_info=True
            )
            raise ConnectionError(f"Could not connect to Ollama host: {self.host}") from e

    def _strip_content(self, resp: Any) -> str:
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
        return content if isinstance(content, str) else ""

    def _prepare_options(self, **kwargs: Any) -> dict[str, Any]:
        opts = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_predict": kwargs.pop("max_tokens", self.max_tokens),
            "seed": kwargs.pop("seed", self.seed),
        }
        stop_list = kwargs.pop("stop", self.stop)
        if stop_list:
            opts["stop"] = stop_list

        opts.update(kwargs)
        # Filter out None values which might cause issues with the API
        return {k: v for k, v in opts.items() if v is not None}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        try:
            resp = self._client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            return self._strip_content(resp).strip()
        except Exception as e:
            logger.error(f"Ollama generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama generate failed: {e}") from e

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        try:
            resp = await self._async_client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            return self._strip_content(resp).strip()
        except Exception as e:
            logger.error(f"Ollama async generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama async generate failed: {e}") from e

    def _make_response_options_and_schema(self, kwargs: Any, response_model: Type[BaseModel]):
        # Use Pydantic v2 method for schema generation
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
        return {k: v for k, v in opts.items() if v is not None}, schema

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        try:
            # Passing the schema dict to 'format' is correct for ollama-python >= 0.2.0
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",  # Explicitly set format to json
                options=opts,
                # template=f"{{{{ .Prompt }}}}\nRespond strictly with JSON matching this schema:\n{json.dumps(schema)}" # Optionally guide model with schema
            )
            # Ollama with format='json' should return the JSON string directly in content
            return self._strip_content(resp)
        except Exception as e:
            logger.error(
                f"Ollama JSON generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama JSON generation failed: {e}") from e

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        try:
            resp = await self._async_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",  # Explicitly set format to json
                options=opts,
                # template=f"{{{{ .Prompt }}}}\nRespond strictly with JSON matching this schema:\n{json.dumps(schema)}"
            )
            return self._strip_content(resp)
        except Exception as e:
            logger.error(
                f"Ollama async JSON generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama async JSON generation failed: {e}") from e

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
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
        except Exception as e:
            logger.error(
                f"Ollama stream generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama stream generation failed: {e}") from e

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
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
        except Exception as e:
            logger.error(
                f"Ollama async stream generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama async stream generation failed: {e}") from e
