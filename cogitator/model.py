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
        # pop any user temperature, default to 0.1
        temp = kwargs.pop("temperature", 0.1)
        json_kwargs = {**kwargs, "temperature": temp}

        for attempt in range(retries + 1):
            raw = ""
            block = ""
            try:
                raw = self._generate_json_internal(prompt, response_model, **json_kwargs)
                block = self._extract_json_block(raw)

                # coerce numeric final_answer → string
                block = re.sub(
                    r'("final_answer"\s*:\s*)(-?\d+(\.\d+)?)',
                    r'\1"\2"',
                    block,
                )
                # rename steps→thoughts for schemas expecting "thoughts"
                block = re.sub(r'"steps"\s*:', '"thoughts":', block)

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
                logger.error("Error generating JSON %d/%d: %s", attempt + 1, retries + 1, e)

            time.sleep(2**attempt)

        raise RuntimeError(f"generate_json failed after {retries + 1} attempts: {last_error}")

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
                block = re.sub(
                    r'("final_answer"\s*:\s*)(-?\d+(\.\d+)?)',
                    r'\1"\2"',
                    block,
                )
                block = re.sub(r'"steps"\s*:', '"thoughts":', block)

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
                logger.error("Error generating JSON async %d/%d: %s", attempt + 1, retries + 1, e)
            await asyncio.sleep(2**attempt)
        raise RuntimeError(f"generate_json_async failed after {retries + 1} attempts: {last_error}")


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.client = SyncOpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

    def _prepare_api_params(self, is_json_mode: bool = False, **kwargs: Any) -> dict[str, Any]:
        params = kwargs.copy()
        if is_json_mode and (
            "gpt-4" in self.model or "gpt-3.5-turbo-1106" in self.model or "gpt-4o" in self.model
        ):
            params["response_format"] = {"type": "json_object"}
        return params

    def _call_api(self, is_json_mode: bool = False, **kwargs: Any) -> Any:
        attempts = 0
        api_params = self._prepare_api_params(is_json_mode=is_json_mode, **kwargs)
        while True:
            try:
                return self.client.chat.completions.create(**api_params)
            except openai.OpenAIError:
                attempts += 1
                if attempts > self.retry_attempts:
                    raise
                time.sleep(self.retry_backoff * (2 ** (attempts - 1)))

    async def _call_api_async(self, is_json_mode: bool = False, **kwargs: Any) -> Any:
        attempts = 0
        api_params = self._prepare_api_params(is_json_mode=is_json_mode, **kwargs)
        while True:
            try:
                return await self.async_client.chat.completions.create(**api_params)
            except openai.OpenAIError:
                attempts += 1
                if attempts > self.retry_attempts:
                    raise
                await asyncio.sleep(self.retry_backoff * (2 ** (attempts - 1)))

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        resp = self._call_api(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("OpenAI missing choices")
        text = choices[0].message.content if choices[0].message else ""
        return text.strip()

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        resp = await self._call_api_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("Async OpenAI missing choices")
        text = choices[0].message.content if choices[0].message else ""
        return text.strip()

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        resp = self._call_api(
            is_json_mode=True,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("OpenAI missing JSON choices")
        return choices[0].message.content or ""

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        resp = await self._call_api_async(
            is_json_mode=True,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("Async OpenAI missing JSON choices")
        return choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        resp = self._call_api(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            stream=True,
            **kwargs,
        )
        for chunk in resp:
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
        resp = await self._call_api_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            stream=True,
            **kwargs,
        )
        async for chunk in resp:
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
        ollama_host: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.host = ollama_host
        self._client = Client(host=self.host)
        self._async_client = AsyncClient(host=self.host)

    def _strip_content(self, resp: Any) -> str:
        if isinstance(resp, dict) and resp.get("message"):
            return resp["message"].get("content", "")
        if hasattr(resp, "message") and getattr(resp.message, "content", None):
            return resp.message.content
        return ""

    def _prepare_options(self, **kwargs: Any) -> dict[str, Any]:
        opts = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_predict": kwargs.pop("max_tokens", self.max_tokens),
        }
        if self.stop:
            opts["stop"] = self.stop
        if "stop" in kwargs:
            opts["stop"] = kwargs.pop("stop")
        opts.update(kwargs)
        return opts

    def generate(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        resp = self._client.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
        )
        return self._strip_content(resp).strip()

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        resp = await self._async_client.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
        )
        return self._strip_content(resp).strip()

    def _make_response(self, kwargs: Any, response_model: Type[BaseModel]):
        schema = response_model.model_json_schema()
        opts = {
            "temperature": kwargs.pop("temperature", 0.1),
            "num_predict": kwargs.pop("max_tokens", self.max_tokens),
        }
        if self.stop:
            opts["stop"] = self.stop
        if "stop" in kwargs:
            opts["stop"] = kwargs.pop("stop")
        opts.update(kwargs)
        return opts, schema

    def _generate_json_internal(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        opts, schema = self._make_response(kwargs, response_model)
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            options=opts,
        )
        return self._strip_content(resp)

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> str:
        opts, schema = self._make_response(kwargs, response_model)
        resp = await self._async_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            options=opts,
        )
        return self._strip_content(resp)

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        opts = self._prepare_options(**kwargs)
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

    async def generate_stream_async(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        opts = self._prepare_options(**kwargs)
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
