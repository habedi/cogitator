import abc
import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Iterator, List, Optional

import ollama
import openai
from ollama import AsyncClient
from openai import AsyncOpenAI, OpenAI

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

    def generate_json(self, prompt: str, retries: int = 2, **kwargs: Any) -> Any:
        last_error = None
        for attempt in range(retries + 1):
            try:
                raw = self.generate(prompt, **kwargs)
                match = re.search(
                    r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, re.DOTALL | re.IGNORECASE
                )
                if match:
                    raw_json = match.group(1)
                else:
                    raw_json = raw

                return json.loads(raw_json.strip())
            except json.JSONDecodeError as je:
                last_error = je
                logger.warning(
                    "JSON decode error attempt %d/%d: %s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    je,
                    raw_json if "raw_json" in locals() else raw,
                )
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating JSON attempt %d/%d: %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
            time.sleep(2**attempt)
        raise RuntimeError(f"generate_json failed after {retries + 1} attempts: {last_error}")

    async def generate_json_async(self, prompt: str, retries: int = 2, **kwargs: Any) -> Any:
        last_error = None
        for attempt in range(retries + 1):
            try:
                raw = await self.generate_async(prompt, **kwargs)
                match = re.search(
                    r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, re.DOTALL | re.IGNORECASE
                )
                if match:
                    raw_json = match.group(1)
                else:
                    raw_json = raw

                return json.loads(raw_json.strip())
            except json.JSONDecodeError as je:
                last_error = je
                logger.warning(
                    "Async JSON decode error attempt %d/%d: %s\nRaw: %.200s",
                    attempt + 1,
                    retries + 1,
                    je,
                    raw_json if "raw_json" in locals() else raw,
                )
            except Exception as e:
                last_error = e
                logger.error(
                    "Error generating async JSON attempt %d/%d: %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
            await asyncio.sleep(2**attempt)
        raise RuntimeError(f"generate_json_async failed after {retries + 1} attempts: {last_error}")


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

    def _call_api(self, **kwargs: Any) -> Any:
        attempts = 0
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.OpenAIError as e:
                attempts += 1
                logger.warning("OpenAI API error %d/%d: %s", attempts, self.retry_attempts + 1, e)
                if attempts > self.retry_attempts:
                    raise
                backoff = self.retry_backoff * (2 ** (attempts - 1))
                time.sleep(backoff)

    async def _call_api_async(self, **kwargs: Any) -> Any:
        attempts = 0
        while True:
            try:
                return await self.async_client.chat.completions.create(**kwargs)
            except openai.OpenAIError as e:
                attempts += 1
                logger.warning(
                    "Async OpenAI API error %d/%d: %s", attempts, self.retry_attempts + 1, e
                )
                if attempts > self.retry_attempts:
                    raise
                backoff = self.retry_backoff * (2 ** (attempts - 1))
                await asyncio.sleep(backoff)

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
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("OpenAI missing choices")
        text = choices[0].message.content
        return text.strip() if text else ""

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
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            **kwargs,
        )
        choices = resp.choices or []
        if not choices:
            raise RuntimeError("Async OpenAI missing choices")
        text = choices[0].message.content
        return text.strip() if text else ""

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
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            stream=True,
            **kwargs,
        )
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and hasattr(delta, "content") and delta.content is not None:
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
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=stop or self.stop,
            stream=True,
            **kwargs,
        )
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and hasattr(delta, "content") and delta.content is not None:
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
        self._client = ollama.Client(host=self.host)
        self._async_client = AsyncClient(host=self.host)

    def _strip_content(self, resp: Any) -> str:
        if isinstance(resp, dict) and resp.get("message"):
            msg = resp["message"]
            return msg.get("content", "")
        elif hasattr(resp, "message") and hasattr(resp.message, "content"):
            return resp.message.content or ""
        return ""

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        opts = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.stop:
            opts["stop"] = self.stop
        opts.update(kwargs)
        resp = self._client.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
        )
        return self._strip_content(resp).strip()

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        opts = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.stop:
            opts["stop"] = self.stop
        opts.update(kwargs)
        resp = await self._async_client.chat(
            model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
        )
        return self._strip_content(resp).strip()

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        opts = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.stop:
            opts["stop"] = self.stop
        opts.update(kwargs)
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
        opts = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.stop:
            opts["stop"] = self.stop
        opts.update(kwargs)
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
