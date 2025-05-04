import abc
import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Type

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
    ) -> Tuple[str, Optional[str]]: ...

    @abc.abstractmethod
    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]: ...

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
            mode_used = None
            try:
                raw, mode_used = self._generate_json_internal(prompt, response_model, **json_kwargs)

                if mode_used in ["json_schema", "json_object", "ollama_schema_format"]:
                    block = raw
                else:
                    block = self._extract_json_block(raw)

                return response_model.model_validate_json(block.strip())
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
            mode_used = None
            try:
                raw, mode_used = await self._generate_json_internal_async(
                    prompt, response_model, **json_kwargs
                )

                if mode_used in ["json_schema", "json_object", "ollama_schema_format"]:
                    block = raw
                else:
                    block = self._extract_json_block(raw)

                return response_model.model_validate_json(block.strip())
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
            await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"generate_json_async failed after {retries + 1} attempts. Last error: {type(last_error).__name__}: {last_error}"
        )
