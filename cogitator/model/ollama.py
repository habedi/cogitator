import logging
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Type

from ollama import AsyncClient, Client
from pydantic import BaseModel

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gemma3:4b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = 33,
        ollama_host: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.seed = seed
        self.host = ollama_host
        try:
            self._client = Client(host=self.host)
            self._async_client = AsyncClient(host=self.host)
            logger.debug("Attempting to check Ollama models...")

            logger.debug(f"Ollama client initialized for host: {self.host}")
        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama client (host: {self.host}): {e}", exc_info=True
            )

            logger.warning(
                f"Could not establish initial connection to Ollama host: {self.host}. Client created, but connection may fail later."
            )

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

        return str(content).strip() if isinstance(content, (str, int, float)) else ""

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

        return {k: v for k, v in opts.items() if v is not None}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        try:
            resp = self._client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            return self._strip_content(resp)
        except Exception as e:
            logger.error(f"Ollama generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama generate failed: {e}") from e

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        opts = self._prepare_options(**kwargs)
        try:
            resp = await self._async_client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], options=opts
            )
            return self._strip_content(resp)
        except Exception as e:
            logger.error(f"Ollama async generate failed for model {self.model}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama async generate failed: {e}") from e

    def _make_response_options_and_schema(self, kwargs: Any, response_model: Type[BaseModel]):
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
    ) -> Tuple[str, Optional[str]]:
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        mode_used = "ollama_schema_format"
        logger.debug(f"Using Ollama structured output with schema for model: {self.model}")
        try:
            resp = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options=opts,
            )
            raw_content = ""
            if isinstance(resp, dict) and resp.get("message"):
                raw_content = resp["message"].get("content", "")
            elif hasattr(resp, "message") and hasattr(resp.message, "content"):
                raw_content = getattr(resp.message, "content", "")
            return raw_content, mode_used
        except Exception as e:
            logger.error(
                f"Ollama JSON generation failed for model {self.model}: {e}", exc_info=True
            )
            raise RuntimeError(f"Ollama JSON generation failed: {e}") from e

    async def _generate_json_internal_async(
        self, prompt: str, response_model: Type[BaseModel], **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        opts, schema = self._make_response_options_and_schema(kwargs, response_model)
        mode_used = "ollama_schema_format"
        logger.debug(f"Using Ollama async structured output with schema for model: {self.model}")
        try:
            resp = await self._async_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options=opts,
            )
            raw_content = ""
            if isinstance(resp, dict) and resp.get("message"):
                raw_content = resp["message"].get("content", "")
            elif hasattr(resp, "message") and hasattr(resp.message, "content"):
                raw_content = getattr(resp.message, "content", "")
            return raw_content, mode_used
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
