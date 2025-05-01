import argparse
import asyncio
import logging
import os
from typing import Any, Callable, Coroutine

from cogitator.model import BaseLLM, OllamaLLM, OpenAILLM
from cogitator.schemas import ExtractedAnswer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)


def get_llm(provider: str, openai_key: str, ollama_model: str) -> BaseLLM:
    if provider == "openai":
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable or --openai-key argument is required for OpenAI provider"
            )
        key = openai_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key must be provided via --openai-key or OPENAI_API_KEY environment variable."
            )
        base = OpenAILLM(api_key=key, model="gpt-4.1-nano")
    elif provider == "ollama":
        base = OllamaLLM(model=ollama_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return JSONOnlyLLM(base)


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY environment variable if set)",
    )
    parser.add_argument(
        "--ollama-model", default="gemma3", help="Ollama model name (default: gemma3)"
    )
    parser.add_argument(
        "--use-async", action="store_true", help="Run the asynchronous version of the example"
    )
    return parser.parse_args()


def run_main(
    main_sync_func: Callable[[argparse.Namespace], None],
    main_async_func: Callable[[argparse.Namespace], Coroutine[Any, Any, None]],
    description: str,
):
    args = parse_common_args(description)
    setup_logging()

    if args.use_async:
        asyncio.run(main_async_func(args))
    else:
        main_sync_func(args)


class JSONOnlyLLM:
    def __init__(self, real_llm):
        self._real = real_llm

    def generate(self, prompt: str, **kwargs) -> str:
        json_prompt = (
            prompt
            + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
        )
        kwargs.pop("temperature", None)
        parsed = self._real.generate_json(json_prompt, response_model=ExtractedAnswer, **kwargs)
        return parsed.final_answer

    async def generate_async(self, prompt: str, **kwargs) -> str:
        json_prompt = (
            prompt
            + '\n\nReturn exactly one JSON object with a single key "final_answer" whose value is the answer string.\n\nJSON Answer:'
        )
        kwargs.pop("temperature", None)
        parsed = await self._real.generate_json_async(
            json_prompt, response_model=ExtractedAnswer, **kwargs
        )
        return parsed.final_answer

    def generate_json(self, prompt: str, response_model: Any, **kwargs) -> Any:
        return self._real.generate_json(prompt, response_model=response_model, **kwargs)

    async def generate_json_async(self, prompt: str, response_model: Any, **kwargs) -> Any:
        return await self._real.generate_json_async(prompt, response_model=response_model, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
