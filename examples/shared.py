import argparse
import asyncio
import logging
import os
from typing import Any, Callable, Coroutine

from cogitator.model import BaseLLM, OllamaLLM, OpenAILLM


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        return OpenAILLM(api_key=key, model="gpt-4.1-nano")
    elif provider == "ollama":
        return OllamaLLM(model=ollama_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


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
