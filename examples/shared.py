import argparse
import asyncio
import logging
import os
from typing import Any, Callable, Coroutine, Optional

from cogitator import BaseLLM, OllamaLLM, OpenAILLM

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)


def get_llm(provider: str, model_name: str, openai_key: Optional[str] = None) -> BaseLLM:
    logger.info(f"Initializing LLM for examples: provider={provider}, model={model_name}")
    if provider == "openai":
        key = openai_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key must be provided via --openai-key or "
                "OPENAI_API_KEY environment variable."
            )
        return OpenAILLM(api_key=key, model=model_name)
    elif provider == "ollama":
        return OllamaLLM(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name of the model (default: 'gemma3:4b' for ollama, 'gpt-4.1-nano' for openai)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (reads OPENAI_API_KEY env var if not set)",
    )
    parser.add_argument(
        "--use-async", action="store_true", help="Run the asynchronous version of the example"
    )

    args = parser.parse_args()

    if not args.model_name:
        args.model_name = "gpt-4.1-nano" if args.provider == "openai" else "gemma3:4b"
        logger.info(
            f"Model name not specified, using default for {args.provider}: {args.model_name}"
        )

    return args


def run_main(
    main_sync_func: Callable[[argparse.Namespace], None],
    main_async_func: Callable[[argparse.Namespace], Coroutine[Any, Any, None]],
    description: str,
):
    args = parse_common_args(description)

    if args.use_async:
        asyncio.run(main_async_func(args))
    else:
        main_sync_func(args)
