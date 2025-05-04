#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator import BaseLLM, SelfConsistency

from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_sc(llm: BaseLLM) -> SelfConsistency:
    return SelfConsistency(
        llm,
        n_samples=5,
        temperature=0.7,
        max_tokens=100,
        stop=["\n\n"],
        internal_extraction_format="json",
    )


PROMPT = (
    "Q: A farmer had 17 sheep. All but 9 died. How many are left?\nA: Let's think step by step."
)


async def main_async(args: argparse.Namespace) -> None:
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    sc = setup_sc(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Running SelfConsistency asynchronously...")
    await sc.run_async(PROMPT, semaphore=semaphore)


def main_sync(args: argparse.Namespace) -> None:
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    sc = setup_sc(llm)

    logger.info("Running SelfConsistency synchronously...")
    sc.run(PROMPT)


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Self-Consistency example")
