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
        n_samples=10,
        temperature=0.8,  # More randomness leads to more diverse answers
        max_tokens=200,
        internal_extraction_format="json",
        # "heuristic" or "json" (JSON option is more robust but might not be supported by all LLMs)
    )


QUESTIONS = [
    "Q: A farmer had 17 sheep. All but 9 died. How many sheep are left?\nA: Let's think step by step.",
    "Q: If a train travels 60 miles in 1 hour, how far can it travel in 2.5 hours?\nA: Let's break it down."
]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    sc = setup_sc(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Running SelfConsistency concurrently for multiple questions...")

    async def run_single_question(prompt: str):
        logger.info(f"Processing async: {prompt[:50]}...")
        answer = await sc.run_async(prompt, semaphore=semaphore)
        print(f"\nPrompt: {prompt}")
        print(f"Final Answer (async self-consistency): {answer}")
        return answer

    tasks = [run_single_question(q) for q in QUESTIONS]
    await asyncio.gather(*tasks)
    logger.info("Async processing complete.")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    sc = setup_sc(llm)

    logger.info("Running SelfConsistency sequentially for multiple questions...")
    for i, prompt in enumerate(QUESTIONS):
        logger.info(f"Processing sync Q{i + 1}: {prompt[:50]}...")
        answer = sc.run(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Final Answer (sync self-consistency): {answer}")
    logger.info("Sync processing complete.")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Self-Consistency example")
