#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator import LeastToMost

from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

QUESTIONS = [
    "A box has 3 red balls and 5 blue balls. How many balls in total?",
    "If x plus 7 equals 12, what is x?",
]


async def main_async(args: argparse.Namespace) -> None:
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    ltm = LeastToMost(llm, intermediate_output_format="json")
    semaphore = asyncio.Semaphore(5)

    logger.info("Running LeastToMost asynchronously...")
    tasks = [ltm.run_async(q, semaphore=semaphore) for q in QUESTIONS]
    answers = await asyncio.gather(*tasks)

    for _q, _a in zip(QUESTIONS, answers, strict=False):
        pass


def main_sync(args: argparse.Namespace) -> None:
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    ltm = LeastToMost(llm, intermediate_output_format="json")

    logger.info("Running LeastToMost synchronously...")
    for q in QUESTIONS:
        ltm.run(q)


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Least-to-Most example")
