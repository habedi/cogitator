#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator.graph_of_thoughts import GraphOfThoughts
from cogitator.model import BaseLLM
from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_got(llm: BaseLLM) -> GraphOfThoughts:
    return GraphOfThoughts(
        llm, max_iters=2, num_branches=2, beam_width=2, use_json=True, merge_threshold=0.9
    )


QUESTIONS = [
    "A baker made 2 dozen cookies (24) and sold 8. How many left?",
    "If 7 times z equals 56, what is z?",
]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    got = setup_got(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Running GraphOfThoughts asynchronously...")
    tasks = [got.run_async(q, semaphore=semaphore) for q in QUESTIONS]
    answers = await asyncio.gather(*tasks)

    for q, a in zip(QUESTIONS, answers):
        print(f"Q: {q}\nA: {a}\n")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    got = setup_got(llm)

    logger.info("Running GraphOfThoughts synchronously...")
    for q in QUESTIONS:
        a = got.run(q)
        print(f"Q: {q}\nA: {a}\n")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Graph-of-Thoughts example")
