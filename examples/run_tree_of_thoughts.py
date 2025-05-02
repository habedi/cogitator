#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator.model import BaseLLM
from cogitator.tree_of_thoughts import TreeOfThoughts
from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_tot(llm: BaseLLM) -> TreeOfThoughts:
    return TreeOfThoughts(llm, max_depth=2, num_branches=2, sims=4, c_puct=1.0)


QUESTIONS = [
    "A garden has 4 rows of 6 plants each. How many plants total?",
    "If y minus 5 equals 10, what is y?",
]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    tot = setup_tot(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Running TreeOfThoughts asynchronously...")
    tasks = [tot.run_async(q, semaphore=semaphore) for q in QUESTIONS]
    answers = await asyncio.gather(*tasks)

    for q, a in zip(QUESTIONS, answers):
        print(f"Q: {q}\nA: {a}\n")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    tot = setup_tot(llm)

    logger.info("Running TreeOfThoughts synchronously...")
    for q in QUESTIONS:
        a = tot.run(q)
        print(f"Q: {q}\nA: {a}\n")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Tree-of-Thoughts example")
