#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator import AutoCoT
from cogitator import BaseLLM
from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_auto_cot(llm: BaseLLM) -> AutoCoT:
    return AutoCoT(
        llm,
        n_demos=4,
        max_q_tokens=100,
        max_steps=8,
        max_retries=3,
    )


QUESTIONS_POOL = [
    "A merchant had 10 apples. He sold 3. How many remain?",
    "There are 7 days in a week. How many days in 3 weeks?",
    "If you buy 4 pens at $2 each, what's the total cost?",
    "A car travels 60 km in 1 hour. How far in 2.5 hours?",
    "A rectangle is 3 by 5. What is its area?",
    "5 birds are on a wire. 2 fly away. How many left?",
    "A baker made 12 buns and packed 3 per box. How many boxes?",
    "You read 20 pages per day. How many pages in 5 days?",
]

TEST_QUESTIONS = [
    "John has 8 oranges and gives 3 away. How many does he have?",
    "You run 5 km per day. How far in 7 days?",
]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    auto = setup_auto_cot(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Fitting AutoCoT asynchronously...")
    await auto.fit_async(QUESTIONS_POOL, semaphore=semaphore)

    logger.info("Running test questions asynchronously...")
    tasks = [auto.run_async(q) for q in TEST_QUESTIONS]
    answers = await asyncio.gather(*tasks)

    for q, a in zip(TEST_QUESTIONS, answers):
        print(f"Q: {q}\nA: {a}\n")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    auto = setup_auto_cot(llm)

    logger.info("Fitting AutoCoT synchronously...")
    auto.fit(QUESTIONS_POOL)

    logger.info("Running test questions synchronously...")
    for q in TEST_QUESTIONS:
        result = auto.run(q)
        print(f"Q: {q}\nA: {result}\n")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Auto-CoT example")
