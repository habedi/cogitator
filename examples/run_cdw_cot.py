#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator.cdw_cot import CDWCoT
from cogitator.model import BaseLLM

from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_cdw_cot(llm: BaseLLM) -> CDWCoT:
    return CDWCoT(llm, pool_size=8, n_clusters=2, lr=0.1, temp=0.5, sample_size=3, seed=42)


TRAIN_QUESTIONS = [
    "A pot has 5 liters. You add 2 liters. How many liters now?",  # 7
    "If x+3=7, what is x?",  # 4
    "There are 4 pens in a box. How many pens in 3 boxes?",  # 12
    "You walk 2 km in 30 minutes. Distance in 1 hour?",  # 4
    "5 apples + 3 apples = ?",  # 8
    "Solve 2y = 10",  # 5
    "Area of 2x4 rectangle?",  # 8
    "Cost of 3 items at $5 each?",  # 15
]
TRAIN_ANSWERS = ["7", "4", "12", "4", "8", "5", "8", "15"]

TEST_QUERIES = ["If you have 3 boxes of 5 pens each, how many pens?", "Solve for y: y – 2 = 4"]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.openai_key, args.ollama_model)
    cdw = setup_cdw_cot(llm)
    semaphore = asyncio.Semaphore(5)

    logger.info("Initializing CDW-CoT pool asynchronously...")
    try:
        await cdw.init_pool_async(TRAIN_QUESTIONS, TRAIN_ANSWERS, semaphore=semaphore)
        logger.info("Training CDW-CoT asynchronously...")
        await cdw.train_async(val_split=0.4, epochs=5, patience=3, semaphore=semaphore)
        logger.info("Answering test questions asynchronously...")
        tasks = [cdw.answer_async(q, semaphore=semaphore) for q in TEST_QUERIES]
        answers = await asyncio.gather(*tasks)
        for q, a in zip(TEST_QUERIES, answers):
            print(f"Q: {q}\nA: {a}\n")
    except Exception as e:
        logger.error(f"CDW-CoT async example failed: {e}", exc_info=True)


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.openai_key, args.ollama_model)
    cdw = setup_cdw_cot(llm)

    logger.info("Initializing CDW-CoT pool synchronously...")
    try:
        cdw.init_pool(TRAIN_QUESTIONS, TRAIN_ANSWERS)
        logger.info("Training CDW-CoT synchronously...")
        cdw.train(val_split=0.4, epochs=5, patience=3)
        logger.info("Answering test questions synchronously...")
        for q in TEST_QUERIES:
            out = cdw.answer(q)
            print(f"Q: {q}\nA: {out}\n")
    except Exception as e:
        logger.error(f"CDW-CoT sync example failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run CDW-CoT example")
