#!/usr/bin/env python3
import argparse
import asyncio
import logging
from typing import List, Dict, Tuple

from cogitator import BaseLLM, GraphOfThoughts, ThoughtExpansion
from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_got(llm: BaseLLM) -> GraphOfThoughts:
    return GraphOfThoughts(
        llm,
        final_answer_format="json",  # "text" or "json"
        # embedder can be added here if needed by GoO
    )


# Define a sample Graph of Operations (GoO)
# This defines the steps GoT will take. Adjust as needed for the problem.
# Example: Generate thoughts, score them, keep the best one.
EXAMPLE_GOO: List[Tuple[str, Dict]] = [
    ('Generate',
     {'k': 3, 'target_set': 'frontier', 'output_set': 'generated_thoughts', 'prompt_key': 'expand',
      'response_schema': ThoughtExpansion}),
    ('Score', {'target_set': 'generated_thoughts', 'prompt_key': 'evaluate'}),
    ('KeepBest', {'N': 1, 'target_set': 'generated_thoughts', 'output_set': 'frontier'})
    # Keep the best node in the frontier for the final answer
]

QUESTIONS = [
    "A baker made 2 dozen cookies (24) and sold 8. How many left?",
    "If 7 times z equals 56, what is z?",
]


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    got = setup_got(llm)
    semaphore = asyncio.Semaphore(5)  # Concurrency limit for LLM calls

    logger.info("Running GraphOfThoughts asynchronously...")
    tasks = [got.run_async(q, graph_of_operations=EXAMPLE_GOO, semaphore=semaphore) for q in
             QUESTIONS]
    answers = await asyncio.gather(*tasks)

    for q, a in zip(QUESTIONS, answers):
        print(f"Q: {q}\nA: {a}\n")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    got = setup_got(llm)

    logger.info("Running GraphOfThoughts synchronously...")
    for q in QUESTIONS:
        try:
            a = got.run(q, graph_of_operations=EXAMPLE_GOO)
            print(f"Q: {q}\nA: {a}\n")
        except NotImplementedError as e:
            logger.debug(f"GraphOfThoughts sync run failed correctly: {e}", exc_info=True)
            print("GraphOfThoughts run failed correctly."
                  " The implementation does not support synchronous execution.")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Graph-of-Thoughts example")
