#!/usr/bin/env python3
import argparse
import asyncio
import logging
from typing import Dict, List, Tuple

from cogitator import BaseLLM, GraphOfThoughts, ThoughtExpansion
from examples.shared import get_llm, run_main, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TO_BE_SORTED_STR = "[0, 2, 6, 3, 8, 7, 1, 1, 6, 7, 7, 7, 7, 9, 3, 0, 1, 7, 9, 1, 3, 5, 1, 3, 6, 4, 5, 4, 7, 3, 5, 7]"

PROMPTS_FOR_SORTING: Dict[str, str] = {
    "generate_sort_attempt": (
        "The input is a string representation of a list of numbers: {ctx}\n"
        "Your task is to sort this list of numbers in ascending order. "
        "Return your single best sorted list as a JSON array containing one string, where that string is the sorted list. "
        "For example, if the input implies sorting [3,1,2], your output should be: {{\"thoughts\": [\"[1, 2, 3]\"]}}"
    ),
    "evaluate_list_format": (
        "Evaluate the following thought: \"{steps}\". "
        "Is it a well-formed string representation of a list of numbers (e.g., '[1, 2, 3]')? "
        "Score its quality from 1 (bad format/not a list) to 10 (perfectly formatted list of numbers). "
        "Provide a brief justification. "
        "Return as a JSON object with 'score' (int) and 'justification' (str)."
    )
}

SORTING_GOO_V2: List[Tuple[str, Dict]] = [
    ('Generate', {
        'k': 1,
        'target_set': 'frontier',
        'output_set': 'sort_attempts',
        'prompt_key': 'generate_sort_attempt',
        'response_schema': ThoughtExpansion
    }),
    ('Score', {
        'target_set': 'sort_attempts',
        'prompt_key': 'evaluate_list_format',
    }),
    ('KeepBest', {
        'N': 1,
        'target_set': 'sort_attempts',
        'output_set': 'frontier'
    })
]


def setup_got_v2(llm: BaseLLM) -> GraphOfThoughts:
    return GraphOfThoughts(
        llm,
        prompts=PROMPTS_FOR_SORTING,
        final_answer_format="direct_content",  # or "text" or "json"
    )


async def main_async(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    got_strategy = setup_got_v2(llm)
    semaphore = asyncio.Semaphore(args.concurrency if hasattr(args, 'concurrency') else 3)

    logger.info(
        f"Running GraphOfThoughts V2 (Sorting Example) asynchronously for: {TO_BE_SORTED_STR}")

    final_sorted_list_str = await got_strategy.run_async(
        question=TO_BE_SORTED_STR,
        graph_of_operations=SORTING_GOO_V2,
        semaphore=semaphore
    )

    print(f"\nUnsorted List: {TO_BE_SORTED_STR}")
    print(f"Final Sorted List (from GoT V2 async): {final_sorted_list_str}\n")


def main_sync(args: argparse.Namespace):
    logger.info(f"Running GraphOfThoughts V2 (Sorting Example) synchronously for:"
                f" {TO_BE_SORTED_STR}")
    logger.info("The implementation does not support synchronous execution."
                " Re-run with `--use-async` flag.")
    return None


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Graph-of-Thoughts V2 (Sorting) example")
