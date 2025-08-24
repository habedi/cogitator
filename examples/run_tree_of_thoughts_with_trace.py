#!/usr/bin/env python3
import argparse
import asyncio
import logging

from cogitator import BaseLLM, TreeOfThoughts
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

    logger.info("Running TreeOfThoughts asynchronously with trace...")
    tasks = [tot.run_async(q, semaphore=semaphore, with_trace=True) for q in QUESTIONS]
    results = await asyncio.gather(*tasks)

    for q, (a, trace) in zip(QUESTIONS, results):
        print(f"Q: {q}\nA: {a}\n")
        if trace:
            print("--- TRACE ---")
            for node in trace.nodes:
                score_str = f"{node.score:.2f}" if node.score is not None else "N/A"
                print(
                    f"  Node {node.node_id} (Parent: {node.parent_id}): "
                    f"Score={score_str}, Visits={node.visits}, "
                    f"Content='{node.content[:50]}...'"
                )
            print("-------------\n")


def main_sync(args: argparse.Namespace):
    llm = get_llm(args.provider, args.model_name, args.openai_key)
    tot = setup_tot(llm)

    logger.info("Running TreeOfThoughts synchronously with trace...")
    for q in QUESTIONS:
        a, trace = tot.run(q, with_trace=True)
        print(f"Q: {q}\nA: {a}\n")
        if trace:
            print("--- TRACE ---")
            for node in trace.nodes:
                score_str = f"{node.score:.2f}" if node.score is not None else "N/A"
                print(
                    f"  Node {node.node_id} (Parent: {node.parent_id}): "
                    f"Score={score_str}, "
                    f"Visits={node.visits}, Content='{node.content[:50]}...'"
                )
            print("-------------\n")


if __name__ == "__main__":
    run_main(main_sync, main_async, "Run Tree-of-Thoughts with trace example")
