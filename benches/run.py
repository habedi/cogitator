#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from benches.shared import (
    setup_logging, Datasets, get_llm, RANDOM_SEED, MAX_TOKEN,
    add_common_args, add_generation_args
)
from cogitator import (
    AutoCoT, CDWCoT, GraphOfThoughts, LeastToMost, SelfConsistency,
    TreeOfThoughts, BaseLLM
)

logger = logging.getLogger("benchmark_run")


async def run_setup_phase(
    args: argparse.Namespace, llm: BaseLLM, methods_to_run: List[str],
    questions: List[str], golds: List[str], semaphore: asyncio.Semaphore,
    instances: Dict[str, Any]
) -> bool:
    logger.info("Running setup phase (fit/train)...")
    needs_auto_setup = any("AutoCoT" in name for name in methods_to_run)
    needs_cdw_setup = any("CDWCoT" in name for name in methods_to_run)
    if not needs_auto_setup and not needs_cdw_setup:
        logger.info("No methods requiring setup found. Skipping setup phase.")
        return True
    auto_instance = instances.get("AutoCoT")
    cdw_instance = instances.get("CDWCoT")
    if needs_auto_setup and not auto_instance: logger.error(
        "AutoCoT setup needed but instance not found."); return False
    if needs_cdw_setup and not cdw_instance: logger.error(
        "CDWCoT setup needed but instance not found."); return False
    try:
        setup_tasks = []
        if needs_auto_setup and auto_instance:
            if args.use_async:
                logger.info("Scheduling AutoCoT fit_async...")
                setup_tasks.append(
                    auto_instance.fit_async(questions, semaphore=semaphore))
            else:
                logger.info("Running AutoCoT fit...")
                auto_instance.fit(questions)
                logger.info("AutoCoT fit complete.")
        if needs_cdw_setup and cdw_instance:
            if args.use_async:
                async def cdw_setup_wrapper():
                    logger.info("Scheduling CDWCoT init_pool_async...")
                    await cdw_instance.init_pool_async(questions, golds, semaphore=semaphore)
                    logger.info("CDWCoT init_pool_async complete. Scheduling train_async...")
                    await cdw_instance.train_async(epochs=5, patience=2, semaphore=semaphore)
                    logger.info("CDWCoT train_async complete.")

                setup_tasks.append(cdw_setup_wrapper())
            else:
                logger.info("Running CDWCoT init_pool...")
                cdw_instance.init_pool(questions, golds)
                logger.info("CDWCoT init_pool complete. Running train...")
                cdw_instance.train(epochs=5, patience=2)
                logger.info("CDWCoT train complete.")
        if args.use_async and setup_tasks:
            logger.info(f"Awaiting {len(setup_tasks)} async setup tasks...")
            await asyncio.gather(*setup_tasks)
            logger.info("Async setup tasks complete.")
        logger.info("Setup phase complete.")
        return True
    except Exception as e:
        logger.error(f"Error during setup phase (fit/train): {e}", exc_info=True)
        logger.warning("Setup failed. Methods requiring setup will be skipped.")
        return False


def get_methods(
    args: argparse.Namespace, llm: BaseLLM, instances: Dict[str, Any]
) -> List[Tuple[str, Optional[Callable]]]:
    auto = instances.get("AutoCoT")
    cdw = instances.get("CDWCoT")
    scot = instances.get("SelfConsistency")
    ltm = instances.get("LeastToMost")
    tot = instances.get("TreeOfThoughts")
    got = instances.get("GraphOfThoughts")

    zero_shot_prompt = lambda q: f"Q: {q}\nA: Let's think step by step."

    methods = [
        ("Zero-Shot-CoT",
         (lambda q, **kw: llm.generate_async(zero_shot_prompt(q), **kw)) if args.use_async else
         (lambda q, **kw: llm.generate(zero_shot_prompt(q), **kw))),
        ("AutoCoT", auto.run_async if args.use_async else auto.run if auto else None),
        ("CDWCoT", cdw.run_async if args.use_async else cdw.run if cdw else None),
        ("SelfConsistency", scot.run_async if args.use_async else scot.run if scot else None),
        ("LeastToMost", ltm.run_async if args.use_async else ltm.run if ltm else None),
        ("TreeOfThoughts", tot.run_async if args.use_async else tot.run if tot else None),
        ("GraphOfThoughts", got.run_async if args.use_async else got.run if got else None),
    ]

    return [(name, func) for name, func in methods if func is not None]


async def run_single_async(q: str, model_func: Callable, semaphore: asyncio.Semaphore) -> Tuple[
    str, float]:
    t0 = time.time()
    raw_output = "[ERROR]"
    try:
        async with semaphore:
            raw_output = await model_func(q)
    except Exception as e:
        logger.error(f"Error during async generation: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
    time_taken = time.time() - t0
    return str(raw_output) if raw_output is not None else "[ERROR]", time_taken


def run_single_sync(q: str, model_func: Callable) -> Tuple[str, float]:
    t0 = time.time()
    raw_output = "[ERROR]"
    try:
        raw_output = model_func(q)
    except Exception as e:
        logger.error(f"Error during sync generation: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
    time_taken = time.time() - t0
    return str(raw_output) if raw_output is not None else "[ERROR]", time_taken


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitatør Benchmarks - Generation Phase")
    add_common_args(parser)
    add_generation_args(parser)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Generation Phase with config: {vars(args)}")

    if args.cutoff is not None and args.cutoff < 0: args.cutoff = None
    if not args.model_name:
        args.model_name = "gpt-4o-mini" if args.provider == "openai" else "gemma3:4b"
        logger.info(
            f"Generation model name not specified, using default for {args.provider}: {args.model_name}")

    try:
        questions, golds = Datasets.load_dataset_by_name(args.dataset, args.cutoff)
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset}': {e}", exc_info=True)
        sys.exit(1)

    try:
        llm = get_llm(args.provider, args.model_name, args.openai_key)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Initializing CoT strategies...")
    common_gen_params = {"max_tokens": MAX_TOKEN, "seed": RANDOM_SEED}
    sc_extraction_format = "json" if args.use_json else "heuristic"
    ltm_intermediate_format = "json" if args.use_json else "text"
    got_final_format = "json" if args.use_json else "text"

    instances = {
        "AutoCoT": AutoCoT(llm, n_demos=5, max_retries=3, max_tokens=MAX_TOKEN,
                           rand_seed=RANDOM_SEED),
        "CDWCoT": CDWCoT(llm, pool_size=10, n_clusters=5, sample_size=10, max_tokens=MAX_TOKEN,
                         seed=RANDOM_SEED),
        "SelfConsistency": SelfConsistency(llm, n_samples=10, **common_gen_params,
                                           internal_extraction_format=sc_extraction_format),
        "LeastToMost": LeastToMost(llm, **common_gen_params,
                                   intermediate_output_format=ltm_intermediate_format),
        "TreeOfThoughts": TreeOfThoughts(llm, sims=10, max_depth=3, num_branches=3,
                                         **common_gen_params),
        "GraphOfThoughts": GraphOfThoughts(llm, max_iters=5, num_branches=3, beam_width=5,
                                           **common_gen_params,
                                           final_answer_format=got_final_format),
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    all_methods = get_methods(args, llm, instances)
    methods_to_run_names = [name for name, func in all_methods if func is not None]

    setup_successful = await run_setup_phase(args, llm, methods_to_run_names, questions, golds,
                                             semaphore, instances)

    logger.info(f"Starting generation, saving to {args.output_file}...")
    generation_results = []

    with open(args.output_file, 'w') as outfile:
        if args.use_async:
            tasks = []
            for i, q in enumerate(questions):
                for name, model_func_maybe in all_methods:
                    if name == "AutoCoT" and (
                        not setup_successful or not instances.get("AutoCoT") or not getattr(
                        instances["AutoCoT"], 'demos', None)): continue
                    if name == "CDWCoT" and (
                        not setup_successful or not instances.get("CDWCoT") or not getattr(
                        instances["CDWCoT"], 'PC', None)): continue
                    if model_func_maybe is None: continue
                    tasks.append(run_single_async(q, model_func_maybe, semaphore))

            gen_outputs = await asyncio.gather(*tasks)
            task_idx = 0
            for i, q in enumerate(questions):
                for name, model_func_maybe in all_methods:
                    if name == "AutoCoT" and (
                        not setup_successful or not instances.get("AutoCoT") or not getattr(
                        instances["AutoCoT"], 'demos', None)): continue
                    if name == "CDWCoT" and (
                        not setup_successful or not instances.get("CDWCoT") or not getattr(
                        instances["CDWCoT"], 'PC', None)): continue
                    if model_func_maybe is None: continue

                    raw_output, time_taken = gen_outputs[task_idx]
                    record = {"question": q, "gold": golds[i], "method": name,
                              "dataset": args.dataset, "raw_output": raw_output, "time": time_taken}
                    outfile.write(json.dumps(record) + '\n')
                    task_idx += 1
                    logger.info(f"Generated: Q{i + 1} - {name} ({time_taken:.2f}s)")

        else:  # Sync execution
            for i, q in enumerate(questions):
                logger.info(f"Processing Question {i + 1}/{len(questions)}")
                for name, model_func_maybe in all_methods:
                    if name == "AutoCoT" and (
                        not setup_successful or not instances.get("AutoCoT") or not getattr(
                        instances["AutoCoT"], 'demos', None)):
                        logger.warning(f"Skipping {name} for Q{i + 1} (setup failed/incomplete).")
                        continue
                    if name == "CDWCoT" and (
                        not setup_successful or not instances.get("CDWCoT") or not getattr(
                        instances["CDWCoT"], 'PC', None)):
                        logger.warning(f"Skipping {name} for Q{i + 1} (setup failed/incomplete).")
                        continue
                    if model_func_maybe is None:
                        logger.warning(f"Skipping {name} for Q{i + 1} (instance missing).")
                        continue

                    logger.info(f"Running: Q{i + 1} - {name}")
                    raw_output, time_taken = run_single_sync(q, model_func_maybe)
                    record = {"question": q, "gold": golds[i], "method": name,
                              "dataset": args.dataset, "raw_output": raw_output, "time": time_taken}
                    outfile.write(json.dumps(record) + '\n')

    logger.info(f"Generation complete. Raw results saved to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
