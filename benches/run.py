#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import polars as pl
from datasets import load_dataset

from cogitator.auto_cot import AutoCoT
from cogitator.cdw_cot import CDWCoT
from cogitator.graph_of_thoughts import GraphOfThoughts
from cogitator.least_to_most import LeastToMost
from cogitator.model import BaseLLM, OllamaLLM, OpenAILLM
from cogitator.sc_cot import SelfConsistency
from cogitator.tree_of_thoughts import TreeOfThoughts
from cogitator.utils import accuracy

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)


class Datasets:
    registry = {
        "gsm8k": ("load_gsm8k", ["math"]),
        "multiarith": ("load_multiarith", ["math"]),
        "aqua": ("load_aqua_rat", ["math"]),
        "csqa": ("load_commonsense_qa", ["commonsense"]),
        "strategyqa": ("load_strategy_qa", ["commonsense", "symbolic"]),
        "coin": ("load_coin_flip", ["symbolic"]),
        "letter": ("load_last_letter", ["symbolic", "text"]),
    }

    @staticmethod
    def apply_cutoff(xs: List[Any], ys: List[Any], cutoff: Optional[int]):
        if cutoff is not None and cutoff >= 0:
            max_len = min(len(xs), len(ys), cutoff)
            return xs[:max_len], ys[:max_len]
        return xs, ys

    @staticmethod
    def load_dataset_by_name(name: str, cutoff: Optional[int]):
        if name not in Datasets.registry:
            raise ValueError(f"Dataset '{name}' not found in registry.")
        logger.info(f"Loading dataset: {name} (cutoff: {cutoff})")
        loader_name, _ = Datasets.registry[name]
        loader_fn = getattr(Datasets, loader_name)
        qs, golds = loader_fn()
        golds = [str(g) for g in golds]
        qs, golds = Datasets.apply_cutoff(qs, golds, cutoff)
        logger.info(f"Loaded {len(qs)} questions and {len(golds)} gold answers.")
        if len(qs) == 0:
            raise ValueError("Loaded 0 questions. Check dataset name and cutoff.")
        return qs, golds

    @staticmethod
    def load_gsm8k():
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        answers = []
        for answer_text in ds["answer"]:
            match = re.search(r"####\s*([+-]?\d+(?:,\d+)*\.?\d*)", answer_text)
            if match:
                extracted = match.group(1).replace(",", "")
                answers.append(extracted)
            else:
                logger.warning(f"Could not extract GSM8K answer from: {answer_text}")
                answers.append("[extraction_error]")
        return ds["question"], answers

    @staticmethod
    def load_multiarith():
        ds = load_dataset("ChilleD/MultiArith", split="train", trust_remote_code=True)
        return ds["question"], ds["final_ans"]

    @staticmethod
    def load_aqua_rat():
        ds = load_dataset("deepmind/aqua_rat", split="test", trust_remote_code=True)
        return ds["question"], [opt[0] for opt in ds["correct"]]

    @staticmethod
    def load_commonsense_qa():
        ds = load_dataset("tau/commonsense_qa", split="train", trust_remote_code=True)
        qs, golds = [], []
        for item in ds:
            qs.append(item["question"])
            try:
                idx = ord(item["answerKey"]) - ord("A")
                golds.append(item["choices"]["text"][idx])
            except (KeyError, IndexError, TypeError, ValueError):
                qs.pop()
                logger.warning(f"Skipping invalid item in commonsense_qa: {item.get('id', 'N/A')}")
        return qs, golds

    @staticmethod
    def load_strategy_qa():
        ds = load_dataset("voidful/StrategyQA", split="train", trust_remote_code=True)
        return ds["question"], ds["answer"]

    @staticmethod
    def load_coin_flip():
        ds = load_dataset("skrishna/coin_flip", split="train", trust_remote_code=True)
        return ds["question"], ds["answer"]

    @staticmethod
    def load_last_letter():
        ds = load_dataset("ChilleD/LastLetterConcat", split="train", trust_remote_code=True)
        cols = ds.column_names
        q_col, a_col = None, None
        if "question" in cols and "answer" in cols:
            q_col, a_col = "question", "answer"
        elif "input" in cols and "output" in cols:
            q_col, a_col = "input", "output"
        elif len(cols) >= 2:
            q_col, a_col = cols[0], cols[1]

        if q_col and a_col:
            return ds[q_col], ds[a_col]
        raise ValueError("Could not determine question/answer columns for LastLetterConcat")


def get_llm(provider: str, model_name: str, openai_key: Optional[str] = None) -> BaseLLM:
    logger.info(f"Initializing LLM: provider={provider}, model={model_name}")
    if provider == "openai":
        key = openai_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key must be provided via --openai-key or "
                "OPENAI_API_KEY environment variable."
            )
        return OpenAILLM(api_key=key, model=model_name)
    elif provider == "ollama":
        return OllamaLLM(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_final_answer(raw_output: str) -> str:
    if raw_output == "[ERROR]":
        return "[ERROR]"

    text = str(raw_output)
    num_pattern = r'[+-]?\d+(?:,\d+)*(?:\.\d+)?'

    patterns = [
        r'(?:final answer(?: is)?|answer:)\s*(' + num_pattern + r')',
        r'\\boxed\{(' + num_pattern + r')\}',
        r'(?:is|:|=)\s*(' + num_pattern + r')\s*\.?\s*$',
        ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")

    numbers = re.findall(num_pattern, text)
    if numbers:
        last_num = numbers[-1]
        marker_pattern = rf'(?:final answer(?: is)?|answer:|is|=)\s*{re.escape(last_num)}\s*\.?\s*$'
        if re.search(marker_pattern, text, re.IGNORECASE):
            return last_num.replace(",", "")

        lines = text.strip().splitlines()
        if lines:
            last_line = lines[-1]
            if re.fullmatch(r'\s*(' + num_pattern + r')\s*\.?\s*', last_line):
                return last_line.strip().replace(',','').rstrip('.')

        return last_num.replace(",", "")

    logger.debug(f"Could not extract numerical answer from: '{text[:100]}...'")
    return text


def log_single_result(
    show_details: bool,
    idx: int,
    name: str,
    mode: str,
    question: str,
    gold: str,
    raw_pred: str,
    extracted_pred: str,
    time_taken: float,
    is_correct: bool,
):
    if not show_details:
        return

    print(f"--- Q {idx + 1} ({name} {mode}) ---")
    print(f"  Q: {question}")
    print(f"  Gold: '{gold}'")
    print(f"  Raw Pred: '{raw_pred}'")
    print(f"  Extracted: '{extracted_pred}'")
    print(f"  Correct: {is_correct}")
    print(f"  Time: {time_taken:.2f}s")
    print("-" * 20)


def run_benchmark(
    name: str,
    model_func: Callable[[str], str],
    questions: List[str],
    golds: List[str],
    show_details: bool,
) -> Dict[str, Any]:
    logger.info(f"Running benchmark for method: {name} (sync)")
    extracted_preds: List[str] = []
    times: List[float] = []

    for i, q in enumerate(questions):
        gold_answer = golds[i]
        t0 = time.time()
        raw_out = "[ERROR]"
        extracted_out = "[ERROR]"
        is_correct = False
        time_taken = 0.0

        try:
            raw_out = model_func(q)
            extracted_out = extract_final_answer(raw_out)
            is_correct = accuracy([extracted_out], [gold_answer]) > 0
        except Exception as e:
            logger.error(f"Error running {name} on question {i}: {e}", exc_info=False)
        finally:
            t1 = time.time()
            time_taken = t1 - t0
            times.append(time_taken)
            extracted_preds.append(extracted_out)
            log_single_result(
                show_details, i, name, "Sync", q, gold_answer, raw_out, extracted_out, time_taken, is_correct
            )

    if not times:
        return {"method": name, "accuracy": 0.0, "avg_time_s": 0.0, "num_queries": 0}

    acc = accuracy(extracted_preds, golds)
    avg_time = sum(times) / len(times)
    logger.info(f"{name} - Final Accuracy: {acc:.3f}, Avg Time: {avg_time:.2f}s")
    return {
        "method": name,
        "accuracy": acc,
        "avg_time_s": avg_time,
        "num_queries": len(questions),
    }


async def run_benchmark_async(
    name: str,
    model_func: Callable[[str], Coroutine[Any, Any, str]],
    questions: List[str],
    golds: List[str],
    semaphore: asyncio.Semaphore,
    show_details: bool,
) -> Dict[str, Any]:
    logger.info(f"Running benchmark for method: {name} (async)")
    results_data: List[Dict[str, Any]] = [{"raw": "[ERROR]", "extracted": "[ERROR]", "time": 0.0} for _ in questions]

    async def run_single(idx: int, q: str):
        t0 = time.time()
        raw_output = "[ERROR]"
        extracted_output = "[ERROR]"
        try:
            async with semaphore:
                raw_output = await model_func(q)
            extracted_output = extract_final_answer(raw_output)
        except Exception as e:
            logger.error(f"Error running {name} on async question {idx}: {e}", exc_info=False)
        finally:
            time_taken = time.time() - t0
            results_data[idx] = {"raw": raw_output, "extracted": extracted_output, "time": time_taken}

    tasks = [run_single(i, q) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks)

    extracted_preds = [res["extracted"] for res in results_data]
    times = [res["time"] for res in results_data]

    for i, q in enumerate(questions):
        is_correct = accuracy([extracted_preds[i]], [golds[i]]) > 0
        log_single_result(
            show_details, i, name, "Async", q, golds[i], results_data[i]["raw"], extracted_preds[i], times[i], is_correct
        )

    if not times:
        return {"method": name, "accuracy": 0.0, "avg_time_s": 0.0, "num_queries": 0}

    acc = accuracy(extracted_preds, golds)
    valid_times = [t for t in times if t > 0]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0

    logger.info(f"{name} - Final Accuracy: {acc:.3f}, Avg Time: {avg_time:.2f}s")
    return {
        "method": name,
        "accuracy": acc,
        "avg_time_s": avg_time,
        "num_queries": len(questions),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Cogitatør Benchmarks")
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=list(Datasets.registry.keys()),
        help="Dataset to run benchmarks on (default: gsm8k)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=50,
        help="Number of samples to load (-1 for all, default: 50)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name of the model to use (e.g., llama3 for ollama, gpt-4o-mini for openai)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (reads OPENAI_API_KEY env var if not set)",
    )
    parser.add_argument(
        "--use-async", action="store_true", help="Run benchmarks asynchronously"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent requests for async mode (default: 5)",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed results (Q, Gold, Pred, Correct, Time) for each question during the run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more verbose output.",
    )

    args = parser.parse_args()

    if args.cutoff < 0:
        args.cutoff = None

    if not args.model_name:
        args.model_name = "gpt-4o-mini" if args.provider == "openai" else "llama3"
        logger.info(f"Model name not specified, using default for {args.provider}: {args.model_name}")

    return args


async def run_setup_phase(
    args: argparse.Namespace,
    llm: BaseLLM,
    methods_to_run: List[str],
    questions: List[str],
    golds: List[str],
    semaphore: asyncio.Semaphore,
    instances: Dict[str, Any],
) -> bool:
    logger.info("Running setup phase (fit/train)...")
    needs_auto_setup = any("AutoCoT" in name for name in methods_to_run)
    needs_cdw_setup = any("CDWCoT" in name for name in methods_to_run)

    if not needs_auto_setup and not needs_cdw_setup:
        logger.info("No methods requiring setup found. Skipping setup phase.")
        return True

    auto = instances["AutoCoT"]
    cdw = instances["CDWCoT"]

    try:
        setup_tasks = []
        if needs_auto_setup:
            if args.use_async:
                setup_tasks.append(auto.fit_async(questions, semaphore=semaphore))
            else:
                auto.fit(questions)

        if needs_cdw_setup:
            if args.use_async:
                async def cdw_setup_wrapper():
                    await cdw.init_pool_async(questions, golds, semaphore=semaphore)
                    await cdw.train_async(epochs=5, patience=2, semaphore=semaphore)
                setup_tasks.append(cdw_setup_wrapper())
            else:
                cdw.init_pool(questions, golds)
                cdw.train(epochs=5, patience=2)

        if args.use_async and setup_tasks:
            await asyncio.gather(*setup_tasks)

        logger.info("Setup phase complete.")
        return True
    except Exception as e:
        logger.error(f"Error during setup phase (fit/train): {e}", exc_info=True)
        logger.warning("Setup failed. Methods requiring setup will be skipped.")
        return False


def get_methods(
    args: argparse.Namespace, llm: BaseLLM, instances: Dict[str, Any]
) -> List[Tuple[str, Optional[Callable]]]:

    auto = instances["AutoCoT"]
    cdw = instances["CDWCoT"]
    scot = instances["SelfConsistency"]
    ltm = instances["LeastToMost"]
    tot = instances["TreeOfThoughts"]
    got = instances["GraphOfThoughts"]

    methods = [
        (
            "Zero-Shot-CoT",
            (lambda q: llm.generate_async(f"Q: {q}\nA: Let's think step by step."))
            if args.use_async
            else (lambda q: llm.generate(f"Q: {q}\nA: Let's think step by step."))
        ),
        ("AutoCoT", auto.answer_async if args.use_async else auto.answer),
        ("CDWCoT", cdw.answer_async if args.use_async else cdw.answer),
        ("SelfConsistency", scot.run_async if args.use_async else scot.run),
        ("LeastToMost", ltm.answer_async if args.use_async else ltm.answer),
        ("TreeOfThoughts", tot.run_async if args.use_async else tot.run),
        ("GraphOfThoughts", got.run_async if args.use_async else got.run),
    ]
    return methods


async def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger.info(f"Running benchmarks with config: {vars(args)}")

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
    instances = {
        "AutoCoT": AutoCoT(llm, n_demos=4, max_retries=1),
        "CDWCoT": CDWCoT(llm, seed=42, pool_size=10, n_clusters=3, sample_size=3),
        "SelfConsistency": SelfConsistency(llm, n_samples=3),
        "LeastToMost": LeastToMost(llm),
        "TreeOfThoughts": TreeOfThoughts(llm, sims=3, max_depth=2, num_branches=2),
        "GraphOfThoughts": GraphOfThoughts(llm, max_iters=2, num_branches=2, beam_width=2),
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    all_methods = get_methods(args, llm, instances)
    methods_to_run_names = [name for name, func in all_methods if func is not None]

    setup_successful = await run_setup_phase(
        args, llm, methods_to_run_names, questions, golds, semaphore, instances
    )

    results = []
    logger.info("Starting benchmarks execution...")
    for name, model_func_maybe in all_methods:
        if "AutoCoT" in name and (not setup_successful or not hasattr(instances["AutoCoT"], 'demos') or instances["AutoCoT"].demos is None):
            logger.warning(f"Skipping {name} because AutoCoT setup failed or wasn't run successfully.")
            continue
        if "CDWCoT" in name and (not setup_successful or not hasattr(instances["CDWCoT"], 'PC') or not instances["CDWCoT"].PC):
            logger.warning(f"Skipping {name} because CDWCoT setup failed or wasn't run successfully.")
            continue
        if model_func_maybe is None:
            logger.warning(f"Skipping {name} because model function is None.")
            continue

        model_func = model_func_maybe

        try:
            if args.use_async:
                async_func: Callable[[str], Coroutine[Any, Any, str]] = model_func
                result = await run_benchmark_async(
                    name, async_func, questions, golds, semaphore, args.show_details
                )
            else:
                sync_func: Callable[[str], str] = model_func
                result = run_benchmark(name, sync_func, questions, golds, args.show_details)
            results.append(result)
        except Exception as e:
            logger.error(f"Benchmark failed during execution for {name}: {e}", exc_info=True)

    if not results:
        logger.error("No benchmark results were collected.")
        sys.exit(1)

    df = pl.DataFrame(results)
    logger.info("Benchmarking complete. Final Results:")
    with pl.Config(tbl_width_chars=120, tbl_rows=len(df) + 1, tbl_formatting="ASCII_MARKDOWN"):
        print(df)


if __name__ == "__main__":
    asyncio.run(main())
