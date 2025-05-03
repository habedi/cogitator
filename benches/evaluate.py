# benches/evaluate.py
import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from typing import Optional, Dict, Any  # Added Dict, Any

import polars as pl

from benches.extractors import extract_answer_heuristic_custom, get_llm_extraction_prompt
from benches.shared import (
    setup_logging, get_llm,
    log_single_result,
    add_common_args, add_evaluation_args,
    RANDOM_SEED
)
# Keep existing imports
from cogitator import BaseLLM, ExtractedAnswer  # Make sure ExtractedAnswer is imported
from cogitator.utils import accuracy

logger = logging.getLogger("benchmark_evaluate")

# Keep existing constants/markers
EXTRACTION_OK = "OK"
EXTRACTION_ERROR_MARKER = "[EXTRACTION_ERROR]"
EXTRACTION_NULL_MARKER = "[EXTRACTION_NULL]"
EXTRACTION_HEURISTIC_FAILURE_MARKER = "[EXTRACTION_HEURISTIC_FAILURE]"
EXTRACTION_LLM_INIT_ERROR_MARKER = "[EXTRACTION_LLM_INIT_ERROR]"
EXTRACTION_EXCEPTION_MARKER = "[EXTRACTION_EXCEPTION]"

FAILURE_MARKERS = [
    EXTRACTION_ERROR_MARKER,
    EXTRACTION_NULL_MARKER,
    EXTRACTION_HEURISTIC_FAILURE_MARKER,
    EXTRACTION_LLM_INIT_ERROR_MARKER,
    EXTRACTION_EXCEPTION_MARKER,
    "[ERROR]",
]


# --- Define the async extraction task function OUTSIDE main ---
async def _run_llm_extraction_task(
        llm_instance: BaseLLM,
        semaphore: asyncio.Semaphore,
        prompt: str,
        args_dict: Dict[str, Any]
) -> str:
    """Helper async function to run a single LLM extraction task."""
    async with semaphore:
        try:
            # Explicitly pass the response model here
            result = await llm_instance.generate_json_async(
                prompt,
                response_model=ExtractedAnswer,  # Pass the actual class
                **args_dict
            )
            if result and result.final_answer is not None:
                extracted = str(result.final_answer).strip()
                # Apply MCQ label formatting if needed
                if re.fullmatch(r'[A-Ea-e]', extracted):
                    return extracted.upper()
                return extracted
            else:
                logger.warning(f"Async LLM extraction returned null for prompt starting with: {prompt[:50]}...")
                return EXTRACTION_NULL_MARKER
        except Exception as async_err:
            logger.error(f"Async LLM extraction task failed: {async_err}", exc_info=False)
            return EXTRACTION_EXCEPTION_MARKER


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitatør Benchmarks - Evaluation Phase")
    add_common_args(parser)
    add_evaluation_args(parser)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Evaluation Phase with config: {vars(args)}")

    extractor_llm: Optional[BaseLLM] = None
    if args.extractor_type == "llm":
        try:
            if not args.model_name:
                # Default model logic remains the same
                default_model = "gpt-4o-mini" if args.provider == "openai" else "gemma3:4b"
                args.model_name = default_model
                logger.info(
                    f"Extractor model name not specified, using default for {args.provider}: {args.model_name}")

            # Call get_llm - this is where the mock will be injected during test
            extractor_llm = get_llm(
                args.provider,
                args.model_name,
                args.openai_key,
                is_extractor=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize extractor LLM: {e}", exc_info=True)
            extractor_llm = None  # Ensure it's None if init fails

    results_by_method = defaultdict(
        lambda: {"extracted": [], "times": [], "golds": [], "questions": [], "datasets": []}
    )
    total_records = 0
    processed_records = 0

    logger.info(f"Reading results from {args.input_file}...")
    try:
        with open(args.input_file, 'r') as infile:
            records_to_process = []
            for line in infile:
                total_records += 1
                try:
                    record = json.loads(line.strip())
                    records_to_process.append(record)
                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON line {total_records} in {args.input_file}")
                    logger.debug(f"Invalid line content: {line.strip()[:200]}...")

            # --- Prepare and run async LLM extraction tasks ---
            llm_result_map = {}
            if args.extractor_type == "llm" and extractor_llm:
                logger.info("Preparing async LLM extraction tasks...")
                concurrency_limit = getattr(args, 'concurrency', 3)
                semaphore = asyncio.Semaphore(concurrency_limit)
                tasks_to_run = []
                record_indices_for_tasks = []

                for i, record in enumerate(records_to_process):
                    try:
                        # Skip records where extraction is not needed/possible
                        raw_output = record["raw_output"]
                        if raw_output in FAILURE_MARKERS or not raw_output:
                            continue

                        question = record["question"]
                        dataset = record.get("dataset", "unknown")
                        extraction_prompt = get_llm_extraction_prompt(question, raw_output, dataset)
                        local_kwargs = {}  # Keep local_kwargs definition if needed
                        extraction_llm_args = {
                            "temperature": local_kwargs.pop("temperature", 0.1),
                            "max_tokens": local_kwargs.pop("max_tokens", 64),
                            "seed": local_kwargs.pop("seed", RANDOM_SEED),
                            **local_kwargs
                        }

                        # Create the coroutine using the helper function defined outside
                        coro = _run_llm_extraction_task(
                            extractor_llm,  # Pass the LLM instance
                            semaphore,
                            extraction_prompt,
                            extraction_llm_args
                        )
                        tasks_to_run.append(coro)
                        record_indices_for_tasks.append(i)  # Store index

                    except KeyError as e:
                        logger.error(f"Skipping task creation for record {i + 1} due to missing key: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error preparing task for record {i + 1}: {e}")

                # --- Execute async LLM extraction tasks ---
                if tasks_to_run:
                    logger.info(f"Running {len(tasks_to_run)} async LLM extraction tasks...")
                    llm_extraction_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
                    logger.info("Async LLM extraction tasks complete.")

                    # --- Create a map from record index to LLM result ---
                    for task_idx, result in enumerate(llm_extraction_results):
                        record_idx = record_indices_for_tasks[task_idx]
                        if isinstance(result, Exception):
                            logger.error(
                                f"Async extraction task for record {record_idx + 1} resulted in exception: {result}")
                            llm_result_map[record_idx] = EXTRACTION_EXCEPTION_MARKER
                        else:
                            llm_result_map[record_idx] = result  # Store the actual result (string or marker)
                else:
                    logger.info("No LLM extraction tasks needed.")

            # --- Process all records (using LLM results map if available) ---
            for i, record in enumerate(records_to_process):
                try:
                    question = record["question"]
                    gold = record["gold"]
                    method = record["method"]
                    raw_output = record["raw_output"]
                    time_taken = record.get("time", 0.0)
                    dataset = record.get("dataset", "unknown")

                    extracted_output = EXTRACTION_ERROR_MARKER  # Default
                    extraction_status = EXTRACTION_OK
                    is_correct = False

                    # Determine extracted output
                    if raw_output in FAILURE_MARKERS or not raw_output:
                        extracted_output = raw_output if raw_output else "[EMPTY_RAW_OUTPUT]"
                        extraction_status = extracted_output
                    elif args.extractor_type == "llm":
                        if extractor_llm:
                            if i in llm_result_map:  # Use pre-computed async result
                                extracted_output = llm_result_map[i]
                                if extracted_output in FAILURE_MARKERS:
                                    extraction_status = extracted_output
                            else:
                                # This case handles records skipped earlier (e.g., bad raw_output)
                                # or if task preparation failed. Assume heuristic fallback or error.
                                logger.warning(
                                    f"LLM result missing for record {i + 1}, using heuristic fallback or error state.")
                                # Fallback to heuristic for records that didn't run LLM extraction
                                try:
                                    extracted_output = extract_answer_heuristic_custom(raw_output, dataset)
                                    if extracted_output == EXTRACTION_HEURISTIC_FAILURE_MARKER:
                                        extraction_status = EXTRACTION_HEURISTIC_FAILURE_MARKER
                                except Exception as heuristic_err:
                                    extracted_output = EXTRACTION_EXCEPTION_MARKER
                                    extraction_status = EXTRACTION_EXCEPTION_MARKER

                        else:  # extractor_llm initialization failed
                            extracted_output = EXTRACTION_LLM_INIT_ERROR_MARKER
                            extraction_status = EXTRACTION_LLM_INIT_ERROR_MARKER
                    else:  # Heuristic case (default)
                        try:
                            extracted_output = extract_answer_heuristic_custom(raw_output, dataset)
                            if extracted_output == EXTRACTION_HEURISTIC_FAILURE_MARKER:
                                extraction_status = EXTRACTION_HEURISTIC_FAILURE_MARKER
                        except Exception as heuristic_err:
                            logger.error(f"Heuristic extraction failed for Q{i + 1}: {heuristic_err}", exc_info=True)
                            extracted_output = EXTRACTION_EXCEPTION_MARKER
                            extraction_status = EXTRACTION_EXCEPTION_MARKER

                    # Calculate correctness only if extraction was OK
                    if extraction_status == EXTRACTION_OK:
                        is_correct = accuracy([extracted_output], [gold]) > 0
                    else:
                        is_correct = False  # Mark as incorrect if extraction failed

                    # Store results
                    results_by_method[method]["extracted"].append(extracted_output)
                    results_by_method[method]["times"].append(time_taken)
                    results_by_method[method]["golds"].append(gold)
                    results_by_method[method]["questions"].append(question)
                    results_by_method[method]["datasets"].append(dataset)

                    log_single_result(
                        args.show_details, i, method, f"Eval ({dataset})", question, gold,
                        raw_output, extracted_output, time_taken, is_correct
                    )
                    processed_records += 1

                except KeyError as e:
                    logger.error(
                        f"Skipping record {i + 1} due to missing key: {e} - Record: {record}")
                except Exception as e:
                    logger.error(f"Unexpected error processing record {i + 1}: {e}", exc_info=True)

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read or process input file {args.input_file}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Processed {processed_records}/{total_records} records from {args.input_file}.")

    if not results_by_method:
        logger.error("No valid results found in the input file.")
        sys.exit(1)

    # --- Final Summary Calculation (remains the same) ---
    # ... (keep the existing summary logic) ...
    final_summary = []
    for method, data in results_by_method.items():
        num_queries = len(data["golds"])
        if num_queries == 0:
            logger.warning(f"No data found for method: {method}")
            final_summary.append({
                "method": method, "accuracy": 0.0, "avg_time_s": 0.0,
                "num_queries": 0, "successful_runs": 0, "extraction_failures": 0
            })
            continue

        valid_indices = [idx for idx, ext in enumerate(data["extracted"]) if
                         ext not in FAILURE_MARKERS and ext != "[EMPTY_RAW_OUTPUT]"]
        successful_extractions = len(valid_indices)
        extraction_failures = num_queries - successful_extractions

        correct_extractions = 0
        if successful_extractions > 0:
            correct_extractions = sum(1 for idx in valid_indices if
                                      accuracy([data["extracted"][idx]], [data["golds"][idx]]) > 0)
            acc = correct_extractions / successful_extractions
        else:
            acc = 0.0

        valid_times = [t for t in data["times"] if t is not None and t > 0]
        successful_runs_timed = len(valid_times)
        avg_time = sum(valid_times) / successful_runs_timed if successful_runs_timed > 0 else 0.0

        logger.info(
            f"{method} - Accuracy (on successful extractions): {acc:.3f} ({correct_extractions}/{successful_extractions}), "
            f"Avg Gen Time: {avg_time:.2f}s ({successful_runs_timed}/{num_queries} timed), "
            f"Extraction Failures: {extraction_failures}"
        )
        final_summary.append({
            "method": method,
            "accuracy": acc,
            "avg_time_s": avg_time,
            "num_queries": num_queries,
            "successful_runs": successful_runs_timed,
            "extraction_failures": extraction_failures,
            "correct_extractions": correct_extractions,
            "successful_extractions": successful_extractions,
        })

    # --- Display Results (remains the same) ---
    # ... (keep the existing display logic) ...
    try:
        df = pl.DataFrame(final_summary)
        required_cols = [
            "method", "accuracy", "correct_extractions", "successful_extractions",
            "extraction_failures", "num_queries", "avg_time_s"
        ]
        present_cols = df.columns
        final_cols = [col for col in required_cols if col in present_cols]

        for col in required_cols:
            if col not in df.columns:
                dtype = pl.Float64 if col in ["accuracy", "avg_time_s"] else pl.Int64
                df = df.with_columns(pl.lit(0).cast(dtype).alias(col))

        logger.info("Evaluation complete. Final Results:")
        df = df.sort("accuracy", descending=True)
        with pl.Config(tbl_width_chars=160, tbl_rows=len(df) + 1, tbl_formatting="ASCII_MARKDOWN"):
            print(df.select(final_cols))
    except Exception as e:
        logger.error(f"Failed to create or display results DataFrame: {e}")
        logger.info("Raw summary list:")
        print(final_summary)


# Keep the entry point
if __name__ == "__main__":
    asyncio.run(main())
