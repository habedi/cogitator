import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np  # Add import
import polars as pl

from benches.extractors import extract_answer_heuristic_custom, get_llm_extraction_prompt
from benches.shared import (
    setup_logging, get_llm,
    log_single_result,
    add_common_args, add_evaluation_args,
    load_and_merge_config,
    DEFAULT_OPENAI_ENV_VAR,
    FAILURE_MARKERS,
    EXTRACTION_OK,
    EXTRACTION_ERROR_MARKER,
    EXTRACTION_NULL_MARKER,
    EXTRACTION_HEURISTIC_FAILURE_MARKER,
    EXTRACTION_LLM_INIT_ERROR_MARKER,
    EXTRACTION_EXCEPTION_MARKER
)
from cogitator import BaseLLM, ExtractedAnswer
from cogitator.utils import accuracy

logger = logging.getLogger("benchmark_evaluate")


async def _run_llm_extraction_task(
    llm_instance: BaseLLM,
    semaphore: asyncio.Semaphore,
    prompt: str,
    args_dict: Dict[str, Any]
) -> str:
    """Helper coroutine to run a single LLM extraction task with semaphore."""
    async with semaphore:
        try:
            # Use generate_json_async from the LLM instance
            result = await llm_instance.generate_json_async(
                prompt,
                response_model=ExtractedAnswer,  # Use the predefined schema
                **args_dict  # Pass merged LLM parameters
            )
            # Process the result
            if result and result.final_answer is not None:
                extracted = str(result.final_answer).strip()
                # Specific handling for multiple choice labels
                if re.fullmatch(r'[A-Ea-e]', extracted):
                    return extracted.upper()
                return extracted
            else:
                logger.warning(
                    f"Async LLM extraction returned null for prompt starting with: {prompt[:50]}...")
                return EXTRACTION_NULL_MARKER
        except Exception as async_err:
            logger.error(f"Async LLM extraction task failed: {async_err}", exc_info=False)
            # Return a specific marker for exceptions during extraction
            return EXTRACTION_EXCEPTION_MARKER


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitator Benchmarks - Evaluation Phase")
    add_common_args(parser)
    add_evaluation_args(parser)
    args = parser.parse_args()

    config = load_and_merge_config(args, parser, config_section="evaluation")

    log_level = logging.DEBUG if config['debug'] else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Evaluation Phase with effective config: {config}")

    # --- Extractor LLM Initialization (if needed) ---
    extractor_llm: Optional[BaseLLM] = None
    if config['extractor_type'] == "llm":
        try:
            # Get API key respecting CLI > ENV precedence
            key_from_cli = getattr(args, 'openai_key', None)
            env_var_name = config.get('openai_key_env_var', DEFAULT_OPENAI_ENV_VAR)
            key_from_env = os.getenv(env_var_name) if env_var_name else None
            openai_api_key = key_from_cli or key_from_env

            # Initialize the LLM designated for extraction
            extractor_llm = get_llm(
                provider=config['extractor_provider'],
                model_name=config['extractor_model_name'],
                openai_key=openai_api_key,
                is_extractor=True,  # Flag for logging/potential different defaults
                ollama_host=config.get('extractor_ollama_host', None),
                llm_params=config.get('extractor_llm_params', {})  # Pass specific extractor params
            )
        except Exception as e:
            # Log error but continue, heuristic fallback might still work
            logger.error(f"Failed to initialize extractor LLM: {e}", exc_info=True)

    # --- Result Aggregation Setup ---
    # Initialize defaultdict to include token lists
    results_by_method = defaultdict(
        lambda: {
            "extracted": [], "times": [], "golds": [], "questions": [],
            "datasets": [], "prompt_tokens": [], "completion_tokens": []
        }
    )
    total_records = 0
    processed_records = 0
    input_filepath = config['input_file']

    logger.info(f"Reading results from {input_filepath}...")
    try:
        with open(input_filepath, 'r') as infile:
            # --- Read Records ---
            records_to_process = []
            for line in infile:
                total_records += 1
                try:
                    record = json.loads(line.strip())
                    records_to_process.append(record)
                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON line {total_records} in {input_filepath}")
                    logger.debug(f"Invalid line content: {line.strip()[:200]}...")

            # --- LLM Extraction (if enabled) ---
            llm_result_map = {}  # Map record index to extracted LLM result
            if config['extractor_type'] == "llm" and extractor_llm:
                logger.info("Preparing async LLM extraction tasks...")
                eval_concurrency = config.get("concurrency", 3)  # Concurrency for evaluation
                semaphore = asyncio.Semaphore(eval_concurrency)
                tasks_to_run = []
                record_indices_for_tasks = []  # Keep track of which record index each task corresponds to

                # Get extractor specific LLM parameters
                extractor_params = config.get('extractor_llm_params', {})
                ext_seed = extractor_params.get('seed')  # Allow specific seed for extractor
                ext_max_tokens = extractor_params.get('max_tokens', 64)  # Default small max tokens
                ext_temp = extractor_params.get('temperature', 0.1)  # Default low temp

                for i, record in enumerate(records_to_process):
                    try:
                        raw_output = record["raw_output"]
                        # Skip extraction if raw output itself indicates failure
                        if raw_output in FAILURE_MARKERS or not raw_output:
                            continue

                        question = record["question"]
                        dataset = record.get("dataset", "unknown")
                        # Get the appropriate prompt template based on dataset type
                        extraction_prompt = get_llm_extraction_prompt(question, raw_output, dataset)

                        # Prepare args for the extraction LLM call
                        extraction_call_args = {
                            "temperature": ext_temp,
                            "max_tokens": ext_max_tokens,
                            # Vary seed per record if base seed is provided
                            "seed": (ext_seed + i) if ext_seed is not None else None,
                            # Can add other extractor-specific overrides here
                        }

                        # Create the coroutine for the extraction task
                        coro = _run_llm_extraction_task(
                            extractor_llm,
                            semaphore,
                            extraction_prompt,
                            extraction_call_args
                        )
                        tasks_to_run.append(coro)
                        record_indices_for_tasks.append(i)

                    except KeyError as e:
                        logger.error(
                            f"Skipping task creation for record {i + 1} due to missing key: {e}")
                    except Exception as e:  # Catch other potential errors during task prep
                        logger.error(f"Unexpected error preparing task for record {i + 1}: {e}")

                # Run all extraction tasks concurrently
                if tasks_to_run:
                    logger.info(
                        f"Running {len(tasks_to_run)} async LLM extraction tasks with concurrency {eval_concurrency}...")
                    llm_extraction_results = await asyncio.gather(*tasks_to_run,
                                                                  return_exceptions=True)
                    logger.info("Async LLM extraction tasks complete.")
                    # Map results back to original record index
                    for task_idx, result in enumerate(llm_extraction_results):
                        record_idx = record_indices_for_tasks[task_idx]
                        if isinstance(result, Exception):
                            logger.error(
                                f"Async extraction task for record {record_idx + 1} resulted in exception: {result}")
                            llm_result_map[record_idx] = EXTRACTION_EXCEPTION_MARKER
                        else:
                            llm_result_map[
                                record_idx] = result  # Store successful extraction or specific failure marker
                else:
                    logger.info("No LLM extraction tasks needed.")

            # --- Process Each Record ---
            for i, record in enumerate(records_to_process):
                try:
                    question = record["question"]
                    gold = record["gold"]
                    method = record["method"]
                    raw_output = record["raw_output"]
                    time_taken = record.get("time", 0.0)
                    dataset = record.get("dataset", "unknown")
                    # Get token counts, default to None if missing
                    prompt_tokens = record.get("prompt_tokens")
                    completion_tokens = record.get("completion_tokens")

                    extracted_output = EXTRACTION_ERROR_MARKER  # Default to error
                    extraction_status = EXTRACTION_OK  # Assume ok unless changed
                    is_correct = False

                    # --- Answer Extraction Logic ---
                    if raw_output in FAILURE_MARKERS or not raw_output:
                        # If the generation itself failed, use that as the extracted output
                        extracted_output = raw_output if raw_output else "[EMPTY_RAW_OUTPUT]"
                        extraction_status = extracted_output  # Mark extraction status as failure
                    elif config['extractor_type'] == "llm":
                        if extractor_llm:
                            if i in llm_result_map:  # Check if LLM extraction was run and has result
                                extracted_output = llm_result_map[i]
                                # If LLM extraction returned a failure marker, update status
                                if extracted_output in FAILURE_MARKERS:
                                    extraction_status = extracted_output
                            else:
                                # LLM extraction wasn't run (e.g., raw was error) or result missing
                                # Fallback to heuristic only if raw output wasn't an error initially
                                logger.warning(
                                    f"LLM result missing for record {i + 1}, using heuristic fallback.")
                                try:
                                    extracted_output = extract_answer_heuristic_custom(raw_output,
                                                                                       dataset)
                                    if extracted_output == EXTRACTION_HEURISTIC_FAILURE_MARKER:
                                        extraction_status = EXTRACTION_HEURISTIC_FAILURE_MARKER
                                except Exception as h_err:  # Catch potential errors in heuristic
                                    extracted_output = EXTRACTION_EXCEPTION_MARKER
                                    extraction_status = EXTRACTION_EXCEPTION_MARKER
                                    logger.error(f"Heuristic fallback failed for Q{i + 1}: {h_err}")
                        else:
                            # Extractor LLM failed to initialize
                            extracted_output = EXTRACTION_LLM_INIT_ERROR_MARKER
                            extraction_status = EXTRACTION_LLM_INIT_ERROR_MARKER
                    else:  # extractor_type == "heuristic"
                        try:
                            extracted_output = extract_answer_heuristic_custom(raw_output, dataset)
                            # If heuristic returns its specific failure marker, update status
                            if extracted_output == EXTRACTION_HEURISTIC_FAILURE_MARKER:
                                extraction_status = EXTRACTION_HEURISTIC_FAILURE_MARKER
                        except Exception as heuristic_err:  # Catch unexpected errors in heuristic
                            logger.error(
                                f"Heuristic extraction failed for Q{i + 1}: {heuristic_err}",
                                exc_info=True)
                            extracted_output = EXTRACTION_EXCEPTION_MARKER
                            extraction_status = EXTRACTION_EXCEPTION_MARKER

                    # --- Accuracy Check ---
                    # Only consider correct if extraction was successful
                    if extraction_status == EXTRACTION_OK:
                        is_correct = accuracy([extracted_output], [gold]) > 0
                    else:
                        is_correct = False  # Mark as incorrect if extraction failed

                    # --- Store Results ---
                    results_by_method[method]["extracted"].append(extracted_output)
                    results_by_method[method]["times"].append(time_taken)
                    results_by_method[method]["golds"].append(gold)
                    results_by_method[method]["questions"].append(question)
                    results_by_method[method]["datasets"].append(dataset)
                    # Store token counts (can be None)
                    results_by_method[method]["prompt_tokens"].append(prompt_tokens)
                    results_by_method[method]["completion_tokens"].append(completion_tokens)

                    # --- Log Single Result Detail (if enabled) ---
                    log_single_result(
                        config['show_details'], i, method, f"Eval ({dataset})", question, gold,
                        raw_output, extracted_output, time_taken, is_correct
                    )
                    processed_records += 1

                except KeyError as e:
                    logger.error(
                        f"Skipping record {i + 1} due to missing key: {e} - Record: {record}")
                except Exception as e:
                    logger.error(f"Unexpected error processing record {i + 1}: {e}", exc_info=True)

    # --- Handle File Reading Errors ---
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filepath}")
        sys.exit(1)
    except Exception as e:  # Catch other file reading/parsing errors
        logger.error(f"Failed to read or process input file {input_filepath}: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Processed {processed_records}/{total_records} records from {input_filepath}.")

    if not results_by_method:
        logger.error("No valid results found in the input file.")
        sys.exit(1)

    # --- Calculate Final Summary ---
    final_summary = []
    for method, data in results_by_method.items():
        num_queries = len(data["golds"])
        if num_queries == 0:
            logger.warning(f"No data found for method: {method}")
            # Append with zero/default values
            final_summary.append({
                "method": method, "accuracy": 0.0, "avg_time_s": 0.0,
                "num_queries": 0, "successful_runs": 0, "extraction_failures": 0,
                "correct_extractions": 0, "successful_extractions": 0,
                "avg_prompt_tokens": 0.0, "avg_completion_tokens": 0.0, "avg_total_tokens": 0.0
            })
            continue

        # Identify successful extractions (not marked as failure)
        valid_indices = [idx for idx, ext in enumerate(data["extracted"]) if
                         ext not in FAILURE_MARKERS and ext != "[EMPTY_RAW_OUTPUT]"]
        successful_extractions = len(valid_indices)
        extraction_failures = num_queries - successful_extractions

        # Calculate accuracy based ONLY on successful extractions
        correct_extractions = 0
        if successful_extractions > 0:
            correct_extractions = sum(1 for idx in valid_indices if
                                      accuracy([data["extracted"][idx]], [data["golds"][idx]]) > 0)
            # Accuracy is correct / successfully_extracted
            acc = correct_extractions / successful_extractions
        else:
            acc = 0.0  # Accuracy is 0 if no successful extractions

        # Calculate average time for runs that recorded time
        valid_times = [t for t in data["times"] if t is not None and t > 0]
        successful_runs_timed = len(valid_times)
        avg_time = sum(valid_times) / successful_runs_timed if successful_runs_timed > 0 else 0.0

        # Calculate token stats (handling None values using np.nanmean)
        prompt_tokens_np = np.array([t if t is not None else np.nan for t in data["prompt_tokens"]],
                                    dtype=float)
        completion_tokens_np = np.array(
            [t if t is not None else np.nan for t in data["completion_tokens"]], dtype=float)
        total_tokens_np = prompt_tokens_np + completion_tokens_np  # NaN propagates

        avg_prompt_tokens = np.nanmean(prompt_tokens_np) if not np.all(
            np.isnan(prompt_tokens_np)) else 0.0
        avg_completion_tokens = np.nanmean(completion_tokens_np) if not np.all(
            np.isnan(completion_tokens_np)) else 0.0
        avg_total_tokens = np.nanmean(total_tokens_np) if not np.all(
            np.isnan(total_tokens_np)) else 0.0

        logger.info(
            f"{method} - Accuracy (on successful extractions): {acc:.3f} ({correct_extractions}/{successful_extractions}), "
            f"Avg Gen Time: {avg_time:.2f}s ({successful_runs_timed}/{num_queries} timed), "
            f"Extraction Failures: {extraction_failures}, "
            f"Avg Tokens (P/C/T): {avg_prompt_tokens:.0f} / {avg_completion_tokens:.0f} / {avg_total_tokens:.0f}"
        )
        # Append summary dictionary
        final_summary.append({
            "method": method,
            "accuracy": acc,
            "avg_time_s": avg_time,
            "num_queries": num_queries,
            "successful_runs": successful_runs_timed,  # Note: based on timing data, not extraction
            "extraction_failures": extraction_failures,
            "correct_extractions": correct_extractions,
            "successful_extractions": successful_extractions,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens
        })

    # --- Display Final Results Table ---
    try:
        df = pl.DataFrame(final_summary)
        # Update required cols to include token stats
        required_cols = [
            "method", "accuracy", "correct_extractions", "successful_extractions",
            "extraction_failures", "num_queries", "avg_time_s",
            "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens"
        ]
        present_cols = df.columns
        # Ensure all required columns exist, filling with 0 if necessary
        final_cols = []
        for col in required_cols:
            if col not in present_cols:
                dtype = pl.Float64 if col in ["accuracy", "avg_time_s", "avg_prompt_tokens",
                                              "avg_completion_tokens",
                                              "avg_total_tokens"] else pl.Int64
                df = df.with_columns(pl.lit(0).cast(dtype).alias(col))
                logger.debug(f"Added missing column '{col}' to summary DataFrame.")
            final_cols.append(col)

        logger.info("Evaluation complete. Final Results:")
        df = df.sort("accuracy", descending=True)
        # Format token columns as integers for display
        df = df.with_columns([
            pl.col("avg_prompt_tokens").round(0).cast(pl.Int64),
            pl.col("avg_completion_tokens").round(0).cast(pl.Int64),
            pl.col("avg_total_tokens").round(0).cast(pl.Int64),
        ])
        # Print the DataFrame using Polars pretty printing
        with pl.Config(tbl_width_chars=200, tbl_rows=len(df) + 1,
                       tbl_formatting="ASCII_MARKDOWN"):  # Increased width
            print(df.select(final_cols))  # Select columns in desired order
    except Exception as e:  # Catch potential Polars errors
        logger.error(f"Failed to create or display results DataFrame: {e}")
        # Fallback to printing the raw list of dictionaries
        logger.info("Raw summary list:")
        print(final_summary)


if __name__ == "__main__":
    asyncio.run(main())
