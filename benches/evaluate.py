#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import re
import sys
from collections import defaultdict
from typing import Optional

import polars as pl

from cogitator import BaseLLM
from cogitator.schemas import ExtractedAnswer
from cogitator.utils import accuracy
from extractors import extract_answer_heuristic_custom, get_llm_extraction_prompt
from shared import (
    setup_logging, get_llm,
    log_single_result,
    add_common_args, add_evaluation_args,
    RANDOM_SEED
)

logger = logging.getLogger("benchmark_evaluate")

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


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitatør Benchmarks - Evaluation Phase")
    add_common_args(parser)
    add_evaluation_args(parser)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Evaluation Phase with config: {vars(args)}")

    extractor_llm: Optional[BaseLLM] = None
    if args.extractor == "llm":
        try:
            if not args.extractor_model_name:
                if args.extractor_provider == "openai":
                    args.extractor_model_name = "gpt-4o-mini"
                else:
                    args.extractor_model_name = "llama3"
                logger.info(
                    f"Extractor model name not specified, using default for {args.extractor_provider}: {args.extractor_model_name}")

            extractor_llm = get_llm(
                args.extractor_provider,
                args.extractor_model_name,
                args.openai_key,
                is_extractor=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize extractor LLM: {e}", exc_info=True)
            extractor_llm = None

    results_by_method = defaultdict(
        lambda: {"extracted": [], "times": [], "golds": [], "questions": [], "datasets": []}
    )
    total_records = 0
    processed_records = 0

    logger.info(f"Reading results from {args.input_file}...")
    try:
        with open(args.input_file, 'r') as infile:
            # Use asyncio.gather for concurrent processing if LLM extractor is used
            # Note: File reading itself is sequential, but LLM calls can be concurrent
            # This requires restructuring to gather tasks, simplified here for clarity

            records_to_process = []
            for line in infile:
                total_records += 1
                try:
                    record = json.loads(line)
                    records_to_process.append(record)
                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON line {total_records} in {args.input_file}")

            for i, record in enumerate(records_to_process):
                try:
                    question = record["question"]
                    gold = record["gold"]
                    method = record["method"]
                    raw_output = record["raw_output"]
                    time_taken = record.get("time", 0.0)
                    dataset = record.get("dataset", "unknown")

                    if dataset == "unknown":
                        logger.warning(
                            f"Record {i + 1} missing 'dataset' field. Using generic extractors if applicable.")

                    extracted_output = EXTRACTION_ERROR_MARKER
                    extraction_status = EXTRACTION_OK
                    is_correct = False

                    try:
                        if args.extractor == "llm":
                            if extractor_llm:
                                extraction_prompt = get_llm_extraction_prompt(question, raw_output,
                                                                              dataset)
                                local_kwargs = {}
                                extraction_llm_args = {
                                    "temperature": local_kwargs.pop("temperature", 0.1),
                                    "max_tokens": local_kwargs.pop("max_tokens", 64),
                                    "seed": local_kwargs.pop("seed", RANDOM_SEED),
                                    **local_kwargs
                                }

                                result = await extractor_llm.generate_json_async(
                                    extraction_prompt,
                                    response_model=ExtractedAnswer,
                                    **extraction_llm_args
                                )

                                if result and result.final_answer is not None:
                                    extracted = str(result.final_answer).strip()
                                    if re.fullmatch(r'[A-Ea-e]', extracted):
                                        extracted_output = extracted.upper()
                                    else:
                                        extracted_output = extracted
                                    logger.debug(f"LLM extraction successful: '{extracted_output}'")
                                else:
                                    extracted_output = EXTRACTION_NULL_MARKER
                                    logger.warning(
                                        f"LLM extraction returned null for Q{i + 1} ({method}, {dataset})")
                                    extraction_status = EXTRACTION_NULL_MARKER
                            else:
                                extracted_output = EXTRACTION_LLM_INIT_ERROR_MARKER
                                extraction_status = EXTRACTION_LLM_INIT_ERROR_MARKER
                        else:
                            extracted_output = extract_answer_heuristic_custom(raw_output, dataset)
                            if extracted_output == EXTRACTION_HEURISTIC_FAILURE_MARKER:
                                extraction_status = EXTRACTION_HEURISTIC_FAILURE_MARKER

                    except Exception as extraction_err:
                        logger.error(
                            f"Error during extraction for Q{i + 1} ({method}, {dataset}): {extraction_err}",
                            exc_info=True)
                        extracted_output = EXTRACTION_EXCEPTION_MARKER
                        extraction_status = EXTRACTION_EXCEPTION_MARKER

                    if extraction_status == EXTRACTION_OK:
                        is_correct = accuracy([extracted_output], [gold]) > 0
                    else:
                        is_correct = False

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
                         ext not in FAILURE_MARKERS]
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


if __name__ == "__main__":
    asyncio.run(main())
