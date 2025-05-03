import argparse
import logging
import os
import re
from typing import Any, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

from cogitator import BaseLLM, OllamaLLM, OpenAILLM
from cogitator.schemas import ExtractedAnswer

logger = logging.getLogger("benchmark_shared")

MAX_TOKEN = 1024
RANDOM_SEED = 33

EXTRACTION_PROMPT_TEMPLATE = """
Original Question:
{question}

LLM Reasoning and Output:
\"\"\"
{raw_output}
\"\"\"

Analyze the LLM Reasoning and Output based on the Original Question.
Extract the definitive final answer stated in the text.
**IMPORTANT: If the original question is multiple choice and the reasoning identifies a specific choice label (e.g., A, B, C, D, E) as the answer, extract ONLY that single capital letter label.**
Otherwise, extract the numerical or short text answer.
Return the result as a JSON object with a single key "final_answer" containing the final answer as a string (or null if no answer found).
Avoid re-interpreting or re-solving the problem. Focus solely on extracting the answer provided in the text.

JSON Output:
"""


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
    # Store full HF path, config name (if any), category tags, and available splits
    registry = {
        "gsm8k": ("openai/gsm8k", "main", ["math"], ["train", "test"]),
        "multiarith": ("ChilleD/MultiArith", None, ["math"], ["train"]),
        "aqua": ("deepmind/aqua_rat", None, ["math"], ["train", "validation", "test"]),
        "csqa": ("tau/commonsense_qa", None, ["commonsense"], ["train", "validation"]),
        "strategyqa": ("ChilleD/StrategyQA", None, ["commonsense", "symbolic"], ["train"]),
        "coin": ("skrishna/coin_flip", None, ["symbolic"], ["train"]),
        "letter": ("ChilleD/LastLetterConcat", None, ["symbolic", "text"], ["train"]),
    }

    @staticmethod
    def apply_cutoff(xs: List[Any], ys: List[Any], cutoff: Optional[int]):
        if cutoff is not None and cutoff >= 0:
            max_len = min(len(xs), len(ys), cutoff)
            logger.info(f"Applying cutoff: Using {max_len} samples.")
            return xs[:max_len], ys[:max_len]
        logger.info(f"No cutoff applied. Using all {len(xs)} samples.")
        return xs, ys

    @staticmethod
    def _load_and_combine_splits(
            dataset_hf_path: str, config_name: Optional[str], available_splits: List[str]
    ) -> Dataset:
        loaded_splits = []
        # Prioritize loading 'train' and 'test' if they exist in available_splits
        splits_to_try = [s for s in available_splits if s in ["train", "test"]]
        # If neither train nor test are listed, try all available splits
        if not splits_to_try:
            splits_to_try = available_splits

        logger.info(f"Attempting to load splits for {dataset_hf_path}: {splits_to_try}")
        for split in splits_to_try:
            try:
                # Use the full hf_path and config_name here
                ds_split = load_dataset(
                    dataset_hf_path, config_name, split=split, trust_remote_code=True
                )
                if isinstance(ds_split, Dataset):
                    logger.info(f"Successfully loaded split '{split}' ({len(ds_split)} rows)")
                    loaded_splits.append(ds_split)
                else:
                    logger.warning(f"Loaded object for split '{split}' is not a Dataset: {type(ds_split)}")

            except ValueError as e:
                # Catch config errors specifically for gsm8k or similar
                if "Config name is missing" in str(e) and config_name is None:
                    logger.error(f"Dataset {dataset_hf_path} requires a config name, but None was provided.")
                logger.warning(f"Split '{split}' not found or failed to load for {dataset_hf_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading split '{split}' for {dataset_hf_path}: {e}", exc_info=True)

        if not loaded_splits:
            raise ValueError(f"Could not load any of the specified splits for dataset {dataset_hf_path}")

        if len(loaded_splits) > 1:
            logger.info(f"Concatenating {len(loaded_splits)} loaded splits...")
            combined_ds = concatenate_datasets(loaded_splits)
            logger.info(f"Concatenated dataset size: {len(combined_ds)} rows.")
            return combined_ds
        else:
            logger.info("Only one split loaded, using it directly.")
            return loaded_splits[0]

    @staticmethod
    def load_dataset_by_name(name: str, cutoff: Optional[int]):
        if name not in Datasets.registry:
            raise ValueError(f"Dataset '{name}' not found in registry.")

        # Unpack registry info: hf_path, config, tags, splits
        hf_path, config_name, _, available_splits = Datasets.registry[name]
        logger.info(
            f"Loading dataset: {name} (HF Path: {hf_path}, Config: {config_name}, Available Splits: {available_splits}, Cutoff: {cutoff})")

        # Load and combine splits first
        combined_ds = Datasets._load_and_combine_splits(hf_path, config_name, available_splits)

        # Now process the combined dataset based on its type (name)
        processor_fn_name = f"_process_{name}"  # Construct the expected processor name
        if not hasattr(Datasets, processor_fn_name):
            # Handle cases where the processor name might need adjustment (like coin_flip -> coin)
            # This is a fallback, the primary fix is renaming the method below.
            logger.warning(f"Processor function '{processor_fn_name}' not found directly. Attempting fallback.")
            # Example fallback (adjust if needed):
            # if name == "coin": processor_fn_name = "_process_coin" # Adjusted fallback
            # Add other mappings if necessary

        # Check again after potential fallback adjustment
        if not hasattr(Datasets, processor_fn_name):
            raise AttributeError(
                f"Could not find processor function for dataset '{name}' (tried '{processor_fn_name}')")

        processor_fn = getattr(Datasets, processor_fn_name)
        qs, golds = processor_fn(combined_ds)

        golds = [str(g) for g in golds]
        logger.info(f"Processed {len(qs)} total questions and {len(golds)} gold answers before cutoff.")

        qs, golds = Datasets.apply_cutoff(qs, golds, cutoff)
        logger.info(f"Final dataset size after cutoff: {len(qs)} samples.")

        if len(qs) == 0:
            raise ValueError("Loaded 0 questions after processing. Check dataset name and cutoff.")
        return qs, golds

    # --- Individual Dataset Processors ---
    # These methods now take the loaded Dataset object as input

    @staticmethod
    def _process_gsm8k(ds: Dataset):
        answers = []
        questions = []
        for item in ds:
            try:
                answer_text = item["answer"]
                question_text = item["question"]
                match = re.search(r"####\s*([+-]?\d+(?:,\d+)*\.?\d*)", answer_text)
                if match:
                    extracted = match.group(1).replace(",", "")
                    answers.append(extracted)
                    questions.append(question_text)
                else:
                    logger.warning(f"Could not extract GSM8K answer from: {answer_text}")
            except KeyError as e:
                logger.warning(f"Skipping item due to missing key {e} in gsm8k")
            except Exception as e:
                logger.warning(f"Skipping item due to error {e} in gsm8k")
        return questions, answers

    @staticmethod
    def _process_multiarith(ds: Dataset):
        return ds["question"], ds["final_ans"]

    # FIX: Renamed from _process_aqua_rat to _process_aqua
    @staticmethod
    def _process_aqua(ds: Dataset):
        questions = ds["question"]
        options_list = ds["options"]
        questions_with_options = [q + "\n" +
                                  "It's a multiple choice question. Pick the correct label (character before ')')\n" +
                                  " ".join(opts) for q, opts in zip(questions, options_list)]
        return questions_with_options, ds["correct"]

    # Renamed from _process_commonsense_qa to match key 'csqa'
    @staticmethod
    def _process_csqa(ds: Dataset):
        qs, golds = [], []
        for item in ds:
            try:
                question_text = item["question"]
                choices = item["choices"]["text"]
                answer_key = item["answerKey"]
                if not question_text or not choices or not answer_key or len(choices) < 1:
                    logger.warning(
                        f"Skipping incomplete item in commonsense_qa: {item.get('id', 'N/A')}")
                    continue

                question_with_choices = f"{question_text}\nChoices: {' '.join([f'({chr(ord('A') + i)}) {c}' for i, c in enumerate(choices)])}"
                qs.append(question_with_choices)

                idx = ord(answer_key) - ord("A")
                if 0 <= idx < len(choices):
                    golds.append(choices[idx])
                else:
                    qs.pop()
                    logger.warning(
                        f"Skipping invalid answer key '{answer_key}' for item in commonsense_qa: {item.get('id', 'N/A')}")
            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.warning(
                    f"Skipping invalid item in commonsense_qa due to error: {e} - Item: {item.get('id', 'N/A')}")
                if len(qs) > len(golds):
                    qs.pop()
        return qs, golds

    @staticmethod
    def _process_strategyqa(ds: Dataset):
        answers = ["yes" if ans else "no" for ans in ds["answer"]]
        return ds["question"], answers

    @staticmethod
    def _process_coin(ds: Dataset):
        return ds["inputs"], ds["targets"]

    # Renamed from _process_last_letter to match key 'letter'
    @staticmethod
    def _process_letter(ds: Dataset):
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


# --- Rest of the file (get_llm, extractors, args, etc.) remains the same ---
# ... (Keep the rest of the functions: get_llm, extract_final_answer, etc.) ...

def get_llm(provider: str, model_name: str, openai_key: Optional[str] = None,
            is_extractor: bool = False) -> BaseLLM:
    role = "Extractor" if is_extractor else "Primary"
    logger.info(
        f"Initializing {role} LLM: provider={provider}, model={model_name}, max_tokens={MAX_TOKEN}, seed={RANDOM_SEED}")
    common_kwargs = {"max_tokens": MAX_TOKEN, "seed": RANDOM_SEED}
    if provider == "openai":
        key = openai_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key must be provided via --openai-key or "
                "OPENAI_API_KEY environment variable."
            )
        return OpenAILLM(api_key=key, model=model_name, **common_kwargs)
    elif provider == "ollama":
        return OllamaLLM(model=model_name, **common_kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_final_answer(raw_output: str) -> str:
    text = str(raw_output).strip()
    if not text or text == "[ERROR]" or text.startswith("[ERROR:"):
        return "[ERROR]"

    mcq_patterns = [
        r'(?:answer|choice|option) is\s+([A-Ea-e])\b\.?$',
        r'final answer\s*:\s*([A-Ea-e])\b\.?$',
        r'\(([A-Ea-e])\)$',
        r'correct option is\s+([A-Ea-e])\b',
        r'\b([A-Ea-e])\s*is the correct answer',
    ]
    lines = text.strip().splitlines()
    if lines:
        last_line = lines[-1].strip()
        for pattern in mcq_patterns:
            match = re.search(pattern, last_line, re.IGNORECASE)
            if match:
                ans = match.group(1).upper()
                logger.debug(
                    f"Extracted MCQ answer '{ans}' from last line using pattern: '{pattern}'")
                return ans
        if re.fullmatch(r'[A-Ea-e]', last_line):
            ans = last_line.upper()
            logger.debug(f"Extracted MCQ answer '{ans}' as last line content.")
            return ans

    for pattern in mcq_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ans = match.group(1).upper()
            logger.debug(f"Extracted MCQ answer '{ans}' from full text using pattern: '{pattern}'")
            return ans

    num_pattern = r'[+-]?\d+(?:,\d+)*(?:\.\d+)?'
    numerical_patterns = [
        r'(?:final answer is|the final answer is|final answer:|answer:|the answer is)\s*(' + num_pattern + r')\b',
        r'\\boxed\{(' + num_pattern + r')\}',
        r'(?:is|equals|result is|=)\s*(' + num_pattern + r')\s*\.?\s*$',
    ]
    for pattern in numerical_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted_num = match.group(1).replace(",", "")
            logger.debug(f"Extracted numerical answer '{extracted_num}' using pattern: '{pattern}'")
            try:
                f_val = float(extracted_num)
                if f_val.is_integer(): return str(int(f_val))
            except ValueError:
                pass
            return extracted_num

    numbers = re.findall(num_pattern, text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "")
        if lines:
            last_line = lines[-1].strip()
            if re.fullmatch(r'.*?(' + re.escape(last_num_str) + r')\s*\.?\s*', last_line,
                            re.IGNORECASE):
                logger.debug(
                    f"Extracted numerical answer '{last_num_str}' from last line direct match.")
                try:
                    f_val = float(last_num_str)
                    if f_val.is_integer(): return str(int(f_val))
                except ValueError:
                    pass
                return last_num_str
            if re.search(r'\b(?:is|answer|equals)\s+(' + re.escape(last_num_str) + r')\s*\.?\s*$',
                         last_line, re.IGNORECASE):
                logger.debug(
                    f"Extracted numerical answer '{last_num_str}' from last line phrasing.")
                try:
                    f_val = float(last_num_str)
                    if f_val.is_integer(): return str(int(f_val))
                except ValueError:
                    pass
                return last_num_str
        logger.debug(f"Extracted numerical answer '{last_num_str}' as last number found.")
        try:
            f_val = float(last_num_str)
            if f_val.is_integer(): return str(int(f_val))
        except ValueError:
            pass
        return last_num_str

    yes_no_match = re.search(r'\b(?:answer|result) is\s+(yes|no)\b\.?', text, re.IGNORECASE)
    if yes_no_match:
        ans = yes_no_match.group(1).lower()
        logger.debug(f"Extracted yes/no answer '{ans}'")
        return ans

    if lines:
        last_line_content = lines[-1].strip()
        if last_line_content and len(last_line_content) < 20:
            logger.debug(f"Falling back to short last line content: '{last_line_content}'")
            return last_line_content

    logger.warning(
        f"Could not extract a definitive answer heuristically from: '{text[:150]}...' Returning error.")
    return "[EXTRACTION_HEURISTIC_FAILURE]"


def extract_final_answer_by_llm(
        raw_output: str, llm: BaseLLM, question: str, **kwargs
) -> str:
    if not raw_output or raw_output == "[ERROR]" or raw_output.startswith("[ERROR:"):
        logger.warning("Skipping LLM extraction for input marked as error.")
        return "[ERROR]"

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        question=question, raw_output=raw_output
    )
    logger.debug("Attempting LLM extraction with prompt:\n%s", prompt)

    try:
        local_kwargs = kwargs.copy()
        extraction_llm_args = {
            "temperature": local_kwargs.pop("temperature", 0.1),
            "max_tokens": local_kwargs.pop("max_tokens", 64),
            "seed": local_kwargs.pop("seed", RANDOM_SEED),
            **local_kwargs
        }
        result = llm.generate_json(
            prompt,
            response_model=ExtractedAnswer,
            **extraction_llm_args
        )
        if result and result.final_answer is not None:
            extracted = str(result.final_answer).strip()
            logger.debug(f"LLM extraction successful: '{extracted}'")
            if re.fullmatch(r'[A-Ea-e]', extracted):
                return extracted.upper()
            return extracted
        else:
            logger.warning(
                f"LLM extraction returned null or invalid object for output: {raw_output[:100]}...")
            return "[EXTRACTION_NULL]"
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}", exc_info=True)
        return "[EXTRACTION_ERROR]"


async def extract_final_answer_by_llm_async(
        raw_output: str, llm: BaseLLM, question: str, **kwargs
) -> str:
    if not raw_output or raw_output == "[ERROR]" or raw_output.startswith("[ERROR:"):
        logger.warning("Skipping async LLM extraction for input marked as error.")
        return "[ERROR]"

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        question=question, raw_output=raw_output
    )
    logger.debug("Attempting async LLM extraction with prompt:\n%s", prompt)

    try:
        local_kwargs = kwargs.copy()
        extraction_llm_args = {
            "response_model": ExtractedAnswer,
            "temperature": local_kwargs.pop("temperature", 0.1),
            "max_tokens": local_kwargs.pop("max_tokens", 64),
            "seed": local_kwargs.pop("seed", RANDOM_SEED),
            **local_kwargs
        }

        semaphore = local_kwargs.pop("semaphore", None)

        if semaphore:
            async with semaphore:
                result = await llm.generate_json_async(prompt, **extraction_llm_args)
        else:
            result = await llm.generate_json_async(prompt, **extraction_llm_args)

        if result and result.final_answer is not None:
            extracted = str(result.final_answer).strip()
            logger.debug(f"Async LLM extraction successful: '{extracted}'")
            if re.fullmatch(r'[A-Ea-e]', extracted):
                return extracted.upper()
            return extracted
        else:
            logger.warning(
                f"Async LLM extraction returned null or invalid object for output: {raw_output[:100]}...")
            return "[EXTRACTION_NULL]"
    except Exception as e:
        logger.error(f"Async LLM extraction failed: {e}", exc_info=True)
        return "[EXTRACTION_ERROR]"


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
    print(f"  Q: {question[:200]}...")
    print(f"  Gold: '{gold}'")
    print(f"  Raw Pred: '{str(raw_pred)[:200]}...'")
    print(f"  Extracted: '{extracted_pred}'")
    print(f"  Correct: {is_correct}")
    print(f"  Time: {time_taken:.2f}s")
    print("-" * 80)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key (reads OPENAI_API_KEY env var if not set)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")


def add_generation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default="gsm8k", choices=list(Datasets.registry.keys()),
                        help="Dataset to run benchmarks on (default: gsm8k)")
    parser.add_argument("--cutoff", type=int, default=50,
                        help="Number of samples to load (-1 for all, default: 50)")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama",
                        help="LLM provider for generation (default: ollama)")
    parser.add_argument("--model-name", default=None,
                        help="Name of the generation model (default: gemma3:4b or gpt-4o-mini)")
    parser.add_argument("--use-async", action="store_true", help="Run generation asynchronously")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent requests for async generation (default: 3)")
    parser.add_argument("--use-json", action="store_true",
                        help="Use JSON format and parsing within strategies where applicable"
                             " (LtM intermediates and final, GoT final, SC internal).")
    parser.add_argument("--output-file", default="benchmark_results.jsonl",
                        help="File to save raw generation results (default: benchmark_results.jsonl)")


def add_evaluation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-file", required=True,
                        help="Path to the JSONL file containing raw generation results.")
    parser.add_argument("--extractor-type", choices=["heuristic", "llm"], default="heuristic",
                        help="Extractor type (default: heuristic)")
    # Renamed arguments for clarity when using LLM extractor
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama",
                        help="LLM provider for LLM-based extraction (default: ollama)")
    parser.add_argument("--model-name", default=None,
                        help="Name of the model for LLM-based extraction (default: gemma3:4b or gpt-4o-mini)")
    parser.add_argument("--show-details", action="store_true",
                        help="Show detailed results (question, answer, predicted answer, was answer correct,"
                             " processing time) for each question during evaluation.")


logger_samples = logging.getLogger("dataset_sampler")


def show_dataset_samples(ds_name: str, num_samples: int = 5):
    """
    Loads a specified dataset from the registry and prints the first few samples.

    Args:
        ds_name: The name of the dataset as defined in Datasets.registry.
        num_samples: The number of samples to display.
    """
    if ds_name not in Datasets.registry:
        logger_samples.error(f"Dataset '{ds_name}' not found in registry.")
        return

    logger_samples.info(f"Loading raw samples for dataset: {ds_name}")
    ds = None
    # Get available splits and HF path from registry
    hf_path, config_name, _, available_splits = Datasets.registry[ds_name]

    try:
        # Use the internal helper to load and combine splits
        ds = Datasets._load_and_combine_splits(hf_path, config_name, available_splits)

    except Exception as e:
        logger_samples.error(f"Failed to load dataset '{ds_name}' (Path: {hf_path}): {e}", exc_info=True)
        return

    if ds is None:
        logger_samples.error(f"Dataset object for '{ds_name}' is None after loading attempt.")
        return

    print(f"\n--- First {num_samples} Samples for Dataset: {ds_name} (Path: {hf_path}, Combined Splits) ---")
    print(f"Total rows loaded: {len(ds)}")

    try:
        actual_samples = min(num_samples, len(ds))
        if actual_samples == 0:
            print("Dataset appears to be empty.")
            return

        # Efficiently select the first few samples
        sample_data = ds.select(range(actual_samples))

        # Print column names (features)
        if sample_data.features:
            print("Columns:", list(sample_data.features.keys()))
        else:
            print("Could not determine columns.")
            return  # Cannot proceed without features

        print("-" * 80)  # Separator

        # Iterate and print rows
        for i in range(actual_samples):
            print(f"Sample {i + 1}:")
            row = sample_data[i]
            for col_name, value in row.items():
                # Basic formatting to handle potentially long text or complex objects
                value_str = str(value)
                truncated_value = value_str[:150] + ('...' if len(value_str) > 150 else '')
                print(f"  {col_name}: {truncated_value}")
            print("-" * 80)  # Separator between rows

    except Exception as e:
        logger_samples.error(
            f"Failed to process or display samples for dataset '{ds_name}': {e}",
            exc_info=True)


if __name__ == "__main__":
    setup_logging()
    for dataset_name in Datasets.registry.keys():
        show_dataset_samples(dataset_name, num_samples=5)
