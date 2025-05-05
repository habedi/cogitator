import argparse
import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset

from cogitator import BaseLLM, OllamaLLM, OpenAILLM
from cogitator.schemas import ExtractedAnswer

logger = logging.getLogger("benchmark_shared")

DEFAULT_MAX_TOKEN = 1024
DEFAULT_RANDOM_SEED = 33
DEFAULT_PROVIDER = "ollama"
DEFAULT_DATASET = "gsm8k"
DEFAULT_CUTOFF = 50
DEFAULT_CONCURRENCY = 3
DEFAULT_OUTPUT_FILE = "benchmark_results.jsonl"
DEFAULT_EXTRACTOR_TYPE = "heuristic"
DEFAULT_OPENAI_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_BENCH_CONFIG_PATH = "benches.yml"

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
        splits_to_try = [s for s in available_splits if s in ["train", "test"]]
        if not splits_to_try:
            splits_to_try = available_splits

        logger.info(f"Attempting to load splits for {dataset_hf_path}: {splits_to_try}")
        for split in splits_to_try:
            try:
                ds_split = load_dataset(
                    dataset_hf_path, config_name, split=split, trust_remote_code=True
                )
                if isinstance(ds_split, Dataset):
                    logger.info(f"Successfully loaded split '{split}' ({len(ds_split)} rows)")
                    loaded_splits.append(ds_split)
                else:
                    logger.warning(
                        f"Loaded object for split '{split}' is not a Dataset: {type(ds_split)}"
                    )
            except ValueError as e:
                if "Config name is missing" in str(e) and config_name is None:
                    logger.error(
                        f"Dataset {dataset_hf_path} requires a config name, but None was provided."
                    )
                logger.warning(
                    f"Split '{split}' not found or failed to load for {dataset_hf_path}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error loading split '{split}' for {dataset_hf_path}: {e}",
                    exc_info=True,
                )

        if not loaded_splits:
            raise ValueError(
                f"Could not load any of the specified splits for dataset {dataset_hf_path}"
            )

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

        hf_path, config_name, _, available_splits = Datasets.registry[name]
        logger.info(
            f"Loading dataset: {name} (HF Path: {hf_path}, Config: {config_name}, Available Splits: {available_splits}, Cutoff: {cutoff})"
        )
        combined_ds = Datasets._load_and_combine_splits(hf_path, config_name, available_splits)
        processor_fn_name = f"_process_{name}"
        if not hasattr(Datasets, processor_fn_name):
            raise AttributeError(
                f"Could not find processor function for dataset '{name}' (tried '{processor_fn_name}')"
            )

        processor_fn = getattr(Datasets, processor_fn_name)
        qs, golds = processor_fn(combined_ds)
        golds = [str(g) for g in golds]
        logger.info(
            f"Processed {len(qs)} total questions and {len(golds)} gold answers before cutoff."
        )
        qs, golds = Datasets.apply_cutoff(qs, golds, cutoff)
        logger.info(f"Final dataset size after cutoff: {len(qs)} samples.")
        if len(qs) == 0:
            raise ValueError("Loaded 0 questions after processing. Check dataset name and cutoff.")
        return qs, golds

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

    @staticmethod
    def _process_aqua(ds: Dataset):
        questions = ds["question"]
        options_list = ds["options"]
        questions_with_options = [
            q + "\n" + "It's a multiple choice question. Pick the correct label (character before ')')\n" + " ".join(
                opts)
            for q, opts in zip(questions, options_list)
        ]
        return questions_with_options, ds["correct"]

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

                choices_str = ' '.join(
                    [f'({chr(ord("A") + i)}) {c}' for i, c in enumerate(choices)])
                question_with_choices = f"{question_text}\nChoices: {choices_str}"

                qs.append(question_with_choices)
                idx = ord(answer_key) - ord("A")
                if 0 <= idx < len(choices):
                    golds.append(answer_key.upper())
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


def get_llm(
    provider: str,
    model_name: Optional[str] = None,
    openai_key: Optional[str] = None,
    ollama_host: Optional[str] = None,
    is_extractor: bool = False,
    llm_params: Optional[Dict[str, Any]] = None
) -> BaseLLM:
    role = "Extractor" if is_extractor else "Primary"
    if model_name is None:
        model_name = "gpt-4o-mini" if provider == "openai" else "gemma3:4b"
        logger.info(f"{role} model name not specified, using default for {provider}: {model_name}")

    base_llm_params = {
        "max_tokens": DEFAULT_MAX_TOKEN,
        "seed": DEFAULT_RANDOM_SEED,
    }
    if llm_params:
        base_llm_params.update(llm_params)

    logger.info(
        f"Initializing {role} LLM: provider={provider}, model={model_name}, "
        f"ollama_host={ollama_host}, params={base_llm_params}"
    )

    if provider == "openai":
        if not openai_key:
            openai_key = os.getenv(DEFAULT_OPENAI_ENV_VAR)
            if not openai_key:
                raise ValueError(
                    "OpenAI provider selected but no API key found in args or env var.")
        return OpenAILLM(api_key=openai_key, model=model_name, **base_llm_params)
    elif provider == "ollama":
        return OllamaLLM(model=model_name, ollama_host=ollama_host, **base_llm_params)
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
                if f_val.is_integer():
                    return str(int(f_val))
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
                    if f_val.is_integer():
                        return str(int(f_val))
                except ValueError:
                    pass
                return last_num_str
            if re.search(r'\b(?:is|answer|equals)\s+(' + re.escape(last_num_str) + r')\s*\.?\s*$',
                         last_line,
                         re.IGNORECASE):
                logger.debug(
                    f"Extracted numerical answer '{last_num_str}' from last line phrasing.")
                try:
                    f_val = float(last_num_str)
                    if f_val.is_integer():
                        return str(int(f_val))
                except ValueError:
                    pass
                return last_num_str
        logger.debug(f"Extracted numerical answer '{last_num_str}' as last number found.")
        try:
            f_val = float(last_num_str)
            if f_val.is_integer():
                return str(int(f_val))
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


def extract_final_answer_by_llm(
    raw_output: str, llm: BaseLLM, question: str, **kwargs
) -> str:
    if not raw_output or raw_output == "[ERROR]" or raw_output.startswith("[ERROR:"):
        logger.warning("Skipping LLM extraction for input marked as error.")
        return "[ERROR]"
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(question=question, raw_output=raw_output)
    logger.debug("Attempting LLM extraction with prompt:\n%s", prompt)
    try:
        local_kwargs = kwargs.copy()
        extraction_llm_args = {
            "temperature": local_kwargs.pop("temperature", 0.1),
            "max_tokens": local_kwargs.pop("max_tokens", 64),
            "seed": local_kwargs.pop("seed", DEFAULT_RANDOM_SEED),
            **local_kwargs
        }
        result = llm.generate_json(prompt, response_model=ExtractedAnswer, **extraction_llm_args)
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
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(question=question, raw_output=raw_output)
    logger.debug("Attempting async LLM extraction with prompt:\n%s", prompt)
    try:
        local_kwargs = kwargs.copy()
        extraction_llm_args = {
            "response_model": ExtractedAnswer,
            "temperature": local_kwargs.pop("temperature", 0.1),
            "max_tokens": local_kwargs.pop("max_tokens", 64),
            "seed": local_kwargs.pop("seed", DEFAULT_RANDOM_SEED),
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
    show_details: bool, idx: int, name: str, mode: str, question: str, gold: str,
    raw_pred: str, extracted_pred: str, time_taken: float, is_correct: bool
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


def load_and_merge_config(
    args: argparse.Namespace,
    parser_definition: argparse.ArgumentParser,
    config_path: str = DEFAULT_BENCH_CONFIG_PATH,
    config_section: str = "generation"
) -> Dict[str, Any]:
    config = {}
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config = yaml_config
                logger.info(f"Loaded benchmark configuration from {config_path}")
            else:
                logger.info(f"{config_path} is empty. Using command-line arguments and defaults.")
    except FileNotFoundError:
        logger.info(f"{config_path} not found. Using command-line arguments and defaults.")
    except yaml.YAMLError as e:
        logger.warning(
            f"Error parsing {config_path}: {e}. Using command-line arguments and defaults.")

    final_config = {}
    common_config = config.get("common", {})
    section_config = config.get(config_section, {})
    strategies_config = config.get("strategies", {})
    final_config['strategies'] = strategies_config

    parser_defaults_ns = parser_definition.parse_args([])
    args_vars = vars(args)
    defaults_vars = vars(parser_defaults_ns)

    cli_set_args = set()
    for key, value in args_vars.items():
        if key in defaults_vars and value != defaults_vars.get(key):
            action = next((a for a in parser_definition._actions if a.dest == key), None)
            if isinstance(action, argparse._StoreTrueAction) and value is True:
                cli_set_args.add(key)
            elif not isinstance(action, argparse._StoreTrueAction):
                cli_set_args.add(key)

    def get_value(arg_name: str, default_value: Any,
                  primary_config: Dict = section_config,
                  secondary_config: Dict = common_config,
                  cli_arg_name: Optional[str] = None):

        check_cli_arg_name = cli_arg_name or arg_name
        cli_value = getattr(args, check_cli_arg_name, None)
        cli_is_set = check_cli_arg_name in cli_set_args

        yaml_value = primary_config.get(arg_name, secondary_config.get(arg_name, None))

        if cli_is_set:
            logger.debug(f"Using CLI value for '{check_cli_arg_name}': {cli_value}")
            return cli_value
        elif yaml_value is not None:
            logger.debug(f"Using YAML value for '{arg_name}': {yaml_value}")
            return yaml_value
        else:
            logger.debug(f"Using default value for '{arg_name}': {default_value}")
            return default_value

    final_config['debug'] = getattr(args, 'debug',
                                    False) if 'debug' in cli_set_args else common_config.get(
        'debug',
        False)
    final_config['openai_key_env_var'] = get_value('openai_key_env_var', DEFAULT_OPENAI_ENV_VAR,
                                                   cli_arg_name='openai_key_env_var')

    if config_section == "generation":
        final_config['dataset'] = get_value('dataset', DEFAULT_DATASET, cli_arg_name='dataset')
        final_config['cutoff'] = get_value('cutoff', DEFAULT_CUTOFF, cli_arg_name='cutoff')
        final_config['provider'] = get_value('provider', DEFAULT_PROVIDER, cli_arg_name='provider')
        final_config['model_name'] = get_value('model_name', None, cli_arg_name='model_name')
        final_config['ollama_host'] = get_value('ollama_host', None)
        final_config['use_async'] = getattr(args, 'use_async',
                                            False) if 'use_async' in cli_set_args else section_config.get(
            'use_async',
            False)
        final_config['concurrency'] = get_value('concurrency', DEFAULT_CONCURRENCY,
                                                cli_arg_name='concurrency')
        final_config['use_json_strategies'] = getattr(args, 'use_json_strategies',
                                                      False) if 'use_json_strategies' in cli_set_args else section_config.get(
            'use_json_strategies', False)
        final_config['output_file'] = get_value('output_file', DEFAULT_OUTPUT_FILE,
                                                cli_arg_name='output_file')
        final_config['llm_params'] = section_config.get('llm_params', {})

    elif config_section == "evaluation":
        gen_output_file = config.get("generation", {}).get("output_file", DEFAULT_OUTPUT_FILE)
        default_input = gen_output_file if 'input_file' not in cli_set_args and hasattr(args,
                                                                                        'input_file') and args.input_file is None else None
        final_config['input_file'] = get_value('input_file', default_input or DEFAULT_OUTPUT_FILE,
                                               cli_arg_name='input_file')

        extractor_yaml_config = section_config.get("extractor", {})

        final_config['extractor_type'] = get_value(
            'type',
            DEFAULT_EXTRACTOR_TYPE,
            primary_config=extractor_yaml_config,
            secondary_config={},
            cli_arg_name='extractor_type'
        )
        final_config['extractor_provider'] = get_value(
            'provider', DEFAULT_PROVIDER,
            primary_config=extractor_yaml_config,
            cli_arg_name='provider'
        )
        final_config['extractor_model_name'] = get_value(
            'model_name', None,
            primary_config=extractor_yaml_config,
            cli_arg_name='model_name'
        )
        final_config['extractor_ollama_host'] = get_value(
            'ollama_host', None,
            primary_config=extractor_yaml_config
        )
        final_config['extractor_llm_params'] = extractor_yaml_config.get('llm_params', {})

        final_config['concurrency'] = get_value('concurrency', DEFAULT_CONCURRENCY,
                                                cli_arg_name='concurrency')
        final_config['show_details'] = getattr(args, 'show_details',
                                               False) if 'show_details' in cli_set_args else section_config.get(
            'show_details', False)

    logger.debug(f"Final merged config ({config_section}): {final_config}")
    return final_config


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key (overrides config/env var)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logging (overrides config).")


def add_generation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        choices=list(Datasets.registry.keys()),
                        help=f"Dataset name (overrides config, default: {DEFAULT_DATASET})")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF,
                        help=f"Number of samples (-1 for all, overrides config, default: {DEFAULT_CUTOFF})")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=DEFAULT_PROVIDER,
                        help=f"LLM provider for generation (overrides config, default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model-name", default=None,
                        help="Generation model name (overrides config, default: gemma3:4b for ollama, gpt-4o-mini for openai)")
    parser.add_argument("--use-async", action='store_true', default=False,
                        help="Run generation asynchronously (overrides config)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max concurrent async requests (overrides config, default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--use-json-strategies", action='store_true', default=False,
                        help="Use JSON mode within strategies (overrides config)")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE,
                        help=f"File to save raw generation results (overrides config, default: {DEFAULT_OUTPUT_FILE})")


def add_evaluation_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-file", default=None,
                        help="Path to the JSONL file with generation results (overrides config default)")
    parser.add_argument("--extractor-type", choices=["heuristic", "llm"],
                        default=DEFAULT_EXTRACTOR_TYPE,
                        help=f"Extractor type (overrides config, default: {DEFAULT_EXTRACTOR_TYPE})")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=DEFAULT_PROVIDER,
                        help=f"LLM provider for LLM-based extraction (overrides config, default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model-name", default=None,
                        help="Model name for LLM-based extraction (overrides config, default: gemma3:4b for ollama, gpt-4o-mini for openai)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max concurrent async requests for LLM extractor (overrides config, default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--show-details", action='store_true', default=False,
                        help="Show detailed results per question (overrides config).")


logger_samples = logging.getLogger("dataset_sampler")


def show_dataset_samples(ds_name: str, num_samples: int = 5):
    if ds_name not in Datasets.registry:
        logger_samples.error(f"Dataset '{ds_name}' not found in registry.")
        return
    logger_samples.info(f"Loading raw samples for dataset: {ds_name}")
    ds = None
    hf_path, config_name, _, available_splits = Datasets.registry[ds_name]
    try:
        ds = Datasets._load_and_combine_splits(hf_path, config_name, available_splits)
    except Exception as e:
        logger_samples.error(f"Failed to load dataset '{ds_name}' (Path: {hf_path}): {e}",
                             exc_info=True)
        return
    if ds is None:
        logger_samples.error(f"Dataset object for '{ds_name}' is None after loading attempt.")
        return
    print(
        f"\n--- First {num_samples} Samples for Dataset: {ds_name} (Path: {hf_path}, Combined Splits) ---")
    print(f"Total rows loaded: {len(ds)}")
    try:
        actual_samples = min(num_samples, len(ds))
        if actual_samples == 0:
            print("Dataset appears to be empty.")
            return
        sample_data = ds.select(range(actual_samples))
        if sample_data.features:
            print("Columns:", list(sample_data.features.keys()))
        else:
            print("Could not determine columns.")
            return
        print("-" * 80)
        for i in range(actual_samples):
            print(f"Sample {i + 1}:")
            row = sample_data[i]
            for col_name, value in row.items():
                value_str = str(value)
                truncated_value = value_str[:150] + ('...' if len(value_str) > 150 else '')
                print(f"  {col_name}: {truncated_value}")
            print("-" * 80)
    except Exception as e:
        logger_samples.error(f"Failed to process or display samples for dataset '{ds_name}': {e}",
                             exc_info=True)


if __name__ == "__main__":
    setup_logging()
    for dataset_name in Datasets.registry.keys():
        show_dataset_samples(dataset_name, num_samples=5)
