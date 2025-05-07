## Benchmarks

Benchmarks are run using the `benches/run.py` script, which handles the generation phase.
A separate script, `benches/evaluate.py`, is used afterward to calculate accuracy from the generated results.

Run the generation script from the project root directory:

```bash
poetry run python benches/run.py [OPTIONS]
```

Available Options for `run.py`:

* `--dataset <name>`: Dataset to use (default: `gsm8k`). Available options: `gsm8k`, `multiarith`, `aqua`, `csqa`,
  `strategyqa`, `coin`, and `letter`.
* `--cutoff <number>`: Number of samples to load from the dataset (-1 for all; default: `50`). These samples are used
  for both setup (if needed) and generation and testing.
* `--provider <provider>`: LLM provider (`ollama` or `openai`; default: `ollama`).
* `--model-name <model>`: Model name for the provider (default: `gemma2:9b` for ollama, `gpt-4o-mini` for openai). Verify model
  availability.
* `--openai-key <key>`: OpenAI API key (needed for `--provider openai`, can use `OPENAI_API_KEY` environment variable if
  it is set).
* `--use-async`: Use asynchronous execution for LLM calls (default: sync). Highly recommended for speed.
* `--concurrency <number>`: Max concurrent LLM requests when using `--use-async` (default: `3`).
* `--use-json-strategies`: Use JSON mode within strategies where applicable (LtM, GoT, SC) (default: disabled).
* `--output-file <path>`: File to save raw generation results in JSONL format (default: `benchmark_results.jsonl`).
* `--debug`: Enable debug-level logging for more verbose output.
* `--help`: Show help message for all available options.

Check out OpenAI's [API documentation](https://platform.openai.com/docs/api-reference) for more details on the models
and their capabilities. Use `ollama list` to see the available models for the `ollama` provider.

### Benchmark Workflow (`run.py`)

The `run.py` script executes the following steps:

1. **Configuration:** Loads settings from `benches.yml` and merges them with command-line arguments (CLI > YAML > Defaults).
2. **Dataset Loading:** Loads the specified dataset subset based on the final configuration.
3. **Model & CoT Strategies:** Sets up the language model and instances of enabled CoT strategies, configured according to
   `benches.yml`.
4. **Setup Phase (One-Time Cost per Run):** Before generating answers for the test questions, strategies that need fitting
   or training (like AutoCoT and CDWCoT) perform this step *once* using the loaded dataset samples and configured
   parameters.
5. **Generation Phase (Per Question):** The script iterates through each loaded question:
    * For each question, it executes all *enabled* CoT strategies using their configured parameters.
    * If run synchronously, strategies execute one after another. If async, calls run concurrently.
    * The raw text output from the LLM and the execution time are recorded.
6. **Output:** Results are saved line-by-line in JSONL format to the specified output file.

See `poetry run python benches/run.py --help` to see all available options.

### Evaluation (`evaluate.py`)

After `run.py` generates the result file, use `evaluate.py` to calculate metrics:

```bash
poetry run python benches/evaluate.py --input-file <path_to_results.jsonl> [EVAL_OPTIONS]
```

This script reads the JSONL file, loads its configuration from `benches.yml` and merges with CLI options, extracts the
final answer from the raw model output (using the configured extractor type: heuristic or LLM), compares it to the
correct answer, and calculates the accuracy for each CoT strategy present in the result file. It then shows a summary
table. See `poetry run python benches/evaluate.py --help` for evaluation-specific options.

### Configuration (`benches.yml`)

Benchmark runs are configured using `benches.yml` in the project root, combined with command-line arguments.
**Configuration Precedence:**

1. **Command-Line Arguments:** Highest priority (e.g., `--dataset`, `--provider`).
2. **`benches.yml`:** Values from this file are used if not specified via CLI.
3. **Code Defaults:** Lowest priority, used if not set in CLI or YAML.

**YAML Structure:**

* **`common`**: Shared settings.
    * `debug`: `true` or `false` for verbose logging.
    * `openai_key_env_var`: Name of the environment variable holding the OpenAI key.
* **`generation`**: Settings for the generation script (`run.py`).
    * `dataset`: Name of the dataset (e.g., `gsm8k`).
    * `cutoff`: Max number of samples to use (-1 for all).
    * `provider`: `ollama` or `openai`.
    * `model_name`: Specific model for the provider.
    * `ollama_host`: (**Optional**) Specify the host address for the Ollama server (e.g., `http://192.168.1.100:11434`). If `null`
      or omitted, uses `OLLAMA_HOST` env var or defaults to `http://localhost:11434`.
    * `use_async`: `true` to run LLM calls concurrently.
    * `concurrency`: Max parallel requests for async runs.
    * `use_json_strategies`: `true` or `false`. Default for strategies supporting JSON output (can be overridden per strategy).
    * `output_file`: Path to save raw results (JSONL).
    * `llm_params`: Global LLM settings (`max_tokens`, `seed`, `temperature`) applied unless overridden per strategy.
* **`evaluation`**: Settings for the evaluation script (`evaluate.py`).
    * `input_file`: Path to the results JSONL file (defaults to `generation.output_file`).
    * `extractor`: Configures how final answers are extracted.
        * `type`: `heuristic` or `llm`.
        * `provider`, `model_name`: Settings for the LLM extractor if `type` is `llm`.
        * `ollama_host`: Specify the Ollama host for the extractor LLM, if using `type: llm` and
          `provider: ollama`. Defaults apply if null/omitted.
        * `llm_params`: Settings for the LLM extractor if `type` is `llm`.
    * `show_details`: `true` to print per-question evaluation details.
    * `concurrency`: Max parallel requests for the LLM extractor.
* **`strategies`**: Configure individual CoT strategies.
    * Each key is the strategy's class name (e.g., `AutoCoT`).
    * Including a section enables the strategy by default. Add `enabled: false` to disable.
    * Set strategy-specific parameters (e.g., `n_demos`, `pool_size`, `n_samples`, `max_depth`).
    * Strategy-specific LLM parameters (like `temperature` for `SelfConsistency`) or format choices (`internal_extraction_format`,
      `intermediate_output_format`, `final_answer_format`) set here override global settings from the `generation` section for
      that specific strategy.

See the example `benches.yml` in the repository for detailed options.

> [!NOTE]
> For parameters like OpenAI keys, it is recommended to specify the *environment variable name* in `benches.yml` (e.g.,
`openai_key_env_var: "MY_API_KEY"`) rather than pasting the key directly into the file.
> The scripts will then read the key from the specified environment variable.
> You can still override this by passing `--openai-key` on the command line.

### Example Usages (`run.py`)

* Run using configuration defined in `benches.yml`:
    ```bash
    # Assumes benches.yml is configured as desired
    poetry run python benches/run.py
    ```

* Run using `benches.yml` but override the dataset via command line:
    ```bash
    poetry run python benches/run.py --dataset aqua
    ```

### Dependencies

To run the benchmarks, you might want to install the development dependencies along with Cogitator itself.

```bash
poetry install --with dev
```

Additionally, any model used in the benchmarks must be available.
Make sure the Ollama server is **running** and pull desired models using `ollama pull <model_name>`.
Make sure the OpenAI key is set correctly if using the OpenAI models.

### More Examples

```bash
# Run using benches.yml (assuming it's configured for Ollama, gemma2:9b, aqua, async, etc.)
poetry run python benches/run.py --output-file my_ollama_results.jsonl

# Evaluate the results using heuristic (as configured in benches.yml or default)
poetry run python benches/evaluate.py --input-file my_ollama_results.jsonl --show-details

# Evaluate the results using LLM extractor (override benches.yml extractor setting)
poetry run python benches/evaluate.py --extractor-type llm --provider ollama --model-name llama3 --input-file my_ollama_results.jsonl

# Run specifically with OpenAI, overriding YAML if necessary
poetry run python benches/run.py --provider openai --model-name gpt-4o-mini --dataset csqa --cutoff 10 --use-async --output-file my_openai_results.jsonl

# Evaluate the OpenAI results using an OpenAI extractor model
poetry run python benches/evaluate.py --input-file my_openai_results.jsonl --extractor-type llm --provider openai --model-name gpt-4o-mini
```

## Performance Metric

Accuracy is the primary metric reported by the `evaluate.py` script.
It is defined as the percentage of correctly answered questions out of the total number of successfully extracted answers for a
given CoT strategy.

> [!NOTE]
> This definition means accuracy reflects performance *only* on runs where the final answer could be successfully extracted.
> Runs resulting in extraction errors (e.g., the extractor fails to find an answer in the raw output) are excluded from the
> accuracy calculation, which is important when comparing strategies with different extraction success rates.

## Datasets

The following datasets can be used for values in the `--dataset` argument of the `run.py` script.

| Dataset Name | Source Link                                                                          | Category Tags             | Description                                       |
|:-------------|:-------------------------------------------------------------------------------------|:--------------------------|:--------------------------------------------------|
| `gsm8k`      | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                         | `math`                    | Grade school math word problems                   |
| `multiarith` | [ChilleD/MultiArith](https://huggingface.co/datasets/ChilleD/MultiArith)             | `math`                    | Multi-step arithmetic problems                    |
| `aqua`       | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat)               | `math`                    | Algebraic word problems with rationales           |
| `csqa`       | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)             | `commonsense`             | Multiple-choice commonsense questions             |
| `strategyqa` | [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA)             | `commonsense`, `symbolic` | Yes and no questions requiring implicit reasoning |
| `coin`       | [skrishna/coin_flip](https://huggingface.co/datasets/skrishna/coin_flip)             | `symbolic`                | Symbolic tasks involving state tracking           |
| `letter`     | [ChilleD/LastLetterConcat](https://huggingface.co/datasets/ChilleD/LastLetterConcat) | `symbolic`, `text`        | Extract and concatenate last letters from words   |
