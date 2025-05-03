## Benchmarks

Benchmarks are primarily run using the `benches/run.py` script, which handles the generation phase.
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
* `--model-name <model>`: Model name for the provider (default: `gemma3:4b` for ollama, `gpt-4o-mini` for openai).
* `--openai-key <key>`: OpenAI API key (needed for `--provider openai`, can use `OPENAI_API_KEY` environment variable if
  it is set).
* `--use-async`: Use asynchronous execution for LLM calls (default: sync). Highly recommended for speed.
* `--concurrency <number>`: Max concurrent LLM requests when using `--use-async` (default: `3`).
* `--use-json-strategies`: Use JSON mode within strategies where applicable (LtM, GoT, SC) (default: disabled).
* `--output-file <path>`: File to save raw generation results in JSONL format (default: `benchmark_results.jsonl`).
* `--debug`: Enable debug logging for more verbose output.

Check out OpenAI's [API documentation](https://platform.openai.com/docs/api-reference) for more details on the models
and their capabilities. Use `ollama list` to see the available models for the `ollama` provider.

### Benchmark Workflow (`run.py`)

The `run.py` script executes the following steps:

1. **Configuration:** Loads settings from `benches.yml` and merges them with command-line arguments (CLI > YAML >
   Defaults).
2. **Dataset Loading:** Loads the specified dataset subset based on the final configuration.
3. **Model & CoT Methods:** Sets up the language model and instances of enabled CoT methods, configured according to
   `benches.yml`.
4. **Setup Phase (One-Time Cost per Run):** Before generating answers for the test questions, methods that need fitting
   or training (like AutoCoT, CDWCoT) perform this step *once* using the loaded dataset samples and configured
   parameters.
5. **Generation Phase (Per Question):** The script iterates through each loaded question:
    * For each question, it executes all *enabled* CoT methods using their configured parameters.
    * If run synchronously, methods execute one after another. If async, calls run concurrently.
    * The raw text output from the LLM and the execution time are recorded.
6. **Output:** Results are saved line-by-line in JSONL format to the specified output file.

### Evaluation (`evaluate.py`)

After `run.py` generates the result file, use `evaluate.py` to calculate metrics:

```bash
poetry run python benches/evaluate.py --input-file <path_to_results.jsonl> [EVAL_OPTIONS]
```

This script reads the JSONL file, loads its configuration from `benches.yml` and merges with CLI options, extracts the
final answer from the raw model output (using the configured extractor type: heuristic or LLM), compares it to the
correct answer, and calculates the accuracy for each CoT method present in the result file. It then shows a summary
table. See `poetry run python benches/evaluate.py --help` for evaluation-specific options.

### Configuration (`benches.yml`)

Benchmark runs can be configured using command-line arguments or a `benches.yml` file placed in the project root.

**Configuration Precedence:**

1. **Command-Line Arguments:** Flags provided directly when running `run.py` or `evaluate.py` take the highest
   precedence for global settings like dataset, provider, cutoff, etc.
2. **`benches.yml`:** If a global parameter is not specified on the command line, its value is taken from the `common`,
   `generation`, or `evaluation` sections of `benches.yml`. Strategy-specific parameters (like `n_demos` for `AutoCoT`)
   are configured under the `strategies` section. CLI flags do *not* override strategy-specific parameters inside the
   YAML.
3. **Code Defaults:** If a parameter is not specified via CLI or YAML, a default value defined in the code is used.

**YAML Structure:**

* **`common`**: Shared settings like `debug`, `openai_key_env_var`.
* **`generation`**: Settings for the generation phase (`run.py`) like `dataset`, `cutoff`, `provider`, `model_name`,
  `use_async`, `concurrency`, `use_json_strategies`, `output_file`, and global `llm_params` (like `max_tokens`, `seed`).
* **`evaluation`**: Settings for the evaluation phase (`evaluate.py`) like `input_file`, `extractor` settings (including
  its own `provider`, `model_name`, `llm_params`), `show_details`, and `concurrency` for the extractor LLM.
* **`strategies`**: Configure individual CoT methods.
    * Each key is the class name (e.g., `AutoCoT`, `SelfConsistency`).
    * If a strategy section exists, the strategy is run by default. Add `enabled: false` inside its section to disable
      it.
    * Specify strategy-specific parameters (e.g., `n_demos`, `pool_size`, `n_samples`, `max_depth`).
    * Strategy-specific LLM parameters (like `temperature` for `SelfConsistency`) set here will override the global
      `generation.llm_params` for that strategy's execution.

See the example `benches.yml` in the repository for detailed options.

**Note on Secrets:** For parameters like OpenAI keys, it's recommended to specify the *environment variable name* in
`benches.yml` (e.g., `openai_key_env_var: "MY_API_KEY"`) rather than pasting the key directly into the file. The scripts
will then read the key from the specified environment variable. You can still override this by passing `--openai-key` on
the command line.

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

To run the benchmarks, you might want to install the development dependencies along with the main package.

```bash
pip install cogitator[dev]
# Or poetry install --with dev
```

Additionally, any model used in the benchmarks must be available. For Ollama, pull models using
`ollama pull <model_name>`. Ensure your OpenAI key is set correctly if using that provider.

### More Examples

```bash
# Run using benches.yml (assuming it's configured for Ollama, gemma3:4b, aqua, async, etc.)
poetry run python benches/run.py --output-file my_ollama_results.jsonl

# Evaluate the results using heuristic (as configured in benches.yml or default)
poetry run python benches/evaluate.py --input-file my_ollama_results.jsonl --show-details

# Evaluate the results using LLM extractor (override benches.yml extractor setting)
poetry run python benches/evaluate.py --extractor-type llm --provider ollama --model-name qwen2:7b --input-file my_ollama_results.jsonl

# Run specifically with OpenAI, overriding YAML if necessary
poetry run python benches/run.py --provider openai --model-name gpt-4o-mini --dataset csqa --cutoff 10 --use-async --output-file my_openai_results.jsonl

# Evaluate the OpenAI results using an OpenAI extractor model
poetry run python benches/evaluate.py --input-file my_openai_results.jsonl --extractor-type llm --provider openai --model-name gpt-4o-mini
```

## Performance Metric

Accuracy is the primary metric reported by the `evaluate.py` script. It is defined as the percentage of correctly
answered questions out of the total number of successfully extracted answers for a given method. Failed extractions are
reported separately.

## Datasets

The following datasets can be used in the benchmarking process:

| Dataset Name | Source Link                                                                          | Category Tags             | Description                                       |
|:-------------|:-------------------------------------------------------------------------------------|:--------------------------|:--------------------------------------------------|
| `gsm8k`      | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                         | `math`                    | Grade school math word problems                   |
| `multiarith` | [ChilleD/MultiArith](https://huggingface.co/datasets/ChilleD/MultiArith)             | `math`                    | Multi-step arithmetic problems                    |
| `aqua`       | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat)               | `math`                    | Algebraic word problems with rationales           |
| `csqa`       | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)             | `commonsense`             | Multiple-choice commonsense questions             |
| `strategyqa` | [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA)             | `commonsense`, `symbolic` | Yes and no questions requiring implicit reasoning |
| `coin`       | [skrishna/coin_flip](https://huggingface.co/datasets/skrishna/coin_flip)             | `symbolic`                | Symbolic tasks involving state tracking           |
| `letter`     | [ChilleD/LastLetterConcat](https://huggingface.co/datasets/ChilleD/LastLetterConcat) | `symbolic`, `text`        | Extract and concatenate last letters from words   |
