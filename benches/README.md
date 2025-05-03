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
* `--use-json`: Use JSON mode for LLM interactions where applicable (affects LeastToMost intermediates and final,
  GraphOfThoughts final, and SelfConsistency internal extraction results). Default: disabled.
* `--output-file <path>`: File to save raw generation results in JSONL format (default: `benchmark_results.jsonl`).
* `--debug`: Enable debug logging for more verbose output.

Check out OpenAI's [API documentation](https://platform.openai.com/docs/api-reference) for more details on the models
and their capabilities. Use `ollama list` to see the available models for the `ollama` provider.

### Benchmark Workflow (`run.py`)

The `run.py` script executes the following steps:

1. **Dataset Loading:** Loads the specified `--dataset` subset (up to `--cutoff` samples).
2. **Model & CoT Methods:** Sets up the language model and instances of all available CoT methods (Zero-Shot, AutoCoT,
   CDW-CoT, SelfConsistency, LeastToMost, TreeOfThoughts, GraphOfThoughts).
3. **Setup Phase (One-Time Cost per Run):** Before generating answers for the test questions, methods that need fitting
   or training perform this step *once* using the loaded dataset samples:
    * **AutoCoT:** Runs its `fit` method to select demo examples and generate their reasoning steps based on the samples
      in the dataset.
    * **CDWCoT:** Runs its `init_pool` and `train` methods to build a prompt pool and optimize selection probabilities,
      again using the samples in the dataset.
    * *Note:* This setup phase, especially CDW-CoT training, involves multiple LLM calls and can be a bit
      time-consuming, especially when run synchronously (without `--use-async`).
4. **Generation Phase (Per Question):** The script iterates through each loaded question:
    * For each question, it executes *all* configured CoT method.
    * If run synchronously, methods execute one after another for the question. If `--use-async` is enabled, LLM calls
      within and across method can run concurrently up to the specified limit.
    * Methods like AutoCoT and CDW-CoT use the results from their one-time setup phase.
    * Methods like SelfConsistency, LeastToMost, TreeOfThoughts, and GraphOfThoughts may perform multiple LLM calls *per
      question* as part of their core logic.
    * The raw text output from the LLM and the execution time are recorded for each method and question.
5. **Output:** Results (question, correct answer, method name, raw model output, time) are saved line-by-line in JSONL
   format to the specified `--output-file`.

### Evaluation (`evaluate.py`)

After `run.py` generates the result file, use `evaluate.py` to calculate metrics:

```bash
poetry run python benches/evaluate.py --input-file <path_to_results.jsonl> [EVAL_OPTIONS]
```

This script reads the JSONL file, extracts the final answer from the raw model output (using either a heuristic or
another LLM), compares it to the correct answer, and calculates the accuracy for each CoT method.
It then shows a summary table.
See `poetry run python benches/evaluate.py --help` for evaluation-specific options like `--extractor-type`,
`--provider` (for LLM extractor), `--model-name` (for LLM extractor), and `--show-details`.

### Example Usages (`run.py`)

* Run with defaults settings (Ollama with `gemma3:4b` model, `gsm8k` dataset, 50 samples):
    ```bash
    poetry run python benches/run.py
    ```

* Run using OpenAI as provider with `gpt-4o-mini` model, async, cutoff 100, on `csqa` dataset with concurrency 5:
    ```bash
    # Make sure OPENAI_API_KEY is set or pass --openai-key
    poetry run python benches/run.py --provider openai --model-name gpt-4o-mini --dataset csqa --cutoff 100 --use-async --concurrency 5
    ```

* Run using Ollama with `gemma3:4b` model, on all samples in the on `multiarith` dataset, with async and JSON mode
  enabled:
    ```bash
    poetry run python benches/run.py --provider ollama --model-name gemma3:4b --dataset multiarith --cutoff -1 --use-async --use-json
    ```

### Dependencies

To run the benchmarks, you might want to install the development dependencies along with the main package.

```bash
pip install cogitator[dev]
```

Additionally, any model used in the benchmarks must be available.
For Ollama, this means that the models like `gemma3:4b`, `gemma3:12b`, `qwen3:14b`, etc. might need to be pulled first.

### More Examples

```bash
# Run the benchmark using Ollama with gemma3:4b model, async, JSON mode
poetry run python benches/run.py --provider ollama --dataset aqua --cutoff 20 --use-async --use-json --model-name gemma3:4b --output-file interim_results.jsonl

# Evaluate the results using heuristic
poetry run python benches/evaluate.py --extractor-type heuristic --input-file interim_results.jsonl --show-details

# Evaluate the results using LLM extractor (ollama)
poetry run python benches/evaluate.py --extractor-type llm --provider ollama --model-name qwen3:14b --input-file interim_results.jsonl --show-details

# Run the benchmark using OpenAI with gpt-4o-mini model, async, JSON mode
poetry run python benches/run.py --provider openai --dataset aqua --cutoff 20 --use-async --use-json --model-name gpt-4o-mini --output-file interim_results_openai.jsonl

# Evaluate the results using LLM extractor (openai)
poetry run python benches/evaluate.py --extractor-type llm --provider openai --model-name gpt-4o-mini --input-file interim_results_openai.jsonl --show-details
```

## Performance Metric

Accuracy is the primary metric reported by the `evaluate.py` script. It is defined as the percentage of correctly
answered questions out of the total number of samples evaluated. The script extracts the final answer from the raw model
output stored in the JSONL file before comparing it to the gold standard.

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
