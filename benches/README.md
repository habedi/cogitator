## Benchmarks

Benchmarks are primarily run using the `benches/run.py` script, which handles the generation phase. A separate script,
`benches/evaluate.py`, is used afterward to calculate accuracy from the generated results.

Run the generation script from the project root directory:
`poetry run python benches/run.py [OPTIONS]`

Available Options for `run.py`:

* `--dataset <name>`: Dataset to use (default: `gsm8k`). Available options: `gsm8k`, `multiarith`, `aqua`, `csqa`,
  `strategyqa`, `coin`, and `letter`.
* `--cutoff <number>`: Number of samples to load from the dataset (-1 for all; default: `50`). These samples are used
  for both setup (if needed) and generation and testing.
* `--provider <provider>`: LLM provider (`ollama` or `openai`; default: `ollama`).
* `--model-name <model>`: Model name for the provider (default: `gemma3:4b` for ollama, `gpt-4.1-nano` for openai).
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
2. **LLM & Strategy Initialization:** Sets up the language model and instances of all available CoT strategies (
   Zero-Shot, AutoCoT, CDW-CoT, SelfConsistency, LeastToMost, TreeOfThoughts, GraphOfThoughts).
3. **Setup Phase (One-Time Cost per Run):** Before generating answers for the test questions, strategies requiring
   fitting or training perform this step *once* using the loaded dataset samples:
    * **AutoCoT:** Runs its `fit` method to select demonstration examples and generate their reasoning steps based on
      the dataset samples.
    * **CDWCoT:** Runs its `init_pool` and `train` methods to build a prompt pool and optimize selection probabilities,
      again using the dataset samples.
    * *Note:* This setup phase, especially CDW-CoT training, involves multiple LLM calls and can be time-consuming,
      particularly when run synchronously (without `--use-async`).
4. **Generation Phase (Per Question):** The script iterates through each loaded question:
    * For each question, it executes *all* configured CoT strategies.
    * If run synchronously, strategies execute one after another for the question. If `--use-async` is enabled, LLM
      calls within and across strategies can run concurrently up to the specified limit.
    * Strategies like AutoCoT and CDW-CoT utilize the results from their one-time setup phase.
    * Strategies like SelfConsistency, LeastToMost, TreeOfThoughts, and GraphOfThoughts may perform multiple LLM calls
      *per question* as part of their core logic.
    * The raw text output from the LLM and the execution time are recorded for each strategy and question.
5. **Output:** Results (question, gold answer, strategy name, raw LLM output, time) are saved line-by-line in JSONL
   format to the specified `--output-file`.

### Evaluation (`evaluate.py`)

After `run.py` generates the results file, use `evaluate.py` to calculate metrics:
`poetry run python benches/evaluate.py --input-file <path_to_results.jsonl> [EVAL_OPTIONS]`

This script reads the JSONL file, extracts the final answer from the raw LLM output (using either a heuristic or another
LLM, configurable via options), compares it to the gold answer, and calculates the accuracy for each CoT method. It then
displays a summary table. See `poetry run python benches/evaluate.py --help` for evaluation-specific options.

### Example Usages (`run.py`)

* Run with defaults (Ollama, llama3, gsm8k, cutoff 50, sync):
    ```bash
    poetry run python benches/run.py
    ```

* Run using OpenAI (`gpt-4o-mini`), async, cutoff 100, on `csqa` (CommonsenseQA):
    ```bash
    # Make sure OPENAI_API_KEY is set or pass --openai-key
    poetry run python benches/run.py --provider openai --model-name gpt-4o-mini --dataset csqa --cutoff 100 --use-async
    ```

* Run using Ollama with `gemma3:4b` model, all samples (-1), on `multiarith`, async with higher concurrency, using JSON
  mode:
    ```bash
    poetry run python benches/run.py --provider ollama --model-name gemma3:4b --dataset multiarith --cutoff -1 --use-async --concurrency 5 --use-json
    ```

### Dependencies

To run the benchmarks, you might want to install the development dependencies along with the main package.

```bash
pip install cogitator[dev]
```

### More Examples

```bash
# Ollama example with gemma3:4b, async, JSON mode
poetry run python benches/run.py --provider ollama --dataset aqua --cutoff 20 --use-async --use-json --model-name gemma3:4b --output-file interim_results.jsonl
poetry run python benches/evaluate.py --extractor heuristic --input-file interim_results.jsonl
poetry run python benches/evaluate.py --extractor llm --extractor-provider ollama --extractor-model-name qwen3:14b --input-file interim_results.jsonl --show-details

# OpenAI example with gpt-4o-mini, async, JSON mode
poetry run python benches/run.py --provider openai --dataset aqua --cutoff 20 --use-async --use-json --model-name gpt-4.1-nano --output-file interim_results_openai.jsonl
poetry run python benches/evaluate.py --extractor llm --extractor-provider openai --extractor-model-name gpt-4o-mini --input-file interim_results_openai.jsonl
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
| `aqua`       | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat)               | `math`                    | Algebraic word problems with rationales (MCQ)     |
| `csqa`       | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)             | `commonsense`             | Multiple-choice commonsense questions             |
| `strategyqa` | [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA)             | `commonsense`, `symbolic` | Yes and no questions requiring implicit reasoning |
| `coin`       | [skrishna/coin_flip](https://huggingface.co/datasets/skrishna/coin_flip)             | `symbolic`                | Symbolic tasks involving state tracking           |
| `letter`     | [ChilleD/LastLetterConcat](https://huggingface.co/datasets/ChilleD/LastLetterConcat) | `symbolic`, `text`        | Extract and concatenate last letters from words   |
