## Benchmarks

Benchmarks are provided to evaluate the performance of different LLMs and prompting methods on various datasets.

### Running Benchmarks

Benchmarks can be run using the `benches/run.py` script.
Run the script from the project root directory using `poetry run python benches/run.py [OPTIONS]`.

Available Options:

* `--dataset <name>`: Dataset to use (default: `gsm8k`). Available: `gsm8k`, `multiarith`, `aqua`, `csqa`, `strategyqa`,
  `coin`, and `letter`.
* `--cutoff <number>`: Number of samples (-1 for all; default: `50`).
* `--provider <provider>`: LLM provider (`ollama` or `openai`, default: `ollama`).
* `--model-name <model>`: Model name for the provider (default: `llama3` for ollama, `gpt-4o-mini` for openai).
* `--openai-key <key>`: OpenAI API key (required for `--provider openai`, can use `OPENAI_API_KEY` env var).
* `--use-async`: Use asynchronous execution (default: sync).
* `--concurrency <number>`: Max concurrent requests for async mode (default: `5`).
* `--show-details`: Show detailed results (Q, Gold, Pred, Correct, Time) for each question.
* `--debug`: Enable debug logging for more verbose output.

Check out OpenAI's [API documentation](https://platform.openai.com/docs/api-reference) for more details on the models
and their capabilities.
Use `ollama list` to see the available models for the `ollama` provider.

#### Example Usages

* Run with defaults (Ollama, llama3, gsm8k, cutoff 50, sync):
    ```bash
    poetry run python benches/run.py
    ```

* Run using OpenAI (`gpt-4o-mini`), async, cutoff 100, on `csqa` (CommonsenseQA), showing details:
    ```bash
    # Make sure OPENAI_API_KEY is set in your environment OR pass the key directly
    poetry run python benches/run.py --provider openai --model-name gpt-4o-mini --dataset csqa --cutoff 100 --use-async --show-details

    # Alternatively, passing the key:
    # poetry run python benches/run.py --provider openai --openai-key "sk-..." \
    # --model-name gpt-4o-mini --dataset csqa --cutoff 100 --use-async --show-details
    ```

* Run using Ollama with `llama3` model, all samples (-1), on `multiarith` dataset, async with higher concurrency:
    ```bash
    poetry run python benches/run.py --provider ollama --model-name llama3 --dataset multiarith \
    --cutoff -1 --use-async --concurrency 10
    ```

* Run sync benchmark on `aqua` (AQUA-RAT) dataset with default Ollama:
    ```bash
    poetry run python benches/run.py --dataset aqua
    ```

### Performance Metric

Accuracy is the primary metric used to evaluate the performance of the models on the datasets.
It is defined as the percentage of correct answers out of the total number of samples (questions) used during the
evaluation. For numerical answers, the final number is extracted from the model output before comparison.

### Datasets

The following datasets can be used in the benchmarking process:

| Dataset Name | Source Link                                                                          | Category Tags             | Description                                     |
|--------------|--------------------------------------------------------------------------------------|---------------------------|-------------------------------------------------|
| `gsm8k`      | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                         | `math`                    | Grade school math word problems                 |
| `multiarith` | [ChilleD/MultiArith](https://huggingface.co/datasets/ChilleD/MultiArith)             | `math`                    | Multi-step arithmetic problems                  |
| `aqua`       | [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat)               | `math`                    | Algebraic word problems with rationales (MCQ)   |
| `csqa`       | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)             | `commonsense`             | Multiple-choice commonsense questions           |
| `strategyqa` | [voidful/StrategyQA](https://huggingface.co/datasets/voidful/StrategyQA)             | `commonsense`, `symbolic` | Yes/no questions requiring implicit reasoning   |
| `coin`       | [skrishna/coin_flip](https://huggingface.co/datasets/skrishna/coin_flip)             | `symbolic`                | Symbolic tasks involving state tracking         |
| `letter`     | [ChilleD/LastLetterConcat](https://huggingface.co/datasets/ChilleD/LastLetterConcat) | `symbolic`, `text`        | Extract and concatenate last letters from words |
