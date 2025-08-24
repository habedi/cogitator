# Cogitator Documentation

## Overview
Cogitator is a Python toolkit for experimenting and working with
[chain-of-thought (CoT) prompting](https://arxiv.org/abs/2201.11903)
methods in large language models (LLMs).
CoT prompting improves LLM performance on complex tasks (like question-answering, reasoning, and problem-solving)
by guiding the models to generate intermediate reasoning steps before arriving at the final answer.
Additionally, it can be used to improve the interpretability of LLMs by providing insight into the model's reasoning process.
The toolkit aims to make it easier to use popular CoT strategies and frameworks for research or integrating them into AI
applications.

### Features

* Provides unified sync/async API for CoT strategies
* Supports using OpenAI and Ollama as LLM providers
* Supports structured model output with Pydantic validation
* Includes a customizable benchmarking framework (see [benches](https://github.com/habedi/cogitator/blob/main/benches))
* Includes implementations of popular CoT strategies and frameworks like
    - [Self-Consistency CoT (ICLR 2023)](https://arxiv.org/abs/2203.11171)
    - [Automatic CoT (ICLR 2023)](https://arxiv.org/abs/2210.03493)
    - [Least-to-Most Prompting (ICLR 2023)](https://arxiv.org/abs/2205.10625)
    - [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
    - [Graph of Thoughts (AAAI 2024)](https://arxiv.org/abs/2308.09687)
    - [Clustered Distance-Weighted CoT (AAAI 2025)](https://arxiv.org/abs/2501.12226)

The diagram below shows a high-level overview of Cogitator's workflow.

![Cogitator Architecture](assets/images/cogitator_v2.svg)

## Getting Started

You can install Cogitator with

```bash
pip install cogitator
```

Or, if you want to install from the latest version with examples and benchmarks included

```bash
git clone https://github.com/habedi/cogitator && cd cogitator

# Set up Python environment (use Poetry 2.0+)
pip install poetry
poetry install --all-extras

# Run the tests to make sure everything is working (optional)
poetry run pytest
```

### Examples

Below is a simple example of using the Self-Consistency CoT with Ollama.

```python
import logging
from cogitator import SelfConsistency, OllamaLLM

# Step 1: Configure logging (optional, but helpful)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTPX logs

# Step 2: Initialize the LLM (using Ollama)
# Needs Ollama running locally with the model pulled (e.g., `ollama pull gemma3:4b`)
try:
    llm = OllamaLLM(model="gemma3:4b")
except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    print("Please make sure Ollama is running and the model is pulled.")
    exit(1)

# Step 3: Choose a CoT strategy (Self-Consistency in this case)
# Self-Consistency generates multiple reasoning paths and finds the most common answer
sc_strategy = SelfConsistency(
    llm,
    n_samples=5,  # Number of reasoning paths to generate
    temperature=0.7  # Higher temperature can lead to more diverse answers
)

# Step 4: Define the prompt (with a basic CoT trigger)
question = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
prompt = f"Q: {question}\nA: Let's think step by step."

# Step 5: Run the CoT prompting sc_strategy
print(f"\nQuestion: {question}")
print("Running Self-Consistency CoT...")
final_answer = sc_strategy.run(prompt)  # Returns the most consistent (repeated) answer

# Expected output: $0.05 or 0.05 (may vary slightly based on model and temperature)
print(f"\nCogitator's Answer (Self-Consistency): {final_answer}")
```

Check out the [examples](https://github.com/habedi/cogitator/blob/main/examples) directory for more examples.

## API Reference

For a complete list of all available modules, classes, and functions, please refer to the API Reference section.

## Extra Resources

* **[Benchmarking](benchmarking.md):** Learn how to configure and run the performance evaluation framework.
* **[Contributing](contributing.md):** Find guidelines for contributing to the Cogitator project.

<!-- end list -->
