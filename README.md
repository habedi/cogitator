<div align="center">
  <picture>
    <img alt="Cogitator Logo" src="logo.svg" height="30%" width="30%">
  </picture>
<br>

<h2>Cogitator</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/cogitator/tests.yml?label=tests&style=flat&labelColor=333333&logo=github&logoColor=white)](https://github.com/habedi/cogitator/actions/workflows/tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/cogitator?style=flat&label=coverage&labelColor=333333&logo=codecov&logoColor=white)](https://codecov.io/gh/habedi/cogitator)
[![Code Quality](https://img.shields.io/codefactor/grade/github/habedi/cogitator?style=flat&label=code%20quality&labelColor=333333&logo=codefactor&logoColor=white)](https://www.codefactor.io/repository/github/habedi/cogitator)
[![Python Version](https://img.shields.io/badge/python-%3E=3.10-3776ab?style=flat&labelColor=333333&logo=python&logoColor=white)](https://github.com/habedi/cogitator)
[![PyPI Version](https://img.shields.io/pypi/v/cogitator.svg?style=flat&label=pypi&labelColor=333333&logo=pypi&logoColor=white&color=3775a9)](https://pypi.org/project/cogitator)
<br>
[![Docs](https://img.shields.io/badge/docs-read-8ca0d7?style=flat&labelColor=333333&logo=readthedocs&logoColor=white)](https://habedi.github.io/cogitator)
[![License](https://img.shields.io/badge/license-MIT-00acc1?style=flat&labelColor=333333&logo=open-source-initiative&logoColor=white)](https://github.com/habedi/cogitator/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/doi-10.5281/zenodo.15331821-6f42c1.svg?style=flat&labelColor=333333&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15331821)

A Python Toolkit for Chain-of-Thought Prompting

</div>

---

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
* Supports using OpenAI, Ollama, and OpenRouter as LLM providers
* Supports structured model output with Pydantic validation
* Includes a customizable benchmarking framework (see [benches](benches))
* Includes implementations of popular CoT strategies and frameworks like
    - [Self-Consistency CoT (ICLR 2023)](https://arxiv.org/abs/2203.11171)
    - [Automatic CoT (ICLR 2023)](https://arxiv.org/abs/2210.03493)
    - [Least-to-Most Prompting (ICLR 2023)](https://arxiv.org/abs/2205.10625)
    - [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
    - [Graph of Thoughts (AAAI 2024)](https://arxiv.org/abs/2308.09687)
    - [Clustered Distance-Weighted CoT (AAAI 2025)](https://arxiv.org/abs/2501.12226)

---

### Getting Started

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

#### Examples

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

Check out the [examples](examples) directory for more examples.

---

### Documentation

Cogitator documentation is available [here](https://habedi.github.io/cogitator).

---

### Benchmarking Framework

This project includes a customizable and extensible benchmarking framework to evaluate the performance of different
CoT strategies on various datasets like [GSM8K](https://arxiv.org/abs/2110.14168) and
[StrategyQA](https://arxiv.org/abs/2101.02235).

Check out the [benches](benches) directory for more details about the framework and how it could be used.

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### Citations

If you find this project useful, please give it a star!
If you have any questions or feedback, please use the discussion section of the repository or open an issue.
If you use this project in your research, please consider citing using the following information:

```bibtex
@software{abedi_cogitator_2025,
    author = {Abedi Firouzjaei, Hassan},
    title = {{Cogitator: A Python Toolkit for Chain-of-Thought Prompting}},
    year = {2025--},
    publisher = {Zenodo},
    doi = {10.5281/zenodo.15331821},
    url = {https://github.com/habedi/cogitator}
}
```

### License

Cogitator is licensed under the [MIT License](LICENSE).

### Acknowledgements

* The logo is named "Cognition" and was originally created by [vectordoodle](https://www.svgrepo.com/author/vectordoodle).
