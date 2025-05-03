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
[![PyPI Version](https://img.shields.io/pypi/v/cogitator.svg?style=flat&label=pypi&labelColor=333333&logo=pypi&logoColor=white&color=3775a9)](https://pypi.org/project/cogitator/)
[![Downloads](https://img.shields.io/pypi/dm/cogitator.svg?style=flat&label=downloads&labelColor=333333&logo=pypi&logoColor=white&color=cc8400)](https://pypi.org/project/cogitator/)
[![License](https://img.shields.io/badge/license-MIT-00acc1?style=flat&labelColor=333333&logo=open-source-initiative&logoColor=white)](https://github.com/habedi/cogitator/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-8ca0d7?style=flat&labelColor=333333&logo=readthedocs&logoColor=white)](https://github.com/habedi/cogitator/blob/main/docs)
[![DOI](https://img.shields.io/badge/doi-10.5281/zenodo.15331821-6f42c1.svg?style=flat&labelColor=333333&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15331821)

A Python toolkit for chain-of-thought prompting

</div>

---

Cogitator is a Python toolkit for experimenting and working with
[chain-of-thought (CoT) prompting](https://arxiv.org/abs/2201.11903)
techniques in large language models (LLMs).
CoT prompting improves LLM performance on complex tasks (like question-answering, reasoning, and problem-solving)
by guiding the models to generate intermediate reasoning steps before arriving at the final answer.
The toolkit aims to make it easier to try out popular CoT methods and frameworks for research or integrate them into AI
applications.

### Features

- Provides unified sync/async API for CoT methods
- Supports using OpenAI and Ollama as LLM providers
- Supports structured JSON model output with Pydantic validation
- Includes a customizable benchmarking framework (see [benches](benches))
- Includes implementations of popular CoT methods and frameworks like
    - [Self-Consistency CoT (ICLR 2023)](https://arxiv.org/abs/2203.11171)
    - [Automatic CoT (ICLR 2023)](https://arxiv.org/abs/2210.03493)
    - [Least-to-Most Prompting (ICLR 2023)](https://arxiv.org/abs/2205.10625)
    - [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
    - [Graph of Thoughts (AAAI 2024)](https://arxiv.org/abs/2308.09687)
    - [Clustered Distance-Weighted CoT (AAAI 2025)](https://arxiv.org/abs/2501.12226)

---

### Getting Started

```bash
pip install cogitator
```

Or, if you want to install from the latest version with examples and benchmarks included

```bash
git clone --depth=1 https://github.com/habedi/cogitator && cd cogitator

# Set up Python environment
pip install poetry
poetry install --with dev

# Run the tests to make sure everything is working (optional)
poetry run pytest
```

#### Examples

See the [examples](examples) directory for examples.

---

### Documentation

See the [docs](docs) directory for the documentation for the Cogitator toolkit.

---

### Benchmarking Framework

Cogitator project includes a customizable and extensible benchmarking framework to evaluate the performance of different
CoT methods on various datasets like [GSM8K](https://arxiv.org/abs/2110.14168) and [StrategyQA](https://arxiv.org/abs/2101.02235).

Check out the [benches](benches) directory for more details about the framework and how it could be used.

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### Logo

The logo is named "Cognition" and was originally created by
[vectordoodle](https://www.svgrepo.com/author/vectordoodle).

### License

This project is licensed under the [MIT License](LICENSE).
