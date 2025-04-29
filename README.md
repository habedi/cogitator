<div align="center">
  <picture>
    <img alt="Cogitator Logo" src="logo.svg" height="30%" width="30%">
  </picture>
<br>

<h2>Cogitatør</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/cogitator/tests.yml?label=tests&style=flat&labelColor=555555&logo=github)](https://github.com/habedi/cogitator/actions/workflows/tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/habedi/cogitator?style=flat&labelColor=555555&logo=codecov)](https://codecov.io/gh/habedi/cogitator)
[![Code Quality](https://img.shields.io/codefactor/grade/github/habedi/cogitator?style=flat&labelColor=555555&logo=codefactor)](https://www.codefactor.io/repository/github/habedi/cogitator)
[![PyPI Version](https://img.shields.io/pypi/v/cogitator.svg?style=flat&labelColor=555555&logo=pypi)](https://pypi.org/project/cogitator/)
[![Downloads](https://img.shields.io/pypi/dm/cogitator.svg?style=flat&labelColor=555555&logo=pypi)](https://pypi.org/project/cogitator/)
[![Python Version](https://img.shields.io/badge/python-%3E=3.10-3776ab?style=flat&labelColor=555555&logo=python)](https://github.com/habedi/cogitator)
[![Documentation](https://img.shields.io/badge/docs-latest-3776ab?style=flat&labelColor=555555&logo=read-the-docs)](https://github.com/habedi/cogitator/blob/main/docs)
[![License](https://img.shields.io/badge/license-MIT-00acc1?style=flat&labelColor=555555&logo=open-source-initiative)](https://github.com/habedi/cogitator/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange?style=flat&labelColor=555555&logo=github)](https://github.com/habedi/cogitator)

A Python toolkit for chain-of-thought prompting

</div>

---

Cogitatør is a Python toolkit for experimenting with chain-of-thought (CoT) prompting techniques in large language
models (LLMs).
CoT prompting improves LLM performance on complex tasks (like question-answering, reasoning, and problem-solving)
by guiding the models to generate intermediate reasoning steps before arriving at the final answer.
The toolkit aims to make it easy to try out different popular CoT strategies (or methods) and integrate them
into your AI applications.

### Features

- Simple unified API for different CoT prompting methods
- Support for remote and local LLM providers, including:
    - OpenAI
    - Ollama
- Supported CoT prompting methods include:
    - [Self-Consistency CoT (ICLR 2023)](https://arxiv.org/abs/2203.11171)
    - [Automatic CoT (ICLR 2023)](https://arxiv.org/abs/2210.03493)
    - [Least-to-Most Prompting (ICLR 2023)](https://arxiv.org/abs/2205.10625)
    - [Tree of Thoughts (NeurIPS 2023)](https://arxiv.org/abs/2305.10601)
    - [Graph of Thoughts (AAAI 2024)](https://arxiv.org/abs/2308.09687)
    - [Clustered Distance-Weighted CoT (AAAI 2025)](https://arxiv.org/abs/2501.12226)

---

### Getting Started

```bash
pip install cogitator[dev]
```

#### Examples

| File                                                          | Description                                                           |
|---------------------------------------------------------------|-----------------------------------------------------------------------|
| [run_least_to_most.py](examples/run_least_to_most.py)         | Example of using the Least-to-Most prompting method                   |
| [run_sc_cot.py](examples/run_sc_cot.py)                       | Example of using the Self-Consistency prompting method                |
| [run_auto_cot.py](examples/run_auto_cot.py)                   | Example of using the Automatic CoT prompting method                   |
| [run_tree_of_thoughts.py](examples/run_tree_of_thoughts.py)   | Example of using the Tree of Thoughts prompting method                |
| [run_graph_of_thoughts.py](examples/run_graph_of_thoughts.py) | Example of using the Graph of Thoughts prompting method               |
| [run_cdw_cot.py](examples/run_cdw_cot.py)                     | Example of using the Clustered Distance-Weighted CoT prompting method |
| [shared.py](examples/shared.py)                               | Shared utilies and settings for the examples                          |

---

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### Logo

The logo is named the "Cognition" and was created by [vectordoodle](https://www.svgrepo.com/author/vectordoodle).

### License

This project is licensed under the [MIT License](LICENSE).
