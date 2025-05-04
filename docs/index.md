# Welcome to Cogitator

Cogitator is a Python library designed for implementing, experimenting with, and benchmarking various Chain-of-Thought (CoT)
prompting techniques for Large Language Models (LLMs).

## Overview

The library provides:

* Implementations of popular CoT strategies (Self-Consistency, AutoCoT, Least-to-Most, Tree of Thoughts, Graph of Thoughts,
  CDW-CoT).
* A unified API supporting both synchronous and asynchronous operations.
* Support for different LLM backends (OpenAI, Ollama) via a common interface (`BaseLLM`).
* Utilities for text embedding and clustering (`SentenceTransformerEmbedder`, `KMeansClusterer`).
* Robust handling of structured JSON output using Pydantic validation (`schemas.py`).
* A comprehensive benchmarking framework (`benches/`) to evaluate and compare CoT methods.

## Getting Started

To install the library:

```bash
pip install cogitator
````

Check the main [README.md](https://github.com/habedi/cogitator/blob/main/README.md) on GitHub for basic usage examples, or explore
the [examples](https://www.google.com/search?q=./examples/) directory in the repository.

## Navigation

* **API Reference:** Detailed documentation for modules, classes, and functions.
    * [BaseLLM](https://www.google.com/search?q=api/model.md): The abstract base class for LLM providers.
    * [OpenAI](https://www.google.com/search?q=api/openai.md): OpenAI LLM implementation.
    * [Ollama](https://www.google.com/search?q=api/ollama.md): Ollama LLM implementation.
    * [Schemas](https://www.google.com/search?q=api/schemas.md): Pydantic models for structured data.
    * [Embedding](https://www.google.com/search?q=api/embedding.md): Text embedding utilities.
    * [Clustering](https://www.google.com/search?q=api/clustering.md): Clustering utilities.
    * Strategies:
        * [AutoCoT](https://www.google.com/search?q=api/auto_cot.md)
        * [CDWCoT](https://www.google.com/search?q=api/cdw_cot.md)
        * [GraphOfThoughts](https://www.google.com/search?q=api/graph_of_thoughts.md)
        * [LeastToMost](https://www.google.com/search?q=api/least_to_most.md)
        * [SelfConsistency](https://www.google.com/search?q=api/sc_cot.md)
        * [TreeOfThoughts](https://www.google.com/search?q=api/tree_of_thoughts.md)
* **Benchmarking:** Information on how to use the [benchmarking framework](https://www.google.com/search?q=benchmarking.md).
* **Contributing:** Guidelines for [contributing](https://www.google.com/search?q=contributing.md) to the project.

<!-- end list -->
