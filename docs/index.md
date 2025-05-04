# Cogitator Documentation

Cogitator is a Python library designed for implementing, experimenting with, and benchmarking various Chain-of-Thought (CoT)
prompting techniques for Large Language Models (LLMs).

## Overview

* Implementations of popular CoT strategies (Self-Consistency, AutoCoT, Least-to-Most, Tree of Thoughts, Graph of Thoughts,
  CDW-CoT).
* A unified API supporting both synchronous and asynchronous operations.
* Support for different LLM backends (OpenAI, Ollama) via a common interface (`BaseLLM`).
* Utilities for text embedding and clustering (`SentenceTransformerEmbedder`, `KMeansClusterer`).
* Robust handling of structured JSON output using Pydantic validation (`schemas.py`).
* A comprehensive benchmarking framework (`benches/`) to evaluate and compare CoT methods.

## Installation

Cogitator can be installed via pip using the following command:

```shell
pip install cogitator
```

### Examples

Check the [examples](https://github.com/habedi/cogitator/blob/main/examples) for usage examples on how to use the library with
different LLM providers and CoT strategies.

## Documentation

* **API Reference:** Detailed documentation for modules, classes, and functions.
    * [BaseLLM](api/model.md): The abstract base class for LLM providers.
    * [OpenAI](api/openai.md): OpenAI LLM implementation.
    * [Ollama](api/ollama.md): Ollama LLM implementation.
    * [Schemas](api/schemas.md): Pydantic models for structured data.
    * [Embedding](api/embedding.md): Text embedding utilities.
    * [Clustering](api/clustering.md): Clustering utilities.
    * Strategies:
        * [AutoCoT](api/auto_cot.md)
        * [CDWCoT](api/cdw_cot.md)
        * [GraphOfThoughts](api/graph_of_thoughts.md)
        * [LeastToMost](api/least_to_most.md)
        * [SelfConsistency](api/sc_cot.md)
        * [TreeOfThoughts](api/tree_of_thoughts.md)
* **Benchmarks:** Information on how to use the [benchmarking framework](benchmarks.md).
* **Contributing:** Guidelines for [contributing](contributing.md) to the project.

<!-- end list -->
