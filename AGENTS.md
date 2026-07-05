# AGENTS.md

This file provides guidance to coding agents collaborating on this repository.

## Mission

Cogitator is a Python toolkit for experimenting and working with chain-of-thought prompting methods in large language models.
It provides unified interfaces and implementations of prompting strategies to improve reasoning performance and interpretability of model outputs.

The priority order of tasks is defined below.

1. Correctness of chain-of-thought strategies, including tree of thoughts, graph of thoughts, and self-consistency.
2. Standard-compliant implementations of local and API-based model providers.
3. Test suite coverage, including unit and integration cases.
4. Detailed error handling, structured schemas, and parameter validation.

## Core Rules

- Use English for code, comments, documentation, and tests.
- Prefer small, focused changes over broad refactoring.
- Add comments only when they clarify non-obvious behavior.
- Do not add features, error handling, or abstractions beyond what is needed for the current task.
- Keep dependencies small. Do not add heavy machine learning libraries or external packages without prior discussion.

## Writing Style

- Use Oxford commas in inline lists, such as "a, b, and c" instead of "a, b, c".
- Do not use em dashes. Restructure the sentence, or use a semicolon or period instead.
- Avoid colorful adjectives and adverbs, writing "graph generator" instead of "powerful graph generator".
- Prefer using noun phrases for checklist items instead of imperative verbs, such as writing "negative weight detection" instead of "detect negative
  weights".
- Write headings in markdown files in title case, for example, "Build from Source" instead of "Build from source". Minor words (a, an, the, and, but,
  or, for, in, on, at, to, by, of, from, and with) stay lowercase unless they are the first word.
- Write correct and complete sentences.
- Avoid made-up words, abbreviations, and colons in the middle of sentences.
- Do not use pretentious language.

## Repository Layout

- `cogitator/__init__.py` for package initialization and top-level exports.
- `cogitator/clustering.py` for clustering algorithms used by prompting strategies.
- `cogitator/embedding.py` for text embedding interfaces and implementations.
- `cogitator/schemas.py` for Pydantic data schemas for evaluations, outputs, and decompositions.
- `cogitator/utils.py` for evaluation metrics, token length estimation, and utility functions.
- `cogitator/model/` for language model provider interfaces and implementations.
- `cogitator/strategies/` for chain-of-thought strategy implementations.
- `tests/` for unit and integration test cases.
- `examples/` for example scripts demonstrating prompting strategies.
- `benches/` for the benchmarking framework to evaluate strategy performance.
- `docs/` for repository documentation.
- `Makefile` for script runner definitions for development tasks.

## Architecture

### Model Providers and Interfaces

The toolkit defines an abstract base class for language models.
Concrete implementations handle interaction with OpenAI, Ollama, and OpenRouter.
All providers support both synchronous and asynchronous execution paths.

### Prompting Strategies

Strategies implement specific reasoning workflows.
These strategies coordinate multiple model queries, manage intermediate states, and parse final answers.
Implementations include automatic chain-of-thought prompting, self-consistency, and tree of thoughts.

### Data Schemas and Validation

Incoming arguments and intermediate data structures are validated using Pydantic models.
This validation ensures that structural outputs, decompositions, and evaluation results conform to expected types.

## Python Conventions

- Python version `>=3.10` as declared in the package configuration.
- Dependency management using Poetry.
- Formatting, linting, and import sorting using Ruff.
- Typechecking using MyPy.
- Test execution using PyTest.
- Preference for pathlib Path objects, typed function signatures, and deterministic ordering in outputs.

## Required Validation

Run the relevant targets for any change:

| Target          | Command               | What It Runs                                           |
|-----------------|-----------------------|--------------------------------------------------------|
| Lint check      | `make lint`           | Ruff checks and automatic style corrections.           |
| Formatting      | `make format`         | Ruff code formatting alignment.                        |
| Type checks     | `make typecheck`      | MyPy static analysis.                                  |
| Unit tests      | `make test`           | PyTest validation of tools, strategies, and models.    |
| OpenAI examples | `make example-openai` | Verification of OpenAI provider using example scripts. |
| Ollama examples | `make example-ollama` | Verification of Ollama provider using example scripts. |

## First Contribution Flow

1. Review of relevant modules under `cogitator/`.
2. Minimal code changes required for the feature or fix.
3. Added or updated tests targeting the modified components.
4. Syntax, formatting, and type checks passing status.
5. Local test suite validation via `make test`.

## Testing Expectations

- Strategy validation using mock language model responses.
- Schema validation for intermediate outputs and Pydantic models.
- Execution verification for both synchronous and asynchronous modes.
- Coverage level maintenance for strategy and utility modules.

## Change Design Checklist

### Before Coding

1. Identification of affected modules, including providers, strategies, schemas, or utilities.
2. Definition of prompt templates and parameter boundaries.
3. Verification of token estimation and cost implications.
4. Backward compatibility check.

### Before Submitting

1. Success status for linter and formatter checks.
2. Strict type compliance validation with `mypy`.
3. Test suite coverage verification.
4. Execution validation using example scripts.

## Commit and Pull Request Hygiene

- Commit scope limitation to one logical change.
- Pull Request description details that include a behavioral change summary, verification of added or updated tests, and execution logs for local runs
  and tests.
