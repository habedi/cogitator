# Variables
PYTHON = python
PIP = pip
POETRY = poetry
SHELL = /bin/bash
BENCH_DIR = benches
EXAMPLE_DIR = examples
OLLAMA_MODEL ?= gemma3

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show help messages for all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; \
 	{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Setup and installation
.PHONY: setup
setup: ## Install system dependencies
	sudo apt-get update
	sudo apt-get install -y python3-pip
	$(PIP) install poetry

.PHONY: install
install: ## Install Python dependencies
	$(POETRY) install --with dev

.PHONY: update
update: ## Update Python dependencies
	$(POETRY) update

# Testing and linting
.PHONY: test
test: ## Run the tests
	$(POETRY) run pytest

.PHONY: lint
lint: ## Run the linter checks
	$(POETRY) run ruff check --fix

.PHONY: format
format: ## Format the Python files
	$(POETRY) run ruff format .

.PHONY: typecheck
typecheck: ## Typecheck the code
	$(POETRY) run mypy .

# Cleaning
.PHONY: clean
clean: ## Remove temporary files and directories
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov coverage.xml junit

# Build and publish
.PHONY: build
build: ## Build the wheel and source distribution
	$(POETRY) build

.PHONY: publish
publish: ## Publish the library to PyPI (requires PYPI_TOKEN to be set)
	$(POETRY) config pypi-token.pypi $(PYPI_TOKEN)
	$(POETRY) publish --build

# Combined checks
.PHONY: check
check: lint typecheck test ## Run linter checks, typechecking, and tests

.PHONY: precommit
precommit: ## Install and run pre-commit hooks
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit run --all-files

# Benchmarks and examples
.PHONY: bench
bench: ## Run the benchmarks
	@$(POETRY) run python $(BENCH_DIR)/run.py

.PHONY: example
example: ## Run the examples
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script..."; \
	   $(POETRY) run python $$script --openai-key $(OPENAI_API_KEY); \
	done

example-ollama: ## Run the examples using Ollama (accepts OLLAMA_MODEL as an argument)
	@echo "Running examples with Ollama provider (Model: $(OLLAMA_MODEL))"
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script --provider ollama --ollama-model $(OLLAMA_MODEL)..."; \
	   $(POETRY) run python $$script --provider ollama --ollama-model $(OLLAMA_MODEL); \
	done

# All-in-one target
.PHONY: all
all: install check build ## Install Python dependencies, run lint, typecheck, tests, and build the library
