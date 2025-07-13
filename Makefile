# Variables
PYTHON = python
PIP = pip
POETRY = poetry
SHELL = /bin/bash
BENCH_DIR = benches
EXAMPLE_DIR = examples
OLLAMA_MODEL ?= gemma3:12b
OPENAI_MODE ?= gpt-4o-mini

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show help messages for all available targets
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Setup and installation
.PHONY: setup
setup: ## Install system dependencies
	sudo apt-get update
	sudo apt-get install -y python3-pip
	$(PIP) install poetry

.PHONY: install
install: ## Install Python dependencies
	$(POETRY) install --all-extras

# Testing and linting
.PHONY: test
test: ## Run the tests
	CUDA_VISIBLE_DEVICES="" $(POETRY) run pytest

.PHONY: lint
lint: ## Run the linter checks
	$(POETRY) run ruff check --fix

.PHONY: format
format: ## Format the Python files
	$(POETRY) run ruff format .

.PHONY: typecheck
typecheck: ## Typecheck the Python files
	$(POETRY) run mypy .

# Cleaning
.PHONY: clean
clean: ## Remove build artifacts, caches, and temporary files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf site dist
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov coverage.xml junit

# Build and publish
.PHONY: build
build: ## Build the wheel and source distribution
	$(POETRY) build

.PHONY: publish
publish: ## Publish the library to PyPI (requires PYPI_TOKEN to be set)
	$(POETRY) config pypi-token.pypi $(PYPI_TOKEN)
	$(POETRY) publish --build

.PHONY: example-openai
example-openai: ## Run the examples using OpenAI (needs OPENAI_API_KEY to be set)
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script --provider openai --openai-key ******** --model-name $(OPENAI_MODE) --use-async"; \
	   $(POETRY) run python $$script --provider openai --openai-key $(OPENAI_API_KEY) --model-name $(OPENAI_MODE) --use-async; \
	done

example-ollama: ## Run the examples using Ollama
	@echo "Running examples with Ollama provider (Model: $(OLLAMA_MODEL))"
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script --provider ollama --model-name $(OLLAMA_MODEL)"; \
	   $(POETRY) run python $$script --provider ollama --model-name $(OLLAMA_MODEL); \
	done

.PHONY: docs
docs: ## Generate the project documentation
	$(POETRY) run mkdocs build

# All-in-one target
.PHONY: all
all: install typecheck build ## Install Python dependencies, run lint, typecheck, tests, and build the library

.PHONY: setup-hooks
setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	$(POETRY) run pre-commit install --hook-type pre-commit
	$(POETRY) run pre-commit install --hook-type pre-push
	$(POETRY) run pre-commit install

.PHONY: test-hooks
test-hooks: ## Run Git hooks manually on all files
	$(POETRY) run pre-commit run --all-files --show-diff-on-failure
