# Variables
PYTHON = python
PIP = pip
UV ?= uv
SHELL = /bin/bash
BENCH_DIR = benches
EXAMPLE_DIR = examples
OLLAMA_MODEL ?= gemma3:12b
OPENAI_MODE ?= gpt-4o-mini

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the help messages for all targets
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' Makefile | \
	awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Setup and installation
.PHONY: setup
setup: ## Install system dependencies
	sudo apt-get update
	sudo apt-get install -y python3-pip
	$(PIP) install uv

.PHONY: install
install: ## Install Python dependencies
	$(UV) sync --locked

# Testing and linting
.PHONY: test
test: ## Run the tests
	CUDA_VISIBLE_DEVICES="" $(UV) run pytest

.PHONY: lint
lint: ## Run the linter checks
	$(UV) run ruff check --fix

.PHONY: format
format: ## Format the Python files
	$(UV) run ruff format .

.PHONY: typecheck
typecheck: ## Typecheck the Python files
	$(UV) run mypy .

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
	$(UV) build

.PHONY: publish
publish: build ## Publish the library to PyPI (requires PYPI_TOKEN to be set)
	$(UV) publish --token "$(PYPI_TOKEN)"

.PHONY: example-openai
example-openai: ## Run the examples using OpenAI (needs OPENAI_API_KEY to be set)
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script --provider openai --openai-key ******** --model-name $(OPENAI_MODE) --use-async"; \
	   $(UV) run python $$script --provider openai --openai-key $(OPENAI_API_KEY) --model-name $(OPENAI_MODE) --use-async; \
	done

example-ollama: ## Run the examples using Ollama
	@echo "Running examples with Ollama provider (Model: $(OLLAMA_MODEL))"
	@for script in $(EXAMPLE_DIR)/run_*.py; do \
	   echo "Running $$script --provider ollama --model-name $(OLLAMA_MODEL)"; \
	   $(UV) run python $$script --provider ollama --model-name $(OLLAMA_MODEL); \
	done

.PHONY: docs
docs: ## Generate the project documentation
	$(UV) run mkdocs build

.PHONY: setup-hooks
setup-hooks: ## Install Git hooks (pre-commit and pre-push)
	$(UV) run pre-commit install --hook-type pre-commit
	$(UV) run pre-commit install --hook-type pre-push
	$(UV) run pre-commit install

.PHONY: test-hooks
test-hooks: ## Run Git hooks manually on all files
	$(UV) run pre-commit run --all-files --show-diff-on-failure
