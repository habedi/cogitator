# Variables
PYTHON = python
PIP = pip
POETRY = poetry

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show help messages for all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; \
 	{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install system dependencies
	sudo apt-get update
	sudo apt-get install -y python3-pip
	$(PIP) install poetry

.PHONY: install
install: ## Install Python dependencies
	$(POETRY) install --no-root --with=dev

.PHONY: update
update: ## Update Python dependencies
	$(POETRY) update

.PHONY: test
test: ## Run the tests
	$(POETRY) run pytest

.PHONY: lint
lint: ## Run the linter checks
	$(POETRY) run ruff check .

.PHONY: format
format: ## Format the Python files
	$(POETRY) run ruff format .

.PHONY: typecheck
typecheck: ## Type check the code
	$(POETRY) run mypy .

.PHONY: clean
clean: ## Remove temporary files and directories
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf junit

.PHONY: coverage
coverage: ## Run the tests with code coverage
	$(POETRY) run pytest --cov=src --cov-report=term-missing

.PHONY: codecov
codecov: ## Run tests with code coverage and generate report in XML format
	$(POETRY) run pytest --cov=src --cov-branch --cov-report=xml

.PHONY: build
build: ## Build the wheel and source distribution
	$(POETRY) build

.PHONY: publish
publish: ## Publish the library to PyPI (requires PYPI_TOKEN to be set)
	$(POETRY) config pypi-token.pypi $(PYPI_TOKEN)
	$(POETRY) publish --build

.PHONY: check
check: lint typecheck test ## Run linter checks, typechecking, and tests

.PHONY: precommit
precommit: ## Install and run pre-commit hooks
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit run --all-files

.PHONY: all
all: install check build ## Install Python dependencies, run lint, typecheck, tests, and build the library
