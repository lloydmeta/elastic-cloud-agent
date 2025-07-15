.PHONY: install format lint lint-fix test clean run help

PYTHON = poetry run python
APP_NAME = elastic_cloud_agent

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make format    - Format code"
	@echo "  make lint      - Run linters"
	@echo "  make lint-fix  - Auto-fix linting errors"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up cache files"
	@echo "  make run       - Run the application"

install:
	poetry install

format:
	$(PYTHON) -m black $(APP_NAME) tests
	$(PYTHON) -m isort $(APP_NAME) tests

lint:
	$(PYTHON) -m ruff check $(APP_NAME) tests
	$(PYTHON) -m mypy $(APP_NAME) tests

lint-fix:
	$(PYTHON) -m ruff check --fix $(APP_NAME) tests

test:
	$(PYTHON) -m pytest $(ARGS)

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf **/__pycache__

run:
	$(PYTHON) -m $(APP_NAME).main

.DEFAULT_GOAL := help