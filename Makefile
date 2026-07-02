.PHONY: help install install-dev test lint format type-check clean setup pre-commit

help:
	@echo "Available commands:"
	@echo "  make setup          - Initial project setup (venv + dependencies + pre-commit)"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with ruff"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make pre-commit     - Install pre-commit hooks"
	@echo "  make clean          - Remove generated files"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	@echo "Then run: make install-dev && make pre-commit"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

type-check:
	mypy gateway services shared tests

pre-commit:
	pre-commit install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ build/ 2>/dev/null || true
