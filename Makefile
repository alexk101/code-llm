.PHONY: lint format test install dev clean setup-hooks

# Default Python interpreter
PYTHON = python

# Install dependencies
install:
	uv pip install -e .

# Install development dependencies
dev:
	uv pip install -e ".[dev]"

# Install pre-commit hooks
setup-hooks:
	pre-commit install

# Format code
format:
	$(PYTHON) lint.py

# Lint code without making changes
lint-check:
	@echo "Running linting checks..."
	$(PYTHON) lint.py --check

# Run tests with coverage
test:
	pytest --cov=. --cov-report=term-missing

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Setup everything for development
setup: install dev setup-hooks
	@echo "Development environment set up successfully!" 