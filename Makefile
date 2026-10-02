.PHONY: install install-hf lint test eval

install:
	pip install -c constraints.txt -e ".[dev]"

install-hf:
	pip install --index-url https://download.pytorch.org/whl/cpu -c constraints.txt torch
	pip install -c constraints.txt --upgrade-strategy only-if-needed -e ".[dev,hf]"

lint:
	ruff check src tests evals && ruff format --check src tests evals && mypy

test:
	pytest

eval:
	python evals/run_evals.py
