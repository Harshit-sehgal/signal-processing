# PG-AMCD Reproducibility and Validation Orchestration Makefile

PYTHON_ENV = Python/venv/bin/python
PIP_ENV = Python/venv/bin/pip

.PHONY: help install test run evaluate reproduce lint clean

help:
	@echo "PG-AMCD Reproducibility targets:"
	@echo "  install    - Install package in editable mode and dev dependencies"
	@echo "  test       - Run all unit and regression tests"
	@echo "  run        - Execute the CLI on the validation test directory"
	@echo "  evaluate   - Run ML training and evaluation script"
	@echo "  reproduce  - Run the entire verification and evaluation pipeline"
	@echo "  lint       - Run formatting and code quality checks (Ruff)"
	@echo "  clean      - Clean up temporary files and caches"

install:
	$(PIP_ENV) install -e .
	$(PIP_ENV) install -e .[dev]

test:
	PYTHONPATH=src $(PYTHON_ENV) -m pytest tests/

run:
	$(PYTHON_ENV) -m pg_amcd.cli run \
		--input-dir "Vibration - ML" \
		--output-dir "outputs/makefile_run" \
		--continue-on-error

evaluate:
	PYTHONPATH=src $(PYTHON_ENV) scripts/evaluate.py

reproduce: test evaluate
	@echo "=================================================="
	@echo "🎉 FULL REPRODUCTION RUN COMPLETED SUCCESSFULLY! 🎉"
	@echo "=================================================="

lint:
	$(PYTHON_ENV) -m ruff check src/ tests/ scripts/ || echo "Ruff check done"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ outputs/makefile_run/
