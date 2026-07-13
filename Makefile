# PG-AMCD Reproducibility and Validation Orchestration Makefile

PYTHON_ENV = Python/venv/bin/python
PIP_ENV = Python/venv/bin/pip

.PHONY: help install test run baseline evaluate report statistics reproduce lint clean validate

help:
	@echo "PG-AMCD Reproducibility targets:"
	@echo "  install    - Install package in editable mode and dev dependencies"
	@echo "  test       - Run all unit and regression tests"
	@echo "  run        - Execute the CLI on the validation test directory"
	@echo "  baseline   - Run synthetic denoising baseline comparison (Segment 5)"
	@echo "  evaluate   - Run evaluation pipeline on labelled data"
	@echo "  report     - Generate Markdown evaluation report"
	@echo "  statistics  - Bootstrap CI + McNemar significance (Segment 7)"
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
	PYTHONPATH=src $(PYTHON_ENV) scripts/evaluate_dataset.py \
		--mat-dir Vibration_Clean --output outputs/evaluation_results.json

validate:
	PYTHONPATH=src $(PYTHON_ENV) scripts/validate_research.py

baseline:
	PYTHONPATH=src $(PYTHON_ENV) scripts/compare_baselines.py

report:
	PYTHONPATH=src $(PYTHON_ENV) scripts/generate_report.py

statistics:
	PYTHONPATH=src $(PYTHON_ENV) scripts/run_statistics.py

reproduce: test baseline evaluate report statistics
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
