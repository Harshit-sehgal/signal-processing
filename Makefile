# PG-AMCD Stage 1--4 reproducibility targets

PYTHON ?= python3
OUTPUT_DIR ?= outputs
CONFIG ?= configs/default.json

.PHONY: help install test typecheck lint quality run validate report clean

help:
	@echo "PG-AMCD Stage 1--4 targets"
	@echo "  install   Install the package and development dependencies"
	@echo "  quality   Run Ruff, MyPy, and the 90% coverage gate"
	@echo "  run       Run through Stage 4 (requires INPUT_DIR and METADATA in physics mode)"
	@echo "  validate  Validate INPUT_DIR and METADATA without processing"
	@echo "  report    Regenerate a report for RUN_DIR"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=pg_amcd --cov-report=term-missing --cov-fail-under=90

typecheck:
	$(PYTHON) -m mypy src/pg_amcd

lint:
	$(PYTHON) -m ruff check .

quality: lint typecheck test

run:
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required" && exit 2)
	$(PYTHON) -m pg_amcd.cli run \
		--input-dir "$(INPUT_DIR)" \
		$(if $(METADATA),--metadata "$(METADATA)",) \
		--output-dir "$(OUTPUT_DIR)" \
		--config "$(CONFIG)" \
		--through-stage 4

validate:
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required" && exit 2)
	$(PYTHON) -m pg_amcd.cli validate \
		--input-dir "$(INPUT_DIR)" \
		$(if $(METADATA),--metadata "$(METADATA)",) \
		--config "$(CONFIG)" \
		--output validation_report.json

report:
	@test -n "$(RUN_DIR)" || (echo "RUN_DIR is required" && exit 2)
	$(PYTHON) -m pg_amcd.cli report --run-dir "$(RUN_DIR)"

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
