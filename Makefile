PY ?= .venv/bin/python

.PHONY: setup test lint \
	test-config test-ingest test-validate test-features test-forecast test-optimise \
	test-backtest test-api test-dashboard \
	ingest build features forecast backtest refresh serve dashboard

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

test-config:
	$(PY) -m pytest -m config

test-ingest:
	$(PY) -m pytest -m ingest

test-validate:
	$(PY) -m pytest -m validate

test-features:
	$(PY) -m pytest -m features

test-forecast:
	$(PY) -m pytest -m forecast

test-optimise:
	$(PY) -m pytest -m optimise

test-backtest:
	$(PY) -m pytest -m backtest

test-api:
	$(PY) -m pytest -m api

test-dashboard:
	$(PY) -m pytest -m dashboard

ingest:
	$(PY) -m ercot_bess.ingest $(ARGS)

build:
	$(PY) -m ercot_bess.validate $(ARGS)

features:
	$(PY) -m ercot_bess.features $(ARGS)

forecast:
	$(PY) -m ercot_bess.forecast $(ARGS)

backtest:
	$(PY) -m ercot_bess.backtest $(ARGS)

# refresh every table end to end from the latest data, the local pipeline run
refresh:
	$(PY) -m ercot_bess.api.orchestrator $(ARGS)

serve:
	$(PY) -m uvicorn ercot_bess.api.app:app $(ARGS)

dashboard:
	$(PY) -m streamlit run src/ercot_bess/dashboard/app.py $(ARGS)
