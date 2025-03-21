.PHONY: install data forecast train optimize serve test lint fmt all clean

install:
	pip install -r requirements-dev.txt

data:
	python data/generate_data.py --n-days 913

forecast train:
	PYTHONPATH=src python -m demand_forecast.forecast

optimize:
	PYTHONPATH=src python -m demand_forecast.optimize

serve:
	PYTHONPATH=src uvicorn demand_forecast.api:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

fmt:
	ruff check . --fix

# Full pipeline from scratch, in order.
all: data forecast optimize test

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
