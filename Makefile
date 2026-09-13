.PHONY: test lint ci scenarios monte-carlo vision media

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

ci: lint test

scenarios:
	python -m sim.runner --all

monte-carlo:
	python -m sim.runner --all --monte-carlo 50

media:
	python tools/record_run.py sim/scenarios/nominal_calm.yaml --out media
	python tools/record_run.py sim/scenarios/wind_gust_8ms.yaml --out media

vision:
	cmake -S vision -B vision/build && cmake --build vision/build
