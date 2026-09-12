.PHONY: test lint ci scenarios monte-carlo vision

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

vision:
	cmake -S vision -B vision/build && cmake --build vision/build
