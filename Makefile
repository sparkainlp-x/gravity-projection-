PYTHON ?= python3

.PHONY: test clean

test:
	$(PYTHON) -m pytest tests/ -vv

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__
