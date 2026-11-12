.PHONY: all data test run
all: data test run
data:
	python data/download.py
test:
	python -m pytest -q tests
run:
	python run_all.py
