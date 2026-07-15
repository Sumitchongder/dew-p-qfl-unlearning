.PHONY: install test lint pipeline clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 src/ scripts/ --max-line-length=110 --extend-ignore=E203,W503

pipeline:
	bash scripts/run_full_pipeline.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf *.egg-info build dist
