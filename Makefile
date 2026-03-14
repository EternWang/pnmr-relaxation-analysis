.PHONY: install analysis test notebook

install:
	python -m pip install -e .[dev]

analysis:
	python scripts/run_full_analysis.py

test:
	pytest -q

notebook:
	jupyter notebook notebooks/01_pnmr_analysis_showcase.ipynb
