PYTHON ?= python3

.PHONY: data run smoke test

data:
	bash scripts/fetch_data.sh

run:
	$(PYTHON) scripts/00_inventory.py

smoke:
	$(PYTHON) -c "import cohortshift; from cohortshift import data, labels, features, models, calibration, conformal, shift, adapt, experiments, plots; print('cohortshift imports OK')"

test:
	$(PYTHON) -m pytest
