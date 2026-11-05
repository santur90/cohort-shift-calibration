PYTHON ?= python3

.PHONY: data run check-data check-selection check-apl prepare e0 smoke test

data:
	bash scripts/fetch_data.sh

run:
	$(PYTHON) scripts/00_inventory.py

check-data:
	$(PYTHON) scripts/00_inventory.py --check-only

check-selection:
	$(PYTHON) scripts/01_check_selection.py

check-apl:
	$(PYTHON) scripts/check_apl.py

prepare:
	$(PYTHON) scripts/01_prepare.py

e0:
	$(PYTHON) scripts/02_e0_shift.py

smoke:
	$(PYTHON) -c "import cohortshift; from cohortshift import data, labels, features, models, calibration, conformal, shift, adapt, experiments, plots; print('cohortshift imports OK')"

test:
	$(PYTHON) -m pytest
