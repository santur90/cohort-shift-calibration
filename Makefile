PYTHON ?= python3

.PHONY: data run check-data check-selection check-apl prepare e0 e1 e2 e3 e5 e6 smoke reproduce-check test

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

e1:
	$(PYTHON) scripts/03_source_experiments.py --experiment e1

e2:
	$(PYTHON) scripts/03_source_experiments.py --experiment e2

e3:
	$(PYTHON) scripts/04_cross_cohort.py

e5:
	$(PYTHON) scripts/05_few_label.py

e6:
	$(PYTHON) scripts/01_prepare.py --analysis-set e6a
	$(PYTHON) scripts/04_cross_cohort.py --analysis-set e6a
	$(PYTHON) scripts/01_prepare.py --analysis-set e6b
	$(PYTHON) scripts/04_cross_cohort.py --analysis-set e6b

smoke:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/smoke_pipeline.py

reproduce-check:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/smoke_pipeline.py --reproduce

test:
	$(PYTHON) -m pytest
