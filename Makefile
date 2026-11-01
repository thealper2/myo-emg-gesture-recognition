# MYO EMG gesture recognition - run inside WSL/Ubuntu from the project root.
#
#   make setup          install the project (editable) + test deps into $(VENV)
#   make test           run the unit tests (synthetic data, no hardware)
#   make audit          audit the local EMG-EPN-612 dataset (read-only)
#   make prepare-data   parse the JSON dataset into data/processed + participant split
#   make hpo            Optuna hyper-parameter search for EMGNet (resumable; N_TRIALS=40 TIMEOUT_H=)
#   make train          final EMGNet training with the best HPO parameters
#   make finalize       re-run the post-training checks on the saved final_model (no training)
#   make evaluate       deploy the trained model + one-time held-out test evaluation
#   make pipeline       prepare-data -> hpo -> train -> evaluate (FULL run)
#   make smoke          tiny end-to-end run on a few users (separate data/artifact dirs)
#   make demo           dashboard with a SIMULATED stream (no device needed)
#   make bt-attach      attach the USB Bluetooth dongle to WSL (usbipd-win), then make bt-status
#   make myo-scan       list BLE devices and mark Myo armbands
#   make myo-check      headless 10 s streaming test with the physical Myo (writes a report)
#   make live           dashboard connected to the physical Myo through dl-myo
#   make clean          delete caches/temp files (keeps data and models)

VENV       ?= $(HOME)/venv
PYTHON     ?= $(VENV)/bin/python
PIP        ?= $(VENV)/bin/pip
EPN612_DIR ?= /mnt/d/Datasets/EMG-EPN612 Dataset
DL_MYO_DIR ?= /mnt/d/work2/dl-myo
WORKERS    ?= 8
N_TRIALS   ?=
TIMEOUT_H  ?=
ARGS       ?=
MAC        ?=
PORT       ?= 7860
DURATION   ?= 10
USBIPD     ?= /mnt/c/Program Files/usbipd-win/usbipd.exe
BT_HWID    ?= 0a12:0001

export EPN612_DIR
export DL_MYO_DIR
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

LIVE_PY    = $(PYTHON) scripts/live_demo.py
MAC_ARG    = $(if $(MAC),--mac $(MAC),)

SMOKE_ENV  = MYO_EMG_DATA_DIR=$(CURDIR)/data/smoke MYO_EMG_ARTIFACTS_DIR=$(CURDIR)/artifacts_smoke
SMOKE_SET  = --set hpo.epochs=2 --set hpo.patience=2 --set hpo.max_windows_per_epoch=5000              --set hpo.startup_trials=1 --set hpo.study_name=smoke --set train.epochs=2 --set train.patience=2              --set train.max_windows_per_epoch=20000 --set selection.max_latency_ms=50

.PHONY: help setup check-env test audit prepare-data hpo train finalize evaluate pipeline \
        smoke demo live myo-scan myo-check bt-attach bt-detach bt-status \
        clean clean-smoke clean-artifacts clean-data

help:
	@sed -n '1,19p' Makefile

check-env:
	@test -x "$(PYTHON)" || { echo "Python not found at $(PYTHON). Set VENV=/path/to/venv"; exit 1; }
	@test -d "$(EPN612_DIR)" || { echo "Dataset not found: $(EPN612_DIR) (set EPN612_DIR=...)"; exit 1; }

setup:
	@test -x "$(PYTHON)" || { echo "Python not found at $(PYTHON). Set VENV=/path/to/venv"; exit 1; }
	$(PIP) install -e ".[dev]"
	$(PYTHON) -c "import torch, sklearn, scipy, numpy, yaml; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

test:
	$(PYTHON) -m pytest tests $(ARGS)

audit: check-env
	$(PYTHON) scripts/audit_dataset.py --workers $(WORKERS) $(ARGS)

prepare-data: check-env
	$(PYTHON) scripts/prepare_dataset.py --workers $(WORKERS) $(ARGS)

hpo:
	$(PYTHON) scripts/hpo.py $(if $(N_TRIALS),--n-trials $(N_TRIALS),) $(if $(TIMEOUT_H),--timeout-h $(TIMEOUT_H),) $(ARGS)

train:
	$(PYTHON) scripts/train_model.py $(ARGS)

finalize:
	$(PYTHON) scripts/train_model.py --finalize-only $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate_model.py $(if $(FORCE),--force,) $(ARGS)

pipeline: prepare-data hpo train evaluate

smoke: check-env
	$(SMOKE_ENV) $(PYTHON) scripts/prepare_dataset.py --max-users 8 --workers 4
	$(SMOKE_ENV) $(PYTHON) scripts/hpo.py --n-trials 3 $(SMOKE_SET)
	$(SMOKE_ENV) $(PYTHON) scripts/train_model.py $(SMOKE_SET)
	$(SMOKE_ENV) $(PYTHON) scripts/evaluate_model.py $(SMOKE_SET) --force
	@echo "Smoke run finished: artifacts_smoke/ (8+8 users, NOT representative results)"

demo:
	$(LIVE_PY) $(ARGS) dashboard --mode demo --port $(PORT)

live:
	$(LIVE_PY) $(ARGS) dashboard --mode live --port $(PORT) --auto-connect $(MAC_ARG)

myo-scan:
	$(LIVE_PY) $(ARGS) scan --seconds $(DURATION)

myo-check:
	$(LIVE_PY) $(ARGS) myo-check --seconds $(DURATION) $(MAC_ARG)

# The dongle must be shared once from an *administrator* PowerShell: usbipd bind --hardware-id 0a12:0001
bt-attach:
	"$(USBIPD)" attach --wsl --hardware-id $(BT_HWID)
	@sleep 3; $(MAKE) --no-print-directory bt-status

bt-detach:
	"$(USBIPD)" detach --hardware-id $(BT_HWID)

bt-status:
	@ls /sys/class/bluetooth 2>/dev/null | grep -q hci && echo "adapter: $$(ls /sys/class/bluetooth | tr '\n' ' ')" \
	  || { echo "No Bluetooth adapter in WSL. Run: make bt-attach"; exit 1; }
	@echo "bluetooth.service: $$(systemctl is-active bluetooth)"
	@bluetoothctl --timeout 2 show 2>/dev/null | grep -E "Controller|Powered|LE" || true

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf build dist src/*.egg-info

clean-smoke:
	rm -rf data/smoke artifacts_smoke

# Destructive: removes trained models/reports (asks for confirmation).
clean-artifacts:
	@read -p "Delete artifacts/ (models, reports, splits)? [y/N] " a; [ "$$a" = y ] && rm -rf artifacts || echo skipped

# Destructive: removes the prepared dataset cache (the original dataset is never touched).
clean-data:
	@read -p "Delete data/processed (prepared cache)? [y/N] " a; [ "$$a" = y ] && rm -rf data/processed || echo skipped
