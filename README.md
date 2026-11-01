# MYO EMG Gesture Recognition (EMG-EPN-612)

Recognises six hand/wrist gestures from 8-channel Myo Armband EMG:

| index | label | meaning |
|---|---|---|
| 0 | `noGesture` | relaxed hand |
| 1 | `fist` | closed fist |
| 2 | `waveIn` | wrist flexion |
| 3 | `waveOut` | wrist extension |
| 4 | `open` | open hand |
| 5 | `pinch` | thumb-and-finger pinch |

The project has two strictly separated stages:

* **Stage A – offline model development (implemented).** One model, **EMGNet** (a residual dilated
  temporal CNN with channel attention, an attention/BiGRU head and an in-network time-domain
  feature branch), is tuned with an Optuna hyper-parameter search and trained only on the local
  EMG-EPN-612 dataset with participant-independent splits. No Myo device is needed.
* **Stage B – real-time inference on the user's Myo (implemented, connectivity verified).** The
  saved model is loaded (never retrained) and fed live EMG from the physical Myo through the local
  `dl-myo` library. A Gradio dashboard shows the signal, predictions and diagnostics. A mock stream
  (clearly labelled SIMULATED) allows testing without hardware. The physical Myo is never used for
  training, calibration or model selection.

The four modes are kept separate:

| mode | command | data |
|---|---|---|
| offline search + training | `make hpo`, `make train` | EMG-EPN-612 train/validation participants |
| offline held-out evaluation | `make evaluate` | EMG-EPN-612 test participants (once) |
| mock-stream inference | `make demo` | **SIMULATED**: replay of test recordings or synthetic noise |
| physical Myo inference | `make live` | live BLE stream from your Myo via dl-myo |

## Requirements

* WSL2 Ubuntu 24.04, Python 3.12 virtualenv at `~/venv` (override with `VENV=...`).
* Dataset at `/mnt/d/Datasets/EMG-EPN612 Dataset` (override with `EPN612_DIR=...`). It is only read.
* Optional NVIDIA GPU for the neural models (CPU fallback is automatic).

Install the project (editable) and test dependencies into the virtualenv:

```bash
make setup
```

All dependencies (numpy, scipy, scikit-learn, torch, pyyaml, optuna, matplotlib, pytest) were already
present in `~/venv`; `make setup` only registers this package. The Makefile also sets
`PYTHONPATH=src`, so every target works even without `make setup`.

## Commands

Run all commands from the project root inside WSL:

| command | what it does |
|---|---|
| `make test` | unit + integration tests on synthetic arrays (no dataset, no hardware) |
| `make audit` | read-only audit → `artifacts/reports/dataset/dataset_audit.{json,md}` |
| `make prepare-data` | parse JSON → `data/processed/epn612_{dev,test}.npz`, create `artifacts/splits/participant_split.json` |
| `make hpo` | Optuna search for EMGNet on the validation participants (`N_TRIALS=40`, `TIMEOUT_H=`; resumable) → `artifacts/hpo/` |
| `make train` | final EMGNet training with the best HPO parameters on all training participants → `artifacts/checkpoints/final_model/` |
| `make finalize` | no training: re-run the reload check, CPU latency and reports on the saved `final_model` (if `make train` stopped after saving) |
| `make evaluate` | deployment checks → `artifacts/checkpoints/best_model/` → one held-out test evaluation |
| `make pipeline` | `prepare-data` → `hpo` → `train` → `evaluate` (full run) |
| `make smoke` | tiny end-to-end run on 8+8 users into `data/smoke/`, `artifacts_smoke/` (not representative) |
| `make demo` | dashboard at http://localhost:7860 with a SIMULATED stream (no device needed) |
| `make bt-attach` / `bt-status` / `bt-detach` | forward the USB Bluetooth dongle into WSL (usbipd-win) / check / release |
| `make myo-scan` | list BLE devices and mark Myo armbands |
| `make myo-check MAC=...` | 10 s headless stream from the physical Myo → `artifacts/reports/hardware/` |
| `make live MAC=...` | dashboard connected to the physical Myo (`MAC` optional: scan by service UUID) |
| `make clean` | delete `__pycache__`, pytest/ruff caches, build files (keeps data and models) |
| `make clean-smoke` | delete smoke-run outputs |
| `make clean-artifacts` / `make clean-data` | delete trained models / prepared cache (asks first) |

Extra arguments go through `ARGS`, configuration overrides through `--set`:

```bash
make hpo N_TRIALS=20 TIMEOUT_H=2
make train ARGS="--set train.epochs=80"        # best HPO params + your overrides
make train ARGS="--no-hpo"                       # ignore the HPO result, use configs/training.yaml
```

### Full training run

```bash
cd /mnt/d/work2/myo-emg-gesture-recognition
make pipeline
```

or step by step (`make prepare-data`, `make hpo`, `make train`, `make evaluate`). Logs
go to the terminal and to `artifacts/logs/`. `make hpo` stores the study in `artifacts/hpo/study.db`:
running it again adds trials to the same study (delete the file to start a new search). If you retrain after a test evaluation, re-run the
evaluation with `make evaluate FORCE=1`. That re-run is a new use of the test set, so record it.

## Dataset

See [docs/dataset.md](docs/dataset.md) for the verified format. In short: `trainingJSON/` and
`testingJSON/` each hold 306 users, one JSON per user with 150 `trainingSamples` and 150
`testingSamples` (25 per class), 8 int8 EMG channels at 200 Hz, 5 s per recording, and a per-sample
`groundTruth` mask for gesture recordings. `testingJSON` users have unlabelled `testingSamples`,
which are excluded.

* train/validation: `trainingJSON` users split by participant: 260 train / 46 validation (seed 42);
* test: 304 `testingJSON` users (labelled `trainingSamples`), never used for fitting or selection;
* `testingJSON/user290` (byte-identical copy of `trainingJSON/user172`) and `testingJSON/user298`
  (copy of `testingJSON/user210`) are removed from the test set automatically.
* With the default windowing: 1.90 M training windows and 0.32 M validation windows, roughly
  balanced across the six classes.

## Method

See [docs/methodology.md](docs/methodology.md). Key choices:

* 200 ms windows (40 samples) with a 50 ms stride, configurable to 100 / 250 ms.
* Window labels come from the `groundTruth` mask (≥ 75 % overlap); transition windows are dropped.
* One shared preprocessing implementation (`myo_emg.data.preprocess`, `myo_emg.features`) for
  training and inference; normalisation is fitted on training participants only.
* Hyper-parameters (window length, architecture, optimiser, class weighting, label smoothing,
  augmentation) are chosen by TPE + median pruning on validation macro F1. Configurations slower than
  10 ms per window on one CPU thread are rejected. The test participants are never used for this.
* Augmentation for cross-user robustness: per-window gain, per-channel gain jitter, ±1 electrode
  channel rotation (armband rotation) and Gaussian noise.

## Outputs

```
artifacts/
  reports/dataset/            dataset audit (JSON + Markdown) and exclusion list
  splits/participant_split.json
  hpo/study.db, best_params.yaml, hpo_report.md, trials.csv, best_trial.json
  checkpoints/final_model/    trained EMGNet bundle (bundle.json + model.pt) with validation metrics
  checkpoints/best_model/     deployed copy incl. the held-out test metrics (used by demo/live)
  checkpoints/archive/        previously deployed models (kept, not deleted)
  scalers/emgnet.json         fitted normalisation parameters
  reports/training_history.json, validation_metrics.json, validation_confusion.png
  reports/test_report.md, test_metrics.json, test_confusion.png
  logs/
```

`bundle.json` holds everything inference needs: class order, sampling rate, channel order, window
length/stride, windowing rules, filter, normalisation parameters, feature settings, architecture,
training configuration, dataset version, split participant lists and hash, validation/test metrics,
CPU latency and software versions. Load it with:

```python
from myo_emg.inference.predictor import load_pipeline, StreamingPredictor

pipe = load_pipeline("artifacts/checkpoints/best_model")          # no training, no re-fitting
scores = pipe.predict_proba(windows)        # windows: (N, pipe.window_length, 8) raw counts
live = StreamingPredictor(pipe, threshold=0.6, smoother={"method": "ema", "alpha": 0.5})
predictions = live.push(samples)                                   # samples: (n, 8) as they arrive
```

`data/`, `artifacts/` and `artifacts_smoke/` are git-ignored.

## Results

**EMGNet: not trained yet.** Until then `make demo` / `make live` stop with a clear error, because
the previously deployed CNN bundle uses the old format. Run `make hpo`, `make train` and `make evaluate`; the results will be in
`artifacts/hpo/hpo_report.md`, `artifacts/reports/validation_metrics.json` and
`artifacts/reports/test_report.md`.

A short timing probe (2 epochs of 500k windows each, not a full training) gave a validation macro F1
of 0.673 / 0.730 / 0.774 for a small / medium / large EMGNet configuration.

Previous run with the removed models (validation macro F1): 1-D CNN 0.691, MLP 0.637, Extra Trees
0.625, Random Forest 0.621, SVM 0.616, LogReg 0.577. The CNN reached a held-out test macro F1 of
0.706 (details in [docs/experiments.md](docs/experiments.md)).

## Interpreting predictions

* Scores are softmax/estimator outputs, **not calibrated probabilities**.
* Below `confidence_threshold` (`configs/inference.yaml`) a prediction is shown as `uncertain`.
* `noGesture` is the relaxed hand of the dataset, not a detector for arbitrary other movements.
* The model was trained on other people. Performance on a new wearer depends on armband position and
  rotation, arm anatomy and electrode contact, and may be lower than the held-out test numbers.

## Physical Myo (Stage B)

Full guide: [docs/hardware_setup.md](docs/hardware_setup.md); problems:
[docs/troubleshooting.md](docs/troubleshooting.md).

* WSL2 cannot use the PC's Bluetooth radio. A USB Bluetooth dongle (here CSR `0a12:0001`) is
  forwarded into WSL with usbipd-win (`make bt-attach`). BlueZ and `btusb` then work without `sudo`.
* `src/myo_emg/device/dl_myo_adapter.py` uses the local dl-myo checkout (`DL_MYO_DIR`, default
  `/mnt/d/work2/dl-myo`, unmodified): `MyoClient.with_device` → `setup(EMGMode.SEND_EMG)` →
  `start()` → `on_emg_data` (2 int8 samples per BLE notification) → thread-safe queue → inference
  thread → dashboard.
* **Verified on 2026-11-01** with the physical armband `D7:50:04:F0:88:9A`: connect, stream
  (3194 samples in 15 s, 0 invalid or duplicated packets, 0 drops), live dashboard (~22 predictions/s,
  end-to-end latency mean 3.2 ms / p95 10.3 ms) and disconnect all worked.
* The armband delivers ~212 samples/s over BLE, while the dataset says 200 Hz. No resampling is
  applied (same firmware stream); the dashboard shows the rate and warns at >10 % deviation.
* Gesture accuracy on your own arm has **not** been measured.

Real-time pipeline: BLE thread → `SampleQueue` (bounded, drops oldest and counts) → inference thread
(`StreamingPredictor`: rolling buffer, saved filter/normalisation, model, smoothing, threshold) →
the dashboard reads snapshots every 0.25 s (plot every 0.5 s). The UI cannot block acquisition. The
live mode only accepts the physical Myo and never falls back to mock data. Connection failures are
shown in the dashboard. Measured: preprocessing and model latency per window, end-to-end latency
(BLE packet arrival → prediction), prediction rate, queue drops and missing samples relative to the
expected rate.

## Tests

```bash
make test
```

The tests cover parsing, invalid and missing data, 8-channel validation, windowing, label mapping,
split integrity (no participant overlap), train-only normalisation, features, model forward passes,
EMGNet variants, augmentation, class weights, the LR schedule, an HPO → train → test smoke run
(including study resume), bundle save/reload consistency, rejection of old bundle formats, streaming prediction,
confidence threshold, smoothing, the sample queue, mock streams, disconnection handling (including
no fallback to mock data in live mode), inference without a device and dashboard startup. They use
synthetic arrays only and never touch Bluetooth hardware.
