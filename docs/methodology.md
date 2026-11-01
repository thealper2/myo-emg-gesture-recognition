# Methodology

## Signal pipeline (identical offline and online)

```
raw int8 EMG (n, 8) @200 Hz
  -> sanitize (live stream only: non-finite samples replaced by the last valid sample and counted)
  -> optional causal Butterworth filter over the whole stream (default: none)
  -> window: 200/250/300 ms (chosen by HPO), new prediction every 10 samples (50 ms)
  -> per-channel z-score (fitted on training participants only)
  -> EMGNet -> 6 softmax scores
```

* Internal window layout is `(n_samples, 8)` with channel order `ch1..ch8` as stored in the dataset.
* The normalisation parameters are stored in the bundle (`bundle.json → normalization`) and in
  `artifacts/scalers/emgnet.json`: mean/std over all samples of the training participants' recordings.
* The filter (if enabled in `configs/training.yaml → preprocessing.filter`) is a causal
  `scipy.signal.sosfilt` Butterworth filter. Offline it runs over each full recording from a zero
  state; online `StreamingPredictor` carries the filter state across BLE packets, so no future
  samples are used. It is disabled by default because the Myo stream is already band-limited.

## Labelling windows

EMG-EPN-612 gives a per-sample `groundTruth` mask for every gesture recording.

* Gesture recordings: a window receives the gesture label when ≥ 75 % of its samples are inside the
  mask (`windowing.gesture_min_overlap`). Partially overlapping (transition) windows are dropped.
  Windows completely outside the mask are dropped by default
  (`windowing.rest_from_gesture_recordings: false`) because rest periods around a gesture are not
  explicitly validated.
* `noGesture` recordings: every window is `noGesture`, taken with a 40-sample stride
  (`rest_stride_samples`) so the class does not dominate the training set.
* Windows never cross recording boundaries; every window keeps its recording and participant ID.

## Leakage prevention

* Split by participant, never by window or recording. Train/validation come from `trainingJSON`;
  test participants are the official `testingJSON` users.
* Byte-identical recordings shared by different participants are detected during preparation:
  development users sharing recordings are kept in the same partition; test users sharing recordings
  with development users are removed from the test set.
* Model selection, early stopping, hyper-parameter search and thresholds use validation participants
  only. `scripts/evaluate_model.py` evaluates the selected bundle on the test participants once and
  refuses to overwrite existing test metrics unless `--force` is given.
* Every bundle stores the participant lists and a hash of the split; evaluation refuses a bundle
  trained with a different split.

## Model: EMGNet

One model is used: `src/myo_emg/models/emgnet.py`.

```
(B, L, 8) z-scored window
  stem: Conv1d(8 -> C, k) + BatchNorm + GELU
  n_blocks x residual block: Conv1d(k, dilation g^i) - BN - GELU - Dropout - Conv1d - BN - Squeeze-Excitation, + skip
  temporal head: attention pooling + mean pooling  |  BiGRU -> attention + mean pooling
  optional time-domain branch: log MAV, log RMS, log waveform length per channel (computed in-network)
  classifier: Dropout - Linear(head_dim) - GELU - Dropout - Linear(6)
```

Global pooling makes the network independent of the window length. The window only contains past
samples, so the non-causal convolutions inside it never see the future.

Training (`src/myo_emg/training/trainer.py`): windows are gathered on the GPU from the shared int8
signal; AdamW with linear warm-up + cosine decay per step; cross-entropy with optional class weights
(`none` / `balanced` / `sqrt`) and label smoothing; bf16 autocast on CUDA; gradient clipping; early
stopping on validation macro F1 with best-checkpoint restore; non-finite losses abort training.

Augmentation (training only), aimed at transfer to new wearers: per-window gain (electrode contact),
per-channel gain jitter, rotation of the 8 channels by ±1 electrode (armband worn rotated) and
Gaussian noise.

Note on class weights: with the windowing above, the six classes have 286k–334k training windows
each, so class weighting has a small effect. The main difficulty is variation between people, not
class imbalance. The HPO therefore treats the weighting as one more searched parameter.

## Hyper-parameter search

`src/myo_emg/training/hpo.py`, search space in `configs/training.yaml → hpo.search_space` (window
length, width, depth, kernel, dilation, SE, temporal head, GRU size, time-domain branch, dropout,
head size, learning rate, weight decay, batch size, label smoothing, class weighting, four augmentation
strengths). Optuna TPE (multivariate, 8 random start-up trials) + median pruning after 3 epochs. Each
trial trains for 12 epochs on a random 500k-window subset per epoch and is scored by its best
validation macro F1. Configurations slower than `selection.max_latency_ms` (10 ms, one CPU thread)
are rejected before training. The study is stored in SQLite and can be resumed. `make train` then
retrains the best configuration on all training windows for up to 60 epochs with patience 10.

Because many configurations are compared on the same 46 validation participants, the best HPO score
is optimistically biased. The held-out test participants give the unbiased estimate.

## Metrics

On validation and test: accuracy, macro F1, weighted F1, balanced accuracy, per-class
precision/recall/F1, confusion matrix and recording-level accuracy (mean window scores per recording),
plus training time, single-window CPU latency (normalisation, model, end-to-end) and model size.
Before deployment the model must stay under the latency (10 ms) and size (50 MB) limits.

## Interpreting scores

Model outputs are softmax / estimator scores. They are **not calibrated probabilities** (calibration
has not been evaluated). The live predictor reports `uncertain` when the (optionally smoothed) top
score is below `inference.confidence_threshold`.

`noGesture` means the relaxed hand as recorded in EMG-EPN-612. It is not an "unknown movement" class:
other movements (typing, grasping objects, rotating the forearm) may be classified as any of the five
gestures. Models trained on other people may also transfer imperfectly to a new user because of arm
anatomy, armband position/rotation and skin contact; the held-out test metrics estimate this only for
EMG-EPN-612 participants, not for a new Myo wearer.
