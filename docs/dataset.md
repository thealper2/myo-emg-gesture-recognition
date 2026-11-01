# EMG-EPN-612 – verified format

All facts below were verified against the local copy at `/mnt/d/Datasets/EMG-EPN612 Dataset` by
`make audit` (full report: `artifacts/reports/dataset/dataset_audit.{json,md}`, exclusion list:
`artifacts/reports/dataset/excluded_recordings.json`). The original files are only read.

## Layout

```
EMG-EPN612 Dataset/
  trainingJSON/userN/userN.json   306 users (user1..user306)  + photo.png / photo.jpg
  testingJSON/userN/userN.json    306 users (user1..user306)  + photo.png
```

User folder names repeat across the two partitions but are different people, so participant IDs
are always `<partition>/<user>` (e.g. `trainingJSON/user1`).

Each JSON file contains:

| key | content |
|---|---|
| `generalInfo` | `deviceModel: "Myo Armband"`, `samplingFrequencyInHertz: 200`, `recordingTimeInSeconds: 5`, `myoPredictionLabel: {noGesture:0, fist:1, waveIn:2, waveOut:3, open:4, pinch:5}` |
| `userInfo` | name, age, gender, handedness, arm measurements, recording date |
| `synchronizationGesture.samples` | 5 synchronisation recordings (~10 s, no class label) – not used |
| `trainingSamples` | 150 recordings: 25 per class |
| `testingSamples` | 150 recordings: 25 per class (labelled only in `trainingJSON`) |

Each recording: `emg = {ch1..ch8: [int]}` (int8 values −128..127, ~1000 samples = 5 s at 200 Hz),
`gestureName`, plus IMU streams at 50 Hz (`quaternion`, `gyroscope`, `accelerometer`) that are not used.
Gesture recordings also have `groundTruth` (per-sample 0/1 mask, same length as the EMG) and
`groundTruthIndex` (1-based, inclusive start/end). The mask and index agree for every recording.

## Audit results (full dataset)

| | trainingJSON | testingJSON |
|---|---|---|
| users | 306 | 306 |
| valid labelled recordings | 91 799 (300/user, 1 excluded) | 45 900 (`trainingSamples` only) |
| excluded | 1 (`trainingJSON/user10 trainingSamples/idx_113`: length 480 < 600) | 45 900 `testingSamples` without labels |
| sampling-rate metadata | 200 Hz (all) | 200 Hz (all) |
| recording length (samples) | 694–1112, median 996 | 616–1132, median 996 |
| gesture segment length (samples) | 77–600, mean ≈ 238–274 per class | 77–600 |
| EMG range | −128..127 | −128..127 |
| missing / non-finite values | none | none |

Class balance is exact: 25 repetitions × 6 classes × 2 sections per development user.

### Duplicated participants

`make prepare-data` hashes every recording and found two duplicated users:

| removed from test | duplicate of | evidence |
|---|---|---|
| `testingJSON/user290` | `trainingJSON/user172` | identical `userInfo` incl. timestamp; all 150 labelled recordings byte-identical (would leak training data into the test set) |
| `testingJSON/user298` | `testingJSON/user210` | identical `userInfo` and recordings (would be counted twice) |

The reasons are stored in `artifacts/splits/participant_split.json` → `test_excluded`. No development
users share recordings.

### Window counts (default 200 ms windows, 50 ms stride)

| split | participants | windows | noGesture | fist | waveIn | waveOut | open | pinch |
|---|---|---|---|---|---|---|---|---|
| train | 260 | 1 900 119 | 312 961 | 329 653 | 319 868 | 333 788 | 286 102 | 317 747 |
| validation | 46 | 317 890 | 55 233 | 54 063 | 53 863 | 55 735 | 48 434 | 50 562 |
| test | 304 | computed by `make evaluate` | | | | | | |

## Partitions used

* **Development (train + validation)**: all `trainingJSON` users, both labelled sections.
  Split by participant (seed 42, 15 % validation) → `artifacts/splits/participant_split.json`.
* **Held-out test**: all `testingJSON` users, labelled `trainingSamples` (their `testingSamples` have
  no labels in this release and are excluded). These participants are never used for fitting,
  normalisation or model selection.
