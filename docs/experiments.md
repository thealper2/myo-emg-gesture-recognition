# Experiments

## EMGNet (current model)

Not trained yet. `make hpo` writes the search results to `artifacts/hpo/hpo_report.md`; `make train`
and `make evaluate` write `artifacts/reports/`.

A timing probe on 2026-11-01 (full data, 2 epochs of 500k windows each, seed 0) gave:

| probe config | epoch time (RTX 5060 Ti) | CPU latency / window | val macro F1 after 2 epochs |
|---|---|---|---|
| 200 ms, 48 ch, 2 blocks, attention | ~5 s | 0.7 ms | 0.673 |
| 250 ms, 64 ch, 4 blocks, attention | ~7 s | 1.4 ms | 0.730 |
| 300 ms, 128 ch, 6 blocks, k=7, BiGRU 96 | ~30 s | 5.8 ms | 0.774 |

The first epoch takes 10–15 s longer because of cuDNN autotuning.

## Previous run (removed models, kept for reference)

Full run: `make pipeline` on 2026-11-01 (RTX 5060 Ti for the neural models, 12 CPU cores), default
`configs/training.yaml` (200 ms windows, 50 ms stride). Source: `artifacts/reports/model_comparison.md`,
`artifacts/reports/test_report.md`, `artifacts/logs/`.

## Validation participants (46 users, 317 890 windows), used for selection

| model | macro F1 | accuracy | balanced acc | recording acc | min class recall | CPU latency / window | size |
|---|---|---|---|---|---|---|---|
| **cnn** | **0.6906** | 0.6955 | 0.6921 | 0.8881 | 0.594 | 0.29 ms | 0.14 MB |
| mlp | 0.6369 | 0.6440 | 0.6401 | 0.8504 | 0.512 | 0.13 ms | 0.06 MB |
| extra_trees | 0.6250 | 0.6333 | 0.6297 | 0.8409 | 0.503 | 4.63 ms | 49.4 MB |
| random_forest | 0.6211 | 0.6280 | 0.6243 | 0.8359 | 0.494 | 4.48 ms | 47.1 MB |
| svm | 0.6158 | 0.6239 | 0.6194 | 0.8275 | 0.468 | 0.86 ms | 2.9 MB |
| logreg | 0.5772 | 0.5879 | 0.5828 | 0.7745 | 0.409 | 0.13 ms | <0.01 MB |

The CNN stopped early after epoch 22 (best epoch 16) and the MLP after epoch 15 (best epoch 9).
The CNN was selected because it has the highest validation macro F1 and the highest minimum
per-class recall, and it is fast and small.

## Held-out test participants (304 testingJSON users, 1 088 416 windows), evaluated once

| metric | validation | test |
|---|---|---|
| macro F1 | 0.6906 | 0.7055 |
| accuracy | 0.6955 | 0.7092 |
| balanced accuracy | 0.6921 | 0.7082 |
| recording-level accuracy | 0.8881 | 0.9052 |
| min class recall | 0.5935 | 0.6108 |

Per class (test): noGesture recall 0.965, fist 0.611, waveIn 0.725, waveOut 0.702, open 0.615,
pinch 0.632. `open` and `pinch` are the hardest classes.

Window-level numbers score every 200 ms window inside a gesture, including the onset. Recording-level
accuracy averages the window scores of one repetition. These numbers are for EMG-EPN-612 participants
recorded under the dataset protocol. They are not a measurement on a new Myo wearer.

## Physical Myo

See [hardware_setup.md](hardware_setup.md): connectivity and streaming were verified (212.5 samples/s,
0 invalid or duplicated packets, ~22 predictions/s, 3 ms mean end-to-end latency). Recognition
accuracy for the wearer was not measured, because no labelled data from the wearer is collected (by
design).
