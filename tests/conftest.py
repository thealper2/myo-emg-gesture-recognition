"""Shared fixtures: a tiny *synthetic* dataset in the EMG-EPN-612 JSON layout.

Synthetic arrays are used only to test code paths; they say nothing about real EMG-EPN-612 or
Myo performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myo_emg.utils.config import load_config

CLASSES = ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]
# per-class channel gain patterns so that synthetic gestures are separable
GAINS = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [8, 8, 8, 8, 2, 2, 2, 2],
        [2, 2, 8, 8, 8, 2, 2, 2],
        [2, 2, 2, 2, 8, 8, 8, 2],
        [8, 2, 2, 2, 2, 2, 8, 8],
        [2, 8, 2, 8, 2, 8, 2, 8],
    ],
    dtype=float,
)


def synth_sample(rng: np.random.Generator, label: int, n: int = 1000, with_label: bool = True) -> dict:
    """One JSON sample dict with a gesture segment (for gestures) and int8-range EMG."""
    emg = rng.normal(0, 1.5, size=(n, 8))
    sample: dict = {"startPointforGestureExecution": 100, "myoDetection": [0] * (n // 4)}
    if label > 0:
        start = int(rng.integers(200, 500))
        length = int(rng.integers(200, 350))
        emg[start : start + length] *= GAINS[label]
        gt = np.zeros(n, dtype=int)
        gt[start : start + length] = 1
        sample["groundTruth"] = gt.tolist()
        sample["groundTruthIndex"] = [start + 1, start + length]
    emg = np.clip(np.round(emg), -128, 127).astype(int)
    sample["emg"] = {f"ch{c + 1}": emg[:, c].tolist() for c in range(8)}
    if with_label:
        sample["gestureName"] = CLASSES[label]
    return sample


def write_user(path: Path, name: str, rng: np.random.Generator, reps: int = 2, test_unlabeled: bool = False) -> Path:
    """Write ``path/name/name.json`` with ``reps`` recordings per class in each section."""
    def section(labelled: bool) -> dict:
        samples, i = {}, 1
        for label in range(6):
            for _ in range(reps):
                samples[f"idx_{i}"] = synth_sample(rng, label, with_label=labelled)
                i += 1
        return samples

    data = {
        "generalInfo": {
            "deviceModel": "Myo Armband",
            "samplingFrequencyInHertz": 200,
            "recordingTimeInSeconds": 5,
            "repetitionsForSynchronizationGesture": 5,
            "myoPredictionLabel": {c: i for i, c in enumerate(CLASSES)},
        },
        "userInfo": {"name": name, "age": int(rng.integers(18, 60)), "gender": "man", "handedness": "right",
                     "date": f"{name}-{rng.integers(1_000_000)}"},
        "synchronizationGesture": {"samples": {}},
        "trainingSamples": section(True),
        "testingSamples": section(not test_unlabeled),
    }
    user_dir = path / name
    user_dir.mkdir(parents=True, exist_ok=True)
    out = user_dir / f"{name}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A synthetic dataset: 6 development users and 3 test users (test 'testingSamples' unlabeled)."""
    root = tmp_path_factory.mktemp("epn612_synth")
    rng = np.random.default_rng(0)
    for i in range(1, 7):
        write_user(root / "trainingJSON", f"user{i}", rng)
    for i in range(1, 4):
        write_user(root / "testingJSON", f"user{i}", rng, test_unlabeled=True)
    return root


def make_config(root: Path, work: Path, extra: list[str] | None = None) -> dict:
    overrides = [
        f"dataset.root={root}",
        f"paths.processed_dir={work / 'processed'}",
        f"paths.artifacts_dir={work / 'artifacts'}",
        "split.val_fraction=0.34",
        "prepare.workers=2",
        "train.epochs=2",
        "train.patience=2",
        "train.batch_size=128",
        "train.device=cpu",
        "train.max_windows_per_epoch=2000",
        "model.channels=16",
        "model.n_blocks=2",
        "model.head_dim=16",
        "hpo.epochs=2",
        "hpo.patience=2",
        "hpo.max_windows_per_epoch=1000",
        "hpo.startup_trials=1",
        "hpo.pruner_startup_trials=10",
        "hpo.search_space={model.channels: {type: categorical, choices: [8, 16]}, "
        "model.temporal_head: {type: categorical, choices: [attention, gru]}, "
        "windowing.window_ms: {type: categorical, choices: [200, 250]}, "
        "train.lr: {type: float, low: 0.001, high: 0.003, log: true}}",
        "selection.max_latency_ms=1000",
        *(extra or []),
    ]
    return load_config(None, overrides)


@pytest.fixture(scope="session")
def prepared_config(synthetic_root, tmp_path_factory) -> dict:
    """Config whose processed data/split were created from the synthetic dataset."""
    from myo_emg.data.prepare import prepare_dataset

    work = tmp_path_factory.mktemp("work")
    config = make_config(synthetic_root, work)
    prepare_dataset(config, workers=2)
    return config


TINY_ARCH = {"type": "emgnet", "n_channels": 8, "n_classes": 6, "channels": 8, "n_blocks": 1, "kernel_size": 3,
             "dilation_growth": 1, "se": True, "temporal_head": "attention", "gru_hidden": 8,
             "td_features": True, "dropout": 0.0, "head_dim": 8}


def tiny_pipeline(filter_cfg: dict | None = None, name: str = "tiny_emgnet", **arch_overrides):
    """Untrained EMGNet pipeline for inference tests (synthetic, no dataset needed)."""
    import torch

    from myo_emg.data.preprocess import Standardizer
    from myo_emg.inference.predictor import GesturePipeline
    from myo_emg.models import build_neural_model

    torch.manual_seed(0)
    arch = {**TINY_ARCH, **arch_overrides}
    meta = {"name": name, "input_type": "raw", "architecture": arch, "classes": CLASSES, "sampling_rate_hz": 200,
            "channel_order": [f"ch{i}" for i in range(1, 9)], "window_ms": 200, "window_length": 40, "stride": 10,
            "filter": filter_cfg or {"type": "none"}}
    scaler = Standardizer(np.zeros(8, np.float32), np.full(8, 5.0, np.float32))
    return GesturePipeline(meta, build_neural_model(arch).eval(), scaler)
