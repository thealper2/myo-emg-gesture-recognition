"""Loading prepared partitions as labelled window sets for training and evaluation."""

from __future__ import annotations

import logging
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np

from myo_emg.data.parse_dataset import PreparedPartition
from myo_emg.data.prepare import DEV_FILE, TEST_FILE, split_path
from myo_emg.data.preprocess import design_filter, filter_recording
from myo_emg.data.splits import load_split
from myo_emg.data.windowing import WindowIndex, WindowingSpec, build_window_index, gather_windows
from myo_emg.utils.config import processed_dir

logger = logging.getLogger(__name__)


@dataclass
class WindowSet:
    """Labelled windows of one participant group, stored as offsets into a shared signal array."""

    name: str
    signal: np.ndarray  # (T, 8) int8 (no filter) or float32 (filtered)
    index: WindowIndex
    window_length: int
    participants: list[str]
    n_recordings: int

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> np.ndarray:
        return self.index.labels

    def windows(self, start: int = 0, stop: int | None = None, idx: np.ndarray | None = None) -> np.ndarray:
        """Materialise windows ``(n, L, 8)`` as float32 for a slice or an explicit index array."""
        starts = self.index.starts[idx] if idx is not None else self.index.starts[start:stop]
        return gather_windows(self.signal, starts, self.window_length).astype(np.float32)

    def class_counts(self, n_classes: int) -> list[int]:
        return np.bincount(self.labels, minlength=n_classes).tolist()


def _filtered_signal(prep: PreparedPartition, filter_cfg: dict[str, Any], fs: float) -> np.ndarray:
    """Apply the causal filter to every recording independently (zero initial state)."""
    sos = design_filter(filter_cfg, fs)
    if sos is None:
        return prep.signal
    out = np.empty(prep.signal.shape, dtype=np.float32)
    for i in range(prep.n_recordings):
        s, n = int(prep.rec_offset[i]), int(prep.rec_length[i])
        out[s : s + n] = filter_recording(prep.signal[s : s + n].astype(np.float32), sos)
    return out


def make_window_set(
    name: str, prep: PreparedPartition, spec: WindowingSpec, filter_cfg: dict[str, Any], fs: float
) -> WindowSet:
    index = build_window_index(prep, spec)
    ws = WindowSet(
        name=name,
        signal=_filtered_signal(prep, filter_cfg, fs),
        index=index,
        window_length=spec.window_length,
        participants=prep.participants.tolist(),
        n_recordings=prep.n_recordings,
    )
    logger.info("%s: %d participants, %d recordings, %d windows", name, len(ws.participants), prep.n_recordings, len(ws))
    return ws


def load_split_for(config: dict[str, Any]) -> dict[str, Any]:
    return load_split(split_path(config))


class DevData:
    """Training and validation participants loaded once; windows are built per windowing spec.

    The (optionally filtered) signals are shared by every windowing variant, which lets the
    hyper-parameter search try different window lengths without re-loading the data.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.split = load_split_for(config)
        dev = PreparedPartition.load(processed_dir(config) / DEV_FILE)
        missing = (set(self.split["train"]) | set(self.split["val"])) - set(dev.participants.tolist())
        if missing:
            raise ValueError(f"Split references {len(missing)} participants absent from the prepared data; "
                             "re-run `make prepare-data`")
        self.fs = float(config["dataset"]["sampling_rate_hz"])
        self.filter_cfg = dict(config["preprocessing"]["filter"])
        self.parts = {name: dev.subset(self.split[name]) for name in ("train", "val")}
        del dev
        if set(self.parts["train"].participants.tolist()) & set(self.parts["val"].participants.tolist()):
            raise AssertionError("train/val participant overlap")
        self.signals = {name: _filtered_signal(p, self.filter_cfg, self.fs) for name, p in self.parts.items()}
        self._cache: dict[tuple, WindowSet] = {}

    def window_set(self, name: str, spec: WindowingSpec) -> WindowSet:
        key = (name, astuple(spec))
        if key not in self._cache:
            prep = self.parts[name]
            index = build_window_index(prep, spec)
            self._cache[key] = WindowSet(name, self.signals[name], index, spec.window_length,
                                         prep.participants.tolist(), prep.n_recordings)
            logger.info("%s windows (L=%d, stride=%d): %d", name, spec.window_length, spec.stride, len(index))
        return self._cache[key]


def load_dev_sets(config: dict[str, Any]) -> tuple[WindowSet, WindowSet, dict[str, Any]]:
    """Training and validation window sets (development users only) plus the split."""
    dev = DevData(config)
    spec = WindowingSpec.from_config(config)
    return dev.window_set("train", spec), dev.window_set("val", spec), dev.split


def load_test_set(config: dict[str, Any], spec: WindowingSpec, filter_cfg: dict[str, Any], fs: float) -> WindowSet:
    """Held-out test participants, windowed with the *model's* saved windowing/filter settings."""
    split = load_split_for(config)
    test = PreparedPartition.load(processed_dir(config) / TEST_FILE)
    return make_window_set("test", test.subset(split["test"]), spec, filter_cfg, fs)
