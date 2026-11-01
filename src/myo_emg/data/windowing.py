"""Window segmentation of recordings into labelled ``(window_length, 8)`` EMG windows.

Labelling rules (EMG-EPN-612 provides a per-sample ``groundTruth`` mask for gesture recordings):

* ``noGesture`` recordings: every window is ``noGesture`` (stride ``rest_stride_samples``).
* gesture recordings: a window gets the gesture label when at least ``gesture_min_overlap`` of its
  samples lie inside the ground-truth segment. Windows that only partially overlap (transitions)
  are dropped. Windows with no overlap are dropped too unless ``rest_from_gesture_recordings`` is set,
  in which case they become ``noGesture`` windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from myo_emg.data.parse_dataset import PreparedPartition


def window_length_samples(window_ms: float, fs: float) -> int:
    """Convert a window duration in milliseconds into an integer number of samples."""
    n = window_ms * fs / 1000.0
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"{window_ms} ms at {fs} Hz is not an integer number of samples ({n})")
    n = int(round(n))
    if n < 4:
        raise ValueError(f"Window of {n} samples is too short")
    return n


@dataclass
class WindowingSpec:
    """Window geometry and labelling parameters."""

    window_length: int
    stride: int
    rest_stride: int
    gesture_min_overlap: float = 0.75
    rest_from_gesture_recordings: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WindowingSpec:
        w = config["windowing"]
        fs = float(config["dataset"]["sampling_rate_hz"])
        return cls(
            window_length=window_length_samples(float(w["window_ms"]), fs),
            stride=int(w["stride_samples"]),
            rest_stride=int(w.get("rest_stride_samples", w["stride_samples"])),
            gesture_min_overlap=float(w.get("gesture_min_overlap", 0.75)),
            rest_from_gesture_recordings=bool(w.get("rest_from_gesture_recordings", False)),
        )


@dataclass
class WindowIndex:
    """Window start offsets into ``PreparedPartition.signal`` with labels and recording ids."""

    starts: np.ndarray  # (N,) int64, global offsets
    labels: np.ndarray  # (N,) int64
    rec_index: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def take(self, idx: np.ndarray) -> WindowIndex:
        return WindowIndex(self.starts[idx], self.labels[idx], self.rec_index[idx])


def recording_windows(
    length: int,
    label: int,
    gt_start: int,
    gt_end: int,
    spec: WindowingSpec,
    rest_label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local window starts and labels for one recording."""
    L = spec.window_length
    if length < L:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if label == rest_label or gt_start < 0:
        starts = np.arange(0, length - L + 1, spec.rest_stride, dtype=np.int64)
        return starts, np.full(starts.shape, rest_label, np.int64)

    starts = np.arange(0, length - L + 1, spec.stride, dtype=np.int64)
    overlap = np.clip(np.minimum(starts + L, gt_end) - np.maximum(starts, gt_start), 0, None) / L
    gesture = overlap >= spec.gesture_min_overlap
    out_starts = [starts[gesture]]
    out_labels = [np.full(int(gesture.sum()), label, np.int64)]
    if spec.rest_from_gesture_recordings:
        rest = starts[(overlap == 0) & (starts % spec.rest_stride == 0)]
        out_starts.append(rest)
        out_labels.append(np.full(rest.shape, rest_label, np.int64))
    s, y = np.concatenate(out_starts), np.concatenate(out_labels)
    order = np.argsort(s, kind="stable")
    return s[order], y[order]


def build_window_index(prep: PreparedPartition, spec: WindowingSpec) -> WindowIndex:
    """Enumerate labelled windows of every recording in ``prep``."""
    starts, labels, recs = [], [], []
    for i in range(prep.n_recordings):
        s, y = recording_windows(
            int(prep.rec_length[i]), int(prep.rec_label[i]), int(prep.gt_start[i]), int(prep.gt_end[i]), spec
        )
        if s.size:
            starts.append(s + int(prep.rec_offset[i]))
            labels.append(y)
            recs.append(np.full(s.shape, i, np.int64))
    if not starts:
        return WindowIndex(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    return WindowIndex(np.concatenate(starts), np.concatenate(labels), np.concatenate(recs))


def gather_windows(signal: np.ndarray, starts: np.ndarray, window_length: int) -> np.ndarray:
    """Return windows ``(N, window_length, n_channels)`` starting at the given global offsets."""
    idx = starts[:, None] + np.arange(window_length)[None, :]
    return signal[idx]


def sliding_windows(x: np.ndarray, window_length: int, stride: int) -> np.ndarray:
    """All windows of ``x`` ``(T, C)`` with the given stride, shape ``(N, window_length, C)``."""
    if x.shape[0] < window_length:
        return np.empty((0, window_length, x.shape[1]), dtype=x.dtype)
    starts = np.arange(0, x.shape[0] - window_length + 1, stride)
    return gather_windows(x, starts, window_length)
