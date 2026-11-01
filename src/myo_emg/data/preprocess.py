"""Signal preprocessing shared by offline training and real-time inference.

Order of operations (identical offline and online)::

    raw int8 samples -> sanitize -> optional causal filter (whole stream) -> window
        -> per-channel z-score -> EMGNet

The optional filter is causal (``scipy.signal.sosfilt`` with carried state), so it never uses
future samples. Offline it runs once over each full recording from a zero initial state; online it
runs over the continuous stream. No filter is applied by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sps


@dataclass
class Standardizer:
    """Per-dimension z-score normalisation whose parameters are fitted on training data only."""

    mean: np.ndarray
    std: np.ndarray

    MIN_STD = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray, chunk: int = 1_000_000) -> Standardizer:
        """Fit on a 2-D array ``(n_observations, n_dims)``; accumulates in float64 chunks."""
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError(f"Standardizer.fit needs a 2-D array with >=2 rows, got {x.shape}")
        total = np.zeros(x.shape[1])
        for i in range(0, x.shape[0], chunk):
            total += x[i : i + chunk].astype(np.float64).sum(axis=0)
        mean = total / x.shape[0]
        sq = np.zeros(x.shape[1])
        for i in range(0, x.shape[0], chunk):
            sq += np.square(x[i : i + chunk].astype(np.float64) - mean).sum(axis=0)
        std = np.sqrt(sq / x.shape[0])
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError("Non-finite values while fitting Standardizer")
        std = np.where(std < cls.MIN_STD, 1.0, std)  # constant dimensions are left unscaled
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalise along the last axis; works for ``(N, D)`` and ``(N, L, D)``."""
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"Expected last dimension {self.mean.shape[0]}, got {x.shape[-1]}")
        return ((x.astype(np.float32, copy=False) - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Standardizer:
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))


def design_filter(config: dict[str, Any], fs: float) -> np.ndarray | None:
    """Return second-order sections for the configured Butterworth filter, or ``None``.

    ``config`` keys: ``type`` (none|highpass|lowpass|bandpass), ``cutoff_hz`` (float or [lo, hi]),
    ``order`` (int).
    """
    ftype = str(config.get("type", "none")).lower()
    if ftype in ("none", "", "null"):
        return None
    if ftype not in ("highpass", "lowpass", "bandpass"):
        raise ValueError(f"Unsupported filter type {ftype!r}")
    cutoff = config["cutoff_hz"]
    nyq = fs / 2.0
    cut = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if np.any(cut <= 0) or np.any(cut >= nyq):
        raise ValueError(f"Filter cutoff {cutoff} must lie in (0, {nyq}) Hz for fs={fs}")
    return sps.butter(int(config.get("order", 4)), cut if ftype == "bandpass" else float(cut[0]),
                      btype=ftype, fs=fs, output="sos")


class CausalFilter:
    """Stateful causal IIR filter applied per channel; safe for streaming."""

    def __init__(self, sos: np.ndarray | None, n_channels: int = 8) -> None:
        self.sos = sos
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        self._zi = None if self.sos is None else np.zeros((self.sos.shape[0], 2, self.n_channels))

    @property
    def enabled(self) -> bool:
        return self.sos is not None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter ``(n_samples, n_channels)`` continuing from the previous call's state."""
        if self.sos is None:
            return x
        y, self._zi = sps.sosfilt(self.sos, x.astype(np.float64), axis=0, zi=self._zi)
        return y.astype(np.float32)


def filter_recording(x: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    """Apply the causal filter to a full recording from a zero initial state."""
    if sos is None:
        return x
    return CausalFilter(sos, x.shape[1])(x)


def sanitize_samples(x: np.ndarray, last_valid: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Replace non-finite samples by the last valid sample (or 0) and return the count replaced.

    Used on the live stream only; offline recordings with invalid values are excluded instead.
    """
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x)
    n_bad = int(bad.any(axis=1).sum())
    if n_bad == 0:
        return x, 0
    x = x.copy()
    prev = np.zeros(x.shape[1], dtype=np.float32) if last_valid is None else last_valid.astype(np.float32)
    for i in range(x.shape[0]):
        row_bad = bad[i]
        if row_bad.any():
            x[i, row_bad] = prev[row_bad]
        prev = x[i]
    return x, n_bad
