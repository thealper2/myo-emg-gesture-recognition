"""Validation helpers for EMG arrays and recording metadata."""

from __future__ import annotations

import numpy as np

N_CHANNELS = 8


class EMGValidationError(ValueError):
    """Raised when an EMG array or recording does not satisfy the expected structure."""


def validate_emg_array(
    emg: np.ndarray,
    n_channels: int = N_CHANNELS,
    value_range: tuple[float, float] | None = None,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    """Check that ``emg`` has shape ``(n_samples, n_channels)`` and finite, in-range values.

    Returns the array unchanged so the call can be chained.
    """
    if not isinstance(emg, np.ndarray):
        raise EMGValidationError(f"EMG must be a numpy array, got {type(emg).__name__}")
    if emg.ndim != 2:
        raise EMGValidationError(f"EMG must be 2-D (samples, channels), got shape {emg.shape}")
    if emg.shape[1] != n_channels:
        raise EMGValidationError(f"Expected {n_channels} channels, got {emg.shape[1]}")
    if emg.shape[0] == 0:
        raise EMGValidationError("EMG array is empty")
    if not allow_nonfinite and np.issubdtype(emg.dtype, np.floating) and not np.isfinite(emg).all():
        raise EMGValidationError(f"EMG contains {int((~np.isfinite(emg)).sum())} non-finite values")
    if value_range is not None:
        lo, hi = value_range
        finite = emg[np.isfinite(emg)] if np.issubdtype(emg.dtype, np.floating) else emg
        if finite.size and (finite.min() < lo or finite.max() > hi):
            raise EMGValidationError(
                f"EMG values outside [{lo}, {hi}]: min={finite.min()}, max={finite.max()}"
            )
    return emg


def ground_truth_bounds(ground_truth: list[int] | np.ndarray, n_samples: int) -> tuple[int, int]:
    """Return ``(start, end)`` (0-based, end exclusive) of the single contiguous gesture segment.

    Raises ``EMGValidationError`` when the mask is empty, has the wrong length or is not contiguous.
    """
    mask = np.asarray(ground_truth).astype(bool)
    if mask.shape != (n_samples,):
        raise EMGValidationError(f"groundTruth length {mask.size} != EMG length {n_samples}")
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise EMGValidationError("groundTruth mask contains no gesture samples")
    start, end = int(idx[0]), int(idx[-1]) + 1
    if idx.size != end - start:
        raise EMGValidationError("groundTruth mask is not a single contiguous segment")
    return start, end
