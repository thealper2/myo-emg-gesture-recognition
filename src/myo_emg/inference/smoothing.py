"""Temporal smoothing of consecutive class-score vectors."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class Smoother:
    """Identity smoother; subclasses keep a short causal history."""

    def update(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)

    def reset(self) -> None:
        pass


class EMASmoother(Smoother):
    """Exponential moving average: ``s_t = alpha * p_t + (1 - alpha) * s_{t-1}``."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def update(self, scores: np.ndarray) -> np.ndarray:
        p = np.asarray(scores, dtype=np.float64)
        self._state = p.copy() if self._state is None else self.alpha * p + (1 - self.alpha) * self._state
        return self._state

    def reset(self) -> None:
        self._state = None


class MovingAverageSmoother(Smoother):
    """Mean of the last ``window`` score vectors."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._hist: deque[np.ndarray] = deque(maxlen=window)

    def update(self, scores: np.ndarray) -> np.ndarray:
        self._hist.append(np.asarray(scores, dtype=np.float64))
        return np.mean(self._hist, axis=0)

    def reset(self) -> None:
        self._hist.clear()


class MajorityVoteSmoother(Smoother):
    """Fraction of votes each class received among the last ``window`` argmax decisions."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._votes: deque[int] = deque(maxlen=window)

    def update(self, scores: np.ndarray) -> np.ndarray:
        p = np.asarray(scores)
        self._votes.append(int(p.argmax()))
        return np.bincount(np.asarray(self._votes), minlength=p.size) / len(self._votes)

    def reset(self) -> None:
        self._votes.clear()


def build_smoother(config: dict[str, Any] | None) -> Smoother:
    """``{"method": none|ema|moving_average|majority, "alpha": .., "window": ..}``."""
    config = config or {}
    method = str(config.get("method", "none")).lower()
    if method in ("none", "", "null"):
        return Smoother()
    if method == "ema":
        return EMASmoother(float(config.get("alpha", 0.5)))
    if method == "moving_average":
        return MovingAverageSmoother(int(config.get("window", 5)))
    if method == "majority":
        return MajorityVoteSmoother(int(config.get("window", 5)))
    raise ValueError(f"Unknown smoothing method {method!r}")
