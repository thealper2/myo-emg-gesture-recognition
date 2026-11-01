"""Device-independent EMG source interface and the thread-safe sample queue.

Acquisition (BLE callbacks or the mock generator) only pushes ``SampleBatch`` objects into a
``SampleQueue``; inference and visualisation consume them in other threads, so a slow consumer can
never block the sensor stream. When the queue is full the *oldest* batches are dropped and counted.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

N_CHANNELS = 8
NOMINAL_RATE_HZ = 200.0


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class SampleBatch:
    """Consecutive EMG samples ``(n, 8)`` received together (one BLE notification = 2 samples)."""

    samples: np.ndarray
    received_at: float  # wall-clock time.time() when the batch reached the host (display only)
    seq: int
    received_mono: float = field(default_factory=time.monotonic)  # for latency/rate measurements


class SampleQueue:
    """Bounded FIFO between acquisition and inference; drops the oldest batches on overflow."""

    def __init__(self, max_batches: int = 2000) -> None:
        if max_batches < 1:
            raise ValueError("max_batches must be >= 1")
        self._items: deque[SampleBatch] = deque()
        self._max = max_batches
        self._cond = threading.Condition()
        self.dropped_batches = 0
        self.dropped_samples = 0

    def put(self, batch: SampleBatch) -> None:
        with self._cond:
            if len(self._items) >= self._max:
                old = self._items.popleft()
                self.dropped_batches += 1
                self.dropped_samples += int(old.samples.shape[0])
            self._items.append(batch)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> SampleBatch | None:
        """Return the next batch, or ``None`` if nothing arrived within ``timeout`` seconds."""
        with self._cond:
            if not self._items and not self._cond.wait_for(lambda: bool(self._items), timeout):
                return None
            return self._items.popleft()

    def clear(self) -> None:
        with self._cond:
            self._items.clear()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


class RateMeter:
    """Measures the received sample rate over a sliding time window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def add(self, n: int, t: float) -> None:
        with self._lock:
            self._events.append((t, n))
            while self._events and t - self._events[0][0] > self.window_s:
                self._events.popleft()

    def rate(self, now: float | None = None) -> float | None:
        now = time.monotonic() if now is None else now
        with self._lock:
            events = [(t, n) for t, n in self._events if now - t <= self.window_s]
        if len(events) < 2:
            return None
        span = max(now - events[0][0], 1e-6)
        return sum(n for _, n in events[1:]) / span


@dataclass
class SourceStatus:
    """Snapshot of a source's state for the UI and diagnostics."""

    state: ConnectionState
    simulated: bool
    description: str
    error: str | None = None
    samples_received: int = 0
    invalid_packets: int = 0
    measured_rate_hz: float | None = None
    last_sample_age_s: float | None = None
    started_at: float | None = None
    streaming_seconds: float | None = None  # monotonic time since streaming started
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def average_rate_hz(self) -> float | None:
        """Mean sample rate since streaming started (more stable than the bursty 2 s rate)."""
        if not self.streaming_seconds or self.streaming_seconds < 1.0:
            return None
        return self.samples_received / self.streaming_seconds

    @property
    def expected_samples(self) -> int | None:
        """Samples a 200 Hz stream would have delivered since streaming started."""
        if self.streaming_seconds is None:
            return None
        return int(self.streaming_seconds * NOMINAL_RATE_HZ)


class EMGSource(ABC):
    """Base class for EMG producers (physical Myo or mock). Subclasses call ``_emit``."""

    simulated: bool = False
    description: str = "EMG source"

    def __init__(self, queue: SampleQueue) -> None:
        self.queue = queue
        self._lock = threading.Lock()
        self._state = ConnectionState.DISCONNECTED
        self._error: str | None = None
        self._seq = 0
        self._samples = 0
        self._invalid = 0
        self._last_t: float | None = None
        self._started_at: float | None = None
        self._started_mono: float | None = None
        self._rate = RateMeter()
        self.info: dict[str, Any] = {}

    @abstractmethod
    def start(self) -> None:
        """Begin acquisition in a background thread (returns immediately)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition and release the device; safe to call repeatedly."""

    def _set_state(self, state: ConnectionState, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            if error is not None or state != ConnectionState.ERROR:
                self._error = error
            if state == ConnectionState.STREAMING and self._started_at is None:
                self._started_at = time.time()
                self._started_mono = time.monotonic()

    def _emit(self, samples: Any, received_at: float | None = None) -> bool:
        """Validate and enqueue samples ``(n, 8)``; invalid packets are counted and dropped."""
        t = time.time() if received_at is None else received_at
        t_mono = time.monotonic()
        try:
            arr = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError):
            arr = None
        if arr is None or arr.ndim != 2 or arr.shape[1] != N_CHANNELS or arr.shape[0] == 0:
            with self._lock:
                self._invalid += 1
            return False
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._samples += arr.shape[0]
            self._last_t = t_mono
        self._rate.add(arr.shape[0], t_mono)
        self.queue.put(SampleBatch(arr, t, seq, t_mono))
        return True

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    def status(self) -> SourceStatus:
        now = time.monotonic()
        with self._lock:
            return SourceStatus(
                state=self._state,
                simulated=self.simulated,
                description=self.description,
                error=self._error,
                samples_received=self._samples,
                invalid_packets=self._invalid,
                measured_rate_hz=self._rate.rate(now),
                last_sample_age_s=None if self._last_t is None else now - self._last_t,
                started_at=self._started_at,
                streaming_seconds=None if self._started_mono is None else time.monotonic() - self._started_mono,
                info=dict(self.info),
            )
