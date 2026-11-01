"""Real-time inference engine and session management (independent of any UI).

``RealtimeEngine`` consumes ``SampleBatch`` objects from a ``SampleQueue`` in its own thread, runs the
saved pipeline through ``StreamingPredictor`` and keeps a short raw-signal buffer for plotting.
``LiveSession`` wires an EMG source (physical Myo or mock) to an engine. The model is only loaded,
never trained or adapted.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from myo_emg.device.interface import N_CHANNELS, ConnectionState, EMGSource, SampleBatch, SampleQueue
from myo_emg.inference.predictor import GesturePipeline, Prediction, StreamingPredictor

logger = logging.getLogger(__name__)


def _summary(values: deque) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.fromiter(values, dtype=float)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)), "last_ms": float(arr[-1])}


class RealtimeEngine:
    """Inference thread: queue -> streaming predictor -> latest prediction + statistics."""

    def __init__(
        self,
        pipeline: GesturePipeline,
        queue: SampleQueue,
        threshold: float = 0.6,
        smoothing: dict[str, Any] | None = None,
        stride: int | None = None,
        plot_seconds: float = 20.0,
    ) -> None:
        self.pipeline = pipeline
        self.queue = queue
        self.predictor = StreamingPredictor(pipeline, stride=stride, threshold=threshold, smoother=smoothing)
        self.fs = pipeline.sampling_rate_hz
        self._plot = np.zeros((int(plot_seconds * self.fs), N_CHANNELS), dtype=np.float32)
        self._plot_n = 0
        self._lock = threading.Lock()
        self._latest: Prediction | None = None
        self._latency = {k: deque(maxlen=200) for k in ("preprocess", "model", "end_to_end")}
        self._pred_times: deque[float] = deque(maxlen=500)
        self.errors: deque[str] = deque(maxlen=20)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.batches_processed = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self.queue.get(timeout=0.1)
            if batch is not None:
                self.process(batch)

    def process(self, batch: SampleBatch) -> list[Prediction]:
        """Run inference on one batch (called by the thread; public for tests)."""
        try:
            preds = self.predictor.push(batch.samples, timestamp=batch.received_at)
        except Exception as exc:  # never kill the inference thread on a bad packet
            msg = f"{time.strftime('%H:%M:%S')} inference error: {exc}"
            logger.warning(msg)
            self.errors.append(msg)
            return []
        done = time.monotonic()
        n = batch.samples.shape[0]
        with self._lock:
            self.batches_processed += 1
            if n >= self._plot.shape[0]:
                self._plot[:] = batch.samples[-self._plot.shape[0] :]
            else:
                self._plot = np.roll(self._plot, -n, axis=0)
                self._plot[-n:] = batch.samples
            self._plot_n += n
            for p in preds:
                self._latency["preprocess"].append(p.preprocess_ms)
                self._latency["model"].append(p.model_ms)
                self._latency["end_to_end"].append((done - batch.received_mono) * 1000.0)
                self._pred_times.append(done)
            if preds:
                self._latest = preds[-1]
        return preds

    def set_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.predictor.threshold = threshold

    def set_smoothing(self, smoothing: dict[str, Any] | None) -> None:
        from myo_emg.inference.smoothing import build_smoother

        self.predictor.smoother = build_smoother(smoothing)

    def snapshot(self, seconds: float = 5.0) -> dict[str, Any]:
        """Thread-safe copy of everything the UI needs."""
        now = time.monotonic()
        with self._lock:
            n = min(int(seconds * self.fs), self._plot.shape[0], self._plot_n)
            recent = self._plot[self._plot.shape[0] - n :].copy()
            rate = sum(1 for t in self._pred_times if now - t <= 2.0) / 2.0
            return {
                "signal": recent,
                "prediction": self._latest,
                "windows_processed": self.predictor.windows_processed,
                "samples_processed": self.predictor.samples_seen,
                "invalid_samples": self.predictor.invalid_samples,
                "latency": {k: _summary(v) for k, v in self._latency.items()},
                "prediction_rate_hz": rate,
                "threshold": self.predictor.threshold,
                "errors": list(self.errors),
            }


SourceFactory = Callable[[str, SampleQueue, dict[str, Any]], EMGSource]


def default_source_factory(kind: str, queue: SampleQueue, options: dict[str, Any]) -> EMGSource:
    """Create a source; ``options`` come from ``configs/inference.yaml`` plus UI inputs."""
    if kind == "myo":
        from myo_emg.device.dl_myo_adapter import DLMyoSource

        dev = options.get("device", {})
        return DLMyoSource(
            queue,
            mac=options.get("mac") or dev.get("mac"),
            scan_timeout_s=float(dev.get("scan_timeout_s", 20)),
            emg_mode=str(dev.get("emg_mode", "emg")),
            channel_shift=int(dev.get("channel_shift", 0)),
            stall_timeout_s=float(dev.get("stall_timeout_s", 3.0)),
            lib_dir=dev.get("dl_myo_dir"),
        )
    if kind in ("mock_replay", "mock_synthetic"):
        from myo_emg.device.mock_stream import MockEMGStream

        mock = options.get("mock", {})
        return MockEMGStream(
            queue,
            mode="replay" if kind == "mock_replay" else "synthetic",
            data_file=Path(mock["replay_file"]) if mock.get("replay_file") else None,
            participants=mock.get("participants"),
            classes=options.get("classes"),
            seed=int(mock.get("seed", 0)),
        )
    raise ValueError(f"Unknown source kind {kind!r}")


class LiveSession:
    """Owns the current source + engine. ``mode='live'`` only allows the physical Myo;
    ``mode='demo'`` only allows simulated sources, so mock data is never shown as hardware data."""

    LIVE_SOURCES = ("myo",)
    DEMO_SOURCES = ("mock_replay", "mock_synthetic")

    def __init__(
        self,
        pipeline: GesturePipeline,
        mode: str,
        options: dict[str, Any],
        source_factory: SourceFactory = default_source_factory,
    ) -> None:
        if mode not in ("live", "demo"):
            raise ValueError("mode must be 'live' or 'demo'")
        self.pipeline = pipeline
        self.mode = mode
        self.options = {**options, "classes": pipeline.classes}
        self.source_factory = source_factory
        self.source: EMGSource | None = None
        self.engine: RealtimeEngine | None = None
        self.queue: SampleQueue | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def allowed_sources(self) -> tuple[str, ...]:
        return self.LIVE_SOURCES if self.mode == "live" else self.DEMO_SOURCES

    def connect(self, kind: str | None = None, mac: str | None = None) -> str:
        """Start a source and the inference engine; failures are reported, never substituted."""
        kind = kind or self.allowed_sources[0]
        if kind not in self.allowed_sources:
            raise ValueError(f"Source {kind!r} is not allowed in {self.mode} mode (allowed: {self.allowed_sources})")
        self.disconnect()
        inf = self.options.get("inference", {})
        with self._lock:
            self.queue = SampleQueue(int(self.options.get("device", {}).get("queue_max_batches", 2000)))
            try:
                self.source = self.source_factory(kind, self.queue, {**self.options, "mac": mac})
            except Exception as exc:
                self.last_error = f"Could not create source {kind}: {exc}"
                logger.error(self.last_error)
                self.source = None
                return self.last_error
            self.engine = RealtimeEngine(
                self.pipeline,
                self.queue,
                threshold=float(inf.get("confidence_threshold", 0.6)),
                smoothing=inf.get("smoothing"),
                stride=inf.get("stride_samples"),
                plot_seconds=float(self.options.get("dashboard", {}).get("max_plot_seconds", 20)),
            )
            self.engine.start()
            self.source.start()
            self.last_error = None
        return f"Started {self.source.description}"

    def disconnect(self) -> str:
        with self._lock:
            if self.source is not None:
                self.source.stop()
            if self.engine is not None:
                self.engine.stop()
            stopped = self.source is not None
            self.source = None
            self.engine = None
        return "Disconnected" if stopped else "Not connected"

    def snapshot(self, seconds: float = 5.0) -> dict[str, Any]:
        with self._lock:
            source, engine, queue = self.source, self.engine, self.queue
        status = source.status() if source is not None else None
        snap = engine.snapshot(seconds) if engine is not None else None
        out = {
            "mode": self.mode,
            "status": status,
            "engine": snap,
            "queue_len": len(queue) if queue is not None else 0,
            "queue_dropped_samples": queue.dropped_samples if queue is not None else 0,
            "last_error": self.last_error,
            "truth": getattr(source, "current_truth", None),
        }
        if status is not None and status.state == ConnectionState.STREAMING and status.expected_samples:
            out["estimated_missing_samples"] = max(0, status.expected_samples - status.samples_received)
        return out
