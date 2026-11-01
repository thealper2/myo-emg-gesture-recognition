"""Simulated EMG stream for testing the application without a Myo. Never real hardware data.

Two modes:

* ``replay``: plays back recordings of held-out EMG-EPN-612 *test* participants at 200 Hz in
  2-sample packets (like Myo BLE notifications). The ground-truth label of the recording being
  replayed is exposed so the demo can be checked visually.
* ``synthetic``: Gaussian noise with random bursts of higher amplitude. No gesture meaning.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from myo_emg.data.parse_dataset import PreparedPartition
from myo_emg.device.interface import N_CHANNELS, NOMINAL_RATE_HZ, ConnectionState, EMGSource, SampleQueue


class MockEMGStream(EMGSource):
    """Background thread emitting simulated packets in real time."""

    simulated = True

    def __init__(
        self,
        queue: SampleQueue,
        mode: str = "synthetic",
        data_file: Path | None = None,
        participants: list[str] | None = None,
        classes: list[str] | None = None,
        rate_hz: float = NOMINAL_RATE_HZ,
        packet_size: int = 2,
        seed: int = 0,
        realtime: bool = True,
    ) -> None:
        super().__init__(queue)
        if mode not in ("replay", "synthetic"):
            raise ValueError(f"Unknown mock mode {mode!r}")
        self.mode = mode
        self.rate_hz = rate_hz
        self.packet_size = packet_size
        self.realtime = realtime
        self.classes = classes or ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]
        self._rng = np.random.default_rng(seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._truth = "-"
        self._signal: np.ndarray | None = None
        self.description = f"SIMULATED mock stream ({mode})"
        if mode == "replay":
            self._load_replay(data_file, participants)

    def _load_replay(self, data_file: Path | None, participants: list[str] | None) -> None:
        if data_file is None or not Path(data_file).is_file():
            raise FileNotFoundError(f"Replay data not found: {data_file}. Run `make prepare-data` or use mode=synthetic.")
        prep = PreparedPartition.load(Path(data_file))
        if participants:
            prep = prep.subset(participants)
        if prep.n_recordings == 0:
            raise ValueError("No recordings available for replay")
        order = self._rng.permutation(prep.n_recordings)
        parts, labels = [], []
        for i in order[: min(600, order.size)]:
            rec = prep.recording(int(i)).astype(np.float32)
            truth = np.full(rec.shape[0], self.classes[0], dtype=object)
            if prep.gt_start[i] >= 0:
                truth[prep.gt_start[i] : prep.gt_end[i]] = self.classes[int(prep.rec_label[i])]
            parts.append(rec)
            labels.append(truth)
        self._signal = np.concatenate(parts)
        self._labels = np.concatenate(labels)
        self.info = {"replay_recordings": len(parts), "replay_participants": prep.participants.size,
                     "replay_source": str(data_file)}

    @property
    def current_truth(self) -> str:
        """Ground-truth label at the current replay position ('-' for synthetic data)."""
        return self._truth

    def _next_packet(self, pos: int) -> np.ndarray:
        if self._signal is not None:
            idx = (pos + np.arange(self.packet_size)) % self._signal.shape[0]
            self._truth = str(self._labels[idx[-1]])
            return self._signal[idx]
        burst = 1.0 + 6.0 * (np.sin(pos / self.rate_hz * 2 * np.pi * 0.25) > 0.6)
        return np.clip(np.round(self._rng.normal(0, 2.0 * burst, (self.packet_size, N_CHANNELS))), -128, 127)

    def _run(self) -> None:
        self._set_state(ConnectionState.STREAMING)
        period = self.packet_size / self.rate_hz
        t0 = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            # emit every packet that is due (catches up after scheduling hiccups), then sleep ~10 ms
            due = int((time.perf_counter() - t0) / period) + 1 if self.realtime else k + 1
            while k < due and not self._stop.is_set():
                self._emit(self._next_packet(k * self.packet_size))
                k += 1
            if self.realtime:
                self._stop.wait(max(0.0, t0 + k * period - time.perf_counter()))
        self._set_state(ConnectionState.DISCONNECTED)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-emg", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._set_state(ConnectionState.DISCONNECTED)

    def simulate_disconnect(self, reason: str = "simulated disconnect") -> None:
        """Test hook: stop emitting and report an error, as a lost BLE link would."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._set_state(ConnectionState.ERROR, reason)
