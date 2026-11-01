"""Thin adapter from the local ``dl-myo`` library to the project's ``EMGSource`` interface.

dl-myo API used (verified in ``/mnt/d/work2/dl-myo/myo/core.py`` and ``myo/types.py``):

* ``MyoClient`` is an abstract asyncio client; subclasses implement the ``on_*`` callbacks.
* ``await Client.with_device(mac=..., aggregate_emg=False)`` scans with Bleak (by MAC, or by the Myo
  service UUID ``d5060001-...``) and connects. It retries forever, so it is wrapped in a timeout here.
* ``await client.setup(classifier_mode, emg_mode, imu_mode)`` configures the armband (LED, vibration,
  never-sleep) and ``await client.start()`` subscribes to the four EMG characteristics.
* Each EMG notification calls ``on_emg_data(EMGData)`` with ``sample1``/``sample2``: two consecutive
  samples of 8 signed int8 values (``struct "<8b"``), i.e. the same units as EMG-EPN-612.
  No device timestamps are provided; the host receive time is used.
* ``EMGMode.SEND_EMG`` (2, "filtered, unrectified") matches the Myo SDK EMG stream used to record
  EMG-EPN-612 at 200 Hz; ``SEND_RAW`` (3) is available for experiments.
* ``await client.stop()`` / ``await client.disconnect()`` end the session. dl-myo exposes no
  disconnect callback, so the link is supervised by polling the underlying Bleak client.

The library itself is not modified; it is imported from ``DL_MYO_DIR`` (default
``/mnt/d/work2/dl-myo``) when present, otherwise from the installed ``dl-myo`` package.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from myo_emg.device.interface import ConnectionState, EMGSource, SampleQueue

logger = logging.getLogger(__name__)

DEFAULT_DL_MYO_DIR = "/mnt/d/work2/dl-myo"


class DeviceDependencyError(RuntimeError):
    """dl-myo or its Bluetooth backend cannot be imported."""


def import_dl_myo(lib_dir: str | None = None) -> ModuleType:
    """Import ``myo`` from the local dl-myo checkout (preferred) or the installed package."""
    path = Path(lib_dir or os.environ.get("DL_MYO_DIR", DEFAULT_DL_MYO_DIR))
    if (path / "myo" / "__init__.py").is_file() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        myo = importlib.import_module("myo")
        importlib.import_module("myo.core")
    except ImportError as exc:
        raise DeviceDependencyError(
            f"Cannot import dl-myo ({exc}). Expected the library at {path} or `pip install` of its "
            "requirements (bleak) in the project virtualenv."
        ) from exc
    if not hasattr(myo, "MyoClient"):
        raise DeviceDependencyError(f"Imported module {myo.__file__} is not dl-myo (no MyoClient)")
    return myo


def packet_to_samples(sample1: Any, sample2: Any, channel_shift: int = 0) -> np.ndarray:
    """Convert one dl-myo EMG packet into ``(2, 8)`` float32 samples in dataset channel order.

    ``channel_shift`` cyclically rotates the channels (``ch_k <- sensor_(k+shift) mod 8``) to
    compensate for an armband worn rotated relative to the EMG-EPN-612 placement. 0 = identity.
    """
    arr = np.asarray([sample1, sample2], dtype=np.float32)
    if arr.shape != (2, 8):
        raise ValueError(f"EMG packet must contain 2 x 8 values, got {arr.shape}")
    return np.roll(arr, -channel_shift, axis=1) if channel_shift else arr


class DLMyoSource(EMGSource):
    """Streams EMG from a physical Myo Armband via dl-myo in a dedicated asyncio thread."""

    simulated = False

    def __init__(
        self,
        queue: SampleQueue,
        mac: str | None = None,
        scan_timeout_s: float = 20.0,
        emg_mode: str = "emg",
        channel_shift: int = 0,
        stall_timeout_s: float = 3.0,
        lib_dir: str | None = None,
    ) -> None:
        super().__init__(queue)
        if emg_mode not in ("emg", "raw"):
            raise ValueError("emg_mode must be 'emg' (SEND_EMG) or 'raw' (SEND_RAW)")
        self.mac = mac or None
        self.scan_timeout_s = scan_timeout_s
        self.emg_mode = emg_mode
        self.channel_shift = int(channel_shift) % 8
        self.stall_timeout_s = stall_timeout_s
        self.lib_dir = lib_dir
        self.description = "Myo Armband via dl-myo (BLE)"
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    # -- thread / lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._set_state(ConnectionState.CONNECTING)
        self._thread = threading.Thread(target=self._thread_main, name="dl-myo", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        loop, ev = self._loop, self._stop_event
        if loop is not None and ev is not None and loop.is_running():
            loop.call_soon_threadsafe(ev.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("dl-myo thread did not stop within %.0f s", timeout)
        if self.state != ConnectionState.ERROR:
            self._set_state(ConnectionState.DISCONNECTED)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # keep the app alive; report the failure in the status
            logger.exception("dl-myo session failed")
            self._set_state(ConnectionState.ERROR, f"{type(exc).__name__}: {exc}")

    # -- asyncio session ------------------------------------------------------------------------
    def _client_class(self, myo: ModuleType) -> type:
        source = self

        class _Client(myo.MyoClient):
            async def on_emg_data(self, emg):  # two samples per notification
                try:
                    source._emit(packet_to_samples(emg.sample1, emg.sample2, source.channel_shift))
                except ValueError:
                    source._count_invalid()

            async def on_emg_data_aggregated(self, eds):
                pass

            async def on_classifier_event(self, ce):
                pass

            async def on_aggregated_data(self, ad):
                pass

            async def on_fv_data(self, fvd):
                pass

            async def on_imu_data(self, imu):
                pass

            async def on_motion_event(self, me):
                pass

        return _Client

    def _count_invalid(self) -> None:
        with self._lock:
            self._invalid += 1

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        myo = import_dl_myo(self.lib_dir)
        from myo.types import ClassifierMode, EMGMode, IMUMode

        self.info = {"dl_myo_module": myo.__file__, "dl_myo_version": getattr(myo, "__version__", "?"),
                     "emg_mode": "SEND_EMG" if self.emg_mode == "emg" else "SEND_RAW",
                     "channel_shift": self.channel_shift, "target_mac": self.mac or "(scan by service UUID)"}
        logger.info("Scanning for Myo (%s, timeout %.0f s)", self.info["target_mac"], self.scan_timeout_s)
        connect = asyncio.ensure_future(self._client_class(myo).with_device(mac=self.mac, aggregate_emg=False))
        stop_wait = asyncio.ensure_future(self._stop_event.wait())
        done, _ = await asyncio.wait({connect, stop_wait}, timeout=self.scan_timeout_s,
                                     return_when=asyncio.FIRST_COMPLETED)
        stop_wait.cancel()
        if connect not in done:
            connect.cancel()
            await asyncio.gather(connect, return_exceptions=True)
            if self._stop_event.is_set():
                return
            self._set_state(ConnectionState.ERROR, f"No Myo found within {self.scan_timeout_s:.0f} s. Is it awake "
                            "(LED on) and in range, and is the Bluetooth adapter attached to WSL?")
            return
        client = connect.result()  # re-raises connection errors (handled in _thread_main)
        try:
            self.info.update({"device_name": client.device.name, "device_address": client.device.address})
            try:
                self.info["battery_percent"] = await client.battery_level()
            except Exception as exc:  # battery is informational only
                logger.debug("battery read failed: %s", exc)
            mode = EMGMode.SEND_EMG if self.emg_mode == "emg" else EMGMode.SEND_RAW
            await client.setup(classifier_mode=ClassifierMode.DISABLED, emg_mode=mode, imu_mode=IMUMode.NONE)
            await client.start()
            self._set_state(ConnectionState.STREAMING)
            logger.info("Streaming EMG from %s (%s)", self.info["device_name"], self.info["device_address"])
            await self._supervise(client)
        finally:
            await self._shutdown(client)

    async def _supervise(self, client: Any) -> None:
        """Wait for stop; detect link loss and stalled data."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            bleak_client = getattr(client, "_client", None)
            if bleak_client is None or not bleak_client.is_connected:
                self._set_state(ConnectionState.ERROR, "Myo disconnected (BLE link lost)")
                return
            st = self.status()
            stalled = st.last_sample_age_s is not None and st.last_sample_age_s > self.stall_timeout_s
            self.info["stalled"] = bool(stalled)

    async def _shutdown(self, client: Any) -> None:
        bleak_client = getattr(client, "_client", None)
        if bleak_client is None or not bleak_client.is_connected:
            return
        for step in (client.stop, client.disconnect):
            try:
                await asyncio.wait_for(step(), timeout=5.0)
            except Exception as exc:
                logger.warning("dl-myo %s failed during shutdown: %s", step.__name__, exc)
        logger.info("Myo disconnected at %s", time.strftime("%H:%M:%S"))
