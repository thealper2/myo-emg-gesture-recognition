"""Device layer, mock stream, real-time engine, live session and dashboard startup (no hardware)."""

import time

import numpy as np
import pytest

from conftest import tiny_pipeline
from myo_emg.device.dl_myo_adapter import DLMyoSource, DeviceDependencyError, import_dl_myo, packet_to_samples
from myo_emg.device.interface import ConnectionState, EMGSource, RateMeter, SampleBatch, SampleQueue
from myo_emg.device.mock_stream import MockEMGStream
from myo_emg.inference.realtime import LiveSession, RealtimeEngine

CLASSES = ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]


@pytest.fixture
def pipeline():
    return tiny_pipeline(name="tiny_cnn")


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# --- queue ------------------------------------------------------------------------------------
def test_queue_fifo_and_timeout():
    q = SampleQueue(10)
    assert q.get(timeout=0.01) is None
    for i in range(3):
        q.put(SampleBatch(np.zeros((2, 8)), time.time(), i))
    assert [q.get(0.1).seq for _ in range(3)] == [0, 1, 2]


def test_queue_overflow_drops_oldest_and_counts():
    q = SampleQueue(max_batches=4)
    for i in range(10):
        q.put(SampleBatch(np.zeros((2, 8)), 0.0, i))
    assert len(q) == 4 and q.dropped_batches == 6 and q.dropped_samples == 12
    assert q.get(0.1).seq == 6


def test_rate_meter():
    m = RateMeter(window_s=2.0)
    for k in range(100):
        m.add(2, 100.0 + k * 0.01)
    assert m.rate(now=100.99) == pytest.approx(200, rel=0.05)
    assert m.rate(now=200.0) is None


# --- sources ----------------------------------------------------------------------------------
class DummySource(EMGSource):
    description = "dummy"

    def start(self):
        self._set_state(ConnectionState.STREAMING)

    def stop(self):
        self._set_state(ConnectionState.DISCONNECTED)


def test_invalid_packets_are_counted_not_queued():
    q = SampleQueue()
    s = DummySource(q)
    assert not s._emit(np.zeros((2, 7)))
    assert not s._emit("garbage")
    assert not s._emit(np.zeros((0, 8)))
    assert s._emit(np.zeros((2, 8)))
    st = s.status()
    assert st.invalid_packets == 3 and st.samples_received == 2 and len(q) == 1


def test_mock_synthetic_stream_runs_in_real_time():
    q = SampleQueue()
    m = MockEMGStream(q, mode="synthetic")
    assert m.simulated and "SIMULATED" in m.description
    m.start()
    assert wait_until(lambda: m.status().samples_received >= 100)
    m.stop()
    st = m.status()
    assert st.state == ConnectionState.DISCONNECTED
    b = q.get(0.1)
    assert b.samples.shape == (2, 8) and np.all(np.abs(b.samples) <= 128)


def test_mock_replay_uses_prepared_test_data(prepared_config):
    from myo_emg.utils.config import processed_dir

    q = SampleQueue()
    m = MockEMGStream(q, mode="replay", data_file=processed_dir(prepared_config) / "epn612_test.npz",
                      realtime=False)
    m.start()
    assert wait_until(lambda: m.status().samples_received > 5000)
    m.stop()
    assert m.current_truth in CLASSES and m.info["replay_recordings"] > 0


def test_mock_replay_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Replay data not found"):
        MockEMGStream(SampleQueue(), mode="replay", data_file=tmp_path / "nope.npz")


def test_simulated_disconnect_is_reported():
    m = MockEMGStream(SampleQueue(), mode="synthetic")
    m.start()
    assert wait_until(lambda: m.state == ConnectionState.STREAMING)
    m.simulate_disconnect()
    st = m.status()
    assert st.state == ConnectionState.ERROR and "disconnect" in st.error


# --- dl-myo adapter (no hardware) -----------------------------------------------------------
def test_packet_conversion_and_channel_shift():
    s1, s2 = tuple(range(8)), tuple(range(10, 18))
    np.testing.assert_array_equal(packet_to_samples(s1, s2), [s1, s2])
    shifted = packet_to_samples(s1, s2, channel_shift=2)
    np.testing.assert_array_equal(shifted[0], [2, 3, 4, 5, 6, 7, 0, 1])
    with pytest.raises(ValueError):
        packet_to_samples((1, 2, 3), s2)


def test_dl_myo_is_importable_from_local_checkout():
    try:
        myo = import_dl_myo()
    except DeviceDependencyError as exc:
        pytest.skip(f"dl-myo not available: {exc}")
    assert hasattr(myo, "MyoClient")


def test_dl_myo_source_rejects_bad_mode():
    with pytest.raises(ValueError):
        DLMyoSource(SampleQueue(), emg_mode="filtered")


# --- engine and session ---------------------------------------------------------------------
def test_engine_produces_predictions_and_stats(pipeline):
    q = SampleQueue()
    eng = RealtimeEngine(pipeline, q, threshold=0.0, plot_seconds=2)
    x = np.random.default_rng(0).normal(0, 5, (200, 8)).astype(np.float32)
    for i in range(0, 200, 2):
        eng.process(SampleBatch(x[i : i + 2], time.time(), i))
    snap = eng.snapshot(seconds=1)
    assert snap["windows_processed"] == (200 - 40) // 10 + 1
    assert snap["signal"].shape == (200, 8)
    np.testing.assert_array_equal(snap["signal"], x)
    assert snap["prediction"].label in CLASSES
    assert snap["latency"]["model"]["mean_ms"] >= 0 and snap["latency"]["end_to_end"]["mean_ms"] >= 0


def test_engine_survives_bad_batches(pipeline):
    eng = RealtimeEngine(pipeline, SampleQueue())
    assert eng.process(SampleBatch(np.zeros((2, 5), np.float32), time.time(), 0)) == []
    assert eng.snapshot()["errors"]


def test_demo_session_end_to_end_without_device(pipeline):
    session = LiveSession(pipeline, "demo", {"inference": {"confidence_threshold": 0.0}})
    assert session.allowed_sources == ("mock_replay", "mock_synthetic")
    session.connect("mock_synthetic")
    assert wait_until(lambda: (session.snapshot()["engine"] or {}).get("windows_processed", 0) >= 3)
    snap = session.snapshot()
    assert snap["status"].simulated and snap["status"].state == ConnectionState.STREAMING
    session.disconnect()
    assert session.snapshot()["status"] is None


def test_live_session_never_falls_back_to_mock(pipeline):
    def failing_factory(kind, queue, options):
        raise DeviceDependencyError("bluetooth unavailable")

    session = LiveSession(pipeline, "live", {}, source_factory=failing_factory)
    with pytest.raises(ValueError, match="not allowed"):
        session.connect("mock_synthetic")
    msg = session.connect("myo")
    assert "bluetooth unavailable" in msg
    snap = session.snapshot()
    assert snap["status"] is None and snap["engine"] is None and "bluetooth unavailable" in snap["last_error"]


def test_live_session_reports_device_disconnect(pipeline):
    holder = {}

    def factory(kind, queue, options):
        holder["src"] = MockEMGStream(queue, mode="synthetic")
        holder["src"].simulated = False  # stand-in for a physical device in this test only
        return holder["src"]

    session = LiveSession(pipeline, "live", {}, source_factory=factory)
    session.connect("myo")
    assert wait_until(lambda: session.snapshot()["status"].state == ConnectionState.STREAMING)
    holder["src"].simulate_disconnect("Myo disconnected (BLE link lost)")
    st = session.snapshot()["status"]
    assert st.state == ConnectionState.ERROR and "BLE link lost" in st.error
    session.disconnect()


def test_dashboard_starts_without_device(pipeline):
    gr = pytest.importorskip("gradio")
    from myo_emg.app.dashboard import build_dashboard, render, render_plot

    session = LiveSession(pipeline, "live", {"inference": {"confidence_threshold": 0.6}})
    app = build_dashboard(session)
    assert isinstance(app, gr.Blocks)
    dev, pred_html, scores, diag = render(session)
    assert "Not connected" in dev and scores is None and "tiny_cnn" in diag
    assert render_plot(session, 5) is not None


def test_dashboard_render_marks_simulated_data(pipeline):
    pytest.importorskip("gradio")
    from myo_emg.app.dashboard import render, render_plot

    session = LiveSession(pipeline, "demo", {"inference": {"confidence_threshold": 0.0}})
    session.connect("mock_synthetic")
    assert wait_until(lambda: (session.snapshot()["engine"] or {}).get("windows_processed", 0) >= 1)
    dev, pred_html, scores, diag = render(session)
    fig = render_plot(session, 2)
    session.disconnect()
    assert "SIMULATED" in fig.axes[0].get_title()
    assert "SIMULATED" in dev and "SIMULATED" in pred_html and set(scores) == set(CLASSES)
