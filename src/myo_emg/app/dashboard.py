"""Gradio dashboard: device connection, live 8-channel EMG plot, predictions and diagnostics.

The UI only *reads* snapshots from ``LiveSession`` on a timer; acquisition and inference run in their
own threads, so rendering can never block the sensor stream.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from myo_emg.device.interface import ConnectionState
from myo_emg.inference.realtime import LiveSession

SOURCE_LABELS = {
    "myo": "Myo Armband (dl-myo, Bluetooth LE)",
    "mock_replay": "SIMULATED: replay of EMG-EPN-612 test recordings",
    "mock_synthetic": "SIMULATED: synthetic noise",
}
# The Myo used during development delivered ~212 samples/s over BLE (no duplicated packets) while the
# dataset metadata says 200 Hz; deviations beyond this tolerance are flagged in the UI.
RATE_TOLERANCE = 0.10
STATE_ICON = {
    ConnectionState.DISCONNECTED: "⚪",
    ConnectionState.CONNECTING: "🟡",
    ConnectionState.STREAMING: "🟢",
    ConnectionState.ERROR: "🔴",
}


def _fmt(v: float | None, unit: str = "", digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}{unit}"


def emg_figure(signal: np.ndarray, fs: float, simulated: bool, problem: str | None):
    """8 channels stacked with vertical offsets on one axis (cheap to draw; display only)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5.2))
    ax = fig.add_subplot()
    n = signal.shape[0]
    # common scale for all channels so amplitudes stay comparable; at least +-10 counts, at most +-128
    scale = float(np.clip(np.percentile(np.abs(signal), 99.5) * 1.2, 10, 128)) if n else 10.0
    if n:
        t = (np.arange(n) - n) / fs
        step = max(1, n // 800)  # decimation for drawing speed only
        color = "#ff7f0e" if simulated else "#1f77b4"
        for c in range(8):
            ax.plot(t[::step], signal[::step, c] / scale - 2.2 * c, lw=0.6, color=color)
    ax.set_yticks([-2.2 * c for c in range(8)], [f"ch{c + 1}" for c in range(8)])
    ax.set_ylim(-2.2 * 7 - 1.2, 1.2)
    ax.set_xlabel(f"time [s] (0 = newest sample)   •   scale ±{scale:.0f} counts per channel")
    ax.grid(alpha=0.2)
    title = "SIMULATED DATA (mock stream)" if simulated else "Myo EMG (raw counts)"
    if problem:
        title += f" — {problem}"
    ax.set_title(title, color="red" if (simulated or problem) else "black", fontsize=10)
    fig.tight_layout()
    return fig


def _plot_problem(st) -> str | None:
    if st is not None and st.state == ConnectionState.STREAMING and st.last_sample_age_s is not None \
            and st.last_sample_age_s > 1.0:
        return f"no data for {st.last_sample_age_s:.1f} s"
    if st is not None and st.state == ConnectionState.ERROR:
        return "connection error"
    return None


def render_plot(session: LiveSession, seconds: float):
    """Figure with the last ``seconds`` of raw EMG (refreshed less often than the predictions)."""
    snap = session.snapshot(seconds)
    st, eng = snap["status"], snap["engine"]
    signal = eng["signal"] if eng else np.zeros((0, 8))
    return emg_figure(signal, session.pipeline.sampling_rate_hz, bool(st and st.simulated), _plot_problem(st))


def render(session: LiveSession, seconds: float = 1.0) -> tuple[str, str, dict[str, float] | None, str]:
    """Build (device markdown, prediction html, class scores, diagnostics markdown)."""
    snap = session.snapshot(seconds)
    st, eng = snap["status"], snap["engine"]
    pipe = session.pipeline
    simulated = bool(st and st.simulated)

    # --- device section
    if st is None:
        dev = "⚪ **Not connected**"
        if snap["last_error"]:
            dev += f"\n\n🔴 {snap['last_error']}"
    else:
        dev = f"{STATE_ICON[st.state]} **{st.state.value.upper()}** — {st.description}"
        if simulated:
            dev += "\n\n⚠️ **SIMULATED DATA — this is not a hardware connection.**"
        if st.error:
            dev += f"\n\n🔴 **Error:** {st.error}"
        age = st.last_sample_age_s
        dev += (f"\n\n| samples received | rate (last 2 s / average) | last sample | invalid packets |\n"
                f"|---|---|---|---|\n"
                f"| {st.samples_received} | {_fmt(st.measured_rate_hz, '', 1)} / "
                f"{_fmt(st.average_rate_hz, ' Hz', 1)} (model: {pipe.sampling_rate_hz:.0f} Hz) | "
                f"{_fmt(age, ' s ago', 2)} | {st.invalid_packets} |")
        avg = st.average_rate_hz
        if avg is not None and abs(avg / pipe.sampling_rate_hz - 1) > RATE_TOLERANCE:
            dev += (f"\n\n⚠️ **Sample rate {avg:.0f} Hz differs from the training rate "
                    f"{pipe.sampling_rate_hz:.0f} Hz by more than {RATE_TOLERANCE:.0%}.** Predictions may be unreliable.")
        if st.info:
            dev += "\n\n" + ", ".join(f"`{k}`: {v}" for k, v in st.info.items())
    # --- prediction section
    pred = eng["prediction"] if eng else None
    scores = None
    if pred is None:
        pred_html = "<div style='font-size:2em'>—</div>"
    else:
        color = "#999" if pred.uncertain else "#0a7d28"
        label = "uncertain" if pred.uncertain else pred.label
        best = pipe.classes[pred.index]
        sub = f"top class {best}, score {pred.confidence:.2f} < threshold {eng['threshold']:.2f}" if pred.uncertain \
            else f"score {pred.confidence:.2f} (threshold {eng['threshold']:.2f})"
        ts = datetime.fromtimestamp(pred.timestamp).strftime("%H:%M:%S.%f")[:-3]
        truth = f"<br>replay ground truth: <b>{snap['truth']}</b>" if simulated and snap.get("truth") else ""
        tag = "<b style='color:#d9480f'>[SIMULATED]</b> " if simulated else ""
        pred_html = (f"<div style='font-size:2.6em;font-weight:700;color:{color}'>{tag}{label}</div>"
                     f"<div>{sub}<br>prediction time {ts}{truth}</div>")
        scores = dict(pred.scores)

    # --- diagnostics
    lat = eng["latency"] if eng else {}
    def lat_row(name: str) -> str:
        s = lat.get(name) if lat else None
        return f"| {name} | {_fmt(s and s['mean_ms'], ' ms', 3)} | {_fmt(s and s['p95_ms'], ' ms', 3)} |"

    diag = [
        f"**Model:** `{pipe.name}` ({pipe.meta['architecture'].get('type')}, input={pipe.input_type}, "
        f"window {pipe.window_length} samples / stride {session.engine.predictor.stride if session.engine else pipe.stride})  ",
        f"**Mode:** {session.mode}  •  **Connection:** {st.state.value if st else 'disconnected'}",
        "",
        "| latency | mean | p95 |",
        "|---|---|---|",
        lat_row("preprocess"),
        lat_row("model"),
        lat_row("end_to_end"),
        "",
        f"windows processed: **{eng['windows_processed'] if eng else 0}**  •  "
        f"prediction rate: **{_fmt(eng and eng['prediction_rate_hz'], ' Hz', 1)}**  •  "
        f"queue length: {snap['queue_len']}  •  dropped (queue overflow): {snap['queue_dropped_samples']} samples  •  "
        f"estimated missing vs 200 Hz: {snap.get('estimated_missing_samples', '-')}  •  "
        f"non-finite samples repaired: {eng['invalid_samples'] if eng else 0}",
    ]
    errors = (eng["errors"] if eng else []) + ([snap["last_error"]] if snap["last_error"] else [])
    if errors:
        diag += ["", "**Errors / warnings:**"] + [f"- {e}" for e in errors[-5:]]
    diag += ["", "_Scores are softmax outputs, not calibrated probabilities. `noGesture` = relaxed hand; "
             "other movements are not rejected automatically._"]
    return dev, pred_html, scores, "\n".join(diag)


def build_dashboard(session: LiveSession, refresh_s: float = 0.25, default_seconds: float = 5.0,
                    plot_refresh_s: float = 0.5):
    """Create (but do not launch) the Gradio Blocks app."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # local app: no usage telemetry
    import gradio as gr

    pipe = session.pipeline
    title = "MYO EMG gesture recognition — " + ("LIVE (physical Myo)" if session.mode == "live" else
                                                  "DEMO (SIMULATED data)")
    choices = [(SOURCE_LABELS[k], k) for k in session.allowed_sources]
    inf = session.options.get("inference", {})
    smoothing = (inf.get("smoothing") or {}).get("method", "none")

    with gr.Blocks(title="MYO EMG", analytics_enabled=False) as demo:
        gr.Markdown(f"## {title}\nModel `{pipe.name}` trained offline on EMG-EPN-612 (no calibration, no retraining). "
                    f"Classes: {', '.join(pipe.classes)}.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Device")
                source = gr.Dropdown(choices=choices, value=choices[0][1], label="Source")
                mac = gr.Textbox(label="Myo MAC address (optional, empty = scan by service UUID)",
                                 value=str(session.options.get("device", {}).get("mac") or ""),
                                 visible=session.mode == "live")
                with gr.Row():
                    connect_btn = gr.Button("Connect", variant="primary")
                    disconnect_btn = gr.Button("Disconnect")
                action = gr.Markdown()
                device_md = gr.Markdown()
                gr.Markdown("### Settings")
                seconds = gr.Slider(1, float(session.options.get("dashboard", {}).get("max_plot_seconds", 20)),
                                    value=default_seconds, step=1, label="Plot interval [s]")
                threshold = gr.Slider(0.0, 1.0, value=float(inf.get("confidence_threshold", 0.6)), step=0.05,
                                      label="Confidence threshold")
                smooth = gr.Dropdown(["none", "ema", "moving_average", "majority"], value=smoothing,
                                     label="Temporal smoothing")
            with gr.Column(scale=2):
                pred_html = gr.HTML()
                scores = gr.Label(num_top_classes=len(pipe.classes), label="Class scores (smoothed)")
        plot = gr.Plot(label="Live EMG (8 channels)", format="png")
        diag = gr.Markdown()

        def do_connect(kind: str, mac_value: str) -> str:
            try:
                return session.connect(kind, mac_value.strip() or None)
            except ValueError as exc:
                return f"🔴 {exc}"

        def do_threshold(v: float) -> None:
            inf["confidence_threshold"] = v
            if session.engine is not None:
                session.engine.set_threshold(v)

        def do_smoothing(method: str) -> None:
            cfg = {**(inf.get("smoothing") or {}), "method": method}
            inf["smoothing"] = cfg
            if session.engine is not None:
                session.engine.set_smoothing(cfg)

        connect_btn.click(do_connect, [source, mac], action)
        disconnect_btn.click(lambda: session.disconnect(), None, action)
        threshold.change(do_threshold, threshold, None)
        smooth.change(do_smoothing, smooth, None)
        outputs = [device_md, pred_html, scores, diag]
        gr.Timer(refresh_s).tick(lambda: render(session), None, outputs)
        gr.Timer(plot_refresh_s).tick(lambda s: render_plot(session, s), seconds, plot)
        demo.load(lambda: render(session), None, outputs)
    return demo


def wait_for_state(session: LiveSession, states: set[ConnectionState], timeout: float) -> ConnectionState | None:
    """Helper for scripts/tests: block until the source reaches one of ``states``."""
    end = time.time() + timeout
    while time.time() < end:
        st = session.snapshot(1)["status"]
        if st is not None and st.state in states:
            return st.state
        time.sleep(0.05)
    return None
