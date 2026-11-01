"""Entry point for the dashboard (``make demo`` / ``make live``) and the headless Myo check."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from myo_emg.data.prepare import TEST_FILE, split_path
from myo_emg.device.interface import ConnectionState, SampleQueue
from myo_emg.inference.predictor import StreamingPredictor, load_pipeline
from myo_emg.inference.realtime import LiveSession, default_source_factory
from myo_emg.utils.config import add_config_args, artifacts_dir, load_config, processed_dir, resolve_path
from myo_emg.utils.logging import setup_logging

logger = logging.getLogger(__name__)

CONFIG_FILES = ("configs/dataset.yaml", "configs/inference.yaml")


def session_options(config: dict[str, Any]) -> dict[str, Any]:
    """Options for sources/engine: inference, device, dashboard and mock-replay settings."""
    mock = dict(config.get("mock", {}))
    replay = processed_dir(config) / TEST_FILE
    mock.setdefault("replay_file", str(replay) if replay.is_file() else None)
    if not mock.get("participants") and split_path(config).is_file():
        test = json.loads(split_path(config).read_text(encoding="utf-8"))["test"]
        mock["participants"] = test[: int(mock.get("n_participants", 5))]
    return {"inference": dict(config["inference"]), "device": dict(config.get("device", {})),
            "dashboard": dict(config.get("dashboard", {})), "mock": mock}


def load_model(config: dict[str, Any]):
    model_dir = resolve_path(config["inference"]["model_dir"])
    if not (model_dir / "bundle.json").is_file():
        raise SystemExit(f"No trained model at {model_dir}. Run `make pipeline` (or set MYO_EMG_MODEL_DIR).")
    import torch

    # one window at a time: extra intra-op threads only compete with the BLE/UI threads
    torch.set_num_threads(int(config["inference"].get("torch_threads", 1)))
    pipe = load_pipeline(model_dir, device=config["inference"].get("device", "cpu"))
    logger.info("Loaded model %s from %s (no training is performed)", pipe.name, model_dir)
    return pipe


def run_dashboard(args: argparse.Namespace) -> None:
    config = load_config(args.config or list(CONFIG_FILES), args.overrides)
    pipe = load_model(config)
    options = session_options(config)
    if args.mac:
        options["device"]["mac"] = args.mac
    session = LiveSession(pipe, args.mode, options)
    from myo_emg.app.dashboard import build_dashboard

    dash = config.get("dashboard", {})
    app = build_dashboard(session, refresh_s=float(dash.get("refresh_s", 0.25)),
                          default_seconds=float(dash.get("plot_seconds", 5)),
                          plot_refresh_s=float(dash.get("plot_refresh_s", 0.5)))
    if args.mode == "demo" or args.auto_connect:
        logger.info(session.connect(session.allowed_sources[0], args.mac))
    try:
        app.launch(server_name=args.host or dash.get("host", "127.0.0.1"),
                   server_port=int(args.port or dash.get("port", 7860)), inbrowser=False)
    finally:
        session.disconnect()


def run_myo_check(args: argparse.Namespace) -> None:
    """Connect to the physical Myo, stream for N seconds and write a hardware report (no UI)."""
    config = load_config(args.config or list(CONFIG_FILES), args.overrides)
    options = session_options(config)
    if args.mac:
        options["device"]["mac"] = args.mac
    pipe = load_model(config) if (resolve_path(config["inference"]["model_dir"]) / "bundle.json").is_file() else None
    queue = SampleQueue()
    source = default_source_factory("myo", queue, {**options, "classes": pipe.classes if pipe else None})
    source.start()
    t_end = time.monotonic() + float(options["device"].get("scan_timeout_s", 20)) + 5
    while source.state == ConnectionState.CONNECTING and time.monotonic() < t_end:
        time.sleep(0.1)
    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_requested": args.seconds}
    if source.state != ConnectionState.STREAMING:
        st = source.status()
        source.stop()
        report.update({"result": "FAILED", "state": st.state.value, "error": st.error, "info": st.info})
        _write_report(config, report)
        raise SystemExit(f"Myo check FAILED: {st.error or st.state.value}")

    predictor = StreamingPredictor(pipe, threshold=float(config["inference"]["confidence_threshold"])) if pipe else None
    samples, counts = [], {}
    queue.clear()  # drop packets buffered during device setup so the rate covers exactly the window
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds and source.state == ConnectionState.STREAMING:
        batch = queue.get(timeout=0.2)
        if batch is None:
            continue
        samples.append(batch.samples)
        for p in predictor.push(batch.samples) if predictor else []:
            counts[p.label] = counts.get(p.label, 0) + 1
    elapsed = time.monotonic() - t0
    st = source.status()
    source.stop()
    x = np.concatenate(samples) if samples else np.zeros((0, 8))
    report.update({
        "result": "OK" if x.shape[0] > 0 else "NO_DATA",
        "final_state": st.state.value, "error": st.error, "info": st.info,
        "elapsed_s": elapsed, "samples": int(x.shape[0]),
        "effective_rate_hz": x.shape[0] / elapsed if elapsed else None,
        "source_measured_rate_hz_last_2s": st.measured_rate_hz,
        "source_average_rate_hz": st.average_rate_hz,
        "saturated_fraction": float((np.abs(x) >= 127).mean()) if x.size else None,
        "invalid_packets": st.invalid_packets, "queue_dropped_samples": queue.dropped_samples,
        "value_min": x.min(axis=0).tolist() if x.size else None,
        "value_max": x.max(axis=0).tolist() if x.size else None,
        "mean_abs_per_channel": np.abs(x).mean(axis=0).round(3).tolist() if x.size else None,
        "prediction_counts": counts, "model": pipe.name if pipe else None,
    })
    path = _write_report(config, report)
    print(json.dumps(report, indent=1))
    print(f"\nReport written to {path}")
    if report["result"] != "OK":
        raise SystemExit("Myo check: connected but no EMG samples received")


def run_scan(args: argparse.Namespace) -> None:
    """List nearby BLE devices with Bleak (the backend dl-myo uses) and mark Myo armbands."""
    import asyncio

    from bleak import BleakScanner

    myo_uuid = "d5060001-a904-deb9-4748-2c7f4a124842"

    async def scan() -> None:
        found = await BleakScanner.discover(timeout=args.seconds, return_adv=True)
        n_myo = 0
        for addr, (dev, adv) in sorted(found.items(), key=lambda kv: -(kv[1][1].rssi or -999)):
            is_myo = myo_uuid in [u.lower() for u in adv.service_uuids or []]
            n_myo += is_myo
            print(f"{addr}  rssi={adv.rssi:>4}  name={dev.name!r}{'   <-- MYO' if is_myo else ''}")
        print(f"\n{len(found)} BLE devices, {n_myo} Myo armband(s)")
        if not n_myo:
            print("No Myo found: wake it (tap/move it or plug in USB; the logo LED lights up) and keep it close.")

    asyncio.run(scan())


def _write_report(config: dict[str, Any], report: dict[str, Any]) -> Path:
    out = artifacts_dir(config) / "reports" / "hardware"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"myo_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    dash = sub.add_parser("dashboard", help="start the Gradio dashboard")
    dash.add_argument("--mode", choices=["demo", "live"], required=True)
    dash.add_argument("--mac", default=None)
    dash.add_argument("--host", default=None)
    dash.add_argument("--port", type=int, default=None)
    dash.add_argument("--auto-connect", action="store_true", help="live mode: connect to the Myo on start")
    check = sub.add_parser("myo-check", help="headless physical Myo streaming test")
    check.add_argument("--mac", default=None)
    check.add_argument("--seconds", type=float, default=10.0)
    scan = sub.add_parser("scan", help="list nearby BLE devices and mark Myo armbands")
    scan.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    setup_logging(args.log_level)
    {"dashboard": run_dashboard, "myo-check": run_myo_check, "scan": run_scan}[args.command](args)


if __name__ == "__main__":
    main()
