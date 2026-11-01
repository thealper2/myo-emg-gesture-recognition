"""Saving the trained model, deployment checks and the one-time held-out test evaluation."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from myo_emg.data.prepare import split_path
from myo_emg.data.splits import split_hash
from myo_emg.data.windowing import WindowingSpec
from myo_emg.inference.predictor import GesturePipeline, load_bundle_meta, load_pipeline, software_versions
from myo_emg.training.data import WindowSet, load_test_set
from myo_emg.utils.config import artifacts_dir
from myo_emg.utils.metrics import classification_metrics, measure_latency, save_confusion_matrix_png

logger = logging.getLogger(__name__)


def final_model_dir(config: dict[str, Any]) -> Path:
    """Model produced by ``make train`` (validated, not yet tested)."""
    return artifacts_dir(config) / "checkpoints" / "final_model"


def best_model_dir(config: dict[str, Any]) -> Path:
    """Deployed model: ``final_model`` after its one-time test evaluation (used by demo/live)."""
    return artifacts_dir(config) / "checkpoints" / "best_model"


def reports_dir(config: dict[str, Any]) -> Path:
    return artifacts_dir(config) / "reports"


def evaluate_pipeline(pipeline: GesturePipeline, wset: WindowSet, chunk: int = 65536) -> tuple[dict[str, Any], np.ndarray]:
    """Run the full inference pipeline on every window of ``wset`` and compute metrics."""
    parts = [pipeline.predict_proba(wset.windows(i, i + chunk)) for i in range(0, len(wset), chunk)]
    proba = np.concatenate(parts)
    metrics = classification_metrics(wset.labels, proba, pipeline.classes, wset.index.rec_index)
    return metrics, proba


def measure_pipeline_latency(pipeline: GesturePipeline, window: np.ndarray, n_runs: int = 300) -> dict[str, Any]:
    """Single-window latency of normalisation, model and both together (CPU pipeline expected)."""
    import torch

    w = window[None].astype(np.float32)
    x = pipeline.transform(w)
    prev = torch.get_num_threads()
    torch.set_num_threads(1)  # as in live inference
    try:
        return {
            "device": str(pipeline.device),
            "torch_threads": 1,
            "preprocess": measure_latency(lambda: pipeline.transform(w), n_runs=n_runs),
            "model": measure_latency(lambda: pipeline.predict_proba_from_input(x), n_runs=n_runs),
            "end_to_end": measure_latency(lambda: pipeline.predict_proba(w), n_runs=n_runs),
        }
    finally:
        torch.set_num_threads(prev)


def base_meta(
    config: dict[str, Any],
    name: str,
    architecture: dict[str, Any],
    split: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    """Metadata stored in every bundle (preprocessing, dataset, split identifiers)."""
    ds = config["dataset"]
    spec = WindowingSpec.from_config(config)
    return {
        "name": name,
        "input_type": "raw",
        "architecture": architecture,
        "classes": list(ds["classes"]),
        "sampling_rate_hz": int(ds["sampling_rate_hz"]),
        "channel_order": list(ds["channel_order"]),
        "window_ms": float(config["windowing"]["window_ms"]),
        "window_length": spec.window_length,
        "stride": spec.stride,
        "windowing": {
            "window_length": spec.window_length,
            "stride": spec.stride,
            "rest_stride": spec.rest_stride,
            "gesture_min_overlap": spec.gesture_min_overlap,
            "rest_from_gesture_recordings": spec.rest_from_gesture_recordings,
        },
        "filter": dict(config["preprocessing"]["filter"]),
        "training": training,
        "dataset": {"name": ds["name"], "version": ds["version"], "root": ds["root"],
                    "train_dir": ds["train_dir"], "test_dir": ds["test_dir"]},
        "split": {
            "file": str(split_path(config)),
            "sha256_16": split_hash(split),
            "seed": split["seed"],
            "val_fraction": split["val_fraction"],
            "train": split["train"],
            "val": split["val"],
            "test": split["test"],
        },
        "software": software_versions(),
        "test_metrics": None,
    }


def finalize_model(
    config: dict[str, Any],
    pipeline: GesturePipeline,
    save_fn: Callable[[Path, dict[str, Any]], None],
    val_set: WindowSet,
    train_seconds: float,
    out: Path,
) -> dict[str, Any]:
    """Validate through the deployed pipeline, save the bundle, verify reload and measure CPU latency."""
    name = pipeline.name
    val_metrics, val_proba = evaluate_pipeline(pipeline, val_set)
    logger.info("[%s] val macro-F1 %.4f acc %.4f bal-acc %.4f min-recall %.3f rec-acc %.4f", name,
                val_metrics["macro_f1"], val_metrics["accuracy"], val_metrics["balanced_accuracy"],
                val_metrics["min_class_recall"], val_metrics["recording_accuracy"])
    if out.exists():
        shutil.rmtree(out)
    save_fn(out, {"validation_metrics": val_metrics, "train_seconds": train_seconds})
    return verify_bundle(config, out, val_set, val_proba)


def verify_bundle(config: dict[str, Any], out: Path, val_set: WindowSet, val_proba: np.ndarray) -> dict[str, Any]:
    """Reload the saved bundle on CPU, compare with ``val_proba``, measure latency and write the reports."""
    reloaded = load_pipeline(out, device="cpu")
    name = reloaded.name
    n_check = min(2048, len(val_set))
    p_reload = reloaded.predict_proba(val_set.windows(0, n_check))
    max_diff = float(np.abs(p_reload - val_proba[:n_check]).max()) if n_check else 0.0
    agree = float((p_reload.argmax(1) == val_proba[:n_check].argmax(1)).mean()) if n_check else 1.0
    # GPU (fp32, TF32 off) vs CPU kernels differ by ~1e-6; anything larger means a broken bundle
    if max_diff > 1e-3 or agree < 0.999:
        raise AssertionError(f"[{name}] reloaded bundle predictions differ (max |dp|={max_diff:.2e}, agree={agree}). "
                             f"The bundle is kept at {out}; after fixing the cause run `make finalize`.")
    latency = measure_pipeline_latency(reloaded, val_set.windows(0, 1)[0])

    meta = load_bundle_meta(out)
    val_metrics, train_seconds = meta["validation_metrics"], meta["train_seconds"]
    meta["reload_check"] = {"n_windows": n_check, "max_abs_score_diff": max_diff, "argmax_agreement": agree}
    meta["latency_cpu"] = latency
    (out / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    scalers = artifacts_dir(config) / "scalers"
    scalers.mkdir(parents=True, exist_ok=True)
    (scalers / f"{name}.json").write_text(json.dumps(meta["normalization"], indent=1), encoding="utf-8")
    rdir = reports_dir(config)
    rdir.mkdir(parents=True, exist_ok=True)
    report = {"name": name, "val": val_metrics, "train_seconds": train_seconds,
              "latency_ms": latency["end_to_end"]["mean_ms"], "size_mb": meta["model_size_bytes"] / 1e6,
              "bundle": str(out)}
    (rdir / "validation_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    save_confusion_matrix_png(val_metrics["confusion_matrix"], reloaded.classes, f"{name} - validation",
                              rdir / "validation_confusion.png")
    logger.info("[%s] saved to %s (%.2f MB, CPU latency %.3f ms/window)", name, out, report["size_mb"],
                report["latency_ms"])
    return report


def check_constraints(config: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    """Deployment limits from ``selection``; returns the list of violations."""
    sel = config["selection"]
    problems = []
    lat = meta["latency_cpu"]["end_to_end"]["mean_ms"]
    if lat > float(sel["max_latency_ms"]):
        problems.append(f"CPU latency {lat:.2f} ms > {sel['max_latency_ms']} ms")
    size = meta["model_size_bytes"] / 1e6
    if size > float(sel["max_model_size_mb"]):
        problems.append(f"size {size:.1f} MB > {sel['max_model_size_mb']} MB")
    return problems


def bundle_spec(meta: dict[str, Any]) -> WindowingSpec:
    w = meta["windowing"]
    return WindowingSpec(w["window_length"], w["stride"], w["rest_stride"], w["gesture_min_overlap"],
                         w["rest_from_gesture_recordings"])


def evaluate_test(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Deploy ``final_model`` to ``best_model`` and evaluate it once on the held-out participants."""
    src, dst = final_model_dir(config), best_model_dir(config)
    meta = load_bundle_meta(src)
    if "latency_cpu" not in meta:
        raise RuntimeError(f"{src} did not pass the post-training checks; run `make finalize` first")
    problems = check_constraints(config, meta)
    if problems:
        raise RuntimeError(f"Trained model violates deployment limits: {problems}")
    split = json.loads(split_path(config).read_text(encoding="utf-8"))
    if meta["split"]["sha256_16"] != split_hash(split):
        raise RuntimeError("Model was trained with a different participant split than the current split file")
    if set(meta["split"]["test"]) & (set(meta["split"]["train"]) | set(meta["split"]["val"])):
        raise AssertionError("Test participants overlap training/validation participants")

    if dst.exists():
        old = load_bundle_meta(dst)
        same = old.get("created_at") == meta.get("created_at") and old.get("name") == meta.get("name")
        if same and old.get("test_metrics") is not None and not force:
            raise RuntimeError(f"{dst} already holds the test evaluation of this model; the held-out test set "
                               "is evaluated once. Use `make evaluate FORCE=1` to repeat it deliberately.")
        if not same:  # keep the previously deployed model instead of deleting it
            stamp = str(old.get("created_at", "unknown")).replace(":", "").replace("-", "")[:15]
            archive = dst.parent / "archive" / f"{old.get('name', 'model')}_{stamp}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                shutil.rmtree(archive)
            shutil.move(str(dst), str(archive))
            logger.info("Archived previous deployed model to %s", archive)
        else:
            shutil.rmtree(dst)
    shutil.copytree(src, dst)

    pipeline = load_pipeline(dst, device=config["train"].get("device", "auto"))
    meta = pipeline.meta
    test_set = load_test_set(config, bundle_spec(meta), meta["filter"], float(meta["sampling_rate_hz"]))
    metrics, _ = evaluate_pipeline(pipeline, test_set)
    metrics["n_participants"] = len(test_set.participants)
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["test_metrics"] = metrics
    (dst / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    rdir = reports_dir(config)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "test_metrics.json").write_text(json.dumps({"model": meta["name"], **metrics}, indent=1), encoding="utf-8")
    save_confusion_matrix_png(metrics["confusion_matrix"], pipeline.classes, f"{meta['name']} - held-out test",
                              rdir / "test_confusion.png")
    v = meta["validation_metrics"]
    md = [
        f"# Held-out test evaluation: {meta['name']}",
        "",
        f"Architecture: `{json.dumps(meta['architecture'])}`  ",
        f"Window: {meta['window_length']} samples ({meta['window_ms']:.0f} ms), stride {meta['stride']}  ",
        f"Hyper-parameters from HPO: {meta['training'].get('hpo_params') is not None}  ",
        f"Test participants: {metrics['n_participants']} (official testingJSON users, labelled trainingSamples); "
        f"windows: {metrics['n_windows']}; recordings: {metrics['n_recordings']}",
        "",
        "| metric | validation | test |",
        "|---|---|---|",
    ]
    for k in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1", "recording_accuracy", "min_class_recall"):
        md.append(f"| {k} | {v[k]:.4f} | {metrics[k]:.4f} |")
    md += ["", "| class | precision | recall | f1 |", "|---|---|---|---|"]
    for c in pipeline.classes:
        pc = metrics["per_class"][c]
        md.append(f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} |")
    (rdir / "test_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("TEST %s: macro F1 %.4f, accuracy %.4f, recording accuracy %.4f", meta["name"],
                metrics["macro_f1"], metrics["accuracy"], metrics["recording_accuracy"])
    return metrics
