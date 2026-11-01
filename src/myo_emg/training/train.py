"""Final EMGNet training with the best HPO parameters (or the config defaults) on the full training set."""

from __future__ import annotations

import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

from myo_emg.data.preprocess import Standardizer
from myo_emg.data.splits import split_hash
from myo_emg.data.windowing import WindowingSpec
from myo_emg.inference.predictor import GesturePipeline, load_bundle_meta, load_pipeline, resolve_device, save_bundle
from myo_emg.models.emgnet import architecture_from_config
from myo_emg.training.data import DevData
from myo_emg.training.evaluate import (base_meta, bundle_spec, evaluate_pipeline, final_model_dir, finalize_model,
                                       reports_dir, verify_bundle)
from myo_emg.training.hpo import apply_params, best_params_path, load_best_params
from myo_emg.training.trainer import Augment, GPUWindows, train_emgnet
from myo_emg.utils.config import apply_override

logger = logging.getLogger(__name__)

MODEL_NAME = "emgnet"


def resolve_config(
    config: dict[str, Any], use_hpo: bool = True, overrides: list[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Config defaults < best HPO parameters < explicit CLI ``--set`` overrides."""
    best = load_best_params(config) if use_hpo else None
    if best is None:
        if use_hpo:
            logger.warning("No HPO result at %s: training with the defaults from configs/training.yaml",
                           best_params_path(config))
        return config, None
    logger.info("Using HPO trial %s (val macro F1 %.4f during search): %s", best["best_trial"],
                best["best_val_macro_f1"], best["params"])
    cfg = apply_params(config, best["params"])
    for override in overrides or []:  # the user's --set wins over the searched value
        apply_override(cfg, override)
        logger.info("CLI override applied on top of HPO parameters: %s", override)
    return cfg, best


def run(config: dict[str, Any], use_hpo: bool = True, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg, best = resolve_config(config, use_hpo, overrides)
    tcfg = cfg["train"]
    device = resolve_device(tcfg.get("device", "auto"))
    classes = list(cfg["dataset"]["classes"])
    dev = DevData(cfg)
    spec = WindowingSpec.from_config(cfg)
    train_ws, val_ws = dev.window_set("train", spec), dev.window_set("val", spec)
    standardizer = Standardizer.fit(dev.signals["train"])  # training participants only
    arch = architecture_from_config(cfg["model"], len(cfg["dataset"]["channel_order"]), len(classes))
    logger.info("Final training on %s: arch %s, window %d samples, %d train / %d val windows", device, arch,
                spec.window_length, len(train_ws), len(val_ws))

    t0 = time.perf_counter()
    model, history, best_epoch, best_f1 = train_emgnet(
        arch, GPUWindows(train_ws, device), GPUWindows(val_ws, device), standardizer, tcfg,
        Augment.from_config(cfg.get("augment")), epochs=int(tcfg["epochs"]), patience=int(tcfg["patience"]),
        max_windows_per_epoch=tcfg.get("max_windows_per_epoch"), seed=int(cfg["seed"]),
    )
    train_s = time.perf_counter() - t0
    GPUWindows.clear_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rdir = reports_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "training_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

    n_params = sum(p.numel() for p in model.parameters())
    training = {
        **{k: tcfg[k] for k in ("batch_size", "epochs", "patience", "lr", "weight_decay", "warmup_epochs",
                                "label_smoothing", "class_weighting", "amp", "max_windows_per_epoch") if k in tcfg},
        "optimizer": "AdamW", "schedule": "linear warm-up + cosine", "loss": "cross-entropy",
        "augment": cfg.get("augment"), "best_epoch": best_epoch, "epochs_run": len(history),
        "best_val_macro_f1_training_loop": best_f1, "n_parameters": n_params, "seed": int(cfg["seed"]),
        "n_train_windows": len(train_ws), "train_class_counts": train_ws.class_counts(len(classes)),
        "device": str(device), "hpo_params": best["params"] if best else None,
        "hpo_trial": best["best_trial"] if best else None,
    }
    meta = base_meta(cfg, MODEL_NAME, arch, dev.split, training)
    pipeline = GesturePipeline(meta, model.float(), standardizer, device)
    save_fn = partial(_save, meta=meta, model=model, standardizer=standardizer)
    return finalize_model(cfg, pipeline, save_fn, val_ws, train_s, final_model_dir(cfg))


def refinalize(config: dict[str, Any]) -> dict[str, Any]:
    """Re-run the post-training checks on the saved ``final_model`` without training again."""
    out = final_model_dir(config)
    meta = load_bundle_meta(out)
    dev = DevData(config)
    if meta["split"]["sha256_16"] != split_hash(dev.split):
        raise RuntimeError(f"{out} was trained with a different participant split than the current split file")
    if meta["filter"] != dict(config["preprocessing"]["filter"]):
        raise RuntimeError(f"{out} was trained with filter {meta['filter']}, config has a different one")
    pipeline = load_pipeline(out, device=config["train"].get("device", "auto"))
    val_ws = dev.window_set("val", bundle_spec(meta))
    val_metrics, val_proba = evaluate_pipeline(pipeline, val_ws)
    logger.info("[%s] val macro-F1 %.4f acc %.4f (re-evaluated saved bundle)", pipeline.name,
                val_metrics["macro_f1"], val_metrics["accuracy"])
    meta["validation_metrics"] = val_metrics
    (out / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return verify_bundle(config, out, val_ws, val_proba)


def _save(out_dir: Path, updates: dict[str, Any], meta, model, standardizer) -> None:
    save_bundle(out_dir, {**meta, **updates}, model, standardizer)
