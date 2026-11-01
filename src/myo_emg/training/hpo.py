"""Optuna hyper-parameter search for EMGNet on the validation participants.

The search space lives in ``configs/training.yaml -> hpo.search_space``; each key is a dotted config
path (e.g. ``model.channels``, ``train.lr``, ``windowing.window_ms``) and each value a spec::

    {type: categorical, choices: [...]} | {type: int, low, high, step?} | {type: float, low, high, log?}

TPE sampling + median pruning on the per-epoch validation macro F1. Configurations whose untrained
single-window CPU latency exceeds ``selection.max_latency_ms`` are rejected before training. The study
is stored in SQLite (``artifacts/hpo/study.db``), so an interrupted search resumes where it stopped.
The test participants are never used here.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from myo_emg.data.preprocess import Standardizer
from myo_emg.data.windowing import WindowingSpec
from myo_emg.inference.predictor import resolve_device
from myo_emg.models import build_neural_model
from myo_emg.models.emgnet import architecture_from_config
from myo_emg.training.data import DevData
from myo_emg.training.trainer import Augment, GPUWindows, cpu_latency_ms, train_emgnet
from myo_emg.utils.config import artifacts_dir

logger = logging.getLogger(__name__)


def hpo_dir(config: dict[str, Any]) -> Path:
    return artifacts_dir(config) / "hpo"


def best_params_path(config: dict[str, Any]) -> Path:
    return hpo_dir(config) / "best_params.yaml"


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def apply_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``config`` with dotted-key parameters applied."""
    cfg = copy.deepcopy(config)
    for k, v in params.items():
        set_dotted(cfg, k, v)
    return cfg


def suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    raise ValueError(f"Unknown search-space type {kind!r} for {name}")


def load_best_params(config: dict[str, Any]) -> dict[str, Any] | None:
    path = best_params_path(config)
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Objective:
    """One Optuna trial = sample parameters -> latency check -> short training -> best val macro F1."""

    def __init__(self, config: dict[str, Any], dev: DevData, standardizer: Standardizer) -> None:
        self.config = config
        self.dev = dev
        self.standardizer = standardizer
        self.device = resolve_device(config["train"].get("device", "auto"))
        self.hpo = config["hpo"]
        self.classes = list(config["dataset"]["classes"])

    def __call__(self, trial: Any) -> float:
        import optuna

        params = {name: suggest(trial, name, spec) for name, spec in self.hpo["search_space"].items()}
        cfg = apply_params(self.config, params)
        spec = WindowingSpec.from_config(cfg)
        arch = architecture_from_config(cfg["model"], len(cfg["dataset"]["channel_order"]), len(self.classes))
        latency = cpu_latency_ms(arch, spec.window_length)
        n_params = sum(p.numel() for p in build_neural_model(arch).parameters())
        trial.set_user_attr("cpu_latency_ms", latency)
        trial.set_user_attr("n_parameters", n_params)
        limit = float(cfg["selection"]["max_latency_ms"])
        if latency > limit:
            trial.set_user_attr("rejected", f"latency {latency:.2f} ms > {limit} ms")
            raise optuna.TrialPruned(f"too slow on CPU ({latency:.2f} ms)")

        logger.info("trial %d: %s (%d params, %.2f ms CPU)", trial.number, params, n_params, latency)
        t0 = time.perf_counter()
        try:
            train = GPUWindows(self.dev.window_set("train", spec), self.device)
            val = GPUWindows(self.dev.window_set("val", spec), self.device)
            _, history, best_epoch, best_f1 = train_emgnet(
                arch, train, val, self.standardizer, cfg["train"], Augment.from_config(cfg.get("augment")),
                epochs=int(self.hpo["epochs"]), patience=int(self.hpo.get("patience", self.hpo["epochs"])),
                max_windows_per_epoch=self.hpo.get("max_windows_per_epoch"), seed=int(cfg["seed"]),
                trial=trial, log_prefix=f"[trial {trial.number}] ",
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            trial.set_user_attr("rejected", "CUDA out of memory")
            raise optuna.TrialPruned("CUDA OOM") from exc
        finally:
            trial.set_user_attr("seconds", time.perf_counter() - t0)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("history", [round(h["val_macro_f1"], 5) for h in history])
        return best_f1


def run_hpo(config: dict[str, Any], n_trials: int | None = None, timeout_h: float | None = None) -> dict[str, Any]:
    """Run (or resume) the study and write ``best_params.yaml`` plus a report."""
    import warnings

    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    hpo = config["hpo"]
    out = hpo_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    n_trials = int(n_trials if n_trials is not None else hpo["n_trials"])
    timeout_h = timeout_h if timeout_h is not None else hpo.get("timeout_h")

    dev = DevData(config)
    standardizer = Standardizer.fit(dev.signals["train"])  # training participants only
    storage = f"sqlite:///{out / 'study.db'}"
    study = optuna.create_study(
        study_name=str(hpo.get("study_name", "emgnet")),
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"]), multivariate=True,
                                           n_startup_trials=int(hpo.get("startup_trials", 8))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=int(hpo.get("pruner_startup_trials", 5)),
                                           n_warmup_steps=int(hpo.get("pruner_warmup_epochs", 3))),
    )
    logger.info("Study %s: %d existing trials; running %d more (timeout %s h) on %s", study.study_name,
                len(study.trials), n_trials, timeout_h, storage)
    study.optimize(Objective(config, dev, standardizer), n_trials=n_trials,
                   timeout=None if timeout_h is None else float(timeout_h) * 3600, gc_after_trial=True)
    GPUWindows.clear_cache()
    return write_report(config, study)


def write_report(config: dict[str, Any], study: Any) -> dict[str, Any]:
    import optuna

    out = hpo_dir(config)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError("No completed HPO trial; increase n_trials/epochs or check the logs")
    best = study.best_trial
    result = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_val_macro_f1": best.value,
        "n_trials": len(study.trials),
        "n_complete": len(done),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "params": best.params,
        "hpo_settings": {k: config["hpo"][k] for k in ("epochs", "max_windows_per_epoch") if k in config["hpo"]},
    }
    best_params_path(config).write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    try:
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    except Exception as exc:  # pandas missing etc.; the SQLite study still has everything
        logger.warning("could not export trials.csv: %s", exc)
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:  # needs >= 2 completed trials with varying params
        logger.info("parameter importance unavailable: %s", exc)
        importance = {}

    top = sorted(done, key=lambda t: -t.value)[:15]
    keys = list(config["hpo"]["search_space"])
    lines = [
        "# EMGNet hyper-parameter search (validation participants)",
        "",
        f"Trials: {result['n_trials']} (complete {result['n_complete']}, pruned/rejected {result['n_pruned']}). "
        f"Each trial: {config['hpo']['epochs']} epochs x {config['hpo'].get('max_windows_per_epoch')} windows.",
        f"Best: trial {best.number}, val macro F1 **{best.value:.4f}**",
        "",
        "| trial | val macro F1 | best epoch | CPU ms | params | " + " | ".join(k.split(".")[-1] for k in keys) + " |",
        "|" + "---|" * (5 + len(keys)),
    ]
    for t in top:
        vals = " | ".join(_fmt(t.params.get(k)) for k in keys)
        ua = t.user_attrs
        lines.append(f"| {t.number} | {t.value:.4f} | {ua.get('best_epoch', '-')} | "
                     f"{ua.get('cpu_latency_ms', float('nan')):.2f} | {ua.get('n_parameters', '-')} | {vals} |")
    if importance:
        lines += ["", "Parameter importance (fANOVA):", ""]
        lines += [f"- {k}: {v:.3f}" for k, v in importance.items()]
    (out / "hpo_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "best_trial.json").write_text(json.dumps({**result, "user_attrs": best.user_attrs}, indent=1),
                                         encoding="utf-8")
    logger.info("Best trial %d: val macro F1 %.4f, params %s", best.number, best.value, best.params)
    return result


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)
