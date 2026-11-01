"""EMGNet training loop shared by the hyper-parameter search and the final training.

* windows are gathered on the GPU from the shared int8 signal (no per-epoch copies);
* augmentation (training only): per-window gain, per-channel gain jitter, +-1 channel rotation
  (simulates an armband worn rotated by one electrode) and Gaussian noise;
* loss: cross-entropy with optional class weights (none | balanced | sqrt) and label smoothing;
* AdamW, linear warm-up + cosine decay per step, bf16 autocast on CUDA, gradient clipping;
* early stopping on validation macro F1 with best-checkpoint restore; optional Optuna pruning.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from myo_emg.data.preprocess import Standardizer
from myo_emg.models import build_neural_model
from myo_emg.training.data import WindowSet

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


class GPUWindows:
    """Window starts/labels of a ``WindowSet`` plus its signal, resident on the training device."""

    _signal_cache: dict[tuple[int, str], torch.Tensor] = {}

    def __init__(self, wset: WindowSet, device: torch.device) -> None:
        key = (id(wset.signal), str(device))
        if key not in self._signal_cache:  # the signal is shared by all window lengths
            self._signal_cache[key] = torch.from_numpy(wset.signal).to(device)
        self.signal = self._signal_cache[key]
        self.starts = torch.from_numpy(wset.index.starts).to(device)
        self.labels = torch.from_numpy(wset.labels).to(device)
        self.offsets = torch.arange(wset.window_length, device=device)
        self.device = device

    @classmethod
    def clear_cache(cls) -> None:
        cls._signal_cache.clear()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def raw(self, idx: torch.Tensor) -> torch.Tensor:
        """Unnormalised windows ``(B, L, 8)`` as float32."""
        return self.signal[self.starts[idx, None] + self.offsets].float()


@dataclass
class Augment:
    gain_range: float = 0.0  # per-window gain sampled log-uniformly in [1/(1+r), 1+r]
    channel_gain_jitter: float = 0.0  # std of log per-channel gain
    rotate_p: float = 0.0  # probability of rotating channels by +-1 electrode
    noise_std: float = 0.0  # Gaussian noise added after normalisation (z units)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> Augment:
        cfg = cfg or {}
        return cls(**{k: float(cfg.get(k, 0.0)) for k in ("gain_range", "channel_gain_jitter", "rotate_p",
                                                           "noise_std")})

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Augment unnormalised windows ``(B, L, C)``."""
        b, _, c = x.shape
        if self.rotate_p > 0:
            rot = torch.rand(b, device=x.device) < self.rotate_p
            if rot.any():
                direction = torch.where(torch.rand(b, device=x.device) < 0.5, 1, -1)
                for shift in (1, -1):
                    sel = rot & (direction == shift)
                    if sel.any():
                        x[sel] = torch.roll(x[sel], shifts=shift, dims=2)
        if self.gain_range > 0:
            lim = math.log1p(self.gain_range)
            x = x * torch.empty(b, 1, 1, device=x.device).uniform_(-lim, lim).exp()
        if self.channel_gain_jitter > 0:
            x = x * (torch.randn(b, 1, c, device=x.device) * self.channel_gain_jitter).exp()
        return x

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x


def class_weights(labels: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    """``balanced``: n / (k * count_c); ``sqrt``: its square root; ``none``: no weighting."""
    if mode in (None, "none", "null"):
        return None
    counts = np.maximum(np.bincount(labels, minlength=n_classes).astype(np.float64), 1)
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(f"Unknown class_weighting {mode!r}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def predict_logits(model: nn.Module, data: GPUWindows, mean: torch.Tensor, std: torch.Tensor,
                   batch_size: int, amp_dtype: torch.dtype | None) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(data), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(data)), device=data.device)
        x = (data.raw(idx) - mean) / std
        with torch.autocast(data.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out.append(model(x).float())
    return torch.cat(out)


def train_emgnet(
    arch: dict[str, Any],
    train: GPUWindows,
    val: GPUWindows,
    standardizer: Standardizer,
    train_cfg: dict[str, Any],
    augment: Augment,
    epochs: int,
    patience: int,
    max_windows_per_epoch: int | None,
    seed: int,
    trial: Any | None = None,
    log_prefix: str = "",
) -> tuple[nn.Module, list[dict[str, Any]], int, float]:
    """Train one EMGNet; returns ``(best_model, history, best_epoch, best_val_macro_f1)``.

    With ``trial`` (Optuna) the validation macro F1 is reported every epoch and the trial is pruned
    when it falls behind earlier trials.
    """
    device = train.device
    set_seed(seed, bool(train_cfg.get("deterministic", False)))
    n_classes = int(arch["n_classes"])
    model = build_neural_model(arch).to(device)
    mean = torch.from_numpy(standardizer.mean).to(device)
    std = torch.from_numpy(standardizer.std).to(device)

    weights = class_weights(train.labels.cpu().numpy(), n_classes, str(train_cfg.get("class_weighting", "none")))
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                                  label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                            weight_decay=float(train_cfg["weight_decay"]))
    bs = int(train_cfg["batch_size"])
    per_epoch = min(len(train), int(max_windows_per_epoch)) if max_windows_per_epoch else len(train)
    steps_per_epoch = math.ceil(per_epoch / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(epochs * steps_per_epoch, int(float(train_cfg.get("warmup_epochs", 1)) * steps_per_epoch)))
    use_bf16 = device.type == "cuda" and train_cfg.get("amp", "bf16") == "bf16" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else None
    clip = float(train_cfg.get("grad_clip", 1.0))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    y_val = val.labels.cpu().numpy()
    labels = list(range(n_classes))

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(train), generator=gen)[:per_epoch].to(device)
        total = torch.zeros((), device=device)
        for i in range(0, per_epoch, bs):
            idx = perm[i : i + bs]
            x = augment.normalized((augment.raw(train.raw(idx)) - mean) / std)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                loss = loss_fn(model(x).float(), train.labels[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {i // bs}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            total += loss.detach() * idx.numel()

        logits = predict_logits(model, val, mean, std, 8192, amp_dtype)
        val_loss = float(nn.functional.cross_entropy(logits, val.labels))
        pred = logits.argmax(1).cpu().numpy()
        f1 = float(f1_score(y_val, pred, average="macro", labels=labels, zero_division=0))
        rec = {"epoch": epoch, "train_loss": float(total) / per_epoch, "val_loss": val_loss, "val_macro_f1": f1,
               "val_accuracy": float((pred == y_val).mean()), "lr": opt.param_groups[0]["lr"],
               "seconds": time.perf_counter() - t0}
        history.append(rec)
        logger.info("%sepoch %3d  loss %.4f  val_loss %.4f  val_F1 %.4f  val_acc %.4f  lr %.2e  (%.1fs)",
                    log_prefix, epoch, rec["train_loss"], val_loss, f1, rec["val_accuracy"], rec["lr"], rec["seconds"])
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        if f1 > best_f1 + float(train_cfg.get("min_delta", 1e-4)):
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if trial is not None:
            import optuna

            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (val F1 {f1:.4f})")
        if bad >= patience:
            logger.info("%searly stopping after epoch %d (best epoch %d, val F1 %.4f)", log_prefix, epoch,
                        best_epoch, best_f1)
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, best_f1


def cpu_latency_ms(arch: dict[str, Any], window_length: int, n_runs: int = 50) -> float:
    """Mean single-window CPU forward time of an untrained model (1 thread, as in live inference)."""
    model = build_neural_model(arch).eval()
    x = torch.randn(1, window_length, int(arch["n_channels"]))
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for _ in range(5):
                model(x)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(x)
        return (time.perf_counter() - t0) * 1000 / n_runs
    finally:
        torch.set_num_threads(prev)
