"""Self-contained inference pipeline: model bundle save/load, batch and streaming prediction.

A *bundle* is a directory with ``bundle.json`` (all preprocessing parameters, class order, metadata,
metrics) and the PyTorch weights ``model.pt``. ``load_pipeline`` restores everything needed for
inference; nothing is re-fitted.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from myo_emg.data.preprocess import CausalFilter, Standardizer, design_filter, sanitize_samples
from myo_emg.inference.smoothing import Smoother, build_smoother
from myo_emg.models import build_neural_model

BUNDLE_FORMAT_VERSION = 2  # v2: single EMGNet model on raw windows (v1 bundles are no longer loadable)
UNCERTAIN_LABEL = "uncertain"


def software_versions() -> dict[str, str]:
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextmanager
def full_fp32(device: torch.device) -> Iterator[None]:
    """Disable TF32 on CUDA so GPU scores match the CPU deployment (cuDNN conv/GRU use TF32 by default,
    which shifts the scores by ~1e-3); training keeps TF32 for speed."""
    if device.type != "cuda":
        yield
        return
    prev = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


class GesturePipeline:
    """Normalisation + network for raw windows of shape ``(N, window_length, 8)``.

    Windows must already have passed the optional causal stream filter (``self.make_filter()``);
    ``StreamingPredictor`` takes care of that for live data.
    """

    def __init__(
        self,
        meta: dict[str, Any],
        model: nn.Module,
        standardizer: Standardizer,
        device: str | torch.device = "cpu",
    ) -> None:
        self.meta = meta
        self.model = model
        self.standardizer = standardizer
        self.input_type: str = meta.get("input_type", "raw")
        self.classes: list[str] = list(meta["classes"])
        self.window_length: int = int(meta["window_length"])
        self.stride: int = int(meta["stride"])
        self.sampling_rate_hz: float = float(meta["sampling_rate_hz"])
        self.n_channels: int = len(meta["channel_order"])
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device).eval()

    @property
    def name(self) -> str:
        return str(self.meta["name"])

    def make_filter(self) -> CausalFilter:
        """A fresh stream filter matching the training configuration (identity if disabled)."""
        return CausalFilter(design_filter(self.meta.get("filter", {}), self.sampling_rate_hz), self.n_channels)

    def validate_windows(self, windows: np.ndarray) -> np.ndarray:
        w = np.asarray(windows)
        if w.ndim == 2:
            w = w[None]
        if w.ndim != 3 or w.shape[1:] != (self.window_length, self.n_channels):
            raise ValueError(f"Expected windows (N, {self.window_length}, {self.n_channels}), got {w.shape}")
        return w

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Apply the per-channel z-score fitted on the training participants."""
        return self.standardizer.transform(self.validate_windows(windows))

    def predict_proba_from_input(self, x: np.ndarray) -> np.ndarray:
        """Softmax scores for already normalised windows, columns in ``self.classes`` order."""
        with torch.no_grad(), full_fp32(self.device):
            logits = self.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
            return torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float64)

    def predict_proba(self, windows: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Softmax scores. These are *not* calibrated probabilities."""
        w = self.validate_windows(windows)
        parts = [
            self.predict_proba_from_input(self.transform(w[i : i + batch_size]))
            for i in range(0, w.shape[0], batch_size)
        ]
        return np.concatenate(parts) if parts else np.empty((0, len(self.classes)))

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self.predict_proba(windows).argmax(axis=1)


def save_bundle(out_dir: Path, meta: dict[str, Any], model: nn.Module, standardizer: Standardizer) -> Path:
    """Write ``bundle.json`` and ``model.pt`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("format_version", BUNDLE_FORMAT_VERSION)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta.setdefault("software", software_versions())
    meta["input_type"] = "raw"
    meta["normalization"] = {"type": "channel_zscore", "fitted_on": "training participants only",
                             **standardizer.to_dict()}
    weights = out_dir / "model.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
    meta["weights_file"] = weights.name
    meta["model_size_bytes"] = weights.stat().st_size
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out_dir


def load_bundle_meta(bundle_dir: Path) -> dict[str, Any]:
    path = Path(bundle_dir) / "bundle.json"
    if not path.is_file():
        raise FileNotFoundError(f"No model bundle at {bundle_dir} (missing bundle.json). Train and evaluate first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_pipeline(bundle_dir: str | Path, device: str = "cpu") -> GesturePipeline:
    """Restore the complete inference pipeline (normalisation + model) from a bundle directory."""
    bundle_dir = Path(bundle_dir)
    meta = load_bundle_meta(bundle_dir)
    if meta.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Bundle at {bundle_dir} has format {meta.get('format_version')}, expected {BUNDLE_FORMAT_VERSION}. "
            "It was produced by an older version of the project; retrain with `make hpo train evaluate`."
        )
    weights = bundle_dir / meta["weights_file"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights missing: {weights}")
    model = build_neural_model(meta["architecture"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return GesturePipeline(meta, model, Standardizer.from_dict(meta["normalization"]), resolve_device(device))


@dataclass
class Prediction:
    """One streaming prediction. ``scores`` are smoothed model scores, not calibrated probabilities."""

    label: str
    index: int
    confidence: float
    uncertain: bool
    scores: dict[str, float]
    raw_scores: list[float]
    sample_index: int
    timestamp: float
    preprocess_ms: float
    model_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


class StreamingPredictor:
    """Rolling-buffer classifier for a continuous 8-channel EMG stream.

    Every ``stride`` new samples (once ``window_length`` samples are available) the most recent
    window is classified. Only past samples are used. Predictions whose (smoothed) top score is below
    ``threshold`` are reported as ``uncertain``.
    """

    def __init__(
        self,
        pipeline: GesturePipeline,
        stride: int | None = None,
        threshold: float = 0.5,
        smoother: Smoother | dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.pipeline = pipeline
        self.stride = int(stride or pipeline.stride)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.threshold = threshold
        self.smoother = smoother if isinstance(smoother, Smoother) else build_smoother(smoother)
        self.reset()

    def reset(self) -> None:
        L, C = self.pipeline.window_length, self.pipeline.n_channels
        self._history = np.zeros((0, C), dtype=np.float32)
        self._filter = self.pipeline.make_filter()
        self._last_valid: np.ndarray | None = None
        self.samples_seen = 0
        self.invalid_samples = 0
        self.windows_processed = 0
        self.smoother.reset()
        self._max_history = L - 1

    def push(self, samples: np.ndarray, timestamp: float | None = None) -> list[Prediction]:
        """Add ``(n, 8)`` samples (a single ``(8,)`` sample is accepted) and return new predictions."""
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        if x.ndim != 2 or x.shape[1] != self.pipeline.n_channels:
            raise ValueError(f"Expected samples (n, {self.pipeline.n_channels}), got {x.shape}")
        if x.shape[0] == 0:
            return []
        t_in = time.time() if timestamp is None else timestamp
        x, n_bad = sanitize_samples(x, self._last_valid)
        self.invalid_samples += n_bad
        self._last_valid = x[-1]
        x = self._filter(x)

        L = self.pipeline.window_length
        ext = np.concatenate([self._history, x])
        first_new = self.samples_seen  # global index of x[0]
        self.samples_seen += x.shape[0]
        # global sample counts n at which a window ends: n >= L and (n - L) % stride == 0
        ends_global = [
            n for n in range(first_new + 1, self.samples_seen + 1) if n >= L and (n - L) % self.stride == 0
        ]
        self._history = ext[-self._max_history :] if self._max_history else ext[:0]
        if not ends_global:
            return []
        offset = first_new - (ext.shape[0] - x.shape[0])  # global index of ext[0]
        windows = np.stack([ext[n - offset - L : n - offset] for n in ends_global])

        t0 = time.perf_counter()
        model_in = self.pipeline.transform(windows)
        t1 = time.perf_counter()
        proba = self.pipeline.predict_proba_from_input(model_in)
        t2 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000 / len(ends_global)
        model_ms = (t2 - t1) * 1000 / len(ends_global)

        preds = []
        for n, p in zip(ends_global, proba):
            s = self.smoother.update(p)
            idx = int(np.argmax(s))
            conf = float(s[idx])
            uncertain = conf < self.threshold
            preds.append(
                Prediction(
                    label=UNCERTAIN_LABEL if uncertain else self.pipeline.classes[idx],
                    index=idx,
                    confidence=conf,
                    uncertain=uncertain,
                    scores={c: float(v) for c, v in zip(self.pipeline.classes, s)},
                    raw_scores=p.tolist(),
                    sample_index=n,
                    timestamp=t_in,
                    preprocess_ms=pre_ms,
                    model_ms=model_ms,
                )
            )
        self.windows_processed += len(preds)
        return preds
