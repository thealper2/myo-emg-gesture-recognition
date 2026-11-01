"""Classification metrics, latency measurement and reporting helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    rec_index: np.ndarray | None = None,
) -> dict[str, Any]:
    """Window-level metrics plus (optionally) recording-level accuracy.

    Recording-level prediction = argmax of the mean class scores over the recording's labelled
    windows, which mirrors classifying one whole gesture repetition.
    """
    labels = np.arange(len(classes))
    y_pred = proba.argmax(axis=1)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n_windows": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "min_class_recall": float(rec.min()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if rec_index is not None:
        out.update(recording_level_metrics(y_true, proba, rec_index))
    return out


def recording_level_metrics(y_true: np.ndarray, proba: np.ndarray, rec_index: np.ndarray) -> dict[str, Any]:
    """Accuracy and macro F1 after averaging window scores within each recording."""
    uniq, inv = np.unique(rec_index, return_inverse=True)
    sums = np.zeros((uniq.size, proba.shape[1]))
    np.add.at(sums, inv, proba)
    rec_true = np.zeros(uniq.size, dtype=np.int64)
    rec_true[inv] = y_true
    rec_pred = sums.argmax(axis=1)
    return {
        "n_recordings": int(uniq.size),
        "recording_accuracy": float((rec_pred == rec_true).mean()),
        "recording_macro_f1": float(f1_score(rec_true, rec_pred, average="macro", zero_division=0)),
    }


def measure_latency(fn: Callable[[], Any], n_warmup: int = 10, n_runs: int = 200) -> dict[str, float]:
    """Wall-clock latency of ``fn()`` in milliseconds (mean, median, p95)."""
    for _ in range(n_warmup):
        fn()
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_runs": n_runs,
    }


def save_confusion_matrix_png(cm: list[list[int]], classes: list[str], title: str, path) -> None:
    """Save a row-normalised confusion-matrix heatmap (matplotlib, headless)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
