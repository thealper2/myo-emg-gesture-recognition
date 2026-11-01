"""Deterministic participant-level train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_participant_split(
    dev_participants: list[str],
    test_participants: list[str],
    val_fraction: float,
    seed: int,
    dev_groups: list[list[str]] | None = None,
    test_exclude: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Split development participants into train/validation; test participants stay held out.

    ``dev_groups`` lists development participants that must land in the same partition (e.g. users
    sharing duplicated recordings). ``test_exclude`` maps test participants to the reason they are
    removed from the test set (e.g. duplicates of development users). The shuffle depends only on
    the sorted participant list and the seed, so it is reproducible.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    dev = sorted(set(dev_participants))
    if len(dev) < 2:
        raise ValueError("Need at least two development participants for a train/validation split")
    grouped = {p for g in dev_groups or [] for p in g}
    units = [sorted(g) for g in dev_groups or []] + [[p] for p in dev if p not in grouped]
    units.sort()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    n_val = min(max(1, int(round(val_fraction * len(dev)))), len(dev) - 1)
    val: list[str] = []
    train: list[str] = []
    for i in order:
        (val if len(val) < n_val else train).extend(units[i])
    if not train:
        raise ValueError("Train partition is empty")
    test_exclude = test_exclude or {}
    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "description": "train/val: random participant split of trainingJSON users; "
        "test: official testingJSON users (labelled trainingSamples)",
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(set(test_participants) - set(test_exclude)),
        "dev_groups": [sorted(g) for g in dev_groups or []],
        "test_excluded": dict(sorted(test_exclude.items())),
    }
    assert_disjoint(split)
    return split


def assert_disjoint(split: dict[str, Any]) -> None:
    """Raise ``ValueError`` if any participant appears in more than one partition."""
    names = ("train", "val", "test")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(split[a]) & set(split[b])
            if common:
                raise ValueError(f"Participants overlap between {a} and {b}: {sorted(common)[:5]}")


def split_hash(split: dict[str, Any]) -> str:
    """Short SHA-256 of the participant lists, stored with every model artifact."""
    payload = json.dumps({k: split[k] for k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_split(split: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**split, "sha256_16": split_hash(split)}, indent=1), encoding="utf-8")


def load_split(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run `make prepare-data` first.")
    split = json.loads(path.read_text(encoding="utf-8"))
    assert_disjoint(split)
    return split
