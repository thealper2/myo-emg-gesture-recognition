"""Parse the raw JSON dataset once into compact ``.npz`` caches and create the participant split."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from myo_emg.data.parse_dataset import (
    ParsedUser,
    ParseSettings,
    PreparedPartition,
    build_prepared,
    list_user_files,
    parse_user_file,
)
from myo_emg.data.splits import load_split, make_participant_split, save_split, split_hash
from myo_emg.utils.config import artifacts_dir, processed_dir

logger = logging.getLogger(__name__)

DEV_FILE = "epn612_dev.npz"
TEST_FILE = "epn612_test.npz"


def split_path(config: dict[str, Any]) -> Path:
    return artifacts_dir(config) / "splits" / "participant_split.json"


def _parse_partition(
    root: Path, partition: str, settings: ParseSettings, workers: int, max_users: int | None
) -> tuple[list[ParsedUser], list[dict[str, str]]]:
    files, problems = list_user_files(root / partition)
    files = files[:max_users] if max_users else files
    logger.info("Parsing %s: %d user files with %d workers", partition, len(files), workers)
    users: list[ParsedUser] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        fn = partial(parse_user_file, partition=partition, settings=settings)
        for i, user in enumerate(pool.map(fn, files, chunksize=2), 1):
            users.append(user)
            if i % 50 == 0:
                logger.info("  %s: %d/%d", partition, i, len(files))
    excluded = [{"participant": p["path"], "key": "*", "reason": p["reason"]} for p in problems]
    excluded += [e for u in users for e in u.excluded]
    return users, excluded


def find_duplicates(dev_users: list[ParsedUser], test_users: list[ParsedUser]) -> tuple[list[list[str]], dict[str, str]]:
    """Detect participants that share byte-identical EMG recordings.

    Returns ``(dev_groups, test_exclude)``: development participants sharing recordings must stay in
    the same split partition; test participants sharing recordings with a development participant
    (leakage) or with an earlier test participant (double counting) are excluded from the test set.
    """
    owners: dict[bytes, set[str]] = {}
    for u in [*dev_users, *test_users]:
        for r in u.recordings:
            owners.setdefault(hashlib.blake2b(r.emg.tobytes(), digest_size=16).digest(), set()).add(u.participant)
    dev_names = {u.participant for u in dev_users}
    test_order = [u.participant for u in test_users]
    links: dict[str, set[str]] = {}
    for parts in owners.values():
        if len(parts) > 1:
            for p in parts:
                links.setdefault(p, set()).update(parts - {p})

    dev_groups: list[list[str]] = []
    seen: set[str] = set()
    for p in sorted(links):
        if p in dev_names and p not in seen:
            group = {q for q in links[p] | {p} if q in dev_names}
            if len(group) > 1:
                dev_groups.append(sorted(group))
            seen |= group
    test_exclude: dict[str, str] = {}
    for i, p in enumerate(test_order):
        others = links.get(p, set())
        dev_dupes = sorted(others & dev_names)
        earlier = sorted(q for q in others if q in test_order[:i] and q not in test_exclude)
        if dev_dupes:
            test_exclude[p] = f"shares identical recordings with development participant(s) {dev_dupes}"
        elif earlier:
            test_exclude[p] = f"shares identical recordings with test participant(s) {earlier}"
    return dev_groups, test_exclude


def prepare_dataset(config: dict[str, Any], workers: int = 8, max_users: int | None = None) -> dict[str, Any]:
    """Create ``epn612_dev.npz`` / ``epn612_test.npz`` and the participant split file.

    The original dataset is only read, never modified.
    """
    ds = config["dataset"]
    root = Path(ds["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    settings = ParseSettings.from_config(config)
    out = processed_dir(config)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"root": str(root), "max_users": max_users}
    excluded_all: list[dict[str, str]] = []
    prepared: dict[str, PreparedPartition] = {}
    parsed: dict[str, list[ParsedUser]] = {}
    for name, partition, fname in (("dev", ds["train_dir"], DEV_FILE), ("test", ds["test_dir"], TEST_FILE)):
        users, excluded = _parse_partition(root, partition, settings, workers, max_users)
        excluded_all += excluded
        prep = build_prepared(users)
        prep.save(out / fname)
        prepared[name] = prep
        parsed[name] = users
        summary[name] = {
            "partition": partition,
            "file": str(out / fname),
            "n_participants": int(prep.participants.size),
            "n_recordings": prep.n_recordings,
            "n_samples": int(prep.signal.shape[0]),
            "n_excluded": len(excluded),
        }
        logger.info("%s: %s", name, summary[name])

    dev_groups, test_exclude = find_duplicates(parsed["dev"], parsed["test"])
    del parsed
    for p, reason in test_exclude.items():
        logger.warning("Excluding test participant %s: %s", p, reason)
        excluded_all.append({"participant": p, "key": "*", "reason": f"excluded from test split: {reason}"})
    if dev_groups:
        logger.warning("Development participants kept together (shared recordings): %s", dev_groups)
    split = make_participant_split(
        prepared["dev"].participants.tolist(),
        prepared["test"].participants.tolist(),
        float(config["split"]["val_fraction"]),
        int(config["split"]["seed"]),
        dev_groups=dev_groups,
        test_exclude=test_exclude,
    )
    path = split_path(config)
    if path.is_file() and split_hash(load_split(path)) != split_hash(split):
        logger.warning("Existing split at %s differs (different users/seed); overwriting", path)
    save_split(split, path)
    summary["split"] = {k: len(split[k]) for k in ("train", "val", "test")} | {"file": str(path)}
    (out / "excluded_recordings.json").write_text(json.dumps(excluded_all, indent=1), encoding="utf-8")
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Split: %s", summary["split"])
    return summary
