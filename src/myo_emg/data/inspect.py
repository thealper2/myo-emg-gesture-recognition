"""Dataset audit: parses every user file and reports structure, counts and exclusions."""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from myo_emg.data.parse_dataset import ParseSettings, list_user_files, parse_user_file

logger = logging.getLogger(__name__)

FINGERPRINT_FIELDS = (
    "age",
    "gender",
    "handedness",
    "distanceFromElbowToMyoInCm",
    "distanceFromElbowToUlnaInCm",
    "armPerimeterInCm",
    "date",
)


def summarize_user(path: Path, partition: str, settings: ParseSettings) -> dict[str, Any]:
    """Parse one user file and return a small, picklable summary (no raw arrays)."""
    user = parse_user_file(path, partition, settings)
    recs = []
    channel_abs = defaultdict(lambda: np.zeros(settings.n_channels))
    channel_n = Counter()
    vmin, vmax = 0, 0
    for r in user.recordings:
        gt_len = r.gt_end - r.gt_start if r.gt_start >= 0 else 0
        recs.append((r.key.split("/")[0], r.label, int(r.emg.shape[0]), int(gt_len)))
        channel_abs[r.label] += np.abs(r.emg.astype(np.float64)).sum(axis=0)
        channel_n[r.label] += r.emg.shape[0]
        vmin, vmax = min(vmin, int(r.emg.min())), max(vmax, int(r.emg.max()))
    other_files = [p.name for p in path.parent.iterdir() if p.is_file() and p != path]
    return {
        "participant": user.participant,
        "partition": partition,
        "path": user.path,
        "sampling_rate_hz": user.general_info.get("samplingFrequencyInHertz"),
        "device": user.general_info.get("deviceModel"),
        "recording_time_s": user.general_info.get("recordingTimeInSeconds"),
        "user_info": {k: user.user_info.get(k) for k in ("name", *FINGERPRINT_FIELDS, "ethnicGroup")},
        "n_sync_samples": user.n_sync_samples,
        "recordings": recs,
        "excluded": user.excluded,
        "warnings": user.warnings,
        "channel_abs_sum": {int(k): v.tolist() for k, v in channel_abs.items()},
        "channel_n": dict(channel_n),
        "value_range": [vmin, vmax],
        "other_files": other_files,
    }


def _describe(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(statistics.fmean(values), 2),
        "median": float(statistics.median(values)),
    }


def aggregate(summaries: list[dict[str, Any]], settings: ParseSettings, file_problems: list[dict]) -> dict[str, Any]:
    """Combine per-user summaries into the dataset-level audit dictionary."""
    classes = settings.classes
    audit: dict[str, Any] = {"partitions": {}, "classes": classes}
    excluded = [*file_problems]
    fingerprints: dict[tuple, list[str]] = defaultdict(list)

    for partition in sorted({s["partition"] for s in summaries}):
        users = [s for s in summaries if s["partition"] == partition]
        by_section: dict[str, Counter] = defaultdict(Counter)
        lengths: list[int] = []
        gt_by_class: dict[str, list[int]] = defaultdict(list)
        abs_sum = defaultdict(lambda: np.zeros(settings.n_channels))
        abs_n: Counter = Counter()
        reasons: Counter = Counter()
        n_excluded_by_section: Counter = Counter()
        for u in users:
            for section, label, n, gt_len in u["recordings"]:
                by_section[section][classes[label]] += 1
                lengths.append(n)
                if gt_len:
                    gt_by_class[classes[label]].append(gt_len)
            for k, v in u["channel_abs_sum"].items():
                abs_sum[int(k)] += np.asarray(v)
            for k, v in u["channel_n"].items():
                abs_n[int(k)] += v
            for e in u["excluded"]:
                excluded.append(e)
                reasons[e["reason"].split(":")[0]] += 1
                n_excluded_by_section[e["key"].split("/")[0]] += 1
            fp = tuple(u["user_info"].get(k) for k in FINGERPRINT_FIELDS)
            fingerprints[fp].append(u["participant"])

        audit["partitions"][partition] = {
            "n_users": len(users),
            "n_users_with_valid_recordings": sum(1 for u in users if u["recordings"]),
            "n_valid_recordings": sum(len(u["recordings"]) for u in users),
            "n_excluded_recordings": int(sum(n_excluded_by_section.values())),
            "excluded_by_section": dict(n_excluded_by_section),
            "exclusion_reasons": dict(reasons),
            "class_counts_by_section": {k: dict(v) for k, v in by_section.items()},
            "class_counts_total": dict(sum(by_section.values(), Counter())),
            "recording_length_samples": _describe(lengths),
            "gesture_segment_length_samples": {c: _describe(v) for c, v in gt_by_class.items()},
            "sampling_rate_values": dict(Counter(str(u["sampling_rate_hz"]) for u in users)),
            "device_values": dict(Counter(str(u["device"]) for u in users)),
            "recording_time_s_values": dict(Counter(str(u["recording_time_s"]) for u in users)),
            "emg_value_range": [
                min(u["value_range"][0] for u in users),
                max(u["value_range"][1] for u in users),
            ],
            "mean_abs_emg_per_class_and_channel": {
                classes[k]: (abs_sum[k] / max(abs_n[k], 1)).round(3).tolist() for k in sorted(abs_sum)
            },
            "sync_samples_per_user": dict(Counter(u["n_sync_samples"] for u in users)),
            "gender": dict(Counter(str(u["user_info"].get("gender")) for u in users)),
            "handedness": dict(Counter(str(u["user_info"].get("handedness")) for u in users)),
            "age": _describe([u["user_info"]["age"] for u in users if isinstance(u["user_info"].get("age"), int)]),
            "users_with_warnings": sum(1 for u in users if u["warnings"]),
            "warning_examples": [w for u in users for w in u["warnings"]][:20],
            "other_files": dict(Counter(f.rsplit(".", 1)[-1] for u in users for f in u["other_files"])),
        }

    dupes = {" | ".join(map(str, fp)): ps for fp, ps in fingerprints.items() if len(ps) > 1}
    audit["possible_duplicate_participants"] = {
        "description": "users whose userInfo (excluding name) is identical; informational only",
        "groups": dupes,
        "cross_partition_groups": {
            k: v for k, v in dupes.items() if len({p.split("/")[0] for p in v}) > 1
        },
    }
    audit["n_users_total"] = len(summaries)
    audit["n_excluded_total"] = len(excluded)
    return {"audit": audit, "excluded": excluded}


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    """Render a human-readable summary of the audit."""
    lines = [
        "# EMG-EPN-612 dataset audit",
        "",
        f"Generated: {audit['generated_at']}  ",
        f"Root: `{audit['root']}`",
        "",
        f"Total users: {audit['n_users_total']}; excluded recordings/files: {audit['n_excluded_total']}",
        "",
    ]
    classes = audit["classes"]
    for name, p in audit["partitions"].items():
        lines += [
            f"## {name}",
            "",
            f"- Users: {p['n_users']} (with valid recordings: {p['n_users_with_valid_recordings']})",
            f"- Valid labelled recordings: {p['n_valid_recordings']}",
            f"- Excluded recordings: {p['n_excluded_recordings']} {p['excluded_by_section']}",
            f"- Exclusion reasons: {p['exclusion_reasons']}",
            f"- Sampling rate metadata: {p['sampling_rate_values']}; device: {p['device_values']}",
            f"- Recording length (samples): {p['recording_length_samples']}",
            f"- EMG value range: {p['emg_value_range']}",
            f"- Synchronization samples per user: {p['sync_samples_per_user']}",
            f"- Gender: {p['gender']}; handedness: {p['handedness']}; age: {p['age']}",
            f"- Users with warnings: {p['users_with_warnings']}",
            f"- Other files in user folders: {p['other_files']}",
            "",
            "| section | " + " | ".join(classes) + " |",
            "|---|" + "---|" * len(classes),
        ]
        for section, counts in p["class_counts_by_section"].items():
            lines.append(f"| {section} | " + " | ".join(str(counts.get(c, 0)) for c in classes) + " |")
        lines += ["", "Gesture segment length (samples, from groundTruth):", ""]
        for c, d in p["gesture_segment_length_samples"].items():
            lines.append(f"- {c}: {d}")
        lines += ["", "Mean |EMG| per class and channel (ch1..ch8):", ""]
        for c, v in p["mean_abs_emg_per_class_and_channel"].items():
            lines.append(f"- {c}: {v}")
        if p["warning_examples"]:
            lines += ["", "Warning examples:", ""] + [f"- {w}" for w in p["warning_examples"]]
        lines.append("")
    cross = audit["possible_duplicate_participants"]["cross_partition_groups"]
    lines += [
        "## Possible duplicate participants",
        "",
        f"Groups with identical userInfo (excluding name): {len(audit['possible_duplicate_participants']['groups'])}; "
        f"spanning both partitions: {len(cross)}",
        "",
    ]
    for k, v in list(cross.items())[:20]:
        lines.append(f"- {v} ({k})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config: dict[str, Any], out_dir: Path, workers: int = 8, max_users: int | None = None) -> dict[str, Any]:
    """Audit both partitions of the dataset and write JSON/Markdown reports to ``out_dir``."""
    ds = config["dataset"]
    root = Path(ds["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    settings = ParseSettings.from_config(config)
    summaries: list[dict[str, Any]] = []
    file_problems: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for partition in (ds["train_dir"], ds["test_dir"]):
            files, problems = list_user_files(root / partition)
            file_problems += [{"participant": p["path"], "key": "*", "reason": p["reason"]} for p in problems]
            files = files[:max_users] if max_users else files
            logger.info("Auditing %s: %d user files", partition, len(files))
            fn = partial(summarize_user, partition=partition, settings=settings)
            for i, s in enumerate(pool.map(fn, files, chunksize=2), 1):
                summaries.append(s)
                if i % 50 == 0:
                    logger.info("  %s: %d/%d", partition, i, len(files))

    result = aggregate(summaries, settings, file_problems)
    audit = result["audit"]
    audit["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit["root"] = str(root)
    audit["max_users"] = max_users
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "excluded_recordings.json").write_text(json.dumps(result["excluded"], indent=1), encoding="utf-8")
    write_markdown(audit, out_dir / "dataset_audit.md")
    logger.info("Audit written to %s", out_dir)
    return audit
