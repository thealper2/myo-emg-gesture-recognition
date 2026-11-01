"""Parser for the EMG-EPN-612 JSON release and the compact prepared-dataset cache.

Verified file layout (see ``docs/dataset.md``)::

    <root>/trainingJSON/userN/userN.json   306 users, trainingSamples + testingSamples labelled
    <root>/testingJSON/userN/userN.json    306 users, only trainingSamples labelled

Each JSON file holds ``generalInfo``, ``userInfo``, ``synchronizationGesture``, ``trainingSamples`` and
``testingSamples``. Every sample has ``emg`` as ``{"ch1": [...], ..., "ch8": [...]}`` (int8 values,
200 Hz); gesture samples additionally carry a per-sample ``groundTruth`` mask and a 1-based inclusive
``groundTruthIndex``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from myo_emg.data.validate import EMGValidationError, ground_truth_bounds, validate_emg_array

logger = logging.getLogger(__name__)

SAMPLE_SECTIONS = ("trainingSamples", "testingSamples")


@dataclass
class Recording:
    """One labelled 5-second EMG recording."""

    participant: str
    key: str
    label: int
    emg: np.ndarray  # (n_samples, 8) int8
    gt_start: int = -1  # 0-based gesture onset, -1 for noGesture
    gt_end: int = -1  # exclusive gesture end, -1 for noGesture


@dataclass
class ParsedUser:
    """Everything extracted from a single user JSON file."""

    participant: str
    path: str
    general_info: dict[str, Any]
    user_info: dict[str, Any]
    recordings: list[Recording] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_sync_samples: int = 0


@dataclass
class ParseSettings:
    """Validation parameters taken from ``configs/dataset.yaml``."""

    classes: list[str]
    n_channels: int = 8
    sampling_rate_hz: int = 200
    emg_min: int = -128
    emg_max: int = 127
    min_samples: int = 600
    max_samples: int = 1200

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParseSettings:
        ds = config["dataset"]
        return cls(
            classes=list(ds["classes"]),
            n_channels=int(ds["n_channels"]),
            sampling_rate_hz=int(ds["sampling_rate_hz"]),
            emg_min=int(ds["emg_min"]),
            emg_max=int(ds["emg_max"]),
            min_samples=int(ds["min_samples"]),
            max_samples=int(ds["max_samples"]),
        )


def _natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 1 << 30, path.name)


def list_user_files(partition_dir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Return the JSON file of every user directory plus a list of directories without one."""
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Dataset partition directory not found: {partition_dir}")
    files: list[Path] = []
    problems: list[dict[str, str]] = []
    for user_dir in sorted((p for p in partition_dir.iterdir() if p.is_dir()), key=_natural_key):
        candidates = sorted(user_dir.glob("*.json"))
        preferred = user_dir / f"{user_dir.name}.json"
        if preferred.is_file():
            files.append(preferred)
        elif len(candidates) == 1:
            files.append(candidates[0])
        else:
            problems.append({"path": str(user_dir), "reason": f"expected one JSON file, found {len(candidates)}"})
    return files, problems


def emg_dict_to_array(emg: Any, n_channels: int) -> np.ndarray:
    """Convert ``{"ch1": [...], ...}`` into a float64 ``(n_samples, n_channels)`` array."""
    if not isinstance(emg, dict):
        raise EMGValidationError(f"emg must be a dict of channels, got {type(emg).__name__}")
    names = [f"ch{i}" for i in range(1, n_channels + 1)]
    missing = [n for n in names if n not in emg]
    if missing:
        raise EMGValidationError(f"missing EMG channels {missing}")
    lengths = {len(emg[n]) for n in names}
    if len(lengths) != 1:
        raise EMGValidationError(f"EMG channels have different lengths {sorted(lengths)}")
    try:
        arr = np.array([emg[n] for n in names], dtype=np.float64).T
    except (TypeError, ValueError) as exc:
        raise EMGValidationError(f"non-numeric EMG values: {exc}") from exc
    return arr


def parse_sample(sample: dict[str, Any], settings: ParseSettings) -> tuple[int, np.ndarray, int, int, list[str]]:
    """Validate one sample and return ``(label, emg_int8, gt_start, gt_end, warnings)``.

    Raises ``EMGValidationError`` when the sample must be excluded.
    """
    name = sample.get("gestureName")
    if name is None:
        raise EMGValidationError("unlabeled: no gestureName")
    if name not in settings.classes:
        raise EMGValidationError(f"unknown gestureName {name!r}")
    label = settings.classes.index(name)

    emg = emg_dict_to_array(sample.get("emg"), settings.n_channels)
    validate_emg_array(emg, settings.n_channels, (settings.emg_min, settings.emg_max))
    n = emg.shape[0]
    if not settings.min_samples <= n <= settings.max_samples:
        raise EMGValidationError(f"length {n} outside [{settings.min_samples}, {settings.max_samples}]")
    if not np.array_equal(emg, np.round(emg)):
        raise EMGValidationError("EMG values are not integers")

    warnings: list[str] = []
    gt_start = gt_end = -1
    if name != settings.classes[0]:
        if "groundTruth" not in sample:
            raise EMGValidationError("gesture recording without groundTruth")
        gt_start, gt_end = ground_truth_bounds(sample["groundTruth"], n)
        idx = sample.get("groundTruthIndex")
        if isinstance(idx, list) and len(idx) == 2 and (int(idx[0]) - 1, int(idx[1])) != (gt_start, gt_end):
            warnings.append(f"groundTruthIndex {idx} disagrees with groundTruth mask [{gt_start}, {gt_end})")
    elif "groundTruth" in sample and np.asarray(sample["groundTruth"]).any():
        warnings.append("noGesture recording has a non-empty groundTruth mask (ignored)")
    return label, emg.astype(np.int8), gt_start, gt_end, warnings


def parse_user_file(path: Path, partition: str, settings: ParseSettings) -> ParsedUser:
    """Parse and validate every sample of one user file.

    Invalid samples are not raised; they are listed in ``ParsedUser.excluded`` with the reason.
    """
    participant = f"{partition}/{path.parent.name}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        user = ParsedUser(participant, str(path), {}, {})
        user.excluded.append({"participant": participant, "key": "*", "reason": f"unreadable JSON: {exc}"})
        return user

    general = data.get("generalInfo", {}) or {}
    user = ParsedUser(participant, str(path), general, data.get("userInfo", {}) or {})

    fs = general.get("samplingFrequencyInHertz")
    if fs != settings.sampling_rate_hz:
        user.warnings.append(f"samplingFrequencyInHertz={fs}, expected {settings.sampling_rate_hz}")
    mapping = general.get("myoPredictionLabel")
    if isinstance(mapping, dict) and sorted(mapping, key=mapping.get) != settings.classes:
        user.warnings.append(f"myoPredictionLabel order {mapping} differs from configured classes")
    name = user.user_info.get("name")
    if name is not None and name != path.parent.name:
        user.warnings.append(f"userInfo.name={name!r} differs from directory {path.parent.name!r}")
    user.n_sync_samples = len((data.get("synchronizationGesture") or {}).get("samples", {}) or {})

    for section in SAMPLE_SECTIONS:
        samples = data.get(section)
        if samples is None:
            user.warnings.append(f"section {section} missing")
            continue
        for idx_key, sample in samples.items():
            key = f"{section}/{idx_key}"
            try:
                label, emg, gs, ge, warns = parse_sample(sample, settings)
            except EMGValidationError as exc:
                user.excluded.append({"participant": participant, "key": key, "reason": str(exc)})
                continue
            user.warnings.extend(f"{key}: {w}" for w in warns)
            user.recordings.append(Recording(participant, key, label, emg, gs, ge))
    return user


@dataclass
class PreparedPartition:
    """All recordings of one partition concatenated into a single compact array.

    ``signal[rec_offset[i] : rec_offset[i] + rec_length[i]]`` is recording ``i``.
    """

    signal: np.ndarray  # (total_samples, 8) int8
    rec_offset: np.ndarray  # (R,) int64
    rec_length: np.ndarray  # (R,) int32
    rec_label: np.ndarray  # (R,) int16
    gt_start: np.ndarray  # (R,) int32, -1 for noGesture
    gt_end: np.ndarray  # (R,) int32, -1 for noGesture
    rec_participant: np.ndarray  # (R,) int32 index into participants
    rec_key: np.ndarray  # (R,) str
    participants: np.ndarray  # (P,) str

    @property
    def n_recordings(self) -> int:
        return int(self.rec_offset.shape[0])

    def recording(self, i: int) -> np.ndarray:
        start = int(self.rec_offset[i])
        return self.signal[start : start + int(self.rec_length[i])]

    def participant_of(self) -> np.ndarray:
        """Participant ID string for every recording."""
        return self.participants[self.rec_participant]

    def subset(self, participants: list[str] | set[str]) -> PreparedPartition:
        """Return only the recordings belonging to ``participants`` (signal is re-packed)."""
        wanted = set(participants)
        keep_p = np.array([p in wanted for p in self.participants])
        keep = keep_p[self.rec_participant]
        idx = np.flatnonzero(keep)
        lengths = self.rec_length[idx]
        offsets = np.zeros(idx.size, dtype=np.int64)
        if idx.size:
            offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
        signal = np.concatenate([self.recording(i) for i in idx]) if idx.size else self.signal[:0]
        new_index = -np.ones(self.participants.size, dtype=np.int32)
        new_index[keep_p] = np.arange(int(keep_p.sum()), dtype=np.int32)
        return PreparedPartition(
            signal=signal,
            rec_offset=offsets,
            rec_length=lengths,
            rec_label=self.rec_label[idx],
            gt_start=self.gt_start[idx],
            gt_end=self.gt_end[idx],
            rec_participant=new_index[self.rec_participant[idx]],
            rec_key=self.rec_key[idx],
            participants=self.participants[keep_p],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> PreparedPartition:
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset not found: {path}. Run `make prepare-data` first.")
        with np.load(path, allow_pickle=False) as npz:
            return cls(**{k: npz[k] for k in cls.__dataclass_fields__})


def build_prepared(users: list[ParsedUser]) -> PreparedPartition:
    """Concatenate the valid recordings of the given users into a ``PreparedPartition``."""
    participants = [u.participant for u in users if u.recordings]
    pindex = {p: i for i, p in enumerate(participants)}
    recs = [r for u in users for r in u.recordings]
    if not recs:
        raise ValueError("No valid recordings to prepare")
    lengths = np.array([r.emg.shape[0] for r in recs], dtype=np.int32)
    offsets = np.zeros(len(recs), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    return PreparedPartition(
        signal=np.concatenate([r.emg for r in recs]).astype(np.int8),
        rec_offset=offsets,
        rec_length=lengths,
        rec_label=np.array([r.label for r in recs], dtype=np.int16),
        gt_start=np.array([r.gt_start for r in recs], dtype=np.int32),
        gt_end=np.array([r.gt_end for r in recs], dtype=np.int32),
        rec_participant=np.array([pindex[r.participant] for r in recs], dtype=np.int32),
        rec_key=np.array([r.key for r in recs]),
        participants=np.array(participants),
    )
