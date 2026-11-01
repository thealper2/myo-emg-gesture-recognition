import json

import numpy as np
import pytest

from conftest import CLASSES, synth_sample, write_user
from myo_emg.data.parse_dataset import (
    ParseSettings,
    PreparedPartition,
    build_prepared,
    emg_dict_to_array,
    list_user_files,
    parse_sample,
    parse_user_file,
)
from myo_emg.data.validate import EMGValidationError, ground_truth_bounds, validate_emg_array

SETTINGS = ParseSettings(classes=CLASSES)


def test_parse_valid_user(tmp_path):
    path = write_user(tmp_path / "trainingJSON", "user7", np.random.default_rng(1), reps=2)
    user = parse_user_file(path, "trainingJSON", SETTINGS)
    assert user.participant == "trainingJSON/user7"
    assert len(user.recordings) == 24 and not user.excluded and not user.warnings
    labels = sorted({r.label for r in user.recordings})
    assert labels == list(range(6))
    for r in user.recordings:
        assert r.emg.dtype == np.int8 and r.emg.shape == (1000, 8)
        assert (r.gt_start >= 0) == (r.label != 0)


def test_unlabeled_test_samples_are_excluded(tmp_path):
    path = write_user(tmp_path / "testingJSON", "user1", np.random.default_rng(2), test_unlabeled=True)
    user = parse_user_file(path, "testingJSON", SETTINGS)
    assert len(user.recordings) == 12
    assert len(user.excluded) == 12
    assert all(e["key"].startswith("testingSamples/") and "unlabeled" in e["reason"] for e in user.excluded)


def test_ground_truth_bounds_match_one_based_index():
    rng = np.random.default_rng(3)
    s = synth_sample(rng, 2)
    label, emg, gs, ge, warns = parse_sample(s, SETTINGS)
    assert label == 2 and not warns
    assert (gs + 1, ge) == tuple(s["groundTruthIndex"])
    assert np.asarray(s["groundTruth"])[gs:ge].all()


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda s: s["emg"].pop("ch8"), "missing EMG channels"),
        (lambda s: s["emg"]["ch3"].pop(), "different lengths"),
        (lambda s: s["emg"]["ch1"].__setitem__(5, None), "non-finite"),
        (lambda s: s["emg"]["ch1"].__setitem__(5, 500), "outside"),
        (lambda s: s["emg"]["ch1"].__setitem__(5, "x"), "non-numeric"),
        (lambda s: s.pop("groundTruth"), "without groundTruth"),
        (lambda s: s.__setitem__("gestureName", "thumbsUp"), "unknown gestureName"),
        (lambda s: s.pop("gestureName"), "unlabeled"),
    ],
)
def test_invalid_samples_raise(mutate, msg):
    s = synth_sample(np.random.default_rng(4), 1)
    mutate(s)
    with pytest.raises(EMGValidationError, match=msg):
        parse_sample(s, SETTINGS)


def test_too_short_recording_is_rejected():
    s = synth_sample(np.random.default_rng(5), 0, n=400)
    with pytest.raises(EMGValidationError, match="length"):
        parse_sample(s, SETTINGS)


def test_ground_truth_index_mismatch_is_a_warning():
    s = synth_sample(np.random.default_rng(6), 3)
    s["groundTruthIndex"] = [1, 2]
    *_, warns = parse_sample(s, SETTINGS)
    assert warns and "disagrees" in warns[0]


def test_non_contiguous_ground_truth_rejected():
    mask = np.zeros(100, int)
    mask[10:20] = 1
    mask[30:40] = 1
    with pytest.raises(EMGValidationError, match="contiguous"):
        ground_truth_bounds(mask, 100)


def test_unreadable_json_file(tmp_path):
    d = tmp_path / "trainingJSON" / "user9"
    d.mkdir(parents=True)
    (d / "user9.json").write_text("{not json", encoding="utf-8")
    user = parse_user_file(d / "user9.json", "trainingJSON", SETTINGS)
    assert not user.recordings and "unreadable JSON" in user.excluded[0]["reason"]


def test_list_user_files_reports_missing_json(tmp_path):
    write_user(tmp_path, "user2", np.random.default_rng(7), reps=1)
    (tmp_path / "user10").mkdir()
    files, problems = list_user_files(tmp_path)
    assert [f.name for f in files] == ["user2.json"]
    assert problems and "user10" in problems[0]["path"]


def test_eight_channel_validation():
    with pytest.raises(EMGValidationError, match="Expected 8 channels"):
        validate_emg_array(np.zeros((10, 7)))
    with pytest.raises(EMGValidationError, match="2-D"):
        validate_emg_array(np.zeros(10))
    with pytest.raises(EMGValidationError, match="channels"):
        emg_dict_to_array({f"ch{i}": [0] for i in range(1, 8)}, 8)


def test_prepared_partition_roundtrip_and_subset(tmp_path):
    users = [
        parse_user_file(write_user(tmp_path / "trainingJSON", f"user{i}", np.random.default_rng(i), reps=1),
                        "trainingJSON", SETTINGS)
        for i in (1, 2, 3)
    ]
    prep = build_prepared(users)
    assert prep.n_recordings == 36 and prep.signal.dtype == np.int8
    assert np.array_equal(prep.recording(13), users[1].recordings[1].emg)
    prep.save(tmp_path / "p.npz")
    loaded = PreparedPartition.load(tmp_path / "p.npz")
    assert np.array_equal(loaded.signal, prep.signal)
    sub = loaded.subset(["trainingJSON/user3"])
    assert sub.participants.tolist() == ["trainingJSON/user3"]
    assert sub.n_recordings == 12
    assert np.array_equal(sub.recording(0), users[2].recordings[0].emg)
    assert set(sub.participant_of()) == {"trainingJSON/user3"}


def test_label_mapping_matches_dataset_metadata(tmp_path):
    path = write_user(tmp_path / "trainingJSON", "user1", np.random.default_rng(8), reps=1)
    mapping = json.loads(path.read_text())["generalInfo"]["myoPredictionLabel"]
    assert sorted(mapping, key=mapping.get) == CLASSES
    user = parse_user_file(path, "trainingJSON", ParseSettings(classes=list(reversed(CLASSES))))
    assert any("myoPredictionLabel order" in w for w in user.warnings)
