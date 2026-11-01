import numpy as np
import pytest

from conftest import write_user
from myo_emg.data.parse_dataset import ParseSettings, parse_user_file
from myo_emg.data.prepare import find_duplicates, split_path
from myo_emg.data.preprocess import Standardizer
from myo_emg.data.splits import assert_disjoint, load_split, make_participant_split, split_hash

DEV = [f"trainingJSON/user{i}" for i in range(1, 21)]
TEST = [f"testingJSON/user{i}" for i in range(1, 11)]


def test_split_is_disjoint_complete_and_deterministic():
    a = make_participant_split(DEV, TEST, 0.15, seed=1)
    b = make_participant_split(list(reversed(DEV)), TEST, 0.15, seed=1)
    assert a["train"] == b["train"] and a["val"] == b["val"]
    assert sorted(a["train"] + a["val"]) == sorted(DEV)
    assert len(a["val"]) == 3 and a["test"] == sorted(TEST)
    assert_disjoint(a)
    c = make_participant_split(DEV, TEST, 0.15, seed=2)
    assert c["val"] != a["val"]


def test_overlap_is_detected():
    split = make_participant_split(DEV, TEST, 0.2, seed=0)
    split["test"].append(split["train"][0])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(split)


def test_groups_stay_together_and_test_exclusion():
    groups = [["trainingJSON/user1", "trainingJSON/user2"]]
    for seed in range(10):
        s = make_participant_split(DEV, TEST, 0.3, seed, dev_groups=groups, test_exclude={TEST[0]: "dup"})
        in_val = {p in s["val"] for p in groups[0]}
        assert len(in_val) == 1
        assert TEST[0] not in s["test"] and s["test_excluded"] == {TEST[0]: "dup"}


def test_find_duplicates_flags_leaking_test_participant(tmp_path):
    settings = ParseSettings(classes=["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
    rng = np.random.default_rng(0)
    dev = [parse_user_file(write_user(tmp_path / "trainingJSON", f"user{i}", rng, reps=1), "trainingJSON", settings)
           for i in (1, 2)]
    test = [parse_user_file(write_user(tmp_path / "testingJSON", f"user{i}", rng, reps=1), "testingJSON", settings)
            for i in (1, 2)]
    test[1].recordings[0].emg = dev[0].recordings[3].emg.copy()  # simulated duplicated recording
    groups, excluded = find_duplicates(dev, test)
    assert groups == []
    assert list(excluded) == ["testingJSON/user2"]


def test_prepared_split_has_no_overlap_and_scaler_uses_train_only(prepared_config):
    from myo_emg.training.data import load_dev_sets

    split = load_split(split_path(prepared_config))
    assert_disjoint(split)
    assert len(split["sha256_16"]) == 16 and split["sha256_16"] == split_hash(split)
    train, val, _ = load_dev_sets(prepared_config)
    assert set(train.participants) == set(split["train"])
    assert set(val.participants) == set(split["val"])
    assert not set(train.participants) & set(val.participants)
    # normalisation fitted on training participants differs from one fitted on all dev data
    s_train = Standardizer.fit(train.signal)
    s_all = Standardizer.fit(np.concatenate([train.signal, val.signal]))
    assert not np.allclose(s_train.std, s_all.std)
