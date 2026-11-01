import numpy as np
import pytest

from myo_emg.data.parse_dataset import PreparedPartition
from myo_emg.data.windowing import (
    WindowingSpec,
    build_window_index,
    gather_windows,
    recording_windows,
    sliding_windows,
    window_length_samples,
)


def test_window_length_samples():
    assert window_length_samples(200, 200) == 40
    assert window_length_samples(100, 200) == 20
    assert window_length_samples(250, 200) == 50
    with pytest.raises(ValueError):
        window_length_samples(203, 200)
    with pytest.raises(ValueError):
        window_length_samples(10, 200)


def test_rest_recording_windows():
    spec = WindowingSpec(window_length=40, stride=10, rest_stride=40)
    s, y = recording_windows(1000, 0, -1, -1, spec)
    assert s[0] == 0 and s[-1] + 40 <= 1000 and np.all(np.diff(s) == 40)
    assert (y == 0).all() and s.size == 25


def test_gesture_windows_respect_overlap_and_drop_transitions():
    spec = WindowingSpec(window_length=40, stride=10, rest_stride=40, gesture_min_overlap=0.75)
    s, y = recording_windows(1000, 3, 300, 500, spec)
    assert (y == 3).all()
    overlap = (np.minimum(s + 40, 500) - np.maximum(s, 300)) / 40
    assert (overlap >= 0.75).all()
    assert s.min() == 290 and s.max() == 470  # 290: 30/40 inside; 470: 30/40 inside


def test_rest_from_gesture_recordings_option():
    spec = WindowingSpec(40, 10, 40, 0.75, rest_from_gesture_recordings=True)
    s, y = recording_windows(1000, 2, 300, 500, spec)
    rest = s[y == 0]
    assert rest.size > 0
    assert ((rest + 40 <= 300) | (rest >= 500)).all()  # no overlap with the gesture
    assert np.all(np.diff(s) > 0)


def test_short_recording_yields_no_windows():
    spec = WindowingSpec(40, 10, 40)
    s, y = recording_windows(30, 0, -1, -1, spec)
    assert s.size == 0 and y.size == 0


def test_build_index_and_gather_preserve_recording_grouping():
    lengths = np.array([100, 120], np.int32)
    signal = np.arange(220 * 8, dtype=np.int64).reshape(220, 8) % 127
    prep = PreparedPartition(
        signal=signal.astype(np.int8),
        rec_offset=np.array([0, 100]),
        rec_length=lengths,
        rec_label=np.array([0, 1], np.int16),
        gt_start=np.array([-1, 20], np.int32),
        gt_end=np.array([-1, 100], np.int32),
        rec_participant=np.array([0, 0], np.int32),
        rec_key=np.array(["a", "b"]),
        participants=np.array(["p"]),
    )
    spec = WindowingSpec(20, 10, 20)
    idx = build_window_index(prep, spec)
    for start, rec in zip(idx.starts, idx.rec_index):
        lo = prep.rec_offset[rec]
        assert lo <= start and start + 20 <= lo + prep.rec_length[rec]  # never crosses recordings
    w = gather_windows(prep.signal, idx.starts, 20)
    assert w.shape == (len(idx), 20, 8)
    np.testing.assert_array_equal(w[3], prep.signal[idx.starts[3] : idx.starts[3] + 20])
    assert set(idx.labels[idx.rec_index == 1]) == {1}


def test_sliding_windows_shape():
    x = np.zeros((100, 8))
    assert sliding_windows(x, 40, 10).shape == (7, 40, 8)
    assert sliding_windows(x[:10], 40, 10).shape == (0, 40, 8)
