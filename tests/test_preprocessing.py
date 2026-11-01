import numpy as np
import pytest

from myo_emg.data.preprocess import CausalFilter, Standardizer, design_filter, filter_recording, sanitize_samples


def test_standardizer_fit_transform_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(5000, 8))
    s = Standardizer.fit(x, chunk=777)
    z = s.transform(x)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-3)
    s2 = Standardizer.from_dict(s.to_dict())
    np.testing.assert_array_equal(s2.transform(x), z)
    assert s.transform(x.reshape(50, 100, 8)).shape == (50, 100, 8)


def test_standardizer_constant_dimension_is_safe():
    x = np.zeros((100, 3))
    x[:, 1] = np.arange(100)
    s = Standardizer.fit(x)
    z = s.transform(x)
    assert np.isfinite(z).all() and s.std[0] == 1.0


def test_standardizer_rejects_bad_input():
    with pytest.raises(ValueError):
        Standardizer.fit(np.zeros(10))
    s = Standardizer.fit(np.random.default_rng(1).normal(size=(10, 8)))
    with pytest.raises(ValueError):
        s.transform(np.zeros((4, 7)))


def test_streaming_filter_equals_offline_filter():
    x = np.random.default_rng(2).normal(size=(997, 8)).astype(np.float32)
    sos = design_filter({"type": "highpass", "cutoff_hz": 10, "order": 4}, 200)
    offline = filter_recording(x, sos)
    f = CausalFilter(sos)
    chunks = [f(x[i : i + 13]) for i in range(0, len(x), 13)]
    np.testing.assert_allclose(np.concatenate(chunks), offline, atol=1e-5)


def test_filter_is_causal():
    sos = design_filter({"type": "bandpass", "cutoff_hz": [10, 90], "order": 2}, 200)
    x = np.zeros((200, 8), np.float32)
    x[150] = 1.0
    y = filter_recording(x, sos)
    assert np.allclose(y[:150], 0)


def test_no_filter_and_invalid_cutoff():
    assert design_filter({"type": "none"}, 200) is None
    x = np.ones((5, 8), np.float32)
    assert CausalFilter(None)(x) is x
    with pytest.raises(ValueError):
        design_filter({"type": "highpass", "cutoff_hz": 150}, 200)
    with pytest.raises(ValueError):
        design_filter({"type": "comb", "cutoff_hz": 10}, 200)


def test_sanitize_replaces_missing_values():
    x = np.arange(24, dtype=np.float32).reshape(3, 8)
    x[1, 2] = np.nan
    x[2, 5] = np.inf
    clean, n_bad = sanitize_samples(x)
    assert n_bad == 2 and np.isfinite(clean).all()
    assert clean[1, 2] == x[0, 2] and clean[2, 5] == x[1, 5]
    first = np.full((1, 8), np.nan, np.float32)
    clean, _ = sanitize_samples(first, last_valid=np.full(8, 7.0))
    assert (clean == 7).all()
