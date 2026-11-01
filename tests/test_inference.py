import numpy as np
import pytest

from conftest import CLASSES, tiny_pipeline
from myo_emg.data.windowing import sliding_windows
from myo_emg.inference.predictor import (
    UNCERTAIN_LABEL,
    StreamingPredictor,
    load_pipeline,
    save_bundle,
)
from myo_emg.inference.smoothing import (
    EMASmoother,
    MajorityVoteSmoother,
    MovingAverageSmoother,
    Smoother,
    build_smoother,
)


@pytest.fixture
def cnn_pipeline():
    return tiny_pipeline()


@pytest.fixture
def gru_pipeline():
    return tiny_pipeline(temporal_head="gru", td_features=False)


@pytest.mark.parametrize("which", ["cnn_pipeline", "gru_pipeline"])
def test_save_reload_gives_identical_predictions(which, request, tmp_path):
    pipe = request.getfixturevalue(which)
    save_bundle(tmp_path / "b", pipe.meta, pipe.model, pipe.standardizer)
    loaded = load_pipeline(tmp_path / "b")
    w = np.random.default_rng(1).normal(0, 6, size=(64, 40, 8)).astype(np.float32)
    np.testing.assert_allclose(loaded.predict_proba(w), pipe.predict_proba(w), atol=1e-6)
    # the reloaded pipeline uses exactly the saved preprocessing configuration
    np.testing.assert_array_equal(loaded.standardizer.mean, pipe.standardizer.mean)
    np.testing.assert_array_equal(loaded.standardizer.std, pipe.standardizer.std)
    assert loaded.window_length == 40 and loaded.classes == CLASSES and loaded.input_type == pipe.input_type
    assert loaded.meta["normalization"]["type"] == "channel_zscore"


def test_missing_bundle_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="bundle.json"):
        load_pipeline(tmp_path)


def test_pipeline_validates_window_shape(cnn_pipeline):
    with pytest.raises(ValueError):
        cnn_pipeline.predict_proba(np.zeros((2, 39, 8)))
    with pytest.raises(ValueError):
        cnn_pipeline.predict_proba(np.zeros((2, 40, 7)))
    assert cnn_pipeline.predict_proba(np.zeros((40, 8))).shape == (1, 6)


def test_streaming_matches_offline_windows(cnn_pipeline):
    x = np.random.default_rng(2).normal(0, 6, size=(203, 8)).astype(np.float32)
    sp = StreamingPredictor(cnn_pipeline, threshold=0.0)
    preds = []
    for i in range(0, len(x), 2):  # Myo delivers two samples per BLE notification
        preds += sp.push(x[i : i + 2])
    offline = cnn_pipeline.predict_proba(sliding_windows(x, 40, 10))
    assert len(preds) == len(offline) == (203 - 40) // 10 + 1
    assert [p.sample_index for p in preds] == list(range(40, 204, 10))
    np.testing.assert_allclose(np.array([p.raw_scores for p in preds]), offline, atol=1e-5)
    assert sp.windows_processed == len(preds) and sp.samples_seen == 203


def test_streaming_is_independent_of_chunking(gru_pipeline):
    x = np.random.default_rng(3).normal(0, 6, size=(150, 8)).astype(np.float32)
    a = StreamingPredictor(gru_pipeline, stride=5)
    b = StreamingPredictor(gru_pipeline, stride=5)
    pa = a.push(x)
    pb = [p for s in x for p in b.push(s)]
    assert [p.sample_index for p in pa] == [p.sample_index for p in pb]
    np.testing.assert_allclose([p.raw_scores for p in pa], [p.raw_scores for p in pb], atol=1e-6)


def test_confidence_threshold_marks_uncertain(cnn_pipeline):
    x = np.random.default_rng(4).normal(0, 6, size=(40, 8))
    certain = StreamingPredictor(cnn_pipeline, threshold=0.0).push(x)[0]
    unsure = StreamingPredictor(cnn_pipeline, threshold=1.0).push(x)[0]
    assert not certain.uncertain and certain.label in CLASSES
    assert unsure.uncertain and unsure.label == UNCERTAIN_LABEL
    assert unsure.index == certain.index and abs(sum(unsure.scores.values()) - 1) < 1e-6
    with pytest.raises(ValueError):
        StreamingPredictor(cnn_pipeline, threshold=1.5)


def test_streaming_handles_invalid_samples(cnn_pipeline):
    sp = StreamingPredictor(cnn_pipeline)
    with pytest.raises(ValueError):
        sp.push(np.zeros((3, 7)))
    x = np.ones((40, 8), np.float32)
    x[10, 3] = np.nan
    preds = sp.push(x)
    assert len(preds) == 1 and sp.invalid_samples == 1 and np.isfinite(preds[0].raw_scores).all()
    assert sp.push(np.zeros((0, 8))) == []
    sp.reset()
    assert sp.samples_seen == 0 and sp.invalid_samples == 0


def test_streaming_applies_saved_filter_statefully():
    from myo_emg.data.preprocess import filter_recording

    pipe = tiny_pipeline({"type": "highpass", "cutoff_hz": 10, "order": 2})
    x = np.random.default_rng(5).normal(size=(100, 8)).astype(np.float32)
    filtered = filter_recording(x, pipe.make_filter().sos)
    preds = StreamingPredictor(pipe, threshold=0).push(x)
    np.testing.assert_allclose(preds[-1].raw_scores, pipe.predict_proba(filtered[None, 60:100])[0], atol=1e-5)


def test_smoothers():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    ema = EMASmoother(0.5)
    ema.update(a)
    np.testing.assert_allclose(ema.update(b), [0.5, 0.5, 0])
    ma = MovingAverageSmoother(2)
    ma.update(a)
    ma.update(b)
    np.testing.assert_allclose(ma.update(b), [0, 1, 0])
    mv = MajorityVoteSmoother(3)
    for p in (a, b, b):
        out = mv.update(p)
    np.testing.assert_allclose(out, [1 / 3, 2 / 3, 0])
    mv.reset()
    np.testing.assert_allclose(mv.update(a), [1, 0, 0])
    assert type(build_smoother(None)) is Smoother
    assert isinstance(build_smoother({"method": "ema", "alpha": 0.3}), EMASmoother)
    with pytest.raises(ValueError):
        build_smoother({"method": "kalman"})
    with pytest.raises(ValueError):
        EMASmoother(0)


def test_smoothing_reduces_flicker(cnn_pipeline):
    x = np.random.default_rng(6).normal(0, 6, size=(400, 8))
    raw = StreamingPredictor(cnn_pipeline, threshold=0).push(x)
    smooth = StreamingPredictor(cnn_pipeline, threshold=0, smoother={"method": "majority", "window": 5}).push(x)
    changes = lambda ps: sum(p.index != q.index for p, q in zip(ps, ps[1:]))  # noqa: E731
    assert changes(smooth) <= changes(raw)
