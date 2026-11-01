import json

import numpy as np
import pytest
import torch

from conftest import TINY_ARCH
from myo_emg.models import build_neural_model
from myo_emg.models.emgnet import EMGNet, architecture_from_config
from myo_emg.training.trainer import Augment, class_weights, warmup_cosine


@pytest.mark.parametrize("length", [20, 40, 50, 60])
@pytest.mark.parametrize("head", ["attention", "gru"])
@pytest.mark.parametrize("td", [True, False])
def test_emgnet_forward_any_window_length(length, head, td):
    model = EMGNet(channels=16, n_blocks=3, temporal_head=head, gru_hidden=8, td_features=td, head_dim=16).eval()
    out = model(torch.randn(4, length, 8))
    assert out.shape == (4, 6) and torch.isfinite(out).all()


def test_emgnet_validates_input_and_config():
    with pytest.raises(ValueError):
        EMGNet()(torch.randn(2, 40, 7))
    with pytest.raises(ValueError):
        EMGNet(kernel_size=4)
    with pytest.raises(ValueError):
        EMGNet(temporal_head="transformer")


def test_factory_and_config_roundtrip():
    cfg = {k: v for k, v in TINY_ARCH.items() if k not in ("type", "n_channels", "n_classes")}
    arch = architecture_from_config(cfg, 8, 6)
    assert arch == TINY_ARCH
    model = build_neural_model(arch)
    assert isinstance(model, EMGNet)
    with pytest.raises(ValueError):
        build_neural_model({"type": "cnn"})


def test_augment_rotation_gain_and_noise():
    torch.manual_seed(0)
    x = torch.arange(8, dtype=torch.float32).repeat(4, 10, 1)  # (4, 10, 8), channel c has value c
    rotated = Augment(rotate_p=1.0).raw(x.clone())
    for b in range(4):  # every window is rotated by exactly one electrode
        first = rotated[b, 0]
        assert torch.equal(first, torch.roll(x[b, 0], 1)) or torch.equal(first, torch.roll(x[b, 0], -1))
    gained = Augment(gain_range=0.5).raw(torch.ones(1000, 2, 8))
    assert gained.min() >= 1 / 1.5 - 1e-5 and gained.max() <= 1.5 + 1e-5 and gained.std() > 0
    assert torch.equal(Augment().raw(x.clone()), x)  # no-op by default
    noisy = Augment(noise_std=0.1).normalized(torch.zeros(10000))
    assert abs(float(noisy.std()) - 0.1) < 0.01


def test_class_weights_modes():
    labels = np.array([0] * 30 + [1] * 10)
    assert class_weights(labels, 2, "none") is None
    bal = class_weights(labels, 2, "balanced")
    sq = class_weights(labels, 2, "sqrt")
    assert bal[1] > sq[1] > sq[0] > bal[0]  # rare class up-weighted, sqrt is milder
    assert abs(float(bal.mean()) - 1) < 1e-6
    with pytest.raises(ValueError):
        class_weights(labels, 2, "focal")


def test_warmup_cosine_schedule():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(55) == pytest.approx(0.5, abs=0.01) and f(100) == pytest.approx(0.0)


def test_hpo_train_evaluate_smoke(prepared_config):
    """Tiny end-to-end run on synthetic data: HPO (2 trials) -> final training -> one test evaluation."""
    from myo_emg.inference.predictor import load_bundle_meta, load_pipeline
    from myo_emg.training import hpo, train
    from myo_emg.training.evaluate import best_model_dir, evaluate_test

    cfg = prepared_config
    result = hpo.run_hpo(cfg, n_trials=2)
    assert result["n_trials"] == 2 and set(result["params"]) == set(cfg["hpo"]["search_space"])
    assert hpo.best_params_path(cfg).is_file() and (hpo.hpo_dir(cfg) / "hpo_report.md").is_file()
    assert hpo.run_hpo(cfg, n_trials=1)["n_trials"] == 3  # the study resumes from SQLite

    report = train.run(cfg)
    assert 0.0 <= report["val"]["macro_f1"] <= 1.0
    meta = load_bundle_meta(train.final_model_dir(cfg))
    best = hpo.load_best_params(cfg)
    assert meta["training"]["hpo_params"] == best["params"]
    assert meta["window_ms"] == best["params"]["windowing.window_ms"]
    assert meta["architecture"]["channels"] == best["params"]["model.channels"]
    assert meta["reload_check"]["argmax_agreement"] >= 0.999

    # a bundle saved without the post-training checks is refused, and `make finalize` completes it
    bundle_json = train.final_model_dir(cfg) / "bundle.json"
    bundle_json.write_text(json.dumps({k: v for k, v in meta.items() if k not in ("reload_check", "latency_cpu")}))
    with pytest.raises(RuntimeError, match="make finalize"):
        evaluate_test(cfg)
    train.refinalize(cfg)
    redone = load_bundle_meta(train.final_model_dir(cfg))
    assert redone["reload_check"]["max_abs_score_diff"] <= 1e-3 and "latency_cpu" in redone
    assert redone["created_at"] == meta["created_at"]

    metrics = evaluate_test(cfg)
    assert metrics["n_participants"] == 3 and 0 <= metrics["accuracy"] <= 1
    deployed = load_bundle_meta(best_model_dir(cfg))
    assert deployed["test_metrics"] is not None
    assert not set(deployed["split"]["test"]) & set(deployed["split"]["train"] + deployed["split"]["val"])
    with pytest.raises(RuntimeError, match="evaluated once"):
        evaluate_test(cfg)
    pipe = load_pipeline(best_model_dir(cfg))
    assert pipe.classes == cfg["dataset"]["classes"] and pipe.window_length == meta["window_length"]
    history = json.loads((best_model_dir(cfg).parents[1] / "reports" / "training_history.json").read_text())
    assert len(history) >= 1 and np.isfinite(history[0]["train_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_inference_matches_cpu_without_tf32():
    from conftest import tiny_pipeline
    from myo_emg.inference.predictor import full_fp32

    torch.backends.cudnn.allow_tf32 = True
    cpu = tiny_pipeline(channels=64, n_blocks=4, kernel_size=7, temporal_head="gru")
    gpu = tiny_pipeline(channels=64, n_blocks=4, kernel_size=7, temporal_head="gru")
    gpu.model.to("cuda")
    gpu.device = torch.device("cuda")
    x = np.random.default_rng(0).normal(0, 20, (512, cpu.window_length, 8))
    assert np.abs(gpu.predict_proba(x) - cpu.predict_proba(x)).max() < 1e-4
    assert torch.backends.cudnn.allow_tf32  # restored after inference
    with full_fp32(torch.device("cuda")):
        assert not torch.backends.cudnn.allow_tf32


def test_old_bundle_format_is_rejected(tmp_path):
    from myo_emg.inference.predictor import load_pipeline

    (tmp_path / "bundle.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="older version"):
        load_pipeline(tmp_path)


def test_cli_override_wins_over_hpo_params(tmp_path, synthetic_root):
    import yaml

    from conftest import make_config
    from myo_emg.training.hpo import best_params_path
    from myo_emg.training.train import resolve_config

    cfg = make_config(synthetic_root, tmp_path)
    path = best_params_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"best_trial": 3, "best_val_macro_f1": 0.8,
                                    "params": {"model.temporal_head": "attention", "train.lr": 0.004}}))
    resolved, best = resolve_config(cfg, True, ["model.temporal_head=gru"])
    assert resolved["model"]["temporal_head"] == "gru"  # explicit --set wins
    assert resolved["train"]["lr"] == 0.004  # other searched values are kept
    resolved, best = resolve_config(cfg, False, ["model.temporal_head=gru"])
    assert best is None
