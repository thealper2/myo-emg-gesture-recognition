"""Model definition and the factory that rebuilds it from a saved architecture dict."""

from __future__ import annotations

from typing import Any

from torch import nn

from myo_emg.models.emgnet import ARCH_KEYS, EMGNet


def build_neural_model(arch: dict[str, Any]) -> nn.Module:
    """Instantiate a network from an architecture dict (as stored in ``bundle.json``)."""
    if arch.get("type") != "emgnet":
        raise ValueError(f"Unknown architecture type {arch.get('type')!r}; this version only supports 'emgnet'")
    return EMGNet(n_channels=int(arch["n_channels"]), n_classes=int(arch["n_classes"]),
                  **{k: arch[k] for k in ARCH_KEYS})
