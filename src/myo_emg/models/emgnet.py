"""EMGNet: residual dilated temporal CNN for 8-channel EMG windows.

Architecture (input ``(B, L, 8)`` z-scored raw EMG, any window length L)::

    stem Conv1d(8 -> C, k) + BN + GELU
    n_blocks x ResidualBlock: [Conv1d(dilation d) -> BN -> GELU -> Dropout -> Conv1d(dilation d) -> BN
                               -> Squeeze-Excitation] + skip, dilation d = growth**i
    temporal head: attention pooling + mean pooling        (temporal_head = "attention")
                   or BiGRU -> attention + mean pooling     (temporal_head = "gru")
    optional time-domain branch: log MAV / RMS / waveform length per channel, computed in-network
    classifier: Dropout -> Linear(head_dim) -> GELU -> Dropout -> Linear(n_classes)

The window only contains past samples, so the non-causal convolutions inside it never use future data.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    """Channel attention: global average over time -> bottleneck MLP -> sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        return x * self.fc(x.mean(dim=2)).unsqueeze(2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, se: bool) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            SqueezeExcite(channels) if se else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class AttentionPool(nn.Module):
    """Softmax-weighted average over time with a learned scoring layer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L) -> (B, C)
        w = torch.softmax(self.score(x).float(), dim=2).to(x.dtype)
        return (x * w).sum(dim=2)


class TimeDomainFeatures(nn.Module):
    """log(MAV), log(RMS), log(mean |diff|) per channel of the normalised input window."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(3 * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, 3C)
        x = x.float()
        eps = 1e-4
        mav = x.abs().mean(dim=1)
        rms = x.pow(2).mean(dim=1).sqrt()
        wl = (x[:, 1:] - x[:, :-1]).abs().mean(dim=1)
        return self.norm(torch.log(torch.cat([mav, rms, wl], dim=1) + eps))


class EMGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 6,
        channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 5,
        dilation_growth: int = 2,
        se: bool = True,
        temporal_head: str = "attention",
        gru_hidden: int = 64,
        td_features: bool = True,
        dropout: float = 0.3,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if temporal_head not in ("attention", "gru"):
            raise ValueError(f"temporal_head must be 'attention' or 'gru', got {temporal_head!r}")
        self.n_channels = n_channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, dilation_growth ** i, dropout, se) for i in range(n_blocks)]
        )
        self.temporal_head = temporal_head
        if temporal_head == "gru":
            self.gru = nn.GRU(channels, gru_hidden, batch_first=True, bidirectional=True)
            pooled = 2 * gru_hidden
        else:
            self.gru = None
            pooled = channels
        self.pool = AttentionPool(pooled)
        self.td = TimeDomainFeatures(n_channels) if td_features else None
        in_dim = 2 * pooled + (3 * n_channels if td_features else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.n_channels:
            raise ValueError(f"EMGNet expects (B, L, {self.n_channels}), got {tuple(x.shape)}")
        h = self.blocks(self.stem(x.transpose(1, 2)))  # (B, C, L)
        if self.gru is not None:
            with torch.autocast(device_type=h.device.type, enabled=False):
                h, _ = self.gru(h.float().transpose(1, 2))  # (B, L, 2H)
            h = h.transpose(1, 2)
        parts = [self.pool(h), h.mean(dim=2)]
        if self.td is not None:
            parts.append(self.td(x).to(h.dtype))
        return self.classifier(torch.cat(parts, dim=1))


ARCH_KEYS = ("channels", "n_blocks", "kernel_size", "dilation_growth", "se", "temporal_head", "gru_hidden",
             "td_features", "dropout", "head_dim")


def architecture_from_config(model_cfg: dict[str, Any], n_channels: int, n_classes: int) -> dict[str, Any]:
    """Normalise the ``model`` config section into the architecture dict stored in ``bundle.json``."""
    arch = {"type": "emgnet", "n_channels": n_channels, "n_classes": n_classes}
    casts = {"channels": int, "n_blocks": int, "kernel_size": int, "dilation_growth": int, "se": bool,
             "temporal_head": str, "gru_hidden": int, "td_features": bool, "dropout": float, "head_dim": int}
    for k in ARCH_KEYS:
        arch[k] = casts[k](model_cfg[k])
    return arch
