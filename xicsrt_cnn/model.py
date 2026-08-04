"""
CNN for regressing spline-knot targets from a detector image.

A compact VGG-style convolutional encoder followed by global average pooling
and an MLP head. Output activation is a sigmoid, because all targets are
min-max normalized into [0, 1] in labels.py -- this guarantees predictions
stay in-range and makes free x-knots automatically respect their bounds.
(Ordering of free x-knots is enforced at decode time via sorting.)
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# torch core + its neural-network building blocks.
import torch
import torch.nn as nn

# The model hyperparameters (channels, blocks, dropout).
from .config import ModelConfig


# One repeatable "block": two conv layers then a 2x downsample. Stacking several
# of these shrinks the image while growing the number of feature channels.
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        # Always call the parent constructor first in an nn.Module.
        super().__init__()
        # nn.Sequential runs these layers in order.
        self.block = nn.Sequential(
            # 3x3 convolution; padding=1 keeps H,W the same. bias=False because
            # the BatchNorm right after has its own shift.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # Normalizes activations per channel -> faster, more stable training.
            nn.BatchNorm2d(out_ch),
            # ReLU nonlinearity (inplace saves a little memory).
            nn.ReLU(inplace=True),
            # Second conv, now in_ch == out_ch.
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Halve the height and width (downsample).
            nn.MaxPool2d(2),
        )

    # Defines what happens when data flows through the block.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# The full network: image in, knot vector out.
class XicsCNN(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_targets: int):
        super().__init__()
        # Guard: the output size must be positive.
        if n_targets <= 0:
            raise ValueError("n_targets must be positive")

        # Build the list of channel counts per stage. Starts at in_channels,
        # then base_channels, 2x, 4x, ... doubling each block.
        chans = [model_cfg.in_channels] + [
            model_cfg.base_channels * (2 ** i) for i in range(model_cfg.n_blocks)
        ]
        # Create one ConvBlock per consecutive channel pair (chans[i]->chans[i+1]).
        blocks = [ConvBlock(chans[i], chans[i + 1]) for i in range(model_cfg.n_blocks)]
        # Chain the blocks into the convolutional feature extractor.
        self.features = nn.Sequential(*blocks)
        # Collapse each feature map to a single number (global average pooling),
        # giving a fixed-length vector regardless of input image size.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # After pooling, the feature vector has this many entries.
        feat_dim = chans[-1]
        # The regression head: a small MLP ending in n_targets values in [0,1].
        self.head = nn.Sequential(
            # Flatten [B, C, 1, 1] -> [B, C].
            nn.Flatten(),
            # Fully-connected hidden layer.
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            # Dropout randomly zeros some activations during training (regularizer).
            nn.Dropout(model_cfg.dropout),
            # Final layer produces exactly n_targets outputs.
            nn.Linear(feat_dim, n_targets),
            # Squash outputs to [0, 1] since labels are normalized to that range.
            nn.Sigmoid(),
        )

    # Full forward pass: features -> pool -> head.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract convolutional features (shrinks H,W, grows channels).
        x = self.features(x)
        # Global average pool to a per-channel vector.
        x = self.pool(x)
        # Map to the knot-coordinate predictions.
        return self.head(x)
