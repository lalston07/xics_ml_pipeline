"""
CNN for regressing spline-knot targets from a detector image.

--- PHYSICS-INFORMED DESIGN NOTE (read this first) ---------------------------
The XICS detector image has TWO physically different axes:

  * ROWS  (the long/tall axis, y)  -> SPATIAL / line-of-sight direction.
        Different rows look at different radial positions in the plasma, so
        this axis carries the *profile shape* information (Ti vs rho).
  * COLS  (the short axis, x)      -> SPECTRAL / dispersion direction.
        The line's Doppler *broadening* across columns encodes temperature.

The native binned image is tall and narrow (~1475 x 195), resized to 512 x 128
before the CNN. Because we are predicting a RADIAL profile, the spatial (row)
resolution is precious -- it maps to radial resolution. So this model pools the
SPECTRAL (column) axis but PRESERVES the SPATIAL (row) axis, instead of the
usual symmetric 2x2 pooling that shrinks both equally.

This file is written out block-by-block (no loop) on purpose, so you can edit
each block's convolutions and pooling by hand while experimenting. The older
loop-based, symmetric-pooling version is kept COMMENTED OUT at the bottom for
reference.
-----------------------------------------------------------------------------
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# torch core + its neural-network building blocks.
import torch
import torch.nn as nn

# The model hyperparameters (channels, blocks, dropout).
from .config import ModelConfig


# One repeatable "block": two conv layers then a pooling downsample.
# `pool` is a (row_pool, col_pool) pair so you can reduce the two axes by
# DIFFERENT amounts. For example:
#     pool=(1, 2)  -> keep rows (spatial), halve cols (spectral)   [default]
#     pool=(2, 2)  -> halve both (the classic symmetric behavior)
#     pool=(2, 1)  -> halve rows only (rarely what you want here)
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (1, 2)):
        # Always call the parent constructor first in an nn.Module.
        super().__init__()
        # nn.Sequential runs these layers in order.
        self.block = nn.Sequential(
            # 3x3 convolution; padding=1 keeps rows,cols the same size. bias=False
            # because the BatchNorm right after has its own shift.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # Normalizes activations per channel -> faster, more stable training.
            nn.BatchNorm2d(out_ch),
            # ReLU nonlinearity (inplace saves a little memory).
            nn.ReLU(inplace=True),
            # Second conv, now in_ch == out_ch.
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Downsample. MaxPool2d((r, c)) reduces rows by r and cols by c.
            # With (1, 2): rows (spatial) are untouched, cols (spectral) halve.
            nn.MaxPool2d(kernel_size=pool),
        )

    # Defines what happens when data flows through the block.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# The full network: image in, knot vector out.
#
# The four blocks are written out EXPLICITLY (not built in a loop) so you can
# edit each one independently while experimenting -- change its channel counts,
# add/remove convolutions, or change how much it pools each axis.
class XicsCNN(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_targets: int):
        super().__init__()
        # Guard: the output size must be positive.
        if n_targets <= 0:
            raise ValueError("n_targets must be positive")

        # Number of input image channels (1 = grayscale detector image).
        c_in = model_cfg.in_channels

        # ------------------------------------------------------------------
        # FEATURE EXTRACTOR -- four blocks, each written out by hand.
        #
        # Channel counts double each block (32 -> 64 -> 128 -> 256), matching
        # the previous model so results stay comparable. Edit these freely.
        #
        # Pooling: each block halves the SPECTRAL (column) axis and KEEPS the
        # SPATIAL (row) axis. So a 512 (spatial) x 128 (spectral) input becomes:
        #     block1 -> 512 x 64
        #     block2 -> 512 x 32
        #     block3 -> 512 x 16
        #     block4 -> 512 x 8
        # The spatial resolution (512 rows) is fully preserved for later use.
        #
        # To experiment, change the `pool=` on any block. e.g. set block3 and
        # block4 to pool=(2, 2) if you decide to start compressing spatial
        # information in the deeper layers.
        # ------------------------------------------------------------------
        self.block1 = ConvBlock(c_in, 32,  pool=(1, 2))   # 1  -> 32 channels
        self.block2 = ConvBlock(32,   64,  pool=(1, 2))   # 32 -> 64
        self.block3 = ConvBlock(64,   128, pool=(1, 2))   # 64 -> 128
        self.block4 = ConvBlock(128,  256, pool=(1, 2))   # 128 -> 256

        # Number of feature channels after the last block (edit if you change
        # the last block's output channels above).
        feat_dim = 256

        # ------------------------------------------------------------------
        # BRIDGE from the 2-D feature maps to a flat vector for the head.
        #
        # We now USE the preserved spatial resolution instead of averaging it
        # away. AdaptiveAvgPool2d((SPATIAL_BINS, 1)) averages each channel's map
        # down to SPATIAL_BINS rows x 1 col -> it summarizes the sightline axis
        # into a handful of spatial bins (keeping coarse "where along the
        # profile" information) while collapsing the remaining spectral columns.
        #
        # After flatten, the head sees feat_dim * SPATIAL_BINS numbers, so it
        # can learn how the features vary along the spatial/radial direction --
        # the physics-informed point of preserving spatial resolution.
        #
        # SPATIAL_BINS is easy to experiment with: 4 (coarser, fewer params),
        # 8 (default), or 16 (finer, larger head -> more overfitting risk).
        # ------------------------------------------------------------------
        self.spatial_bins = 8
        self.pool = nn.AdaptiveAvgPool2d((self.spatial_bins, 1))

        # The head's input is now the channels times the number of spatial bins.
        head_in = feat_dim * self.spatial_bins   # 256 * 8 = 2048

        # ------------------------------------------------------------------
        # REGRESSION HEAD -- a small MLP ending in n_targets values in [0,1].
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            # Flatten [B, C, SPATIAL_BINS, 1] -> [B, C * SPATIAL_BINS].
            nn.Flatten(),
            # Fully-connected hidden layer. First Linear now takes head_in
            # inputs (channels x spatial bins) and mixes across BOTH channels
            # and spatial position.
            nn.Linear(head_in, feat_dim),
            nn.ReLU(inplace=True),
            # Dropout randomly zeros some activations during training (regularizer).
            nn.Dropout(model_cfg.dropout),
            # Final layer produces exactly n_targets outputs.
            nn.Linear(feat_dim, n_targets),
            # Squash outputs to [0, 1] since labels are normalized to that range.
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # PREVIOUS BRIDGE/HEAD (kept for reference) -- collapsed ALL spatial
        # rows to a single value, so the head saw only `feat_dim` features and
        # the preserved spatial resolution went unused. Uncomment to revert.
        # ------------------------------------------------------------------
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Sequential(
        #     nn.Flatten(),                       # [B, C, 1, 1] -> [B, C]
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(model_cfg.dropout),
        #     nn.Linear(feat_dim, n_targets),
        #     nn.Sigmoid(),
        # )

    # Full forward pass: run the four blocks in order, then pool, then head.
    # Writing the blocks out here (instead of a Sequential) makes it easy to
    # print/inspect intermediate shapes while you experiment -- just add
    # `print(x.shape)` between the calls.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, 32,  rows, cols/2)
        x = self.block2(x)   # (B, 64,  rows, cols/4)
        x = self.block3(x)   # (B, 128, rows, cols/8)
        x = self.block4(x)   # (B, 256, rows, cols/16)
        x = self.pool(x)     # (B, 256, spatial_bins, 1)  -> keeps spatial info
        return self.head(x)  # (B, n_targets)


# ===========================================================================
# PREVIOUS VERSION (kept for reference) -- loop-built blocks with SYMMETRIC
# 2x2 pooling that shrank BOTH axes equally. This is what earlier model
# versions (v00 / v01) used. To go back to it, comment out the class above
# and uncomment this one.
# ===========================================================================
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # <-- symmetric: halves BOTH rows and cols
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)
#
#
# class XicsCNN(nn.Module):
#     def __init__(self, model_cfg: ModelConfig, n_targets: int):
#         super().__init__()
#         if n_targets <= 0:
#             raise ValueError("n_targets must be positive")
#         # channel counts, doubling each block, built in a loop
#         chans = [model_cfg.in_channels] + [
#             model_cfg.base_channels * (2 ** i) for i in range(model_cfg.n_blocks)
#         ]
#         blocks = [ConvBlock(chans[i], chans[i + 1]) for i in range(model_cfg.n_blocks)]
#         self.features = nn.Sequential(*blocks)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         feat_dim = chans[-1]
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(feat_dim, feat_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(model_cfg.dropout),
#             nn.Linear(feat_dim, n_targets),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.pool(x)
#         return self.head(x)
