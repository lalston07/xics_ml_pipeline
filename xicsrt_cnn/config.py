"""Central configuration for the XICS CNN training pipeline (ion-temp only)."""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# `dataclass` builds the config containers; `field` provides per-instance
# defaults for the nested config objects (see note below).
from dataclasses import dataclass, field
# `Path` handles filesystem paths in an OS-independent way.
from pathlib import Path

# Detector-image settings live in image.py; xarray variable names in schema.py.
from .image import ImageConfig
from .schema import XarraySchema

# --- work out repo-relative default paths ----------------------------------
# Absolute path to THIS file (config.py).
_THIS = Path(__file__).resolve()
# Go up two folders: cnn/ -> xics_ml_pipeline/ (the pipeline root).
_PIPELINE_ROOT = _THIS.parent.parent  # xics_ml_pipeline/
# Default place to look for training data.
_DATA_ROOT = _PIPELINE_ROOT / "nn_training_data"


# Everything about WHERE the data is and HOW to turn it into model inputs.
@dataclass
class DataConfig:
    # Path to an xarray file (.nc) OR a directory of per-sample .nc files.
    # Point this at the mentor's delivered dataset.
    xarray_path: Path = _DATA_ROOT / "training.nc"
    # If xarray_path is a directory, this glob picks the per-sample files.
    file_glob: str = "*.nc"
    # The variable/coordinate names inside the xarray (edit schema.py, not here).
    # `field(default_factory=...)` gives each DataConfig its own fresh schema
    # object instead of sharing one mutable instance across all configs.
    schema: XarraySchema = field(default_factory=XarraySchema)
    # Detector-image binning/normalization settings (its own fresh instance).
    image: ImageConfig = field(default_factory=ImageConfig)
    # Resize every image to this (rows, cols) before the CNN. None = keep the
    # native binned size.
    resize_to: tuple[int, int] | None = (512, 128)
    # Fraction of samples held out for validation.
    val_fraction: float = 0.15
    # Random seed so the train/val split is reproducible.
    seed: int = 0


# Everything about the CNN's shape/capacity.
@dataclass
class ModelConfig:
    # Number of input image channels (1 = grayscale detector image).
    in_channels: int = 1
    # Number of filters in the first conv block (doubles each block).
    base_channels: int = 32
    # How many downsampling conv blocks to stack.
    n_blocks: int = 4
    # Dropout probability in the head (regularization to reduce overfitting).
    dropout: float = 0.1
    # Note: the number of outputs is NOT set here; it's derived at runtime from
    # the TiLabelSchema (number of free knot coordinates).


# Everything about the optimization / training run.
@dataclass
class TrainConfig:
    # How many samples per gradient step.
    batch_size: int = 16
    # How many full passes over the data.
    epochs: int = 100
    # Learning rate for the AdamW optimizer.
    lr: float = 1e-3
    # Weight decay (L2 regularization strength).
    weight_decay: float = 1e-4
    # DataLoader worker processes. On Windows keep 0 unless the entry point is
    # guarded by `if __name__ == "__main__"` (avoids spawn issues).
    num_workers: int = 0
    # Device to train on: "cpu" or "cuda".
    device: str = "cpu"
    # Print a progress line every this many epochs.
    log_every: int = 10
    # Where to save model checkpoints.
    ckpt_dir: Path = _PIPELINE_ROOT / "cnn" / "checkpoints"


# Top-level bundle that holds all three config groups together.
@dataclass
class PipelineConfig:
    # Each sub-config gets its own fresh instance via default_factory.
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
