"""XICS detector-image -> ion-temperature spline-knot CNN pipeline.

Inputs and ground truth come from an xarray Dataset (see schema.py). The CNN
input is the detector image, built by binning ray-detector intersection points
(already in local coordinates) via image.py. The output/ground truth is the set
of free ion-temperature spline knot locations.
"""

# This file makes `cnn` a package. The imports below "lift" the most useful
# names up to the top level, so you can write e.g. `from cnn import train`
# instead of `from cnn.train import train`.

# Xarray variable-name schema + a ready-made default instance.
from .schema import XarraySchema, DEFAULT_SCHEMA
# Configuration dataclasses (paths, model shape, training hyperparameters).
from .config import PipelineConfig, DataConfig, ModelConfig, TrainConfig
# Detector-image building + resizing helpers.
from .image import ImageConfig, build_detector_image, image_shape_from_config, resize_image
# Label schema + encode/decode between xarray knots and the target vector.
from .labels import (
    TiLabelSchema,
    build_ti_schema,
    encode_sample,
    decode_targets,
)
# The PyTorch Dataset + sample discovery/splitting + per-sample image builder.
from .dataset import (
    XicsXarrayDataset,
    SampleRef,
    discover_samples,
    split_samples,
    split_train_test,
    build_image_from_sample,
)
# The CNN model.
from .model import XicsCNN
# Training + prediction + evaluation entry points.
from .train import (
    train,
    load_model,
    predict_image,
    predict_sample,
    evaluate_test,
)
# Rich evaluation report (comparable metrics for before/after comparisons).
from .evaluate_report import evaluate_report

# `__all__` lists the public API: the names exported by `from cnn import *`
# and what tools treat as this package's official surface.
__all__ = [
    "XarraySchema",
    "DEFAULT_SCHEMA",
    "PipelineConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ImageConfig",
    "build_detector_image",
    "image_shape_from_config",
    "resize_image",
    "TiLabelSchema",
    "build_ti_schema",
    "encode_sample",
    "decode_targets",
    "XicsXarrayDataset",
    "SampleRef",
    "discover_samples",
    "split_samples",
    "split_train_test",
    "build_image_from_sample",
    "XicsCNN",
    "train",
    "load_model",
    "predict_image",
    "predict_sample",
    "evaluate_test",
    "evaluate_report",
]
