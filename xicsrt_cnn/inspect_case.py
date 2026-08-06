"""
Inspect a single test case: predicted vs. true ion-temperature knots.

Loads a trained checkpoint, runs the model on one sample, and prints the
predicted knot locations next to the ground-truth knot locations.

How to run (from the xics_ml_pipeline directory):
    # first held-out test case of the default checkpoint:
    python -m xicsrt_cnn.inspect_case

    # a specific sample index (e.g. 16):
    python -m xicsrt_cnn.inspect_case 16

    # choose which model to use (version-specific checkpoints):
    python -m xicsrt_cnn.inspect_case 16 --checkpoint xicsrt_cnn/checkpoints/best_v01.pt

`pred` and `truth` are full knot arrays (fixed knots filled in), ready to feed
into a PCHIP / CubicHermiteSpline to rebuild and plot the profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from xicsrt_cnn import PipelineConfig, discover_samples
from xicsrt_cnn.train import predict_sample

# Default checkpoint if none is given on the command line.
DEFAULT_CHECKPOINT = "xicsrt_cnn/checkpoints/best.pt"


def held_out_indices(checkpoint_path):
    """Return the list of sample indices that were held out for testing.

    These are stored inside the checkpoint at training time.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("test_sample_indices", [])


def _config_for_checkpoint(checkpoint_path: str) -> PipelineConfig:
    """Build a config whose data file matches the one this checkpoint used.

    A v00 checkpoint's test samples come from the v00 .nc file, a v01's from the
    v01 file. We read the data file the checkpoint recorded (via test_paths) so
    discover_samples looks in the correct file regardless of the config default.
    """
    cfg = PipelineConfig()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    paths = ckpt.get("test_paths", [])
    if paths:
        # All test samples share the same source file; use the first.
        cfg.data.xarray_path = Path(paths[0])
    return cfg


def inspect(sample_index: int | None = None,
            checkpoint_path: str = DEFAULT_CHECKPOINT) -> None:
    # Use a config whose data file matches this checkpoint's training data.
    cfg = _config_for_checkpoint(checkpoint_path)

    # Discover all samples so we can pick the one we want by its index.
    refs, _ = discover_samples(cfg.data)

    # If no index was given, default to the first held-out test case.
    if sample_index is None:
        test_ids = held_out_indices(checkpoint_path)
        if not test_ids:
            raise RuntimeError(
                "No held-out test indices found in the checkpoint. "
                "Train first with: python -m xicsrt_cnn.run_training"
            )
        sample_index = test_ids[0]

    # Find the SampleRef matching the requested sample index.
    try:
        ref = next(r for r in refs if r.sample_index == sample_index)
    except StopIteration:
        raise ValueError(f"Sample index {sample_index} not found in the dataset.")

    # Warn (but continue) if this case was actually part of training.
    test_ids = held_out_indices(checkpoint_path)
    where = "HELD-OUT TEST" if sample_index in test_ids else "TRAINING"
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Inspecting sample {sample_index}  ({where} case)\n")

    # Run the model: returns predicted and true knots in physical units.
    pred, truth = predict_sample(checkpoint_path, cfg, sample_ref=ref)

    # Print a tidy comparison.
    np.set_printoptions(precision=4, suppress=True)
    print("x_knots (normalized flux radius rho):")
    print(f"  predicted: {pred['x_knots']}")
    print(f"  true     : {truth['x_knots']}")
    print()
    print("y_knots (ion temperature, eV):")
    print(f"  predicted: {pred['y_knots']}")
    print(f"  true     : {truth['y_knots']}")
    print()

    # Simple error summaries.
    x_err = np.abs(pred["x_knots"] - truth["x_knots"])
    y_err = np.abs(pred["y_knots"] - truth["y_knots"])
    print(f"mean |x error| (rho): {x_err.mean():.4f}")
    print(f"mean |y error| (eV) : {y_err.mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect predicted vs. true knots for one sample."
    )
    parser.add_argument(
        "sample_index", nargs="?", type=int, default=None,
        help="which sample to inspect (default: first held-out test case)",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="path to the trained model checkpoint "
             "(e.g. xicsrt_cnn/checkpoints/best_v01.pt)",
    )
    args = parser.parse_args()
    inspect(args.sample_index, args.checkpoint)


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    main()
