"""
Inspect a single test case: predicted vs. true ion-temperature knots.

Loads a trained checkpoint, runs the model on one sample from
xicsrt_training_set_v00.nc, and prints the predicted knot locations next to the
ground-truth knot locations.

How to run (from the xics_ml_pipeline directory):
    # inspect the first held-out test case:
    python -m xicsrt_cnn.inspect_case

    # inspect a specific sample index (e.g. 16):
    python -m xicsrt_cnn.inspect_case 16

`pred` and `truth` are full knot arrays (fixed knots filled in), ready to feed
into a PCHIP / CubicHermiteSpline to rebuild and plot the profile.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from xicsrt_cnn import PipelineConfig, discover_samples
from xicsrt_cnn.train import predict_sample

# Path to the trained model saved by run_training.py.
CHECKPOINT = "xicsrt_cnn/checkpoints/best.pt"


def held_out_indices(checkpoint_path):
    """Return the list of sample indices that were held out for testing.

    These are stored inside the checkpoint at training time.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("test_sample_indices", [])


def inspect(sample_index: int | None = None, checkpoint_path: str = CHECKPOINT) -> None:
    # Use the default configuration (same data file + split as training).
    cfg = PipelineConfig()

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
    # Optional command-line argument: the sample index to inspect.
    sample_index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    inspect(sample_index)


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    main()
