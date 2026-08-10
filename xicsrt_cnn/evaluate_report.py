"""
Rich evaluation report for a trained XICS ion-temperature CNN.

Computes comparable, physically-meaningful metrics on a model's held-out TEST
set (the exact samples stored in the checkpoint), so you can fairly compare two
models (e.g. the 100-case model vs. a later 1000-case model). Each model is
evaluated on ITS OWN held-out test set.

Metrics (all averaged over the test samples):
  * knot RMSE in physical units, split into:
      - x-knots (normalized flux radius rho)
      - y-knots (ion temperature, eV)
  * profile-space RMSE: rebuild the Ti(rho) curve from the knots (PCHIP, the
    same interpolation used to generate the data) and compare predicted vs.
    true curves on a dense rho grid. This is the "how wrong is the actual
    profile" number and is the fairest single quality measure.

Results are printed AND saved to a JSON file so you can diff two runs later.

How to run (from the xics_ml_pipeline directory):
    # baseline (current 100-case model), save under a clear name:
    python -m xicsrt_cnn.evaluate_report --tag v00_100cases

    # later, the 1000-case model (point config/checkpoint as needed):
    python -m xicsrt_cnn.evaluate_report --tag v01_1000cases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from xicsrt_cnn import PipelineConfig, SampleRef
from xicsrt_cnn.train import predict_sample, load_model
import torch


# Dense grid of normalized flux radius on which we sample the Ti curve for the
# profile-space error. 201 points matches the generator's default resolution.
_RHO_GRID = np.linspace(0.0, 1.0, 201)


def _ti_curve(x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    """Rebuild the Ti(rho) profile from spline knots using PCHIP.

    PCHIP is the monotone, shape-preserving interpolator the data generator
    uses, so this reproduces the intended profile from the knots.
    """
    # PCHIP needs strictly increasing x; the knots are already sorted on decode,
    # but guard against duplicates just in case.
    x = np.asarray(x_knots, dtype=np.float64)
    y = np.asarray(y_knots, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Drop any duplicate x to keep PchipInterpolator happy.
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    spline = PchipInterpolator(x, y)
    return spline(_RHO_GRID)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-squared error between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate_report(
    checkpoint_path: str,
    cfg: PipelineConfig | None = None,
    tag: str | None = None,
    out_dir: str | None = None,
) -> dict:
    """Evaluate a model on its held-out test set and return + save a report."""
    cfg = cfg or PipelineConfig()

    # Reload the exact test samples stored in the checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    test_indices = ckpt.get("test_sample_indices", [])
    test_paths = ckpt.get("test_paths", [])
    refs = [SampleRef(Path(p), i) for p, i in zip(test_paths, test_indices)]
    if not refs:
        raise RuntimeError("No test samples recorded in this checkpoint.")

    # Per-sample metric accumulators.
    per_sample = []
    x_rmses, y_rmses, prof_rmses = [], [], []

    for ref in refs:
        # Predicted vs. true knots (physical units) for this test sample.
        pred, truth = predict_sample(checkpoint_path, cfg, sample_ref=ref)

        # Knot-level RMSE, split by coordinate type.
        x_rmse = _rmse(pred["x_knots"], truth["x_knots"])  # rho units
        y_rmse = _rmse(pred["y_knots"], truth["y_knots"])  # eV units

        # Profile-space RMSE: rebuild both Ti curves and compare on the grid.
        pred_curve = _ti_curve(pred["x_knots"], pred["y_knots"])
        true_curve = _ti_curve(truth["x_knots"], truth["y_knots"])
        prof_rmse = _rmse(pred_curve, true_curve)  # eV units

        x_rmses.append(x_rmse)
        y_rmses.append(y_rmse)
        prof_rmses.append(prof_rmse)
        per_sample.append({
            "sample": int(ref.sample_index),
            "x_knot_rmse_rho": x_rmse,
            "y_knot_rmse_eV": y_rmse,
            "profile_rmse_eV": prof_rmse,
        })

    # The data file the checkpoint actually used (from its saved test paths),
    # not the config default -- these can differ (e.g. a v01 checkpoint whose
    # test samples live in the 1000-case file even if the config points at v00).
    data_file = str(refs[0].path)

    # Aggregate metrics across the whole test set.
    report = {
        "tag": tag or Path(checkpoint_path).stem,
        "checkpoint": str(checkpoint_path),
        "data_file": data_file,
        "n_test_samples": len(refs),
        "test_sample_indices": [int(i) for i in test_indices],
        "aggregate": {
            # Mean over test samples of each per-sample RMSE.
            "x_knot_rmse_rho_mean": float(np.mean(x_rmses)),
            "y_knot_rmse_eV_mean": float(np.mean(y_rmses)),
            "profile_rmse_eV_mean": float(np.mean(prof_rmses)),
            # Also report the spread (std) so you can see consistency.
            "profile_rmse_eV_std": float(np.std(prof_rmses)),
            "profile_rmse_eV_median": float(np.median(prof_rmses)),
            "profile_rmse_eV_worst": float(np.max(prof_rmses)),
            "profile_rmse_eV_best": float(np.min(prof_rmses)),
        },
        "per_sample": per_sample,
    }

    _print_report(report)

    # Save to JSON next to the checkpoint (or a chosen directory).
    out_dir = Path(out_dir) if out_dir else Path(checkpoint_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"eval_{report['tag']}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {out_path}")

    return report


def _print_report(report: dict) -> None:
    """Pretty-print the report to the console."""
    agg = report["aggregate"]
    print(f"=== Evaluation report: {report['tag']} ===")
    print(f"data file      : {report['data_file']}")
    print(f"test samples   : {report['n_test_samples']}  "
          f"(indices {report['test_sample_indices']})")
    print()
    print("Aggregate (mean over test set):")
    print(f"  x-knot RMSE  : {agg['x_knot_rmse_rho_mean']:.4f}  (rho)")
    print(f"  y-knot RMSE  : {agg['y_knot_rmse_eV_mean']:.1f}  eV")
    print(f"  profile RMSE : {agg['profile_rmse_eV_mean']:.1f}  eV  "
          f"(median {agg['profile_rmse_eV_median']:.1f}, "
          f"best {agg['profile_rmse_eV_best']:.1f}, "
          f"worst {agg['profile_rmse_eV_worst']:.1f})")
    print()
    print("Per-sample profile RMSE (eV):")
    for r in report["per_sample"]:
        print(f"  sample {r['sample']:>4}: {r['profile_rmse_eV']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rich evaluation report.")
    parser.add_argument(
        "--checkpoint", default="xicsrt_cnn/checkpoints/best.pt",
        help="path to the trained model checkpoint",
    )
    parser.add_argument(
        "--tag", default=None,
        help="label for this evaluation (used in the saved filename)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="directory to save the JSON report (default: next to checkpoint)",
    )
    args = parser.parse_args()
    evaluate_report(args.checkpoint, tag=args.tag, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
