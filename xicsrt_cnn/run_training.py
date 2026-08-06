"""
Ready-to-run training script for the XICS ion-temperature CNN.

What it does (current settings = 1000-case dataset):
  1. Trains the CNN on 800 of the 1000 samples in xicsrt_training_set_v01.nc.
  2. Holds out 100 samples as a final test set (never seen during training).
  3. Uses 100 samples for validation during training.
  4. Saves the best model to xicsrt_cnn/checkpoints/best.pt.
  5. Evaluates that best model on the held-out 100 test cases.

The older 100-case (v00) settings are kept below but commented out, so you can
still see / revert to them.

How to run (from the xics_ml_pipeline directory):
    python -m xicsrt_cnn.run_training

Adjust the settings in the CONFIG section below as needed.
"""

from __future__ import annotations

from xicsrt_cnn import PipelineConfig
from xicsrt_cnn.train import train, evaluate_test


def main() -> None:
    # ----------------------------------------------------------------------
    # CONFIG - edit these to change how training runs
    # ----------------------------------------------------------------------
    cfg = PipelineConfig()

    # ======================================================================
    # CURRENT: 1000-case dataset (xicsrt_training_set_v01.nc)
    #   split -> 800 train / 100 validation / 100 test
    # ======================================================================
    # --- data / split ---
    # Point at the 1000-case file (sits next to the v00 file in the repo root).
    cfg.data.xarray_path = cfg.data.xarray_path.parent / "xicsrt_training_set_v01.nc"
    # Hold out 100 of 1000 samples for the final test (leaves 900).
    cfg.data.test_count = 100
    # Fraction of the remaining 900 used for validation. 100/900 gives exactly
    # 100 validation / 800 train.
    cfg.data.val_fraction = 100 / 900
    # Detector image size (rows, cols) fed to the CNN. Smaller = faster.
    cfg.data.resize_to = (512, 128)
    # Seed controls which samples are held out (kept fixed for reproducibility).
    cfg.data.seed = 0
    # Save this model under a version-specific name so it does NOT overwrite the
    # 100-case (v00) model. Both survive in xicsrt_cnn/checkpoints/.
    cfg.train.ckpt_name = "best_v01.pt"

    # ======================================================================
    # OLD: 100-case dataset (xicsrt_training_set_v00.nc)
    #   split -> 80 train / 10 validation / 10 test
    # Kept for reference; uncomment this block (and comment the one above) to
    # reproduce the original 100-case run.
    # ======================================================================
    # cfg.data.xarray_path = cfg.data.xarray_path.parent / "xicsrt_training_set_v00.nc"
    # cfg.data.test_count = 10
    # cfg.data.val_fraction = 0.11   # 10 validation / 80 train
    # cfg.data.resize_to = (512, 128)
    # cfg.data.seed = 0
    # cfg.train.ckpt_name = "best_v00.pt"

    # --- training ---
    cfg.train.epochs = 100          # number of passes over the data
    cfg.train.batch_size = 16       # samples per gradient step
    cfg.train.lr = 1e-3             # learning rate
    cfg.train.log_every = 5         # print progress every N epochs
    cfg.train.device = "cuda"       # use the GPU (falls back to CPU if none)

    # ----------------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------------
    # Step 1: train (returns the path to the best saved checkpoint).
    print("=== Training ===")
    ckpt_path = train(cfg)

    # Step 2: evaluate the best model on the held-out 10 test cases.
    print("\n=== Testing on held-out cases ===")
    evaluate_test(ckpt_path, cfg)

    print(f"\nDone. Best model saved to: {ckpt_path}")


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    main()
