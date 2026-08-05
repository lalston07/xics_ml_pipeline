"""
Ready-to-run training script for the XICS ion-temperature CNN.

What it does:
  1. Trains the CNN on 90 of the 100 samples in xicsrt_training_set_v00.nc.
  2. Holds out 10 samples as a final test set (never seen during training).
  3. Saves the best model to xicsrt_cnn/checkpoints/best.pt.
  4. Evaluates that best model on the held-out 10 test cases.

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

    # --- data / split ---
    # 10 of 100 samples are held out for the final test (leaves 90 to train on).
    cfg.data.test_count = 10
    # Fraction of the 90 training samples used for validation during training
    # (monitoring only). 0.11 gives 10 validation / 80 train. Set 0.0 to use
    # all 90 for training with no validation.
    cfg.data.val_fraction = 0.11
    # Detector image size (rows, cols) fed to the CNN. Smaller = faster.
    cfg.data.resize_to = (512, 128)
    # Seed controls which 10 samples are held out (kept fixed for reproducibility).
    cfg.data.seed = 0

    # --- training ---
    cfg.train.epochs = 100          # number of passes over the data
    cfg.train.batch_size = 16       # samples per gradient step
    cfg.train.lr = 1e-3             # learning rate
    cfg.train.log_every = 5         # print progress every N epochs
    cfg.train.device = "cpu"        # "cuda" if you have a GPU

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
