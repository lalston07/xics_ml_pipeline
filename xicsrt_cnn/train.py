"""
Training + prediction for the XICS ion-temperature CNN.

Usage (from the xics_ml_pipeline directory):
    python -m cnn.train          # train using PipelineConfig defaults

Loss: MSE on the normalized free ion-temp knot target vector (knot-space).
A profile-space loss (sampling the reconstructed PCHIP/Hermite spline on a rho
grid) is a planned upgrade; decode_targets in labels.py already supports it.
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# `time` for per-epoch timing; `Path` for the checkpoint location.
import time
from pathlib import Path

# NumPy + torch; DataLoader batches and shuffles the Dataset.
import numpy as np
import torch
from torch.utils.data import DataLoader

# Our config + pipeline pieces.
from .config import PipelineConfig
from .dataset import (
    XicsXarrayDataset,
    SampleRef,
    build_image_from_sample,
    discover_samples,
    split_samples,
    split_train_test,
)
from .labels import TiLabelSchema, build_ti_schema, decode_targets
from .model import XicsCNN


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
# Wrap the samples in train/val DataLoaders that feed batches to the model.
# `refs` here are the TRAINING samples only (the test set is held out earlier).
def make_loaders(cfg: PipelineConfig, label_schema: TiLabelSchema, refs):
    # Carve a validation set out of the training samples for monitoring.
    train_refs, val_refs = split_samples(refs, cfg.data.val_fraction, cfg.data.seed)
    # Build a Dataset for each split (val only if there are val samples).
    train_ds = XicsXarrayDataset(train_refs, label_schema, cfg.data)
    val_ds = XicsXarrayDataset(val_refs, label_schema, cfg.data) if val_refs else None

    # Training loader: shuffles each epoch for better learning.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    # Validation loader: no shuffle needed; None if there's no val set.
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )
        if val_ds is not None
        else None
    )
    # Return the loaders plus the split sizes (for logging).
    return train_loader, val_loader, len(train_refs), len(val_refs)


# Compute the average loss over a loader WITHOUT updating weights.
def evaluate(model, loader, loss_fn, device) -> float:
    # Put the model in eval mode (turns off dropout, freezes BatchNorm stats).
    model.eval()
    # Running total loss and sample count.
    total, n = 0.0, 0
    # `no_grad` disables gradient tracking -> faster, less memory.
    with torch.no_grad():
        # Loop over batches.
        for images, targets in loader:
            # Move data to the chosen device (cpu/cuda).
            images, targets = images.to(device), targets.to(device)
            # Forward pass -> predictions.
            preds = model(images)
            # Accumulate loss weighted by batch size (so the average is correct
            # even if the last batch is smaller).
            total += loss_fn(preds, targets).item() * images.size(0)
            n += images.size(0)
    # Average loss (max(n,1) avoids divide-by-zero on an empty loader).
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# The main training routine. Returns the path to the best saved checkpoint.
def train(cfg: PipelineConfig | None = None) -> Path:
    # Use provided config, or fall back to defaults.
    cfg = cfg or PipelineConfig()

    # Find all samples and build the label schema from a reference sample.
    refs, ref_ds = discover_samples(cfg.data)
    label_schema = build_ti_schema(ref_ds, cfg.data.schema)
    # Print the schema summary so you can see the target layout.
    print(label_schema.describe())

    # Hold out the final TEST set first (e.g. 10 of 100 samples). The model
    # never sees these during training; they are for final evaluation only.
    train_refs, test_refs = split_train_test(
        refs, cfg.data.test_count, cfg.data.seed
    )
    print(f"Total samples: {len(refs)} -> train {len(train_refs)}, "
          f"held-out test {len(test_refs)}")

    # Choose the device: honor the request, but fall back to cpu if cuda is
    # asked for but unavailable.
    device = torch.device(
        cfg.train.device
        if (cfg.train.device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )

    # Build the data loaders from the TRAINING samples (val carved from them).
    train_loader, val_loader, n_train, n_val = make_loaders(cfg, label_schema, train_refs)
    print(f"Training samples: {n_train}, validation samples: {n_val}")

    # Create the model with the right number of outputs, on the chosen device.
    model = XicsCNN(cfg.model, n_targets=label_schema.size).to(device)
    # AdamW optimizer with the configured learning rate and weight decay.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # Mean-squared-error loss between predicted and true knot vectors.
    loss_fn = torch.nn.MSELoss()

    # Make sure the checkpoint directory exists.
    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Track the best (lowest) loss seen so far; start at infinity.
    best_val = float("inf")
    # Where the best model will be written (name is configurable so different
    # datasets/versions can each keep their own checkpoint).
    best_path = ckpt_dir / cfg.train.ckpt_name

    # Main epoch loop.
    for epoch in range(1, cfg.train.epochs + 1):
        # Put the model in training mode (enables dropout, updates BN stats).
        model.train()
        # Start the epoch timer and reset running totals.
        t0 = time.time()
        running, n = 0.0, 0
        # Loop over training batches.
        for images, targets in train_loader:
            # Move batch to device.
            images, targets = images.to(device), targets.to(device)
            # Clear gradients from the previous step.
            optimizer.zero_grad()
            # Forward pass.
            preds = model(images)
            # Compute the loss.
            loss = loss_fn(preds, targets)
            # Backpropagate to get gradients.
            loss.backward()
            # Update the weights.
            optimizer.step()
            # Accumulate weighted loss for the epoch average.
            running += loss.item() * images.size(0)
            n += images.size(0)
        # Average training loss for this epoch.
        train_loss = running / max(n, 1)

        # Validation loss (NaN if there's no validation set).
        val_loss = (
            evaluate(model, val_loader, loss_fn, device)
            if val_loader is not None
            else float("nan")
        )

        # Print a progress line on the first epoch and every log_every after.
        if epoch % cfg.train.log_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | train {train_loss:.6f} | "
                f"val {val_loss:.6f} | {time.time() - t0:.1f}s"
            )

        # Decide what to track for "best": val loss if available, else train.
        current = val_loss if val_loader is not None else train_loss
        # If this is the best so far, save a checkpoint.
        if current < best_val:
            best_val = current
            # Save the weights PLUS everything needed to rebuild the model and
            # decode predictions later (schema masks, ranges, output size).
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": cfg.model,
                    "n_targets": label_schema.size,
                    "label_x_free": label_schema.x_free,
                    "label_y_free": label_schema.y_free,
                    "label_x_range": label_schema.x_range,
                    "label_y_range": label_schema.y_range,
                    "epoch": epoch,
                    "val_loss": best_val,
                    # Record which samples were held out for testing, so
                    # evaluate_test() can reload exactly the same 10.
                    "test_sample_indices": [r.sample_index for r in test_refs],
                    "test_paths": [str(r.path) for r in test_refs],
                },
                best_path,
            )

    # Report and return where the best model was saved.
    print(f"Best loss {best_val:.6f} -> {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
# Rebuild a trained model (and its label schema) from a saved checkpoint.
def load_model(checkpoint_path, cfg: PipelineConfig | None = None):
    """Load a trained model + its label schema from a checkpoint."""
    # Use provided config, or defaults.
    cfg = cfg or PipelineConfig()
    # Load the checkpoint dict onto CPU. weights_only=False because we saved
    # non-tensor objects (schema arrays, model_cfg) alongside the weights.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Reconstruct the label schema from the saved masks and ranges.
    label_schema = TiLabelSchema(
        x_free=np.asarray(ckpt["label_x_free"], dtype=bool),
        y_free=np.asarray(ckpt["label_y_free"], dtype=bool),
        x_range=tuple(ckpt["label_x_range"]),
        y_range=tuple(ckpt["label_y_range"]),
    )
    # Recreate the model with the same architecture and output size...
    model = XicsCNN(ckpt.get("model_cfg", cfg.model), n_targets=ckpt["n_targets"])
    # ...load the trained weights...
    model.load_state_dict(ckpt["model_state"])
    # ...and switch to eval mode for inference.
    model.eval()
    return model, label_schema


# `@torch.no_grad()` wraps the whole function so no gradients are tracked.
@torch.no_grad()
def predict_image(model, image: np.ndarray, label_schema: TiLabelSchema,
                  x_knots_template: np.ndarray, y_knots_template: np.ndarray):
    """Predict ion-temp knots from a single detector image (H, W).

    Returns physical-unit {"x_knots", "y_knots"} via decode_targets.
    """
    # Make a 4-D tensor [batch=1, channel=1, H, W] from the image.
    t = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    # Run the model, drop the batch dim, back to a NumPy vector.
    vec = model(t).squeeze(0).cpu().numpy()
    # Turn the normalized vector into full physical-unit knot arrays.
    return decode_targets(vec, label_schema, x_knots_template, y_knots_template)


# End-to-end convenience: checkpoint + one sample -> predicted vs true knots.
@torch.no_grad()
def predict_sample(checkpoint_path, cfg: PipelineConfig | None = None,
                   sample_ref=None):
    """End-to-end: load model, build image for one sample, predict knots.

    If sample_ref is None, uses the first discovered sample. Returns
    (predicted_knots, truth_knots) both in physical units.
    """
    # Use provided config, or defaults.
    cfg = cfg or PipelineConfig()
    # Load the trained model and its label schema.
    model, label_schema = load_model(checkpoint_path, cfg)

    # Find samples; use the given one or default to the first.
    refs, ref_ds = discover_samples(cfg.data)
    ref = sample_ref or refs[0]

    # Open the file this sample lives in (use the configured backend engine).
    import xarray as xr
    ds = (
        xr.open_dataset(ref.path, engine=cfg.data.engine)
        if cfg.data.engine
        else xr.open_dataset(ref.path)
    )
    schema = cfg.data.schema

    # Local helper: read a knot array for this sample as a flat float array.
    def knots(name):
        da = ds[name]
        if ref.sample_index is not None and schema.sample_dim in da.dims:
            da = da.isel({schema.sample_dim: ref.sample_index})
        return np.asarray(da.values, dtype=np.float64).ravel()

    # The true knot arrays; also used as templates to fill fixed knots on decode.
    x_tmpl = knots(schema.ti_x_knots)
    y_tmpl = knots(schema.ti_y_knots)

    # Build the detector image for this sample.
    image = build_image_from_sample(ds, ref.sample_index, cfg.data)
    # Predict knots from the image.
    pred = predict_image(model, image, label_schema, x_tmpl, y_tmpl)
    # Package the ground truth in the same format.
    truth = {"x_knots": x_tmpl, "y_knots": y_tmpl}
    # Return (prediction, truth) for easy comparison/plotting.
    return pred, truth


# Evaluate a trained model on the held-out TEST set saved in the checkpoint.
@torch.no_grad()
def evaluate_test(checkpoint_path, cfg: PipelineConfig | None = None):
    """Run the trained model on the 10 held-out test samples.

    Reloads exactly the samples that were held out during training (their
    indices are stored in the checkpoint). Returns a list of per-sample dicts
    with predicted vs. true knots and the mean-squared knot error, and prints
    a short summary.
    """
    # Use provided config, or defaults.
    cfg = cfg or PipelineConfig()
    # Load the trained model + label schema.
    model, label_schema = load_model(checkpoint_path, cfg)
    # Load the checkpoint again to read the stored test-sample indices.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    test_indices = ckpt.get("test_sample_indices", [])
    test_paths = ckpt.get("test_paths", [])

    # Rebuild SampleRef objects for the held-out test samples.
    refs = [SampleRef(Path(p), i) for p, i in zip(test_paths, test_indices)]
    if not refs:
        print("No test samples recorded in the checkpoint.")
        return []

    # Evaluate each test sample and collect the results.
    results = []
    for ref in refs:
        pred, truth = predict_sample(checkpoint_path, cfg, sample_ref=ref)
        # Mean-squared error between predicted and true knots (physical units).
        mse = float(
            np.mean(
                (pred["x_knots"] - truth["x_knots"]) ** 2
                + (pred["y_knots"] - truth["y_knots"]) ** 2
            )
        )
        results.append({"sample": ref.sample_index, "pred": pred,
                        "truth": truth, "mse": mse})

    # Print a compact summary.
    mean_mse = float(np.mean([r["mse"] for r in results]))
    print(f"Held-out test set: {len(results)} samples, mean knot MSE {mean_mse:.4f}")
    for r in results:
        print(f"  sample {r['sample']}: knot MSE {r['mse']:.4f}")
    return results


# Only run training when this file is executed directly
# (python -m xicsrt_cnn.train), not when it is imported.
if __name__ == "__main__":
    train()
