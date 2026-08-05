"""
View or save the binned detector image for one sample.

The CNN never uses image files -- it bins the local-coordinate ray
intersections into a 2-D array in memory (see image.py). This script lets you
*look* at that array: display it on screen and/or save it as a real image file
(.tiff, .png, etc.) so you can visually check what the CNN sees.

How to run (from the xics_ml_pipeline directory):
    # display sample 0 on screen:
    python -m xicsrt_cnn.view_image

    # display a specific sample index (e.g. 16):
    python -m xicsrt_cnn.view_image 16

    # save sample 16 to a file instead of (or as well as) showing it:
    python -m xicsrt_cnn.view_image 16 --save sample16.tiff
"""

from __future__ import annotations

import argparse

import numpy as np

from xicsrt_cnn import PipelineConfig, discover_samples, build_image_from_sample


def get_image(sample_index: int, cfg: PipelineConfig | None = None) -> np.ndarray:
    """Build and return the binned detector image (2-D float array) for a sample."""
    cfg = cfg or PipelineConfig()
    # Discover all samples and their reference dataset.
    refs, ds = discover_samples(cfg.data)
    # Find the SampleRef whose sample_index matches the request.
    try:
        ref = next(r for r in refs if r.sample_index == sample_index)
    except StopIteration:
        raise ValueError(f"Sample index {sample_index} not found in the dataset.")
    # Bin that sample's local-coord intersections into the 2-D image array.
    return build_image_from_sample(ds, ref.sample_index, cfg.data)


def save_image(image: np.ndarray, path: str) -> None:
    """Save the image array to a file (.tiff, .png, ...) using PIL.

    The array is float in [0, 1]; TIFF keeps it as float32, while formats like
    PNG need 8-bit, so we scale to 0-255 for those.
    """
    from PIL import Image

    lower = path.lower()
    if lower.endswith((".tif", ".tiff")):
        # TIFF can store the raw float32 values directly.
        Image.fromarray(image.astype(np.float32)).save(path)
    else:
        # Other formats (png/jpg) need 8-bit integers.
        arr8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(arr8).save(path)
    print(f"Saved image to {path}  (shape {image.shape})")


def show_image(image: np.ndarray, sample_index: int) -> None:
    """Display the image on screen with matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # aspect='auto' because the detector is tall and narrow; origin='lower'
    # puts local y increasing upward.
    im = ax.imshow(image, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"Detector image - sample {sample_index}")
    ax.set_xlabel("local x (pixels)")
    ax.set_ylabel("local y (pixels)")
    fig.colorbar(im, ax=ax, label="normalized counts")
    plt.show()


def main() -> None:
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="View/save a detector image.")
    parser.add_argument(
        "sample_index", nargs="?", type=int, default=0,
        help="which sample to view (default: 0)",
    )
    parser.add_argument(
        "--save", metavar="PATH", default=None,
        help="save the image to this file (.tiff, .png, ...)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="do not open the on-screen viewer (useful with --save)",
    )
    args = parser.parse_args()

    # Build the image for the requested sample.
    image = get_image(args.sample_index)
    print(
        f"sample {args.sample_index}: image shape {image.shape}, "
        f"min {image.min():.3f}, max {image.max():.3f}, "
        f"nonzero pixels {int((image > 0).sum())}"
    )

    # Save if requested.
    if args.save:
        save_image(image, args.save)

    # Show unless suppressed.
    if not args.no_show:
        show_image(image, args.sample_index)


# Only run when executed directly, not when imported.
if __name__ == "__main__":
    main()
