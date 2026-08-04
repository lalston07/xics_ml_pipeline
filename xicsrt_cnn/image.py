"""
Detector image helper.

Builds the 2-D detector image (the CNN input) from ray-detector intersection
points that are already expressed in LOCAL detector coordinates (the machine
-> local conversion is done upstream by the mentor's code).

The detector is a rectangle centered on its local origin, spanning
[-xsize/2, +xsize/2] x [-ysize/2, +ysize/2]. Each intersection is a photon
count; we histogram them into a pixel grid whose resolution is set by
`pixel_size` (matching XICSRT's own binning) or by an explicit shape.
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# `dataclass` builds the small settings container ImageConfig below.
from dataclasses import dataclass

# NumPy does all the array math and the 2-D histogram (binning).
import numpy as np


# Settings that describe the detector and how to turn intersections into pixels.
@dataclass
class ImageConfig:
    # Physical detector width in meters (local x direction). Default matches the
    # W7-X XICS detector in the sample configs; override per dataset if needed.
    xsize: float = 0.03354
    # Physical detector height in meters (local y direction).
    ysize: float = 0.25370000000000004
    # Pixel pitch in meters; xsize/pixel_size and ysize/pixel_size set the grid.
    pixel_size: float = 0.000172
    # Optional (rows, cols) override. If set, it wins over pixel_size so you can
    # force an exact image size. `None` means "compute size from pixel_size".
    shape: tuple[int, int] | None = None
    # If True, divide the finished image by its max so pixel values land in
    # [0, 1] (helps the CNN train).
    normalize: bool = True


# Work out the pixel grid size (rows, cols) implied by an ImageConfig.
def image_shape_from_config(cfg: ImageConfig) -> tuple[int, int]:
    """(rows, cols) = (y bins, x bins)."""
    # If the caller forced an explicit shape, just use it.
    if cfg.shape is not None:
        return cfg.shape
    # Number of columns = detector width divided by pixel size (at least 1).
    n_x = max(1, int(round(cfg.xsize / cfg.pixel_size)))
    # Number of rows = detector height divided by pixel size (at least 1).
    n_y = max(1, int(round(cfg.ysize / cfg.pixel_size)))
    # Images are indexed (row, col) = (y, x), so return y first.
    return (n_y, n_x)


# Turn a list of ray hit positions into a 2-D image by counting hits per pixel.
def build_detector_image(
    local_x: np.ndarray,          # x positions of hits on the detector (meters)
    local_y: np.ndarray,          # y positions of hits on the detector (meters)
    cfg: ImageConfig,             # geometry + normalization settings
    weights: np.ndarray | None = None,  # optional per-hit weight (e.g. counts)
) -> np.ndarray:
    """Bin local-coordinate intersections into a 2-D detector image.

    Parameters
    ----------
    local_x, local_y : 1-D arrays of in-plane local coordinates (meters).
        Points outside the detector extent are dropped.
    cfg : ImageConfig with detector extent / pixel size / optional shape.
    weights : optional per-ray weights (e.g. photon counts). Defaults to 1.

    Returns
    -------
    image : float32 array of shape (rows, cols) = (y bins, x bins).
    """
    # Force inputs to flat 1-D float arrays (ravel flattens any extra shape).
    local_x = np.asarray(local_x, dtype=np.float64).ravel()
    local_y = np.asarray(local_y, dtype=np.float64).ravel()
    # x and y must describe the same set of points, so lengths must match.
    if local_x.shape != local_y.shape:
        raise ValueError("local_x and local_y must have the same length")

    # Build a boolean mask that is True only where BOTH x and y are real numbers
    # (rays that missed the detector may be stored as NaN/inf).
    finite = np.isfinite(local_x) & np.isfinite(local_y)
    # Keep only the finite points.
    local_x = local_x[finite]
    local_y = local_y[finite]
    # If weights were given, keep the weights for those same points.
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()[finite]

    # Decide the output grid size (rows = y bins, cols = x bins).
    n_y, n_x = image_shape_from_config(cfg)

    # Bin edges along x: n_x pixels need n_x+1 edges, spanning the detector width
    # centered on 0 (so from -xsize/2 to +xsize/2).
    x_edges = np.linspace(-cfg.xsize / 2.0, cfg.xsize / 2.0, n_x + 1)
    # Same idea along y, spanning the detector height.
    y_edges = np.linspace(-cfg.ysize / 2.0, cfg.ysize / 2.0, n_y + 1)

    # histogram2d counts how many points fall in each (x_bin, y_bin) cell.
    # Its output is indexed [x, y]; we transpose below to get [y, x] = [row,col].
    hist, _, _ = np.histogram2d(
        local_x, local_y, bins=[x_edges, y_edges], weights=weights
    )
    # Transpose so rows=y, cols=x, and store as float32 (what the CNN expects).
    image = hist.T.astype(np.float32)  # shape (n_y, n_x)

    # Optionally rescale so the brightest pixel becomes 1.0.
    if cfg.normalize:
        # Find the max pixel value.
        mx = float(image.max())
        # Only divide if there's at least one hit (avoids divide-by-zero).
        if mx > 0:
            image = image / mx
    # Hand back the finished image.
    return image


# Resize an image to a fixed (rows, cols) so every sample feeds the CNN at the
# same resolution. Kept separate so the binning above stays free of torch.
def resize_image(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Bilinearly resize a (H, W) image to `size` (rows, cols) using torch.

    Kept separate from build_detector_image so binning stays torch-free.
    """
    # Nothing to do if no target size, or it already matches.
    if size is None or tuple(image.shape) == tuple(size):
        return image
    # Import torch only when we actually need it (keeps this module lightweight).
    import torch

    # Make a 4-D tensor [batch=1, channel=1, H, W] because interpolate wants it.
    t = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    # Bilinear resize to the requested (rows, cols).
    t = torch.nn.functional.interpolate(
        t, size=size, mode="bilinear", align_corners=False
    )
    # Drop the batch and channel dims and return a plain NumPy array.
    return t.squeeze(0).squeeze(0).numpy()
