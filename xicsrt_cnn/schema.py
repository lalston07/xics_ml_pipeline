"""
Xarray schema for the XICS CNN.

The training data is an xarray Dataset (per raytrace sample, or a stacked set
of samples) that holds BOTH the CNN input (detector-plane ray intersections in
LOCAL detector coordinates) and the ground-truth output (ion-temperature spline
knots).

>>> THIS FILE IS THE SINGLE PLACE TO UPDATE WHEN THE MENTOR'S DATA ARRIVES. <<<
The exact variable / coordinate names in the delivered xarray are not final yet.
Adjust the string names in `XarraySchema` below to match, and nothing else in
the cnn/ package needs to change.

Assumptions baked in (all overridable here):
  * Ray intersections are already converted to LOCAL detector coordinates
    upstream. We only need the in-plane (x_local, y_local) components to bin
    into an image.
  * Only the ion-temperature profile is used as the label for now.
  * A "sample" dimension indexes independent raytrace runs when multiple are
    stored in one Dataset. If the Dataset holds a single sample, `sample_dim`
    may be absent and the loaders treat it as one sample.
"""

# Lets us write type hints like `tuple[float, float]` even on older Pythons.
from __future__ import annotations

# `dataclass` turns a class into a simple, auto-initialized container of fields.
from dataclasses import dataclass


# `frozen=True` makes instances read-only (immutable), so a schema can't be
# accidentally changed after it is created. Each attribute below is a *field*
# with a default value, so `XarraySchema()` gives you all the defaults at once.
@dataclass(frozen=True)
class XarraySchema:
    # --- dimensions ---------------------------------------------------------
    # Name of the xarray dimension that indexes independent samples/raytraces.
    # If the delivered dataset has no such dimension, the loaders treat each
    # file as a single sample.
    sample_dim: str = "sample"
    # Name of the dimension that indexes individual ray intersection points
    # (i.e. how many photons/rays hit the detector in one sample).
    ray_dim: str = "ray"

    # --- CNN input: detector-plane intersections (LOCAL coords) -------------
    # Name of the variable holding each ray's x position on the detector,
    # already in local coordinates (meters).
    intersect_x: str = "detector_local_x"
    # Same, for the y position on the detector (meters).
    intersect_y: str = "detector_local_y"

    # --- detector geometry (used to size/bin the image) ---------------------
    # Names of the scalar variables giving the detector's physical width...
    det_xsize: str = "detector_xsize"
    # ...height...
    det_ysize: str = "detector_ysize"
    # ...and pixel pitch (all in meters). If any are missing from the xarray,
    # the loader falls back to the defaults in ImageConfig (see image.py).
    det_pixel_size: str = "detector_pixel_size"

    # --- ground-truth label: ion-temperature spline knots -------------------
    # Variable holding the knot x-positions (normalized flux radius rho).
    ti_x_knots: str = "ion_temp_x_knots"
    # Variable holding the knot y-values (ion temperature in keV).
    ti_y_knots: str = "ion_temp_y_knots"
    # Boolean-per-knot: True where the x-position is free (predicted by the CNN).
    ti_x_free: str = "ion_temp_x_free"
    # Boolean-per-knot: True where the y-value is free (predicted by the CNN).
    ti_y_free: str = "ion_temp_y_free"
    # Name of the dimension that runs over the (5) knots.
    knot_dim: str = "knot"

    # --- physical normalization ranges for the Ti knots ---------------------
    # x is normalized flux radius rho, which already lives in [0, 1].
    ti_x_range: tuple[float, float] = (0.0, 1.0)
    # y is ion temperature in keV; upper bound matches generate_random_ion_temp
    # (y_max = 5.0 keV). These ranges scale the labels into [0, 1] for training.
    ti_y_range: tuple[float, float] = (0.0, 5.0)


# A ready-made default instance so other modules can just import DEFAULT_SCHEMA
# instead of constructing XarraySchema() themselves every time.
DEFAULT_SCHEMA = XarraySchema()
