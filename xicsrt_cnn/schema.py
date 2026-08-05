"""
Xarray schema for the XICS CNN.

The training data is an xarray Dataset (a stacked set of samples) that holds
BOTH the CNN input (detector-plane ray intersections in LOCAL detector
coordinates) and the ground-truth output (ion-temperature spline knots).

>>> THIS FILE IS THE SINGLE PLACE TO UPDATE IF THE DATA FORMAT CHANGES. <<<
The names below match the delivered dataset `xicsrt_training_set_v00.nc`.
Adjust the string names in `XarraySchema` to match a new file, and nothing
else in the xicsrt_cnn/ package needs to change.

Layout of xicsrt_training_set_v00.nc (100 samples):
  * `sample`      dim = 100 independent raytraces.
  * `ray`         dim = 3223 ray slots per sample (padded; unused slots = NaN).
  * `axis`        dim = 2, coordinate values ['x', 'y'].
  * `intersect`   variable, dims (sample, ray, axis): each ray's detector-plane
                  hit position in LOCAL coordinates (meters). Already converted
                  from machine coords upstream. Padding rays are NaN and get
                  dropped when we bin the image.
  * detector geometry + ion-temp knots live under the flattened
    `config__...` variable names (double-underscore separators).

Assumptions baked in (all overridable here):
  * Only the ion-temperature profile is used as the label for now.
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
    sample_dim: str = "sample"
    # Name of the dimension that indexes individual ray slots within a sample.
    ray_dim: str = "ray"

    # --- CNN input: detector-plane intersections (LOCAL coords) -------------
    # Name of the single variable holding all ray hit positions. Its dims are
    # (sample, ray, axis); the `axis` dimension separates x from y.
    intersect: str = "intersect"
    # Name of the dimension that separates the x and y components.
    axis_dim: str = "axis"
    # The coordinate labels along `axis_dim` that pick the x and y components.
    axis_x: str = "x"
    axis_y: str = "y"
    # Optional: variable giving how many rays are actually valid per sample.
    # We don't strictly need it (padding rays are NaN and dropped anyway), but
    # it's recorded here for reference. Set to None if absent.
    sample_count: str = "sample_count"

    # --- detector geometry (used to size/bin the image) ---------------------
    # Scalar-per-sample variables giving the detector's physical width...
    det_xsize: str = "config__optics__detector__xsize"
    # ...height...
    det_ysize: str = "config__optics__detector__ysize"
    # ...and pixel pitch (all in meters). If any are missing from the xarray,
    # the loader falls back to the defaults in ImageConfig (see image.py).
    det_pixel_size: str = "config__optics__detector__pixel_size"

    # --- ground-truth label: ion-temperature spline knots -------------------
    # Variable holding the knot x-positions (normalized flux radius rho).
    ti_x_knots: str = "config__sources__plasma__profile_ion_temp__x_knots"
    # Variable holding the knot y-values (ion temperature in eV).
    ti_y_knots: str = "config__sources__plasma__profile_ion_temp__y_knots"
    # Boolean-per-knot: True where the x-position is free (predicted by the CNN).
    ti_x_free: str = "config__sources__plasma__profile_ion_temp__x_free"
    # Boolean-per-knot: True where the y-value is free (predicted by the CNN).
    ti_y_free: str = "config__sources__plasma__profile_ion_temp__y_free"

    # --- physical normalization ranges for the Ti knots ---------------------
    # x is normalized flux radius rho, which already lives in [0, 1].
    ti_x_range: tuple[float, float] = (0.0, 1.0)
    # y is ion temperature in eV. In this dataset temperature_scale = 1.0 and
    # the knot y-values span ~0..5000 eV, so we scale by 5000 to map into [0, 1].
    ti_y_range: tuple[float, float] = (0.0, 5000.0)


# A ready-made default instance so other modules can just import DEFAULT_SCHEMA
# instead of constructing XarraySchema() themselves every time.
DEFAULT_SCHEMA = XarraySchema()
