"""
Label encoding / decoding for the XICS CNN (ion-temperature only).

The network regresses the *free* ion-temperature spline knot coordinates.
Fixed coordinates (x=0 at the axis, x=1 at the LCFS, edge y=0) are NOT
predicted; they are reconstructed from the knot layout at decode time.

Labels are read from an xarray Dataset (one sample) using the names defined
in schema.py. This is the single contract between the delivered xarray and the
network's output vector, so encoding (training) and decoding (plotting) stay in
sync.

Flat target vector ordering (ion temp only):
    all free x_knots (in knot index order), then all free y_knots.

Each free coordinate is min-max normalized into [0, 1] using the ranges in the
schema so x (rho) and y (keV) contribute comparably to the loss.
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# `dataclass` builds the small TiLabelSchema container below.
from dataclasses import dataclass

# NumPy handles all array indexing/normalization.
import numpy as np

# The names/ranges that tell us where the knots live in the xarray.
from .schema import XarraySchema, DEFAULT_SCHEMA


# Describes which knot coordinates are "free" (predicted) vs "fixed", plus the
# physical ranges used to scale them into [0, 1]. Immutable (frozen=True) so it
# can't change after being built from a reference sample.
@dataclass(frozen=True)
class TiLabelSchema:
    """Free/fixed layout of the ion-temp knots + normalization ranges.

    Derived once from a reference sample and assumed constant across the
    dataset (the generators enforce a fixed layout).
    """

    # Boolean array, one entry per knot: True where the knot's x is predicted.
    x_free: np.ndarray  # bool per knot
    # Boolean array, one entry per knot: True where the knot's y is predicted.
    y_free: np.ndarray  # bool per knot
    # (min, max) physical range of x used to normalize/denormalize (rho).
    x_range: tuple[float, float]
    # (min, max) physical range of y used to normalize/denormalize (keV).
    y_range: tuple[float, float]

    # `@property` lets you access these like attributes (schema.n_knots) even
    # though they are computed on the fly from x_free/y_free.
    @property
    def n_knots(self) -> int:
        # Total number of knots = length of the per-knot boolean array.
        return int(self.x_free.size)

    @property
    def n_x_free(self) -> int:
        # How many x-coordinates are free = number of True entries in x_free.
        return int(self.x_free.sum())

    @property
    def n_y_free(self) -> int:
        # How many y-coordinates are free = number of True entries in y_free.
        return int(self.y_free.sum())

    @property
    def size(self) -> int:
        # Length of the CNN output vector = free x count + free y count.
        return self.n_x_free + self.n_y_free

    def describe(self) -> str:
        # Human-readable one-liner showing counts and which slots hold x vs y.
        return (
            f"TiLabelSchema: {self.n_knots} knots, "
            f"{self.n_x_free} free-x + {self.n_y_free} free-y = {self.size} targets "
            f"(x slots 0..{self.n_x_free - 1}, y slots "
            f"{self.n_x_free}..{self.size - 1})"
        )


# Small helper: pull one variable out of the xarray, optionally picking a single
# sample. Underscore prefix means "internal to this module".
def _get(ds, name: str, sample_dim: str, sample_index=None):
    """Fetch a variable from an xarray Dataset, optionally selecting a sample.

    Returns a numpy array (or scalar). Raises KeyError with a clear message if
    the name is missing so schema mismatches are easy to diagnose.

    Parameters
    ----------
    ds : xarray Dataset to read from.
    name : variable/coordinate name to fetch.
    sample_dim : name of the dimension indexing samples (from the schema).
    sample_index : if given and `sample_dim` is present on the variable,
        select that single sample along `sample_dim`.
    """
    # If the requested name isn't a data variable OR a coordinate, fail loudly
    # with a hint that schema.py probably needs updating.
    if name not in ds.variables and name not in getattr(ds, "coords", {}):
        raise KeyError(
            f"'{name}' not found in xarray. Update cnn/schema.py to match the "
            f"delivered dataset. Available: {list(ds.variables)}"
        )
    # Grab the DataArray for this variable.
    da = ds[name]
    # If we were told which sample to use, and this variable actually has the
    # sample dimension, slice out just that one sample.
    if sample_index is not None and sample_dim in da.dims:
        da = da.isel({sample_dim: sample_index})
    # Return the plain NumPy values (drops xarray metadata).
    return np.asarray(da.values)


# Build the TiLabelSchema (free/fixed layout) by reading the boolean masks from
# a reference sample in the xarray.
def build_ti_schema(ds, schema: XarraySchema = DEFAULT_SCHEMA) -> TiLabelSchema:
    """Build the ion-temp label schema from a reference xarray Dataset."""
    # If the dataset stacks many samples, just look at sample 0; otherwise None.
    sample_index = 0 if schema.sample_dim in ds.dims else None
    # Read the per-knot "is x free?" mask and force it to a 1-D bool array.
    x_free = np.asarray(
        _get(ds, schema.ti_x_free, schema.sample_dim, sample_index), dtype=bool
    ).ravel()
    # Read the per-knot "is y free?" mask likewise.
    y_free = np.asarray(
        _get(ds, schema.ti_y_free, schema.sample_dim, sample_index), dtype=bool
    ).ravel()
    # The two masks must have the same number of knots.
    if x_free.shape != y_free.shape:
        raise ValueError("ion_temp x_free / y_free length mismatch")
    # Bundle the masks + ranges into the immutable schema object.
    return TiLabelSchema(
        x_free=x_free,
        y_free=y_free,
        x_range=schema.ti_x_range,
        y_range=schema.ti_y_range,
    )


# Scale physical values into [0, 1] given a (lo, hi) range.
def _normalize(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # Guard against a zero-width range (would divide by zero); else min-max.
    return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)


# The inverse of _normalize: turn [0, 1] values back into physical units.
def _denormalize(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return v * (hi - lo) + lo


# Turn ONE xarray sample into the flat, normalized target vector the CNN learns.
def encode_sample(
    ds,                                        # the xarray Dataset
    label_schema: TiLabelSchema,               # which knots are free + ranges
    schema: XarraySchema = DEFAULT_SCHEMA,     # variable names in the xarray
    sample_index: int | None = None,           # which sample (None = auto/0)
) -> np.ndarray:
    """Xarray sample -> flat, normalized ion-temp target vector (float32)."""
    # If no sample was specified but the data is stacked, default to sample 0.
    if sample_index is None and schema.sample_dim in ds.dims:
        sample_index = 0
    # Read this sample's knot x-positions as a 1-D float array.
    x_knots = np.asarray(
        _get(ds, schema.ti_x_knots, schema.sample_dim, sample_index),
        dtype=np.float64,
    ).ravel()
    # Read this sample's knot y-values as a 1-D float array.
    y_knots = np.asarray(
        _get(ds, schema.ti_y_knots, schema.sample_dim, sample_index),
        dtype=np.float64,
    ).ravel()

    # Allocate the output vector (length = number of free coords).
    out = np.empty(label_schema.size, dtype=np.float32)
    # `n` marks the boundary between the x-block and the y-block in the vector.
    n = label_schema.n_x_free
    # First n slots: the free x-knots, normalized. (`*x_range` unpacks lo, hi.)
    out[:n] = _normalize(
        x_knots[label_schema.x_free], *label_schema.x_range
    )
    # Remaining slots: the free y-knots, normalized.
    out[n:] = _normalize(
        y_knots[label_schema.y_free], *label_schema.y_range
    )
    # Return the assembled label vector.
    return out


# Turn a CNN output vector back into full knot arrays (with fixed knots filled
# in) so you can rebuild and plot the ion-temperature curve.
def decode_targets(
    vector: np.ndarray,                # the CNN's predicted (or true) vector
    label_schema: TiLabelSchema,       # which knots are free + ranges
    x_knots_template: np.ndarray,      # a full knot-x array to copy fixed values
    y_knots_template: np.ndarray,      # a full knot-y array to copy fixed values
) -> dict[str, np.ndarray]:
    """Flat normalized target vector -> ion-temp knot arrays (physical units).

    Fixed coordinates are taken from the template knot arrays (any sample with
    the same layout works: fixed values x=0, x=1, edge y=0 are constant).
    Free x-knots are re-sorted so the reconstructed spline stays valid.

    Returns {"x_knots": ..., "y_knots": ...} ready for the PCHIP/Hermite
    spline builders used for plotting.
    """
    # Make sure the vector is a flat 1-D float array.
    vector = np.asarray(vector, dtype=np.float64).ravel()
    # Start from copies of the template knot arrays so we keep the fixed knots
    # (x=0, x=1, edge y=0) and only overwrite the free ones. `.copy()` avoids
    # mutating the caller's arrays.
    x_knots = np.asarray(x_knots_template, dtype=np.float64).copy()
    y_knots = np.asarray(y_knots_template, dtype=np.float64).copy()

    # Same split point as in encode_sample: first n entries are x, rest are y.
    n = label_schema.n_x_free
    # Convert the normalized x-block back to physical units (rho).
    x_pred = _denormalize(vector[:n], *label_schema.x_range)
    # Convert the normalized y-block back to physical units (keV).
    y_pred = _denormalize(vector[n:], *label_schema.y_range)

    # Drop the predicted x-values into the free x-knot slots.
    x_knots[label_schema.x_free] = x_pred
    # Drop the predicted y-values into the free y-knot slots.
    y_knots[label_schema.y_free] = y_pred

    # A spline needs its x-knots sorted; sort x and reorder y to match so the
    # curve is always valid even if predicted x's came out slightly out of order.
    order = np.argsort(x_knots)
    return {"x_knots": x_knots[order], "y_knots": y_knots[order]}
