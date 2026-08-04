"""
PyTorch Dataset that reads XICS training samples from xarray.

Each sample is (image_tensor, target_vector):
    image_tensor : float32 [1, H, W], detector image built from local-coord
                   ray intersections (see image.py), normalized to [0, 1]
    target_vector: float32 [n_targets], free ion-temp knot coords (labels.py)

Two storage layouts are supported:
  1. A single .nc file with a `sample_dim` indexing many raytraces.
  2. A directory of per-sample .nc files (one raytrace each), matched by glob.

The mentor's delivered dataset determines which; both are handled transparently.
"""

# Enable modern type-hint syntax on older Python versions.
from __future__ import annotations

# `Path` handles filesystem paths portably.
from pathlib import Path

# NumPy for arrays; torch for tensors; xarray to read the .nc data.
import numpy as np
import torch
import xarray as xr
# The base class every PyTorch dataset inherits from.
from torch.utils.data import Dataset

# Our own config + helpers.
from .config import DataConfig
from .image import ImageConfig, build_detector_image, resize_image
from .labels import TiLabelSchema, encode_sample
from .schema import XarraySchema


# ---------------------------------------------------------------------------
# Sample references: (dataset-or-path, sample_index_or_None)
# ---------------------------------------------------------------------------
# A tiny object that says "this sample lives in file `path`, at position
# `sample_index` (or the whole file if sample_index is None)".
class SampleRef:
    """Points to one sample, either inside a stacked Dataset or a file."""

    # `__slots__` restricts the attributes to just these two, saving memory and
    # preventing accidental typos creating new attributes.
    __slots__ = ("path", "sample_index")

    def __init__(self, path: Path, sample_index: int | None):
        # The .nc file this sample comes from.
        self.path = path
        # Which index along the sample dimension (None = single-sample file).
        self.sample_index = sample_index

    def __repr__(self) -> str:
        # Friendly printout, e.g. SampleRef(training.nc, i=3).
        return f"SampleRef({self.path.name}, i={self.sample_index})"


# Thin wrapper around xarray's open function (one place to change if needed).
def _open(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


# Scan the configured data location and return a list of every sample, plus one
# open Dataset to use as a reference for building the schema.
def discover_samples(data_cfg: DataConfig) -> tuple[list[SampleRef], xr.Dataset]:
    """Enumerate all samples and return them plus a reference Dataset.

    The reference Dataset is used to build the label schema and image/geometry
    defaults. Caller is responsible for the returned Dataset's lifetime.
    """
    # Where to look, and the variable-name schema.
    path = Path(data_cfg.xarray_path)
    schema = data_cfg.schema

    # CASE 1: a directory of per-sample .nc files.
    if path.is_dir():
        # Find and sort all matching files.
        files = sorted(path.glob(data_cfg.file_glob))
        # Bail out clearly if the directory has none.
        if not files:
            raise RuntimeError(f"No '{data_cfg.file_glob}' files in {path}")
        # One SampleRef per file (sample_index None = whole file is one sample).
        refs = [SampleRef(f, None) for f in files]
        # Open the first file to serve as the schema reference.
        ref_ds = _open(files[0])
        return refs, ref_ds

    # If it's neither a directory nor an existing file, explain how to fix it.
    if not path.exists():
        raise RuntimeError(
            f"xarray_path does not exist: {path}. Point DataConfig.xarray_path "
            f"at the delivered dataset (a .nc file or a directory of .nc files)."
        )

    # CASE 2: a single .nc file.
    ds = _open(path)
    # If it has a sample dimension, make one SampleRef per index along it.
    if schema.sample_dim in ds.dims:
        n = ds.sizes[schema.sample_dim]
        refs = [SampleRef(path, i) for i in range(n)]
    # Otherwise the whole file is a single sample.
    else:
        refs = [SampleRef(path, None)]
    return refs, ds


# Build an ImageConfig for one dataset, overriding detector geometry with any
# values stored in the xarray (falling back to the defaults in `base`).
def _image_config_from_ds(ds: xr.Dataset, base: ImageConfig, schema: XarraySchema) -> ImageConfig:
    """Override ImageConfig detector geometry from the xarray if present."""
    # Small local helper: read a scalar variable if it exists, else return None.
    def scalar(name):
        if name in ds.variables or name in ds.coords:
            # `.ravel()[0]` grabs the single value even if it's 0-D/1-D.
            return float(np.asarray(ds[name].values).ravel()[0])
        return None

    # Try to read detector width, height, and pixel size from the xarray.
    xs = scalar(schema.det_xsize)
    ys = scalar(schema.det_ysize)
    px = scalar(schema.det_pixel_size)
    # Build a new ImageConfig, using the xarray value where present, else `base`.
    return ImageConfig(
        xsize=xs if xs is not None else base.xsize,
        ysize=ys if ys is not None else base.ysize,
        pixel_size=px if px is not None else base.pixel_size,
        shape=base.shape,
        normalize=base.normalize,
    )


# Read the ray intersections for one sample and turn them into a finished image.
def build_image_from_sample(
    ds: xr.Dataset,                 # the open xarray Dataset
    sample_index: int | None,       # which sample (None = whole file)
    data_cfg: DataConfig,           # names + image settings
) -> np.ndarray:
    """Read local-coord intersections for one sample and bin into an image."""
    # The variable-name schema.
    schema = data_cfg.schema

    # Local helper: fetch a variable and slice out this sample if needed.
    def get(name):
        da = ds[name]
        if sample_index is not None and schema.sample_dim in da.dims:
            da = da.isel({schema.sample_dim: sample_index})
        return np.asarray(da.values).ravel()

    # Ray hit x-positions and y-positions on the detector (local coords).
    lx = get(schema.intersect_x)
    ly = get(schema.intersect_y)

    # Figure out detector geometry (from xarray if available).
    img_cfg = _image_config_from_ds(ds, data_cfg.image, schema)
    # Bin the hits into a 2-D image.
    image = build_detector_image(lx, ly, img_cfg)
    # Resize to the fixed CNN input size (no-op if resize_to is None/matches).
    image = resize_image(image, data_cfg.resize_to)
    return image


# The actual PyTorch Dataset: given a list of SampleRefs, it yields
# (image_tensor, target_tensor) pairs on demand.
class XicsXarrayDataset(Dataset):
    def __init__(
        self,
        refs: list[SampleRef],          # which samples this dataset covers
        label_schema: TiLabelSchema,    # free/fixed knot layout for labels
        data_cfg: DataConfig,           # paths, names, image settings
        cache: bool = True,             # keep computed items in memory?
    ):
        # Store the inputs on the instance for use in __getitem__.
        self.refs = refs
        self.label_schema = label_schema
        self.data_cfg = data_cfg
        self._cache = cache
        # Cache of opened Datasets keyed by file path (avoid re-opening files).
        self._ds_cache: dict[str, xr.Dataset] = {}
        # Cache of finished (image, target) tensors keyed by item index.
        self._item_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # Open a file once and reuse it on later calls.
    def _get_ds(self, path: Path) -> xr.Dataset:
        key = str(path)
        if key not in self._ds_cache:
            self._ds_cache[key] = _open(path)
        return self._ds_cache[key]

    # PyTorch calls this to learn how many samples exist.
    def __len__(self) -> int:
        return len(self.refs)

    # PyTorch calls this to fetch sample number `i`.
    def __getitem__(self, i: int):
        # Return the cached tensors if we've built this item before.
        if self._cache and i in self._item_cache:
            return self._item_cache[i]

        # Which sample this is, and its source file.
        ref = self.refs[i]
        ds = self._get_ds(ref.path)

        # Build the input image and add a channel dimension -> [1, H, W].
        image = build_image_from_sample(ds, ref.sample_index, self.data_cfg)
        image_t = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]

        # Build the label vector (free knot coords) and make it a tensor.
        vec = encode_sample(
            ds, self.label_schema, self.data_cfg.schema, ref.sample_index
        )
        target_t = torch.from_numpy(vec)

        # Save to cache if enabled, then return the pair.
        if self._cache:
            self._item_cache[i] = (image_t, target_t)
        return image_t, target_t


# Split the samples into training and validation sets, reproducibly.
def split_samples(
    refs: list[SampleRef], val_fraction: float, seed: int
) -> tuple[list[SampleRef], list[SampleRef]]:
    """Deterministic train/val split of samples."""
    # A seeded random generator so the split is the same every run.
    rng = np.random.default_rng(seed)
    # Random ordering of sample indices.
    order = rng.permutation(len(refs))
    # How many go to validation: fraction of total, at least 1 if we have >1
    # sample; 0 if there's only a single sample (nothing to hold out).
    n_val = max(1, int(round(len(refs) * val_fraction))) if len(refs) > 1 else 0
    # The first n_val shuffled indices are the validation set.
    val_idx = set(order[:n_val].tolist())
    # Walk all samples and drop each into train or val.
    train, val = [], []
    for i, r in enumerate(refs):
        (val if i in val_idx else train).append(r)
    return train, val
