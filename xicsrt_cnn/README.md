# cnn

A CNN pipeline that predicts **ion-temperature spline knots** from an XICS
**detector image**.

- **Input**  = detector image, built by binning ray-detector intersection
  points (already in local coordinates) into pixels.
- **Output** = the free ion-temperature spline knot locations (the ground
  truth comes from the same xarray).

Everything reads from a single **xarray Dataset** (`.nc` file) that holds both
the intersections and the knots.

---

## The big picture

```
 xarray (.nc file)                          CNN                    knots you plot
┌──────────────────┐                   ┌──────────┐            ┌──────────────────┐
│ ray intersections│──build image────▶ │ XicsCNN  │──▶ vector ─│ decode → x_knots │
│  (local coords)  │   (image.py)      │(model.py)│  (numbers) │        y_knots   │
│                  │                                            └──────────────────┘
│ ion-temp knots   │──encode label───▶ compared to prediction        (labels.py)
│  (ground truth)  │   (labels.py)         via MSE loss
└──────────────────┘                       (train.py)
```

Two directions through the same schema:

- **Training**: image -> CNN -> predicted vector, compared against the
  *encoded* true knots.
- **Prediction / plotting**: image -> CNN -> vector -> *decoded* back into knot
  arrays.

---

## Where to start reading

Read the files in this order (easy -> hard). Each builds on the previous one.

| # | File | Lines | What it is |
|---|------|-------|------------|
| 1 | `schema.py` | ~83 | **Start here.** Just names + numbers: which xarray variables hold the intersections and the ion-temp knots, plus physical ranges. Everything refers back to this. |
| 2 | `config.py` | ~91 | All the knobs: data path, image size, model size, learning rate, epochs. Pure settings, no logic. |
| 3 | `image.py` | ~141 | The input side. `build_detector_image` = a 2-D histogram counting ray hits per pixel. |
| 4 | `labels.py` | ~233 | **The key concept file.** `encode_sample` turns knots -> a flat number vector (training target); `decode_targets` turns the CNN's numbers back -> knots. |
| 5 | `model.py` | ~95 | The CNN itself. `ConvBlock` is one repeated unit; `XicsCNN` stacks them + a small head. Can be a black box at first. |
| 6 | `dataset.py` | ~238 | The glue: `XicsXarrayDataset.__getitem__` returns one `(image, target)` pair by combining `image.py` + `labels.py`. |
| 7 | `train.py` | ~292 | **The entry point.** `train()` ties it all together; `predict_sample()` runs a trained model. |
| 8 | `__init__.py` | ~66 | Reference only. Lists what the package exports. |

### Fastest way in

Open **`train.py`** and read the `train()` function top to bottom. It calls, in
order:

1. `discover_samples()` - find the data
2. `build_ti_schema()` - figure out the label layout
3. `make_loaders()` - wrap data into batches
4. `XicsCNN(...)` - build the model
5. the epoch loop - train it
6. `torch.save(...)` - save the best model

Following those calls into the other files is the quickest way to see how
everything connects. `train()` is the map; the other files are the territory.

---

## How to run it

1. **When the mentor's data arrives**, open `cnn/schema.py` and set the variable
   names to match the delivered xarray. (This is the only file you should need
   to touch for a schema change.)

2. In `cnn/config.py`, point `DataConfig.xarray_path` at the data file
   (a single `.nc` file, or a directory of per-sample `.nc` files).

3. From the `xics_ml_pipeline` directory, train:

   ```
   python -m cnn.train
   ```

   This trains and saves the best model to `cnn/checkpoints/best.pt`.

4. To see a prediction vs. the ground truth:

   ```python
   from cnn import predict_sample

   pred, truth = predict_sample("cnn/checkpoints/best.pt")
   # pred["x_knots"],  pred["y_knots"]
   # truth["x_knots"], truth["y_knots"]
   ```

   `pred` and `truth` are dicts of full knot arrays (fixed knots filled in),
   ready to feed into a PCHIP / CubicHermiteSpline to rebuild and plot the
   ion-temperature profile.

---

## Notes / assumptions

- **Ion temperature only** for now. The label layout is fixed by the xarray's
  `x_free` / `y_free` masks (which knots the CNN predicts vs. which are fixed at
  x=0, x=1, edge y=0).
- **Local coordinates**: the machine -> local coordinate conversion happens
  **upstream** (in the mentor's code). This package only bins the local-coord
  intersections into an image.
- **No external XICS imports**: this package depends only on the standard
  library, `numpy`, `torch`, and `xarray` - nothing from the other XICS repos.
- **Loss** is MSE on the normalized knot vector (knot-space). A profile-space
  loss (sampling the reconstructed spline on a rho grid) is a planned upgrade;
  `decode_targets` in `labels.py` already provides what that would need.
