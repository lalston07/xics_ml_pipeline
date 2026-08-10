# XICS Ion-Temperature CNN — Observations Report

**Author:** L. Alston
**Project:** Predicting W7-X XICS ion-temperature spline knots from synthetic
detector images (SULI 2026)
**Scope of this report:** the effect of 
(1) increasing the training-set size and
(2) a physics-informed change to the network architecture.

---

## 1. Problem summary

The goal is a convolutional neural network (CNN) that takes a **XICS detector
image** and predicts the **ion-temperature (Ti) profile** of the plasma,
represented as the free knots of a PCHIP spline.

- **Input:** a 2-D detector image, built by binning the ray–detector
  intersection points (already in local detector coordinates) from the XICSRT
  forward model into a pixel grid. Images are resized to 512 × 128 before the
  network.
- **Output:** 7 numbers — the *free* spline knots of the Ti profile
  (3 free knot x-positions in normalized flux radius ρ, and 4 free knot
  y-values in eV). The fixed knots (axis at ρ = 0, edge at ρ = 1 with Ti = 0)
  are filled in afterward, and the full knot set is used to reconstruct the
  Ti(ρ) curve.
- **Task type:** this is a **regression / inverse problem** (image → continuous
  profile parameters), not classification.

This work is methodologically similar to the FSIM_NN / FIDASIM neural-network
surrogate approach: a physics forward model generates synthetic diagnostic data,
and a network learns the inverse mapping back to the plasma profile.

---

## 2. Method and evaluation

### 2.1 Data splits

Each dataset is split into three groups, using a fixed random seed so the split
is reproducible:

- **Train** — the network updates its weights on these.
- **Validation** — checked after every epoch to select the best model and detect
  overfitting; the network never trains on these.
- **Test** — held out entirely and used only for the final evaluation.

### 2.2 "Best model" selection

Training saves a checkpoint every time the **validation loss** reaches a new
low, so the saved model is the epoch that generalized best — not necessarily the
last epoch. This protects against overfitting late in training.

### 2.3 Evaluation metric

The headline metric is **profile-space RMSE (eV)**: for each test case the
predicted knots and the true knots are each used to rebuild the Ti(ρ) curve via
PCHIP interpolation (the same interpolation used to generate the data), and the
two curves are compared on a dense 201-point ρ grid. This measures the error in
the physical quantity of interest — the reconstructed temperature profile — and
is directly comparable between models. Supporting metrics are the knot-level
RMSE split into x-knots (ρ) and y-knots (eV).

All models were evaluated with the identical procedure (`evaluate_report`), each
on its own held-out test set.

---

## 3. The three models

| Model | Training data | Architecture | Trainable params |
|-------|---------------|--------------|------------------|
| **v00** | 100 cases (80 train / 10 val / 10 test) | baseline CNN, symmetric pooling | ~1.24 M |
| **v01** | 1000 cases (800 train / 100 val / 100 test) | baseline CNN, symmetric pooling | ~1.24 M |
| **v02** | 1000 cases (800 train / 100 val / 100 test) | physics-informed, spatial-preserving | ~1.70 M |

- **v00 → v01** isolates the effect of **training-set size** (same architecture,
  10× more data).
- **v01 → v02** isolates the effect of the **architecture change** (same data,
  different network).

### 3.1 Baseline architecture (v00, v01)

A compact VGG-style encoder: four convolutional blocks, each with two 3×3
convolutions (Conv → BatchNorm → ReLU, twice) followed by a **2×2 max-pool that
halves both image axes equally**. Channels double each block
(1 → 32 → 64 → 128 → 256). A global average pool then collapses each feature map
to a single value, and a small fully-connected head outputs the 7 knots with a
final sigmoid (matching the [0, 1]-normalized labels).

### 3.2 Physics-informed architecture (v02)

The key observation motivating v02 is that the two image axes are **physically
different**:

- The **spectral (column) axis** carries the Doppler *broadening* of the line,
  which encodes temperature.
- The **spatial / line-of-sight (row) axis** corresponds to different radial
  positions in the plasma, and therefore carries the *profile-shape*
  information — exactly what we are trying to predict.

The baseline network pooled both axes equally, discarding spatial resolution
just as fast as spectral resolution. In v02 the pooling is made **asymmetric**:
each block pools only the **spectral** axis and **preserves the full spatial
resolution**. The bridge to the head then keeps **8 spatial bins** (rather than
averaging everything to a single value), so the prediction head can use how the
features vary along the spatial/radial direction. This raised the parameter
count from ~1.24 M to ~1.70 M (a wider first head layer).

---

## 4. Results

### 4.1 Headline comparison (profile-space RMSE)

| Model | Data | Architecture | Mean RMSE (eV) | Median (eV) | Best (eV) | Worst (eV) | Std (eV) |
|-------|------|--------------|---------------:|------------:|----------:|-----------:|---------:|
| v00 | 100  | baseline | **544.2** | 463.7 | 129.5 | 1502.5 | 364.1 |
| v01 | 1000 | baseline | **466.6** | 432.5 | 80.2  | 1575.6 | 278.6 |
| v02 | 1000 | spatial-preserving | **439.5** | 408.7 | 59.0 | 1540.0 | 274.2 |

*(v00 is evaluated on 10 test cases; v01 and v02 on 100 test cases each. The
three test sets are different samples, so comparisons are of aggregate error
statistics rather than case-by-case.)*

### 4.2 Supporting knot-level metrics

| Model | x-knot RMSE (ρ) | y-knot RMSE (eV) | Best epoch | Best val loss |
|-------|----------------:|-----------------:|-----------:|--------------:|
| v00 | 0.1234 | 606.7 | 26 | 0.02795 |
| v01 | 0.1073 | 496.9 | 21 | 0.01812 |
| v02 | 0.1058 | 467.3 | 13 | 0.01914 |

---

## 5. Observations

### 5.1 Effect of training-set size (v00 → v01)

Increasing the dataset from 100 to 1000 cases produced the **largest single
improvement**:

- Mean profile RMSE dropped from **544 → 467 eV (≈ 14 %)**.
- The spread tightened substantially: standard deviation fell from 364 → 279 eV,
  and the best case improved from 130 → 80 eV.
- The validation loss of the best model roughly halved (0.0280 → 0.0181).

This is consistent with the v00 model being **data-starved**: with only 80
training samples for a ~1.24 M-parameter network, it could not see enough
variety of profiles and tended to predict something close to the *average*
profile. More data gave it the variety needed to do better across the board.

Interestingly, the **worst-case** error did **not** improve with more data
(1503 → 1576 eV). The hardest cases are extreme profiles (e.g. very hot,
sharply-peaked cores near the top of the 0–5000 eV range); these remain rare in
the training set even at 1000 cases, so the network is still cautious about
them.

### 5.2 Effect of the physics-informed architecture (v01 → v02)

With the dataset held fixed at 1000 cases, preserving the spatial axis and
feeding it to the prediction head gave a **further, consistent improvement**:

- Mean profile RMSE dropped from **467 → 440 eV (≈ 6 %)**.
- **Every** summary statistic improved: median 432 → 409 eV, and notably the
  **best case fell from 80 → 59 eV (≈ 26 %)**.
- The y-knot (temperature) RMSE improved from 497 → 467 eV.

The interpretation is that the spatial/line-of-sight axis genuinely carries
profile-shape information that the baseline network was discarding during
pooling. Keeping that resolution and exposing it to the head let the model make
sharper predictions, especially on the easier cases.

### 5.3 Combined effect and ranking of what mattered

The two changes **stack**: profile RMSE improved 544 → 467 → 440 eV across the
three models. In terms of impact:

1. **More training data** was the bigger lever (≈ 14 % improvement).
2. **The physics-informed architecture** added a further ≈ 6 % on top.

A convenient way to read the current best model (v02): a mean profile RMSE of
~440 eV against a temperature scale of 0–5000 eV is roughly **9 % of full
scale**, with the best individual reconstructions near ~1 % (59 eV).

### 5.4 Overfitting behavior

All models showed the classic overfitting signature: training loss kept
decreasing steadily while validation loss bottomed out early (best epochs were
26, 21, and 13 respectively) and then drifted upward and became noisy. Because
the pipeline saves the *best-validation* checkpoint rather than the last epoch,
the deployed models correspond to those early best epochs, not the overfit later
ones. Early stopping was implemented to halt training automatically at the best
point; it was turned off for these runs only so the full loss curves could be
observed.

### 5.5 The persistent limitation: extreme profiles

Across all three models, the **worst-case error stayed around ~1500 eV** and was
consistently driven by extreme (very hot / sharply-peaked) profiles. Neither
more data nor the architecture change resolved this. The most likely cause is
that such profiles are underrepresented in the randomly generated training set,
so the network rarely learns them. This points to the most promising future
work.

---

## 6. Recommendations / future work

In rough order of expected impact:

1. **More training data, weighted toward the extremes.** The largest gains came
   from more data, and the remaining failures are on rare profiles. Generating
   additional cases — or deliberately over-sampling hot / sharply-peaked cores —
   should directly attack the worst-case error.
2. **Noise / dropout augmentation** (as in FSIM_NN). Adding realistic detector
   noise during training would both improve robustness on real data and
   effectively enlarge the dataset.
3. **Further architecture exploration.** The spatial-preserving change helped;
   natural next steps are tuning the number of spatial bins (4 / 8 / 16) and
   experimenting with how aggressively each block pools the spectral vs. spatial
   axes.
4. **Predict along the radial axis directly (longer-term).** Because the spatial
   axis maps to radial position, a natural extension is to predict Ti as a
   function of ρ along that axis, rather than only the spline knots. This would
   require a detector-row → ρ mapping and per-position supervision, which the
   current dataset does not yet provide.
5. **Regularization for the larger model.** The v02 head added parameters; if
   overfitting worsens, increasing dropout / weight decay or reducing model size
   are straightforward levers.

---

## 7. Reproducibility notes

- **Environment:** Python (Anaconda `desc-env`), PyTorch with CUDA 11.8, trained
  on an NVIDIA RTX 3050 Ti Laptop GPU (4 GB). Batch size 8 was used for v02 to
  fit the preserved-spatial feature maps in 4 GB of VRAM.
- **Data files:** `xicsrt_training_set_v00.nc` (100 cases),
  `xicsrt_training_set_v01.nc` (1000 cases). Both store the ray intersections
  (`intersect`, in local detector coordinates) and the ground-truth spline knots
  under the flattened `config__sources__plasma__profile_ion_temp__*` variables.
- **Code:** the `xicsrt_cnn/` package. Train with
  `python -m xicsrt_cnn.run_training`; evaluate with
  `python -m xicsrt_cnn.evaluate_report --checkpoint <ckpt> --tag <name>`;
  inspect a single case with
  `python -m xicsrt_cnn.inspect_case <sample_index> --checkpoint <ckpt>`.
- **Saved evaluation reports:** `eval_v00_100cases.json`,
  `eval_v01_1000cases.json`, `eval_v02_spatial.json` (kept locally under
  `xicsrt_cnn/checkpoints/`).
- **Note on one saved file:** `eval_v01_1000cases.json` records its data file as
  `..._v00.nc`; this is a cosmetic labeling artifact from an earlier version of
  the evaluation script (it printed the config default rather than the file the
  checkpoint actually used). v01 was in fact trained and tested on the 1000-case
  file. The labeling was subsequently corrected in `evaluate_report.py`.
```
Profile RMSE (eV), lower is better:

  v00 (100 cases, baseline)          |##################  544
  v01 (1000 cases, baseline)         |###############     467
  v02 (1000 cases, spatial-aware)    |##############      440
```
