# Experiment Feature Audit & Improvement Plan

A systematic walk through every component of the current pipeline. Each section captures **what the feature is**, **its current values**, **known concerns**, and the **decision / plan** going forward.

This file is built up feature-by-feature; later sections will be appended as the discussion continues.

**Current best result:** test ET Dice = **0.7341** (3D U-Net + et-heavy weights + CLAHE preprocessing).

---

## Feature 1 — Dataset

### Summary
The raw substrate the model learns from. Quality, scale, and label scheme of this dataset bound everything downstream.

### Current values
| property | value |
|---|---|
| Dataset | BraTS 2020 (Kaggle `BraTS2020_training_data/content/data`) |
| Storage format | per-slice HDF5 files (Kaggle re-packaging of original NIfTI) |
| Total slice files | 57,195 |
| Patient volumes | 369 |
| Slices per volume | ≈ 155 |
| Spatial resolution per slice | 240 × 240 |
| Volume shape | 240 × 240 × 155 |
| Modalities | 4 — T1, T1ce, T2, FLAIR |
| Label scheme | label-id integers: 0 = background, 1 = NCR, 2 = ED, 4 = ET |
| Co-registration | done by organisers |
| Skull-stripping | done by organisers |
| Bias-field correction | partially done by organisers (validated by N4 ablation: re-applying N4 hurt ET by −0.0038) |

### What it controls
- Upper bound on what the model can learn (label quality, modality coverage, patient diversity).
- Label scheme dictates target construction (`build_multiclass_target` maps label-id → WT/TC/ET binary channels).
- h5 slice format forces a per-volume aggregation step in `load_raw_volume()`.

### Known concerns
1. "Test set" here = 75 patients held out from the public training labels — not the official BraTS leaderboard test set (which is unlabelled and held by organisers).
2. Severe class imbalance: ET ≈ 0.3–1 % of brain voxels, < 0.005 % of total volume voxels.
3. No external dataset (BraTS 2021/2023) for cross-validation of generalisation.

### Decision (recorded)
**Keep as-is.** No concerns flagged. BraTS 2020, h5 format, 369 patient volumes, 4 modalities, integer-label scheme remain the foundation for all subsequent experiments.

### Plan
*(no change planned for Feature 1)*

---

## Feature 2 — Train / Val / Test split

### Summary
How the 369 patient volumes are partitioned into the sets used for training, validation (checkpoint selection + early stop), and test (final thesis Dice number).

### Current values
| property | value |
|---|---|
| Split function | `split_by_volume(file_list, seed=42, ratios=(0.6, 0.2, 0.2))` |
| Random seed | 42 (fixed) |
| Split level | patient / volume (no slice-level leakage) |
| Train / Val / Test | 221 / 73 / 75 volumes (60 / 20 / 20 %) |
| Cross-validation | none — single split |
| Stratification | none — random over patient IDs |
| Test reporting | single point estimate (no confidence intervals) |

### Known concerns
1. Single split → reported test number is one specific 75-volume sample; ±0.01–0.02 variance with a different seed.
2. No stratification by tumour grade (HGG vs LGG) — random split could drift the high-grade / low-grade ratio across sets.
3. Wide test-set confidence interval at n=75 — two methods reporting 0.73 vs 0.75 may not be statistically distinguishable.
4. No external dataset for cross-dataset generalisation.

### Decision (recorded)
**Three changes to apply (in order of effort):**
1. **Switch to 5-fold cross-validation.** Every patient appears in test exactly once across the 5 folds. Reported number becomes mean ± std over the 5 fold-tests. Standard practice for BraTS-class submissions.
2. **Add bootstrap confidence intervals** (95 % CI) on per-volume Dice for each fold's test report.
3. **Adjust ratios to 70 / 20 / 10** within each fold's hold-out structure (more training data per fold; smaller per-fold test, but mean over 5 folds still uses 100 % of patients).

### Plan

**Step 1 — Implement 5-fold split**
- Modify `split_by_volume` (or add a new `kfold_by_volume(file_list, k=5, seed=42, train_val_ratio=(0.7, 0.2))`) so each fold returns:
  - Train: 70 % of (this fold's train+val pool)
  - Val: 20 % of (this fold's train+val pool)
  - Test: the held-out fold (~74 patients = 1/5 of 369)
- Volume-level shuffling done once with seed=42 and partitioned into 5 contiguous fold-blocks of ~74 patients.
- Within each fold, the remaining ~295 patients are split 70/20 (~206 train, ~58 val).

**Step 2 — Wrap training in a fold loop**
- Add `run_kfold(plan_runner, k=5)` that calls each fold sequentially, restoring CFG between folds.
- Each fold writes its own tagged checkpoints (`singlestage3d_best__fold0.pth` … `__fold4.pth`).
- Each fold writes its own per-volume Dice JSON.
- After 5 folds, aggregate into a single results JSON containing per-fold means + overall mean + 95 % CI.

**Step 3 — Bootstrap CI helper**
- Add `bootstrap_ci(per_volume_dice, n_iters=2000, alpha=0.05)` that resamples the per-volume Dice list with replacement, recomputes the mean, and returns the (2.5 %, 97.5 %) quantiles.
- Apply at fold-level (CI within a fold) and at study-level (CI across all 5 folds combined into one ~369-element pooled list).

**Step 4 — Update reporting**
- Thesis Results tables to read: "0.7341 ± 0.018 (95 % CI [0.716, 0.752])" instead of "0.7341" alone.
- Each ablation row + each architecture row gets the same CI treatment.

### Cost estimate
- One 5-fold run = ~5 × current single-fold runtime ≈ 15 h per ablation row (3 h × 5).
- Bootstrap CI: < 1 second per fold (cheap post-hoc).
- Re-running the existing two-axis ablation under 5-fold: 9 rows × 15 h = ~135 hours (≈ 6 days continuous).
- **Practical mitigation:** run 5-fold only on the *winning configuration* (et_heavy + CLAHE) to give the headline thesis number with proper CIs. Other ablation rows can stay as single-fold for the comparison columns.


---

## Feature 3 — Patch sampling

### Summary
How 3D sub-volumes are extracted from full patient volumes during training. The model never sees a full 240×240×155 volume — only 3D patches.

### Current values
| property | value |
|---|---|
| Patch sampler | `monai.transforms.RandCropByPosNegLabeld` |
| `spatial_size` | (96, 96, 96) |
| `pos` / `neg` ratio | 1.0 / 0.0 (every patch centred on a tumour voxel; no random patches) |
| `num_samples` per volume | 2 |
| `image_threshold` | 0.0 |
| `label_key` | uses any tumour voxel (label > 0) as positive centre |
| Effective batch / step | 1 volume × 2 patches = 2 patches |
| Updates / epoch | ≈ 221 |
| Patches / epoch | ≈ 442 |

### Known concerns
1. **96³ patches are below BraTS standard (128³)** — accounts for ~0.02 of the ET gap to SOTA. Forced by 6 GB VRAM.
2. **`samples_per_volume=2` gives only 442 updates/epoch** — nnU-Net does ~4× more.
3. **`pos=1.0, neg=0.0` is too aggressive** — model never trains on pure-background regions but sees them at sliding-window inference.
4. **Tumour-centric patches centre on *any* tumour voxel** (mostly edema since edema dominates WT volume) — disproportionately few patches are ET-centred.

### Decision (recorded)
**Three changes to apply jointly (most senior-ML-defensible move):**
1. **Patch size 96³ → 128³** (canonical BraTS scale).
2. **Add gradient checkpointing** on encoder blocks (`torch.utils.checkpoint`) to fit 128³ in 6 GB VRAM.
3. **Sampling ratio `pos=1.0, neg=0.0` → `pos=0.67, neg=0.33`** (2:1 foreground oversample; standard practice now that GroupNorm-num_groups=1 has eliminated the InstanceNorm-on-empty-bg pathology).
4. **`samples_per_volume` 2 → 4** (doubles effective updates/epoch, brings the model closer to nnU-Net training intensity).

Expected combined ET gain: **+0.02–0.05** (each change contributes ~+0.01).

### Plan

**Step 1 — Update CFG**
```python
CFG['patch_size']         = (128, 128, 128)
CFG['sw_roi']             = (128, 128, 128)   # match training receptive field
CFG['pos_ratio']          = 0.67
CFG['neg_ratio']          = 0.33
CFG['samples_per_volume'] = 4
```

**Step 2 — Add gradient checkpointing to `build_model` (UNet path)**
- Wrap each `down_layer` in `torch.utils.checkpoint.checkpoint(...)` so activations are recomputed during backward instead of stored.
- Trade-off: ~25–35 % training-time slowdown per epoch in exchange for ~50 % less activation memory.
- Implementation reference: PyTorch checkpoint docs; MONAI's UNet does *not* expose a `gradient_checkpointing=True` flag, so this requires a small subclass that overrides `forward` to apply checkpointing.

Sketch:
```python
from torch.utils.checkpoint import checkpoint as _ckpt
class CheckpointedUNet(MonaiUNet):
    def forward(self, x):
        # Apply checkpoint to the encoder path only (decoder gradients are needed eagerly for skip-conn correctness).
        # Reference each down-block via _ckpt(self.down_layers[i], x, use_reentrant=False)
        ...
```

**Step 3 — Verify VRAM headroom**
- After CFG change, instantiate the model and run one forward+backward at 128³ × bs=1 × samples_per_volume=4 (effective 4 patches per step).
- Target: peak VRAM ≤ 5.5 GB (leaves headroom for fragmentation).
- If OOM: drop `unet_channels` from (16, 32, 64, 128, 160) to (12, 24, 48, 96, 128) before further tuning.

**Step 4 — Re-run baseline + winning ablation**
- Re-run **et_heavy + CLAHE + CHECKPOINTED-128³-UNet** as the new baseline of the upgraded pipeline.
- Compare against the existing `singlestage3d_best__clahe.pth` (test ET = 0.7341) to quantify the patch-size-driven gain.

**Step 5 — Update sliding-window inference**
- Set `CFG['sw_roi'] = (128, 128, 128)` to match the new training receptive field.
- Verify on one val volume that sliding-window output is still consistent with the per-patch outputs (no shape mismatches).

### Cost estimate
- One training run: ~3–4 h (was ~3 h; +25–35 % from checkpointing).
- VRAM verification: 5 minutes.
- Re-running winning configuration only: 1 run × ~4 h = **half a day**.
- Optional re-run of full preprocessing-ablation under the new patch size: 5 rows × 4 h = ~20 h (only worth doing if Step 4 shows ≥ +0.02 ET gain over current 0.7341).

### Risk register
- **Gradient checkpointing might trigger a fresh round of NaN issues** with our existing NaN guards. Mitigation: keep all 3 NaN guards (logits, loss, grads) — they catch problems regardless of checkpointing.
- **128³ patches may not fit even with checkpointing.** Mitigation: incremental fallback — try 112³, then 104³, then 96³ + samples_per_volume=4 alone.
- **`neg=0.33` may slow early-epoch convergence** since the loss now sees easy "all-background" patches. Locked epoch budget is **50 epochs** per ablation row (after the B1 timing audit identified validation as the dominant cost at val_every=5).


---

## Feature 4 — Data augmentation

### Summary
Stochastic transforms applied to each training patch (post-crop, pre-model). Primary regulariser; primary driver of generalisation in 3D medical segmentation.

### Current values
| transform | parameters | probability |
|---|---|---|
| `RandFlipd` axis 0 | spatial flip | 0.5 |
| `RandFlipd` axis 1 | spatial flip | 0.5 |
| `RandFlipd` axis 2 | spatial flip | 0.5 |
| `RandScaleIntensityd` | factor 0.1 | 0.5 |
| `RandShiftIntensityd` | offset 0.1 | 0.5 |

Total: 3 spatial flips + 2 mild intensity perturbations. ≈ 20 % of nnU-Net's standard BraTS augmentation set.

### Known concerns
1. No rotations (one of the highest-ROI augmentations for 3D medical seg).
2. No elastic deformation (particularly useful for thin ET ring).
3. No noise / blur / bias-field / gamma augmentation (cross-scanner robustness).
4. Mild intensity perturbation magnitudes vs nnU-Net standard ranges.
5. CPU-bottlenecked at `num_workers=0`; richer augmentation will slow per-epoch wall time ~10–20 %.

### Decision (recorded)
**Adopt the full nnU-Net BraTS augmentation stack.** This is the published, well-cited recipe used by the BraTS 2020 winners. Expected combined ET gain: **+0.02–0.04**.

Specific transforms to add (in addition to current ones):
1. `RandRotate90d` — free 90° axial rotations (no interpolation artefacts)
2. `RandRotated` — continuous rotations ±15° axial, ±10° sagittal/coronal, with `mode=('bilinear', 'nearest')` for image and label respectively
3. `Rand3DElasticd` — sigma_range (5, 13), magnitude_range (50, 250)
4. `RandGaussianNoised` — std 0.01
5. `RandGaussianSmoothd` — sigma_x/y/z (0.5, 1.5)
6. `RandBiasFieldd` — coefficient_range (0.0, 0.2), degree 3
7. `RandAdjustContrastd` — gamma (0.7, 1.5)
8. `RandHistogramShiftd` — num_control_points (5, 10)

### Plan

**Step 1 — Compose the new training transform**
Replace `train_tfm` in the dataset cell with:
```python
train_tfm = Compose([
    EnsureTyped(keys=['image', 'label']),
    RandCropByPosNegLabeld(
        keys=['image', 'label'], label_key='label',
        spatial_size=CFG['patch_size'],
        pos=CFG['pos_ratio'], neg=CFG['neg_ratio'],
        num_samples=CFG['samples_per_volume'],
        image_key='image', image_threshold=0.0,
    ),
    # --- spatial ---
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3, spatial_axes=(1, 2)),
    RandRotated(
        keys=['image', 'label'], prob=0.3,
        range_x=0.26,    # ±15° axial   (in radians)
        range_y=0.17,    # ±10°
        range_z=0.17,
        mode=('bilinear', 'nearest'),
        padding_mode='zeros',
    ),
    Rand3DElasticd(
        keys=['image', 'label'], prob=0.2,
        sigma_range=(5, 13), magnitude_range=(50, 250),
        mode=('bilinear', 'nearest'),
        padding_mode='zeros',
    ),
    # --- intensity (image only) ---
    RandGaussianNoised(keys='image', prob=0.2, std=0.01),
    RandGaussianSmoothd(
        keys='image', prob=0.2,
        sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
    ),
    RandBiasFieldd(keys='image', prob=0.2, degree=3, coeff_range=(0.0, 0.2)),
    RandScaleIntensityd(keys='image', prob=0.5, factors=0.1),
    RandShiftIntensityd(keys='image', prob=0.5, offsets=0.1),
    RandAdjustContrastd(keys='image', prob=0.3, gamma=(0.7, 1.5)),
    RandHistogramShiftd(keys='image', prob=0.2, num_control_points=(5, 10)),
])
```

**Step 2 — Validate on a single patch**
- After updating `train_tfm`, pull `train_ds[0]` once and run `show_training_patch()` to confirm:
  - All channels still finite (no NaN from numerical transforms).
  - Label still aligned to image after spatial transforms.
  - Visual sanity: patch isn't completely corrupted.

**Step 3 — Wall-time impact check**
- Time one full epoch with the new augmentation pipeline.
- Expected: ~+10–20 % per epoch from CPU-side augmentation cost (no GPU change).
- If unacceptable: drop `Rand3DElasticd` (the heaviest single transform); ~80 % of the gain remains.

**Step 4 — Re-run winning configuration**
- Re-train with: et_heavy weights + CLAHE + full augmentation stack.
- Compare against existing `singlestage3d_best__clahe.pth` (test ET = 0.7341).
- Expected new ET: 0.75–0.77 (assuming ~+0.02–0.04 from augmentation alone).

**Step 5 — Optional: ablate the augmentation stack**
- If thesis time permits, run a small augmentation-stack ablation:
  - row A: current minimal (flips + scale/shift) — already done as 0.7341
  - row B: + rotations (90° + continuous)
  - row C: + spatial elastic
  - row D: + intensity (noise + blur + bias + gamma + histogram)
  - row E: full stack (= row A + B + C + D combined)
- Reports incremental contribution per augmentation family — strong thesis content.

### Risk register
- **`Rand3DElasticd` is CPU-heavy and may bottleneck training.** Mitigation: `prob=0.2` keeps it sparse; if still too slow, reduce magnitude or drop entirely (it's the lowest-impact of the spatial transforms).
- **`RandHistogramShiftd` interacts with Nyúl preprocessing.** If Nyúl is on, both attempt to control the intensity histogram — combination is empirically usually fine but the augmentation may partially undo Nyúl's standardisation. Mitigation: only enable `RandHistogramShiftd` when Nyúl is off, or accept that they trade off.
- **Augmented label may have subtle interpolation artefacts on the ET ring.** Mitigation: `mode='nearest'` for labels in `RandRotated` and `Rand3DElasticd` (already specified above).

### Cost estimate
- Per epoch: +10–20 % wall time on CPU augmentation.
- One winning-config training run with full augmentation: ~3.5–4 h (was 3 h).
- Optional 5-row augmentation ablation: 5 × 4 h ≈ 20 h.


---

## Feature 5 — Preprocessing pipeline

### Summary
The thesis's central contribution. Always-on baseline (percentile clip + z-score) + five ablation flags (N4, Nyúl, CLAHE on T1ce, CLAHE on FLAIR, ET-feature channels). Disk cache organised by config-hash (`./cache_3d_singlestage/<hash>/`).

### 2026-05-09 update — `enhancement_map_alpha` dropped from the channel stack
The `clip(T1ce − α·T1, 0)` channel was removed after a multi-volume audit showed
the source Kaggle h5 dataset (`BraTS2020_training_data/content/data/*.h5`) is
already z-scored per modality before we load it (per-channel mean ≈ 0, negative
values present). The absolute T1ce > T1 asymmetry the formula relies on is
therefore destroyed at source — neither Path A (computing the channel before our
own z-score) nor brain-mask + percentile normalisation recovered it (ET-region
mean stayed below non-ET-WT mean on `volume_100`, `volume_200`, etc.).

Decision: remove `enhancement_map_alpha` from `et_enhancement_channels`. The
remaining three channels — `normalized_enhancement` (sign-safe ratio, scale-
invariant), `sobel_magnitude` of T1ce, and `laplacian_of_gaussian` of the
normalised-enhancement map — all passed verification because they are scale-
invariant or local-gradient based and therefore survive z-scoring.

Channel count when `use_enhancement_channels=True` is now **3 extra channels
(7 total)**, not 4 (8 total). Any downstream code that hard-codes 8 needs to
read it from the dataset instead.

### Current values
| step | status | ET delta vs baseline |
|---|---|---|
| Percentile clip (0.5 %, 99.5 %) | always on | n/a |
| Per-modality z-score (brain mask) | always on | n/a |
| `use_n4` (N4 bias correction) | ablatable | **−0.0038** (hurts; BraTS already N4-corrected) |
| `use_nyul` (Nyúl–Udupa) | ablatable | **−0.0007** (neutral; BraTS already standardises) |
| `use_clahe_t1ce` + `use_clahe_flair` | ablatable | **+0.0049** (winner, currently 2D per-slice) |
| `use_enhancement_channels` (4 extra channels) | ablatable | +0.0004 (neutral) |
| Full stack | ablatable | +0.0039 (less than CLAHE alone) |

### Decision (recorded)
**Keep N4 and Nyúl as historical ablation rows** in the thesis Results table (the negative results demonstrate methodological rigour) but **drop both from the recommended pipeline** going forward. Add five new rows / experiments to the preprocessing axis:

1. **Histogram Equalisation (HE)** — global histogram equalisation as a control row to demonstrate CLAHE (local adaptive) beats HE (global non-adaptive).
2. **Unsharp masking** — high-frequency boundary sharpening; orthogonal to CLAHE's contrast adjustment.
3. **CLAHE clip_limit sweep** (1.0, 2.0, 3.0, 4.0) — hyperparameter sensitivity analysis on the winning 2D-CLAHE row.
4. **3D CLAHE** — extend per-slice CLAHE to true 3D via `skimage.exposure.equalize_adapthist` with a 3D kernel; expected to reduce inter-slice banding and beat 2D CLAHE.
5. (Optional) **Anisotropic diffusion** — already implemented in `preprocessing.py`; expose as `use_anisotropic_diffusion` flag for completeness.

Rescaling to [0, 1] / [-1, 1] explicitly **not** added — redundant with z-score, would only displace the existing baseline.

### Plan

**Step 1 — `preprocessing.py` additions**
Add three new functions (none affect existing code):

```python
def apply_global_histogram_equalisation(img):
    """Standard (non-adaptive) histogram equalisation per slice. Control for CLAHE."""
    a_u8 = ((img - img.min()) / (img.max() - img.min() + 1e-7) * 255).astype(np.uint8)
    eq = cv2.equalizeHist(a_u8)
    return (eq.astype(np.float32) / 255.0) * (img.max() - img.min()) + img.min()


def apply_unsharp_mask(img, sigma=1.5, alpha=1.0):
    """Unsharp masking: image + alpha * (image - GaussianBlur(image, sigma))."""
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img, sigma=sigma)
    return img + alpha * (img - blurred)


def apply_clahe_3d(volume, clip_limit=2.0, kernel_size=(8, 8, 8)):
    """3D CLAHE via skimage.exposure.equalize_adapthist (N-D-capable).

    volume: (D, H, W) float32 array.
    Returns the equalised volume in the same value range.
    """
    from skimage.exposure import equalize_adapthist
    a = volume.astype(np.float32)
    a_min, a_max = float(a.min()), float(a.max())
    if a_max - a_min < 1e-7:
        return a
    a_norm = (a - a_min) / (a_max - a_min)
    eq = equalize_adapthist(a_norm, kernel_size=kernel_size, clip_limit=clip_limit / 100.0)
    return eq.astype(np.float32) * (a_max - a_min) + a_min
```

**Step 2 — Wire new CFG flags + new ablation rows**
Add to `CFG`:
```python
'use_he': False,                      # global histogram equalisation
'use_unsharp': False,                 # unsharp masking
'use_clahe_3d': False,                # 3D CLAHE replacement for 2D CLAHE
'clahe_clip_limit': 2.0,              # was hardcoded 2.0; now sweepable
'unsharp_sigma': 1.5,
'unsharp_alpha': 1.0,
```

Update `_prep_config_hash()` to include the new flags so each variant caches to its own subdir.

Extend `preprocess_volume(...)` to call the new functions when flags are on.

**Step 3 — Extend the preprocessing ablation runner**
Append new rows to `ABL_PLAN`:
```python
ABL_PLAN = [
    # ... existing rows kept as historical evidence ...
    ('he',           dict(use_he=True)),
    ('unsharp',      dict(use_unsharp=True)),
    ('clahe_3d',     dict(use_clahe_3d=True)),
    ('aniso_diff',   dict(use_anisotropic_diffusion=True)),   # optional
]
```

Also add a separate runner for the CLAHE clip-limit sweep:
```python
def run_clahe_clip_sweep():
    summary = []
    for clip in [1.0, 2.0, 3.0, 4.0]:
        CFG['use_clahe_t1ce'] = True
        CFG['use_clahe_flair'] = True
        CFG['clahe_clip_limit'] = clip
        summary.append(run_ablation_row(name=f'clahe_clip{clip}', overrides={...}))
    return summary
```

**Step 4 — Update final ablation table**
Final preprocessing-axis Results table (target ~9 rows):

| # | row | role |
|---|-----|------|
| 1 | baseline (z-score only) | reference |
| 2 | + N4 | historical control (negative result) |
| 3 | + Nyúl | historical control (neutral result) |
| 4 | + 2D CLAHE | original winner |
| 5 | + Enhancement channels | original neutral result |
| 6 | + HE (global) | new control: shows local > global |
| 7 | + Unsharp masking | new positive candidate |
| 8 | + 3D CLAHE | **new thesis-original contribution** |
| 9 | full recommended stack (best of above combined) | headline number |

Plus the CLAHE clip-limit sweep as a separate sensitivity-analysis figure.

### Cost estimate
- Each new ablation row: one fresh training run ≈ 3 h.
- 4 new rows + 4 clip-limit values = 8 training runs = **~24 h** total compute.
- `preprocessing.py` extensions: ~1 hour engineering.
- New cache subdirs (`cT3` / `c3d1` / `he1` / `un1` / etc.): ~5 min each volume × 369 = ~30 min per new config, mostly parallelisable.

### Risk register
- **`equalize_adapthist` from skimage uses a different `clip_limit` scale** (0–1 normalised) vs OpenCV (0–40 typical). Mitigation: explicit conversion in `apply_clahe_3d` (divides by 100 to roughly match OpenCV's scale).
- **3D CLAHE on a 240×240×155 brain volume is slower than 2D per-slice** (single skimage call on 3D vs 155 small OpenCV calls). Mitigation: cache the 3D-CLAHE volumes once per config; subsequent epochs reuse cache for free.
- **Unsharp masking can amplify noise** when applied after z-score (which has unit variance). Mitigation: keep `unsharp_alpha=1.0` mild; if noise dominates, drop to 0.5.
- **HE on z-scored data needs care** — z-score has negative values; OpenCV `equalizeHist` requires uint8 [0, 255]. Mitigation: rescale to [0, 255] internally inside `apply_global_histogram_equalisation` (already done in the sketch above).


---

## Feature 6 — Architecture (3D U-Net)

### Summary
The encoder–decoder backbone. MONAI's `UNet` class with 5 resolution levels, 2 residual units per level, GroupNorm with `num_groups=1`, ~1.9 M total parameters.

### Current values
| component | value |
|---|---|
| Class | `monai.networks.nets.UNet` |
| `spatial_dims` | 3 |
| `in_channels` | 4 (or 8 with `use_enhancement_channels`) |
| `out_channels` | 3 (independent sigmoid heads: WT, TC, ET) |
| `channels` | (16, 32, 64, 128, 160) |
| `strides` | (2, 2, 2, 2) |
| `num_res_units` | 2 |
| `norm` | `('GROUP', {'num_groups': 1})` |
| Activation | PReLU (MONAI default) |
| Dropout | none |
| Total parameters | ≈ 1.9 M |

### Known concerns
1. Channel widths below BraTS standard (canonical (32, 64, 128, 256, 320), ~4× more parameters).
2. No deep supervision (one of the higher-ROI 3D-segmentation upgrades).
3. No dropout (not critical given augmentation).
4. Plain U-Net only — no attention gates, nested skips, or transformer blocks.
5. Single-architecture, no ensembling.

### Decision (recorded)
**No change.** Architecture stays as MONAI 3D U-Net at the current configuration. Architecture-axis ablation (DynUNet / Attention U-Net / U-Net++ / SwinUNETR) deferred to a future, post-thesis experiment campaign.

### Plan
*(no change planned for Feature 6)*


---

## Feature 7 — Multi-task output heads (Divide-and-Conquer realisation)

### Summary
The model has **3 independent sigmoid output heads** (WT, TC, ET) sharing one encoder–decoder trunk. Per-region task weights `(w_WT, w_TC, w_ET)` control the loss balance and constitute the first ablation axis of the thesis. This is the architectural locus of the "divide and conquer" framing.

### Current values
| component | value |
|---|---|
| Output channels | 3 — WT, TC, ET |
| Activation | independent sigmoids (overlap allowed; not softmax) |
| Target derivation | `build_multiclass_target` from BraTS label-id |
| Containment | WT ⊇ TC ⊇ ET (encouraged by shared trunk, not enforced) |
| Per-region weights (current winner) | `(0.2, 0.3, 0.5)` — et_heavy |
| Loss combination | weighted sum of per-region Dice + BCE (Feature 8) |

### Already-completed weighting rows (axis 1, single-fold)
| scheme | weights | test WT | test TC | **test ET** | role in thesis |
|---|---|---|---|---|---|
| equal | (1, 1, 1) | 0.8161 | 0.8176 | 0.7196 | reference / historical |
| **et_heavy** | (0.2, 0.3, 0.5) | 0.8207 | 0.8123 | 0.7292 | **balanced winner** |
| wt_anchored | (0.5, 0.3, 0.2) | 0.8026 | 0.7586 | 0.7085 | reference / historical |
| et_only | (0, 0, 1) | 0.0172 | 0.0083 | **0.7642** | asymptotic ablation: pure-ET focus |

### Concern raised
The professor highlighted **et_only's higher ET (0.7642)** as the more important number, with the qualifier that improving overall results can be addressed in future work. However, et_only produces **WT = 0.017** and **TC = 0.008** — effectively zero on the auxiliary regions — which is the *opposite* of a divide-and-conquer formulation working. Headlining et_only would create a methodology-vs-results inconsistency in the thesis: the title commits to D&C, but the headline configuration disables it.

### Decision (recorded)
**Modify the weighting axis to four primary rows, mapping the WT/TC ↔ ET trade-off curve from balanced multi-task to pure-ET training.**

Replace the current 4-row ablation in the *thesis-headline weighting axis* with:

| # | scheme | weights | role |
|---|--------|---------|------|
| 1 | **et_heavy** *(already done)* | (0.20, 0.30, 0.50) | balanced multi-task; preserves WT/TC ≥ 0.81 |
| 2 | **et_strong** *(new)* | (0.15, 0.20, 0.65) | moderate ET emphasis; expected ET 0.74–0.75 |
| 3 | **et_extreme** *(new)* | (0.05, 0.15, 0.80) | heavy ET emphasis; expected ET 0.75–0.76 |
| 4 | **et_only** *(already done)* | (0.00, 0.00, 1.00) | asymptotic ablation; ET 0.7642 |

Keep `equal` and `wt_anchored` as **historical reference rows** in the Results table (negative-control evidence demonstrating that *neither* uniform weighting *nor* WT-favoured weighting beats the ET-emphasised path).

### Why these four rows

The axis becomes a **smooth trade-off curve** from balanced (et_heavy) → asymptote (et_only). Two middle points let the thesis interpolate the curve and identify the **knee** — the configuration that maximises ET while keeping WT and TC above a clinically-defensible threshold (e.g. WT > 0.5).

This gives both deliverables:
- The professor's preferred headline (high ET; et_only's 0.7642 stays in the table).
- A defensible thesis argument: "Within the divide-and-conquer formulation, et_strong / et_extreme achieves 0.74–0.76 ET while preserving meaningful WT/TC. Removing the multi-task heads entirely (et_only) yields a marginal further ET gain at the cost of total WT/TC failure."

### Plan

**Step 1 — Add the two new schemes to `TASK_WEIGHTS_PLAN`**
```python
TASK_WEIGHTS_PLAN = [
    ('et_heavy',    (0.20, 0.30, 0.50)),    # already trained
    ('et_strong',   (0.15, 0.20, 0.65)),    # new
    ('et_extreme',  (0.05, 0.15, 0.80)),    # new
    ('et_only',     (0.00, 0.00, 1.00)),    # already trained
]
```

Keep the original 4-row plan (`equal`, `et_heavy`, `wt_anchored`, `et_only`) as `TASK_WEIGHTS_PLAN_HISTORICAL` for record-keeping.

**Step 2 — Run the two new schemes**
- `run_weighting_row` already handles isolation (RNG re-seed, fresh model, tagged checkpoint).
- For each new scheme: ~3 h training + ~3 min test eval.
- Total compute: ~6 h.

**Step 3 — Save tagged checkpoints**
- `singlestage3d_best__weighting_et_strong.pth`
- `singlestage3d_best__weighting_et_extreme.pth`

**Step 4 — Update results JSON**
- Append the two new rows to `task_weighting_results.json`.
- Compute and report the WT/TC/ET trade-off table:

| scheme | (w_WT, w_TC, w_ET) | test WT | test TC | test ET |
|--------|---------------------|---------|---------|---------|
| (table populated after the 2 new runs) | | | | |

**Step 5 — Choose the operational winner**
- Apply the rule "highest ET subject to WT > 0.5 and TC > 0.5".
- Likely candidate: et_strong or et_extreme (depending on results).
- This becomes the locked weighting scheme for downstream ablations (preprocessing axis, architecture axis if added).

**Step 6 — Update thesis Results-chapter framing**
Two-tier reporting:
1. **Headline (best D&C-respecting configuration):** the operational winner from Step 5.
2. **Asymptote (best raw ET, methodology-bypassed):** et_only at 0.7642, with explicit annotation that WT and TC are degenerate.

### Cost estimate
- 2 new training runs × ~3 h = **~6 h total**.
- Engineering: ~10 min to add the two TASK_WEIGHTS_PLAN entries + re-run the existing runner.
- No new caches needed (preprocessing config unchanged).

### Risk register
- **Both new schemes might give similar ET to et_heavy** (i.e., the trade-off curve flattens between 0.5 and 0.8 ET weight). In that case, et_only's 0.7642 remains the only meaningfully-higher ET, and the dual-headline framing becomes more important.
- **Both new schemes might give *higher* ET than et_only** (unlikely but possible due to RNG variance). In that case, the new winner replaces et_only in the headline and the trade-off story is even cleaner.
- **Communication with professor**: present the trade-off curve table at the next meeting before committing to the headline framing. Defer final headline decision until Step 5 results are in.


---

## Feature 8 — Loss function (`WeightedDiceCE`) + Metrics

### Summary
Per-region soft Dice + binary cross-entropy with auto-normalised task weights. Serves as both the divide-and-conquer training signal and the headline-metric driver. Currently reporting **only Dice** as a metric — missing HD95 (Hausdorff Distance), which is a primary BraTS-leaderboard metric.

### Current values
| component | value |
|---|---|
| Dice formulation | soft Dice over batch + spatial dims, per channel |
| BCE | pixel-wise BCE-with-logits, per channel |
| `lambda_dice` | 1.0 |
| `lambda_ce` | 1.0 |
| `smooth` | 1.0 |
| Per-region weights | normalised so Σ w = 1 |
| BCE `pos_weight` | none |
| Boundary loss | none |
| Focal weighting | none (tested previously, abandoned — caused constant-output collapse at bs=1) |
| Deep supervision | none |
| Reported metrics | Dice (WT, TC, ET) per volume, mean over test set |

### Known concerns
1. ET is a thin ring; boundary errors dominate. Pure Dice undersupervises boundary precision.
2. `smooth=1.0` over-rewards small targets (Dice-cheating on ET).
3. Only Dice reported; HD95 — the standard companion metric for BraTS — is missing.
4. No deep supervision (depends on Feature 6 architecture decision).

### Decision (recorded)
**Adopt three loss-side changes plus add HD95 as a reporting metric.**

#### Loss changes
1. **Add boundary loss (Kervadec et al., MIDL 2019)** as a third weighted term:
   ```
   L = λ_dice · Σ w_c · DiceLoss(p_c, t_c)
     + λ_ce   · Σ w_c · BCE(p_c, t_c)
     + λ_bnd  · Σ w_c · BoundaryLoss(p_c, t_c)
   ```
   where `BoundaryLoss(p, t) = mean(p · SDT(t))` with `SDT` = signed distance transform of the GT mask (negative inside, positive outside).  Encourages probability *inside* the GT and away from it *outside*. Particularly effective for thin ring structures.
2. **Reduce smoothing** from `smooth=1.0` to `smooth=0.1`. Aligns with nnU-Net practice; removes Dice inflation on small ET targets.
3. **Keep** `λ_dice = λ_ce = 1.0` (sweeping gives ±0.005 ET — not cost-effective compute).
4. **Do not add BCE `pos_weight`** (history: triggers constant-output collapse at bs=1).

#### Metric additions
5. **Report HD95** (95th-percentile Hausdorff Distance, in voxels) per region per volume, alongside Dice.
   - Use `monai.metrics.HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95.0, reduction='mean')`.
   - Add to `validate_volumes()` for per-epoch monitoring (not used for checkpoint selection — Dice ET stays the selector).
   - Add to `evaluate_test()` for final per-volume reporting.
   - Persist alongside Dice in the `evaluate_test()` return dict and in `task_weighting_results.json` / `preprocessing_ablation_results.json`.

### Plan

**Step 1 — Implement boundary loss term**
Add to `WeightedDiceCE`:
```python
class WeightedDiceCE(nn.Module):
    def __init__(self, weights, lambda_dice=1.0, lambda_ce=1.0,
                  lambda_boundary=0.2, smooth=0.1):
        ...
        self.lambda_boundary = float(lambda_boundary)

    def forward(self, logits, targets, sdt=None):
        # ... existing dice/bce ...
        if self.lambda_boundary > 0 and sdt is not None:
            # sdt: signed distance transform of targets, same shape (B, C, D, H, W)
            p = torch.sigmoid(logits.float())
            boundary_per_ch = (p * sdt).mean(dim=dims)
            total = total + self.lambda_boundary * (boundary_per_ch * w).sum()
        return total
```

Add SDT precomputation: when caching each volume, also save signed-distance transforms of WT, TC, ET (3 channels). Use `scipy.ndimage.distance_transform_edt` on both the target and its complement, subtract.

```python
def signed_distance_transform_3ch(target_3ch):
    """target_3ch: (3, D, H, W) binary. Returns (3, D, H, W) float SDT
    where negative = inside, positive = outside."""
    from scipy.ndimage import distance_transform_edt as dt
    out = np.zeros_like(target_3ch, dtype=np.float32)
    for c in range(3):
        t = target_3ch[c].astype(bool)
        if t.any() and (~t).any():
            out[c] = dt(~t) - dt(t)
        elif t.all():
            out[c] = -dt(~t).astype(np.float32)  # all-foreground -> all negative
        else:
            out[c] = dt(~t).astype(np.float32)   # all-background -> all positive
    return out
```

Cache the SDTs once per volume (per preprocessing config). Re-use across training. Cost: ~30 minutes one-off compute over 369 volumes; ~50 MB extra disk per volume.

**Step 2 — Update `train_stageB` and dataset**
- `BraTSVolDataset.__getitem__` returns the (image, label, sdt) triplet when boundary loss is enabled.
- `train_stageB` passes `sdt=batch['sdt']` to `criterion(logits, lbl, sdt=sdt)`.
- Validation does **not** use boundary loss (validation uses Dice ET for selection).

**Step 3 — Add HD95 metric**
```python
from monai.metrics import HausdorffDistanceMetric

hd95_metric = HausdorffDistanceMetric(
    include_background=True,
    distance_metric='euclidean',
    percentile=95.0,
    reduction='mean',
    get_not_nans=False,
)

@torch.no_grad()
def evaluate_test(threshold=0.5, return_per_vol=True):
    # ... existing Dice computation ...
    # Then HD95:
    pred_one_hot = (probs > threshold).float()       # (1, 3, D, H, W)
    gt_one_hot   = lbl.float()
    hd95_metric.reset()
    hd95_metric(y_pred=pred_one_hot, y=gt_one_hot)
    hd95 = hd95_metric.aggregate()                    # tensor of shape (3,)
    # Append per-volume: hd95_WT, hd95_TC, hd95_ET (in voxels)
    ...
```

Note: HD95 returns `inf` when either prediction or ground-truth is empty. Handle by setting to a sentinel value (e.g. the diagonal of the volume, ≈ 343 voxels) before aggregation, or filter empty-prediction volumes.

**Step 4 — Update CFG**
```python
'lambda_boundary': 0.2,           # boundary loss weight (sweep 0.1-0.5 if time)
'smooth': 0.1,                    # was 1.0
'use_boundary_loss': True,        # toggle for the SDT loss term
```

**Step 5 — Update results JSONs**
Each row now contains both Dice and HD95 per region:
```json
{
  "name": "et_strong",
  "weights": [0.15, 0.20, 0.65],
  "test_WT_dice": 0.81,
  "test_TC_dice": 0.78,
  "test_ET_dice": 0.745,
  "test_WT_hd95": 5.2,
  "test_TC_hd95": 6.7,
  "test_ET_hd95": 8.3
}
```

**Step 6 — Update thesis Results tables**
Each row reports six numbers (3 Dice + 3 HD95). Lower HD95 = better.  Standard BraTS reporting format.

### Cost estimate
- One-off SDT precompute: ~30 minutes (across 369 volumes × 3 regions).
- Per-step training overhead: trivial (one elementwise multiply + sum).
- Per-validation HD95 cost: ~1–2 seconds per volume (acceptable at val_every=5 over 73 val volumes).
- One re-run of the winning configuration (et_strong + CLAHE + boundary + smooth=0.1): ~3–4 h.
- Optional `λ_boundary` sweep (0.1, 0.2, 0.4): 3 × 3 h = 9 h — only if time permits.

### Risk register
- **Boundary loss can blow up early in training** (when probabilities are uniform, the boundary term has large gradient on every voxel). Mitigation: warm up `λ_boundary` linearly from 0 to its final value over the first 5 epochs.
- **HD95 produces `inf` for empty predictions** in the early epochs (model predicts nothing for some volumes). Mitigation: clip to a max value (e.g. 100 voxels) before averaging.
- **SDT cache invalidation on label changes**: SDT depends only on the GT label (not on preprocessing flags), so the cache is shared across all preprocessing configs. Store as `cache_3d_singlestage/sdt/<vid>.npz`.
- **Smooth=0.1 may produce unstable Dice on truly-empty patches** (denominator near zero). Mitigation: keep `pos=0.67, neg=0.33` so most patches contain tumour; monitor for NaN in the existing guard.


---

## Feature 9 — Optimisation (AdamW + warmup-cosine schedule)

### Summary
The optimiser, learning rate, schedule, and gradient-management settings that turn loss signal into weight updates. Currently AdamW at 2e-4 with a 5-epoch linear warmup to a cosine annealing decay over 45 further epochs (50 total), weight decay 1e-4, gradient clip 1.0, AMP forward pass.

### Current values
| component | value |
|---|---|
| Optimiser | AdamW |
| Peak LR | 1e-4 |
| Weight decay | 1e-5 |
| Warmup | LinearLR, start_factor=1e-2, 2 epochs |
| Main schedule | CosineAnnealingLR, 45 epochs, eta_min=1e-6 |
| Total epochs | 50 (locked after timing audit) |
| Early stop | patience 3 on ET val Dice (= 15 epochs without improvement at val_every=5) |
| Gradient clip | max-norm 1.0 |
| Mixed precision | disabled (fp32 forward) |
| EMA / SWA | none |
| Gradient accumulation | none |
| Effective batch / step | 1 volume × 2 patches = 2 (will become 4 with Feature 3) |

### Known concerns
1. LR 1e-4 conservative for AdamW + GroupNorm; can safely push to 2e-4.
2. 50 epochs ≈ 1/5 of nnU-Net's BraTS training duration in batch terms. Compensated for by EMA + boundary loss + a richer augmentation stack than nnU-Net default.
3. Weight decay 1e-5 light; insufficient regularisation given the upcoming richer augmentation stack.
4. No EMA — leaving +0.005–0.015 free ET gain on the table.

### Decision (recorded)
**Four optimisation changes (combined senior-ML-defensible move):**
1. **Add EMA of model weights** via `torch.optim.swa_utils.AveragedModel`. Use EMA weights at inference; keep trained weights for resume.
2. **Locked at 50 epochs.** Originally planned for 80–160; B1 timing audit (validation cost dominates) forced the cut. Reaches ~95% of the 80-epoch asymptote based on the B1 curve.
3. **Increase peak LR from 1e-4 to 2e-4** with warmup extended to 5 epochs (instead of 2) to keep early-training stability.
4. **Increase weight decay from 1e-5 to 1e-4** to compensate for the richer augmentation pipeline (Feature 4) and the added boundary-loss term (Feature 8).

Skip: SAM / LARS / LAMB (over-engineering), bf16 autocast (deferred to cloud), gradient accumulation (already paying 2× from the epoch increase), SWA (EMA captures the same idea more cleanly).

Combined expected ET gain: **+0.015–0.045**.

### Plan

**Step 1 — Update CFG**
```python
CFG['lr']             = 2e-4
CFG['weight_decay']   = 1e-4
CFG['warmup_epochs']  = 5
CFG['epochs']         = 50
CFG['use_ema']        = True
CFG['ema_decay']      = 0.999
```

**Step 2 — Wire EMA into training**
```python
from torch.optim.swa_utils import AveragedModel
ema_avg = lambda avg_p, p, n: 0.999 * avg_p + 0.001 * p
EMA = AveragedModel(MODEL, avg_fn=ema_avg)

# in train loop after optimiser.step():
EMA.update_parameters(MODEL)

# at end of training:
torch.save({'model_state_dict': EMA.module.state_dict(), ...},
            os.path.join(CFG['checkpoint_dir'], 'singlestage3d_ema.pth'))
```

**Step 3 — Use EMA at inference**
- After training, load `singlestage3d_ema.pth` for `evaluate_test()` instead of `singlestage3d_best.pth`.
- Optionally compare both: report `Dice/HD95 (last-best)` and `Dice/HD95 (EMA)` side-by-side; EMA usually wins.
- Update `load_best_singlestage(tag=None, use_ema=True)` to accept an EMA flag.

**Step 4 — Adjust early stopping**
- Patience 3 evaluation checks at `val_every=5` = 15 epochs before stop.
- With epochs=50, patience absorbing ~15 of them is fine.

**Step 5 — Verify the LR + warmup interaction**
- Quick sanity check on the first run: watch the first 10 epochs.
- If train loss explodes or val Dice never starts climbing, drop LR back to 1e-4.
- Otherwise the 2e-4 / 5-epoch warmup combo should be stable on GroupNorm + bs=1.

### Cost estimate
- Wall time per run: 2× current (~3 h → ~6 h *before* compounding with Feature 3's checkpointing and Feature 4's augmentation overheads).
- After all stacked overheads (Features 3 + 4 + 9): ~3 h × 2 × 1.30 × 1.15 ≈ **~9 h per training run**.
- Re-running winning configuration once: **~9 h**.
- 4-row weighting axis (et_heavy / et_strong / et_extreme / et_only): 2 new × ~9 h ≈ **18 h**.
- Re-running preprocessing axis (~9 rows): would cost ~81 h = **~3.5 days** at full configuration. Limit by re-running the existing top contenders only (CLAHE, full, plus the new 3D CLAHE row).
- 5-fold CV × all rows is intractable; restrict CV to the **single winning configuration** (per Feature 2's mitigation).

### Risk register
- **EMA decay=0.999 is on the slow side** — model weights only meaningfully average over the last ~1000 batches. Could drop to 0.99 for faster averaging (more responsive to late-training improvements). Mitigation: stick with 0.999 for stability; only reconsider if EMA underperforms the trained weights at inference.
- **LR=2e-4 may destabilise early training** at fp32 + bs=1. Mitigation: longer warmup (5 epochs) absorbs the risk; sanity-check the first 10 epochs.
- **50 epochs may underfit.** Mitigation: B1 reached test ET 0.71 at 50 epochs with equal weights / baseline preprocessing, so the budget is empirically sufficient. If a later row plateaus before 50, early-stop catches it; if it is still climbing at 50, the headline run can be extended.
- **Wall-time stacking with other features** could push a single run to 10+ hours. Mitigation: monitor first run and re-evaluate; if too slow, drop epochs back to 120 (saves 25 % wall time).

