# Chapter 3 — Methodology — Writing Guide

This file briefs another Claude (or another human collaborator) who has not seen
the project history. It contains:

1. What the thesis is about and the divide-and-conquer formulation in one page
2. The state of the codebase as of 2026-05 (what is implemented, what was tried
   and rejected, what is still pending)
3. The agreed Chapter 3 outline (section-by-section) with notes on what each
   subsection should cover and which pitfalls to avoid
4. The Stage-1 verification status and the known data caveats that constrain
   what can honestly be claimed in Chapter 3

If you are about to write a section, read the matching subsection here *first*.
Many of the design choices look arbitrary in isolation but were forced by
findings recorded only in `EXPERIMENT_FEATURES.md` and `PROGRESS.md`.

---

## 1. Project overview

**Thesis title.** *Pre-Processing Aided Divide and Conquer for Brain Tumor
Segmentation: The Case of Enhancing Tumor Using 3D U-Net.*

**Author.** Bahaa Medhat (GUC, Bachelor thesis).

**Dataset.** BraTS 2020 (UPenn), 369 patient volumes, four co-registered
skull-stripped modalities (T1, T1ce, T2, FLAIR), 240 × 240 × 155 voxels each.
The Kaggle h5 distribution used here (`BraTS2020_training_data/content/data/`)
stores **per-modality z-scored** intensities, not raw HU/SI units. This single
fact constrains several preprocessing decisions and **must be acknowledged in
§3.1.3**. See `EXPERIMENT_FEATURES.md` Feature 5 for the audit.

**Target.** The **Enhancing Tumor (ET)** sub-region — the BraTS label-id 4
voxels, the post-contrast-enhancing tumour rim that is the clinical target for
gadolinium-uptake assessment, surgical planning, and treatment-response
monitoring. ET is ~ 0.3–1 % of brain volume — the rarest of the three BraTS
sub-regions and the hardest to segment.

**The divide-and-conquer formulation.** The three BraTS sub-regions
*Whole Tumour* (WT), *Tumour Core* (TC), and *Enhancing Tumour* (ET) are
**nested anatomical sets**, not disjoint classes. Containment:

> WT ⊇ TC ⊇ ET

The thesis *divides* the segmentation problem into these three nested binary
sub-tasks and *conquers* them jointly by a single 3D U-Net with three
**independent sigmoid output heads** that share one encoder–decoder trunk.
The natural containment is *encouraged* by shared features, not *enforced* by
a softmax constraint. A softmax over the three regions would force them to be
mutually exclusive, which is biologically wrong by construction (every ET voxel
is also a TC voxel and a WT voxel).

**Two-axis ablation methodology.** The thesis evaluates contributions along
two orthogonal axes:

1. **Per-region loss weighting** (the divide-and-conquer balance) — schemes
   include `equal`, `et_heavy`, `et_strong`, `et_extreme`, `wt_anchored`,
   `et_only`. Best scheme is locked, then…
2. **Preprocessing technique** — baseline → +CLAHE-2D → +CLAHE-3D → +HE →
   +Unsharp → +ET-channels → +N4 → +Nyúl → full positive stack.

A third axis (architecture: U-Net vs SegResNet vs Attention U-Net vs U-Net++
vs SwinUNETR) is **optional**; the author has not yet committed to including
it (`PROGRESS.md`).

**Hardware constraint.** Single consumer GPU, **6 GB VRAM**. This forces
patch size 96³ (not the conventional 128³), narrow encoder widths
`(16, 32, 64, 128, 160)`, gradient checkpointing, mixed-precision (AMP),
gradient accumulation (`grad_accum_steps = 4`, effective batch ≈ 4),
batch_size = 1, GroupNorm with `num_groups=1` (LayerNorm-over-features —
stable at batch_size=1 where InstanceNorm collapses sparse-class signal).
**The hardware constraint must be declared in §3.2.5 as a methodology
limitation, up front.**

---

## 2. State of the codebase (2026-05)

### 2.1 Files

| file | purpose |
|---|---|
| `brats2020_singlestage_3d.ipynb` | the active notebook (38 cells after polish) |
| `preprocessing.py` | reusable preprocessing module (CLAHE, CLAHE-3D, HE, unsharp, N4, Nyúl, enhancement channels, percentile-z-score) |
| `EXPERIMENT_FEATURES.md` | per-feature design audit; the **source of truth** for *why* each design decision was made and what was tried and rejected |
| `PROGRESS.md` | summary of historical results and lessons |
| `TODO.md` | stage-1 verification → stage-2 polish → stage-3 experiment phases |
| `CHAPTER3_GUIDE.md` | this file |

Old 2D notebook (`brats2020unet2D.ipynb`) and old scripts have been retired.

### 2.2 Notebook structure (post-polish, 38 cells)

The notebook is the methodology in executable form. Section-to-cell mapping
when you write Chapter 3 captions / cross-references:

| § | cells | what is there |
|---|---|---|
| 3.1 Dataset | 0 (title md), 1 (imports), 2 (CFG), 3 (set_seed + split_by_volume), 4 (group_slices_by_volume) | the dataset code lives in cells 3–4; splits are seed-42 60/20/20 volume-level |
| 3.3 Preprocessing | 5 (md), 6 (`preprocess_volume`, `_prep_config_hash`, `cache_all_volumes`, `load_cached_volume`) | preprocessing logic, with the optional flags wired through CFG |
| 3.1 viz | 7 (md), 8 (`show_raw_volume`), 9 (md), 10 (`show_preprocessing_comparison`) | visual sanity probes |
| 3.4/3.6 dataset class | 11 (md), 12 (`build_multiclass_target`, `et_enhancement_channels`, `BraTSVolDataset`, transforms), 13 (SDT helpers + HD95 metric) | the dataset and the MONAI transform pipeline |
| 3.3.4 viz | 14 (md), 15 (`show_enhancement_channels`) | ET-channel illustration |
| 3.6 viz | 16 (md), 17 (`show_training_patch`) | a 96³ post-augmentation patch |
| 3.4 + 3.5 + 3.6 | 18 (md), 19 (`build_model` + `WeightedDiceCE` + criterion + optimiser + scheduler), 20 (EMA helper), 21 (gradient checkpointing helper + invocation) | model, loss, optimiser, EMA, memory-saving features |
| 3.4 summary | 22 (md), 23 (`model_summary`) | prints param count + heaviest modules |
| 3.6 training | 24 (md), 25 (`train_model`, `validate_volumes`), 26 (`run_kfold_cv`), 27 (EMA-aware validate_volumes wrapper) | training loop, k-fold orchestration, EMA validation |
| 3.7 curves | 28 (md), 29 (`plot_training_curves`) | per-epoch loss + val Dice |
| 3.7 test eval | 30 (md), 31 (`load_best_singlestage` + `evaluate_test`) | sliding-window inference, per-region Dice + HD95 |
| 3.7 viz | 32 (md), 33 (`show_test_prediction`) | qualitative grid per test volume |
| 3.8 ablations | 34 (md, preprocessing axis), 35 (`ABL_PLAN` + `run_ablation_row` + `run_preprocessing_table`), 36 (md, weighting axis), 37 (`TASK_WEIGHTS_PLAN` + `run_weighting_row`) | the two ablation runners |

If you cite a cell number in Chapter 3, **use the cell's leading markdown
title** (e.g. *"§ Training procedure (notebook cell 24)"*) rather than the
bare integer — the cell index will drift if the notebook is reordered.

### 2.3 What is implemented

- 3D U-Net (MONAI's `UNet`) with 3 sigmoid output channels (WT/TC/ET)
- `WeightedDiceCE` loss = per-region Dice + BCE + (optional) Kervadec
  boundary loss using precomputed signed-distance transforms
- AdamW + LinearLR warmup (5 epochs) → CosineAnnealingLR
- AMP (mixed precision) + gradient checkpointing + gradient accumulation
- Exponential Moving Average of weights (decay 0.999); validation uses
  the EMA shadow
- NaN guards on logits, loss, and gradients
- Sliding-window inference (96³ ROI, 0.5 overlap, Gaussian weighting)
- HD95 metric via `monai.metrics.HausdorffDistanceMetric`
- Per-config disk cache keyed by `_prep_config_hash()`
- Resume-aware training (`singlestage3d_last.pth` with `prep_hash` guard)
- Two ablation runners (`run_ablation_row`, `run_weighting_row`) with
  RNG re-seed and tagged checkpoint saves per row
- 5-fold cross-validation orchestrator (`run_kfold_cv`) — to be applied
  to the winning configuration only (cost: ~ 45 h on a 6 GB GPU)

### 2.4 What was tried and rejected (do **not** describe in Chapter 3 as if these are options the thesis evaluates)

- **Two-stage cascade (Stage-A WT localisation → Stage-B ET on cropped
  ROI).** Tried first; abandoned. The single-stage 3D pipeline reaches
  comparable ET Dice without the cascade-coverage failure mode. The old
  2D-cascade notebook (`brats2020unet2D.ipynb`) exists only as a historical
  artefact; do not cite it in Chapter 3.
- **Absolute-intensity enhancement map `clip(T1ce − α·T1, 0)`.** Dropped
  because the source data is z-scored at the file level, destroying the
  cross-modality magnitude asymmetry the formula relies on. The remaining
  three ET feature channels (normalised enhancement, Sobel, LoG) survive
  z-scoring and are kept. **This decision is described in §3.3.4.4 as a
  consequence of the dataset's intensity state, not as a methodology
  failure** — Chapter 3 frames it as "the channel stack is restricted to
  scale-invariant features", with the why in a footnote that points to
  `EXPERIMENT_FEATURES.md` Feature 5.
- **Softmax over WT/TC/ET.** Considered and rejected. Mentioned in §3.2.2
  as the alternative formulation, with the containment-incompatibility
  argument for why it is wrong here.
- **SegResNet, blocks_down=(1,2,2,4).** Tried as the initial model; replaced
  by U-Net for better small-region (ET) performance. The notebook now
  supports U-Net only; the `build_model` function takes no `model` switch.
  If §3.4 mentions SegResNet, it is only as the "alternative the architecture
  axis could compare against" — *not* as part of the current pipeline.

### 2.5 What is pending

- Phase 0 verification run (20 epochs, equal weights, baseline preprocessing,
  3D U-Net) — was queued at the time of writing
- Phase 1–5 thesis runs (see `TODO.md` § Stage 3)

Chapter 3 must read as if Phase 0+ is complete and successful — it is a
methodology chapter, not a status report.

---

## 3. Chapter 3 outline with per-subsection notes

The outline below is the agreed structure. For each subsection, the
**Cover** bullet lists what must be in the section; the **Pitfall** bullet
lists the trap to avoid; the **Source** bullet points at the part of the
codebase or docs that is authoritative.

### 3.1 Dataset

#### 3.1.1 BraTS 2020 overview
- **Cover.** UPenn organisation; 369 patient volumes; four co-registered
  skull-stripped modalities (T1, T1ce, T2, FLAIR); 240 × 240 × 155 voxels.
- **Pitfall.** Do not cite slice counts from the older 2D pipeline —
  always cite *volume* counts.
- **Source.** Notebook cell 0 markdown; `EXPERIMENT_FEATURES.md` Feature 1.

#### 3.1.2 Label hierarchy
- **Cover.** Raw label-ids `1 = NCR`, `2 = ED`, `4 = ET`; derived regions
  via `build_multiclass_target` (cell 12): WT = `label > 0`, TC =
  `label ∈ {1, 4}`, ET = `label == 4`. The containment WT ⊇ TC ⊇ ET is a
  *property of the labels*, not an assumption.
- **Pitfall.** Do **not** call WT/TC/ET classes; they are regions / sets.
- **Source.** `build_multiclass_target` in cell 12.

#### 3.1.3 Data source and intensity state
- **Cover.** The Kaggle h5 distribution stores **per-modality z-scored**
  intensities (per-channel mean ≈ 0 within brain, negative values present
  outside). This is documented; do not pretend the data is raw.
  Implication: any preprocessing step or feature channel that relies on
  *absolute cross-modality magnitudes* is ill-defined and dropped — see
  §3.3.4.4.
- **Pitfall.** This is the single most important caveat in the whole
  thesis. Skipping it makes §3.3.4.4 look arbitrary.
- **Source.** `EXPERIMENT_FEATURES.md` Feature 5 (2026-05 update).

#### 3.1.4 Volume-level train / validation / test split
- **Cover.** 60 / 20 / 20 = 221 / 73 / 75 volumes. Seed 42. **Volume-level**
  split (a patient's slices are never split across splits — no spatial
  leakage between train/val/test). Implemented in `split_by_volume` (cell 3).
- **Pitfall.** Do not describe the split as "slice-level" — it is not, and
  saying so undermines the leakage argument.
- **Source.** `split_by_volume` in cell 3.

#### 3.1.5 Storage and access pattern
- **Cover.** Each preprocessing configuration writes to its own
  `./cache_3d_singlestage/<hash>/` subdirectory (hash via
  `_prep_config_hash`, cell 6). Each volume is an `.npz` with `image`,
  `mask`, optional `enh`, optional `sdt` arrays. The cache is hash-keyed
  so reproducibility across ablation rows is trivial.
- **Pitfall.** Do not promise the on-disk schema is portable across
  refactors — it is keyed to the current CFG flag set.
- **Source.** `_prep_config_hash`, `cache_all_volumes` in cell 6.

### 3.2 Problem formulation: divide-and-conquer

#### 3.2.1 The nested-region segmentation problem
- **Cover.** State the WT ⊇ TC ⊇ ET hierarchy with a small diagram. State
  the per-region typical share of brain volume (WT 5–10 %, TC 1–3 %, ET
  0.3–1 %) — this motivates the class-imbalance choices later.
- **Source.** Notebook cell 0; `EXPERIMENT_FEATURES.md` Feature 7.

#### 3.2.2 Why three independent sigmoid heads, not a softmax
- **Cover.** Softmax forces mutual exclusivity. Containment requires every
  ET voxel to *also* be a TC voxel and a WT voxel. Therefore independent
  sigmoids — one per region — are the right output parameterisation.
  Containment is *encouraged* by the shared encoder/decoder trunk, not
  *enforced* by an output constraint.
- **Pitfall.** Do not claim the network *guarantees* containment at
  inference — it does not. Containment is an empirical property of the
  trained features.
- **Source.** `WeightedDiceCE` and `build_model` in cell 19; cell 0 md.

#### 3.2.3 ET as primary clinical target
- **Cover.** ET is the gadolinium-uptake region — surgical resection target,
  treatment-response indicator, survival-correlate. WT and TC are
  *auxiliary*: they regularise the encoder, they are not the thesis target.
  This is why `task_weights` defaults to ET-heavy `(0.2, 0.3, 0.5)` and
  why the operational winner rule is "highest ET subject to WT > 0.5 AND
  TC > 0.5".
- **Source.** `EXPERIMENT_FEATURES.md` Feature 7.

#### 3.2.4 Two-axis ablation methodology
- **Cover.** Axis 1 (weighting) is swept first; the winner is locked; axis
  2 (preprocessing) is then swept under that locked weighting. The full
  4 × 9 grid would be 36 runs ≈ 6 days of compute on the available
  hardware; the sequential design produces 4 + 8 = 12 runs ≈ 1.5 days
  while still measuring each axis under conditions favourable to the other.
  Justify the sequential design before §3.6 (Training procedure) so a
  reader does not ask "why not full grid" later.
- **Source.** `EXPERIMENT_FEATURES.md` cost estimates; `TODO.md` Stage 3.

#### 3.2.5 Hardware constraints and methodological implications
- **Cover.** Declare 6 GB VRAM up front. Direct consequences: 96³ patch
  size (vs 128³ in competitive submissions), narrow channel widths
  `(16, 32, 64, 128, 160)`, batch_size = 1, gradient checkpointing, AMP,
  gradient accumulation `4×` (effective batch ≈ 4), GroupNorm with
  num_groups = 1.
- **Pitfall.** Do not apologise for the constraint — present it as the
  scope condition that the methodology was *designed for*. The thesis
  contribution is what is achievable within this constraint.
- **Source.** CFG in cell 2; cell 0 md.

### 3.3 Preprocessing pipeline

For each step list: **formula** (or one-line description), **role in the
thesis** (always-on vs ablation flag vs control), **citation**, and the
CFG flag that toggles it. Use `EXPERIMENT_FEATURES.md` Feature 5 as the
authoritative reference for design decisions.

#### 3.3.1 Always-applied baseline
- Percentile clipping `[0.5, 99.5]` within brain mask; per-modality
  z-score within brain mask. Implemented in
  `pp.percentile_clip_zscore`. Runs *last* in the pipeline so its
  statistics are computed on the fully-processed signal.

#### 3.3.2 Optional steps — control rows
- **3.3.2.1 N4 bias correction** — SimpleITK; per modality, ≈ 15 s/vol.
  CFG flag `use_n4`. Citation: Tustison et al. 2010.
- **3.3.2.2 Nyúl–Udupa histogram standardisation** — fits landmarks on
  training set, applies to each volume. CFG `use_nyul`. Citation: Nyúl,
  Udupa, Zhang 2000.
- **3.3.2.3 Global histogram equalisation** — non-adaptive HE on T1ce +
  FLAIR. **Included as a control row** to show CLAHE > HE empirically.
  CFG `use_hist_eq`. Implemented in `pp.apply_global_histogram_equalisation`.

#### 3.3.3 Optional steps — sharpening / contrast
- **3.3.3.1 CLAHE 2D** — per-slice, on T1ce and FLAIR. CFG
  `use_clahe_t1ce` / `use_clahe_flair`. Citation: Zuiderveld 1994.
- **3.3.3.2 CLAHE 3D (thesis-original variant)** — per-slice CLAHE
  followed by axial-neighbour smoothing. Adjacent-slice correlation
  rises from 0.93 (naive 2D) to 0.98 (3D-smoothed). CFG `use_clahe_3d`.
  Implemented in `pp.apply_clahe_3d`. **Present as a thesis
  contribution** — it is a small but novel piece.
- **3.3.3.3 Unsharp masking** — `out = img + amount · (img − blur(img, σ))`.
  CFG `use_unsharp`. Implemented in `pp.apply_unsharp_mask`. Verified to
  approximately double Laplacian energy (high-frequency boost).

#### 3.3.4 ET-specific feature channels
- **3.3.4.1 Normalised enhancement** — `(T1ce − T1) / (|T1ce| + |T1| + ε)`,
  bounded in `[-1, +1]` by triangle inequality. Scale-invariant — robust
  to per-patient T1ce gain variation.
- **3.3.4.2 Sobel magnitude on T1ce** — `‖∇T1ce‖`. Verified to peak on
  the WT boundary (boundary mean / interior mean > 1.5 on the verification
  suite).
- **3.3.4.3 Laplacian-of-Gaussian on normalised enhancement** — ring
  detector for the enhancing rim.
- **3.3.4.4 Note on the dropped absolute-difference channel.** The
  classical `clip(T1ce − α·T1, 0)` enhancement map relies on absolute
  cross-modality magnitudes. The source data is z-scored per modality
  (§3.1.3), destroying that asymmetry. The channel was therefore
  excluded; the three retained channels are all scale-invariant or
  local-gradient based. **State this as a methodological consequence
  of §3.1.3, not as a failure.**

#### 3.3.5 Application order and rationale
- **Order.** N4 → Nyúl → (CLAHE-3D | CLAHE-2D | HE | Unsharp) → ET
  feature channels (built on the post-CLAHE, pre-z-score signal) →
  percentile-clip + z-score. Justify each junction: N4 corrects
  multiplicative drift first so later steps inherit a clean signal;
  Nyúl canonicalises histograms; CLAHE etc. sharpens locally;
  enhancement channels are built before z-score so they see
  physics-meaningful intensities; z-score last so its statistics see
  the fully-processed signal.

### 3.4 Network architecture

#### 3.4.1 3D U-Net (MONAI)
- **Cover.** `monai.networks.nets.UNet`, `spatial_dims=3`.
- **Source.** `build_model` in cell 19.

#### 3.4.2 Encoder / decoder configuration
- **Cover.** Channels `(16, 32, 64, 128, 160)` — 5 resolution levels;
  strides `(2, 2, 2, 2)` halve at each downsample; 2 residual units per
  level (`num_res_units=2`); PReLU (MONAI default). Total parameters
  ≈ 3.2 M (printed at runtime by `model_summary`).

#### 3.4.3 Normalisation choice
- **Cover.** GroupNorm with `num_groups = 1`. At `num_groups = 1`,
  GroupNorm is equivalent to LayerNorm over features. Required because
  batch_size = 1 (hardware constraint, §3.2.5) makes BatchNorm undefined
  and InstanceNorm degenerate (per-feature mean / variance collapses to
  zero for sparse-class signal — ET in particular).

#### 3.4.4 Activation, output heads, parameter count
- **Cover.** PReLU activations. Output: 3 channels — independent sigmoids
  for WT, TC, ET. Parameter count ≈ 3.2 M.

#### 3.4.5 Input-channel composition under the preprocessing ablation
- **Cover.** Input is `(B, C, 96, 96, 96)` with `C = 4` (baseline) or
  `C = 7` (when `use_enhancement_channels = True` adds the three ET
  feature channels). `build_model` adapts to whatever `in_channels_for_cfg()`
  reports, so the *same* architecture is reused across rows; only the
  first conv layer's input channel count changes.

### 3.5 Loss function

#### 3.5.1 Per-region Dice loss
- `D_c = 1 − (2 Σ p · t + s) / (Σ p + Σ t + s)` per region, smoothing
  `s = 0.1`.

#### 3.5.2 Per-region binary cross-entropy
- Pixel-wise BCE-with-logits per region.

#### 3.5.3 Task weighting and auto-normalisation
- Weights `(w_WT, w_TC, w_ET)` set by `CFG['task_weights']` and
  **normalised inside `WeightedDiceCE` so they sum to 1**. This lets
  the thesis report weights as percentages and keeps total loss
  magnitude comparable across configurations. Default `(0.2, 0.3, 0.5)`
  = `et_heavy`; the weighting axis sweeps this.

#### 3.5.4 Boundary loss (Kervadec et al. 2019)
- `L_boundary = mean(σ(logit) · |SDT|_normalised)` per region.
  Signed-distance transforms are precomputed and cached per volume
  (see §3.6.x SDT cache). Toggle via `CFG['use_boundary_loss']` with
  weight `lambda_boundary = 0.2`.

#### 3.5.5 Combined objective
- `L_total = λ_dice · Σ w_c · DiceLoss_c + λ_ce · Σ w_c · BCE_c +
  λ_b · Σ w_c · BoundaryLoss_c`, all `λ` from CFG. Default
  `λ_dice = λ_ce = 1.0`, `λ_b = 0.2`. Justify each coefficient.

### 3.6 Training procedure

#### 3.6.1 Patch-based 3D training
- 96³ patches via MONAI's `RandCropByPosNegLabeld`. `pos = 0.67,
  neg = 0.33` (two-thirds tumour-centric, one-third background). Two
  samples per volume per epoch.

#### 3.6.2 Data augmentation
- Random flips on all three spatial axes; random 90° rotations;
  random scale + shift intensity; random Gaussian noise; random
  bias-field (MONAI's `RandBiasFieldd`). All `EnsureTyped` with
  `allow_missing_keys=True` so the `sdt` key is tolerated whether
  or not boundary loss is enabled.

#### 3.6.3 Optimiser and schedule
- AdamW (decoupled weight decay), `lr = 2e-4`, `weight_decay = 1e-4`.
  LinearLR warmup for 5 epochs (`start_factor = 1e-2`), then
  CosineAnnealingLR for the remaining epochs to `eta_min = 1e-6`.

#### 3.6.4 Stability features
- Three independent NaN guards per training step (logits, loss,
  gradients) — each one skips the batch without poisoning weights.
  Gradient-norm clipping at `max_norm = 1.0`.

#### 3.6.5 Memory-saving features
- Mixed-precision (AMP) forward; gradient checkpointing on the
  encoder via `torch.utils.checkpoint.checkpoint_sequential` (helper
  in cell 21); gradient accumulation `grad_accum_steps = 4` to
  simulate effective batch ≈ 4 at `batch_size = 1`.

#### 3.6.6 EMA of weights and EMA-based validation
- `torch.optim.swa_utils.AveragedModel` with decay 0.999.
  **Validation runs against the EMA shadow** — this stabilises the
  checkpoint-selection signal across the last few epochs and is the
  reason the val metric tracks test more tightly than the live
  weights would.

#### 3.6.7 Checkpoint selection and early stopping
- Save best on validation ET Dice (the thesis target). Each ablation
  row copies the best file to a tagged filename
  (`singlestage3d_best__<row>.pth`). Early-stop after 15 consecutive
  validation checks without improvement (= 15 epochs at `val_every = 5`).

#### 3.6.8 Resume mechanics and configuration-hash safety
- `singlestage3d_last.pth` is overwritten every epoch with model weights,
  EMA shadow, optimiser, scheduler, history, **and the
  preprocessing-config hash** (`_prep_config_hash()`). A subsequent
  call to `train_model()` refuses to resume if the saved hash differs
  from the current CFG — protects against resuming a model trained on
  different inputs.

### 3.7 Evaluation metrics

#### 3.7.1 Sliding-window inference
- `monai.inferers.sliding_window_inference`. ROI `(96, 96, 96)`
  (matches training patch size so the receptive field carries over),
  overlap 0.5, Gaussian-weighted seams.

#### 3.7.2 Per-region Dice coefficient
- Per-region Dice on the full reconstructed volume, computed at
  threshold `CFG['et_threshold']` (default 0.5). Aggregated across
  test volumes as the arithmetic mean (per-volume Dice → mean Dice).

#### 3.7.3 95th-percentile Hausdorff distance (HD95)
- `monai.metrics.HausdorffDistanceMetric` with `percentile=95.0`,
  `distance_metric='euclidean'`. Per-region, per-volume; reported
  alongside Dice. HD95 captures boundary mistakes that Dice averages
  over.

#### 3.7.4 Threshold choice and its sensitivity
- Default threshold 0.5. Note that with three independent sigmoid heads
  the natural threshold per head is 0.5; the alternative of sweeping
  per-region thresholds on a held-out subset is *not* used in the thesis
  (kept simple, lets the divide-and-conquer story stand on the
  architectural choice, not a tuned hyperparameter).

#### 3.7.5 Reporting convention
- Per-volume Dice and HD95 → arithmetic mean across test volumes →
  reported as the headline number per region. Mention that ET sample
  size for HD95 may be smaller when ET is absent in a test volume
  (HD95 is undefined for empty masks); state the convention used
  (`HausdorffDistanceMetric(include_background=True, get_not_nans=False)`).

### 3.8 Summary

#### 3.8.1 End-to-end pipeline recap
- One paragraph, no formulas: volume → preprocessing → 96³ patch sampling →
  3D U-Net trunk → three sigmoid heads → joint loss → sliding-window
  inference → per-region Dice and HD95. Re-state the two-axis ablation
  in one sentence.

#### 3.8.2 Reproducibility statement
- Seed 42 (Python, NumPy, PyTorch, CUDA, MONAI). Volume-level split.
  Resume-aware training keyed by preprocessing-config hash. All ablation
  rows produce tagged checkpoints (`singlestage3d_best__<row>.pth`) and
  per-row result JSONs. The notebook runs end-to-end from a clean kernel
  via "Run All".

#### 3.8.3 Forward reference to Chapter 4
- Two sentences. *Chapter 4 reports the two ablation axes operationalised
  on this methodology. The headline number (Phase 5) is produced by
  the all-winners-stacked configuration; 5-fold cross-validation is
  applied to that headline number only.*

---

## 4. Pitfalls that apply to the whole chapter

- **No results in Chapter 3.** No Dice numbers, no "we found that…",
  no comparison rows. Numbers belong in Chapter 4. If a section feels
  empty without a number, write the *expected role* of that step
  instead — what it is supposed to do, not what it did.
- **No research diary language.** Replace any "earlier we tried X",
  "previous version", "first experiment" with present-tense
  methodology. The reader is told what the pipeline *is*, not what
  it once was.
- **Present-tense, plural-of-modesty or impersonal.** "The network
  outputs three sigmoid maps", not "we made the network output
  three sigmoid maps".
- **Cite the right paper at each step.** Zuiderveld 1994 (CLAHE),
  Nyúl, Udupa, Zhang 2000 (histogram standardisation), Tustison et
  al. 2010 (N4), Kervadec et al. 2019 (boundary loss), Ronneberger,
  Fischer, Brox 2015 (U-Net), Çiçek et al. 2016 (3D U-Net),
  Loshchilov & Hutter 2019 (AdamW), Loshchilov & Hutter 2016 (cosine
  annealing), Polyak & Juditsky 1992 (EMA / Polyak averaging).
- **Acknowledge the dataset's intensity state once (§3.1.3) and refer
  back to it.** Every preprocessing decision affected by it should
  point back at §3.1.3 with a short cross-reference, not re-explain.
- **One short paragraph on the hardware constraint up front (§3.2.5).**
  Patch size, batch size, GroupNorm, and the memory-saving features
  all flow from this constraint; introducing it once at the start
  makes the later choices look principled rather than ad-hoc.

---

## 5. How to use this guide with Claude

If you are an LLM continuing this thesis, the recommended workflow is:

1. Read this whole file once.
2. Skim `EXPERIMENT_FEATURES.md` for the design audit (the *why*).
3. Skim `PROGRESS.md` for historical numbers and lessons.
4. For the section you are about to write, re-read the matching
   subsection notes above.
5. Cross-check the implementation against the notebook cell listed in
   §2.2 of this guide.
6. Write the section in present-tense methodology language. No
   numbers. No research-diary phrasing. Every preprocessing decision
   that looks arbitrary should have a one-sentence justification, and
   the justification should cite either a paper (for standard
   techniques) or `EXPERIMENT_FEATURES.md` (for choices forced by
   this dataset / hardware).

If a subsection's notes here disagree with what is currently in the
notebook, the **notebook is the ground truth**. Update this guide
rather than describing a defunct version of the pipeline.
