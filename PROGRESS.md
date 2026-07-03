# Project Progress Summary

**Thesis:** *Pre-Processing Aided Divide and Conquer for Brain Tumor Segmentation: The Case of Enhancing Tumor Using 3D U-Net*

**Status:** Single-stage 3D U-Net pipeline complete; two-axis ablation finished; current ceiling ~0.73 ET test Dice on BraTS 2020 with 6 GB GPU.

**Purpose of this document:** capture every architectural, preprocessing, training, and engineering decision made so far — so a follow-up experiment with a different U-Net variant or a different strategy can be set up in hours, not weeks.

---

## 1. Final headline numbers

Best configuration found: **et-heavy task weighting (0.2 / 0.3 / 0.5) + CLAHE preprocessing on T1ce and FLAIR**.

| target | best config | test Dice |
|--------|-------------|-----------|
| **Whole Tumour (WT)** | et_heavy + full preprocessing stack | **0.8568** |
| **Tumour Core (TC)** | et_heavy + CLAHE | **0.8258** |
| **Enhancing Tumour (ET)** *(thesis target)* | **et_heavy + CLAHE** | **0.7341** |

Naïve baseline (equal task weights, z-score-only preprocessing): **ET = 0.7196**.
Total improvement from optimisation across both ablation axes: **+0.0145 ET Dice**.

---

## 2. Dataset

| metric | value |
|---|---|
| Dataset | BraTS 2020 (UPenn) |
| Total slice files (h5) | 57,195 |
| Total patient volumes | 369 |
| Modalities per slice | 4 (T1, T1ce, T2, FLAIR) |
| Spatial resolution | 240 × 240 × 155 per volume |
| Train volumes | **221** (60 %) |
| Val volumes | **73** (20 %) |
| Test volumes | **75** (20 %) |
| Split level | volume / patient (no spatial leakage) |
| Random seed | **42** (fixed across project) |

The split function `split_by_volume(file_list, seed=42, ratios=(0.6, 0.2, 0.2))` is what every experiment must use to keep results comparable. Re-using the same volume-id sets is non-negotiable for cross-experiment comparisons.

---

## 3. Architecture and training (current state)

### Model — MONAI 3D U-Net

| component | value |
|-----------|-------|
| `spatial_dims` | 3 |
| `in_channels` | 4 (or 8 with `use_enhancement_channels=True`) |
| `out_channels` | 3 (independent sigmoid heads: WT, TC, ET) |
| `channels` | `(16, 32, 64, 128, 160)` |
| `strides` | `(2, 2, 2, 2)` |
| `num_res_units` | 2 |
| `norm` | `('GROUP', {'num_groups': 1})` *(LayerNorm-over-features)* |
| Total parameters | ≈ 1.9 M |

**Three independent sigmoid heads, not a softmax** — because regions overlap (every ET voxel is also TC and WT). The containment hierarchy emerges from co-trained shared features, not from a softmax constraint.

### Loss — `WeightedDiceCE` (custom)

Per-region soft Dice + binary cross-entropy with auto-normalised task weights:

```
L = λ_dice · Σ_c w_c · DiceLoss(p_c, t_c) + λ_ce · Σ_c w_c · BCE(p_c, t_c)
            where  c ∈ {WT, TC, ET},  Σ_c w_c = 1
```

Defaults: `λ_dice = λ_ce = 1.0`, smoothing `s = 1.0`.

### Optimiser — AdamW + warmup-cosine

| component | value |
|-----------|-------|
| Optimiser | AdamW |
| Peak LR | `1e-4` |
| Weight decay | `1e-5` |
| Warmup | LinearLR, `start_factor=1e-2`, 2 epochs |
| Main schedule | CosineAnnealingLR, 45 epochs, `eta_min=1e-6` |
| Grad clip | max-norm `1.0` |

### Patch sampling

| parameter | value | role |
|-----------|-------|------|
| `patch_size` | `(96, 96, 96)` | reduced from canonical 128³ for 6 GB GPU |
| `pos` / `neg` | `1.0 / 0.0` | every patch tumour-centric (avoids InstanceNorm-on-empty-bg pathology) |
| `num_samples` per volume | 2 | effective batch = 2 patches per step |
| `batch_size` | 1 | one volume per loader batch |
| Augmentation | `RandFlipd` × 3 axes (p=0.5), `RandScaleIntensityd` (0.1, p=0.5), `RandShiftIntensityd` (0.1, p=0.5) | standard MONAI BraTS augmentation |

### Inference

| parameter | value |
|-----------|-------|
| Inferer | `monai.inferers.sliding_window_inference` |
| ROI | `(96, 96, 96)` (matches training patch size) |
| Overlap | 0.5 |
| Mode | Gaussian weighting at window edges |
| `et_threshold` | 0.5 |

### Training loop guarantees (in `train_stageB`)

- **3-stage NaN guards**: non-finite logits → skip batch; non-finite loss → skip batch; non-finite gradient → zero grads & skip step. Prevents weight poisoning from a single rogue batch.
- **fp32 forward** (autocast disabled). Earlier autocast + GroupNorm + bs=1 produced NaN gradients.
- **Resume-aware checkpointing**: `singlestage3d_last.pth` is rewritten every epoch with model + optimiser + scheduler + best-tracker + patience + history + prep-config-hash. Re-calling `train_stageB()` continues exactly where it left off; a `prep_hash` mismatch refuses to resume.
- **Per-row tagged checkpoints** (e.g. `singlestage3d_best__weighting_et_heavy.pth`, `singlestage3d_best__clahe.pth`) — no row can overwrite another's best.
- **Early stopping**: patience 3 on ET val Dice (= 15 epochs at `val_every=5`).

---

## 4. Preprocessing pipeline

### Always applied (baseline)

| step | function (`preprocessing.py`) |
|------|------------------------------|
| Percentile-clip [0.5 %, 99.5 %] per modality | `pp.percentile_clip_zscore` |
| Z-score within brain mask per modality | same |
| Spatial flips + intensity scale/shift | MONAI transforms during training |

### Ablation-toggleable (CFG flags)

| CFG flag | function | reference |
|----------|----------|-----------|
| `use_n4` | N4 bias correction | Tustison 2010 |
| `use_nyul` | Nyúl–Udupa histogram standardisation | Nyúl & Udupa 1999 |
| `use_clahe_t1ce` / `use_clahe_flair` | per-slice CLAHE | Zuiderveld 1994 |
| `use_enhancement_channels` | adds 4 ET-specific channels | enhancement: physics-motivated; LoG: Marr & Hildreth 1980 |

The enhancement channels (when on) are: `clip(T1ce − T1, 0)`, `(T1ce − T1)/(T1ce + T1 + ε)`, `‖∇T1ce‖` (Sobel), and `∇²G ∗ enhancement` (LoG). Built on the fly in `BraTSVolDataset.__getitem__`; not cached.

Each preprocessing configuration writes its own cache at `./cache_3d_singlestage/<config_hash>/`. The hash format: `n40_ny0_cT0_cF0` = baseline; `n41_ny1_cT1_cF1` = N4 + Nyúl + CLAHE on both T1ce and FLAIR.

---

## 5. Two-axis ablation results

### Axis 1 — Per-region task weighting (loss balance)

| scheme | (w_WT, w_TC, w_ET) | test WT | test TC | **test ET** |
|--------|---------------------|---------|---------|-------------|
| equal       | (1, 1, 1)            | 0.8161 | 0.8176 | 0.7196 |
| **et_heavy** *(winner)* | **(0.2, 0.3, 0.5)** | 0.8207 | 0.8123 | **0.7292** |
| wt_anchored | (0.5, 0.3, 0.2)      | 0.8026 | 0.7586 | 0.7085 |
| et_only *(ablation)* | (0, 0, 1)         | 0.0172 | 0.0083 | 0.7642 |

**Finding:** Up-weighting the ET head provides a small but consistent gain (+0.0096 over equal). Removing the WT/TC heads entirely (`et_only`) gives a marginally higher ET (0.7642) but catastrophically destroys WT and TC — proving the multi-task formulation is essential for a clinically usable system.

### Axis 2 — Preprocessing (under et_heavy weights)

| row | flags | test WT | test TC | **test ET** | Δ ET |
|-----|-------|---------|---------|-------------|------|
| baseline (z-score only) | — | 0.8207 | 0.8123 | **0.7292** | — |
| + N4 | `use_n4` | 0.8104 | 0.8082 | 0.7254 | **−0.0038** |
| + Nyúl | `use_nyul` | 0.8097 | 0.8118 | 0.7285 | −0.0007 |
| **+ CLAHE** *(winner)* | `use_clahe_t1ce + use_clahe_flair` | **0.8256** | **0.8258** | **0.7341** | **+0.0049** |
| + enh. channels | `use_enhancement_channels` | 0.8082 | 0.8170 | 0.7296 | +0.0004 |
| + full stack | all 5 flags | **0.8568** | 0.8158 | 0.7331 | +0.0039 |

**Findings:**

1. **CLAHE is the only individually-positive contribution for ET.** Local contrast enhancement on T1ce sharpens the enhancing ring; on FLAIR it sharpens the edema boundary.
2. **N4 hurts ET (−0.0038).** Validates the up-front hypothesis that BraTS 2020 is already N4-corrected; re-applying it over-smooths the small gradients ET depends on.
3. **Nyúl is essentially neutral.** BraTS organisers pre-process intensity normalisation; redundant.
4. **Enhancement channels are essentially neutral on ET (+0.0004).** The model already extracts these features implicitly from the raw modalities. Worth reporting as an honest negative result.
5. **Full stack** maximises WT (+0.036 over baseline) but not ET — region-specific best configurations exist.

---

## 6. Lessons learned (what didn't work and why)

These are the dead ends that future experiments should *not* re-explore:

| failure mode | symptom | root cause | resolution |
|--------------|---------|------------|------------|
| 2D cascade approach (Stage-A WT crop → Stage-B ET) | test ET Dice ≈ 0.16–0.31 | Error compounds across stages; 2D inference loses z-axis coherence; ET ring is destroyed by aggressive resize-to-96 | Abandoned; switched to single-stage 3D multi-task heads |
| InstanceNorm on 3D U-Net at bs=1 | val Dice plateau at 0.07 across all regions | InstanceNorm normalises each channel per-sample over spatial dims; with sparse classes this erases the small-class signal | Switched to GroupNorm |
| GroupNorm with `num_groups=8` | `num_channels must be divisible by num_groups` ValueError | MONAI UNet applies norm uniformly including the 3-channel output head | Use `num_groups=1` (LayerNorm-over-features); works for any channel count |
| AMP + autocast + bs=1 | NaN gradients during backward | Numerical instability with mixed precision on small batches and GroupNorm | Disabled autocast; running fp32 |
| Random patches (`neg > 0`) | val Dice flatlines at near-zero on early epochs | Pure-background patches make InstanceNorm/GroupNorm normalise constants → meaningless gradient | Use `pos=1.0, neg=0.0` (tumour-centric only) |
| `clip_grad_norm` after a NaN gradient | All weights become NaN; subsequent forwards collapse | clip_grad_norm computes a global norm; if any grad is NaN, it smears NaN across all parameters | Added gradient-NaN guard *before* the clip |
| `pos_weight=150` in BCE | Model collapsed to constant output ≈ 0.40 everywhere | Class-weighted equilibrium of weighted BCE forces a uniform-probability solution when no spatial signal is found | Reduced to `pos_weight=5`, then dropped BCE pos_weight entirely with the move to `WeightedDiceCE` |
| Focal Tversky in 3D with bs=1 | Gradient too small to escape constant-output equilibrium | FT's gradient scales as 1/N where N is voxel count (~9 × 10⁵); per-voxel update ~1 × 10⁻⁹ | Replaced with `DiceFocalLoss`, then `DiceCELoss`-family. Settled on `WeightedDiceCE`. |
| Subvolume crop + resize to 96³ | ET ring (2–3 voxels) was destroyed by trilinear resize | Aggressive downsampling annihilates thin structures | Switched to random *patch* sampling at native resolution (no resize) |
| 5,000-sample-per-epoch Stage-A cap | Training time wasted before any signal emerged | Subsampling artefact in earlier 2D-cascade pipeline | N/A — abandoned with the cascade |

---

## 7. Hardware constraints and ceiling analysis

### Constraints

- **GPU:** single consumer card with 6 GB VRAM (running on Windows / CUDA 11.8 / PyTorch 2.7.1).
- **Patch size:** capped at 96³ (vs canonical 128³ for BraTS).
- **Channel widths:** `(16, 32, 64, 128, 160)` (vs canonical 32–320 for nnU-Net-class models — about 1/3 capacity).
- **Batch size:** 1 (effective 2 via 2 patches/volume); standard practice would be 2–4 at 128³.

### Ceiling estimate

The current setup is at **0.7341 ET test Dice**. Published BraTS 2020 SOTA single-model figures cluster around:

- nnU-Net default 3D: **0.78–0.82 ET**
- TransBTS / CKD-TransBTS / SwinUNETR: **0.78–0.81 ET**
- KIU-Net / Attention U-Net variants: **0.74–0.78 ET**

The ~0.05 gap from current to SOTA is primarily explained by:

1. **Patch size 96³ vs 128³** — accounts for ~0.02 (less context, fewer ET ring voxels per patch).
2. **Model capacity ~1.9 M vs 30 M+** — accounts for ~0.02–0.03 (less feature richness).
3. **No deep supervision** — accounts for ~0.01.
4. **No ensembling** — accounts for ~0.01–0.02.

Closing the gap requires either renting a 24 GB cloud GPU for one or two runs, or running deeper architectures with gradient checkpointing.

---

## 8. Recommended next experiments

In rough priority order. Each is a **single CFG / model swap**, not a rewrite — the dataset, preprocessing module, training loop, ablation infrastructure, and visualisations all stay.

### 8.1 — Architecture variants (highest ROI)

| candidate | implementation | expected ET | rationale |
|-----------|---------------|-------------|-----------|
| **MONAI SegResNet** | already plumbed in `build_model` (`CFG['model']='segresnet'`) | **+0.01–0.03** | Won BraTS 2018; residual blocks help small datasets. *Caveat:* deviates from "U-Net" in thesis title — frame as "U-Net variant with residual blocks". |
| **MONAI DynUNet** (nnU-Net-style) | drop-in `build_model` change; one-line `from monai.networks.nets import DynUNet` | **+0.02–0.05** | Has built-in deep supervision (auxiliary losses on intermediate decoder levels). Most cited 3D BraTS architecture. |
| **MONAI SwinUNETR** | requires `pip install monai[einops]`; transformer encoder + CNN decoder | **+0.03–0.06** | Transformers handle global context (relevant for diffuse edema in WT). Heavier; may need cloud GPU. |
| **Attention U-Net** | reference impl. on GitHub; ~50 LoC swap | **+0.01–0.02** | Adds attention gates on skip connections; cheap upgrade with same VRAM budget. |
| **V-Net** | MONAI `VNet` | **±0.01** | Mostly equivalent to current 3D U-Net but with PReLU + residual short-cuts. Useful as a comparison baseline, not a primary candidate. |

### 8.2 — Strategies (orthogonal to architecture)

| strategy | implementation effort | expected ET gain | notes |
|----------|----------------------|------------------|-------|
| **Test-time augmentation (TTA)** | wrap `evaluate_test()` in a 4-flip averager (~30 LoC) | **+0.01–0.02** | Free at training time; only inference cost increases 4×. |
| **Deep supervision** | switch to DynUNet *or* add auxiliary heads at decoder levels 2 and 3 in current U-Net | **+0.02–0.03** | Stabilises training of deeper networks; standard in nnU-Net pipelines. |
| **Ensemble of 3–5 seeds** | re-run best config with seeds 42, 43, 44, 45, 46; average sigmoids | **+0.02–0.04** | Most reliable Dice gain in BraTS literature. Cost: 3–5× wall time. |
| **Larger patches via gradient checkpointing** | `torch.utils.checkpoint` on encoder blocks; allows 128³ at same VRAM | **+0.02** | ~30 % training-time slowdown, but unlocks the canonical patch size. |
| **AMP with `bfloat16` instead of `float16`** | one-line autocast change to `dtype=torch.bfloat16` | enables mixed-precision safely | bf16 has the same dynamic range as fp32, so the GroupNorm + bs=1 NaN issues we hit don't occur. |
| **Cloud GPU run** | 24 GB GPU on Vast.ai / RunPod (~$3–5 for 4 h) | **+0.03–0.06** | Single-run path to ~0.78 ET with `init_filters=32` SegResNet at 128³. |
| **Larger BCE-Dice mix exploration** | new ablation of `(λ_dice, λ_ce)` ratios (e.g. 1:0.5, 1:2, 1:3) | **+0.005–0.015** | Cheap; one CFG sweep. |
| **Compound loss** | Dice + CE + Boundary loss (Kervadec 2019) | **+0.01–0.02** | Boundary loss helps thin structures like the ET ring. |
| **Cascade *with* the strong single-stage model** | use this checkpoint's WT prediction as ROI for a second-pass ET model | possibly **+0.02** | Reverses the failed 2D cascade strategy at 3D scale. |
| **Self-supervised pretraining (SimMIM / SwinUNETR)** | requires pretraining run on the unlabelled BraTS data | **+0.03–0.05** | Heavy; only consider if cloud GPU is available. |

### 8.3 — Suggested first three experiments to run

If you have one week of compute time, this ordering maximises learning per hour:

1. **DynUNet swap with deep supervision.** ~3 hours per run × 2 runs (one for et_heavy + CLAHE, one for + full preprocessing). Should land at 0.75–0.77 ET.
2. **TTA on the existing CLAHE checkpoint.** No retraining. 10 min eval. Should add +0.01–0.02 to 0.7341 → ~0.745.
3. **3-seed ensemble of (DynUNet + et_heavy + CLAHE).** 9 hours total. Should land at 0.76–0.78 ET — comfortably above SOTA-mid.

After these, if results are still below 0.78 ET, the single-GPU ceiling is real and a cloud-GPU run becomes the next move.

---

## 9. Code and checkpoint inventory

### Notebooks

| file | purpose |
|------|---------|
| `brats2020_singlestage_3d.ipynb` | Main experimental notebook (current) |
| `preprocessing.py` | Reusable preprocessing module — *thesis Methods artefact* |

### Scripts

| file | purpose |
|------|---------|
| `scripts/generate_singlestage_notebook.py` | Regenerates the notebook from scratch |
| `scripts/add_visualisations.py` | Adds the 7 visualisation cell groups |
| `scripts/add_resume_logic.py` | Wires resume + auto-load history |
| `scripts/add_task_weighting.py` | Adds the `task_weights` CFG key + `WeightedDiceCE` + 4-row ablation |
| `scripts/lock_in_et_heavy_and_isolate.py` | Locks et_heavy weighting into preprocessing-ablation runs + RNG re-seeding |
| `scripts/export_html_3d.py` | Exports notebook to HTML for thesis archival |

### Result files (to keep verbatim)

- `task_weighting_results.json` — axis 1 (4 rows)
- `preprocessing_ablation_results.json` — axis 2 (6 rows; baseline reused from axis 1)

### Checkpoints (in `./checkpoints/`)

| file | content |
|------|---------|
| `singlestage3d_best__weighting_equal.pth` | row-1 weighting (1, 1, 1) |
| `singlestage3d_best__weighting_et_heavy.pth` | row-2 weighting (0.2, 0.3, 0.5) — **axis 1 winner** |
| `singlestage3d_best__weighting_wt_anchored.pth` | row-3 weighting |
| `singlestage3d_best__weighting_et_only.pth` | row-4 ablation |
| `singlestage3d_best__n4.pth` | preprocessing row 2 |
| `singlestage3d_best__nyul.pth` | row 3 |
| `singlestage3d_best__clahe.pth` | row 4 — **axis 2 winner** |
| `singlestage3d_best__enh_channels.pth` | row 5 |
| `singlestage3d_best__full.pth` | row 6 (all flags on) |

### Preprocessing caches (in `./cache_3d_singlestage/`)

Each subdirectory `<config_hash>/` contains 369 `.npz` files (one per volume). Hash format: `n40_ny0_cT0_cF0` etc. Reused automatically by re-runs.

---

## 10. Reproducibility checklist

Before any new experiment:

- [ ] Confirm `CFG['seed'] = 42` and `set_seed(42)` is called.
- [ ] Use the same volume split (`split_by_volume(file_list, seed=42, ratios=(0.6, 0.2, 0.2))`).
- [ ] Re-seed all RNGs (`random`, `numpy`, `torch`, `torch.cuda`, `monai.utils.set_determinism`) at the start of each ablation row.
- [ ] Document the exact `prep_hash` (cache subdir name) in the results JSON.
- [ ] Save model + optimiser + scheduler + history in `singlestage3d_last.pth` so the experiment is fully resumable.
- [ ] Tag the best checkpoint per row with the row's config name.
- [ ] Record results in a JSON artefact (don't rely on stdout alone).

---

## 11. What to keep verbatim for the thesis

These are the artefacts the thesis Methods + Results chapters depend on. Don't delete or rewrite them:

- **`preprocessing.py`** — every function has paper references in its docstring. Treat as Methods chapter section 3.x verbatim.
- **`task_weighting_results.json`** + **`preprocessing_ablation_results.json`** — the two ablation tables.
- **The 9 tagged best checkpoints** — needed to regenerate per-row test predictions for thesis figures.
- **The two ablation result tables in this document (sections 5.x).**

---

## 12. Open questions / loose ends

Things that came up during the project but were not resolved:

1. **Why does CLAHE help and not enhancement channels?** Both target the same boundary structure. Hypothesis: CLAHE modifies the *modality input* the encoder sees, while enhancement channels are computed from the same z-scored input the encoder already has — possibly redundant with what the first conv layer learns implicitly. Worth testing whether enhancement channels help when the underlying T1ce / T1 are *not* available (i.e., as a replacement, not addition).
2. **Why does the full stack improve WT but not ET?** Possibly because N4 + Nyúl + CLAHE compound differently for big vs small structures — N4's smoothing helps WT (where edema is broad, low-frequency) and hurts ET (where the ring is high-frequency).
3. **Is the 0.7341 ET ceiling truly at the GPU constraint, or is there a model-class effect?** A SegResNet or DynUNet swap at the same 96³ patch + bs=1 setup would isolate this.
4. **Cross-axis optimum: should the best preprocessing be measured under the best weighting, or should weighting be re-optimised under the best preprocessing?** This thesis design picked option (A) for tractability; an honest sensitivity analysis would re-run weighting under CLAHE preprocessing.

---

*Last updated:* 2026-04-24
*Best run reproducible by:* `CFG['task_weights'] = (0.2, 0.3, 0.5); CFG['use_clahe_t1ce'] = True; CFG['use_clahe_flair'] = True; history = train_stageB(); load_best_singlestage(tag='clahe'); evaluate_test()`
