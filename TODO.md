# Thesis TODO — Pre-experiment hygiene → Notebook polish → Experiment phases

Three sequential workstreams. Do **not** start the experiment phases until both verification and polish are complete — running 15 experiments and then discovering CLAHE was buggy or a markdown was wrong wastes a week of compute.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done

---

## STAGE 1 — Preprocessing technique verification

**Goal:** prove every preprocessing transform produces the expected output before any experiment uses it. Catch silent bugs (wrong channel order, wrong intensity scale, alignment drift between image and label, etc.) now, not after thesis runs are done.

### 1.1 — Build a single-volume probe cell

A new diagnostic cell that, given one example volume, runs each preprocessing variant in turn and shows:
- 3-slice axial mosaic per modality (raw + processed side by side)
- Intensity histograms inside the brain mask before/after
- Numerical sanity (mean, std, min, max) per modality
- Tumour overlay (WT/TC/ET contours) to confirm spatial alignment is preserved

- [ ] Add cell `verify_preprocessing(vid)` rendering all of the above for one volume
- [ ] Add cell `verify_preprocessing_pipeline(vid)` that runs the *full active pipeline* (whatever flags are on) and shows the cumulative effect

### 1.2 — Per-technique unit checks

For each technique, an explicit assertion-based check that the output passes basic sanity tests. **Each item needs a small Python check that prints PASS/FAIL.**

#### Baseline (always on)
- [ ] **Percentile clip [0.5, 99.5]**: max output ≤ p99.5 of brain voxels; min ≥ p0.5
- [ ] **Z-score per modality**: brain-mask mean ≈ 0 (|µ| < 1e-3) and std ≈ 1 (|σ−1| < 1e-2) for each modality
- [ ] **Brain mask preserved**: zero-voxels outside the brain mask still ≈ 0 after preprocessing (no leakage)

#### Optional (ablation flags)
- [ ] **N4 bias correction**: output preserves the spatial label-mask alignment (label voxels still align with their tissue), and the brain-internal histogram width contracts vs raw (bias removal flattens the distribution)
- [ ] **Nyúl–Udupa**: post-Nyúl percentile landmarks of the volume match the reference landmarks within tolerance (this is what Nyúl is *supposed* to do)
- [ ] **CLAHE 2D**: per-slice variance ≥ 90 % of raw (no information loss); local-window contrast (e.g. mean abs gradient inside the brain) increases vs raw
- [ ] **CLAHE 3D**: same checks as 2D + adjacent-slice intensity correlation ≥ that of CLAHE-2D (3D should *reduce* per-slice banding artefacts)
- [ ] **Histogram Equalisation**: output histogram inside brain ≈ uniform (chi-square against uniform within tolerance); per-slice contrast much higher than CLAHE (will likely *hurt* later — that's the point of including it)
- [ ] **Unsharp masking**: high-frequency content (Laplacian energy) increases; low-frequency content unchanged
- [ ] **Enhancement map (α=1.0)**: `clip(T1ce − T1, 0)` is non-negative everywhere; peaks coincide with GT ET region (verify via masked mean: enh inside ET > enh inside non-ET WT)
- [ ] **Normalised enhancement**: values in `[−1, +1]`; high values cluster on GT ET
- [ ] **Sobel | T1ce**: peaks coincide with tumour boundary (compute mean Sobel magnitude on `dilated(WT) ∖ eroded(WT)` ring vs interior — boundary band should be > 1.5× interior)
- [ ] **LoG of enhancement**: peak response on the enhancing ring (boundary mean > interior mean)

### 1.3 — Cross-cutting checks

- [ ] **Determinism**: running the full pipeline twice on the same volume produces bit-identical output (assert with `np.array_equal`)
- [ ] **Cache hash**: each unique CFG flag combination resolves to a unique cache subdirectory (test by flipping one flag at a time and confirming a new dir appears)
- [ ] **SDT cache**: signed-distance transform produces negative values inside GT, positive outside, zero on the boundary (verify on a synthetic ball mask before trusting on real data)
- [ ] **HD95 metric**: synthetic test — known shapes with known HD95 (a 1-voxel-shifted ball should have HD95 ≈ 1)

### 1.4 — Sign-off

- [ ] All 1.2 checks PASS for at least 3 different volumes (small, medium, large tumour)
- [ ] Visual inspection of 1.1 mosaics confirms no obvious artefacts
- [ ] Add a `## Preprocessing verification report` markdown cell summarising the PASS/FAIL table — this becomes Methods Chapter Section 3.x.6 evidence

---

## STAGE 2 — Notebook polishing

**Goal:** the notebook should be presentable to the supervisor / examiner. Remove debugging artefacts, replace dev-time print statements with thesis-quality markdown context, and ensure top-to-bottom execution order is correct.

### 2.1 — Inventory & cleanup

- [ ] List every cell with its current purpose (markdown vs code), and tag each one as: **keep**, **rewrite**, **merge with adjacent**, **delete**
- [ ] **Delete** dev-time exploration cells (smoke tests, one-off debugging prints, intermediate checkpoint inspection cells, anything labelled "test" or "debug")
- [ ] **Delete** commented-out code blocks left from earlier iterations (every `# Uncomment to ...` directive that's now obsolete)
- [ ] **Delete** redundant import cells; consolidate all imports into the single imports cell at the top
- [ ] **Merge** small consecutive code cells that logically belong together (e.g. the model build + EMA setup + criterion + optimiser + scheduler should be one cell, not five)

### 2.2 — Markdown polishing

The notebook has 15+ markdown cells (M1–M15 from earlier audit). Each needs a final review.

- [ ] **M1 (title)** — already in thesis-defence form; confirm references list is current
- [ ] **M2–M15** — re-read each. Replace any references to "early experiments", "we tried X earlier", "previous version" with present-tense thesis-method language. Examiners should see a clean methodology, not a research diary
- [ ] **Add a short markdown header before *every* code cell group** explaining what that section does (Dataset, Preprocessing, Architecture, Loss, Training, Inference, Visualisation, Ablation runners). The notebook should read like a paper, not a script
- [ ] **Number the sections explicitly** (3.1 Dataset, 3.2 Preprocessing, …) so they map 1-to-1 onto Chapter 3 of the thesis

### 2.3 — Code cell polishing

- [ ] Remove inline `# debug` / `# TODO` / `# fix later` comments
- [ ] Rename ad-hoc variables (e.g. `_smoke_vid`, `_x`, `_tmp`) to thesis-quality names
- [ ] Consolidate `print(...)` statements that fire on import (e.g. "EMA enabled with decay=...") into a single `notebook_summary()` call at end of setup
- [ ] Remove `warnings.warn` calls that no longer fire (legacy guards from features now resolved)
- [ ] Ensure every cell can run with **`Run All`** top-to-bottom from a fresh kernel without errors

### 2.4 — Visualisation cells

- [ ] Each visualisation has a concise markdown caption above it (one sentence: what it shows, why it matters) — like a figure caption in the thesis
- [ ] Visualisations should *not* show debug overlays (e.g. raw probability heatmaps that don't appear in the thesis); only show outputs you'd put in Chapter 4
- [ ] Add a final visualisation cell: **"Headline test prediction"** — once trained, picks the median-Dice volume from the test set and renders the WT / TC / ET overlays. This figure goes directly into Chapter 4

### 2.5 — Reproducibility checks

- [ ] Restart kernel + Run All works end-to-end on a freshly cloned repo
- [ ] All cells that produce thesis figures save those figures to `./figures/` in addition to displaying inline (so they can be embedded in the LaTeX thesis without re-running the notebook)
- [ ] `nbconvert --to html` works without warnings (already wired via `scripts/export_html_3d.py`)

### 2.6 — Sign-off

- [ ] Notebook table-of-contents auto-renders (the markdown headings produce a navigable outline)
- [ ] Total cell count down to ~50 (currently 44+; aim *down*, not up)
- [ ] No more than **one** code cell exceeds 60 lines — anything longer is a sign that logic should be in a `.py` module

---

## STAGE 3 — Experiment phases (the actual thesis runs)

**Pre-conditions:**
- Stage 1 complete (preprocessing verified)
- Stage 2 complete (notebook polished)
- One Phase-0 verification run finished without NaN/OOM

**Conventions for every run:**
- Each row saves: `singlestage3d_best__<phase>_<row>.pth` and `results__<phase>_<row>.json`
- Each result JSON includes per-volume Dice **and** HD95 for WT, TC, ET
- All non-varying CFG values stay at the locked-in baseline (Methodology Chapter 3.x)

### Phase 0 — Verification run *(do this last, after Stage 1+2 are done)*

- [ ] Short run (~20 epochs) at the locked baseline config (equal weights, baseline preprocessing, 3D U-Net) to confirm: training loss decreases, NaN guards do not fire, EMA shadow updates, val_loop computes both Dice and HD95, checkpoint saves include all expected fields
- [ ] **Do not** include this in thesis numbers

### Phase 1 — Baseline establishment (Chapter 4.1)

- [ ] **Run B1**: equal weights (1.0, 1.0, 1.0), baseline preprocessing (z-score only), 3D U-Net — full **50 epochs** (locked epoch budget after the Phase-0 + B1 timing audit), save as `__base.pth`. This is the reference number every other row is compared against.

Cost: ~2.5 h.

### Phase 2 — Axis 1: Divide-and-Conquer (Chapter 4.2)

5 additional runs varying `task_weights`, baseline preprocessing, 3D U-Net.

- [ ] **W2**: wt_anchored `(0.5, 0.3, 0.2)`
- [ ] **W3**: et_heavy `(0.2, 0.3, 0.5)`
- [ ] **W4**: et_strong `(0.15, 0.20, 0.65)`
- [ ] **W5**: et_extreme `(0.05, 0.15, 0.80)`
- [ ] **W6**: et_only `(0.0, 0.0, 1.0)`
- [ ] Pick the **operational winner** `W_best`: highest ET Dice subject to WT > 0.5 AND TC > 0.5

Cost: ~12.5 h (5 runs × ~2.5 h).

### Phase 3 — Axis 2: Preprocessing (Chapter 4.3)

8 runs at `W_best`, 3D U-Net.

- [ ] **P1**: baseline at `W_best` (clean re-run for fair comparison)
- [ ] **P2**: + N4 *(control / negative result)*
- [ ] **P3**: + Nyúl–Udupa *(control)*
- [ ] **P4**: + CLAHE on T1ce + FLAIR
- [ ] **P5**: + Histogram Equalisation *(control: shows local > global)*
- [ ] **P6**: + Unsharp masking
- [ ] **P7**: + ET-specific feature channels (enhancement, normEnh, Sobel, LoG)
- [ ] **P8**: + **3D CLAHE** *(thesis-original)*
- [ ] **P9**: full positive stack (best individually-positive rows combined)
- [ ] Pick the **preprocessing winner** `P_best`

Cost: ~20 h (8 runs × ~2.5 h).

#### Optional sub-experiment (if time permits)
- [ ] **CLAHE clip_limit sweep**: re-run `P4` (or `P8` if 3D wins) at clip ∈ {1.0, 3.0, 4.0} (2.0 is the default in `P4`/`P8`). Reports hyperparameter sensitivity. **Skip if compute is tight.**

### Phase 4 — Axis 3: Architecture (Chapter 4.4)

4 runs at `W_best`, `P_best`, varying architecture.

- [ ] **A2**: SegResNet
- [ ] **A3**: Attention U-Net
- [ ] **A4**: U-Net++ (`BasicUNetPlusPlus`)
- [ ] **A5**: SwinUNETR (small, `feature_size=24`)
- [ ] Pick the **architecture winner** `A_best`

Cost: ~10 h (4 runs × ~2.5 h).

### Phase 5 — Headline configuration (Chapter 4.5)

- [ ] **H1**: `(A_best, W_best, P_best)` — the all-winners-stacked run. This is the thesis' top-line ET Dice number.

Cost: ~2.5 h.

### Phase 6 — Literature comparison (Chapter 4.6)

No new training. Just a markdown table.

- [ ] Compile published BraTS 2020 ET Dice numbers (nnU-Net 0.82, SegResNet ensemble ~0.78, A4-Unet 0.73, others) alongside `H1`'s number
- [ ] Discuss the methodological trade-offs (single-stage, no ensembling, 6 GB GPU)

### Phase 7 — Optional ensembles (Chapter 4.7) *(if time permits)*

- [ ] **E1, E2**: train two more seeds (43, 44) of the headline configuration
- [ ] Average sigmoids at inference; report ensemble ET Dice
- [ ] Expected gain: +0.02–0.04 ET Dice

Cost: ~5 h (2 extra seeds × ~2.5 h).

---

## Total compute estimate (3 tiers)

| tier | included phases | total runs | wall time |
|---|---|---|---|
| Minimum viable | 1, 2, 5 | 7 | ~17.5 h (~1 day) |
| **Recommended** | 1, 2, 3, 5, 6 | 15 | ~37.5 h (~1.5 days) |
| Comprehensive | all phases (incl. clip-limit sweep + arch + ensembles) | ~25 | ~62 h (~2.5 days) |

---

## Thesis-writing parallel work

Chapter 3 (Methodology) does **not** depend on any experiment results — it can be written in parallel with Stage 1 and Stage 2.

- [ ] Chapter 3.1 — Dataset (BraTS 2020, splits, preprocessing) — write while Stage 1 verification runs
- [ ] Chapter 3.2 — Preprocessing pipeline (each technique with citation, formula, role) — write while Stage 1 verification runs
- [ ] Chapter 3.3 — Architecture (3D U-Net, channels, GroupNorm, multi-task heads) — write during Stage 2 polish
- [ ] Chapter 3.4 — Loss (`WeightedDiceCE` + boundary loss + HD95 metric) — write during Stage 2 polish
- [ ] Chapter 3.5 — Training procedure (AdamW, schedule, EMA) — write during Stage 2 polish
- [ ] Chapter 3.6 — Two-axis ablation methodology (sequential ablation rationale, why not full grid) — write during Stage 2 polish
- [ ] Chapter 4 — Results — write *after* each phase completes; don't wait until all phases are done

---

## Checkpoint inventory (current state)

Existing checkpoints from earlier runs that should **not** be overwritten before they're either (a) included in the thesis as historical evidence or (b) explicitly retired:

| file | what it represents |
|---|---|
| `singlestage3d_best__weighting_equal.pth` | old run, equal weights, baseline preproc |
| `singlestage3d_best__weighting_et_heavy.pth` | old run, et_heavy, baseline preproc — **prior best ET 0.7292** |
| `singlestage3d_best__weighting_wt_anchored.pth` | old run, control |
| `singlestage3d_best__weighting_et_only.pth` | old run, ET 0.7642 (degenerate WT/TC) |
| `singlestage3d_best__clahe.pth` | old run, et_heavy + CLAHE — **prior best ET 0.7341** |
| (others) | listed in PROGRESS.md |

For thesis purposes, **rename or move all `__legacy_<date>/` before starting Phase 1** so the new naming scheme is clean.

- [ ] Move existing checkpoints to `./checkpoints/legacy_pre_thesis_runs/`
- [ ] Move existing JSONs (`task_weighting_results.json`, `preprocessing_ablation_results.json`) to `./legacy_results/`
- [ ] Confirm fresh runs cannot accidentally load a legacy checkpoint (verify by deleting the active `singlestage3d_last.pth`)
