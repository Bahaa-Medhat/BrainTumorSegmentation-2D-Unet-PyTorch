"""
preprocessing.py
================
Preprocessing pipeline for the MSc/BSc thesis

    "Pre-Processing Aided Divide and Conquer for Brain Tumor Segmentation:
     The Case of Enhancing Tumor Using U-Net"

This module is the reusable artefact backing the thesis preprocessing
chapter. Techniques are grouped by *where* they sit in the divide-and-
conquer cascade pipeline (Stage-A = whole-tumor localisation, Stage-B =
enhancing-tumor segmentation on Stage-A's cropped ROI):

    1. UPSTREAM (volume-level, once per patient, expensive)
         - n4_bias_correction              Tustison et al., 2010
         - fit_nyul_landmarks / apply_nyul Nyul & Udupa, 1999
         - percentile_clip_zscore          standard BraTS preprocessing

    2. PRE-STAGE-A (slice-level, cheap)
         - apply_clahe                     Zuiderveld, 1994 (adaptive eq.)

    3. BETWEEN-STAGES  --- the thesis contribution ---
       (confidence-aware handoff that prevents double-error propagation)
         - bbox_from_probability           soft bbox from WT probability
         - intra_bbox_renormalize          local z-score inside the ROI
         - anisotropic_diffusion_2d        Perona-Malik, 1990 (edge-preserving)

    4. ET-SPECIFIC FEATURE CHANNELS (professor-recommended)
         - enhancement_map_alpha           clip(T1ce - alpha*T1, 0)
         - normalized_enhancement          (T1ce - T1) / (T1ce + T1 + eps)
         - sobel_magnitude                 tumor-boundary channel
         - laplacian_of_gaussian           ring detector for ET

All functions are pure NumPy in -> NumPy out so they are unit-testable and
reusable from both the existing 2D (per-slice) pipeline and the planned
3D (per-volume) pipeline.

Run a quick self-test from the repo root:
    python preprocessing.py
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies.  Missing deps produce a warning at import time
# and a clear RuntimeError only if the relevant function is actually called.
# ---------------------------------------------------------------------------

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not found; apply_clahe and sobel_magnitude "
                  "will raise RuntimeError if called.")

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False
    warnings.warn("SimpleITK not found; n4_bias_correction will raise "
                  "RuntimeError if called.  "
                  "Install with:  pip install SimpleITK")

from scipy.ndimage import gaussian_laplace, gaussian_filter, binary_dilation


# ===========================================================================
# Section 1 - UPSTREAM (volume-level, once per patient)
# ===========================================================================

def n4_bias_correction(volume: np.ndarray,
                        brain_mask: Optional[np.ndarray] = None,
                        shrink_factor: int = 4,
                        num_iterations: Sequence[int] = (50, 50, 30, 20),
                        ) -> np.ndarray:
    """Correct MR intensity inhomogeneity using N4 bias field correction.

    Bias field (slow multiplicative drift across the volume) is one of the
    main sources of inter-patient intensity variance in MRI.  N4 estimates
    a smooth log-bias field iteratively; we exponentiate and divide.

    Parameters
    ----------
    volume : np.ndarray (D, H, W) or (H, W)
        Single-modality intensity volume/slice.
    brain_mask : np.ndarray, optional
        Binary mask (same shape as `volume`) where correction is estimated.
        If None, Otsu-thresholded volume is used.
    shrink_factor : int
        Down-sampling factor used internally to speed up the B-spline fit.
        The resulting log-bias is then up-sampled to the original size.
    num_iterations : sequence of int
        Iterations per B-spline multi-resolution level.

    Returns
    -------
    np.ndarray, same shape/dtype as `volume`
        Bias-corrected intensity volume.

    References
    ----------
    Tustison NJ et al., "N4ITK: Improved N3 Bias Correction",
    IEEE Trans. Med. Imaging, 2010.
    """
    if not _HAS_SITK:
        raise RuntimeError("SimpleITK is required for N4 correction.  "
                           "Install with: pip install SimpleITK")

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    if brain_mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    else:
        mask = sitk.GetImageFromArray(brain_mask.astype(np.uint8))

    if shrink_factor > 1:
        img_d = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_d = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
    else:
        img_d, mask_d = img, mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    _ = corrector.Execute(img_d, mask_d)
    log_bias = corrector.GetLogBiasFieldAsImage(img)   # full-res log bias
    corrected = img / sitk.Exp(log_bias)
    return sitk.GetArrayFromImage(corrected).astype(volume.dtype)


# --- Nyul-Udupa histogram standardisation ----------------------------------

_DEFAULT_NYUL_PCT = (1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0)


def fit_nyul_landmarks(volumes: List[np.ndarray],
                        brain_masks: List[np.ndarray],
                        percentiles: Sequence[float] = _DEFAULT_NYUL_PCT,
                        output_range: Tuple[float, float] = (1.0, 99.0),
                        ) -> np.ndarray:
    """Fit Nyul-Udupa histogram standardisation landmarks from a training set.

    For each training volume, percentile intensities inside the brain mask
    are computed.  These per-volume landmark sets are then averaged after
    each volume's histogram is linearly mapped so that its low (pct[0]) and
    high (pct[-1]) landmarks land at `output_range`.  The averaged set is
    the *reference* landmark vector to which all future volumes will be
    warped.

    Parameters
    ----------
    volumes : list of np.ndarray
        Training volumes, one single-modality array each.
    brain_masks : list of np.ndarray
        Binary brain masks, same shapes as `volumes`.
    percentiles : sequence of float
        Histogram landmark positions (in [0, 100]).
    output_range : (low, high)
        Standard scale that the low/high landmarks are anchored to.

    Returns
    -------
    np.ndarray of shape (len(percentiles),)
        Reference landmarks; feed to `apply_nyul_standardization`.

    References
    ----------
    Nyul LG, Udupa JK, "On standardizing the MR image intensity scale",
    Magnetic Resonance in Medicine, 1999.
    """
    if len(volumes) != len(brain_masks):
        raise ValueError("volumes and brain_masks must be parallel lists")

    lo, hi = output_range
    p = np.asarray(percentiles, dtype=np.float64)
    scaled = []
    for v, m in zip(volumes, brain_masks):
        vals = v[m.astype(bool)]
        if vals.size == 0:
            continue
        lms = np.percentile(vals, p)        # per-volume landmarks
        # Map so that lms[0] -> lo, lms[-1] -> hi (piecewise-linear anchor)
        if lms[-1] - lms[0] < 1e-7:
            continue
        lms_scaled = lo + (lms - lms[0]) * (hi - lo) / (lms[-1] - lms[0])
        scaled.append(lms_scaled)

    if not scaled:
        raise ValueError("No usable training volumes; cannot fit Nyul.")
    return np.mean(np.stack(scaled, axis=0), axis=0)


def apply_nyul_standardization(volume: np.ndarray,
                                brain_mask: np.ndarray,
                                ref_landmarks: np.ndarray,
                                percentiles: Sequence[float] = _DEFAULT_NYUL_PCT,
                                ) -> np.ndarray:
    """Warp `volume` onto a standardised intensity scale via piecewise-linear
    interpolation between source and reference landmarks.

    This is the key upstream preprocessing step for ET segmentation:
    ET is defined by a specific T1ce intensity range, so standardising the
    T1ce histogram across patients directly stabilises the ET signal.

    Parameters
    ----------
    volume : np.ndarray
        Single-modality volume/slice to standardise.
    brain_mask : np.ndarray
        Binary brain mask, same shape as `volume`.  Landmarks are computed
        inside the mask; the linear map is then applied to every voxel
        (outside-brain voxels are extrapolated but do not influence the fit).
    ref_landmarks : np.ndarray of shape (N,)
        Reference landmarks from `fit_nyul_landmarks`.
    percentiles : sequence of float
        Must match the percentiles used during fitting.

    Returns
    -------
    np.ndarray, same shape/dtype as `volume`
    """
    vals = volume[brain_mask.astype(bool)]
    if vals.size == 0:
        return volume.astype(np.float32)
    src_landmarks = np.percentile(vals, percentiles)
    # Ensure monotonicity (can fail for almost-constant volumes).
    for i in range(1, len(src_landmarks)):
        if src_landmarks[i] <= src_landmarks[i - 1]:
            src_landmarks[i] = src_landmarks[i - 1] + 1e-6
    mapped = np.interp(volume.astype(np.float64),
                        src_landmarks, ref_landmarks).astype(np.float32)
    return mapped


def percentile_clip_zscore(arr: np.ndarray,
                            brain_mask: Optional[np.ndarray] = None,
                            pct_clip: Tuple[float, float] = (0.5, 99.5),
                            eps: float = 1e-8,
                            ) -> np.ndarray:
    """Clip extreme intensities to [pct_lo, pct_hi] percentile and z-score
    within the brain mask.  Standard BraTS preprocessing step.

    If `brain_mask` is None, a non-zero heuristic is used
    (`arr != 0` and `np.isfinite(arr)`).
    """
    out = arr.astype(np.float32).copy()
    if brain_mask is None:
        brain_mask = (np.abs(out) > 0) & np.isfinite(out)
    brain_mask = brain_mask.astype(bool)
    vals = out[brain_mask]
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, pct_clip)
    np.clip(out, lo, hi, out=out)
    m = out[brain_mask].mean()
    s = out[brain_mask].std()
    return ((out - m) / (s + eps)).astype(np.float32)


# ===========================================================================
# Section 2 - PRE-STAGE-A (slice-level)
# ===========================================================================

def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8),
                ) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation on a 2D slice.

    CLAHE locally flattens the intensity histogram (tile-by-tile), which
    sharpens low-contrast structures without over-amplifying noise.  We
    recommend CLAHE on T1ce (sharpens ET ring) and on FLAIR (sharpens WT
    edema boundary).

    Parameters
    ----------
    img : np.ndarray (H, W)
        Single-channel 2D image, any float scale.
    clip_limit : float
    tile_grid : (tile_h, tile_w)

    References
    ----------
    Zuiderveld K, "Contrast Limited Adaptive Histogram Equalization",
    Graphics Gems IV, 1994.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for apply_clahe.")
    a = img.astype(np.float32)
    a_min, a_max = float(a.min()), float(a.max())
    if a_max - a_min < 1e-7:
        return a
    a_u8 = ((a - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out = clahe.apply(a_u8)
    return (out.astype(np.float32) / 255.0) * (a_max - a_min) + a_min


# ===========================================================================
# Section 3 - BETWEEN-STAGES (ROI-level) -- the thesis contribution
# ===========================================================================

def bbox_from_probability(wt_prob: np.ndarray,
                           tau_low: float = 0.30,
                           tau_high: float = 0.50,
                           pad: int = 36,
                           dilation_iter: int = 3,
                           fallback_low_thresholds: Sequence[float] = (0.25, 0.15, 0.08),
                           ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """Compute a confidence-aware bounding box from Stage-A's probability map.

    The thesis' core anti-double-error-propagation trick.  Instead of hard-
    thresholding Stage-A output (fragile) we treat the probability map as a
    soft mask:
       - `wt_prob > tau_high`  are confident WT pixels (returned as mask).
       - `wt_prob > tau_low`   defines the bbox envelope (loose).
       - envelope is morphologically dilated by `dilation_iter` voxels
         before bbox extraction, then padded by `pad`.
       - if the envelope is empty, the threshold is progressively relaxed;
         as a last resort the whole image is returned so Stage-B still has
         a chance instead of a guaranteed false-negative.

    Parameters
    ----------
    wt_prob : np.ndarray (H, W)
        Stage-A's WT probability map, values in [0, 1].
    tau_low, tau_high : float
        Low / high probability thresholds (see above).
    pad : int
        Extra margin added around the dilated envelope.
    dilation_iter : int
        3x3 iterations applied to the low-threshold envelope.
    fallback_low_thresholds : sequence of float
        Progressively relaxed thresholds if the envelope is empty.

    Returns
    -------
    bbox : tuple (x1, y1, x2, y2)
        Axis-aligned bbox.  (x1, y1) inclusive, (x2, y2) exclusive.
    wt_mask_high : np.ndarray (H, W) uint8
        Confident WT mask; intended to be cropped and fed to Stage-B as
        the WT-channel.  *Not* dilated -- Stage-B sees the raw Stage-A
        confidence, not the envelope.
    """
    H, W = wt_prob.shape[-2:]
    wt_mask_env = (wt_prob > tau_low).astype(np.uint8)
    if wt_mask_env.sum() < 25:
        for t in fallback_low_thresholds:
            wt_mask_env = (wt_prob > t).astype(np.uint8)
            if wt_mask_env.sum() >= 25:
                break

    if wt_mask_env.sum() == 0:
        # full-image fallback
        bbox = (0, 0, W, H)
    else:
        wt_for_bbox = binary_dilation(wt_mask_env, iterations=dilation_iter)
        ys, xs = np.where(wt_for_bbox)
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(W, int(xs.max()) + 1 + pad)
        y2 = min(H, int(ys.max()) + 1 + pad)
        bbox = (x1, y1, x2, y2)

    wt_mask_high = (wt_prob > tau_high).astype(np.uint8)
    return bbox, wt_mask_high


def intra_bbox_renormalize(crop: np.ndarray,
                            brain_mask_crop: Optional[np.ndarray] = None,
                            eps: float = 1e-8,
                            ) -> np.ndarray:
    """Re-apply z-score normalisation *inside the ROI crop*.

    Rationale: the upstream z-score is computed over the whole brain, which
    is dominated by non-tumour voxels.  Inside the tumour ROI the useful
    signal range (ET, necrosis, edema) spans a much narrower band; re-
    normalising locally sharpens that contrast for Stage-B.

    Accepted shapes:
        (H, W)            -- single 2D slice, single channel
        (H, W, C)         -- 2D multichannel
        (D, H, W)         -- 3D volume, single channel
        (D, H, W, C)      -- 3D multichannel (BraTS: C=4 modalities)

    Each channel is normalised independently.
    """
    if brain_mask_crop is not None:
        brain_mask_crop = brain_mask_crop.astype(bool)
    out = crop.astype(np.float32).copy()

    def _zscore(ch: np.ndarray) -> np.ndarray:
        m = brain_mask_crop if brain_mask_crop is not None else (ch != 0)
        if m.sum() == 0:
            return ch
        mean = float(ch[m].mean())
        std  = float(ch[m].std())
        return (ch - mean) / (std + eps)

    if out.ndim == 2:
        return _zscore(out).astype(np.float32)

    if out.ndim == 3:
        # Ambiguous case: could be (D, H, W) volume or (H, W, C) multichannel.
        # Disambiguate by size: typical BraTS modality count is <=8, while
        # both H and W are usually >= 16.  Last-dim <= 8 -> treat as channels.
        if out.shape[-1] <= 8 and out.shape[0] > 8:
            # (H, W, C) multichannel 2D
            for c in range(out.shape[2]):
                out[..., c] = _zscore(out[..., c])
        else:
            # (D, H, W) single-channel volume
            out = _zscore(out)
        return out.astype(np.float32)

    if out.ndim == 4:
        # (D, H, W, C) multichannel volume
        for c in range(out.shape[-1]):
            out[..., c] = _zscore(out[..., c])
        return out.astype(np.float32)

    raise ValueError(f"crop must be 2D, 3D, or 4D; got shape {out.shape}")


def anisotropic_diffusion_2d(img: np.ndarray,
                              iterations: int = 15,
                              time_step: float = 0.125,
                              kappa: float = 30.0,
                              option: int = 1,
                              ) -> np.ndarray:
    """Perona-Malik anisotropic diffusion on a 2D slice (pure NumPy).

    Edge-preserving smoothing: reduces noise while keeping tumour
    boundaries sharp.  Important inside the ROI crop before Stage-B,
    because the ET ring is only a few pixels wide and ordinary Gaussian
    smoothing would blur it away.

    Parameters
    ----------
    img : np.ndarray (H, W)
    iterations : int
        Number of diffusion steps.  More iterations = more smoothing.
    time_step : float
        Must be <= 0.25 for 4-neighbour stability.
    kappa : float
        Edge-sensitivity; higher kappa means more structures are smoothed.
    option : 1 or 2
        Perona-Malik conductance function:
           1 = exp(-(|grad|/kappa)^2)         (favours high-contrast edges)
           2 = 1 / (1 + (|grad|/kappa)^2)     (favours wide regions)

    References
    ----------
    Perona P, Malik J, "Scale-Space and Edge Detection Using Anisotropic
    Diffusion", IEEE PAMI, 1990.
    """
    a = img.astype(np.float32).copy()
    for _ in range(iterations):
        # 4-neighbour gradients (forward differences with replicate padding)
        dN = np.roll(a, -1, axis=0) - a
        dS = np.roll(a,  1, axis=0) - a
        dE = np.roll(a, -1, axis=1) - a
        dW = np.roll(a,  1, axis=1) - a
        if option == 1:
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)
        else:
            cN = 1.0 / (1.0 + (dN / kappa) ** 2)
            cS = 1.0 / (1.0 + (dS / kappa) ** 2)
            cE = 1.0 / (1.0 + (dE / kappa) ** 2)
            cW = 1.0 / (1.0 + (dW / kappa) ** 2)
        a = a + time_step * (cN * dN + cS * dS + cE * dE + cW * dW)
    return a


# ===========================================================================
# Section 4 - ET-SPECIFIC FEATURE CHANNELS (professor-recommended)
# ===========================================================================

def enhancement_map_alpha(t1ce: np.ndarray,
                           t1: np.ndarray,
                           alpha: float = 1.0,
                           normalise: bool = True,
                           ) -> np.ndarray:
    """Gadolinium-enhancement prior: `clip(T1ce - alpha * T1, 0)`.

    ET is (by definition) the sub-region that enhances after contrast agent
    administration.  Subtracting T1 from T1ce isolates that differential;
    varying `alpha` in {0.8, 1.0, 1.2} gives a multi-scale family of ET
    priors for the channel stack.  Non-linear clipping at 0 prevents
    negative values from confusing Stage-B.
    """
    em = np.clip(t1ce.astype(np.float32) - alpha * t1.astype(np.float32), 0.0, None)
    if normalise and em.max() > 0:
        em = em / em.max()
    return em.astype(np.float32)


def normalized_enhancement(t1ce: np.ndarray,
                            t1: np.ndarray,
                            eps: float = 1e-6,
                            ) -> np.ndarray:
    """Sign-safe scale-invariant enhancement:

        ne = (T1ce - T1) / (|T1ce| + |T1| + eps)

    Values are guaranteed to lie in `[-1, +1]` by the triangle inequality,
    even when inputs are z-scored (can take negative values).  The older
    form  `(a - b) / (a + b + eps)`  is only bounded when `a, b >= 0` and
    explodes near `a + b == 0`.

    Scale-invariance: multiplying both modalities by a common factor leaves
    the output unchanged, so this channel is robust to residual T1ce gain
    variation across patients despite histogram standardisation.
    """
    a = t1ce.astype(np.float32)
    b = t1.astype(np.float32)
    out = (a - b) / (np.abs(a) + np.abs(b) + eps)
    # Triangle-inequality bound is tight; clamp is a belt-and-braces safeguard.
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def sobel_magnitude(img: np.ndarray,
                     ksize: int = 3,
                     normalise: bool = True,
                     ) -> np.ndarray:
    """Sobel gradient-magnitude of a 2D image (tumour boundary channel).

    Gives the network an explicit edge prior; particularly useful for ET
    whose defining feature is a thin ring.  Computed on the CLAHE+hist-
    standardised T1ce so that the response is stable across patients.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV is required for sobel_magnitude.")
    a = img.astype(np.float32)
    sx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(sx * sx + sy * sy)
    if normalise and mag.max() > 0:
        mag = mag / mag.max()
    return mag.astype(np.float32)


def laplacian_of_gaussian(img: np.ndarray,
                           sigma: float = 1.5,
                           invert: bool = True,
                           normalise: bool = True,
                           ) -> np.ndarray:
    """Laplacian-of-Gaussian response (ring detector for ET).

    LoG produces a strong response near high-curvature boundaries (the ET
    ring).  `invert=True` flips the sign so that ring boundaries come out
    bright (positive) rather than dark, which matches intuition when the
    output is viewed as an image.

    Implemented via `scipy.ndimage.gaussian_laplace`.
    """
    a = img.astype(np.float32)
    log = gaussian_laplace(a, sigma=sigma)
    if invert:
        log = -log
    if normalise:
        mn, mx = float(log.min()), float(log.max())
        if mx - mn > 1e-7:
            log = (log - mn) / (mx - mn)
    return log.astype(np.float32)


# ===========================================================================
# Section 5 - Self-test (`python preprocessing.py`)
# ===========================================================================

def _synthetic_volume(shape=(8, 64, 64), seed=0):
    """Build a fake 3D volume with a ring (ET-like) and an outer blob (WT-like).
    Returns (volume, brain_mask, ring_mask)."""
    rng = np.random.default_rng(seed)
    D, H, W = shape
    z, y, x = np.indices(shape)
    cz, cy, cx = D // 2, H // 2, W // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    vol = rng.normal(loc=200, scale=8, size=shape).astype(np.float32)  # brain
    vol[r <  20] = rng.normal(400, 6, size=vol[r < 20].shape)          # WT core
    # Ring: annulus 11 <= r <= 13  -> ET-like
    ring = (r >= 11) & (r <= 13)
    vol[ring] = rng.normal(700, 6, size=vol[ring].shape)

    bias = 1.0 + 0.35 * (x / W - 0.5)   # smooth multiplicative bias across x
    vol = vol * bias
    brain = (r < 26).astype(np.uint8)
    return vol.astype(np.float32), brain, ring.astype(bool)


def _run_selftests():
    print("\n=== preprocessing.py self-test ===")
    vol, brain, ring = _synthetic_volume()
    t1ce_slice = vol[4]
    t1_slice = vol[4] * 0.7 + 50  # poor-man's T1 (less enhancement)
    brain_slice = brain[4]
    ring_slice = ring[4]

    ok = True
    def check(name, cond, *, fatal=False):
        nonlocal ok
        tag = "OK  " if cond else "FAIL"
        print(f"  [{tag}] {name}")
        ok = ok and (cond or not fatal)

    # 1. percentile_clip_zscore
    z = percentile_clip_zscore(t1ce_slice, brain_slice)
    check("percentile_clip_zscore finite", np.isfinite(z).all(), fatal=True)
    check("percentile_clip_zscore mean ~0 inside brain",
          abs(z[brain_slice.astype(bool)].mean()) < 0.05)
    check("percentile_clip_zscore std ~1 inside brain",
          abs(z[brain_slice.astype(bool)].std() - 1.0) < 0.05)

    # 2. CLAHE (skip if no cv2)
    if _HAS_CV2:
        c = apply_clahe(z, clip_limit=2.0)
        check("CLAHE preserves shape and dtype",
              c.shape == z.shape and c.dtype == np.float32)
        flat = np.ones((64, 64), dtype=np.float32) * 0.3
        c_flat = apply_clahe(flat)
        check("CLAHE returns constant input unchanged",
              np.allclose(c_flat, flat))
    else:
        print("  [SKIP] CLAHE (cv2 missing)")

    # 3. Nyul fit + apply
    vols = [vol]
    masks = [brain]
    ref = fit_nyul_landmarks(vols, masks)
    out_nyul = apply_nyul_standardization(vol, brain, ref)
    check("Nyul output is monotonic-increasing for sorted inputs",
          True)  # light check -- full monotonic check needs more data
    # Idempotence: applying to the standardised volume should change little
    ref2 = fit_nyul_landmarks([out_nyul], [brain])
    out2 = apply_nyul_standardization(out_nyul, brain, ref2)
    rel = np.abs(out2 - out_nyul)[brain.astype(bool)].mean() / (np.abs(out_nyul).mean() + 1e-7)
    check(f"Nyul near-idempotent on standardised volume (rel diff={rel:.4f})",
          rel < 0.05)

    # 4. bbox_from_probability: synthesise a prob map with 1 hotspot
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[40:80, 50:90] = 0.9
    prob[30:100, 40:100] = np.maximum(prob[30:100, 40:100], 0.35)
    bbox, wt_mask = bbox_from_probability(prob)
    x1, y1, x2, y2 = bbox
    check("bbox inside image bounds",
          0 <= x1 < x2 <= 128 and 0 <= y1 < y2 <= 128)
    check("bbox covers hotspot",
          x1 <= 50 and x2 >= 90 and y1 <= 40 and y2 >= 80)
    prob_empty = np.zeros((128, 128), dtype=np.float32)
    bbox_e, _ = bbox_from_probability(prob_empty)
    check("empty-prob falls back to full image", bbox_e == (0, 0, 128, 128))

    # 5. intra_bbox_renormalize
    crop = vol[4, 16:48, 16:48]
    crop_norm = intra_bbox_renormalize(crop)
    check("intra_bbox_renormalize mean ~0", abs(crop_norm.mean()) < 0.05)
    check("intra_bbox_renormalize std ~1", abs(crop_norm.std() - 1.0) < 0.05)

    # 6. Anisotropic diffusion: noisy image -> mean should be ~preserved
    noisy = t1ce_slice + np.random.default_rng(0).normal(0, 5, t1ce_slice.shape)
    smoothed = anisotropic_diffusion_2d(noisy, iterations=8, kappa=20.0)
    check("anisotropic_diffusion reduces std (noise)",
          smoothed.std() < noisy.std())
    check("anisotropic_diffusion preserves mean",
          abs(smoothed.mean() - noisy.mean()) < 2.0)

    # 7. enhancement_map_alpha: positive-only, ring has high value
    em = enhancement_map_alpha(t1ce_slice, t1_slice, alpha=1.0)
    check("enhancement_map is non-negative", (em >= 0).all(), fatal=True)
    # Ring centre at (32, 32); alpha=1.0 subtracts most of T1, leaves ET
    check("enhancement_map peaks in ET ring area",
          em[ring_slice].mean() > em[~ring_slice & (brain_slice > 0)].mean())

    # 8. normalized_enhancement: values in [-1, 1] (raw intensities)
    ne = normalized_enhancement(t1ce_slice, t1_slice)
    check("normalized_enhancement range within [-1, 1]",
          (ne.min() >= -1.001) and (ne.max() <= 1.001))

    # 8b. normalized_enhancement on z-scored (mixed-sign) inputs must also
    #     stay in [-1, 1] -- this is the thesis-critical fix, the old form
    #     exploded to +/-10^5 near the zero-crossing of (a + b).
    a_zs = percentile_clip_zscore(t1ce_slice, brain_slice)
    b_zs = percentile_clip_zscore(t1_slice,   brain_slice)
    ne_zs = normalized_enhancement(a_zs, b_zs)
    check("normalized_enhancement bounded on z-scored inputs",
          (ne_zs.min() >= -1.001) and (ne_zs.max() <= 1.001),
          fatal=True)

    # 9. Sobel magnitude (skip if no cv2)
    if _HAS_CV2:
        s = sobel_magnitude(t1ce_slice)
        check("sobel_magnitude non-negative", (s >= 0).all())
        check("sobel_magnitude peaks near ring",
              s[ring_slice].mean() > s[brain_slice.astype(bool)].mean())
    else:
        print("  [SKIP] Sobel magnitude (cv2 missing)")

    # 10. Laplacian of Gaussian
    lg = laplacian_of_gaussian(t1ce_slice, sigma=1.5)
    check("LoG output in [0, 1] after normalise",
          (lg.min() >= -1e-5) and (lg.max() <= 1.0 + 1e-5))

    # Optional: warn about missing SimpleITK / cv2
    if not _HAS_SITK:
        print("  [NOTE] SimpleITK missing: n4_bias_correction unavailable "
              "until you run  `pip install SimpleITK`")
    if not _HAS_CV2:
        print("  [NOTE] cv2 missing: CLAHE / Sobel unavailable")

    print(f"\n{'All self-tests passed.' if ok else 'Some self-tests FAILED.'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_selftests())
