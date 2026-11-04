"""Frozen constants, statistics and losses for Section 5.2 (LLV).

Source of truth: section5.2-cleaned-specification.md, frozen 2026-10-06.
Where the older section5-restructure-plan.md or llv-section5.2-test-design.md
disagree, this module follows the cleaned specification.

Everything here is float64 on the reduction path; the archive stays float32.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

# --------------------------------------------------------------- geometry --
T_WIN = 400                 # one scored window, in saved steps
K_LAGS = 400                # retained lags, k = 0 .. 399
RAW_ROLLOUT = 1200          # raw recurrent steps in the final rollout
RAW_TRAIN_UNROLL = 800      # raw recurrent steps during training

# Explicit raw-step slices. Never "the last `lags` steps" (spec section 8).
SLICE_WARMUP = (0, 400)
SLICE_G_IN = (400, 800)
SLICE_G_OUT = (800, 1200)
SLICE_COMBINED = (400, 1200)

# Target archive array indices. Array index 0 is physical t = 1, so physical
# t = 401..800 is traj[400:800] (spec section 3).
W_K = 400
W_G = 400
TARGET_G_IN = (400, 800)
TARGET_G_OUT = (800, 1200)
TARGET_COMBINED = (400, 1200)

# ------------------------------------------------------------------ split --
SPLIT_F = (0, 400)
SPLIT_V = (4000, 8000)
SPLIT_T_EV = (8000, 20000)
N_REF_BLOCKS = 9


def ref_block(j: int) -> tuple:
    """Trajectory index range of reference block R_j, j = 1..9."""
    if not 1 <= j <= N_REF_BLOCKS:
        raise ValueError("reference block index must be 1..9, got %r" % (j,))
    return (400 + 400 * (j - 1), 400 + 400 * j)


# ---------------------------------------------------------------- families --
FAMILIES = ("ACF", "CCF", "ACF2", "PDF", "PROJ")

# ------------------------------------------------------------------- frames --
# Three non-coordinate orthonormal frames. The paper reports n_f = 4 frames in
# total (one coordinate frame plus these three); the code argument is
# n_noncoordinate_frames = 3. A bare `n_frames` field is forbidden.
N_NONCOORDINATE_FRAMES = 3
N_TOTAL_FRAMES_REPORTED = 4

_S2 = np.sqrt(2.0)
_S5 = np.sqrt(5.0)
FRAMES = np.array(
    [
        [[1.0 / _S2, 1.0 / _S2], [1.0 / _S2, -1.0 / _S2]],
        [[1.0 / _S5, 2.0 / _S5], [2.0 / _S5, -1.0 / _S5]],
        [[1.0 / _S5, -2.0 / _S5], [2.0 / _S5, 1.0 / _S5]],
    ],
    dtype=np.float64,
)
#: the six non-coordinate directions, in frame/row order
PROJ_DIRECTIONS = FRAMES.reshape(6, 2)
N_PROJ = PROJ_DIRECTIONS.shape[0]

# ---------------------------------------------------------------------- KDE --
GRID_N = 513
GRID_LO, GRID_HI = -5.0, 5.0
GRID = np.linspace(GRID_LO, GRID_HI, GRID_N, dtype=np.float64)

# Frozen EFFECTIVE bandwidth values, never exponent denominators and never
# recomputed from a generated ensemble's sample count (spec section 4).
_TB = 160000
BANDWIDTH = 0.091028210151304        # (T*B)^(-1/5)
BANDWIDTH_2D = 0.13572088082974534   # (T*B)^(-1/6), evaluation only

# ----------------------------------------------------------------- streams --
SEED_BASE = {
    "param_init": 0,
    "train_noise": 1000,
    "checkpoint_monitor": 2000,
    "classification": 3000,
    "final_generation": 4000,
    "density_visualization": 5000,
    "stability": 6000,
}
SEED_LONGLAG_BOOTSTRAP = 7000
#: The burn-in/stationarity resampling seed is the one that produced the frozen
#: spec 3.1 table in verify_and_calibrate_v2.py. It is deliberately NOT 7001:
#: that gate reproduces an existing calibration with real provenance, so its
#: generator is inherited rather than renumbered.
SEED_BURNIN_BOOTSTRAP = 20260913

#: Spec 3.1, frozen. Late reference half, tested half, and the diagnostics.
BURNIN_REFERENCE_HALF = (4000, 6000)
BURNIN_TESTED_HALF = (6000, 8000)
BURNIN_LATE_T = (901, 1200)            # physical, inclusive
BURNIN_TESTED_T = 400                  # physical, the last discarded step
BURNIN_N_REPLICATES = 200
BURNIN_FROZEN = {                      # observed, limit
    "mean": (0.0407, 0.1141),
    "variance": (0.0782, 0.1692),
    "ks": (0.0234, 0.0337),
}
BURNIN_FROZEN_DWIN = {"ACF": 0.0192, "CCF": 0.0199, "ACF2": 0.0291}
DWIN_TOLERANCE = 0.10

N_RUNS = 20
RUN_SEEDS = tuple(range(N_RUNS))

# --------------------------------------------------- training configuration --
TRAIN_CONFIG = {
    "lstm_layers": 2,
    "hidden_units": 10,
    "input_width": 4,
    "d_in_active": 2,
    "d_out": 2,
    "iterations": 10000,
    "optimizer": "Adam",
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "batch": 400,
    "raw_train_unroll": RAW_TRAIN_UNROLL,
    "scored_slice": SLICE_G_IN,
    "loss_form": "relative",
    "n_noncoordinate_frames": N_NONCOORDINATE_FRAMES,
    "loss_balancing": True,
    "sigma_parameterization": "exp",   # sigma_s = exp(alpha_s), alpha_s(0) = 0
    "sigma_alpha_init": 0.0,
    "validation_cadence": 200,         # events at 200, 400, ..., 10000
    "validation_events": 50,
    "n_runs": N_RUNS,
}

CLASSIFICATION_THRESHOLD = 10.0        # r_val <= 10, inclusive. Never 3.
B_GEN = 400
B_VIS = 12000

LONGLAG_K_LO, LONGLAG_K_HI = 400, 600
LONGLAG_N_BOOT = 2000
LONGLAG_BLOCK = 400
LONGLAG_CURVES = ("ACF_A", "ACF_B", "CCF_AB", "CCF_BA")

SLIDING_STARTS = tuple(range(400, 801, 20))   # 21 windows


# ============================================================ statistics ====
def _as_block(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 2:
        raise ValueError("expected (T, B, 2) block, got shape %r" % (a.shape,))
    return a


def pooled_moments(block) -> tuple:
    """Pooled mean and population s.d. over (time, trajectory), float64."""
    a = _as_block(block)
    mean = a.mean(axis=(0, 1), dtype=np.float64)
    var = ((a - mean) ** 2).mean(axis=(0, 1), dtype=np.float64)
    return mean, np.sqrt(var)


def corr_window(block, n_lags: int = K_LAGS) -> np.ndarray:
    """Window-normalized correlation matrix, manuscript Eq. (corr_estimator).

    C_ij(k) = 1/(sigma_i sigma_j) * 1/(N T) *
              sum_n sum_{t=0}^{T-1-k} (y_i[t,n] - ybar_i)(y_j[t+k,n] - ybar_j)

    The divisor is N*T for every lag, so the estimator carries the finite
    window factor (1 - k/T). Both ensembles in any comparison use it, so the
    comparison is unbiased even though the curves are not.

    Returns (2, 2, n_lags): out[i, j, k] = C_ij(k).
    """
    a = _as_block(block)
    n_t, n_b, dim = a.shape
    if n_lags > n_t:
        raise ValueError("n_lags %d exceeds window length %d" % (n_lags, n_t))
    mean, sd = pooled_moments(a)
    xc = a - mean

    n_fft = 1 << int(np.ceil(np.log2(2 * n_t)))
    f = np.fft.rfft(xc, n_fft, axis=0)
    out = np.empty((dim, dim, n_lags), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            # conj(F_i) * F_j gives sum_t xc_i[t] xc_j[t+k] at index k.
            cc = np.fft.irfft(np.conj(f[:, :, i]) * f[:, :, j], n_fft, axis=0)
            out[i, j] = cc[:n_lags].sum(axis=1) / (n_b * n_t) / (sd[i] * sd[j])
    return out


def corr_window_bruteforce(block, n_lags: int = K_LAGS) -> np.ndarray:
    """Direct O(T*K) reference implementation of corr_window, for fixtures."""
    a = _as_block(block)
    n_t, n_b, dim = a.shape
    mean, sd = pooled_moments(a)
    xc = a - mean
    out = np.empty((dim, dim, n_lags), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            for k in range(n_lags):
                s = (xc[: n_t - k, :, i] * xc[k:, :, j]).sum(dtype=np.float64)
                out[i, j, k] = s / (n_b * n_t) / (sd[i] * sd[j])
    return out


def corr_paircount(block, k_lo: int, k_hi: int) -> np.ndarray:
    """Pair-count-normalized estimator, spec section 10. Evaluation only.

    Divisor is B*(T_win - k), so each lag is normalized by the pairs it
    actually has. Returns (4, k_hi - k_lo + 1) in the fixed curve order
    ACF_A, ACF_B, CCF_AB, CCF_BA.
    """
    a = _as_block(block)
    n_t, n_b, _ = a.shape
    if not 0 <= k_lo <= k_hi < n_t:
        raise ValueError("lag range %d..%d invalid for length %d"
                         % (k_lo, k_hi, n_t))
    mean, sd = pooled_moments(a)
    xc = a - mean

    n_fft = 1 << int(np.ceil(np.log2(2 * n_t)))
    f = np.fft.rfft(xc, n_fft, axis=0)
    lags = np.arange(k_lo, k_hi + 1)
    pairs = (n_t - lags).astype(np.float64) * n_b

    def cross(i, j):
        cc = np.fft.irfft(np.conj(f[:, :, i]) * f[:, :, j], n_fft, axis=0)
        return cc[k_lo:k_hi + 1].sum(axis=1) / pairs / (sd[i] * sd[j])

    return np.stack([cross(0, 0), cross(1, 1), cross(0, 1), cross(1, 0)])


def corr_paircount_bruteforce(block, k_lo: int, k_hi: int) -> np.ndarray:
    """Direct reference implementation of corr_paircount, for fixtures."""
    a = _as_block(block)
    n_t, n_b, _ = a.shape
    mean, sd = pooled_moments(a)
    xc = a - mean
    rows = []
    for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
        row = []
        for k in range(k_lo, k_hi + 1):
            s = (xc[: n_t - k, :, i] * xc[k:, :, j]).sum(dtype=np.float64)
            row.append(s / (n_b * (n_t - k)) / (sd[i] * sd[j]))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def kde_direct(samples, bandwidth: float = BANDWIDTH,
               grid: np.ndarray = GRID, chunk: int = 200000) -> np.ndarray:
    """Gaussian KDE, spec section 4 / manuscript Eq. (kde1d), evaluated on grid.

    f(z) = 1/(M h) sum_m K((z - s_m)/h), K the standard Gaussian.

    `bandwidth` is an EFFECTIVE bandwidth value, never an exponent denominator
    and never recomputed from len(samples).

    samples: (..., n_series). All leading axes pool into the sample axis.
    returns: (len(grid), n_series) float64.
    """
    _check_bandwidth(bandwidth)
    v = np.asarray(samples, dtype=np.float64)
    v = v.reshape(-1, v.shape[-1])
    n_samp, n_ser = v.shape
    norm = 1.0 / (n_samp * bandwidth * np.sqrt(2.0 * np.pi))
    out = np.zeros((grid.size, n_ser), dtype=np.float64)
    for lo in range(0, n_samp, chunk):
        blk = v[lo:lo + chunk]
        for c in range(n_ser):
            u = (grid[:, None] - blk[None, :, c]) / bandwidth
            out[:, c] += np.exp(-0.5 * u * u).sum(axis=1)
    return out * norm


#: target element count per KDE chunk; bounds the (grid x chunk x series)
#: temporary, which is also the autograd tape footprint on the training path.
KDE_CHUNK_ELEMENTS = 2.0e7


def _kde_chunk(n_grid: int, n_ser: int, requested) -> int:
    if requested is not None:
        c = int(requested)
        if c < 1:
            raise ValueError("sample_chunk_size must be >= 1")
        return c
    return max(1, int(KDE_CHUNK_ELEMENTS // max(1, n_grid * n_ser)))


def _check_bandwidth(bandwidth: float) -> None:
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be a positive value, got %r"
                         % (bandwidth,))
    if bandwidth >= 1.0:
        raise ValueError(
            "bandwidth %r looks like an exponent denominator, not a bandwidth; "
            "the legacy `bw` interface is forbidden here" % (bandwidth,))


def kde_torch(samples, bandwidth: float = BANDWIDTH,
              grid: np.ndarray = GRID, sample_chunk_size=None,
              checkpoint: bool = False):
    """The spec section 4 canonical Gaussian-kernel sum, in torch.

    This is the same literal estimator as :func:`kde_direct`, evaluated in
    sample chunks so that the (grid x chunk x series) temporary stays bounded.
    It is differentiable with respect to `samples`: the only operations are
    subtraction, scaling, `exp` and summation, all smooth everywhere. No
    binning, no `floor`, no FFT.

    `checkpoint=True` wraps each chunk in `torch.utils.checkpoint`, so the
    backward pass recomputes the kernel instead of storing it; the autograd
    footprint then drops to a single chunk. This is the training path.

    Chunking changes only the summation order, so the result differs from the
    unchunked sum by float rounding alone; gate 5 asserts that against the
    NumPy direct formula.
    """
    import torch

    _check_bandwidth(bandwidth)
    v = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
    v = v.reshape(-1, v.shape[-1])
    n_samp, n_ser = int(v.shape[0]), int(v.shape[1])
    g = torch.as_tensor(np.asarray(grid, dtype=np.float64),
                        dtype=v.dtype, device=v.device)
    n_grid = int(g.shape[0])
    chunk = _kde_chunk(n_grid, n_ser, sample_chunk_size)

    def kernel_sum(blk):
        u = (g[:, None, None] - blk[None, :, :]) / bandwidth
        return torch.exp(-0.5 * u * u).sum(dim=1)

    out = torch.zeros(n_grid, n_ser, dtype=v.dtype, device=v.device)
    for lo in range(0, n_samp, chunk):
        blk = v[lo:lo + chunk]
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as _ckpt
            part = _ckpt(kernel_sum, blk, use_reentrant=False)
        else:
            part = kernel_sum(blk)
        out = out + part
    return out / (n_samp * bandwidth * float(np.sqrt(2.0 * np.pi)))


def kde_eval(samples, bandwidth: float = BANDWIDTH,
             grid: np.ndarray = GRID) -> np.ndarray:
    """Evaluation-side canonical KDE, returned as float64 NumPy.

    Routes to :func:`kde_torch` when torch is importable, because the same
    kernel sum runs orders of magnitude faster there (and on a GPU when one is
    visible), and falls back to :func:`kde_direct` otherwise. Both branches
    evaluate the identical formula in float64; gate 5 is what licenses the
    substitution.
    """
    try:
        import torch
    except ImportError:
        return kde_direct(samples, bandwidth, grid)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.as_tensor(np.ascontiguousarray(
        np.asarray(samples, dtype=np.float64)), dtype=torch.float64, device=dev)
    with torch.no_grad():
        return kde_torch(t, bandwidth, grid).cpu().numpy()


def projection_scales(fit_block) -> np.ndarray:
    """Six projection scales from the fitting block, float64, ddof = 0.

    Directions and scales are returned separately and must stay separate: do
    not fold the scale into the direction vector (spec section 4).
    """
    a = _as_block(fit_block)
    proj = np.einsum("tbd,wd->tbw", a, PROJ_DIRECTIONS)
    mean = proj.mean(axis=(0, 1), dtype=np.float64)
    var = ((proj - mean) ** 2).mean(axis=(0, 1), dtype=np.float64)
    return np.sqrt(var)


def projected_samples(block, scales) -> np.ndarray:
    """u_w = (w . x) / sigma_w for the six directions."""
    a = _as_block(block)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (N_PROJ,):
        raise ValueError("expected %d projection scales, got %r"
                         % (N_PROJ, scales.shape))
    return np.einsum("tbd,wd->tbw", a, PROJ_DIRECTIONS) / scales


def family_arrays(block, scales, n_lags: int = K_LAGS,
                  families=FAMILIES) -> dict:
    """The five family arrays in the frozen order of spec section 4.

    ACF  : ACF_A, ACF_B                          (2, n_lags)
    CCF  : CCF_AB, CCF_BA                        (2, n_lags)
    ACF2 : ACF2_A, ACF2_B                        (2, n_lags)
    PDF  : PDF_A, PDF_B                          (2, 513)
    PROJ : the six directions in frame/row order (6, 513)
    """
    want = tuple(families)
    for s in want:
        if s not in FAMILIES:
            raise ValueError("unknown family %r" % (s,))
    a = _as_block(block)
    out = {}
    if {"ACF", "CCF"} & set(want):
        c = corr_window(a, n_lags)
        if "ACF" in want:
            out["ACF"] = np.stack([c[0, 0], c[1, 1]])
        if "CCF" in want:
            out["CCF"] = np.stack([c[0, 1], c[1, 0]])
    if "ACF2" in want:
        c2 = corr_window(a ** 2, n_lags)
        out["ACF2"] = np.stack([c2[0, 0], c2[1, 1]])
    if "PDF" in want:
        out["PDF"] = kde_eval(a).T
    if "PROJ" in want:
        out["PROJ"] = kde_eval(projected_samples(a, scales)).T
    return {s: out[s] for s in FAMILIES if s in out}


def relative_loss(arr_a, arr_b) -> float:
    """L_s(A, B) = ||S(A) - S(B)||_2^2 / ||S(B)||_2^2, spec section 4.

    Squared relative discrepancy. No square root, no second square, no L1
    term, no per-curve normalization. The entry count cancels, so this equals
    the ratio of the two mean squares.
    """
    a = np.asarray(arr_a, dtype=np.float64).ravel()
    b = np.asarray(arr_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("family arrays differ in shape: %r vs %r"
                         % (a.shape, b.shape))
    den = float(np.dot(b, b))
    if not np.isfinite(den) or den <= 0.0:
        raise ValueError("target family norm is not finite and positive")
    return float(np.dot(a - b, a - b) / den)


def family_losses(stats_a: dict, stats_b: dict) -> dict:
    return {s: relative_loss(stats_a[s], stats_b[s]) for s in FAMILIES}


def q_val(losses: dict) -> float:
    """Q_val = sum_s L_s. The L_s are already squared; do not square again."""
    return float(sum(losses[s] for s in FAMILIES))


def ratios(losses: dict, ref: dict) -> dict:
    return {s: losses[s] / ref[s] for s in FAMILIES}


def worst_ratio(r: dict) -> float:
    return float(max(r[s] for s in FAMILIES))


# ------------------------------------------------------------ admissibility --
def exceedance(rho) -> np.ndarray:
    """v = max(-rho_A, -rho_B, rho_A - 1, rho_B - 1, rho_A + rho_B - 1)."""
    r = np.asarray(rho, dtype=np.float64)
    a, b = r[..., 0], r[..., 1]
    return np.maximum.reduce([-a, -b, a - 1.0, b - 1.0, a + b - 1.0])


def admissibility(rho) -> dict:
    """Rates and maximum exceedance, spec section 11.

    Non-finite states count as invalid and stay in the denominator;
    max_exceedance is taken over finite states only so NaN cannot propagate.
    """
    r = np.asarray(rho, dtype=np.float64)
    total = int(r.shape[0] * r.shape[1])
    finite = np.isfinite(r).all(axis=-1)
    n_nonfinite = int(total - finite.sum())
    v = np.full(finite.shape, np.nan)
    v[finite] = exceedance(r[finite])
    violating = finite & (v > 0.0)
    n_violating = int(violating.sum())
    if finite.any():
        max_exc = float(np.maximum(v[finite], 0.0).max())
    else:
        max_exc = float("nan")
    return {
        "nonfinite_rate": n_nonfinite / total,
        "constraint_rate": n_violating / total,
        "invalid_rate": (n_nonfinite + n_violating) / total,
        "max_exceedance": max_exc,
        "n_samples": total,
    }


# ------------------------------------------------------------------ manifest --
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


#: names inside a manifest directory
MANIFEST_NAME = "manifest.json"
FREEZE_TOKEN_NAME = "GATES_PASSED"
SUPERSEDED_NAME = "SUPERSEDED"
#: the modules whose checksums the freeze covers
GATED_SOURCES = ("llv_spec.py", "llv_streams.py", "llv_gates.py")


class FinalTargetLocked(RuntimeError):
    """Raised when T_ev is touched before the gates and manifest freeze."""


class ArchiveLoader:
    """Archive access with the final segment locked until the gates pass.

    `open_final()` raises unless a manifest freeze token has been written, so
    an accidental read of [8000, 20000) fails loudly rather than silently.
    """

    def __init__(self, path: str, manifest_dir: str = None):
        self.path = path
        self.manifest_dir = manifest_dir
        self._traj = None
        self._unlocked = None

    @property
    def traj(self) -> np.ndarray:
        if self._traj is None:
            self._traj = np.load(self.path)["traj"]
        return self._traj

    def _slice(self, idx: tuple, window: tuple) -> np.ndarray:
        return np.asarray(
            self.traj[window[0]:window[1], idx[0]:idx[1], :], dtype=np.float64)

    def fitting(self, window=TARGET_G_IN):
        return self._slice(SPLIT_F, window)

    def validation(self, window=TARGET_G_IN):
        return self._slice(SPLIT_V, window)

    def reference(self, j: int, window=TARGET_G_IN):
        return self._slice(ref_block(j), window)

    def assert_unlocked(self) -> dict:
        """Verify the full chain before T_ev may be read.

        The existence of a freeze token proves nothing on its own: it can
        outlive the manifest it was written for, the archive can be replaced,
        and the estimator code or the specification can change underneath it.
        All four are therefore re-checked, every time:

          token -> manifest    the token must hold the manifest's current sha
          manifest -> archive  the recorded archive sha must still match
          manifest -> code     llv_spec/llv_gates/llv_streams must be unchanged
          manifest -> spec     the specification document must be unchanged

        Returns the verified manifest.
        """
        if self._unlocked is not None:
            return self._unlocked
        if not self.manifest_dir:
            raise FinalTargetLocked(
                "T_ev is locked: no manifest directory was given to "
                "ArchiveLoader, so the gate result cannot be verified")
        d = self.manifest_dir
        tok_p = os.path.join(d, FREEZE_TOKEN_NAME)
        man_p = os.path.join(d, MANIFEST_NAME)
        for p in (tok_p, man_p):
            if not os.path.exists(p):
                raise FinalTargetLocked("T_ev is locked: %r does not exist" % p)
        if os.path.exists(os.path.join(d, SUPERSEDED_NAME)):
            raise FinalTargetLocked(
                "T_ev is locked: manifest directory %r is marked superseded" % d)

        token = open(tok_p).read().strip()
        man_sha = sha256_file(man_p)
        if token != man_sha:
            raise FinalTargetLocked(
                "T_ev is locked: freeze token %s... does not match the current "
                "manifest sha %s...; the manifest changed after the gates ran"
                % (token[:12], man_sha[:12]))
        with open(man_p) as fh:
            man = json.load(fh)

        want_arch = man.get("archive_sha256")
        got_arch = sha256_file(self.path)
        if want_arch != got_arch:
            raise FinalTargetLocked(
                "T_ev is locked: archive %r has sha %s... but the manifest was "
                "frozen against %s..."
                % (self.path, got_arch[:12], str(want_arch)[:12]))

        for name, want in (man.get("source_sha256") or {}).items():
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
            if not os.path.exists(path):
                raise FinalTargetLocked(
                    "T_ev is locked: gated source %r is missing" % name)
            got = sha256_file(path)
            if got != want:
                raise FinalTargetLocked(
                    "T_ev is locked: %s has changed since the gates ran "
                    "(%s... now %s...); rerun the gates into a new manifest "
                    "directory" % (name, str(want)[:12], got[:12]))

        spec_p = man.get("specification_path")
        spec_want = man.get("specification_sha256")
        if not spec_p or not spec_want:
            raise FinalTargetLocked(
                "T_ev is locked: the manifest records no specification "
                "checksum; rerun the gates with --spec")
        if not os.path.exists(spec_p):
            raise FinalTargetLocked(
                "T_ev is locked: specification %r is missing" % spec_p)
        spec_got = sha256_file(spec_p)
        if spec_got != spec_want:
            raise FinalTargetLocked(
                "T_ev is locked: the specification has changed since the gates "
                "ran (%s... now %s...). Per its own preamble a changed "
                "specification requires a new version and a complete rerun."
                % (str(spec_want)[:12], spec_got[:12]))

        self._unlocked = man
        return man

    def open_final(self, window=TARGET_G_IN):
        self.assert_unlocked()
        return self._slice(SPLIT_T_EV, window)


def extract_windows(rollout: np.ndarray) -> dict:
    """Split one continuous 1200-step rollout into its named raw windows.

    The only sanctioned way to obtain G_in and G_out. Explicit slices, never
    "the last `lags` steps" (spec section 8). The caller must pass the output
    of a SINGLE uninterrupted rollout; the recurrent state is not reset at raw
    step 800, so slicing a concatenation of two separate rollouts here would
    silently answer a different question.
    """
    a = np.asarray(rollout)
    if a.shape[0] != RAW_ROLLOUT:
        raise ValueError("expected %d raw steps, got %d"
                         % (RAW_ROLLOUT, a.shape[0]))
    return {
        "warmup": a[SLICE_WARMUP[0]:SLICE_WARMUP[1]],
        "G_in": a[SLICE_G_IN[0]:SLICE_G_IN[1]],
        "G_out": a[SLICE_G_OUT[0]:SLICE_G_OUT[1]],
        "combined": a[SLICE_COMBINED[0]:SLICE_COMBINED[1]],
    }


def standardize(rho, mu, sd) -> np.ndarray:
    return (np.asarray(rho, dtype=np.float64) - mu) / sd


def inverse_standardize(x, mu, sd) -> np.ndarray:
    """rho = s * x + mu. No clipping, no rounding to the lattice."""
    return np.asarray(x, dtype=np.float64) * sd + mu
