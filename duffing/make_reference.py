#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2 of 3 -- standardisation constants, analytic reference densities and the
Gaussian-copula null, for each target archive written by make_target.py.

No SciPy: the normal quantile function and the root find are implemented here,
so the whole pipeline runs on numpy alone. `--regress-scipy` checks both against
SciPy when it happens to be installed.

Three jobs:

1. Fix the standardisation constants the whole pipeline shares.

   mu, scale   per-component mean and standard deviation, estimated on the
               FITTING BLOCK ONLY. Not on the whole archive: these constants
               enter the training objective, so estimating them from all N
               trajectories would let validation and evaluation data influence
               training preprocessing.

               The mean of this target is known to vanish (V is even), so
               subtracting a sample mean adds about 1% of sigma in noise and
               breaks the exact q -> -q symmetry. It is subtracted anyway, to
               keep one definition across Sections 4, 5.1 and 5.2 -- Section 5.2
               genuinely needs it, since the LLV densities sit near 0.19 and
               0.36. The manuscript should say the mean is known to vanish here
               and is subtracted for uniformity, so a reader does not read it as
               an oversight.

   s_w         one scale per projection direction, also from the fitting block.

   This affine map is a COORDINATE CHANGE, fixed once. It is not the centering
   that the correlation estimators perform: those subtract each measured
   ensemble's own empirical mean, per the manuscript's definition. Two separate
   steps; evaluate.py does both.

   Projections are taken on the STANDARDISED components,
       z_w = (w . q_tilde) / s_w,   q_tilde = (q - mu) / scale,
   not on the raw ones. The manuscript standardises components before forming
   statistics, and scale varies with lambda (0.691 / 0.763 at lambda = 0 against
   0.589 / 0.609 at lambda = 4), so a direction set defined on raw coordinates
   would drift between couplings and the orthogonality of the pairs would mean
   nothing.

2. Evaluate the analytic reference densities by quadrature -- component
   marginals, the six training projections, four held-out projections and the
   bivariate joint -- in the standardised frame, each CONVOLVED WITH THE SAME
   GAUSSIAN KERNEL the KDE losses apply to the generated ensemble. Comparing a
   smoothed estimate against an unsmoothed truth would charge a perfect
   generator a fixed smoothing bias.

3. Compute kappa_1122 / (var1 var2) for the target and for a matched
   Gaussian-copula null (same two marginals, same Pearson correlation, Gaussian
   dependence). The gap in magnitude is the fourth-order dependence that
   marginals plus correlation cannot account for, and it is what picks the
   lambda for the detailed projected-PDF ablation.

A note on the projection set. At lambda = 0 the two components are independent
and both marginals are even, so a*q1 + b*q2 and a*q1 - b*q2 have the same law
and the six directions collapse to three distinct densities. No choice of kappa
prevents this -- it follows from independence alone. That is harmless, since at
lambda = 0 there is no joint structure to capture, but it must not be described
as six independent constraints. For lambda > 0 the coupling breaks the
q2 -> -q2 symmetry and all six differ; measured L1 separations between
sign-flipped pairs are 0.02-0.07 at lambda = 1..4, against 0.0005 from the
kappa asymmetry alone. The coupling does the separating, not the kappa choice.

Provenance. The reference stores the sha256 of the target archive it was built
from and a hash over its own constants. evaluate.py refuses to pair a reference
with a target whose hash differs, and refuses a generated ensemble whose stored
reference hash differs -- otherwise overwriting an archive silently re-pairs old
references with new data.

Outputs, per lambda:
    reference_lam<X>.npz    densities, grids, constants, directions, split, hashes
    reference_lam<X>.json   scalars: moments, kappa_1122, the null, bandwidths

Usage
-----
    python make_reference.py --datadir data
    python make_reference.py --datadir data --regress-scipy
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

import make_target as MT
from make_target import gibbs_grid, gibbs_moments

SCHEMA_VERSION = 3

# Bumped for the loss contract: the constants hash now also covers l_seq, the
# CCF mode and the projection aggregation rule, so a training run cannot pair
# with a reference that defines the loss differently. References built by
# schema 2 must be rebuilt -- seconds, and the trajectory archives are untouched.

ESTIMATOR_VERSION = 1
PROJ_AGGREGATION = "concat-2nf-then-normalized-mse"

# PROJ_AGGREGATION: the selected 2*n_f projected densities are concatenated into
# one stack and a single normalized MSE is taken over it. The direction count
# appears in the numerator and the denominator alike, so raising n_f adds
# constraints without mechanically inflating the term. Do NOT average
# per-direction relative errors instead -- that is a different quantity and
# would no longer match the evaluator.

# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------

B_FIT = 400
N_VAL = 4000
N_REF_BLOCKS = 9
K_LAGS = 200

# The retained lag array is 0 .. L-1, i.e. k_max = 199. This is the FPU/LLV
# convention (k_max = L-1; there 399 of L = 400) applied at this section's
# L = 200 -- audit B9 -- not a new choice.
#
# The earlier K = 100 was defended as "lags beyond ~150 carry no signal". That is
# true of the signal but is not a reason to truncate: the biased estimator tapers
# by (L-k)/L on BOTH ensembles identically, so lag 199 enters weighted by 1/200.
# It can therefore neither dominate the denominator nor inject meaningful noise,
# while it CAN catch spurious long-lag structure in a generated ensemble that
# truncation would hide. Keeping the full range costs nothing and removes a
# cross-section inconsistency.


def split_indices(n_traj, b_fit=B_FIT, n_val=N_VAL):
    """The data-split protocol, by index, four disjoint segments.

        [0, B)              fitting block       -- training objective, mu, scale, s_w
        [B, 10B)            9 held-out blocks   -- the reference-ensemble error
        [10B, 10B+n_val)    validation          -- every decision
        [10B+n_val, N)      evaluation          -- every reported number

    Nine held-out blocks, not ten: the fitting block is excluded so that every
    quantity in the denominator is disjoint from the training objective. This
    keeps hold_start = 10B identical to the FPU code already on disk, so
    Section 5.1 needs no re-splitting -- only its limits.py changes, to take the
    median over the nine blocks instead of ten.
    """
    hold_start = 10 * b_fit
    if n_traj < hold_start + n_val + b_fit:
        raise ValueError(
            f"archive of {n_traj} trajectories is too small for the protocol; "
            f"need at least {hold_start + n_val + b_fit}")
    return {
        "fit": (0, b_fit),
        "reference": (b_fit, hold_start),
        "validation": (hold_start, hold_start + n_val),
        "evaluation": (hold_start + n_val, n_traj),
    }


# Directions in the STANDARDISED frame, three orthogonal pairs.
PROJECTIONS = (
    (1.0, 1.0), (1.0, -1.0),
    (1.0, 2.0), (2.0, -1.0),
    (2.0, 1.0), (1.0, -2.0),
)

# Directions that never enter any training objective, and never enter model
# selection either. They are reported, not optimised against and not used to
# choose anything -- otherwise "held out" would be a word rather than a fact.
HOLDOUT_PROJECTIONS = (
    (1.0, 3.0), (3.0, -1.0),
    (1.0, -3.0), (3.0, 1.0),
)

Z_HALFWIDTH = 5.0
Z_N = 513

# Output grid [-5, 5] with 513 points: dz = 0.019531, so h/dz = 5.35 at this
# section's h = (200*400)^(-1/5) = 0.10453. The criterion is h/dz >= 4, so that
# the kernel is resolved by at least four grid cells and the grid-weighted norm
# is not itself a discretisation of the bandwidth.
#
# 401 points (dz = 0.025, h/dz = 4.18) already satisfies the criterion HERE. 513
# is chosen because the grid is shared with Section 5.2, whose smallest
# bandwidth is h = 0.0793 on the same [-5, 5] range: that needs dz <= 0.0198,
# i.e. N_z >= 506. One grid for both sections -- audit B8, which is on LLV's
# critical path, not Section 4 housekeeping.
Z_FINE_N = 4001
KERNEL = "gaussian, isotropic, h = (L*B)^(-1/(d+4)); analytic reference " \
         "convolved with the same kernel before comparison"


def ccf_mode_for(analytic_corr):
    """'absolute' when the population cross-correlation vanishes identically.

    At lambda = 0 the components are exactly independent, so the relative
    denominator is zero in the population. The decision is made HERE, from the
    analytic correlation, and stored -- never re-derived downstream from a
    filename or from whether an empirical CCF happens to look small. Note that
    the empirical fitting-block CCF is NOT overwritten with zeros: it carries
    sampling noise and that is what the network is asked to match. Only the
    normalisation changes.
    """
    return "absolute" if abs(float(analytic_corr)) < 1e-8 else "relative"


def bandwidth_1d(l_seq, b_fit=B_FIT):
    """h = (L*B)^(-1/5), fixed by the FITTING BLOCK size and shared by the
    reference and the generated ensembles. A mismatch biases the learned
    variance. This differs from Section 5.1's value because L differs (200 here
    against 400 there) -- the bandwidth follows the sample count, so that is
    correct and should not be "fixed" back."""
    return float((l_seq * b_fit) ** (-1.0 / 5.0))


def bandwidth_2d(l_seq, b_fit=B_FIT):
    """h = n^(-1/(d+4)) with d = 2, for the diagnostic bivariate KDE."""
    return float((l_seq * b_fit) ** (-1.0 / 6.0))


def z_fine_grid(h):
    """Deposition grid, wide enough that the kernel tail off the output grid is
    accounted for. Stored in the reference so evaluate.py reconstructs it
    identically rather than re-deriving it."""
    return np.linspace(-Z_HALFWIDTH - 6 * h, Z_HALFWIDTH + 6 * h, Z_FINE_N)


# --------------------------------------------------------------------------
# SciPy-free helpers
# --------------------------------------------------------------------------

def _norm_ppf(u):
    """Inverse standard normal CDF (Acklam's rational approximation).

    Relative error below 1.2e-9 over the whole range. Replaces
    scipy.stats.norm.ppf so the pipeline has no SciPy dependency;
    --regress-scipy checks it against SciPy where available, including the tails
    where a home-made quantile function is most likely to drift.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    lo, hi = u < 0.02425, u > 1 - 0.02425
    mid = ~(lo | hi)

    q = np.sqrt(-2 * np.log(np.where(lo, u, 0.5)))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])[lo] \
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)[lo]
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - u, 0.5)))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])[hi] \
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)[hi]
    q = np.where(mid, u, 0.5) - 0.5
    r = q * q
    out[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q)[mid] \
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)[mid]
    return out


def _bisect(f, lo, hi, tol=1e-12, maxiter=300):
    """Bisection on a monotone function. Replaces scipy.optimize.brentq."""
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError("root not bracketed")
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Standardisation constants
# --------------------------------------------------------------------------

def standardisation(traj, split):
    """mu, scale and the projection scales -- all from the fitting block, all
    stored, all applied unchanged to every later ensemble.

    No squared-process mean is stored: the correlation estimators centre each
    measured ensemble with its own empirical mean, which is the manuscript's
    definition and what the training loss does.
    """
    lo, hi = split["fit"]
    fit = np.asarray(traj[:, lo:hi, :], dtype=np.float64)
    mu = fit.mean(axis=(0, 1))
    scale = fit.std(axis=(0, 1))
    qt = (fit - mu) / scale

    def s_of(a, b):
        return float((a * qt[..., 0] + b * qt[..., 1]).std())

    return {
        "mu": mu,
        "scale": scale,
        "proj_scales": np.array([s_of(a, b) for a, b in PROJECTIONS]),
        "holdout_scales": np.array([s_of(a, b) for a, b in HOLDOUT_PROJECTIONS]),
    }


# --------------------------------------------------------------------------
# Analytic densities in the standardised frame
# --------------------------------------------------------------------------

def _deposit(grid, dq, p, a, b, mu, scale, s, z_fine):
    """Density of (a*q1t + b*q2t)/s with q_t = (q - mu)/scale, by cloud-in-cell
    deposition of the quadrature mass. The O(bin^2) deposition error is washed
    out by the kernel convolution that follows."""
    Q1, Q2 = np.meshgrid(grid, grid, indexing="ij")
    z = ((a * (Q1 - mu[0]) / scale[0] + b * (Q2 - mu[1]) / scale[1]) / s).ravel()
    w = (p * dq * dq).ravel()

    dz = z_fine[1] - z_fine[0]
    pos = (z - z_fine[0]) / dz
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 < len(z_fine) - 1)
    i0, frac, w = i0[ok], frac[ok], w[ok]

    f = (np.bincount(i0, weights=w * (1.0 - frac), minlength=len(z_fine))
         + np.bincount(i0 + 1, weights=w * frac, minlength=len(z_fine)))
    return f[:len(z_fine)] / dz


def _convolve(z_fine, f_fine, h, z_out):
    """Convolve a density on a fine uniform grid with N(0, h^2)."""
    dz = z_fine[1] - z_fine[0]
    ker = np.exp(-0.5 * ((z_out[:, None] - z_fine[None, :]) / h) ** 2)
    return (ker @ (f_fine * dz)) / (np.sqrt(2.0 * np.pi) * h)


def analytic_densities(grid, dq, p, const, h, h2):
    """Every PDF-family reference the evaluation needs, in standardised units,
    each already convolved with the kernel the KDE losses use."""
    mu, scale = const["mu"], const["scale"]
    z_out = np.linspace(-Z_HALFWIDTH, Z_HALFWIDTH, Z_N)
    z_fine = z_fine_grid(h)

    def dens(a, b, s):
        return _convolve(z_fine, _deposit(grid, dq, p, a, b, mu, scale, s, z_fine),
                         h, z_out)

    marg = [dens(1.0, 0.0, 1.0), dens(0.0, 1.0, 1.0)]
    proj = [dens(a, b, s) for (a, b), s in zip(PROJECTIONS, const["proj_scales"])]
    hold = [dens(a, b, s) for (a, b), s in zip(HOLDOUT_PROJECTIONS, const["holdout_scales"])]

    g1 = (grid - mu[0]) / scale[0]
    g2 = (grid - mu[1]) / scale[1]
    k1 = np.exp(-0.5 * ((z_out[:, None] - g1[None, :]) / h2) ** 2)
    k2 = np.exp(-0.5 * ((z_out[:, None] - g2[None, :]) / h2) ** 2)
    joint = k1 @ (p * dq * dq) @ k2.T
    joint /= joint.sum() * (z_out[1] - z_out[0]) ** 2

    return {"z_grid": z_out,
            "marginal": np.stack(marg, axis=-1),
            "projected": np.stack(proj, axis=-1),
            "projected_holdout": np.stack(hold, axis=-1),
            "joint": joint}


# --------------------------------------------------------------------------
# Gaussian-copula null
# --------------------------------------------------------------------------

def gaussian_copula_null(grid, dq, p, target_corr):
    """kappa_1122/(var1 var2) of the distribution with the SAME two marginals
    and the SAME Pearson correlation as the target, but Gaussian dependence.

    This is the null the target has to beat for the projected-PDF loss to be
    testing anything beyond marginals plus correlation. A linear-mixture null
    would not preserve both marginals, so it is not the honest comparison.
    """
    f1 = p.sum(axis=1) * dq
    f2 = p.sum(axis=0) * dq

    def to_z(f):
        F = np.clip(np.cumsum(f) * dq - 0.5 * f * dq, 1e-12, 1 - 1e-12)
        return _norm_ppf(F)

    Z1, Z2 = np.meshgrid(to_z(f1), to_z(f2), indexing="ij")
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    Q1, Q2 = np.meshgrid(grid, grid, indexing="ij")

    def build(r):
        c = np.exp(-(r * r * (Z1 ** 2 + Z2 ** 2) - 2 * r * Z1 * Z2)
                   / (2 * (1 - r * r))) / np.sqrt(1 - r * r)
        q = c * F1 * F2
        return q / (q.sum() * dq * dq)

    def corr_of(q):
        w = q * dq * dq
        m = lambda A: float((A * w).sum())
        m1, m2 = m(Q1), m(Q2)
        return ((m(Q1 * Q2) - m1 * m2)
                / np.sqrt((m(Q1 ** 2) - m1 ** 2) * (m(Q2 ** 2) - m2 ** 2)))

    if abs(target_corr) < 1e-8:
        return 0.0, 0.0
    r = _bisect(lambda r: corr_of(build(r)) - target_corr, 1e-8, 1 - 1e-8)
    return gibbs_moments(grid, dq, build(r))["kappa_1122_over_var1var2"], float(r)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    hh = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            hh.update(blk)
    return hh.hexdigest()


def constants_hash(const, h, h2, k_lags, b_fit, l_seq, ccf_mode):
    """Hash over everything that must agree between training and evaluation.

    Covers the standardisation constants, the directions, both bandwidths, the
    grids, K, B, the sequence length, the kernel convention, the CCF mode and
    the projection aggregation rule. A generated ensemble stores this value;
    evaluate.py refuses one whose hash differs from the reference it is scored
    against, because a mismatch in any of these is invisible in the numbers and
    fatal to the comparison.

    l_seq and ccf_mode are in here deliberately: L fixes the KDE sample count
    (L*B) and hence the bandwidth, and ccf_mode changes the CCF term's
    definition, so a run trained at one setting must not be scored at another.
    """
    hh = hashlib.sha256()
    for arr in (const["mu"], const["scale"], const["proj_scales"],
                const["holdout_scales"], np.array(PROJECTIONS),
                np.array(HOLDOUT_PROJECTIONS)):
        hh.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    hh.update(np.array([h, h2, k_lags, b_fit, l_seq, Z_HALFWIDTH, Z_N, Z_FINE_N,
                        ESTIMATOR_VERSION], dtype=np.float64).tobytes())
    hh.update(KERNEL.encode())
    hh.update(str(ccf_mode).encode())
    hh.update(PROJ_AGGREGATION.encode())
    return hh.hexdigest()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

_PHYSICS_KEYS = (("alpha", MT.ALPHA), ("beta", MT.BETA), ("dt", MT.DT),
                 ("dt_save", MT.DT_SAVE))
_PHYSICS_SEQ = (("kappa", list(MT.KAPPA)), ("gamma", list(MT.GAMMA)))


def check_metadata(meta):
    """Abort if the archive was produced by a different physics than the one
    this module would use for the analytic density.

    Without this the failure is silent and severe: change a potential
    parameter, re-run stage 2 on an old archive, and the reference densities
    become a "truth" for a system the trajectories never obeyed.
    """
    bad = [f"{k}: archive {meta.get(k)!r} vs source {v!r}"
           for k, v in _PHYSICS_KEYS if meta.get(k) != v]
    bad += [f"{k}: archive {meta.get(k)!r} vs source {v!r}"
            for k, v in _PHYSICS_SEQ if list(meta.get(k, [])) != v]
    if meta.get("integrator") != "BAOAB":
        bad.append(f"integrator: archive {meta.get('integrator')!r} vs 'BAOAB'")
    if bad:
        raise SystemExit(
            "make_target.py constants do not match the archive metadata:\n  "
            + "\n  ".join(bad)
            + "\nRegenerate the archive, or check out the make_target.py that "
              "produced it. Refusing to build a reference for physics the "
              "trajectories never obeyed.")


def build_reference(traj_path, outdir, verbose=True):
    tag = re.search(r"traj_(lam[\w]+)\.npy$", os.path.basename(traj_path)).group(1)
    with open(os.path.join(os.path.dirname(traj_path), f"meta_{tag}.json")) as fh:
        meta = json.load(fh)
    check_metadata(meta)
    lam = meta["lambda"]

    traj = np.load(traj_path, mmap_mode="r")
    l_seq, n_traj = traj.shape[0], traj.shape[1]
    if meta.get("L_seq") != l_seq:
        raise SystemExit(f"metadata L_seq={meta.get('L_seq')} but array has {l_seq}")
    split = split_indices(n_traj)

    const = standardisation(traj, split)
    grid, dq, p = gibbs_grid(lam)
    moments = gibbs_moments(grid, dq, p)
    h, h2 = bandwidth_1d(l_seq), bandwidth_2d(l_seq)
    dens = analytic_densities(grid, dq, p, const, h, h2)
    k_null, r_gauss = gaussian_copula_null(grid, dq, p, moments["corr"])

    k_t = moments["kappa_1122_over_var1var2"]
    unexplained = abs(k_t - k_null)

    ccf_mode = ccf_mode_for(moments["corr"])
    c_hash = constants_hash(const, h, h2, K_LAGS, B_FIT, l_seq, ccf_mode)
    t_hash = sha256_file(traj_path)

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, f"reference_{tag}.npz"),
             schema_version=SCHEMA_VERSION,
             mu=const["mu"], scale=const["scale"],
             proj_scales=const["proj_scales"],
             holdout_scales=const["holdout_scales"],
             projections=np.array(PROJECTIONS),
             holdout_projections=np.array(HOLDOUT_PROJECTIONS),
             bandwidth_1d=h, bandwidth_2d=h2,
             z_halfwidth=Z_HALFWIDTH, z_n=Z_N, z_fine_n=Z_FINE_N,
             kernel=KERNEL,
             k_lags=K_LAGS, b_fit=B_FIT, l_seq=l_seq,
             n_ref_blocks=N_REF_BLOCKS,
             ccf_mode=ccf_mode, estimator_version=ESTIMATOR_VERSION,
             proj_aggregation=PROJ_AGGREGATION,
             split_fit=np.array(split["fit"]),
             split_reference=np.array(split["reference"]),
             split_validation=np.array(split["validation"]),
             split_evaluation=np.array(split["evaluation"]),
             # Analytic scalars. analytic_corr is what tells evaluate.py that the
             # target cross-correlation vanishes identically, from the physics
             # rather than from a filename.
             analytic_corr=moments["corr"], analytic_cov=moments["cov"],
             kappa_1122_over_var1var2=k_t,
             kappa_1122_over_var1var2_null=k_null,
             target_sha256=t_hash, constants_sha256=c_hash,
             **dens)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "lambda": lam, "tag": tag, "n_traj": int(n_traj), "L_seq": int(l_seq),
        "split": {k: list(v) for k, v in split.items()},
        "n_held_out_blocks": N_REF_BLOCKS,
        "mu": const["mu"].tolist(), "scale": const["scale"].tolist(),
        "proj_scales": const["proj_scales"].tolist(),
        "holdout_scales": const["holdout_scales"].tolist(),
        "bandwidth_1d": h, "bandwidth_2d": h2, "k_lags": K_LAGS,
        "kernel": KERNEL, "ccf_mode": ccf_mode,
        "estimator_version": ESTIMATOR_VERSION,
        "proj_aggregation": PROJ_AGGREGATION,
        "analytic_moments": moments,
        "kappa_1122_over_var1var2": k_t,
        "kappa_1122_over_var1var2_gaussian_copula_null": k_null,
        "kappa_1122_unexplained_magnitude": unexplained,
        "gaussian_copula_r": r_gauss,
        "target_sha256": t_hash, "constants_sha256": c_hash,
    }
    with open(os.path.join(outdir, f"reference_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        deg = " [ccf_mode=absolute]" if ccf_mode == "absolute" else ""
        print(f"lam={lam:<5g} corr {moments['corr']:+.4f}{deg}   "
              f"k1122/v1v2 {k_t:+.4f}  null {k_null:+.4f}  "
              f"|unexplained| {unexplained:.4f}   "
              f"mu {np.round(const['mu'], 4)}  scale {np.round(const['scale'], 4)}")
    return summary


def regress_scipy(datadir):
    """Check the numpy quantile function and root find against SciPy.

    Not a dependency -- if SciPy is absent this is skipped. It exists because a
    home-made norm.ppf is most likely to drift in the tails, exactly where the
    copula construction evaluates it.
    """
    try:
        from scipy.stats import norm
        from scipy.optimize import brentq
    except ImportError:
        print("SciPy not installed; regression check skipped")
        return True

    u = np.concatenate([np.logspace(-14, -1.6, 400),
                        np.linspace(0.025, 0.975, 600),
                        1 - np.logspace(-14, -1.6, 400)])
    err = float(np.abs(_norm_ppf(u) - norm.ppf(u)).max())
    print(f"norm.ppf  max abs error over [1e-14, 1-1e-14]: {err:.3e}")
    ok = err < 1e-7

    for path in sorted(glob.glob(os.path.join(datadir, "reference_lam*.json"))):
        with open(path) as fh:
            s = json.load(fh)
        lam = s["lambda"]
        grid, dq, p = gibbs_grid(lam)
        m = gibbs_moments(grid, dq, p)
        if abs(m["corr"]) < 1e-8:
            print(f"lam={lam:<5g} corr 0, copula null trivially 0 -- skipped")
            continue
        f1, f2 = p.sum(axis=1) * dq, p.sum(axis=0) * dq

        def to_z(f, ppf):
            F = np.clip(np.cumsum(f) * dq - 0.5 * f * dq, 1e-12, 1 - 1e-12)
            return ppf(F)

        def make(ppf):
            Z1, Z2 = np.meshgrid(to_z(f1, ppf), to_z(f2, ppf), indexing="ij")
            F1, F2 = np.meshgrid(f1, f2, indexing="ij")
            Q1, Q2 = np.meshgrid(grid, grid, indexing="ij")

            def build(r):
                c = np.exp(-(r * r * (Z1 ** 2 + Z2 ** 2) - 2 * r * Z1 * Z2)
                           / (2 * (1 - r * r))) / np.sqrt(1 - r * r)
                q = c * F1 * F2
                return q / (q.sum() * dq * dq)

            def corr_of(q):
                w = q * dq * dq
                mm = lambda A: float((A * w).sum())
                a, b2 = mm(Q1), mm(Q2)
                return ((mm(Q1 * Q2) - a * b2)
                        / np.sqrt((mm(Q1 ** 2) - a ** 2) * (mm(Q2 ** 2) - b2 ** 2)))
            return build, corr_of

        b_np, c_np = make(_norm_ppf)
        b_sp, c_sp = make(norm.ppf)
        r_np = _bisect(lambda r: c_np(b_np(r)) - m["corr"], 1e-8, 1 - 1e-8)
        r_sp = brentq(lambda r: c_sp(b_sp(r)) - m["corr"], 1e-8, 1 - 1e-8, xtol=1e-12)
        k_np = gibbs_moments(grid, dq, b_np(r_np))["kappa_1122_over_var1var2"]
        k_sp = gibbs_moments(grid, dq, b_sp(r_sp))["kappa_1122_over_var1var2"]
        print(f"lam={lam:<5g} copula r  numpy {r_np:.12f}  scipy {r_sp:.12f}  "
              f"diff {abs(r_np - r_sp):.2e}   k1122 diff {abs(k_np - k_sp):.2e}")
        ok &= abs(r_np - r_sp) < 1e-8 and abs(k_np - k_sp) < 1e-8

    print("regression PASSED" if ok else "regression FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default=None, help="defaults to --datadir")
    ap.add_argument("--regress-scipy", action="store_true",
                    help="compare the numpy ppf and root find against SciPy")
    args = ap.parse_args()
    outdir = args.outdir or args.datadir

    if args.regress_scipy:
        raise SystemExit(0 if regress_scipy(args.datadir) else 1)

    paths = sorted(glob.glob(os.path.join(args.datadir, "traj_lam*.npy")))
    if not paths:
        raise SystemExit(f"no traj_lam*.npy in {args.datadir}; run make_target.py first")

    print("kappa_1122/(var1 var2) is the normalised fourth-order joint cumulant; "
          "|unexplained| is the\nmagnitude the matched Gaussian-copula null cannot "
          "account for. Run the detailed\nprojected-PDF ablation where it is largest.\n")
    summaries = [build_reference(p, outdir) for p in paths]

    best = max(summaries, key=lambda s: s["kappa_1122_unexplained_magnitude"])
    print(f"\nlargest unexplained magnitude at lambda = {best['lambda']:g} "
          f"({best['kappa_1122_unexplained_magnitude']:.4f})")


if __name__ == "__main__":
    main()
