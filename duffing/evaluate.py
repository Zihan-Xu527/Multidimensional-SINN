#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3 of 3 -- the statistics, the reference-ensemble error (the "floor") and
the reported errors.

Every per-lambda constant -- mu, scale, the projection scales, the directions,
the bandwidths, K and the four split boundaries -- is read from
reference_lam<X>.npz, never from this file's source. Provenance is checked: the
reference stores the sha256 of the target archive it was built from and a hash
over its own constants, and a generated ensemble carrying a different constants
hash is refused.

Five statistic families, the same five that serve as training losses:

    E_ACF    autocorrelation of each standardised component
    E_CCF    the two directional cross-correlations
    E_ACF2   autocorrelation of the squared standardised components
    E_PDF    component marginal densities
    E_PROJ   densities of the six projected variables

TWO SEPARATE CENTERINGS, not to be confused:

1. The coordinate change q_tilde = (q - mu_fit)/scale_fit. Fixed once in
   make_reference.py from the fitting block, applied unchanged everywhere.
2. The centering inside the correlation estimators: each measured ensemble has
   its OWN empirical mean removed, which is the manuscript's definition of
   C_ij(k) and what the training loss computes. The squared process likewise
   has its own mean removed.

Using a fixed constant for step 2 instead of the ensemble's own mean changes
ACF by up to 0.12 and CCF by up to 0.23 on mean-shifted input, so the two must
match between training and evaluation.

Error definitions. Relative L2 with RMS norms,
||stat(X) - stat(R)|| / ||stat(R)||, family by family.

* PDF and PROJ are measured against the ANALYTIC density from
  make_reference.py, kernel-convolved to match the estimator. There is nothing
  to hold out from an exact density, so they are reported ONCE.

* ACF, CCF and ACF2 are measured against a held-out ensemble -- validation for
  every decision, evaluation for every reported number.

* CCF uses the CROSS-ONLY ordered-pair norm, P_cross = {(1,2),(2,1)}:

      e_CCF = RMS_2K( dC_12, dC_21 ) / RMS_2K( C_12^R, C_21^R ),

  i.e. the two directional curves concatenated into one 2K vector, numerator
  and denominator alike. The ACF is NOT in the denominator: mixing it in would
  let cross-correlation error be judged against autocorrelation strength, and
  would change the definition already used in Sections 5.1 and 5.2.

* When the target cross-correlation vanishes identically -- lambda = 0, where
  the two components are exactly independent -- the relative form is 0/0 in the
  limit and, estimated from a finite reference, divides by pure sampling noise.
  There the ABSOLUTE discrepancy RMS_2K(dC_12, dC_21) is reported instead,
  CCF is excluded from the relative-error success maximum, and max|C_12| is
  reported as the spurious-coupling scale. The degenerate case is detected from
  the analytic correlation stored in the reference, not from a filename.

Decisions happen on validation only. The evaluation pass reports errors and
ratios; it never produces a success verdict.

Diagnostics that are REPORTED BUT NEVER USED FOR SELECTION: the held-out
projection error, the bivariate joint error and kappa_1122/(var1 var2).

Usage
-----
    python evaluate.py --datadir data
    python evaluate.py --datadir data --lam 4 --generated runs/gen_s0.npz
    python evaluate.py --datadir data --selftest
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from make_reference import _convolve, constants_hash

# No default success factor. The 3.0 that used to live here was inherited from
# Section 5.1 as a convention and answered no stated question; calibrate.py
# replaces it with the 95th percentile of the worst-family ratio a genuine
# ensemble of the same size reaches by sampling noise alone. tau depends on
# n_gen and on which floor it is measured against, so it must be passed in.
SUCCESS_FACTOR = None
FAMILIES = ("ACF", "CCF", "ACF2", "PDF", "PROJ")
STATIC = ("PDF", "PROJ")          # measured against the analytic density
DYNAMIC = ("ACF", "CCF", "ACF2")  # measured against a held-out ensemble

GENERATED_SCHEMA = 1


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def _acf_raw(x, n_lags):
    """Unnormalised autocovariance of x, shape (L, N), averaged over N, with the
    ensemble's OWN empirical mean removed.

    Biased estimator: normalised at zero lag only, never divided by (L - k), so
    the curves taper toward zero at large lag. Reference and generated
    ensembles are treated identically, so the comparison is fair -- but these
    are not unbiased ACF estimates and the manuscript must say so in the
    methods, not only in a figure caption.
    """
    x = x - x.mean()
    n = x.shape[0]
    f = np.fft.rfft(x, 2 * n - 1, axis=0)
    ac = np.fft.irfft(f * np.conj(f), 2 * n - 1, axis=0)[:n].mean(axis=1)
    return ac[:n_lags]


def _ccf_raw(x, y, n_lags):
    """Unnormalised cross-covariance C_xy(k) = <x(t) y(t+k)>, each series
    centred by its own empirical mean.

    Note the direction. irfft(F(x) conj(F(y)))[k] is <x(t+k) y(t)> = C_yx(k),
    so the conjugation goes on x, not y, to match the manuscript's
    C_ij(k) = <y_i(t) y_j(t+k)>. Getting this backwards leaves the aggregate
    norm unchanged but swaps the labels of the two directional curves.
    """
    x = x - x.mean()
    y = y - y.mean()
    n = x.shape[0]
    f = np.fft.rfft(x, 2 * n - 1, axis=0)
    g = np.fft.rfft(y, 2 * n - 1, axis=0)
    cc = np.fft.irfft(np.conj(f) * g, 2 * n - 1, axis=0)[:n].mean(axis=1)
    return cc[:n_lags]


def correlations(qt, n_lags):
    """ACF of both standardised components, both directional CCFs (C_12, C_21),
    and the ACF of the squared components. qt has shape (L, N, 2)."""
    x, y = qt[..., 0], qt[..., 1]
    ax, ay = _acf_raw(x, n_lags), _acf_raw(y, n_lags)
    norm = np.sqrt(ax[0] * ay[0])
    acf = np.stack([ax / ax[0], ay / ay[0]], axis=-1)
    ccf = np.stack([_ccf_raw(x, y, n_lags) / norm,
                    _ccf_raw(y, x, n_lags) / norm], axis=-1)
    a2x, a2y = _acf_raw(x ** 2, n_lags), _acf_raw(y ** 2, n_lags)
    acf2 = np.stack([a2x / a2x[0], a2y / a2y[0]], axis=-1)
    return acf, ccf, acf2


def kde(samples, h, z_out, z_fine):
    """1-D KDE by cloud-in-cell deposition followed by Gaussian convolution --
    the same two-step treatment the analytic reference receives, so the
    deposition error is common to both sides."""
    s = np.asarray(samples).ravel()
    dz = z_fine[1] - z_fine[0]
    pos = (s - z_fine[0]) / dz
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 < len(z_fine) - 1)
    i0, frac = i0[ok], frac[ok]
    f = (np.bincount(i0, weights=1.0 - frac, minlength=len(z_fine))
         + np.bincount(i0 + 1, weights=frac, minlength=len(z_fine)))[:len(z_fine)]
    return _convolve(z_fine, f / (s.size * dz), h, z_out)


def kde2d(qt, h2, z_out):
    """Bivariate KDE of the standardised components, separable convolution."""
    dz = z_out[1] - z_out[0]
    edges = np.linspace(z_out[0] - 0.5 * dz, z_out[-1] + 0.5 * dz, len(z_out) + 1)
    hist, _, _ = np.histogram2d(qt[..., 0].ravel(), qt[..., 1].ravel(),
                                bins=[edges, edges])
    hist = hist / (hist.sum() * dz * dz)
    ker = (np.exp(-0.5 * ((z_out[:, None] - z_out[None, :]) / h2) ** 2)
           / (np.sqrt(2.0 * np.pi) * h2))
    out = ker @ hist @ ker.T * dz * dz
    return out / (out.sum() * dz * dz)


def kappa_1122_norm(qt):
    """Centred fourth-order joint cumulant divided by var1*var2.

    The means are removed here rather than assumed zero: the target's mean
    vanishes by symmetry, but a generated ensemble's need not, and the
    uncentred formula would then be simply wrong.
    """
    x = qt[..., 0].astype(np.float64).ravel()
    y = qt[..., 1].astype(np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    v1, v2, c = (x ** 2).mean(), (y ** 2).mean(), (x * y).mean()
    return float(((x ** 2 * y ** 2).mean() - v1 * v2 - 2 * c ** 2) / (v1 * v2))


# --------------------------------------------------------------------------
# Statistics bundle and errors
# --------------------------------------------------------------------------

def standardise(arr, ref, space):
    """Apply the fixed affine map from the reference file, or not.

    space='standardized' the array is already in q_tilde units and is used as
                         is. This is the DEFAULT, because the training code
                         standardises the target before it reaches the network
                         and returns the network output directly, so a saved
                         SINN ensemble is in q_tilde units.
    space='raw'          the array is in physical coordinates (what
                         make_target.py writes) and (q - mu)/scale is applied.

    Applying the map twice, or not at all, is the silent failure this
    distinction exists to prevent.
    """
    t = np.asarray(arr, dtype=np.float64)
    if space == "standardized":
        return t
    if space == "raw":
        return (t - ref["mu"]) / ref["scale"]
    raise ValueError(f"unknown coordinate space {space!r}")


def z_grids(ref):
    h = float(ref["bandwidth_1d"])
    zh, zfn = float(ref["z_halfwidth"]), int(ref["z_fine_n"])
    return ref["z_grid"], np.linspace(-zh - 6 * h, zh + 6 * h, zfn)


def stats(arr, ref, space):
    """All five families plus the report-only diagnostics."""
    z_out, z_fine = z_grids(ref)
    h, h2 = float(ref["bandwidth_1d"]), float(ref["bandwidth_2d"])
    n_lags = int(ref["k_lags"])

    qt = standardise(arr, ref, space)
    if not np.all(np.isfinite(qt)):
        raise ValueError(f"{int((~np.isfinite(qt)).sum())} non-finite values in "
                         f"the ensemble; refusing to score it")
    acf, ccf, acf2 = correlations(qt, n_lags)

    pdf = np.stack([kde(qt[..., i], h, z_out, z_fine) for i in (0, 1)], axis=-1)
    proj = np.stack([kde((a * qt[..., 0] + b * qt[..., 1]) / s, h, z_out, z_fine)
                     for (a, b), s in zip(ref["projections"], ref["proj_scales"])],
                    axis=-1)
    hold = np.stack([kde((a * qt[..., 0] + b * qt[..., 1]) / s, h, z_out, z_fine)
                     for (a, b), s in zip(ref["holdout_projections"],
                                          ref["holdout_scales"])], axis=-1)

    return {"ACF": acf, "CCF": ccf, "ACF2": acf2, "PDF": pdf, "PROJ": proj,
            "PROJ_HOLDOUT": hold, "JOINT": kde2d(qt, h2, z_out),
            "kappa_1122_over_var1var2": kappa_1122_norm(qt)}


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def _rel(a, b):
    """||a - b|| / ||b||, RMS norms."""
    return _rms(a - b) / _rms(b)


def errors(sg, sr, ccf_degenerate):
    """The five reported errors plus the report-only diagnostics.

    When ccf_degenerate, the CCF entry is the ABSOLUTE discrepancy and is
    flagged so classify() leaves it out of the relative-error maximum.
    """
    d_ccf = sg["CCF"] - sr["CCF"]
    out = {
        "ACF": _rel(sg["ACF"], sr["ACF"]),
        "ACF2": _rel(sg["ACF2"], sr["ACF2"]),
        "PDF": _rel(sg["PDF"], sr["PDF"]),
        "PROJ": _rel(sg["PROJ"], sr["PROJ"]),
        "CCF_abs": _rms(d_ccf),
        "PROJ_HOLDOUT": _rel(sg["PROJ_HOLDOUT"], sr["PROJ_HOLDOUT"]),
        "JOINT": _rel(sg["JOINT"], sr["JOINT"]),
        "max_abs_ccf": float(np.abs(sg["CCF"]).max()),
        "kappa_1122_over_var1var2": sg["kappa_1122_over_var1var2"],
        "ccf_degenerate": bool(ccf_degenerate),
    }
    # Cross-only ordered-pair relative norm: the two directional curves
    # concatenated into one 2K vector, numerator and denominator alike.
    out["CCF"] = float("nan") if ccf_degenerate else _rms(d_ccf) / _rms(sr["CCF"])
    return out


def reference_stats(traj, ref, segment, space="raw"):
    """Correlations from the named held-out segment; densities from the
    analytic reference."""
    lo, hi = segment
    s = stats(traj[:, lo:hi, :], ref, space)
    s["PDF"] = ref["marginal"]
    s["PROJ"] = ref["projected"]
    s["PROJ_HOLDOUT"] = ref["projected_holdout"]
    s["JOINT"] = ref["joint"]
    return s


def ccf_is_degenerate(ref):
    """True when the population cross-correlation vanishes identically.

    Read from the ccf_mode that make_reference.py decided and stored, not
    re-derived here: the loss module reads the same field, so training and
    evaluation cannot disagree about which branch a lambda is on. The field is
    part of the constants hash.
    """
    return str(ref["ccf_mode"]) == "absolute"


def split_from_reference(ref):
    return {k: (int(ref[f"split_{k}"][0]), int(ref[f"split_{k}"][1]))
            for k in ("fit", "reference", "validation", "evaluation")}


def floor(traj, ref, split, against):
    """Reference-ensemble error: the median over the nine held-out blocks.

    Answers "how far from the truth is a real sample of exactly the training
    budget", which is the scale a model error has to be read against. It is NOT
    a bound -- a model can land below it, because the network cannot represent
    the sampling wiggles of a B-trajectory estimate and smooths them, and the
    smoothed curve can sit closer to the truth than the block does. The
    manuscript must not call it an information limit.
    """
    deg = ccf_is_degenerate(ref)
    sr = reference_stats(traj, ref, split[against])
    lo, _ = split["reference"]
    b, nb = int(ref["b_fit"]), int(ref["n_ref_blocks"])
    rows = [errors(stats(traj[:, lo + i * b: lo + (i + 1) * b, :], ref, "raw"),
                   sr, deg) for i in range(nb)]
    med = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        med[k] = (bool(vals[0]) if k == "ccf_degenerate"
                  else float(np.median(vals)))
    return med, rows, sr


def classify(model_err, floor_err, factor=None):
    """Success when every applicable family is within `factor` times its own
    reference-ensemble error.

    CCF drops out of the maximum when the target cross-correlation vanishes
    identically; the absolute discrepancy is reported there instead, and
    CCF_abs / floor CCF_abs is available as a diagnostic ratio but is not part
    of the verdict.

    `factor` is tau, from calibrate.py: the 95th percentile of the worst-family
    ratio a genuine ensemble of the same size reaches by sampling noise alone.
    It is NOT a convention and NOT optional -- it depends on n_gen and on which
    floor it is measured against, so passing it is the caller's job. Report the
    classification gap (worst success / best failure) whenever tau is stated;
    a clean gap is the evidence that tau is not doing the work.

    Any non-finite error or ratio is a failure, never a success by omission.
    """
    if factor is None:
        raise ValueError(
            "classify() needs an explicit tau. There is no defensible default; "
            "run calibrate.py at the n_gen you are scoring and pass its W95.")
    fams = [f for f in FAMILIES if not (f == "CCF" and model_err["ccf_degenerate"])]
    ratios, bad = {}, []
    for k in fams:
        num, den = model_err[k], floor_err[k]
        r = num / den if np.isfinite(num) and np.isfinite(den) and den > 0 else float("nan")
        ratios[k] = r
        if not np.isfinite(r):
            bad.append(k)
    out = {"families": fams, "ratios": ratios,
           "excluded": ["CCF"] if model_err["ccf_degenerate"] else []}
    if bad:
        out.update({"worst_ratio": float("nan"), "success": False,
                    "non_finite": bad})
        return out
    worst = max(ratios.values())
    out.update({"worst_ratio": worst, "success": bool(worst <= factor)})
    if model_err["ccf_degenerate"] and floor_err["CCF_abs"] > 0:
        out["ccf_abs_ratio_diagnostic"] = model_err["CCF_abs"] / floor_err["CCF_abs"]
    return out


# --------------------------------------------------------------------------
# Generated-ensemble contract
# --------------------------------------------------------------------------

def save_generated(path, arr, ref, warmup_discarded=True,
                   coordinate_space="standardized"):
    """Write a self-describing generated ensemble. One place defines the
    contract, so the training driver and the evaluator cannot drift apart."""
    np.savez(path,
             schema_version=GENERATED_SCHEMA,
             data=np.asarray(arr, dtype=np.float32),
             coordinate_space=coordinate_space,
             axis_order="time,trajectory,component",
             warmup_discarded=bool(warmup_discarded),
             constants_sha256=str(ref["constants_sha256"]),
             target_sha256=str(ref["target_sha256"]))


def load_generated(path, ref, cli_space):
    """Load a generated ensemble, honouring its metadata over the CLI.

    A self-describing .npz wins over --generated-space and a conflict is an
    error, never a silent override. A bare .npy has no metadata, so the CLI
    value is used -- defaulting to 'standardized', because the training code
    standardises the target before the network sees it and saves the network
    output directly.
    """
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        if "data" not in d:
            raise SystemExit(f"{path}: .npz has no 'data' array")
        arr = d["data"]
        space = str(d["coordinate_space"]) if "coordinate_space" in d else None
        if space is not None and space != cli_space:
            raise SystemExit(
                f"{path} declares coordinate_space={space!r} but "
                f"--generated-space={cli_space!r} was passed. Metadata wins; "
                f"drop the flag or fix the file rather than overriding.")
        space = space or cli_space
        if "axis_order" in d and str(d["axis_order"]) != "time,trajectory,component":
            raise SystemExit(f"{path}: axis_order {str(d['axis_order'])!r}, "
                             f"expected 'time,trajectory,component'")
        if "warmup_discarded" in d and not bool(d["warmup_discarded"]):
            raise SystemExit(f"{path}: warmup_discarded is False. Keep only the "
                             f"scored window before saving.")
        for key in ("constants_sha256", "target_sha256"):
            if key in d and str(d[key]) != str(ref[key]):
                raise SystemExit(
                    f"{path}: {key} {str(d[key])[:12]}... does not match the "
                    f"reference {str(ref[key])[:12]}.... The generated ensemble "
                    f"was produced against different constants or a different "
                    f"target archive; the comparison would be meaningless.")
        return arr, space
    return np.load(path, mmap_mode="r"), cli_space


def check_generated(gen, traj, ref, space):
    """Shape, finiteness and coordinate-space sanity, before anything is scored."""
    if gen.ndim != 3 or gen.shape[0] != traj.shape[0] or gen.shape[2] != 2:
        raise SystemExit(
            f"generated array has shape {gen.shape}; expected "
            f"({traj.shape[0]}, N, 2). If the network was fed a recurrent "
            f"warm-up, keep only the last {traj.shape[0]} steps.")
    g = np.asarray(gen, dtype=np.float64)
    if not np.all(np.isfinite(g)):
        raise SystemExit(f"generated array has {int((~np.isfinite(g)).sum())} "
                         f"non-finite values; refusing to score it")
    sd = g.std(axis=(0, 1))
    if space == "raw" and np.all(np.abs(sd - 1.0) < 0.15):
        print(f"  WARNING: space='raw' but the per-component standard "
              f"deviations are {np.round(sd, 3)}, close to one -- this run may "
              f"be dividing by scale a second time.")
    if space == "standardized" and np.any(np.abs(sd - 1.0) > 0.5):
        print(f"  WARNING: space='standardized' but the per-component standard "
              f"deviations are {np.round(sd, 3)}, far from one.")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_reference(datadir, tag, traj_path=None):
    d = np.load(os.path.join(datadir, f"reference_{tag}.npz"), allow_pickle=False)
    ref = {k: d[k] for k in d.files}
    ref["constants_sha256"] = str(ref["constants_sha256"])
    ref["target_sha256"] = str(ref["target_sha256"])
    # The stored constants hash must reproduce from the stored constants; and the
    # target must be the archive the reference was built from.
    got = constants_hash({"mu": ref["mu"], "scale": ref["scale"],
                          "proj_scales": ref["proj_scales"],
                          "holdout_scales": ref["holdout_scales"]},
                         float(ref["bandwidth_1d"]), float(ref["bandwidth_2d"]),
                         int(ref["k_lags"]), int(ref["b_fit"]),
                         int(ref["l_seq"]), str(ref["ccf_mode"]))
    if got != ref["constants_sha256"]:
        raise SystemExit(
            f"reference_{tag}.npz was written by a different make_reference.py "
            f"(constants hash {ref['constants_sha256'][:12]}... but this source "
            f"produces {got[:12]}...). Rebuild the reference.")
    if traj_path is not None:
        from make_reference import sha256_file
        if sha256_file(traj_path) != ref["target_sha256"]:
            raise SystemExit(
                f"{traj_path} does not match the archive reference_{tag}.npz was "
                f"built from. The target was overwritten; rebuild the reference.")
    return ref


def run(datadir, tag, generated=None, space="standardized", verify_target=True,
        tau=None):
    traj_path = os.path.join(datadir, f"traj_{tag}.npy")
    ref = load_reference(datadir, tag, traj_path if verify_target else None)
    traj = np.load(traj_path, mmap_mode="r")
    split = split_from_reference(ref)
    deg = ccf_is_degenerate(ref)

    report = {"tag": tag, "k_lags": int(ref["k_lags"]), "ccf_degenerate": deg,
              "split": {k: list(v) for k, v in split.items()},
              "constants_sha256": ref["constants_sha256"]}
    if deg:
        print(f"[{tag}] target cross-correlation vanishes identically: CCF is "
              f"reported as an absolute discrepancy and excluded from the "
              f"success maximum")

    printed_static = False
    for against in ("validation", "evaluation"):
        E, rows, _ = floor(traj, ref, split, against)
        report[f"floor_{against}"] = E
        report[f"floor_blocks_{against}"] = rows
        if not printed_static:
            print(f"[{tag}] floor  static   " +
                  "  ".join(f"{k}={E[k]:.4f}" for k in STATIC))
            printed_static = True
        dyn = [f"ACF={E['ACF']:.4f}",
               (f"CCF_abs={E['CCF_abs']:.4f}" if deg else f"CCF={E['CCF']:.4f}"),
               f"ACF2={E['ACF2']:.4f}"]
        print(f"[{tag}] floor  vs {against:<10} " + "  ".join(dyn))
        if against == "evaluation":
            print(f"[{tag}]   held-out blocks: k1122/v1v2 median "
                  f"{np.median([r['kappa_1122_over_var1var2'] for r in rows]):+.4f} "
                  f"(analytic {float(ref['kappa_1122_over_var1var2']):+.4f}, "
                  f"null {float(ref['kappa_1122_over_var1var2_null']):+.4f}), "
                  f"max|C_12| median "
                  f"{np.median([r['max_abs_ccf'] for r in rows]):.5f}")

    if generated is not None:
        gen, space = load_generated(generated, ref, space)
        check_generated(gen, traj, ref, space)
        sg = stats(gen, ref, space)

        # Static families are measured against the analytic density and are the
        # same for both passes, so they are reported once.
        e_val = errors(sg, reference_stats(traj, ref, split["validation"]), deg)
        e_eva = errors(sg, reference_stats(traj, ref, split["evaluation"]), deg)
        cls = classify(e_val, report["floor_validation"], tau)  # decisions: validation only
        report.update({"model_validation": e_val, "model_evaluation": e_eva,
                       "class_validation": cls, "generated_space": space})

        print(f"[{tag}] model  static   " +
              "  ".join(f"{k}={e_eva[k]:.4f}" for k in STATIC))
        for name, e in (("validation", e_val), ("evaluation", e_eva)):
            dyn = [f"ACF={e['ACF']:.4f}",
                   (f"CCF_abs={e['CCF_abs']:.4f}" if deg else f"CCF={e['CCF']:.4f}"),
                   f"ACF2={e['ACF2']:.4f}"]
            print(f"[{tag}] model  vs {name:<10} " + "  ".join(dyn))
        print(f"[{tag}]   verdict (validation only): " +
              "  ".join(f"{k}={cls['ratios'][k]:.2f}x" for k in cls["families"]) +
              (f"  [CCF excluded]" if cls["excluded"] else "") +
              f"  -> {'SUCCESS' if cls['success'] else 'FAIL'} "
              f"(worst {cls['worst_ratio']:.2f}x)")
        if "ccf_abs_ratio_diagnostic" in cls:
            print(f"[{tag}]   diagnostic only: CCF_abs / floor "
                  f"{cls['ccf_abs_ratio_diagnostic']:.2f}x")
        print(f"[{tag}]   report-only, never used for selection: "
              f"PROJ_HOLDOUT={e_eva['PROJ_HOLDOUT']:.4f}  "
              f"JOINT={e_eva['JOINT']:.4f}  "
              f"k1122/v1v2={e_eva['kappa_1122_over_var1var2']:+.4f}")
    return report


def selftest(datadir, tag, tau):
    """Three synthetic 'generated' ensembles the pipeline must separate.

    good: a genuine held-out block of the target. By construction it is a real
          sample of exactly the training budget, so every ratio must land near
          one and it must be classified a success. Anything else means the error
          machinery is broken, not that the sample is bad.
    bad:  the same block with q2 replaced by 0.9*q1 + 0.1*q2 -- marginals barely
          move, coupling destroyed. Projection and joint errors must blow up.
    nan:  the same block with one value set to NaN. Must be refused, and must
          never be reported as a success.
    """
    traj_path = os.path.join(datadir, f"traj_{tag}.npy")
    ref = load_reference(datadir, tag, traj_path)
    traj = np.load(traj_path, mmap_mode="r")
    split = split_from_reference(ref)
    deg = ccf_is_degenerate(ref)
    lo, _ = split["reference"]
    b = int(ref["b_fit"])
    good = np.asarray(traj[:, lo:lo + b, :], dtype=np.float64)
    bad = good.copy()
    bad[..., 1] = 0.9 * good[..., 0] + 0.1 * good[..., 1]
    nan = good.copy()
    nan[0, 0, 0] = np.nan

    E, _, _ = floor(traj, ref, split, "validation")
    sr = reference_stats(traj, ref, split["validation"])
    ok = True
    for name, arr, want in (("good", good, True), ("bad", bad, False)):
        c = classify(errors(stats(arr, ref, "raw"), sr, deg), E, tau)
        got = c["success"]
        ok &= (got == want)
        print(f"  {name:<5} " +
              "  ".join(f"{k}={c['ratios'][k]:.2f}x" for k in c["families"]) +
              (" [CCF excluded]" if c["excluded"] else "") +
              f"  -> {'SUCCESS' if got else 'FAIL'}"
              f"{'' if got == want else '   <-- UNEXPECTED'}")
    try:
        stats(nan, ref, "raw")
        print("  nan   scored without complaint   <-- UNEXPECTED")
        ok = False
    except ValueError:
        print("  nan   refused, as required")
    print("  selftest PASSED" if ok else "  selftest FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--lam", type=float, default=None, help="default: all")
    ap.add_argument("--generated", default=None,
                    help="a generated ensemble to score (.npz preferred, see "
                         "save_generated); requires --lam")
    ap.add_argument("--generated-space", choices=("standardized", "raw"),
                    default="standardized",
                    help="coordinate space of a bare .npy --generated; ignored "
                         "when the file declares its own (metadata wins)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tau", type=float, default=None,
                    help="success threshold from calibrate.py, at the n_gen "
                         "being scored and against the floor being used. "
                         "Required whenever a verdict is produced; there is no "
                         "default because tau does not transfer across n_gen.")
    ap.add_argument("--no-verify-target", action="store_true",
                    help="skip the target sha256 check (slow on large archives)")
    ap.add_argument("--out", default=None, help="write the report as JSON")
    args = ap.parse_args()

    if args.lam is not None:
        tags = [f"lam{args.lam:g}".replace(".", "p")]
    else:
        tags = [re.search(r"traj_(lam[\w]+)\.npy$", p).group(1)
                for p in sorted(glob.glob(os.path.join(args.datadir, "traj_lam*.npy")))]
        if args.generated:
            raise SystemExit("--generated requires --lam")

    if args.selftest:
        if args.tau is None:
            raise SystemExit(
                "--selftest needs --tau: the good/bad separation is a verdict, "
                "and a verdict needs the calibrated threshold. Use the W95 that "
                "calibrate.py reports at n_gen = B (the block size the selftest "
                "scores).")
        allok = True
        for t in tags:
            print(f"[{t}] selftest  (tau = {args.tau:g})")
            allok &= selftest(args.datadir, t, args.tau)
        raise SystemExit(0 if allok else 1)

    if args.generated is not None and args.tau is None:
        raise SystemExit("scoring a generated ensemble produces a verdict; "
                         "pass --tau from calibrate.py")
    reports = [run(args.datadir, t, args.generated, args.generated_space,
                   not args.no_verify_target, args.tau) for t in tags]

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(reports, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
