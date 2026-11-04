#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2b of 3 -- precompute the TRAINING target statistics from the fitting
block, once, so that hundreds of sweep jobs read a cache instead of each
rebuilding it.

What this is and is not. make_reference.py produces the ANALYTIC densities: the
truth, used for evaluation. This file produces the EMPIRICAL statistics of the
fitting block [0, B): the quantities the training objective actually compares
against. The distinction is the protocol -- SINN must learn from B trajectories
of data, not from a density nobody would have in a real application. Both exist
and they are not interchangeable.

Five target statistic families, on the standardised fitting block:

    ACF      autocorrelation of each component, K lags
    CCF      both directional cross-correlations C_12, C_21, K lags
    ACF2     autocorrelation of the squared components
    PDF      the two component marginal KDEs
    PROJ     the six projected KDEs (a training run uses the first 2*n_f)

and, for each family, the denominator D_s = mean(S_fit^2) that turns a plain
MSE into the squared relative L2 discrepancy the loss minimises,

    L_s = mean( (S_theta - S_fit)^2 ) / D_s ,        e_s = sqrt(L_s).

The CCF family carries ccf_mode from the reference. When it is 'absolute'
(lambda = 0, where the population cross-correlation vanishes identically) the
loss drops the division by D_CCF. The cached target CCF is NOT zeroed: the
fitting block's empirical cross-correlation carries sampling noise and that is
exactly what the network is asked to reproduce. Only the normalisation changes.

The estimators are imported from evaluate.py, so there is exactly one numpy
definition of each statistic in the repository and the training target cannot
drift from the reported error by an implementation difference. sinn_loss.py
mirrors them in torch and `python sinn_loss.py --verify` checks the two agree.

The cache is bound to target_sha256 + constants_sha256 + the fitting range +
the estimator version. sinn_loss.py refuses a cache whose binding does not match
the reference it was handed.

Usage
-----
    python make_target_stats.py --datadir data
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

import evaluate as EV
from make_reference import ESTIMATOR_VERSION, PROJ_AGGREGATION, Z_FINE_N, Z_HALFWIDTH

FAMILIES = ("ACF", "CCF", "ACF2", "PDF", "PROJ")


def target_statistics(traj, ref):
    """The five families on the standardised fitting block, plus denominators."""
    lo, hi = int(ref["split_fit"][0]), int(ref["split_fit"][1])
    h = float(ref["bandwidth_1d"])
    z_out = ref["z_grid"]
    z_fine = np.linspace(-Z_HALFWIDTH - 6 * h, Z_HALFWIDTH + 6 * h, Z_FINE_N)

    fit = np.asarray(traj[:, lo:hi, :], dtype=np.float64)
    qt = (fit - ref["mu"]) / ref["scale"]
    if not np.all(np.isfinite(qt)):
        raise SystemExit("non-finite values in the fitting block")

    acf, ccf, acf2 = EV.correlations(qt, int(ref["k_lags"]))
    pdf = np.stack([EV.kde(qt[..., i], h, z_out, z_fine) for i in (0, 1)], axis=-1)
    proj = np.stack([EV.kde((a * qt[..., 0] + b * qt[..., 1]) / s, h, z_out, z_fine)
                     for (a, b), s in zip(ref["projections"], ref["proj_scales"])],
                    axis=-1)

    stats = {"ACF": acf, "CCF": ccf, "ACF2": acf2, "PDF": pdf, "PROJ": proj}
    denom = {k: float(np.mean(v ** 2)) for k, v in stats.items()}
    # Per-n_f PROJ denominators: the loss over the first 2*n_f directions
    # normalises by the mean square of exactly those columns, so that the
    # aggregation rule is the same at every n_f.
    denom_proj_nf = {nf: float(np.mean(proj[:, :2 * nf] ** 2)) for nf in (1, 2, 3)}
    return stats, denom, denom_proj_nf


def cache_binding(ref):
    """Identity of the data and constants this cache is valid for."""
    hh = hashlib.sha256()
    hh.update(str(ref["target_sha256"]).encode())
    hh.update(str(ref["constants_sha256"]).encode())
    hh.update(np.asarray(ref["split_fit"], dtype=np.int64).tobytes())
    hh.update(np.array([ESTIMATOR_VERSION], dtype=np.int64).tobytes())
    return hh.hexdigest()


def build(datadir, tag, verbose=True):
    ref = EV.load_reference(datadir, tag,
                            os.path.join(datadir, f"traj_{tag}.npy"))
    traj = np.load(os.path.join(datadir, f"traj_{tag}.npy"), mmap_mode="r")
    if traj.shape[0] != int(ref["l_seq"]):
        raise SystemExit(f"archive L={traj.shape[0]} but reference says "
                         f"l_seq={int(ref['l_seq'])}")

    stats, denom, denom_proj_nf = target_statistics(traj, ref)
    binding = cache_binding(ref)

    out = os.path.join(datadir, f"target_stats_{tag}.npz")
    np.savez(out,
             estimator_version=ESTIMATOR_VERSION,
             proj_aggregation=PROJ_AGGREGATION,
             ccf_mode=str(ref["ccf_mode"]),
             k_lags=int(ref["k_lags"]), l_seq=int(ref["l_seq"]),
             b_fit=int(ref["b_fit"]),
             bandwidth_1d=float(ref["bandwidth_1d"]),
             z_grid=ref["z_grid"],
             mu=ref["mu"], scale=ref["scale"],
             projections=ref["projections"], proj_scales=ref["proj_scales"],
             target_ACF=stats["ACF"], target_CCF=stats["CCF"],
             target_ACF2=stats["ACF2"], target_PDF=stats["PDF"],
             target_PROJ=stats["PROJ"],
             denom_ACF=denom["ACF"], denom_CCF=denom["CCF"],
             denom_ACF2=denom["ACF2"], denom_PDF=denom["PDF"],
             denom_PROJ_nf1=denom_proj_nf[1],
             denom_PROJ_nf2=denom_proj_nf[2],
             denom_PROJ_nf3=denom_proj_nf[3],
             cache_binding=binding,
             target_sha256=str(ref["target_sha256"]),
             constants_sha256=str(ref["constants_sha256"]))

    summary = {"tag": tag, "ccf_mode": str(ref["ccf_mode"]),
               "estimator_version": ESTIMATOR_VERSION,
               "denominators": denom,
               "denominators_PROJ_by_nf": denom_proj_nf,
               "cache_binding": binding}
    with open(os.path.join(datadir, f"target_stats_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if verbose:
        print(f"{tag:<8} ccf_mode={str(ref['ccf_mode']):<8} "
              + "  ".join(f"D_{k}={denom[k]:.4g}" for k in FAMILIES)
              + f"  D_PROJ(nf=1,2)={denom_proj_nf[1]:.4g},{denom_proj_nf[2]:.4g}")
    return summary


def verify_kde(datadir, tag):
    """Check the CIC estimator against a direct Gaussian sum on real data.

    The direct sum is the definition; CIC plus convolution is what both the
    training loss and the evaluator use, because the direct form costs
    n_samples * n_grid Gaussians per density and is not affordable per training
    step. This shows the two agree to well inside any error being reported.
    """
    ref = EV.load_reference(datadir, tag)
    traj = np.load(os.path.join(datadir, f"traj_{tag}.npy"), mmap_mode="r")
    lo, hi = int(ref["split_fit"][0]), int(ref["split_fit"][1])
    h = float(ref["bandwidth_1d"])
    z_out = ref["z_grid"]
    z_fine = np.linspace(-Z_HALFWIDTH - 6 * h, Z_HALFWIDTH + 6 * h, Z_FINE_N)
    qt = (np.asarray(traj[:, lo:hi, :], dtype=np.float64) - ref["mu"]) / ref["scale"]

    def direct(s):
        s = s.ravel()
        acc = np.zeros_like(z_out)
        for j in range(0, s.size, 20000):          # chunked to bound memory
            blk = s[j:j + 20000]
            acc += np.exp(-0.5 * ((z_out[:, None] - blk[None, :]) / h) ** 2).sum(axis=1)
        return acc / (s.size * np.sqrt(2.0 * np.pi) * h)

    worst = 0.0
    cases = [("q1", qt[..., 0]), ("q2", qt[..., 1])]
    for (a, b), s in zip(ref["projections"], ref["proj_scales"]):
        cases.append((f"({a:g},{b:g})", (a * qt[..., 0] + b * qt[..., 1]) / s))
    for name, samp in cases:
        f_cic = EV.kde(samp, h, z_out, z_fine)
        f_dir = direct(samp)
        rel = float(np.sqrt(np.mean((f_cic - f_dir) ** 2) / np.mean(f_dir ** 2)))
        worst = max(worst, rel)
        print(f"  {name:<10} CIC vs direct sum: relative L2 {rel:.3e}")
    print(f"  worst {worst:.3e} -- "
          + ("OK" if worst < 1e-4 else "TOO LARGE, investigate"))
    return worst < 1e-4


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--verify-kde", action="store_true",
                    help="compare the CIC estimator against a direct Gaussian sum")
    args = ap.parse_args()

    tags = [re.search(r"traj_(lam[\w]+)\.npy$", p).group(1)
            for p in sorted(glob.glob(os.path.join(args.datadir, "traj_lam*.npy")))]
    if not tags:
        raise SystemExit(f"no traj_lam*.npy in {args.datadir}")

    if args.verify_kde:
        ok = True
        for t in tags[:1] + ([tags[-1]] if len(tags) > 1 else []):
            print(f"[{t}] KDE estimator check")
            ok &= verify_kde(args.datadir, t)
        raise SystemExit(0 if ok else 1)

    print("training targets: EMPIRICAL statistics of the fitting block [0, B).\n"
          "The analytic densities in reference_lam*.npz are the evaluation "
          "truth and are a different thing.\n")
    for t in tags:
        build(args.datadir, t)


if __name__ == "__main__":
    main()
