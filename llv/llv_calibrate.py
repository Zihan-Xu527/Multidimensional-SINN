#!/usr/bin/env python3
"""Stage 3 of spec section 13: validation-side calibration.

Two deliverables, both validation-only:

  1. the burn-in / empirical-stationarity summary of spec section 3.1;
  2. the 2000-replicate long-lag envelope of spec section 10.

It never reads T_ev, never touches a checkpoint, and nothing it computes
depends on any training or pilot outcome. Every threshold it emits was
predeclared in the frozen specification before any run existed.

Spec section 10, the envelope:

  Before opening T_ev, draw 2000 trajectory-level bootstrap blocks of size 400
  from R = union_j R_j, preserving each trajectory's complete 800-step
  dependence. Compare each block with V over the same interval and compute

      M_ref   = max over k = 400..600 and over
                ACF_A, ACF_B, CCF_AB, CCF_BA
                of |curve_block - curve_V|
      eps_lag = empirical q_0.95(M_ref)

  The bootstrap generator uses seed 7000. No factor of three and no pointwise
  band are added. ACF2 is not part of this maximum statistic.

The curves use the pair-count estimator of section 10 on the combined retained
interval G_in u G_out, i.e. physical t = 401..1200, length 800: divisor
B*(800 - k), means and standard deviations computed separately for each
ensemble whose curve is measured, never borrowed from F.

usage:
    ARCHIVE=... SPEC=... python3 llv_calibrate.py \
        --manifest-dir llv_manifest_v2 --out calib_llv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import llv_spec as S

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                    # pragma: no cover
    torch = None
    HAVE_TORCH = False

#: the combined retained interval, physical t = 401..1200 -> array [400:1200)
COMBINED = S.SLICE_COMBINED
#: R = union_j R_j, as one pool of trajectories
R_POOL = (S.ref_block(1)[0], S.ref_block(9)[1])


# ------------------------------------------------------- pair-count on torch --
def paircount_torch(block, k_lo: int, k_hi: int):
    """Same estimator as S.corr_paircount, on the device, float64.

    Returns (4, k_hi - k_lo + 1) in the fixed curve order
    ACF_A, ACF_B, CCF_AB, CCF_BA.
    """
    a = block
    n_t, n_b, _ = a.shape
    mean = a.mean(dim=(0, 1))
    sd = torch.sqrt(((a - mean) ** 2).mean(dim=(0, 1)))
    xc = a - mean
    n_fft = 1 << int(np.ceil(np.log2(2 * n_t)))
    f = torch.fft.rfft(xc, n=n_fft, dim=0)
    lags = torch.arange(k_lo, k_hi + 1, device=a.device, dtype=a.dtype)
    pairs = (n_t - lags) * n_b

    def cross(i, j):
        cc = torch.fft.irfft(torch.conj(f[:, :, i]) * f[:, :, j], n=n_fft,
                             dim=0)
        return cc[k_lo:k_hi + 1].sum(dim=1) / pairs / (sd[i] * sd[j])

    return torch.stack([cross(0, 0), cross(1, 1), cross(0, 1), cross(1, 0)])


# ------------------------------------------------------------------ envelope --
def long_lag_envelope(traj, mu, sd, device, out_dir):
    k_lo, k_hi = S.LONGLAG_K_LO, S.LONGLAG_K_HI
    n_boot, blk = S.LONGLAG_N_BOOT, S.LONGLAG_BLOCK
    lo, hi = COMBINED

    v = S.standardize(
        np.asarray(traj[lo:hi, S.SPLIT_V[0]:S.SPLIT_V[1], :], np.float64),
        mu, sd)
    pool = S.standardize(
        np.asarray(traj[lo:hi, R_POOL[0]:R_POOL[1], :], np.float64), mu, sd)
    n_pool = pool.shape[1]
    print("  V %s over physical t = %d..%d;  R pool %d trajectories"
          % (v.shape, lo + 1, hi, n_pool))

    # The target curve the blocks are compared against: V, computed once.
    curve_v_np = S.corr_paircount(v, k_lo, k_hi)

    if HAVE_TORCH:
        dev = device
        v_t = torch.as_tensor(v, dtype=torch.float64, device=dev)
        pool_t = torch.as_tensor(pool, dtype=torch.float64, device=dev)
        curve_v = paircount_torch(v_t, k_lo, k_hi)
        # the torch path must reproduce the NumPy estimator exactly
        dv = float(np.max(np.abs(curve_v.cpu().numpy() - curve_v_np)))
        if not dv < 1e-10:
            raise SystemExit("torch pair-count deviates from NumPy by %.3e"
                             % dv)
        print("  torch pair-count matches NumPy to %.1e" % dv)
    else:
        curve_v = None
        print("  no torch; running the bootstrap in NumPy (slow)")

    rng = np.random.default_rng(S.SEED_LONGLAG_BOOTSTRAP)

    m_ref = np.empty(n_boot, dtype=np.float64)
    arg_curve = np.empty(n_boot, dtype=np.int64)
    arg_lag = np.empty(n_boot, dtype=np.int64)
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n_pool, blk)          # with replacement
        if HAVE_TORCH:
            sel = torch.as_tensor(idx, device=device)
            c = paircount_torch(pool_t.index_select(1, sel), k_lo, k_hi)
            d = (c - curve_v).abs()
            flat = int(torch.argmax(d))
            m_ref[b] = float(d.reshape(-1)[flat])
        else:
            c = S.corr_paircount(pool[:, idx, :], k_lo, k_hi)
            d = np.abs(c - curve_v_np)
            flat = int(np.argmax(d))
            m_ref[b] = float(d.reshape(-1)[flat])
        arg_curve[b] = flat // (k_hi - k_lo + 1)
        arg_lag[b] = k_lo + flat % (k_hi - k_lo + 1)
        if (b + 1) % 500 == 0:
            print("    %4d/%d  %.0fs" % (b + 1, n_boot, time.time() - t0),
                  flush=True)

    eps_lag = float(np.quantile(m_ref, 0.95))
    curve_v_out = (curve_v.cpu().numpy() if HAVE_TORCH else curve_v_np)

    np.savez(os.path.join(out_dir, "longlag_envelope.npz"),
             M_ref=m_ref, curve_V=curve_v_out,
             lags=np.arange(k_lo, k_hi + 1),
             argmax_curve=arg_curve, argmax_lag=arg_lag)

    counts = {S.LONGLAG_CURVES[i]: int((arg_curve == i).sum())
              for i in range(4)}
    return {
        "eps_lag": eps_lag,
        "quantile": 0.95,
        "factor": 1.0,
        "no_multiplier": "no factor of three and no pointwise band",
        "n_boot": n_boot,
        "block_size": blk,
        "resampling": "trajectory-level, with replacement, each trajectory's "
                      "complete 800-step dependence preserved",
        "pool": {"indices": list(R_POOL), "n_trajectories": int(n_pool),
                 "source": "R = union_j R_j"},
        "compared_against": {"set": "V", "indices": list(S.SPLIT_V)},
        "interval": {"array_slice": list(COMBINED),
                     "physical_t": [lo + 1, hi], "length": hi - lo},
        "estimator": "pair-count normalized, divisor B*(800 - k); means and "
                     "s.d. per ensemble, never borrowed from F",
        "lag_range": [k_lo, k_hi],
        "k_hi_justification": "800 - k >= (800 - 400)/2, so every reported lag "
                              "keeps at least 200 time pairs per trajectory",
        "curves": list(S.LONGLAG_CURVES),
        "acf2_excluded": True,
        "seed": S.SEED_LONGLAG_BOOTSTRAP,
        "M_ref_summary": {
            "min": float(m_ref.min()), "q05": float(np.quantile(m_ref, 0.05)),
            "median": float(np.median(m_ref)),
            "q95": eps_lag, "max": float(m_ref.max()),
            "mean": float(m_ref.mean())},
        "argmax_curve_counts": counts,
        "argmax_lag_summary": {"min": int(arg_lag.min()),
                               "median": float(np.median(arg_lag)),
                               "max": int(arg_lag.max())},
        "curve_V_absmax": float(np.abs(curve_v_out).max()),
        "wall_seconds": time.time() - t0,
    }


# ------------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--out", default="calib_llv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.archive or not os.path.exists(args.archive):
        raise SystemExit("set ARCHIVE to the processed LLV archive")
    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    loader.assert_unlocked()           # verifies the chain; does not open T_ev
    man_path = os.path.join(args.manifest_dir, S.MANIFEST_NAME)
    with open(man_path) as fh:
        man = json.load(fh)

    dev = args.device or ("cuda" if HAVE_TORCH and torch.cuda.is_available()
                          else "cpu")
    print("LLV validation-side calibration, spec section 13 stage 3")
    print("device %s  torch %s" % (dev, HAVE_TORCH))
    os.makedirs(args.out, exist_ok=True)

    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    # ---- 1. burn-in / stationarity summary ------------------------------
    # Gate 10 already computed this, under the frozen spec 3.1 procedure, and
    # reproduced all six frozen values. It is carried through from the gate
    # manifest rather than reimplemented here: a second implementation of the
    # same calculation is a source of disagreement, not of assurance.
    burn = man.get("burnin")
    if not burn:
        raise SystemExit("the gate manifest carries no burn-in block")
    for key, want in (("seed", S.SEED_BURNIN_BOOTSTRAP),
                      ("tested_half", list(S.BURNIN_TESTED_HALF)),
                      ("reference_half", list(S.BURNIN_REFERENCE_HALF)),
                      ("n_replicates", S.BURNIN_N_REPLICATES)):
        if burn.get(key) != want:
            raise SystemExit("gate-10 burn-in %s is %r, expected %r"
                             % (key, burn.get(key), want))
    stationarity = {
        "source": "gate 10 of llv_manifest_v2, carried through unchanged",
        "procedure": "verify_and_calibrate_v2.py, frozen spec 3.1",
        "reads": "validation halves only",
        **burn,
    }
    print("\nburn-in / stationarity (from gate 10)")
    for key in ("mean", "variance", "ks"):
        print("  %-9s observed %.6f   limit %.6f   frozen %s"
              % (key, burn["observed"][key], burn["limits"][key],
                 S.BURNIN_FROZEN[key]))
    for s in ("ACF", "CCF", "ACF2"):
        print("  d_win[%-4s] %.6f  (squared L %.4e)  fixed tolerance %.2f"
              % (s, burn["d_win"][s], burn["L_squared"][s],
                 S.DWIN_TOLERANCE))

    # ---- 2. long-lag envelope -------------------------------------------
    print("\nlong-lag envelope, %d replicates, seed %d"
          % (S.LONGLAG_N_BOOT, S.SEED_LONGLAG_BOOTSTRAP), flush=True)
    env = long_lag_envelope(loader.traj, mu, sd, dev, args.out)

    print("\n  M_ref   min %.6f  median %.6f  q95 %.6f  max %.6f"
          % (env["M_ref_summary"]["min"], env["M_ref_summary"]["median"],
             env["eps_lag"], env["M_ref_summary"]["max"]))
    print("  eps_lag = %.6f   (q_0.95 of M_ref, no multiplier)"
          % env["eps_lag"])
    print("  maximizing curve counts: " + "  ".join(
        "%s %d" % (k, v) for k, v in env["argmax_curve_counts"].items()))
    print("  maximizing lag: min %d  median %.0f  max %d"
          % (env["argmax_lag_summary"]["min"],
             env["argmax_lag_summary"]["median"],
             env["argmax_lag_summary"]["max"]))
    print("  %.0fs" % env["wall_seconds"])

    out = {
        "stage": "calibration",
        "specification": man.get("specification"),
        "specification_sha256": man.get("specification_sha256"),
        "archive_path": args.archive,
        "archive_sha256": man["archive_sha256"],
        "manifest_dir": os.path.abspath(args.manifest_dir),
        "manifest_sha256": S.sha256_file(man_path),
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_gates.py",
                           "llv_calibrate.py") if os.path.exists(n)},
        "standardization": {"mu": mu.tolist(), "s": sd.tolist()},
        "stationarity": stationarity,
        "long_lag_envelope": env,
        "reads": ["R_j", "V"],
        "never_reads": ["T_ev"],
        "depends_on_training": False,
        "note": "every threshold here was predeclared in the frozen "
                "specification; none of it was derived from, or adjusted "
                "after, any development, pilot or formal training outcome",
        "torch_version": torch.__version__ if HAVE_TORCH else None,
        "device": dev,
    }
    path = os.path.join(args.out, "calibration.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    digest = S.sha256_file(path)
    with open(os.path.join(args.out, "calibration.sha256"), "w") as fh:
        fh.write(digest + "\n")
    print("\nwrote %s (sha %s...) and longlag_envelope.npz"
          % (path, digest[:12]))
    print("T_ev was not opened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
