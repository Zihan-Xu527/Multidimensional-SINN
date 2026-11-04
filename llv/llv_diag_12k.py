#!/usr/bin/env python3
"""Diagnostic: does the residual oscillation survive N = 12000, and is it in
the initial segment too?

This answers three questions put to the manuscript and NOTHING else. It is a
DIAGNOSTIC: it does not retrain, does not reclassify, does not reselect a
checkpoint, does not touch the frozen N = 400 evaluation, Figure 13, the
success list or any stored number. Its outputs live in their own directory
and are not inputs to the figure builder.

  1. Is the residual correlation oscillation seen in the continuation segment
     also present in the initial (scored) segment?
  2. Do the joint and marginal densities agree about equally well in the two
     segments?
  3. Does the oscillation survive when the SINN ensemble grows from N = 400
     to N = 12000?

Protocol. The frozen representative checkpoint (run 201, iteration 7000) is
rolled out once on the `density_visualization` stream -- the same stream,
seed and ensemble size the Supplementary density figure already uses, so this
is the same 12000-trajectory ensemble, not a new draw. 400 warm-up steps are
discarded and the next 800 retained continuously, with no state reset. The
retained block is then split

    retained steps   1 .. 400   = G_in,  raw [400, 800)    (scored)
    retained steps 401 .. 800   = G_out, raw [800, 1200)   (continuation)

and each half is compared with the SAME window of the 12000-trajectory
held-out target. The two halves are never concatenated before a correlation
is computed: a short-lag correlation across the join would mix the windows.

Estimators, all frozen and unchanged:

  correlations  S.corr_paircount on each 400-step block, lags 0..399, so the
                divisor is N(400 - k). The curves come out pair-count
                normalised ALREADY -- they are never multiplied by
                T/(T - k) a second time. (The main-text figure stores
                window-normalised arrays and converts at draw time; this
                script does not, and must not.)
  densities     the frozen standardisation, the frozen [-5, 5] grid, the
                frozen bandwidths h and h2, and the same Jacobian transform
                to physical coordinates.

usage:
    ARCHIVE=... SPEC=... python3 llv_diag_12k.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --out diag12k_llv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import llv_density as D
import llv_model as M
import llv_spec as S
import llv_streams as ST

SEGMENTS = (("in", "G_in", "scored", (1, 400)),
            ("out", "G_out", "continuation", (401, 800)))
CURVES = ("ACF_A", "ACF_B", "CCF_AB", "CCF_BA")


def oscillation_report(lag, tgt, sinn, lo=120):
    """A compact, honest description of the residual, per curve.

    No test and no threshold: the amplitude of each series beyond `lo`, their
    ratio, and where the largest gap sits. The caller prints it; nothing
    downstream consumes it.
    """
    out = {}
    for i, nm in enumerate(CURVES):
        t, s = tgt[i], sinn[i]
        tail = lag >= lo
        d = s - t
        j = int(np.argmax(np.abs(d)))
        out[nm] = {
            "target_amplitude_beyond_lag": float(np.abs(t[tail]).max()),
            "sinn_amplitude_beyond_lag": float(np.abs(s[tail]).max()),
            "amplitude_ratio": float(np.abs(s[tail]).max()
                                     / max(np.abs(t[tail]).max(), 1e-300)),
            "rms_difference_beyond_lag": float(np.sqrt((d[tail] ** 2).mean())),
            "max_abs_difference": float(np.abs(d[j])),
            "max_abs_difference_lag": int(lag[j]),
        }
    return out


def densities(block, dev, sample_chunk):
    """Joint and marginal densities in standardized coordinates."""
    f2 = D.kde2d(block, S.BANDWIDTH_2D, S.GRID, dev, sample_chunk)
    m1 = S.kde_eval(block, bandwidth=S.BANDWIDTH)      # (G, 2) at h
    return f2, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--out", default="diag12k_llv")
    ap.add_argument("--sample-chunk", type=int, default=20000)
    ap.add_argument("--reuse", default="",
                    help="path to a saved (1200, 12000, 2) rollout of this "
                         "checkpoint, if one exists; otherwise it is "
                         "regenerated deterministically")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    dev, dtype = args.device, torch.float64

    sel_path = os.path.join(args.select_dir, "representative.json")
    rec = os.path.join(args.select_dir, "representative.sha256")
    for p in (sel_path, rec):
        if not os.path.exists(p):
            raise SystemExit("%r is missing" % p)
    with open(sel_path) as fh:
        sel = json.load(fh)
    sel_sha = S.sha256_file(sel_path)
    if open(rec).read().strip() != sel_sha:
        raise SystemExit("the representative selection changed since it was "
                         "frozen")
    run = int(sel["representative_run"])
    it = int(sel["representative_checkpoint_iteration"])

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    os.makedirs(args.out, exist_ok=True)
    print("LLV diagnostic: N = %d, both segments" % S.B_VIS)
    print("DIAGNOSTIC ONLY -- the frozen N = %d evaluation, Figure 13, the "
          "success list and every stored number are untouched" % S.B_GEN)
    print("device %s  float64" % dev)
    print("representative run %d, checkpoint iteration %d (frozen, sha %s...)"
          % (run, it, sel_sha[:12]))

    # ---- the 12000-trajectory ensemble -----------------------------------
    path = os.path.join(args.runs_dir, "run_%d.pt" % run)
    ckpt_sha = S.sha256_file(path)
    seed = S.SEED_BASE["density_visualization"] + run
    reused = False
    if args.reuse and os.path.exists(args.reuse):
        raw = np.asarray(np.load(args.reuse)["raw"], np.float64)
        if raw.shape != (S.RAW_ROLLOUT, S.B_VIS, 2):
            raise SystemExit("%r has shape %r, expected %r"
                             % (args.reuse, raw.shape,
                                (S.RAW_ROLLOUT, S.B_VIS, 2)))
        reused = True
        print("\nreusing the saved rollout %r" % args.reuse)
    else:
        payload = torch.load(path, map_location="cpu")
        if int(payload["run"]) != run:
            raise SystemExit("checkpoint %r carries run %r"
                             % (path, payload["run"]))
        if int(payload["best"]["iteration"]) != it:
            raise SystemExit("checkpoint iteration disagrees with the frozen "
                             "selection")
        model = M.LLVGenerator(config=payload["metadata"]["config"],
                               dtype=dtype).to(dev)
        model.load_state_dict({k: v.to(dev) for k, v in
                               payload["best"]["model_state"].items()})
        model.eval()
        streams = ST.RunStreams(run, device=dev)
        print("\nregenerating the density_visualization ensemble: N = %d, "
              "%d raw steps, seed %d" % (S.B_VIS, S.RAW_ROLLOUT, seed),
              flush=True)
        print("  (same stream, seed and size as the Supplementary density "
              "figure, so this is that ensemble, not a new draw)")
        t0 = time.time()
        with torch.no_grad():
            noise = streams.noise("density_visualization", S.RAW_ROLLOUT,
                                  S.B_VIS).to(dtype)
            out, _ = model(noise)
        raw = out.detach().cpu().numpy().astype(np.float64)
        del out, noise
        print("  %.0fs" % (time.time() - t0), flush=True)

    if tuple(raw.shape) != (S.RAW_ROLLOUT, S.B_VIS, 2):
        raise AssertionError("rollout shape %r" % (tuple(raw.shape),))
    win = S.extract_windows(raw)
    gen = {"in": win["G_in"], "out": win["G_out"]}
    print("warm-up [%d, %d) discarded; retained 800 steps split into "
          "1..400 and 401..800, no reset at the join"
          % S.SLICE_WARMUP)

    results, curves_npz, dens_npz = {}, {}, {}
    for key, name, role, (t_lo, t_hi) in SEGMENTS:
        window = S.TARGET_G_IN if key == "in" else S.TARGET_G_OUT
        print("\n=== %s segment: %s, retained t = %d..%d"
              % (role, name, t_lo, t_hi), flush=True)
        tev = S.standardize(loader.open_final(window), mu, sd)
        if tev.shape[1] != S.B_VIS:
            raise SystemExit("target window has %d trajectories, expected %d"
                             % (tev.shape[1], S.B_VIS))
        g = gen[key]
        print("  SINN %s   target %s" % ((g.shape,), (tev.shape,)))

        # ---- correlations, pair-count normalised ONCE --------------------
        lag = np.arange(0, S.T_WIN, dtype=np.int64)
        c_s = S.corr_paircount(g, 0, S.T_WIN - 1)
        c_t = S.corr_paircount(tev, 0, S.T_WIN - 1)
        curves_npz["lags"] = lag
        curves_npz["curve_target_" + key] = c_t
        curves_npz["curve_sinn_" + key] = c_s
        osc = oscillation_report(lag, c_t, c_s)
        print("  pair-count ACF/CCF, lags 0..%d, divisor N(%d - k)"
              % (S.T_WIN - 1, S.T_WIN))
        print("  %-7s %10s %10s %7s %10s %12s"
              % ("curve", "|tgt|>120", "|SINN|>120", "ratio", "rms diff",
                 "max diff @k"))
        for nm in CURVES:
            o = osc[nm]
            print("  %-7s %10.4f %10.4f %7.2f %10.4f %8.4f @%d"
                  % (nm, o["target_amplitude_beyond_lag"],
                     o["sinn_amplitude_beyond_lag"], o["amplitude_ratio"],
                     o["rms_difference_beyond_lag"], o["max_abs_difference"],
                     o["max_abs_difference_lag"]))

        # ---- densities, frozen grid and bandwidths -----------------------
        print("  joint KDE at h2 on %dx%d, marginals at h" % (S.GRID_N,
                                                              S.GRID_N),
              flush=True)
        t0 = time.time()
        f2_s, m1_s = densities(g, dev, args.sample_chunk)
        f2_t, m1_t = densities(tev, dev, args.sample_chunk)
        print("    %.0fs" % (time.time() - t0), flush=True)
        jac2 = 1.0 / (sd[0] * sd[1])
        dens_npz["f2d_rho_target_" + key] = f2_t * jac2
        dens_npz["f2d_rho_sinn_" + key] = f2_s * jac2
        for i, comp in enumerate("AB"):
            dens_npz["marg_rho_%s_target_%s" % (comp, key)] = m1_t[:, i]/sd[i]
            dens_npz["marg_rho_%s_sinn_%s" % (comp, key)] = m1_s[:, i]/sd[i]

        # how far apart the two densities are, on the same scale in both
        # segments: the peak of the TARGET density of that segment
        peak = float(f2_t.max())
        d2 = float(np.abs(f2_s - f2_t).max() / peak)
        l1 = float(D.trapezoid(D.trapezoid(np.abs(f2_s - f2_t), S.GRID,
                                           axis=1), S.GRID))
        marg = {}
        for i, comp in enumerate("AB"):
            pk = float(np.abs(m1_t[:, i]).max())
            marg[comp] = {
                "max_abs_difference_over_peak":
                    float(np.abs(m1_s[:, i] - m1_t[:, i]).max() / pk),
                "L1_difference":
                    float(D.trapezoid(np.abs(m1_s[:, i] - m1_t[:, i]),
                                      S.GRID)),
            }
        print("  joint   max|f_SINN - f_KMC| / max f_KMC = %.4f ; "
              "L1 = %.4f" % (d2, l1))
        for comp in "AB":
            print("  marg %s  max diff / peak = %.4f ; L1 = %.4f"
                  % (comp, marg[comp]["max_abs_difference_over_peak"],
                     marg[comp]["L1_difference"]))

        results[key] = {
            "segment": name, "role": role,
            "retained_time": [t_lo, t_hi],
            "archive_window": list(window),
            "n_sinn": int(g.shape[1]), "n_target": int(tev.shape[1]),
            "correlation": {
                "estimator": "S.corr_paircount on the 400-step block, "
                             "divisor N(400 - k)",
                "already_pair_count_normalised": True,
                "converted_again": False,
                "lag_range": [0, S.T_WIN - 1],
                "oscillation": osc,
            },
            "density": {
                "joint_max_abs_difference_over_target_peak": d2,
                "joint_L1_difference": l1,
                "marginal": marg,
                "normalised_by": "the target peak of THIS segment",
            },
        }
        del tev, f2_s, f2_t, m1_s, m1_t

    npz_c = os.path.join(args.out, "diag12k_curves.npz")
    np.savez_compressed(npz_c, curve_names=np.array(CURVES), **curves_npz)
    npz_d = os.path.join(args.out, "diag12k_density.npz")
    np.savez_compressed(npz_d, grid_x=S.GRID,
                        grid_rho_A=S.GRID * sd[0] + mu[0],
                        grid_rho_B=S.GRID * sd[1] + mu[1],
                        mu=mu, s=sd, **dens_npz)

    doc = {
        "stage": "diagnostic, N = %d, both segments" % S.B_VIS,
        "purpose": ["is the continuation residual oscillation present in the "
                    "initial segment too",
                    "do the two segments' densities agree about equally well",
                    "does the oscillation survive N = 400 -> N = %d"
                    % S.B_VIS],
        "diagnostic_only": True,
        "changes_formal_evaluation": False,
        "retrained_anything": False,
        "reclassified_anything": False,
        "reselected_checkpoint": False,
        "feeds_the_figure_builder": False,
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "representative_run": run,
        "representative_checkpoint_iteration": it,
        "checkpoint_file_sha256": ckpt_sha,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_model.py",
                           "llv_density.py", "llv_diag_12k.py")
                          if os.path.exists(n)},
        "ensemble": {
            "N": S.B_VIS, "prose_symbol": "N",
            "stream": "density_visualization", "seed": seed,
            "raw_rollout": S.RAW_ROLLOUT,
            "warmup_discarded": list(S.SLICE_WARMUP),
            "retained": "800 steps, continuous, no reset at the join",
            "reused_saved_rollout": reused,
            "same_ensemble_as": "the Supplementary density figure",
        },
        "segments_never_concatenated": True,
        "segments": results,
        "artifacts": {
            "curves": {"path": os.path.abspath(npz_c),
                       "sha256": S.sha256_file(npz_c)},
            "density": {"path": os.path.abspath(npz_d),
                        "sha256": S.sha256_file(npz_d)},
        },
        "caveat": "the per-family paired losses of the formal evaluation are "
                  "not evidence that any individual curve or contour agrees; "
                  "the numbers here describe the curves and densities "
                  "themselves",
        "torch_version": torch.__version__, "device": dev,
    }
    p = os.path.join(args.out, "diag12k.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    print("\nwrote %s (sha %s...)" % (npz_c, S.sha256_file(npz_c)[:12]))
    print("wrote %s (sha %s...)" % (npz_d, S.sha256_file(npz_d)[:12]))
    print("wrote %s" % p)
    print("\nnow draw the four figures:")
    print("    python3 llv_diag_figures.py --diag-dir %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
