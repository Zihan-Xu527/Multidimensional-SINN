#!/usr/bin/env python3
"""Export the long-lag correlation curves the main-text figure needs.

Stage 4 recorded M_SINN and where its maximum fell, but not the curves
themselves, so the four-panel main-text figure had no data to be drawn from.
The figure builder may only read saved products (spec section 13), so the
curves are exported here rather than regenerated inside the builder.

This step regenerates ONE rollout: the frozen representative checkpoint, the
same final_generation stream and the same N = 400, 1200-step protocol stage 4
used. It therefore reproduces stage 4's M_SINN exactly, and that agreement is
asserted rather than assumed -- if the two disagree by anything at all, this
exits nonzero instead of quietly publishing a second version of the number.

Nothing is reclassified, no checkpoint is reselected, and no summary statistic
is recomputed: the only new artifact is the curve data.

What the figure needs, per spec section 10: the four curves
ACF_A, ACF_B, CCF_AB, CCF_BA over lags 400..600 for the representative run and
for T_ev, plus the frozen eps_lag so the band C_target(k) +/- eps_lag can be
drawn. The band is equal-width and transferred from the validation side; it was
calibrated from R-versus-V, not from final data.

usage:
    ARCHIVE=... SPEC=... python3 llv_curves.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --calib-dir calib_llv \
        --final-dir final_llv --out curves_llv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import llv_model as M
import llv_spec as S
import llv_streams as ST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--calib-dir", default="calib_llv")
    ap.add_argument("--final-dir", default="final_llv")
    ap.add_argument("--out", default="curves_llv")
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

    calib_path = os.path.join(args.calib_dir, "calibration.json")
    with open(calib_path) as fh:
        calib = json.load(fh)
    eps_lag = float(calib["long_lag_envelope"]["eps_lag"])

    final_path = os.path.join(args.final_dir, "final_evaluation.json")
    if not os.path.exists(final_path):
        raise SystemExit("stage-4 result %r is missing; the M_SINN "
                         "cross-check cannot be made" % final_path)
    with open(final_path) as fh:
        final = json.load(fh)
    rows = [r for r in final["runs"] if int(r["run"]) == run]
    if len(rows) != 1:
        raise SystemExit("stage 4 has %d records for run %d" % (len(rows), run))
    stage4 = rows[0]

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    os.makedirs(args.out, exist_ok=True)
    print("LLV long-lag curve export for the main-text figure")
    print("device %s  float64" % dev)
    print("representative run %d, checkpoint iteration %d (frozen, sha %s...)"
          % (run, sel["representative_checkpoint_iteration"], sel_sha[:12]))
    print("eps_lag = %.6f (frozen in stage 3, calibrated R-vs-V)" % eps_lag)
    print("lags %d..%d, curves %s"
          % (S.LONGLAG_K_LO, S.LONGLAG_K_HI, ", ".join(S.LONGLAG_CURVES)))

    # ---- the representative rollout, identical protocol to stage 4 -------
    path = os.path.join(args.runs_dir, "run_%d.pt" % run)
    ckpt_sha = S.sha256_file(path)
    payload = torch.load(path, map_location="cpu")
    if int(payload["run"]) != run:
        raise SystemExit("checkpoint %r carries run %r" % (path,
                                                           payload["run"]))
    if int(payload["best"]["iteration"]) != \
            int(sel["representative_checkpoint_iteration"]):
        raise SystemExit("checkpoint iteration disagrees with the frozen "
                         "selection")
    cfg = payload["metadata"]["config"]
    model = M.LLVGenerator(config=cfg, dtype=dtype).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in
                           payload["best"]["model_state"].items()})
    model.eval()
    streams = ST.RunStreams(run, device=dev)

    print("\nregenerating the stage-4 rollout: N = %d, %d raw steps, "
          "final_generation seed %d"
          % (S.B_GEN, S.RAW_ROLLOUT, S.SEED_BASE["final_generation"] + run),
          flush=True)
    t0 = time.time()
    with torch.no_grad():
        noise = streams.noise("final_generation", S.RAW_ROLLOUT,
                              S.B_GEN).to(dtype)
        out, _ = model(noise)
    raw = out.detach().cpu().numpy().astype(np.float64)
    win = S.extract_windows(raw)
    curve_sinn = S.corr_paircount(win["combined"], S.LONGLAG_K_LO,
                                  S.LONGLAG_K_HI)
    print("  %.0fs" % (time.time() - t0), flush=True)

    tev = S.standardize(loader.open_final(S.TARGET_COMBINED), mu, sd)
    curve_tev = S.corr_paircount(tev, S.LONGLAG_K_LO, S.LONGLAG_K_HI)
    del tev

    # ---- the cross-check against stage 4 ---------------------------------
    dif = np.abs(curve_sinn - curve_tev)
    flat = int(np.argmax(dif))
    n_lag = S.LONGLAG_K_HI - S.LONGLAG_K_LO + 1
    m_sinn = float(dif.reshape(-1)[flat])
    arg_curve = S.LONGLAG_CURVES[flat // n_lag]
    arg_lag = S.LONGLAG_K_LO + flat % n_lag

    s4_m = float(stage4["M_SINN"])
    delta = abs(m_sinn - s4_m)
    print("\ncross-check against stage 4")
    print("  M_SINN here      %.17g" % m_sinn)
    print("  M_SINN stage 4   %.17g" % s4_m)
    print("  absolute difference %.3e" % delta)
    print("  argmax here      %s at lag %d" % (arg_curve, arg_lag))
    print("  argmax stage 4   %s at lag %d"
          % (stage4["M_SINN_argmax"]["curve"],
             int(stage4["M_SINN_argmax"]["lag"])))
    if delta != 0.0 or arg_curve != stage4["M_SINN_argmax"]["curve"] \
            or arg_lag != int(stage4["M_SINN_argmax"]["lag"]):
        raise SystemExit(
            "this rollout does not reproduce stage 4's long-lag statistic. "
            "The same seed and the same protocol must give the same number; "
            "refusing to publish a second version of it.")
    print("  identical. The exported curves are the ones stage 4 measured.")

    lags = np.arange(S.LONGLAG_K_LO, S.LONGLAG_K_HI + 1)
    npz = os.path.join(args.out, "longlag_curves.npz")
    np.savez_compressed(
        npz, lags=lags, curve_target=curve_tev, curve_sinn=curve_sinn,
        band_lower=curve_tev - eps_lag, band_upper=curve_tev + eps_lag,
        eps_lag=np.float64(eps_lag),
        curve_names=np.array(S.LONGLAG_CURVES))
    npz_sha = S.sha256_file(npz)

    doc = {
        "stage": "long-lag curve export for the main-text figure",
        "purpose": "stage 4 stored M_SINN but not the curves; the figure "
                   "builder may only read saved products, so the curves are "
                   "exported here",
        "regenerated": "one rollout of the frozen representative checkpoint "
                       "under stage 4's exact protocol",
        "reclassified_anything": False,
        "reselected_checkpoint": False,
        "recomputed_any_summary_statistic": False,
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "calibration_sha256": S.sha256_file(calib_path),
        "final_evaluation_sha256": S.sha256_file(final_path),
        "representative_run": run,
        "representative_checkpoint_iteration":
            int(sel["representative_checkpoint_iteration"]),
        "checkpoint_file_sha256": ckpt_sha,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_model.py",
                           "llv_curves.py") if os.path.exists(n)},
        "protocol": {
            "ensemble_size_N": S.B_GEN,
            "prose_symbol": "N",
            "raw_rollout": S.RAW_ROLLOUT,
            "stream": "final_generation",
            "seed": S.SEED_BASE["final_generation"] + run,
            "interval": "combined G_in u G_out, physical t = 401..1200",
            "estimator": "pair-count normalized, divisor N(800 - k)",
            "lag_range": [S.LONGLAG_K_LO, S.LONGLAG_K_HI],
            "curve_order": list(S.LONGLAG_CURVES),
        },
        "band": {
            "eps_lag": eps_lag,
            "form": "C_target(k) +/- eps_lag, equal width at every lag",
            "provenance": "calibrated on the validation side from R versus V "
                          "in stage 3, before T_ev was opened; not from final "
                          "data",
            "no_multiplier": True,
        },
        "cross_check_vs_stage_4": {
            "M_SINN": m_sinn,
            "M_SINN_stage_4": s4_m,
            "absolute_difference": delta,
            "argmax_curve": arg_curve,
            "argmax_lag": arg_lag,
            "identical": True,
        },
        "M_SINN_over_eps_lag": m_sinn / eps_lag,
        "figure_requirements": {
            "panels": list(S.LONGLAG_CURVES),
            "x_axis_ends_at": S.LONGLAG_K_HI,
            "mark_lag": S.LONGLAG_K_LO,
            "target_label": "Target",
            "model_label": "SINN",
            "caption_must_state": [
                "the representative run's seed",
                "M_SINN / eps_lag = %.3f" % (m_sinn / eps_lag),
                "the band was calibrated from R-versus-V, not from final data",
            ],
        },
        "artifact": {"path": os.path.abspath(npz), "sha256": npz_sha},
        "torch_version": torch.__version__,
        "device": dev,
    }
    p = os.path.join(args.out, "longlag_curves.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "longlag_curves.sha256"), "w") as fh:
        fh.write(digest + "\n")

    print("\nM_SINN / eps_lag = %.3f" % (m_sinn / eps_lag))
    print("wrote %s (sha %s...)" % (npz, npz_sha[:12]))
    print("wrote %s (sha %s...)" % (p, digest[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
