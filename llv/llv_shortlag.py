#!/usr/bin/env python3
"""Export the short-lag G_out correlation curves for the main-text figure.

Scope. This is a PRESENTATION step. It regenerates nothing that any decision
rests on: no training, no archive rebuild, no reclassification, no reselection
of the representative checkpoint, and no recomputation of an endpoint summary.
It reproduces one deterministic rollout of the already frozen representative
checkpoint so that the curves behind the stored ACF and CCF losses can be
drawn, and it asserts that those stored losses come back exactly.

What is computed, per spec sections 4 and 8:

    C_AA(k), C_BB(k), C_AB(k), C_BA(k),   0 <= k <= 399

on the continuous held-out window G_out (raw steps [800, 1200)), for the SINN
ensemble and for T_ev,out, with `S.corr_window` -- the SAME estimator and the
SAME standardization the final ACF and CCF family losses used. The pair-count
long-lag estimator (`S.corr_paircount`) is NOT used here; it belongs to the
Supplementary 400..600 figure and answers a different question.

The cross-check is the point of this script. Stacking the curves into the
frozen family order gives

    ACF = [C_AA, C_BB],    CCF = [C_AB, C_BA]

and `S.relative_loss` against the T_ev,out arrays must reproduce the stored
run-201 G_out ACF and CCF losses with zero difference. If it does not, this
exits nonzero rather than publishing curves that do not belong to the numbers
already in the table.

usage:
    ARCHIVE=... SPEC=... python3 llv_shortlag.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --final-dir final_llv --out shortlag_llv
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

# panel order; the index pair is into the (2, 2, K) correlation matrix
CURVES = (("ACF_A", 0, 0), ("ACF_B", 1, 1),
          ("CCF_AB", 0, 1), ("CCF_BA", 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--final-dir", default="final_llv")
    ap.add_argument("--out", default="shortlag_llv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    dev, dtype = args.device, torch.float64

    # ---- the frozen selection, verified ----------------------------------
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

    # ---- the stored endpoint record this export must reproduce -----------
    final_path = os.path.join(args.final_dir, "final_evaluation.json")
    if not os.path.exists(final_path):
        raise SystemExit("stage-4 result %r is missing; the loss cross-check "
                         "cannot be made" % final_path)
    with open(final_path) as fh:
        final = json.load(fh)
    rows = [r for r in final["runs"] if int(r["run"]) == run]
    if len(rows) != 1:
        raise SystemExit("stage 4 has %d records for run %d" % (len(rows), run))
    stage4 = rows[0]
    stored = {s: float(stage4["L_eval"]["out"][s]) for s in ("ACF", "CCF")}

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    os.makedirs(args.out, exist_ok=True)
    print("LLV short-lag G_out correlation export for the main-text figure")
    print("device %s  float64" % dev)
    print("representative run %d, checkpoint iteration %d (frozen, sha %s...)"
          % (run, sel["representative_checkpoint_iteration"], sel_sha[:12]))
    print("estimator S.corr_window, lags 0..%d, window G_out = raw [%d, %d)"
          % (S.K_LAGS - 1, *S.SLICE_G_OUT))
    print("standardized coordinates, the same ones the ACF/CCF losses used")
    print("the pair-count long-lag estimator is NOT used here\n")

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

    print("regenerating the stage-4 rollout: N = %d, %d raw steps, "
          "final_generation seed %d"
          % (S.B_GEN, S.RAW_ROLLOUT, S.SEED_BASE["final_generation"] + run),
          flush=True)
    t0 = time.time()
    with torch.no_grad():
        noise = streams.noise("final_generation", S.RAW_ROLLOUT,
                              S.B_GEN).to(dtype)
        out, _ = model(noise)
    raw = out.detach().cpu().numpy().astype(np.float64)
    if tuple(raw.shape) != (S.RAW_ROLLOUT, S.B_GEN, 2):
        raise AssertionError("rollout shape %r" % (tuple(raw.shape),))
    win = S.extract_windows(raw)
    gen_out = win["G_out"]
    print("  %.0fs; G_out %s" % (time.time() - t0, (gen_out.shape,)),
          flush=True)

    # ---- T_ev,out, sliced exactly as stage 4 sliced it -------------------
    tev = S.standardize(loader.open_final(S.TARGET_COMBINED), mu, sd)
    lo_out = S.TARGET_G_OUT[0] - S.TARGET_COMBINED[0]
    tev_out = np.ascontiguousarray(tev[lo_out:lo_out + S.T_WIN])
    del tev
    print("T_ev,out %s" % (tev_out.shape,))

    # ---- the curves ------------------------------------------------------
    c_sinn = S.corr_window(gen_out, S.K_LAGS)
    c_tgt = S.corr_window(tev_out, S.K_LAGS)
    curve_sinn = np.stack([c_sinn[i, j] for _, i, j in CURVES])
    curve_tgt = np.stack([c_tgt[i, j] for _, i, j in CURVES])

    # ---- the cross-check against the stored endpoint losses --------------
    fam_sinn = {"ACF": np.stack([c_sinn[0, 0], c_sinn[1, 1]]),
                "CCF": np.stack([c_sinn[0, 1], c_sinn[1, 0]])}
    fam_tgt = {"ACF": np.stack([c_tgt[0, 0], c_tgt[1, 1]]),
               "CCF": np.stack([c_tgt[0, 1], c_tgt[1, 0]])}
    here = {s: S.relative_loss(fam_sinn[s], fam_tgt[s]) for s in ("ACF", "CCF")}

    print("\ncross-check against the stored run-%d G_out losses" % run)
    bad = []
    for s in ("ACF", "CCF"):
        delta = abs(here[s] - stored[s])
        print("  L_%-4s here %.17g   stage 4 %.17g   |difference| %.3e"
              % (s, here[s], stored[s], delta))
        if delta != 0.0:
            bad.append(s)
    if bad:
        raise SystemExit(
            "these curves do not reproduce the stored %s loss(es) for run %d. "
            "The same checkpoint, the same seed and the same estimator must "
            "give the same number; refusing to publish curves that do not "
            "belong to the figures and tables already frozen."
            % (", ".join(bad), run))
    print("  identical. The exported curves are the ones the stored ACF and "
          "CCF losses were computed from.")

    lags = np.arange(0, S.K_LAGS, dtype=np.int64)
    npz = os.path.join(args.out, "shortlag_curves.npz")
    np.savez_compressed(
        npz, lags=lags, curve_target=curve_tgt, curve_sinn=curve_sinn,
        curve_names=np.array([n for n, _, _ in CURVES]),
        corr_target_full=c_tgt, corr_sinn_full=c_sinn)
    npz_sha = S.sha256_file(npz)

    doc = {
        "stage": "short-lag G_out correlation export for the main-text figure",
        "purpose": "the main-text correlation figure shows the scored lag "
                   "range; stage 4 stored the ACF and CCF losses but not the "
                   "curves, and the figure builder may only read saved "
                   "products",
        "regenerated": "one deterministic rollout of the frozen "
                       "representative checkpoint under stage 4's exact "
                       "protocol",
        "retrained_anything": False,
        "rebuilt_archive": False,
        "reclassified_anything": False,
        "reselected_checkpoint": False,
        "changed_accepted_mask": False,
        "recomputed_any_endpoint_summary": False,
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "final_evaluation_sha256": S.sha256_file(final_path),
        "representative_run": run,
        "representative_checkpoint_iteration":
            int(sel["representative_checkpoint_iteration"]),
        "checkpoint_file_sha256": ckpt_sha,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_model.py",
                           "llv_shortlag.py") if os.path.exists(n)},
        "protocol": {
            "ensemble_size_N": S.B_GEN,
            "prose_symbol": "N",
            "raw_rollout": S.RAW_ROLLOUT,
            "stream": "final_generation",
            "seed": S.SEED_BASE["final_generation"] + run,
            "window": "G_out, raw [%d, %d), physical t = 801..1200"
                      % S.SLICE_G_OUT,
            "continuous": "the recurrent state is carried through raw step "
                          "800; G_out is not a fresh rollout",
            "estimator": "S.corr_window, window-normalized, divisor N*T at "
                         "every lag, pooled means and pooled s.d. per "
                         "ensemble",
            "estimator_note": "this is the estimator the ACF and CCF family "
                              "losses use; the pair-count long-lag estimator "
                              "is not used in this figure",
            "coordinates": "standardized, mu and s from the frozen manifest",
            "lag_range": [0, S.K_LAGS - 1],
            "curve_order": [n for n, _, _ in CURVES],
        },
        "cross_check_vs_stage_4": {
            "window": "G_out",
            "families": ["ACF", "CCF"],
            "L_here": here,
            "L_stage_4": stored,
            "absolute_difference": {s: abs(here[s] - stored[s])
                                    for s in ("ACF", "CCF")},
            "identical": True,
            "note": "the ACF array is [C_AA, C_BB] and the CCF array is "
                    "[C_AB, C_BA], the frozen family order",
        },
        "figure_requirements": {
            "panels": [n for n, _, _ in CURVES],
            "x_axis": [0, S.K_LAGS - 1],
            "envelope": False,
            "envelope_note": "no finite-sample envelope is drawn in this "
                             "figure",
            "target_label": "Target",
            "model_label": "SINN",
        },
        "artifact": {"path": os.path.abspath(npz), "sha256": npz_sha},
        "torch_version": torch.__version__,
        "device": dev,
    }
    p = os.path.join(args.out, "shortlag_curves.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "shortlag_curves.sha256"), "w") as fh:
        fh.write(digest + "\n")

    print("\nwrote %s (sha %s...)" % (npz, npz_sha[:12]))
    print("wrote %s (sha %s...)" % (p, digest[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
