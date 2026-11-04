#!/usr/bin/env python3
"""Stage 2 of spec section 13: the independent validation classifier.

Spec section 7, verbatim:

  After the best checkpoint is frozen, classify it using a NEW B=400,
  length-800 generated ensemble from the independent classification stream;
  discard raw [0:400) and use [400:800) as G_in. Do not reuse the ensemble
  that attained the minimum checkpoint score.

      L_{s,ref}^{val,in} = median_{j=1..9} L_s(R_{j,in}, V_in)
      r_s^{val,(m)}      = L_s(G_in^(m), V_in) / L_{s,ref}^{val,in}
      r^{val,(m)}        = max_s r_s^{val,(m)}
      run m is validation-successful iff r^{val,(m)} <= 10   (inclusive)

UNITS. Every L_s is the SQUARED relative discrepancy of spec section 4, so
every r_s is a ratio of squared quantities. The unsquared discrepancy ratio is
sqrt(r_s). Both are reported, always labelled, and never interchanged: r = 63.8
corresponds to an unsquared ratio of 8.0.

This stage reads R_j and V. It never opens T_ev. It also never re-selects a
checkpoint: the checkpoint is whatever the trainer's Q_val minimum retained,
and the diagnostics saved here (sigma_s, the effective weights, the six
per-direction PROJ contributions) are recorded for interpretation only and must
not be used to revise a checkpoint or an accepted mask after the fact.

usage:
    ARCHIVE=... SPEC=... python3 llv_classify.py \
        --manifest-dir llv_manifest_v2 --runs-dir runs_llv --out class_llv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

import llv_model as M
import llv_spec as S
import llv_streams as ST
import llv_torch as T

#: labels for the six projected directions, in the frozen frame/row order
PROJ_LABELS = tuple(
    "frame%d_row%d" % (f + 1, r + 1) for f in range(3) for r in range(2))


def per_direction_losses(gen_proj: np.ndarray, tgt_proj: np.ndarray) -> dict:
    """Relative squared loss of each PROJ direction on its own.

    Diagnostic only. The family loss L_PROJ is NOT the mean of these: it
    divides one pooled numerator by one pooled denominator.
    """
    out = {}
    for i, name in enumerate(PROJ_LABELS):
        out[name] = {
            "L_squared": S.relative_loss(gen_proj[i], tgt_proj[i]),
            "target_norm_sq": float(np.dot(tgt_proj[i], tgt_proj[i])),
        }
        out[name]["unsquared"] = float(np.sqrt(out[name]["L_squared"]))
    return out


def classify_one(payload: dict, ctx: dict) -> dict:
    run = int(payload["run"])
    cfg = payload["metadata"]["config"]
    dev, dtype = ctx["device"], ctx["dtype"]

    # Rebuild from the checkpoint's own recorded configuration, not from the
    # module-level default, so a checkpoint trained at another hidden width
    # classifies correctly without editing this file.
    model = M.LLVGenerator(config=cfg, dtype=dtype).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in
                           payload["best"]["model_state"].items()})
    model.eval()

    unroll = int(cfg["raw_train_unroll"])
    batch = int(S.B_GEN)
    lo, hi = cfg["scored_slice"]

    streams = ST.RunStreams(run, device=dev)
    with torch.no_grad():
        noise = streams.noise("classification", unroll, batch).to(dtype)
        out, _ = model(noise)
        g_in = out[lo:hi].detach().cpu().numpy().astype(np.float64)

    if g_in.shape != (hi - lo, batch, 2):
        raise AssertionError("G_in shape %r" % (g_in.shape,))

    sc = ctx["scales"]
    fam = S.family_arrays(g_in, sc)
    tgt = ctx["target_val_in"]
    ref = ctx["L_ref"]

    losses = {s: S.relative_loss(fam[s], tgt[s]) for s in S.FAMILIES}
    r = {s: losses[s] / ref[s] for s in S.FAMILIES}
    worst_family = max(S.FAMILIES, key=lambda s: r[s])
    r_val = r[worst_family]
    accepted = bool(r_val <= S.CLASSIFICATION_THRESHOLD)

    alpha = payload["best"]["sigma_state"]["alpha"]
    alpha = np.asarray(alpha.detach().cpu().numpy()
                       if torch.is_tensor(alpha) else alpha, np.float64)
    sigma = np.exp(alpha)
    diagnostics = {
        "adaptive_scales_at_checkpoint": {
            s: {"alpha": float(alpha[i]), "sigma": float(sigma[i]),
                "effective_weight": float(1.0 / (2.0 * sigma[i] ** 2))}
            for i, s in enumerate(S.FAMILIES)},
        "proj_per_direction": per_direction_losses(fam["PROJ"], tgt["PROJ"]),
        "note": "diagnostics only; they may not be used to revise the "
                "checkpoint or the accepted mask after the fact",
    }

    return {
        "run": run,
        "checkpoint_iteration": int(payload["best"]["iteration"]),
        "Q_val_at_checkpoint": float(payload["best"]["q_val"]),
        "generation": {
            "stream": "classification",
            "seed": S.SEED_BASE["classification"] + run,
            "B_gen": batch, "raw_unroll": unroll,
            "scored_slice": list(cfg["scored_slice"]),
            "reused_checkpoint_ensemble": False,
        },
        "L_val_in": losses,
        "L_ref_val_in": dict(ref),
        "r_squared_ratio": r,
        "r_unsquared_ratio": {s: float(np.sqrt(r[s])) for s in S.FAMILIES},
        "r_val": float(r_val),
        "r_val_unsquared": float(np.sqrt(r_val)),
        "worst_family": worst_family,
        "threshold": S.CLASSIFICATION_THRESHOLD,
        "threshold_inclusive": True,
        "accepted": accepted,
        "units": "every L_s and every r_s is a SQUARED relative quantity; the "
                 "unsquared discrepancy ratio is sqrt(r_s)",
        "diagnostics": diagnostics,
        "monitoring_L_val_at_checkpoint": payload["best"]["L_val"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="runs_llv")
    ap.add_argument("--out", default="class_llv")
    ap.add_argument("--pattern", default="run_*.pt",
                    help="checkpoint glob inside --runs-dir; use "
                         "'pilot_w030_run_*.pt' for one pilot width")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float64",
                                                           "float32"))
    args = ap.parse_args()

    if not args.archive or not os.path.exists(args.archive):
        raise SystemExit("set ARCHIVE to the processed LLV archive")
    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    loader.assert_unlocked()          # verifies the chain; does not open T_ev
    with open(os.path.join(args.manifest_dir, S.MANIFEST_NAME)) as fh:
        man = json.load(fh)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = args.device
    print("LLV independent validation classifier, spec section 13 stage 2")
    print("device %s  dtype %s" % (dev, args.dtype))

    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)
    fit = S.standardize(loader.fitting(S.TARGET_G_IN), mu, sd)
    scales = S.projection_scales(fit)
    del fit

    print("computing V_in and the nine R_j reference losses", flush=True)
    t0 = time.time()
    v_in = S.standardize(loader.validation(S.TARGET_G_IN), mu, sd)
    tgt = S.family_arrays(v_in, scales)
    del v_in
    per_block = []
    for j in range(1, 10):
        rj = S.standardize(loader.reference(j, S.TARGET_G_IN), mu, sd)
        per_block.append({s: S.relative_loss(a, tgt[s]) for s, a in
                          S.family_arrays(rj, scales).items()})
        del rj
    ref = {s: float(np.median([b[s] for b in per_block])) for s in S.FAMILIES}
    print("  %.0fs" % (time.time() - t0))

    # Cross-check against gate 9, which computed the same quantity.
    gate_ref = man.get("L_ref_val_in") or {}
    drift = {s: abs(ref[s] - gate_ref[s]) / gate_ref[s]
             for s in S.FAMILIES if s in gate_ref}
    if drift and max(drift.values()) > 1e-9:
        raise SystemExit(
            "reference losses differ from gate 9 by up to %.3e (relative): %r"
            % (max(drift.values()), {s: "%.3e" % d for s, d in drift.items()}))
    print("reference losses match gate 9 to %.1e relative"
          % (max(drift.values()) if drift else 0.0))
    for s in S.FAMILIES:
        print("  L_ref[%-5s] %.4e" % (s, ref[s]))

    ctx = {"device": dev, "dtype": dtype, "scales": scales,
           "target_val_in": tgt, "L_ref": ref}

    paths = sorted(glob.glob(os.path.join(args.runs_dir, args.pattern)))
    if not paths:
        raise SystemExit("no run_*.pt in %r" % args.runs_dir)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for p in paths:
        if not re.search(r"run_\d+\.pt$", p):
            continue
        tag = os.path.basename(p)[:-3]
        payload = torch.load(p, map_location="cpu")
        res = classify_one(payload, ctx)
        res["source_checkpoint"] = os.path.abspath(p)
        res["archive_sha256"] = man["archive_sha256"]
        res["manifest_sha256"] = S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME))
        res["specification_sha256"] = man.get("specification_sha256")
        results.append(res)
        res["is_pilot"] = bool((payload["metadata"] or {}).get("pilot"))
        res["pilot"] = (payload["metadata"] or {}).get("pilot")
        res["hidden_units"] = int(payload["metadata"].get(
            "hidden_units", payload["metadata"]["config"]["hidden_units"]))
        with open(os.path.join(args.out,
                               "classification_%s.json" % tag), "w") as fh:
            json.dump(res, fh, indent=2, default=float)

        print("\nrun %d  checkpoint iteration %d  Q_val %.4e"
              % (res["run"], res["checkpoint_iteration"],
                 res["Q_val_at_checkpoint"]))
        print("  %-5s %12s %12s %10s %10s"
              % ("family", "L_val", "L_ref", "r (sq)", "sqrt(r)"))
        for s in S.FAMILIES:
            print("  %-5s %12.4e %12.4e %10.2f %10.2f"
                  % (s, res["L_val_in"][s], ref[s],
                     res["r_squared_ratio"][s], res["r_unsquared_ratio"][s]))
        print("  r_val = %.2f (squared ratio), sqrt = %.2f; worst family %s; "
              "threshold %g -> %s"
              % (res["r_val"], res["r_val_unsquared"], res["worst_family"],
                 res["threshold"],
                 "ACCEPTED" if res["accepted"] else "not successful"))
        d = res["diagnostics"]["proj_per_direction"]
        print("  PROJ per direction, r_s is not defined per direction; "
              "relative squared loss only:")
        for name in PROJ_LABELS:
            print("    %-12s L %.4e  unsquared %.4f"
                  % (name, d[name]["L_squared"], d[name]["unsquared"]))

    mask = [r["accepted"] for r in results]
    summary = {
        "runs": [r["run"] for r in results],
        "accepted_mask": mask,
        "M": int(sum(mask)),
        "n_runs_classified": len(results),
        "pattern": args.pattern,
        "hidden_units": sorted({r["hidden_units"] for r in results}),
        "is_pilot": any(r["is_pilot"] for r in results),
        "r_val": {str(r["run"]): r["r_val"] for r in results},
        "L_ref_val_in": ref,
        "threshold": S.CLASSIFICATION_THRESHOLD,
        "units": "r is a ratio of squared relative discrepancies; the "
                 "unsquared ratio is sqrt(r)",
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "specification_sha256": man.get("specification_sha256"),
        "reads": ["R_j", "V"],
        "never_reads": ["T_ev"],
    }
    stag = re.sub(r"[^A-Za-z0-9]+", "_", args.pattern).strip("_")
    with open(os.path.join(args.out,
                           "classification_summary_%s.json" % stag), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\nM = %d of %d classified" % (summary["M"], len(results)))
    if summary["M"] == 0 and len(results) == S.N_RUNS:
        print("M = 0 over the full population. Spec section 9 requires "
              "stopping: no representative figure and no conditional main "
              "result.")
    print("no final-evaluation quantity was computed; T_ev was not opened.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
