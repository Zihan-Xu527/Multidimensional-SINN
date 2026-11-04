#!/usr/bin/env python3
"""Stage 4 of spec section 13: the final evaluator.

This is the ONLY stage that opens T_ev = [8000, 20000). It refuses to start
unless the gate freeze verifies and the representative selection of spec
section 9 already exists on disk, so the representative can never be chosen
after any final number has been seen.

What it computes, once, for all 20 retained checkpoints irrespective of
validation success (spec section 8):

  one continuous B_gen = 400, 1200-step rollout per run from zero recurrent
  states, state NOT reset at raw step 800;
      raw [0:400)    warm-up, discarded
      raw [400:800)  G_in
      raw [800:1200) G_out
  L_{s,ref}^{eval,w} = median_j L_s(R_{j,w}, T_ev,w)         w in {in, out}
  r_{s,w}            = L_s(G_w, T_ev,w) / L_{s,ref}^{eval,w}
  r_w                = max_s r_{s,w}
  Lbar_{s,ref}^{eval} = (L_{s,ref}^{eval,in} + L_{s,ref}^{eval,out}) / 2
  Delta_s            = [L_s(G_out, T_ev,out) - L_s(G_in, T_ev,in)]
                       / Lbar_{s,ref}^{eval}

plus, for the Supplementary, the 21 sliding windows a = 400, 420, ..., 800 with
slices [a:a+400]; the section 10 long-lag statistic M_SINN against the frozen
eps_lag from stage 3; and the section 11 admissibility rates in each window.

UNITS. Every L_s is the squared relative discrepancy of spec section 4, so
every r and every Delta is built from squared quantities. The unsquared
discrepancy ratio is sqrt(r). Both are reported and never interchanged.

The two endpoint ratios use their own window-specific denominators and describe
accuracy within each window; they do not classify runs. Delta_s is descriptive,
is not a non-inferiority test, and has no pass/fail threshold.

usage:
    ARCHIVE=... SPEC=... python3 llv_final.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --calib-dir calib_llv --out final_llv
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


# --------------------------------------------------------------- generation --
def rollout(payload: dict, run: int, device, dtype):
    """One continuous B_gen x 1200 rollout from zero recurrent states."""
    cfg = payload["metadata"]["config"]
    model = M.LLVGenerator(config=cfg, dtype=dtype).to(device)
    model.load_state_dict({k: v.to(device) for k, v in
                           payload["best"]["model_state"].items()})
    model.eval()
    streams = ST.RunStreams(run, device=device)
    with torch.no_grad():
        noise = streams.noise("final_generation", S.RAW_ROLLOUT,
                              S.B_GEN).to(dtype)
        out, state = model(noise)            # state kept, never reset mid-way
    if tuple(out.shape) != (S.RAW_ROLLOUT, S.B_GEN, 2):
        raise AssertionError("rollout shape %r" % (tuple(out.shape),))
    return out.detach().cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------- helpers --
def median_ref(blocks, target):
    """L_{s,ref} = median_j L_s(R_j, target), per family."""
    per = [{s: S.relative_loss(a[s], target[s]) for s in S.FAMILIES}
           for a in blocks]
    return {s: float(np.median([p[s] for p in per])) for s in S.FAMILIES}, per


def window_slice(arr, lo, hi):
    return np.ascontiguousarray(arr[lo:hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--calib-dir", default="calib_llv")
    ap.add_argument("--out", default="final_llv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float64",
                                                           "float32"))
    ap.add_argument("--skip-sliding", action="store_true",
                    help="omit the 21-window Supplementary diagnostic")
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = args.device

    # ---- preconditions ---------------------------------------------------
    if not args.archive or not os.path.exists(args.archive):
        raise SystemExit("set ARCHIVE to the processed LLV archive")
    sel_path = os.path.join(args.select_dir, "representative.json")
    if not os.path.exists(sel_path):
        raise SystemExit(
            "the representative selection %r does not exist. Spec section 9 "
            "requires it to be frozen BEFORE T_ev is opened; run llv_select.py "
            "first." % sel_path)
    with open(sel_path) as fh:
        sel = json.load(fh)
    sel_sha = S.sha256_file(sel_path)
    recorded = os.path.join(args.select_dir, "representative.sha256")
    if os.path.exists(recorded):
        want = open(recorded).read().strip()
        if want != sel_sha:
            raise SystemExit("the representative selection has changed since "
                             "it was frozen (%s... now %s...)"
                             % (want[:12], sel_sha[:12]))
    calib_path = os.path.join(args.calib_dir, "calibration.json")
    if not os.path.exists(calib_path):
        raise SystemExit("stage 3 calibration %r is missing" % calib_path)
    with open(calib_path) as fh:
        calib = json.load(fh)
    eps_lag = float(calib["long_lag_envelope"]["eps_lag"])

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    for key, got in (("archive_sha256", calib["archive_sha256"]),
                     ("archive_sha256", sel["archive_sha256"])):
        if man[key] != got:
            raise SystemExit("archive sha mismatch between manifest and an "
                             "earlier stage")

    os.makedirs(args.out, exist_ok=True)
    print("LLV final evaluator, spec section 13 stage 4")
    print("device %s  dtype %s" % (dev, args.dtype))
    print("representative run %d, checkpoint iteration %d (frozen, sha %s...)"
          % (sel["representative_run"],
             sel["representative_checkpoint_iteration"], sel_sha[:12]))
    print("M = %d of %d; accepted mask %s"
          % (sel["M"], sel["n_runs"],
             "".join("Y" if a else "N" for a in sel["accepted_mask"])))
    print("eps_lag = %.6f (frozen in stage 3)" % eps_lag)

    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)
    fit = S.standardize(loader.fitting(S.TARGET_G_IN), mu, sd)
    scales = S.projection_scales(fit)
    del fit

    # ---- the final target, opened here and only here ---------------------
    print("\nopening T_ev = [%d, %d)" % S.SPLIT_T_EV, flush=True)
    t0 = time.time()
    tev = S.standardize(loader.open_final(S.TARGET_COMBINED), mu, sd)
    print("  T_ev combined %s  (%.0fs)" % (tev.shape, time.time() - t0))

    lo_in = S.TARGET_G_IN[0] - S.TARGET_COMBINED[0]
    lo_out = S.TARGET_G_OUT[0] - S.TARGET_COMBINED[0]
    tev_in = window_slice(tev, lo_in, lo_in + S.T_WIN)
    tev_out = window_slice(tev, lo_out, lo_out + S.T_WIN)

    print("computing target family statistics", flush=True)
    t0 = time.time()
    tgt = {"in": S.family_arrays(tev_in, scales),
           "out": S.family_arrays(tev_out, scales)}
    print("  %.0fs" % (time.time() - t0), flush=True)

    # ---- the nine reference blocks, in both windows ----------------------
    print("computing the nine reference blocks", flush=True)
    t0 = time.time()
    ref_arr = {"in": [], "out": []}
    ref_comb = []
    for j in range(1, 10):
        rj = S.standardize(loader.reference(j, S.TARGET_COMBINED), mu, sd)
        ref_arr["in"].append(S.family_arrays(
            window_slice(rj, lo_in, lo_in + S.T_WIN), scales))
        ref_arr["out"].append(S.family_arrays(
            window_slice(rj, lo_out, lo_out + S.T_WIN), scales))
        ref_comb.append(rj)
    l_ref, ref_per_block = {}, {}
    for w in ("in", "out"):
        l_ref[w], ref_per_block[w] = median_ref(ref_arr[w], tgt[w])
    l_bar = {s: 0.5 * (l_ref["in"][s] + l_ref["out"][s]) for s in S.FAMILIES}
    print("  %.0fs" % (time.time() - t0))
    print("  %-5s %12s %12s %12s" % ("family", "L_ref in", "L_ref out",
                                     "common Lbar"))
    for s in S.FAMILIES:
        print("  %-5s %12.4e %12.4e %12.4e"
              % (s, l_ref["in"][s], l_ref["out"][s], l_bar[s]))

    # ---- long-lag target curve ------------------------------------------
    curve_tev = S.corr_paircount(tev, S.LONGLAG_K_LO, S.LONGLAG_K_HI)

    # ---- sliding windows: targets and references ------------------------
    slide = None
    if not args.skip_sliding:
        print("\ncomputing %d sliding-window targets and references"
              % len(S.SLIDING_STARTS), flush=True)
        t0 = time.time()
        slide = {"starts": list(S.SLIDING_STARTS), "target": {}, "L_ref": {},
                 "block_ratios": {}}
        for a in S.SLIDING_STARTS:
            off = a - S.TARGET_COMBINED[0]
            ta = S.family_arrays(window_slice(tev, off, off + S.T_WIN), scales)
            blocks = [S.family_arrays(window_slice(rj, off, off + S.T_WIN),
                                      scales) for rj in ref_comb]
            ref_a, per = median_ref(blocks, ta)
            slide["target"][a] = ta
            slide["L_ref"][a] = ref_a
            slide["block_ratios"][a] = {
                s: [p[s] / ref_a[s] for p in per] for s in S.FAMILIES}
            print("  a = %d  %.0fs" % (a, time.time() - t0), flush=True)
    del tev

    # ---- per-run evaluation ---------------------------------------------
    paths = sorted(p for p in glob.glob(os.path.join(args.runs_dir,
                                                     "run_*.pt"))
                   if re.search(r"run_\d+\.pt$", p))
    if len(paths) != S.N_RUNS:
        raise SystemExit("expected %d checkpoints in %r, found %d"
                         % (S.N_RUNS, args.runs_dir, len(paths)))
    accepted = {r: a for r, a in zip(sel["runs"], sel["accepted_mask"])}

    results = []
    for p in paths:
        payload = torch.load(p, map_location="cpu")
        run = int(payload["run"])
        t0 = time.time()
        raw = rollout(payload, run, dev, dtype)
        win = S.extract_windows(raw)

        fam = {"in": S.family_arrays(win["G_in"], scales),
               "out": S.family_arrays(win["G_out"], scales)}
        loss = {w: {s: S.relative_loss(fam[w][s], tgt[w][s])
                    for s in S.FAMILIES} for w in ("in", "out")}
        r = {w: {s: loss[w][s] / l_ref[w][s] for s in S.FAMILIES}
             for w in ("in", "out")}
        r_w = {w: max(r[w].values()) for w in ("in", "out")}
        delta = {s: (loss["out"][s] - loss["in"][s]) / l_bar[s]
                 for s in S.FAMILIES}

        curve = S.corr_paircount(win["combined"], S.LONGLAG_K_LO,
                                 S.LONGLAG_K_HI)
        dif = np.abs(curve - curve_tev)
        flat = int(np.argmax(dif))
        m_sinn = float(dif.reshape(-1)[flat])

        adm = {}
        for w, key in (("in", "G_in"), ("out", "G_out")):
            rho = S.inverse_standardize(win[key], mu, sd)
            adm[w] = S.admissibility(rho)

        row = {
            "run": run,
            "accepted": bool(accepted[run]),
            "checkpoint_iteration": int(payload["best"]["iteration"]),
            "Q_val_at_checkpoint": float(payload["best"]["q_val"]),
            "final_generation_seed": S.SEED_BASE["final_generation"] + run,
            "B_gen": S.B_GEN,
            "raw_rollout": S.RAW_ROLLOUT,
            "state_reset_at_step_800": False,
            "L_eval": loss,
            "r_eval_squared": r,
            "r_eval_unsquared": {w: {s: float(np.sqrt(r[w][s]))
                                     for s in S.FAMILIES}
                                 for w in ("in", "out")},
            "r_worst_squared": r_w,
            "r_worst_unsquared": {w: float(np.sqrt(r_w[w]))
                                  for w in ("in", "out")},
            "Delta": delta,
            "M_SINN": m_sinn,
            "M_SINN_over_eps_lag": m_sinn / eps_lag,
            "M_SINN_argmax": {"curve": S.LONGLAG_CURVES[
                flat // (S.LONGLAG_K_HI - S.LONGLAG_K_LO + 1)],
                "lag": S.LONGLAG_K_LO + flat % (
                    S.LONGLAG_K_HI - S.LONGLAG_K_LO + 1)},
            "admissibility": adm,
            "wall_seconds": time.time() - t0,
        }

        if slide is not None:
            sl = {}
            for a in S.SLIDING_STARTS:
                # The generated rollout is the full 1200 raw steps, so the
                # spec slice [a:a+400] applies to it directly. The target and
                # the reference blocks were sliced out of TARGET_COMBINED,
                # which starts at 400, hence the a - 400 offset used above.
                ga = S.family_arrays(window_slice(raw, a, a + S.T_WIN), scales)
                sl[a] = {s: S.relative_loss(ga[s], slide["target"][a][s])
                         / slide["L_ref"][a][s] for s in S.FAMILIES}
            row["sliding_r_squared"] = sl

        results.append(row)
        print("run %3d  %s  r_in %7.2f  r_out %7.2f  M_SINN/eps %6.3f  "
              "invalid in %.2e out %.2e  %.0fs"
              % (run, "acc" if row["accepted"] else "rej",
                 r_w["in"], r_w["out"], row["M_SINN_over_eps_lag"],
                 adm["in"]["invalid_rate"], adm["out"]["invalid_rate"],
                 row["wall_seconds"]), flush=True)
        with open(os.path.join(args.out, "final_run_%d.json" % run), "w") as fh:
            json.dump(row, fh, indent=2, default=float)

    # ---- main-text summary, conditioned on the successful runs ----------
    acc = [x for x in results if x["accepted"]]

    def q(vals):
        v = np.asarray(vals, np.float64)
        return {"median": float(np.median(v)),
                "q1": float(np.quantile(v, 0.25)),
                "q3": float(np.quantile(v, 0.75))}

    table = {}
    for s in S.FAMILIES:
        d = [x["Delta"][s] for x in acc]
        table[s] = {
            "r_eval_in": q([x["r_eval_squared"]["in"][s] for x in acc]),
            "r_eval_out": q([x["r_eval_squared"]["out"][s] for x in acc]),
            "Delta": q(d),
            "n_plus": int(sum(1 for v in d if v > 0)),
            "M": len(acc),
        }

    longlag = {
        "eps_lag": eps_lag,
        "ratio": q([x["M_SINN_over_eps_lag"] for x in acc]),
        "n_lag": int(sum(1 for x in acc
                         if x["M_SINN"] <= eps_lag)),
        "M": len(acc),
    }
    worst = {w: q([x["r_worst_squared"][w] for x in acc])
             for w in ("in", "out")}
    adm_sum = {
        w: {k: q([x["admissibility"][w][k] for x in acc])
            for k in ("nonfinite_rate", "constraint_rate", "invalid_rate",
                      "max_exceedance")}
        for w in ("in", "out")}
    adm_sum["n_runs_with_any_invalid"] = int(sum(
        1 for x in acc if x["admissibility"]["in"]["invalid_rate"] > 0
        or x["admissibility"]["out"]["invalid_rate"] > 0))
    adm_sum["paired_change_invalid_rate"] = q([
        x["admissibility"]["out"]["invalid_rate"]
        - x["admissibility"]["in"]["invalid_rate"] for x in acc])

    print("\nmain-text endpoint table, conditioned on the %d successful runs"
          % len(acc))
    print("  %-5s %22s %22s %22s %6s"
          % ("family", "r_in med [Q1,Q3]", "r_out med [Q1,Q3]",
             "Delta med [Q1,Q3]", "n+/M"))
    for s in S.FAMILIES:
        t = table[s]
        print("  %-5s %7.2f [%5.2f,%5.2f] %7.2f [%5.2f,%5.2f] "
              "%+7.2f [%+5.2f,%+5.2f] %3d/%d"
              % (s, t["r_eval_in"]["median"], t["r_eval_in"]["q1"],
                 t["r_eval_in"]["q3"], t["r_eval_out"]["median"],
                 t["r_eval_out"]["q1"], t["r_eval_out"]["q3"],
                 t["Delta"]["median"], t["Delta"]["q1"], t["Delta"]["q3"],
                 t["n_plus"], t["M"]))
    print("  run-wise worst-family r: in %.2f [%.2f,%.2f]   "
          "out %.2f [%.2f,%.2f]  (these are not maxima of the medians above)"
          % (worst["in"]["median"], worst["in"]["q1"], worst["in"]["q3"],
             worst["out"]["median"], worst["out"]["q1"], worst["out"]["q3"]))
    print("  long-lag M_SINN/eps_lag %.3f [%.3f,%.3f];  n_lag = %d of %d"
          % (longlag["ratio"]["median"], longlag["ratio"]["q1"],
             longlag["ratio"]["q3"], longlag["n_lag"], longlag["M"]))
    print("  admissibility invalid_rate  in %.3e  out %.3e  "
          "(median); runs with any invalid state: %d of %d"
          % (adm_sum["in"]["invalid_rate"]["median"],
             adm_sum["out"]["invalid_rate"]["median"],
             adm_sum["n_runs_with_any_invalid"], len(acc)))

    out = {
        "stage": "final evaluator",
        "specification": man.get("specification"),
        "specification_sha256": man.get("specification_sha256"),
        "archive_path": args.archive,
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "representative_run": sel["representative_run"],
        "calibration_sha256": S.sha256_file(calib_path),
        "eps_lag": eps_lag,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_gates.py",
                           "llv_torch.py", "llv_model.py", "llv_final.py",
                           "llv_calibrate.py") if os.path.exists(n)},
        "windows": {"G_in": list(S.SLICE_G_IN), "G_out": list(S.SLICE_G_OUT),
                    "combined": list(S.SLICE_COMBINED),
                    "raw_rollout": S.RAW_ROLLOUT,
                    "target_in": list(S.TARGET_G_IN),
                    "target_out": list(S.TARGET_G_OUT)},
        "target": {"set": "T_ev", "indices": list(S.SPLIT_T_EV),
                   "n_trajectories": S.SPLIT_T_EV[1] - S.SPLIT_T_EV[0]},
        "L_ref_eval": l_ref,
        "L_ref_eval_per_block": ref_per_block,
        "L_bar_ref_eval": l_bar,
        "accepted_mask": sel["accepted_mask"],
        "M": sel["M"],
        "runs": results,
        "endpoint_table": table,
        "run_wise_worst_family_r": worst,
        "long_lag": longlag,
        "admissibility_summary": adm_sum,
        "sliding_windows": (None if slide is None else {
            "starts": slide["starts"],
            "L_ref": {str(a): slide["L_ref"][a] for a in slide["starts"]},
            "block_ratios": {str(a): slide["block_ratios"][a]
                             for a in slide["starts"]},
        }),
        "units": "every L_s, r and Delta is built from SQUARED relative "
                 "discrepancies; the unsquared ratio is sqrt(r)",
        "reporting_symbol": {
            "ensemble_size": S.B_GEN,
            "prose_symbol": "N",
            "code_field": "B_gen",
            "visualization_ensemble_size": S.B_VIS,
            "note": "the main text and the Supplementary write N = 400 and "
                    "N_vis = 12000. B_gen and B_VIS are internal code field "
                    "names only; the symbol B must not appear in prose, "
                    "tables, figures or captions."},
        "notes": [
            "all 20 checkpoints were evaluated once, irrespective of "
            "validation success",
            "the two endpoint ratios use window-specific denominators and do "
            "not classify runs",
            "Delta_s is descriptive, is not a non-inferiority test and has no "
            "pass/fail threshold",
            "the long-lag statistic and the sliding-window figure are "
            "descriptive and do not alter the accepted mask",
            "no checkpoint was reselected and no run was reclassified here",
        ],
        "torch_version": torch.__version__,
        "device": dev,
        "dtype": args.dtype,
    }
    path = os.path.join(args.out, "final_evaluation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    digest = S.sha256_file(path)
    with open(os.path.join(args.out, "final_evaluation.sha256"), "w") as fh:
        fh.write(digest + "\n")
    print("\nwrote %s (sha %s...)" % (path, digest[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
