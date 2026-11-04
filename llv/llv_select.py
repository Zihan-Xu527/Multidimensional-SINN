#!/usr/bin/env python3
"""Freeze the reporting population and the representative run, spec section 9.

This step exists so that "selected before T_ev is opened" is a fact on disk
rather than a claim. It reads only the stage-2 classification products and the
stage-1 monitoring histories, writes one JSON plus its sha256, and REFUSES to
overwrite an existing selection. Once T_ev has been opened, any later rerun of
this script fails rather than silently reselecting.

Spec section 9:

  One representative run is selected before T_ev is opened. Sort only the
  validation-successful runs by their stored checkpoint-monitoring Q_val,
  ascending, and select zero-based element `M // 2`, i.e. the one-based rank
  floor(M/2) + 1. If M = 0, stop: no representative figure or conditional main
  result is produced.

  The same representative checkpoint is used for every single-run figure, the
  smooth joint density, and the 10^6-step stability diagnostic. No seed may be
  reselected after viewing final, out-of-window, density, long-lag, or long-run
  results.

usage:
    python3 llv_select.py --class-dir class_formal --runs-dir formal_llv \
        --manifest-dir llv_manifest_v2 --out select_formal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import llv_spec as S

SELECTION_NAME = "representative.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-dir", default="class_formal")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--out", default="select_formal")
    args = ap.parse_args()

    path = os.path.join(args.out, SELECTION_NAME)
    if os.path.exists(path):
        raise SystemExit(
            "a representative selection already exists at %r. Spec section 9 "
            "forbids reselecting after final, out-of-window, density, long-lag "
            "or long-run results have been seen, so this script will not "
            "overwrite it. Delete it deliberately only if T_ev has never been "
            "opened." % path)

    # One classification file per run, from stage 2.
    cls = {}
    for p in sorted(glob.glob(os.path.join(args.class_dir,
                                           "classification_run_*.json"))):
        m = re.search(r"classification_run_(\d+)\.json$", p)
        if not m:
            continue
        with open(p) as fh:
            d = json.load(fh)
        if int(m.group(1)) != int(d["run"]):
            raise SystemExit("file %r disagrees with its run field" % p)
        if d.get("is_pilot"):
            raise SystemExit("%r is a pilot classification; pilot output may "
                             "not enter the reporting population" % p)
        cls[int(d["run"])] = d

    if len(cls) != S.N_RUNS:
        raise SystemExit("expected %d classified runs, found %d: %r"
                         % (S.N_RUNS, len(cls), sorted(cls)))

    # Cross-check every Q_val against the trainer's own history, so the
    # ordering cannot be built on a stale or mismatched classification file.
    for run, d in cls.items():
        hp = os.path.join(args.runs_dir, "run_%d_history.json" % run)
        if not os.path.exists(hp):
            raise SystemExit("missing history %r" % hp)
        with open(hp) as fh:
            h = json.load(fh)
        if int(h["selected_iteration"]) != int(d["checkpoint_iteration"]):
            raise SystemExit("run %d: history iteration %r vs classification "
                             "%r" % (run, h["selected_iteration"],
                                     d["checkpoint_iteration"]))
        if abs(float(h["selected_Q_val"]) - float(d["Q_val_at_checkpoint"])) \
                > 0:
            raise SystemExit("run %d: Q_val differs between history and "
                             "classification" % run)
        # The retained checkpoint must be the history's Q_val minimum.
        q = min(e["Q_val"] for e in h["history"])
        if abs(q - float(h["selected_Q_val"])) > 0:
            raise SystemExit("run %d: retained Q_val is not the history "
                             "minimum" % run)

    runs = sorted(cls)
    mask = [bool(cls[r]["accepted"]) for r in runs]
    m_count = int(sum(mask))
    if m_count == 0:
        raise SystemExit(
            "M = 0. Spec section 9 requires stopping: no representative "
            "figure and no conditional main result is produced.")

    # Sort the successful runs by stored Q_val ascending. Python's sort is
    # stable and the input is in ascending run order, so an exact Q_val tie
    # resolves to the lower run id deterministically; ties are recorded.
    succ = [cls[r] for r in runs if cls[r]["accepted"]]
    ordered = sorted(succ, key=lambda d: float(d["Q_val_at_checkpoint"]))
    qs = [float(d["Q_val_at_checkpoint"]) for d in ordered]
    ties = [[ordered[i]["run"], ordered[i + 1]["run"]]
            for i in range(len(qs) - 1) if qs[i] == qs[i + 1]]

    k = m_count // 2
    rep = ordered[k]

    with open(os.path.join(args.manifest_dir, S.MANIFEST_NAME)) as fh:
        man = json.load(fh)

    out = {
        "stage": "reporting population and representative selection",
        "rule": "sort validation-successful runs by stored "
                "checkpoint-monitoring Q_val ascending; take zero-based "
                "element M // 2 (one-based rank floor(M/2) + 1)",
        "selected_before_T_ev_opened": True,
        "M": m_count,
        "n_runs": len(runs),
        "runs": runs,
        "accepted_mask": mask,
        "threshold": S.CLASSIFICATION_THRESHOLD,
        "threshold_inclusive": True,
        "zero_based_index": k,
        "one_based_rank": k + 1,
        "representative_run": int(rep["run"]),
        "representative_checkpoint_iteration":
            int(rep["checkpoint_iteration"]),
        "representative_Q_val": float(rep["Q_val_at_checkpoint"]),
        "representative_r_val_squared": float(rep["r_val"]),
        "representative_r_val_unsquared": float(rep["r_val_unsquared"]),
        "representative_worst_family": rep["worst_family"],
        "representative_seeds": {
            p: S.SEED_BASE[p] + int(rep["run"]) for p in S.SEED_BASE},
        "ordering": [
            {"index": i, "run": int(d["run"]),
             "Q_val": float(d["Q_val_at_checkpoint"]),
             "r_val_squared": float(d["r_val"]),
             "checkpoint_iteration": int(d["checkpoint_iteration"])}
            for i, d in enumerate(ordered)],
        "q_val_ties": ties,
        "rejected_runs": [
            {"run": int(cls[r]["run"]), "r_val_squared": float(cls[r]["r_val"]),
             "r_val_unsquared": float(cls[r]["r_val_unsquared"]),
             "worst_family": cls[r]["worst_family"]}
            for r in runs if not cls[r]["accepted"]],
        "units": "r is a ratio of squared relative discrepancies; the "
                 "unsquared ratio is sqrt(r). Q_val is a sum of squared "
                 "relative discrepancies and is not normalized by L_ref",
        "same_checkpoint_used_for": [
            "every single-run figure", "the smooth joint density",
            "the 10^6-step stability diagnostic"],
        "no_reselection": "no seed may be reselected after viewing final, "
                          "out-of-window, density, long-lag or long-run "
                          "results",
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "specification_sha256": man.get("specification_sha256"),
        "class_dir": os.path.abspath(args.class_dir),
        "runs_dir": os.path.abspath(args.runs_dir),
        "reads": ["stage-2 classification products", "stage-1 histories"],
        "never_reads": ["T_ev"],
    }

    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    digest = S.sha256_file(path)
    with open(os.path.join(args.out, "representative.sha256"), "w") as fh:
        fh.write(digest + "\n")

    print("reporting population and representative run, spec section 9")
    print("  M = %d of %d   mask %s"
          % (m_count, len(runs), "".join("Y" if a else "N" for a in mask)))
    print("  rejected: " + ", ".join(
        "run %d (r_val %.2f, %s)"
        % (d["run"], d["r_val_squared"], d["worst_family"])
        for d in out["rejected_runs"]))
    print("  zero-based element M // 2 = %d of the %d successful runs"
          % (k, m_count))
    print("  representative = run %d, checkpoint iteration %d, "
          "Q_val %.4e, r_val %.2f (squared; unsquared %.2f)"
          % (rep["run"], rep["checkpoint_iteration"],
             float(rep["Q_val_at_checkpoint"]), float(rep["r_val"]),
             float(rep["r_val_unsquared"])))
    if ties:
        print("  Q_val ties present, resolved by lower run id: %r" % ties)
    if 0 < k < len(ordered) - 1:
        print("  neighbours: index %d run %d Q_val %.4e  |  index %d run %d "
              "Q_val %.4e"
              % (k - 1, ordered[k - 1]["run"],
                 float(ordered[k - 1]["Q_val_at_checkpoint"]),
                 k + 1, ordered[k + 1]["run"],
                 float(ordered[k + 1]["Q_val_at_checkpoint"])))
    print("\nfrozen to %s (sha %s...)" % (path, digest[:12]))
    print("T_ev may be opened only after this file exists; this script "
          "refuses to overwrite it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
