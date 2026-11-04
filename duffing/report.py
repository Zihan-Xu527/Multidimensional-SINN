#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final evaluation: retained checkpoint -> generated ensemble -> errors against T.

Everything the grid produced is VALIDATION-side. Section 3.6 reports against the
evaluation segment, and nothing has ever been measured there. This script is the
only place that opens it.

WHAT IT PRODUCES, per configuration:

    family      e_mod        e_ref^eval    ratio
    ACF         0.0xxx       0.0xxx        x.xx
    ...

  * e_mod       the model error of Eq. (21) against T, median over seeds
  * e_ref^eval  the reference-ensemble error against T: the median over the nine
                blocks of R. Computed once per lambda, independent of the model.
  * ratio       their quotient, on the scale Section 3.6 asks the reader to use.

THREE THINGS THIS SCRIPT IS CAREFUL ABOUT.

1. A THIRD NOISE STREAM. Training uses seed+10000, validation seed+20000. The
   reported ensemble uses seed+30000, so the numbers below are not measured on
   the same noise the checkpoint was selected with. Reusing the validation
   stream would report the ensemble that won the selection, which is a mild but
   real optimism.

2. MATCHED SIZE. --n-gen defaults to B, the fitting budget, because the
   denominator is the error of a B-trajectory REAL sample. An ensemble larger
   than B is smoother for reasons that have nothing to do with the model, and
   its ratio is depressed by roughly sqrt(B/n_gen). --n-gen-precise additionally
   reports a high-precision point estimate from a large ensemble; it is printed
   in its own block, labelled, and carries no pass/fail.

3. THE DISPLAYED RUN IS CHOSEN ON VALIDATION. The figure run is the one whose
   VALIDATION aggregate is the median over the runs that succeed on validation.
   Choosing it by its evaluation error would be selecting on the reported
   quantity -- the one real weakness of the Section 5.1 runs, and the reason the
   protocol says every decision uses V alone.

Usage
-----
    python report.py --lam 4 --nf 3 --lb on --din 2 --tau 3
    python report.py --lam 4 --nf 3 --lb on --din 2 --tau 3 --n-gen-precise 12000
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import evaluate as EV
from train_sinn import SINN, white_noise

# Quantities computed by evaluate.errors() that never enter any training loss.
OUT_OF_OBJECTIVE = ("JOINT", "PROJ_HOLDOUT", "kappa_1122_over_var1var2")


def cfg_tag(nf, lb, din, layers, hidden, lam):
    tag = f"lam{lam:g}".replace(".", "p")
    return f"nf{nf}_lb{lb}_din{din}_L{layers}H{hidden}_{tag}"


def generate(ckpt_path, ref, n_gen, din, layers, hidden, warmup, seed, device):
    """Regenerate an ensemble from a saved checkpoint, on its own noise stream."""
    ck = torch.load(ckpt_path, map_location=device)
    if str(ck.get("constants_sha256", "")) != ref["constants_sha256"]:
        raise SystemExit(
            f"{os.path.basename(ckpt_path)} was trained against constants "
            f"{str(ck.get('constants_sha256',''))[:12]}..., the reference here is "
            f"{ref['constants_sha256'][:12]}.... Refusing to score across them.")
    net = SINN(hidden, layers).to(device=device, dtype=torch.float32)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    l_seq = int(ref["l_seq"])
    g = torch.Generator(device=device)
    g.manual_seed(seed + 30_000)          # third stream; see the header
    with torch.no_grad():
        x = white_noise(l_seq + warmup, n_gen, din, g, device, torch.float32)
        y, _ = net(x)
        arr = y[-l_seq:].detach().cpu().numpy().astype(np.float64)
    return arr, int(ck["iter"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--rundir", default="runs/grid")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--nf", type=int, default=3)
    ap.add_argument("--lb", choices=("on", "off"), default="on")
    ap.add_argument("--din", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=None,
                    help="default: B from the reference, i.e. matched size")
    ap.add_argument("--n-gen-precise", type=int, default=0,
                    help="extra high-precision point estimate; no pass/fail")
    ap.add_argument("--tau", type=float, required=True)
    # WHICH ENSEMBLE THE ERRORS ARE MEASURED AGAINST. The three correlation
    # families need a real ensemble to compare with; the density families are
    # compared with the analytic reference either way and are untouched by this
    # switch. "evaluation" is the large held-out segment; "validation" is the
    # segment the checkpoint was selected on, which is the right choice once the
    # evaluation segment is no longer part of the protocol. The two give
    # different raw errors -- the anchor's own sampling noise enters them -- but
    # nearly the same ratios, since numerator and denominator move together.
    # NEVER MIX THEM IN ONE TABLE. The anchor is recorded in the json and
    # printed at the top of every report so a number can be traced to it.
    ap.add_argument("--against", choices=("evaluation", "validation"),
                    default="evaluation",
                    help="ensemble the correlation families are scored against")
    ap.add_argument("--save-display", action="store_true",
                    help="write the displayed run's ensemble for the figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # THE DEVICE IS PART OF THE RESULT, not an implementation detail. A CPU LSTM
    # and a cuDNN LSTM do not agree bitwise, so the generated ensemble differs
    # and the KDE errors move by a percent or two -- enough to make two tables
    # disagree while every input is identical. It is recorded here and printed,
    # so a number can always be traced to the machine that produced it. Every
    # reported table must come from one device; the grid trained on A100s, so
    # that is the one.
    dev = str(args.device)
    gpu = (torch.cuda.get_device_name(0)
           if dev.startswith("cuda") and torch.cuda.is_available() else None)
    if not dev.startswith("cuda"):
        print("WARNING: generating on the CPU. The reported errors will differ "
              "from GPU-generated ones at the percent level. Do not mix them in "
              "one table.")

    tag = f"lam{args.lam:g}".replace(".", "p")
    traj_path = os.path.join(args.datadir, f"traj_{tag}.npy")
    ref = EV.load_reference(args.datadir, tag, traj_path)
    traj = np.load(traj_path, mmap_mode="r")
    split = EV.split_from_reference(ref)
    deg = EV.ccf_is_degenerate(ref)
    B = int(ref["b_fit"])
    n_gen = int(args.n_gen) if args.n_gen else B
    os.makedirs(args.outdir, exist_ok=True)

    # ---- the yardstick, computed once, model-independent --------------------
    floor_eval, floor_rows, sr_eval = EV.floor(traj, ref, split, args.against)
    print(f"lambda={args.lam:g}   B={B}   n_gen={n_gen}   tau={args.tau:g}")
    print(f"anchor: {args.against} segment {split[args.against]}   "
          f"(density families are scored against the analytic reference at "
          f"either setting)")
    if args.against == "validation":
        print("  NOTE: the checkpoint was selected on this segment, so the "
              "correlation-family\n        errors reported here are not "
              "out-of-sample in the strict sense.")
    if deg:
        print("  CCF degenerate: reported as absolute error")

    # ---- the runs of this configuration -------------------------------------
    ctag = cfg_tag(args.nf, args.lb, args.din, args.layers, args.hidden, args.lam)
    ckpts = sorted(glob.glob(os.path.join(args.rundir, f"net_{ctag}_s*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching net_{ctag}_s*.pt in {args.rundir}")

    fams = [f for f in EV.FAMILIES if not (f == "CCF" and deg)]
    per_seed = []
    for ck in ckpts:
        run_tag = os.path.basename(ck)[len("net_"):-len(".pt")]
        seed = int(run_tag.rsplit("_s", 1)[1])
        rec_path = os.path.join(args.rundir, f"run_{run_tag}.json")
        rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}

        arr, it = generate(ck, ref, n_gen, args.din, args.layers, args.hidden,
                           args.warmup, seed, args.device)
        e = EV.errors(EV.stats(arr, ref, "standardized"), sr_eval, deg)
        cls = EV.classify(e, floor_eval, args.tau)
        per_seed.append({
            "run_tag": run_tag, "seed": seed, "iter": it,
            "errors": {k: float(e[k]) for k in fams},
            # OUT-OF-OBJECTIVE DIAGNOSTICS. None of these enters any loss, which
            # is what makes them evidence rather than bookkeeping. PROJ improving
            # when n_f rises shows only that a term was added to the objective
            # and then minimised; the joint density, the held-out directions and
            # the fourth-order joint cumulant are what actually test whether the
            # dependence structure was captured.
            "diagnostics": {k: float(e[k]) for k in OUT_OF_OBJECTIVE},
            "ratios": {k: float(v) for k, v in cls["ratios"].items()},
            "worst_ratio": float(cls["worst_ratio"]),
            "success_eval": bool(cls["success"]),
            # validation side, carried over from the training record
            "val_agg": float(rec.get("final_val_agg", float("nan"))),
            "val_best_agg": float(rec.get("best", {}).get("val_agg", float("nan"))),
            "val_worst_ratio": float(rec.get("best", {}).get("worst_ratio", float("nan"))),
        })
        if deg:
            per_seed[-1]["errors"]["CCF_abs"] = float(e["CCF_abs"])

    # ---- the three-column table, medians over seeds --------------------------
    print(f"\n{'family':<10}{'e_mod':<12}{'e_ref^eval':<14}{'ratio':<8}")
    table = {}
    for k in fams:
        em = float(np.median([p["errors"][k] for p in per_seed]))
        er = float(floor_eval[k])
        table[k] = {"e_mod": em, "e_ref_eval": er, "ratio": em / er}
        print(f"{k:<10}{em:<12.4f}{er:<14.4f}{em/er:<8.2f}")
    if deg:
        em = float(np.median([p["errors"]["CCF_abs"] for p in per_seed]))
        print(f"{'CCF (abs)':<10}{em:<12.4f}{float(floor_eval['CCF_abs']):<14.4f}"
              f"{'--':<8}  excluded from the criterion")

    # ---- the out-of-objective block, kept visually separate ------------------
    print(f"\n{'out-of-objective':<20}{'model':<12}{'reference':<12}{'ratio':<8}")
    diag = {}
    for k in ("JOINT", "PROJ_HOLDOUT"):
        m = float(np.median([p["diagnostics"][k] for p in per_seed]))
        r = float(floor_eval[k])
        diag[k] = {"model": m, "reference": r, "ratio": m / r}
        print(f"{k:<20}{m:<12.4f}{r:<12.4f}{m/r:<8.2f}")
    kt = float(sr_eval["kappa_1122_over_var1var2"])
    km = float(np.median([p["diagnostics"]["kappa_1122_over_var1var2"]
                          for p in per_seed]))
    diag["kappa_1122_over_var1var2"] = {"model": km, "target": kt}
    print(f"{'kappa_(2,2) norm':<20}{km:<12.4f}{kt:<12.4f}"
          f"{'--':<8}  target from the evaluation ensemble")

    ws = sorted(p["worst_ratio"] for p in per_seed)
    n_ok = sum(p["success_eval"] for p in per_seed)
    print(f"\nseeds {len(per_seed)}   worst-family ratio: median {ws[len(ws)//2]:.3f}"
          f"   min {ws[0]:.3f}   max {ws[-1]:.3f}")
    print(f"successful on EVALUATION at tau={args.tau:g}: {n_ok}/{len(per_seed)}"
          "   (the classification of record is the validation one)")

    # ---- the displayed run, chosen on validation -----------------------------
    # THE RULE, without a conditional: the run whose validation aggregate is the
    # median over ALL of this configuration's runs.
    #
    # It used to take the median over the validation-SUCCESSFUL runs, which has
    # no value for a configuration where nothing succeeds -- and one of those,
    # n_f = 0, is precisely the panel a figure about the joint density has to
    # show. A rule with a fallback branch is two rules; this is one, it applies
    # to every panel alike, and it still uses validation alone. The success
    # count is printed beside it so a caption can state it.
    ranked = sorted(per_seed, key=lambda p: p["val_best_agg"])
    display = ranked[len(ranked) // 2]
    n_ok_val = sum(1 for p in per_seed
                   if np.isfinite(p["val_worst_ratio"])
                   and p["val_worst_ratio"] <= args.tau)
    print(f"\ndisplayed run (median validation aggregate over all "
          f"{len(per_seed)} seeds; {n_ok_val} of them succeed on validation): "
          f"{display['run_tag']}")

    if args.save_display:
        arr, _ = generate(os.path.join(args.rundir, f"net_{display['run_tag']}.pt"),
                          ref, n_gen, args.din, args.layers, args.hidden,
                          args.warmup, display["seed"], args.device)
        gpath = os.path.join(args.outdir, f"gen_{display['run_tag']}.npz")
        EV.save_generated(gpath, arr, ref)
        print(f"wrote {gpath}")

    # ---- optional high-precision point estimate ------------------------------
    precise = None
    if args.n_gen_precise:
        big = int(args.n_gen_precise)
        rows = []
        for p in per_seed:
            arr, _ = generate(os.path.join(args.rundir, f"net_{p['run_tag']}.pt"),
                              ref, big, args.din, args.layers, args.hidden,
                              args.warmup, p["seed"], args.device)
            e = EV.errors(EV.stats(arr, ref, "standardized"), sr_eval, deg)
            rows.append({k: float(e[k]) for k in fams})
        precise = {k: float(np.median([r[k] for r in rows])) for k in fams}
        print(f"\nhigh-precision point estimates, n_gen={big} "
              f"(NOT comparable to e_ref^eval; no pass/fail):")
        for k in fams:
            print(f"  {k:<10}{precise[k]:.4f}")

    out = {
        "lambda": args.lam, "config": ctag, "tau": args.tau,
        "b_fit": B, "n_gen": n_gen, "n_seeds": len(per_seed),
        "constants_sha256": ref["constants_sha256"],
        "device": dev, "gpu_name": gpu, "torch": torch.__version__,
        "anchor": args.against,
        "anchor_segment": [int(v) for v in split[args.against]],
        "ccf_degenerate": bool(deg),
        "floor_eval": {k: float(v) for k, v in floor_eval.items()
                       if k != "ccf_degenerate"},
        "floor_blocks": [{k: float(v) for k, v in r.items()
                          if k != "ccf_degenerate"} for r in floor_rows],
        "table": table,
        "out_of_objective": diag,
        "per_seed": per_seed,
        "display_run": display["run_tag"],
        "display_rule": "median validation aggregate over all runs of the cell",
        "n_success_validation": n_ok_val,
        "precise": precise, "n_gen_precise": int(args.n_gen_precise),
    }
    opath = os.path.join(args.outdir, f"report_{ctag}.json")
    with open(opath, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {opath}")


if __name__ == "__main__":
    main()
