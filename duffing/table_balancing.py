#!/usr/bin/env python3
"""Export the S1 late-training loss/success table from saved run records.

Each run contributes the median of its final five recorded
``train_unweighted`` values; the table reports the median over all runs.
Success uses the retained checkpoint's squared family ratios, not the old
``best.success`` boolean and not any-step threshold attainment.

No model is loaded, sampled, trained or re-evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
FAMILIES = {"ACF", "CCF", "ACF2", "PDF", "PROJ"}
RUN_NAME = re.compile(
    r"run_nf3_lb(on|off)_din2_L2H10_lam([0-9p]+)_s([0-9]+)\.json$"
)


def retained_ratio(best, lam):
    """Convert the stored unsquared ratios once, then take the maximum."""
    expected = FAMILIES - ({"CCF"} if lam == 0 else set())
    squared = {
        key: float(value) ** 2
        for key, value in best["ratios"].items()
        if key in FAMILIES and np.isfinite(float(value))
    }
    if set(squared) != expected:
        raise ValueError(f"lambda={lam:g}: unexpected applicable families")
    worst = max(squared.values())
    if not np.isclose(worst, float(best["worst_ratio"]) ** 2,
                      rtol=1e-12, atol=0):
        raise ValueError("stored worst ratio disagrees with the family ratios")
    return worst


def collect(rundir, threshold):
    cells = {(lam, lb): [] for lam in LAMBDAS for lb in ("off", "on")}
    for path in sorted(rundir.glob("run_*.json")):
        match = RUN_NAME.fullmatch(path.name)
        if not match:
            continue
        lb, tag, seed = match.groups()
        lam = float(tag.replace("p", "."))
        if lam not in LAMBDAS:
            continue
        record = json.loads(path.read_text())
        identity = (int(record["n_f"]), int(record["d_in"]),
                    int(record["layers"]), int(record["hidden"]),
                    float(record["lambda"]), bool(record["loss_balancing"]),
                    int(record["seed_init"]))
        if identity != (3, 2, 2, 10, lam, lb == "on", int(seed)):
            raise ValueError(f"filename/configuration mismatch: {path}")
        hist = path.with_name(path.name.replace("run_", "hist_", 1))
        hist = hist.with_suffix(".npz")
        with np.load(hist, allow_pickle=False) as saved:
            train = np.asarray(saved["train_unweighted"], dtype=float)
            iterations = np.asarray(saved["iter"])
        if train.ndim != 1 or train.size < 5 or train.shape != iterations.shape:
            raise ValueError(f"invalid recorded training history: {hist}")
        if not np.isfinite(train).all():
            raise ValueError(f"non-finite recorded training loss: {hist}")
        ratio = retained_ratio(record["best"], lam)
        cells[(lam, lb)].append({
            "lambda": lam, "balancing": lb, "seed": int(seed),
            "late_training_loss": float(np.median(train[-5:])),
            "retained_squared_ratio": ratio,
            "successful": ratio <= threshold,
            "tail_iterations": iterations[-5:].tolist(),
            "run_file": path.name, "history_file": hist.name,
        })
    for (lam, lb), runs in cells.items():
        expected_n = 20 if lam == 4 else 10
        if sorted(r["seed"] for r in runs) != list(range(expected_n)):
            raise ValueError(f"lambda={lam:g}, balancing={lb}: incomplete run set")
    return cells


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rundir", type=Path, default=Path("runs/grid"))
    parser.add_argument("--outdir", type=Path, default=Path("tables"))
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be positive and finite")
    cells = collect(args.rundir, args.threshold)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    per_run = []
    for (lam, lb), runs in cells.items():
        summary.append({
            "lambda": lam, "balancing": lb, "n_runs": len(runs),
            "median_late_training_loss": float(np.median(
                [r["late_training_loss"] for r in runs])),
            "successes": sum(r["successful"] for r in runs),
            "squared_ratio_threshold": args.threshold,
        })
        per_run.extend(runs)
    write_csv(args.outdir / "table_loss_balancing.csv", summary, list(summary[0]))
    write_csv(args.outdir / "table_loss_balancing_runs.csv", per_run,
              list(per_run[0]))

    lookup = {(r["lambda"], r["balancing"]): r for r in summary}
    lines = [
        "% Saved-history summary; no model generation or re-evaluation.",
        r"\begin{tabular}{lcc@{\hspace{1.5em}}cc}", r"\hline",
        r"& \multicolumn{2}{c}{balancing off} & \multicolumn{2}{c}{balancing on} \\",
        r"\cline{2-3} \cline{4-5}",
        r"$\lambda$ & loss & successes & loss & successes \\", r"\hline",
    ]
    for lam in LAMBDAS:
        values = [f"{lam:g}"]
        for lb in ("off", "on"):
            row = lookup[(lam, lb)]
            values += [scientific(row["median_late_training_loss"]),
                       f"{row['successes']}/{row['n_runs']}"]
        lines.append(" & ".join(values) + r" \\")
    totals = {lb: (sum(r["successes"] for r in summary if r["balancing"] == lb),
                   sum(r["n_runs"] for r in summary if r["balancing"] == lb))
              for lb in ("off", "on")}
    off, on = totals["off"], totals["on"]
    lines += [r"\hline", f"Total & & {off[0]}/{off[1]} & & {on[0]}/{on[1]}"
              + r" \\", r"\hline", r"\end{tabular}"]
    (args.outdir / "table_loss_balancing.tex").write_text("\n".join(lines) + "\n")
    print(f"Read {len(per_run)} saved runs; threshold {args.threshold:g}")
    print(f"Validation successes: off {off[0]}/{off[1]}, on {on[0]}/{on[1]}")
    print(f"Wrote summary CSV, per-run CSV and LaTeX under {args.outdir}")


if __name__ == "__main__":
    main()
