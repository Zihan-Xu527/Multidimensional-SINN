#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The training driver for the Section 4 sweep: one run = one (n_f, LB, d_in,
capacity, lambda, seed) cell, one JSON line, one checkpoint, one history file.

WHAT THIS DRIVER GUARANTEES
---------------------------
1. Every decision is made on VALIDATION. The reported checkpoint is the one with
   the lowest validation aggregate; nothing on the evaluation segment is ever
   read here. Evaluation numbers are produced later by evaluate.py, from the
   saved checkpoint's generated ensemble.

2. No threshold is baked in. The run executes a fixed iteration budget and logs
   the full validation history, so "iterations to reach tolerance tau" can be
   computed afterwards for any tau. This matters because tau is to be fixed by
   oracle calibration, not by looking at model results -- a run that stopped
   early at a guessed threshold could not be re-scored.

3. d_in does not change the parameter count. The LSTM input is ALWAYS 4 wide;
   channels d_in..3 are fed exact zeros. Their input weights receive no gradient
   and contribute nothing, so the network is functionally identical to one with
   input_size = d_in, while the parameter count is constant across the d_in
   sweep. Nobody can attribute a d_in effect to capacity. The dead-parameter
   fraction is +0.4% to +5.5% depending on capacity and is logged.

4. The full L = 200 steps go to every loss. The recurrent warm-up is extra
   steps fed before the scored window and discarded here, never by the loss.
   K = 200 (lags 0..199, i.e. k_max = L-1) does not truncate anything either;
   it is the full lag range, matching the FPU/LLV convention. Cutting the
   trajectory itself would halve the KDE sample count, which is why the warm-up
   is extra steps rather than a slice of the scored window.

5. Everything the objective needs comes from reference_lam*.npz and
   target_stats_lam*.npz, whose constants hash is logged with the run. A run
   cannot be silently paired with a different normalisation later.

RUN TAG carries every factor, including capacity, because block 1b differs from
block 1 only by capacity and a tag that omitted it would collide:

    nf3_lbon_din2_L3H75_lam4_s07

SELECTION AGGREGATE. The five squared relative discrepancies against the
validation reference -- the same statistics and normalisation the evaluator
reports, with the analytic density for PDF/PROJ and the validation ensemble for
ACF/CCF/ACF2 -- summed. Smooth, and on the same quantities as the final report.

Usage
-----
    python train_sinn.py --datadir data --lam 4 --nf 3 --lb on --din 2 \
        --layers 3 --hidden 75 --seed 7 --iters 5000 --outdir runs

    python train_sinn.py --datadir data --lam 4 --smoke      # 60 iterations
"""

import argparse
import json
import os
import platform
import socket
import time

import numpy as np
import torch
from torch import nn

import evaluate as EV
from sinn_loss import SINNObjective

INPUT_WIDTH = 4          # always; d_in active channels, the rest fed zeros
DIN_MAX = 4


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class SINN(nn.Module):
    """Stacked LSTM with a linear readout -- the architecture of
    toolbox/_sinn.py, with the input width pinned at INPUT_WIDTH so that the
    parameter count does not move with d_in."""

    def __init__(self, hidden, layers, output=2, input_width=INPUT_WIDTH):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_width, hidden_size=hidden,
                            num_layers=layers)
        self.readout = nn.Linear(hidden, output)

    def forward(self, x, state=None):
        y, state = self.lstm(x, state)
        return self.readout(y), state


def hardware_record(device):
    """Everything needed to say later what a number was measured on."""
    rec = {"device": str(device), "torch": torch.__version__,
           "hostname": socket.gethostname(), "python": platform.python_version(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        rec.update({"gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count_visible": torch.cuda.device_count(),
                    "cuda": torch.version.cuda})
    return rec


def param_counts(hidden, layers, d_in):
    """Total parameters and the fraction that is dead because d_in < 4."""
    total = sum(p.numel() for p in SINN(hidden, layers).parameters())
    dead = 4 * hidden * (INPUT_WIDTH - d_in)      # first-layer input weights only
    return total, dead


def white_noise(n_steps, n_traj, d_in, gen, device, dtype):
    """(n_steps, n_traj, INPUT_WIDTH) Gaussian noise with channels >= d_in
    exactly zero. A fresh batch every iteration, which is what keeps the network
    from overfitting a particular noise realisation."""
    x = torch.zeros((n_steps, n_traj, INPUT_WIDTH), device=device, dtype=dtype)
    x[:, :, :d_in] = torch.randn((n_steps, n_traj, d_in), generator=gen,
                                 device=device, dtype=dtype)
    return x


# --------------------------------------------------------------------------
# Validation scoring
# --------------------------------------------------------------------------

class Validator:
    """The five errors of a generated ensemble against the VALIDATION reference.

    Uses evaluate.py's numpy estimators, the same code the final report runs, so
    selection and reporting cannot diverge. The evaluation segment is never
    touched here.
    """

    def __init__(self, datadir, tag, n_gen, tau):
        traj_path = os.path.join(datadir, f"traj_{tag}.npy")
        self.ref = EV.load_reference(datadir, tag, traj_path)
        traj = np.load(traj_path, mmap_mode="r")
        self.split = EV.split_from_reference(self.ref)
        self.deg = EV.ccf_is_degenerate(self.ref)

        # The yardstick is the WHOLE validation segment, always, because that is
        # what evaluate.floor uses for the nine held-out blocks. An earlier
        # version let --val-traj shrink both the generated ensemble AND this
        # reference, so the model was scored against a 2000-trajectory yardstick
        # while the floor was scored against a 4000-trajectory one -- numerator
        # and denominator measured with different rulers, and their ratio
        # meaningless. n_gen now controls only how many trajectories the network
        # generates. It must never touch the reference.
        self.sr = EV.reference_stats(traj, self.ref, self.split["validation"])
        self.n_gen = int(n_gen)

        # tau is calibrated, never a convention: it is the 95th percentile of the
        # worst-family ratio a GENUINE ensemble of exactly n_gen trajectories
        # reaches by sampling noise alone (calibrate.py). It does not transfer
        # across n_gen, so the two travel together and both are logged.
        self.tau = float(tau)
        self.floor, _, _ = EV.floor(traj, self.ref, self.split, "validation")

    def __call__(self, gen_np):
        e = EV.errors(EV.stats(gen_np, self.ref, "standardized"), self.sr, self.deg)
        fams = [f for f in EV.FAMILIES if not (f == "CCF" and self.deg)]
        agg = float(sum(e[f] ** 2 for f in fams))       # selection scalar
        cls = EV.classify(e, self.floor, self.tau)
        return e, agg, cls


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(args):
    tag = f"lam{args.lam:g}".replace(".", "p")
    run_tag = (f"nf{args.nf}_lb{'on' if args.lb == 'on' else 'off'}"
               f"_din{args.din}_L{args.layers}H{args.hidden}_{tag}_s{args.seed:02d}")
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    dtype = torch.float32

    obj = SINNObjective.from_files(args.datadir, tag, n_f=args.nf,
                                   loss_balancing=(args.lb == "on"),
                                   device=device, dtype=dtype)
    val = Validator(args.datadir, tag, args.n_gen, args.tau)

    # seed_init and seed_data are separate knobs: the archive is fixed (its own
    # seeds live in meta_lam*.json), this one only drives the initialisation and
    # the input noise.
    torch.manual_seed(args.seed)
    net = SINN(args.hidden, args.layers).to(device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 10_000)
    # A SEPARATE stream for the validation noise. Sharing one generator made the
    # training noise sequence depend on --val-every, so the same seed gave
    # different runs at different validation cadences. Seed-for-seed comparison
    # across the grid is the whole point of the seed axis.
    gen_val = torch.Generator(device=device)
    gen_val.manual_seed(args.seed + 20_000)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr_theta)
    opt_sigma = (torch.optim.Adam(obj.balance_parameters(), lr=args.lr_sigma)
                 if args.lb == "on" else None)

    total_p, dead_p = param_counts(args.hidden, args.layers, args.din)
    n_steps = obj.l_seq + args.warmup
    hist = {"iter": [], "train_unweighted": [], "val_agg": [],
            "val_errors": [], "sigma": []}
    best = {"iter": -1, "val_agg": float("inf")}
    t0 = time.time()

    for it in range(1, args.iters + 1):
        x = white_noise(n_steps, args.batch, args.din, gen, device, dtype)
        y, _ = net(x)
        out = obj(y[-obj.l_seq:])                       # warm-up discarded here

        opt.zero_grad(set_to_none=True)
        if opt_sigma is not None:
            opt_sigma.zero_grad(set_to_none=True)
        out["total"].backward()
        opt.step()
        if opt_sigma is not None:
            opt_sigma.step()

        if it % args.val_every == 0 or it == args.iters:
            net.eval()
            with torch.no_grad():
                xv = white_noise(n_steps, val.n_gen, args.din, gen_val, device, dtype)
                yv, _ = net(xv)
                gen_np = yv[-obj.l_seq:].detach().cpu().numpy().astype(np.float64)
            net.train()
            e, agg, cls = val(gen_np)

            hist["iter"].append(it)
            hist["train_unweighted"].append(float(out["unweighted_total"]))
            hist["val_agg"].append(agg)
            hist["val_errors"].append({k: float(e[k]) for k in EV.FAMILIES
                                       if not (k == "CCF" and val.deg)})
            hist["sigma"].append({k: float(torch.exp(v).detach())
                                  for k, v in obj.log_sigma.items()}
                                 if args.lb == "on" else {})

            if agg < best["val_agg"]:
                best = {"iter": it, "val_agg": agg,
                        "val_errors": hist["val_errors"][-1],
                        "worst_ratio": cls["worst_ratio"],
                        "success": bool(cls["success"]),
                        "ratios": {k: float(v) for k, v in cls["ratios"].items()}}
                torch.save({"state_dict": net.state_dict(), "iter": it,
                            "run_tag": run_tag,
                            "constants_sha256": obj.constants_sha256},
                           os.path.join(args.outdir, f"net_{run_tag}.pt"))

            if args.verbose:
                print(f"[{run_tag}] it={it:<6} train={float(out['unweighted_total']):.4e} "
                      f"val_agg={agg:.4e} worst={cls['worst_ratio']:.2f}x"
                      f"{' *' if best['iter'] == it else ''}")

    wall = time.time() - t0
    np.savez(os.path.join(args.outdir, f"hist_{run_tag}.npz"),
             iter=np.array(hist["iter"]),
             train_unweighted=np.array(hist["train_unweighted"]),
             val_agg=np.array(hist["val_agg"]),
             val_errors=json.dumps(hist["val_errors"]),
             sigma=json.dumps(hist["sigma"]))

    record = {
        "run_tag": run_tag, "lambda": args.lam, "tag": tag,
        "n_f": args.nf, "loss_balancing": args.lb == "on", "d_in": args.din,
        "layers": args.layers, "hidden": args.hidden,
        "params_total": total_p, "params_dead_from_din": dead_p,
        "params_dead_fraction": dead_p / total_p,
        "seed_init": args.seed, "batch": args.batch, "warmup": args.warmup,
        "l_seq": obj.l_seq, "k_lags": obj.k_lags, "ccf_mode": obj.ccf_mode,
        "iters": args.iters, "lr_theta": args.lr_theta, "lr_sigma": args.lr_sigma,
        "n_gen": val.n_gen, "tau": val.tau, "val_every": args.val_every,
        "families": list(obj.families),
        "best": best,
        "final_val_agg": hist["val_agg"][-1] if hist["val_agg"] else None,
        # No threshold is applied here. The full history is saved so
        # iterations-to-tolerance can be computed for any tau afterwards.
        "wall_time_s": wall, "s_per_iter": wall / args.iters,
        # Hardware provenance. This is BOOKKEEPING ONLY: accuracy runs are
        # packed several per GPU, so s_per_iter from here must never be quoted
        # in the manuscript. All reported timing comes from timing.py on one
        # A100 of dept.appliedmath with --exclusive. Section 5.1 had to infer
        # its hardware after
        # the fact because none of this was logged; that is the mistake this
        # block exists to avoid.
        **hardware_record(device),
        "constants_sha256": obj.constants_sha256,
        "selection": "validation only; evaluation segment never read by this driver",
    }
    # Two writes, on purpose. The per-run file is AUTHORITATIVE: the sweep packs
    # 6-10 processes per GPU, all finishing around the same time, and concurrent
    # appends to one shared file are only atomic below PIPE_BUF -- these records
    # are close enough to that limit that a torn line is a real possibility, and
    # a torn line is a silently lost run. runs.jsonl is kept because it is
    # convenient to grep, but it is a convenience copy: rebuild it from the
    # per-run files (collect_runs.py) before any analysis.
    with open(os.path.join(args.outdir, f"run_{run_tag}.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    with open(os.path.join(args.outdir, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")
    if args.verbose:
        print(f"[{run_tag}] best it={best['iter']} val_agg={best['val_agg']:.4e} "
              f"worst={best.get('worst_ratio', float('nan')):.2f}x "
              f"success={best.get('success')}  {wall:.0f}s "
              f"({record['s_per_iter']:.3f} s/iter)")
    return record


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--nf", type=int, default=3, choices=(0, 1, 2, 3),
                    help="orthogonal projection FRAMES; 2*nf directions")
    ap.add_argument("--lb", choices=("on", "off"), default="on")
    ap.add_argument("--din", type=int, default=2, choices=(1, 2, 3, 4),
                    help="active white-noise channels; the LSTM input stays 4 wide")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=400,
                    help="generated trajectories per training step")
    ap.add_argument("--warmup", type=int, default=200,
                    help="recurrent warm-up steps, fed then discarded "
                         "(= L, the FPU/LLV convention; >= 2.5 correlation "
                         "times of the target)")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=400,
                    help="trajectories the network generates for scoring. 400 = "
                         "the fitting budget B, so the model is compared "
                         "like-for-like against real samples of the same size. "
                         "This does NOT change the validation reference, which "
                         "is always the whole segment.")
    ap.add_argument("--tau", type=float, required=True,
                    help="success threshold, from calibrate.py at this --n-gen. "
                         "Required on purpose: there is no defensible default, "
                         "and tau does not transfer across n_gen.")
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--lr-sigma", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="60 iterations, tiny net, for a pipeline check")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    if args.smoke:
        args.iters, args.val_every = 60, 20
        args.layers, args.hidden, args.batch, args.n_gen = 2, 10, 200, 400

    run(args)


if __name__ == "__main__":
    main()
