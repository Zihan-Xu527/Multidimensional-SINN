#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The two timing benchmarks, measured separately from the accuracy sweep.

WHY SEPARATELY. Per-iteration cost is not comparable across GPU models or across
a shared device, so it is never taken from the accuracy runs -- those are spread
over the four A100s of dept.appliedmath, several processes packed per GPU. The
timing runs use the SAME GPU model, one A100 of dept.appliedmath, but reserved
--exclusive and drained of sweep jobs. One model for the whole manuscript: the
accuracy sweep, the Section 4 timing table and the Section 5.1 timing column are
then all A100, and no number has to be qualified by which node it came from.

This workload is a sequential LSTM and is launch-bound, so CPU contention
pollutes s/iter directly even when the GPU itself is free -- hence --exclusive
rather than a bare --gres=gpu:1.

    sbatch --partition=dept.appliedmath --nodelist=gnode009 --gres=gpu:a100:1 \
           --exclusive --time=00:30:00 --wrap "python timing.py --datadir data --all"

gnode010 is the same model and is an equally valid host; pin ONE of them and use
it for every timing run reported, including any re-measurement of 5.1.

BENCHMARK A -- training step cost.
Per-iteration wall time of the real training step, split into forward / loss /
backward, over a grid of capacities and with loss balancing on and off. Supports
the Section 4.4.1 claim that balancing costs slightly more per iteration but
reaches tolerance in fewer iterations, so the total is shorter. That claim needs
BOTH halves: this file measures the per-iteration half, and the iteration half
comes from the validation histories saved by train_sinn.py, for whatever
tolerance is finally adopted.

The old numbers are obsolete and must not be reused: L went 400 -> 200 and the
recurrent warm-up 400 -> 200 (warm-up = L is kept, so the LSTM runs 400
sequential steps per iteration, not 800), and the KDE went from direct summation
to CIC plus a cached convolution. Nothing about the previous 0.06 / 0.07 s/iter
carries over.

BENCHMARK B -- the cost of a multivariate KDE against projected 1-D KDEs.
Section 3 asserts that direct multivariate KDE "can be computationally
expensive" with no evidence. This measures it, and it measures it FAIRLY: both
sides use the same cloud-in-cell deposition plus Gaussian convolution, so the
comparison is about dimensionality and not about one route being hand-optimised
while the other is naive. Comparing our CIC 1-D against a direct-summation d-D
would be a statement about implementations.

The dominant effect is not the deposition -- that is O(n_samples * 2^d) and grows
mildly. It is the GRID: N_z^d cells at fixed per-axis resolution. At N_z = 401
that is 1.6e5 at d = 2, 6.4e7 at d = 3 and 2.6e10 at d = 4, the last of which
does not fit in any GPU. To stay affordable at higher d one must coarsen the
axis, i.e. give up resolution; the projected route never faces that trade-off,
since n_f * d one-dimensional grids grow linearly in d. The benchmark therefore
reports time AND grid footprint, and marks the configurations that are infeasible
rather than silently skipping them.

Usage
-----
    python timing.py --datadir data --all
    python timing.py --datadir data --step        # benchmark A only
    python timing.py --kde                       # benchmark B only (no data needed)
    python timing.py --kde --numpy               # CPU/numpy, for a sanity check
"""

import argparse
import json
import os
import platform
import socket
import time

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def has_whole_node(env):
    """True when the job holds every core on its node, which is what exclusivity
    buys a launch-bound measurement. Returns None -- neither true nor false --
    for a record written before these fields existed, so that the caller can say
    "cannot be determined from this file" instead of guessing either way."""
    a, t = env.get("cpus_affinity"), env.get("cpus_total")
    if a is None or t is None:
        return None
    return a == t


def environment():
    """Recorded once per table. A timing number without this is not quotable --
    Section 5.1 had to infer its hardware after the fact because none of it was
    logged."""
    env = {"hostname": socket.gethostname(), "python": platform.python_version(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
           "slurm_exclusive": os.environ.get("SLURM_JOB_EXCLUSIVE"),
           # SLURM_JOB_EXCLUSIVE is not exported by every Slurm configuration.
           # It is unset on this cluster even for jobs that sacct reports with
           # AllocCPUS equal to all 56 cores of the node, so it cannot settle the
           # question on its own. These two can: if the job holds every core the
           # node has, nothing else is on its CPUs, which is the thing a
           # launch-bound measurement actually needs. It is a measurement rather
           # than a self-declaration, which is why exporting SLURM_JOB_EXCLUSIVE
           # by hand from the submit script would have been the wrong fix -- that
           # only records what the script claimed, not what Slurm granted.
           "cpus_affinity": len(os.sched_getaffinity(0)),
           "cpus_total": os.cpu_count(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch is not None:
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env.update({"gpu_name": p.name, "gpu_memory_GB": p.total_memory / 1e9,
                        "gpu_count_visible": torch.cuda.device_count(),
                        "cuda": torch.version.cuda})
    return env


def _sync(device):
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, device, reps, warmup=5):
    """Median and IQR of a timed call, after warm-up. Median rather than mean:
    a stray scheduler hiccup should not set the reported number."""
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 75) - np.percentile(ts, 25))


# --------------------------------------------------------------------------
# Benchmark A: the training step
# --------------------------------------------------------------------------

def bench_step(datadir, lam, capacities, n_f=3, d_in=2, batch=400, warmup_steps=200,
               reps=30, device=None):
    if torch is None:
        raise SystemExit("benchmark A needs PyTorch")
    import train_sinn as TS
    from sinn_loss import SINNObjective

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"lam{lam:g}".replace(".", "p")
    rows = []

    for layers, hidden in capacities:
        for lb in (False, True):
            obj = SINNObjective.from_files(datadir, tag, n_f=n_f,
                                           loss_balancing=lb, device=device,
                                           dtype=torch.float32)
            net = TS.SINN(hidden, layers).to(device=device, dtype=torch.float32)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            opt_s = (torch.optim.Adam(obj.balance_parameters(), lr=1e-3)
                     if lb else None)
            gen = torch.Generator(device=device)
            gen.manual_seed(0)
            n_steps = obj.l_seq + warmup_steps
            x = TS.white_noise(n_steps, batch, d_in, gen, device, torch.float32)

            def fwd():
                return net(x)[0][-obj.l_seq:]

            def full():
                opt.zero_grad(set_to_none=True)
                if opt_s is not None:
                    opt_s.zero_grad(set_to_none=True)
                out = obj(net(x)[0][-obj.l_seq:])
                out["total"].backward()
                opt.step()
                if opt_s is not None:
                    opt_s.step()

            t_fwd, _ = _time(fwd, device, reps)

            y = fwd().detach()

            def lossonly():
                obj(y)

            t_loss, _ = _time(lossonly, device, reps)
            t_all, iqr = _time(full, device, reps)

            rows.append({"layers": layers, "hidden": hidden,
                         "params": sum(p.numel() for p in net.parameters()),
                         "loss_balancing": lb, "n_f": n_f, "d_in": d_in,
                         "batch": batch, "l_seq": obj.l_seq,
                         "warmup_steps": warmup_steps,
                         "s_per_iter": t_all, "iqr": iqr,
                         "s_forward": t_fwd, "s_loss_only": t_loss})
            print(f"  L{layers}H{hidden:<4} lb={'on ' if lb else 'off'} "
                  f"{t_all * 1e3:7.2f} ms/iter (IQR {iqr * 1e3:.2f})   "
                  f"fwd {t_fwd * 1e3:6.2f}  loss {t_loss * 1e3:6.2f}")

    # The claim Section 4.4.1 makes needs the ratio stated, not just the pair.
    for layers, hidden in capacities:
        off = next(r for r in rows if r["layers"] == layers
                   and r["hidden"] == hidden and not r["loss_balancing"])
        on = next(r for r in rows if r["layers"] == layers
                  and r["hidden"] == hidden and r["loss_balancing"])
        print(f"  L{layers}H{hidden:<4} balancing overhead "
              f"{100 * (on['s_per_iter'] / off['s_per_iter'] - 1):+.1f}% per iteration")
    return rows


# --------------------------------------------------------------------------
# Benchmark B: multivariate KDE against projected 1-D KDEs
# --------------------------------------------------------------------------

def _cic_1d_np(s, lo, dz, n_fine, conv):
    pos = (s - lo) / dz
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 < n_fine - 1)
    i0, frac = i0[ok], frac[ok]
    f = (np.bincount(i0, weights=1 - frac, minlength=n_fine)
         + np.bincount(i0 + 1, weights=frac, minlength=n_fine))[:n_fine]
    return conv @ f


def _cic_dd_np(x, lo, dz, n_ax, conv):
    """d-dimensional CIC deposition onto an n_ax^d grid, then a separable
    Gaussian convolution along each axis. Same estimator family as the 1-D
    route -- that is the point of the comparison."""
    n_samp, d = x.shape
    pos = (x - lo) / dz
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    cells = n_ax ** d
    flat_grid = np.zeros(cells, dtype=np.float64)
    for corner in range(2 ** d):
        idx = []
        w = np.ones(n_samp)
        ok = np.ones(n_samp, dtype=bool)
        for a in range(d):
            bit = (corner >> a) & 1
            ia = i0[:, a] + bit
            ok &= (ia >= 0) & (ia < n_ax)
            idx.append(np.clip(ia, 0, n_ax - 1))
            w = w * (frac[:, a] if bit else 1 - frac[:, a])
        flat = np.ravel_multi_index(tuple(idx), (n_ax,) * d)
        # bincount, not np.add.at: add.at is 10-50x slower, and using it here
        # while the 1-D route uses bincount would make the comparison a
        # statement about numpy internals rather than about dimensionality.
        flat_grid += np.bincount(flat[ok], weights=w[ok], minlength=cells)
    grid = flat_grid.reshape((n_ax,) * d)
    for a in range(d):
        grid = np.moveaxis(np.tensordot(conv, np.moveaxis(grid, a, 0), axes=1), 0, a)
    return grid


def bench_kde(dims=(2, 3, 4), batches=(400, 800), l_seq=200, n_f=3,
              n_ax_1d=401, n_fine=4001, use_numpy=False, reps=5,
              mem_budget_GB=40.0):
    """Cost of one d-dimensional joint KDE against n_f*d one-dimensional ones.

    Reports the grid footprint as well as the time, and marks a configuration
    infeasible rather than pretending it ran: at fixed per-axis resolution the
    joint grid is N_z^d cells, which is 2.6e10 at d = 4.
    """
    rows = []
    lo, hi = -5.0, 5.0
    z_out = np.linspace(lo, hi, n_ax_1d)
    z_fine = np.linspace(lo - 0.6, hi + 0.6, n_fine)
    dzf = z_fine[1] - z_fine[0]
    h = (l_seq * 400) ** (-1.0 / 5.0)
    conv_1d = (np.exp(-0.5 * ((z_out[:, None] - z_fine[None, :]) / h) ** 2)
               * dzf / (np.sqrt(2 * np.pi) * h))
    rng = np.random.default_rng(0)

    print(f"  per-axis grid {n_ax_1d}, deposition grid {n_fine}, h = {h:.4f}")
    print("  NOTE: this runs on CPU/numpy. Both sides use bincount deposition, so "
          "the comparison\n        is fair, but the TIME ratio is not the quotable "
          "number -- re-measure it with\n        torch scatter_add on the A100. The "
          "grid cell count and memory footprint are\n        exact and "
          "implementation-independent; that is where the argument actually lives.")
    print("  Measured on CPU: d=2 gives only ~1.9x, so 'multivariate KDE is "
          "prohibitive' is NOT\n        supported at d=2. The scaling from d=2 to "
          "d=3 to d=4 is.")
    for d in dims:
        h_d = (l_seq * 400) ** (-1.0 / (d + 4))
        cells = n_ax_1d ** d
        gb = cells * 8 / 1e9
        conv_d = (np.exp(-0.5 * ((z_out[:, None] - z_out[None, :]) / h_d) ** 2)
                  * (z_out[1] - z_out[0]) / (np.sqrt(2 * np.pi) * h_d))
        for batch in batches:
            n_samp = l_seq * batch
            x = rng.standard_normal((n_samp, d))
            n_proj = n_f * d

            t1, _ = _time(lambda: [_cic_1d_np(x[:, i % d], z_fine[0], dzf, n_fine,
                                              conv_1d) for i in range(n_proj)],
                          "cpu", reps, warmup=2)
            feasible = gb <= mem_budget_GB
            td = float("nan")
            if feasible:
                td, _ = _time(lambda: _cic_dd_np(x, lo, (hi - lo) / (n_ax_1d - 1),
                                                 n_ax_1d, conv_d),
                              "cpu", max(1, reps // 2), warmup=1)
            rows.append({"d": d, "batch": batch, "n_samples": n_samp,
                         "n_projections": n_proj, "grid_cells_joint": cells,
                         "grid_GB_joint_float64": gb, "feasible": bool(feasible),
                         "s_projected_1d_total": t1, "s_joint_dd": td,
                         "ratio_joint_over_projected": (td / t1 if feasible
                                                        else float("inf"))})
            if feasible:
                print(f"  d={d} batch={batch:<5} {n_proj:2d} x 1-D: "
                      f"{t1 * 1e3:8.2f} ms   joint {d}-D: {td * 1e3:9.2f} ms  "
                      f"({td / t1:6.1f}x)   grid {cells:.3e} cells, {gb:.2f} GB")
            else:
                print(f"  d={d} batch={batch:<5} {n_proj:2d} x 1-D: "
                      f"{t1 * 1e3:8.2f} ms   joint {d}-D: INFEASIBLE          "
                      f"grid {cells:.3e} cells, {gb:.1f} GB > {mem_budget_GB} GB")
    return rows


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--step", action="store_true", help="benchmark A")
    ap.add_argument("--kde", action="store_true", help="benchmark B")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--numpy", action="store_true", help="benchmark B on CPU/numpy")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--capacities", default="2x10,2x30,3x30,3x50,3x75,3x110")
    ap.add_argument("--mem-budget-GB", type=float, default=40.0)
    args = ap.parse_args()

    env = environment()
    print("environment:", json.dumps(env, indent=2))
    if env.get("gpu_name") and has_whole_node(env) is False:
        print(f"  NOTE: this job holds {env.get('cpus_affinity')} of the node's "
              f"{env.get('cpus_total')} cores, so another job can share its CPUs "
              f"and inflate s/iter -- this workload is launch-bound. Do not "
              f"quote these numbers.")
    gpu = (env.get("gpu_name") or "")
    if gpu and "A100" not in gpu:
        print(f"  NOTE: GPU is '{gpu}', not an A100. Every timing number in the "
              "manuscript is\n        specified on an A100 of dept.appliedmath "
              "(gnode009/gnode010). Numbers from a\n        different model are "
              "not comparable with the rest of the table. Do not quote these.")

    report = {"environment": env}
    caps = [tuple(int(v) for v in c.split("x")) for c in args.capacities.split(",")]

    if args.step or args.all:
        print("\nBENCHMARK A -- training step")
        report["step"] = bench_step(args.datadir, args.lam, caps, reps=args.reps)
    if args.kde or args.all:
        print("\nBENCHMARK B -- multivariate KDE vs projected 1-D KDEs "
              "(same CIC estimator on both sides)")
        report["kde"] = bench_kde(use_numpy=args.numpy,
                                 mem_budget_GB=args.mem_budget_GB)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
