#!/usr/bin/env python3
"""Parse all SPPARKS logs into one archive file, FPU-style.

Reads  $OUT/log/log<i>.spparks   for i = 1 .. NRUN
Writes $OUT/llv_theta_T<T>_B<NRUN>.npz  with

    traj   float32 (T, NRUN, 2)   theta_A, theta_B at t = 1 .. T
    sites  ('A', 'B')
    meta   json string: lattice, rates, Nstep, deltat, seeds, burn-in, ...

This replaces the 02_parse_log -> 05_SINN_traj text pipeline for the 20000-run
archive: 40000 small text files cost far more I/O than one 192 MB array, and a
single npz matches the data contract the FPU chain already uses.

usage:  OUT=/data/zxu64/LLV_KMC_20k NRUN=20000 python3 build_llv_archive.py
"""

import json
import os
import sys
import time

import numpy as np

OUT = os.environ.get("OUT", "/data/zxu64/LLV_KMC_20k")
NRUN = int(os.environ.get("NRUN", "20000"))
NSTEP = int(os.environ.get("NSTEP", "1200"))
NSITE = int(os.environ.get("NSITE", "10000"))

LOGDIR = os.path.join(OUT, "log")


def parse_one(path):
    """Return (T,2) float64 array of theta_A, theta_B at t = 1 .. NSTEP.

    SPPARKS stats lines are:
        Time  Naccept  Nreject  Nsweeps  CPU  vac  spec1  spec2
    The per-timestep `run` loop emits the same time twice (end of one run,
    start of the next), so rows are de-duplicated on time, keeping the last.
    """
    seen = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 8:
                continue
            try:
                t = float(parts[0])
                n1 = int(parts[6])
                n2 = int(parts[7])
            except ValueError:
                continue
            seen[t] = (n1, n2)

    times = sorted(k for k in seen if k >= 1.0)
    if len(times) != NSTEP:
        raise ValueError(
            "%s: %d rows after dedup, expected %d" % (path, len(times), NSTEP)
        )
    out = np.empty((NSTEP, 2), dtype=np.float64)
    for row, t in enumerate(times):
        n1, n2 = seen[t]
        out[row, 0] = n1 / NSITE
        out[row, 1] = n2 / NSITE
    return out


def main():
    missing = [i for i in range(1, NRUN + 1)
               if not os.path.exists(os.path.join(LOGDIR, "log%d.spparks" % i))]
    if missing:
        sys.exit("missing %d logs, first few: %s" % (len(missing), missing[:5]))

    traj = np.empty((NSTEP, NRUN, 2), dtype=np.float32)
    t0 = time.time()
    for i in range(1, NRUN + 1):
        traj[:, i - 1, :] = parse_one(os.path.join(LOGDIR, "log%d.spparks" % i))
        if i % 1000 == 0:
            rate = i / (time.time() - t0)
            print("  %6d / %d   %.0f logs/s   eta %.1f min"
                  % (i, NRUN, rate, (NRUN - i) / rate / 60), flush=True)

    meta = {
        "model": "lattice Lotka-Volterra, SPPARKS app_style surfchemtest, solve_style tree",
        "lattice": [100, 100], "periodic": True, "n_site": NSITE,
        "k1": 0.5, "k2": 0.3, "k3": 0.1,
        "Nstep": NSTEP, "deltat": 1.0, "Nrun": NRUN, "seeds": "1..%d" % NRUN,
        "components": ["theta_A (prey)", "theta_B (predator)"],
        "t_index": "traj[j] is t = j+1, j = 0 .. %d" % (NSTEP - 1),
        "W_G": 400,
        "note": "burn-in W_K is NOT removed here; the full t = 1..%d is stored" % NSTEP,
    }

    path = os.path.join(OUT, "llv_theta_T%d_B%d.npz" % (NSTEP, NRUN))
    np.savez_compressed(path, traj=traj, sites=np.array(["A", "B"]),
                        meta=json.dumps(meta))
    print("\nwrote %s  (%.0f MB)" % (path, os.path.getsize(path) / 1e6))

    # ---- sanity ----------------------------------------------------------
    print("\nshape", traj.shape, traj.dtype)
    late = traj[800:, :, :]                      # t = 801..1200, safely stationary
    print("stationary window t=801..1200:")
    print("  theta_A  mean %.4f  sd %.4f" % (late[:, :, 0].mean(), late[:, :, 0].std()))
    print("  theta_B  mean %.4f  sd %.4f" % (late[:, :, 1].mean(), late[:, :, 1].std()))
    print("  expected  A 0.1217 +- 0.0228   B 0.3547 +- 0.0275   (400-run archive)")

    bad = ((traj[:, :, 0] < 0) | (traj[:, :, 1] < 0)
           | (traj[:, :, 0] + traj[:, :, 1] > 1)).sum()
    print("\nEq.(35) violations in the KMC reference: %d  (must be 0)" % bad)

    ens = traj.mean(axis=1)
    print("\nensemble mean vs t (transient):")
    for t in (1, 10, 50, 100, 200, 300, 400, 600, 900, 1200):
        print("  t=%5d   A %.4f   B %.4f" % (t, ens[t - 1, 0], ens[t - 1, 1]))


if __name__ == "__main__":
    main()
