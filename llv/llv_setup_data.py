#!/usr/bin/env python3
"""Export the data behind the illustrative LLV setup figure.

Scope. Presentation only. Nothing here is trained, classified, selected or
summarised, and nothing here enters a loss, a ratio, a threshold or a
decision. Two illustrative objects are copied out of material that already
exists:

  1. one representative post-transient KMC lattice configuration, taken from
     a saved SPPARKS frame stack (0 = vacancy O, 1 = prey A, 2 = predator B);
  2. one fixed KMC trajectory of the coarse-grained densities, trajectory 0 of
     the FITTING segment over physical time t = 401 .. 800, in physical
     density coordinates rho, not standardized.

The trajectory is deliberately taken from the fitting segment: the figure
introduces the model system, so it must not display held-out data, and it
must not invite a pathwise comparison with any SINN output. There is no SINN
quantity in this export at all.

The lattice frame and the density trajectory come from two different KMC
runs. They are not two views of one realisation and the caption must not
claim they are; both are illustrations of the same stationary regime.

usage:
    ARCHIVE=... python3 llv_setup_data.py --manifest-dir llv_manifest_v2 \
        --frames frames.npz --out setup_llv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import llv_spec as S

SPECIES = {0: "O (vacancy)", 1: "A (prey)", 2: "B (predator)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--frames", default="frames.npz",
                    help="saved SPPARKS frame stack; see dump_to_frames.py")
    ap.add_argument("--frame-index", type=int, default=-1,
                    help="which frame to show; the default is the last, "
                         "which is the furthest from the transient")
    ap.add_argument("--trajectory", type=int, default=0)
    ap.add_argument("--out", default="setup_llv")
    args = ap.parse_args()

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    # The fitting segment needs no freeze token; assert_unlocked is called
    # anyway so that this export records the same verified chain as every
    # other product, and so a tampered archive is caught here too.
    man = loader.assert_unlocked()

    os.makedirs(args.out, exist_ok=True)
    print("LLV setup-figure data export (illustrative; no SINN quantity)")

    # ---- (b) one fixed KMC density trajectory ----------------------------
    blk = loader.fitting(S.TARGET_G_IN)           # (400, n_traj, 2), physical
    n_traj = blk.shape[1]
    if not 0 <= args.trajectory < n_traj:
        raise SystemExit("trajectory %d is out of range; the fitting segment "
                         "has %d" % (args.trajectory, n_traj))
    rho = np.ascontiguousarray(blk[:, args.trajectory, :])
    t_lo, t_hi = S.TARGET_G_IN[0] + 1, S.TARGET_G_IN[1]
    t_phys = np.arange(t_lo, t_hi + 1, dtype=np.int64)
    if t_phys.shape[0] != rho.shape[0]:
        raise AssertionError("time axis %d vs %d rows"
                             % (t_phys.shape[0], rho.shape[0]))
    if not np.isfinite(rho).all():
        raise SystemExit("the selected trajectory contains non-finite values")
    print("  trajectory %d of the fitting segment, physical t = %d..%d"
          % (args.trajectory, t_lo, t_hi))
    print("    rho_A  mean %.4f  range [%.4f, %.4f]"
          % (rho[:, 0].mean(), rho[:, 0].min(), rho[:, 0].max()))
    print("    rho_B  mean %.4f  range [%.4f, %.4f]"
          % (rho[:, 1].mean(), rho[:, 1].min(), rho[:, 1].max()))

    payload = {
        "rho_trajectory": rho,
        "t_physical": t_phys,
    }
    doc_lattice = None

    # ---- (a) one representative lattice configuration --------------------
    if args.frames and os.path.exists(args.frames):
        z = np.load(args.frames)
        frames, times = z["frames"], z["times"]
        if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
            raise SystemExit("frame stack has shape %r; expected "
                             "(n_frames, L, L)" % (frames.shape,))
        idx = int(args.frame_index) % frames.shape[0]
        lat = np.asarray(frames[idx], dtype=np.uint8)
        bad = sorted(set(np.unique(lat).tolist()) - set(SPECIES))
        if bad:
            raise SystemExit("frame carries unexpected site states %r" % bad)
        occ = {SPECIES[k]: float((lat == k).mean()) for k in SPECIES}
        # the stationary densities recorded in the frozen manifest, for a
        # consistency print only -- nothing is asserted against them, because
        # the lattice dump is a different KMC run from the archive
        mu = np.asarray(man["standardization"]["mu"], np.float64)
        print("  lattice frame %d of %d, t_dump = %.3f, %dx%d"
              % (idx, frames.shape[0], float(times[idx]), *lat.shape))
        for k in sorted(SPECIES):
            print("    %-14s %.4f" % (SPECIES[k], (lat == k).mean()))
        print("    archive stationary rho_A, rho_B: %.4f, %.4f "
              "(different KMC run; printed for orientation only)"
              % (mu[0], mu[1]))
        payload["lattice"] = lat
        doc_lattice = {
            "source_file": os.path.abspath(args.frames),
            "source_sha256": S.sha256_file(args.frames),
            "frame_index": idx,
            "n_frames": int(frames.shape[0]),
            "frame_time_in_dump": float(times[idx]),
            "side": int(lat.shape[0]),
            "encoding": {str(k): v for k, v in SPECIES.items()},
            "occupancy": occ,
            "same_realisation_as_trajectory": False,
            "note": "a different KMC run from the archive; illustrative only",
        }
    else:
        print("  NO lattice frame: %r is missing. Panel (a) will be skipped. "
              "Produce it with dump_to_frames.py and rerun." % args.frames)

    npz = os.path.join(args.out, "setup_data.npz")
    np.savez_compressed(npz, **payload)
    npz_sha = S.sha256_file(npz)

    doc = {
        "stage": "illustrative setup-figure data",
        "role": "introduces the model system; enters no loss, no ratio, no "
                "threshold and no decision",
        "contains_sinn_output": False,
        "pathwise_comparison_implied": False,
        "retrained_anything": False,
        "rebuilt_archive": False,
        "reclassified_anything": False,
        "reselected_checkpoint": False,
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_setup_data.py")
                          if os.path.exists(n)},
        "trajectory": {
            "segment": "fitting, F = [%d, %d)" % S.SPLIT_F,
            "segment_rationale": "the setup figure introduces the system, so "
                                 "it shows fitting data and never the "
                                 "held-out segment",
            "index": int(args.trajectory),
            "n_trajectories_in_segment": int(n_traj),
            "window": list(S.TARGET_G_IN),
            "physical_time": [int(t_lo), int(t_hi)],
            "coordinates": "physical density rho, not standardized",
            "symbol": "rho",
            "y_axis_label": "population density",
            "mean": {"A": float(rho[:, 0].mean()),
                     "B": float(rho[:, 1].mean())},
            "range": {"A": [float(rho[:, 0].min()), float(rho[:, 0].max())],
                      "B": [float(rho[:, 1].min()), float(rho[:, 1].max())]},
        },
        "lattice": doc_lattice,
        "artifact": {"path": os.path.abspath(npz), "sha256": npz_sha},
    }
    p = os.path.join(args.out, "setup_data.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "setup_data.sha256"), "w") as fh:
        fh.write(digest + "\n")

    print("\nwrote %s (sha %s...)" % (npz, npz_sha[:12]))
    print("wrote %s (sha %s...)" % (p, digest[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
