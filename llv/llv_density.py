#!/usr/bin/env python3
"""Stage 6 of spec section 13: the Supplementary joint-density data.

Spec section 10.1, as written:

  Use the same representative checkpoint selected in section 9. Generate one
  separate N_vis = 12000 continuous 1200-step ensemble with the visualization
  stream, discard raw [0:400), and display only G_out = [800:1200). Compare it
  with T_ev,out, also 12,000 trajectories. Reference blocks do not enter this
  figure.

  Compute both bivariate KDEs in standardized (x_A, x_B) coordinates with the
  same frozen h_2, and compute the two marginal KDEs with frozen h. For a
  figure in physical (rho_A, rho_B) coordinates, transform the axes using
  rho_i = s_i x_i + mu_i and transform the bivariate density by the Jacobian
  f_rho = f_x / (s_A s_B); transform each marginal by 1 / s_i. The two panels
  are labeled Target and SINN, never Reference. This visualization-only
  ensemble enters no numerical summary or decision.

This stage therefore produces DATA ONLY, saved as one .npz plus a JSON record.
The figure itself is drawn by the figure builder, which per section 13 reads
only saved evaluation products. Nothing here is a numerical summary: no loss,
no ratio, no threshold, no admissibility rate and no decision.

The ensemble size is written N_vis = 12000 in all human-readable output; the
symbol B is not used for an ensemble size.

usage:
    ARCHIVE=... SPEC=... python3 llv_density.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --out density_llv
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

PANEL_LABELS = ("Target", "SINN")
FORBIDDEN_LABEL = "Reference"


def trapezoid(y, x, axis: int = -1) -> np.ndarray:
    """Trapezoidal integration, independent of the NumPy version.

    `np.trapz` was removed in NumPy 2.0 in favour of `np.trapezoid`, and the
    two are not both present on every machine this runs on, so the rule is
    spelled out here rather than depending on which one exists.
    """
    a = np.moveaxis(np.asarray(y, np.float64), axis, -1)
    dx = np.diff(np.asarray(x, np.float64))
    return ((a[..., :-1] + a[..., 1:]) * 0.5 * dx).sum(axis=-1)


def kde2d(samples, bandwidth: float, grid: np.ndarray, device,
          sample_chunk: int = 20000) -> np.ndarray:
    """Bivariate Gaussian KDE on grid x grid, standardized coordinates.

    f(z1, z2) = 1/(M h^2 2pi) sum_m exp(-((z1-s1m)^2 + (z2-s2m)^2) / (2h^2))

    The Gaussian kernel is separable, so each sample chunk contributes
    A @ B.T with A[i,m] = exp(-(z1_i - s1_m)^2 / 2h^2) and likewise B. That is
    the same literal kernel sum as the 1D estimator -- no binning and no FFT --
    evaluated as two (G x m) temporaries and one matmul instead of a
    (G x G x m) tensor.

    Returns (len(grid), len(grid)) float64, index [i, j] = f(grid[i], grid[j]).
    """
    if not np.isfinite(bandwidth) or bandwidth <= 0 or bandwidth >= 1.0:
        raise ValueError("bandwidth %r is not a positive value below 1; the "
                         "legacy exponent-denominator interface is forbidden"
                         % (bandwidth,))
    v = np.asarray(samples, np.float64).reshape(-1, 2)
    m_tot = v.shape[0]
    g = torch.as_tensor(np.asarray(grid, np.float64), dtype=torch.float64,
                        device=device)
    n_g = int(g.shape[0])
    out = torch.zeros(n_g, n_g, dtype=torch.float64, device=device)
    for lo in range(0, m_tot, sample_chunk):
        blk = torch.as_tensor(np.ascontiguousarray(v[lo:lo + sample_chunk]),
                              dtype=torch.float64, device=device)
        a = torch.exp(-0.5 * ((g[:, None] - blk[None, :, 0]) / bandwidth) ** 2)
        b = torch.exp(-0.5 * ((g[:, None] - blk[None, :, 1]) / bandwidth) ** 2)
        out += a @ b.T
    out /= m_tot * bandwidth * bandwidth * 2.0 * np.pi
    return out.cpu().numpy()


def marginal_consistency(f2d: np.ndarray, samples, bandwidth: float,
                         grid: np.ndarray) -> dict:
    """Check the bivariate estimator against the 1D one at the same bandwidth.

    Integrating the separable kernel over one coordinate gives exactly the 1D
    Gaussian KDE at the same bandwidth, provided the grid covers the kernel
    mass. Trapezoidal integration of `f2d` must therefore reproduce
    `kde_direct(.., bandwidth)` up to quadrature error and whatever mass sits
    outside the grid. This is a correctness check on the matmul formulation,
    not a statistical statement.
    """
    v = np.asarray(samples, np.float64).reshape(-1, 2)
    out = {}
    for axis, comp, name in ((1, 0, "x_A"), (0, 1, "x_B")):
        num = trapezoid(f2d, grid, axis=axis)
        ref = S.kde_direct(v[:, comp:comp + 1], bandwidth=bandwidth,
                           grid=grid)[:, 0]
        den = float(np.linalg.norm(ref))
        out[name] = {
            "relative_l2": float(np.linalg.norm(num - ref) / den),
            "max_abs": float(np.max(np.abs(num - ref))),
            "integral_of_1d_kde": float(trapezoid(ref, grid)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--out", default="density_llv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--sample-chunk", type=int, default=20000)
    args = ap.parse_args()

    dev = args.device
    dtype = torch.float64

    # ---- the frozen representative, verified, never reselected -----------
    sel_path = os.path.join(args.select_dir, "representative.json")
    rec = os.path.join(args.select_dir, "representative.sha256")
    for p in (sel_path, rec):
        if not os.path.exists(p):
            raise SystemExit("%r is missing; the frozen selection cannot be "
                             "verified" % p)
    with open(sel_path) as fh:
        sel = json.load(fh)
    sel_sha = S.sha256_file(sel_path)
    if open(rec).read().strip() != sel_sha:
        raise SystemExit("the representative selection changed since it was "
                         "frozen")
    run = int(sel["representative_run"])

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    os.makedirs(args.out, exist_ok=True)
    print("LLV Supplementary joint-density data, spec section 13 stage 6")
    print("device %s  float64" % dev)
    print("representative run %d, checkpoint iteration %d (frozen, sha %s...)"
          % (run, sel["representative_checkpoint_iteration"], sel_sha[:12]))
    print("N_vis = %d, visualization stream seed %d"
          % (S.B_VIS, S.SEED_BASE["density_visualization"] + run))
    print("frozen bandwidths: h = %.15f (marginals), h2 = %.17f (bivariate)"
          % (S.BANDWIDTH, S.BANDWIDTH_2D))

    # ---- the visualization ensemble --------------------------------------
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
    model = M.LLVGenerator(config=payload["metadata"]["config"],
                           dtype=dtype).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in
                           payload["best"]["model_state"].items()})
    model.eval()
    streams = ST.RunStreams(run, device=dev)

    print("\ngenerating one continuous %d-step ensemble, N_vis = %d"
          % (S.RAW_ROLLOUT, S.B_VIS), flush=True)
    t0 = time.time()
    with torch.no_grad():
        noise = streams.noise("density_visualization", S.RAW_ROLLOUT,
                              S.B_VIS).to(dtype)
        out, _ = model(noise)
        del noise
    raw = out.detach().cpu().numpy().astype(np.float64)
    del out
    win = S.extract_windows(raw)
    gen_out = np.ascontiguousarray(win["G_out"])      # display only G_out
    del raw, win
    print("  G_out %s  (%.0fs)" % (gen_out.shape, time.time() - t0),
          flush=True)

    # ---- the target, same window, same 12000 trajectories ----------------
    tev_out = S.standardize(loader.open_final(S.TARGET_G_OUT), mu, sd)
    if tev_out.shape[1] != S.B_VIS:
        raise SystemExit("T_ev,out has %d trajectories, expected %d"
                         % (tev_out.shape[1], S.B_VIS))
    print("  T_ev,out %s" % (tev_out.shape,))

    # ---- bivariate and marginal densities in standardized coordinates ----
    dens = {}
    for name, block in (("Target", tev_out), ("SINN", gen_out)):
        print("\n%s: bivariate KDE at h2 on %dx%d grid"
              % (name, S.GRID_N, S.GRID_N), flush=True)
        t0 = time.time()
        f2 = kde2d(block, S.BANDWIDTH_2D, S.GRID, dev, args.sample_chunk)
        print("  %.0fs; checking against the 1D estimator at h2" % (
            time.time() - t0), flush=True)
        chk = marginal_consistency(f2, block, S.BANDWIDTH_2D, S.GRID)
        for k, c in chk.items():
            print("    marginalised over the other axis vs kde_direct(%s): "
                  "relative %.3e  max abs %.3e" % (k, c["relative_l2"],
                                                   c["max_abs"]))
        m1 = S.kde_eval(block, bandwidth=S.BANDWIDTH)        # (G, 2) at h
        dens[name] = {"f2d_x": f2, "marg_x": m1, "check": chk}

    # ---- physical coordinates, with the Jacobian -------------------------
    rho_grid = [S.GRID * sd[i] + mu[i] for i in (0, 1)]
    jac2 = 1.0 / (sd[0] * sd[1])
    payload_npz = {
        "grid_x": S.GRID,
        "grid_rho_A": rho_grid[0],
        "grid_rho_B": rho_grid[1],
        "bandwidth_h": S.BANDWIDTH,
        "bandwidth_h2": S.BANDWIDTH_2D,
        "mu": mu, "s": sd,
        "jacobian_bivariate": jac2,
        "jacobian_marginal_A": 1.0 / sd[0],
        "jacobian_marginal_B": 1.0 / sd[1],
    }
    for name in PANEL_LABELS:
        d = dens[name]
        payload_npz["f2d_x_" + name] = d["f2d_x"]
        payload_npz["f2d_rho_" + name] = d["f2d_x"] * jac2
        payload_npz["marg_x_A_" + name] = d["marg_x"][:, 0]
        payload_npz["marg_x_B_" + name] = d["marg_x"][:, 1]
        payload_npz["marg_rho_A_" + name] = d["marg_x"][:, 0] / sd[0]
        payload_npz["marg_rho_B_" + name] = d["marg_x"][:, 1] / sd[1]

    npz = os.path.join(args.out, "joint_density.npz")
    np.savez_compressed(npz, **payload_npz)
    npz_sha = S.sha256_file(npz)

    doc = {
        "stage": "Supplementary joint-density data (spec 10.1)",
        "role": "visualization only; enters no numerical summary, no loss, "
                "no ratio, no threshold and no decision",
        "figure_drawn_here": False,
        "figure_note": "the figure is drawn by the figure builder, which "
                       "reads only saved evaluation products",
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "representative_run": run,
        "representative_checkpoint_iteration":
            int(sel["representative_checkpoint_iteration"]),
        "checkpoint_path": os.path.abspath(path),
        "checkpoint_file_sha256": ckpt_sha,
        "checkpoint_reselected": False,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_gates.py",
                           "llv_model.py", "llv_density.py")
                          if os.path.exists(n)},
        "ensemble": {
            "N_vis": S.B_VIS,
            "prose_symbol": "N_vis",
            "raw_rollout": S.RAW_ROLLOUT,
            "stream": "density_visualization",
            "seed": S.SEED_BASE["density_visualization"] + run,
            "discarded": list(S.SLICE_WARMUP),
            "displayed_window": list(S.SLICE_G_OUT),
            "displayed_window_name": "G_out",
            "separate_from_final_generation": True,
            "note": "a separate ensemble from the final-evaluation rollout, "
                    "drawn from its own stream",
        },
        "target": {"set": "T_ev,out", "indices": list(S.SPLIT_T_EV),
                   "window": list(S.TARGET_G_OUT),
                   "n_trajectories": int(tev_out.shape[1])},
        "reference_blocks_used": False,
        "estimators": {
            "bivariate": "literal separable Gaussian kernel sum at h2, "
                         "evaluated chunk-wise as A @ B.T; no binning, no FFT",
            "marginal": "the canonical 1D Gaussian kernel sum at h",
            "grid": [S.GRID_LO, S.GRID_HI], "grid_n": S.GRID_N,
            "coordinates": "computed in standardized (x_A, x_B)",
        },
        "physical_transform": {
            "axes": "rho_i = s_i x_i + mu_i",
            "bivariate_density": "f_rho = f_x / (s_A s_B)",
            "marginal_density": "f_rho_i = f_x_i / s_i",
            "jacobian_bivariate": jac2,
            "jacobian_marginal": [1.0 / sd[0], 1.0 / sd[1]],
        },
        "panel_labels": list(PANEL_LABELS),
        "forbidden_label": FORBIDDEN_LABEL,
        "label_note": "the two panels are labelled Target and SINN; the word "
                      "Reference must not appear in this figure or caption",
        "marginal_consistency_check": {
            name: dens[name]["check"] for name in PANEL_LABELS},
        "check_note": "integrating the separable bivariate kernel over one "
                      "coordinate equals the 1D Gaussian KDE at the same "
                      "bandwidth, so this validates the matmul formulation. "
                      "It is a correctness check, not a statistical result.",
        "artifact": {"path": os.path.abspath(npz), "sha256": npz_sha},
        "torch_version": torch.__version__,
        "device": dev,
    }
    p = os.path.join(args.out, "joint_density.json")
    with open(p, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "joint_density.sha256"), "w") as fh:
        fh.write(digest + "\n")

    print("\nwrote %s (sha %s...)" % (npz, npz_sha[:12]))
    print("wrote %s (sha %s...)" % (p, digest[:12]))
    print("panels: %s and %s; %r is not used."
          % (PANEL_LABELS[0], PANEL_LABELS[1], FORBIDDEN_LABEL))
    print("no numerical summary, loss, ratio, threshold or decision was "
          "produced from this ensemble.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
