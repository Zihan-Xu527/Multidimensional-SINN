#!/usr/bin/env python3
"""Stage 5 of spec section 13: the 10^6-step stability evaluator.

Spec section 11, second half, as written:

  The stability calculation uses the representative run and 50 independently
  generated trajectories. Starting from zero recurrent states, discard 400
  warm-up steps and then advance for 10^6 evaluated steps without reset.
  At post-warm-up steps 10^5, 2x10^5, ..., 10^6, retain the preceding 10^4
  steps for statistical summaries. Accumulate finiteness, violation count,
  maximum exceedance, and first violation/non-finite step ONLINE over all
  10^6 evaluated steps, not only the ten retained windows.

  For each of the ten retained windows, save the float64 pooled mean and
  standard deviation of x_A, x_B, the finite physical-coordinate minima and
  maxima, the three rates and maximum exceedance above, and the five relative
  family losses against the fixed T_ev,in target statistics. For these family
  statistics, split each continuously generated 10,000-step window into 25
  contiguous, non-overlapping 400-step blocks without resetting the recurrent
  state. Together with the 50 parallel trajectories this gives 1,250 analysis
  blocks of length T = 400, so the correlation estimator and lag taper match
  the target endpoint protocol. Use K = 400 and the frozen KDE settings.

  These ten losses describe drift of stationary statistics; they have no
  reference-error denominator, no threshold, and no interpretation as pathwise
  or 10^6-step target accuracy.

  The long-run calculation reports stability of this one validation-selected
  checkpoint. It supplies neither a success criterion nor evidence of
  10^6-step target accuracy.

The ensemble size is written N = 50 in all human-readable output. No r_s is
computed here, no threshold is applied, and the checkpoint is read from the
frozen representative selection -- never reselected.

No clipping, no projection, no rounding to N_site^-1, and no state reset at
any point after the warm-up.

usage:
    ARCHIVE=... SPEC=... python3 llv_stability.py \
        --manifest-dir llv_manifest_v2 --runs-dir formal_llv \
        --select-dir select_formal --out stability_llv
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

#: spec section 11
N_TRAJ = 50
N_WARMUP = 400
N_STEPS = 1_000_000
CHECKPOINT_EVERY = 100_000          # 10^5, 2x10^5, ..., 10^6
RETAIN_STEPS = 10_000               # the preceding 10^4 steps
BLOCKS_PER_WINDOW = 25              # 25 contiguous 400-step blocks
#: generation chunk: equals RETAIN_STEPS so the retained windows align exactly
#: with chunk boundaries and nothing has to be stitched.
CHUNK = RETAIN_STEPS


def reshape_blocks(window: np.ndarray) -> np.ndarray:
    """(10000, 50, 2) -> (400, 1250, 2): 25 contiguous blocks x 50 trajectories.

    The blocks are contiguous and non-overlapping and the recurrent state was
    never reset between them; stacking them along the batch axis is only a
    reindexing of one continuous rollout.
    """
    n_t, n_b, dim = window.shape
    if n_t != RETAIN_STEPS or n_b != N_TRAJ:
        raise ValueError("window shape %r" % (window.shape,))
    parts = [window[i * S.T_WIN:(i + 1) * S.T_WIN]
             for i in range(BLOCKS_PER_WINDOW)]
    out = np.concatenate(parts, axis=1)
    if out.shape != (S.T_WIN, BLOCKS_PER_WINDOW * N_TRAJ, dim):
        raise AssertionError("block reshape produced %r" % (out.shape,))
    return out


def finite_range(rho: np.ndarray, finite: np.ndarray, component: int) -> list:
    """[min, max] of one physical component over finite states only.

    A retained window in which no state is finite returns [NaN, NaN]; calling
    min()/max() on the empty selection would raise instead.
    """
    if not finite.any():
        return [float("nan"), float("nan")]
    values = rho[..., component][finite]
    return [float(values.min()), float(values.max())]


class OnlineAdmissibility:
    """Accumulates over all 10^6 evaluated steps, never only the windows."""

    def __init__(self):
        self.n_samples = 0
        self.n_nonfinite = 0
        self.n_violating = 0
        self.max_exceedance = -np.inf
        self.first_nonfinite_step = None
        self.first_violation_step = None
        self.n_finite_seen = 0

    def update(self, rho: np.ndarray, step_offset: int):
        """rho: (steps, N, 2) physical coordinates; step_offset is the
        post-warm-up index of rho[0], one-based."""
        finite = np.isfinite(rho).all(axis=-1)
        self.n_samples += int(finite.size)
        n_nf = int(finite.size - finite.sum())
        self.n_nonfinite += n_nf
        self.n_finite_seen += int(finite.sum())

        if n_nf and self.first_nonfinite_step is None:
            rows = np.flatnonzero((~finite).any(axis=1))
            self.first_nonfinite_step = int(step_offset + rows[0])

        if finite.any():
            v = np.full(finite.shape, -np.inf)
            v[finite] = S.exceedance(rho[finite])
            self.max_exceedance = max(self.max_exceedance,
                                      float(np.max(v[finite])))
            viol = finite & (v > 0.0)
            n_v = int(viol.sum())
            self.n_violating += n_v
            if n_v and self.first_violation_step is None:
                rows = np.flatnonzero(viol.any(axis=1))
                self.first_violation_step = int(step_offset + rows[0])

    def result(self) -> dict:
        t = self.n_samples
        return {
            "n_state_time_samples": t,
            "n_nonfinite": self.n_nonfinite,
            "n_finite_violating": self.n_violating,
            "nonfinite_rate": self.n_nonfinite / t,
            "constraint_rate": self.n_violating / t,
            "invalid_rate": (self.n_nonfinite + self.n_violating) / t,
            "max_exceedance": (float("nan") if self.n_finite_seen == 0
                               else max(self.max_exceedance, 0.0)),
            "max_exceedance_raw_over_finite": (
                None if self.n_finite_seen == 0 else self.max_exceedance),
            "first_nonfinite_step": self.first_nonfinite_step,
            "first_violation_step": self.first_violation_step,
            "accumulated_over": "all %d evaluated post-warm-up steps"
                                % N_STEPS,
        }


def selftest_nonfinite(verbose: bool = True) -> dict:
    """Fixtures for the degenerate windows, run before any generation.

    (a) a window in which NO state is finite must complete and must record
        nonfinite_rate = 1, invalid_rate = 1, max_exceedance = NaN and
        [NaN, NaN] physical ranges;
    (b) a mixed window must report the correct first non-finite step and
        first violation step on the one-based post-warm-up index.
    """
    rho = np.full((7, 3, 2), np.nan)
    fin = np.isfinite(rho).all(axis=-1)
    ra, rb = finite_range(rho, fin, 0), finite_range(rho, fin, 1)
    if not (all(np.isnan(ra)) and all(np.isnan(rb))):
        raise AssertionError("all-non-finite physical_range %r %r" % (ra, rb))

    adm = S.admissibility(rho)
    if adm["nonfinite_rate"] != 1.0 or adm["invalid_rate"] != 1.0:
        raise AssertionError("all-non-finite rates %r" % (adm,))
    if not np.isnan(adm["max_exceedance"]):
        raise AssertionError("max_exceedance should be NaN, got %r"
                             % adm["max_exceedance"])

    on = OnlineAdmissibility()
    on.update(rho, 1)
    r = on.result()
    if not (r["nonfinite_rate"] == 1.0 and r["invalid_rate"] == 1.0
            and np.isnan(r["max_exceedance"])
            and r["first_nonfinite_step"] == 1
            and r["first_violation_step"] is None):
        raise AssertionError("online all-non-finite result %r" % (r,))

    mixed = np.full((7, 3, 2), 0.3)
    mixed[3, 1, :] = np.inf          # non-finite at index 3
    mixed[5, 0, 0] = -0.2            # finite violation (rho_A < 0) at index 5
    on2 = OnlineAdmissibility()
    on2.update(mixed, 2)             # rho[0] is post-warm-up step 2
    r2 = on2.result()
    if r2["first_nonfinite_step"] != 5 or r2["first_violation_step"] != 7:
        raise AssertionError("first-event steps %r" % (r2,))
    if abs(r2["max_exceedance"] - 0.2) > 1e-12:
        raise AssertionError("max_exceedance %r" % r2["max_exceedance"])
    if r2["n_nonfinite"] != 1 or r2["n_finite_violating"] != 1:
        raise AssertionError("mixed counts %r" % (r2,))

    out = {"all_nonfinite_window": {
               "nonfinite_rate": adm["nonfinite_rate"],
               "invalid_rate": adm["invalid_rate"],
               "max_exceedance": adm["max_exceedance"],
               "physical_range_rho_A": ra, "physical_range_rho_B": rb},
           "mixed_window": {
               "first_nonfinite_step": r2["first_nonfinite_step"],
               "first_violation_step": r2["first_violation_step"],
               "max_exceedance": r2["max_exceedance"]}}
    if verbose:
        print("selftest: all-non-finite window -> nonfinite 1.0, invalid 1.0, "
              "max exceedance NaN, ranges [nan, nan]")
        print("selftest: mixed window -> first non-finite step 5, first "
              "violation step 7, max exceedance 0.2")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--runs-dir", default="formal_llv")
    ap.add_argument("--select-dir", default="select_formal")
    ap.add_argument("--out", default="stability_llv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float64",
                                                           "float32"))
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = args.device

    # ---- the checkpoint comes from the frozen selection, never reselected --
    sel_path = os.path.join(args.select_dir, "representative.json")
    if not os.path.exists(sel_path):
        raise SystemExit("frozen representative selection %r is missing"
                         % sel_path)
    with open(sel_path) as fh:
        sel = json.load(fh)
    sel_sha = S.sha256_file(sel_path)
    rec = os.path.join(args.select_dir, "representative.sha256")
    if not os.path.exists(rec):
        raise SystemExit(
            "the recorded selection checksum %r is missing. The frozen "
            "selection cannot be verified, so this stage will not run." % rec)
    if open(rec).read().strip() != sel_sha:
        raise SystemExit("the representative selection changed since it was "
                         "frozen (recorded %s..., now %s...)"
                         % (open(rec).read().strip()[:12], sel_sha[:12]))
    run = int(sel["representative_run"])

    loader = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    man = loader.assert_unlocked()
    mu = np.asarray(man["standardization"]["mu"], np.float64)
    sd = np.asarray(man["standardization"]["s"], np.float64)

    os.makedirs(args.out, exist_ok=True)
    print("LLV 10^6-step stability evaluator, spec section 13 stage 5")
    print("device %s  dtype %s" % (dev, args.dtype))
    print("representative run %d, checkpoint iteration %d "
          "(frozen selection, sha %s...)"
          % (run, sel["representative_checkpoint_iteration"], sel_sha[:12]))
    print("N = %d trajectories, %d warm-up steps discarded, %d evaluated "
          "steps, no reset" % (N_TRAJ, N_WARMUP, N_STEPS))
    print()
    fixtures = selftest_nonfinite()

    # ---- fixed target statistics: T_ev,in, computed once -----------------
    fit = S.standardize(loader.fitting(S.TARGET_G_IN), mu, sd)
    scales = S.projection_scales(fit)
    del fit
    print("\ncomputing the fixed T_ev,in target family statistics", flush=True)
    t0 = time.time()
    tev_in = S.standardize(loader.open_final(S.TARGET_G_IN), mu, sd)
    target = S.family_arrays(tev_in, scales)
    del tev_in
    print("  %.0fs" % (time.time() - t0), flush=True)

    # ---- the model, and the warm-up --------------------------------------
    path = os.path.join(args.runs_dir, "run_%d.pt" % run)
    ckpt_sha = S.sha256_file(path)
    payload = torch.load(path, map_location="cpu")
    if int(payload["run"]) != run:
        raise SystemExit("checkpoint %r carries run %r but the frozen "
                         "selection names run %d"
                         % (path, payload["run"], run))
    if int(payload["best"]["iteration"]) != \
            int(sel["representative_checkpoint_iteration"]):
        raise SystemExit("checkpoint iteration in %r disagrees with the "
                         "frozen selection" % path)
    cfg = payload["metadata"]["config"]
    model = M.LLVGenerator(config=cfg, dtype=dtype).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in
                           payload["best"]["model_state"].items()})
    model.eval()
    streams = ST.RunStreams(run, device=dev)

    print("\nwarm-up: %d steps from zero recurrent states, discarded"
          % N_WARMUP, flush=True)
    with torch.no_grad():
        noise = streams.noise("stability", N_WARMUP, N_TRAJ).to(dtype)
        _, state = model(noise)

    # ---- the long run ----------------------------------------------------
    online = OnlineAdmissibility()
    windows = []
    n_chunks = N_STEPS // CHUNK
    if N_STEPS % CHUNK or CHECKPOINT_EVERY % CHUNK:
        raise AssertionError("chunking does not align with the retained "
                             "windows")
    chunks_per_checkpoint = CHECKPOINT_EVERY // CHUNK
    print("advancing %d steps in %d chunks of %d (state carried across every "
          "boundary)" % (N_STEPS, n_chunks, CHUNK), flush=True)

    t0 = time.time()
    for c in range(n_chunks):
        step0 = c * CHUNK + 1                     # one-based post-warm-up
        with torch.no_grad():
            noise = streams.noise("stability", CHUNK, N_TRAJ).to(dtype)
            out, state = model(noise, state)      # state carried, never reset
        x = out.detach().cpu().numpy().astype(np.float64)
        rho = S.inverse_standardize(x, mu, sd)    # no clip, project or round
        online.update(rho, step0)

        if (c + 1) % chunks_per_checkpoint == 0:
            end = (c + 1) * CHUNK
            adm = S.admissibility(rho)
            fin = np.isfinite(rho).all(axis=-1)
            blocks = reshape_blocks(x)
            fam = S.family_arrays(blocks, scales)
            loss = {s: S.relative_loss(fam[s], target[s]) for s in S.FAMILIES}
            mean = x.mean(axis=(0, 1))
            std = np.sqrt(((x - mean) ** 2).mean(axis=(0, 1)))
            rec_w = {
                "checkpoint_step": end,
                "window_steps": [end - RETAIN_STEPS + 1, end],
                "n_analysis_blocks": BLOCKS_PER_WINDOW * N_TRAJ,
                "block_length": S.T_WIN,
                "pooled_mean_x": {"A": float(mean[0]), "B": float(mean[1])},
                "pooled_sd_x": {"A": float(std[0]), "B": float(std[1])},
                "physical_range": {
                    "rho_A": finite_range(rho, fin, 0),
                    "rho_B": finite_range(rho, fin, 1)},
                "admissibility": adm,
                "family_loss_vs_T_ev_in": loss,
                "loss_note": "no reference denominator, no threshold, not an "
                             "accuracy statement about the 10^6-step run",
            }
            windows.append(rec_w)
            print("  step %7d  mean (%+.5f, %+.5f)  sd (%.5f, %.5f)  "
                  "invalid %.2e  L: " % (end, mean[0], mean[1], std[0], std[1],
                                         adm["invalid_rate"])
                  + " ".join("%s %.3e" % (s, loss[s]) for s in S.FAMILIES)
                  + "  %.0fs" % (time.time() - t0), flush=True)

    on = online.result()
    print("\nonline accumulation over all %d evaluated steps "
          "(%d state-time samples)" % (N_STEPS, on["n_state_time_samples"]))
    print("  non-finite        %d   rate %.3e" % (on["n_nonfinite"],
                                                  on["nonfinite_rate"]))
    print("  finite violating  %d   rate %.3e" % (on["n_finite_violating"],
                                                  on["constraint_rate"]))
    print("  invalid rate      %.3e" % on["invalid_rate"])
    print("  max exceedance    %.6e" % on["max_exceedance"])
    print("  first non-finite step  %s" % on["first_nonfinite_step"])
    print("  first violation step   %s" % on["first_violation_step"])

    # ---- drift description, no threshold ---------------------------------
    def spread(vals):
        v = np.asarray(vals, np.float64)
        return {"min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()),
                "median": float(np.median(v))}

    drift = {
        "pooled_mean_x_A": spread([w["pooled_mean_x"]["A"] for w in windows]),
        "pooled_mean_x_B": spread([w["pooled_mean_x"]["B"] for w in windows]),
        "pooled_sd_x_A": spread([w["pooled_sd_x"]["A"] for w in windows]),
        "pooled_sd_x_B": spread([w["pooled_sd_x"]["B"] for w in windows]),
        "family_loss": {s: spread([w["family_loss_vs_T_ev_in"][s]
                                   for w in windows]) for s in S.FAMILIES},
        "note": "descriptive spread across the ten retained windows. No "
                "threshold is defined and none is applied; whether this "
                "constitutes drift is a reporting judgement, not a test.",
    }
    print("\nacross the ten retained windows (descriptive, no threshold)")
    print("  pooled mean x_A  range %.3e   x_B  range %.3e"
          % (drift["pooled_mean_x_A"]["range"],
             drift["pooled_mean_x_B"]["range"]))
    print("  pooled sd   x_A  range %.3e   x_B  range %.3e"
          % (drift["pooled_sd_x_A"]["range"],
             drift["pooled_sd_x_B"]["range"]))
    for s in S.FAMILIES:
        d = drift["family_loss"][s]
        print("  L[%-5s] min %.3e  median %.3e  max %.3e  (max/min %.2f)"
              % (s, d["min"], d["median"], d["max"],
                 d["max"] / d["min"] if d["min"] > 0 else float("nan")))

    out_doc = {
        "stage": "10^6-step stability evaluator",
        "specification": man.get("specification"),
        "specification_sha256": man.get("specification_sha256"),
        "archive_sha256": man["archive_sha256"],
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "representative_selection_sha256": sel_sha,
        "representative_run": run,
        "representative_checkpoint_iteration":
            int(sel["representative_checkpoint_iteration"]),
        "checkpoint_reselected": False,
        "checkpoint_path": os.path.abspath(path),
        "checkpoint_file_sha256": ckpt_sha,
        "verification": {
            "selection_file_sha256_matches_recorded": True,
            "run_id_matches_selection": True,
            "checkpoint_iteration_matches_selection": True,
            "checkpoint_file_sha256_compared_against_frozen_value": False,
            "note": "the frozen selection does not record a checksum of the "
                    "checkpoint file itself, so the checkpoint's SHA was not "
                    "compared against any frozen value. What was verified is "
                    "the selection file's own SHA against its recorded "
                    "checksum, the run id, and the checkpoint iteration. The "
                    "checkpoint file's current SHA is recorded above for "
                    "provenance."},
        "degenerate_window_selftest": fixtures,
        "source_sha256": {n: S.sha256_file(n) for n in
                          ("llv_spec.py", "llv_streams.py", "llv_gates.py",
                           "llv_torch.py", "llv_model.py", "llv_stability.py")
                          if os.path.exists(n)},
        "protocol": {
            "ensemble_size_N": N_TRAJ,
            "warmup_steps_discarded": N_WARMUP,
            "evaluated_steps": N_STEPS,
            "state_reset": "never, at any point after the warm-up",
            "post_processing": "no clipping, no projection, no rounding to "
                               "N_site^-1",
            "generation_chunk": CHUNK,
            "chunking_note": "chunking only bounds memory. The recurrent "
                             "state is carried across every chunk boundary, "
                             "and each chunk's noise comes from successive "
                             "draws on the same continuously advanced "
                             "generator stream, so no draw is reused and none "
                             "is skipped. No claim is made that the chunked "
                             "draws are bit-identical to one single large "
                             "draw; that was not tested and the protocol does "
                             "not require it.",
            "stability_stream_seed": S.SEED_BASE["stability"] + run,
            "retained_windows_at": [CHECKPOINT_EVERY * (k + 1)
                                    for k in range(N_STEPS
                                                   // CHECKPOINT_EVERY)],
            "retained_window_length": RETAIN_STEPS,
            "blocks_per_window": BLOCKS_PER_WINDOW,
            "analysis_blocks_per_window": BLOCKS_PER_WINDOW * N_TRAJ,
            "block_length_T": S.T_WIN,
            "K_lags": S.K_LAGS,
            "kde": {"grid_n": S.GRID_N, "bandwidth": S.BANDWIDTH},
            "target": "fixed T_ev,in family statistics",
        },
        "reporting_symbol": {
            "ensemble_size": N_TRAJ, "prose_symbol": "N",
            "note": "human-readable output uses N = 50; the symbol B is not "
                    "used for an ensemble size anywhere."},
        "online_accumulation": on,
        "retained_windows": windows,
        "drift_description": drift,
        "units": "the ten family losses are squared relative discrepancies "
                 "against a fixed target; they carry no reference denominator "
                 "and no r_s is defined here",
        "claim_boundary": "this reports stability of one validation-selected "
                          "checkpoint. It supplies neither a success "
                          "criterion nor evidence of 10^6-step target "
                          "accuracy.",
        "figure_policy": "one compact Supplementary table by default; a "
                         "stability figure only if there is clear temporal "
                         "drift",
        "torch_version": torch.__version__,
        "device": dev,
        "dtype": args.dtype,
        "wall_seconds": time.time() - t0,
    }
    p = os.path.join(args.out, "stability.json")
    with open(p, "w") as fh:
        json.dump(out_doc, fh, indent=2, default=float)
    digest = S.sha256_file(p)
    with open(os.path.join(args.out, "stability.sha256"), "w") as fh:
        fh.write(digest + "\n")

    # ---- the compact Supplementary table ---------------------------------
    tp = os.path.join(args.out, "stability_table.txt")
    with open(tp, "w") as fh:
        fh.write("Supplementary: 10^6-step stability, representative run %d, "
                 "checkpoint iteration %d, N = %d trajectories\n"
                 % (run, sel["representative_checkpoint_iteration"], N_TRAJ))
        fh.write("Each row summarises the 10^4 steps preceding the listed "
                 "step, split into 25 contiguous 400-step blocks (1250 "
                 "analysis blocks). Losses are squared relative "
                 "discrepancies against fixed T_ev,in statistics; they have "
                 "no reference denominator and no threshold.\n\n")
        hdr = ("%9s %10s %10s %9s %9s %11s %11s %11s %11s %11s %11s\n"
               % ("step", "mean x_A", "mean x_B", "sd x_A", "sd x_B",
                  "L_ACF", "L_CCF", "L_ACF2", "L_PDF", "L_PROJ",
                  "invalid"))
        fh.write(hdr)
        for w in windows:
            L = w["family_loss_vs_T_ev_in"]
            fh.write("%9d %10.5f %10.5f %9.5f %9.5f %11.4e %11.4e %11.4e "
                     "%11.4e %11.4e %11.4e\n"
                     % (w["checkpoint_step"], w["pooled_mean_x"]["A"],
                        w["pooled_mean_x"]["B"], w["pooled_sd_x"]["A"],
                        w["pooled_sd_x"]["B"], L["ACF"], L["CCF"], L["ACF2"],
                        L["PDF"], L["PROJ"],
                        w["admissibility"]["invalid_rate"]))
        fh.write("\nOnline over all %d evaluated steps (%d state-time "
                 "samples): non-finite %d, finite violating %d, invalid rate "
                 "%.3e, maximum exceedance %.6e, first non-finite step %s, "
                 "first violation step %s.\n"
                 % (N_STEPS, on["n_state_time_samples"], on["n_nonfinite"],
                    on["n_finite_violating"], on["invalid_rate"],
                    on["max_exceedance"], on["first_nonfinite_step"],
                    on["first_violation_step"]))

    print("\nwrote %s (sha %s...)" % (p, digest[:12]))
    print("wrote %s" % tp)
    print("no stability figure was produced; the compact table is the default "
          "deliverable per the figure policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
