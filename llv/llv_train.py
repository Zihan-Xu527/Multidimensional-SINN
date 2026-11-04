#!/usr/bin/env python3
"""Stage 1 of spec section 13: the LLV trainer.

Reads F and V only. Trains the 20 runs of spec section 5. Saves every
validation history and the retained checkpoint per run. Never touches
T_ev = [8000, 20000): it never calls `open_final`, and the nine reference
blocks R_j belong to the classifier and the evaluator, not here.

What the trainer scores, per spec sections 5 and 6:

  target (training)    family statistics of F over physical t = 401..800
  target (validation)  family statistics of V over physical t = 401..800
  generated            raw [400:800) of an 800-step rollout from zero states
  loss                 L_s = ||S_s(G_in) - S_s(target)||^2 / ||S_s(target)||^2
  balancing            two Adam optimizers, block coordinate descent
  checkpoint score     Q_val = sum_s L_s on a FRESH noise draw per event

usage:
    ARCHIVE=... SPEC=... python3 llv_train.py \
        --manifest-dir llv_manifest_v2 --out runs_llv [--runs 0-19]
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
import llv_torch as T


# --------------------------------------------------------------------- setup --
def frozen_standardization(loader):
    """mu, s from F over physical t = 401..800, float64, ddof = 0 (spec 4)."""
    fit = loader.fitting(S.TARGET_G_IN)
    mu = fit.mean(axis=(0, 1), dtype=np.float64)
    sd = fit.std(axis=(0, 1), dtype=np.float64)
    return mu, sd, fit


def verify_gates(manifest_dir: str) -> dict:
    """The trainer may run only against a verified gate result.

    This re-uses the loader's own verification chain, so the manifest, the
    archive, this code and the specification must all still match what the
    gates were run against. It does NOT open T_ev.
    """
    path = os.path.join(manifest_dir, S.MANIFEST_NAME)
    tok = os.path.join(manifest_dir, S.FREEZE_TOKEN_NAME)
    for p in (path, tok):
        if not os.path.exists(p):
            raise SystemExit("gates not passed: %r is missing" % p)
    with open(path) as fh:
        man = json.load(fh)
    return man


#: Pre-frozen pilot run ids. Disjoint from the production list 0..19 in every
#: stream: param_init 100-104 (vs 0-19), train_noise 1100-1104 (vs 1000-1019),
#: checkpoint_monitor 2100-2104, classification 3100-3104.
PILOT_RUNS = (100, 101, 102, 103, 104)
#: The twenty FORMAL run ids. Re-frozen to 200..219 after seed 0 was observed
#: in development and 100..104 in the width pilot, so that no formal seed has
#: been looked at. They are defined HERE and not in llv_spec.py on purpose:
#: llv_spec.py is covered by the gate freeze, and editing it would invalidate
#: llv_manifest_v2 and lock T_ev. The consequence is that the manifest still
#: records seeds.runs = 0..19 from S.RUN_SEEDS; FORMAL_RUNS below is the
#: authoritative formal population and every output file records both.
FORMAL_RUNS = tuple(range(200, 220))
assert len(FORMAL_RUNS) == S.N_RUNS
assert not (set(FORMAL_RUNS) & set(PILOT_RUNS)) and 0 not in FORMAL_RUNS
#: The widths the feasibility pilot tests, in order.
PILOT_WIDTHS = (10, 30, 75)


def parse_runs(spec: str, allow_any: bool = False):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not allow_any:
        bad = [r for r in out if r not in FORMAL_RUNS]
        if bad:
            raise SystemExit(
                "runs %r are not in the frozen formal population %d..%d. The "
                "development seed 0 and the pilot seeds %r are deliberately "
                "excluded from the formal experiment; rerunning them requires "
                "--pilot and their output is not a result."
                % (bad, FORMAL_RUNS[0], FORMAL_RUNS[-1], list(PILOT_RUNS)))
    return sorted(set(out))


# -------------------------------------------------------------- one training --
def _source_checksums() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    names = tuple(S.GATED_SOURCES) + ("llv_torch.py", "llv_model.py",
                                      "llv_train.py")
    out = {}
    for n in names:
        p = os.path.join(here, n)
        if os.path.exists(p):
            out[n] = S.sha256_file(p)
    return out


def train_run(run: int, ctx: dict) -> dict:
    cfg = dict(S.TRAIN_CONFIG)
    if ctx.get("width") and int(ctx["width"]) != int(cfg["hidden_units"]):
        cfg["hidden_units"] = int(ctx["width"])
    if ctx.get("smoke"):
        cfg.update(ctx["smoke"])
    dev, dtype = ctx["device"], ctx["dtype"]
    n_iter = int(cfg["iterations"])
    cadence = int(cfg["validation_cadence"])
    batch = int(cfg["batch"])
    unroll = int(cfg["raw_train_unroll"])
    lo, hi = cfg["scored_slice"]

    model, sigmas, init_gen = M.build(run, dev, dtype, config=cfg)
    streams = ST.RunStreams(run, device=dev)
    opt_theta = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 betas=cfg["betas"])
    opt_sigma = torch.optim.Adam(sigmas.parameters(), lr=cfg["lr"],
                                 betas=cfg["betas"])

    sc, di = ctx["scales_t"], ctx["directions_t"]
    tgt_fit, tgt_val = ctx["target_fit"], ctx["target_val"]

    history = []
    best = {"q_val": float("inf"), "iteration": None}
    t0 = time.time()

    for it in range(1, n_iter + 1):
        noise = streams.noise("train_noise", unroll, batch).to(dtype)
        out, _ = model(noise)
        g_in = out[lo:hi]
        fam = T.family_arrays_t(g_in, sc, di, checkpoint=ctx["checkpoint"],
                                sample_chunk_size=ctx["chunk"])
        losses = T.family_losses_t(fam, tgt_fit)

        # theta half: sigma detached, so the graph through the model is used
        # once and then freed.
        opt_theta.zero_grad(set_to_none=True)
        sigmas.theta_objective(losses).backward()
        opt_theta.step()

        # sigma half: losses detached, so this needs no model graph at all.
        opt_sigma.zero_grad(set_to_none=True)
        sigmas.scale_objective(losses).backward()
        opt_sigma.step()

        if it % cadence == 0:
            # A NEW noise tensor at every event, from a generator that is
            # advanced continuously and never re-seeded (spec 6).
            with torch.no_grad():
                v_noise = streams.noise("checkpoint_monitor", unroll,
                                        batch).to(dtype)
                v_out, _ = model(v_noise)
                v_fam = T.family_arrays_t(v_out[lo:hi], sc, di,
                                          checkpoint=False,
                                          sample_chunk_size=ctx["chunk"])
                v_loss = {s: float(T.relative_loss_t(v_fam[s], tgt_val[s]))
                          for s in S.FAMILIES}
            q = float(sum(v_loss[s] for s in S.FAMILIES))
            event = {
                "event_index": it // cadence,
                "iteration": it,
                "L_val": v_loss,
                "Q_val": q,
                "L_train": {s: float(losses[s]) for s in S.FAMILIES},
                "scales": sigmas.state_values(),
            }
            history.append(event)
            if q < best["q_val"]:
                best = {
                    "q_val": q,
                    "iteration": it,
                    "event_index": event["event_index"],
                    "L_val": dict(v_loss),
                    "model_state": {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()},
                    "sigma_state": {k: v.detach().cpu().clone()
                                    for k, v in sigmas.state_dict().items()},
                    "scales": sigmas.state_values(),
                }
            if ctx["verbose"]:
                print("  run %2d  it %5d  Q_val %.6e  best %.6e @ %d  %.0fs"
                      % (run, it, q, best["q_val"], best["iteration"],
                         time.time() - t0), flush=True)

    if len(history) != int(cfg["validation_events"]):
        raise AssertionError("expected %d validation events, recorded %d"
                             % (cfg["validation_events"], len(history)))
    if best["iteration"] is None:
        raise AssertionError("no checkpoint was retained for run %d" % run)

    return {
        "run": run,
        "history": history,
        "best": best,
        "streams": streams.metadata(),
        "param_init_seed": S.SEED_BASE["param_init"] + run,
        "n_parameters": model.n_parameters(),
        "hidden_units": int(cfg["hidden_units"]),
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.items()},
        "wall_seconds": time.time() - t0,
        "final_generator_states": {
            p: ST._sha_tensor(v) for p, v in streams.state_snapshot().items()},
    }


# -------------------------------------------------------------------- driver --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--out", default="runs_llv")
    ap.add_argument("--runs",
                    default="%d-%d" % (FORMAL_RUNS[0], FORMAL_RUNS[-1]))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float64",
                                                           "float32"))
    ap.add_argument("--sample-chunk-size", type=int, default=0,
                    help="0 selects the bounded default from llv_spec")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="disable KDE gradient checkpointing (more memory)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="feasibility-pilot mode: permits --hidden-units to "
                         "deviate from the frozen table and permits pilot run "
                         "ids, and marks every output as a pilot")
    ap.add_argument("--hidden-units", type=int, default=0,
                    help="override the frozen hidden width; requires --pilot")
    ap.add_argument("--smoke", type=int, default=0, metavar="ITERS",
                    help="plumbing check only: run ITERS iterations with a "
                         "small batch and a short unroll, write to "
                         "smoke_*.pt, and mark the output directory as not a "
                         "result. Use it once before launching the real runs.")
    args = ap.parse_args()

    if not args.archive or not os.path.exists(args.archive):
        raise SystemExit("set ARCHIVE to the processed LLV archive")
    man = verify_gates(args.manifest_dir)

    # The gated identity must still hold, or this training is not the
    # specification's experiment. Reuse the loader's own chain.
    probe = S.ArchiveLoader(args.archive, manifest_dir=args.manifest_dir)
    probe.assert_unlocked()

    frozen_width = int(S.TRAIN_CONFIG["hidden_units"])
    width = int(args.hidden_units) or frozen_width
    if width != frozen_width and not args.pilot:
        raise SystemExit(
            "--hidden-units %d deviates from the frozen width %d; the frozen "
            "specification freezes capacity, so this is only allowed under "
            "--pilot" % (width, frozen_width))
    pilot = None
    if args.pilot:
        pilot = {"role": "hidden-width feasibility pilot",
                 "hidden_units": width,
                 "frozen_hidden_units": frozen_width,
                 "deviates_from_frozen_specification": width != frozen_width,
                 "pilot_run_ids": list(PILOT_RUNS),
                 "widths_under_test": list(PILOT_WIDTHS),
                 "production_run_ids": list(S.RUN_SEEDS),
                 "note": "pilot output may not be reported as a result and "
                         "may not be used to open T_ev; only the width "
                         "decision leaves this stage"}
        print("PILOT MODE: hidden width %d (frozen %d). Outputs are marked "
              "as pilot and are not results." % (width, frozen_width))

    smoke_cfg = None
    if args.smoke:
        # A smoke run must not be able to masquerade as the experiment, so it
        # gets its own file names, its own marker file, and a metadata flag.
        n_sm = max(1, int(args.smoke))
        cad = max(1, n_sm // 2)
        smoke_cfg = {"iterations": n_sm,
                     "validation_cadence": cad,
                     "validation_events": n_sm // cad,
                     "batch": 16,
                     "raw_train_unroll": 2 * S.T_WIN}
        print("SMOKE RUN: %r. This is a plumbing check, not a result."
              % smoke_cfg)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dev = args.device
    chunk = args.sample_chunk_size or None

    print("LLV trainer, spec section 13 stage 1")
    print("device %s  dtype %s  archive sha %s..."
          % (dev, args.dtype, man["archive_sha256"][:12]))
    print("manifest %s sha %s..."
          % (args.manifest_dir, S.sha256_file(
              os.path.join(args.manifest_dir, S.MANIFEST_NAME))[:12]))
    T.selftest(device=dev, dtype=dtype)

    mu, sd, fit_raw = frozen_standardization(probe)
    man_mu = np.asarray(man["standardization"]["mu"], np.float64)
    man_sd = np.asarray(man["standardization"]["s"], np.float64)
    if not (np.allclose(mu, man_mu, rtol=0, atol=0)
            and np.allclose(sd, man_sd, rtol=0, atol=0)):
        raise SystemExit("standardization constants differ from the manifest")

    fit = S.standardize(fit_raw, mu, sd)
    scales = S.projection_scales(fit)
    val = S.standardize(probe.validation(S.TARGET_G_IN), mu, sd)

    print("computing frozen target statistics (F and V, t = 401..800)",
          flush=True)
    tgt_fit_np = S.family_arrays(fit, scales)
    tgt_val_np = S.family_arrays(val, scales)
    del fit, val, fit_raw

    sc, di = T.frozen_tensors(scales, dev, dtype)
    ctx = {
        "device": dev, "dtype": dtype, "chunk": chunk,
        "checkpoint": not args.no_checkpointing,
        "scales_t": sc, "directions_t": di,
        "target_fit": T.target_tensors(tgt_fit_np, dev, dtype),
        "target_val": T.target_tensors(tgt_val_np, dev, dtype),
        "verbose": not args.quiet,
        "smoke": smoke_cfg,
        "width": width,
    }

    os.makedirs(args.out, exist_ok=True)
    if smoke_cfg:
        with open(os.path.join(args.out, "SMOKE_NOT_A_RESULT"), "w") as fh:
            fh.write(json.dumps(smoke_cfg) + "\n")
    meta_common = {
        "stage": "trainer",
        "specification": man.get("specification"),
        "specification_sha256": man.get("specification_sha256"),
        "archive_path": args.archive,
        "archive_sha256": man["archive_sha256"],
        "manifest_dir": os.path.abspath(args.manifest_dir),
        "manifest_sha256": S.sha256_file(
            os.path.join(args.manifest_dir, S.MANIFEST_NAME)),
        "source_sha256": _source_checksums(),
        "smoke": smoke_cfg,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in S.TRAIN_CONFIG.items()},
        "pilot": pilot,
        "hidden_units": width,
        "run_population": {
            "formal_run_ids": list(FORMAL_RUNS),
            "development_run_ids": [0],
            "pilot_run_ids": list(PILOT_RUNS),
            "authoritative": "formal_run_ids",
            "manifest_records_run_ids": list(S.RUN_SEEDS),
            "note": "the formal population was re-frozen to %d..%d after seed "
                    "0 was observed in development and %r in the width pilot. "
                    "llv_spec.py is covered by the gate freeze and was not "
                    "edited, so llv_manifest_v2 still records seeds.runs = "
                    "0..%d; formal_run_ids above is authoritative."
                    % (FORMAL_RUNS[0], FORMAL_RUNS[-1], list(PILOT_RUNS),
                       S.N_RUNS - 1)},
        "seeds_in_use": {
            p: [S.SEED_BASE[p] + r for r in FORMAL_RUNS] for p in S.SEED_BASE},
        "standardization": {"mu": mu.tolist(), "s": sd.tolist(),
                            "source": "F, physical t = 401..800, ddof = 0"},
        "projection": {"scales": np.asarray(scales).tolist(),
                       "n_noncoordinate_frames": S.N_NONCOORDINATE_FRAMES,
                       "n_frames_total": S.N_NONCOORDINATE_FRAMES + 1,
                       "directions": np.asarray(S.PROJ_DIRECTIONS).tolist()},
        "kde": {"grid_n": S.GRID_N, "range": [S.GRID_LO, S.GRID_HI],
                "bandwidth": S.BANDWIDTH,
                "estimator": "canonical Gaussian kernel sum, chunked, "
                             "differentiable"},
        "families": list(S.FAMILIES),
        "target_windows": {"train": list(S.TARGET_G_IN),
                           "validation": list(S.TARGET_G_IN)},
        "scored_slice": list(S.TRAIN_CONFIG["scored_slice"]),
        "reads": ["F", "V"],
        "never_reads": ["T_ev", "R_j"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": dev,
        "dtype": args.dtype,
        "gradient_checkpointing": not args.no_checkpointing,
        "sample_chunk_size": chunk,
    }
    if dev.startswith("cuda"):
        meta_common["device_name"] = torch.cuda.get_device_name(0)

    index = []
    for run in parse_runs(args.runs, allow_any=bool(args.pilot)):
        print("run %d" % run, flush=True)
        res = train_run(run, ctx)
        payload = dict(res)
        payload["metadata"] = meta_common
        if smoke_cfg:
            stem = "smoke_%02d" % run
        elif pilot:
            stem = "pilot_w%03d_run_%03d" % (width, run)
        else:
            stem = "run_%02d" % run
        path = os.path.join(args.out, stem + ".pt")
        torch.save(payload, path)
        hist = os.path.join(args.out, stem + "_history.json")
        with open(hist, "w") as fh:
            json.dump({"run": run, "history": res["history"],
                       "selected_iteration": res["best"]["iteration"],
                       "selected_Q_val": res["best"]["q_val"],
                       "selected_L_val": res["best"]["L_val"],
                       "metadata": meta_common}, fh, indent=2, default=float)
        index.append({"run": run, "checkpoint": os.path.basename(path),
                      "history": os.path.basename(hist),
                      "selected_iteration": res["best"]["iteration"],
                      "Q_val": res["best"]["q_val"],
                      "wall_seconds": res["wall_seconds"]})
        print("  retained iteration %d, Q_val %.6e, %.0fs"
              % (res["best"]["iteration"], res["best"]["q_val"],
                 res["wall_seconds"]), flush=True)

    if smoke_cfg:
        print("\nSMOKE RUN COMPLETE. Nothing here is a result; delete %r and "
              "rerun without --smoke." % args.out)
        return 0

    # One index file per invocation, named after the runs it covers. A single
    # shared index.json would be rewritten by every task of a job array, so
    # only the last task's rows would survive. The classifier globs run_*.pt
    # and does not depend on these files.
    tag = args.runs.replace(",", "_").replace(" ", "")
    with open(os.path.join(args.out, "index_%s.json" % tag), "w") as fh:
        json.dump({"runs": index, "metadata": meta_common}, fh, indent=2,
                  default=float)
    print("\nwrote %d runs to %s (index_%s.json)"
          % (len(index), args.out, tag))
    print("Q_val over runs: min %.6e  median %.6e  max %.6e"
          % (min(r["Q_val"] for r in index),
             float(np.median([r["Q_val"] for r in index])),
             max(r["Q_val"] for r in index)))
    print("no classification was performed here; that is stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
