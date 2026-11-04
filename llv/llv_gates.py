#!/usr/bin/env python3
"""Pre-evaluation gates for Section 5.2, spec section 12.

Fourteen gates. Every one must PASS before any final generation or any read of
T_ev = [8000, 20000). A gate that cannot run is BLOCKED, and a blocked gate is
not a warning: the driver exits nonzero exactly as for a failure.

usage:
    ARCHIVE=/data/zxu64/LLV_KMC_20k/llv_theta_T1200_B20000.npz \
        python3 llv_gates.py [--manifest-dir DIR]

Gates 1, 4, 9, 10 and 11 read the archive (F, R_j, V only - never T_ev).
Gates 7 and 8 require torch. Everything else is self-contained fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

import llv_spec as S
import llv_streams as ST

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"
_RESULTS = []


def gate(number, title, needs=()):
    def deco(fn):
        def wrapper(ctx):
            for req in needs:
                if not ctx.get(req):
                    return (number, title, BLOCKED, "requires %s" % req)
            try:
                detail = fn(ctx)
                return (number, title, PASS, detail or "")
            except Exception as exc:                   # noqa: BLE001
                ctx.setdefault("tracebacks", {})[number] = traceback.format_exc()
                return (number, title, FAIL, "%s: %s"
                        % (type(exc).__name__, exc))
        wrapper.number, wrapper.title = number, title
        _RESULTS.append(wrapper)
        return wrapper
    return deco


def _close(a, b, tol, what):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not d <= tol:
        raise AssertionError("%s: max deviation %.3e exceeds %.3e"
                             % (what, d, tol))
    return d


# ====================================================================== 1 ====
@gate(1, "archive and split", needs=("archive",))
def gate_archive(ctx):
    ld = ctx["loader"]
    traj = ld.traj
    if traj.shape != (1200, 20000, 2):
        raise AssertionError("archive shape %r, expected (1200, 20000, 2)"
                             % (traj.shape,))
    if traj.dtype != np.float32:
        raise AssertionError("archive dtype %r, expected float32" % traj.dtype)
    n_bad = int((~np.isfinite(traj)).sum())
    if n_bad:
        raise AssertionError("%d non-finite archive values" % n_bad)

    spans = [S.SPLIT_F] + [S.ref_block(j) for j in range(1, 10)] + \
            [S.SPLIT_V, S.SPLIT_T_EV]
    covered = np.zeros(20000, dtype=np.int32)
    for lo, hi in spans:
        covered[lo:hi] += 1
    if covered.max() != 1 or covered.min() != 1:
        raise AssertionError("split is not an exact disjoint partition")

    fit = ld.fitting(S.TARGET_G_IN)
    mu, sd = S.pooled_moments(fit)
    ctx["mu"], ctx["sd"] = mu, sd
    _close(mu, [0.121613, 0.355175], 5e-6, "frozen mu")
    _close(sd, [0.022545, 0.027240], 5e-6, "frozen s")
    ctx["archive_sha"] = S.sha256_file(ld.path)
    return ("shape ok, partition exact, mu %s s %s, sha %s..."
            % (np.round(mu, 6), np.round(sd, 6), ctx["archive_sha"][:12]))


# ====================================================================== 2 ====
@gate(2, "window fixture")
def gate_windows(ctx):
    # A labelled tensor: entry (t, b, c) == t. If any slice is wrong the
    # extracted index set is wrong, and the assertion names which one.
    labelled = np.tile(np.arange(S.RAW_ROLLOUT, dtype=np.float64)[:, None, None],
                       (1, 3, 2))
    w = S.extract_windows(labelled)
    for name, (lo, hi) in (("warmup", S.SLICE_WARMUP), ("G_in", S.SLICE_G_IN),
                           ("G_out", S.SLICE_G_OUT),
                           ("combined", S.SLICE_COMBINED)):
        got = w[name][:, 0, 0]
        want = np.arange(lo, hi, dtype=np.float64)
        if got.shape != want.shape or not np.array_equal(got, want):
            raise AssertionError("%s is not raw [%d:%d)" % (name, lo, hi))
    if w["G_in"].shape[0] != 400 or w["G_out"].shape[0] != 400:
        raise AssertionError("windows must be 400 steps each")

    # No reset at raw step 800: one continuous rollout whose state is a running
    # counter must be strictly increasing across the G_in/G_out boundary.
    calls = {"n": 0}

    def step(state):
        calls["n"] += 1
        return state + 1.0, state + 1.0

    state, traj = 0.0, []
    for _ in range(S.RAW_ROLLOUT):
        out, state = step(state)
        traj.append(out)
    traj = np.array(traj)[:, None, None].repeat(2, axis=2)
    ww = S.extract_windows(traj)
    if calls["n"] != S.RAW_ROLLOUT:
        raise AssertionError("rollout must advance exactly %d steps in one call"
                             % S.RAW_ROLLOUT)
    boundary = ww["G_out"][0, 0, 0] - ww["G_in"][-1, 0, 0]
    if boundary != 1.0:
        raise AssertionError(
            "recurrent state was reset at raw step 800 (boundary step %r)"
            % boundary)
    # And the forbidden idiom must not reproduce G_in.
    last_400 = traj[-400:]
    if np.array_equal(last_400, ww["G_in"]):
        raise AssertionError("'last 400 steps' coincides with G_in; the "
                             "window must be an explicit slice")
    return "slices exact, state continuous across step 800"


# ====================================================================== 3 ====
@gate(3, "CCF fixture")
def gate_ccf(ctx):
    rng = np.random.default_rng(31)
    n_t, n_b = 200, 40
    # Deliberately asymmetric: B is a lagged copy of A, so C_AB and C_BA
    # differ markedly and a dropped or swapped direction cannot pass.
    a = rng.standard_normal((n_t, n_b))
    b = np.empty_like(a)
    b[:7] = rng.standard_normal((7, n_b))
    b[7:] = a[:-7] + 0.1 * rng.standard_normal((n_t - 7, n_b))
    blk = np.stack([a, b], axis=2)

    c = S.corr_window(blk, 40)
    cb = S.corr_window_bruteforce(blk, 40)
    d = _close(c, cb, 1e-10, "FFT vs brute force")

    fam = S.family_arrays(blk, S.projection_scales(blk), 40)
    if not np.allclose(fam["CCF"][0], c[0, 1]) or \
       not np.allclose(fam["CCF"][1], c[1, 0]):
        raise AssertionError("CCF rows are not (C_AB, C_BA) in that order")
    if np.allclose(fam["CCF"][0], fam["CCF"][1]):
        raise AssertionError("the two CCF directions are identical; one has "
                             "been duplicated instead of computed")
    # C_AB(k) = E[x_A(t) x_B(t+k)] peaks at k = 7 for this construction.
    if int(np.argmax(c[0, 1])) != 7:
        raise AssertionError("C_AB peaks at lag %d, expected 7: the "
                             "conjugation convention is reversed"
                             % int(np.argmax(c[0, 1])))
    if int(np.argmax(c[1, 0])) == 7:
        raise AssertionError("C_BA also peaks at lag 7; directions are swapped")
    _close([c[0, 0, 0], c[1, 1, 0]], [1.0, 1.0], 1e-12, "C_ii(0)")
    return "FFT matches brute force to %.1e; C_AB peaks at lag 7, C_BA does not" % d


# ====================================================================== 4 ====
@gate(4, "frame fixture", needs=("archive",))
def gate_frames(ctx):
    w = S.PROJ_DIRECTIONS
    if w.shape != (6, 2):
        raise AssertionError("expected 6 non-coordinate directions, got %r"
                             % (w.shape,))
    _close(np.linalg.norm(w, axis=1), np.ones(6), 1e-15, "direction norms")
    for r in range(S.N_NONCOORDINATE_FRAMES):
        f = S.FRAMES[r]
        _close(f @ f.T, np.eye(2), 1e-15, "frame %d orthonormality" % (r + 1))
    want = np.array([[1, 1], [1, -1], [1, 2], [2, -1], [1, -2], [2, 1]],
                    dtype=np.float64)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    _close(w, want, 1e-15, "exact direction set")
    for a in range(6):
        for b in range(a + 1, 6):
            if abs(abs(float(w[a] @ w[b])) - 1.0) < 1e-12:
                raise AssertionError("directions %d and %d are collinear"
                                     % (a, b))

    meta = ctx.setdefault("manifest", {}).setdefault("projection", {})
    meta["n_noncoordinate_frames"] = S.N_NONCOORDINATE_FRAMES
    meta["n_total_frames_reported"] = S.N_TOTAL_FRAMES_REPORTED
    meta["convention"] = ("n_noncoordinate_frames counts frames EXCLUDING the "
                          "coordinate frame; the paper reports n_f = 4 total")
    if "n_frames" in meta:
        raise AssertionError("bare 'n_frames' metadata field is forbidden")

    fit = ctx["loader"].fitting(S.TARGET_G_IN)
    x = S.standardize(fit, ctx["mu"], ctx["sd"])
    scales = S.projection_scales(x)
    ctx["scales"] = scales
    meta["scales"] = [float(v) for v in scales]
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise AssertionError("projection scales are not finite and positive")
    u = S.projected_samples(x, scales)
    sd_u = np.sqrt(((u - u.mean(axis=(0, 1))) ** 2).mean(axis=(0, 1)))
    _close(sd_u, np.ones(6), 1e-12, "unit fitting-block s.d. after scaling")

    # The KDE helper must not renormalize the scales away: feeding it u and
    # 2*u must give measurably different densities.
    f1 = S.kde_direct(u[:, :20, :])
    f2 = S.kde_direct(2.0 * u[:, :20, :])
    if np.allclose(f1, f2, atol=1e-10):
        raise AssertionError("KDE helper renormalized its input; the "
                             "projection scales would be discarded")
    return ("6 unit directions orthonormal by frame; scales %s; std(u)=1 to 1e-12"
            % np.round(scales, 6))


# ====================================================================== 5 ====
def _kde_fixtures():
    """Sample sets a binned/FFT approximation cannot survive.

    smooth    : the easy case, a broad Gaussian cloud.
    discrete  : values on a coarse lattice, so every sample sits at the same
                sub-cell offset and deposition error adds coherently instead
                of averaging out.
    boundary  : mass piled at and beyond both grid ends, where a padded
                deposition grid truncates or wraps.
    collapsed : a single repeated value, the zero-variance limit.
    """
    rng = np.random.default_rng(55)
    smooth = rng.standard_normal((64, 48, 2))

    lattice = np.round(rng.standard_normal((64, 48, 2)) * 4) / 4.0
    lattice[:, :, 1] = np.floor(rng.uniform(-3, 3, (64, 48)) * 20) / 20.0

    edge = np.empty((64, 48, 2))
    edge[:, :, 0] = np.concatenate([
        np.full((64, 16), -5.0), np.full((64, 16), 5.0),
        rng.uniform(-6.5, 6.5, (64, 16))], axis=1)
    edge[:, :, 1] = np.concatenate([
        np.full((64, 24), -4.999), rng.uniform(4.9, 7.0, (64, 24))], axis=1)

    collapsed = np.zeros((32, 32, 2))
    collapsed[:, :, 0] = 0.3173
    collapsed[:, :, 1] = -2.0

    return (("smooth", smooth), ("discrete", lattice),
            ("boundary", edge), ("collapsed", collapsed))


@gate(5, "KDE fixture", needs=("torch",))
def gate_kde(ctx):
    if S.GRID.size != 513 or S.GRID[0] != -5.0 or S.GRID[-1] != 5.0:
        raise AssertionError("grid must be 513 points on [-5, 5]")
    _close(S.BANDWIDTH, 160000 ** (-1 / 5), 1e-15, "h")
    _close(S.BANDWIDTH_2D, 160000 ** (-1 / 6), 1e-15, "h2")

    rng = np.random.default_rng(5)
    s = rng.standard_normal((50, 7, 1))
    f = S.kde_direct(s)
    flat = s.ravel()
    ref = np.array([
        np.exp(-0.5 * ((z - flat) / S.BANDWIDTH) ** 2).sum()
        / (flat.size * S.BANDWIDTH * np.sqrt(2 * np.pi))
        for z in S.GRID])
    d = _close(f[:, 0], ref, 1e-12, "direct Gaussian KDE")

    # Repeating the same samples to mimic a larger ensemble must change
    # neither the bandwidth nor the density.
    f_rep = S.kde_direct(np.concatenate([s, s, s], axis=1))
    _close(f_rep, f, 1e-12, "density under sample repetition")

    # ---- the differentiable chunked implementation vs the direct formula ----
    # The trainer uses S.kde_torch. It must be the same literal kernel sum, at
    # BOTH bandwidths, on sample sets chosen so that a deposition/FFT
    # approximation would be exposed rather than hidden by a smooth cloud.
    import torch

    worst = 0.0
    worst_tag = ""
    lines = []
    for tag, samp in _kde_fixtures():
        for hname, h in (("h", S.BANDWIDTH), ("h2", S.BANDWIDTH_2D)):
            ref_np = S.kde_direct(samp, bandwidth=h)
            t = torch.as_tensor(samp, dtype=torch.float64)
            got = S.kde_torch(t, bandwidth=h).numpy()
            den = np.linalg.norm(ref_np)
            rel = float(np.linalg.norm(got - ref_np) / den) if den > 0 else 0.0
            amax = float(np.max(np.abs(got - ref_np)))
            if rel > worst:
                worst, worst_tag = rel, "%s/%s" % (tag, hname)
            if not (rel < 1e-12 and amax < 1e-10):
                raise AssertionError(
                    "kde_torch deviates from the direct NumPy formula on the "
                    "%s fixture at %s: relative %.3e, absolute %.3e"
                    % (tag, hname, rel, amax))
            lines.append("%s/%s %.1e" % (tag, hname, rel))

            # Chunking may reorder the summation and nothing else.
            small = S.kde_torch(t, bandwidth=h, sample_chunk_size=7).numpy()
            _close(small, got, 1e-11, "chunk-size dependence (%s/%s)"
                   % (tag, hname))
            ck = S.kde_torch(t, bandwidth=h, sample_chunk_size=7,
                             checkpoint=True).numpy()
            _close(ck, got, 1e-11, "checkpointed path (%s/%s)" % (tag, hname))

    # ---- differentiability, against the analytic derivative --------------
    # d f(z) / d s_m = u * phi(u) / (M h^2),  u = (z - s_m)/h.
    # A binned estimator cannot pass this: its deposition weight is piecewise
    # linear in the sample position with a floor() breakpoint, so the gradient
    # is wrong at the cell scale and undefined on cell boundaries.
    for tag, samp in _kde_fixtures():
        sm = torch.tensor(np.asarray(samp, np.float64),
                          dtype=torch.float64, requires_grad=True)
        g_t = torch.as_tensor(S.GRID, dtype=torch.float64)
        w = torch.cos(g_t)[:, None]                 # arbitrary smooth weights
        f = S.kde_torch(sm, bandwidth=S.BANDWIDTH, checkpoint=True)
        (f * w).sum().backward()
        got_g = sm.grad.detach().numpy()

        flat = np.asarray(samp, np.float64).reshape(-1, samp.shape[-1])
        m = flat.shape[0]
        u = (S.GRID[:, None, None] - flat[None, :, :]) / S.BANDWIDTH
        phi = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
        want = (np.cos(S.GRID)[:, None, None] * u * phi).sum(axis=0) \
            / (m * S.BANDWIDTH ** 2)
        tol = 1e-9 * max(1.0, float(np.abs(want).max()))
        _close(got_g.reshape(-1, samp.shape[-1]), want, tol,
               "autograd vs analytic KDE derivative (%s)" % tag)
        if not np.isfinite(got_g).all():
            raise AssertionError("non-finite KDE gradient on %s" % tag)
        if float(np.abs(got_g).max()) == 0.0:
            raise AssertionError("KDE gradient is identically zero on %s; the "
                                 "estimator is not differentiable w.r.t. the "
                                 "samples" % tag)

    # ---- no binned/FFT estimator may remain on the official path ---------
    if hasattr(S, "kde_fast"):
        raise AssertionError("llv_spec still exposes kde_fast; the binned/FFT "
                             "approximation must not sit on the official path")
    import inspect
    src = inspect.getsource(S.family_arrays)
    if "kde_eval" not in src or "fft" in src.lower() or "bincount" in src:
        raise AssertionError("family_arrays must obtain densities from "
                             "kde_eval, the canonical kernel sum")

    # ---- the ambiguous legacy `bw` interface, in both its forms ----------
    for fn in (S.kde_direct, S.kde_eval):
        for bad in (5, 5.0, 6, 1.0, 0.0, -1.0):
            try:
                fn(s, bandwidth=bad)
            except ValueError:
                continue
            raise AssertionError(
                "%s accepted bandwidth=%r; exponent denominators and "
                "non-positive values must be rejected" % (fn.__name__, bad))
    for bad in (5, 5.0, 1.0, 0.0, -1.0):
        try:
            S.kde_torch(torch.as_tensor(s), bandwidth=bad)
        except ValueError:
            continue
        raise AssertionError("kde_torch accepted bandwidth=%r" % (bad,))

    ctx.setdefault("manifest", {})["kde"] = {
        "estimator": "canonical Gaussian kernel sum (kde_torch / kde_direct)",
        "binned_fft_on_official_path": False,
        "fixtures": ["smooth", "discrete", "boundary", "collapsed"],
        "bandwidths": {"h": S.BANDWIDTH, "h2": S.BANDWIDTH_2D},
        "worst_relative_deviation": worst,
        "worst_fixture": worst_tag,
        "gradient_check": "analytic u*phi(u)/(M h^2), tol 1e-9"}
    return ("513-pt grid, h=%.15f, h2=%.17f; direct match %.1e; "
            "chunked+checkpointed torch vs direct, worst %.1e at %s "
            "(%s); autograd == analytic derivative"
            % (S.BANDWIDTH, S.BANDWIDTH_2D, d, worst, worst_tag,
               ", ".join(lines)))


# ====================================================================== 6 ====
@gate(6, "loss-scale fixture")
def gate_loss_scale(ctx):
    rng = np.random.default_rng(6)
    a = rng.standard_normal(97)
    b = rng.standard_normal(97) + 3.0
    want = float(np.sum((a - b) ** 2) / np.sum(b ** 2))
    got = S.relative_loss(a, b)
    _close(got, want, 1e-15, "squared relative discrepancy")

    # It must be the SQUARE, not the root and not the square of the square.
    root = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    if abs(got - root) < 1e-9:
        raise AssertionError("loss equals the unsquared relative error")
    if abs(got - root ** 4) < 1e-9:
        raise AssertionError("loss has been squared twice")
    # And must carry no L1 term.
    l1 = float(np.mean(np.abs(a - b)))
    if abs(got - (want + l1)) < 1e-9:
        raise AssertionError("an L1 term is present in the loss")

    # Scaling the discrepancy by c scales L by exactly c^2.
    c = 2.5
    scaled = S.relative_loss(b + c * (a - b), b)
    _close(scaled, (c ** 2) * want, 1e-12, "quadratic homogeneity")

    losses = {s: 0.1 * (i + 1) for i, s in enumerate(S.FAMILIES)}
    _close(S.q_val(losses), sum(losses.values()), 1e-15, "Q_val = sum(L)")
    ref = {s: 0.05 for s in S.FAMILIES}
    r = S.ratios(losses, ref)
    for s in S.FAMILIES:
        _close(r[s], losses[s] / ref[s], 1e-15, "r = L / L_ref")
    _close(S.worst_ratio(r), max(r.values()), 1e-15, "worst-family ratio")
    if S.CLASSIFICATION_THRESHOLD != 10.0:
        raise AssertionError("classification threshold must be 10, not %r"
                             % S.CLASSIFICATION_THRESHOLD)
    return "L = ||dS||^2/||S||^2 exactly; Q_val = sum(L); r = L/L_ref; threshold 10"


# ====================================================================== 7 ====
@gate(7, "fresh-validation fixture", needs=("torch",))
def gate_fresh_validation(ctx):
    import torch
    a = ST.RunStreams(run=3)
    n1 = a.noise("checkpoint_monitor", 8, 5)
    n2 = a.noise("checkpoint_monitor", 8, 5)
    if torch.equal(n1, n2):
        raise AssertionError("two consecutive validation events drew the same "
                             "tensor; the stream is being re-seeded or a fixed "
                             "val_input is being reused")
    b = ST.RunStreams(run=3)
    m1 = b.noise("checkpoint_monitor", 8, 5)
    m2 = b.noise("checkpoint_monitor", 8, 5)
    if not (torch.equal(n1, m1) and torch.equal(n2, m2)):
        raise AssertionError("rerunning the same run did not reproduce the "
                             "validation noise sequence")
    width, active = S.TRAIN_CONFIG["input_width"], S.TRAIN_CONFIG["d_in_active"]
    if n1.shape[-1] != width:
        raise AssertionError("input width must be %d" % width)
    if not torch.all(n1[:, :, active:] == 0):
        raise AssertionError("channels beyond d_in must be identically zero")
    if torch.all(n1[:, :, :active] == 0):
        raise AssertionError("active noise channels are zero")
    return "consecutive events differ; rerun reproduces the sequence; width %d, %d active" \
        % (width, active)


# ====================================================================== 8 ====
@gate(8, "independent-classification fixture", needs=("torch",))
def gate_independent_classification(ctx):
    import torch
    for run in (0, 7, 19):
        r = ST.RunStreams(run=run)
        if r.seeds["classification"] == r.seeds["checkpoint_monitor"]:
            raise AssertionError("run %d: classification and monitoring share "
                                 "a seed" % run)
        if torch.equal(r.initial_states["classification"],
                       r.initial_states["checkpoint_monitor"]):
            raise AssertionError("run %d: classification and monitoring "
                                 "generators share a state" % run)
        c = r.noise("classification", 8, 5)
        m = r.noise("checkpoint_monitor", 8, 5)
        if torch.equal(c, m):
            raise AssertionError("run %d: classification reused the "
                                 "monitoring ensemble" % run)
    seeds = {p: [ST.seed_for(p, m) for m in S.RUN_SEEDS]
             for p in ST.RunStreams.PURPOSES}
    flat = [s for v in seeds.values() for s in v]
    if len(set(flat)) != len(flat):
        raise AssertionError("seed collision across purposes or runs")
    for bootstrap in (S.SEED_LONGLAG_BOOTSTRAP, S.SEED_BURNIN_BOOTSTRAP):
        if bootstrap in flat:
            raise AssertionError("bootstrap seed %d collides with a model "
                                 "stream" % bootstrap)
    return "%d model seeds distinct across %d purposes; bootstrap seeds disjoint" \
        % (len(flat), len(seeds))


# ====================================================================== 9 ====
@gate(9, "reference fixture", needs=("archive",))
def gate_reference(ctx):
    ld, mu, sd, sc = ctx["loader"], ctx["mu"], ctx["sd"], ctx["scales"]
    v = S.standardize(ld.validation(S.TARGET_G_IN), mu, sd)
    sv = S.family_arrays(v, sc)
    per_block = []
    for j in range(1, 10):
        rj = S.standardize(ld.reference(j, S.TARGET_G_IN), mu, sd)
        per_block.append(S.family_losses(S.family_arrays(rj, sc), sv))
    ref = {s: float(np.median([b[s] for b in per_block])) for s in S.FAMILIES}
    for s in S.FAMILIES:
        if not np.isfinite(ref[s]) or ref[s] <= 0:
            raise AssertionError("L_ref[%s] = %r is not finite and positive"
                                 % (s, ref[s]))
    # Record the denominators themselves. L_s divides by ||S_s(V)||^2, so a
    # family whose population statistic nearly vanishes is numerically fragile
    # even though its norm is formally nonzero. Spec section 4 requires the
    # norms to be finite and nonzero; this makes their size auditable, and
    # flags the CCF case the specification anticipates for independent
    # components.
    norms = {s: float(np.dot(sv[s].ravel(), sv[s].ravel())) for s in S.FAMILIES}
    ctx["L_ref_val_in"], ctx["target_norms_val_in"] = ref, norms
    man = ctx.setdefault("manifest", {})
    man["L_ref_val_in"], man["target_norm_sq_val_in"] = ref, norms
    man["L_ref_per_block_val_in"] = [
        {s: b[s] for s in S.FAMILIES} for b in per_block]
    # Spec section 12 gate 9: record the norms so weak statistics stay
    # auditable, and introduce NO additional magnitude cutoff. A large
    # reference loss is recorded as a diagnostic, never a failure: no such
    # cutoff belongs to the classification rule or to this gate.
    man["L_ref_above_unity"] = [s for s in S.FAMILIES if ref[s] > 1.0]
    return ("  ".join("%s %.3e" % (s, ref[s]) for s in S.FAMILIES)
            + "  |  target ||S||^2 "
            + "  ".join("%s %.3e" % (s, norms[s]) for s in S.FAMILIES))


# ===================================================================== 10 ====
def _ks(a, b):
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, v, "right") / a.size
                        - np.searchsorted(b, v, "right") / b.size).max())


@gate(10, "burn-in / stationarity fixture", needs=("archive",))
def gate_burnin(ctx):
    """Spec 3.1, the frozen verify_and_calibrate_v2.py procedure.

    Late reference half [4000,6000), tested half [6000,8000), NumPy seed
    20260913. The two halves are disjoint trajectory sets, so neither is
    scientifically preferred as the tested one; this assignment is kept
    because it is the one with real provenance and the one the frozen table
    and the Supplementary already report.

    Mean and variance: 200 whole-trajectory resamples, max over physical
    t = 901..1200 and both components. Taking the max over the whole late
    window rather than at the single tested time point is the appropriate
    null for a process with an oscillatory transient.

    KS: an empirical cutoff from 200 sampled late time points. It is not a
    trajectory bootstrap and must not be called one.

    d_s^win: on the tested half, 401..800 against 801..1200, unsquared, fixed
    tolerance 0.10. Both scales are stored; the paper displays four decimals,
    the manifest keeps full float64.
    """
    ld, mu, sd, sc = ctx["loader"], ctx["mu"], ctx["sd"], ctx["scales"]
    traj = ld.traj
    t_lo, t_hi = S.BURNIN_TESTED_HALF
    r_lo, r_hi = S.BURNIN_REFERENCE_HALF
    lt_lo, lt_hi = S.BURNIN_LATE_T[0] - 1, S.BURNIN_LATE_T[1]
    n_rep = S.BURNIN_N_REPLICATES

    tested = S.standardize(
        np.asarray(traj[:, t_lo:t_hi, :], np.float64), mu, sd)
    late = S.standardize(
        np.asarray(traj[lt_lo:lt_hi, r_lo:r_hi, :], np.float64), mu, sd)

    bar = late.mean(axis=(0, 1))
    var = ((late - bar) ** 2).mean(axis=(0, 1))
    pt = tested[S.BURNIN_TESTED_T - 1]
    obs = {"mean": float(np.abs(pt.mean(axis=0) - bar).max()),
           "variance": float((np.abs(pt.var(axis=0) - var) / var).max()),
           "ks": max(_ks(pt[:, c], late[:, :, c].ravel()) for c in (0, 1))}

    rng = np.random.default_rng(S.SEED_BURNIN_BOOTSTRAP)
    n_half = tested.shape[1]
    late_t = tested[lt_lo:lt_hi]
    bm, bv = np.empty(n_rep), np.empty(n_rep)
    for r in range(n_rep):
        sub = late_t[:, rng.integers(0, n_half, n_half), :]
        bm[r] = np.abs(sub.mean(axis=1) - bar).max()
        bv[r] = (np.abs(sub.var(axis=1) - var) / var).max()
    bk = np.empty(n_rep)
    for r in range(n_rep):
        # Transcribed literally from verify_and_calibrate_v2.py: the draw sits
        # INSIDE the generator over components, so each replicate consumes one
        # late time index PER COMPONENT and the two components are compared at
        # different times. Sharing a single index per replicate changes both
        # the statistic and the generator stream.
        bk[r] = max(_ks(tested[int(rng.integers(lt_lo, lt_hi))][:, c],
                        late[:, :, c].ravel()) for c in (0, 1))
    lim = {"mean": float(np.quantile(bm, 0.95)),
           "variance": float(np.quantile(bv, 0.95)),
           "ks": float(np.quantile(bk, 0.95))}

    fam = ("ACF", "CCF", "ACF2")
    b_in = S.family_arrays(tested[400:800], sc, families=fam)
    b_out = S.family_arrays(tested[800:1200], sc, families=fam)
    d_win, l_sq = {}, {}
    for s in fam:
        l_sq[s] = S.relative_loss(b_in[s], b_out[s])
        d_win[s] = float(np.sqrt(l_sq[s]))

    lines, bad = [], []
    labels = {"mean": "max standardized mean deviation",
              "variance": "max relative variance deviation",
              "ks": "max marginal KS distance"}
    for key in ("mean", "variance", "ks"):
        f_obs, f_lim = S.BURNIN_FROZEN[key]
        o, l = obs[key], lim[key]
        lines.append("%s obs %.4f lim %.4f" % (labels[key], o, l))
        if o > l:
            bad.append("%s observed %.4f exceeds its limit %.4f"
                       % (key, o, l))
        if abs(o - f_obs) > 5e-5:
            bad.append("%s observed %.6f does not reproduce the frozen %.4f"
                       % (key, o, f_obs))
        if abs(l - f_lim) > 5e-5:
            bad.append("%s limit %.6f does not reproduce the frozen %.4f"
                       % (key, l, f_lim))
    for s in fam:
        f = S.BURNIN_FROZEN_DWIN[s]
        if d_win[s] > S.DWIN_TOLERANCE:
            bad.append("d_win[%s] = %.4f exceeds the fixed tolerance %.2f"
                       % (s, d_win[s], S.DWIN_TOLERANCE))
        if abs(d_win[s] - f) > 5e-5:
            bad.append("d_win[%s] = %.6f does not reproduce the frozen %.4f"
                       % (s, d_win[s], f))

    ctx.setdefault("manifest", {})["burnin"] = {
        "procedure": "verify_and_calibrate_v2.py, frozen",
        "reference_half": list(S.BURNIN_REFERENCE_HALF),
        "tested_half": list(S.BURNIN_TESTED_HALF),
        "late_reference_t": list(S.BURNIN_LATE_T),
        "tested_t": S.BURNIN_TESTED_T,
        "seed": S.SEED_BURNIN_BOOTSTRAP,
        "n_replicates": n_rep,
        "observed": obs,
        "limits": lim,
        "limit_kind": {
            "mean": "95th percentile, 200 whole-trajectory resamples, max "
                    "over t = 901..1200 and both components",
            "variance": "95th percentile, same resampling",
            "ks": "empirical 95th-percentile cutoff over 200 sampled late "
                  "time points; not a trajectory bootstrap"},
        "d_win": d_win,
        "L_squared": l_sq,
        "d_win_tolerance": S.DWIN_TOLERANCE,
        "d_win_tolerance_kind": "fixed operational tolerance, not a quantile",
        "scales": "d_win is unsquared (spec 3.1); L_squared is the spec 4 "
                  "family loss; (d_win)^2 == L_squared",
        "frozen_reference": {"diagnostics": S.BURNIN_FROZEN,
                             "d_win": S.BURNIN_FROZEN_DWIN},
        "display_precision": "paper shows 4 decimals; stored values are "
                             "full float64"}

    if bad:
        raise AssertionError("; ".join(bad))
    return ("; ".join(lines) + " | d_win " + "  ".join(
        "%s %.4f" % (s, d_win[s]) for s in fam)
        + " | frozen spec 3.1 table reproduced")


# ===================================================================== 11 ====
@gate(11, "float precision", needs=("archive",))
def gate_float_precision(ctx):
    ld, mu, sd, sc = ctx["loader"], ctx["mu"], ctx["sd"], ctx["scales"]
    v = S.standardize(ld.validation(S.TARGET_G_IN), mu, sd)
    r1 = S.standardize(ld.reference(1, S.TARGET_G_IN), mu, sd)
    l64 = S.family_losses(S.family_arrays(r1, sc), S.family_arrays(v, sc))
    l32 = S.family_losses(
        S.family_arrays(r1.astype(np.float32).astype(np.float64), sc),
        S.family_arrays(v.astype(np.float32).astype(np.float64), sc))
    worst, where = 0.0, None
    for s in S.FAMILIES:
        rel = abs(l32[s] - l64[s]) / abs(l64[s])
        if rel > worst:
            worst, where = rel, s
    if worst >= 1e-5:
        raise AssertionError("float32 reporting path differs by %.3e at %s, "
                             "limit 1e-5" % (worst, where))
    ctx.setdefault("manifest", {})["float_precision"] = {
        "worst_relative_difference": worst, "family": where}
    return "worst relative difference %.3e at %s (limit 1e-5)" % (worst, where)


# ===================================================================== 12 ====
@gate(12, "long-lag fixture")
def gate_longlag(ctx):
    rng = np.random.default_rng(12)
    n_t, n_b = 800, 30
    a = rng.standard_normal((n_t, n_b))
    b = np.roll(a, 5, axis=0) + 0.3 * rng.standard_normal((n_t, n_b))
    blk = np.stack([a, b], axis=2)

    k_lo, k_hi = S.LONGLAG_K_LO, S.LONGLAG_K_HI
    c = S.corr_paircount(blk, k_lo, k_hi)
    cb = S.corr_paircount_bruteforce(blk, k_lo, k_hi)
    d = _close(c, cb, 1e-10, "pair-count FFT vs direct formula")
    if c.shape != (4, k_hi - k_lo + 1):
        raise AssertionError("expected 4 curves over %d lags, got %r"
                             % (k_hi - k_lo + 1, c.shape))
    if np.allclose(c[2], c[3]):
        raise AssertionError("CCF_AB and CCF_BA are identical in the long-lag "
                             "estimator")

    # Pair counts at the two declared lags.
    if 800 - k_lo != 400 or 800 - k_hi != 200:
        raise AssertionError("pair counts are %d at k=%d and %d at k=%d; the "
                             "k_max rule requires 400 and 200"
                             % (800 - k_lo, k_lo, 800 - k_hi, k_hi))
    if 800 - k_hi < (800 - 400) / 2:
        raise AssertionError("k_max violates 800 - k >= (800 - 400)/2")

    # Pair-count normalization must NOT taper: for white noise the window
    # estimator decays like (1 - k/T) while this one does not.
    wn = np.stack([rng.standard_normal((n_t, 400)),
                   rng.standard_normal((n_t, 400))], axis=2)
    cw = S.corr_window(wn, n_t)[0, 0]
    taper = cw[600] / cw[0] if cw[0] else np.nan
    if abs(taper) > 0.5:
        raise AssertionError("window estimator is not tapering as expected")

    # Trajectory-level resampling: whole trajectories, never time points.
    boot = ST.longlag_rng()
    if boot.bit_generator.state["state"]["state"] != \
            np.random.default_rng(S.SEED_LONGLAG_BOOTSTRAP) \
            .bit_generator.state["state"]["state"]:
        raise AssertionError("long-lag bootstrap is not seeded with 7000")
    idx = boot.integers(0, n_b, S.LONGLAG_BLOCK)
    if idx.shape != (S.LONGLAG_BLOCK,):
        raise AssertionError("bootstrap block size must be %d" % S.LONGLAG_BLOCK)
    sub = blk[:, idx, :]
    if sub.shape[0] != n_t:
        raise AssertionError("resampling must preserve each trajectory's full "
                             "800-step dependence")
    if S.LONGLAG_N_BOOT != 2000:
        raise AssertionError("long-lag bootstrap must use 2000 replicates")
    if S.LONGLAG_CURVES != ("ACF_A", "ACF_B", "CCF_AB", "CCF_BA"):
        raise AssertionError("long-lag curve order is wrong")
    return ("direct formula matches to %.1e; pair counts 400 at k=400 and 200 "
            "at k=600; 2000 trajectory-level replicates, seed 7000, no factor "
            "three" % d)


# ===================================================================== 13 ====
@gate(13, "admissibility fixture")
def gate_admissibility(ctx):
    states = np.array([
        [0.12, 0.36],      # interior, valid
        [0.00, 0.50],      # boundary rho_A = 0, valid (v = 0, not > 0)
        [0.40, 0.60],      # boundary sum = 1, valid
        [-1e-6, 0.50],     # just below zero, invalid
        [0.60, 0.60],      # sum 1.2, invalid, exceedance 0.2
        [1.50, 0.10],      # rho_A > 1, invalid, exceedance 0.6
        [np.nan, 0.30],    # non-finite
        [0.20, np.inf],    # non-finite
    ], dtype=np.float64)[:, None, :]

    res = S.admissibility(states)
    n = states.shape[0]
    _close(res["nonfinite_rate"], 2 / n, 1e-15, "nonfinite_rate")
    _close(res["constraint_rate"], 3 / n, 1e-15, "constraint_rate")
    _close(res["invalid_rate"], 5 / n, 1e-15, "invalid_rate")
    _close(res["max_exceedance"], 0.6, 1e-12, "max_exceedance")
    if not np.isfinite(res["max_exceedance"]):
        raise AssertionError("NaN propagated into max_exceedance")

    v = S.exceedance(states[:3, 0])
    if np.any(v > 0):
        raise AssertionError("a boundary state was reported as violating")

    allbad = np.full((2, 1, 2), np.nan)
    r2 = S.admissibility(allbad)
    if not np.isnan(r2["max_exceedance"]) or r2["nonfinite_rate"] != 1.0:
        raise AssertionError("an all-non-finite window must give "
                             "max_exceedance = NaN and nonfinite_rate = 1")

    # No clipping anywhere on the inverse-standardization path. The chosen
    # standardized values are far enough out that rho_A < 0 and rho_B > 1:
    # at mu/s = (0.1216, 0.3552)/(0.02254, 0.02724) that needs x_A < -5.40
    # and x_B > 23.67.
    mu_t, sd_t = np.array([0.1216, 0.3552]), np.array([0.02254, 0.02724])
    x = np.array([[[-9.0, 30.0]]])
    rho = S.inverse_standardize(x, mu_t, sd_t)
    if rho[0, 0, 0] >= 0.0:
        raise AssertionError("rho_A was clipped at zero: %r" % rho[0, 0, 0])
    if rho[0, 0, 1] <= 1.0:
        raise AssertionError("rho_B was clipped at one: %r" % rho[0, 0, 1])
    _close(rho[0, 0], mu_t + x[0, 0] * sd_t, 1e-15, "inverse standardization")
    # The generated values must stay off the N_site^-1 lattice: no rounding.
    if np.allclose(rho * 1e4, np.round(rho * 1e4)):
        raise AssertionError("generated densities were rounded onto the "
                             "N_site^-1 lattice")
    return ("rates 2/8 nonfinite, 3/8 constraint, 5/8 invalid; "
            "max exceedance 0.6; boundaries valid; no clipping")


# ===================================================================== 14 ====
@gate(14, "manifest freeze", needs=("archive",))
def gate_manifest(ctx):
    man = ctx.setdefault("manifest", {})
    man["specification"] = "section5.2-cleaned-specification.md (2026-10-06)"
    spec_path = ctx.get("spec_path")
    if not spec_path or not os.path.exists(spec_path):
        raise AssertionError(
            "the specification document must be checksummed into the manifest, "
            "because the loader re-verifies it before unlocking T_ev. Copy "
            "section5.2-cleaned-specification.md next to this code and pass "
            "--spec PATH (or set SPEC=PATH). Got %r" % (spec_path,))
    man["specification_path"] = os.path.abspath(spec_path)
    man["specification_sha256"] = S.sha256_file(spec_path)
    man["archive_path"] = ctx["loader"].path
    man["archive_sha256"] = ctx["archive_sha"]
    man["standardization"] = {
        "mu": [float(v) for v in ctx["mu"]],
        "s": [float(v) for v in ctx["sd"]],
        "source": "F over physical t = 401..800, float64, ddof = 0",
    }
    man["split"] = {
        "F": list(S.SPLIT_F), "V": list(S.SPLIT_V), "T_ev": list(S.SPLIT_T_EV),
        "R": [list(S.ref_block(j)) for j in range(1, 10)],
    }
    man["windows"] = {"G_in": list(S.SLICE_G_IN), "G_out": list(S.SLICE_G_OUT),
                      "combined": list(S.SLICE_COMBINED),
                      "raw_rollout": S.RAW_ROLLOUT}
    man["training"] = dict(S.TRAIN_CONFIG)
    man["training"]["betas"] = list(S.TRAIN_CONFIG["betas"])
    man["training"]["scored_slice"] = list(S.TRAIN_CONFIG["scored_slice"])
    man["seeds"] = {"base": dict(S.SEED_BASE), "runs": list(S.RUN_SEEDS),
                    "longlag_bootstrap": S.SEED_LONGLAG_BOOTSTRAP,
                    "burnin_resampling": S.SEED_BURNIN_BOOTSTRAP}
    man.setdefault("kde", {}).update({
        "grid_n": S.GRID_N, "range": [S.GRID_LO, S.GRID_HI],
        "bandwidth": S.BANDWIDTH, "bandwidth_2d": S.BANDWIDTH_2D,
        "interface": "bandwidth_value, never an exponent denominator",
        "canonical_estimator": "literal Gaussian kernel sum, chunked, "
                               "differentiable (spec 4)"})
    man["classification"] = {"rule": "r_val = max_s L_s/L_s_ref <= 10",
                             "threshold": S.CLASSIFICATION_THRESHOLD,
                             "inclusive": True}
    man["reporting"] = {
        "B_gen": S.B_GEN, "B_vis": S.B_VIS,
        "representative": "sort successful runs by stored Q_val ascending, "
                          "take zero-based element M // 2",
        "families": list(S.FAMILIES),
    }
    man["longlag"] = {"k_lo": S.LONGLAG_K_LO, "k_hi": S.LONGLAG_K_HI,
                      "n_boot": S.LONGLAG_N_BOOT, "block": S.LONGLAG_BLOCK,
                      "curves": list(S.LONGLAG_CURVES),
                      "factor": 1.0}
    man["figures"] = {"target_label": "Target (KMC)",
                      "reference_label": "Reference blocks (R_j)",
                      "main_correlation_k_max": S.LONGLAG_K_HI}
    here = os.path.dirname(os.path.abspath(S.__file__))
    man["source_sha256"] = {
        name: S.sha256_file(os.path.join(here, name))
        for name in S.GATED_SOURCES}

    out = ctx["manifest_dir"]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, S.MANIFEST_NAME)
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True, default=float)
    digest = S.sha256_file(path)
    with open(os.path.join(out, "manifest.sha256"), "w") as fh:
        fh.write(digest + "\n")
    ctx["manifest_sha"] = digest
    ctx["freeze_path"] = os.path.join(out, S.FREEZE_TOKEN_NAME)
    return "manifest written to %s, sha %s..." % (path, digest[:12])


# ======================================================================= ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE", ""))
    ap.add_argument("--manifest-dir", default="llv_manifest_v2")
    ap.add_argument("--spec", default=os.environ.get("SPEC", ""),
                    help="path to section5.2-cleaned-specification.md; its "
                         "checksum is frozen into the manifest and re-verified "
                         "by the loader before T_ev can be opened")
    ap.add_argument("--supersede", action="append", default=[],
                    help="mark an older manifest directory superseded and "
                         "disable its freeze token; repeatable")
    args = ap.parse_args()

    for old in args.supersede:
        if os.path.abspath(old) == os.path.abspath(args.manifest_dir):
            print("refusing to supersede the directory being written: %r" % old,
                  file=sys.stderr)
            return 2
        if not os.path.isdir(old):
            print("supersede: %r is not a directory" % old, file=sys.stderr)
            return 2
        tok = os.path.join(old, S.FREEZE_TOKEN_NAME)
        if os.path.exists(tok):
            os.rename(tok, tok + ".revoked")
        with open(os.path.join(old, S.SUPERSEDED_NAME), "w") as fh:
            fh.write("superseded by %s\n" % os.path.abspath(args.manifest_dir))
        print("superseded %s (token revoked)" % old)

    live = os.path.join(args.manifest_dir, S.FREEZE_TOKEN_NAME)
    if os.path.exists(live):
        print("refusing to overwrite an existing freeze token at %r. Run into "
              "a new --manifest-dir and pass --supersede %s."
              % (live, args.manifest_dir), file=sys.stderr)
        return 2

    ctx = {"manifest_dir": args.manifest_dir, "spec_path": args.spec,
           "notes": []}
    try:
        import torch                                    # noqa: F401
        ctx["torch"] = True
    except ImportError:
        ctx["torch"] = False

    if args.archive and os.path.exists(args.archive):
        ctx["archive"] = True
        ctx["loader"] = S.ArchiveLoader(args.archive,
                                        manifest_dir=args.manifest_dir)
    else:
        ctx["archive"] = False
        if args.archive:
            print("archive %r not found" % args.archive, file=sys.stderr)

    print("Section 5.2 pre-evaluation gates")
    print("archive: %s   torch: %s" % (args.archive or "(none)", ctx["torch"]))
    print("-" * 78)

    results = [fn(ctx) for fn in _RESULTS]
    width = max(len(t) for _, t, _, _ in results)
    n_pass = 0
    for num, title, status, detail in results:
        print("%2d  %-*s  %-7s  %s" % (num, width, title, status, detail))
        n_pass += status == PASS
    print("-" * 78)
    print("%d/%d passed" % (n_pass, len(results)))

    for note in ctx.get("notes", []):
        print("\nNOTE  " + note)
    for num, tb in ctx.get("tracebacks", {}).items():
        print("\n--- traceback for gate %d ---\n%s" % (num, tb))

    if n_pass == len(results):
        with open(ctx["freeze_path"], "w") as fh:
            fh.write(ctx["manifest_sha"] + "\n")
        print("\nALL GATES PASSED. Freeze token written to %s"
              % ctx["freeze_path"])
        if ctx.get("archive"):
            try:
                S.ArchiveLoader(args.archive,
                                manifest_dir=args.manifest_dir).assert_unlocked()
            except S.FinalTargetLocked as exc:
                print("\nbut the loader still refuses to unlock T_ev: %s" % exc,
                      file=sys.stderr)
                return 1
            print("Loader verified token, manifest, archive, code and "
                  "specification checksums; T_ev may now be opened via "
                  "ArchiveLoader(path, manifest_dir=%r)." % args.manifest_dir)
        return 0
    print("\nGATES NOT PASSED. T_ev stays locked. A blocked gate is not a "
          "warning.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
