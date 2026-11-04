#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 of 3 -- generate and save the Section 4 target trajectories.

Target: two harmonically coupled stochastic Duffing oscillators with
asymmetric on-site quartics, in contact with a heat bath at one temperature.

    V(q1,q2) = a/2 (q1^2 + q2^2) + k1/4 q1^4 + k2/4 q2^4 + lam/2 (q1-q2)^2

    dq_i = p_i dt
    dp_i = -dV/dq_i dt - gamma_i p_i dt + sqrt(2 gamma_i / beta) dW_i

This replaces the post-hoc mixing construction q2t = rho*q1 + sqrt(1-rho^2)*q2
of the earlier draft. There, q1 and q2 never interacted: every joint cumulant
was pinned by the two marginals and the single scalar rho, the cross-correlation
was an exact rescaling of the ACF of q1, and the two components had a spurious
instantaneous velocity correlation <dq1/dt dq2/dt> = rho/beta that no genuinely
coupled equilibrium system can have. A potential coupling fixes all three.

This file writes only trajectories. Analytic reference densities and
normalisation scales are stage 2 (make_reference.py); the split, the floor and
the reported errors are stage 3 (evaluate.py).

Outputs, per coupling strength lam:
    traj_lam<X>.npy    float32, shape (L, N, 2) = (time, trajectory, component),
                       matching the (seq_len, batch, input_size) layout the SINN
                       code expects
    meta_lam<X>.json   every constant the manuscript's Table 1 needs, plus an
                       empirical-vs-analytic moment check

Usage
-----
    python make_target.py --outdir data
    python make_target.py --convergence          # dt convergence table
    python make_target.py --outdir data --ntraj 2000   # smoke test
"""

import argparse
import json
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------

ALPHA = 1.0          # on-site harmonic coefficient, both components
KAPPA = (1.0, 0.5)   # on-site quartic coefficients -- deliberately unequal
BETA = 1.0           # inverse temperature
GAMMA = (1.0, 0.5)   # damping, per component

# KAPPA[0] != KAPPA[1] breaks the exchange symmetry of V. With equal kappa the
# static law is invariant under q1 <-> q2, and then
#     rho(q1 + 2 q2) == rho(2 q1 + q2)   and   rho(2 q1 - q2) == rho(q1 - 2 q2)
# identically, so the swapped directions duplicate each other. Unequal kappa
# removes that duplication -- but only weakly: the measured L1 separation it
# buys between a swapped pair is 0.0005, against 0.01-0.07 from the coupling.
#
# It does NOT remove the other degeneracy. At lam = 0 the components are
# independent and both marginals are even, so a*q1 + b*q2 and a*q1 - b*q2 have
# the same law and the six directions collapse to three distinct densities
# (verified: L1 difference exactly 0). No choice of kappa prevents this; it
# follows from independence alone. It is harmless, since at lam = 0 there is no
# joint structure to capture, but the six directions must not be described as
# six independent constraints there. For lam > 0 the coupling breaks the
# q2 -> -q2 symmetry and all six differ.
#
# GAMMA[0] != GAMMA[1] additionally breaks the
# exchange symmetry of the dynamics, giving the two components different
# relaxation behaviour; it does not affect the stationary density, which is the
# Gibbs measure exp(-beta V) whatever the damping, as long as both components
# satisfy the fluctuation-dissipation relation at the same temperature.

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)

# lam = 20 is excluded. The variance-based stiffness scale of the relative mode,
# omega = 1/sqrt(beta <v^2>) with v = (q1-q2)/sqrt(2), reaches 6.5 there, i.e. a
# period below one time unit and about ten saved points per period. That is
# above the sampling limit but leaves no comfortable margin for the correlation
# functions, and lam = 20 sits in the near-degenerate strong-coupling regime.
# Note omega here is an RMS (second-spectral-moment) frequency, not a spectral
# peak; a peak would have to come from a high-resolution power spectrum.

# --------------------------------------------------------------------------
# Numerical constants
# --------------------------------------------------------------------------

DT = 0.01            # BAOAB internal step
DT_SAVE = 0.1        # SINN coarse-grained step
L_SEQ = 200          # saved points per trajectory -> 20 time units
N_TRAJ = 20000       # independent trajectories per lambda

# No physical burn-in is simulated: initial conditions are drawn from the Gibbs
# measure by inverse-CDF sampling on the quadrature grid -- exact for the
# piecewise-constant density the grid defines, which differs from the continuous
# Gibbs measure at O(dq^2), not exact in the strict sense. This does NOT make the
# chain stationary either: BAOAB's invariant measure pi_h differs from the
# continuous pi, so the ensemble drifts by |pi_h - pi| across the window. The
# convergence check below is what licenses calling that negligible -- an
# empirical bound with a stated error bar, not a proof -- and it is not optional. The recurrent warm-up of the
# network (feed extra steps, keep the last L) is a separate matter handled in
# the training code, not here.

GRID_HALFWIDTH = 8.0   # quadrature grid for the Gibbs measure
GRID_N = 2001

SEED_IC = 20260913     # shared across lambda -> paired initial quantiles
SEED_NOISE = 71828     # shared across lambda -> paired noise realisations

# Sharing both seeds across lambda makes the comparison between coupling
# strengths paired rather than between independent samples, which is what the
# data-split protocol asks for when several targets are compared. Trajectory i
# at one lambda and trajectory i at another see the same uniform quantiles and
# the same noise increments.


# --------------------------------------------------------------------------
# Potential
# --------------------------------------------------------------------------

def potential(q1, q2, lam):
    """V(q1, q2). Broadcasts over arrays."""
    return (0.5 * ALPHA * (q1 ** 2 + q2 ** 2)
            + 0.25 * KAPPA[0] * q1 ** 4
            + 0.25 * KAPPA[1] * q2 ** 4
            + 0.5 * lam * (q1 - q2) ** 2)


def grad_potential(q, lam):
    """dV/dq for q of shape (n, 2). Returns an array of the same shape."""
    q1, q2 = q[:, 0], q[:, 1]
    rel = lam * (q1 - q2)
    g = np.empty_like(q)
    g[:, 0] = ALPHA * q1 + KAPPA[0] * q1 ** 3 + rel
    g[:, 1] = ALPHA * q2 + KAPPA[1] * q2 ** 3 - rel
    return g


def gibbs_grid(lam, halfwidth=GRID_HALFWIDTH, n=GRID_N):
    """Normalised joint density exp(-beta V) on a quadrature grid.

    Returns (grid, dq, p) with p of shape (n, n) indexed [q1, q2] and
    integrating to one. This is the analytic truth for every static quantity;
    it exists because the coupling is a potential, and it is the main practical
    gain over the old post-hoc mixing construction.
    """
    grid = np.linspace(-halfwidth, halfwidth, n)
    dq = grid[1] - grid[0]
    Q1, Q2 = np.meshgrid(grid, grid, indexing="ij")
    V = potential(Q1, Q2, lam)
    p = np.exp(-BETA * (V - V.min()))
    p /= p.sum() * dq * dq
    return grid, dq, p


def gibbs_moments(grid, dq, p):
    """Low-order moments of the analytic joint density.

    kappa_1122 = <q1^2 q2^2> - <q1^2><q2^2> - 2 <q1 q2>^2 is the lowest joint
    cumulant that a linear mixture of independent components cannot set
    independently of the correlation, so it is the scalar that says how much
    genuine higher-order dependence the target carries. make_reference.py
    compares it against a matched Gaussian-copula null.
    """
    Q1, Q2 = np.meshgrid(grid, grid, indexing="ij")
    w = p * dq * dq
    m = lambda A: float((A * w).sum())
    v1, v2 = m(Q1 ** 2), m(Q2 ** 2)
    c11 = m(Q1 * Q2)
    return {
        "var1": v1, "var2": v2,
        "std1": float(np.sqrt(v1)), "std2": float(np.sqrt(v2)),
        "cov": c11,
        "corr": c11 / float(np.sqrt(v1 * v2)),
        "excess_kurtosis1": m(Q1 ** 4) / v1 ** 2 - 3.0,
        "excess_kurtosis2": m(Q2 ** 4) / v2 ** 2 - 3.0,
        "kappa_1122_over_var1var2": (m(Q1 ** 2 * Q2 ** 2) - v1 * v2 - 2 * c11 ** 2) / (v1 * v2),
        "std_rel_mode": float(np.sqrt(m((Q1 - Q2) ** 2) / 2.0)),
    }


# --------------------------------------------------------------------------
# Sampling and integration
# --------------------------------------------------------------------------

def sample_gibbs(grid, dq, p, n, rng):
    """Draw n samples from the joint Gibbs density.

    The grid cells are treated as a discrete distribution: one inverse-CDF
    lookup on the flattened cell probabilities picks a cell, then a uniform
    jitter inside the cell removes the grid signature. Exact up to the
    piecewise-constant approximation of the density within a cell, an O(dq^2)
    error -- at dq ~ 8e-3 the jitter contributes dq^2/12 ~ 5e-6 to the variance,
    five orders of magnitude below it.

    Not exact sampling in the strict sense: it is exact for the
    piecewise-constant density the grid defines.

    Fully vectorised and O(GRID_N^2) in memory regardless of n, which matters:
    the obvious marginal-then-conditional implementation needs an (n, GRID_N)
    array and exhausts memory past a few thousand trajectories.
    """
    w = (p * dq * dq).ravel()
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    idx = np.clip(np.searchsorted(cdf, rng.random(n)), 0, w.size - 1)
    i1, i2 = np.unravel_index(idx, p.shape)
    q1 = grid[i1] + (rng.random(n) - 0.5) * dq
    q2 = grid[i2] + (rng.random(n) - 0.5) * dq
    return np.stack([q1, q2], axis=1)


def baoab(q, p, lam, n_steps, dt, save_every, n_save, rng):
    """BAOAB: B (half kick) - A (half drift) - O - A - B.

    The O step integrates dp = -gamma p dt + sqrt(2 gamma/beta) dW exactly, so
    the thermostat contributes no discretisation error; only the symmetric A/B
    splitting does. For this splitting the invariant-measure error on
    configurational averages is known to be higher order than the naive estimate
    in the high-friction limit; whether that regime applies here is not assumed,
    which is exactly what --convergence measures. Euler-Maruyama, by contrast,
    biases the invariant measure at O(dt), and visibly so.
    """
    gam = np.array(GAMMA)
    c1 = np.exp(-gam * dt)
    c2 = np.sqrt((1.0 - c1 ** 2) / BETA)

    out = np.empty((n_save, q.shape[0], 2), dtype=np.float32)
    out[0] = q
    k = 1

    half = 0.5 * dt
    for step in range(1, n_steps + 1):
        p -= half * grad_potential(q, lam)                  # B
        q += half * p                                       # A
        p = c1 * p + c2 * rng.standard_normal(q.shape)      # O
        q += half * p                                       # A
        p -= half * grad_potential(q, lam)                  # B
        if step % save_every == 0 and k < n_save:
            out[k] = q
            k += 1
    return out, q, p


def euler_maruyama(q, p, lam, n_steps, dt, save_every, n_save, rng):
    """The integrator the old Section 4 used. Kept only for the contrast rows
    of the convergence table."""
    gam = np.array(GAMMA)
    sig = np.sqrt(2.0 * dt * gam / BETA)
    out = np.empty((n_save, q.shape[0], 2), dtype=np.float32)
    out[0] = q
    k = 1
    for step in range(1, n_steps + 1):
        q_new = q + p * dt
        p = p + (-grad_potential(q, lam) - gam * p) * dt + sig * rng.standard_normal(q.shape)
        q = q_new
        if step % save_every == 0 and k < n_save:
            out[k] = q
            k += 1
    return out, q, p


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def generate(lam, n_traj, outdir, verbose=True, force=False):
    tag = f"lam{lam:g}".replace(".", "p")
    traj_path = os.path.join(outdir, f"traj_{tag}.npy")
    if os.path.exists(traj_path) and not force:
        raise SystemExit(
            f"{traj_path} already exists. Regenerating silently would leave the "
            f"reference and the runs that used the old archive pointing at data "
            f"that no longer exists. Pass --force to overwrite, or write to a "
            f"new --outdir.")

    t0 = time.time()
    grid, dq, p_gibbs = gibbs_grid(lam)
    moments = gibbs_moments(grid, dq, p_gibbs)

    rng_ic = np.random.default_rng(SEED_IC)
    rng_noise = np.random.default_rng(SEED_NOISE)

    q = sample_gibbs(grid, dq, p_gibbs, n_traj, rng_ic)
    p = rng_ic.standard_normal((n_traj, 2)) / np.sqrt(BETA)

    save_every = int(round(DT_SAVE / DT))
    traj, _, _ = baoab(q, p, lam, (L_SEQ - 1) * save_every, DT,
                       save_every, L_SEQ, rng_noise)

    if not np.all(np.isfinite(traj)):
        n_bad = int((~np.isfinite(traj)).sum())
        raise SystemExit(
            f"lambda={lam}: {n_bad} non-finite values in the generated "
            f"trajectories. BAOAB has gone unstable -- reduce dt, or check that "
            f"lambda is within the range the step size resolves.")

    t = traj.astype(np.float64)
    empirical = {
        "var": [float((t[..., 0] ** 2).mean()), float((t[..., 1] ** 2).mean())],
        "cov": float((t[..., 0] * t[..., 1]).mean()),
    }
    empirical["corr"] = empirical["cov"] / float(
        np.sqrt(empirical["var"][0] * empirical["var"][1]))

    os.makedirs(outdir, exist_ok=True)
    np.save(traj_path, traj)

    meta = {
        "lambda": lam, "tag": tag,
        "alpha": ALPHA, "kappa": list(KAPPA), "beta": BETA, "gamma": list(GAMMA),
        "dt": DT, "dt_save": DT_SAVE, "L_seq": L_SEQ, "n_traj": n_traj,
        "integrator": "BAOAB", "physical_burn_in_steps": 0,
        "initial_condition": "Gibbs to grid resolution, inverse-CDF on the "
                             "quadrature grid, O(dq^2)",
        "seed_ic": SEED_IC, "seed_noise": SEED_NOISE,
        "analytic_moments": moments,
        "empirical_moments": empirical,
        "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(outdir, f"meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    if verbose:
        m = moments
        print(f"lam={lam:<5g} corr {m['corr']:+.4f} / {empirical['corr']:+.4f}   "
              f"var1 {m['var1']:.4f} / {empirical['var'][0]:.4f}   "
              f"var2 {m['var2']:.4f} / {empirical['var'][1]:.4f}   "
              f"k1122 {m['kappa_1122_over_var1var2']:+.4f}   "
              f"[{meta['wall_time_s']:.1f}s]")
    return meta


def convergence_check(lam=4.0, n_traj=4000, t_total=200.0):
    """Verify that the integrator bias is below the sampling noise.

    This is what licenses treating the analytic Gibbs density as the truth and
    skipping the physical burn-in, so it belongs in the appendix, not a drawer.
    BAOAB should scatter about zero with no trend in dt; the Euler-Maruyama rows
    show what a trend looks like when there is one.

    The same seeds are reused across step sizes on purpose, so the sampling
    noise is partly common to every row and a systematic trend stands out
    against it. The Brownian paths are NOT shared across step sizes -- BAOAB's
    exact O step does not decompose across substeps, so a shared path would need
    a different construction -- which is why every row carries a standard error,
    computed from the scatter of per-trajectory time averages so that
    within-trajectory correlation is accounted for.
    """
    grid, dq, p_gibbs = gibbs_grid(lam)
    m = gibbs_moments(grid, dq, p_gibbs)
    c_exact = m["cov"]

    # At lam = 0 the components are exactly independent and <q1 q2> vanishes, so
    # a relative error there is 0/0. Report it in absolute terms instead. The
    # same degeneracy hits the CCF family in evaluate.py, which reports an
    # absolute cross-correlation discrepancy there and drops CCF from the
    # relative-error success maximum.
    degenerate = abs(c_exact) < 1e-6

    def sem(per_traj):
        # Standard error of the ensemble mean from the scatter of per-trajectory
        # time averages, so within-trajectory correlation is accounted for.
        return float(per_traj.std(ddof=1) / np.sqrt(per_traj.size))

    def line(name, dt, t):
        v1p, v2p = (t[..., 0] ** 2).mean(axis=0), (t[..., 1] ** 2).mean(axis=0)
        cp = (t[..., 0] * t[..., 1]).mean(axis=0)
        v1, v2, c = v1p.mean(), v2p.mean(), cp.mean()
        cs = (f"<q1q2>={c:+.6f} +-{sem(cp):.6f} (abs; exact 0)" if degenerate
              else f"<q1q2>={100 * (c / c_exact - 1):+.2f}% "
                   f"+-{100 * sem(cp) / abs(c_exact):.2f}")
        print(f"{name:<6} dt={dt:<7} "
              f"<q1^2>={100 * (v1 / m['var1'] - 1):+.2f}% +-{100 * sem(v1p) / m['var1']:.2f}  "
              f"<q2^2>={100 * (v2 / m['var2'] - 1):+.2f}% +-{100 * sem(v2p) / m['var2']:.2f}  "
              f"{cs}")

    print(f"lambda = {lam}   ({n_traj} trajectories, {t_total} time units each)")
    print(f"exact:  <q1^2>={m['var1']:.6f}  <q2^2>={m['var2']:.6f}  <q1q2>={c_exact:+.6f}")

    for scheme, integrate, steps in (("BAOAB", baoab, (0.01, 0.005, 0.0025)),
                                     ("EM", euler_maruyama, (0.01, 0.005))):
        for dt in steps:
            rng_ic = np.random.default_rng(SEED_IC)
            rng = np.random.default_rng(SEED_NOISE)
            q = sample_gibbs(grid, dq, p_gibbs, n_traj, rng_ic)
            p = rng_ic.standard_normal((n_traj, 2)) / np.sqrt(BETA)
            save_every = max(1, int(round(DT_SAVE / dt)))
            n_save = int(t_total / DT_SAVE)
            traj, _, _ = integrate(q, p, lam, n_save * save_every, dt,
                                   save_every, n_save, rng)
            line(scheme, dt, traj.astype(np.float64))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--ntraj", type=int, default=N_TRAJ)
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--convergence", action="store_true",
                    help="run the dt convergence check instead of generating")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing archive")
    ap.add_argument("--conv-ntraj", type=int, default=4000)
    ap.add_argument("--conv-time", type=float, default=200.0)
    args = ap.parse_args()

    if args.convergence:
        for lam in args.lambdas:
            convergence_check(lam, args.conv_ntraj, args.conv_time)
            print()
        return

    print(f"potential: alpha={ALPHA} kappa={KAPPA} beta={BETA} gamma={GAMMA}")
    print(f"BAOAB dt={DT}, save every {DT_SAVE}, L={L_SEQ}, N={args.ntraj}, "
          f"no physical burn-in")
    print()
    for lam in args.lambdas:
        generate(lam, args.ntraj, args.outdir, force=args.force)


if __name__ == "__main__":
    main()
