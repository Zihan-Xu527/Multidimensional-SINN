#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The training objective for the Section 4 sweep: five statistical losses on the
standardised process, in torch, differentiable, reading every constant from the
reference and the target-statistics cache.

CONTRACT
--------
For each statistic family s, with S_theta the generated statistic and S_fit the
cached fitting-block target,

    M_s = mean[ (S_theta - S_fit)^2 ]
    D_s = mean[ S_fit^2 ]                       (cached, a constant)
    L_s = M_s / D_s                             <- what training minimises
    e_s = sqrt(L_s)                             <- what gets reported

so the objective is the SQUARE of the relative L2 discrepancy the evaluator
reports, with the identical statistic and the identical normalisation. The
squared form keeps a smooth gradient near zero error and matches the
uncertainty-weighting form of the loss-balancing scheme.

Manuscript wording this licenses: "Training minimizes the squared relative L2
discrepancy used for reporting; the statistic and normalization are otherwise
identical."

Five families: ACF, CCF, ACF2, PDF, PROJ.

PROJ aggregation. The selected 2*n_f projected densities are CONCATENATED into
one stack and a single normalized MSE is taken over it, with the matching
D_PROJ(n_f) from the cache. The direction count sits in the numerator and the
denominator alike, so raising n_f adds constraints without mechanically
inflating the term -- which is what makes the n_f x LB interaction in block 1
interpretable rather than an artefact. Averaging per-direction relative errors
instead would be a different quantity and would no longer match the evaluator.

n_f = 0 removes the PROJ term ENTIRELY, together with its balancing parameter.
A zero-valued loss term left in place would still contribute its log-sigma
penalty and drive that parameter without bound.

CCF mode. When the reference says ccf_mode='absolute' (lambda = 0, where the
population cross-correlation vanishes identically) the CCF term is M_CCF with no
division. The cached target CCF is NOT zeroed -- the fitting block's empirical
cross-correlation carries sampling noise and that is what the network is asked
to match; only the normalisation changes. The branch is read from the stored
ccf_mode, which is part of the constants hash, so training and evaluation cannot
disagree about which branch a lambda is on.

This creates a scale discontinuity at lambda = 0. The manuscript should present
lambda = 0 as an independence anchor in its own right, and LB comparisons should
be made within each lambda rather than across the break.

Loss balancing. Self-adaptive weighting after nondimensionalisation:

    L_total = sum_s [ L_s / (2 sigma_s^2) + log sigma_s ],   sigma_s = exp(rho_s)

with rho_s trainable. Because the L_s are already dimensionless and O(1), this
is no longer repairing units -- but it still changes the relative gradient
weights and the final multi-objective trade-off, so describe it as adaptive task
weighting after nondimensionalisation, not merely as a change of convergence
pace. The unweighted sum of the active L_s is reported alongside for both
baseline and balanced runs, so convergence is compared on one quantity.

ESTIMATORS
----------
Mirrors of the numpy definitions in evaluate.py, which are the single source of
truth. `python sinn_loss.py --verify --datadir data` checks every family against
them on random data and on the real fitting block.

KDE is cloud-in-cell deposition followed by a cached Gaussian convolution
matrix, the same estimator evaluate.py uses. Deposition is differentiable almost
everywhere: the bin index is piecewise constant, the interpolation weights are
linear in the sample. `scatter_add` carries the gradient. A direct Gaussian sum
would be the definition, but at L*B = 80000 samples and 401 grid points it is
2.6e8 Gaussians per density per step -- roughly a gigabyte of forward
activations before the backward pass -- and it agrees with CIC to ~1e-6 relative
anyway (make_target_stats.py --verify-kde measures it on the real data).

SHAPES
------
The generated ensemble reaches the loss as (L, N, 2), ALREADY STANDARDISED --
the network is trained on standardised targets and its output is used directly,
so no mu/scale is applied here. L must be the full sequence length (200), never
truncated to K: K = 100 truncates the correlation lags only. Cutting the
trajectory to 100 steps would halve the KDE sample count from 80000 to 40000 and
silently change the bandwidth's meaning.

The recurrent warm-up is the caller's business: feed the network L + warmup
steps, keep the last L, hand those here.

Usage
-----
    from sinn_loss import SINNObjective
    obj = SINNObjective.from_files("data", "lam4", n_f=3, loss_balancing=True,
                                   device="cuda")
    out = obj(generated)            # generated: (200, N, 2) standardised, requires_grad
    out["total"].backward()
    # out["losses"]["PDF"], out["errors"]["PDF"], out["unweighted_total"], ...

    python sinn_loss.py --verify --datadir data --lam 4
"""

import argparse
import os

import numpy as np

try:
    import torch
except ImportError:                                     # pragma: no cover
    torch = None

Z_HALFWIDTH = 5.0
Z_FINE_N = 4001
FAMILIES = ("ACF", "CCF", "ACF2", "PDF", "PROJ")


# --------------------------------------------------------------------------
# Differentiable estimators
# --------------------------------------------------------------------------

def _acf(x, n_lags):
    """Unnormalised autocovariance of x, shape (L, N), averaged over N, with the
    ensemble's own empirical mean removed. Biased estimator: normalised at zero
    lag only, never divided by (L - k). Mirrors evaluate._acf_raw."""
    x = x - x.mean()
    n = x.shape[0]
    f = torch.fft.rfft(x, n=2 * n - 1, dim=0)
    ac = torch.fft.irfft(f * torch.conj(f), n=2 * n - 1, dim=0)[:n].mean(dim=1)
    return ac[:n_lags]


def _ccf(x, y, n_lags):
    """C_xy(k) = <x(t) y(t+k)>, each series centred by its own mean.

    Conjugation on x, not y: irfft(F(x) conj(F(y))) would be <x(t+k) y(t)> =
    C_yx, which swaps the labels of the two directional curves. Mirrors
    evaluate._ccf_raw."""
    x = x - x.mean()
    y = y - y.mean()
    n = x.shape[0]
    f = torch.fft.rfft(x, n=2 * n - 1, dim=0)
    g = torch.fft.rfft(y, n=2 * n - 1, dim=0)
    cc = torch.fft.irfft(torch.conj(f) * g, n=2 * n - 1, dim=0)[:n].mean(dim=1)
    return cc[:n_lags]


def correlations(qt, n_lags):
    """ACF (L,2), CCF (L,2) as [C_12, C_21], ACF2 (L,2). Mirrors
    evaluate.correlations."""
    x, y = qt[..., 0], qt[..., 1]
    ax, ay = _acf(x, n_lags), _acf(y, n_lags)
    norm = torch.sqrt(ax[0] * ay[0])
    acf = torch.stack([ax / ax[0], ay / ay[0]], dim=-1)
    ccf = torch.stack([_ccf(x, y, n_lags) / norm,
                       _ccf(y, x, n_lags) / norm], dim=-1)
    a2x, a2y = _acf(x ** 2, n_lags), _acf(y ** 2, n_lags)
    acf2 = torch.stack([a2x / a2x[0], a2y / a2y[0]], dim=-1)
    return acf, ccf, acf2


def kde_batch(samples, z_fine_lo, dz, n_fine, conv):
    """Differentiable CIC KDE for a batch of sample sets.

    samples : (n_sets, n_samples)
    conv    : (n_out, n_fine)  precomputed Gaussian convolution, already
              carrying the dz and 1/(sqrt(2 pi) h) factors
    returns : (n_out, n_sets)

    scatter_add carries the gradient through the interpolation weights; the
    floor index is piecewise constant and contributes none, which is why this is
    differentiable almost everywhere rather than everywhere.
    """
    n_sets, n_samp = samples.shape
    pos = (samples - z_fine_lo) / dz
    i0 = torch.floor(pos)
    frac = pos - i0
    i0 = i0.long()
    inside = (i0 >= 0) & (i0 < n_fine - 1)
    idx = i0.clamp(0, n_fine - 2)
    zero = torch.zeros((), dtype=samples.dtype, device=samples.device)
    w0 = torch.where(inside, 1.0 - frac, zero)
    w1 = torch.where(inside, frac, zero)

    hist = torch.zeros((n_sets, n_fine), dtype=samples.dtype, device=samples.device)
    hist = hist.scatter_add(1, idx, w0).scatter_add(1, idx + 1, w1)
    hist = hist / (n_samp * dz)
    return conv @ hist.T


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

class SINNObjective:
    """The five statistical losses, optionally self-adaptively balanced.

    Everything the objective needs is fixed at construction from
    reference_lam*.npz and target_stats_lam*.npz: the directions, the projection
    scales, the bandwidth, the grids, K, the CCF mode, the cached targets and
    their denominators. Nothing is recomputed per step and nothing is inferred
    from a filename.
    """

    def __init__(self, ref, tgt, n_f=3, loss_balancing=True, device="cpu",
                 dtype=None):
        if torch is None:                               # pragma: no cover
            raise RuntimeError("sinn_loss needs PyTorch")
        dtype = dtype or torch.float32
        self.device, self.dtype = device, dtype
        self.n_f = int(n_f)
        if self.n_f not in (0, 1, 2, 3):
            raise ValueError("n_f must be 0, 1, 2 or 3 (frames, i.e. 2*n_f directions)")

        binding_ref = str(ref["constants_sha256"])
        if str(tgt["constants_sha256"]) != binding_ref:
            raise RuntimeError(
                f"target_stats constants hash {str(tgt['constants_sha256'])[:12]}... "
                f"does not match the reference {binding_ref[:12]}.... Rebuild the "
                f"cache with make_target_stats.py.")
        if str(tgt["ccf_mode"]) != str(ref["ccf_mode"]):
            raise RuntimeError("ccf_mode disagrees between cache and reference")

        self.l_seq = int(ref["l_seq"])
        self.k_lags = int(ref["k_lags"])
        self.ccf_mode = str(ref["ccf_mode"])
        self.constants_sha256 = binding_ref

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

        # Projection directions and scales, first 2*n_f of them.
        dirs = np.asarray(ref["projections"], dtype=np.float64)[:2 * self.n_f]
        scal = np.asarray(ref["proj_scales"], dtype=np.float64)[:2 * self.n_f]
        self.proj_w = t(dirs / scal[:, None]) if self.n_f else None   # (2n_f, 2)

        # KDE machinery: one cached convolution matrix, used for every density.
        h = float(ref["bandwidth_1d"])
        z_out = np.asarray(ref["z_grid"], dtype=np.float64)
        z_fine = np.linspace(-Z_HALFWIDTH - 6 * h, Z_HALFWIDTH + 6 * h, Z_FINE_N)
        dz = z_fine[1] - z_fine[0]
        conv = (np.exp(-0.5 * ((z_out[:, None] - z_fine[None, :]) / h) ** 2)
                * dz / (np.sqrt(2.0 * np.pi) * h))
        self.conv = t(conv)
        self.z_fine_lo, self.dz, self.n_fine = float(z_fine[0]), float(dz), Z_FINE_N

        # Cached targets and denominators.
        self.target = {k: t(tgt[f"target_{k}"]) for k in ("ACF", "CCF", "ACF2", "PDF")}
        self.denom = {k: float(tgt[f"denom_{k}"]) for k in ("ACF", "CCF", "ACF2", "PDF")}
        self.families = ["ACF", "CCF", "ACF2", "PDF"]
        if self.n_f:
            self.target["PROJ"] = t(np.asarray(tgt["target_PROJ"])[:, :2 * self.n_f])
            self.denom["PROJ"] = float(tgt[f"denom_PROJ_nf{self.n_f}"])
            self.families.append("PROJ")
        # n_f = 0: PROJ is absent from self.families, so it has no loss term and
        # no balancing parameter. A zero-valued term would still contribute its
        # log-sigma penalty.

        self.loss_balancing = bool(loss_balancing)
        if self.loss_balancing:
            # sigma_s = exp(rho_s) keeps sigma positive with no clamping.
            self.log_sigma = {
                k: torch.zeros((), dtype=dtype, device=device, requires_grad=True)
                for k in self.families}
        else:
            self.log_sigma = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_files(cls, datadir, tag, n_f=3, loss_balancing=True, device="cpu",
                   dtype=None):
        ref = dict(np.load(os.path.join(datadir, f"reference_{tag}.npz"),
                           allow_pickle=False))
        tp = os.path.join(datadir, f"target_stats_{tag}.npz")
        if not os.path.exists(tp):
            raise SystemExit(
                f"{tp} is missing. The training targets are built once by "
                f"make_target_stats.py; sweep jobs must not build them, or "
                f"hundreds of tasks will race on the same cache.")
        tgt = dict(np.load(tp, allow_pickle=False))
        return cls(ref, tgt, n_f, loss_balancing, device, dtype)

    # -- statistics --------------------------------------------------------

    def statistics(self, gen):
        """The generated statistics, in the same layout as the cached targets.

        gen: (L, N, 2), ALREADY STANDARDISED. No mu/scale is applied.
        """
        if gen.dim() != 3 or gen.shape[2] != 2:
            raise ValueError(f"expected (L, N, 2), got {tuple(gen.shape)}")
        if gen.shape[0] != self.l_seq:
            raise ValueError(
                f"expected L = {self.l_seq} steps, got {gen.shape[0]}. K = "
                f"{self.k_lags} truncates the correlation lags, never the "
                f"trajectory: cutting it would change the KDE sample count.")

        acf, ccf, acf2 = correlations(gen, self.k_lags)
        flat = gen.reshape(-1, 2)                                  # (L*N, 2)
        out = {"ACF": acf, "CCF": ccf, "ACF2": acf2,
               "PDF": kde_batch(flat.T, self.z_fine_lo, self.dz, self.n_fine,
                                self.conv)}
        if self.n_f:
            proj = (self.proj_w @ flat.T)                          # (2n_f, L*N)
            out["PROJ"] = kde_batch(proj, self.z_fine_lo, self.dz, self.n_fine,
                                    self.conv)
        return out

    # -- losses ------------------------------------------------------------

    def __call__(self, gen):
        sg = self.statistics(gen)
        losses = {}
        for k in self.families:
            m = ((sg[k] - self.target[k]) ** 2).mean()
            if k == "CCF" and self.ccf_mode == "absolute":
                losses[k] = m                    # no division: D_CCF -> 0 in the population
            else:
                losses[k] = m / self.denom[k]

        unweighted = sum(losses.values())
        if self.loss_balancing:
            total = 0.0
            for k in self.families:
                rho = self.log_sigma[k]
                total = total + 0.5 * losses[k] * torch.exp(-2.0 * rho) + rho
        else:
            total = unweighted

        return {
            "total": total,
            "unweighted_total": unweighted,
            "losses": losses,
            "errors": {k: torch.sqrt(v.detach()) for k, v in losses.items()},
            "weights": ({k: torch.exp(-2.0 * self.log_sigma[k]).detach() * 0.5
                         for k in self.families} if self.loss_balancing else None),
            "statistics": sg,
        }

    def balance_parameters(self):
        """The sigma parameters, for their own optimizer."""
        return list(self.log_sigma.values())

    def describe(self):
        return {"n_f": self.n_f, "families": list(self.families),
                "ccf_mode": self.ccf_mode, "loss_balancing": self.loss_balancing,
                "k_lags": self.k_lags, "l_seq": self.l_seq,
                "denominators": dict(self.denom),
                "constants_sha256": self.constants_sha256}


# --------------------------------------------------------------------------
# Verification against the numpy definitions in evaluate.py
# --------------------------------------------------------------------------

def verify(datadir, tag, seed=0):
    """Check every torch estimator against evaluate.py's numpy version.

    Run this once on the machine that will train. The numpy definitions are the
    single source of truth; if this disagrees, the training target and the
    reported error are different quantities and the sweep is not interpretable.
    """
    if torch is None:
        raise SystemExit("PyTorch is not installed here; run --verify where it is")
    import evaluate as EV

    ref = dict(np.load(os.path.join(datadir, f"reference_{tag}.npz"),
                       allow_pickle=False))
    tgt = dict(np.load(os.path.join(datadir, f"target_stats_{tag}.npz"),
                       allow_pickle=False))
    traj = np.load(os.path.join(datadir, f"traj_{tag}.npy"), mmap_mode="r")
    h = float(ref["bandwidth_1d"])
    z_out = ref["z_grid"]
    z_fine = np.linspace(-Z_HALFWIDTH - 6 * h, Z_HALFWIDTH + 6 * h, Z_FINE_N)
    k = int(ref["k_lags"])

    rng = np.random.default_rng(seed)
    lo, hi = int(ref["split_fit"][0]), int(ref["split_fit"][1])
    real = ((np.asarray(traj[:, lo:hi, :], dtype=np.float64) - ref["mu"])
            / ref["scale"])
    fake = rng.standard_normal(real.shape) * 0.8 + 0.3     # deliberately off-centre

    obj = SINNObjective(ref, tgt, n_f=3, loss_balancing=True, device="cpu",
                        dtype=torch.float64)
    worst = 0.0
    for name, qt in (("fitting block", real), ("random, mean-shifted", fake)):
        acf_n, ccf_n, acf2_n = EV.correlations(qt, k)
        pdf_n = np.stack([EV.kde(qt[..., i], h, z_out, z_fine) for i in (0, 1)],
                         axis=-1)
        proj_n = np.stack([EV.kde((a * qt[..., 0] + b * qt[..., 1]) / s, h, z_out,
                                  z_fine)
                           for (a, b), s in zip(ref["projections"],
                                                ref["proj_scales"])], axis=-1)
        sg = obj.statistics(torch.as_tensor(qt, dtype=torch.float64))
        print(f"  {name}:")
        for key, npv in (("ACF", acf_n), ("CCF", ccf_n), ("ACF2", acf2_n),
                         ("PDF", pdf_n), ("PROJ", proj_n)):
            tv = sg[key].detach().numpy()
            rel = float(np.sqrt(np.mean((tv - npv) ** 2) / np.mean(npv ** 2)))
            worst = max(worst, rel)
            print(f"    {key:<5} torch vs numpy: relative L2 {rel:.3e}")

    # ---- gradients, at a point where the loss is NOT already zero ----------
    #
    # The fitting block is useless for this: the cached target IS its statistics,
    # so the loss sits exactly at its minimum, every residual vanishes and the
    # gradient is zero whether or not the KDE's gradient path is connected at
    # all. A held-out block gives losses of order 1e-3 and a real gradient.
    b = int(ref["b_fit"])
    lo = int(ref["split_reference"][0])
    held = ((np.asarray(traj[:, lo:lo + b, :], dtype=np.float64) - ref["mu"])
            / ref["scale"])

    def fresh():
        return torch.as_tensor(held, dtype=torch.float64).clone().requires_grad_(True)

    g = fresh()
    out = obj(g)
    out["total"].backward()
    gnorm = float(g.grad.norm())
    losses = {k: float(v) for k, v in out["losses"].items()}
    print(f"  held-out block losses: "
          + "  ".join(f"{k}={losses[k]:.3e}" for k in obj.families))
    print(f"  total gradient norm {gnorm:.4e}  finite={bool(np.isfinite(gnorm))}")

    # Per family: every term must actually move the input. PDF and PROJ are the
    # ones that matter here -- they are the only path through scatter_add, so a
    # zero here means the CIC deposition is not carrying its gradient.
    per_ok = True
    for k in obj.families:
        gi = fresh()
        obj(gi)["losses"][k].backward()
        gk = float(gi.grad.norm())
        per_ok &= np.isfinite(gk) and gk > 1e-12
        print(f"    d{k}/dx norm {gk:.4e}")

    # Finite-difference check on the PDF term, which exercises the deposition
    # weights directly. The estimator is piecewise linear in each sample, so the
    # step must stay well inside one deposition bin.
    rng2 = np.random.default_rng(seed + 1)
    gi = fresh()
    obj(gi)["losses"]["PDF"].backward()
    ana = gi.grad.detach().numpy()
    eps = 1e-6
    fd_worst = 0.0
    for _ in range(5):
        i, j, c = (rng2.integers(held.shape[0]), rng2.integers(held.shape[1]),
                   rng2.integers(2))
        d = np.zeros_like(held)
        d[i, j, c] = eps
        lp = float(obj(torch.as_tensor(held + d, dtype=torch.float64))["losses"]["PDF"])
        lm = float(obj(torch.as_tensor(held - d, dtype=torch.float64))["losses"]["PDF"])
        num = (lp - lm) / (2 * eps)
        rel = abs(num - ana[i, j, c]) / max(abs(num), 1e-30)
        fd_worst = max(fd_worst, rel)
    print(f"  finite-difference check on dPDF/dx: worst relative error {fd_worst:.3e}")

    # n_f = 0 must drop PROJ and its balancing parameter outright.
    o0 = SINNObjective(ref, tgt, n_f=0, loss_balancing=True, device="cpu",
                       dtype=torch.float64)
    ok0 = "PROJ" not in o0.families and len(o0.balance_parameters()) == 4
    print(f"  n_f=0 drops PROJ and its sigma: {ok0}")

    # e_s must equal the evaluator's relative L2 against the same target -- also
    # checked on the held-out block, so both sides are a real number.
    e = {kk: float(v) for kk, v in obj(torch.as_tensor(held,
                                                       dtype=torch.float64))["errors"].items()}
    pdf_np = np.stack([EV.kde(held[..., i], h, z_out, z_fine) for i in (0, 1)],
                      axis=-1)
    ev = EV._rel(pdf_np, tgt["target_PDF"])
    e_ok = abs(e["PDF"] - ev) < 1e-12
    print(f"  e_PDF from loss {e['PDF']:.6e}  vs evaluate._rel {ev:.6e}  "
          f"diff {abs(e['PDF'] - ev):.2e}")

    good = (worst < 1e-9 and np.isfinite(gnorm) and gnorm > 1e-12 and per_ok
            and fd_worst < 1e-4 and ok0 and e_ok)
    print("  verify PASSED" if good else "  verify FAILED")
    return good


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    tag = f"lam{args.lam:g}".replace(".", "p")
    if args.verify:
        print(f"[{tag}] torch estimators vs the numpy definitions in evaluate.py")
        raise SystemExit(0 if verify(args.datadir, tag) else 1)
    obj = SINNObjective.from_files(args.datadir, tag)
    print(obj.describe())


if __name__ == "__main__":
    main()
