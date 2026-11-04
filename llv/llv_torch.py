"""Differentiable torch statistics for Section 5.2, spec sections 3 and 4.

Every function here mirrors one in llv_spec.py and must agree with it
numerically. `selftest` checks that agreement and the trainer calls it at
startup: if the training-side estimator drifted from the evaluation-side one,
checkpoint selection and classification would no longer be the same quantity.

No function here reads the archive, and none of them knows about T_ev.
"""

from __future__ import annotations

import numpy as np
import torch

import llv_spec as S


def pooled_moments_t(a: torch.Tensor):
    """Pooled mean and population s.d. over (time, batch), per component."""
    mean = a.mean(dim=(0, 1))
    var = ((a - mean) ** 2).mean(dim=(0, 1))
    return mean, torch.sqrt(var)


def corr_window_t(block: torch.Tensor, n_lags: int = S.K_LAGS) -> torch.Tensor:
    """C_ij(k), manuscript Eq. (corr_estimator), differentiable.

    Pooled means subtracted, divisor N*T at every lag so the 1 - k/T taper is
    carried, each ensemble normalized by its own pooled s.d., so C_ii(0) = 1.
    The conjugation is conj(F_i) * F_j, which puts
    sum_t x_i[t] x_j[t+k] at index k; the opposite convention returns C_ji.
    """
    a = block
    n_t, n_b, dim = a.shape
    mean, sd = pooled_moments_t(a)
    xc = a - mean
    n_fft = 1 << int(np.ceil(np.log2(2 * n_t)))
    f = torch.fft.rfft(xc, n=n_fft, dim=0)
    rows = []
    for i in range(dim):
        cols = []
        for j in range(dim):
            cc = torch.fft.irfft(torch.conj(f[:, :, i]) * f[:, :, j],
                                 n=n_fft, dim=0)
            cols.append(cc[:n_lags].sum(dim=1) / (n_b * n_t) / (sd[i] * sd[j]))
        rows.append(torch.stack(cols))
    return torch.stack(rows)


def projected_samples_t(block: torch.Tensor, scales: torch.Tensor,
                        directions: torch.Tensor) -> torch.Tensor:
    """u_w = (w . x) / varsigma_w. Directions and scales stay separate."""
    return torch.einsum("tbd,wd->tbw", block, directions) / scales


def family_arrays_t(block: torch.Tensor, scales: torch.Tensor,
                    directions: torch.Tensor, n_lags: int = S.K_LAGS,
                    checkpoint: bool = True,
                    sample_chunk_size=None) -> dict:
    """The five family arrays in the frozen order of spec section 4."""
    c = corr_window_t(block, n_lags)
    c2 = corr_window_t(block * block, n_lags)
    kw = {"checkpoint": checkpoint, "sample_chunk_size": sample_chunk_size}
    return {
        "ACF": torch.stack([c[0, 0], c[1, 1]]),
        "CCF": torch.stack([c[0, 1], c[1, 0]]),
        "ACF2": torch.stack([c2[0, 0], c2[1, 1]]),
        "PDF": S.kde_torch(block, **kw).T,
        "PROJ": S.kde_torch(
            projected_samples_t(block, scales, directions), **kw).T,
    }


def relative_loss_t(arr_a: torch.Tensor, arr_b: torch.Tensor) -> torch.Tensor:
    """L_s = ||S(A) - S(B)||^2 / ||S(B)||^2, spec section 4.

    `arr_b` is the frozen target array and carries no gradient. No square root,
    no second square, no L1 term, no per-curve normalization.
    """
    a, b = arr_a.reshape(-1), arr_b.reshape(-1).detach()
    if a.shape != b.shape:
        raise ValueError("family arrays differ in shape: %r vs %r"
                         % (tuple(a.shape), tuple(b.shape)))
    den = torch.dot(b, b)
    return torch.dot(a - b, a - b) / den


def family_losses_t(fam, target) -> dict:
    return {s: relative_loss_t(fam[s], target[s]) for s in S.FAMILIES}


def q_val_t(losses) -> torch.Tensor:
    """Q_val = sum_s L_s. Already squared; never squared again."""
    out = None
    for s in S.FAMILIES:
        out = losses[s] if out is None else out + losses[s]
    return out


# ------------------------------------------------------------------ constants
def frozen_tensors(scales, device, dtype=torch.float64):
    """The projection scales and directions as tensors on the compute device."""
    sc = torch.as_tensor(np.asarray(scales, np.float64), dtype=dtype,
                         device=device)
    di = torch.as_tensor(np.asarray(S.PROJ_DIRECTIONS, np.float64), dtype=dtype,
                         device=device)
    return sc, di


def target_tensors(arrays, device, dtype=torch.float64) -> dict:
    """Freeze a NumPy family-array dict onto the device as constants."""
    return {s: torch.as_tensor(np.ascontiguousarray(
        np.asarray(arrays[s], np.float64)), dtype=dtype, device=device)
        for s in S.FAMILIES}


# ------------------------------------------------------------------- selftest
def selftest(device="cpu", dtype=torch.float64, seed=11, verbose=True) -> dict:
    """Assert the torch estimators equal the NumPy ones used for evaluation.

    This is a precondition of the trainer, not an extra audit: the checkpoint
    score and the published classification ratio must be the same functional.
    """
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((S.T_WIN, 32, 2)) * 0.8
    block[:, :, 1] += 0.3 * np.roll(block[:, :, 0], 3, axis=0)
    scales = S.projection_scales(block)

    # The reference must be the pure NumPy path. S.family_arrays routes its
    # densities through kde_eval, which itself prefers torch when torch is
    # importable, so calling it here would compare torch against torch.
    c = S.corr_window(block)
    c2 = S.corr_window(block ** 2)
    want = {
        "ACF": np.stack([c[0, 0], c[1, 1]]),
        "CCF": np.stack([c[0, 1], c[1, 0]]),
        "ACF2": np.stack([c2[0, 0], c2[1, 1]]),
        "PDF": S.kde_direct(block).T,
        "PROJ": S.kde_direct(S.projected_samples(block, scales)).T,
    }
    t = torch.as_tensor(block, dtype=dtype, device=device)
    sc, di = frozen_tensors(scales, device, dtype)
    got = family_arrays_t(t, sc, di, checkpoint=False)

    out = {}
    for s in S.FAMILIES:
        g = got[s].detach().cpu().numpy()
        w = np.asarray(want[s], np.float64)
        den = np.linalg.norm(w)
        rel = float(np.linalg.norm(g - w) / den) if den > 0 else 0.0
        out[s] = rel
        if not rel < 1e-11:
            raise AssertionError(
                "torch %s differs from the NumPy estimator by %.3e" % (s, rel))

    ref = rng.standard_normal(want["ACF"].shape)
    l_np = S.relative_loss(want["ACF"], ref)
    l_t = float(relative_loss_t(
        got["ACF"], torch.as_tensor(ref, dtype=dtype, device=device)))
    if abs(l_t - l_np) > 1e-12 * max(1.0, abs(l_np)):
        raise AssertionError("relative_loss disagrees: %r vs %r" % (l_t, l_np))
    out["relative_loss"] = abs(l_t - l_np)

    # The loss must actually propagate gradients to the samples.
    g = torch.tensor(block, dtype=dtype, device=device, requires_grad=True)
    fam = family_arrays_t(g, sc, di, checkpoint=True, sample_chunk_size=4096)
    tg = target_tensors(want, device, dtype)
    q_val_t(family_losses_t(fam, tg)).backward()
    if g.grad is None or not torch.isfinite(g.grad).all():
        raise AssertionError("Q_val gradient is missing or non-finite")
    if float(g.grad.abs().max()) == 0.0:
        raise AssertionError("Q_val gradient is identically zero")
    out["grad_absmax"] = float(g.grad.abs().max())

    if verbose:
        print("selftest  " + "  ".join(
            "%s %.1e" % (s, out[s]) for s in S.FAMILIES)
            + "  | grad |max %.3e" % out["grad_absmax"])
    return out


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev)
    selftest(device=dev)
    print("OK")
