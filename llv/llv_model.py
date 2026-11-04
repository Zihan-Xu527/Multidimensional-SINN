"""The Section 5.2 generator and its checkpoint format, spec section 5.

2 stacked LSTM layers, 10 hidden units per layer, linear 2-output readout.
Input width 4; the first 2 channels receive independent standard Gaussian
noise and the remaining 2 are identically zero, which keeps the parameter count
fixed across the Section 4 input-dimension comparison.

This is deliberately NOT the FPU configuration (3 layers x 75 hidden). The
constructor reads llv_spec.TRAIN_CONFIG and refuses to be given a different
shape silently.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import llv_spec as S


class LLVGenerator(nn.Module):
    def __init__(self, config=None, dtype=torch.float64):
        super().__init__()
        cfg = dict(S.TRAIN_CONFIG if config is None else config)
        self.input_width = int(cfg["input_width"])
        self.hidden = int(cfg["hidden_units"])
        self.layers = int(cfg["lstm_layers"])
        self.out_dim = int(cfg["d_out"])
        self.lstm = nn.LSTM(self.input_width, self.hidden, self.layers)
        self.readout = nn.Linear(self.hidden, self.out_dim)
        self.to(dtype)

    # ---------------------------------------------------------------- init --
    def reset_parameters(self, generator: torch.Generator):
        """Re-initialize every parameter from an explicit generator.

        PyTorch's own reset_parameters draws from the global RNG, which would
        make `param_init` seed m unreproducible. The distribution is the same
        one PyTorch uses, U(-1/sqrt(hidden), 1/sqrt(hidden)) for the recurrent
        stack and U(-1/sqrt(in_features), ...) for the readout, so this changes
        only which stream the numbers come from.
        """
        k = 1.0 / np.sqrt(self.hidden)

        def fill(p):
            # torch.rand takes an explicit generator on every supported
            # version; Tensor.uniform_ does not, so it is avoided here.
            u = torch.rand(p.shape, generator=generator, device=p.device,
                           dtype=p.dtype)
            p.copy_((u * 2.0 - 1.0) * k)

        with torch.no_grad():
            for _, p in self.lstm.named_parameters():
                fill(p)
            fill(self.readout.weight)
            fill(self.readout.bias)

    # ------------------------------------------------------------- forward --
    def forward(self, noise: torch.Tensor, state=None):
        """noise: (steps, batch, input_width). Returns (output, state).

        `state=None` means zero hidden and cell states, which is how every
        rollout in this specification starts. Passing a state back in continues
        the recurrence without reset; that is what the 10^6-step stability run
        and the 1200-step continuous rollout need, and it is why nothing here
        implements a window as "the last k steps".
        """
        y, state = self.lstm(noise, state)
        return self.readout(y), state

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaptiveScales(nn.Module):
    """sigma_s = exp(alpha_s), all alpha_s = 0 initially (spec section 5).

    The exponential parameterization is the LLV choice and differs from the
    FPU tree's direct sigma with clamp_(min=1e-4); sigma is positive by
    construction here, so no clamp exists and none may be added.
    """

    def __init__(self, families=S.FAMILIES, dtype=torch.float64):
        super().__init__()
        self.families = tuple(families)
        self.alpha = nn.Parameter(
            torch.zeros(len(self.families), dtype=dtype))
        if float(self.alpha.abs().max()) != 0.0:
            raise AssertionError("alpha must start at zero")

    def sigma(self) -> torch.Tensor:
        return torch.exp(self.alpha)

    def theta_objective(self, losses) -> torch.Tensor:
        """sum_s L_s / (2 sigma_s^2), with sigma held fixed.

        Block coordinate descent: this is the theta half. It is gradient
        identical to joint descent on the balanced objective, because the
        log sigma term does not depend on theta.
        """
        w = torch.exp(-2.0 * self.alpha.detach())
        out = None
        for i, s in enumerate(self.families):
            term = losses[s] * w[i] * 0.5
            out = term if out is None else out + term
        return out

    def scale_objective(self, losses) -> torch.Tensor:
        """sum_s [ L_s / (2 sigma_s^2) + log sigma_s ], with L_s held fixed.

        log sigma_s is exactly alpha_s under this parameterization.
        """
        out = None
        for i, s in enumerate(self.families):
            term = losses[s].detach() * torch.exp(-2.0 * self.alpha[i]) * 0.5 \
                + self.alpha[i]
            out = term if out is None else out + term
        return out

    def state_values(self) -> dict:
        a = self.alpha.detach().cpu().numpy()
        return {"alpha": {s: float(a[i]) for i, s in enumerate(self.families)},
                "sigma": {s: float(np.exp(a[i]))
                          for i, s in enumerate(self.families)}}


def build(run: int, device: str, dtype=torch.float64, config=None):
    """Generator plus adaptive scales for one run, parameters seeded by `run`.

    `config` defaults to the frozen S.TRAIN_CONFIG. Passing a modified one is
    how the hidden-width feasibility pilot varies capacity without editing the
    frozen table; the caller is responsible for recording the deviation.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(S.SEED_BASE["param_init"] + int(run))
    model = LLVGenerator(config=config, dtype=dtype).to(device)
    model.reset_parameters(gen)
    scales = AdaptiveScales(dtype=dtype).to(device)
    return model, scales, gen


# --------------------------------------------------------------- step check --
def step_check(device=None, dtype=torch.float64, verbose=True) -> dict:
    """One full training step on synthetic targets, with no archive.

    This exercises every torch API the trainer uses - parameter init from an
    explicit generator, the LSTM rollout, the differentiable family
    statistics, both optimizer halves - so an API or device problem surfaces
    here in seconds instead of after the archive has been loaded.
    """
    import llv_streams as ST
    import llv_torch as T

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sigmas, _ = build(run=0, device=device, dtype=dtype)
    streams = ST.RunStreams(0, device=device)
    cfg = S.TRAIN_CONFIG
    opt_t = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                             betas=cfg["betas"])
    opt_s = torch.optim.Adam(sigmas.parameters(), lr=cfg["lr"],
                             betas=cfg["betas"])

    batch, unroll = 8, int(cfg["raw_train_unroll"])
    lo, hi = cfg["scored_slice"]
    rng = np.random.default_rng(0)
    fake = rng.standard_normal((S.T_WIN, batch, 2))
    scales = S.projection_scales(fake)
    target = T.target_tensors(
        {"ACF": np.stack([np.ones(S.K_LAGS), np.ones(S.K_LAGS)]),
         "CCF": np.stack([np.full(S.K_LAGS, 0.1), np.full(S.K_LAGS, 0.1)]),
         "ACF2": np.stack([np.ones(S.K_LAGS), np.ones(S.K_LAGS)]),
         "PDF": np.full((2, S.GRID_N), 0.2),
         "PROJ": np.full((S.N_PROJ, S.GRID_N), 0.2)}, device, dtype)
    sc, di = T.frozen_tensors(scales, device, dtype)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    noise = streams.noise("train_noise", unroll, batch).to(dtype)
    out, _ = model(noise)
    if tuple(out.shape) != (unroll, batch, model.out_dim):
        raise AssertionError("rollout shape %r" % (tuple(out.shape),))
    fam = T.family_arrays_t(out[lo:hi], sc, di, checkpoint=True,
                            sample_chunk_size=4096)
    losses = T.family_losses_t(fam, target)

    opt_t.zero_grad(set_to_none=True)
    sigmas.theta_objective(losses).backward()
    opt_t.step()
    opt_s.zero_grad(set_to_none=True)
    sigmas.scale_objective(losses).backward()
    opt_s.step()

    moved = sum(1 for k, v in model.state_dict().items()
                if not torch.equal(v, before[k]))
    if moved == 0:
        raise AssertionError("no model parameter changed after one step")
    a = sigmas.alpha.detach().cpu().numpy()
    if float(np.abs(a).max()) == 0.0:
        raise AssertionError("alpha did not move; the sigma half is inert")

    # Reproducibility: the same run and device must rebuild the same weights.
    m2, _, _ = build(run=0, device=device, dtype=dtype)
    for k, v in m2.state_dict().items():
        if not torch.equal(v, before[k]):
            raise AssertionError("parameter init is not reproducible at %s" % k)

    info = {"device": str(device), "n_parameters": model.n_parameters(),
            "params_changed": moved,
            "Q_val": float(T.q_val_t(losses)),
            "alpha_after": a.tolist()}
    if verbose:
        print("step_check  device %s  params %d (%d moved)  Q %.6e"
              % (info["device"], info["n_parameters"], moved, info["Q_val"]))
        print("            alpha after one step " + np.array2string(
            a, precision=6))
    return info


if __name__ == "__main__":
    step_check()
    print("OK")
