"""Deterministic random streams for Section 5.2, spec section 6.

One generator object per purpose per run, created once and advanced
continuously. The checkpoint-monitoring stream in particular must draw a NEW
tensor at every validation event and must not be re-seeded per event: the
archived FPU trainer drew one fixed `val_input` before the loop and reused it
for all 20000 iterations, which lets the checkpoint overfit a single noise
draw. That behaviour is forbidden here.
"""

from __future__ import annotations

from llv_spec import (SEED_BASE, SEED_BURNIN_BOOTSTRAP, SEED_LONGLAG_BOOTSTRAP,
                      TRAIN_CONFIG)

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    HAVE_TORCH = False

import numpy as np


def seed_for(purpose: str, run: int) -> int:
    if purpose not in SEED_BASE:
        raise KeyError("unknown stream purpose %r" % (purpose,))
    return SEED_BASE[purpose] + run


class RunStreams:
    """The model-side generators for one training run.

    Every generator lives on the recorded device and is advanced continuously;
    `initial_states` is captured at construction for the metadata record.
    """

    PURPOSES = tuple(SEED_BASE)

    def __init__(self, run: int, device: str = "cpu"):
        if not HAVE_TORCH:
            raise RuntimeError("RunStreams requires torch")
        self.run = int(run)
        self.device = device
        self.seeds = {p: seed_for(p, self.run) for p in self.PURPOSES}
        self.gen = {}
        for p, s in self.seeds.items():
            g = torch.Generator(device=device)
            g.manual_seed(s)
            self.gen[p] = g
        self.initial_states = {p: g.get_state().clone()
                               for p, g in self.gen.items()}

    def noise(self, purpose: str, steps: int, batch: int) -> "torch.Tensor":
        """Draw a fresh (steps, batch, input_width) noise tensor.

        Channels beyond `d_in_active` are identically zero, matching the
        Section 4 convention that keeps the parameter count fixed across the
        input-dimension comparison.
        """
        width = TRAIN_CONFIG["input_width"]
        active = TRAIN_CONFIG["d_in_active"]
        out = torch.zeros(steps, batch, width, device=self.device)
        out[:, :, :active] = torch.randn(
            steps, batch, active, generator=self.gen[purpose],
            device=self.device)
        return out

    def state_snapshot(self) -> dict:
        return {p: g.get_state().clone() for p, g in self.gen.items()}

    def metadata(self) -> dict:
        meta = {
            "run": self.run,
            "device": str(self.device),
            "seeds": dict(self.seeds),
            "torch_version": torch.__version__ if HAVE_TORCH else None,
            "cuda_version": (torch.version.cuda if HAVE_TORCH else None),
            "initial_states_sha": {
                p: _sha_tensor(v) for p, v in self.initial_states.items()},
        }
        if HAVE_TORCH and torch.cuda.is_available():
            meta["device_name"] = torch.cuda.get_device_name(0)
        return meta


def _sha_tensor(t) -> str:
    import hashlib
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


def longlag_rng() -> np.random.Generator:
    """NumPy generator for the long-lag reference bootstrap (seed 7000).

    Calibrations use their own NumPy generators, so adding one calibration
    cannot advance another calibration's stream.
    """
    return np.random.default_rng(SEED_LONGLAG_BOOTSTRAP)


def burnin_rng() -> np.random.Generator:
    """NumPy generator for the burn-in/stationarity bootstrap (seed 7001)."""
    return np.random.default_rng(SEED_BURNIN_BOOTSTRAP)
