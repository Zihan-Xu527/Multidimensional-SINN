#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistic-based loss functions for multidimensional SINN.

Expected trajectory shape
-------------------------
Most losses expect trajectories with shape

    (time_length, batch_size, dim)

where
    time_length: number of time steps,
    batch_size: number of independent sample trajectories,
    dim: dimension of the stochastic process.

For the 2D Langevin example, dim = 2 and the two channels are

    x[..., 0] = q1_tilde,
    x[..., 1] = q2_tilde.

This file defines losses for:
    - ACF: autocorrelation of each component,
    - CCF: cross-correlation between different components,
    - PDF: marginal PDFs of each component,
    - PDF2/PDF4/PDF6: PDFs of selected linear combinations,
    - PDF2D: 2D joint PDF.
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


class StatLoss(_Loss, metaclass=ABCMeta):
    """Base class for statistic-based losses.

    A ``StatLoss`` object stores a target statistic computed from empirical
    data. During training, the same statistic is computed from generated
    trajectories and compared with the stored target.
    """

    eps: float = 1e-12

    @classmethod
    def from_expr(cls, expr: Callable[[float], float], t: Iterable[float], **options):
        """Create a target statistic by evaluating an expression on a grid."""
        target = np.fromiter(map(expr, t), dtype=np.float32)
        return cls(target, **options)

    def __init__(
        self,
        target: TensorLike,
        pointwise_loss: Union[str, Callable] = "mse_loss",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        # Backward compatibility:
        # Older code calls cls(..., loss=lower_level_loss).  The original file
        # stored this in self.loss but did not use it.  Here we intentionally
        # support both names: `loss` and `pointwise_loss`.
        if "loss" in kwargs:
            pointwise_loss = kwargs.pop("loss")

        self._target = torch.as_tensor(target, dtype=torch.float32)
        if device is not None:
            self._target = self._target.to(device)
        self._target = self._target.clone().detach()

        if callable(pointwise_loss):
            self._loss = pointwise_loss
        else:
            try:
                self._loss = getattr(F, pointwise_loss)
            except AttributeError as exc:
                raise RuntimeError(f"Unrecognized pointwise loss: {pointwise_loss}") from exc

        # Store additional options such as lower, upper, n, bw, sample_lags.
        self.__dict__.update(**kwargs)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _check_trajectory_shape(x: torch.Tensor, name: str = "x") -> None:
        if x.ndim != 3:
            raise ValueError(
                f"{name} must have shape (time_length, batch_size, dim), "
                f"but got shape {tuple(x.shape)}."
            )

    @staticmethod
    def _lag_indices(
        lags: Optional[Union[int, Sequence[int], np.ndarray, torch.Tensor]],
        time_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert a lag specification into valid integer lag indices."""
        if lags is None:
            lag_idx = torch.arange(time_length, device=device)
        elif isinstance(lags, int):
            if lags > time_length:
                raise ValueError(
                    f"lags={lags} is larger than the trajectory length "
                    f"time_length={time_length}. Use a smaller LAGS value, "
                    "for example target.shape[0] // 2."
                )
            lag_idx = torch.arange(lags, device=device)
        else:
            lag_idx = torch.as_tensor(lags, dtype=torch.long, device=device)
            if torch.any(lag_idx >= time_length) or torch.any(lag_idx < 0):
                raise ValueError(
                    f"All lag indices must be between 0 and {time_length - 1}."
                )
        return lag_idx.long()

    @staticmethod
    def _kde_bandwidth(num_samples: int, dim: int, bw: Optional[float]) -> float:
        """Return the Gaussian KDE bandwidth.

        The default bandwidth uses the common rule n^{-1/(d+4)}.
        For backward compatibility with the original code, if ``bw`` is given,
        it is interpreted as the denominator in n^{-1/bw}, not as the bandwidth
        value itself.
        """
        exponent_denominator = dim + 4 if bw is None else bw
        return num_samples ** (-1.0 / exponent_denominator)

    @staticmethod
    def _gaussian_kde_1d(samples: torch.Tensor, lower: float, upper: float, n: int, bw: Optional[float]):
        """Evaluate a 1D Gaussian KDE on a uniform grid."""
        samples = torch.ravel(samples)
        grid = torch.linspace(lower, upper, n, device=samples.device)
        bandwidth = StatLoss._kde_bandwidth(len(samples), dim=1, bw=bw)
        norm_factor = np.sqrt(2.0 * np.pi) * len(samples) * bandwidth
        values = torch.exp(-0.5 * ((samples[:, None] - grid[None, :]) / bandwidth) ** 2)
        return values.sum(dim=0) / norm_factor

    # ------------------------------------------------------------------
    # Correlation statistics
    # ------------------------------------------------------------------
    @staticmethod
    def acf(x: torch.Tensor, lags=None, method: str = "fft") -> torch.Tensor:
        """Compute normalized autocorrelation functions.

        Parameters
        ----------
        x:
            Tensor with shape (time_length, batch_size, dim).
        lags:
            Number of lags or explicit lag indices.
        method:
            ``"fft"`` or ``"bruteforce"``.

        Returns
        -------
        Tensor with shape (num_lags, dim).
        """
        StatLoss._check_trajectory_shape(x)
        time_length = x.shape[0]
        lag_idx = StatLoss._lag_indices(lags, time_length, x.device)

        # Center each component using the mean over time and batch.
        x_centered = x - x.mean(dim=(0, 1), keepdim=True)

        if method == "fft":
            f = torch.fft.fft(x_centered, n=2 * time_length - 1, dim=0)
            acf_full = torch.fft.ifft(f * f.conj(), dim=0).real[:time_length]
            acf_full = acf_full.mean(dim=1)  # average over batch
            return acf_full[lag_idx] / (acf_full[0] + StatLoss.eps)

        if method == "bruteforce":
            corr = torch.zeros((len(lag_idx), x.shape[2]), device=x.device)
            denom = torch.sum(x_centered**2, dim=(0, 1)) + StatLoss.eps
            for i, lag in enumerate(lag_idx.tolist()):
                if lag == 0:
                    u = v = x_centered
                else:
                    u = x_centered[:-lag]
                    v = x_centered[lag:]
                corr[i] = torch.sum(u * v, dim=(0, 1)) / denom
            return corr

        raise NotImplementedError(f"Unknown ACF method: {method}")

    @staticmethod
    def ccf(x: torch.Tensor, lags=None, method: str = "fft") -> torch.Tensor:
        """Compute normalized cross-correlation functions.

        For dim = 2, the output contains two ordered pairs:

            column 0: CCF(q1, q2),
            column 1: CCF(q2, q1).

        For general dim, all ordered pairs (j, k), j != k, are included.

        Returns
        -------
        Tensor with shape (num_lags, dim * (dim - 1)).
        """
        StatLoss._check_trajectory_shape(x)
        time_length, _, dim = x.shape
        lag_idx = StatLoss._lag_indices(lags, time_length, x.device)

        x_centered = x - x.mean(dim=(0, 1), keepdim=True)
        corr = torch.zeros((len(lag_idx), dim * (dim - 1)), device=x.device)

        col = 0
        for j in range(dim):
            xj = x_centered[..., j]
            for k in range(dim):
                if j == k:
                    continue
                xk = x_centered[..., k]

                if method == "fft":
                    fj = torch.fft.fft(xj, n=2 * time_length - 1, dim=0)
                    fk = torch.fft.fft(xk, n=2 * time_length - 1, dim=0)
                    ccf_full = torch.fft.ifft(fj * fk.conj(), dim=0).real[:time_length]
                    ccf_full = ccf_full.mean(dim=1)  # average over batch

                    acf_j0 = torch.fft.ifft(fj * fj.conj(), dim=0).real[0].mean()
                    acf_k0 = torch.fft.ifft(fk * fk.conj(), dim=0).real[0].mean()
                    denom = torch.sqrt(acf_j0 * acf_k0 + StatLoss.eps)
                    corr[:, col] = ccf_full[lag_idx] / denom

                elif method == "bruteforce":
                    denom = torch.sqrt(torch.sum(xj**2) * torch.sum(xk**2) + StatLoss.eps)
                    for i, lag in enumerate(lag_idx.tolist()):
                        if lag == 0:
                            u, v = xj, xk
                        else:
                            u, v = xj[:-lag], xk[lag:]
                        corr[i, col] = torch.sum(u * v) / denom

                else:
                    raise NotImplementedError(f"Unknown CCF method: {method}")

                col += 1

        return corr

    # ------------------------------------------------------------------
    # KDE statistics
    # ------------------------------------------------------------------
    @staticmethod
    def gauss_kde(x: torch.Tensor, lower: float, upper: float, n: int, bw: Optional[float]):
        """Marginal PDFs of each component.

        Returns an array with shape (n, dim).  For dim = 2:
            column 0: PDF of q1_tilde,
            column 1: PDF of q2_tilde.
        """
        StatLoss._check_trajectory_shape(x)
        pdf = torch.zeros((n, x.shape[2]), device=x.device)
        for i in range(x.shape[2]):
            pdf[:, i] = StatLoss._gaussian_kde_1d(x[..., i], lower, upper, n, bw)
        return pdf

    @staticmethod
    def gauss_kde_linear_combinations(
        x: torch.Tensor,
        lower: float,
        upper: float,
        n: int,
        bw: Optional[float],
        coefficients: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        """PDFs of selected linear combinations of two components.

        Each coefficient pair [a, b] defines the projection

            a * q1_tilde + b * q2_tilde.

        This is useful for capturing joint-distribution information without
        estimating the full 2D PDF.
        """
        StatLoss._check_trajectory_shape(x)
        if x.shape[2] != 2:
            raise ValueError("Linear-combination PDF losses currently require dim = 2.")

        q1 = x[..., 0]
        q2 = x[..., 1]
        pdf = torch.zeros((n, len(coefficients)), device=x.device)
        for i, (a, b) in enumerate(coefficients):
            samples = a * q1 + b * q2
            pdf[:, i] = StatLoss._gaussian_kde_1d(samples, lower, upper, n, bw)
        return pdf

    @staticmethod
    def gauss_kde_sum(x, lower, upper, n, bw):
        """PDF of q1_tilde + q2_tilde."""
        return StatLoss.gauss_kde_linear_combinations(x, lower, upper, n, bw, [(1.0, 1.0)]).squeeze(-1)

    @staticmethod
    def gauss_kde_diff(x, lower, upper, n, bw):
        """PDF of q1_tilde - q2_tilde."""
        return StatLoss.gauss_kde_linear_combinations(x, lower, upper, n, bw, [(1.0, -1.0)]).squeeze(-1)

    @staticmethod
    def gauss_kde2(x, lower, upper, n, bw):
        """PDFs of q1_tilde + q2_tilde and q1_tilde - q2_tilde."""
        return StatLoss.gauss_kde_linear_combinations(
            x, lower, upper, n, bw,
            [(1.0, 1.0), (1.0, -1.0)],
        )

    @staticmethod
    def gauss_kde4(x, lower, upper, n, bw):
        """PDFs of four linear combinations.

        Columns are:
            q1 + q2, q1 - q2, q1 + 2 q2, q1 - 2 q2.
        """
        return StatLoss.gauss_kde_linear_combinations(
            x, lower, upper, n, bw,
            [(1.0, 1.0), (1.0, -1.0), (1.0, 2.0), (1.0, -2.0)],
        )

    @staticmethod
    def gauss_kde6(x, lower, upper, n, bw):
        """PDFs of six linear combinations.

        Columns are:
            0: q1 + q2
            1: q1 - q2
            2: q1 + 2 q2
            3: q1 - 2 q2
            4: 2 q1 + q2
            5: 2 q1 - q2
        """
        return StatLoss.gauss_kde_linear_combinations(
            x, lower, upper, n, bw,
            [
                (1.0, 1.0),
                (1.0, -1.0),
                (1.0, 2.0),
                (1.0, -2.0),
                (2.0, 1.0),
                (2.0, -1.0),
            ],
        )

    @staticmethod
    def gauss_kde2D(x: torch.Tensor, lower: float, upper: float, n: int, bw: Optional[float]):
        """Estimate the 2D joint PDF of (q1_tilde, q2_tilde).

        Returns a tensor with shape (n, n), evaluated on a square grid.
        """
        StatLoss._check_trajectory_shape(x)
        if x.shape[2] != 2:
            raise ValueError("2D joint KDE requires dim = 2.")

        q1 = torch.ravel(x[..., 0])
        q2 = torch.ravel(x[..., 1])
        grid_x = torch.linspace(lower, upper, n, device=x.device)
        grid_y = torch.linspace(lower, upper, n, device=x.device)
        bandwidth = StatLoss._kde_bandwidth(len(q1), dim=2, bw=bw)
        norm_factor = 2.0 * np.pi * len(q1) * bandwidth**2

        pdf = torch.zeros((n, n), device=x.device)
        q1_sq = (q1[:, None] - grid_x[None, :]) ** 2
        for i, y_value in enumerate(grid_y):
            radius_sq = q1_sq + (q2[:, None] - y_value) ** 2
            pdf[:, i] = torch.exp(-0.5 * radius_sq / bandwidth**2).sum(dim=0) / norm_factor
        return pdf

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compare generated trajectories with the stored target statistic."""


# ----------------------------------------------------------------------
# Correlation loss classes
# ----------------------------------------------------------------------
class ACFLoss(StatLoss):
    """ACF loss computed using FFT."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.acf(data, lags=lags, method="fft"), **options)

    def forward(self, input):
        return self._loss(self.acf(input, lags=len(self._target), method="fft"), self._target)


class BruteForceACFLoss(StatLoss):
    """ACF loss computed directly from lagged products."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.acf(data, lags=lags, method="bruteforce"), **options)

    def forward(self, input):
        return self._loss(self.acf(input, lags=len(self._target), method="bruteforce"), self._target)


class RandomBruteForceACFLoss(StatLoss):
    """ACF loss using a random subset of lags at each call."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.acf(data, lags=lags, method="bruteforce"), **options)

    def forward(self, input):
        lags = np.random.choice(len(self._target), self.sample_lags, replace=False)
        return self._loss(self.acf(input, lags=lags, method="bruteforce"), self._target[lags])


class CCFLoss(StatLoss):
    """CCF loss computed using FFT."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.ccf(data, lags=lags, method="fft"), **options)

    def forward(self, input):
        return self._loss(self.ccf(input, lags=len(self._target), method="fft"), self._target)


class BruteForceCCFLoss(StatLoss):
    """CCF loss computed directly from lagged products."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.ccf(data, lags=lags, method="bruteforce"), **options)

    def forward(self, input):
        return self._loss(self.ccf(input, lags=len(self._target), method="bruteforce"), self._target)


class RandomBruteForceCCFLoss(StatLoss):
    """CCF loss using a random subset of lags at each call."""

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        return cls(cls.ccf(data, lags=lags, method="bruteforce"), **options)

    def forward(self, input):
        lags = np.random.choice(len(self._target), self.sample_lags, replace=False)
        return self._loss(self.ccf(input, lags=lags, method="bruteforce"), self._target[lags])


# ----------------------------------------------------------------------
# Density loss classes
# ----------------------------------------------------------------------
class DensityLoss(StatLoss):
    """Marginal PDF loss for each output component."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLossSum(StatLoss):
    """PDF loss for q1_tilde + q2_tilde."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde_sum(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde_sum(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLossDiff(StatLoss):
    """PDF loss for q1_tilde - q2_tilde."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde_diff(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde_diff(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLoss2(StatLoss):
    """PDF loss for q1 ± q2."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde2(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde2(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLoss4(StatLoss):
    """PDF loss for four selected linear combinations."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde4(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde4(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLoss6(StatLoss):
    """PDF loss for six selected linear combinations."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde6(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde6(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


class DensityLoss2D(StatLoss):
    """2D joint PDF loss for (q1_tilde, q2_tilde)."""

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        target = cls.gauss_kde2D(data, lower=lower, upper=upper, n=n, bw=bw)
        return cls(target, **options, lower=lower, upper=upper, n=n, bw=bw)

    def forward(self, input):
        current = self.gauss_kde2D(input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw)
        return self._loss(current, self._target)


# ----------------------------------------------------------------------
# Public factory function
# ----------------------------------------------------------------------
def make_loss(stat, data, loss_type=None, **kwargs):
    """Create a statistic-based loss object.

    Parameters
    ----------
    stat:
        Statistic name. Supported values are:

            'pdf', 'pdfsum', 'pdfdiff', 'pdf2', 'pdf4', 'pdf6', 'pdf2D',
            'acf[fft]', 'acf[bruteforce]', 'acf[randombrute]',
            'ccf[fft]', 'ccf[bruteforce]', 'ccf[randombrute]'.

    data:
        Usually an empirical trajectory tensor with shape
        (time_length, batch_size, dim).  The target statistic is computed from
        this tensor and stored inside the returned loss object.

    loss_type:
        List of PyTorch pointwise losses used to compare statistics.
        Default: ['mse_loss'].

    kwargs:
        Extra arguments required by each loss, such as ``lags`` for ACF/CCF and
        ``lower``, ``upper``, ``n``, ``bw`` for PDF losses.
    """
    if loss_type is None:
        loss_type = ["mse_loss"]

    loss_classes = {
        "pdf": DensityLoss,
        "pdfsum": DensityLossSum,
        "pdfdiff": DensityLossDiff,
        "pdf2": DensityLoss2,
        "pdf4": DensityLoss4,
        "pdf6": DensityLoss6,
        "pdf2D": DensityLoss2D,
        "acf[fft]": ACFLoss,
        "acf[bruteforce]": BruteForceACFLoss,
        "acf[randombrute]": RandomBruteForceACFLoss,
        "ccf[fft]": CCFLoss,
        "ccf[bruteforce]": BruteForceCCFLoss,
        "ccf[randombrute]": RandomBruteForceCCFLoss,
    }

    if stat not in loss_classes:
        raise RuntimeError(f"Unknown stat {stat}.")

    def lower_level_loss(a, b):
        # Sum one or more pointwise losses, e.g. MSE + L1.
        return sum(getattr(F, name)(a, b) for name in loss_type)

    loss_cls = loss_classes[stat]

    # Empirical trajectories have shape (time_length, batch_size, dim).
    # For this project, target statistics are normally created from such data.
    if len(data.shape) == 3:
        return loss_cls.from_empirical_data(data, loss=lower_level_loss, **kwargs)

    # If the user passes a precomputed target statistic directly, use it as is.
    return loss_cls(data, loss=lower_level_loss, **kwargs)
