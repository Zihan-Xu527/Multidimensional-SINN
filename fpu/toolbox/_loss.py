#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import scipy.stats as st


class StatLoss(_Loss, metaclass=ABCMeta):
    '''Base class for all statistic-based loss functions. This is an abstract
    class because it only handles the processing for the 'target' properties,
    and leaves the definition of actual comparison operations to concrete
    subclasses.
    '''

    @classmethod
    def from_expr(cls, expr, t, **options):
        '''Specify the target statistic function with an expression
        at given evaluation points.

        Parameters
        ----------
        expr: callable
            A function to be evaluated at the given points.
        t: list-like
            A list of uniformly spaced grid points on which the expression will
            be evaluated.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(np.fromiter(map(expr, t), dtype=np.float32), **options)

    def __init__(self, target, pointwise_loss='mse_loss', device='cpu',
                 **kwargs):
        '''Direct specification of the target statistic function in discretized
        form.

        Parameters
        ----------
        array: list-like The target statistic function.
        '''
        super().__init__()
        self._target = target.clone().detach()
#         self._target = torch.tensor(target, dtype=torch.float32, device=device)

        if callable(pointwise_loss):
            self._loss = pointwise_loss
        else:
            try:
                self._loss = getattr(F, pointwise_loss)
            except AttributeError as e:
                raise RuntimeError(f'Unrecognized pointwise loss. Error:\n{e}')
        self.__dict__.update(**kwargs)

    @staticmethod
    def acf(x, lags=None, method='fft'):
        mean = (x.mean(axis=0)).mean(axis=0)
        x = x - mean[None,None,...]
        if method == 'fft':
            f = torch.fft.fft(x, x.shape[0] * 2 - 1, dim=0)
            acf = torch.fft.ifft(
                f * f.conj(), dim=0
            ).real[
                :x.shape[0]
            ].mean(axis=1)
            return acf[:lags, ...] / acf[0, ...]
        elif method == 'bruteforce':
            if lags is None:
                lags = torch.arange(x.shape[0])
            elif isinstance(lags, int):
                lags = torch.arange(lags)
            else:
                lags = lags.clone().detach()
#                 lags = torch.tensor(lags, dtype=torch.int32)
            corr = torch.zeros((len(lags), *x.shape[2:]), device=x.device)
            for i, lag in enumerate(lags):
                if lag == 0:
                    u = v = x
                elif lag < x.shape[0]:
                    u, v = x[:-lag, ...], x[lag:, ...]
                else:
                    continue
                corr[i, ...] = torch.sum(u * v, axis=[0, 1]) / (
                    torch.sqrt(
                        torch.sum(torch.square(x), axis=[0, 1]) *
                        torch.sum(torch.square(x), axis=[0, 1])
                    )
                )
            return corr
        else:
            raise NotImplementedError(f'Unknown method {method}.')
            

    
    
    @staticmethod
    def ccf(x, lags=None, method='fft'):
        '''Cross-correlation functions of every ordered pair of components.

        Works for any number of components d: the returned tensor has
        d * (d - 1) columns, ordered as (0,1), (0,2), ..., (1,0), (1,2), ...
        For d = 2 this reproduces the previous two-column layout exactly.
        '''
        columns = []
        mean = (x.mean(axis=0)).mean(axis=0)
        x_tmp = x - mean[None,None,...]
        n = 0
        for j in range(x_tmp.size()[2]):
            x = x_tmp[...,j]
            for k in range(x_tmp.size()[2]):
                if j != k:
                    y = x_tmp[...,k]
                    if method == 'fft':
                        f = torch.fft.fft(x, x.shape[0] * 2 - 1, dim=0)
                        g = torch.fft.fft(y, y.shape[0] * 2 - 1, dim=0)
                        ccf = torch.fft.ifft(
                            f * g.conj(), dim=0
                        ).real[
                            :x.shape[0]
                        ].mean(axis=1)

                        acf_f = torch.fft.ifft(
                            f * f.conj(), dim=0
                        ).real[
                            :x.shape[0]
                        ].mean(axis=1)
                    
                        acf_g = torch.fft.ifft(
                            g * g.conj(), dim=0
                        ).real[
                            :y.shape[0]
                        ].mean(axis=1)
                    
                        columns.append(
                            ccf[:lags, ...] / torch.sqrt(acf_f[0, ...])
                            / torch.sqrt(acf_g[0, ...])
                        )
                        n += 1
                    elif method == 'bruteforce':
                        if lags is None:
                            lags = torch.arange(x.shape[0])
                        elif isinstance(lags, int):
                            lags = torch.arange(lags)
                        else:
                            lags = lags.clone().detach()
                        corr_tmp = torch.zeros((len(lags), *x.shape[2:]), device=x.device)
                        for i, lag in enumerate(lags):
                            if lag == 0:
                                u = x
                                v = y
                            elif lag < x.shape[0]:
                                u, v = x[:-lag, ...], y[lag:, ...]
                            else:
                                continue
                            corr_tmp[i, ...] = torch.sum(u * v, axis=[0, 1]) / (
                                torch.sqrt(
                                    torch.sum(torch.square(x), axis=[0, 1]) *
                                    torch.sum(torch.square(y), axis=[0, 1])
                                )
                            )
                        columns.append(corr_tmp)
                        n += 1
                    else:
                        raise NotImplementedError(f'Unknown method {method}.')
        return torch.stack(columns, dim=-1)




    # ------------------------------------------------------------------
    # Dimension-agnostic projected densities.
    #
    # The 1D densities of a collection of linear combinations
    # w . x, |w| = 1, replace the joint PDF of the d components.  The
    # directions are grouped into mutually orthogonal frames: each frame is
    # an orthonormal basis of R^d, so a frame contributes d projections and
    # `n_frames` frames contribute n_frames * d directions in total.
    # ------------------------------------------------------------------

    # Three orthogonal frames in R^2 (six directions), grouped by frame:
    #   frame 1: (1,1), (1,-1)
    #   frame 2: (1,2), (2,-1)
    #   frame 3: (1,-2), (2,1)
    # This is exactly the direction set of the original `gauss_kde6`; only
    # the order of the last three columns differs (old order was (1,-2),
    # (2,1), (2,-1)), so the summed loss is unchanged and only per-column
    # plot labels need to follow the new order.
    _FRAMES_2D = (
        (1., 1.), (1., -1.), (1., 2.), (2., -1.), (1., -2.), (2., 1.)
    )

    # Three orthogonal frames in R^3 (nine directions), grouped by frame:
    #   frame 1: (1,1,1), (1,-1,0), (1,1,-2)
    #   frame 2: (1,1,-1), (0,1,1), (2,-1,1)
    #   frame 3: (1,-1,1), (1,1,0), (1,-1,-2)
    # Within each frame the three vectors are mutually orthogonal, and no
    # two of the nine directions are parallel.
    _FRAMES_3D = (
        (1., 1., 1.), (1., -1., 0.), (1., 1., -2.),
        (1., 1., -1.), (0., 1., 1.), (2., -1., 1.),
        (1., -1., 1.), (1., 1., 0.), (1., -1., -2.),
    )

    @staticmethod
    def _bandwidth(m, bw, bandwidth):
        """Resolve the KDE bandwidth.

        `bandwidth`, when given, is used verbatim.  This matters whenever the
        target and the generated ensemble have different sizes: the historical
        rule below reads the bandwidth off the sample count, so a target built
        from 20000 trajectories and a prediction from 400 would be smoothed by
        different amounts and could never agree.  Pass the same explicit value
        to both.
        """
        if bandwidth is not None:
            return float(bandwidth)
        if bw is None:
            return m**(-1 / 5)
        return m**(-1 / bw)

    @staticmethod
    def projection_directions(dim, n_frames=3, seed=0):
        '''Projection directions for `gauss_kde_proj`, as a unit-norm tensor
        of shape (n_frames * dim, dim).

        Parameters
        ----------
        dim: int
            Number of components d of the process.
        n_frames: int
            Number of mutually orthogonal frames.  Each frame supplies d
            directions, so the total number of projections is n_frames * d.
        seed: int
            Only used for d not in (2, 3), where the frames are obtained by
            QR factorization of pseudo-random matrices.  Fixed by default so
            that the direction set is reproducible.

        Notes
        -----
        For d = 2 and d = 3 the directions are the hard-coded integer frames
        above; `n_frames` then selects the first `n_frames` of the three.
        '''
        if dim == 2:
            table = StatLoss._FRAMES_2D
        elif dim == 3:
            table = StatLoss._FRAMES_3D
        else:
            table = None

        if table is not None:
            if n_frames > len(table) // dim:
                raise RuntimeError(
                    f'Only {len(table) // dim} tabulated frames are available '
                    f'for dim={dim}; got n_frames={n_frames}.'
                )
            directions = torch.tensor(
                table[:n_frames * dim], dtype=torch.get_default_dtype()
            )
        else:
            rng = np.random.default_rng(seed)
            frames = []
            for _ in range(n_frames):
                q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
                frames.append(q.T)
            directions = torch.tensor(
                np.concatenate(frames, axis=0), dtype=torch.get_default_dtype()
            )

        return directions / torch.linalg.vector_norm(
            directions, dim=1, keepdim=True
        )

    @staticmethod
    def projection_scales(x, directions):
        """Standard deviation of each projection w . x over the ensemble.

        Used as a FIXED rescaling derived from the target: the same constants
        are applied to the generated ensemble, so a wrong generated covariance
        still shows up as a projected density that is too wide or too narrow.
        Without it the projections span a range of scales (for three tagged
        particles on a periodic chain, sigma runs from about 0.65 to 1.41), so
        one common grid and one common bandwidth resolve them very unevenly
        and the widest projections get truncated.
        """
        dim = x.shape[-1]
        samples = x.reshape(-1, dim)
        w = torch.as_tensor(directions, device=x.device, dtype=x.dtype)
        if w.ndim == 1:
            w = w[None, :]
        w = w / torch.linalg.vector_norm(w, dim=1, keepdim=True)
        return (samples @ w.T).std(0)

    @staticmethod
    def gauss_kde_proj(x, directions, lower, upper, n, bw, chunk=None,
                       bandwidth=None, scale=None):
        '''KDE of the 1D densities of the projections w . x.

        Parameters
        ----------
        x: tensor of shape (..., d)
            Trajectories; all leading axes are pooled into the sample axis.
        directions: tensor-like of shape (K, d)
            Projection directions.  Normalized to unit Euclidean norm here,
            so the caller may pass unnormalized integer vectors.
        chunk: int, optional
            Accumulate the kernel sum over blocks of at most `chunk` samples.
            The result is unchanged; it only bounds the size of the
            intermediate (n_samples, n) array.  Useful when building the
            target from a large ensemble; leave as None inside the training
            loop, where autograd keeps every block alive anyway.

        Returns
        -------
        pdf: tensor of shape (n, K)
            Column k is the density of w_k . x on the grid, matching the
            layout of `gauss_kde` (grid first, statistic second).
        '''
        dim = x.shape[-1]
        samples = x.reshape(-1, dim)
        w = torch.as_tensor(directions, device=x.device, dtype=x.dtype)
        if w.ndim == 1:
            w = w[None, :]
        if w.shape[-1] != dim:
            raise RuntimeError(
                f'Directions have {w.shape[-1]} components but the data has '
                f'{dim}.'
            )
        w = w / torch.linalg.vector_norm(w, dim=1, keepdim=True)

        projected = samples @ w.T                       # (n_samples, K)
        if scale is not None:
            sc = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
            projected = projected / sc[None, :]
        grid = torch.linspace(lower, upper, n, device=x.device, dtype=x.dtype)

        m = samples.shape[0]
        bwp = StatLoss._bandwidth(m, bw, bandwidth)
        norm_factor = (2 * np.pi)**0.5 * m * bwp

        # One direction at a time: the intermediate (n_samples, n) array is
        # the memory bottleneck, and K of them at once does not fit for the
        # batch sizes used here.
        block = m if chunk is None else int(chunk)
        columns = []
        for k in range(w.shape[0]):
            p_k = projected[:, k]
            col = None
            for s in range(0, m, block):
                part = torch.sum(
                    torch.exp(
                        -0.5 * torch.square(
                            (p_k[s:s + block, None] - grid[None, :]) / bwp
                        )
                    ),
                    axis=0
                )
                col = part if col is None else col + part
            columns.append(col / norm_factor)
        return torch.stack(columns, dim=-1)

    @staticmethod
    def gauss_kde2D(x, lower, upper, n, bw, dims=(0, 1), bandwidth=None):
        pdf = torch.zeros((n, n), device=x.device)
        x_tmp = torch.ravel(x[...,dims[0]])
        y_tmp = torch.ravel(x[...,dims[1]])
        x_grid = torch.linspace(lower, upper, n, device=x.device)
        y_grid = torch.linspace(lower, upper, n, device=x.device)
        if bandwidth is not None:
            bwp = float(bandwidth)
        elif bw is None:
            bwp = len(x_tmp)**(-1 / 6)
        else:
            bwp = len(x_tmp)**(-1 / bw)
        norm_factor = (2 * np.pi) * len(x_tmp) * bwp * bwp 
        x_square = torch.square(x_tmp[:,None]-x_grid[None,:])
        for i in range(n):
            sum_square = x_square + torch.square(y_tmp[:,None]-y_grid[i])
            pdf[...,i] = torch.sum(
                torch.exp( -0.5 * ( sum_square ) / (bwp * bwp)
                         )
                ,axis=0 )/ norm_factor
        return pdf


    @staticmethod
    def gauss_kde_sum(x, lower, upper, n, bw):
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        xy_sum = x_tmp+y_tmp
        grid = torch.linspace(lower, upper, n, device=x.device)
        if bw is None:
            bwp = len(x_tmp)**(-1 / 5)
        else:
            bwp = len(x_tmp)**(-1 / bw)
            
        norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
        return torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor 
    
    @staticmethod
    def gauss_kde_diff(x, lower, upper, n, bw):
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        xy_diff = x_tmp-y_tmp
        grid = torch.linspace(lower, upper, n, device=x.device)
        if bw is None:
            bwp = len(x_tmp)**(-1 / 5)
        else:
            bwp = len(x_tmp)**(-1 / bw)
            
        norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
        return torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor  
    
    @staticmethod
    def gauss_kde2(x, lower, upper, n, bw):
        pdf = torch.zeros((n, 2), device=x.device)
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        xy_sum = x_tmp+y_tmp
        xy_diff = x_tmp-y_tmp
        grid = torch.linspace(lower, upper, n, device=x.device)
        if bw is None:
            bwp = len(x_tmp)**(-1 / 5)
        else:
            bwp = len(x_tmp)**(-1 / bw)
            
        norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
        pdf[...,0]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,1]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        return pdf 
    
    @staticmethod
    def gauss_kde4(x, lower, upper, n, bw):
        pdf = torch.zeros((n, 4), device=x.device)
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        xy_sum = x_tmp+y_tmp
        xy_diff = x_tmp-y_tmp
        xy_sum1 = x_tmp+y_tmp*2
        xy_diff1 = x_tmp-y_tmp*2
        grid = torch.linspace(lower, upper, n, device=x.device)
        if bw is None:
            bwp = len(x_tmp)**(-1 / 5)
        else:
            bwp = len(x_tmp)**(-1 / bw)
            
        norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
        pdf[...,0]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,1]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,2]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum1[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,3]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff1[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        return pdf    

    @staticmethod
    def gauss_kde6(x, lower, upper, n, bw):
        pdf = torch.zeros((n, 6), device=x.device)
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        xy_sum = x_tmp+y_tmp
        xy_diff = x_tmp-y_tmp
        xy_sum1 = x_tmp+y_tmp*2
        xy_diff1 = x_tmp-y_tmp*2 
        xy_sum2 = 2*x_tmp+y_tmp
        xy_diff2 = 2*x_tmp-y_tmp
        grid = torch.linspace(lower, upper, n, device=x.device)
        if bw is None:
            bwp = len(x_tmp)**(-1 / 5)
        else:
            bwp = len(x_tmp)**(-1 / bw)
            
        norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
        pdf[...,0]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,1]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,2]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum1[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,3]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff1[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,4]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_sum2[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        pdf[...,5]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (xy_diff2[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor
        return pdf
    
    @staticmethod
    def gauss_kde(x, lower, upper, n, bw, chunk=None, bandwidth=None):
        '''Marginal densities of the d components.

        `chunk` accumulates the kernel sum over blocks of at most that many
        samples; the result is unchanged, it only bounds the intermediate
        (n_samples, n) array.  See `gauss_kde_proj`.
        '''
        columns = []
        for i in range(x.size()[2]):
            x_tmp = torch.ravel(x[...,i])
            grid = torch.linspace(lower, upper, n, device=x.device)
            m = len(x_tmp)
            bwp = StatLoss._bandwidth(m, bw, bandwidth)
            norm_factor = (2 * np.pi)**0.5 * m * bwp
            block = m if chunk is None else int(chunk)
            col = None
            for s in range(0, m, block):
                part = torch.sum(
                    torch.exp(
                        -0.5 * torch.square(
                            (x_tmp[s:s + block, None] - grid[None, :]) / bwp
                        )
                    ),
                    axis=0
                )
                col = part if col is None else col + part
            columns.append(col / norm_factor)
        return torch.stack(columns, dim=-1)
    

#     def gauss_kde(x, lower, upper, n, bw=None):
#         x = torch.ravel(x)
#         grid = torch.linspace(lower, upper, n, device=x.device)
#         if bw is None:
#             bw = len(x)**(-1 / 5)
#         norm_factor = (2 * np.pi)**0.5 * len(x) * bw
#         return torch.sum(
#             torch.exp(
#                 -0.5 * torch.square(
#                     (x[:, None] - grid[None, :]) / bw
#                 )
#             ),
#             axis=0
#         ) / norm_factor

    @abstractmethod
    def forward(self, input):
        '''Evaluate the input stochastic processes against the target statistic
        function.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''



class ACFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.acf(data, lags=lags), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        _input = self.acf(input, lags=len(self._target))
        return self._loss(_input, self._target)


class BruteForceACFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.acf(data, lags=lags, method='bruteforce'), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        _input = self.acf(input, lags=len(self._target), method='bruteforce')
        return self._loss(_input, self._target)


class RandomBruteForceACFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.acf(data, lags=lags, method='bruteforce'), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        lags = np.random.choice(len(self._target), self.sample_lags, False)
        _input = self.acf(input, lags=lags, method='bruteforce')
        return self._loss(_input, self._target[lags])

class CCFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.ccf(data, lags=lags), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        _input = self.ccf(input, lags=len(self._target))
        return self._loss(_input, self._target)


class BruteForceCCFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.ccf(data, lags=lags, method='bruteforce'), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        _input = self.ccf(input, lags=len(self._target), method='bruteforce')
        return self._loss(_input, self._target)


class RandomBruteForceCCFLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lags, **options):
        '''Create target ACF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(cls.ccf(data, lags=lags, method='bruteforce'), **options)

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, n_variables)
            The input trajectory as generated by an NN.
        '''
        lags = np.random.choice(len(self._target), self.sample_lags, False)
        _input = self.ccf(input, lags=lags, method='bruteforce')
        return self._loss(_input, self._target[lags])
    
class DensityLoss(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, target_chunk=200000,
                            bandwidth=None, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        # One bandwidth for both sides.  When it is not given explicitly it is
        # taken from the TARGET ensemble here and reused in forward(), instead
        # of each side reading it off its own sample count.
        if bandwidth is None:
            bandwidth = cls._bandwidth(
                data.reshape(-1, data.shape[-1]).shape[0], bw, None
            )
        with torch.no_grad():
            target = cls.gauss_kde(
                data, lower=lower, upper=upper, n=n, bw=bw, chunk=target_chunk,
                bandwidth=bandwidth
            )
        return cls(
            target,
            **options, lower=lower, upper=upper, n=n, bw=bw, bandwidth=bandwidth
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw,
            bandwidth=getattr(self, 'bandwidth', None)
        )
        return self._loss(_input, self._target)

class DensityLossSum(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde_sum(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde_sum(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)
    
class DensityLossDiff(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde_diff(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde_diff(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)
    
class DensityLoss2(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde2(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde2(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)

class DensityLoss4(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde4(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde4(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)

class DensityLoss6(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde6(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde6(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)
    
class ProjectedDensityLoss(StatLoss):
    '''Densities of a collection of 1D projections w . x, for any number of
    components.  Replaces the 2D-only `DensityLoss2`/`4`/`6`.
    '''

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, directions=None,
                            n_frames=3, target_chunk=200000, bandwidth=None,
                            standardize=True, scale=None, **options):
        '''Create the target projected PDFs from observed trajectories.

        Parameters
        ----------
        data: tensor of shape (trajectory_length, n_batch, d)
            The empirically observed trajectories.
        directions: tensor-like of shape (K, d), optional
            Projection directions.  Defaults to `n_frames` orthogonal frames
            in R^d, i.e. n_frames * d directions.
        target_chunk: int
            Sample-block size used for the target only.  The target ensemble
            is usually much larger than a training batch, and it needs no
            gradient, so it is built block by block under no_grad.
        '''
        if directions is None:
            directions = cls.projection_directions(
                data.shape[-1], n_frames=n_frames
            )
        with torch.no_grad():
            # Fixed per-projection scale, so every projected density lives
            # on the same grid at the same resolution.  Pass `scale` to reuse
            # one set of constants across several ensembles (training target,
            # held-out reference, a sweep over ensemble sizes); otherwise it
            # is taken from `data`.
            if scale is None and standardize:
                scale = cls.projection_scales(data, directions)
            elif not standardize:
                scale = None
            # One bandwidth for both sides -- see DensityLoss.
            if bandwidth is None:
                bandwidth = cls._bandwidth(
                    data.reshape(-1, data.shape[-1]).shape[0], bw, None
                )
            target = cls.gauss_kde_proj(
                data, directions, lower=lower, upper=upper, n=n, bw=bw,
                chunk=target_chunk, bandwidth=bandwidth, scale=scale
            )
        return cls(
            target,
            **options, lower=lower, upper=upper, n=n, bw=bw,
            directions=directions, scale=scale, bandwidth=bandwidth
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of shape (trajectory_length, n_batch, d)
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde_proj(
            input, self.directions, lower=self.lower, upper=self.upper,
            n=self.n, bw=self.bw, bandwidth=getattr(self, 'bandwidth', None),
            scale=getattr(self, 'scale', None)
        )
        return self._loss(_input, self._target)


class DensityLoss2D(StatLoss):

    @classmethod
    def from_empirical_data(cls, data, lower, upper, n, bw, dims=(0, 1),
                            bandwidth=None, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        if bandwidth is None:
            m = data.reshape(-1, data.shape[-1]).shape[0]
            bandwidth = m**(-1 / 6) if bw is None else m**(-1 / bw)
        with torch.no_grad():
            target = cls.gauss_kde2D(
                data, lower=lower, upper=upper, n=n, bw=bw, dims=dims,
                bandwidth=bandwidth
            )
        return cls(
            target,
            **options, lower=lower, upper=upper, n=n, bw=bw, dims=dims,
            bandwidth=bandwidth
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde2D(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw,
            dims=self.dims, bandwidth=getattr(self, 'bandwidth', None)
        )
        return self._loss(_input, self._target)
    
# def make_loss(stat, data, loss_type=['mse_loss', 'l1_loss'], **kwargs):
def make_loss(stat, data, loss_type=['mse_loss'], **kwargs):
    '''
    Create a loss function.

    Parameters
    ----------
    stat: 'pdf' or 'pdfproj' or 'acf[fft]' or 'acf[bruteforce]' or
        'acf[randombrute]' or 'ccf[...]'
        Statistics to compute.  'pdf' gives the d marginal densities;
        'pdfproj' gives the densities of n_frames * d projections and works
        for any d (the 2D-only 'pdf2'/'pdf4'/'pdf6'/'pdf2D' are kept for
        backward compatibility).
    data: tensor
        Target statistics function or sample trajectories.
    loss_type: list
        Lower-levle loss functions to use.
    kwargs:
        additional arguments to pass to the loss function

    Returns
    -------
    loss: callable
        A loss function
    '''
    if stat == 'pdf':
        loss_cls = DensityLoss
    elif stat == 'pdfsum':
        loss_cls = DensityLossSum
    elif stat == 'pdfdiff':
        loss_cls = DensityLossDiff
    elif stat == 'pdf2':
        loss_cls = DensityLoss2
    elif stat == 'pdf4':
        loss_cls = DensityLoss4
    elif stat == 'pdf6':
        loss_cls = DensityLoss6
    elif stat == 'pdfproj':
        loss_cls = ProjectedDensityLoss
    elif stat == 'pdf2D':
        loss_cls = DensityLoss2D
    elif stat == 'acf[fft]':
        loss_cls = ACFLoss
    elif stat == 'acf[bruteforce]':
        loss_cls = BruteForceACFLoss
    elif stat == 'acf[randombrute]':
        loss_cls = RandomBruteForceACFLoss
    elif stat == 'ccf[fft]':
        loss_cls = CCFLoss
    elif stat == 'ccf[bruteforce]':
        loss_cls = BruteForceCCFLoss
    elif stat == 'ccf[randombrute]':
        loss_cls = RandomBruteForceCCFLoss
    else:
        raise RuntimeError(f'Unknown stat {stat}.')

    def lower_level_loss(a, b):
        return torch.sum([getattr(F, ls)(a, b) for ls in loss_type])

    if len(data.shape) == 1:
        return loss_cls(
            data,
            loss=lower_level_loss,
            **kwargs
        )
    elif len(data.shape) == 3:
        return loss_cls.from_empirical_data(
            data,
            loss=lower_level_loss,
            **kwargs
        )
    else:
        raise RuntimeError('Unknown truth data format.')
