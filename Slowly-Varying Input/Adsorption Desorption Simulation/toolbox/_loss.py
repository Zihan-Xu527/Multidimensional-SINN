#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import scipy.stats as st
from scipy.stats import skew, kurtosis


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

    def __init__(self, target, pointwise_loss='mse_loss', device='cpu', normalize=False,
                 **kwargs):
        '''Direct specification of the target statistic function in discretized
        form.

        Parameters
        ----------
        array: list-like The target statistic function.
        '''
        super().__init__()
        self._target = target.clone().detach()
        
        # Add normalization flag
        self.normalize = normalize


        if callable(pointwise_loss):
            self._loss = pointwise_loss
        else:
            try:
                self._loss = getattr(F, pointwise_loss)
            except AttributeError as e:
                raise RuntimeError(f'Unrecognized pointwise loss. Error:\n{e}')
        self.__dict__.update(**kwargs)

    @staticmethod
    def normalize_data(data):
        '''Normalize the data by subtracting the mean and dividing by the std'''
        mean = data.mean()
        std = data.std()
        return (data - mean) / std if std != 0 else data


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
        corr = torch.zeros((x.shape[0], x.shape[2]*(x.shape[2]-1)), device=x.device)
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
                    
                        corr[...,n] = ccf[:lags, ...] / torch.sqrt(acf_f[0, ...])/ torch.sqrt(acf_g[0,...])
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
                        corr[...,n] = corr_tmp  
                        n += 1
                    else:
                        raise NotImplementedError(f'Unknown method {method}.')
        return corr




    @staticmethod
    def gauss_kde2D(x, lower, upper, n, bw):
        pdf = torch.zeros((n, n), device=x.device)
        x_tmp = torch.ravel(x[...,0])
        y_tmp = torch.ravel(x[...,1])
        x_grid = torch.linspace(lower, upper, n, device=x.device)
        y_grid = torch.linspace(lower, upper, n, device=x.device)
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))
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
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))
            
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
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))
            
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
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))
            
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
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))
            
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
    def gauss_kde6(x, lower, upper, n, bw=None, normalize=False):
        '''Compute 6 KDEs of linear combinations of x[...,0] and x[...,1]'''
        pdf = torch.zeros((n, 10), device=x.device)

        x0 = torch.ravel(x[..., 0])
        x1 = torch.ravel(x[..., 1])

        if normalize:
            x0 = StatLoss.normalize_data(x0)
            x1 = StatLoss.normalize_data(x1)

        combos = [
            x0 + x1,
            x0 - x1,
            x0 + 2 * x1,
            x0 - 2 * x1,
            2 * x0 + x1,
            2 * x0 - x1,
            x0 + 4 * x1,
            x0 - 4 * x1,
            4 * x0 + x1,
            4 * x0 - x1
        ]  
        grid = torch.linspace(lower, upper, n, device=x.device)
        d = x.shape[-1]  # dimension of the data
        if bw is None:
            bwp = len(x0) ** (-1 / (d + 4))
        else:
            bwp = bw * len(x0) ** (-1 / (d + 4))

        norm_factor = (2 * np.pi) ** 0.5 * len(x0) * bwp

        for i, combo in enumerate(combos):
            diff = (combo[:, None] - grid[None, :]) / bwp
            pdf[:, i] = torch.sum(torch.exp(-0.5 * diff ** 2), axis=0) / norm_factor

        return pdf


    
    @staticmethod
    def gauss_kde(x, lower, upper, n, bw, normalize=False):
        '''Kernel Density Estimation with optional normalization.'''
        pdf = torch.zeros((n, *x.shape[2:]), device=x.device)
        d = x.shape[-1]  # dimension of the data
    
        # Normalize data if needed
        if normalize:
            x = torch.stack([StatLoss.normalize_data(x[..., i]) for i in range(x.size()[2])], dim=-1)

        # Calculate KDE for each feature in the data
        for i in range(x.size()[2]):
            x_tmp = torch.ravel(x[..., i])
            grid = torch.linspace(lower, upper, n, device=x.device)
            if bw is None:
                bwp = len(x_tmp)**(-1 / (d + 4))
            else:
                bwp = bw * len(x_tmp)**(-1 / (d + 4))
            norm_factor = len(x_tmp) * bwp *(2 * np.pi)**0.5 
            pdf[..., i] = torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (x_tmp[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor

        return pdf



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
    def from_empirical_data(cls, data, lower, upper, n, bw, normalize=False, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde(data, lower=lower, upper=upper, n=n, bw=bw, normalize=normalize),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw, normalize=self.normalize
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
    def from_empirical_data(cls, data, lower, upper, n, bw, normalize=False, **options):
        '''Create target PDF from a number of empirically observed trajectories.

        Parameters
        ----------
        input: tensor of any shape
            The empirically observed trajectories.
        options: keyword argument list
            To be forwarded to `__init__`.
        '''
        return cls(
            cls.gauss_kde6(data, lower=lower, upper=upper, n=n, bw=bw, normalize=normalize),
            **options, lower=lower, upper=upper, n=n, bw=bw, normalize=normalize
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde6(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw, normalize=self.normalize
        )
        return self._loss(_input, self._target)
    
class DensityLoss2D(StatLoss):

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
            cls.gauss_kde2D(data, lower=lower, upper=upper, n=n, bw=bw),
            **options, lower=lower, upper=upper, n=n, bw=bw
        )

    def forward(self, input):
        '''
        Parameters
        ----------
        input: tensor of any shape
            The input trajectory as generated by an NN.
        '''
        _input = self.gauss_kde2D(
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
        )
        return self._loss(_input, self._target)
    
    
class ConditionalLoss2D(StatLoss):
    @classmethod
    def from_empirical_data(cls, theta, data, **options):
        theta = theta.flatten()
        data = data.view(-1, 2)

        mu = theta.mean()
        sigma = theta.std()
        bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

        # Set KDE parameters
        n = options.pop('n', 100)
        bw = options.pop('bw', 0.01)
        normalize = options.pop('normalize', False)
        lower = 0.22
        upper = 0.28

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        kdes = []
        for mask in masks:
            if mask.any():
                group = data[mask]  # shape (N, 2)
                kde_vals = StatLoss.gauss_kde(group.unsqueeze(0), lower, upper, n=n, bw=bw, normalize=normalize)
#                 range_ = group.max() - group.min()
#                 margin = 0.1 * range_
#                 lower = group.min() - margin
#                 upper = group.max() + margin
#                 kde_vals = cls.gauss_kde(group.unsqueeze(0), lower.item(), upper.item(), n=n, bw=bw, normalize=normalize)  # (n, 2)
            else:
                kde_vals = torch.zeros(n, 2, device=data.device)
            kdes.append(kde_vals.T)  # (2, n)

        target_kdes = torch.stack(kdes)           # (4 groups, 2 dims, n grid)
        target_kdes = target_kdes.permute(2, 0, 1)  # (n grid, 4 groups, 2 dims)

        instance = cls(target_kdes, **options)
        instance.bounds = bounds
        instance.lower = lower
        instance.upper = upper
        instance.n = n
        instance.bw = bw
        instance.normalize = normalize
        return instance
    def forward(self, input):
        theta, data = input
        theta = theta.flatten()
        data = data.view(-1, 2)

        masks = [
            theta <= self.bounds[0],
            (theta > self.bounds[0]) & (theta <= self.bounds[1]),
            (theta > self.bounds[1]) & (theta <= self.bounds[2]),
            theta > self.bounds[2],
        ]

        kdes = []
        for mask in masks:
            if mask.any():
                group = data[mask]  # shape (N_group, 2)
                kde_vals = StatLoss.gauss_kde(group.unsqueeze(0), self.lower, self.upper, n=self.n, bw=self.bw, normalize=self.normalize)
#                 range_ = group.max() - group.min()
#                 margin = 0.1 * range_
#                 lower = group.min() - margin
#                 upper = group.max() + margin
#                 kde_vals = self.gauss_kde(group.unsqueeze(0), lower.item(), upper.item(), n=self.n, bw=self.bw, normalize=self.normalize)  # (n, 2)
            else:
                kde_vals = torch.zeros(self.n, 2, device=data.device)
            kdes.append(kde_vals.T)  # shape (2, n)

        current_kdes = torch.stack(kdes)           # shape (4 groups, 2 dims, n grid)
        current_kdes = current_kdes.permute(2, 0, 1)  # → shape (n grid, 4 groups, 2 dims)
        return self._loss(current_kdes, self._target)



    @staticmethod
    def compute_skewness(samples, dim=0, keepdim=False):
        mean = samples.mean(dim=dim, keepdim=True)
        std = samples.std(dim=dim, keepdim=True)
        z = (samples - mean) / (std + 1e-8)
        skewness = (z ** 3).mean(dim=dim, keepdim=keepdim)
        return skewness

    @staticmethod
    def compute_kurtosis(samples, dim=0, keepdim=False, fisher=True):
        mean = samples.mean(dim=dim, keepdim=True)
        std = samples.std(dim=dim, keepdim=True)
        z = (samples - mean) / (std + 1e-8)
        kurt = (z ** 4).mean(dim=dim, keepdim=keepdim)
        return kurt - 3 if fisher else kurt
# class ConditionalMeanLoss2D(StatLoss):
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         means = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0) * 10
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#             means.append(mean)

#         target_means = torch.stack(means)  # (4 groups, 2 dims)
#         instance = cls(target_means, **options)
#         instance.bounds = bounds
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         means = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0) * 10
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#             means.append(mean)

#         current_means = torch.stack(means)
#         return self._loss(current_means, self._target)
class ConditionalMeanLoss2D(StatLoss):
    @classmethod
    def from_empirical_data(cls, theta, data, bounds=None, scale=10.0, **options):
        theta = theta.flatten()
        data = data.view(-1, 2)

        if bounds is None:
            mu = theta.mean()
            sigma = theta.std()
            bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        means = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0) * scale
            else:
                mean = torch.zeros(2, device=theta.device)
            means.append(mean)

        target_means = torch.stack(means)

        instance = cls(target_means, **options)
        instance.bounds = bounds
        instance.scale = scale   # ✅ store scale
        return instance

    def forward(self, input):
        theta, data = input
        theta = theta.flatten()
        data = data.view(-1, 2)
        bounds = self.bounds
        scale = self.scale  # ✅ use stored scale

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        means = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0) * scale
            else:
                mean = torch.zeros(2, device=theta.device)
            means.append(mean)

        current_means = torch.stack(means)
        return self._loss(current_means, self._target)

# class ConditionalStdLoss2D(StatLoss):
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         vars_ = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 var = group.std(dim=0) * 10
# #                 var = torch.log(group.var(dim=0))
#             else:
#                 var = torch.zeros(2, device=theta.device)
#             vars_.append(var)

#         target_vars = torch.stack(vars_)  # (4 groups, 2 dims)
#         instance = cls(target_vars, **options)
#         instance.bounds = bounds
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         vars_ = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
# #                 var = torch.log(group.var(dim=0))
#                 var = group.std(dim=0) * 10
#             else:
#                 var = torch.zeros(2, device=theta.device)
#             vars_.append(var)

#         current_vars = torch.stack(vars_)
#         return self._loss(current_vars, self._target)

class ConditionalStdLoss2D(StatLoss):
    @classmethod
    def from_empirical_data(cls, theta, data, bounds=None, scale=10.0, **options):
        theta = theta.flatten()
        data = data.view(-1, 2)

        if bounds is None:
            mu = theta.mean()
            sigma = theta.std()
            bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        stds = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                std = group.std(dim=0) * scale
            else:
                std = torch.zeros(2, device=theta.device)
            stds.append(std)

        target_stds = torch.stack(stds)

        instance = cls(target_stds, **options)
        instance.bounds = bounds
        instance.scale = scale
        return instance

    def forward(self, input):
        theta, data = input
        theta = theta.flatten()
        data = data.view(-1, 2)

        bounds = self.bounds
        scale = self.scale

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        stds = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                std = group.std(dim=0) * scale
            else:
                std = torch.zeros(2, device=theta.device)
            stds.append(std)

        current_stds = torch.stack(stds)
        return self._loss(current_stds, self._target)

    
# class ConditionalSkewLoss2D(StatLoss):
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]
#         def signed_cuberoot(x: torch.Tensor):
#             return torch.sign(x) * torch.pow(torch.abs(x), 1.0/3)

#         skews = []
        
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0, keepdim=True)
#                 z = (group - mean)
#                 skew = signed_cuberoot((z ** 3).mean(dim=0)) * 10
#             else:
#                 skew = torch.zeros(2, device=theta.device)
#             skews.append(skew)

#         target_skews = torch.stack(skews)   # (4 groups, 2 dims)
#         instance = cls(target_skews, **options)
#         instance.bounds = bounds
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]
#         def signed_cuberoot(x: torch.Tensor):
#             return torch.sign(x) * torch.pow(torch.abs(x), 1.0/3)
#         skews = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0, keepdim=True)
#                 z = (group - mean)
#                 skew = signed_cuberoot((z ** 3).mean(dim=0)) * 10
#             else:
#                 skew = torch.zeros(2, device=theta.device)
#             skews.append(skew)

#         current_skews = torch.stack(skews)
#         return self._loss(current_skews, self._target)
class ConditionalSkewLoss2D(StatLoss):
    @staticmethod
    def signed_cuberoot(x: torch.Tensor):
        return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3)
    
    @classmethod
    def from_empirical_data(cls, theta, data, bounds=None, scale=10.0, **options):
        theta = theta.flatten()
        data = data.view(-1, 2)

        if bounds is None:
            mu = theta.mean()
            sigma = theta.std()
            bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        skews = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0, keepdim=True)
                z = group - mean
                skew = cls.signed_cuberoot((z ** 3).mean(dim=0)) * scale
            else:
                skew = torch.zeros(2, device=theta.device)
            skews.append(skew)

        target_skews = torch.stack(skews)

        instance = cls(target_skews, **options)
        instance.bounds = bounds
        instance.scale = scale
        return instance

    def forward(self, input):
        theta, data = input
        theta = theta.flatten()
        data = data.view(-1, 2)

        bounds = self.bounds
        scale = self.scale

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        skews = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0, keepdim=True)
                z = group - mean
                skew = self.signed_cuberoot((z ** 3).mean(dim=0)) * scale
            else:
                skew = torch.zeros(2, device=theta.device)
            skews.append(skew)

        current_skews = torch.stack(skews)
        return self._loss(current_skews, self._target)
    
# class ConditionalKurtLoss2D(StatLoss):
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         kurts = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0, keepdim=True)
#                 std = group.std(dim=0, keepdim=True) + 1e-8
#                 z = (group - mean)
#                 kurt = torch.pow((z ** 4).mean(dim=0),1.0/4) * 10
# #                 if fisher:
# #                     kurt = kurt - 3
#             else:
#                 kurt = torch.zeros(2, device=theta.device)
#             kurts.append(kurt)

#         target_kurts = torch.stack(kurts)
#         instance = cls(target_kurts, **options)
#         instance.bounds = bounds
# #         instance.fisher = fisher
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds
# #         fisher = self.fisher

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         kurts = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0, keepdim=True)
#                 z = (group - mean)
#                 kurt = torch.pow((z ** 4).mean(dim=0),1.0/4) * 10
# #                 if fisher:
# #                     kurt = kurt - 3
#             else:
#                 kurt = torch.zeros(2, device=theta.device)
#             kurts.append(kurt)

#         current_kurts = torch.stack(kurts)
#         return self._loss(current_kurts, self._target)
class ConditionalKurtLoss2D(StatLoss):
    @classmethod
    def from_empirical_data(cls, theta, data, bounds=None, scale=10.0, **options):
        theta = theta.flatten()
        data = data.view(-1, 2)

        if bounds is None:
            mu = theta.mean()
            sigma = theta.std()
            bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        kurts = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0, keepdim=True)
                z = group - mean
                kurt = torch.pow((z ** 4).mean(dim=0), 1.0 / 4) * scale
            else:
                kurt = torch.zeros(2, device=theta.device)
            kurts.append(kurt)

        target_kurts = torch.stack(kurts)

        instance = cls(target_kurts, **options)
        instance.bounds = bounds
        instance.scale = scale
        return instance

    def forward(self, input):
        theta, data = input
        theta = theta.flatten()
        data = data.view(-1, 2)

        bounds = self.bounds
        scale = self.scale

        masks = [
            theta <= bounds[0],
            (theta > bounds[0]) & (theta <= bounds[1]),
            (theta > bounds[1]) & (theta <= bounds[2]),
            theta > bounds[2],
        ]

        kurts = []
        for mask in masks:
            if mask.any():
                group = data[mask]
                mean = group.mean(dim=0, keepdim=True)
                z = group - mean
                kurt = torch.pow((z ** 4).mean(dim=0), 1.0 / 4) * scale
            else:
                kurt = torch.zeros(2, device=theta.device)
            kurts.append(kurt)

        current_kurts = torch.stack(kurts)
        return self._loss(current_kurts, self._target)
# class ConditionalMeanLoss2D(StatLoss):
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         stats = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0)*10
#                 var = group.var(dim=0)
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#                 var = torch.zeros(2, device=theta.device)
#             stats.append(torch.stack([mean, var]))

#         target_stats = torch.stack(stats)  # (4 groups, 2 stats, 2 dims)
#         target_stats = target_stats.permute(1, 0, 2)  # → (2 stats, 4 groups, 2 dims)

#         instance = cls(target_stats, **options)
#         instance.bounds = bounds
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         stats = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0)*10
#                 var = group.var(dim=0)
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#                 var = torch.zeros(2, device=theta.device)
#             stats.append(torch.stack([mean, var]))

#         current_stats = torch.stack(stats)  # (4 groups, 2 stats, 2 dims)
#         current_stats = current_stats.permute(1, 0, 2)  # → (2 stats, 4 groups, 2 dims)
#         return self._loss(current_stats, self._target)

# mean, var, skewness, kurt
#     @classmethod
#     def from_empirical_data(cls, theta, data, bounds=None, **options):
#         theta = theta.flatten()
#         data = data.view(-1, 2)

#         if bounds is None:
#             mu = theta.mean()
#             sigma = theta.std()
#             bounds = [mu - 2/3 * sigma, mu, mu + 2/3 * sigma]

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         stats = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0)*10
#                 var = torch.log(group.var(dim=0))
#                 skewness = cls.compute_skewness(group, dim=0).squeeze(0)
#                 kurt = cls.compute_kurtosis(group, dim=0).squeeze(0)
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#                 var = torch.zeros(2, device=theta.device)
#                 skewness = torch.zeros(2, device=theta.device)
#                 kurt = torch.zeros(2, device=theta.device)
#             stats.append(torch.stack([mean, var, skewness, kurt]))

#         target_stats = torch.stack(stats)  # (4 groups, 4 stats, 2 dims)
#         target_stats = target_stats.permute(1, 0, 2) # → (4 stats, 4 groups, 2 dims)


#         instance = cls(target_stats, **options)
#         instance.bounds = bounds
#         return instance

#     def forward(self, input):
#         theta, data = input
#         theta = theta.flatten()
#         data = data.view(-1, 2)
#         bounds = self.bounds

#         masks = [
#             theta <= bounds[0],
#             (theta > bounds[0]) & (theta <= bounds[1]),
#             (theta > bounds[1]) & (theta <= bounds[2]),
#             theta > bounds[2],
#         ]

#         stats = []
#         for mask in masks:
#             if mask.any():
#                 group = data[mask]
#                 mean = group.mean(dim=0)*10
#                 var = torch.log(group.var(dim=0))
#                 skewness = self.compute_skewness(group, dim=0).squeeze(0)
#                 kurt = self.compute_kurtosis(group, dim=0).squeeze(0)
#             else:
#                 mean = torch.zeros(2, device=theta.device)
#                 var = torch.zeros(2, device=theta.device)
#                 skewness = torch.zeros(2, device=theta.device)
#                 kurt = torch.zeros(2, device=theta.device)
#             stats.append(torch.stack([mean, var, skewness, kurt]))

#         current_stats = torch.stack(stats)  # (4 groups, 4 stats, 2 dims)
#         current_stats = current_stats.permute(1, 0, 2) # → (4 stats, 4 groups, 2 dims)
#         return self._loss(current_stats, self._target)



    
# def make_loss(stat, data, loss_type=['mse_loss', 'l1_loss'], **kwargs):
def make_loss(stat, data, loss_type=['mse_loss'], **kwargs):
    '''
    Create a loss function.

    Parameters
    ----------
    stat: 'pdf' or 'acf[fft]' or 'acf[bruteforce]' or 'acf[randombrute]'
        Statistics to compute
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
    elif stat == 'conditional_pdf':
        loss_cls = ConditionalLoss2D
    elif stat == 'conditional_mean':
        loss_cls = ConditionalMeanLoss2D
    elif stat == 'conditional_std':
        loss_cls = ConditionalStdLoss2D
    elif stat == 'conditional_skew':
        loss_cls = ConditionalSkewLoss2D
    elif stat == 'conditional_kurt':
        loss_cls = ConditionalKurtLoss2D
    else:
        raise RuntimeError(f'Unknown stat {stat}.')

    def lower_level_loss(a, b):
        return torch.sum([getattr(F, ls)(a, b) for ls in loss_type])
    
    #  Special handling for conditional losses
    if isinstance(data, (tuple, list)) and len(data) == 2:
        cond_data, target_data = data
        return loss_cls.from_empirical_data(
            cond_data,
            target_data,
            loss=lower_level_loss,
            **kwargs
        )

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
