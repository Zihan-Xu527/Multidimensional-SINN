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


#         x_tmp = torch.ravel(x[...,0])
#         y_tmp = torch.ravel(x[...,1])
#         xs=x_tmp.cpu()
#         ys=y_tmp.cpu()

#         # Peform the kernel density estimate
#         xx, yy = np.mgrid[lower:upper:100j, lower:upper:100j]
#         positions = np.vstack([xx.ravel(), yy.ravel()])
#         values = torch.vstack([xs, ys])
#         kernel = st.gaussian_kde(values)
#         f = np.reshape(kernel(positions).T, xx.shape)
#         return torch.from_numpy(f).to(x.device)

#     @staticmethod
#     def gauss_kde2D(x, lower, upper, n, bw):
#         pdf = torch.zeros((n, n), device=x.device)
#         x_tmp = torch.ravel(x[...,0])
#         y_tmp = torch.ravel(x[...,1])
#         X=torch.cat((x_tmp,y_tmp),0)
#         X=torch.reshape(X,(2,len(x_tmp)))
#         x_grid = torch.linspace(lower, upper, n, device=x.device)
#         y_grid = torch.linspace(lower, upper, n, device=x.device)
#         if bw is None:
#             bwp = len(x_tmp)**(-1 / 5)
#         else:
#             bwp = len(x_tmp)**(-1 / bw)
#         norm_factor = (2 * np.pi) * len(x_tmp) * bwp
#         for i in range(n):
#             for j in range(n):
#                 grid_x = torch.tensor([x_grid[i]], device=x.device)
#                 grid_y = torch.tensor([y_grid[j]], device=x.device)
#                 pdf[i,j] = torch.sum(torch.exp(-0.5*torch.square(
#                     (x_tmp[:,None]-torch.repeat_interleave(grid_x, len(x_tmp)))/bwp))*torch.exp(-0.5*torch.square(
#                     (y_tmp[:,None]-torch.repeat_interleave(grid_y, len(y_tmp)))/bwp)),axis=0 )/ norm_factor
#         return pdf       
    @staticmethod
    def gauss_kde2D(x, lower, upper, n, bw):
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
    def gauss_kde(x, lower, upper, n, bw):
        pdf = torch.zeros((n, *x.shape[2:]), device=x.device)
        for i in range(x.size()[2]):
            x_tmp = torch.ravel(x[...,i])
            grid = torch.linspace(lower, upper, n, device=x.device)
            if bw is None:
                bwp = len(x_tmp)**(-1 / 5)
            else:
                bwp = len(x_tmp)**(-1 / bw)
            norm_factor = (2 * np.pi)**0.5 * len(x_tmp) * bwp
            pdf[...,i]=torch.sum(
                torch.exp(
                    -0.5 * torch.square(
                        (x_tmp[:, None] - grid[None, :]) / bwp
                    )
                ),
                axis=0
            ) / norm_factor 
        return pdf
    

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
            cls.gauss_kde(data, lower=lower, upper=upper, n=n, bw=bw),
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
            input, lower=self.lower, upper=self.upper, n=self.n, bw=self.bw
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
    
def make_loss(stat, data, loss_type=['mse_loss', 'l1_loss'], **kwargs):
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
