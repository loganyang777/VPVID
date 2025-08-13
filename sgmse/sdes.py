"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import math
import os
import warnings

import numpy as np
import tqdm
from sgmse.util.tensors import batch_broadcast
import scipy.special as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sgmse.util.registry import Registry

from sgmse.mlp import MLP


SDERegistry = Registry("SDE")

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N, resolution=None):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N
        self.resolution = resolution

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, y, *args):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """

        dt = 1 / self.N if self.resolution is None else self.resolution
        drift, diffusion = self.sde(x, t, y, *args) # 关键
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, y, *args):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, y)
                rev_F = f - G[:, None, None, None] ** 2 * score_model(x, t, y, *args) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_F, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("vpvid")
class VPVID(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=25,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--sigma_min", type=float, default=0.1, help="The minimum beta to use.")
        parser.add_argument("--sigma_max", type=float, default=2., help="The maximum beta to use.")
        parser.add_argument("--customized_var", action='store_true',  help="Customize your own exponential variance scheduler")
        parser.add_argument("--eta_coff", type=float, default=1.5, help="The minest eta .")

        return parser

    def __init__(self, sigma_min, sigma_max, N=100, 
                 t_eps=0.04, eta_coff=1.5, T=1.,
                 **ignored_kwargs):
        """
        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.t = T
        # 相当于 torch.exp(-0.5*integral_beta)
        self.log_mean_coeff = lambda t: -0.25 * (t ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * t * self.sigma_min 
        # 相当于 torch.exp(-0.5*beta)
        self.d_log_mean_coeff = lambda t: - 0.5 * t * (self.sigma_max - self.sigma_min) - 0.5 * self.sigma_min

        
        self.t_eps = t_eps
        self.eta_coff = eta_coff

    def copy(self):
        return VPVID(self.sigma_min, self.sigma_max, N=self.N, 
                        T=self.t, t_eps=self.t_eps, eta_coff=self.eta_coff)

    @property
    def T(self):
        return self.t

    def sde(self, x, t, y):
        """Compute the drift term of the SDE"""
        beta_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        alpha_t, _ = self.compute_alpha_t(t)

        drift = -0.5 *beta_t*x + self.eta_coff*(y*alpha_t - x)
        diffusion = torch.sqrt(beta_t + 2*self.eta_coff*(1 - alpha_t**2)).squeeze(-1).squeeze(-1).squeeze(-1)

        return drift, diffusion

    def _mean(self, x0, t, y):
        # st = alpha_t * vt + sigma_t * z
        t = expand_t_like_x(t, x0)
        alpha_t, _ = self.compute_alpha_t(t)
        vt, _ = self.compute_vt(t, x0, y)
        return alpha_t * vt

    def _std(self, t):
        std, _ = self.compute_sigma_t(t)
        return std

    def marginal_prob(self, x0, t, y):
        mean = self._mean(x0, t, y)
        std = self._std(t)
        return mean, std.squeeze(-1).squeeze(-1).squeeze(-1)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")

        T = torch.ones(y.shape[0], device=y.device) * self.T
        t = expand_t_like_x(T, y)
        z = torch.randn_like(y)
        st = self.compute_st(t, z, y, y)

        return st

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)


    def compute_drift(self, x, t, y):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        exp_itg_beta = torch.exp(self.log_mean_coeff(t))
        diffusion = beta_t + 2*self.eta_coff*(1 - exp_itg_beta**2)
        # diffusion = torch.sqrt(beta_t**2 + 2*self.eta_coff*(1 - exp_itg_beta**2)*beta_t)
        drift = -.5*beta_t*x + self.eta_coff*(y*exp_itg_beta - x)

        return drift, diffusion
    

    def compute_diffusion(self, x, t, y, form="SBDM", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = torch.ones(x.shape[0], device=x.device) * t
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t, y)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        temp = self.log_mean_coeff(t)
        alpha_t = torch.exp(temp)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        d_sigma_t = torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_lambda_t(self, t):
        """Compute the linear interpolation coefficient"""
        # st = (1 - torch.exp(-self.eta_coff*t)) * x + torch.exp(-self.eta_coff*t) * y
        lambda_t = torch.exp(-self.eta_coff*t)
        d_lambda_t = -self.eta_coff * torch.exp(-self.eta_coff*t)
        return lambda_t, d_lambda_t

    def compute_mu_t(self, t, z, x, y):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        vt, _= self.compute_vt(t, x, y)
        return alpha_t * vt + sigma_t * z
    
    def compute_st(self, t, z, x, y):
        """Sample xt from time-dependent density p_t; rng is required"""
        st = self.compute_mu_t(t, z, x, y)
        return st
    
    def compute_vt(self, t, x, y):
        lambda_t, d_lambda_t = self.compute_lambda_t(t)
        vt = lambda_t*x + (1-lambda_t)*y
        d_vt = d_lambda_t*x - d_lambda_t*y
        return vt, d_vt

    def compute_ut(self, t, z, x, y):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        vt , d_vt = self.compute_vt(t, x, y)

        return d_alpha_t * vt + alpha_t * d_vt + d_sigma_t * z
    
    def _forward(self, t, z, x, y):
        st = self.compute_st(t, z, x, y)
        ut = self.compute_ut(t, z, x, y)
        return t, st, ut


    """velocity and score transform"""
    def get_score_from_velocity(self, velocity, x, t, y):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        lambda_t, d_lambda_t = self.compute_lambda_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)

        A = d_alpha_t * lambda_t + alpha_t * d_lambda_t
        up = alpha_t*lambda_t*velocity - A*x + (alpha_t**2)*d_lambda_t*y
        down = A*(sigma_t**2) - alpha_t*lambda_t*sigma_t*d_sigma_t

        # alpha_ratio = d_alpha_t / alpha_t
        # up = -alpha_t*lambda_t*y - lambda_t*velocity + velocity
        # down = alpha_ratio*(lambda_t-1)*(sigma_t**2+1) + \
        #     d_lambda_t*sigma_t**2 - lambda_t*d_sigma_t*sigma_t + d_lambda_t + sigma_t*d_sigma_t
        # print(f't={t}, alpha_t={alpha_t}, d_alpha_t={d_alpha_t}, sigma_t={sigma_t}, d_sigma_t={d_sigma_t}')
        score = up / down
        return score
    
    def discretize(self, x, t, y, *args):
        raise NotImplementedError("discretize for IC_VPSDE not yet implemented!")

    def reverse(oself, velocity_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                # 首先调用父类的__init__方法，传入所有必需的参数
                super().__init__(
                    sigma_min=oself.sigma_min,
                    sigma_max=oself.sigma_max,
                    N=oself.N,
                    t_eps=oself.t_eps,
                    eta_coff=oself.eta_coff
                )
                
                # 设置RSDE特有的属性
                self.probability_flow = probability_flow
                self.get_diffusion = oself.compute_diffusion
                # 复制原始SDE的方法引用
                self.get_score_from_velocity = oself.get_score_from_velocity
                self.discretize_fn = oself.discretize

            @property
            def T(self):
                return T


            def sde(self, x, t, y, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                # _, diffusion = sde_fn(x, t, y)
                velocity = velocity_model(x, t, y)
                if self.probability_flow:
                    # ODE for probability flow sampling
                    return velocity, torch.zeros_like(t.view(-1, 1, 1, 1))
                G = self.compute_diffusion(x, t, y)
                score = self.get_score_from_velocity(velocity, x, t, y)
                total_drift = velocity - 0.5 * G.view(-1, 1, 1, 1)**2 * score
                total_diffusion = G
                return total_drift, total_diffusion

            def discretize(self, x, t, y, *args):
                raise NotImplementedError("discretize for IC_VPSDE not yet implemented!")
            
        return RSDE()

@SDERegistry.register("vevid")
class VEVID(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=25,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--sigma_min", type=float, default=0.1, help="The minimum beta to use.")
        parser.add_argument("--sigma_max", type=float, default=2., help="The maximum beta to use.")
        parser.add_argument("--customized_var", action='store_true',  help="Customize your own exponential variance scheduler")
        parser.add_argument("--eta_coff", type=float, default=1.5, help="The minest eta .")

        return parser

    def __init__(self, sigma_min, sigma_max, N=100, 
                 t_eps=0.04, eta_coff=1.5, T=1.,
                 **ignored_kwargs):
        """
        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.t = T
        # 相当于 torch.exp(-0.5*integral_beta)
        self.log_mean_coeff = lambda t: -0.25 * (t ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * t * self.sigma_min 
        # 相当于 torch.exp(-0.5*beta)
        self.d_log_mean_coeff = lambda t: - 0.5 * t * (self.sigma_max - self.sigma_min) - 0.5 * self.sigma_min

        
        self.t_eps = t_eps
        self.eta_coff = eta_coff

    def copy(self):
        return VEVID(self.sigma_min, self.sigma_max, N=self.N, 
                        T=self.t, t_eps=self.t_eps, eta_coff=self.eta_coff)

    @property
    def T(self):
        return self.t

    def sde(self, x, t, y):
        """Compute the drift term of the SDE"""
        beta_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        alpha_t, _ = self.compute_alpha_t(t)

        drift = -0.5 *beta_t*x + self.eta_coff*(y*alpha_t - x)
        diffusion = torch.sqrt(beta_t + 2*self.eta_coff*(1 - alpha_t**2)).squeeze(-1).squeeze(-1).squeeze(-1)

        return drift, diffusion

    def _mean(self, x0, t, y):
        # st = alpha_t * vt + sigma_t * z
        t = expand_t_like_x(t, x0)
        alpha_t, _ = self.compute_alpha_t(t)
        vt, _ = self.compute_vt(t, x0, y)
        return alpha_t * vt

    def _std(self, t):
        std, _ = self.compute_sigma_t(t)
        return std

    def marginal_prob(self, x0, t, y):
        mean = self._mean(x0, t, y)
        std = self._std(t)
        return mean, std.squeeze(-1).squeeze(-1).squeeze(-1)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")

        T = torch.ones(y.shape[0], device=y.device) * self.T
        t = expand_t_like_x(T, y)
        z = torch.randn_like(y)
        st = self.compute_st(t, z, y, y)

        return st

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)


    def compute_drift(self, x, t, y):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        exp_itg_beta = torch.exp(self.log_mean_coeff(t))
        diffusion = beta_t + 2*self.eta_coff*(1 - exp_itg_beta**2)
        drift = torch.sqrt(-0.5 *beta_t*x + self.eta_coff*(y*exp_itg_beta - x))
        return drift, diffusion
    

    def compute_diffusion(self, x, t, y, form="SBDM", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = torch.ones(x.shape[0], device=x.device) * t
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t, y)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        temp = self.log_mean_coeff(t)
        alpha_t = torch.exp(temp)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        d_sigma_t = torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_lambda_t(self, t):
        """Compute the linear interpolation coefficient"""
        # st = (1 - torch.exp(-self.eta_coff*t)) * x + torch.exp(-self.eta_coff*t) * y
        lambda_t = torch.exp(-self.eta_coff*t)
        d_lambda_t = -self.eta_coff * torch.exp(-self.eta_coff*t)
        return lambda_t, d_lambda_t

    def compute_mu_t(self, t, z, x, y):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        vt, _= self.compute_vt(t, x, y)
        return alpha_t * vt + sigma_t * z
    
    def compute_st(self, t, z, x, y):
        """Sample xt from time-dependent density p_t; rng is required"""
        st = self.compute_mu_t(t, z, x, y)
        return st
    
    def compute_vt(self, t, x, y):
        lambda_t, d_lambda_t = self.compute_lambda_t(t)
        vt = lambda_t*x + (1-lambda_t)*y
        d_vt = d_lambda_t*x - d_lambda_t*y
        return vt, d_vt

    def compute_ut(self, t, z, x, y):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        vt , d_vt = self.compute_vt(t, x, y)

        return d_alpha_t * vt + alpha_t * d_vt + d_sigma_t * z
    
    def _forward(self, t, z, x, y):
        st = self.compute_st(t, z, x, y)
        ut = self.compute_ut(t, z, x, y)
        return t, st, ut


    """velocity and score transform"""
    def get_score_from_velocity(self, velocity, x, t, y):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        lambda_t, d_lambda_t = self.compute_lambda_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)

        A = d_alpha_t * lambda_t + alpha_t * d_lambda_t
        up = alpha_t*lambda_t*velocity - A*x + (A*alpha_t-alpha_t*d_alpha_t*lambda_t)*y
        down = A*(sigma_t**2) - alpha_t*lambda_t*sigma_t*d_sigma_t

        # alpha_ratio = d_alpha_t / alpha_t
        # up = -alpha_t*lambda_t*y - lambda_t*velocity + velocity
        # down = alpha_ratio*(lambda_t-1)*(sigma_t**2+1) + \
        #     d_lambda_t*sigma_t**2 - lambda_t*d_sigma_t*sigma_t + d_lambda_t + sigma_t*d_sigma_t
        # print(f't={t}, alpha_t={alpha_t}, d_alpha_t={d_alpha_t}, sigma_t={sigma_t}, d_sigma_t={d_sigma_t}')
        score = up / down
        return score
    
    def discretize(self, x, t, y, *args):
        raise NotImplementedError("discretize for IC_VPSDE not yet implemented!")

    def reverse(oself, velocity_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                # 首先调用父类的__init__方法，传入所有必需的参数
                super().__init__(
                    sigma_min=oself.sigma_min,
                    sigma_max=oself.sigma_max,
                    N=oself.N,
                    t_eps=oself.t_eps,
                    eta_coff=oself.eta_coff
                )
                
                # 设置RSDE特有的属性
                self.probability_flow = probability_flow
                self.get_diffusion = oself.compute_diffusion
                # 复制原始SDE的方法引用
                self.get_score_from_velocity = oself.get_score_from_velocity
                self.discretize_fn = oself.discretize

            @property
            def T(self):
                return T

            def sde(self, x, t, y, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                # _, diffusion = sde_fn(x, t, y)
                velocity = velocity_model(x, t, y)
                if self.probability_flow:
                    # ODE for probability flow sampling
                    return velocity, torch.zeros_like(velocity)
                G = self.compute_diffusion(x, t, y)
                score = self.get_score_from_velocity(velocity, x, t, y)
                total_drift = velocity - 0.5 * G.view(-1, 1, 1, 1)**2 * score
                total_diffusion = G
                return total_drift, total_diffusion

            def discretize(self, x, t, y, *args):
                raise NotImplementedError("discretize for IC_VPSDE not yet implemented!")
            
        return RSDE()