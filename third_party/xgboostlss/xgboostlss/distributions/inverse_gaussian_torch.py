#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of the inverse Gaussian (Wald) distribution, written by Hoang Dung Dang
(bachelor thesis) for use in XGBoostLSS.

The sampling routine (`inverseGaussian.rsample`), including its numerical-stability comments, is
ported from TensorFlow Probability:
    https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/inverse_gaussian.py
    Copyright 2018 The TensorFlow Probability Authors. Licensed under the Apache License, Version 2.0.
"""

from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all, _standard_normal
from torch.types import _size
from torch import Tensor
from typing import Tuple

import numpy as np
import scipy.special as spf


class ModifiedBesselFn2(torch.autograd.Function):
    """
    Modified Bessel function of the second kind K_nu(x), see https://en.wikipedia.org/wiki/Bessel_function
    Uses scipy functions, so it will only work on CPU
    """
    
    @staticmethod
    def forward(ctx, inp, nu):
        if inp.is_cuda:
            raise RuntimeError("ModifiedBesselFn2 only supports CPU tensors.")
        ctx._nu = float(nu)
        ctx.save_for_backward(inp)
        out_np = spf.kv(ctx._nu, inp.detach().cpu().numpy())
        return torch.from_numpy(out_np).to(dtype=inp.dtype, device=inp.device)
    

    @staticmethod
    def backward(ctx, grad_out):
        (inp,) = ctx.saved_tensors
        nu = ctx._nu
        if inp.is_cuda:
            raise RuntimeError("ModifiedBesselFn2 only supports CPU tensors.")
        # d/dx log K_ν(x) = K_ν'(x) / K_ν(x)
        kv_np = spf.kv(nu, inp.detach().cpu().numpy())
        kvp_np = spf.kvp(nu, inp.detach().cpu().numpy()) # derivative w.r.t. x
        dlog_np = kvp_np / kv_np
        grad_inp = grad_out.detach().cpu().numpy() * dlog_np
        return torch.from_numpy(grad_inp).to(dtype=inp.dtype, device=inp.device), None

    
    
class LogModifiedBesselDeriv2_num(torch.autograd.Function):
    """
    Derivative of the log of the modified Bessel function of the second kind K_nu(x) w.r.t. order nu, 
    see https://en.wikipedia.org/wiki/Bessel_function
    Implemented by 5-point stencil which has 4-th order error
    Uses scipy functions, so it will only work on CPU
    """
    dx=1e-4 # finite differencing stepsize

    @staticmethod
    def forward(ctx, inp, nu):
        if inp.is_cuda:
            raise RuntimeError("LogModifiedBesselDeriv2_num only supports CPU tensors.")
        dx = LogModifiedBesselDeriv2_num.dx
        ctx._nu = float(nu)
        ctx.save_for_backward(inp)
        x_np = inp.detach().cpu().numpy()
        nu = ctx._nu
        out_np = (
            (np.log(spf.kv(nu - 2 * dx, x_np))
            - 8 * np.log(spf.kv(nu - dx, x_np))
            + 8 * np.log(spf.kv(nu + dx, x_np))
            - np.log(spf.kv(nu + 2 * dx, x_np)))
            / (12.0 * dx)
        )
        return torch.from_numpy(out_np).to(dtype=inp.dtype, device=inp.device)

    @staticmethod
    def backward(ctx, grad_out):
        (inp,) = ctx.saved_tensors
        nu = ctx._nu
        dx = LogModifiedBesselDeriv2_num.dx
        if inp.is_cuda:
            raise RuntimeError("LogModifiedBesselDeriv2_num only supports CPU tensors.")
        x_np = inp.detach().cpu().numpy()

        def dlog(nuu: float) -> np.ndarray:
            return spf.kvp(nuu, x_np) / spf.kv(nuu, x_np)

        deriv_np = (
            dlog(nu - 2 * dx) - 
            8 * dlog(nu - dx) + 
            8 * dlog(nu + dx) - 
            dlog(nu + 2 * dx)
        ) / (12.0 * dx)

        grad_inp = grad_out.detach().cpu().numpy() * deriv_np
        return torch.from_numpy(grad_inp).to(dtype=inp.dtype, device=inp.device), None
        

modified_bessel2 = ModifiedBesselFn2.apply
log_modified_bessel_deriv2 = LogModifiedBesselDeriv2_num.apply


# ----------------------------------
# Inverse Gaussian distribution
# ---------------------------------
class inverseGaussian(ExponentialFamily):
    r"""
    Creates a inverse Gaussian distribution parametrized by :attr:loc (mean) and :attr:concentration (shape parameter).
    It's also known as the Wald distribution. Some, e.g., the Python scipy package, 
    refer to the special case when loc is 1 as the Wald distribution.

    Mapping to Python scipy's parameterization:
    Python: scipy.stats.invgauss
     - mu = loc / concentration
     - scale = concentration

    Samples are positive real. The inverse Gaussian describes the reaction times in a drift-diffusion model.
    
    Example::

        >>> m = inverseGaussian(torch.tensor([0.3]), torch.tensor([2.0]))
        >>> reaction_times = m.sample(size=1000)
        >>> reaction_times.mean() # should be close to 0.3
        >>> reaction_times.var() # should be close to 0.3**3 * 2.0**2
    Args:
        loc (Number, Tensor): the mean value of samples from this distribution.
        shp (Number, Tensor): the deviation around the mean
    """
    arg_constraints = {'loc': constraints.positive,
                       'shp': constraints.positive}
    support = constraints.positive
    has_enumerate_support = False
    has_rsample = True
    
    def __init__(self, loc: Tensor | Number, shp: Tensor | Number, validate_args=None):
        if isinstance(loc, Number) and isinstance(shp, Number):
            self.loc = torch.as_tensor(float(loc))
            self.shp = torch.as_tensor(float(shp))
            batch_shape = torch.Size()
        else:
            self.loc, self.shp = broadcast_all(loc, shp)
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.loc.pow(3) / self.shp

    @property
    def stddev(self) -> Tensor:
        return (self.loc.pow(3) / self.shp).sqrt()
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(inverseGaussian, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.shp = self.shp.expand(batch_shape)

        super(inverseGaussian, new).__init__(batch_shape, validate_args=False)
        
        new._validate_args = self._validate_args
        return new

    # def sample(self, sample_shape=torch.Size()) -> Tensor:
    #     """
    #     Generates a sample_shape shaped sample or sample_shape shaped batch of
    #     samples if the distribution parameters are batched.
    #     """
    #     shape = self._extended_shape(sample_shape)
    #     with torch.no_grad():
    #         loc_np = self.loc.cpu().numpy()
    #         shp_np = self.shp.cpu().numpy()
    #         samples = np.random.wald(loc_np, shp_np, size=shape)
    #         return torch.tensor(samples, dtype=self.loc.dtype, device=self.loc.device)
         
        
    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        See https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution, https://www.jstor.org/stable/2683801
        Sampling via the transformation method of Michael, Schucany & Haas (1976), "Generating random
        variates using transformations with multiple roots", The American Statistician 30(2):88-90.
        Ported from TensorFlow Probability (Apache-2.0):
        https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/inverse_gaussian.py
        """
        
        # Wikipedia defines an intermediate x with the formula
        #   x = loc + loc ** 2 * y / (2 * conc)
        #       - loc / (2 * conc) * sqrt(4 * loc * conc * y + loc ** 2 * y ** 2)
        # where y ~ N(0, 1)**2 (sampled_chi2 above) and conc is the concentration.
        # Let us write
        #   w = loc * y / (2 * conc)
        # Then we can extract the common factor in the last two terms to obtain
        #   x = loc + loc * w * (1 - sqrt(2 / w + 1))
        # Now we see that the Wikipedia formula suffers from catastrphic
        # cancellation for large w (e.g., if conc << loc).
        #
        # Fortunately, we can fix this by multiplying both sides
        # by 1 + sqrt(2 / w + 1).  We get
        #   x * (1 + sqrt(2 / w + 1)) =
        #     = loc * (1 + sqrt(2 / w + 1)) + loc * w * (1 - (2 / w + 1))
        #     = loc * (sqrt(2 / w + 1) - 1)
        # The term sqrt(2 / w + 1) + 1 no longer presents numerical difficulties for large w, 
        # and sqrt(2 / w + 1) - 1 is just sqrt1pm1(2 / w), 
        # which we know how to compute accurately.
        # 
        # This just leaves the matter of small w, where 2 / w may overflow.  
        # In the limit a w -> 0, x -> loc, so we just mask that case.
        shape = self._extended_shape(sample_shape)
        sampled_chi2 = _standard_normal(shape, device=self.loc.device, dtype=self.loc.dtype) ** 2
        sampled_uniform = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)

        sqrt1pm1_arg = 4.0 * self.shp / (self.loc * sampled_chi2)
        safe_arg = torch.where(torch.isfinite(sqrt1pm1_arg), sqrt1pm1_arg, torch.ones_like(sqrt1pm1_arg))
        
        # Compute sqrt1pm1(x) = sqrt(1 + x) - 1 in a numerically stable way
        sqrt_term = torch.sqrt(safe_arg + 1.0)
        ratio = safe_arg / (1.0 + sqrt_term).pow(2)

        sampled = self.loc * torch.where(torch.isfinite(sqrt1pm1_arg), ratio, torch.ones_like(ratio))
        accept = sampled_uniform <= self.loc / (self.loc + sampled)
        return torch.where(accept, sampled, self.loc.pow(2) / sampled)


    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        μ, λ, x = broadcast_all(self.loc, self.shp, value)
        LOG_2PI = x.new_tensor(2.0 * np.pi)

        return (
            0.5 * (torch.log(λ) - torch.log(LOG_2PI) - 3.0 * torch.log(x)) 
            -  (λ * (x - μ).pow(2)) / (2.0 * μ.pow(2) * value))        

    @property
    def _natural_params(self) -> Tuple[Tensor, Tensor]:
        return (-self.shp / (2.0 * self.loc.pow(2)), -self.shp / 2.0)

    def _log_normalizer(self, theta1: Tensor, theta2: Tensor) -> Tensor:
        return -2 * torch.sqrt(theta1 * theta2) - 0.5 * torch.log(- theta2 / np.pi)
    
    def _log_normalizer_grad(self, theta1: Tensor, theta2: Tensor) -> Tuple[Tensor, Tensor]:
        # Both theta1 and theta2 are negative for the inverse Gaussian's natural parameters.
        # When differentiating the log normalizer, care must be taken with the square roots and signs:
        #   d/dtheta1 [-2 * sqrt(theta1 * theta2)] = -2 * (1/2) * theta2 / sqrt(theta1 * theta2) = -theta2 / sqrt(theta1 * theta2)
        #   d/dtheta2 [-2 * sqrt(theta1 * theta2) - 0.5 * log(-theta2 / pi)] = 
        #       -2 * (1/2) * theta1 / sqrt(theta1 * theta2) - 0.5 * (1 / theta2)
        #     = -theta1 / sqrt(theta1 * theta2) - 0.5 / theta2
        # Since both theta1 and theta2 are negative, sqrt(theta1 * theta2) is real and positive.
        g1 = -torch.sqrt(theta2 / theta1)
        g2 = -torch.sqrt(theta1 / theta2) - 0.5 / theta2
        return g1, g2
    
    @property
    def _mean_carrier_measure(self) -> Tensor:
        # carrier measure is - 3/2 * log(x)
        # The mean carrier measure can be computed from the derviative of a modified Bessel function of the second kind, which 
        # is implemented in above. See https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
        
        ls = self.shp / self.loc
        return -1.5 * (torch.log(self.loc) + log_modified_bessel_deriv2(ls, -0.5))
    
    def entropy(self):
        """
        Differentiable version of entropy using exponential family identities
        The exponential family entropy of pytorch is not differentiable w.r.t. the parameters, 
        because the natural parameters are detached (version 1.5.0)

        H = -E[k(X)] + A(θ) - ⟨θ, ∇A(θ)⟩
        """
        result = -self._mean_carrier_measure
        θ1, θ2 = self._natural_params
        A = self._log_normalizer(θ1, θ2)
        g1, g2 = self._log_normalizer_grad(θ1, θ2)
        result = result + A - (θ1 * g1 + θ2 * g2)
        return result