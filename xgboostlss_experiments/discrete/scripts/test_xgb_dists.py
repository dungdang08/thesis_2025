import os
import sys

import torch
import numpy as np
from scipy.stats import invgauss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'third_party', 'xgboostlss'))
from xgboostlss.distributions.inverse_gaussian_torch import inverseGaussian

# Make sure your custom distribution is imported
# from your_module import inverseGaussian

# Parameters
loc = torch.tensor(0.5)
shp = torch.tensor(2.0)
sample_shape = (50,)

# Custom PyTorch distribution
d = inverseGaussian(loc, shp)
torch.manual_seed(2)  # For reproducibility
# Estimated from samples
empirical_mean = d.sample(sample_shape).mean().item()
empirical_var = d.sample(sample_shape).var(unbiased=False).item()

np_samples = np.random.wald(mean=loc, scale=shp, size=sample_shape[0])
np_mean = np.mean(np_samples)
np_var = np.var(np_samples)


# Estimated from rsamples
empirical_rmean = d.rsample(sample_shape).mean().item()
empirical_rvar = d.rsample(sample_shape).var(unbiased=False).item()

# Closed-form (analytical)
theoretical_mean = d.mean.item()
theoretical_var = d.variance.item()

# SciPy reference
mu_scipy = loc.item()
scale_scipy = shp.item()
scipy_dist = invgauss(mu=mu_scipy / scale_scipy, scale=scale_scipy)
scipy_mean = scipy_dist.mean()
scipy_var = scipy_dist.var()


print("Custom Inverse Gaussian (PyTorch)")
print(f"Empirical Mean     = {empirical_mean:.6f} (sample)")
print(f"Empirical Variance = {empirical_var:.6f} (sample)")
print(f"Empirical Mean     = {empirical_rmean:.6f} (rsample)")
print(f"Empirical Variance = {empirical_rvar:.6f} (rsample)")

print("\nNumPy np.random.wald")
print(f"Sample Mean     = {np_mean:.6f}")
print(f"Sample Variance = {np_var:.6f}")

print(f"\nTheoretical Mean   = {theoretical_mean:.6f}")
print(f"Theoretical Variance = {theoretical_var:.6f}")

print("\nSciPy invgauss")
print(f"SciPy Mean         = {scipy_mean:.6f}")
print(f"SciPy Variance     = {scipy_var:.6f}")