"""
Additive Kernels for Gaussian Processes

Implements the full additive kernel from:
Duvenaud et al. (2011) - "Additive Gaussian Processes"

The full additive kernel models interactions of all orders:
k(x, x') = Σ_{d=1}^D Σ_{|S|=d} σ²_S ∏_{i∈S} k_i(x_i, x'_i)

Where:
- S are all non-empty subsets of dimensions
- σ²_S are outputscales for each interaction term
- k_i are base kernels on individual dimensions
"""

import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel
from linear_operator.operators import ZeroLinearOperator, LinearOperator
from linear_operator import to_linear_operator, to_dense
from itertools import combinations
from typing import Optional


class FullAdditiveKernel(Kernel):
    """
    Full Additive Kernel as introduced by Duvenaud et al. (2011).
    
    This kernel decomposes the function into a sum of interactions of all orders:
    - 1st order: individual dimensions
    - 2nd order: pairs of dimensions  
    - 3rd order: triples of dimensions
    - etc.
    
    The kernel is defined as:
    k(x, x') = Σ_{d=1}^D Σ_{|S|=d} σ²_S ∏_{i∈S} k_i(x_i, x'_i)
    
    For D dimensions, there are 2^D - 1 terms (all non-empty subsets).
    
    Args:
        base_kernels: List of scalable base kernels, one per dimension
        
    Example:
        >>> # 3-dimensional input with RBF on each dimension
        >>> k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k3 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> additive_kernel = FullAdditiveKernel([k1, k2, k3])
        >>> 
        >>> # This creates: k = σ²₁k₁ + σ²₂k₂ + σ²₃k₃ + σ²₁₂k₁k₂ + σ²₁₃k₁k₃ + σ²₂₃k₂k₃ + σ²₁₂₃k₁k₂k₃
    
    Reference:
        Duvenaud, D., Nickisch, H., & Rasmussen, C. (2011).
        "Additive Gaussian Processes." NIPS 2011.
    """
    
    def __init__(
        self,
        base_kernel_type: str,
        num_dims: int,
        lengthscale_prior=None,
        outputscale_prior=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_dims = num_dims
        base_kernels = []
    
        for i in range(num_dims):
            if base_kernel_type.lower() == 'rbf':
                k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[i], lengthscale_prior=lengthscale_prior, **kwargs), outputscale_prior=outputscale_prior)
            elif base_kernel_type.lower() == 'matern32':
                k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=1.5, active_dims=[i]), outputscale_prior=outputscale_prior)
            elif base_kernel_type.lower() == 'matern52':
                k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5, active_dims=[i]), outputscale_prior=outputscale_prior)
            else:
                raise ValueError(f"Unknown kernel type: {base_kernel_type}")
            
            base_kernels.append(k)
        self.base_kernels = torch.nn.ModuleList(base_kernels)
        
        # Generate all non-empty subsets of dimensions
        self.subsets = []
        for order in range(1, self.num_dims + 1):
            for subset in combinations(range(self.num_dims), order):
                self.subsets.append(subset)
        
        self.num_terms = len(self.subsets)
    
    def forward(self, x1, x2, diag=False, **params) -> Tensor | LinearOperator:
        """
        Compute the full additive kernel matrix.
        
        Args:
            x1: First input tensor of shape (n, D) where D is num_dims
            x2: Second input tensor of shape (m, D) where D is num_dims
            diag: If True, return only the diagonal
            
        Returns:
            Kernel matrix of shape (n, m) or (n,) if diag=True
        """
        
        # Start with zero
        res = ZeroLinearOperator() if not diag else 0
        
        # Compute each interaction term
        for i, subset in enumerate(self.subsets):
            # Product of base kernels for dimensions in this subset
            next_term = None
            k_0 = self.base_kernels[subset[0]]
            x1_eq_x2 = torch.equal(x1, x2)
            if not x1_eq_x2:
                # If x1 != x2, then we can't make a MulLinearOperator because the kernel won't necessarily be square/symmetric
                res_prod = to_dense(k_0(x1, x2, diag=diag, **params))
            else:
                res_prod = k_0(x1, x2, diag=diag, **params)

                if not diag:
                    res_prod = to_linear_operator(res_prod)
            
            if len(subset) > 1:
                for dim_idx in subset[1:]:
                    # Get the base kernel for this dimension
                    k = self.base_kernels[dim_idx]
                    
                    next_term = k(x1, x2, diag=diag, **params)
                    if not x1_eq_x2:
                        res_prod = res_prod * to_dense(next_term)
                    else:
                        if not diag:
                            res_prod = res_prod * to_linear_operator(next_term)
                        else:
                            res_prod = res_prod * next_term

            if not diag:
                res = res + to_linear_operator(res_prod)
            else:
                res = res + res_prod
        
        return res


# =============================================================================
# ExactGP Model with Additive Kernel
# =============================================================================

class ExactGPAdditiveModel(gpytorch.models.ExactGP):
    """
    Exact GP model using the Full Additive Kernel with RBF base kernels.
    
    This model implements the additive GP from Duvenaud et al. 2011 for exact
    GP inference. Suitable for smaller datasets (up to ~10,000 points).
    
    Args:
        train_x: Training inputs of shape (n, num_dims)
        train_y: Training targets of shape (n,)
        likelihood: GPyTorch likelihood (typically GaussianLikelihood)
        num_dims: Number of input dimensions (inferred if not provided)
        learn_outputscales: Whether to learn the additive interaction outputscales
        
    Example:
        >>> # Create model for 4D input
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>> model = ExactGPAdditiveModel(train_x, train_y, likelihood, num_dims=4)
        >>>
        >>> # Training
        >>> model.train()
        >>> likelihood.train()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> for i in range(100):
        ...     optimizer.zero_grad()
        ...     output = model(train_x)
        ...     loss = -mll(output, train_y)
        ...     loss.backward()
        ...     optimizer.step()
        >>>
        >>> # Prediction
        >>> model.eval()
        >>> likelihood.eval()
        >>> with torch.no_grad(), gpytorch.settings.fast_pred_var():
        ...     predictions = likelihood(model(test_x))
    
    Reference:
        Duvenaud, D., Nickisch, H., & Rasmussen, C. (2011).
        "Additive Gaussian Processes." NIPS 2011.
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        num_dims: Optional[int] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        
        # Infer number of dimensions
        if num_dims is None:
            if train_x.dim() == 1:
                num_dims = 1
            else:
                num_dims = train_x.size(-1)
        
        self.num_dims = num_dims
        
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Covariance module: Full additive kernel with RBF bases
        self.covar_module = FullAdditiveKernel(
            base_kernel_type='rbf',
            num_dims=num_dims,
        )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (n, num_dims)
            
        Returns:
            MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)