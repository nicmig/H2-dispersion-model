"""
Additive Kernels for Gaussian Processes

Implements two different additive kernels based on Duvenaud et al. (2011) - "Additive Gaussian Processes"

"""

import torch
from torch import Tensor
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive
from linear_operator.operators import ZeroLinearOperator, LinearOperator
from linear_operator import to_linear_operator, to_dense
from itertools import combinations
from typing import Optional


class KernelWeightedAdditiveKernel(Kernel):
    """
    Kernel-weighted Additive Kernel.
    
    This kernel decomposes the function into a sum of interactions of all orders:
    - 1st order: individual dimensions
    - 2nd order: pairs of dimensions  
    - 3rd order: triples of dimensions
    - etc.
    
    For D dimensions, there are 2^D - 1 terms. Each dimension has its own kernel with its own output scale σ².
    This results in D output scales to optimize.
    
    Args:
        base_kernels: List of scalable base kernels, one per dimension
        
    Example:
        >>> # 3-dimensional input with RBF on each dimension
        >>> k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k3 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> additive_kernel = FullAdditiveKernel([k1, k2, k3])
        >>> 
        >>> # This creates: k = σ²₁k₁ + σ²₂k₂ + σ²₃k₃ + σ²₁k₁σ²₂k₂ + σ²₁k₁σ²₃k₃ + σ²₂k₂σ²₃k₃ + σ²₁k₁σ²₂k₂σ²₃k₃
    
    """

    @property
    def is_stationary(self) -> bool:
        return all(k.is_stationary for k in self.base_kernels)
    
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
                raise ValueError(f"Unknown kernel type: {base_kernel_type}. Only RBF kernel and Matern kernel are implemented.")
            
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


class SubsetWeightedAdditiveKernel(Kernel):
    """
    Subset-weighted Additive Kernel.
    
    This kernel decomposes the function into a sum of interactions of all orders:
    - 1st order: individual dimensions
    - 2nd order: pairs of dimensions  
    - 3rd order: triples of dimensions
    - etc.
    
    For D dimensions, there are 2^D - 1 terms. Each subset has its own output scale σ².
    This results in 2^D - 1 output scales and limits this method to lower dimension data because this method
    is computationally expensive but increases the interpretability.
    
    Args:
        base_kernels: List of scalable base kernels, one per dimension
        
    Example:
        >>> # 3-dimensional input with RBF on each dimension
        >>> k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> k3 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> additive_kernel = FullAdditiveKernel([k1, k2, k3])
        >>> 
        >>> # This creates: k = σ²₁k₁ + σ²₂k₂ + σ²₃k₃ + σ²₄k₁k₂ + σ²₅k₁k₃ + σ²₆k₂k₃ + σ²₇k₁k₂k₃
    
    """

    @property
    def is_stationary(self) -> bool:
        return all(k.is_stationary for k in self.base_kernels)
    
    def __init__(
        self,
        base_kernel_type: str,
        num_dims: int,
        lengthscale_prior=None,
        lengthscale_constraints=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_dims = num_dims
        base_kernels = []
        
        # lengthscale_constraints: list of constraints, one per dimension
        if lengthscale_constraints is None:
            lengthscale_constraints = [None] * num_dims
    
        for i in range(num_dims):
            lc = lengthscale_constraints[i] if i < len(lengthscale_constraints) else None
            if base_kernel_type.lower() == 'rbf':
                k = gpytorch.kernels.keops.RBFKernel(
                    ard_num_dims=1, active_dims=[i],
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lc,
                    **kwargs
                )
            elif base_kernel_type.lower() == 'matern32':
                k = gpytorch.kernels.keops.MaternKernel(
                    nu=1.5, active_dims=[i],
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lc,
                    **kwargs
                )
            elif base_kernel_type.lower() == 'matern52':
                k = gpytorch.kernels.keops.MaternKernel(
                    nu=2.5, active_dims=[i],
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=lc,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown kernel type: {base_kernel_type}. Only RBF kernel and Matern kernel are implemented.")
            
            base_kernels.append(k)
        self.base_kernels = torch.nn.ModuleList(base_kernels)

        # Generate all non-empty subsets of dimensions
        self.subsets = []
        for order in range(1, self.num_dims + 1):
            for subset in combinations(range(self.num_dims), order):
                self.subsets.append(subset)
        
        self.num_terms = len(self.subsets)
        self.raw_outputscales = torch.nn.Parameter(torch.rand(self.num_terms))
        self.register_constraint("raw_outputscales", Positive())

    def _outputscale_param(self, m):
        return m.outputscales

    def _outputscale_closure(self, m, v):
        m._set_outputscale(v)
    
    @property
    def outputscales(self):
        return self.raw_outputscales_constraint.transform(self.raw_outputscales)

    @outputscales.setter
    def outputscales(self, value):
        self.initialize(raw_outputscales=self.raw_outputscales_constraint.inverse_transform(value))


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
        outputscales = self.outputscales
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

            
            if diag:
                outputscale = outputscales[i].unsqueeze(-1)
                res_prod = to_dense(res_prod) * outputscale
            else:
                outputscale = outputscales[i].view(*outputscales[i].shape, 1, 1)
                res_prod = res_prod.mul(outputscale)

            if not diag:
                res = res + to_linear_operator(res_prod)
            else:
                res = res + res_prod
            
        return res
            
    def num_outputs_per_input(self, x1, x2):
        return self.base_kernels[0].num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernels[0].prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
