"""
Latent Mass Flow GP for H2 Dispersion Prediction

During operation:
- INPUT: time, sensor observations {sensor_id: h2_concentration}
- OUTPUT: concentration field prediction + mass flow estimate + uncertainty

Training:
- INPUT: time, mass_flow, y, z (from CFD + experiments) - x removed as domain is 1D in x
- OUTPUT: h2_concentration
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time as time_module
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


@dataclass
class InferenceResult:
    """Container for inference results"""
    mass_flow_map: float                    # MAP estimate
    mass_flow_posterior_mean: float         # Posterior mean
    mass_flow_posterior_std: float          # Posterior uncertainty
    mass_flow_credible_interval: Tuple[float, float]  # 95% CI
    field_mean: np.ndarray                  # Mean concentration field
    field_std: np.ndarray                   # Std deviation field
    total_uncertainty: np.ndarray           # Combined uncertainty
    danger_zones: Dict                      # Safety-relevant info
    sensor_predictions: Dict[int, Tuple[float, float]]  # Predicted vs observed
    anomalies: Dict                         # detected anomalies in the sensors
    inference_time_ms: float                # Computational time


class SparseH2DispersionGP(gpytorch.models.ApproximateGP):
    """
    Sparse Variational GP for H2 dispersion.
    
    Uses inducing points to approximate the full GP posterior.
    Scales to large datasets (O(m²n) instead of O(n³)).
    
    GP model: f(time, mass_flow, y, z) -> h2_concentration (x removed)
    """
    
    def __init__(self, inducing_points, learn_inducing=True):
        """
        Args:
            inducing_points: Initial inducing point locations [n_inducing, n_features]
            learn_inducing: Whether to optimize inducing point locations
        """
        # Variational distribution q(u)
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        # Variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            inducing_points, 
            variational_dist,
            learn_inducing_locations=learn_inducing
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Composite kernel with different characteristics per dimension
        # Dimension order: [time, mass_flow, y, z]
        # Use default lengthscale initialization (more stable)
        
        # Time kernel: smoother evolution
        self.time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(nu=0.5, active_dims=[0])
        )
              
        # Mass flow kernel: smooth variation
        self.mass_flow_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(active_dims=[1])
        )
        
        # Space kernel: anisotropic dispersion (x removed)
        self.space_kernel_y = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(active_dims=[2])
        )
        
        self.space_kernel_z = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(active_dims=[3])
        )
        
        # Combined kernel
        self.covar_module = (
            self.time_kernel + 
            self.mass_flow_kernel + 
            self.space_kernel_y +
            self.space_kernel_z
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class LatentMassFlowPredictor:
    """
    Operational predictor that infers mass flow from sensor readings
    and predicts concentration field.
    """
    
    def __init__(self, 
                 gp_model: SparseH2DispersionGP,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 sensor_positions: Dict[int, Tuple[float, float, float]],
                 flammability_limit: float = 0.04,
                 device: str = 'cpu'):
        
        self.device = device
        self.sensor_positions = sensor_positions
        self.flammability_limit = flammability_limit
        
        # Move model to specified device and ensure all components are synced
        self.gp = gp_model.to(device)
        self.likelihood = likelihood.to(device)
        
        # For sparse GP, explicitly move inducing points
        if hasattr(self.gp, 'variational_strategy'):
            inducing_points = self.gp.variational_strategy.inducing_points
            self.gp.variational_strategy.inducing_points = inducing_points.to(device)
        
        self.gp.eval()
        self.likelihood.eval()
        
        # Cache for repeated queries
        self._cache = {}
        
        # Mass flow prior (learned from training data or set physically)
        self.mass_flow_prior_mean = 0.5
        self.mass_flow_prior_std = 0.4
        self.mass_flow_bounds = (0.01, 2.0)  # Physical bounds kg/s
        
    def set_mass_flow_prior(self, mean: float, std: float):
        """Update prior based on operational knowledge"""
        self.mass_flow_prior_mean = mean
        self.mass_flow_prior_std = std
    
    def _compute_log_likelihood(self, 
                                mass_flow: float, 
                                time: float,
                                observed_sensors: List[int],
                                observed_values: np.ndarray) -> float:
        """
        Compute log p(y_obs | mass_flow, time) using the GP.
        
        This is the likelihood function for Bayesian inference over mass flow.
        """
        try:
            # Build input: [time, mass_flow, y, z] for each observed sensor
            X_obs = []
            for s in observed_sensors:
                _, y, z = self.sensor_positions[s]
                X_obs.append([time, mass_flow, y, z])
            
            X_obs = torch.tensor(X_obs, dtype=torch.float64, device=self.device)
            y_obs = torch.tensor(observed_values, dtype=torch.float64, device=self.device)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Get GP prediction at observed sensors
                pred = self.likelihood(self.gp(X_obs))
                
                # Log likelihood of observations under GP predictive distribution
                # For BetaLikelihood, log_prob returns per-observation values, sum them
                log_lik = pred.log_prob(y_obs).sum().item()
                
            # Add log prior
            log_prior = -0.5 * ((mass_flow - self.mass_flow_prior_mean) / self.mass_flow_prior_std)**2
            
            return log_lik + log_prior
            
        except Exception as e:
            warnings.warn(f"Error computing likelihood at mf={mass_flow}: {e}")
            return -1e10
    
    def infer_mass_flow_map(self, 
                           time: float,
                           sensor_readings: Dict[int, float],
                           method: str = 'gradient') -> Dict:
        """
        Maximum A Posteriori (MAP) estimate of mass flow.
        
        Methods:
        - 'grid': Exhaustive grid search (robust but slower)
        - 'gradient': Scipy optimization (faster, may get stuck)
        - 'hybrid': Grid then refine with gradient (recommended)
        """
        observed_sensors = list(sensor_readings.keys())
        observed_values = np.array(list(sensor_readings.values()))
        
        if method == 'grid':
            return self._infer_mass_flow_grid(time, observed_sensors, observed_values)
        elif method == 'gradient':
            return self._infer_mass_flow_gradient(time, observed_sensors, observed_values)
        elif method == 'hybrid':
            return self._infer_mass_flow_hybrid(time, observed_sensors, observed_values)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _infer_mass_flow_grid(self, time, observed_sensors, observed_values, n_points=100):
        """Grid search over mass flow values"""
        mf_values = np.linspace(self.mass_flow_bounds[0], self.mass_flow_bounds[1], n_points)
        log_liks = []
        
        for mf in mf_values:
            log_lik = self._compute_log_likelihood(mf, time, observed_sensors, observed_values)
            log_liks.append(log_lik)
        
        log_liks = np.array(log_liks)
        
        # MAP estimate
        map_idx = np.argmax(log_liks)
        map_estimate = mf_values[map_idx]
        
        # Convert to probability distribution
        log_liks_norm = log_liks - np.max(log_liks)  # Numerical stability
        probs = np.exp(log_liks_norm)
        probs = probs / probs.sum()
        
        # Posterior statistics
        posterior_mean = np.sum(mf_values * probs)
        posterior_var = np.sum(probs * (mf_values - posterior_mean)**2)
        posterior_std = np.sqrt(posterior_var)
        
        # Credible interval (95%)
        cum_probs = np.cumsum(probs)
        ci_lower = mf_values[np.searchsorted(cum_probs, 0.025)]
        ci_upper = mf_values[np.searchsorted(cum_probs, 0.975)]
        
        return {
            'map_estimate': map_estimate,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval': (ci_lower, ci_upper),
            'posterior_grid': (mf_values, probs),
            'log_likelihoods': log_liks
        }
    
    def _infer_mass_flow_gradient(self, time, observed_sensors, observed_values):
        """Gradient-based optimization"""
        
        def neg_log_posterior(mf):
            return -self._compute_log_likelihood(mf[0], time, observed_sensors, observed_values)
        
        result = minimize(
            neg_log_posterior,
            x0=[self.mass_flow_prior_mean],
            bounds=[self.mass_flow_bounds],
            method='L-BFGS-B'
        )
        
        map_estimate = result.x[0]
        
        # Estimate uncertainty via Hessian approximation
        # (Second derivative at MAP)
        eps = 0.01
        ll_plus = -neg_log_posterior([map_estimate + eps])
        ll_minus = -neg_log_posterior([map_estimate - eps])
        ll_map = -neg_log_posterior([map_estimate])
        
        hessian = (ll_plus - 2*ll_map + ll_minus) / (eps**2)
        if hessian < 0:
            hessian = -hessian  # Ensure positive
        posterior_std = 1.0 / np.sqrt(hessian + 1e-6)
        
        return {
            'map_estimate': map_estimate,
            'posterior_mean': map_estimate,  # Approximation
            'posterior_std': posterior_std,
            'credible_interval': (map_estimate - 2*posterior_std, map_estimate + 2*posterior_std),
            'optimization_success': result.success
        }
    
    def _infer_mass_flow_hybrid(self, time, observed_sensors, observed_values):
        """Coarse grid then gradient refinement"""
        # Coarse grid
        coarse = self._infer_mass_flow_grid(time, observed_sensors, observed_values, n_points=30)
        
        # Refine around MAP
        mf_center = coarse['map_estimate']
        search_radius = 0.2
        bounds = (
            max(self.mass_flow_bounds[0], mf_center - search_radius),
            min(self.mass_flow_bounds[1], mf_center + search_radius)
        )
        
        def neg_log_posterior(mf):
            return -self._compute_log_likelihood(mf[0], time, observed_sensors, observed_values)
        
        result = minimize(
            neg_log_posterior,
            x0=[mf_center],
            bounds=[bounds],
            method='L-BFGS-B'
        )
        
        # Recompute full posterior on fine grid around refined MAP
        map_estimate = result.x[0]
        mf_fine = np.linspace(
            max(self.mass_flow_bounds[0], map_estimate - 0.3),
            min(self.mass_flow_bounds[1], map_estimate + 0.3),
            50
        )
        
        log_liks = []
        for mf in mf_fine:
            log_liks.append(self._compute_log_likelihood(mf, time, observed_sensors, observed_values))
        
        log_liks = np.array(log_liks)
        log_liks_norm = log_liks - np.max(log_liks)
        probs = np.exp(log_liks_norm)
        probs = probs / probs.sum()
        
        posterior_mean = np.sum(mf_fine * probs)
        posterior_std = np.sqrt(np.sum(probs * (mf_fine - posterior_mean)**2))
        
        cum_probs = np.cumsum(probs)
        ci_lower = mf_fine[np.searchsorted(cum_probs, 0.025)]
        ci_upper = mf_fine[np.searchsorted(cum_probs, 0.975)]
        
        return {
            'map_estimate': map_estimate,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval': (ci_lower, ci_upper),
            'posterior_grid': (mf_fine, probs),
            'optimization_success': result.success
        }
    
    def predict_field(self,
                     time: float,
                     sensor_readings: Dict[int, float],
                     grid_points: np.ndarray,
                     method: str = 'marginalize',
                     n_mcmc_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict concentration field at arbitrary 3D locations.
        
        Args:
            time: Current time
            sensor_readings: {sensor_id: h2_concentration}
            grid_points: [N, 3] array of (x, y, z) coordinates
            method: 'map' (point estimate), 'marginalize' (Bayesian), 'mcmc' (full posterior)
            n_mcmc_samples: Number of samples for marginalization
        
        Returns:
            field_mean: [N] mean predictions
            field_std: [N] total uncertainty (aleatoric + epistemic)
        """
        if method == 'map':
            return self._predict_field_map(time, sensor_readings, grid_points)
        elif method == 'marginalize':
            return self._predict_field_marginalize(time, sensor_readings, grid_points, n_mcmc_samples)
        elif method == 'mcmc':
            return self._predict_field_mcmc(time, sensor_readings, grid_points, n_mcmc_samples)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_field_map(self, time, sensor_readings, grid_points):
        """Predict using MAP mass flow estimate"""
        mf_info = self.infer_mass_flow_map(time, sensor_readings, method='hybrid')
        mass_flow = mf_info['map_estimate']
        
        return self._predict_at_mass_flow(time, mass_flow, grid_points)
    
    def _predict_field_marginalize(self, time, sensor_readings, grid_points, n_samples):
        """Marginalize over mass flow uncertainty"""
        # Get mass flow posterior
        mf_info = self.infer_mass_flow_map(time, sensor_readings, method='hybrid')
        mf_mean = mf_info['posterior_mean']
        mf_std = mf_info['posterior_std']
        
        # Sample from mass flow posterior (truncated Gaussian)
        mf_samples = []
        while len(mf_samples) < n_samples:
            sample = np.random.normal(mf_mean, mf_std) # assumption that mass flow is normal distributed
            if self.mass_flow_bounds[0] <= sample <= self.mass_flow_bounds[1]:
                mf_samples.append(sample)
        
        # Predict for each mass flow sample
        predictions = []
        for mf in mf_samples:
            mean, _ = self._predict_at_mass_flow(time, mf, grid_points)
            predictions.append(mean)
        
        predictions = np.array(predictions)
        
        # Total uncertainty = epistemic (variance over samples) + aleatoric (GP noise)
        field_mean = predictions.mean(axis=0)
        epistemic_var = predictions.var(axis=0)
        _, aleatoric_std = self._predict_at_mass_flow(time, mf_mean, grid_points)
        aleatoric_var = aleatoric_std**2
        
        total_var = epistemic_var + aleatoric_var
        field_std = np.sqrt(total_var)
        
        return field_mean, field_std
    
    def _predict_field_mcmc(self, time, sensor_readings, grid_points, n_samples):
        """Full MCMC sampling (slower but most accurate)"""
        # Get posterior grid
        mf_info = self.infer_mass_flow_map(time, sensor_readings, method='hybrid')
        mf_grid, probs = mf_info['posterior_grid']
        
        # Sample from discrete posterior
        samples_idx = np.random.choice(len(mf_grid), size=n_samples, p=probs)
        mf_samples = mf_grid[samples_idx]
        
        predictions = []
        for mf in mf_samples:
            mean, _ = self._predict_at_mass_flow(time, mf, grid_points)
            predictions.append(mean)
        
        predictions = np.array(predictions)
        field_mean = predictions.mean(axis=0)
        field_std = predictions.std(axis=0)
        
        return field_mean, field_std
    
    def _predict_at_mass_flow(self, time, mass_flow, grid_points):
        """Internal: predict at specific mass flow"""
        n_points = len(grid_points)
        
        # Build input: [time, mass_flow, y, z] - x removed
        # grid_points is [N, 3] with [x, y, z], we only take [y, z]
        X_query = np.column_stack([
            np.full(n_points, time),
            np.full(n_points, mass_flow),
            grid_points[:, 1:]  # Only y, z columns
        ])
        
        # Detect model device and ensure consistency
        model_device = next(self.gp.parameters()).device
        X_query = torch.tensor(X_query, dtype=torch.float64, device=model_device)
        
        # Ensure all model components are on the same device
        self.gp = self.gp.to(model_device)
        self.likelihood = self.likelihood.to(model_device)
        if hasattr(self.gp, 'variational_strategy'):
            induc_points = self.gp.variational_strategy.inducing_points
            self.gp.variational_strategy.inducing_points = induc_points.to(model_device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp.eval()
            self.likelihood.eval()
            pred = self.likelihood(self.gp(X_query))
            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()
        
        return mean, std
    
    def identify_danger_zones(self, 
                             field_mean: np.ndarray,
                             field_std: np.ndarray,
                             grid_points: np.ndarray,
                             confidence_level: float = 0.90) -> Dict:
        """
        Identify regions where P(concentration > flammability_limit) > confidence_level
        """
        # Probability of exceeding limit
        z_scores = (field_mean - self.flammability_limit) / (field_std + 1e-9)
        prob_exceed = 1 - norm.cdf(z_scores)
        
        # High confidence danger zones
        danger_mask = prob_exceed > confidence_level
        
        if not np.any(danger_mask):
            return {
                'has_danger': False,
                'danger_volume_m3': 0.0,
                'max_concentration': field_mean.max(),
                'max_probability': prob_exceed.max()
            }
        
        danger_coords = grid_points[danger_mask]
        danger_concentrations = field_mean[danger_mask]
        danger_probs = prob_exceed[danger_mask]
        
        # Estimate volume (assumes regular grid)
        # For accurate volume, need voxel size
        voxel_volume = self._estimate_voxel_volume(grid_points)
        danger_volume = danger_mask.sum() * voxel_volume
        
        return {
            'has_danger': True,
            'danger_volume_m3': danger_volume,
            'max_concentration': field_mean.max(),
            'max_concentration_std': field_std[np.argmax(field_mean)],
            'max_probability': prob_exceed.max(),
            'danger_centroid': danger_coords.mean(axis=0) if len(danger_coords) > 0 else None,
            'n_voxels_danger': int(danger_mask.sum())
        }
    
    def _estimate_voxel_volume(self, grid_points):
        """Estimate voxel volume from grid spacing"""
        # Simple estimation: assume roughly cubic grid
        n_points = len(grid_points)
        # Approximate bounding box
        x_range = grid_points[:, 0].max() - grid_points[:, 0].min()
        y_range = grid_points[:, 1].max() - grid_points[:, 1].min()
        z_range = grid_points[:, 2].max() - grid_points[:, 2].min()
        
        # Assume uniform grid
        volume = x_range * y_range * z_range
        return volume / n_points
    
    def predict_sensor_values(self, 
                             time: float,
                             sensor_readings: Dict[int, float],
                             target_sensors: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        Predict H2 concentration at specific sensors (useful for validation
        or predicting at faulty/missing sensor locations).
        """
        positions = np.array([self.sensor_positions[s] for s in target_sensors])
        mean, std = self.predict_field(time, sensor_readings, positions, method='marginalize')
        
        # Ensure mean and std are 1D arrays (BetaLikelihood may return extra dimensions)
        mean = np.asarray(mean).flatten()
        std = np.asarray(std).flatten()
        
        return {s: (float(mean[i]), float(std[i])) for i, s in enumerate(target_sensors)}
    
    def detect_sensor_faults(self,
                            time: float,
                            sensor_readings: Dict[int, float],
                            threshold_sigma: float = 3.0) -> Dict[int, Dict]:
        """
        Detect anomalous sensor readings that don't fit the GP model.
        Useful for sensor fault detection.
        """
        # Predict at all observed sensors using leave-one-out
        anomalies = {}
        
        for test_sensor, observed_value in sensor_readings.items():
            # Leave this sensor out
            other_readings = {k: v for k, v in sensor_readings.items() if k != test_sensor}
            
            if len(other_readings) < 3:
                continue  # Not enough sensors for reliable prediction
            
            # Predict at left-out sensor location
            pred = self.predict_sensor_values(time, other_readings, [test_sensor])
            predicted_mean, predicted_std = pred[test_sensor]
            
            # Z-score
            residual = observed_value - predicted_mean
            z_score = abs(residual) / (predicted_std + 1e-6) # very simple sensor fault detection
            
            if z_score > threshold_sigma:
                anomalies[test_sensor] = {
                    'observed': observed_value,
                    'predicted': predicted_mean,
                    'predicted_std': predicted_std,
                    'residual': residual,
                    'z_score': z_score,
                    'severity': 'CRITICAL' if z_score > 5 else 'WARNING'
                }
        
        return anomalies
    
    def full_inference(self,
                      time: float,
                      sensor_readings: Dict[int, float],
                      grid_points: np.ndarray,
                      prediction_method: str = 'marginalize') -> InferenceResult:
        """
        Complete inference pipeline:
        1. Infer mass flow
        2. Predict field
        3. Identify danger zones
        4. Check sensor consistency
        
        Returns structured result for downstream safety systems.
        """
        start_time = time_module.time()
        
        # 1. Mass flow inference
        mf_info = self.infer_mass_flow_map(time, sensor_readings, method='hybrid')
        
        # 2. Field prediction
        field_mean, field_std = self.predict_field(
            time, sensor_readings, grid_points, method=prediction_method
        )
        
        # 3. Total uncertainty includes mass flow uncertainty
        total_uncertainty = field_std
        
        # 4. Danger zones
        danger_info = self.identify_danger_zones(field_mean, field_std, grid_points)
        
        # 5. Sensor predictions (for validation)
        all_sensors = list(self.sensor_positions.keys())
        sensor_preds = self.predict_sensor_values(time, sensor_readings, all_sensors)
        
        # 6. Fault detection
        anomalies = self.detect_sensor_faults(time, sensor_readings)
        
        inference_time = (time_module.time() - start_time) * 1000  # ms
        
        return InferenceResult(
            mass_flow_map=mf_info['map_estimate'],
            mass_flow_posterior_mean=mf_info['posterior_mean'],
            mass_flow_posterior_std=mf_info['posterior_std'],
            mass_flow_credible_interval=mf_info['credible_interval'],
            field_mean=field_mean,
            field_std=field_std,
            total_uncertainty=total_uncertainty,
            danger_zones=danger_info,
            sensor_predictions=sensor_preds,
            anomalies=anomalies,
            inference_time_ms=inference_time
        )
    
    def plot_mass_flow_posterior(self, result: InferenceResult, 
                                  true_mass_flow: Optional[float] = None,
                                  save_path: Optional[str] = None):
        """Plot mass flow posterior distribution."""

        # Recompute posterior grid for visualization
        mf_grid = np.linspace(0.01, 1.5, 100)
        log_liks = []
        
        for mf in mf_grid:
            # This is a placeholder - would need actual log likelihood computation
            # For now, use Gaussian approximation from result
            log_lik = -0.5 * ((mf - result.mass_flow_posterior_mean) / 
                             result.mass_flow_posterior_std)**2
            log_liks.append(log_lik)
        
        log_liks = np.array(log_liks)
        log_liks_norm = log_liks - np.max(log_liks)
        probs = np.exp(log_liks_norm)
        probs = probs / np.sum(probs)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.fill_between(mf_grid, probs, alpha=0.3, color='blue', label='Posterior')
        ax.plot(mf_grid, probs, 'b-', linewidth=2)
        ax.axvline(result.mass_flow_map, color='red', linestyle='--', 
                  linewidth=2, label=f'MAP = {result.mass_flow_map:.3f}')
        ax.axvspan(result.mass_flow_credible_interval[0], 
                  result.mass_flow_credible_interval[1], 
                  alpha=0.2, color='green', 
                  label=f'95% CI: [{result.mass_flow_credible_interval[0]:.3f}, {result.mass_flow_credible_interval[1]:.3f}]')
        
        if true_mass_flow:
            ax.axvline(true_mass_flow, color='black', linestyle='-',
                      linewidth=2, label=f'True = {true_mass_flow:.3f}')
        
        ax.set_xlabel('Mass Flow (kg/s)')
        ax.set_ylabel('Posterior Probability')
        ax.set_title('Mass Flow Inference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_concentration_slices(self, result: InferenceResult,
                                   grid_bounds: Dict,
                                   sensor_readings: Optional[Dict] = None,
                                   ground_truth_field: Optional[np.ndarray] = None,
                                   planes: List[str] = ['xy', 'xz'],
                                   resolution: Tuple[int, int, int] = (25, 25, 8),
                                   save_path: Optional[str] = None):
        """Plot 2D slices of concentration field with optional ground truth comparison.
        
        Args:
            result: InferenceResult with predictions
            grid_bounds: Dict with 'x', 'y', 'z' bounds
            sensor_readings: Observed sensor readings
            ground_truth_field: Optional ground truth concentration field for comparison
            planes: List of planes to plot ['xy', 'xz', 'yz']
            resolution: Grid resolution (nx, ny, nz)
            save_path: Optional path to save figure
        """        
        nx, ny, nz = resolution
        n_total = len(result.field_mean)
        
        # Use actual resolution if it matches, otherwise try to infer
        if n_total == nx * ny * nz:
            mean_3d = result.field_mean.reshape((nx, ny, nz))
            std_3d = result.field_std.reshape((nx, ny, nz))
            if ground_truth_field is not None:
                gt_3d = ground_truth_field.reshape((nx, ny, nz))
        else:
            # Fallback: try cubic-ish reshape
            try:
                nz = int(np.round(n_total ** (1/3)))
                nx = ny = nz
                mean_3d = result.field_mean.reshape((nx, ny, nz))
                std_3d = result.field_std.reshape((nx, ny, nz))
                if ground_truth_field is not None:
                    gt_3d = ground_truth_field.reshape((nx, ny, nz))
            except:
                # Last resort: flatten to 2D
                nx = int(np.sqrt(n_total))
                ny = n_total // nx
                mean_3d = result.field_mean[:nx*ny].reshape((nx, ny, 1))
                std_3d = result.field_std[:nx*ny].reshape((nx, ny, 1))
                if ground_truth_field is not None:
                    gt_3d = ground_truth_field[:nx*ny].reshape((nx, ny, 1))
                nz = 1
        
        # Determine number of columns: prediction + GT (if provided) + error (if GT)
        n_cols = len(planes)
        if ground_truth_field is not None:
            n_cols *= 3  # Pred, GT, Error for each plane
        
        fig, axes = plt.subplots(1 if ground_truth_field is None else 3, 
                                len(planes), 
                                figsize=(6*len(planes), 5*(1 if ground_truth_field is None else 3)))
        
        if len(planes) == 1:
            axes = np.atleast_1d(axes)
        if ground_truth_field is None:
            axes = axes.reshape(1, -1)
        
        slice_configs = {
            'xy': (mean_3d[:, :, nz//2], std_3d[:, :, nz//2], 
                   gt_3d[:, :, nz//2] if ground_truth_field is not None else None,
                   'x', 'y', (grid_bounds['x'], grid_bounds['y'])),
            'xz': (mean_3d[:, ny//2, :], std_3d[:, ny//2, :],
                   gt_3d[:, ny//2, :] if ground_truth_field is not None else None,
                   'x', 'z', (grid_bounds['x'], grid_bounds['z'])),
            'yz': (mean_3d[nx//2, :, :], std_3d[nx//2, :, :],
                   gt_3d[nx//2, :, :] if ground_truth_field is not None else None,
                   'y', 'z', (grid_bounds['y'], grid_bounds['z'])),
        }
        
        for col, plane in enumerate(planes):
            if plane not in slice_configs:
                continue
            
            mean_slice, std_slice, gt_slice, xlabel, ylabel, bounds = slice_configs[plane]
            
            x = np.linspace(*bounds[0], mean_slice.shape[0])
            y = np.linspace(*bounds[1], mean_slice.shape[1])
            xx, yy = np.meshgrid(x, y, indexing='ij')
            
            # Common vmax for consistent color scale
            if ground_truth_field is not None:
                vmax = max(mean_slice.max(), gt_slice.max(), 0.1)
            else:
                vmax = max(mean_slice.max(), 0.1)
            
            # Plot 1: Prediction
            ax_pred = axes[0, col] if ground_truth_field is not None else axes[0, col]
            im1 = ax_pred.contourf(xx, yy, mean_slice, levels=20, cmap='RdYlGn_r',
                                  vmin=0, vmax=vmax)
            ax_pred.contour(xx, yy, mean_slice, levels=[self.flammability_limit],
                           colors='red', linewidths=2, linestyles='--')
            
            # Plot sensors
            if sensor_readings:
                for sid, pos in self.sensor_positions.items():
                    if plane == 'xy' and abs(pos[2] - np.mean(grid_bounds['z'])) < 0.3:
                        color = 'lime' if sid in sensor_readings else 'gray'
                        ax_pred.plot(pos[0], pos[1], 'ko', markersize=6)
            
            ax_pred.set_xlabel(f'{xlabel} (m)')
            ax_pred.set_ylabel(f'{ylabel} (m)')
            title = f'{plane.upper()} Plane'
            if col == 0:
                title = f'Prediction\n{title}'
            ax_pred.set_title(title)
            plt.colorbar(im1, ax=ax_pred, label='H₂ Fraction', fraction=0.046)
            
            # Plot 2 & 3: Ground truth and error (if provided)
            if ground_truth_field is not None:
                # Ground truth
                ax_gt = axes[1, col]
                im2 = ax_gt.contourf(xx, yy, gt_slice, levels=20, cmap='RdYlGn_r',
                                    vmin=0, vmax=vmax)
                ax_gt.contour(xx, yy, gt_slice, levels=[self.flammability_limit],
                             colors='red', linewidths=2, linestyles='--')
                ax_gt.set_xlabel(f'{xlabel} (m)')
                ax_gt.set_ylabel(f'{ylabel} (m)')
                if col == 0:
                    ax_gt.set_title(f'Ground Truth\n{plane.upper()} Plane')
                else:
                    ax_gt.set_title(f'{plane.upper()} Plane')
                plt.colorbar(im2, ax=ax_gt, label='H₂ Fraction', fraction=0.046)
                
                # Error
                ax_err = axes[2, col]
                error = np.abs(mean_slice - gt_slice)
                im3 = ax_err.contourf(xx, yy, error, levels=20, cmap='Reds')
                ax_err.set_xlabel(f'{xlabel} (m)')
                ax_err.set_ylabel(f'{ylabel} (m)')
                if col == 0:
                    ax_err.set_title(f'Absolute Error\n{plane.upper()} Plane')
                else:
                    ax_err.set_title(f'{plane.upper()} Plane')
                plt.colorbar(im3, ax=ax_err, label='|Error|', fraction=0.046)
        
        plt.suptitle('Concentration Field: Prediction vs Ground Truth' 
                     if ground_truth_field is not None else 'Concentration Field Prediction',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_sensor_comparison(self, result: InferenceResult,
                                sensor_readings: Dict[int, float],
                                ground_truth: Optional[Dict[int, float]] = None,
                                save_path: Optional[str] = None):
        """Plot sensor predictions vs observations with optional ground truth comparison.
        
        Args:
            result: InferenceResult with predictions
            sensor_readings: Observed sensor readings (during operation)
            ground_truth: Optional ground truth values at all sensors (for validation)
            save_path: Optional path to save figure
        """
        # If ground truth provided, show all sensors, otherwise just observed
        if ground_truth is not None:
            sensor_ids = sorted(list(ground_truth.keys()))
            has_gt = True
        else:
            sensor_ids = list(sensor_readings.keys())
            has_gt = False
        
        observed = [sensor_readings.get(s, np.nan) for s in sensor_ids]
        predicted = [result.sensor_predictions[s][0] for s in sensor_ids]
        pred_stds = [result.sensor_predictions[s][1] for s in sensor_ids]
        
        if has_gt:
            gt_values = [ground_truth[s] for s in sensor_ids]
        
        fig, axes = plt.subplots(1, 2 if has_gt else 1, figsize=(12 if has_gt else 6, 6))
        if not has_gt:
            axes = [axes]
        
        # Plot 1: Bar chart comparison
        ax1 = axes[0]
        x = np.arange(len(sensor_ids))
        width = 0.35
        
        ax1.bar(x - width/2, observed, width, label='Observed', color='coral', alpha=0.8)
        ax1.bar(x + width/2, predicted, width,
                label='Predicted', color='steelblue', alpha=0.8)
        
        ax1.axhline(y=self.flammability_limit, color='red', linestyle='--',
                   linewidth=2, label=f'4% limit')
        
        ax1.set_xlabel('Sensor ID')
        ax1.set_ylabel('H₂ Volume Fraction')
        ax1.set_title('Sensor Values: Observation vs Prediction')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sensor_ids, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Scatter plot (if ground truth available)
        if has_gt:
            ax2 = axes[1]
            ax2.scatter(gt_values, predicted, c='blue', alpha=0.6, s=100, 
                       edgecolors='black', label='GP vs GT')
            
            # Perfect prediction line
            min_val = min(min(gt_values), min(predicted))
            max_val = max(max(gt_values), max(predicted))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', 
                    label='Perfect prediction', alpha=0.5)
            
            ax2.set_xlabel('Ground Truth H₂ Fraction')
            ax2.set_ylabel('Predicted H₂ Fraction')
            ax2.set_title('Prediction Accuracy vs Ground Truth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def select_inducing_points(X_train, n_inducing=1000, method='kmeans'):
    """
    Select inducing points using clustering or random sampling.
    
    Args:
        X_train: Training inputs [n, d] (numpy array)
        n_inducing: Number of inducing points
        method: 'kmeans' or 'random'
    
    Returns:
        Tensor of inducing points [n_inducing, d]
    """
    n_train = len(X_train)
    
    if method == 'kmeans' and n_train > n_inducing:
        print(f"Selecting {n_inducing} inducing points via K-means...")
        kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=10)
        kmeans.fit(X_train)
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    elif method == 'random':
        print(f"Randomly selecting {n_inducing} inducing points...")
        indices = np.random.choice(n_train, size=n_inducing, replace=False)
        inducing_points = torch.tensor(X_train[indices], dtype=torch.float32)
    else:
        # Use all data if smaller than n_inducing
        inducing_points = torch.tensor(X_train, dtype=torch.float32)
    
    # Remove any duplicate inducing points to avoid singular matrices
    inducing_points_np = inducing_points.numpy()
    inducing_points_np = np.unique(inducing_points_np, axis=0)
    if len(inducing_points_np) < len(inducing_points):
        print(f"  Removed {len(inducing_points) - len(inducing_points_np)} duplicate inducing points")
        inducing_points = torch.tensor(inducing_points_np, dtype=torch.float32)
    
    return inducing_points


def train_h2_dispersion_gp(df_train,
                           df_val=None,
                           val_split: float = 0.1,
                           n_epochs: int = 200,
                           learning_rate: float = 0.001,
                           n_inducing: int = 1000,
                           patience: int = 20,
                           val_freq: int = 10,
                           checkpoint_path: str = 'checkpoint_best.pth',
                           device: str = 'cpu',
                           sensor_positions: Optional[Dict] = None):
    """
    Train the Sparse GP model from training data using inducing points.
    
    Includes validation, early stopping, model checkpointing, and metrics tracking.
    If df_val is not provided, splits df_train into train/val using val_split ratio.
    
    Args:
        df_train: Training dataframe with columns time, mass_flow, y, z, h2_volume_fraction (x removed)
        df_val: Validation dataframe (optional - if None, splits from df_train)
        val_split: Fraction of df_train to use for validation (if df_val is None)
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        n_inducing: Number of inducing points (more = better approximation but slower)
        patience: Early stopping patience (epochs without improvement)
        val_freq: Validation frequency (epochs)
        checkpoint_path: Path to save best model checkpoint
        device: 'cpu' or 'cuda'
        sensor_positions: Dict of sensor positions for visualization (optional)
    
    Returns:
        model, likelihood, history: Trained model, likelihood, and training history
    """
    # Split train/val if validation not provided
    if df_val is None:
        # Split by scenarios to avoid data leakage
        scenarios = df_train['scenario'].unique()
        n_val = max(1, int(len(scenarios) * val_split))
        val_scenarios = np.random.choice(scenarios, size=n_val, replace=False)
        
        df_val = df_train[df_train['scenario'].isin(val_scenarios)].copy()
        df_train = df_train[~df_train['scenario'].isin(val_scenarios)].copy()
        
        print(f"Split train into {len(df_train):,} train / {len(df_val):,} val")
        print(f"Validation scenarios: {list(val_scenarios)}")
    
    # Prepare training data
    X = df_train[['time', 'mass_flow', 'y', 'z']].values
    y = df_train['h2_volume_fraction'].values
    
    X_train = torch.tensor(X, dtype=torch.float32, device=device)
    y_train = torch.tensor(y, dtype=torch.float32, device=device)
    
    print(f"Training data: {len(X_train):,} points")
    
    # Prepare validation data
    has_validation = len(df_val) > 0
    if has_validation:
        X_val_np = df_val[['time', 'mass_flow', 'y', 'z']].values
        y_val_np = df_val['h2_volume_fraction'].values
        X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
        print(f"Validation data: {len(X_val):,} points")
    else:
        print("No validation data available - early stopping disabled")
    
    # Select inducing points
    inducing_points = select_inducing_points(
        X, 
        n_inducing=n_inducing,
        method='random'
    ).to(device)
    
    print(f"Inducing points: {inducing_points.shape}")
    
    # Initialize sparse GP model (use double precision for numerical stability)
    model = SparseH2DispersionGP(
        inducing_points=inducing_points.double(),
        learn_inducing=True  # Optimize inducing point locations
    ).double().to(device)
    
    # Convert training data to double precision
    X_train = X_train.double()
    y_train = y_train.double()
    if has_validation:
        X_val = X_val.double()
        y_val = y_val.double()
    
    """likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-3)  # Minimum 0.001 noise
    ).to(device)"""
    
    likelihood = gpytorch.likelihoods.BetaLikelihood(
        scale_prior=gpytorch.priors.GammaPrior(3.0, 0.5)
    ).double().to(device)
    
    # Training
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)
    
    # Use Variational ELBO instead of ExactMarginalLogLikelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train), beta=1.0)
    
    # Tracking
    history = {
        'train_loss': [],
        'val_loss': [] if has_validation else None,
        'epochs': [],
        'best_epoch': 0,
        'best_val_loss': float('inf') if has_validation else None
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Data validation
    assert torch.isfinite(X_train).all(), "NaN/Inf in training features"
    assert torch.isfinite(y_train).all(), "NaN/Inf in training targets"
    
    print(f"\nTraining for up to {n_epochs} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Validation frequency: every {val_freq} epochs")
    print("-" * 60)
    
    start_time = time_module.time()
    
    # Use numerical stability settings for entire training
    with gpytorch.settings.cholesky_jitter(1e-1), \
         gpytorch.settings.fast_computations(solves=True):
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            history['train_loss'].append(train_loss)
            history['epochs'].append(epoch + 1)
            
            # Validation
            if has_validation and (epoch + 1) % val_freq == 0:
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val)
                    val_loss = -mll(val_output, y_val).item()
                model.train()
                
                history['val_loss'].append(val_loss)
                
                # Print progress
                print(f"Epoch {epoch+1:3d}/{n_epochs}: Train ELBO={train_loss:10.2f}, Val ELBO={val_loss:10.2f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    history['best_epoch'] = epoch + 1
                    history['best_val_loss'] = best_val_loss
                    
                    # Save checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'likelihood_state_dict': likelihood.state_dict(),
                        'inducing_points': model.variational_strategy.inducing_points.detach(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'history': history,
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  -> New best! Checkpoint saved to {checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"  -> No improvement ({patience_counter}/{patience})")
                    
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                        break
            elif (epoch + 1) % val_freq == 0:
                # Print train-only progress (no validation)
                print(f"Epoch {epoch+1:3d}/{n_epochs}: Train ELBO={train_loss:10.2f}")
    
    training_time = time_module.time() - start_time
    
    # Load best model if validation was used
    if has_validation and Path(checkpoint_path).exists():
        print(f"\nLoading best model from epoch {history['best_epoch']}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        # Final validation visualization with best model
        if sensor_positions is not None:
            try:
                print("\nGenerating final validation visualization...")
                
                # Create visualization model on CPU
                viz_model = SparseH2DispersionGP(
                    inducing_points=model.variational_strategy.inducing_points.detach().clone(),
                    learn_inducing=False
                ).double().to('cpu')
                viz_model.load_state_dict(model.state_dict())
                viz_model.eval()
                
                viz_likelihood = gpytorch.likelihoods.BetaLikelihood(
                    scale_prior=gpytorch.priors.GammaPrior(3.0, 0.5)
                ).double().to('cpu')
                viz_likelihood.load_state_dict(likelihood.state_dict())
                viz_likelihood.eval()
                
                # Create predictor
                temp_predictor = LatentMassFlowPredictor(
                    viz_model, viz_likelihood, sensor_positions, device='cpu'
                )
                
                # Sample validation data
                val_scenarios = df_val['scenario'].unique()
                val_scenario = np.random.choice(val_scenarios)
                val_data_scenario = df_val[df_val['scenario'] == val_scenario]
                
                val_times = val_data_scenario['time'].unique()
                val_time = np.random.choice(val_times)
                val_mass_flow = val_data_scenario[val_data_scenario['time'] == val_time]['mass_flow'].iloc[0]
                
                val_at_time = val_data_scenario[val_data_scenario['time'] == val_time]
                
                if len(val_at_time) >= 3:
                    sensor_readings_viz = dict(zip(
                        val_at_time['sensor_id'].values,
                        val_at_time['h2_volume_fraction'].values
                    ))
                    
                    viz_grid = val_at_time[['x', 'y', 'z']].values
                    
                    result_viz = temp_predictor.full_inference(
                        time=val_time,
                        sensor_readings=sensor_readings_viz,
                        grid_points=viz_grid,
                        prediction_method='map'
                    )
                    
                    gt_sensors = sensor_readings_viz
                    
                    viz_dir = Path('validation_viz')
                    viz_dir.mkdir(exist_ok=True)
                    
                    temp_predictor.plot_sensor_comparison(
                        result_viz, sensor_readings_viz,
                        ground_truth=gt_sensors,
                        save_path=viz_dir / 'final_best_model_sensors.png'
                    )
                    plt.close()
                    
                    print(f"  Final validation plot saved to {viz_dir}/final_best_model_sensors.png")
                
                del viz_model, viz_likelihood, temp_predictor
                
            except Exception as e:
                import traceback
                print(f"  Warning: Could not generate final validation plot: {e}")
                traceback.print_exc()
    
    print(f"\nTraining completed in {training_time:.1f}s")
    print("\nLearned hyperparameters:")
    print(f"  Outputscale (time): {model.time_kernel.outputscale.item():.4f}")
    print(f"  Lengthscale (time): {model.time_kernel.base_kernel.lengthscale.item():.4f}")
    print(f"  Outputscale (space_y): {model.space_kernel_y.outputscale.item():.4f}")
    print(f"  Lengthscale (space_y): {model.space_kernel_y.base_kernel.lengthscale.detach().cpu().numpy()}")
    print(f"  Outputscale (space_z): {model.space_kernel_z.outputscale.item():.4f}")
    print(f"  Lengthscale (space_z): {model.space_kernel_z.base_kernel.lengthscale.detach().cpu().numpy()}")
    print(f"  Noise: {likelihood.noise.item():.6f}")
    
    if has_validation:
        print(f"\nBest validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    
    return model, likelihood, history


def evaluate_model(model, likelihood, df_test):
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained GP model
        likelihood: GP likelihood
        df_test: Test dataframe
    
    Returns:
        Dictionary of metrics
    """
    # Get the device from the model to ensure consistency
    model_device = next(model.parameters()).device
    
    X_test = torch.tensor(df_test[['time', 'mass_flow', 'y', 'z']].values, 
                          dtype=torch.float64, device=model_device)
    y_test = torch.tensor(df_test['h2_volume_fraction'].values, 
                          dtype=torch.float64, device=model_device)
    
    # Ensure model and likelihood are on the same device
    model = model.to(model_device)
    likelihood = likelihood.to(model_device)
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_test))
        pred_mean = pred.mean
        pred_std = pred.stddev
    
    # Metrics
    mae = torch.mean(torch.abs(pred_mean - y_test)).item()
    rmse = torch.sqrt(torch.mean((pred_mean - y_test)**2)).item()
    mape = torch.mean(torch.abs((pred_mean - y_test) / (y_test + 1e-6))).item() * 100
    
    # Uncertainty calibration
    residuals = torch.abs(pred_mean - y_test)
    within_1std = torch.mean((residuals < pred_std).float()).item()
    within_2std = torch.mean((residuals < 2 * pred_std).float()).item()
    
    # Metrics by source
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'within_1std': within_1std,
        'within_2std': within_2std,
    }
    
    # Add source-specific metrics if available
    if 'source' in df_test.columns:
        for source in df_test['source'].unique():
            mask = df_test['source'] == source
            source_mae = torch.mean(torch.abs(pred_mean[mask.values] - y_test[mask.values])).item()
            metrics[f'{source}_mae'] = source_mae
    
    return metrics


def save_model(model, likelihood, history, metrics, output_path, scaler=None):
    """
    Save trained model, metrics, and history.
    
    Args:
        model: Trained GP model
        likelihood: GP likelihood
        history: Training history dict
        metrics: Test metrics dict
        output_path: Path to save model
        scaler: Optional scaler for preprocessing
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': model.variational_strategy.inducing_points.detach(),
        'history': history,
        'metrics': metrics,
        'scaler': scaler,
    }
    torch.save(checkpoint, output_path)
    
    # Save metrics as JSON
    metrics_path = output_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save history as JSON
    history_path = output_path.parent / (output_path.stem + '_history.json')
    with open(history_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        history_json = {k: v for k, v in history.items() if v is not None}
        json.dump(history_json, f, indent=2)
    
    print(f"\nModel saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"History saved to {history_path}")


def load_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: 'cpu' or 'cuda'
    
    Returns:
        model, likelihood, checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    inducing_points = checkpoint['inducing_points'].to(device)
    
    model = SparseH2DispersionGP(
        inducing_points=inducing_points,
        learn_inducing=False  # Don't optimize when loading
    ).double().to(device)
    
    likelihood = gpytorch.likelihoods.BetaLikelihood(
        scale_prior=gpytorch.priors.GammaPrior(3.0, 0.5)
    ).double().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    model.eval()
    likelihood.eval()
    
    return model, likelihood, checkpoint


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv('/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model/data/unified_raw.csv')
    
    # Sensor positions (should match your data)
    SENSOR_POSITIONS = {
        1: (0.77, 0.24, 0.8), 2: (0.46, 0.0, 0.8), 3: (0.48, 0.25, 0.52),
        4: (0.13, 0.26, 0.8), 5: (0.48, 1.33, 0.52), 6: (0.46, 1.11, 0.8),
        7: (0.9, 2.22, 0.8), 8: (0.47, 2.46, 0.52), 9: (0.46, 2.22, 0.8),
        10: (0.74, 2.45, 0.8), 11: (0.15, 2.44, 0.8), 12: (0.00, 2.22, 0.8),
        13: (0.47, 3.54, 0.52), 14: (0.46, 3.31, 0.8), 15: (0.78, 4.65, 0.8),
        17: (0.46, 4.39, 0.8), 18: (0.47, 4.63, 0.52), 19: (0.14, 4.68, 0.52),
        20: (0.75, 0.24, 0.0), 21: (0.45, 0.0, 0.0), 22: (0.16, 0.24, 0.0),
        23: (0.47, 0.24, 0.28), 24: (0.46, 1.33, 0.27), 25: (0.47, 1.1, 0.0),
        26: (0.46, 2.46, 0.27), 27: (0.47, 2.23, 0.0), 28: (0.47, 3.56, 0.27),
        29: (0.47, 4.63, 0.27), 30: (0.46, 4.44, 0.0),
    }
    
    # Split data - only train and test available
    df_train_full = df[df['split'] == 'train'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    print(f"Train: {len(df_train_full):,} rows, Test: {len(df_test):,} rows")
    
    # Train (or load pretrained model)
    print("\nTraining GP model with train/val split and early stopping...")
    model, likelihood, history = train_h2_dispersion_gp(
        df_train_full, 
        df_val=None,           # Will split train into train/val automatically
        val_split=0.2,         # Use 20% of train scenarios for validation
        n_epochs=500, 
        n_inducing=1000,
        patience=30,
        val_freq=20,
        checkpoint_path='checkpoint_best.pth',
        device='cuda',
        sensor_positions=SENSOR_POSITIONS  # For validation visualization
    )
    
    # Evaluate on test set (auto-detects device from model)
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, likelihood, df_test)
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test 1σ coverage: {metrics['within_1std']:.1%}")
    
    # Save model
    save_model(model, likelihood, history, metrics, 'models/h2_dispersion_sparse.pth')
    
    # Create predictor (auto-detect device from model)
    model_device = next(model.parameters()).device
    predictor = LatentMassFlowPredictor(model, likelihood, SENSOR_POSITIONS, device=str(model_device))
    
    # Simulate runtime operation
    print("\n" + "="*60)
    print("RUNTIME INFERENCE EXAMPLE")
    print("="*60)
    
    # Simulate sensor readings at t=60s from a leak with true mass_flow ~0.48 kg/s
    runtime_sensor_readings = {
        1: 0.08,    # High concentration near source
        2: 0.05,
        5: 0.12,    # Downstream sensor
        8: 0.03,
        15: 0.01,   # Far sensor, low concentration
        23: 0.06,
    }
    current_time = 60.0
    
    # Create prediction grid
    grid_x = np.linspace(0, 1, 25)
    grid_y = np.linspace(0, 5, 25)
    grid_z = np.linspace(0, 1, 8)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Full inference
    result = predictor.full_inference(
        time=current_time,
        sensor_readings=runtime_sensor_readings,
        grid_points=grid_points,
        prediction_method='mcmc'
    )
    
    # Print results
    print(f"\nMass Flow Inference:")
    print(f"  MAP Estimate: {result.mass_flow_map:.3f} kg/s")
    print(f"  Posterior Mean ± Std: {result.mass_flow_posterior_mean:.3f} ± {result.mass_flow_posterior_std:.3f} kg/s")
    print(f"  95% Credible Interval: [{result.mass_flow_credible_interval[0]:.3f}, {result.mass_flow_credible_interval[1]:.3f}] kg/s")
    
    print(f"\nConcentration Field:")
    print(f"  Max Predicted: {result.field_mean.max():.4f}")
    print(f"  Uncertainty at Max: {result.field_std[np.argmax(result.field_mean)]:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    grid_bounds = {'x': (0, 1), 'y': (0, 5), 'z': (0, 1)}
    
    # Plot 1: Mass flow posterior
    #fig1 = predictor.plot_mass_flow_posterior(result, save_path='viz_mass_flow.png')
    
    # Plot 2: Concentration slices (without ground truth - operational mode)
    fig2 = predictor.plot_concentration_slices(
        result, grid_bounds, runtime_sensor_readings, 
        planes=['xy', 'xz'], resolution=(25, 25, 8), save_path='viz_slices.png'
    )
    
    # Example with ground truth (for validation/testing only):
    # If you have ground truth data from CFD/experiments:
    # ground_truth = df_test[(df_test['time'] == current_time) & 
    #                        (df_test['mass_flow'] == true_mass_flow)]['h2_volume_fraction'].values
    # fig2_with_gt = predictor.plot_concentration_slices(
    #     result, grid_bounds, runtime_sensor_readings,
    #     ground_truth_field=ground_truth,
    #     planes=['xy', 'xz'], resolution=(25, 25, 8), save_path='viz_slices_with_gt.png'
    # )
    
    # Plot 3: Sensor comparison (operational mode)
    fig3 = predictor.plot_sensor_comparison(
        result, runtime_sensor_readings, save_path='viz_sensors.png'
    )
    
    # Example with ground truth (for validation):
    # ground_truth_sensors = {1: 0.085, 2: 0.052, 5: 0.125, ...}  # From CFD/experiment
    # fig3_with_gt = predictor.plot_sensor_comparison(
    #     result, runtime_sensor_readings, 
    #     ground_truth=ground_truth_sensors,
    #     save_path='viz_sensors_with_gt.png'
    # )
    
    print("Saved: viz_mass_flow.png, viz_slices.png, viz_sensors.png")
    
    print(f"\nSafety Assessment:")
    danger = result.danger_zones
    if danger['has_danger']:
        print(f"  ⚠️  DANGER DETECTED")
        print(f"  Danger Volume: {danger['danger_volume_m3']:.2f} m³")
        print(f"  Max Concentration: {danger['max_concentration']:.4f}")
    else:
        print(f"  ✓ No danger zones (below 4% limit)")
    
    print(f"\nSensor Fault Detection:")
    anomalies = predictor.detect_sensor_faults(current_time, runtime_sensor_readings)
    if anomalies:
        for sensor, info in anomalies.items():
            print(f"  ⚠️  Sensor {sensor}: {info['severity']} (z={info['z_score']:.1f}σ)")
    else:
        print(f"  ✓ All sensors consistent")
    
    print(f"\nInference Time: {result.inference_time_ms:.1f} ms")
