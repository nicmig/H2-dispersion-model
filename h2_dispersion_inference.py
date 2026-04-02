"""
H2 Dispersion Inference Module

Operational inference using trained GP models.
Loads trained models and performs mass flow inference and concentration prediction.
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# Log transform epsilon to avoid log(0)
LOG_EPSILON = 1e-6


@dataclass
class InferenceResult:
    """Container for inference results"""
    mass_flow_map: float                    # MAP estimate
    mass_flow_posterior_mean: float         # Posterior mean
    mass_flow_posterior_std: float          # Posterior uncertainty
    mass_flow_credible_interval: Tuple[float, float]  # 95% CI
    field_mean: np.ndarray                  # Mean concentration field
    field_std: np.ndarray                   # Std deviation field
    posterior_grid: Tuple
    log_likelihoods: np.ndarray
    total_uncertainty: np.ndarray           # Combined uncertainty
    inference_time_ms: float                # Computational time


class H2DispersionInference:
    """
    Operational inference class for H2 dispersion.
    
    Loads trained GP models and performs:
    - Mass flow inference from sensor readings
    - Concentration field prediction
    - Danger zone identification
    - Sensor fault detection
    """
    
    def __init__(self, 
                 model,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 sensor_positions: Dict[int, Tuple[float, float, float]],
                 flammability_limit: float = 0.04,
                 device: str = 'cpu',
                 log_epsilon: float = LOG_EPSILON):
        """
        Args:
            model: Trained GP model (SparseH2DispersionGP, ExactGPModel, etc.)
            likelihood: GP likelihood
            sensor_positions: Dict mapping sensor_id -> (x, y, z)
            flammability_limit: H2 flammability limit (default 4%)
            device: 'cpu' or 'cuda'
            log_epsilon: Epsilon for log transform
        """
        self.device = device
        self.sensor_positions = sensor_positions
        self.log_epsilon = log_epsilon
        self.flammability_limit = flammability_limit
        
        # Move model to device and set to eval mode
        self.gp = model.to(device)
        self.likelihood = likelihood.to(device)
        
        self.gp.eval()
        self.likelihood.eval()
        
        # Cache for repeated queries
        self._cache = {}
        
        # Mass flow prior
        self.mass_flow_prior_mean = 0.5
        self.mass_flow_prior_std = 0.4
        self.mass_flow_bounds = (0.01, 2.0)
        
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
        Compute log p(y_obs | mass_flow, time, y, z) using the GP.
        """
        try:
            # Build input: [time, mass_flow, y, z] for each observed sensor
            X_obs = []
            for s in observed_sensors:
                _, y, z = self.sensor_positions[s]
                X_obs.append([time, mass_flow, y, z])
            
            X_obs = torch.tensor(X_obs, dtype=torch.float64, device=self.device)
            y_obs_log = np.log(observed_values + self.log_epsilon)
            y_obs = torch.tensor(y_obs_log, dtype=torch.float64, device=self.device)
            
            with torch.no_grad():
                log_lik = self.likelihood.log_marginal(y_obs, self.gp(X_obs))
                
            # Add log prior
            log_prior = -0.5 * ((mass_flow - self.mass_flow_prior_mean) / self.mass_flow_prior_std)**2
            
            return log_lik.item() + log_prior
            
        except Exception as e:
            return -1e10
    
    def infer_mass_flow(self,
                        time: float,
                        sensor_readings: Dict[int, float],
                        method: str = 'mcmc',
                        n_samples: int = 100) -> Dict:
        """
        Infer mass flow from sensor observations.
        
        Args:
            time: Current time
            sensor_readings: Dict of {sensor_id: h2_concentration}
            method: 'map', 'mcmc', or 'laplace'
            n_samples: Number of MCMC samples (if method='mcmc')
        
        Returns:
            Dictionary with inference results
        """
        import time as time_module
        start_time = time_module.time()
        
        observed_sensors = list(sensor_readings.keys())
        observed_values = np.array([sensor_readings[s] for s in observed_sensors])
        
        # Create grid for posterior evaluation
        mass_flow_grid = np.linspace(self.mass_flow_bounds[0], 
                                     self.mass_flow_bounds[1], 100)
        
        if method == 'map':
            # MAP estimation via optimization
            result = minimize_scalar(
                lambda m: -self._compute_log_likelihood(m, time, observed_sensors, observed_values),
                bounds=self.mass_flow_bounds,
                method='bounded'
            )
            
            map_estimate = result.x
            
            # Approximate posterior std using Laplace approximation
            epsilon = 0.01
            log_p_map = -result.fun
            log_p_plus = -self._compute_log_likelihood(map_estimate + epsilon, time, 
                                                       observed_sensors, observed_values)
            log_p_minus = -self._compute_log_likelihood(map_estimate - epsilon, time,
                                                        observed_sensors, observed_values)
            
            second_deriv = (log_p_plus - 2*log_p_map + log_p_minus) / epsilon**2
            posterior_std = 1.0 / np.sqrt(max(-second_deriv, 1e-6))
            
            result_dict = {
                'map_estimate': map_estimate,
                'posterior_mean': map_estimate,
                'posterior_std': posterior_std,
                'credible_interval': (map_estimate - 1.96*posterior_std,
                                     map_estimate + 1.96*posterior_std),
                'log_likelihood': log_p_map
            }
            
        elif method == 'mcmc':
            # Simple Metropolis-Hastings MCMC
            samples = self._mcmc_sampler(time, observed_sensors, observed_values, n_samples)
            
            result_dict = {
                'map_estimate': float(samples.mean()),  # Approximation
                'posterior_mean': float(samples.mean()),
                'posterior_std': float(samples.std()),
                'credible_interval': (float(np.percentile(samples, 2.5)),
                                     float(np.percentile(samples, 97.5))),
                'samples': samples
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result_dict['inference_time_ms'] = (time_module.time() - start_time) * 1000
        return result_dict
    
    def _mcmc_sampler(self, time, observed_sensors, observed_values, n_samples=100):
        """Simple Metropolis-Hastings sampler for mass flow posterior."""
        samples = []
        current_m = self.mass_flow_prior_mean
        current_log_p = self._compute_log_likelihood(current_m, time, 
                                                      observed_sensors, observed_values)
        
        proposal_std = 0.1
        n_accepted = 0
        
        for i in range(n_samples + 500):  # Burn-in
            proposal = current_m + np.random.normal(0, proposal_std)
            proposal = np.clip(proposal, *self.mass_flow_bounds)
            
            proposal_log_p = self._compute_log_likelihood(proposal, time,
                                                          observed_sensors, observed_values)
            
            log_alpha = proposal_log_p - current_log_p
            if np.log(np.random.uniform()) < log_alpha:
                current_m = proposal
                current_log_p = proposal_log_p
                n_accepted += 1
            
            if i >= 500:  # After burn-in
                samples.append(current_m)
        
        return np.array(samples)
    
    def predict_field(self,
                      time: float,
                      mass_flow: float,
                      grid_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict concentration field at grid points.
        
        Args:
            time: Time value
            mass_flow: Mass flow value
            grid_points: Array of [y, z] coordinates (N x 2)
        
        Returns:
            (mean, std) arrays of shape (N,)
        """
        # Build input: [time, mass_flow, y, z] for each grid point
        X_grid = np.column_stack([
            np.full(len(grid_points), time),
            np.full(len(grid_points), mass_flow),
            grid_points[:, 0],  # y
            grid_points[:, 1]   # z
        ])
        
        X_grid_t = torch.tensor(X_grid, dtype=torch.float64, device=self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(X_grid_t))
            mean_log = pred.mean.cpu().numpy()
            std_log = pred.stddev.cpu().numpy()
        
        # Convert from log space
        mean_original = np.exp(mean_log) - self.log_epsilon
        
        return mean_original, std_log
    
    def full_inference(self,
                       time: float,
                       sensor_readings: Dict[int, float],
                       grid_points: np.ndarray,
                       prediction_method: str = 'marginalize') -> InferenceResult:
        """
        Full inference pipeline.
        
        Args:
            time: Current time
            sensor_readings: Dict of {sensor_id: h2_concentration}
            grid_points: Grid for field prediction (N x 2)
            prediction_method: 'map', 'marginalize', or 'mcmc'
        
        Returns:
            InferenceResult with all predictions
        """
        import time as time_module
        start_time = time_module.time()
        
        # Infer mass flow
        mass_flow_result = self.infer_mass_flow(time, sensor_readings, 
                                                 method='mcmc' if prediction_method == 'mcmc' else 'map')
        
        # Predict field
        if prediction_method == 'map':
            field_mean, field_std = self.predict_field(
                time, mass_flow_result['map_estimate'], grid_points
            )
            total_unc = field_std
            
        elif prediction_method == 'marginalize':
            # Integrate over mass flow uncertainty
            mass_flow_samples = np.random.normal(
                mass_flow_result['posterior_mean'],
                mass_flow_result['posterior_std'],
                20
 )
            
            field_means = []
            for m in mass_flow_samples:
                m = np.clip(m, *self.mass_flow_bounds)
                fm, _ = self.predict_field(time, m, grid_points)
                field_means.append(fm)
            
            field_means = np.array(field_means)
            field_mean = field_means.mean(axis=0)
            epistemic_var = field_means.var(axis=0)
            
            _, aleatoric_std = self.predict_field(time, mass_flow_result['map_estimate'], grid_points)
            total_unc = np.sqrt(epistemic_var + aleatoric_std**2)
            
        else:
            raise ValueError(f"Unknown prediction_method: {prediction_method}")
        
        inference_time = (time_module.time() - start_time) * 1000
        
        return InferenceResult(
            mass_flow_map=mass_flow_result['map_estimate'],
            mass_flow_posterior_mean=mass_flow_result['posterior_mean'],
            mass_flow_posterior_std=mass_flow_result['posterior_std'],
            mass_flow_credible_interval=mass_flow_result['credible_interval'],
            field_mean=field_mean,
            field_std=field_std,
            posterior_grid=(grid_points[:, 0], grid_points[:, 1]),
            log_likelihoods=np.array([]),  # Could store if needed
            total_uncertainty=total_unc,
            inference_time_ms=inference_time
        )
    
    def detect_sensor_faults(self,
                            time: float,
                            sensor_readings: Dict[int, float],
                            threshold_sigma: float = 3.0) -> Dict[int, Dict]:
        """
        Detect inconsistent sensors using leave-one-out cross-validation.
        
        Args:
            time: Current time
            sensor_readings: Dict of {sensor_id: h2_concentration}
            threshold_sigma: Z-score threshold for flagging faults
        
        Returns:
            Dict of faulty sensors with diagnostic info
        """
        anomalies = {}
        all_sensors = list(sensor_readings.keys())
        
        for test_sensor in all_sensors:
            # Use all other sensors to predict this one
            other_sensors = [s for s in all_sensors if s != test_sensor]
            if len(other_sensors) < 2:
                continue
                
            other_readings = {s: sensor_readings[s] for s in other_sensors}
            
            # Infer mass flow without this sensor
            mf_result = self.infer_mass_flow(time, other_readings, method='map')
            
            # Predict at test sensor location
            _, y_test, z_test = self.sensor_positions[test_sensor]
            test_point = np.array([[y_test, z_test]])
            
            pred_mean, pred_std = self.predict_field(
                time, mf_result['map_estimate'], test_point
            )
            
            observed = sensor_readings[test_sensor]
            z_score = (observed - pred_mean[0]) / (pred_std[0] + 1e-6)
            
            if abs(z_score) > threshold_sigma:
                anomalies[test_sensor] = {
                    'observed': observed,
                    'predicted': pred_mean[0],
                    'predicted_std': pred_std[0],
                    'z_score': z_score,
                    'severity': 'CRITICAL' if abs(z_score) > 5 else 'WARNING'
                }
        
        return anomalies
    
    def plot_sensor_comparison(self, time, sensor_readings, save_path=None):
        """Plot comparison of predicted vs observed sensor values."""
        # Placeholder for plotting functionality
        pass


def load_inference_model(checkpoint_path: str, 
                         sensor_positions: Dict,
                         device: str = 'cpu') -> H2DispersionInference:
    """
    Load trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        sensor_positions: Sensor positions dict
        device: 'cpu' or 'cuda'
    
    Returns:
        H2DispersionInference instance ready for inference
    """
    from h2_dispersion_gp import SparseH2DispersionGP, ExactGPModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type from checkpoint
    model_type = checkpoint.get('model_type', 'exact_gp')
    
    # Reconstruct model based on type
    # This is simplified - would need to handle different model types properly
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    
    # For now, create placeholder - in practice would need to reconstruct properly
    if model_type == 'approximate_gp':
        # Would need inducing points from checkpoint
        inducing_points = checkpoint['model_state_dict'].get('variational_strategy.inducing_points')
        model = SparseH2DispersionGP(inducing_points=inducing_points).double().to(device)
    else:
        # Exact GP - need training data or would need to save differently
        raise NotImplementedError("Loading exact GP requires saving training data or implementing differently")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    model.eval()
    likelihood.eval()
    
    return H2DispersionInference(model, likelihood, sensor_positions, device=device)
