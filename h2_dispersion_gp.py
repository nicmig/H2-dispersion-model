"""
GP for H2 Dispersion Prediction

During operation:
- INPUT: time, sensor observations {sensor_id: h2_concentration}
- OUTPUT: concentration field prediction + uncertainty

Training:
- INPUT: time, mass_flow, y, z (from CFD + experiments)
- OUTPUT: h2_concentration
"""
"""import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'"""

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from gpytorch.distributions import MultivariateNormal
import torch
import gpytorch
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy
from gpytorch.constraints import Interval
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time as time_module
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

from scipy.cluster.vq import kmeans2
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Import additive kernels
from additive_kernels import ScaleAdditiveKernel, FullAdditiveKernel

# Epsilon for log-transform to avoid log(0)
LOG_EPSILON = 1e-4


class ExperimentLogger:
    """Logger for tracking experiment runs and their key information."""
    
    def __init__(self, log_dir: str = 'experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'experiment_{self.experiment_id}.json'
        self.entries = []
        
    def log(self, entry: Dict):
        """Add an entry to the log."""
        entry['timestamp'] = datetime.now().isoformat()
        entry['experiment_id'] = self.experiment_id
        self.entries.append(entry)
        self._save()
        
    def log_training_start(self, config: Dict, model_config: Optional[Dict] = None):
        """Log training configuration at start.
        
        Args:
            config: Training hyperparameters (epochs, lr, etc.)
            model_config: Model architecture config (likelihood, kernels, loss)
        """
        entry = {
            'event': 'training_start',
            'config': config
        }
        if model_config is not None:
            entry['model_config'] = model_config
        self.log(entry)
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Log epoch metrics."""
        entry = {
            'event': 'epoch',
            'epoch': epoch,
            'train_loss': train_loss
        }
        if val_loss is not None:
            entry['val_loss'] = val_loss
        self.log(entry)
        
    def log_training_end(self, best_epoch: int, best_val_loss: float, total_time: float):
        """Log training completion."""
        self.log({
            'event': 'training_end',
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_time_seconds': total_time
        })
        
    def log_evaluation(self, metrics: Dict, dataset: str = 'test'):
        """Log evaluation metrics."""
        self.log({
            'event': 'evaluation',
            'dataset': dataset,
            'metrics': metrics
        })
        
    def _save(self):
        """Save log to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.entries, f, indent=2)
            
    def get_log_path(self) -> Path:
        """Get path to log file."""
        return self.log_file


class SparseH2DispersionGP(gpytorch.models.ApproximateGP):
    """
    Sparse Variational GP for H2 dispersion.
    
    Uses inducing points to approximate the full GP posterior.
    Scales to large datasets (O(m²n) instead of O(n³)).
    
    GP model: f(time, mass_flow, y, z) -> h2_concentration
    """
    
    def __init__(self, inducing_points):
        """
        Args:
            inducing_points: Initial inducing point locations [n_inducing, n_features]
            learn_inducing: Whether to optimize inducing point locations
        """
        # Variational distribution q(u)
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        
        # Variational strategy
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            inducing_points, 
            variational_dist,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # additive kernel
        self.covar_module = FullAdditiveKernel(base_kernel_type='rbf', num_dims=4)
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[0])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[1])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[2])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[0]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[1])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[0]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[2])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[0]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[3])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[1]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[2])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[1]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[3])) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[2]))*gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(active_dims=[3]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class VNNGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256, lengthscale_constraints=None):

        m, d = inducing_points.shape
        self.m = m
        self.k = k

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)

        variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution, k=k, training_batch_size=training_batch_size, jitter_val=0.0001)

        super(VNNGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = FullAdditiveKernel(base_kernel_type='rbf', num_dims=4)
        self.covar_module = ScaleAdditiveKernel(
            base_kernel_type='rbf', num_dims=5,
            lengthscale_constraints=lengthscale_constraints
        ) # less output scale parameters and more interpretable

        self.likelihood = likelihood
    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x: torch.Tensor | None, prior: bool = False, **kwargs) -> MultivariateNormal:
        if x is not None:
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=False, **kwargs)


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
        centroid, _ = kmeans2(X_train, k=n_inducing, iter=10, minit='random')
        inducing_points = torch.tensor(centroid, dtype=torch.float64)
    elif method == 'random':
        print(f"Randomly selecting {n_inducing} inducing points...")
        indices = np.random.choice(n_train, size=n_inducing, replace=False)
        inducing_points = torch.tensor(X_train[indices], dtype=torch.float64)
    else:
        # Use all data if smaller than n_inducing
        inducing_points = torch.tensor(X_train, dtype=torch.float64)
    
    # Remove any duplicate inducing points to avoid singular matrices
    inducing_points_np = inducing_points.numpy()
    inducing_points_np = np.unique(inducing_points_np, axis=0)
    if len(inducing_points_np) < len(inducing_points):
        print(f"  Removed {len(inducing_points) - len(inducing_points_np)} duplicate inducing points")
        inducing_points = torch.tensor(inducing_points_np, dtype=torch.float64)
    
    return inducing_points


def train_h2_dispersion_gp(df_train,
                           n_epochs: int = 200,
                           learning_rate: float = 0.005,
                           device: str = 'cpu',
                           logger: Optional[ExperimentLogger] = None,
                           model_path: Optional[str] = None):
    """
    Train Exact GP model on training data only.
    
    Standard GP training without validation during training. GPs typically don't
    overfit, so we train for fixed epochs and evaluate at the end.
    
    Args:
        df_train: Training dataframe with columns time, mass_flow, y, z, h2_volume_fraction
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: 'cpu' or 'cuda'
        logger: Optional ExperimentLogger for tracking experiment
        model_path: Optional path to save model (e.g., 'models/exact_gp.pth')
    
    Returns:
        model, likelihood, history: Trained model, likelihood, and training history
    """
    # Prepare training data
    X = df_train[['time', 'mass_flow', 'y', 'z']].values
    y = df_train['h2_volume_fraction'].values
    
    # Scale inputs and log-transform target
    x_scaler = StandardScaler()
    x_scaler.fit(X)
    X_scaled = x_scaler.transform(X)
    
    y_log = np.log(y + LOG_EPSILON)
    
    X_train = torch.tensor(X_scaled, dtype=torch.float64, device=device)
    y_train = torch.tensor(y_log, dtype=torch.float64, device=device)
    
    print(f"Training data: {len(X_train):,} points")
    print(f"Input dimensions: time, mass_flow, y, z")
    print(f"Target: log(y)")
    
    # Initialize logger
    if logger is None:
        logger = ExperimentLogger()
    
    logger.log_training_start({
        'n_train': len(X_train),
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'device': device,
        'target_transform': f'log(y)',
    })
    
    # Create model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    model = ExactGPModel(X_train, y_train, likelihood).to(device)
    
    print(f"\nModel: {type(model).__name__}")
    
    # Training
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    history = {
        'train_loss': [],
        'epochs': [],
    }
    
    # Data validation
    assert torch.isfinite(X_train).all(), "NaN/Inf in training features"
    assert torch.isfinite(y_train).all(), "NaN/Inf in training targets"
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    start_time = time_module.time()
    
    with gpytorch.settings.cholesky_jitter(1e-3):
        iterator = tqdm(range(n_epochs), desc="Training")
        for epoch in iterator:
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            
            iterator.set_postfix(loss=f"{loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(loss.item())
            history['epochs'].append(epoch + 1)
    
    training_time = time_module.time() - start_time
    
    print(f"\nTraining complete! Duration: {training_time/60:.2f} minutes")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    
    # Save model if path provided
    if model_path:
        save_checkpoint(
            model, likelihood, history, model_path, model_type='exact_gp',
            x_scaler=x_scaler,
            hyperparams={'learning_rate': learning_rate, 'n_epochs': n_epochs}
        )
    
    logger.log_training_end(
        best_epoch=n_epochs,
        best_val_loss=history['train_loss'][-1],
        total_time=training_time
    )
    
    return model, likelihood, history


def evaluate_validation(model, likelihood, val_loader, y_val_t, y_scaler=None,
                        likelihood_type='gaussian'):
    """
    Evaluate model on validation set and return MAE, RMSE.
    
    Args:
        model: GP model
        likelihood: Model likelihood
        val_loader: DataLoader for validation data
        y_val_t: Validation targets tensor
        y_scaler: StandardScaler for inverse-transforming targets (gaussian only)
        likelihood_type: 'gaussian' or 'beta'
        log_epsilon: Epsilon for log-transform inverse
    
    Returns:
        mae, rmse
    """
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            # Free GPU memory after each batch to prevent OOM with large k
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    means = means[1:]

    if likelihood_type == "gaussian":
        pred_mean_orig = torch.exp(torch.from_numpy(
            y_scaler.inverse_transform(means.numpy().reshape(-1, 1))
        ))
        y_val_orig = torch.exp(torch.from_numpy(
            y_scaler.inverse_transform(y_val_t.cpu().numpy().reshape(-1, 1))
        ))
        pred_mean_orig = torch.clamp(pred_mean_orig, min=0)
        mae = torch.mean(torch.abs(pred_mean_orig - y_val_orig)).item()
        rmse = torch.sqrt(torch.mean((pred_mean_orig - y_val_orig) ** 2)).item()
    else:
        mae = torch.mean(torch.abs(means - y_val_t.cpu())).item()
        rmse = torch.sqrt(torch.mean((means - y_val_t.cpu()) ** 2)).item()
    
    model.train()
    likelihood.train()
    return mae, rmse


def stratified_scenario_split(df_train_full, split_ratio=0.2, seed=42):
    """
    Split training scenarios into train and validation ensuring coverage
    across the mass flow range (low, mid, high).
    
    Args:
        df_train_full: DataFrame with 'scenario' and 'mass_flow' columns
        split_ratio: Fraction of scenarios to use for validation
        seed: Random seed
        
    Returns:
        train_scenarios, val_scenarios: arrays of scenario names
    """
    # Get unique scenarios and their mean mass flow
    scenario_info = df_train_full.groupby('scenario')['mass_flow'].mean().reset_index()
    scenario_info = scenario_info.sort_values('mass_flow').reset_index(drop=True)
    
    n_scenarios = len(scenario_info)
    n_val = max(1, int(n_scenarios * split_ratio))
    
    # Stratified selection: pick one scenario from each quantile bin
    np.random.seed(seed)
    val_indices = []
    bin_size = n_scenarios / n_val
    for i in range(n_val):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        end_idx = min(end_idx, n_scenarios)
        # Pick randomly within this bin
        idx = np.random.choice(range(start_idx, end_idx))
        val_indices.append(idx)
    
    val_scenarios = scenario_info.iloc[val_indices]['scenario'].values
    train_scenarios = scenario_info[~scenario_info.index.isin(val_indices)]['scenario'].values
    
    return train_scenarios, val_scenarios


def train_h2_dispersion_gp_approximate_additive(df,
                                        split_ratio: float = 0.2,
                                       n_inducing: int = 500,
                                       k: int = 64,
                                       training_batch_size: int = 512,
                                       model_type: str = "VNNGP",
                                       likelihood_type: str = 'gaussian',
                                       n_epochs: int = 200,
                                       learning_rate: float = 0.01,
                                       device: str = 'cpu',
                                       model_path: Optional[str] = None,
                                       trained_model: Optional[str] = None,
                                       val_every_n_epochs: int = 10,
                                       early_stopping_patience: int = 50,
                                       mass_flow_lengthscale_min: float = 0.1):
    """
    Train Sparse Variational GP (Approximate GP) with inducing points.
    
    This uses the SparseH2DispersionGP model which scales to larger datasets
    using inducing points for variational approximation.
    
    Args:
        df: Full dataframe with columns time, mass_flow, y, z, h2_volume_fraction, split, scenario
        split_ratio: Fraction of training scenarios to use for validation
        n_inducing: Number of inducing points (fewer = faster but less accurate)
        k: Number of nearest neighbors for VNNGP
        training_batch_size: Training batch size for VNNGP
        model_type: 'SVGP' or 'VNNGP'
        likelihood_type: 'gaussian' or 'beta'
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: 'cpu' or 'cuda'
        model_path: Optional path to save model
        trained_model: Optional path to load pre-trained model
        val_every_n_epochs: Evaluate validation metrics every N epochs
        early_stopping_patience: Stop if validation RMSE doesn't improve for this many epochs
        mass_flow_lengthscale_min: Minimum lengthscale for mass_flow dimension (prevents collapse)
    
    Returns:
        model, likelihood, history: Trained model, likelihood, and training history
    """

    # Split data - only train and test available
    df_train_full = df[df['split'] == 'train'].copy()
    #df_test = df[df['split'] == 'test'].copy()
    
    # Filter to active sensors only (two-stage model: classifier handles inactive)
    if 'active' in df_train_full.columns:
        n_before = len(df_train_full)
        df_train_full = df_train_full[df_train_full['active'] == 1].copy()
        print(f"Filtered to active sensors: {n_before:,} -> {len(df_train_full):,} rows")
    
    print(f"Full train: {len(df_train_full):,} rows")
    
    # Stratified split by mass flow
    train_scenarios, val_scenarios = stratified_scenario_split(df_train_full, split_ratio=split_ratio)
    
    df_train = df_train_full[~df_train_full['scenario'].isin(val_scenarios)].copy()
    df_val = df_train_full[df_train_full['scenario'].isin(val_scenarios)].copy()
    
    print(f"Split train: {len(df_train):,} rows, Val: {len(df_val):,} rows")
    print(f"Validation scenarios: {sorted(list(val_scenarios))}")
    print(f"Train scenarios: {sorted(list(train_scenarios))}")
    
    # Create experiment logger
    logger = ExperimentLogger(log_dir='experiments')

    # Prepare training data
    if likelihood_type == "gaussian":
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X = df_train[['time', 'mass_flow', 'x', 'y', 'z']].values
        x_scaler.fit(X)
        x_scaled = x_scaler.transform(X)
        y = df_train['h2_volume_fraction'].values
        y_log = np.log(y + LOG_EPSILON)
        y_scaler.fit(y_log.reshape(-1,1))
        y_scaled = y_scaler.transform(y_log.reshape(-1,1))

        x_train = torch.tensor(x_scaled, dtype=torch.float64, device=device).contiguous()
        y_train = torch.tensor(y_scaled, dtype=torch.float64, device=device).contiguous()

        train_dataset = TensorDataset(x_train, y_train.squeeze())
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Prepare validation data
        X_val = df_val[['time', 'mass_flow', 'x', 'y', 'z']].values
        x_val_scaled = x_scaler.transform(X_val)
        y_val = df_val['h2_volume_fraction'].values
        y_val_log = np.log(y_val + LOG_EPSILON)
        y_val_scaled = y_scaler.transform(y_val_log.reshape(-1,1))

        X_val_t = torch.tensor(x_val_scaled, dtype=torch.float64, device=device).contiguous()
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float64, device=device).contiguous()

        val_dataset = TensorDataset(X_val_t, y_val_t.squeeze())
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        print(f"\nTraining data: {len(x_train):,} points")
        print(f"Input dimensions: time, mass_flow, x,  y, z")
        print(f"Target: log(y)")

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    elif likelihood_type == 'beta':
        x_scaler = StandardScaler()
        X = df_train[['time', 'mass_flow', 'x', 'y', 'z']].values
        x_scaler.fit(X)
        x_scaled = x_scaler.transform(X)
        y = df_train['h2_volume_fraction'].values

        x_train = torch.tensor(x_scaled, dtype=torch.float64, device=device).contiguous()
        y_train = torch.tensor(y, dtype=torch.float64, device=device).contiguous()

        train_dataset = TensorDataset(x_train, y_train.squeeze())
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Prepare validation data
        X_val = df_val[['time', 'mass_flow', 'x', 'y', 'z']].values
        x_val_scaled = x_scaler.transform(X_val)
        y_val = df_val['h2_volume_fraction'].values

        X_val_t = torch.tensor(x_val_scaled, dtype=torch.float64, device=device).contiguous()
        y_val_t = torch.tensor(y_val, dtype=torch.float64, device=device).contiguous()

        val_dataset = TensorDataset(X_val_t, y_val_t.squeeze())
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        print(f"\nTraining data: {len(x_train):,} points")
        print(f"Input dimensions: time, mass_flow, x, y, z")

        likelihood = gpytorch.likelihoods.BetaLikelihood().double().to(device)
    else:
        raise ValueError(f"Unknown likelihhod type: {likelihood_type}")
    # Initialize logger
    if logger is None:
        logger = ExperimentLogger()
    
    logger.log_training_start({
        'n_train': len(x_train),
        'n_inducing': n_inducing,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'device': device,
        'model_type': model_type,
        'likelihood_type': likelihood_type
    })
    
    if model_type == "SVGP":
        # Select inducing points using k-means
        inducing_points = select_inducing_points(X, n_inducing=n_inducing, method='kmeans')
        inducing_points = inducing_points.to(device) 
        model = SparseH2DispersionGP(inducing_points=inducing_points).double().to(device)
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW([{'params' : model.parameters()}, {'params': likelihood.parameters()}], lr=learning_rate)

        print(f"\nModel: SparseH2DispersionGP (Approximate GP)")
        print(f"Variational parameters: {n_inducing} inducing points")

        #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train))
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=y_train.size(0))
        
        history = {
            'train_epoch_loss': [],
            'epochs': [],
            'RMSE': [],
            'MAE': []
        }
        
        print(f"\nTraining for {n_epochs} epochs...")
        print("-" * 60)
        
        start_time = time_module.time()
        
        iterator = tqdm(range(n_epochs), desc="Training  SVGP")
        for epoch in iterator:
            loss_epoch = 0.
            minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            with gpytorch.settings.cholesky_jitter():
                for x_batch, y_batch in minibatch_iter:
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    loss_train = loss.detach().item()
                    minibatch_iter.set_postfix(loss=loss_train)
                    loss_epoch += loss_train
                    loss.backward()
                    optimizer.step()

            epoch_loss = float(loss_epoch) / float(len(minibatch_iter))    
            iterator.set_postfix(epoch_loss=f"{epoch_loss:.4f}")
            history['train_epoch_loss'].append(epoch_loss)
            history['epochs'].append(epoch)
            if model_path is not None:
                path = model_path + str(epoch)
                save_checkpoint(
                    model, likelihood, history, path, model_type=model_type,
                    x_scaler=x_scaler, y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
                    hyperparams={
                        'k': k,
                        'training_batch_size': training_batch_size,
                        'n_inducing': n_inducing,
                        'likelihood_type': likelihood_type,
                        'lengthscale_constraints': None  # Not serializable; stored in model state
                    }
                )
        
        training_time = time_module.time() - start_time

    if model_type == "VNNGP":
        k = k
        training_batch_size = training_batch_size
        y_train = y_train.squeeze()
        
        # Build lengthscale constraints: mass_flow (dim 1) gets lower bound
        lengthscale_constraints = [None, None, None, None]
        if mass_flow_lengthscale_min > 0:
            lengthscale_constraints[1] = Interval(
                lower_bound=mass_flow_lengthscale_min,
                upper_bound=10.0,
                initial_value=1.0
            )
            print(f"\nMass flow lengthscale constraint: [{mass_flow_lengthscale_min}, 10.0]")
        
        # Select inducing points as subset of training data
        if trained_model is not None:
            print("Use pre-trained model.")
            checkpoint = torch.load(trained_model, map_location=device)
            # Use saved inducing points
            inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points']
            model = VNNGP(
                inducing_points=inducing_points, likelihood=likelihood,
                k=k, training_batch_size=training_batch_size,
                lengthscale_constraints=lengthscale_constraints
            ).double().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        else:
            model = VNNGP(
                inducing_points=x_train, likelihood=likelihood,
                k=k, training_batch_size=training_batch_size,
                lengthscale_constraints=lengthscale_constraints
            ).double().to(device)
        num_batches = model.variational_strategy._total_training_batches
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
        print(f"\nModel: VNNGP (Approximate GP)")

        #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=y_train.size(0))

        history = {
            'train_epoch_loss': [],
            'epochs': [],
            'RMSE': [],
            'MAE': [],
            'val_RMSE': [],
            'val_MAE': [],
            'val_epochs': [],
            'best_epoch': None
        }
        
        print(f"\nTraining for {n_epochs} epochs...")
        print(f"Validation every {val_every_n_epochs} epochs, early stopping patience={early_stopping_patience}")
        print("-" * 60)

        best_val_rmse = float('inf')
        epochs_since_improvement = 0
        best_model_state = None
        best_likelihood_state = None
        best_optimizer_state = None

        start_time = time_module.time()
        iterator = tqdm(range(n_epochs), desc="Training VNNGP")
        for epoch in iterator:
            loss_epoch = 0.
            minibatch_iter = tqdm(range(num_batches), desc="Minibatch", leave=False)
            with gpytorch.settings.cholesky_jitter():
                for i in minibatch_iter:
                    optimizer.zero_grad()
                    output = model(x=None)
                    current_training_indices = model.variational_strategy.current_training_indices
                    y_batch = y_train[..., current_training_indices]
                    loss = -mll(output, y_batch)
                    loss_train = loss.detach().item()
                    minibatch_iter.set_postfix(loss=loss_train)
                    loss_epoch += loss_train
                    loss.backward()
                    optimizer.step()

            epoch_loss = float(loss_epoch) / float(len(minibatch_iter))
            iterator.set_postfix(epoch_loss=f"{epoch_loss:.4f}")
            history['train_epoch_loss'].append(epoch_loss)
            history['epochs'].append(epoch)
            
            # Validation evaluation every N epochs
            if (epoch + 1) % val_every_n_epochs == 0:
                mae, rmse = evaluate_validation(
                    model, likelihood, val_loader, y_val_t,
                    y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
                    likelihood_type=likelihood_type
                )
                history['val_MAE'].append(mae)
                history['val_RMSE'].append(rmse)
                history['val_epochs'].append(epoch)
                
                iterator.set_postfix(
                    epoch_loss=f"{epoch_loss:.4f}",
                    val_rmse=f"{rmse:.4f}",
                    val_mae=f"{mae:.4f}"
                )
                print(f"\nEpoch {epoch+1}: Val MAE={mae:.4f}, Val RMSE={rmse:.4f}")
                
                # Early stopping check
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    epochs_since_improvement = 0
                    history['best_epoch'] = epoch + 1
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_likelihood_state = {k: v.cpu().clone() for k, v in likelihood.state_dict().items()}
                    best_optimizer_state = optimizer.state_dict()
                    if model_path is not None:
                        best_path = model_path + "_best"
                        save_checkpoint(
                            model, likelihood, history, best_path, model_type=model_type,
                            x_scaler=x_scaler, y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
                            hyperparams={
                                'k': k,
                                'training_batch_size': training_batch_size,
                                'n_inducing': n_inducing,
                                'likelihood_type': likelihood_type
                            }
                        )
                else:
                    epochs_since_improvement += val_every_n_epochs
                
                logger.log_epoch(epoch, epoch_loss, val_loss=rmse)
                
                if epochs_since_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered! No improvement for {epochs_since_improvement} epochs.")
                    print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {history['best_epoch']}")
                    break
            else:
                logger.log_epoch(epoch, epoch_loss)
            
            if model_path is not None:
                path = model_path + str(epoch)
                save_checkpoint(
                    model, likelihood, history, path, model_type=model_type,
                    x_scaler=x_scaler, y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
                    hyperparams={
                        'k': k,
                        'training_batch_size': training_batch_size,
                        'n_inducing': n_inducing,
                        'likelihood_type': likelihood_type
                    }
                )
        
        training_time = time_module.time() - start_time
        
        # Restore best model if available
        if best_model_state is not None:
            print(f"\nRestoring best model from epoch {history['best_epoch']} (val RMSE={best_val_rmse:.4f})")
            model.load_state_dict(best_model_state)
            likelihood.load_state_dict(best_likelihood_state)
            optimizer.load_state_dict(best_optimizer_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining complete! Duration: {training_time/60:.2f} minutes")
    print(f"Final training loss: {history['train_epoch_loss'][-1]:.4f}")
    
    logger.log_training_end(
        best_epoch=history.get('best_epoch', n_epochs),
        best_val_loss=best_val_rmse if best_val_rmse != float('inf') else history['train_epoch_loss'][-1],
        total_time=training_time
    )

    # Final validation evaluation
    final_mae, final_rmse = evaluate_validation(
        model, likelihood, val_loader, y_val_t,
        y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
        likelihood_type=likelihood_type
    )

    history['RMSE'].append(final_rmse)
    history['MAE'].append(final_mae)
    
    # Save model if path provided
    if model_path:
        save_checkpoint(
            model, likelihood, history, model_path, model_type=model_type,
            x_scaler=x_scaler, y_scaler=y_scaler if likelihood_type == 'gaussian' else None,
            hyperparams={
                'k': k,
                'training_batch_size': training_batch_size,
                'n_inducing': n_inducing,
                'likelihood_type': likelihood_type
            }
        )

    plt.figure(figsize=(10, 5))
    plt.plot(history['epochs'], history['train_epoch_loss'], label='Train Epoch Loss', marker='s', markersize=3)
    if history['val_epochs']:
        plt.plot(history['val_epochs'], history['val_RMSE'], label='Val RMSE', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / RMSE')
    plt.title('Training Loss and Validation RMSE over Epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig_name = model_type + str(k)
    plt.savefig(fig_name)

    print(f"Final Validation MAE: {final_mae:.4f}")
    print(f"Final Validation RMSE: {final_rmse:.4f}")
    if history['best_epoch'] is not None:
        print(f"Best Validation RMSE: {best_val_rmse:.4f} at epoch {history['best_epoch']}")
    
    return model, likelihood, history


def save_checkpoint(model, likelihood, history, path, model_type='gp',
                     x_scaler=None, y_scaler=None,
                     hyperparams=None):
    """
    Save model checkpoint with training history and metadata.
    
    Args:
        model: Trained GP model
        likelihood: Model likelihood
        history: Training history dict
        path: Save path
        model_type: Type of model for metadata
        x_scaler: StandardScaler for inputs [time, mass_flow, y, z]
        y_scaler: StandardScaler for target log(C + epsilon)
        hyperparams: Dict of additional hyperparameters (k, n_inducing, etc.)
    """
    import json
    from pathlib import Path
    import pickle
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'history': history,
        'model_type': model_type,
    }
    
    # Save scalers using pickle (StandardScaler is not JSON-serializable)
    if x_scaler is not None:
        checkpoint['x_scaler'] = pickle.dumps(x_scaler)
    if y_scaler is not None:
        checkpoint['y_scaler'] = pickle.dumps(y_scaler)
    
    # Save hyperparameters
    if hyperparams is not None:
        checkpoint['hyperparams'] = hyperparams
    
    torch.save(checkpoint, path)
    print(f"\nModel saved to {path}")
    
    # Also save history as JSON
    history_path = path.parent / (path.stem + '_history.json')
    with open(history_path, 'w') as f:
        # Convert any tensors in history to lists
        history_serializable = {}
        for k, v in history.items():
            if v is None:
                continue
            if isinstance(v, list):
                # Convert each element if it's a tensor
                history_serializable[k] = [
                    float(x.item()) if isinstance(x, torch.Tensor) else x 
                    for x in v
                ]
            elif isinstance(v, torch.Tensor):
                history_serializable[k] = v.detach().cpu().tolist()
            else:
                history_serializable[k] = v
        
        json.dump(history_serializable, f, indent=2)
    print(f"History saved to {history_path}")


def evaluate_gp_model(model, likelihood, df_test, device='cpu', x_scaler=None, y_scaler=None):
    """
    Evaluate trained GP model on test set.
    
    This should only be called for the final selected model (best on validation).
    Returns comprehensive metrics including MAE, RMSE, R².
    
    Args:
        model: Trained GP model
        likelihood: Model likelihood
        df_test: Test dataframe
        device: 'cpu' or 'cuda'
        x_scaler: StandardScaler for input features (if used during training)
        y_scaler: StandardScaler for log-targets (if used during training)
    
    Returns:
        metrics: Dict with 'mae', 'rmse', 'r2', 'nll', 'predictions', 'targets'
    """
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    # Prepare test data
    X_test = df_test[['time', 'mass_flow', 'x', 'y', 'z']].values
    y_test = df_test['h2_volume_fraction'].values
    y_test_log = np.log(y_test + LOG_EPSILON)
    
    # Apply input scaling if scaler was used during training
    if x_scaler is not None:
        X_test = x_scaler.transform(X_test)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float64, device=device)
    y_test_t = torch.tensor(y_test_log, dtype=torch.float64, device=device)
    
    print(f"Test data: {len(X_test_t):,} points")
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # Get predictions
        pred = likelihood(model(X_test_t))
        pred_mean = pred.mean
        pred_std = pred.stddev
        
        # Compute NLL (in log space)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        nll = -mll(pred, y_test_t)
        
        # Convert to original scale for metrics
        if y_scaler is not None:
            pred_mean_orig = torch.exp(torch.from_numpy(
                y_scaler.inverse_transform(pred_mean.cpu().numpy().reshape(-1, 1))
            ).to(device))
            y_test_orig = torch.exp(torch.from_numpy(
                y_scaler.inverse_transform(y_test_t.cpu().numpy().reshape(-1, 1))
            ).to(device))
        else:
            pred_mean_orig = torch.exp(pred_mean)
            y_test_orig = torch.exp(y_test_t)
        
        pred_std_orig = pred_std * pred_mean_orig  # Delta method: std in original space ≈ std_log * mean_orig
        
        # Ensure non-negative
        pred_mean_orig = torch.clamp(pred_mean_orig, min=0)
        
        # Metrics
        mae = torch.mean(torch.abs(pred_mean_orig - y_test_orig)).item()
        rmse = torch.sqrt(torch.mean((pred_mean_orig - y_test_orig) ** 2)).item()
        
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'nll': nll.item(),
        'predictions': pred_mean_orig.cpu().numpy(),
        'targets': y_test_orig.cpu().numpy(),
        'std': pred_std_orig.cpu().numpy(),
    }
    
    print(f"\nTest Metrics:")
    print(f"  NLL:      {nll:.4f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    
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
    
    Auto-detects model type from checkpoint metadata and reconstructs
    the correct model class with all hyperparameters.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: 'cpu' or 'cuda'
    
    Returns:
        model, likelihood, checkpoint dict
    """
    import pickle
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get('model_type', 'SVGP')
    hyperparams = checkpoint.get('hyperparams', {})
    
    if model_type == 'VNNGP' or model_type.endswith('vnngp'):
        # VNNGP reconstruction
        inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points'].to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
        
        # Read hyperparameters from checkpoint, with sensible defaults
        k = hyperparams.get('k', 64)
        training_batch_size = hyperparams.get('training_batch_size', 1024)
        lengthscale_constraints = hyperparams.get('lengthscale_constraints', None)
        
        model = VNNGP(
            inducing_points=inducing_points,
            likelihood=likelihood,
            k=k,
            training_batch_size=training_batch_size,
            lengthscale_constraints=lengthscale_constraints
        ).double().to(device)
        
    elif model_type == 'ExactGP':
        # ExactGP requires train_x and train_y to reconstruct
        raise NotImplementedError(
            "ExactGP model loading is not supported from checkpoint alone. "
            "Please use load_inference_model() with the training CSV."
        )
    else:
        # Default: SparseH2DispersionGP (SVGP)
        inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points'].to(device)
        model = SparseH2DispersionGP(inducing_points=inducing_points).double().to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    model.eval()
    likelihood.eval()
    
    # Restore scalers if saved
    if 'x_scaler' in checkpoint:
        checkpoint['x_scaler'] = pickle.loads(checkpoint['x_scaler'])
    if 'y_scaler' in checkpoint:
        checkpoint['y_scaler'] = pickle.loads(checkpoint['y_scaler'])
    
    return model, likelihood, checkpoint


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv('/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model/data/unified_raw_cut_off.csv')
    
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
    
    # Train on training set only
    print("\nTraining GP model...")
    """model, likelihood, history = train_h2_dispersion_gp(
        df_train, 
        n_epochs=500,
        learning_rate=0.1,
        device='cuda',
        logger=logger,
        model_path='models/exact_gp.pth'
    )"""

    """model, likelihood, history = train_h2_dispersion_gp_additive(
        df_train=df_train,
        n_epochs=100,
        learning_rate=0.1,
        device='cuda',
        logger=logger,
        model_path='models/additive_gp.pth'
    )"""

    model, likelihood, history = train_h2_dispersion_gp_approximate_additive(
        df=df,
        split_ratio=0.3,
        n_inducing=6000,
        k=32,
        training_batch_size=1024,
        model_type="VNNGP",
        likelihood_type="beta",
        n_epochs=1,
        learning_rate=0.008,
        device='cuda:0',
        model_path='models/approximate_scaleAdditive_vnngp_k32_beta.pth',
        trained_model=None
    )