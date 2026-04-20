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

# Log transform epsilon to avoid log(0) - data has small values ~1e-8
LOG_EPSILON = 1e-6


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
    def __init__(self, inducing_points, likelihood, k=256, training_batch_size=256):

        m, d = inducing_points.shape
        self.m = m
        self.k = k

        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)

        variational_strategy = NNVariationalStrategy(self, inducing_points, variational_distribution, k=k, training_batch_size=training_batch_size, jitter_val=0.001)

        super(VNNGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        #self.covar_module = FullAdditiveKernel(base_kernel_type='rbf', num_dims=4)
        self.covar_module = ScaleAdditiveKernel(base_kernel_type='rbf', num_dims=4) # less output scale parameters and more interpretable

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
        inducing_points = torch.tensor(X_train[indices], dtype=torch.float32)
    else:
        # Use all data if smaller than n_inducing
        inducing_points = torch.tensor(X_train, dtype=torch.float32)
    
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
    
    # Log-transform target to handle skewed distribution
    y_log = np.log(y + LOG_EPSILON)
    
    X_train = torch.tensor(X, dtype=torch.float64, device=device)
    y_train = torch.tensor(y_log, dtype=torch.float64, device=device)
    
    print(f"Training data: {len(X_train):,} points")
    print(f"Input dimensions: time, mass_flow, y, z")
    print(f"Target: log(y + {LOG_EPSILON})")
    
    # Initialize logger
    if logger is None:
        logger = ExperimentLogger()
    
    logger.log_training_start({
        'n_train': len(X_train),
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'device': device,
        'log_epsilon': LOG_EPSILON,
        'target_transform': f'log(y + {LOG_EPSILON})',
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
        save_checkpoint(model, likelihood, history, model_path, model_type='exact_gp')
    
    logger.log_training_end(
        best_epoch=n_epochs,
        best_val_loss=history['train_loss'][-1],
        total_time=training_time
    )
    
    return model, likelihood, history


def train_h2_dispersion_gp_approximate_additive(df,
                                        split_ratio: float = 0.3,
                                       n_inducing: int = 500,
                                       k: int = 64,
                                       training_batch_size: int = 512,
                                       model_type: str = "VNNGP",
                                       likelihood_type: str = 'gaussian',
                                       n_epochs: int = 200,
                                       learning_rate: float = 0.01,
                                       device: str = 'cpu',
                                       model_path: Optional[str] = None,
                                       trained_model: Optional[str] = None):
    """
    Train Sparse Variational GP (Approximate GP) with inducing points.
    
    This uses the SparseH2DispersionGP model which scales to larger datasets
    using inducing points for variational approximation.
    
    Args:
        df_train: Training dataframe with columns time, mass_flow, y, z, h2_volume_fraction
        n_inducing: Number of inducing points (fewer = faster but less accurate)
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: 'cpu' or 'cuda'
        logger: Optional ExperimentLogger
        model_path: Optional path to save model (e.g., 'models/approximate_gp.pth')
    
    Returns:
        model, likelihood, history: Trained model, likelihood, and training history
    """

    # Split data - only train and test available
    df_train_full = df[df['split'] == 'train'].copy()
    #df_test = df[df['split'] == 'test'].copy()
    
    print(f"Full train: {len(df_train_full):,} rows")
    
    # Split training data into train and validation (80/20 split by scenarios)
    split_ratio = split_ratio
    train_scenarios = df_train_full['scenario'].unique()
    n_val = max(1, int(len(train_scenarios) * split_ratio))
    np.random.seed(42)
    val_scenarios = np.random.choice(train_scenarios, size=n_val, replace=False)
    
    df_train = df_train_full[~df_train_full['scenario'].isin(val_scenarios)].copy()
    df_val = df_train_full[df_train_full['scenario'].isin(val_scenarios)].copy()
    
    print(f"Split train: {len(df_train):,} rows, Val: {len(df_val):,} rows")
    print(f"Validation scenarios: {list(val_scenarios)}")
    
    # Create experiment logger
    logger = ExperimentLogger(log_dir='experiments')

    # Prepare training data
    if likelihood_type == "gaussian":
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X = df_train[['time', 'mass_flow', 'y', 'z']].values
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
        X_val = df_val[['time', 'mass_flow', 'y', 'z']].values
        x_val_scaled = x_scaler.transform(X_val)
        y_val = df_val['h2_volume_fraction'].values
        y_val_log = np.log(y_val + LOG_EPSILON)
        y_val_scaled = y_scaler.transform(y_val_log.reshape(-1,1))

        X_val_t = torch.tensor(x_val_scaled, dtype=torch.float64, device=device).contiguous()
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float64, device=device).contiguous()

        val_dataset = TensorDataset(X_val_t, y_val_t.squeeze())
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        print(f"\nTraining data: {len(x_train):,} points")
        print(f"Input dimensions: time, mass_flow, y, z")
        print(f"Target: log(y + {LOG_EPSILON})")

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    elif likelihood_type == 'beta':
        x_scaler = StandardScaler()
        X = df_train[['time', 'mass_flow', 'y', 'z']].values
        x_scaler.fit(X)
        x_scaled = x_scaler.transform(X)
        y = df_train['h2_volume_fraction'].values
        y_beta = np.clip(y, 1e-6, 1 - 1e-6)

        x_train = torch.tensor(x_scaled, dtype=torch.float64, device=device).contiguous()
        y_train = torch.tensor(y_beta, dtype=torch.float64, device=device).contiguous()

        train_dataset = TensorDataset(x_train, y_train.squeeze())
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Prepare validation data
        X_val = df_val[['time', 'mass_flow', 'y', 'z']].values
        x_val_scaled = x_scaler.transform(X_val)
        y_val = df_val['h2_volume_fraction'].values
        y_val_beta = np.clip(y_val, 1e-6, 1 - 1e-6)

        X_val_t = torch.tensor(x_val_scaled, dtype=torch.float64, device=device).contiguous()
        y_val_t = torch.tensor(y_val_beta, dtype=torch.float64, device=device).contiguous()

        val_dataset = TensorDataset(X_val_t, y_val_t.squeeze())
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        print(f"\nTraining data: {len(x_train):,} points")
        print(f"Input dimensions: time, mass_flow, y, z")

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
                save_checkpoint(model, likelihood, history, path, model_type=model_type)
        
        training_time = time_module.time() - start_time

    elif model_type == "VNNGP":
        k = k
        training_batch_size = training_batch_size
        y_train = y_train.squeeze()
        if trained_model is not None:
            print("Use pre-trained model.")
            checkpoint = torch.load(trained_model, map_location=device)
            model = VNNGP(inducing_points=x_train,likelihood=likelihood, k=k, training_batch_size=training_batch_size).double().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        else:
            model = VNNGP(inducing_points=x_train, likelihood=likelihood, k=k, training_batch_size=training_batch_size).double().to(device)
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
        'MAE': []
        }
        
        print(f"\nTraining for {n_epochs} epochs...")
        print("-" * 60)

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
            if model_path is not None:
                path = model_path + str(epoch)
                save_checkpoint(model, likelihood, history, path, model_type=model_type)
        
        training_time = time_module.time() - start_time
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining complete! Duration: {training_time/60:.2f} minutes")
    print(f"Final training loss: {history['train_epoch_loss'][-1]:.4f}")
    
    logger.log_training_end(
        best_epoch=n_epochs,
        best_val_loss=history['train_epoch_loss'][-1],
        total_time=training_time
    )

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    print("Ready to evaluate.")
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]

    if likelihood_type == "gaussian":
        # Convert to original scale for MAE/RMSE
        pred_mean_orig = torch.exp(torch.from_numpy(y_scaler.inverse_transform(means.numpy().reshape(-1, 1)))) - LOG_EPSILON
        y_val_orig = torch.exp(torch.from_numpy(y_scaler.inverse_transform(y_val_t.cpu().numpy().reshape(-1, 1)))) - LOG_EPSILON
        
        # Ensure non-negative for MAE/RMSE calculation
        pred_mean_orig = torch.clamp(pred_mean_orig, min=0)
        
        mae = torch.mean(torch.abs(pred_mean_orig - y_val_orig)).item()
        rmse = torch.sqrt(torch.mean((pred_mean_orig - y_val_orig) ** 2)).item()
    else:
        mae = torch.mean(torch.abs(means - y_val_t.cpu()))
        rmse = torch.sqrt(torch.mean(means - y_val_t.cpu()))


    history['RMSE'].append(rmse)
    history['MAE'].append(mae)
    # Save model if path provided
    if model_path:
        save_checkpoint(model, likelihood, history, model_path, model_type=model_type)

    plt.figure(figsize=(7, 5))
    plt.plot(history['epochs'], history['train_epoch_loss'], label='Train Epoch Loss', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig_name = model_type + str(k)
    plt.savefig(fig_name)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    return model, likelihood, history


def save_checkpoint(model, likelihood, history, path, model_type='gp'):
    """
    Save model checkpoint with training history.
    
    Args:
        model: Trained GP model
        likelihood: Model likelihood
        history: Training history dict
        path: Save path
        model_type: Type of model for metadata
    """
    import json
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'history': history,
        'epoch': history['epochs'],
        'model_type': model_type
    }
    
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


def train_h2_dispersion_gp_exact_additive(df_train,
                                    n_epochs: int = 200,
                                    learning_rate: float = 0.01,
                                    device: str = 'cpu',
                                    logger: Optional[ExperimentLogger] = None,
                                    model_path: Optional[str] = None):
    """
    Train H2 dispersion model using Full Additive Kernel (Duvenaud et al. 2011).
    
    This model learns interactions of all orders between input dimensions:
    - 1st order: time, mass_flow, y, z individually
    - 2nd order: pairs (time×mass_flow, time×y, etc.)
    - 3rd order: triples, etc.
    
    The kernel structure is:
    k(x,x') = Σ_{d=1}^4 Σ_{|S|=d} σ²_S ∏_{i∈S} k_i(x_i, x'_i)
    
    For 4D input, this creates 2^4 - 1 = 15 interaction terms.
    
    Args:
        df_train: Training dataframe with columns time, mass_flow, y, z, h2_volume_fraction
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam
        device: 'cpu' or 'cuda'
        logger: Optional ExperimentLogger
        learn_interactions: If False, uses only 1st order terms (no interactions)
        model_path: Optional path to save model (e.g., 'models/additive_gp.pth')
    
    Returns:
        model, likelihood, history: Trained additive model and training history
    """
    # Prepare data
    X = df_train[['time', 'mass_flow', 'y', 'z']].values
    y = df_train['h2_volume_fraction'].values
    y_log = np.log(y + LOG_EPSILON)
    
    X_train = torch.tensor(X, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_log, dtype=torch.float32, device=device)
    
    print(f"\nTraining data: {len(X_train):,} points")
    print(f"Input dimensions: {X_train.shape[1]} (time, mass_flow, y, z)")
    
    # Initialize logger
    if logger is None:
        logger = ExperimentLogger()
    
    logger.log_training_start({
        'n_train': len(X_train),
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'device': device,
        'kernel': 'FullAdditiveKernel',
        'num_dims': 4,
        'num_terms': 15,
        'target_transform': f'log(y + {LOG_EPSILON})',
    })
    
    # Create additive GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPAdditiveModel(
        X_train, y_train, likelihood, 
        num_dims=4
    ).to(device)
    
    print(f"\nModel: ExactGP with Full Additive Kernel")
    print(f"Number of interaction terms: {model.covar_module.num_terms}")
    
    # Training
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    history = {
        'train_loss': [],
        'epochs': [],
    }
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    start_time = time_module.time()
    
    iterator = tqdm(range(n_epochs), desc="Training Additive GP")
    for epoch in iterator:
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        
        iterator.set_postfix(loss=f"{loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        history['epochs'].append(epoch)
    
    elapsed = time_module.time() - start_time
    
    print(f"\nTraining complete! Duration: {elapsed/60:.2f} minutes")
    
    # Save model if path provided
    if model_path:
        save_checkpoint(model, likelihood, history, model_path, model_type='additive_gp')
    
    logger.log_training_end(
        best_epoch=n_epochs,
        best_val_loss=history['train_loss'][-1],
        total_time=elapsed
    )
    
    return model, likelihood, history


def evaluate_gp_model(model, likelihood, df_test, device='cpu'):
    """
    Evaluate trained GP model on test set.
    
    This should only be called for the final selected model (best on validation).
    Returns comprehensive metrics including MAE, RMSE, R².
    
    Args:
        model: Trained GP model
        likelihood: Model likelihood
        df_test: Test dataframe
        device: 'cpu' or 'cuda'
    
    Returns:
        metrics: Dict with 'mae', 'rmse', 'r2', 'nll', 'predictions', 'targets'
    """
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    # Prepare test data
    X_test = df_test[['time', 'mass_flow', 'y', 'z']].values
    y_test = df_test['h2_volume_fraction'].values
    y_test_log = np.log(y_test + LOG_EPSILON)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_log, dtype=torch.float32, device=device)
    
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
        pred_mean_orig = torch.exp(pred_mean) - LOG_EPSILON
        y_test_orig = torch.exp(y_test_t) - LOG_EPSILON
        pred_std_orig = pred_std * torch.exp(pred_mean)  # Approximate std in original space
        
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


def plot_gp_predictive_distributions(model, X_train, y_train, device='cpu', 
                                     save_path='gp_predictive_dists.png',
                                     log_epsilon=LOG_EPSILON,
                                     n_test_points=200,
                                     n_samples=10):
    """
    Plot GP predictive distributions for each dimension.
    
    Creates 1D slice plots showing:
    - Training data as blue crosses
    - Posterior mean ± 2*std as red shaded area
    - Posterior samples as thin black lines
    
    Args:
        model: Trained SparseH2DispersionGP
        X_train: Training inputs [N, 4] (time, mass_flow, y, z)
        y_train: Training targets [N] (log-transformed)
        device: Device for computation
        save_path: Where to save the figure
        log_epsilon: Epsilon used for log transform
        n_test_points: Number of test points per dimension
        n_samples: Number of posterior samples to draw
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    
    # Dimension labels and indices - ONLY for dimensions with active kernels
    # Kernels: time (dim 0), mass_flow (dim 1), space_z (dim 3)
    # Note: y (dim 2) has no active kernel
    dim_names = ['Mass Flow (kg/s)', 'Y Position (m)', 'Z Position (m)']
    dim_indices =  [1, 2, 3]  # Skip dimension 2 (y)
    
    # Compute median values for fixing other dimensions
    X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    median_values = np.median(X_train_np, axis=0)
    
    # Convert training targets back from log space for plotting
    y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    y_train_original = np.exp(y_train_np) - log_epsilon
    
    # Create 3 subplots (one per kernel) in a 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (dim_idx, dim_name) in enumerate(zip(dim_indices, dim_names)):
        ax = axes[idx]
        
        # Get min/max for this dimension from training data
        dim_min = X_train_np[:, dim_idx].min()
        dim_max = X_train_np[:, dim_idx].max()
        
        # Create test points varying only this dimension
        X_test_1d = np.linspace(dim_min, dim_max, n_test_points)
        X_test = np.tile(median_values, (n_test_points, 1))
        X_test[:, dim_idx] = X_test_1d
        
        # Convert to tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=device)
        
        with torch.no_grad():
            # Get posterior distribution
            posterior = model(X_test_tensor)
            mean = posterior.mean.cpu().numpy()
            std = posterior.stddev.cpu().numpy()
            
            # Sample from posterior
            samples = posterior.sample(sample_shape=torch.Size([n_samples])).cpu().numpy()
        
        # Inverse log-transform to get original scale
        mean_orig = np.exp(mean) - log_epsilon
        std_orig = mean_orig * std  # Approximate delta method
        samples_orig = np.exp(samples) - log_epsilon
        
        # Clip negative values
        mean_orig = np.maximum(mean_orig, 0)
        std_orig = np.maximum(std_orig, 0)
        samples_orig = np.maximum(samples_orig, 0)
        
        # Plot training data (projected onto this dimension)
        ax.scatter(X_train_np[:, dim_idx], y_train_original, 
                  c='blue', marker='x', s=30, alpha=0.5, 
                  label='Training Data', zorder=3)
        
        # Plot posterior samples as black lines
        for i in range(n_samples):
            ax.plot(X_test_1d, samples_orig[i], 'k-', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Plot posterior mean ± 2*std as red shaded area
        ax.fill_between(X_test_1d, 
                       mean_orig - 2*std_orig, 
                       mean_orig + 2*std_orig,
                       color='red', alpha=0.2, label='Posterior ±2σ', zorder=2)
        
        # Plot posterior mean as red line
        ax.plot(X_test_1d, mean_orig, 'r-', linewidth=2, label='Posterior Mean', zorder=4)
        
        # Add critical threshold line (4% = 0.04)
        ax.axhline(y=0.04, color='green', linestyle='--', linewidth=1.5, 
                  label='Critical 4%', alpha=0.7, zorder=0)
        
        # Formatting
        ax.set_xlabel(dim_name, fontsize=11)
        ax.set_ylabel('H₂ Volume Fraction', fontsize=11)
        ax.set_title(f'Predictive Distribution vs {dim_name.split(" (")[0]}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.suptitle('GP Predictive Distributions (1D Slices at Median Values)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGP predictive distributions saved to: {save_path}")
    plt.close()


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
        'log_epsilon': LOG_EPSILON,
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
    
    inducing_points = checkpoint['model_state_dict']['variational_strategy.inducing_points'].to(device)
    
    model = SparseH2DispersionGP(inducing_points=inducing_points)
    #likelihood = gpytorch.likelihoods.BetaLikelihood().double().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

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