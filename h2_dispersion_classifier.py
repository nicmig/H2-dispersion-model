"""
Two-Stage GP: Stage 1 - H2 Dispersion Sensor Activity Classifier

Classifies whether a sensor location is "active" (H2 detected) or "inactive"
based on time, mass_flow, y, z inputs.

Models:
- ExactGP with KeOps RBF kernel (for smaller datasets)
- VNNGP with Bernoulli likelihood (scalable alternative)

Target: binary active/inactive label (0 or 1)
"""

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import torch
import gpytorch
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time as time_module
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from h2_dispersion_gp import ExperimentLogger, save_checkpoint, stratified_scenario_split, select_inducing_points

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

class H2DispersionClassifierExactGP(gpytorch.models.ExactGP):
    """
    Exact GP classifier for H2 sensor activity.
    
    Uses KeOps RBF kernel for memory-efficient computation.
    Only suitable for datasets up to ~20k points due to O(n^3) complexity.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(ard_num_dims=4)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class H2DispersionClassifierVNNGP(gpytorch.models.ApproximateGP):
    """
    VNNGP classifier for H2 sensor activity.
    
    Scalable variational approximation with Bernoulli likelihood.
    Suitable for large datasets (100k+ points).
    """
    def __init__(self, inducing_points, k=256, training_batch_size=256):
        m, d = inducing_points.shape
        self.m = m
        self.k = k
        
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(m)
        variational_strategy = NNVariationalStrategy(
            self, inducing_points, variational_distribution,
            k=k, training_batch_size=training_batch_size, jitter_val=0.001
        )
        
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(ard_num_dims=4)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x=None, prior=False, **kwargs):
        if x is not None and x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.variational_strategy(x=x, prior=prior, **kwargs)


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_classifier_data(df, device='cpu', max_exact_points=20000):
    """
    Prepare data for classifier training.
    
    Args:
        df: DataFrame with columns time, mass_flow, y, z, active, split, scenario
        device: 'cpu' or 'cuda'
        max_exact_points: Maximum points for Exact GP (falls back to VNNGP if exceeded)
    
    Returns:
        Dict with train/val/test tensors and metadata
    """
    # Use all data (both train and test splits) but keep test held out
    df_train_full = df[df['split'] == 'train'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    print(f"Full data: {len(df):,} rows")
    print(f"Train pool: {len(df_train_full):,} rows")
    print(f"Test pool: {len(df_test):,} rows")
    
    # Stratified split: 80% scenarios for training, 20% for test
    # But test experiments are already held out, so we use those as test
    # From train scenarios, take 20% as validation
    all_train_scenarios = df_train_full['scenario'].unique()
    n_val = max(1, int(len(all_train_scenarios) * 0.20))
    np.random.seed(42)
    
    # Sort by mass flow for stratified selection
    scenario_info = df_train_full.groupby('scenario')['mass_flow'].mean().reset_index()
    scenario_info = scenario_info.sort_values('mass_flow').reset_index(drop=True)
    
    val_indices = []
    bin_size = len(scenario_info) / n_val
    for i in range(n_val):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        end_idx = min(end_idx, len(scenario_info))
        idx = np.random.choice(range(start_idx, end_idx))
        val_indices.append(idx)
    
    val_scenarios = scenario_info.iloc[val_indices]['scenario'].values
    train_scenarios = scenario_info[~scenario_info.index.isin(val_indices)]['scenario'].values
    test_scenarios = df_test['scenario'].unique()
    
    df_train = df_train_full[df_train_full['scenario'].isin(train_scenarios)].copy()
    df_val = df_train_full[df_train_full['scenario'].isin(val_scenarios)].copy()
    
    print(f"\nSplit:")
    print(f"  Train: {len(df_train):,} rows ({len(train_scenarios)} scenarios)")
    print(f"  Val:   {len(df_val):,} rows ({len(val_scenarios)} scenarios)")
    print(f"  Test:  {len(df_test):,} rows ({len(test_scenarios)} scenarios)")
    print(f"  Train scenarios: {sorted(list(train_scenarios))}")
    print(f"  Val scenarios:   {sorted(list(val_scenarios))}")
    print(f"  Test scenarios:  {sorted(list(test_scenarios))}")
    
    # Fit scaler on train data only
    x_scaler = StandardScaler()
    X_train = df_train[['time', 'mass_flow', 'y', 'z']].values
    x_scaler.fit(X_train)
    
    X_val = df_val[['time', 'mass_flow', 'y', 'z']].values
    X_test = df_test[['time', 'mass_flow', 'y', 'z']].values
    
    x_train_scaled = x_scaler.transform(X_train)
    x_val_scaled = x_scaler.transform(X_val)
    x_test_scaled = x_scaler.transform(X_test)
    
    y_train = df_train['active'].values.astype(np.float64)
    y_val = df_val['active'].values.astype(np.float64)
    y_test = df_test['active'].values.astype(np.float64)
    
    # Check class balance
    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        print(f"  {name} active: {n_pos:,} ({100*n_pos/len(y):.1f}%), inactive: {n_neg:,} ({100*n_neg/len(y):.1f}%)")
    
    # Determine if Exact GP is feasible
    use_exact = len(x_train_scaled) <= max_exact_points
    if not use_exact:
        print(f"\nWARNING: Train set ({len(x_train_scaled):,}) exceeds max_exact_points ({max_exact_points}).")
        print("         Falling back to VNNGP.")
    
    data = {
        'x_train': torch.tensor(x_train_scaled, dtype=torch.float64, device=device).contiguous(),
        'y_train': torch.tensor(y_train, dtype=torch.float64, device=device).contiguous(),
        'x_val': torch.tensor(x_val_scaled, dtype=torch.float64, device=device).contiguous(),
        'y_val': torch.tensor(y_val, dtype=torch.float64, device=device).contiguous(),
        'x_test': torch.tensor(x_test_scaled, dtype=torch.float64, device=device).contiguous(),
        'y_test': torch.tensor(y_test, dtype=torch.float64, device=device).contiguous(),
        'x_scaler': x_scaler,
        'use_exact': use_exact,
        'train_scenarios': train_scenarios,
        'val_scenarios': val_scenarios,
        'test_scenarios': test_scenarios
    }
    
    return data


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_classifier(model, likelihood, x_eval, y_eval, device='cpu', dataset_name='eval'):
    """
    Evaluate classifier on a dataset.
    
    Returns dict with metrics: accuracy, precision, recall, f1, auroc, confusion_matrix
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get latent function predictions
        f_pred = model(x_eval)
        f_mean = f_pred.mean
        
        # Get probability predictions via likelihood
        pred_probs = likelihood(f_pred).probs.cpu().numpy()
        pred_labels = (pred_probs >= 0.5).astype(int)
    
    y_true = y_eval.cpu().numpy().astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_true, pred_labels),
        'precision': precision_score(y_true, pred_labels, zero_division=0),
        'recall': recall_score(y_true, pred_labels, zero_division=0),
        'f1': f1_score(y_true, pred_labels, zero_division=0),
        'auroc': roc_auc_score(y_true, pred_probs) if len(np.unique(y_true)) > 1 else float('nan'),
        'confusion_matrix': confusion_matrix(y_true, pred_labels).tolist(),
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_predicted_positive': int(pred_labels.sum())
    }
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    return metrics


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_classifier_exact(data, n_epochs=100, learning_rate=0.01, device='cpu',
                           model_path=None, logger=None):
    """Train Exact GP classifier."""
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().double().to(device)
    model = H2DispersionClassifierExactGP(x_train, y_train, likelihood).double().to(device)
    
    model.train()
    likelihood.train()
    
    # Use variational ELBO for Bernoulli (Laplace approximation can be unstable)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nModel: ExactGP Classifier (KeOps RBF)")
    print(f"Training points: {len(x_train):,}")
    
    history = {
        'train_epoch_loss': [],
        'epochs': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auroc': []
    }
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    start_time = time_module.time()
    best_val_f1 = 0.0
    best_model_state = None
    best_likelihood_state = None
    
    iterator = tqdm(range(n_epochs), desc="Training ExactGP Classifier")
    for epoch in iterator:
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        history['train_epoch_loss'].append(loss_val)
        history['epochs'].append(epoch)
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate_classifier(model, likelihood, x_val, y_val, device=device, dataset_name='Validation')
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auroc'].append(val_metrics['auroc'])
            
            iterator.set_postfix(loss=f"{loss_val:.4f}", val_f1=f"{val_metrics['f1']:.4f}")
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_likelihood_state = {k: v.cpu().clone() for k, v in likelihood.state_dict().items()}
        else:
            iterator.set_postfix(loss=f"{loss_val:.4f}")
    
    training_time = time_module.time() - start_time
    
    # Restore best model
    if best_model_state is not None:
        print(f"\nRestoring best model (val F1={best_val_f1:.4f})")
        model.load_state_dict(best_model_state)
        likelihood.load_state_dict(best_likelihood_state)
    
    return model, likelihood, history, training_time


def train_classifier_vnngp(data, n_inducing=1000, k=64, training_batch_size=1024,
                           n_epochs=100, learning_rate=0.005, device='cpu',
                           model_path=None, logger=None):
    """Train VNNGP classifier."""
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    
    # Select inducing points via k-means
    inducing_points = select_inducing_points(x_train.cpu().numpy(), n_inducing=n_inducing, method='kmeans')
    inducing_points = inducing_points.to(device)
    
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().double().to(device)
    model = H2DispersionClassifierVNNGP(
        inducing_points=inducing_points, k=k, training_batch_size=training_batch_size
    ).double().to(device)
    
    num_batches = model.variational_strategy._total_training_batches
    model.train()
    likelihood.train()
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nModel: VNNGP Classifier (KeOps RBF)")
    print(f"Training points: {len(x_train):,}")
    print(f"Inducing points: {n_inducing}")
    print(f"K (neighbors): {k}")
    print(f"Training batches: {num_batches}")
    
    history = {
        'train_epoch_loss': [],
        'epochs': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auroc': []
    }
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    start_time = time_module.time()
    best_val_f1 = 0.0
    best_model_state = None
    best_likelihood_state = None
    
    iterator = tqdm(range(n_epochs), desc="Training VNNGP Classifier")
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
        history['train_epoch_loss'].append(epoch_loss)
        history['epochs'].append(epoch)
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate_classifier(model, likelihood, x_val, y_val, device=device, dataset_name='Validation')
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auroc'].append(val_metrics['auroc'])
            
            iterator.set_postfix(loss=f"{epoch_loss:.4f}", val_f1=f"{val_metrics['f1']:.4f}")
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_likelihood_state = {k: v.cpu().clone() for k, v in likelihood.state_dict().items()}
                if model_path is not None:
                    best_path = model_path + "_best"
                    save_checkpoint(model, likelihood, history, best_path, model_type='vnngp_classifier')
        else:
            iterator.set_postfix(loss=f"{epoch_loss:.4f}")
        
        if model_path is not None:
            path = model_path + str(epoch)
            save_checkpoint(model, likelihood, history, path, model_type='vnngp_classifier')
    
    training_time = time_module.time() - start_time
    
    # Restore best model
    if best_model_state is not None:
        print(f"\nRestoring best model (val F1={best_val_f1:.4f})")
        model.load_state_dict(best_model_state)
        likelihood.load_state_dict(best_likelihood_state)
    
    return model, likelihood, history, training_time


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def train_h2_dispersion_classifier(
    df,
    model_type='VNNGP',
    n_inducing=1000,
    k=64,
    training_batch_size=1024,
    n_epochs=100,
    learning_rate=0.005,
    device='cpu',
    model_path=None,
    max_exact_points=20000
):
    """
    Train H2 dispersion sensor activity classifier.
    
    Args:
        df: Full dataframe with columns time, mass_flow, y, z, active, split, scenario
        model_type: 'ExactGP' or 'VNNGP'
        n_inducing: Number of inducing points for VNNGP
        k: Number of nearest neighbors for VNNGP
        training_batch_size: Training batch size for VNNGP
        n_epochs: Training epochs
        learning_rate: Adam learning rate
        device: 'cpu' or 'cuda'
        model_path: Path to save model checkpoints
        max_exact_points: Max train points for ExactGP (falls back to VNNGP if exceeded)
    
    Returns:
        model, likelihood, history, metrics dict
    """
    # Prepare data
    data = prepare_classifier_data(df, device=device, max_exact_points=max_exact_points)
    
    # Override model type if ExactGP is infeasible
    if model_type == 'ExactGP' and not data['use_exact']:
        print(f"\nSwitching from ExactGP to VNNGP (dataset too large)")
        model_type = 'VNNGP'
    
    # Initialize logger
    logger = ExperimentLogger(log_dir='experiments')
    logger.log_training_start({
        'model_type': model_type,
        'n_train': len(data['x_train']),
        'n_val': len(data['x_val']),
        'n_test': len(data['x_test']),
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'device': device
    })
    
    # Train model
    if model_type == 'ExactGP':
        model, likelihood, history, training_time = train_classifier_exact(
            data, n_epochs=n_epochs, learning_rate=learning_rate,
            device=device, model_path=model_path, logger=logger
        )
    elif model_type == 'VNNGP':
        model, likelihood, history, training_time = train_classifier_vnngp(
            data, n_inducing=n_inducing, k=k, training_batch_size=training_batch_size,
            n_epochs=n_epochs, learning_rate=learning_rate,
            device=device, model_path=model_path, logger=logger
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on all splits
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    train_metrics = evaluate_classifier(model, likelihood, data['x_train'], data['y_train'],
                                        device=device, dataset_name='Train')
    val_metrics = evaluate_classifier(model, likelihood, data['x_val'], data['y_val'],
                                      device=device, dataset_name='Validation')
    test_metrics = evaluate_classifier(model, likelihood, data['x_test'], data['y_test'],
                                       device=device, dataset_name='Test')
    
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    logger.log_training_end(
        best_epoch=history['epochs'][-1] if history['epochs'] else 0,
        best_val_loss=history['train_epoch_loss'][-1] if history['train_epoch_loss'] else 0,
        total_time=training_time
    )
    
    # Save final model
    if model_path:
        save_checkpoint(model, likelihood, history, model_path, model_type=f'{model_type}_classifier')
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['epochs'], history['train_epoch_loss'], label='Train Loss', marker='s', markersize=3)
    if history['val_f1']:
        val_epochs = history['epochs'][9::10][:len(history['val_f1'])]
        plt.plot(val_epochs, history['val_f1'], label='Val F1', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / F1')
    plt.title(f'Classifier Training ({model_type})')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig('classifier_training.png', dpi=150)
    print("\nSaved: classifier_training.png")
    
    print(f"\nTraining complete! Duration: {training_time/60:.2f} minutes")
    
    return model, likelihood, history, all_metrics


# ---------------------------------------------------------------------------
# INFERENCE HELPER
# ---------------------------------------------------------------------------

def predict_active(model, likelihood, x_input, device='cpu'):
    """
    Predict probability of sensor being active.
    
    Args:
        model: Trained classifier model
        likelihood: Model likelihood
        x_input: Input tensor [n, 4] (time, mass_flow, y, z) - MUST BE SCALED
        device: 'cpu' or 'cuda'
    
    Returns:
        probs: Probability of active [n]
        labels: Binary predictions [n]
    """
    model.eval()
    likelihood.eval()
    
    if not isinstance(x_input, torch.Tensor):
        x_input = torch.tensor(x_input, dtype=torch.float64, device=device).contiguous()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred = model(x_input)
        probs = likelihood(f_pred).probs.cpu().numpy()
    
    labels = (probs >= 0.5).astype(int)
    return probs, labels


if __name__ == '__main__':
    # Quick test
    print("Loading data...")
    df = pd.read_csv('data/unified_raw_cut_off_2d.csv')
    print(f"Loaded {len(df)} rows")
    
    model, likelihood, history, metrics = train_h2_dispersion_classifier(
        df=df,
        model_type='VNNGP',
        n_inducing=500,
        k=64,
        training_batch_size=1024,
        n_epochs=10,
        learning_rate=0.005,
        device='cuda:0',
        model_path='models/h2_classifier_vnngp.pth'
    )
