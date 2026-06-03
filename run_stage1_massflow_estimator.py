#!/usr/bin/env python3
"""
Stage 1: Early-time mass flow estimation using temporal features.

Trains and evaluates:
1. Random Forest on handcrafted features + temporal derivatives
2. Improved MLP on log-transformed sensors + handcrafted features

Uses leave-one-experiment-out cross-validation.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, List, Optional

from early_time_features import build_dataset_temporal, ALL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Improved MLP with batch norm and skip connections
# ---------------------------------------------------------------------------

class ImprovedMLP(nn.Module):
    """
    MLP on combined features: log-transformed sensors + handcrafted features.
    Input: 87 log-sensors + 15 handcrafted features = 102 elements.
    """
    def __init__(self, input_dim: int = 102, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_sample_weights(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Compute inverse-frequency weights based on mass flow quantile bins."""
    bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6  # Ensure max value falls in last bin
    bin_indices = np.digitize(y, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    # Avoid division by zero
    bin_counts = np.maximum(bin_counts, 1)
    weights = 1.0 / bin_counts[bin_indices]
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sample_weights: Optional[np.ndarray] = None,
    n_epochs: int = 300,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    patience: int = 50,
    loss_type: str = "mae",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, Dict]:
    """Train a PyTorch model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5, min_lr=1e-6)

    if loss_type == "mae":
        base_criterion = nn.L1Loss(reduction='none')
    elif loss_type == "huber":
        base_criterion = nn.HuberLoss(delta=0.1, reduction='none')
    else:
        base_criterion = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        batch_idx = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss_per_sample = base_criterion(pred, yb)
            if sample_weights is not None:
                # Get weights for this batch
                start_idx = batch_idx * train_loader.batch_size
                end_idx = start_idx + len(yb)
                w = torch.from_numpy(sample_weights[start_idx:end_idx]).float().to(device)
                loss = (loss_per_sample * w).mean()
            else:
                loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            batch_idx += 1

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(base_criterion(pred, yb).mean().item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# LOEO-CV for PyTorch model with ensemble
# ---------------------------------------------------------------------------

def loeo_cv_torch(
    X: np.ndarray,
    y: np.ndarray,
    exp_ids: np.ndarray,
    model_class,
    model_kwargs: Dict,
    n_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    patience: int = 50,
    loss_type: str = "mae",
    n_ensemble: int = 7,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict:
    unique_exps = np.unique(exp_ids)
    fold_results = []
    all_y_true, all_y_pred, all_exp = [], [], []

    for idx, holdout in enumerate(unique_exps):
        mask_train = exp_ids != holdout
        mask_test = exp_ids == holdout

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask_train])
        X_test = scaler.transform(X[mask_test])

        # Ensemble: train n_ensemble models with different seeds
        ensemble_preds = []

        sample_weights = compute_sample_weights(y[mask_train])

        for ens_idx in range(n_ensemble):
            train_ds = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y[mask_train]).float(),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y[mask_test]).float(),
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = model_class(**model_kwargs)
            # Set different seed per ensemble member
            torch.manual_seed(42 + ens_idx)
            model, _ = train_torch_model(
                model, train_loader, val_loader,
                sample_weights=sample_weights,
                n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
                patience=patience, loss_type=loss_type, device=device,
            )

            model.eval()
            with torch.no_grad():
                x_t = torch.from_numpy(X_test).float().to(device)
                y_pred = model(x_t).cpu().numpy()
            ensemble_preds.append(y_pred)

        # Average ensemble predictions
        y_pred = np.mean(ensemble_preds, axis=0)

        m = compute_metrics(y[mask_test], y_pred)
        m["holdout"] = str(holdout)
        fold_results.append(m)
        all_y_true.extend(y[mask_test].tolist())
        all_y_pred.extend(y_pred.tolist())
        all_exp.extend([str(holdout)] * len(y_pred))

        if verbose:
            print(f"[MLP] Fold {idx+1}/{len(unique_exps)} | {holdout} | MAE={m['MAE']:.4f} | MAPE={m['MAPE']:.1f}%")

    overall = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    return {
        "folds": fold_results,
        "overall": overall,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "exp_ids": np.array(all_exp),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results_mlp: Dict, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axmin, axmax = 0, max(results_mlp["y_true"].max(), results_mlp["y_pred"].max()) * 1.1

    # Panel 1: Scatter predicted vs true
    ax = axes[0, 0]
    ax.scatter(results_mlp["y_true"], results_mlp["y_pred"], alpha=0.4, s=15, c="green", label=f"MLP (MAE={results_mlp['overall']['MAE']:.3f})")
    ax.plot([axmin, axmax], [axmin, axmax], "k--", lw=2)
    ax.set_xlabel("True Mass Flow (kg/s)")
    ax.set_ylabel("Predicted Mass Flow (kg/s)")
    ax.set_title("MLP: Predicted vs True Mass Flow")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-experiment MAE
    ax = axes[0, 1]
    exp_names = [f["holdout"] for f in results_mlp["folds"]]
    mlp_maes = [f["MAE"] for f in results_mlp["folds"]]

    x = np.arange(len(exp_names))
    ax.bar(x, mlp_maes, color="green", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAE (kg/s)")
    ax.set_title("Per-Experiment MAE")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: MLP scatter zoomed
    ax = axes[1, 0]
    ax.scatter(results_mlp["y_true"], results_mlp["y_pred"], alpha=0.4, s=15, c="green")
    ax.plot([axmin, axmax], [axmin, axmax], "k--", lw=2)
    ax.set_xlabel("True Mass Flow (kg/s)")
    ax.set_ylabel("Predicted Mass Flow (kg/s)")
    ax.set_title(f"MLP: MAE={results_mlp['overall']['MAE']:.4f}, RMSE={results_mlp['overall']['RMSE']:.4f}, MAPE={results_mlp['overall']['MAPE']:.1f}%")
    ax.grid(True, alpha=0.3)

    # Panel 4: Error distribution
    ax = axes[1, 1]
    mlp_err = results_mlp["y_pred"] - results_mlp["y_true"]
    ax.hist(mlp_err, bins=30, alpha=0.7, color="green", label=f"MLP (μ={np.mean(mlp_err):.3f})")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Prediction Error (kg/s)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = Path("data/unified_raw_two_modes.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(data_path)

    print("\nBuilding temporal datasets (t <= 10s)...")
    X_mlp, X_combined, y, exp_ids, _= build_dataset_temporal(df, time_max=10.0, n_timesteps=3, active_threshold=0.009)
    print(f"  MLP features: {X_mlp.shape}")
    print(f"  Combined features: {X_combined.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  Unique experiments: {len(np.unique(exp_ids))}")

    # ------------------------------------------------------------------
    # Improved MLP on combined features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MLP (log-sensors + handcrafted features, balanced, ensemble=3, MAE loss)")
    print("=" * 60)
    results_mlp = loeo_cv_torch(
        X_combined, y, exp_ids,
        model_class=ImprovedMLP,
        model_kwargs={"input_dim": X_combined.shape[1], "dropout": 0.2},
        n_epochs=300, batch_size=32, lr=5e-4, weight_decay=5e-4, patience=50,
        loss_type="mae", n_ensemble=7,
        device=device, verbose=True,
    )
    print(f"\nMLP Overall: MAE={results_mlp['overall']['MAE']:.4f}, RMSE={results_mlp['overall']['RMSE']:.4f}, MAPE={results_mlp['overall']['MAPE']:.1f}%")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    viz_dir = Path("validation_viz")
    viz_dir.mkdir(exist_ok=True)
    plot_path = viz_dir / "stage1_massflow_estimator.png"
    plot_results(results_mlp, plot_path)

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / f"stage1_massflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {"time_max": 10.0, "n_timesteps": 3, "n_samples": int(len(y)), "n_experiments": int(len(np.unique(exp_ids)))},
        "mlp_overall": results_mlp["overall"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
