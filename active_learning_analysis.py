#!/usr/bin/env python3
"""
Active Learning Analysis for Stage 1 Mass Flow Estimation.

Trains an ensemble MLP using LOEO-CV and computes epistemic uncertainty
(ensemble standard deviation) for each out-of-sample prediction.
Groups uncertainty by mass flow, fits a smooth uncertainty curve, and
recommends mass flows for additional CFD simulations.
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple, List, Optional

from early_time_features import build_dataset_temporal
from run_stage1_massflow_estimator import ImprovedMLP, compute_sample_weights, train_torch_model


# ---------------------------------------------------------------------------
# LOEO-CV with per-sample ensemble uncertainty
# ---------------------------------------------------------------------------

def loeo_cv_with_uncertainty(
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
    """
    Leave-one-experiment-out CV that returns per-sample ensemble mean & std.
    """
    unique_exps = np.unique(exp_ids)
    all_y_true, all_y_mean, all_y_std, all_mass_flows, all_exp = [], [], [], [], []
    fold_results = []

    for idx, holdout in enumerate(unique_exps):
        mask_train = exp_ids != holdout
        mask_test = exp_ids == holdout

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask_train])
        X_test = scaler.transform(X[mask_test])

        sample_weights = compute_sample_weights(y[mask_train])
        ensemble_preds = []

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

        ensemble_preds = np.array(ensemble_preds)  # (n_ensemble, n_test)
        y_mean = ensemble_preds.mean(axis=0)
        y_std = ensemble_preds.std(axis=0)

        mae = float(np.mean(np.abs(y[mask_test] - y_mean)))
        rmse = float(np.sqrt(np.mean((y[mask_test] - y_mean) ** 2)))
        mape = float(np.mean(np.abs((y[mask_test] - y_mean) / (np.abs(y[mask_test]) + 1e-8))) * 100)

        fold_results.append({
            "holdout": str(holdout),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "mean_uncertainty": float(y_std.mean()),
            "max_uncertainty": float(y_std.max()),
        })

        all_y_true.extend(y[mask_test].tolist())
        all_y_mean.extend(y_mean.tolist())
        all_y_std.extend(y_std.tolist())
        all_mass_flows.extend(y[mask_test].tolist())
        all_exp.extend([str(holdout)] * len(y_mean))

        if verbose:
            print(f"Fold {idx+1:2d}/{len(unique_exps)} | {str(holdout):15s} | MAE={mae:.4f} | Uncertainty={y_std.mean():.4f}")

    return {
        "folds": fold_results,
        "y_true": np.array(all_y_true),
        "y_mean": np.array(all_y_mean),
        "y_std": np.array(all_y_std),
        "mass_flows": np.array(all_mass_flows),
        "exp_ids": np.array(all_exp),
    }


# ---------------------------------------------------------------------------
# Uncertainty modeling
# ---------------------------------------------------------------------------

def compute_uncertainty_by_mass_flow(
    mass_flows: np.ndarray,
    uncertainties: np.ndarray,
    min_samples: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group uncertainties by unique mass flow values.
    Returns: unique_mass_flows, mean_uncertainty, sample_counts
    """
    unique_mf = np.unique(mass_flows)
    mean_unc = []
    counts = []

    for mf in unique_mf:
        mask = np.abs(mass_flows - mf) < 1e-6
        if mask.sum() >= min_samples:
            mean_unc.append(uncertainties[mask].mean())
            counts.append(int(mask.sum()))
        else:
            mean_unc.append(np.nan)
            counts.append(int(mask.sum()))

    return unique_mf, np.array(mean_unc), np.array(counts)


def fit_uncertainty_gp(
    mass_flows: np.ndarray,
    uncertainties: np.ndarray,
    mass_flow_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a GP to uncertainty as a function of mass flow.
    Returns: mean_prediction, std_prediction on the grid.
    """
    # Reshape for sklearn
    X = mass_flows.reshape(-1, 1)
    y = uncertainties

    # Kernel: smooth RBF + noise
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.3, (1e-2, 1.0)) + WhiteKernel(1e-5, (1e-10, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3, normalize_y=True)
    gp.fit(X, y)

    X_grid = mass_flow_grid.reshape(-1, 1)
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    return y_mean, y_std


def fit_uncertainty_spline(
    mass_flows: np.ndarray,
    uncertainties: np.ndarray,
    mass_flow_grid: np.ndarray,
    smoothing: float = 2.0,
) -> np.ndarray:
    """
    Fit a smoothing spline to uncertainty vs mass flow.
    """
    # Sort by mass flow
    order = np.argsort(mass_flows)
    x = mass_flows[order]
    y = uncertainties[order]

    # Use log-mass-flow for better spacing at low values
    # But keep linear for physical interpretability
    spline = UnivariateSpline(x, y, s=smoothing)
    return spline(mass_flow_grid)


def compute_data_density(
    mass_flows: np.ndarray,
    mass_flow_grid: np.ndarray,
    bandwidth: float = 0.05,
) -> np.ndarray:
    """
    Compute a smoothed data density (number of samples per mass flow).
    Uses Gaussian kernel smoothing.
    """
    density = np.zeros_like(mass_flow_grid)
    for mf in mass_flows:
        density += np.exp(-0.5 * ((mass_flow_grid - mf) / bandwidth) ** 2)
    # Normalize to [0, 1]
    density = density / (density.max() + 1e-8)
    return density


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_mass_flows(
    mass_flow_grid: np.ndarray,
    uncertainty_mean: np.ndarray,
    uncertainty_std: np.ndarray,
    data_density: np.ndarray,
    n_recommend: int = 10,
    strategy: str = "uncertainty_density",
) -> List[Dict]:
    """
    Recommend mass flows for new CFD simulations.

    Strategies:
    - "uncertainty": pure uncertainty sampling
    - "uncertainty_density": uncertainty weighted by (1 - density)
    - "margin": regions where uncertainty is high but GP is confident
    """
    if strategy == "uncertainty":
        score = uncertainty_mean
    elif strategy == "uncertainty_density":
        score = uncertainty_mean * (1.0 - data_density)
    elif strategy == "margin":
        score = uncertainty_mean + 2 * uncertainty_std
    else:
        score = uncertainty_mean

    # Exclude regions where we already have lots of data (density > 0.8)
    # unless uncertainty is extremely high
    mask = (data_density < 0.8) | (uncertainty_mean > np.percentile(uncertainty_mean, 90))
    score_masked = np.where(mask, score, -np.inf)

    top_indices = np.argsort(score_masked)[::-1][:n_recommend]
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "mass_flow": float(mass_flow_grid[idx]),
            "uncertainty": float(uncertainty_mean[idx]),
            "uncertainty_std": float(uncertainty_std[idx]),
            "data_density": float(data_density[idx]),
            "score": float(score_masked[idx]),
        })
    return recommendations


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_active_learning_analysis(
    results: Dict,
    mass_flow_grid: np.ndarray,
    uncertainty_mean: np.ndarray,
    uncertainty_std: np.ndarray,
    data_density: np.ndarray,
    recommendations: List[Dict],
    save_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mass_flows = results["mass_flows"]
    uncertainties = results["y_std"]
    y_true = results["y_true"]
    y_mean = results["y_mean"]

    # --- Panel 1: Uncertainty vs Mass Flow ---
    ax = axes[0, 0]
    # Scatter of raw per-sample uncertainties
    ax.scatter(mass_flows, uncertainties, alpha=0.3, s=10, c="gray", label="Per-sample uncertainty")
    # Smoothed GP curve
    ax.plot(mass_flow_grid, uncertainty_mean, "b-", lw=2, label="GP mean uncertainty")
    ax.fill_between(
        mass_flow_grid,
        uncertainty_mean - 2 * uncertainty_std,
        uncertainty_mean + 2 * uncertainty_std,
        alpha=0.2, color="blue", label="GP 95% CI"
    )
    # Recommended points
    rec_mf = [r["mass_flow"] for r in recommendations]
    rec_unc = [r["uncertainty"] for r in recommendations]
    ax.scatter(rec_mf, rec_unc, c="red", s=100, marker="*", zorder=5, label=f"Top-{len(recommendations)} recommendations")

    ax.set_xlabel("Mass Flow (kg/s)")
    ax.set_ylabel("Ensemble Std Dev (kg/s)")
    ax.set_title("Epistemic Uncertainty vs Mass Flow")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Data Density ---
    ax = axes[0, 1]
    # Histogram
    ax.hist(mass_flows, bins=30, alpha=0.5, color="green", edgecolor="black")
    # Smoothed density
    ax2 = ax.twinx()
    ax2.plot(mass_flow_grid, data_density, "r-", lw=2, label="Smoothed density")
    ax2.set_ylabel("Normalized Density", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.set_xlabel("Mass Flow (kg/s)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Training Data Distribution")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Acquisition Score ---
    ax = axes[1, 0]
    acquisition_score = uncertainty_mean * (1.0 - data_density)
    ax.plot(mass_flow_grid, acquisition_score, "purple", lw=2)
    ax.fill_between(mass_flow_grid, 0, acquisition_score, alpha=0.3, color="purple")
    ax.scatter(rec_mf, [uncertainty_mean * (1.0 - data_density) for uncertainty_mean, data_density in zip(
        [r["uncertainty"] for r in recommendations],
        [r["data_density"] for r in recommendations]
    )], c="red", s=100, marker="*", zorder=5)
    ax.set_xlabel("Mass Flow (kg/s)")
    ax.set_ylabel("Acquisition Score")
    ax.set_title("Active Learning Acquisition Score\n(Uncertainty × (1 - Density))")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Prediction Error vs Uncertainty ---
    ax = axes[1, 1]
    errors = np.abs(y_true - y_mean)
    ax.scatter(uncertainties, errors, alpha=0.3, s=10, c="orange")
    # Correlation
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    ax.set_xlabel("Ensemble Std Dev (kg/s)")
    ax.set_ylabel("Absolute Error (kg/s)")
    ax.set_title(f"Uncertainty vs Actual Error (r={corr:.3f})")
    ax.grid(True, alpha=0.3)

    # Add diagonal reference line
    max_val = max(uncertainties.max(), errors.max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y=x (perfect calibration)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved active learning plot to {save_path}")


def print_recommendations(recommendations: List[Dict]):
    print("\n" + "=" * 70)
    print("TOP RECOMMENDED MASS FLOWS FOR NEW CFD SIMULATIONS")
    print("=" * 70)
    print(f"{'Rank':>4} | {'Mass Flow (kg/s)':>16} | {'Uncertainty':>12} | {'Data Density':>12} | {'Score':>12}")
    print("-" * 70)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:>4} | {rec['mass_flow']:>16.4f} | {rec['uncertainty']:>12.4f} | {rec['data_density']:>12.3f} | {rec['score']:>12.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = Path("data/unified_raw_two_modes.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(data_path)

    print("\nBuilding temporal datasets (t <= 10s)...")
    X_rf, X_mlp, X_combined, y, exp_ids, times = build_dataset_temporal(
        df, time_max=10.0, n_timesteps=3, active_threshold=0.009
    )
    print(f"  Combined features: {X_combined.shape}")
    print(f"  Unique experiments: {len(np.unique(exp_ids))}")

    # ------------------------------------------------------------------
    # LOEO-CV with ensemble uncertainty
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running LOEO-CV with ensemble uncertainty (n_ensemble=7)")
    print("=" * 70)
    results = loeo_cv_with_uncertainty(
        X_combined, y, exp_ids,
        model_class=ImprovedMLP,
        model_kwargs={"input_dim": X_combined.shape[1], "dropout": 0.2},
        n_epochs=300, batch_size=32, lr=5e-4, weight_decay=5e-4, patience=50,
        loss_type="mae", n_ensemble=7,
        device=device, verbose=True,
    )

    overall_mae = float(np.mean(np.abs(results["y_true"] - results["y_mean"])))
    overall_mape = float(np.mean(np.abs((results["y_true"] - results["y_mean"]) / (np.abs(results["y_true"]) + 1e-8))) * 100)
    print(f"\nOverall LOEO-CV: MAE={overall_mae:.4f} kg/s, MAPE={overall_mape:.1f}%")
    print(f"Mean ensemble uncertainty: {results['y_std'].mean():.4f} kg/s")

    # ------------------------------------------------------------------
    # Uncertainty by mass flow
    # ------------------------------------------------------------------
    print("\nAnalyzing uncertainty by mass flow...")
    unique_mf, mean_unc, counts = compute_uncertainty_by_mass_flow(
        results["mass_flows"], results["y_std"], min_samples=1
    )

    print(f"\n{'Mass Flow':>12} | {'Samples':>8} | {'Mean Uncertainty':>16} | {'Status'}")
    print("-" * 60)
    for mf, cnt, unc in zip(unique_mf, counts, mean_unc):
        status = "HIGH" if unc > 0.15 else ("MED" if unc > 0.08 else "LOW")
        print(f"{mf:>12.4f} | {cnt:>8} | {unc:>16.4f} | {status}")

    # ------------------------------------------------------------------
    # Fit uncertainty model on grid
    # ------------------------------------------------------------------
    mass_flow_grid = np.linspace(0.01, 1.4, 200)

    # Use all per-sample points for GP (not just binned means)
    # This gives finer resolution
    uncertainty_mean, uncertainty_std = fit_uncertainty_gp(
        results["mass_flows"], results["y_std"], mass_flow_grid
    )

    data_density = compute_data_density(results["mass_flows"], mass_flow_grid, bandwidth=0.05)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    recommendations = recommend_mass_flows(
        mass_flow_grid, uncertainty_mean, uncertainty_std, data_density,
        n_recommend=10, strategy="uncertainty_density"
    )
    print_recommendations(recommendations)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    viz_dir = Path("validation_viz")
    viz_dir.mkdir(exist_ok=True)
    plot_path = viz_dir / "active_learning_analysis.png"
    plot_active_learning_analysis(
        results, mass_flow_grid, uncertainty_mean, uncertainty_std,
        data_density, recommendations, plot_path
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / f"active_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall_mae": overall_mae,
        "overall_mape": overall_mape,
        "mean_uncertainty": float(results["y_std"].mean()),
        "max_uncertainty": float(results["y_std"].max()),
        "uncertainty_by_mass_flow": [
            {"mass_flow": float(mf), "samples": int(cnt), "mean_uncertainty": float(unc)}
            for mf, cnt, unc in zip(unique_mf, counts, mean_unc)
        ],
        "recommendations": recommendations,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
