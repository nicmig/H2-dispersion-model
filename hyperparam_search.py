#!/usr/bin/env python3
"""
Hyperparameter search for Stage 1 MLP ensemble.
Tests combinations of n_ensemble and learning_rate.
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from early_time_features import build_dataset_temporal
from run_stage1_massflow_estimator import ImprovedMLP, compute_sample_weights, train_torch_model
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def loeo_cv_fast(
    X: np.ndarray,
    y: np.ndarray,
    exp_ids: np.ndarray,
    n_ensemble: int,
    lr: float,
    n_epochs: int = 200,
    batch_size: int = 32,
    weight_decay: float = 5e-4,
    patience: int = 40,
    loss_type: str = "mae",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> dict:
    """Fast LOEO-CV with specified hyperparameters."""
    unique_exps = np.unique(exp_ids)
    all_y_true, all_y_pred = [], []
    fold_maes = []

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

            model = ImprovedMLP(input_dim=X.shape[1], dropout=0.2)
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
                pred = model(x_t).cpu().numpy()
            ensemble_preds.append(pred)

        y_pred = np.mean(ensemble_preds, axis=0)
        mae = float(np.mean(np.abs(y[mask_test] - y_pred)))
        fold_maes.append(mae)
        all_y_true.extend(y[mask_test].tolist())
        all_y_pred.extend(y_pred.tolist())

        if verbose:
            print(f"  Fold {idx+1:2d}/{len(unique_exps)} | {str(holdout):15s} | MAE={mae:.4f}")

    overall_mae = float(np.mean(np.abs(np.array(all_y_true) - np.array(all_y_pred))))
    overall_mape = float(np.mean(np.abs((np.array(all_y_true) - np.array(all_y_pred)) / (np.abs(np.array(all_y_true)) + 1e-8))) * 100)
    return {
        "overall_mae": overall_mae,
        "overall_mape": overall_mape,
        "fold_maes": fold_maes,
        "mean_fold_mae": float(np.mean(fold_maes)),
        "std_fold_mae": float(np.std(fold_maes)),
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = Path("data/unified_raw_two_modes.csv")
    df = pd.read_csv(data_path)

    print("Building dataset...")
    X_rf, X_mlp, X_combined, y, exp_ids, times = build_dataset_temporal(
        df, time_max=10.0, n_timesteps=3, active_threshold=0.009
    )
    print(f"  Combined features: {X_combined.shape}")

    # Hyperparameter grid
    ensemble_sizes = [3, 5, 7, 10]
    learning_rates = [1e-4, 5e-4, 1e-3]

    results = []
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"{'Ensemble':>8} | {'LR':>10} | {'MAE':>8} | {'MAPE':>8} | {'Fold Std':>8} | {'Time'}")
    print("-" * 70)

    for n_ens in ensemble_sizes:
        for lr in learning_rates:
            import time as time_mod
            t0 = time_mod.time()

            res = loeo_cv_fast(
                X_combined, y, exp_ids,
                n_ensemble=n_ens, lr=lr,
                n_epochs=200, patience=40,
                device=device, verbose=False,
            )
            elapsed = time_mod.time() - t0

            results.append({
                "n_ensemble": n_ens,
                "lr": lr,
                "mae": res["overall_mae"],
                "mape": res["overall_mape"],
                "mean_fold_mae": res["mean_fold_mae"],
                "std_fold_mae": res["std_fold_mae"],
                "time_sec": elapsed,
            })

            print(f"{n_ens:>8} | {lr:>10.4f} | {res['overall_mae']:>8.4f} | {res['overall_mape']:>7.1f}% | {res['std_fold_mae']:>8.4f} | {elapsed:>5.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print("RANKED BY MAE")
    print("=" * 70)
    results_sorted = sorted(results, key=lambda x: x["mae"])
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. ensemble={r['n_ensemble']}, lr={r['lr']:.4f} → MAE={r['mae']:.4f}, MAPE={r['mape']:.1f}%")

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE heatmap
    mae_matrix = np.zeros((len(ensemble_sizes), len(learning_rates)))
    for r in results:
        i = ensemble_sizes.index(r["n_ensemble"])
        j = learning_rates.index(r["lr"])
        mae_matrix[i, j] = r["mae"]

    ax = axes[0]
    im = ax.imshow(mae_matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in learning_rates])
    ax.set_yticks(range(len(ensemble_sizes)))
    ax.set_yticklabels(ensemble_sizes)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Ensemble Size")
    ax.set_title("MAE (lower is better)")
    for i in range(len(ensemble_sizes)):
        for j in range(len(learning_rates)):
            ax.text(j, i, f"{mae_matrix[i, j]:.4f}", ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax)

    # MAPE heatmap
    mape_matrix = np.zeros((len(ensemble_sizes), len(learning_rates)))
    for r in results:
        i = ensemble_sizes.index(r["n_ensemble"])
        j = learning_rates.index(r["lr"])
        mape_matrix[i, j] = r["mape"]

    ax = axes[1]
    im = ax.imshow(mape_matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in learning_rates])
    ax.set_yticks(range(len(ensemble_sizes)))
    ax.set_yticklabels(ensemble_sizes)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Ensemble Size")
    ax.set_title("MAPE % (lower is better)")
    for i in range(len(ensemble_sizes)):
        for j in range(len(learning_rates)):
            ax.text(j, i, f"{mape_matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    viz_dir = Path("validation_viz")
    viz_dir.mkdir(exist_ok=True)
    plot_path = viz_dir / "hyperparam_search_ensemble_lr.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")

    # Save JSON
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / f"hyperparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        import json
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "best": results_sorted[0],
        }, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
