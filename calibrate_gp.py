#!/usr/bin/env python3
"""
Calibration analysis of the GP dispersion model (beta likelihood) on the
held-out test set.

Draws MC samples from the full predictive distribution (latent GP posterior +
beta observation noise) at every test point and computes:

  - PICP (prediction interval coverage probability) at several nominal levels
  - a calibration / reliability curve (nominal vs empirical coverage over all
    central interval levels)
  - a PIT histogram (probability integral transform; uniform if calibrated)
  - mean prediction interval width (sharpness) for the 95% interval

Metrics are reported for all test rows and for exposed (active == 1) sensors;
the plots use the exposed rows, matching the GP's training distribution.

Usage:
    python calibrate_gp.py
    python calibrate_gp.py --checkpoint models/<other>.pth_best --device cpu
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from h2_dispersion_gp import load_model

DEFAULT_CKPT = "models/approximate_indvAdditive_svgp_rbf_6000_beta_unified_preprocessed_9e-3_200_lr85e-2.pth_best"
DEFAULT_CSV = "data/unified_preprocessed_9e-3.csv"
FIG_PATH = Path("visualizations/gp_calibration.pdf")

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

N_SAMPLES = 1000
PICP_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]


def predictive_samples(model, likelihood, x_scaler, df_test, device,
                       n_samples=N_SAMPLES, batch=4096):
    """MC samples from the predictive distribution, shape (n_samples, n_points)."""
    # Match the checkpoint's training features (see evaluate_gp_model):
    # relative coordinates if the scaler's x mean is ~0, raw otherwise.
    if abs(x_scaler.mean_[2]) < 0.1:
        cols = ["time_since_release", "mass_flow", "x_rel", "y_rel", "z_rel"]
    else:
        cols = ["time_since_release", "mass_flow", "x", "y", "z"]
    X_scaled = x_scaler.transform(df_test[cols].values)

    samples = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(n_samples):
        for s in range(0, len(X_scaled), batch):
            xb = torch.tensor(X_scaled[s:s + batch], dtype=torch.float64, device=device)
            dist = likelihood(model(xb))          # batch shape (n_samples, B)
            samples.append(dist.sample().cpu())   # (n_samples, B) observation draws
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.cat(samples, dim=1).numpy()


def coverage_table(samples, y_true, levels=PICP_LEVELS):
    """PICP and mean interval width per nominal central-coverage level."""
    rows = []
    for gamma in levels:
        alpha = (1.0 - gamma) / 2.0
        lower = np.quantile(samples, alpha, axis=0)
        upper = np.quantile(samples, 1.0 - alpha, axis=0)
        covered = (y_true >= lower) & (y_true <= upper)
        rows.append({"nominal": gamma, "empirical": float(covered.mean()),
                     "mean_width": float((upper - lower).mean())})
    return rows


def calibration_curve(samples, y_true):
    """Empirical coverage for all central levels in (0, 1) — for the reliability plot."""
    taus = np.linspace(0.01, 0.99, 99)          # quantile levels of the predictive CDF
    qs = np.quantile(samples, taus, axis=0)     # (99, n_points)
    cdf_at_y = (qs >= y_true[None, :]).mean(axis=1)  # empirical P(y <= Q(tau)); ideal: tau
    return taus, cdf_at_y


def pit_values(samples, y_true):
    """Probability integral transform: predictive CDF evaluated at the target."""
    return (samples <= y_true[None, :]).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="GP calibration analysis.")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, likelihood, checkpoint = load_model(args.checkpoint, device=args.device)
    df = pd.read_csv(args.csv)
    df_test = df[df["split"] == "test"].copy()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test rows: {len(df_test):,}")

    print(f"Drawing {N_SAMPLES} predictive samples per test point...")
    samples = predictive_samples(model, likelihood, checkpoint["x_scaler"], df_test, args.device)
    y_true = df_test["h2_volume_fraction"].values
    active = df_test["active"].values == 1

    # ------------------------------------------------------------------
    # PICP tables
    # ------------------------------------------------------------------
    results = {}
    for label, mask in [("all", np.ones(len(y_true), bool)), ("active_only", active)]:
        rows = coverage_table(samples[:, mask], y_true[mask])
        results[label] = rows
        print(f"\nPICP ({label}, n={mask.sum():,}):")
        print(f"  {'nominal':>8} {'empirical':>10} {'gap':>8} {'mean width':>11}")
        for r in rows:
            print(f"  {r['nominal']:>8.2f} {r['empirical']:>10.4f} "
                  f"{r['empirical'] - r['nominal']:>+8.3f} {r['mean_width']:>11.4f}")

    # ------------------------------------------------------------------
    # Calibration curve + PIT on exposed rows
    # ------------------------------------------------------------------
    taus, cdf_at_y = calibration_curve(samples[:, active], y_true[active])
    cal_error = float(np.mean(np.abs(cdf_at_y - taus)))
    pit = pit_values(samples[:, active], y_true[active])
    print(f"\nMean absolute calibration error (active rows): {cal_error:.4f}")
    print(f"PIT mean={pit.mean():.3f} (0.5 ideal), std={pit.std():.3f} "
          f"({np.sqrt(1/12):.3f} ideal)")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal calibration")
    ax.plot(taus, cdf_at_y, color="tab:blue", lw=2,
            label=f"GP (MAE$ cal$={cal_error:.3f})")
    for r in results["active_only"]:
        ax.scatter([r["nominal"]], [r["empirical"]], color="tab:red", zorder=3)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration curve (exposed sensors, test set)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(pit, bins=20, range=(0, 1), density=True, color="tab:blue",
            alpha=0.7, edgecolor="k")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform (ideal)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("density")
    ax.set_title("PIT histogram (exposed sensors, test set)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Calibration of the GP dispersion model (beta likelihood)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"\nSaved figure to {FIG_PATH}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "n_samples_per_point": N_SAMPLES,
        "picp": results,
        "calibration_error_active": cal_error,
        "pit_mean_active": float(pit.mean()),
        "pit_std_active": float(pit.std()),
    }
    out_path = Path("experiments") / f"gp_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
