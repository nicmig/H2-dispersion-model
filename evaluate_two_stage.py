#!/usr/bin/env python3
"""
Two-stage end-to-end evaluation on the held-out test experiments.

Stage 1: MLP ensemble estimates the H2 leakage mass flow (g/s) from early-time
         sensor observations (<= 10 s after release onset). The saved ensemble
         (models/stage1_ensemble.pth, written by massflow_estimator.py) is
         loaded; if missing, it is trained on all non-test experiments and
         saved. Use --retrain-stage1 after changing the dataset.

Stage 2: a trained GP (SVGP, Beta or Gaussian likelihood) forecasts the
         spatiotemporal H2 concentration field, using the Stage-1 mass-flow
         estimate instead of the true mass flow.

Reports MAE/RMSE of the predicted H2 volume fraction on the held-out test set,
both for the full two-stage pipeline and for an "oracle" run where the GP
receives the true mass flow (to isolate the error introduced by Stage 1).

Usage:
    python evaluate_two_stage.py
    python evaluate_two_stage.py --gp-checkpoint models/a.pth_best models/b.pth_best
    python evaluate_two_stage.py --retrain-stage1 --device cpu
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_dataset import HELD_OUT_TEST_EXPERIMENTS
from early_time_features import build_dataset_temporal
from massflow_estimator import (ImprovedMLP, compute_metrics,
                                train_final_ensemble, save_ensemble, load_ensemble)
from h2_dispersion_gp import load_model, evaluate_gp_model

DEFAULT_CSV = "data/unified_preprocessed_9e-3.csv"
DEFAULT_GP_CKPT = "models/approximate_indvAdditive_svgp_rbf_6000_beta_unified_preprocessed_9e-3_200_lr85e-2.pth_best"
DEFAULT_STAGE1_CACHE = "models/stage1_ensemble.pth"


# ---------------------------------------------------------------------------
# Stage 1: mass-flow ensemble
# ---------------------------------------------------------------------------

def load_or_train_stage1(cache_path, X, y, exp_ids, n_ensemble, device, retrain=False):
    """Load the saved Stage 1 ensemble, or train and save it if missing."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not retrain:
        print(f"Loading Stage 1 ensemble from {cache_path}")
        return load_ensemble(cache_path)

    ensemble = train_final_ensemble(X, y, exp_ids, n_ensemble=n_ensemble, device=device)
    save_ensemble(ensemble, cache_path)
    return ensemble


def predict_mass_flow(ensemble, X, device):
    """Ensemble-mean mass-flow prediction in g/s (clamped at 0)."""
    X_scaled = ensemble["scaler"].transform(X)
    x_t = torch.from_numpy(X_scaled).float().to(device)
    preds = []
    with torch.no_grad():
        for state in ensemble["state_dicts"]:
            model = ImprovedMLP(input_dim=ensemble["input_dim"], dropout=0.2)
            model.load_state_dict(state)
            model.to(device).eval()
            preds.append(model(x_t).cpu().numpy())
    return np.maximum(np.mean(preds, axis=0), 0.0)


# ---------------------------------------------------------------------------
# Stage 2: GP evaluation
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_with_gp_model(model, likelihood, checkpoint, df_test, device):
    """Evaluate with evaluate_gp_model from h2_dispersion_gp.py.

    Runs per test experiment to bound GPU memory (evaluate_gp_model predicts
    in a single shot, which needs an N x N covariance matrix), then aggregates.
    'mass_flow' in df_test must already be set (true or Stage-1 estimate).
    """
    y_pred_all, y_true_all, active_all = [], [], []
    nlls, ns = [], []
    per_exp = {}
    for exp_id, g in df_test.groupby("experiment_id", sort=True):
        m = evaluate_gp_model(model, likelihood, g, device=device,
                              x_scaler=checkpoint["x_scaler"],
                              y_scaler=checkpoint.get("y_scaler"))
        per_exp[exp_id] = regression_metrics(m["targets"], m["predictions"])
        y_pred_all.append(m["predictions"])
        y_true_all.append(m["targets"])
        active_all.append(g["active"].values if "active" in g.columns
                          else np.ones(len(g), dtype=bool))
        nlls.append(m["nll"])
        ns.append(len(g))

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    active = np.concatenate(active_all)
    return {
        "all": regression_metrics(y_true, y_pred),
        "active_only": regression_metrics(y_true[active], y_pred[active]),
        "nll": float(np.average(nlls, weights=ns)),
        "per_experiment": per_exp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-stage end-to-end evaluation.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Preprocessed dataset CSV.")
    parser.add_argument("--gp-checkpoint", nargs="+", default=[DEFAULT_GP_CKPT],
                        help="Stage 2 GP checkpoint(s) (.pth / .pth_best).")
    parser.add_argument("--stage1-cache", default=DEFAULT_STAGE1_CACHE,
                        help="Where to cache the re-trained Stage 1 ensemble.")
    parser.add_argument("--retrain-stage1", action="store_true",
                        help="Ignore the cache and retrain the Stage 1 ensemble.")
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Dataset: {args.csv}")

    df = pd.read_csv(args.csv)
    df_test = df[df["split"] == "test"].copy()
    print(f"Loaded {len(df):,} rows; test set: {len(df_test):,} rows, "
          f"experiments: {sorted(df_test['experiment_id'].unique())}")

    # ------------------------------------------------------------------
    # Stage 1: estimate mass flow for each test experiment
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: mass-flow estimation (early-time MLP ensemble)")
    print("=" * 60)
    _, X_combined, y, exp_ids, _ = build_dataset_temporal(
        df, time_max=10.0, n_timesteps=3, active_threshold=0.009,
        time_col="time_since_release",
    )
    mask_test = np.isin(exp_ids, HELD_OUT_TEST_EXPERIMENTS)
    print(f"Early-time samples: {mask_test.sum()} test / {(~mask_test).sum()} train")

    ensemble = load_or_train_stage1(
        args.stage1_cache, X_combined, y, exp_ids, args.n_ensemble, args.device,
        retrain=args.retrain_stage1,
    )

    mf_pred_samples = predict_mass_flow(ensemble, X_combined[mask_test], args.device)
    stage1 = pd.DataFrame({
        "experiment_id": exp_ids[mask_test],
        "mass_flow_true": y[mask_test],
        "mass_flow_pred": mf_pred_samples,
    })
    # One leakage-rate estimate per experiment: mean over its early-time samples.
    mf_per_exp = stage1.groupby("experiment_id").agg(
        mass_flow_true=("mass_flow_true", "first"),
        mass_flow_pred=("mass_flow_pred", "mean"),
        n_samples=("mass_flow_pred", "size"),
    )
    s1 = compute_metrics(mf_per_exp["mass_flow_true"].values, mf_per_exp["mass_flow_pred"].values)
    print("\nStage 1 per-experiment mass-flow estimates (g/s):")
    print(mf_per_exp.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nStage 1 (per-experiment): MAE={s1['MAE']:.4f} g/s, "
          f"RMSE={s1['RMSE']:.4f} g/s, MAPE={s1['MAPE']:.1f}%")

    # Two-stage test frame: mass_flow overwritten with the Stage-1 estimate.
    df_two_stage = df_test.copy()
    df_two_stage["mass_flow"] = df_two_stage["experiment_id"].map(mf_per_exp["mass_flow_pred"])

    # ------------------------------------------------------------------
    # Stage 2: GP forecast with estimated vs true mass flow
    # ------------------------------------------------------------------
    all_results = {}
    for ckpt_path in args.gp_checkpoint:
        print("\n" + "=" * 60)
        print(f"STAGE 2: GP dispersion forecast — {ckpt_path}")
        print("=" * 60)
        model, likelihood, checkpoint = load_model(ckpt_path, device=args.device)

        res_two_stage = evaluate_with_gp_model(model, likelihood, checkpoint, df_two_stage, args.device)
        res_oracle = evaluate_with_gp_model(model, likelihood, checkpoint, df_test, args.device)

        print("\nRESULTS (H2 volume fraction; multiply by 100 for %):")
        for name, res in [("Two-stage (estimated mass flow)", res_two_stage),
                          ("Oracle   (true mass flow)     ", res_oracle)]:
            print(f"{name}: "
                  f"MAE={res['all']['mae']:.6f} RMSE={res['all']['rmse']:.6f} (all rows) | "
                  f"MAE={res['active_only']['mae']:.6f} RMSE={res['active_only']['rmse']:.6f} (active only) | "
                  f"NLL={res['nll']:.4f}")
        print("Two-stage per-experiment:")
        for exp_id, m in res_two_stage["per_experiment"].items():
            print(f"  {exp_id}: MAE={m['mae']:.6f} RMSE={m['rmse']:.6f}")

        all_results[ckpt_path] = {
            "two_stage": res_two_stage,
            "oracle_true_mass_flow": res_oracle,
        }

        del model, likelihood
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "timestamp": datetime.now().isoformat(),
        "csv": args.csv,
        "stage1_cache": args.stage1_cache,
        "test_experiments": sorted(df_test["experiment_id"].unique()),
        "stage1": {
            "per_experiment": mf_per_exp.reset_index().to_dict(orient="records"),
            "metrics_g_per_s": s1,
        },
        "stage2": all_results,
    }
    out_path = Path("experiments") / f"two_stage_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
