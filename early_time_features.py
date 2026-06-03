"""
Feature extraction for early-time mass flow estimation.

Supports:
1. Handcrafted features (single snapshot + temporal derivatives)
2. Full 29-sensor vector over 3 consecutive timesteps (for MLP)
3. Combined vector: raw sensors + handcrafted features
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# Sensor positions from create_unified_raw.py
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

# Sensors ordered by y-position for MLP (preserves some spatial structure)
_y_sorted = sorted(SENSOR_POSITIONS.items(), key=lambda x: x[1][1])
SENSOR_ORDER = [sid for sid, _ in _y_sorted]

# Thresholds
ACTIVE_THRESHOLD = 0.009
FLOOR_VALUE = 1e-4  # Below threshold, sensors get this floor instead of exact zero

FEATURE_NAMES_SINGLE = [
    "time", "n_active", "max_conc", "mean_conc", "std_conc", "top3_mean",
    "weighted_y", "weighted_z", "spread_y", "spread_z",
]

FEATURE_NAMES_TEMPORAL = [
    "dn_active_dt", "dmax_conc_dt", "dmean_conc_dt", "dspread_z_dt",
    "dweighted_z_dt"
]

ALL_FEATURE_NAMES = FEATURE_NAMES_SINGLE + FEATURE_NAMES_TEMPORAL


def apply_floor(sensor_readings: Dict[int, float], threshold: float = ACTIVE_THRESHOLD, floor: float = FLOOR_VALUE) -> Dict[int, float]:
    """Replace values below threshold with floor value to ensure numerical stability."""
    return {sid: max(val, floor) if val < threshold else val for sid, val in sensor_readings.items()}


def extract_features(
    sensor_readings: Dict[int, float],
    time: float,
    sensor_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
    active_threshold: float = ACTIVE_THRESHOLD,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Extract handcrafted features from a single timestep's sensor observations."""
    if sensor_positions is None:
        sensor_positions = SENSOR_POSITIONS

    active_ids = []
    active_concs = []
    active_positions = []

    for sid, conc in sensor_readings.items():
        if sid not in sensor_positions:
            continue
        if conc > active_threshold:
            active_ids.append(sid)
            active_concs.append(conc)
            active_positions.append(sensor_positions[sid])

    active_concs = np.array(active_concs, dtype=np.float64)
    n_active = len(active_concs)

    if n_active == 0:
        return np.zeros(10, dtype=np.float64)

    active_positions = np.array(active_positions, dtype=np.float64)
    max_conc = float(np.max(active_concs))
    mean_conc = float(np.mean(active_concs))
    std_conc = float(np.std(active_concs)) if n_active > 1 else 0.0
    top3_mean = float(np.mean(np.partition(active_concs, -min(3, n_active))[-min(3, n_active):]))
    weights = active_concs / np.sum(active_concs)
    weighted_y = float(np.sum(weights * active_positions[:, 1]))
    weighted_z = float(np.sum(weights * active_positions[:, 2]))
    spread_y = float(np.var(active_positions[:, 1])) if n_active > 1 else 0.0
    spread_z = float(np.var(active_positions[:, 2])) if n_active > 1 else 0.0

    return np.array([
        time, float(n_active), max_conc, mean_conc, std_conc, top3_mean,
        weighted_y, weighted_z, spread_y, spread_z,
    ], dtype=np.float64)


def extract_temporal_features(
    readings_now: Dict[int, float],
    readings_prev: Dict[int, float],
    time: float,
    dt: float,
    active_threshold: float = ACTIVE_THRESHOLD,
) -> np.ndarray:
    """
    Extract features from current snapshot plus temporal derivatives.
    Returns vector of shape (15,) = 10 single + 5 temporal features.
    """
    feats_now = extract_features(readings_now, time, active_threshold=active_threshold)
    feats_prev = extract_features(readings_prev, time - dt, active_threshold=active_threshold)

    dt_safe = dt if dt > 1e-6 else 1.0
    dn_active_dt = (feats_now[1] - feats_prev[1]) / dt_safe
    dmax_conc_dt = (feats_now[2] - feats_prev[2]) / dt_safe
    dmean_conc_dt = (feats_now[3] - feats_prev[3]) / dt_safe
    dspread_z_dt = (feats_now[9] - feats_prev[9]) / dt_safe
    dweighted_z_dt = (feats_now[7] - feats_prev[7]) / dt_safe

    temporal = np.array([
        dn_active_dt, dmax_conc_dt, dmean_conc_dt,
        dspread_z_dt, dweighted_z_dt,
    ], dtype=np.float64)

    return np.concatenate([feats_now, temporal])


def build_sensor_vector(
    sensor_readings: Dict[int, float],
    sensor_order: Optional[List[int]] = None,
    floor: float = FLOOR_VALUE,
) -> np.ndarray:
    """
    Build a fixed-size 29-element vector from sensor readings, ordered by y-position.
    Missing sensors are filled with floor value (not zero, for numerical stability).
    """
    if sensor_order is None:
        sensor_order = SENSOR_ORDER
    return np.array([
        sensor_readings.get(sid, floor) if sensor_readings.get(sid, 0.0) >= floor else floor
        for sid in sensor_order
    ], dtype=np.float64)


def build_temporal_sensor_vectors(
    readings_list: List[Dict[int, float]],
    sensor_order: Optional[List[int]] = None,
    floor: float = FLOOR_VALUE,
) -> np.ndarray:
    """
    Build MLP input from 3 consecutive timesteps.
    Input: list of 3 sensor reading dicts [t_0, t_1, t_2] (oldest to newest).
    Returns shape (87,) = 29 sensors × 3 timesteps.
    """
    vectors = [build_sensor_vector(r, sensor_order, floor) for r in readings_list]
    return np.concatenate(vectors)


def log_transform_sensors(sensor_vector: np.ndarray, floor: float = FLOOR_VALUE) -> np.ndarray:
    """Log-transform sensor values to compress dynamic range."""
    return np.log(sensor_vector + 1.0)


def build_dataset_temporal(
    df,
    time_max: float = 10.0,
    n_timesteps: int = 3,
    active_threshold: float = ACTIVE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build datasets with temporal features and full sensor vectors.
    Uses n_timesteps consecutive snapshots (default 3).

    Returns
    -------
    X_rf : (n_samples, 15) — handcrafted + temporal features for Random Forest
    X_mlp : (n_samples, 87) — full sensor vectors for MLP
    X_combined : (n_samples, 104) — concatenated [log_sensors, handcrafted_features]
    y : (n_samples,) — mass flow
    exp_ids : (n_samples,) — experiment ids
    times : (n_samples,) — timestamps
    """
    import pandas as pd

    df = df[df["time"] <= time_max].copy()

    mlp_list = []
    combined_list = []
    mass_flows = []
    exp_ids = []
    times = []

    for exp_id in df["experiment_id"].unique():
        exp_df = df[df["experiment_id"] == exp_id].sort_values("time")
        unique_times = exp_df["time"].unique()

        for i in range(n_timesteps - 1, len(unique_times)):
            t_now = unique_times[i]

            # Collect n_timesteps consecutive groups
            groups = []
            valid = True
            for j in range(n_timesteps):
                t = unique_times[i - j]
                group = exp_df[exp_df["time"] == t]
                if len(group) < 29:
                    valid = False
                    break
                groups.append(group)

            if not valid:
                continue

            # Apply floor to all readings
            readings_list = []
            for group in reversed(groups):  # oldest to newest
                raw = dict(zip(group["sensor_id"].values, group["h2_volume_fraction"].values))
                readings_list.append(apply_floor(raw, threshold=active_threshold, floor=FLOOR_VALUE))

            # RF features: current snapshot + temporal derivative from most recent interval
            dt = unique_times[i] - unique_times[i - 1]
            rf_feats = extract_temporal_features(
                readings_list[-1], readings_list[-2], t_now, dt, active_threshold
            )

            # MLP input: 29 sensors × 3 timesteps
            mlp_feats = build_temporal_sensor_vectors(readings_list)

            # Combined: log-transformed sensors + handcrafted features
            log_sensors = log_transform_sensors(mlp_feats)
            combined = np.concatenate([log_sensors, rf_feats])

            mlp_list.append(mlp_feats)
            combined_list.append(combined)
            mass_flows.append(groups[0]["mass_flow"].iloc[0])
            exp_ids.append(exp_id)
            times.append(t_now)

    return (
        np.vstack(mlp_list),
        np.vstack(combined_list),
        np.array(mass_flows, dtype=np.float64),
        np.array(exp_ids),
        np.array(times, dtype=np.float64),
    )
