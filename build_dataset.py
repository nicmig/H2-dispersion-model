"""
Build unified datasets for the H2 dispersion pipeline.

Supports two dataset modes (selected in config.json):
- raw: combine resampled CFD and/or experimental data with minimal processing.
- preprocessed: raw preprocessing plus end-of-release cut-off and release-onset
  features (t_release, time_since_release, h2_lag_1, optional h2_initial).
"""

import pandas as pd
import numpy as np
from scipy import signal
import re
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CFD to Experiment mapping
CFD_EXP_MAPPING = {
    'A': ['23_FFI_P101_T00012', '23_FFI_P101_T00013'],
    'E': ['23_FFI_P101_T00003', '23_FFI_P101_T00004'],
    'F': ['23_FFI_P101_T00007'],
    'H': ['23_FFI_P101_T00011'],
    'O1': ['23_FFI_P101_T00027', '23_FFI_P101_T00028'],
    'O2': ['23_FFI_P101_T00029', '23_FFI_P101_T00030'],
}

CFD_MASS_FLOW = {
    'A': 0.086, 'B': 0.15, 'C': 0.20, 'D': 0.30, 'E': 0.48,
    'F': 0.74, 'G': 1.00, 'H': 1.27, 'O1': 0.091, 'O2': 0.45,
}

# Experimental mass flow start
FORTY_SEC_START = ['23_FFI_P101_T00003', '23_FFI_P101_T00005', 
                           '23_FFI_P101_T00006', '23_FFI_P101_T00009']

THIRTY_FIVE_SEC_START = ['23_FFI_P101_T00004', '23_FFI_P101_T00007', 
                           '23_FFI_P101_T00008', '23_FFI_P101_T00011',
                           '23_FFI_P101_T00012', '23_FFI_P101_T00013',
                           '23_FFI_P101_T00014', '23_FFI_P101_T00026',
                           '23_FFI_P101_T00027', '23_FFI_P101_T00027 2',
                           '23_FFI_P101_T00028', '23_FFI_P101_T00029',
                           '23_FFI_P101_T00030', '23_FFI_P101_T00031',
                           '23_FFI_P101_T00040', '23_FFI_P101_T00041',
                           '23_FFI_P101_T00042', '23_FFI_P101_T00044',
                           '23_FFI_P101_T00045']

# experimental cut-off when experiment is "dead" == leakage stops
THIRTY_SEC = ['23_FFI_P101_T00005']
SIXTY_SEC = ['23_FFI_P101_T00003', '23_FFI_P101_T00007', '23_FFI_P101_T00009', '23_FFI_P101_T00006', '23_FFI_P101_T00008']
NINETY_SEC = ['23_FFI_P101_T00011']
HUNDRED_TWENTY_SEC = ['23_FFI_P101_T00004', '23_FFI_P101_T00026', '23_FFI_P101_T00027', '23_FFI_P101_T00028', '23_FFI_P101_T00014', '23_FFI_P101_T00012', '23_FFI_P101_T00013', 
                      '23_FFI_P101_T00029', '23_FFI_P101_T00030', '23_FFI_P101_T00031']
TWO_HUNDRED_FORTY_SEC = ['23_FFI_P101_T00040', '23_FFI_P101_T00041', '23_FFI_P101_T00042', '23_FFI_P101_T00044', '23_FFI_P101_T00045']

# Splits
HELD_OUT_TEST_EXPERIMENTS = ['23_FFI_P101_T00006', '23_FFI_P101_T00011', 
                             '23_FFI_P101_T00045', '23_FFI_P101_T00040']

# Threshold for active/inactive sensor classification
# Values above this threshold are considered "active" (H2 detected)
ACTIVE_THRESHOLD = 0.009

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


def load_config(config_path: str = 'config.json') -> dict:
    """Load pipeline configuration from JSON.

    Defaults are provided so the script still runs if the config file is
    missing, but in normal use the repository ships with config.json at the
    root.
    """
    default_config = {
        "data_sources": {
            "include_experiments": True,
            "include_cfd": False,
        },
        "dataset_type": "raw",
        "release_onset": {
            "threshold": 0.009,
            "include_initial": False,
            "include_lag1": True,
        },
        "paths": {
            "data_dir": "data",
            "experiments_dir": "data/raw",
            "cfd_dir": "data/CFD",
            "raw_output_csv": "data/unified_raw.csv",
            "raw_output_summary": "data/unified_raw_summary.txt",
            "preprocessed_output_csv": "data/unified_preprocessed.csv",
            "preprocessed_output_summary": "data/unified_preprocessed_summary.txt",
        },
    }

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        # Merge nested structures rather than replacing wholesale
        for key in default_config:
            if isinstance(default_config[key], dict):
                default_config[key].update(user_config.get(key, {}))
            else:
                default_config[key] = user_config.get(key, default_config[key])
    else:
        logger.warning(
            f"Config file {config_path} not found; using default settings."
        )

    return default_config


def process_and_resample_cfd_sensor(data_file, sensor_num, scenario_name, mass_flow):
    """Process and resample a single CFD sensor's data."""
    df = pd.read_csv(data_file)
    x, y, z = SENSOR_POSITIONS[sensor_num]
    
    # Shift time to start from 0
    t_min = df['time'].min()
    df['time'] = df['time'] - t_min
    
    # Resample from 100Hz to 1.6Hz
    original_freq = 100
    target_freq = 1.6
    num_samples = int(len(df) * target_freq / original_freq)
    
    # Resample h2_volume_fraction
    resampled_h2 = signal.resample(df['h2_volume_fraction'].values, num_samples)
    resampled_h2[0] = 0.
    new_time = np.linspace(df['time'].iloc[0], df['time'].iloc[-1], num_samples)
    
    # Add small noise to zero values to avoid exact zeros
    resampled_h2 = np.maximum(resampled_h2, 0)
    resampled_h2 = np.where(resampled_h2 == 0, 1e-8, resampled_h2)
    
    # Create resampled dataframe for this sensor
    records = []
    for i in range(num_samples):
        records.append({
            'time': new_time[i],
            'mass_flow': mass_flow,
            'sensor_id': sensor_num,
            'x': x, 'y': y, 'z': z,
            'h2_volume_fraction': resampled_h2[i],
            'source': 'cfd',
            'scenario': scenario_name,
            'experiment_id': f'CFD_{scenario_name}'
        })
    
    return pd.DataFrame(records)


def process_cfd_scenario(scenario_dir, scenario_name):
    """Process all sensors in a CFD scenario."""
    all_sensor_dfs = []
    mass_flow = CFD_MASS_FLOW.get(scenario_name, np.nan)
    
    sensor_dirs = [d for d in scenario_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('h2Sensor')]
    
    for sensor_dir in sorted(sensor_dirs):
        sensor_num = int(sensor_dir.name.replace('h2Sensor', ''))
        
        data_file = sensor_dir / '0.1' / 'data.csv'
        if not data_file.exists():
            continue
        
        # Process and resample each sensor individually
        df_sensor = process_and_resample_cfd_sensor(data_file, sensor_num, scenario_name, mass_flow)
        if not df_sensor.empty:
            all_sensor_dfs.append(df_sensor)
    
    all_sensor_dfs = pd.concat(all_sensor_dfs, ignore_index=True)

    return all_sensor_dfs


def extract_mean_mass_flow(df_exp):
    """Extract mean mass flow from experimental data."""
    # The mass flow info is in the first few rows under 'mass flow meter 1' column
    # Look for pattern like "mean mass flow - 0.4757728572042119"
    if 'mass flow meter 1' in df_exp.columns:
        col_values = df_exp['mass flow meter 1'].astype(str)
        for val in col_values.head(5):  # Check first 5 rows
            match = re.search(r'mean mass flow[\s-]+([\d.]+)', val.lower())
            if match:
                return float(match.group(1))
            # Also check "total mean mass flow"
            match = re.search(r'total mean mass flow[\s-]+([\d.]+)', val.lower())
            if match:
                return float(match.group(1))
    elif 'mass flow meter 2' in df_exp.columns:
        col_values = df_exp['mass flow meter 2'].astype(str)
        for val in col_values.head(5):  # Check first 5 rows
            match = re.search(r'mean mass flow[\s-]+([\d.]+)', val.lower())
            if match:
                return float(match.group(1))
            # Also check "total mean mass flow"
            match = re.search(r'total mean mass flow[\s-]+([\d.]+)', val.lower())
            if match:
                return float(match.group(1))
    return np.nan


def process_experiment(exp_file, exp_name):
    """Process experimental data file."""
    df = pd.read_csv(exp_file, low_memory=False)
    
    # Extract mean mass flow from metadata
    mass_flow = extract_mean_mass_flow(df)
    
    # Use h2 sensor time
    time_col = 'h2 sensor time'
    if time_col not in df.columns:
        return pd.DataFrame()
    
    # Get sensor columns
    sensor_cols = [c for c in df.columns if 'h2 concentration [%]' in c.lower()]
    
    # Find last row with valid H2 sensor data
    # Check where ALL sensor columns are NaN/empty
    last_valid_idx = None
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        has_valid = False
        for col in sensor_cols:
            val = row[col]
            # Check if value is valid (not NaN and not empty string)
            if pd.notna(val) and str(val).strip() != '':
                has_valid = True
                break
        if has_valid:
            last_valid_idx = idx
            break
    
    if last_valid_idx is None:
        logger.warning(f"  No valid H2 sensor data found in {exp_name}")
        return pd.DataFrame()
    
    # Trim dataframe to only include rows up to last valid H2 measurement
    if last_valid_idx < len(df) - 1:
        logger.info(f"  Trimming from {len(df)} to {last_valid_idx + 1} rows (H2 sensors stopped)")
        df = df.iloc[:last_valid_idx + 1].copy()
    
    all_records = []
    
    for sensor_col in sensor_cols:
        parts = sensor_col.split()
        sensor_num = int(parts[1])
        
        x, y, z = SENSOR_POSITIONS[sensor_num]
        
        # Experimental data is in volume fraction [%], convert to fraction
        h2_vf_percent = df[sensor_col].values
        h2_vf = h2_vf_percent / 100.0  # Convert % to fraction
        h2_vf = np.maximum(h2_vf, 0)  # clip negative H2 concentrations
        # Add small noise to zero values to avoid exact zeros
        h2_vf = np.where(h2_vf == 0, 1e-8, h2_vf)
        
        n_rows = len(df)
        records = pd.DataFrame({
            'time': df[time_col].values,
            'mass_flow': [mass_flow] * n_rows,
            'sensor_id': [sensor_num] * n_rows,
            'x': [x] * n_rows, 'y': [y] * n_rows, 'z': [z] * n_rows,
            'h2_volume_fraction': h2_vf,
            'source': ['experiment'] * n_rows,
            'scenario': [exp_name] * n_rows,
            'experiment_id': [exp_name] * n_rows
        })
        
        all_records.append(records)
    all_records = pd.concat(all_records, ignore_index=True)

    return all_records


def shift_time_to_mass_flow_start(df, exp_name):
    """Shift time so that time starts when mass flow starts."""
    df = df.copy()
    if exp_name in FORTY_SEC_START:
        df = df[df['time'] >= 39]
        t_min = df['time'].min()
        df['time'] = df['time'] - t_min
    else:
        df = df[df['time'] >= 35]
        t_min = df['time'].min()
        df['time'] = df['time'] - t_min
    return df

def cut_off_time(df, exp_name):
    """Shift time so that time starts when mass flow starts."""
    df = df.copy()
    if exp_name in THIRTY_SEC:
        df = df[df['time'] <= 30]
    elif exp_name in SIXTY_SEC:
        df = df[df['time'] <= 60]
    elif exp_name in NINETY_SEC:
        df = df[df['time'] <= 90]
    elif exp_name in HUNDRED_TWENTY_SEC:
        df = df[df['time'] <= 120]
    else:
        df = df[df['time'] <= 240]
    return df

def assign_split(df, exp_name):
    """Assign train/val/test split based on experiment name."""
    if exp_name.startswith('CFD_'):
        return 'train'
    elif exp_name in HELD_OUT_TEST_EXPERIMENTS:
        return 'test'
    else:
        return 'train'


def add_release_onset_features(
    df: pd.DataFrame,
    threshold: float = 0.009,
    include_initial: bool = False,
    include_lag1: bool = True,
) -> pd.DataFrame:
    """
    Add release-onset features to a unified raw DataFrame.

    For each scenario the first time any sensor exceeds ``threshold`` becomes
    ``t_release``. ``time_since_release = time - t_release`` then replaces the
    original ``time`` column. Optionally adds ``h2_initial`` and ``h2_lag_1``.

    Parameters
    ----------
    df : pd.DataFrame
        Unified raw dataset with 'scenario', 'experiment_id', 'sensor_id',
        'time' and 'h2_volume_fraction' columns.
    threshold : float
        H2 volume-fraction threshold used to detect the release onset.
    include_initial : bool
        Whether to add an ``h2_initial`` column per (scenario, experiment, sensor).
    include_lag1 : bool
        Whether to add ``h2_lag_1`` and drop rows where it is missing.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset with ``time_since_release`` instead of ``time``.
    """
    df = df.copy()

    # Per-scenario release onset: first time any sensor exceeds threshold
    active = df[df['h2_volume_fraction'] > threshold]
    if len(active) == 0:
        raise ValueError(f"No readings exceed threshold {threshold}")

    scenario_onset = active.groupby('scenario')['time'].min().reset_index()
    scenario_onset.columns = ['scenario', 't_release']
    logger.info(f"Release onset times (threshold={threshold}):")
    logger.info('\n' + scenario_onset.to_string(index=False))

    df = df.merge(scenario_onset, on='scenario', how='left')

    # Scenarios with no readings above threshold get t_release = min time
    if df['t_release'].isna().any():
        logger.warning(
            "Some scenarios have no reading above threshold; using min time for t_release."
        )
        scenario_min_time = df.groupby('scenario')['time'].min().reset_index()
        scenario_min_time.columns = ['scenario', 'min_time']
        df = df.merge(scenario_min_time, on='scenario', how='left')
        df['t_release'] = df['t_release'].fillna(df['min_time'])
        df = df.drop(columns=['min_time'])

    # Time since release replaces the original time coordinate
    df['time_since_release'] = df['time'] - df['t_release']

    group_cols = ['scenario', 'experiment_id', 'sensor_id']
    df = df.sort_values(group_cols + ['time'])

    # Optional: initial condition at t_release
    if include_initial:
        df['time_diff_to_release'] = (df['time'] - df['t_release']).abs()
        idx_closest = df.groupby(group_cols)['time_diff_to_release'].idxmin()
        initial_conditions = df.loc[
            idx_closest, group_cols + ['h2_volume_fraction']
        ].copy()
        initial_conditions = initial_conditions.rename(
            columns={'h2_volume_fraction': 'h2_initial'}
        )
        df = df.merge(initial_conditions, on=group_cols, how='left')
        df = df.drop(columns=['time_diff_to_release'])

    # Optional: lag-1 concentration
    if include_lag1:
        df['h2_lag_1'] = df.groupby(group_cols)['h2_volume_fraction'].shift(1)
        n_before = len(df)
        df = df.dropna(subset=['h2_lag_1'])
        logger.info(
            f"Dropped {n_before - len(df)} rows with missing h2_lag_1"
        )

    # Replace the original time column with time_since_release
    df = df.drop(columns=['time'])

    return df


def main():
    config = load_config()
    data_sources = config['data_sources']
    paths = config['paths']
    dataset_type = config.get('dataset_type', 'raw')
    release_onset_cfg = config.get('release_onset', {})

    if dataset_type not in ('raw', 'preprocessed'):
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}' in config.json. "
            "Use 'raw' or 'preprocessed'."
        )

    # Resolve relative paths against the config file location, falling back to CWD
    config_file = Path('config.json').resolve()
    base_dir = config_file.parent if config_file.exists() else Path.cwd()
    data_dir = base_dir / paths['data_dir']
    cfd_dir = base_dir / paths['cfd_dir']
    exp_dir = base_dir / paths['experiments_dir']

    if dataset_type == 'raw':
        output_file = base_dir / paths['raw_output_csv']
        summary_file = base_dir / paths['raw_output_summary']
    else:
        output_file = base_dir / paths['preprocessed_output_csv']
        summary_file = base_dir / paths['preprocessed_output_summary']

    include_cfd = data_sources.get('include_cfd', False)
    include_experiments = data_sources.get('include_experiments', True)

    logger.info("Configuration:")
    logger.info(f"  dataset_type: {dataset_type}")
    logger.info(f"  include_experiments: {include_experiments}")
    logger.info(f"  include_cfd: {include_cfd}")
    logger.info(f"  data_dir: {data_dir}")
    logger.info(f"  output_file: {output_file}")
    logger.info("")

    if not include_experiments and not include_cfd:
        raise ValueError(
            "At least one data source must be enabled in config.json."
        )

    all_data = []

    # ------------------------------------------------------------------
    # CFD scenarios
    # ------------------------------------------------------------------
    if include_cfd:
        if not cfd_dir.exists():
            raise FileNotFoundError(
                f"CFD data directory not found: {cfd_dir}\n"
                "Set include_cfd to false in config.json if you do not have the CFD data."
            )

        logger.info("Processing CFD data...")
        logger.info("=" * 50)

        for scenario in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'O1', 'O2']:
            scenario_dir = cfd_dir / scenario
            if not scenario_dir.exists():
                continue

            logger.info(f"Processing CFD scenario {scenario}...")
            df_cfd = process_cfd_scenario(scenario_dir, scenario)
            if not df_cfd.empty:
                # Each sensor is already resampled individually in process_cfd_scenario
                df_cfd['split'] = 'train'
                all_data.append(df_cfd)
                logger.info(
                    f"  Added {len(df_cfd)} records "
                    f"(mass_flow: {df_cfd['mass_flow'].iloc[0]:.3f}, "
                    f"H2 max: {df_cfd['h2_volume_fraction'].max():.4f})"
                )
    else:
        logger.info("Skipping CFD data (include_cfd is false).")

    # ------------------------------------------------------------------
    # Experimental data
    # ------------------------------------------------------------------
    if include_experiments:
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experimental data directory not found: {exp_dir}\n"
                "Download the open dataset and place it under the configured experiments_dir."
            )

        logger.info("Processing experimental data...")
        logger.info("=" * 50)

        for exp_folder in sorted(exp_dir.iterdir()):
            if not exp_folder.is_dir():
                continue
            exp_name = exp_folder.name
            csv_files = list(exp_folder.glob('*.csv'))
            if not csv_files:
                continue
            exp_file = csv_files[0]

            logger.info(f"Processing {exp_name}...")
            df_exp = process_experiment(exp_file, exp_name)
            if not df_exp.empty:
                if dataset_type == 'preprocessed':
                    # Cut off the tail where the experiment has ended.
                    # Release-onset alignment replaces the mass-flow-start shift.
                    df_exp = cut_off_time(df_exp, exp_name)
                df_exp['split'] = assign_split(df_exp, exp_name)
                split_name = df_exp['split'].iloc[0]
                all_data.append(df_exp)
                logger.info(
                    f"  Added {len(df_exp)} records -> {split_name} "
                    f"(mass_flow: {df_exp['mass_flow'].iloc[0]:.3f}, "
                    f"H2 max: {df_exp['h2_volume_fraction'].max():.4f})"
                )
    else:
        logger.info("Skipping experimental data (include_experiments is false).")

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    if not all_data:
        raise ValueError(
            "No data was processed. Check that the configured directories "
            "contain the expected files and that at least one data source is enabled."
        )

    logger.info("Combining data...")
    df_all = pd.concat(all_data, ignore_index=True)

    # Add active/inactive label based on H2 volume fraction threshold
    df_all['active'] = (df_all['h2_volume_fraction'] > ACTIVE_THRESHOLD).astype(int)
    n_active = df_all['active'].sum()
    n_inactive = len(df_all) - n_active
    logger.info(f"Active/inactive label added (threshold={ACTIVE_THRESHOLD})")
    logger.info(f"  Active:   {n_active:,} ({100*n_active/len(df_all):.1f}%)")
    logger.info(f"  Inactive: {n_inactive:,} ({100*n_inactive/len(df_all):.1f}%)")

    # ------------------------------------------------------------------
    # Release-onset preprocessing (preprocessed mode only)
    # ------------------------------------------------------------------
    if dataset_type == 'preprocessed':
        logger.info("Adding release-onset features...")
        df_all = add_release_onset_features(
            df_all,
            threshold=release_onset_cfg.get('threshold', 0.009),
            include_initial=release_onset_cfg.get('include_initial', False),
            include_lag1=release_onset_cfg.get('include_lag1', True),
        )

        n_before_release = (df_all['time_since_release'] < 0).sum()
        n_after_release = (df_all['time_since_release'] >= 0).sum()
        logger.info(f"  Records before release: {n_before_release:,}")
        logger.info(f"  Records at/after release: {n_after_release:,}")

    # Save dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_file, index=False)
    logger.info(f"Saved {dataset_type} dataset: {output_file} ({len(df_all)} records)")

    # Generate summary statistics
    logger.info("Generating summary statistics...")

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"Summary Statistics ({dataset_type.capitalize()} Data)")
    summary_lines.append("=" * 60)
    summary_lines.append(f"\nTotal records: {len(df_all):,}")
    summary_lines.append(f"Columns: {list(df_all.columns)}")
    summary_lines.append("")

    cfd_records = df_all[df_all['source'] == 'cfd']
    if not cfd_records.empty:
        summary_lines.append("CFD Scenarios:")
        summary_lines.append("-" * 40)
        cfd_summary = cfd_records.groupby('scenario').agg({
            'mass_flow': 'first',
            'h2_volume_fraction': ['min', 'max', 'mean'],
            'experiment_id': 'count'
        }).round(4)
        summary_lines.append(cfd_summary.to_string())
    else:
        summary_lines.append("CFD Scenarios: none included")

    summary_lines.append("\nExperiments (all):")
    summary_lines.append("-" * 40)
    exp_records = df_all[df_all['source'] == 'experiment']
    if not exp_records.empty:
        exp_summary = exp_records.groupby('experiment_id').agg({
            'mass_flow': 'first',
            'h2_volume_fraction': ['min', 'max', 'mean'],
            'split': 'first'
        }).round(4)
        summary_lines.append(exp_summary.to_string())
    else:
        summary_lines.append("Experiments: none included")

    if dataset_type == 'preprocessed' and 'time_since_release' in df_all.columns:
        summary_lines.append("\nRelease-onset timing:")
        summary_lines.append("-" * 40)
        summary_lines.append(
            df_all.groupby('scenario')['time_since_release']
            .agg(['min', 'max', 'mean'])
            .round(4)
            .to_string()
        )

    summary_lines.append("\nOverall Statistics:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Sources: {df_all['source'].value_counts().to_dict()}")
    summary_lines.append(f"\nActive/Inactive (threshold={ACTIVE_THRESHOLD}):")
    summary_lines.append(df_all['active'].value_counts().to_string())
    summary_lines.append(f"\nH2 Volume Fraction by source:")
    summary_lines.append(df_all.groupby('source')['h2_volume_fraction'].describe().to_string())
    summary_lines.append(f"\nMass Flow by source:")
    summary_lines.append(df_all.groupby('source')['mass_flow'].describe().to_string())

    # Save summary to file
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"Saved summary statistics: {summary_file}")

    # Also log the summary
    logger.info("\n" + "\n".join(summary_lines))


if __name__ == '__main__':
    main()
