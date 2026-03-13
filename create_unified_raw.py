#!/usr/bin/env python3
"""
Create unified raw data file without any scaling.
- CFD: use h2_volume_fraction as-is, mass_flow from column
- Experiments: use H2 concentration / 100, mass flow from "mean mass flow"
- No scaling, no delay embeddings, just unified raw data
"""

import pandas as pd
import numpy as np
from scipy import signal
import re
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

# Splits
HELD_OUT_TEST_EXPERIMENTS = ['23_FFI_P101_T00005', '23_FFI_P101_T00014', 
                             '23_FFI_P101_T00031', '23_FFI_P101_T00045']

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


def process_cfd_scenario(scenario_dir, scenario_name):
    """Process all sensors in a CFD scenario."""
    records = []
    mass_flow = CFD_MASS_FLOW.get(scenario_name, np.nan)
    
    sensor_dirs = [d for d in scenario_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('h2Sensor')]
    
    for sensor_dir in sorted(sensor_dirs):
        sensor_num = int(sensor_dir.name.replace('h2Sensor', ''))
        
        data_file = sensor_dir / '0.1' / 'data.csv'
        if not data_file.exists():
            continue
        
        df = pd.read_csv(data_file)
        x, y, z = SENSOR_POSITIONS[sensor_num]
        
        # Use h2_volume_fraction directly (already in correct unit)
        # Add small noise to zero values to avoid exact zeros
        for _, row in df.iterrows():
            h2_vf = row['h2_volume_fraction']
            if h2_vf == 0:
                h2_vf = 1e-8  # Small noise to avoid exact zero
            records.append({
                'time': row['time'],
                'mass_flow': mass_flow,
                'sensor_id': sensor_num,
                'x': x, 'y': y, 'z': z,
                'h2_volume_fraction': h2_vf,
                'source': 'cfd',
                'scenario': scenario_name,
                'experiment_id': f'CFD_{scenario_name}'
            })
    
    return pd.DataFrame(records)


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
        
        if sensor_num not in SENSOR_POSITIONS:
            continue
        
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
    
    if all_records:
        return pd.concat(all_records, ignore_index=True)
    return pd.DataFrame()


def shift_time_to_zero(df):
    """Shift time so that minimum time is 0."""
    df = df.copy()
    t_min = df['time'].min()
    df['time'] = df['time'] - t_min
    return df

def resample_data(df, original_freq=100, target_freq=1.6):
    num_samples = int(len(df) * target_freq / original_freq)
    resampled_data = signal.resample(df['h2_volume_fraction'].values, num_samples)
    new_time = np.linspace(df['time'].iloc[0], df['time'].iloc[-1], num_samples)
    resampled_data[0] = 0.0
    resampled_data = np.maximum(resampled_data, 0)
    resampled_data = np.where(resampled_data == 0, 1e-8, resampled_data)
    df_resampled = pd.DataFrame({'time': new_time, 'h2_volume_fraction': resampled_data})
    df_other = df.drop(columns=['h2_volume_fraction']) 
    return pd.merge_asof(
        df_resampled.sort_values('time'),
        df_other.sort_values('time'),
        on='time',
        direction='nearest'
    )

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

def assign_split(df, exp_name):
    """Assign train/val/test split based on experiment name."""
    if exp_name.startswith('CFD_'):
        return 'train'
    elif exp_name in HELD_OUT_TEST_EXPERIMENTS:
        return 'test'
    else:
        return 'train'


def main():
    data_dir = Path('/home/niclasflehmig/VisualCodeProjects/H2-dispersion-model/data')
    cfd_dir = data_dir / 'CFD'
    exp_dir = data_dir / 'raw'
    
    all_data = []
    
    logger.info("Processing CFD data...")
    logger.info("=" * 50)
    
    # Process CFD scenarios
    for scenario in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'O1', 'O2']:
        scenario_dir = cfd_dir / scenario
        if not scenario_dir.exists():
            continue
        
        logger.info(f"Processing CFD scenario {scenario}...")
        df_cfd = process_cfd_scenario(scenario_dir, scenario)
        if not df_cfd.empty:
            df_cfd = shift_time_to_zero(df_cfd)
            df_cfd = resample_data(df_cfd)
            df_cfd['split'] = 'train'
            all_data.append(df_cfd)
            logger.info(f"  Added {len(df_cfd)} records (mass_flow: {df_cfd['mass_flow'].iloc[0]:.3f}, H2 max: {df_cfd['h2_volume_fraction'].max():.4f})")
    
    logger.info("Processing experimental data...")
    logger.info("=" * 50)
    
    # Process experimental data
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
            df_exp = shift_time_to_mass_flow_start(df_exp, exp_name)
            # Filter: keep only data up to 200 seconds after mass flow start
            df_exp = df_exp[df_exp['time'] <= 200]
            df_exp['split'] = assign_split(df_exp, exp_name)
            split_name = df_exp['split'].iloc[0]
            all_data.append(df_exp)
            logger.info(f"  Added {len(df_exp)} records -> {split_name} (mass_flow: {df_exp['mass_flow'].iloc[0]:.3f}, H2 max: {df_exp['h2_volume_fraction'].max():.4f})")
    
    logger.info("Combining data...")
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Save unified raw data
    output_file = data_dir / 'unified_raw.csv'
    df_all.to_csv(output_file, index=False)
    logger.info(f"Saved unified raw data: {output_file} ({len(df_all)} records)")
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("Summary Statistics (Raw Data)")
    summary_lines.append("=" * 60)
    summary_lines.append(f"\nTotal records: {len(df_all):,}")
    summary_lines.append(f"Columns: {list(df_all.columns)}")
    summary_lines.append("")
    
    summary_lines.append("CFD Scenarios:")
    summary_lines.append("-" * 40)
    cfd_summary = df_all[df_all['source'] == 'cfd'].groupby('scenario').agg({
        'mass_flow': 'first',
        'h2_volume_fraction': ['min', 'max', 'mean'],
        'experiment_id': 'count'
    }).round(4)
    summary_lines.append(cfd_summary.to_string())
    
    summary_lines.append("\nExperiments (all):")
    summary_lines.append("-" * 40)
    exp_summary = df_all[df_all['source'] == 'experiment'].groupby('experiment_id').agg({
        'mass_flow': 'first',
        'h2_volume_fraction': ['min', 'max', 'mean'],
        'split': 'first'
    }).round(4)
    summary_lines.append(exp_summary.to_string())
    
    summary_lines.append("\nOverall Statistics:")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Sources: {df_all['source'].value_counts().to_dict()}")
    summary_lines.append(f"\nH2 Volume Fraction by source:")
    summary_lines.append(df_all.groupby('source')['h2_volume_fraction'].describe().to_string())
    summary_lines.append(f"\nMass Flow by source:")
    summary_lines.append(df_all.groupby('source')['mass_flow'].describe().to_string())
    
    # Save summary to file
    summary_file = data_dir / 'unified_raw_summary.txt'
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"Saved summary statistics: {summary_file}")
    
    # Also log the summary
    logger.info("\n" + "\n".join(summary_lines))


if __name__ == '__main__':
    main()
