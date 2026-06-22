"""Create a dataset with per-scenario release onset and optional lag/initial features."""

import argparse
import pandas as pd


def create_release_onset_dataset(
    input_path: str,
    output_path: str,
    threshold: float = 0.009,
    include_initial: bool = False,
    include_lag1: bool = True,
):
    """
    Create a dataset where each row has:
      - t_release: first time in the scenario where any sensor exceeds threshold
      - time_since_release: time - t_release
      - h2_initial: (optional) concentration at t_release for this sensor
      - h2_lag_1: (optional) concentration at t-1 for this sensor
    """
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records from {input_path}")

    # Per-scenario release onset: first time any sensor exceeds threshold
    active = df[df['h2_volume_fraction'] > threshold]
    if len(active) == 0:
        raise ValueError(f"No readings exceed threshold {threshold}")

    scenario_onset = active.groupby('scenario')['time'].min().reset_index()
    scenario_onset.columns = ['scenario', 't_release']
    print(f"\nRelease onset times (threshold={threshold}):")
    print(scenario_onset.to_string(index=False))

    # Merge release onset back
    df = df.merge(scenario_onset, on='scenario', how='left')

    # Scenarios with no readings above threshold get t_release = min time
    if df['t_release'].isna().any():
        print("\nWarning: some scenarios have no reading above threshold; using min time.")
        scenario_min_time = df.groupby('scenario')['time'].min().reset_index()
        scenario_min_time.columns = ['scenario', 'min_time']
        df = df.merge(scenario_min_time, on='scenario', how='left')
        df['t_release'] = df['t_release'].fillna(df['min_time'])
        df = df.drop(columns=['min_time'])

    # Time since release
    df['time_since_release'] = df['time'] - df['t_release']

    group_cols = ['scenario', 'experiment_id', 'sensor_id']
    df = df.sort_values(group_cols + ['time'])

    # Optional: initial condition at t_release
    if include_initial:
        df['time_diff_to_release'] = (df['time'] - df['t_release']).abs()
        idx_closest = df.groupby(group_cols)['time_diff_to_release'].idxmin()
        initial_conditions = df.loc[idx_closest, group_cols + ['h2_volume_fraction']].copy()
        initial_conditions = initial_conditions.rename(columns={'h2_volume_fraction': 'h2_initial'})
        df = df.merge(initial_conditions, on=group_cols, how='left')
        df = df.drop(columns=['time_diff_to_release'])

    # Optional: lag-1 concentration
    if include_lag1:
        df['h2_lag_1'] = df.groupby(group_cols)['h2_volume_fraction'].shift(1)
        # Drop rows where lag is missing
        n_before = len(df)
        df = df.dropna(subset=['h2_lag_1'])
        print(f"\nDropped {n_before - len(df)} rows with missing h2_lag_1")

    # Sanity checks
    print(f"\nDataset summary:")
    print(f"  Total records: {len(df):,}")
    print(f"  Records before release (time_since_release < 0): {(df['time_since_release'] < 0).sum():,}")
    print(f"  Records at/after release (time_since_release >= 0): {(df['time_since_release'] >= 0).sum():,}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Columns: {df.columns.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Create release-onset dataset")
    parser.add_argument('--input', type=str, default='data/unified_raw_two_modes.csv',
                        help='Input unified CSV')
    parser.add_argument('--output', type=str, default='data/unified_raw_two_modes_release_onset.csv',
                        help='Output CSV')
    parser.add_argument('--threshold', type=float, default=0.009,
                        help='Threshold for release detection')
    parser.add_argument('--include_initial', action='store_true',
                        help='Include h2_initial column')
    parser.add_argument('--include_lag1', action='store_true', default=True,
                        help='Include h2_lag_1 column')
    args = parser.parse_args()

    create_release_onset_dataset(
        args.input,
        args.output,
        args.threshold,
        args.include_initial,
        args.include_lag1,
    )


if __name__ == '__main__':
    main()
