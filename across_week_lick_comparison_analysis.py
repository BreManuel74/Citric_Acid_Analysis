"""
Weekly Comparison Analysis

Performs statistical comparisons across multiple weeks of capacitive sensor data.
Loads a master CSV with metadata for all weeks and aligns it with multiple capacitive log files
to perform cross-week statistical analysis.

Usage:
    python weekly_comparison_analysis.py
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Try importing pingouin for mixed ANOVA
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("WARNING: pingouin not installed. Mixed ANOVA will not be available.")
    print("Install with: pip install pingouin")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def load_capacitive_csv(csv_path: Path) -> pd.DataFrame:
    """Load and clean a capacitive CSV file.
    
    Returns DataFrame with Time_sec column and sensor readings.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if "Arduino_Timestamp" not in df.columns:
        raise ValueError(f"No Arduino_Timestamp column found in {csv_path}")

    # Clean timestamp data
    df["Arduino_Timestamp"] = pd.to_numeric(df["Arduino_Timestamp"], errors="coerce")
    df = df.dropna(subset=["Arduino_Timestamp"]).copy()
    df["Arduino_Timestamp"] = df["Arduino_Timestamp"].astype("int64")

    # Compute time in seconds
    df["Time_sec"] = df["Arduino_Timestamp"] / 1000.0
    
    return df


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return list of sensor columns in sorted numeric order."""
    sensor_cols = [c for c in df.columns if c.startswith("Sensor_") and not c.endswith("_deviation")]
    
    def key(c: str) -> int:
        try:
            return int(c.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    sensor_cols.sort(key=key)
    return sensor_cols


def compute_sensor_KDE(df: pd.DataFrame, sensor_cols: List[str]) -> pd.Series:
    """Compute the KDE (Kernel Density Estimation) peak for each sensor column.
    
    Returns a Series indexed by sensor column names with their KDE peak values.
    The KDE peak represents the most probable value in the distribution.
    """
    kdes = {}
    for col in sensor_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        # Drop NaN values before computing KDE
        series = series.dropna()
        if len(series) > 1:  # Need at least 2 points for KDE
            try:
                # Create KDE
                kde = stats.gaussian_kde(series)
                # Create evaluation points around the data range
                min_val, max_val = series.min(), series.max()
                x_eval = np.linspace(min_val, max_val, 1000)
                # Find the peak of the KDE
                density = kde(x_eval)
                peak_idx = np.argmax(density)
                kdes[col] = x_eval[peak_idx]
            except Exception:
                # Fall back to mean if KDE fails
                kdes[col] = series.mean()
        else:
            kdes[col] = series.iloc[0] if len(series) == 1 else None
    return pd.Series(kdes)


def compute_KDE_normalizations(df: pd.DataFrame, sensor_cols: List[str], sensor_kdes: pd.Series) -> pd.DataFrame:
    """Compute KDE normalization for each sensor: abs((value - KDE) / KDE).
    
    For each sensor column, creates a new column with suffix '_deviation' containing
    the absolute normalized value: abs((capacitance_value - KDE) / KDE).
    
    Returns a copy of the dataframe with the new normalization columns added.
    """
    df_with_normalizations = df.copy()
    
    for col in sensor_cols:
        kde_val = sensor_kdes[col]
        if kde_val is not None and kde_val != 0 and col in df.columns:
            # Compute KDE normalization: abs((value - KDE) / KDE)
            sensor_series = pd.to_numeric(df[col], errors="coerce")
            normalization_col_name = f"{col}_deviation"  # Keep same column name for compatibility
            df_with_normalizations[normalization_col_name] = abs((sensor_series - kde_val) / kde_val)
        else:
            # If KDE is None or zero, set normalization to NaN
            normalization_col_name = f"{col}_deviation"
            df_with_normalizations[normalization_col_name] = pd.NA
    
    return df_with_normalizations


# Function kept for backward compatibility but not used
def compute_dynamic_thresholds(df: pd.DataFrame, sensor_cols: List[str], z_threshold: float = 4.0) -> pd.Series:
    """Compute dynamic thresholds for event detection (DEPRECATED - use fixed threshold)."""
    thresholds = {}
    
    for sensor_col in sensor_cols:
        deviation_col = f"{sensor_col}_deviation"
        if deviation_col in df.columns:
            dev_values = df[deviation_col].dropna()
            if len(dev_values) > 0:
                mean_dev = dev_values.mean()
                std_dev = dev_values.std()
                threshold = mean_dev + (z_threshold * std_dev)
                thresholds[sensor_col] = threshold
            else:
                thresholds[sensor_col] = np.nan
        else:
            thresholds[sensor_col] = np.nan
    
    return pd.Series(thresholds)


def detect_events_above_threshold(df: pd.DataFrame, sensor_cols: List[str], thresholds: pd.Series) -> pd.DataFrame:
    """Detect time points where deviation exceeds the threshold for each sensor.
    
    Creates boolean columns indicating when each sensor's deviation peaks above the threshold.
    Uses scipy.signal.find_peaks for robust peak detection in discrete sampled data.
    
    Parameters:
        df: DataFrame with Time_sec and deviation columns
        sensor_cols: List of sensor column names
        thresholds: Series of threshold values per sensor
        
    Returns:
        DataFrame with columns:
            - Time_sec
            - For each sensor: {sensor}_event (boolean indicating detected peaks above threshold)
            - For each sensor: {sensor}_deviation (original deviation value)
            - For each sensor: {sensor}_derivative (first-order derivative of deviation)
    """
    result = pd.DataFrame()
    result['Time_sec'] = df['Time_sec']
    
    for sensor_col in sensor_cols:
        dev_col = f"{sensor_col}_deviation"
        event_col = f"{sensor_col}_event"
        deriv_col = f"{sensor_col}_derivative"
        
        if dev_col not in df.columns:
            result[event_col] = False
            result[dev_col] = np.nan
            result[deriv_col] = np.nan
            continue
        
        threshold = thresholds.get(sensor_col)
        if threshold is None or not np.isfinite(threshold):
            result[event_col] = False
            result[dev_col] = df[dev_col]
            result[deriv_col] = np.nan
            continue
        
        # Get deviations and calculate first-order derivative
        deviations = pd.to_numeric(df[dev_col], errors="coerce")
        result[dev_col] = deviations
        
        # Calculate first-order derivative using forward difference
        clean_deviations = deviations.fillna(0)
        derivative = np.zeros_like(clean_deviations)
        derivative[:-1] = np.diff(clean_deviations)  # Forward difference
        derivative[-1] = derivative[-2] if len(derivative) > 1 else 0  # Handle last point
        result[deriv_col] = derivative
        
        # Find peaks in the deviation signal using scipy.signal.find_peaks
        peaks, _ = find_peaks(clean_deviations, height=threshold, distance=1)
        
        # Create boolean mask for detected peaks
        peak_mask = np.zeros(len(clean_deviations), dtype=bool)
        peak_mask[peaks] = True
        
        result[event_col] = peak_mask
    
    return result


def compute_lick_bouts(events_df: pd.DataFrame, sensor_cols: List[str], ili_cutoff: float = 0.3) -> dict:
    """Compute lick bouts for each sensor."""
    bout_results = {}
    
    for sensor_col in sensor_cols:
        event_col = f"{sensor_col}_event"
        if event_col not in events_df.columns:
            bout_results[sensor_col] = {
                'bout_count': 0,
                'bout_sizes': np.array([]),
                'bout_durations': np.array([]),
                'bout_start_times': np.array([]),
                'bout_end_times': np.array([])
            }
            continue
        
        # Get event times
        event_times = events_df[events_df[event_col]]['Time_sec'].values
        
        if len(event_times) < 2:
            bout_results[sensor_col] = {
                'bout_count': 1 if len(event_times) == 1 else 0,
                'bout_sizes': np.array([1]) if len(event_times) == 1 else np.array([]),
                'bout_durations': np.array([0.0]) if len(event_times) == 1 else np.array([]),
                'bout_start_times': event_times if len(event_times) == 1 else np.array([]),
                'bout_end_times': event_times if len(event_times) == 1 else np.array([])
            }
            continue
        
        # Calculate inter-lick intervals
        ilis = np.diff(event_times)
        
        # Find bout boundaries (where ILI >= cutoff)
        bout_ends = np.where(ilis >= ili_cutoff)[0]
        
        # Create bout segments
        bout_starts = [0] + (bout_ends + 1).tolist()
        bout_ends_list = bout_ends.tolist() + [len(event_times) - 1]
        
        bout_sizes = []
        bout_durations = []
        bout_start_times = []
        bout_end_times = []
        
        for start_idx, end_idx in zip(bout_starts, bout_ends_list):
            bout_size = end_idx - start_idx + 1
            bout_start_time = event_times[start_idx]
            bout_end_time = event_times[end_idx]
            bout_duration = bout_end_time - bout_start_time
            
            bout_sizes.append(bout_size)
            bout_durations.append(bout_duration)
            bout_start_times.append(bout_start_time)
            bout_end_times.append(bout_end_time)
        
        bout_results[sensor_col] = {
            'bout_count': len(bout_sizes),
            'bout_sizes': np.array(bout_sizes),
            'bout_durations': np.array(bout_durations),
            'bout_start_times': np.array(bout_start_times),
            'bout_end_times': np.array(bout_end_times)
        }
    
    return bout_results


def compute_weekly_averages(weekly_results: Dict) -> Dict:
    """Compute average licks, bouts, and weight metrics across all 12 animals for each week.
    
    Parameters:
        weekly_results: Dictionary with results from process_single_week for each date
        
    Returns:
        Dictionary with weekly averages containing:
            - avg_total_licks: Average total licks across 12 animals
            - avg_total_bouts: Average total bouts across 12 animals
            - avg_fecal_count: Average fecal count across 12 animals
            - avg_bottle_weight_loss: Average bottle weight loss across 12 animals
            - avg_total_weight_loss: Average total weight loss across 12 animals
            - ca_percent: Citric acid percentage for this week
            - std_licks: Standard deviation of licks across animals
            - std_bouts: Standard deviation of bouts across animals
            - std_fecal: Standard deviation of fecal counts across animals
            - std_bottle_weight: Standard deviation of bottle weight loss across animals
            - std_total_weight: Standard deviation of total weight loss across animals
    """
    averages = {}
    
    for date, result in weekly_results.items():
        lick_counts = result['lick_counts']
        bout_counts = result['bout_counts']
        fecal_counts = result['fecal_counts']
        bottle_weights = result['weights']  # bottle weight change
        total_weights = result['weight_losses']  # total weight change
        
        # Calculate averages and statistics for all metrics
        avg_licks = np.mean(lick_counts)
        avg_bouts = np.mean(bout_counts)
        avg_fecal = np.mean(fecal_counts)
        
        # For bottle weights, exclude zeros that represent excluded outliers for 11/12/25
        if date == '11/12/25':
            # Filter out the zero placeholder for R9O outlier
            bottle_weights_filtered = bottle_weights[bottle_weights > 0]
            avg_bottle_weight = np.mean(bottle_weights_filtered) if len(bottle_weights_filtered) > 0 else 0
            std_bottle_weight = np.std(bottle_weights_filtered) if len(bottle_weights_filtered) > 0 else 0
        else:
            avg_bottle_weight = np.mean(bottle_weights)
            std_bottle_weight = np.std(bottle_weights)
        
        avg_total_weight = np.mean(total_weights)
        
        std_licks = np.std(lick_counts)
        std_bouts = np.std(bout_counts)
        std_fecal = np.std(fecal_counts)
        std_total_weight = np.std(total_weights)
        
        averages[date] = {
            'date': date,
            'ca_percent': result['ca_percent'],
            'avg_total_licks': avg_licks,
            'avg_total_bouts': avg_bouts,
            'avg_fecal_count': avg_fecal,
            'avg_bottle_weight_loss': avg_bottle_weight,
            'avg_total_weight_loss': avg_total_weight,
            'avg_licks_per_animal': lick_counts,
            'avg_bouts_per_animal': bout_counts,
            'avg_fecal_per_animal': fecal_counts,
            'avg_bottle_weight_per_animal': bottle_weights,
            'avg_total_weight_per_animal': total_weights,
            'std_licks': std_licks,
            'std_bouts': std_bouts,
            'std_fecal': std_fecal,
            'std_bottle_weight': std_bottle_weight,
            'std_total_weight': std_total_weight,
            'total_animals': len(lick_counts),
            'sum_total_licks': result['total_licks'],
            'sum_total_bouts': result['total_bouts'],
            'animal_ids': result.get('animal_ids', []),  # Preserve for repeated measures
            'animal_sexes': result.get('animal_sexes', [])  # Preserve for mixed ANOVA
        }
    
    return averages


def perform_anova_analysis(weekly_averages: Dict) -> Dict:
    """
    Perform MIXED ANOVA tests for each of the 5 measures:
    - Between-subjects factor: Sex (each animal has one sex)
    - Within-subjects factor: CA% (repeated measures across concentrations)
    
    Uses pingouin.mixed_anova() to properly account for both factors and repeated measurements.
    Falls back to standard ANOVA with warning if pingouin is not available.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        
    Returns:
        Dictionary containing Mixed ANOVA results for each measure
    """
    # Check if pingouin is available for mixed ANOVA
    if not HAS_PINGOUIN:
        print("\n" + "="*80)
        print("WARNING: Mixed ANOVA requires pingouin library")
        print("Falling back to standard ANOVA (does NOT account for repeated measures or sex)")
        print("Install pingouin with: pip install pingouin")
        print("="*80 + "\n")
        return _perform_standard_anova_fallback(weekly_averages)
    
    # Build long-format dataframe for mixed ANOVA
    # Each row: one animal at one CA% with one measurement
    # Need to track Sex for each animal (constant across CA% levels)
    long_data = []
    animal_sex_map = {}  # Map animal ID to sex (should be constant across CA%)
    
    # Sort by CA% concentration
    sorted_dates = sorted(weekly_averages.keys(), key=lambda d: weekly_averages[d]['ca_percent'])
    
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking Animal IDs and Sex Data")
    print("="*80)
    
    for date in sorted_dates:
        data = weekly_averages[date]
        ca_percent = data['ca_percent']
        animal_ids = data.get('animal_ids', [])
        animal_sexes = data.get('animal_sexes', [])
        
        print(f"\n{date} (CA {ca_percent}%):")
        print(f"  Number of animals: {len(animal_ids) if animal_ids else 0}")
        print(f"  Animal IDs: {animal_ids}")
        print(f"  Sexes: {animal_sexes}")
        
        # If no animal IDs, create generic ones based on position
        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(len(data['avg_licks_per_animal']))]
            print(f"  WARNING: No animal IDs found, using generic IDs: {animal_ids}")
        
        # If no sex data, use Unknown
        if not animal_sexes or len(animal_sexes) != len(animal_ids):
            animal_sexes = ["Unknown"] * len(animal_ids)
            print(f"  WARNING: No sex data found, using 'Unknown'")
        
        for i, animal_id in enumerate(animal_ids):
            # Track sex for each animal (should be consistent across CA%)
            if animal_id not in animal_sex_map:
                animal_sex_map[animal_id] = animal_sexes[i]
            elif animal_sex_map[animal_id] != animal_sexes[i]:
                print(f"  WARNING: Animal {animal_id} has inconsistent sex: {animal_sex_map[animal_id]} vs {animal_sexes[i]}")
            
            # Skip bottle weight outlier for 11/12/25 if weight is 0 (placeholder)
            bottle_weight = data['avg_bottle_weight_per_animal'][i]
            
            long_data.append({
                'Animal': animal_id,
                'Sex': animal_sex_map[animal_id],
                'CA_Percent': ca_percent,
                'Date': date,
                'licks': data['avg_licks_per_animal'][i],
                'bouts': data['avg_bouts_per_animal'][i],
                'fecal': data['avg_fecal_per_animal'][i],
                'bottle_weight': bottle_weight if bottle_weight > 0 else np.nan,  # Exclude outliers as NaN
                'total_weight': data['avg_total_weight_per_animal'][i]
            })
    
    print("\n" + "="*80)
    print(f"Total unique animals across all weeks: {len(animal_sex_map)}")
    print(f"Animal → Sex mapping:")
    for animal_id, sex in sorted(animal_sex_map.items()):
        print(f"  {animal_id}: {sex}")
    print("="*80 + "\n")
    
    df_long = pd.DataFrame(long_data)
    
    # Check if we have sex data
    has_sex_data = 'Sex' in df_long.columns and df_long['Sex'].nunique() > 1 and 'Unknown' not in df_long['Sex'].values
    
    if not has_sex_data:
        print("\n" + "="*80)
        print("WARNING: Sex data not found or incomplete in master CSV")
        print("Falling back to one-way repeated measures ANOVA (CA% only)")
        print("="*80 + "\n")
    
    # Perform repeated measures ANOVA for each measure
    anova_results = {}
    measures = ['licks', 'bouts', 'fecal', 'bottle_weight', 'total_weight']
    measure_names = {
        'licks': 'Total Licks',
        'bouts': 'Total Bouts', 
        'fecal': 'Fecal Count',
        'bottle_weight': 'Bottle Weight Loss',
        'total_weight': 'Total Weight Loss'
    }
    
    print("\n" + "="*80)
    if has_sex_data:
        print("PERFORMING MIXED ANOVA")
        print("Between-subjects factor: Sex")
        print("Within-subjects factor: CA% (repeated measures)")
    else:
        print("PERFORMING REPEATED MEASURES ANOVA (ONE-WAY)")
        print("Within-subjects factor: CA%")
    print("="*80)
    
    for measure in measures:
        print(f"\nAnalyzing: {measure_names[measure]}")
        
        # For bottle_weight, drop rows with NaN (outliers)
        # Include Sex column if available
        if has_sex_data:
            if measure == 'bottle_weight':
                df_measure = df_long[['Animal', 'Sex', 'CA_Percent', measure]].dropna()
            else:
                df_measure = df_long[['Animal', 'Sex', 'CA_Percent', measure]].copy()
        else:
            if measure == 'bottle_weight':
                df_measure = df_long[['Animal', 'CA_Percent', measure]].dropna()
            else:
                df_measure = df_long[['Animal', 'CA_Percent', measure]].copy()
        
        # Check if we have enough data
        n_animals = df_measure['Animal'].nunique()
        n_ca_levels = df_measure['CA_Percent'].nunique()
        
        # DIAGNOSTIC: Check for animals with incomplete data across CA%
        animals_per_ca = df_measure.groupby('Animal')['CA_Percent'].nunique()
        complete_animals = animals_per_ca[animals_per_ca == n_ca_levels]
        incomplete_animals = animals_per_ca[animals_per_ca < n_ca_levels]
        
        if len(incomplete_animals) > 0:
            print(f"  WARNING: {len(incomplete_animals)} animals missing data at some CA% levels")
            print(f"  Complete animals (all {n_ca_levels} CA% levels): {len(complete_animals)}")
            print(f"  Incomplete animals:")
            for animal_id, n_levels in incomplete_animals.items():
                ca_levels_present = df_measure[df_measure['Animal'] == animal_id]['CA_Percent'].unique()
                print(f"    {animal_id}: present in {n_levels}/{n_ca_levels} CA% levels ({sorted(ca_levels_present)})")
            print(f"  Filtering to only animals with complete data...")
            
            # Filter to only animals with complete data across all CA% levels
            df_measure = df_measure[df_measure['Animal'].isin(complete_animals.index)]
            n_animals = len(complete_animals)
            print(f"  After filtering: n_animals={n_animals}, n_ca_levels={n_ca_levels}")
        else:
            print(f"  All {n_animals} animals have complete data across all {n_ca_levels} CA% levels")
        
        if n_ca_levels < 2:
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'error': 'Insufficient data for ANOVA (need at least 2 CA% levels)',
                'significant': False,
                'is_mixed_anova': has_sex_data
            }
            print(f"  ERROR: Insufficient data ({n_ca_levels} CA% levels)")
            continue
        
        try:
            if has_sex_data:
                # MIXED ANOVA with Sex (between) and CA% (within)
                result_table = pg.mixed_anova(
                    dv=measure,
                    within='CA_Percent',
                    between='Sex',
                    subject='Animal',
                    data=df_measure
                )
                
                # Extract main effects and interaction
                sex_row = result_table[result_table['Source'] == 'Sex']
                ca_row = result_table[result_table['Source'] == 'CA_Percent']
                interaction_row = result_table[result_table['Source'] == 'Interaction']
                
                # Sex main effect (between-subjects)
                f_stat_sex = sex_row['F'].values[0] if len(sex_row) > 0 else np.nan
                p_value_sex = sex_row['p-unc'].values[0] if len(sex_row) > 0 else np.nan
                
                # CA% main effect (within-subjects)
                f_stat_ca = ca_row['F'].values[0] if len(ca_row) > 0 else np.nan
                p_value_ca = ca_row['p-unc'].values[0] if len(ca_row) > 0 else np.nan
                
                # Interaction
                f_stat_interaction = interaction_row['F'].values[0] if len(interaction_row) > 0 else np.nan
                p_value_interaction = interaction_row['p-unc'].values[0] if len(interaction_row) > 0 else np.nan
                
            else:
                # ONE-WAY REPEATED MEASURES ANOVA (CA% only)
                result_table = pg.rm_anova(
                    dv=measure,
                    within='CA_Percent',
                    subject='Animal',
                    data=df_measure,
                    detailed=True
                )
                
                # Extract results
                f_stat_ca = result_table.loc[result_table['Source'] == 'CA_Percent', 'F'].values[0]
                p_value_ca = result_table.loc[result_table['Source'] == 'CA_Percent', 'p-unc'].values[0]
                
                # Set sex/interaction to NaN for one-way design
                f_stat_sex = np.nan
                p_value_sex = np.nan
                f_stat_interaction = np.nan
                p_value_interaction = np.nan
            
            # Compute descriptive statistics per CA%
            group_stats = []
            for ca_percent in sorted(df_measure['CA_Percent'].unique()):
                ca_data = df_measure[df_measure['CA_Percent'] == ca_percent][measure]
                group_stats.append({
                    'ca_percent': ca_percent,
                    'n': len(ca_data),
                    'mean': ca_data.mean(),
                    'std': ca_data.std(ddof=1),
                    'min': ca_data.min(),
                    'max': ca_data.max()
                })
            
            # If mixed ANOVA, also compute descriptive stats per Sex
            sex_stats = []
            if has_sex_data:
                for sex_val in sorted(df_measure['Sex'].unique()):
                    sex_data = df_measure[df_measure['Sex'] == sex_val][measure]
                    sex_stats.append({
                        'sex': sex_val,
                        'n': len(sex_data),
                        'mean': sex_data.mean(),
                        'std': sex_data.std(ddof=1)
                    })
            
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                # CA% effect (main for mixed, only for RM)
                'f_statistic': f_stat_ca,
                'p_value': p_value_ca,
                'significant': p_value_ca < 0.05,
                # Sex effect (only for mixed ANOVA)
                'f_statistic_sex': f_stat_sex,
                'p_value_sex': p_value_sex,
                'significant_sex': p_value_sex < 0.05 if not np.isnan(p_value_sex) else False,
                # Interaction effect (only for mixed ANOVA)
                'f_statistic_interaction': f_stat_interaction,
                'p_value_interaction': p_value_interaction,
                'significant_interaction': p_value_interaction < 0.05 if not np.isnan(p_value_interaction) else False,
                # Descriptive stats
                'group_stats': group_stats,
                'sex_stats': sex_stats,
                'ca_labels': sorted(df_measure['CA_Percent'].unique()),
                'n_animals': n_animals,
                'n_ca_levels': n_ca_levels,
                'is_mixed_anova': has_sex_data,
                'anova_table': result_table
            }
            
            # Print results
            print(f"  Results:")
            if has_sex_data:
                sig_marker_sex = "***" if p_value_sex < 0.05 else ""
                sig_marker_ca = "***" if p_value_ca < 0.05 else ""
                sig_marker_int = "***" if p_value_interaction < 0.05 else ""
                print(f"    Sex: F = {f_stat_sex:.3f}, p = {p_value_sex:.4f} {sig_marker_sex}")
                print(f"    CA%: F = {f_stat_ca:.3f}, p = {p_value_ca:.4f} {sig_marker_ca}")
                print(f"    Sex × CA%: F = {f_stat_interaction:.3f}, p = {p_value_interaction:.4f} {sig_marker_int}")
            else:
                sig_marker = "***" if p_value_ca < 0.05 else ""
                print(f"  F = {f_stat_ca:.3f}, p = {p_value_ca:.4f} {sig_marker}")
            
            print(f"  Animals: {n_animals}, CA% levels: {n_ca_levels}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'error': f'ANOVA failed: {str(e)}',
                'significant': False,
                'is_mixed_anova': has_sex_data
            }
    
    return anova_results


def _perform_standard_anova_fallback(weekly_averages: Dict) -> Dict:
    """Fallback to standard ANOVA when pingouin is not available.
    
    WARNING: This does NOT account for repeated measures and treats observations
    as independent. Results may be invalid for repeated measures designs.
    """
    # Original one-way ANOVA implementation
    ca_groups = {}
    
    for date, data in weekly_averages.items():
        ca_percent = data['ca_percent']
        
        if ca_percent not in ca_groups:
            ca_groups[ca_percent] = {
                'licks': [],
                'bouts': [],
                'fecal': [],
                'bottle_weight': [],
                'total_weight': []
            }
        
        # Add individual animal data for each measure
        ca_groups[ca_percent]['licks'].extend(data['avg_licks_per_animal'])
        ca_groups[ca_percent]['bouts'].extend(data['avg_bouts_per_animal'])
        ca_groups[ca_percent]['fecal'].extend(data['avg_fecal_per_animal'])
        
        # For bottle weight, handle the 11/12/25 outlier exclusion
        if date == '11/12/25':
            # Filter out zeros that represent excluded outliers
            bottle_weights_filtered = [w for w in data['avg_bottle_weight_per_animal'] if w > 0]
            ca_groups[ca_percent]['bottle_weight'].extend(bottle_weights_filtered)
        else:
            ca_groups[ca_percent]['bottle_weight'].extend(data['avg_bottle_weight_per_animal'])
            
        ca_groups[ca_percent]['total_weight'].extend(data['avg_total_weight_per_animal'])
    
    # Perform ANOVA for each measure
    anova_results = {}
    measures = ['licks', 'bouts', 'fecal', 'bottle_weight', 'total_weight']
    measure_names = {
        'licks': 'Total Licks',
        'bouts': 'Total Bouts', 
        'fecal': 'Fecal Count',
        'bottle_weight': 'Bottle Weight Loss',
        'total_weight': 'Total Weight Loss'
    }
    
    for measure in measures:
        # Prepare data for ANOVA (groups must have data)
        groups_data = []
        ca_labels = []
        
        for ca_percent in sorted(ca_groups.keys()):
            if len(ca_groups[ca_percent][measure]) > 0:
                groups_data.append(ca_groups[ca_percent][measure])
                ca_labels.append(ca_percent)
        if len(groups_data) >= 2:  # Need at least 2 groups for ANOVA
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups_data)
            
            # Calculate descriptive statistics for each group
            group_stats = []
            for i, ca_percent in enumerate(ca_labels):
                data = groups_data[i]
                group_stats.append({
                    'ca_percent': ca_percent,
                    'n': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),  # Sample standard deviation
                    'min': np.min(data),
                    'max': np.max(data)
                })
            
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group_stats': group_stats,
                'ca_labels': ca_labels,
                'groups_data': groups_data
            }
        else:
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data for ANOVA (need at least 2 groups)'
            }
    
    return anova_results


def perform_tukey_hsd(anova_results: Dict, weekly_averages: Dict) -> Dict:
    """
    Perform Tukey's HSD post-hoc test for significant CA% main effects.
    
    For mixed ANOVA or repeated measures ANOVA with significant CA% effects,
    performs pairwise comparisons across CA% levels.
    
    Note: For repeated measures data, this is an approximation. Proper RM post-hoc
    tests would account for the correlated nature of repeated observations.
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        weekly_averages: Dictionary from compute_weekly_averages (needed to reconstruct data)
        
    Returns:
        Dictionary containing Tukey HSD results for significant measures
    """
    tukey_results = {}
    
    for measure, anova_data in anova_results.items():
        # Check if CA% effect is significant (either in one-way RM or mixed ANOVA)
        if not anova_data.get('significant', False):
            continue
        
        # Skip if there's an error
        if 'error' in anova_data:
            continue
        
        try:
            # Reconstruct long-format data from weekly_averages
            all_data = []
            group_labels = []
            
            # Get group stats from ANOVA results (contains CA% info)
            if 'group_stats' not in anova_data or not anova_data['group_stats']:
                continue
            
            # Map measure name to data key in weekly_averages
            measure_to_key = {
                'licks': 'avg_licks_per_animal',
                'bouts': 'avg_bouts_per_animal',
                'fecal': 'avg_fecal_per_animal',
                'bottle_weight': 'avg_bottle_weight_per_animal',
                'total_weight': 'avg_total_weight_per_animal'
            }
            
            data_key = measure_to_key.get(measure)
            if not data_key:
                continue
            
            # Reconstruct data for each CA%
            sorted_dates = sorted(weekly_averages.keys(), key=lambda d: weekly_averages[d]['ca_percent'])
            
            for date in sorted_dates:
                ca_percent = weekly_averages[date]['ca_percent']
                ca_data = weekly_averages[date][data_key]
                
                # For bottle_weight, exclude zeros (outliers)
                if measure == 'bottle_weight':
                    ca_data = [x for x in ca_data if x > 0]
                
                # Add to combined data
                all_data.extend(ca_data)
                group_labels.extend([f"{ca_percent}%"] * len(ca_data))
            
            if len(all_data) == 0 or len(set(group_labels)) < 2:
                continue
            
            # Perform Tukey HSD
            tukey_result = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
            
            # Parse results into a more accessible format
            comparisons = []
            
            # Extract pairwise comparison results directly from tukey_result
            # The tukey HSD result contains arrays for each comparison
            summary_data = tukey_result.summary().data[1:]  # Skip header row
            
            for row in summary_data:
                group1, group2, meandiff, p_adj, lower_ci, upper_ci, reject = row
                
                comparisons.append({
                    'group1': str(group1),
                    'group2': str(group2), 
                    'meandiff': float(meandiff),
                    'p_adj': float(p_adj),
                    'lower_ci': float(lower_ci),
                    'upper_ci': float(upper_ci),
                    'significant': bool(reject)
                })
            
            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'comparisons': comparisons,
                'tukey_object': tukey_result  # Store the full result for summary table
            }
            
        except Exception as e:
            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'error': f"Error performing Tukey HSD: {str(e)}"
            }
    
    return tukey_results


def display_tukey_results(tukey_results: Dict) -> str:
    """
    Display Tukey HSD results in a formatted table and return formatted string.
    
    Parameters:
        tukey_results: Dictionary from perform_tukey_hsd
        
    Returns:
        Formatted string with Tukey HSD results
    """
    if not tukey_results:
        return "\n" + "=" * 80 + "\nNo significant CA% effects found - Tukey HSD post-hoc tests not needed.\n" + "=" * 80 + "\n"
    
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TUKEY HSD POST-HOC TEST RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Tukey HSD performed for measures with significant CA% main effects.")
    lines.append("Pairwise comparisons across CA% levels (collapsed across Sex if applicable).")
    lines.append("Alpha = 0.05 (family-wise error rate controlled)")
    lines.append("")
    
    for measure, results in tukey_results.items():
        if 'error' in results:
            lines.append(f"{results['measure_name']}: {results['error']}")
            lines.append("")
            continue
            
        lines.append(f"MEASURE: {results['measure_name'].upper()}")
        lines.append("-" * 60)
        
        # Header for pairwise comparisons
        header = f"{'Comparison':<15} {'Mean Diff':<12} {'Adj p-value':<12} {'95% CI Lower':<12} {'95% CI Upper':<12} {'Significant':<12}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Sort comparisons by p-value for better readability
        sorted_comparisons = sorted(results['comparisons'], key=lambda x: x['p_adj'])
        
        for comp in sorted_comparisons:
            comparison_name = f"{comp['group1']} vs {comp['group2']}"
            significance = "Yes" if comp['significant'] else "No"
            
            row = (f"{comparison_name:<15} "
                   f"{comp['meandiff']:<12.3f} "
                   f"{comp['p_adj']:<12.6f} "
                   f"{comp['lower_ci']:<12.3f} "
                   f"{comp['upper_ci']:<12.3f} "
                   f"{significance:<12}")
            lines.append(row)
        
        lines.append("")
        
        # Summary of significant comparisons
        significant_comps = [comp for comp in results['comparisons'] if comp['significant']]
        if significant_comps:
            lines.append("Significant pairwise differences:")
            for comp in significant_comps:
                direction = "higher" if comp['meandiff'] > 0 else "lower"
                lines.append(f"  • {comp['group1']} vs {comp['group2']}: {comp['group1']} is {direction} (p = {comp['p_adj']:.4f})")
        else:
            lines.append("No significant pairwise differences found (despite significant ANOVA).")
            lines.append("This may indicate the effect is distributed across groups rather than")
            lines.append("concentrated in specific pairwise comparisons.")
        
        lines.append("")
        lines.append("")
    
    # Overall summary
    total_measures = len(tukey_results)
    measures_with_sig_pairs = len([r for r in tukey_results.values() 
                                   if 'comparisons' in r and 
                                   any(comp['significant'] for comp in r['comparisons'])])
    
    lines.append("SUMMARY OF TUKEY HSD RESULTS:")
    lines.append("-" * 40)
    lines.append(f"Measures tested: {total_measures}")
    lines.append(f"Measures with significant pairwise differences: {measures_with_sig_pairs}")
    
    if measures_with_sig_pairs > 0:
        lines.append("\nSignificant pairwise differences were found, indicating specific")
        lines.append("CA% concentrations that differ significantly from each other.")
    else:
        lines.append("\nNo significant pairwise differences found, despite significant ANOVA results.")
        lines.append("This suggests the overall effect may be more complex than simple pairwise differences.")
    
    lines.append("\n" + "=" * 80)
    lines.append("")
    
    # Join lines and print
    formatted_output = "\n".join(lines)
    print(formatted_output)
    
    return formatted_output


def display_anova_results(anova_results: Dict) -> str:
    """
    Display ANOVA results in a formatted table and return formatted string.
    Handles mixed ANOVA, repeated measures, and standard ANOVA output formats.
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        
    Returns:
        Formatted string with ANOVA results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    
    # Check if results are mixed ANOVA or standard
    is_mixed = any(results.get('is_mixed_anova', False) for results in anova_results.values())
    
    if is_mixed:
        lines.append("MIXED ANOVA RESULTS ACROSS CA% CONCENTRATIONS")
        lines.append("=" * 80)
        lines.append("Between-subjects factor: Sex")
        lines.append("Within-subjects factor: CA% (repeated measures)")
        lines.append("Subject tracking: Animal ID")
    else:
        lines.append("REPEATED MEASURES ANOVA RESULTS ACROSS CA% CONCENTRATIONS")
        lines.append("=" * 80)
        lines.append("Within-subjects factor: CA%")
        lines.append("Subject tracking: Animal ID")
    
    lines.append("")
    
    for measure, results in anova_results.items():
        if 'error' in results:
            lines.append(f"{results['measure_name']}: {results['error']}")
            lines.append("")
            continue
            
        lines.append(f"MEASURE: {results['measure_name'].upper()}")
        lines.append("-" * 50)
        
        # Show design type
        if results.get('is_mixed_anova', False):
            lines.append("Design: Mixed ANOVA (Sex between-subjects × CA% within-subjects)")
            if 'n_animals' in results and 'n_ca_levels' in results:
                lines.append(f"Animals: {results['n_animals']}, CA% levels: {results['n_ca_levels']}")
            lines.append("")
            
            # Sex main effect (between-subjects)
            lines.append("SEX MAIN EFFECT (between-subjects):")
            lines.append(f"  F-statistic: {results.get('f_statistic_sex', np.nan):.4f}")
            lines.append(f"  p-value: {results.get('p_value_sex', np.nan):.6f}")
            if results.get('significant_sex', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - Significant differences between sexes")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant sex differences")
            lines.append("")
            
            # CA% main effect (within-subjects)
            lines.append("CA% MAIN EFFECT (within-subjects):")
            lines.append(f"  F-statistic: {results['f_statistic']:.4f}")
            lines.append(f"  p-value: {results['p_value']:.6f}")
            if results.get('significant', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - Significant differences across CA% concentrations")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant CA% differences")
            lines.append("")
            
            # Interaction (Sex × CA%)
            lines.append("SEX × CA% INTERACTION:")
            lines.append(f"  F-statistic: {results.get('f_statistic_interaction', np.nan):.4f}")
            lines.append(f"  p-value: {results.get('p_value_interaction', np.nan):.6f}")
            if results.get('significant_interaction', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - CA% effect differs by sex")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - CA% effect similar across sexes")
            lines.append("")
            
        else:
            # One-way repeated measures ANOVA
            lines.append("Design: One-way repeated measures (within-subjects)")
            if 'n_animals' in results and 'n_ca_levels' in results:
                lines.append(f"Animals: {results['n_animals']}, CA% levels: {results['n_ca_levels']}")
            lines.append("")
            
            lines.append(f"F-statistic: {results['f_statistic']:.4f}")
            lines.append(f"p-value: {results['p_value']:.6f}")
            
            if results['significant']:
                lines.append(f"Result: SIGNIFICANT (p < 0.05) - Significant differences across CA% concentrations")
            else:
                lines.append(f"Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant differences")
            lines.append("")
        
        # Group descriptive statistics
        lines.append("CA% Statistics:")
        header = f"{'CA%':<6} {'N':<4} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8}"
        lines.append(header)
        lines.append("-" * 50)
        
        for group_stat in results['group_stats']:
            row = (f"{group_stat['ca_percent']:<6} "
                   f"{group_stat['n']:<4} "
                   f"{group_stat['mean']:<10.2f} "
                   f"{group_stat['std']:<10.2f} "
                   f"{group_stat['min']:<8.2f} "
                   f"{group_stat['max']:<8.2f}")
            lines.append(row)
        
        # If mixed ANOVA, also show sex stats
        if results.get('is_mixed_anova', False) and 'sex_stats' in results and results['sex_stats']:
            lines.append("")
            lines.append("Sex Statistics:")
            sex_header = f"{'Sex':<8} {'N':<4} {'Mean':<10} {'Std':<10}"
            lines.append(sex_header)
            lines.append("-" * 35)
            
            for sex_stat in results['sex_stats']:
                sex_row = (f"{sex_stat['sex']:<8} "
                           f"{sex_stat['n']:<4} "
                           f"{sex_stat['mean']:<10.2f} "
                           f"{sex_stat['std']:<10.2f}")
                lines.append(sex_row)
        
        lines.append("")
        lines.append("")
    
    # Summary of significant results
    significant_ca = [results['measure_name'] for results in anova_results.values() 
                      if results.get('significant', False)]
    significant_sex = [results['measure_name'] for results in anova_results.values() 
                       if results.get('significant_sex', False)]
    significant_int = [results['measure_name'] for results in anova_results.values() 
                       if results.get('significant_interaction', False)]
    
    lines.append("SUMMARY OF SIGNIFICANT RESULTS:")
    lines.append("-" * 40)
    
    if is_mixed:
        if significant_ca:
            lines.append(f"CA% main effects: {', '.join(significant_ca)}")
        if significant_sex:
            lines.append(f"Sex main effects: {', '.join(significant_sex)}")
        if significant_int:
            lines.append(f"Sex × CA% interactions: {', '.join(significant_int)}")
        
        if not significant_ca and not significant_sex and not significant_int:
            lines.append("No significant effects found.")
    else:
        if significant_ca:
            lines.append(f"CA% effects: {', '.join(significant_ca)}")
        else:
            lines.append("No significant CA% effects found.")
    
    if significant_ca:
        lines.append("\nConsider Tukey HSD post-hoc tests for pairwise CA% comparisons.")
    
    lines.append("\n" + "=" * 80)
    lines.append("")
    
    # Join lines and print
    formatted_output = "\n".join(lines)
    print(formatted_output)
    
    return formatted_output


def display_weekly_averages(weekly_averages: Dict) -> str:
    """Display weekly averages in a formatted table and return formatted string."""
    # Build formatted string
    lines = []
    lines.append("=" * 80)
    lines.append("WEEKLY AVERAGES SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Headers for comprehensive table
    lines.append("LICK AND BOUT AVERAGES:")
    header1 = f"{'Date':<12} {'Avg Licks':<12} {'Std Licks':<12} {'Avg Bouts':<12} {'Std Bouts':<12} {'Animals':<8}"
    lines.append(header1)
    lines.append("-" * 80)
    
    # Sort dates chronologically for proper display
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    
    for date in sorted_dates:
        avg_data = weekly_averages[date]
        row = (f"{date:<12} "
               f"{avg_data['avg_total_licks']:<12.1f} "
               f"{avg_data['std_licks']:<12.1f} "
               f"{avg_data['avg_total_bouts']:<12.1f} "
               f"{avg_data['std_bouts']:<12.1f} "
               f"{avg_data['total_animals']:<8}")
        lines.append(row)
    
    lines.append("")
    lines.append("WEIGHT AND FECAL AVERAGES:")
    header2 = f"{'Date':<12} {'Avg Fecal':<12} {'Std Fecal':<12} {'Avg Bottle':<12} {'Std Bottle':<12} {'Avg Total':<12} {'Std Total':<12}"
    lines.append(header2)
    lines.append("-" * 90)
    
    for date in sorted_dates:
        avg_data = weekly_averages[date]
        row = (f"{date:<12} "
               f"{avg_data['avg_fecal_count']:<12.1f} "
               f"{avg_data['std_fecal']:<12.1f} "
               f"{avg_data['avg_bottle_weight_loss']:<12.1f} "
               f"{avg_data['std_bottle_weight']:<12.1f} "
               f"{avg_data['avg_total_weight_loss']:<12.1f} "
               f"{avg_data['std_total_weight']:<12.1f}")
        lines.append(row)
    
    lines.append("-" * 80)
    
    # Additional summary statistics
    if len(weekly_averages) > 1:
        all_avg_licks = [avg['avg_total_licks'] for avg in weekly_averages.values()]
        all_avg_bouts = [avg['avg_total_bouts'] for avg in weekly_averages.values()]
        all_avg_fecal = [avg['avg_fecal_count'] for avg in weekly_averages.values()]
        all_avg_bottle = [avg['avg_bottle_weight_loss'] for avg in weekly_averages.values()]
        all_avg_total = [avg['avg_total_weight_loss'] for avg in weekly_averages.values()]
        
        lines.append("")
        lines.append("CROSS-WEEK STATISTICS:")
        lines.append(f"Licks - Mean: {np.mean(all_avg_licks):.1f}, "
                    f"Std: {np.std(all_avg_licks):.1f}, "
                    f"Range: {np.min(all_avg_licks):.1f} - {np.max(all_avg_licks):.1f}")
        lines.append(f"Bouts - Mean: {np.mean(all_avg_bouts):.1f}, "
                    f"Std: {np.std(all_avg_bouts):.1f}, "
                    f"Range: {np.min(all_avg_bouts):.1f} - {np.max(all_avg_bouts):.1f}")
        lines.append(f"Fecal - Mean: {np.mean(all_avg_fecal):.1f}, "
                    f"Std: {np.std(all_avg_fecal):.1f}, "
                    f"Range: {np.min(all_avg_fecal):.1f} - {np.max(all_avg_fecal):.1f}")
        lines.append(f"Bottle Weight - Mean: {np.mean(all_avg_bottle):.1f}, "
                    f"Std: {np.std(all_avg_bottle):.1f}, "
                    f"Range: {np.min(all_avg_bottle):.1f} - {np.max(all_avg_bottle):.1f}")
        lines.append(f"Total Weight - Mean: {np.mean(all_avg_total):.1f}, "
                    f"Std: {np.std(all_avg_total):.1f}, "
                    f"Range: {np.min(all_avg_total):.1f} - {np.max(all_avg_total):.1f}")
    
    lines.append("=" * 80)
    lines.append("")
    
    # Join lines and print
    formatted_output = "\n".join(lines)
    print("\n" + formatted_output)
    
    return formatted_output


def plot_weekly_averages(weekly_averages: Dict, save_path: Optional[Path] = None, show: bool = True) -> Optional[Path]:
    """Create two separate subplot figures: one for behavioral metrics, one for physiological metrics.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure (will create two files)
        show: Whether to display the plots
        
    Returns:
        Path to saved file if save_path provided, None otherwise
    """
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    
    # Extract data for plotting
    dates = []
    ca_percents = []
    avg_licks = []
    std_licks = []
    avg_bouts = []
    std_bouts = []
    avg_fecal = []
    std_fecal = []
    avg_bottle_weight = []
    std_bottle_weight = []
    avg_total_weight = []
    std_total_weight = []
    
    for date in sorted_dates:
        data = weekly_averages[date]
        dates.append(date)
        ca_percents.append(data['ca_percent'])
        avg_licks.append(data['avg_total_licks'])
        std_licks.append(data['std_licks'])
        avg_bouts.append(data['avg_total_bouts'])
        std_bouts.append(data['std_bouts'])
        avg_fecal.append(data['avg_fecal_count'])
        std_fecal.append(data['std_fecal'])
        avg_bottle_weight.append(data['avg_bottle_weight_loss'])
        std_bottle_weight.append(data['std_bottle_weight'])
        avg_total_weight.append(data['avg_total_weight_loss'])
        std_total_weight.append(data['std_total_weight'])
    
    # Convert to numpy arrays
    avg_licks = np.array(avg_licks)
    std_licks = np.array(std_licks)
    avg_bouts = np.array(avg_bouts)
    std_bouts = np.array(std_bouts)
    avg_fecal = np.array(avg_fecal)
    std_fecal = np.array(std_fecal)
    avg_bottle_weight = np.array(avg_bottle_weight)
    std_bottle_weight = np.array(std_bottle_weight)
    avg_total_weight = np.array(avg_total_weight)
    std_total_weight = np.array(std_total_weight)
    
    # ===== FIGURE 1: BEHAVIORAL METRICS =====
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    x_pos = np.arange(len(dates))
    
    # Plot 1: Average Licks
    ax1.errorbar(x_pos, avg_licks, yerr=std_licks, 
                marker='o', markersize=8, linewidth=2, capsize=5, 
                color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue')
    ax1.set_xlabel('Citric Acid Concentration (%)')
    ax1.set_ylabel('Average Licks per Animal')
    ax1.set_title('Average Licks vs Citric Acid Concentration (±SD)', fontsize=13, weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{int(ca)}%" for ca in ca_percents], rotation=45)
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Average Bouts
    ax2.errorbar(x_pos, avg_bouts, yerr=std_bouts,
                marker='s', markersize=8, linewidth=2, capsize=5,
                color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    ax2.set_xlabel('Citric Acid Concentration (%)')
    ax2.set_ylabel('Average Bouts per Animal')
    ax2.set_title('Average Lick Bouts vs Citric Acid Concentration (±SD)', fontsize=13, weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{int(ca)}%" for ca in ca_percents], rotation=45)
    ax2.set_ylim(bottom=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig1.suptitle('Behavioral Metrics vs Citric Acid Concentration', fontsize=16, weight='bold', y=0.96)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig1.subplots_adjust(hspace=0.4)
    
    # ===== FIGURE 2: PHYSIOLOGICAL METRICS =====
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot 3: Average Fecal Count
    ax3.errorbar(x_pos, avg_fecal, yerr=std_fecal,
                marker='^', markersize=8, linewidth=2, capsize=5,
                color='saddlebrown', markerfacecolor='tan', markeredgecolor='saddlebrown')
    ax3.set_xlabel('Citric Acid Concentration (%)')
    ax3.set_ylabel('Average Fecal Count per Animal')
    ax3.set_title('Average Fecal Count vs Citric Acid Concentration (±SD)', fontsize=13, weight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{int(ca)}%" for ca in ca_percents], rotation=45)
    ax3.set_ylim(bottom=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Average Bottle Weight Loss
    ax4.errorbar(x_pos, avg_bottle_weight, yerr=std_bottle_weight,
                marker='D', markersize=8, linewidth=2, capsize=5,
                color='purple', markerfacecolor='plum', markeredgecolor='purple')
    ax4.set_xlabel('Citric Acid Concentration (%)')
    ax4.set_ylabel('Average Bottle Weight Loss per Animal (g)')
    ax4.set_title('Average Bottle Weight Loss vs Citric Acid Concentration (±SD)', fontsize=13, weight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{int(ca)}%" for ca in ca_percents], rotation=45)
    ax4.set_ylim(bottom=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Plot 5: Average Total Weight Loss
    ax5.errorbar(x_pos, avg_total_weight, yerr=std_total_weight,
                marker='v', markersize=8, linewidth=2, capsize=5,
                color='darkorange', markerfacecolor='orange', markeredgecolor='darkorange')
    ax5.set_xlabel('Citric Acid Concentration (%)')
    ax5.set_ylabel('Average Total Weight Loss per Animal (g)')
    ax5.set_title('Average Total Weight Loss vs Citric Acid Concentration (±SD)', fontsize=13, weight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"{int(ca)}%" for ca in ca_percents], rotation=45)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    fig2.suptitle('Physiological Metrics vs Citric Acid Concentration', fontsize=16, weight='bold', y=0.97)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig2.subplots_adjust(hspace=0.45)
    
    # Save if requested
    if save_path is not None:
        # Create separate file names for the two figures
        base_path = save_path.with_suffix('')
        behavioral_path = base_path.with_name(f"{base_path.name}_behavioral.svg")
        physiological_path = base_path.with_name(f"{base_path.name}_physiological.svg")
        
        fig1.savefig(behavioral_path, format='svg', dpi=200, bbox_inches='tight')
        fig2.savefig(physiological_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Behavioral metrics plot saved to: {behavioral_path}")
        print(f"Physiological metrics plot saved to: {physiological_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
    
    return save_path


def save_weekly_averages_to_file(weekly_averages: Dict, formatted_output: str, save_path: Path) -> Path:
    """Save weekly averages summary to a text file.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        formatted_output: Formatted string from display_weekly_averages
        save_path: Path where to save the text file
        
    Returns:
        Path to the saved text file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("WEEKLY LICK AND BOUT AVERAGES ANALYSIS\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Write the formatted table
        f.write(formatted_output)
        
        # Additional detailed breakdown
        f.write("\nDETAILED BREAKDOWN BY WEEK\n")
        f.write("=" * 80 + "\n")
        
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
        
        for date in sorted_dates:
            data = weekly_averages[date]
            f.write(f"\nWeek: {date}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Licks per Animal: {data['avg_total_licks']:.2f} (±{data['std_licks']:.2f})\n")
            f.write(f"Average Bouts per Animal: {data['avg_total_bouts']:.2f} (±{data['std_bouts']:.2f})\n")
            f.write(f"Average Fecal Count per Animal: {data['avg_fecal_count']:.2f} (±{data['std_fecal']:.2f})\n")
            f.write(f"Average Bottle Weight Loss per Animal: {data['avg_bottle_weight_loss']:.2f} (±{data['std_bottle_weight']:.2f})\n")
            f.write(f"Average Total Weight Loss per Animal: {data['avg_total_weight_loss']:.2f} (±{data['std_total_weight']:.2f})\n")
            f.write(f"Total Animals: {data['total_animals']}\n")
            f.write(f"Total Licks (All Animals): {data['sum_total_licks']}\n")
            f.write(f"Total Bouts (All Animals): {data['sum_total_bouts']}\n")
            
            # Individual animal data
            f.write(f"Individual Animal Lick Counts: {', '.join([str(int(x)) for x in data['avg_licks_per_animal']])}\n")
            f.write(f"Individual Animal Bout Counts: {', '.join([str(int(x)) for x in data['avg_bouts_per_animal']])}\n")
            f.write(f"Individual Animal Fecal Counts: {', '.join([str(x) for x in data['avg_fecal_per_animal']])}\n")
            f.write(f"Individual Animal Bottle Weight Loss: {', '.join([str(x) for x in data['avg_bottle_weight_per_animal']])}\n")
            f.write(f"Individual Animal Total Weight Loss: {', '.join([str(x) for x in data['avg_total_weight_per_animal']])}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
    
    return save_path


def save_brief_lick_summary(weekly_averages: Dict, save_path: Path) -> Path:
    """Save a brief summary of weekly lick counts by animal and averages.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Path where to save the text file
        
    Returns:
        Path to the saved text file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("WEEKLY LICK COUNTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
        
        for i, date in enumerate(sorted_dates, 1):
            data = weekly_averages[date]
            
            f.write(f"Week {i} ({date}):" + "\n")
            
            # Individual animal lick counts
            lick_counts = [int(x) for x in data['avg_licks_per_animal']]
            f.write(f"  Individual licks by animal: {', '.join(map(str, lick_counts))}" + "\n")
            
            # Average licks
            f.write(f"  Average licks per animal: {data['avg_total_licks']:.1f}" + "\n")
            
            f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    return save_path


def parse_date_string(date_str: str) -> datetime:
    """Parse MM/DD/YY date string to datetime object for proper sorting.
    
    Parameters:
        date_str: Date string in MM/DD/YY format (e.g., '11/5/25', '11/12/25')
        
    Returns:
        datetime object
    """
    try:
        # Parse MM/DD/YY format - assumes 20XX for years
        return datetime.strptime(date_str, '%m/%d/%y')
    except ValueError:
        try:
            # Try M/D/YY format (single digit month/day)
            return datetime.strptime(date_str, '%m/%d/%y')
        except ValueError:
            # Fallback: return a default date if parsing fails
            print(f"Warning: Could not parse date '{date_str}', using fallback")
            return datetime(2025, 1, 1)


def sort_dates_chronologically(dates: List[str]) -> List[str]:
    """Sort date strings in chronological order.
    
    Parameters:
        dates: List of date strings in MM/DD/YY format
        
    Returns:
        List of date strings sorted chronologically
    """
    # Create list of tuples (original_string, datetime_object)
    date_tuples = [(date_str, parse_date_string(date_str)) for date_str in dates]
    
    # Sort by datetime object
    date_tuples.sort(key=lambda x: x[1])
    
    # Return sorted original strings
    return [date_str for date_str, _ in date_tuples]


def bin_licks_by_minute(events_df: pd.DataFrame, sensor_col: str, max_duration_min: float = 30.0, bin_size_min: float = 5.0) -> np.ndarray:
    """Bin lick events into time intervals for a single sensor.
    
    Parameters:
        events_df: DataFrame with Time_sec and event columns
        sensor_col: Sensor column name (e.g., 'Sensor_20')
        max_duration_min: Maximum duration to consider in minutes (default: 30)
        bin_size_min: Size of each bin in minutes (default: 5)
        
    Returns:
        Array of lick counts for each bin (length = max_duration_min / bin_size_min)
    """
    event_col = f"{sensor_col}_event"
    
    # Calculate number of bins
    num_bins = int(max_duration_min / bin_size_min)
    
    if event_col not in events_df.columns:
        # Return zeros if sensor has no events
        return np.zeros(num_bins)
    
    # Get event times in seconds
    event_times = events_df[events_df[event_col]]['Time_sec'].values
    
    # Convert to minutes
    event_times_min = event_times / 60.0
    
    # Filter to only events within max_duration
    event_times_min = event_times_min[event_times_min < max_duration_min]
    
    # Create bins (0-5 min, 5-10 min, ..., 25-30 min for bin_size=5)
    bins = np.arange(0, max_duration_min + bin_size_min, bin_size_min)
    
    # Count licks in each bin
    lick_counts, _ = np.histogram(event_times_min, bins=bins)
    
    return lick_counts


def compute_weekly_lick_rate_averages(weekly_results: Dict, max_duration_min: float = 30.0, bin_size_min: float = 5.0) -> Dict:
    """Compute average lick rate across all animals for each week.
    
    For each week, bins licks into time intervals and computes mean and SEM
    across all animals.
    
    Parameters:
        weekly_results: Dictionary from process_single_week
        max_duration_min: Maximum duration in minutes (default: 30)
        bin_size_min: Size of each bin in minutes (default: 5)
        
    Returns:
        Dictionary with keys:
            - date: Date string
            - ca_percent: CA percentage
            - mean_licks: Mean licks per bin (array of length max_duration_min/bin_size_min)
            - sem_licks: SEM of licks per bin (array of length max_duration_min/bin_size_min)
            - n_animals: Number of animals
            - animal_ids: List of animal IDs
            - bin_size_min: Bin size in minutes
    """
    lick_rate_data = {}
    
    print("\n" + "=" * 80)
    print(f"COMPUTING LICK RATE AVERAGES ({bin_size_min}-MINUTE BINS)")
    print("=" * 80)
    
    for date, result in weekly_results.items():
        print(f"\nProcessing {date} (CA {result['ca_percent']}%):")
        
        events_df = result['events_df']
        selected_sensors = result['selected_sensors']
        animal_ids = result.get('animal_ids', [])
        
        # Collect lick counts per minute for each animal
        all_animal_lick_rates = []
        
        for sensor in selected_sensors:
            lick_counts = bin_licks_by_minute(events_df, sensor, max_duration_min, bin_size_min)
            all_animal_lick_rates.append(lick_counts)
            print(f"  {sensor}: Total licks in {int(max_duration_min)} min = {lick_counts.sum()}")
        
        # Convert to array (animals x time_bins)
        all_animal_lick_rates = np.array(all_animal_lick_rates)
        
        # Compute mean and SEM across animals for each time bin
        mean_licks = np.mean(all_animal_lick_rates, axis=0)
        sem_licks = np.std(all_animal_lick_rates, axis=0, ddof=1) / np.sqrt(len(selected_sensors))
        
        print(f"  Mean licks per {bin_size_min}-min bin (across {len(selected_sensors)} animals):")
        print(f"    Total in {int(max_duration_min)} min: {mean_licks.sum():.1f}")
        print(f"    Mean per {bin_size_min}-min bin: {mean_licks.mean():.2f} ± {sem_licks.mean():.2f}")
        
        lick_rate_data[date] = {
            'date': date,
            'ca_percent': result['ca_percent'],
            'mean_licks': mean_licks,
            'sem_licks': sem_licks,
            'n_animals': len(selected_sensors),
            'animal_ids': animal_ids,  # NEW: Track animal IDs for unique count
            'all_animal_data': all_animal_lick_rates,  # Keep for comprehensive plot
            'bin_size_min': bin_size_min
        }
    
    print("\n" + "=" * 80)
    return lick_rate_data


def plot_lick_rate_histogram(
    date: str,
    mean_licks: np.ndarray,
    sem_licks: np.ndarray,
    ca_percent: int,
    n_animals: int,
    bin_size_min: float = 5.0,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot lick rate histogram for a single week.
    
    Parameters:
        date: Date string
        mean_licks: Mean licks per bin (array)
        sem_licks: SEM of licks per bin (array)
        ca_percent: CA percentage
        n_animals: Number of animals
        bin_size_min: Size of each bin in minutes
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # X-axis: bin centers (for positioning bars)
    num_bins = len(mean_licks)
    bin_edges = np.arange(0, num_bins * bin_size_min + bin_size_min, bin_size_min)
    bin_centers = bin_edges[:-1] + bin_size_min / 2
    
    # Create bar plot with error bars - width equals bin size for continuous bars
    ax.bar(bin_centers, mean_licks, width=bin_size_min, color='steelblue', alpha=0.7, 
           edgecolor='black', linewidth=1.2, yerr=sem_licks, capsize=3,
           error_kw={'elinewidth': 1.5, 'capthick': 1.5}, align='center')
    
    # Labels and title
    ax.set_xlabel('Time (minutes)', fontsize=12, weight='bold')
    ax.set_ylabel(f'Licks per {bin_size_min}-Minute Bin (Mean ± SEM)', fontsize=12, weight='bold')
    ax.set_title(f'Lick Rate - Week: {date} (CA {ca_percent}%, n={n_animals})', 
                fontsize=14, weight='bold')
    
    # Format x-axis to show bin edges
    ax.set_xlim(0, num_bins * bin_size_min)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(x)}' for x in bin_edges])
    
    # Format y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Lick rate plot saved to: {save_path}")
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_comprehensive_lick_rate_by_ca(
    lick_rate_data: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot comprehensive lick rate histogram collapsing across CA% concentrations.
    
    Averages lick rates across all CA% concentrations and all animals.
    
    Parameters:
        lick_rate_data: Dictionary from compute_weekly_lick_rate_averages
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Figure object
    """
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE LICK RATE PLOT (ALL CA% COMBINED)")
    print("=" * 80)
    
    # Organize data by CA% concentration
    ca_groups = {}
    bin_size_min = None
    all_unique_animal_ids = set()  # Track unique animal IDs across all weeks/CA%
    
    for date, data in lick_rate_data.items():
        ca_percent = data['ca_percent']
        
        if ca_percent not in ca_groups:
            ca_groups[ca_percent] = []
        
        ca_groups[ca_percent].append(data['all_animal_data'])
        
        # Track unique animal IDs
        if 'animal_ids' in data and data['animal_ids']:
            print(f"  {date} (CA {ca_percent}%): Adding {len(data['animal_ids'])} animal IDs: {data['animal_ids'][:3]}..." if len(data['animal_ids']) > 3 else f"  {date} (CA {ca_percent}%): Adding animal IDs: {data['animal_ids']}")
            all_unique_animal_ids.update(data['animal_ids'])
        else:
            print(f"  {date} (CA {ca_percent}%): No animal IDs found in data")
        
        if bin_size_min is None:
            bin_size_min = data.get('bin_size_min', 5.0)
    
    print(f"\nCollected {len(all_unique_animal_ids)} unique animal IDs total")
    if all_unique_animal_ids:
        print(f"Unique IDs: {sorted(all_unique_animal_ids)}")
    
    # Print summary of data by CA%
    for ca_percent in sorted(ca_groups.keys()):
        total_animals = sum(arr.shape[0] for arr in ca_groups[ca_percent])
        print(f"  CA {ca_percent}%: {total_animals} observations across {len(ca_groups[ca_percent])} week(s)")
    
    # Concatenate all animal data from all CA% concentrations
    all_data = []
    for ca_percent in sorted(ca_groups.keys()):
        all_data.extend(ca_groups[ca_percent])
    
    all_animals_all_ca = np.vstack(all_data)
    n_unique_animals = len(all_unique_animal_ids) if all_unique_animal_ids else all_animals_all_ca.shape[0]
    
    print(f"\nTotal observations: {all_animals_all_ca.shape[0]}")
    print(f"Unique mice: {n_unique_animals}")
    
    # Compute mean and SEM across all animals from all CA% concentrations
    mean_licks = np.mean(all_animals_all_ca, axis=0)
    sem_licks = np.std(all_animals_all_ca, axis=0, ddof=1) / np.sqrt(all_animals_all_ca.shape[0])
    
    print(f"Overall mean licks per {bin_size_min}-min bin: {mean_licks.mean():.2f} ± {sem_licks.mean():.2f}")
    print(f"Total licks in 30 min: {mean_licks.sum():.1f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # X-axis: bin centers (for positioning bars)
    num_bins = len(mean_licks)
    bin_edges = np.arange(0, num_bins * bin_size_min + bin_size_min, bin_size_min)
    bin_centers = bin_edges[:-1] + bin_size_min / 2
    
    # Create bar plot with error bars - width equals bin size for continuous bars
    ax.bar(bin_centers, mean_licks, width=bin_size_min, color='darkgreen', alpha=0.7,
           edgecolor='black', linewidth=1.2, yerr=sem_licks, capsize=3,
           error_kw={'elinewidth': 1.5, 'capthick': 1.5}, align='center')
    
    # Labels and title
    ax.set_xlabel('Time (minutes)', fontsize=12, weight='bold')
    ax.set_ylabel(f'Licks per {bin_size_min}-Minute Bin (Mean ± SEM)', fontsize=12, weight='bold')
    ax.set_title(f'Comprehensive Lick Rate - All CA% Combined (n={n_unique_animals} unique mice, {len(ca_groups)} CA% concentrations)',
                fontsize=14, weight='bold')
    
    # Format x-axis to show bin edges
    ax.set_xlim(0, num_bins * bin_size_min)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(x)}' for x in bin_edges])
    
    # Format y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    print("=" * 80 + "\n")
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Comprehensive lick rate plot saved to: {save_path}")
    
    if not show:
        plt.close(fig)
    
    return fig


def load_master_csv(csv_path: Path) -> pd.DataFrame:
    """Load the master metadata CSV with all weeks of data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Clean date column
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).str.strip()
    
    return df


def get_available_dates(master_df: pd.DataFrame) -> List[str]:
    """Get chronologically sorted list of unique dates from master CSV."""
    if 'date' not in master_df.columns:
        raise ValueError("Master CSV must have a 'date' column")
    
    unique_dates = master_df['date'].unique().tolist()
    return sort_dates_chronologically(unique_dates)


def select_multiple_files_tkinter(title: str, filetypes: List[Tuple[str, str]], initial_dir: Optional[Path] = None) -> List[Path]:
    """Open file dialog to select multiple files."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_paths = filedialog.askopenfilenames(
            title=title,
            filetypes=filetypes,
            initialdir=str(initial_dir) if initial_dir else None
        )
        
        root.destroy()
        
        return [Path(fp) for fp in file_paths] if file_paths else []
    
    except Exception:
        return []


def select_single_file_tkinter(title: str, filetypes: List[Tuple[str, str]], initial_dir: Optional[Path] = None) -> Optional[Path]:
    """Open file dialog to select a single file."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=str(initial_dir) if initial_dir else None
        )
        
        root.destroy()
        
        return Path(file_path) if file_path else None
    
    except Exception:
        return None


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Try to extract date from capacitive log filename.
    
    Expected format: capacitive_log_2025-MM-DD.csv or capacitive_log_2025-M-D.csv
    Returns date in MM/DD/YY format to match master CSV.
    """
    # Pattern for capacitive_log_YYYY-M-D.csv or capacitive_log_YYYY-MM-DD.csv
    # Allows single or double digit months/days
    pattern = r'capacitive_log_(\d{4})-(\d{1,2})-(\d{1,2})\.csv'
    match = re.search(pattern, filename)
    
    if match:
        year, month, day = match.groups()
        # Convert to MM/DD/YY format
        year_short = year[2:]  # Get last 2 digits
        return f"{int(month)}/{int(day)}/{year_short}"
    
    return None


def match_capacitive_files_to_dates(capacitive_files: List[Path], available_dates: List[str]) -> Dict[str, Path]:
    """Match capacitive files to dates from master CSV."""
    date_to_file = {}
    unmatched_files = []
    
    print("\n" + "=" * 60)
    print("MATCHING CAPACITIVE FILES TO DATES")
    print("=" * 60)
    
    for file_path in capacitive_files:
        extracted_date = extract_date_from_filename(file_path.name)
        
        if extracted_date and extracted_date in available_dates:
            date_to_file[extracted_date] = file_path
            print(f"✓ {file_path.name} → {extracted_date}")
        else:
            unmatched_files.append(file_path)
            print(f"✗ {file_path.name} → Could not match (extracted: {extracted_date})")
    
    print(f"\nMatched: {len(date_to_file)} files")
    print(f"Unmatched: {len(unmatched_files)} files")
    
    if unmatched_files:
        print("\nUnmatched files:")
        for file_path in unmatched_files:
            print(f"  - {file_path.name}")
        
        # Allow manual matching
        print("\nAvailable dates from master CSV:")
        for i, date in enumerate(available_dates, 1):
            if date not in date_to_file:
                print(f"  {i}. {date} (unassigned)")
        
        print("\nYou can manually assign unmatched files if needed.")
    
    print("=" * 60 + "\n")
    
    return date_to_file


def process_single_week(
    date: str,
    capacitive_file: Path,
    master_df: pd.DataFrame,
    fixed_threshold: float = 0.01,
    ili_cutoff: float = 0.3
) -> Dict:
    """Process a single week's data and return summary statistics."""
    print(f"Processing {date}: {capacitive_file.name}")
    
    # Load and process capacitive data
    df = load_capacitive_csv(capacitive_file)
    sensor_cols = get_sensor_columns(df)
    
    # Compute KDE baselines and normalizations (same as lick_detection.py)
    sensor_kdes = compute_sensor_KDE(df, sensor_cols)
    df = compute_KDE_normalizations(df, sensor_cols, sensor_kdes)
    
    # Use fixed threshold (same as lick_detection.py)
    thresholds = pd.Series({sensor: fixed_threshold for sensor in sensor_cols})
    events_df = detect_events_above_threshold(df, sensor_cols, thresholds)
    
    # Compute bouts
    bout_dict = compute_lick_bouts(events_df, sensor_cols, ili_cutoff)
    
    # Get metadata for this date
    date_metadata = master_df[master_df['date'] == date].copy()
    
    # Extract sensor mappings, animal IDs, sex, and CA%
    sensor_to_weight = {}
    sensor_to_weight_loss = {}
    sensor_to_fecal = {}
    sensor_to_animal_id = {}  # Track animal IDs for repeated measures
    sensor_to_sex = {}  # Track sex for mixed ANOVA
    ca_percent = None
    
    for _, row in date_metadata.iterrows():
        sensor_num = int(row['selected_sensors'])
        sensor_name = f"Sensor_{sensor_num}"
        sensor_to_weight[sensor_name] = float(row['bottle_weight_change'])
        sensor_to_weight_loss[sensor_name] = float(row['total_weight_change'])
        
        # Extract CA% (should be the same for all rows in this date)
        if ca_percent is None and 'CA_%' in row and pd.notna(row['CA_%']):
            ca_percent = int(row['CA_%'])
        
        # Extract animal ID (check multiple possible column names)
        if 'animal_ID' in row and pd.notna(row['animal_ID']):
            sensor_to_animal_id[sensor_name] = str(row['animal_ID']).strip()
        elif 'animal_id' in row and pd.notna(row['animal_id']):
            sensor_to_animal_id[sensor_name] = str(row['animal_id']).strip()
        elif 'ID' in row and pd.notna(row['ID']):
            sensor_to_animal_id[sensor_name] = str(row['ID']).strip()
        elif 'mouse_id' in row and pd.notna(row['mouse_id']):
            sensor_to_animal_id[sensor_name] = str(row['mouse_id']).strip()
        else:
            sensor_to_animal_id[sensor_name] = sensor_name  # Fallback to sensor name
        
        # Extract sex (use sex or Sex column)
        if 'sex' in row and pd.notna(row['sex']):
            sensor_to_sex[sensor_name] = str(row['sex']).upper().strip()
        elif 'Sex' in row and pd.notna(row['Sex']):
            sensor_to_sex[sensor_name] = str(row['Sex']).upper().strip()
        else:
            sensor_to_sex[sensor_name] = "Unknown"
        
        # Add fecal count if available in the CSV
        if 'fecal_count' in row and pd.notna(row['fecal_count']):
            sensor_to_fecal[sensor_name] = int(row['fecal_count'])
        else:
            sensor_to_fecal[sensor_name] = 0.0  # Default to 0 if not available
    
    # Default CA% to 0 if not found
    if ca_percent is None:
        ca_percent = 0
    
    # Compute summary statistics for selected sensors only
    selected_sensors = list(sensor_to_weight.keys())
    
    # Calculate total licks and bouts for selected sensors
    total_licks = 0
    total_bouts = 0
    total_bout_licks = 0
    
    lick_counts = []
    bout_counts = []
    weights = []
    weight_losses = []
    fecal_counts = []
    animal_ids = []  # Track animal IDs in same order as measurements
    
    for sensor in selected_sensors:
        # Lick counts
        event_col = f"{sensor}_event"
        if event_col in events_df.columns:
            sensor_licks = events_df[event_col].sum()
            total_licks += sensor_licks
            lick_counts.append(sensor_licks)
        else:
            lick_counts.append(0)
        
        # Bout counts
        if sensor in bout_dict:
            sensor_bouts = bout_dict[sensor]['bout_count']
            sensor_bout_licks = bout_dict[sensor]['bout_sizes'].sum() if len(bout_dict[sensor]['bout_sizes']) > 0 else 0
            total_bouts += sensor_bouts
            total_bout_licks += sensor_bout_licks
            bout_counts.append(sensor_bouts)
        else:
            bout_counts.append(0)
        
        # Weights and fecal counts
        # Exclude R9O bottle weight outlier for 11/12/25 (R9O maps to Sensor_10)
        if date == '11/12/25' and sensor == 'Sensor_10':
            # Skip R9O bottle weight outlier (3.49) for this week
            weights.append(0)  # Use 0 as placeholder to exclude from average
        else:
            weights.append(sensor_to_weight.get(sensor, 0))
        
        weight_losses.append(sensor_to_weight_loss.get(sensor, 0))
        fecal_counts.append(sensor_to_fecal.get(sensor, 0))
        animal_ids.append(sensor_to_animal_id.get(sensor, sensor))
    
    # Extract sex information for each animal (in same order as other data)
    animal_sexes = [sensor_to_sex.get(sensor, "Unknown") for sensor in selected_sensors]
    
    # Calculate correlations
    lick_weight_corr = np.corrcoef(lick_counts, weights)[0, 1] if len(lick_counts) > 1 else np.nan
    bout_weight_corr = np.corrcoef(bout_counts, weights)[0, 1] if len(bout_counts) > 1 else np.nan
    lick_weightloss_corr = np.corrcoef(lick_counts, weight_losses)[0, 1] if len(lick_counts) > 1 else np.nan
    bout_weightloss_corr = np.corrcoef(bout_counts, weight_losses)[0, 1] if len(bout_counts) > 1 else np.nan
    weight_weightloss_corr = np.corrcoef(weights, weight_losses)[0, 1] if len(weights) > 1 else np.nan
    
    return {
        'date': date,
        'ca_percent': ca_percent,
        'capacitive_file': capacitive_file.name,
        'selected_sensors': selected_sensors,
        'total_licks': total_licks,
        'total_bouts': total_bouts,
        'total_bout_licks': total_bout_licks,
        'avg_licks_per_bout': total_bout_licks / total_bouts if total_bouts > 0 else 0,
        'lick_counts': np.array(lick_counts),
        'bout_counts': np.array(bout_counts),
        'fecal_counts': np.array(fecal_counts),
        'weights': np.array(weights),
        'weight_losses': np.array(weight_losses),
        'animal_ids': animal_ids,  # NEW: Track animal IDs for repeated measures
        'animal_sexes': animal_sexes,  # NEW: Track sex for mixed ANOVA
        'correlations': {
            'lick_weight': lick_weight_corr,
            'bout_weight': bout_weight_corr,
            'lick_weightloss': lick_weightloss_corr,
            'bout_weightloss': bout_weightloss_corr,
            'weight_weightloss': weight_weightloss_corr
        },
        'sensor_to_weight': sensor_to_weight,
        'sensor_to_weight_loss': sensor_to_weight_loss,
        'bout_dict': bout_dict,
        'events_df': events_df,
        'thresholds': thresholds
    }


def main():
    """Main function for weekly comparison analysis."""
    print("=" * 80)
    print("WEEKLY COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Select master CSV file
    print("\nStep 1: Select Master CSV File")
    print("-" * 40)
    master_csv = select_single_file_tkinter(
        title="Select Master CSV File (with all weeks metadata)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initial_dir=Path.cwd()
    )
    
    if master_csv is None:
        print("No master CSV selected. Exiting.")
        return
    
    print(f"Selected master CSV: {master_csv.name}")
    
    # Load master CSV
    try:
        master_df = load_master_csv(master_csv)
        available_dates = get_available_dates(master_df)
        print(f"Available dates: {', '.join(available_dates)}")
    except Exception as e:
        print(f"Error loading master CSV: {e}")
        return
    
    # Select capacitive log files
    print("\nStep 2: Select Capacitive Log Files")
    print("-" * 40)
    capacitive_files = select_multiple_files_tkinter(
        title="Select Capacitive Log Files (hold Ctrl for multiple)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initial_dir=master_csv.parent
    )
    
    if not capacitive_files:
        print("No capacitive files selected. Exiting.")
        return
    
    print(f"Selected {len(capacitive_files)} capacitive files:")
    for file_path in capacitive_files:
        print(f"  - {file_path.name}")
    
    # Match files to dates
    date_to_file = match_capacitive_files_to_dates(capacitive_files, available_dates)
    
    if not date_to_file:
        print("No files could be matched to dates. Exiting.")
        return
    
    # Process each week
    print("\nStep 3: Processing Weekly Data")
    print("-" * 40)
    
    weekly_results = {}
    
    for date in sort_dates_chronologically(list(date_to_file.keys())):
        capacitive_file = date_to_file[date]
        try:
            result = process_single_week(date, capacitive_file, master_df)
            weekly_results[date] = result
            print(f"✓ Processed {date}")
        except Exception as e:
            print(f"✗ Error processing {date}: {e}")
    
    print(f"\nSuccessfully processed {len(weekly_results)} weeks of data")
    
    # Calculate weekly averages
    print("\nStep 4: Computing Weekly Averages")
    print("-" * 40)
    
    weekly_averages = compute_weekly_averages(weekly_results)
    formatted_output = display_weekly_averages(weekly_averages)
    
    # Perform ANOVA analysis
    print("\nStep 5: Performing One-Way ANOVA Analysis")
    print("-" * 40)
    
    anova_results = perform_anova_analysis(weekly_averages)
    anova_output = display_anova_results(anova_results)
    
    # Perform Tukey HSD for significant results
    print("\nStep 6: Performing Tukey HSD Post-Hoc Tests")
    print("-" * 40)
    
    tukey_results = perform_tukey_hsd(anova_results, weekly_averages)
    tukey_output = display_tukey_results(tukey_results)
    
    # Compute and plot lick rate histograms (5-minute bins over 30 minutes)
    print("\nStep 7: Computing Lick Rate Averages (5-Minute Bins)")
    print("-" * 40)
    
    lick_rate_data = compute_weekly_lick_rate_averages(weekly_results, max_duration_min=30.0, bin_size_min=5.0)
    
    # Plot individual week lick rate histograms
    print("\nPlotting individual week lick rate histograms...")
    sorted_dates = sort_dates_chronologically(list(lick_rate_data.keys()))
    
    for date in sorted_dates:
        data = lick_rate_data[date]
        plot_lick_rate_histogram(
            date=date,
            mean_licks=data['mean_licks'],
            sem_licks=data['sem_licks'],
            ca_percent=data['ca_percent'],
            n_animals=data['n_animals'],
            bin_size_min=data['bin_size_min'],
            show=True  # Show plots as they're created
        )
    
    # Plot comprehensive lick rate histogram (all CA% combined)
    print("\nPlotting comprehensive lick rate histogram (all CA% combined)...")
    plot_comprehensive_lick_rate_by_ca(lick_rate_data, show=True)
    
    # Plot comprehensive weekly averages
    print("\nPlotting comprehensive weekly averages (licks, bouts, fecal, weights)...")
    plot_weekly_averages(weekly_averages, show=True)
    
    # Optional: Save comprehensive summary with all statistical results
    save_table = input("\nSave weekly averages, ANOVA, and Tukey HSD results as text file? (y/n): ").strip().lower()
    if save_table in ['y', 'yes']:
        table_path = master_csv.parent / "comprehensive_statistical_analysis_summary.txt"
        # Combine all outputs
        combined_output = formatted_output + "\n" + anova_output + "\n" + tukey_output
        save_weekly_averages_to_file(weekly_averages, combined_output, table_path)
        print(f"Comprehensive statistical analysis saved to: {table_path}")
    
    # Always save brief lick summary
    brief_path = master_csv.parent / "weekly_lick_summary_brief.txt"
    save_brief_lick_summary(weekly_averages, brief_path)
    print(f"Brief lick summary saved to: {brief_path}")
    
    # Optional: Save plots
    save_plot = input("\nSave weekly averages plots as SVG? (y/n): ").strip().lower()
    if save_plot in ['y', 'yes']:
        save_path = master_csv.parent / "weekly_averages_plots.svg"
        plot_weekly_averages(weekly_averages, save_path=save_path, show=False)
        print("Note: Two separate files created - one for behavioral metrics, one for physiological metrics")
    
    # Optional: Save lick rate histogram plots
    save_lick_rate = input("\nSave lick rate histogram plots as SVG? (y/n): ").strip().lower()
    if save_lick_rate in ['y', 'yes']:
        # Save individual week plots
        sorted_dates = sort_dates_chronologically(list(lick_rate_data.keys()))
        for i, date in enumerate(sorted_dates, 1):
            data = lick_rate_data[date]
            save_path = master_csv.parent / f"lick_rate_week_{i}_{date.replace('/', '-')}.svg"
            fig = plot_lick_rate_histogram(
                date=date,
                mean_licks=data['mean_licks'],
                sem_licks=data['sem_licks'],
                ca_percent=data['ca_percent'],
                n_animals=data['n_animals'],
                bin_size_min=data['bin_size_min'],
                save_path=save_path,
                show=False
            )
            plt.close(fig)  # Close after saving to free memory
        
        # Save comprehensive plot
        comprehensive_save_path = master_csv.parent / "lick_rate_comprehensive_all_ca.svg"
        fig_comp = plot_comprehensive_lick_rate_by_ca(lick_rate_data, save_path=comprehensive_save_path, show=False)
        plt.close(fig_comp)  # Close after saving
        print(f"Saved {len(sorted_dates)} individual week plots + 1 comprehensive plot")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Analyzed: Licks, Bouts, Fecal Counts, Bottle Weight Loss, Total Weight Loss")
    print(f"Statistical Analysis: One-way ANOVA and Tukey HSD post-hoc tests completed")
    print(f"All pairwise comparisons identified for significant measures")
    print(f"Lick Rate Analysis: 5-minute bins over 30 minutes for each week + comprehensive combined plot")
    
    return weekly_results, weekly_averages, anova_results, tukey_results


if __name__ == "__main__":
    main()