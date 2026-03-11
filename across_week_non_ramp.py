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

# Try importing pingouin for repeated measures ANOVA
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("WARNING: pingouin not installed. Repeated measures ANOVA will not be available.")
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


def compute_sensor_KDE(df: pd.DataFrame, sensor_cols: List[str], verbose: bool = False) -> pd.Series:
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
                
                # Verbose output for troubleshooting
                if verbose:
                    print(f"  {col}: KDE={kdes[col]:.2f}, mean={series.mean():.2f}, "
                          f"std={series.std():.2f}, min={min_val:.2f}, max={max_val:.2f}")
            except Exception:
                # Fall back to mean if KDE fails
                kdes[col] = series.mean()
                if verbose:
                    print(f"  {col}: KDE failed, using mean={kdes[col]:.2f}")
        else:
            kdes[col] = series.iloc[0] if len(series) == 1 else None
            if verbose:
                print(f"  {col}: Insufficient data, KDE={kdes[col]}")
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


def compute_dynamic_thresholds(df: pd.DataFrame, sensor_cols: List[str], z_threshold: float = 4.0, verbose: bool = False) -> pd.Series:
    """Compute dynamic thresholds for event detection."""
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
                
                # Verbose output for troubleshooting
                if verbose and threshold > 20:  # Flag unusually high thresholds
                    print(f"\n  *** HIGH THRESHOLD WARNING for {sensor_col} ***")
                    print(f"      Threshold: {threshold:.2f}")
                    print(f"      Mean deviation: {mean_dev:.2f}")
                    print(f"      Std deviation: {std_dev:.2f}")
                    print(f"      Formula: {mean_dev:.2f} + ({z_threshold} * {std_dev:.2f}) = {threshold:.2f}")
            else:
                thresholds[sensor_col] = np.nan
        else:
            thresholds[sensor_col] = np.nan
    
    return pd.Series(thresholds)


def detect_events_above_threshold(df: pd.DataFrame, sensor_cols: List[str], thresholds: pd.Series) -> pd.DataFrame:
    """Detect time points where deviation exceeds the dynamic threshold for each sensor.
    
    Creates boolean columns indicating when each sensor's deviation peaks above the threshold.
    Uses scipy.signal.find_peaks for robust peak detection in discrete sampled data.
    
    Parameters:
        df: DataFrame with Time_sec and deviation columns
        sensor_cols: List of sensor column names
        thresholds: Series of threshold values per sensor (from compute_dynamic_thresholds)
        
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
        # This aligns the derivative with the current time point where the change begins
        # derivative[i] = deviations[i+1] - deviations[i]
        clean_deviations = deviations.fillna(0)
        derivative = np.zeros_like(clean_deviations)
        derivative[:-1] = np.diff(clean_deviations)  # Forward difference
        derivative[-1] = derivative[-2] if len(derivative) > 1 else 0  # Handle last point
        result[deriv_col] = derivative
        
        # Find peaks in the deviation signal using scipy.signal.find_peaks
        # This is much more robust than simple threshold comparison
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
    
    NOW PRESERVES ANIMAL IDs FOR REPEATED MEASURES ANALYSIS.
    
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
            - animal_ids: List of animal IDs (for repeated measures tracking)
    """
    averages = {}
    
    print("\n" + "=" * 80)
    print("DETAILED CALCULATION VERIFICATION")
    print("=" * 80)
    
    for date, result in weekly_results.items():
        print(f"\n--- Processing Date: {date} (CA {result['ca_percent']}%) ---")
        
        lick_counts = result['lick_counts']
        bout_counts = result['bout_counts']
        fecal_counts = result['fecal_counts']
        bottle_weights = result['weights']  # bottle weight change
        total_weights = result['weight_losses']  # total weight change
        
        print(f"\nNumber of animals: {len(lick_counts)}")
        print(f"Individual lick counts: {lick_counts}")
        print(f"Individual bout counts: {bout_counts}")
        print(f"Individual fecal counts: {fecal_counts}")
        print(f"Individual bottle weights: {bottle_weights}")
        print(f"Individual total weights: {total_weights}")
        
        # Calculate averages and statistics for all metrics
        avg_licks = np.mean(lick_counts)
        avg_bouts = np.mean(bout_counts)
        avg_fecal = np.mean(fecal_counts)
        
        print(f"\nLick calculation: sum={np.sum(lick_counts)}, mean={avg_licks:.2f}")
        print(f"Bout calculation: sum={np.sum(bout_counts)}, mean={avg_bouts:.2f}")
        print(f"Fecal calculation: sum={np.sum(fecal_counts)}, mean={avg_fecal:.2f}")
        
        # For bottle weights, exclude zeros that represent excluded outliers for 11/12/25
        if date == '11/12/25':
            print(f"\n*** SPECIAL HANDLING FOR {date}: Excluding R9O outlier (zeros) from bottle weight ***")
            # Filter out the zero placeholder for R9O outlier
            bottle_weights_filtered = bottle_weights[bottle_weights > 0]
            print(f"Original bottle weights: {bottle_weights}")
            print(f"Filtered bottle weights (excluding zeros): {bottle_weights_filtered}")
            print(f"Number of animals included: {len(bottle_weights_filtered)}")
            avg_bottle_weight = np.mean(bottle_weights_filtered) if len(bottle_weights_filtered) > 0 else 0
            std_bottle_weight = np.std(bottle_weights_filtered) if len(bottle_weights_filtered) > 0 else 0
            print(f"Bottle weight calculation: sum={np.sum(bottle_weights_filtered)}, mean={avg_bottle_weight:.2f}")
        else:
            avg_bottle_weight = np.mean(bottle_weights)
            std_bottle_weight = np.std(bottle_weights)
            print(f"\nBottle weight calculation: sum={np.sum(bottle_weights)}, mean={avg_bottle_weight:.2f}")
        
        avg_total_weight = np.mean(total_weights)
        print(f"Total weight calculation: sum={np.sum(total_weights)}, mean={avg_total_weight:.2f}")
        
        std_licks = np.std(lick_counts)
        std_bouts = np.std(bout_counts)
        std_fecal = np.std(fecal_counts)
        std_total_weight = np.std(total_weights)
        
        print(f"\nStandard deviations:")
        print(f"  Licks: {std_licks:.2f}")
        print(f"  Bouts: {std_bouts:.2f}")
        print(f"  Fecal: {std_fecal:.2f}")
        print(f"  Bottle weight: {std_bottle_weight:.2f}")
        print(f"  Total weight: {std_total_weight:.2f}")
        
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
            'animal_ids': result.get('animal_ids', [])  # NEW: preserve animal IDs for repeated measures
        }
    
    return averages


def perform_anova_analysis(weekly_averages: Dict) -> Dict:
    """
    Perform one-way REPEATED MEASURES ANOVA tests for each of the 5 measures across weeks.
    
    Uses pingouin.rm_anova() to properly account for repeated measurements from the same
    animals across different weeks. Falls back to standard ANOVA with warning if pingouin
    is not available.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        
    Returns:
        Dictionary containing ANOVA results for each measure
    """
    # Check if pingouin is available for repeated measures ANOVA
    if not HAS_PINGOUIN:
        print("\n" + "="*80)
        print("WARNING: Repeated measures ANOVA requires pingouin library")
        print("Falling back to standard ANOVA (does NOT account for repeated measures)")
        print("Install pingouin with: pip install pingouin")
        print("="*80 + "\n")
        return _perform_standard_anova_fallback(weekly_averages)
    
    # Build long-format dataframe for repeated measures ANOVA
    # Each row: one animal at one week with one measurement
    long_data = []
    
    sorted_weeks = sort_dates_chronologically(list(weekly_averages.keys()))
    
    for week_idx, date in enumerate(sorted_weeks):
        data = weekly_averages[date]
        animal_ids = data.get('animal_ids', [])
        
        # If no animal IDs, create generic ones based on position
        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(len(data['avg_licks_per_animal']))]
        
        for i, animal_id in enumerate(animal_ids):
            # Skip bottle weight outlier for 11/12/25 if weight is 0 (placeholder)
            bottle_weight = data['avg_bottle_weight_per_animal'][i]
            
            long_data.append({
                'Animal': animal_id,
                'Week': week_idx + 1,  # 1-indexed week number
                'Date': date,
                'licks': data['avg_licks_per_animal'][i],
                'bouts': data['avg_bouts_per_animal'][i],
                'fecal': data['avg_fecal_per_animal'][i],
                'bottle_weight': bottle_weight if bottle_weight > 0 else np.nan,  # Exclude outliers as NaN
                'total_weight': data['avg_total_weight_per_animal'][i]
            })
    
    df_long = pd.DataFrame(long_data)
    
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
    print("PERFORMING REPEATED MEASURES ANOVA (ONE-WAY)")
    print("Within-subjects factor: Week")
    print("="*80)
    
    for measure in measures:
        print(f"\nAnalyzing: {measure_names[measure]}")
        
        # For bottle_weight, drop rows with NaN (outliers)
        if measure == 'bottle_weight':
            df_measure = df_long[['Animal', 'Week', measure]].dropna()
        else:
            df_measure = df_long[['Animal', 'Week', measure]].copy()
        
        # Check if we have enough data
        n_animals = df_measure['Animal'].nunique()
        n_weeks = df_measure['Week'].nunique()
        
        # DIAGNOSTIC: Check for animals with incomplete data across weeks
        animals_per_week = df_measure.groupby('Animal')['Week'].nunique()
        complete_animals = animals_per_week[animals_per_week == n_weeks]
        incomplete_animals = animals_per_week[animals_per_week < n_weeks]
        
        if len(incomplete_animals) > 0:
            print(f"  WARNING: {len(incomplete_animals)} animals missing data in some weeks:")
            for animal_id, n_present in incomplete_animals.items():
                weeks_present = sorted(df_measure[df_measure['Animal'] == animal_id]['Week'].unique())
                print(f"    {animal_id}: present in {n_present}/{n_weeks} weeks (weeks: {weeks_present})")
            print(f"  Complete animals (all {n_weeks} weeks): {len(complete_animals)}")
            print(f"  Filtering to only animals with complete data...")
            
            # Filter to only animals with complete data across all weeks
            df_measure = df_measure[df_measure['Animal'].isin(complete_animals.index)]
            n_animals = len(complete_animals)
            print(f"  After filtering: n_animals={n_animals}, n_weeks={n_weeks}")
        
        if n_weeks < 2:
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data for ANOVA (need at least 2 weeks)',
                'is_repeated_measures': True
            }
            print(f"  ERROR: Insufficient data ({n_weeks} weeks)")
            continue
        
        try:
            # Perform repeated measures ANOVA
            # dv = dependent variable (the measure)
            # within = within-subjects factor (Week)
            # subject = subject identifier (Animal)
            rm_result = pg.rm_anova(
                dv=measure,
                within='Week',
                subject='Animal',
                data=df_measure,
                detailed=True
            )
            
            # Extract results
            f_stat = rm_result.loc[rm_result['Source'] == 'Week', 'F'].values[0]
            p_value = rm_result.loc[rm_result['Source'] == 'Week', 'p-unc'].values[0]
            
            # Check for sphericity violation and use corrected p-value if needed
            if 'p-GG-corr' in rm_result.columns:
                p_gg = rm_result.loc[rm_result['Source'] == 'Week', 'p-GG-corr'].values[0]
                sphericity_violated = (p_gg != p_value)
            else:
                p_gg = p_value
                sphericity_violated = False
            
            # Get effect size
            if 'np2' in rm_result.columns:  # Partial eta-squared
                effect_size = rm_result.loc[rm_result['Source'] == 'Week', 'np2'].values[0]
            else:
                effect_size = np.nan
            
            # Compute descriptive statistics per week
            group_stats = []
            for week_num in sorted(df_measure['Week'].unique()):
                week_data = df_measure[df_measure['Week'] == week_num][measure]
                group_stats.append({
                    'week': sorted_weeks[week_num - 1],  # Convert back to date
                    'week_number': week_num,
                    'n': len(week_data),
                    'mean': week_data.mean(),
                    'std': week_data.std(ddof=1),
                    'min': week_data.min(),
                    'max': week_data.max()
                })
            
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': f_stat,
                'p_value': p_value,
                'p_value_gg_corrected': p_gg,
                'sphericity_violated': sphericity_violated,
                'effect_size': effect_size,
                'significant': p_gg < 0.05,  # Use GG-corrected p-value for significance
                'group_stats': group_stats,
                'n_animals': n_animals,
                'n_weeks': n_weeks,
                'is_repeated_measures': True,
                'rm_anova_table': rm_result
            }
            
            sig_marker = "***" if p_gg < 0.05 else ""
            print(f"  F({n_weeks-1}, {(n_weeks-1)*(n_animals-1)}) = {f_stat:.3f}, p = {p_gg:.4f} {sig_marker}")
            if sphericity_violated:
                print(f"  (Greenhouse-Geisser corrected p-value used due to sphericity violation)")
            print(f"  Partial η² = {effect_size:.3f}")
            print(f"  Animals: {n_animals}, Complete observations across {n_weeks} weeks")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': f'Repeated measures ANOVA failed: {str(e)}',
                'is_repeated_measures': True
            }
    
    return anova_results


def _perform_standard_anova_fallback(weekly_averages: Dict) -> Dict:
    """Fallback to standard ANOVA when pingouin is not available.
    
    WARNING: This does NOT account for repeated measures and treats observations
    as independent. Results may be invalid for repeated measures designs.
    """
    # Organize data by week (date)
    week_groups = {}
    
    for date, data in weekly_averages.items():
        if date not in week_groups:
            week_groups[date] = {
                'licks': [],
                'bouts': [],
                'fecal': [],
                'bottle_weight': [],
                'total_weight': [],
                'ca_percent': data['ca_percent']
            }
        
        # Add individual animal data for each measure
        week_groups[date]['licks'].extend(data['avg_licks_per_animal'])
        week_groups[date]['bouts'].extend(data['avg_bouts_per_animal'])
        week_groups[date]['fecal'].extend(data['avg_fecal_per_animal'])
        
        # For bottle weight, handle the 11/12/25 outlier exclusion
        if date == '11/12/25':
            # Filter out zeros that represent excluded outliers
            bottle_weights_filtered = [w for w in data['avg_bottle_weight_per_animal'] if w > 0]
            week_groups[date]['bottle_weight'].extend(bottle_weights_filtered)
        else:
            week_groups[date]['bottle_weight'].extend(data['avg_bottle_weight_per_animal'])
            
        week_groups[date]['total_weight'].extend(data['avg_total_weight_per_animal'])
    
    # Perform standard ANOVA for each measure
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
        week_labels = []
        
        # Sort weeks chronologically
        sorted_weeks = sort_dates_chronologically(list(week_groups.keys()))
        
        for week in sorted_weeks:
            if len(week_groups[week][measure]) > 0:
                groups_data.append(week_groups[week][measure])
                week_labels.append(week)
        
        if len(groups_data) >= 2:  # Need at least 2 groups for ANOVA
            # Perform standard one-way ANOVA (DOES NOT ACCOUNT FOR REPEATED MEASURES)
            f_stat, p_value = stats.f_oneway(*groups_data)
            
            # Calculate descriptive statistics for each group
            group_stats = []
            for i, week in enumerate(week_labels):
                data = groups_data[i]
                group_stats.append({
                    'week': week,
                    'ca_percent': week_groups[week]['ca_percent'],
                    'n': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),
                    'min': np.min(data),
                    'max': np.max(data)
                })
            
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group_stats': group_stats,
                'week_labels': week_labels,
                'groups_data': groups_data,
                'is_repeated_measures': False,
                'warning': 'Standard ANOVA used - does NOT account for repeated measures'
            }
        else:
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data for ANOVA (need at least 2 groups)',
                'is_repeated_measures': False
            }
    
    return anova_results


def perform_tukey_hsd(anova_results: Dict) -> Dict:
    """
    Perform Tukey's HSD post-hoc test for significant ANOVA results.
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        
    Returns:
        Dictionary containing Tukey HSD results for significant measures
    """
    tukey_results = {}
    
    for measure, anova_data in anova_results.items():
        # Only perform Tukey HSD for significant ANOVA results
        if anova_data.get('significant', False) and 'groups_data' in anova_data:
            try:
                # Prepare data for Tukey HSD
                all_data = []
                group_labels = []
                
                for i, week in enumerate(anova_data['week_labels']):
                    group_data = anova_data['groups_data'][i]
                    all_data.extend(group_data)
                    group_labels.extend([week] * len(group_data))
                
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
        return "\n" + "=" * 80 + "\nNo significant ANOVA results found - Tukey HSD not performed.\n" + "=" * 80 + "\n"
    
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TUKEY HSD POST-HOC TEST RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Tukey HSD performed only for measures with significant ANOVA results.")
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
        lines.append("weeks that differ significantly from each other.")
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
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        
    Returns:
        Formatted string with ANOVA results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    
    # Check if results are repeated measures or standard
    is_rm = any(results.get('is_repeated_measures', False) for results in anova_results.values())
    
    if is_rm:
        lines.append("ONE-WAY REPEATED MEASURES ANOVA RESULTS ACROSS WEEKS")
        lines.append("=" * 80)
        lines.append("Within-subjects factor: Week")
        lines.append("Subject tracking: Animal ID")
    else:
        lines.append("ONE-WAY ANOVA RESULTS ACROSS WEEKS")
        lines.append("=" * 80)
        lines.append("WARNING: Standard ANOVA used - does NOT account for repeated measures")
    
    lines.append("")
    
    for measure, results in anova_results.items():
        if 'error' in results:
            lines.append(f"{results['measure_name']}: {results['error']}")
            lines.append("")
            continue
        
        if 'warning' in results:
            lines.append(f"WARNING: {results['warning']}")
            lines.append("")
            
        lines.append(f"MEASURE: {results['measure_name'].upper()}")
        lines.append("-" * 50)
        
        # Show design type
        if results.get('is_repeated_measures', False):
            lines.append("Design: One-way repeated measures (within-subjects)")
            if 'n_animals' in results and 'n_weeks' in results:
                lines.append(f"Animals: {results['n_animals']}, Weeks: {results['n_weeks']}")
        else:
            lines.append("Design: Standard one-way (between-groups)")
        
        # ANOVA summary
        lines.append(f"F-statistic: {results['f_statistic']:.4f}")
        
        # Show both uncorrected and corrected p-values if available
        if 'p_value_gg_corrected' in results and results.get('sphericity_violated', False):
            lines.append(f"p-value (uncorrected): {results['p_value']:.6f}")
            lines.append(f"p-value (GG-corrected): {results['p_value_gg_corrected']:.6f}")
            lines.append("  (Greenhouse-Geisser correction applied for sphericity violation)")
            p_for_sig = results['p_value_gg_corrected']
        else:
            lines.append(f"p-value: {results.get('p_value', results.get('p_value_gg_corrected', np.nan)):.6f}")
            p_for_sig = results.get('p_value', results.get('p_value_gg_corrected', np.nan))
        
        # Effect size for repeated measures
        if 'effect_size' in results and not np.isnan(results['effect_size']):
            effect_interpretation = "large" if results['effect_size'] > 0.14 else "medium" if results['effect_size'] > 0.06 else "small"
            lines.append(f"Partial η²: {results['effect_size']:.4f} ({effect_interpretation} effect size)")
        
        if results['significant']:
            lines.append(f"Result: SIGNIFICANT (p < 0.05) - There are significant differences across weeks")
        else:
            lines.append(f"Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant differences across weeks")
        
        lines.append("")
        
        # Group descriptive statistics
        lines.append("Week Statistics:")
        
        # Check if we have CA% information (old format) or week_number (new format)
        has_ca_percent = 'group_stats' in results and len(results['group_stats']) > 0 and 'ca_percent' in results['group_stats'][0]
        
        if has_ca_percent:
            header = f"{'Week':<12} {'CA%':<6} {'N':<4} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8}"
        else:
            header = f"{'Week':<12} {'Week#':<7} {'N':<4} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8}"
        
        lines.append(header)
        lines.append("-" * 60)
        
        for group_stat in results['group_stats']:
            if has_ca_percent:
                row = (f"{group_stat['week']:<12} "
                       f"{group_stat['ca_percent']:<6} "
                       f"{group_stat['n']:<4} "
                       f"{group_stat['mean']:<10.2f} "
                       f"{group_stat['std']:<10.2f} "
                       f"{group_stat['min']:<8.2f} "
                       f"{group_stat['max']:<8.2f}")
            else:
                week_num = group_stat.get('week_number', '')
                row = (f"{group_stat['week']:<12} "
                       f"{week_num:<7} "
                       f"{group_stat['n']:<4} "
                       f"{group_stat['mean']:<10.2f} "
                       f"{group_stat['std']:<10.2f} "
                       f"{group_stat['min']:<8.2f} "
                       f"{group_stat['max']:<8.2f}")
            lines.append(row)
        
        lines.append("")
        lines.append("")
    
    # Summary of significant results
    significant_measures = [results['measure_name'] for results in anova_results.values() 
                          if results.get('significant', False)]
    
    lines.append("SUMMARY OF SIGNIFICANT RESULTS:")
    lines.append("-" * 40)
    
    if significant_measures:
        lines.append(f"Significant differences found in: {', '.join(significant_measures)}")
        lines.append("\nThese measures show statistically significant differences across weeks.")
        lines.append("Consider post-hoc tests (e.g., Tukey's HSD) for pairwise comparisons.")
    else:
        lines.append("No significant differences found in any measure across weeks.")
    
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
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Average Licks per Animal')
    ax1.set_title('Average Licks Across Weeks (±SD)', fontsize=13, weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Average Bouts
    ax2.errorbar(x_pos, avg_bouts, yerr=std_bouts,
                marker='s', markersize=8, linewidth=2, capsize=5,
                color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Average Bouts per Animal')
    ax2.set_title('Average Lick Bouts Across Weeks (±SD)', fontsize=13, weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax2.set_ylim(bottom=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig1.suptitle('Behavioral Metrics Across Weeks', fontsize=16, weight='bold', y=0.96)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig1.subplots_adjust(hspace=0.4)
    
    # ===== FIGURE 2: PHYSIOLOGICAL METRICS =====
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot 3: Average Fecal Count
    ax3.errorbar(x_pos, avg_fecal, yerr=std_fecal,
                marker='^', markersize=8, linewidth=2, capsize=5,
                color='saddlebrown', markerfacecolor='tan', markeredgecolor='saddlebrown')
    ax3.set_xlabel('Week')
    ax3.set_ylabel('Average Fecal Count per Animal')
    ax3.set_title('Average Fecal Count Across Weeks (±SD)', fontsize=13, weight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax3.set_ylim(bottom=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Average Bottle Weight Loss
    ax4.errorbar(x_pos, avg_bottle_weight, yerr=std_bottle_weight,
                marker='D', markersize=8, linewidth=2, capsize=5,
                color='purple', markerfacecolor='plum', markeredgecolor='purple')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Average Bottle Weight Loss per Animal (g)')
    ax4.set_title('Average Bottle Weight Loss Across Weeks (±SD)', fontsize=13, weight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax4.set_ylim(bottom=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Plot 5: Average Total Weight Loss
    ax5.errorbar(x_pos, avg_total_weight, yerr=std_total_weight,
                marker='v', markersize=8, linewidth=2, capsize=5,
                color='darkorange', markerfacecolor='orange', markeredgecolor='darkorange')
    ax5.set_xlabel('Week')
    ax5.set_ylabel('Average Total Weight Loss per Animal (g)')
    ax5.set_title('Average Total Weight Loss Across Weeks (±SD)', fontsize=13, weight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    fig2.suptitle('Physiological Metrics Across Weeks', fontsize=16, weight='bold', y=0.97)
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
            
            f.write(f"Week {i} ({date}):\n")
            
            # Individual animal lick counts
            lick_counts = [int(x) for x in data['avg_licks_per_animal']]
            f.write(f"  Individual licks by animal: {', '.join(map(str, lick_counts))}\n")
            
            # Average licks
            f.write(f"  Average licks per animal: {data['avg_total_licks']:.1f}\n")
            
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


def plot_comprehensive_lick_rate(
    lick_rate_data: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot comprehensive lick rate histogram collapsing all weeks.
    
    Averages lick rates across all weeks and all animals.
    
    Parameters:
        lick_rate_data: Dictionary from compute_weekly_lick_rate_averages
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Figure object
    """
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE LICK RATE PLOT (ALL WEEKS COMBINED)")
    print("=" * 80)
    
    # Collect all animal data from all weeks
    all_data = []
    total_animals = 0
    bin_size_min = None
    
    sorted_dates = sort_dates_chronologically(list(lick_rate_data.keys()))
    
    for date in sorted_dates:
        data = lick_rate_data[date]
        all_data.append(data['all_animal_data'])
        total_animals += data['n_animals']
        if bin_size_min is None:
            bin_size_min = data.get('bin_size_min', 5.0)
        print(f"  Week {date}: {data['n_animals']} animals, CA {data['ca_percent']}%")
    
    # Concatenate all animal data (all weeks, all animals)
    all_animals_all_weeks = np.vstack(all_data)
    print(f"\nTotal data: {all_animals_all_weeks.shape[0]} animals across {len(sorted_dates)} weeks")
    
    # Compute mean and SEM across all animals from all weeks
    mean_licks = np.mean(all_animals_all_weeks, axis=0)
    sem_licks = np.std(all_animals_all_weeks, axis=0, ddof=1) / np.sqrt(all_animals_all_weeks.shape[0])
    
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
    ax.set_title(f'Comprehensive Lick Rate - All Weeks Combined (n={all_animals_all_weeks.shape[0]} animals, {len(sorted_dates)} weeks)',
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
    print(f"\n{'='*60}")
    print(f"Processing {date}: {capacitive_file.name}")
    print(f"{'='*60}")
    
    # Load and process capacitive data
    df = load_capacitive_csv(capacitive_file)
    sensor_cols = get_sensor_columns(df)
    print(f"\nComputing sensor KDE baselines and raw statistics:")
    sensor_kdes = compute_sensor_KDE(df, sensor_cols, verbose=True)
    df = compute_KDE_normalizations(df, sensor_cols, sensor_kdes)
    
    # Use fixed threshold (same as lick_detection.py)
    print(f"\nUsing fixed threshold: {fixed_threshold}")
    thresholds = pd.Series({sensor: fixed_threshold for sensor in sensor_cols})
    events_df = detect_events_above_threshold(df, sensor_cols, thresholds)
    
    print(f"\nEvent detection results (fixed threshold = {fixed_threshold}):")
    for sensor in sensor_cols:
        event_col = f"{sensor}_event"
        if event_col in events_df.columns:
            num_events = events_df[event_col].sum()
            print(f"  {sensor}: threshold={thresholds.get(sensor, 'N/A'):.2f}, events detected={num_events}")
    
    # Compute bouts
    bout_dict = compute_lick_bouts(events_df, sensor_cols, ili_cutoff)
    
    # Get metadata for this date
    date_metadata = master_df[master_df['date'] == date].copy()
    
    print(f"\nMetadata loaded: {len(date_metadata)} animals found in master CSV for {date}")
    
    # Extract sensor mappings, animal IDs, and CA%
    sensor_to_weight = {}
    sensor_to_weight_loss = {}
    sensor_to_fecal = {}
    sensor_to_animal_id = {}  # NEW: Track animal IDs for repeated measures
    ca_percent = None
    
    print(f"\nSensor assignments from master CSV:")
    for idx, row in date_metadata.iterrows():
        sensor_num = int(row['selected_sensors'])
        sensor_name = f"Sensor_{sensor_num}"
        sensor_to_weight[sensor_name] = float(row['bottle_weight_change'])
        sensor_to_weight_loss[sensor_name] = float(row['total_weight_change'])
        
        # Extract animal ID if available
        if 'animal_ID' in row and pd.notna(row['animal_ID']):
            sensor_to_animal_id[sensor_name] = str(row['animal_ID'])
        else:
            # Fallback to generic ID based on sensor if animal_ID not in CSV
            sensor_to_animal_id[sensor_name] = f"Animal_{sensor_num}"
        
        print(f"  Animal {row.get('animal_ID', idx)}: {sensor_name} (bottle_wt={row['bottle_weight_change']}, total_wt={row['total_weight_change']})")
        
        # Extract CA% (should be the same for all rows in this date)
        if ca_percent is None and 'CA_%' in row and pd.notna(row['CA_%']):
            ca_percent = int(row['CA_%'])
        
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
    
    print(f"\nSelected sensors for this week: {selected_sensors}")
    print(f"CA%: {ca_percent}")
    
    # Calculate total licks and bouts for selected sensors
    total_licks = 0
    total_bouts = 0
    total_bout_licks = 0
    
    lick_counts = []
    bout_counts = []
    weights = []
    weight_losses = []
    fecal_counts = []
    animal_ids = []  # NEW: Track animal IDs in same order as measurements
    
    print(f"\nPer-sensor data (only selected sensors):")
    for sensor in selected_sensors:
        # Lick counts
        event_col = f"{sensor}_event"
        if event_col in events_df.columns:
            sensor_licks = events_df[event_col].sum()
            total_licks += sensor_licks
            lick_counts.append(sensor_licks)
            sensor_status = f"OK - {sensor_licks} licks detected"
        else:
            lick_counts.append(0)
            sensor_status = f"WARNING - Event column '{event_col}' not found in data!"
        
        # Bout counts
        if sensor in bout_dict:
            sensor_bouts = bout_dict[sensor]['bout_count']
            sensor_bout_licks = bout_dict[sensor]['bout_sizes'].sum() if len(bout_dict[sensor]['bout_sizes']) > 0 else 0
            total_bouts += sensor_bouts
            total_bout_licks += sensor_bout_licks
            bout_counts.append(sensor_bouts)
        else:
            bout_counts.append(0)
            sensor_status += " / WARNING - No bout data!"
        
        if lick_counts[-1] == 0:
            print(f"  {sensor}: *** {sensor_status} ***")
        else:
            print(f"  {sensor}: {sensor_status}")
        
        # Weights and fecal counts
        bottle_wt = sensor_to_weight.get(sensor, 0)
        total_wt = sensor_to_weight_loss.get(sensor, 0)
        fecal = sensor_to_fecal.get(sensor, 0)
        
        # Exclude R9O bottle weight outlier for 11/12/25 (R9O maps to Sensor_10)
        if date == '11/12/25' and sensor == 'Sensor_10':
            # Skip R9O bottle weight outlier (3.49) for this week
            weights.append(0)  # Use 0 as placeholder to exclude from average
        else:
            weights.append(bottle_wt)
        
        weight_losses.append(total_wt)
        fecal_counts.append(fecal)
        animal_ids.append(sensor_to_animal_id.get(sensor, f"Unknown_{sensor}"))  # NEW: Track animal ID
        
        # Additional detail line for metadata
        if lick_counts[-1] == 0:
            print(f"       Metadata: bottle_wt={bottle_wt}, total_wt={total_wt}, fecal={fecal}")
            print(f"       >>> INVESTIGATE: This sensor was selected in master CSV but detected 0 licks! <<<")
    
    # Calculate correlations
    lick_weight_corr = np.corrcoef(lick_counts, weights)[0, 1] if len(lick_counts) > 1 else np.nan
    bout_weight_corr = np.corrcoef(bout_counts, weights)[0, 1] if len(bout_counts) > 1 else np.nan
    lick_weightloss_corr = np.corrcoef(lick_counts, weight_losses)[0, 1] if len(lick_counts) > 1 else np.nan
    bout_weightloss_corr = np.corrcoef(bout_counts, weight_losses)[0, 1] if len(bout_counts) > 1 else np.nan
    weight_weightloss_corr = np.corrcoef(weights, weight_losses)[0, 1] if len(weights) > 1 else np.nan
    
    print(f"\nWeek Summary for {date}:")
    print(f"  Total licks (all sensors): {total_licks}")
    print(f"  Total bouts (all sensors): {total_bouts}")
    print(f"  Average licks per bout: {total_bout_licks / total_bouts if total_bouts > 0 else 0:.2f}")
    print(f"  Lick-weight correlation: {lick_weight_corr:.3f}")
    print(f"  Bout-weight correlation: {bout_weight_corr:.3f}")
    
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
        'animal_ids': animal_ids,  # NEW: Include animal IDs for repeated measures tracking
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
    
    tukey_results = perform_tukey_hsd(anova_results)
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
    
    # Plot comprehensive lick rate histogram (all weeks combined)
    print("\nPlotting comprehensive lick rate histogram (all weeks combined)...")
    plot_comprehensive_lick_rate(lick_rate_data, show=True)
    
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
        comprehensive_save_path = master_csv.parent / "lick_rate_comprehensive_all_weeks.svg"
        fig_comp = plot_comprehensive_lick_rate(lick_rate_data, save_path=comprehensive_save_path, show=False)
        plt.close(fig_comp)  # Close after saving
        print(f"Saved {len(sorted_dates)} individual week plots + 1 comprehensive plot")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Analyzed: Licks, Bouts, Fecal Counts, Bottle Weight Loss, Total Weight Loss")
    print(f"Statistical Analysis: One-way ANOVA across weeks and Tukey HSD post-hoc tests completed")
    print(f"All pairwise week comparisons identified for significant measures")
    print(f"Lick Rate Analysis: 5-minute bins over 30 minutes for each week + comprehensive combined plot")
    
    return weekly_results, weekly_averages, anova_results, tukey_results


if __name__ == "__main__":
    main()