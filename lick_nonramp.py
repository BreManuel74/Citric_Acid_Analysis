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


def compute_sensor_KDE(df: pd.DataFrame, sensor_cols: List[str], cache_file: Path = None, verbose: bool = False) -> pd.Series:
    """Compute the KDE (Kernel Density Estimation) peak for each sensor column.
    
    Returns a Series indexed by sensor column names with their KDE peak values.
    The KDE peak represents the most probable value in the distribution.
    
    Parameters:
        df: DataFrame containing sensor data
        sensor_cols: List of sensor column names
        cache_file: Optional path to save/load cached KDE values (speeds up subsequent runs)
        verbose: Whether to print detailed processing info
    """
    # Try to load from cache if available
    if cache_file and cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file, index_col=0)
            cached_kdes = cached_df['KDE_Peak']  # Convert to Series
            # Verify cache has all required sensors
            if all(col in cached_kdes.index for col in sensor_cols):
                if verbose:
                    print(f"  ✓ Loaded KDE values from cache: {cache_file.name}")
                return cached_kdes[sensor_cols]
            else:
                if verbose:
                    print(f"  ⚠ Cache incomplete, recomputing KDE values")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading cache ({e}), recomputing KDE values")
    
    # Compute KDE values
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
    
    result = pd.Series(kdes)
    
    # Save to cache if path provided
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(cache_file, header=['KDE_Peak'])
            if verbose:
                print(f"  ✓ Saved KDE values to cache: {cache_file.name}")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not save cache: {e}")
    
    return result


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


def compute_fixed_thresholds(sensor_cols: List[str], fixed_threshold: float = 0.01) -> pd.Series:
    """Compute fixed thresholds for event detection.
    
    Uses the EXACT lick detection algorithm from lick_detection.py.
    Fixed threshold of 0.01 for all sensors (not dynamic z-score based).
    """
    thresholds = {sensor_col: fixed_threshold for sensor_col in sensor_cols}
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


def calculate_time_to_50_percent_licks(events_df: pd.DataFrame, sensor_col: str) -> float:
    """Calculate the time (in minutes) when a sensor reaches 50% of its total licks.
    
    Parameters:
        events_df: DataFrame with Time_sec and event columns
        sensor_col: Sensor column name (e.g., 'Sensor_1')
        
    Returns:
        Time in minutes when 50% of total licks is reached, or np.nan if no licks
    """
    event_col = f"{sensor_col}_event"
    
    if event_col not in events_df.columns:
        return np.nan
    
    # Get all event times for this sensor
    event_times = events_df[events_df[event_col]]['Time_sec'].values
    
    if len(event_times) == 0:
        return np.nan
    
    # Sort times (should already be sorted, but just in case)
    event_times = np.sort(event_times)
    
    # Calculate total licks and 50% threshold
    total_licks = len(event_times)
    licks_at_50_percent = total_licks / 2.0
    
    # Find the index where cumulative licks reaches 50%
    # If total is 100, we want time at lick #50
    idx_50_percent = int(np.ceil(licks_at_50_percent)) - 1  # -1 for 0-indexing
    
    # Ensure index is valid
    if idx_50_percent >= len(event_times):
        idx_50_percent = len(event_times) - 1
    
    # Get the time at 50% in seconds, then convert to minutes
    time_at_50_percent_sec = event_times[idx_50_percent]
    time_at_50_percent_min = time_at_50_percent_sec / 60.0
    
    return time_at_50_percent_min


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
            - avg_first_5min_lick_pct: Average percentage of licks in first 5 minutes
            - avg_first_5min_bout_pct: Average percentage of bouts in first 5 minutes
            - ca_percent: Citric acid percentage for this week
            - std_licks: Standard deviation of licks across animals
            - std_bouts: Standard deviation of bouts across animals
            - std_fecal: Standard deviation of fecal counts across animals
            - std_bottle_weight: Standard deviation of bottle weight loss across animals
            - std_total_weight: Standard deviation of total weight loss across animals
            - std_first_5min_lick_pct: Standard deviation of first 5 min lick percentages
            - std_first_5min_bout_pct: Standard deviation of first 5 min bout percentages
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
        first_5min_lick_pcts = result.get('first_5min_lick_pcts', np.zeros(len(lick_counts)))  # NEW
        first_5min_bout_pcts = result.get('first_5min_bout_pcts', np.zeros(len(bout_counts)))  # NEW
        time_to_50pct_licks = result.get('time_to_50pct_licks', np.full(len(lick_counts), np.nan))  # NEW: Time to 50% (minutes)
        
        print(f"\nNumber of animals: {len(lick_counts)}")
        print(f"Individual lick counts: {lick_counts}")
        print(f"Individual bout counts: {bout_counts}")
        print(f"Individual fecal counts: {fecal_counts}")
        print(f"Individual bottle weights: {bottle_weights}")
        print(f"Individual total weights: {total_weights}")
        print(f"Individual first-5-min lick %: {first_5min_lick_pcts}")  # NEW
        print(f"Individual first-5-min bout %: {first_5min_bout_pcts}")  # NEW
        print(f"Individual time to 50% licks (min): {time_to_50pct_licks}")  # NEW
        
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
        
        # Calculate first-5-min percentage averages
        avg_first_5min_lick_pct = np.mean(first_5min_lick_pcts)
        avg_first_5min_bout_pct = np.mean(first_5min_bout_pcts)
        
        # Calculate time to 50% statistics (excluding NaN values)
        time_to_50pct_valid = time_to_50pct_licks[~np.isnan(time_to_50pct_licks)]
        avg_time_to_50pct = np.mean(time_to_50pct_valid) if len(time_to_50pct_valid) > 0 else np.nan
        std_time_to_50pct = np.std(time_to_50pct_valid) if len(time_to_50pct_valid) > 0 else np.nan
        
        # Print detailed calculation breakdown for first-5-min percentages
        print(f"\n--- First 5-Minute Average Calculation Details ---")
        print(f"Individual animal lick percentages: {first_5min_lick_pcts}")
        print(f"Sum of all percentages: {np.sum(first_5min_lick_pcts):.2f}")
        print(f"Number of animals: {len(first_5min_lick_pcts)}")
        print(f"Average = {np.sum(first_5min_lick_pcts):.2f} / {len(first_5min_lick_pcts)} = {avg_first_5min_lick_pct:.2f}%")
        print(f"\nIndividual animal bout percentages: {first_5min_bout_pcts}")
        print(f"Sum of all bout percentages: {np.sum(first_5min_bout_pcts):.2f}")
        print(f"Number of animals: {len(first_5min_bout_pcts)}")
        print(f"Average bout % = {np.sum(first_5min_bout_pcts):.2f} / {len(first_5min_bout_pcts)} = {avg_first_5min_bout_pct:.2f}%")
        print(f"--------------------------------------------------\n")
        
        std_licks = np.std(lick_counts)
        std_bouts = np.std(bout_counts)
        std_fecal = np.std(fecal_counts)
        std_total_weight = np.std(total_weights)
        std_first_5min_lick_pct = np.std(first_5min_lick_pcts)
        std_first_5min_bout_pct = np.std(first_5min_bout_pcts)
        
        n = len(lick_counts)
        n_time_to_50pct = len(time_to_50pct_valid)
        sem_licks = std_licks / np.sqrt(n) if n > 0 else 0
        sem_bouts = std_bouts / np.sqrt(n) if n > 0 else 0
        sem_fecal = std_fecal / np.sqrt(n) if n > 0 else 0
        sem_total_weight = std_total_weight / np.sqrt(n) if n > 0 else 0
        sem_first_5min_lick_pct = std_first_5min_lick_pct / np.sqrt(n) if n > 0 else 0
        sem_first_5min_bout_pct = std_first_5min_bout_pct / np.sqrt(n) if n > 0 else 0
        sem_time_to_50pct = std_time_to_50pct / np.sqrt(n_time_to_50pct) if n_time_to_50pct > 0 else np.nan
        if date == '11/12/25':
            n_bottle = len(bottle_weights_filtered)
            sem_bottle_weight = std_bottle_weight / np.sqrt(n_bottle) if n_bottle > 0 else 0
        else:
            sem_bottle_weight = std_bottle_weight / np.sqrt(n) if n > 0 else 0
        
        print(f"\nStandard deviations:")
        print(f"  Licks: {std_licks:.2f}")
        print(f"  Bouts: {std_bouts:.2f}")
        print(f"  Fecal: {std_fecal:.2f}")
        print(f"  Bottle weight: {std_bottle_weight:.2f}")
        print(f"  Total weight: {std_total_weight:.2f}")
        print(f"  First 5 min lick %: {std_first_5min_lick_pct:.2f}")
        print(f"  First 5 min bout %: {std_first_5min_bout_pct:.2f}")
        
        averages[date] = {
            'date': date,
            'ca_percent': result['ca_percent'],
            'avg_total_licks': avg_licks,
            'avg_total_bouts': avg_bouts,
            'avg_fecal_count': avg_fecal,
            'avg_bottle_weight_loss': avg_bottle_weight,
            'avg_total_weight_loss': avg_total_weight,
            'avg_first_5min_lick_pct': avg_first_5min_lick_pct,  # NEW
            'avg_first_5min_bout_pct': avg_first_5min_bout_pct,  # NEW
            'avg_time_to_50pct_licks': avg_time_to_50pct,  # NEW: Average time to reach 50% of total licks (minutes)
            'avg_licks_per_animal': lick_counts,
            'avg_bouts_per_animal': bout_counts,
            'avg_fecal_per_animal': fecal_counts,
            'avg_bottle_weight_per_animal': bottle_weights,
            'avg_total_weight_per_animal': total_weights,
            'first_5min_lick_pcts_per_animal': first_5min_lick_pcts,  # NEW
            'first_5min_bout_pcts_per_animal': first_5min_bout_pcts,  # NEW
            'time_to_50pct_licks_per_animal': time_to_50pct_licks,  # NEW: Per-animal time to 50% (minutes)
            'std_licks': std_licks,
            'std_bouts': std_bouts,
            'std_fecal': std_fecal,
            'std_bottle_weight': std_bottle_weight,
            'std_total_weight': std_total_weight,
            'std_first_5min_lick_pct': std_first_5min_lick_pct,  # NEW
            'std_first_5min_bout_pct': std_first_5min_bout_pct,  # NEW
            'std_time_to_50pct_licks': std_time_to_50pct,  # NEW: Std dev of time to 50% (minutes)
            'sem_licks': sem_licks,
            'sem_bouts': sem_bouts,
            'sem_fecal': sem_fecal,
            'sem_bottle_weight': sem_bottle_weight,
            'sem_total_weight': sem_total_weight,
            'sem_first_5min_lick_pct': sem_first_5min_lick_pct,  # NEW
            'sem_first_5min_bout_pct': sem_first_5min_bout_pct,  # NEW
            'sem_time_to_50pct_licks': sem_time_to_50pct,  # NEW: SEM of time to 50% (minutes)
            'total_animals': len(lick_counts),
            'sum_total_licks': result['total_licks'],
            'sum_total_bouts': result['total_bouts'],
            'animal_ids': result.get('animal_ids', []),  # Preserve animal IDs for repeated measures
            'animal_sexes': result.get('animal_sexes', [])  # NEW: preserve sex for mixed ANOVA
        }
    
    return averages


def perform_anova_analysis(weekly_averages: Dict) -> Dict:
    """
    Perform MIXED ANOVA tests for each of the 5 measures:
    - Between-subjects factor: Sex (each animal has one sex)
    - Within-subjects factor: Week (repeated measures across weeks)
    
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
    # Each row: one animal at one week with one measurement
    # Need to track Sex for each animal (constant across weeks)
    long_data = []
    animal_sex_map = {}  # Map animal ID to sex (should be constant across weeks)
    
    sorted_weeks = sort_dates_chronologically(list(weekly_averages.keys()))
    
    for week_idx, date in enumerate(sorted_weeks):
        data = weekly_averages[date]
        animal_ids = data.get('animal_ids', [])
        animal_sexes = data.get('animal_sexes', [])
        
        # If no animal IDs, create generic ones based on position
        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(len(data['avg_licks_per_animal']))]
        
        # If no sex data, use Unknown
        if not animal_sexes or len(animal_sexes) != len(animal_ids):
            animal_sexes = ["Unknown"] * len(animal_ids)
        
        for i, animal_id in enumerate(animal_ids):
            # Track sex for each animal (should be consistent across weeks)
            if animal_id not in animal_sex_map:
                animal_sex_map[animal_id] = animal_sexes[i]
            
            # Skip bottle weight outlier for 11/12/25 if weight is 0 (placeholder)
            bottle_weight = data['avg_bottle_weight_per_animal'][i]
            
            long_data.append({
                'Animal': animal_id,
                'Sex': animal_sex_map[animal_id],
                'Week': week_idx + 1,  # 1-indexed week number
                'Date': date,
                'licks': data['avg_licks_per_animal'][i],
                'bouts': data['avg_bouts_per_animal'][i],
                'fecal': data['avg_fecal_per_animal'][i],
                'bottle_weight': bottle_weight if bottle_weight > 0 else np.nan,  # Exclude outliers as NaN
                'total_weight': data['avg_total_weight_per_animal'][i]
            })
    
    df_long = pd.DataFrame(long_data)
    
    # Check if we have sex data
    has_sex_data = 'Sex' in df_long.columns and df_long['Sex'].nunique() > 1 and 'Unknown' not in df_long['Sex'].values
    
    if not has_sex_data:
        print("\n" + "="*80)
        print("WARNING: Sex data not found or incomplete in master CSV")
        print("Falling back to one-way repeated measures ANOVA (Week only)")
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
        print("Within-subjects factor: Week (repeated measures)")
    else:
        print("PERFORMING REPEATED MEASURES ANOVA (ONE-WAY)")
        print("Within-subjects factor: Week")
    print("="*80)
    
    for measure in measures:
        print(f"\nAnalyzing: {measure_names[measure]}")
        
        # For bottle_weight, drop rows with NaN (outliers)
        # Include Sex column if available
        if has_sex_data:
            if measure == 'bottle_weight':
                df_measure = df_long[['Animal', 'Sex', 'Week', measure]].dropna()
            else:
                df_measure = df_long[['Animal', 'Sex', 'Week', measure]].copy()
        else:
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
                'f_statistic_sex': np.nan,
                'p_value_sex': np.nan,
                'significant_sex': False,
                'f_statistic_interaction': np.nan,
                'p_value_interaction': np.nan,
                'significant_interaction': False,
                'error': 'Insufficient data for ANOVA (need at least 2 weeks)',
                'is_repeated_measures': True,
                'is_mixed_anova': has_sex_data
            }
            print(f"  ERROR: Insufficient data ({n_weeks} weeks)")
            continue
        
        try:
            if has_sex_data:
                # MIXED ANOVA with Sex (between) and Week (within)
                result_table = pg.mixed_anova(
                    dv=measure,
                    within='Week',
                    between='Sex',
                    subject='Animal',
                    data=df_measure
                )
                
                # Extract main effects and interaction
                # Sources: 'Sex', 'Week', 'Interaction'
                sex_row = result_table[result_table['Source'] == 'Sex']
                week_row = result_table[result_table['Source'] == 'Week']
                interaction_row = result_table[result_table['Source'] == 'Interaction']
                
                # Sex main effect (between-subjects)
                f_stat_sex = sex_row['F'].values[0] if len(sex_row) > 0 else np.nan
                p_value_sex = sex_row['p-unc'].values[0] if len(sex_row) > 0 else np.nan
                np2_sex = sex_row['np2'].values[0] if len(sex_row) > 0 and 'np2' in sex_row.columns else np.nan
                
                # Week main effect (within-subjects, with sphericity)
                f_stat_week = week_row['F'].values[0] if len(week_row) > 0 else np.nan
                p_value_week = week_row['p-unc'].values[0] if len(week_row) > 0 else np.nan
                np2_week = week_row['np2'].values[0] if len(week_row) > 0 and 'np2' in week_row.columns else np.nan
                
                # Check for sphericity (GG-corrected p-value for Week)
                if 'p-GG-corr' in week_row.columns and len(week_row) > 0:
                    p_gg_week = week_row['p-GG-corr'].values[0]
                    sphericity_violated_week = (p_gg_week != p_value_week)
                else:
                    p_gg_week = p_value_week
                    sphericity_violated_week = False
                
                # Interaction (within-subjects, with sphericity)
                f_stat_interaction = interaction_row['F'].values[0] if len(interaction_row) > 0 else np.nan
                p_value_interaction = interaction_row['p-unc'].values[0] if len(interaction_row) > 0 else np.nan
                np2_interaction = interaction_row['np2'].values[0] if len(interaction_row) > 0 and 'np2' in interaction_row.columns else np.nan
                
                # Check for sphericity (GG-corrected p-value for Interaction)
                if 'p-GG-corr' in interaction_row.columns and len(interaction_row) > 0:
                    p_gg_interaction = interaction_row['p-GG-corr'].values[0]
                    sphericity_violated_interaction = (p_gg_interaction != p_value_interaction)
                else:
                    p_gg_interaction = p_value_interaction
                    sphericity_violated_interaction = False
                
                # For backwards compatibility, keep 'f_statistic' and 'p_value' as the Week effect
                f_stat = f_stat_week
                p_value = p_value_week
                p_gg = p_gg_week
                sphericity_violated = sphericity_violated_week
                effect_size = np2_week
                
            else:
                # ONE-WAY REPEATED MEASURES ANOVA (Week only)
                result_table = pg.rm_anova(
                    dv=measure,
                    within='Week',
                    subject='Animal',
                    data=df_measure,
                    detailed=True
                )
                
                # Extract results
                f_stat = result_table.loc[result_table['Source'] == 'Week', 'F'].values[0]
                p_value = result_table.loc[result_table['Source'] == 'Week', 'p-unc'].values[0]
                
                # Check for sphericity violation and use corrected p-value if needed
                if 'p-GG-corr' in result_table.columns:
                    p_gg = result_table.loc[result_table['Source'] == 'Week', 'p-GG-corr'].values[0]
                    sphericity_violated = (p_gg != p_value)
                else:
                    p_gg = p_value
                    sphericity_violated = False
                
                # Get effect size
                if 'np2' in result_table.columns:  # Partial eta-squared
                    effect_size = result_table.loc[result_table['Source'] == 'Week', 'np2'].values[0]
                else:
                    effect_size = np.nan
                
                # Set sex/interaction to NaN for one-way design
                f_stat_sex = np.nan
                p_value_sex = np.nan
                np2_sex = np.nan
                f_stat_interaction = np.nan
                p_value_interaction = np.nan
                p_gg_interaction = np.nan
                np2_interaction = np.nan
                sphericity_violated_interaction = False
            
            # Compute descriptive statistics per week (and per sex if available)
            group_stats = []
            for week_num in sorted(df_measure['Week'].unique()):
                week_data = df_measure[df_measure['Week'] == week_num][measure]
                n = len(week_data)
                mean = week_data.mean()
                std = week_data.std(ddof=1)
                sem = std / np.sqrt(n) if n > 0 else np.nan
                # 95% CI: mean ± 1.96 * SEM
                ci_lower = mean - 1.96 * sem
                ci_upper = mean + 1.96 * sem
                
                group_stats.append({
                    'week': sorted_weeks[week_num - 1],  # Convert back to date
                    'week_number': week_num,
                    'n': n,
                    'mean': mean,
                    'median': week_data.median(),
                    'std': std,
                    'sem': sem,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'q25': week_data.quantile(0.25),
                    'q75': week_data.quantile(0.75),
                    'min': week_data.min(),
                    'max': week_data.max()
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
                # Week effect (main for mixed, only for RM)
                'f_statistic': f_stat,
                'p_value': p_value,
                'p_value_gg_corrected': p_gg,
                'sphericity_violated': sphericity_violated,
                'effect_size': effect_size,
                'significant': p_gg < 0.05,  # Use GG-corrected p-value for significance
                # Sex effect (only for mixed ANOVA)
                'f_statistic_sex': f_stat_sex,
                'p_value_sex': p_value_sex,
                'effect_size_sex': np2_sex,
                'significant_sex': p_value_sex < 0.05 if not np.isnan(p_value_sex) else False,
                # Interaction effect (only for mixed ANOVA)
                'f_statistic_interaction': f_stat_interaction,
                'p_value_interaction': p_value_interaction,
                'p_value_gg_corrected_interaction': p_gg_interaction,
                'sphericity_violated_interaction': sphericity_violated_interaction,
                'effect_size_interaction': np2_interaction,
                'significant_interaction': p_gg_interaction < 0.05 if not np.isnan(p_gg_interaction) else False,
                # Descriptive stats
                'group_stats': group_stats,
                'sex_stats': sex_stats,
                'n_animals': n_animals,
                'n_weeks': n_weeks,
                'is_repeated_measures': True,
                'is_mixed_anova': has_sex_data,
                'anova_table': result_table
            }
            
            # Print results
            print(f"  Results:")
            if has_sex_data:
                # Sex main effect
                sig_marker_sex = "***" if p_value_sex < 0.05 else ""
                print(f"    Sex: F = {f_stat_sex:.3f}, p = {p_value_sex:.4f} {sig_marker_sex}, partial η² = {np2_sex:.3f}")
                
                # Week main effect
                sig_marker_week = "***" if p_gg < 0.05 else ""
                print(f"    Week: F = {f_stat:.3f}, p = {p_gg:.4f} {sig_marker_week}, partial η² = {effect_size:.3f}")
                if sphericity_violated:
                    print(f"      (Greenhouse-Geisser corrected due to sphericity violation)")
                
                # Interaction
                sig_marker_int = "***" if p_gg_interaction < 0.05 else ""
                print(f"    Sex × Week: F = {f_stat_interaction:.3f}, p = {p_gg_interaction:.4f} {sig_marker_int}, partial η² = {np2_interaction:.3f}")
                if sphericity_violated_interaction:
                    print(f"      (Greenhouse-Geisser corrected due to sphericity violation)")
            else:
                # One-way RM ANOVA
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
                'f_statistic_sex': np.nan,
                'p_value_sex': np.nan,
                'significant_sex': False,
                'f_statistic_interaction': np.nan,
                'p_value_interaction': np.nan,
                'significant_interaction': False,
                'error': f'ANOVA failed: {str(e)}',
                'is_repeated_measures': True,
                'is_mixed_anova': has_sex_data
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


def perform_tukey_hsd(anova_results: Dict, weekly_averages: Dict) -> Dict:
    """
    Perform Bonferroni-corrected paired t-test post-hoc tests for significant
    Week main effects. Animal IDs are preserved across weeks, so paired
    t-tests are used to properly account for the repeated-measures structure.

    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        weekly_averages: Dictionary from compute_weekly_averages

    Returns:
        Dictionary containing post-hoc results for significant measures
    """
    import itertools
    tukey_results = {}

    for measure, anova_data in anova_results.items():
        if not anova_data.get('significant', False):
            continue

        if 'error' in anova_data:
            continue

        try:
            if 'group_stats' not in anova_data or not anova_data['group_stats']:
                continue

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

            # Build {week_label: {animal_id: value}} mapping
            sorted_weeks = sort_dates_chronologically(list(weekly_averages.keys()))
            week_maps = {}  # week_label -> {id: value}
            for week_idx, date in enumerate(sorted_weeks):
                raw = weekly_averages[date][data_key]
                ids = weekly_averages[date].get('animal_ids', [])
                if measure == 'bottle_weight':
                    id_val = {aid: v for aid, v in zip(ids, raw) if v > 0}
                else:
                    id_val = {aid: v for aid, v in zip(ids, raw)
                              if not (isinstance(v, float) and np.isnan(v))}
                if id_val:
                    week_maps[f"Week {week_idx + 1}"] = id_val

            week_labels = sorted(week_maps.keys())
            pairs = list(itertools.combinations(week_labels, 2))
            k = len(pairs)
            if k == 0:
                continue

            comparisons = []
            for g1, g2 in pairs:
                map1, map2 = week_maps[g1], week_maps[g2]
                common = sorted(set(map1.keys()) & set(map2.keys()))
                if len(common) < 2:
                    continue
                v1 = np.array([map1[aid] for aid in common], dtype=float)
                v2 = np.array([map2[aid] for aid in common], dtype=float)
                diffs = v1 - v2
                mean_diff = float(np.mean(diffs))
                se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                t_stat, p_raw = stats.ttest_rel(v1, v2)
                df_val = len(diffs) - 1
                p_adj = min(float(p_raw) * k, 1.0)
                t_crit = stats.t.ppf(0.975, df_val)
                lower_ci = mean_diff - t_crit * se_diff
                upper_ci = mean_diff + t_crit * se_diff
                comparisons.append({
                    'group1': g1,
                    'group2': g2,
                    'meandiff': mean_diff,
                    't_stat': float(t_stat),
                    'df': df_val,
                    'p_raw': float(p_raw),
                    'p_adj': p_adj,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'significant': p_adj < 0.05,
                    'n_pairs': len(common),
                })

            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'comparisons': comparisons,
            }

        except Exception as e:
            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'error': f"Error performing post-hoc tests: {str(e)}"
            }

    return tukey_results


def display_tukey_results(tukey_results: Dict) -> str:
    """
    Display Bonferroni-corrected post-hoc test results in a formatted table.

    Parameters:
        tukey_results: Dictionary from perform_tukey_hsd

    Returns:
        Formatted string with post-hoc results
    """
    if not tukey_results:
        return "\n" + "=" * 80 + "\nNo significant Week effects found - post-hoc tests not needed.\n" + "=" * 80 + "\n"

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BONFERRONI POST-HOC TEST RESULTS (PAIRED T-TESTS)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Paired t-tests with Bonferroni correction for within-subjects Week comparisons.")
    lines.append("Same animals measured each week; observations paired by animal ID.")
    lines.append("Alpha = 0.05 (Bonferroni-adjusted family-wise error rate)")
    lines.append("")

    for measure, results in tukey_results.items():
        if 'error' in results:
            lines.append(f"{results['measure_name']}: {results['error']}")
            lines.append("")
            continue

        lines.append(f"MEASURE: {results['measure_name'].upper()}")
        lines.append("-" * 60)
        header = f"{'Comparison':<20} {'Mean Diff':<12} {'T':<8} {'df':<6} {'n':<5} {'p(raw)':<10} {'p(Bonf.)':<10} {'Sig':<6}"
        lines.append(header)
        lines.append("-" * 80)

        sorted_comparisons = sorted(results['comparisons'], key=lambda x: x['p_adj'])
        for comp in sorted_comparisons:
            comparison_name = f"{comp['group1']} vs {comp['group2']}"
            sig = "*" if comp['significant'] else "ns"
            row = (f"{comparison_name:<20} "
                   f"{comp['meandiff']:<12.3f} "
                   f"{comp.get('t_stat', float('nan')):<8.3f} "
                   f"{comp.get('df', 0):<6} "
                   f"{comp.get('n_pairs', ''):<5} "
                   f"{comp.get('p_raw', float('nan')):<10.4f} "
                   f"{comp['p_adj']:<10.4f} "
                   f"{sig:<6}")
            lines.append(row)

        lines.append("")
        significant_comps = [comp for comp in results['comparisons'] if comp['significant']]
        if significant_comps:
            lines.append("Significant pairwise differences:")
            for comp in significant_comps:
                direction = "higher" if comp['meandiff'] > 0 else "lower"
                lines.append(f"  • {comp['group1']} vs {comp['group2']}: {comp['group1']} is {direction} (p(Bonf.) = {comp['p_adj']:.4f})")
        else:
            lines.append("No significant pairwise differences found (despite significant ANOVA).")
            lines.append("This may indicate the effect is distributed across groups rather than")
            lines.append("concentrated in specific pairwise comparisons.")

        lines.append("")
        lines.append("")

    total_measures = len(tukey_results)
    measures_with_sig_pairs = len([r for r in tukey_results.values()
                                   if 'comparisons' in r and
                                   any(comp['significant'] for comp in r['comparisons'])])

    lines.append("SUMMARY OF POST-HOC RESULTS:")
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
    formatted_output = "\n".join(lines)
    print(formatted_output)
    return formatted_output


def display_anova_results(anova_results: Dict) -> str:
    """
    Display ANOVA results in a formatted table and return formatted string.
    Supports both Mixed ANOVA (Sex × Week) and one-way repeated measures (Week only).
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        
    Returns:
        Formatted string with ANOVA results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    
    # Check if results are repeated measures, mixed, or standard
    is_rm = any(results.get('is_repeated_measures', False) for results in anova_results.values())
    is_mixed = any(results.get('is_mixed_anova', False) for results in anova_results.values())
    
    if is_mixed:
        lines.append("MIXED ANOVA RESULTS ACROSS WEEKS")
        lines.append("=" * 80)
        lines.append("Between-subjects factor: Sex")
        lines.append("Within-subjects factor: Week (repeated measures)")
        lines.append("Subject tracking: Animal ID")
    elif is_rm:
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
        if results.get('is_mixed_anova', False):
            lines.append("Design: Mixed ANOVA (Sex between-subjects × Week within-subjects)")
            if 'n_animals' in results and 'n_weeks' in results:
                lines.append(f"Animals: {results['n_animals']}, Weeks: {results['n_weeks']}")
            lines.append("")
            
            # Sex main effect (between-subjects)
            lines.append("SEX MAIN EFFECT (between-subjects):")
            lines.append(f"  F-statistic: {results.get('f_statistic_sex', np.nan):.4f}")
            lines.append(f"  p-value: {results.get('p_value_sex', np.nan):.6f}")
            if 'effect_size_sex' in results and not np.isnan(results.get('effect_size_sex', np.nan)):
                eta_sex = results['effect_size_sex']
                effect_interp_sex = "large" if eta_sex > 0.14 else "medium" if eta_sex > 0.06 else "small"
                lines.append(f"  Partial η²: {eta_sex:.4f} ({effect_interp_sex} effect size)")
            if results.get('significant_sex', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - Significant differences between sexes")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant sex differences")
            lines.append("")
            
            # Week main effect (within-subjects, with sphericity)
            lines.append("WEEK MAIN EFFECT (within-subjects):")
            lines.append(f"  F-statistic: {results['f_statistic']:.4f}")
            if 'p_value_gg_corrected' in results and results.get('sphericity_violated', False):
                lines.append(f"  p-value (uncorrected): {results['p_value']:.6f}")
                lines.append(f"  p-value (GG-corrected): {results['p_value_gg_corrected']:.6f}")
                lines.append("    (Greenhouse-Geisser correction applied for sphericity violation)")
                p_for_sig_week = results['p_value_gg_corrected']
            else:
                lines.append(f"  p-value: {results.get('p_value', results.get('p_value_gg_corrected', np.nan)):.6f}")
                p_for_sig_week = results.get('p_value', results.get('p_value_gg_corrected', np.nan))
            if 'effect_size' in results and not np.isnan(results['effect_size']):
                eta_week = results['effect_size']
                effect_interp_week = "large" if eta_week > 0.14 else "medium" if eta_week > 0.06 else "small"
                lines.append(f"  Partial η²: {eta_week:.4f} ({effect_interp_week} effect size)")
            if results.get('significant', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - Significant differences across weeks")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - No significant week differences")
            lines.append("")
            
            # Interaction (Sex × Week)
            lines.append("SEX × WEEK INTERACTION:")
            lines.append(f"  F-statistic: {results.get('f_statistic_interaction', np.nan):.4f}")
            if 'p_value_gg_corrected_interaction' in results and results.get('sphericity_violated_interaction', False):
                lines.append(f"  p-value (uncorrected): {results.get('p_value_interaction', np.nan):.6f}")
                lines.append(f"  p-value (GG-corrected): {results['p_value_gg_corrected_interaction']:.6f}")
                lines.append("    (Greenhouse-Geisser correction applied for sphericity violation)")
                p_for_sig_int = results['p_value_gg_corrected_interaction']
            else:
                lines.append(f"  p-value: {results.get('p_value_interaction', np.nan):.6f}")
                p_for_sig_int = results.get('p_value_interaction', np.nan)
            if 'effect_size_interaction' in results and not np.isnan(results.get('effect_size_interaction', np.nan)):
                eta_int = results['effect_size_interaction']
                effect_interp_int = "large" if eta_int > 0.14 else "medium" if eta_int > 0.06 else "small"
                lines.append(f"  Partial η²: {eta_int:.4f} ({effect_interp_int} effect size)")
            if results.get('significant_interaction', False):
                lines.append(f"  Result: SIGNIFICANT (p < 0.05) - Week effect differs by sex")
            else:
                lines.append(f"  Result: NOT SIGNIFICANT (p ≥ 0.05) - Week effect similar across sexes")
            lines.append("")
            
        elif results.get('is_repeated_measures', False):
            lines.append("Design: One-way repeated measures (within-subjects)")
            if 'n_animals' in results and 'n_weeks' in results:
                lines.append(f"Animals: {results['n_animals']}, Weeks: {results['n_weeks']}")
            lines.append("")
            
            # ANOVA summary (one-way RM)
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
        else:
            lines.append("Design: Standard one-way (between-groups)")
            lines.append("")
            
            # ANOVA summary (standard)
            lines.append(f"F-statistic: {results['f_statistic']:.4f}")
            lines.append(f"p-value: {results.get('p_value', np.nan):.6f}")
            
            if results['significant']:
                lines.append(f"Result: SIGNIFICANT (p < 0.05)")
            else:
                lines.append(f"Result: NOT SIGNIFICANT (p ≥ 0.05)")
            lines.append("")
        
        # Group descriptive statistics
        lines.append("Week Statistics:")
        
        # Check if we have CA% information (old format) or week_number (new format)
        has_ca_percent = 'group_stats' in results and len(results['group_stats']) > 0 and 'ca_percent' in results['group_stats'][0]
        
        if has_ca_percent:
            header = f"{'Week':<12} {'CA%':<6} {'N':<4} {'Mean':<10} {'Median':<10} {'Std':<10} {'SEM':<10}"
        else:
            header = f"{'Week':<12} {'Week#':<7} {'N':<4} {'Mean':<10} {'Median':<10} {'Std':<10} {'SEM':<10}"
        
        lines.append(header)
        lines.append("-" * 72)
        
        for group_stat in results['group_stats']:
            if has_ca_percent:
                row = (f"{group_stat['week']:<12} "
                       f"{group_stat['ca_percent']:<6} "
                       f"{group_stat['n']:<4} "
                       f"{group_stat['mean']:<10.2f} "
                       f"{group_stat.get('median', np.nan):<10.2f} "
                       f"{group_stat['std']:<10.2f} "
                       f"{group_stat.get('sem', np.nan):<10.2f}")
            else:
                week_num = group_stat.get('week_number', '')
                row = (f"{group_stat['week']:<12} "
                       f"{week_num:<7} "
                       f"{group_stat['n']:<4} "
                       f"{group_stat['mean']:<10.2f} "
                       f"{group_stat.get('median', np.nan):<10.2f} "
                       f"{group_stat['std']:<10.2f} "
                       f"{group_stat.get('sem', np.nan):<10.2f}")
            lines.append(row)
        
        lines.append("")
        lines.append("95% Confidence Intervals and Range:")
        if has_ca_percent:
            header2 = f"{'Week':<12} {'CA%':<6} {'95% CI Lower':<14} {'95% CI Upper':<14} {'Q25':<10} {'Q75':<10}"
        else:
            header2 = f"{'Week':<12} {'Week#':<7} {'95% CI Lower':<14} {'95% CI Upper':<14} {'Q25':<10} {'Q75':<10}"
        lines.append(header2)
        lines.append("-" * 80)
        
        for group_stat in results['group_stats']:
            if has_ca_percent:
                row2 = (f"{group_stat['week']:<12} "
                        f"{group_stat['ca_percent']:<6} "
                        f"{group_stat.get('ci_lower', np.nan):<14.2f} "
                        f"{group_stat.get('ci_upper', np.nan):<14.2f} "
                        f"{group_stat.get('q25', np.nan):<10.2f} "
                        f"{group_stat.get('q75', np.nan):<10.2f}")
            else:
                week_num = group_stat.get('week_number', '')
                row2 = (f"{group_stat['week']:<12} "
                        f"{week_num:<7} "
                        f"{group_stat.get('ci_lower', np.nan):<14.2f} "
                        f"{group_stat.get('ci_upper', np.nan):<14.2f} "
                        f"{group_stat.get('q25', np.nan):<10.2f} "
                        f"{group_stat.get('q75', np.nan):<10.2f}")
            lines.append(row2)
        
        lines.append("")
        
        # Sex-specific descriptive statistics (for mixed ANOVA)
        if results.get('is_mixed_anova', False) and 'sex_stats' in results and results['sex_stats']:
            lines.append("Sex Statistics:")
            header = f"{'Sex':<6} {'N':<6} {'Mean':<10} {'Std':<10}"
            lines.append(header)
            lines.append("-" * 35)
            
            for sex_stat in results['sex_stats']:
                row = (f"{sex_stat['sex']:<6} "
                       f"{sex_stat['n']:<6} "
                       f"{sex_stat['mean']:<10.2f} "
                       f"{sex_stat['std']:<10.2f}")
                lines.append(row)
            
            lines.append("")
        
        lines.append("")
    
    # Summary of significant results
    significant_measures_week = [results['measure_name'] for results in anova_results.values() 
                          if results.get('significant', False)]
    significant_measures_sex = [results['measure_name'] for results in anova_results.values() 
                          if results.get('significant_sex', False)]
    significant_measures_interaction = [results['measure_name'] for results in anova_results.values() 
                          if results.get('significant_interaction', False)]
    
    is_mixed = any(results.get('is_mixed_anova', False) for results in anova_results.values())
    
    lines.append("SUMMARY OF SIGNIFICANT RESULTS:")
    lines.append("-" * 40)
    
    if is_mixed:
        if significant_measures_sex:
            lines.append(f"Sex main effect significant in: {', '.join(significant_measures_sex)}")
        else:
            lines.append("Sex main effect: No significant differences")
        
        if significant_measures_week:
            lines.append(f"Week main effect significant in: {', '.join(significant_measures_week)}")
        else:
            lines.append("Week main effect: No significant differences")
        
        if significant_measures_interaction:
            lines.append(f"Sex × Week interaction significant in: {', '.join(significant_measures_interaction)}")
            lines.append("  (Week effect differs by sex - consider simple effects analysis)")
        else:
            lines.append("Sex × Week interaction: Not significant (week effect similar across sexes)")
    else:
        if significant_measures_week:
            lines.append(f"Significant differences found in: {', '.join(significant_measures_week)}")
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


def perform_mixed_anova_posthoc(data: pd.DataFrame, dv: str, within: str, between: str, subject: str, 
                                 alpha: float = 0.05, correction: str = 'fdr_bh') -> Dict:
    """
    Perform proper post-hoc tests for mixed ANOVA with repeated measures.
    
    This function performs comprehensive post-hoc analyses including:
    1. Within-subjects pairwise comparisons (across levels of within factor)
    2. Between-subjects pairwise comparisons (between levels of between factor)
    3. Simple effects analysis (within effect at each level of between factor)
    
    Parameters:
        data: DataFrame with long-format data
        dv: Dependent variable column name
        within: Within-subjects factor column (e.g., 'Week' or 'CA%')
        between: Between-subjects factor column (e.g., 'Sex')
        subject: Subject identifier column (e.g., 'Animal_ID')
        alpha: Significance level (default 0.05)
        correction: Multiple comparison correction method ('fdr_bh', 'bonferroni', 'holm')
    
    Returns:
        Dictionary with post-hoc test results
    """
    import pingouin as pg
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    results = {
        'within_pairwise': None,
        'between_pairwise': None,
        'simple_effects': None,
        'correction_method': correction,
        'alpha': alpha,
        'within_factor': within,
        'between_factor': between
    }
    
    # 1. Within-subjects pairwise comparisons (e.g., Week or CA% pairs, collapsed across Sex)
    try:
        within_pw = pg.pairwise_tests(
            data=data,
            dv=dv,
            within=within,
            subject=subject,
            parametric=True,
            padjust=correction,
            effsize='hedges'
        )
        results['within_pairwise'] = within_pw
    except Exception as e:
        results['within_pairwise'] = f"Error in within-subjects pairwise: {str(e)}"
    
    # 2. Between-subjects pairwise comparisons (Sex pairs, collapsed across within factor)
    try:
        # Average across within-factor levels for each subject first
        between_data = data.groupby([subject, between])[dv].mean().reset_index()
        
        between_levels = between_data[between].unique()
        if len(between_levels) == 2:
            # For 2 groups, do independent t-test
            group1 = between_data[between_data[between] == between_levels[0]][dv]
            group2 = between_data[between_data[between] == between_levels[1]][dv]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1) - 1) * group1.std()**2 + 
                                   (len(group2) - 1) * group2.std()**2) / 
                                  (len(group1) + len(group2) - 2))
            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
            
            between_pw = pd.DataFrame({
                'Contrast': [f"{between_levels[0]} vs {between_levels[1]}"],
                'A': [between_levels[0]],
                'B': [between_levels[1]],
                'T': [t_stat],
                'p-unc': [p_val],
                'p-corr': [p_val],  # Only one comparison
                "cohen's d": [cohens_d],
                'n1': [len(group1)],
                'n2': [len(group2)]
            })
            results['between_pairwise'] = between_pw
        else:
            # For >2 groups, use pairwise_tests
            between_pw = pg.pairwise_tests(
                data=between_data,
                dv=dv,
                between=between,
                parametric=True,
                padjust=correction,
                effsize='cohen'
            )
            results['between_pairwise'] = between_pw
    except Exception as e:
        results['between_pairwise'] = f"Error in between-subjects pairwise: {str(e)}"
    
    # 3. Simple effects: within-factor effect at each level of between factor
    try:
        between_levels = data[between].unique()
        simple_effects = []
        
        for between_level in between_levels:
            level_data = data[data[between] == between_level]
            
            # Perform repeated measures ANOVA for this level
            within_levels = level_data[within].unique()
            if len(within_levels) >= 2:
                # Use pingouin's repeated measures ANOVA
                rm_aov = pg.rm_anova(
                    data=level_data,
                    dv=dv,
                    within=within,
                    subject=subject,
                    detailed=True
                )
                
                simple_effects.append({
                    between: between_level,
                    'F': rm_aov.loc[0, 'F'],
                    'df1': rm_aov.loc[0, 'ddof1'],
                    'df2': rm_aov.loc[0, 'ddof2'],
                    'p-unc': rm_aov.loc[0, 'p-unc'],
                    'n_subjects': level_data[subject].nunique(),
                    f'n_{within}_levels': len(within_levels)
                })
        
        if simple_effects:
            simple_effects_df = pd.DataFrame(simple_effects)
            
            # Apply multiple comparison correction across simple effects
            if len(simple_effects_df) > 1:
                _, p_corrected, _, _ = multipletests(
                    simple_effects_df['p-unc'],
                    alpha=alpha,
                    method=correction
                )
                simple_effects_df['p-corr'] = p_corrected
            else:
                simple_effects_df['p-corr'] = simple_effects_df['p-unc']
            
            simple_effects_df['significant'] = simple_effects_df['p-corr'] < alpha
            results['simple_effects'] = simple_effects_df
        else:
            results['simple_effects'] = "Insufficient data for simple effects analysis"
            
    except Exception as e:
        results['simple_effects'] = f"Error in simple effects: {str(e)}"
    
    return results


def display_posthoc_results(posthoc_results: Dict, measure_name: str) -> str:
    """
    Display post-hoc test results in formatted output.
    
    Parameters:
        posthoc_results: Dictionary from perform_mixed_anova_posthoc
        measure_name: Name of the measure being analyzed
        
    Returns:
        Formatted string with post-hoc results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"POST-HOC TESTS FOR: {measure_name.upper()}")
    lines.append("=" * 80)
    lines.append(f"Correction method: {posthoc_results['correction_method']}")
    lines.append(f"Alpha level: {posthoc_results['alpha']}")
    lines.append("")
    
    within_name = posthoc_results.get('within_factor', 'Within factor')
    between_name = posthoc_results.get('between_factor', 'Between factor')
    
    # 1. Within-subjects pairwise
    lines.append(f"1. WITHIN-SUBJECTS PAIRWISE COMPARISONS ({within_name})")
    lines.append("-" * 80)
    within_pw = posthoc_results['within_pairwise']
    if isinstance(within_pw, str):
        lines.append(f"   {within_pw}")
    elif within_pw is not None and not within_pw.empty:
        lines.append(f"   Pairwise comparisons across {within_name} levels (repeated measures):")
        lines.append("")
        for idx, row in within_pw.iterrows():
            contrast = f"   {row['A']} vs {row['B']}"
            lines.append(f"{contrast}")
            lines.append(f"      T-statistic: {row['T']:.4f}")
            lines.append(f"      p-uncorrected: {row['p-unc']:.6f}")
            lines.append(f"      p-corrected: {row.get('p-corr', row['p-unc']):.6f}")
            lines.append(f"      Effect size (Hedges' g): {row.get('hedges', 0):.4f}")
            if row.get('p-corr', row['p-unc']) < posthoc_results['alpha']:
                lines.append(f"      *** SIGNIFICANT ***")
            lines.append("")
    else:
        lines.append("   No within-subjects comparisons available")
    
    lines.append("")
    
    # 2. Between-subjects pairwise
    lines.append(f"2. BETWEEN-SUBJECTS PAIRWISE COMPARISONS ({between_name})")
    lines.append("-" * 80)
    between_pw = posthoc_results['between_pairwise']
    if isinstance(between_pw, str):
        lines.append(f"   {between_pw}")
    elif between_pw is not None and not between_pw.empty:
        lines.append(f"   Comparison between {between_name} levels (collapsed across {within_name}):")
        lines.append("")
        for idx, row in between_pw.iterrows():
            contrast = f"   {row['A']} vs {row['B']}"
            lines.append(f"{contrast}")
            lines.append(f"      T-statistic: {row['T']:.4f}")
            lines.append(f"      p-uncorrected: {row['p-unc']:.6f}")
            lines.append(f"      p-corrected: {row.get('p-corr', row['p-unc']):.6f}")
            cohens_d = row.get("cohen's d", row.get('cohen', 0))
            lines.append(f"      Effect size (Cohen's d): {cohens_d:.4f}")
            lines.append(f"      n1: {row.get('n1', 'N/A')}, n2: {row.get('n2', 'N/A')}")
            if row.get('p-corr', row['p-unc']) < posthoc_results['alpha']:
                lines.append(f"      *** SIGNIFICANT ***")
            lines.append("")
    else:
        lines.append("   No between-subjects comparisons available")
    
    lines.append("")
    
    # 3. Simple effects
    lines.append("3. SIMPLE EFFECTS ANALYSIS")
    lines.append("-" * 80)
    simple_eff = posthoc_results['simple_effects']
    if isinstance(simple_eff, str):
        lines.append(f"   {simple_eff}")
    elif simple_eff is not None and not simple_eff.empty:
        lines.append(f"   {within_name} effect separately for each {between_name} level:")
        lines.append("")
        for idx, row in simple_eff.iterrows():
            lines.append(f"   {between_name.upper()}: {row[between_name]}")
            lines.append(f"      F({row['df1']:.0f}, {row['df2']:.0f}) = {row['F']:.4f}")
            lines.append(f"      p-uncorrected: {row['p-unc']:.6f}")
            lines.append(f"      p-corrected: {row['p-corr']:.6f}")
            lines.append(f"      n subjects: {row['n_subjects']}")
            if row['significant']:
                lines.append(f"      *** SIGNIFICANT {within_name} effect for {row[between_name]} ***")
            else:
                lines.append(f"      Not significant")
            lines.append("")
    else:
        lines.append("   No simple effects analysis available")
    
    lines.append("=" * 80)
    lines.append("")
    
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
        all_avg_first_5min_lick = [avg['avg_first_5min_lick_pct'] for avg in weekly_averages.values()]
        all_avg_first_5min_bout = [avg['avg_first_5min_bout_pct'] for avg in weekly_averages.values()]
        
        lines.append("")
        lines.append("CROSS-WEEK STATISTICS:")
        lines.append(f"Licks - Mean: {np.mean(all_avg_licks):.1f}, "
                    f"Std: {np.std(all_avg_licks):.1f}, "
                    f"Range: {np.min(all_avg_licks):.1f} - {np.max(all_avg_licks):.1f}")
        lines.append(f"Bouts - Mean: {np.mean(all_avg_bouts):.1f}, "
                    f"Std: {np.std(all_avg_bouts):.1f}, "
                    f"Range: {np.min(all_avg_bouts):.1f} - {np.max(all_avg_bouts):.1f}")
        lines.append(f"First 5-min Lick % - Mean: {np.mean(all_avg_first_5min_lick):.1f}%, "
                    f"Std: {np.std(all_avg_first_5min_lick):.1f}%, "
                    f"Range: {np.min(all_avg_first_5min_lick):.1f}% - {np.max(all_avg_first_5min_lick):.1f}%")
        lines.append(f"First 5-min Bout % - Mean: {np.mean(all_avg_first_5min_bout):.1f}%, "
                    f"Std: {np.std(all_avg_first_5min_bout):.1f}%, "
                    f"Range: {np.min(all_avg_first_5min_bout):.1f}% - {np.max(all_avg_first_5min_bout):.1f}%")
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
        std_licks.append(data['sem_licks'])
        avg_bouts.append(data['avg_total_bouts'])
        std_bouts.append(data['sem_bouts'])
        avg_fecal.append(data['avg_fecal_count'])
        std_fecal.append(data['sem_fecal'])
        avg_bottle_weight.append(data['avg_bottle_weight_loss'])
        std_bottle_weight.append(data['sem_bottle_weight'])
        avg_total_weight.append(data['avg_total_weight_loss'])
        std_total_weight.append(data['sem_total_weight'])
    
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
    ax1.set_title('Average Licks Across Weeks (±SEM)', fontsize=13, weight='bold')
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
    ax2.set_title('Average Lick Bouts Across Weeks (±SEM)', fontsize=13, weight='bold')
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
    ax3.set_title('Average Fecal Count Across Weeks (±SEM)', fontsize=13, weight='bold')
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
    ax4.set_title('Average Bottle Weight Loss Across Weeks (±SEM)', fontsize=13, weight='bold')
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
    ax5.set_title('Average Total Weight Loss Across Weeks (±SEM)', fontsize=13, weight='bold')
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


def perform_frontloading_anova(weekly_averages: Dict) -> Dict:
    """
    Perform ONE-WAY REPEATED MEASURES ANOVA for front-loading measures:
    - % of licks in first 5 minutes
    - % of bouts in first 5 minutes
    - Time to 50% of total licks (minutes)
    
    Within-subjects factor: Week (repeated measures across weeks)
    
    Uses pingouin.rm_anova() for repeated measures analysis.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        
    Returns:
        Dictionary containing repeated measures ANOVA results for front-loading measures
    """
    # Check if pingouin is available
    if not HAS_PINGOUIN:
        print("\n" + "="*80)
        print("WARNING: Repeated measures ANOVA requires pingouin library")
        print("Front-loading ANOVA cannot be performed without pingouin")
        print("Install pingouin with: pip install pingouin")
        print("="*80 + "\n")
        return {}
    
    # Build long-format dataframe for repeated measures ANOVA
    long_data = []
    
    # Sort by date chronologically
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    
    print("\n" + "="*80)
    print("FRONT-LOADING ANOVA: Building Data")
    print("="*80)
    
    for week_idx, date in enumerate(sorted_dates, 1):
        data = weekly_averages[date]
        animal_ids = data.get('animal_ids', [])
        
        # If no animal IDs, create generic ones
        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(len(data.get('first_5min_lick_pcts_per_animal', [])))]
        
        first_5min_lick_pcts = data.get('first_5min_lick_pcts_per_animal', [])
        first_5min_bout_pcts = data.get('first_5min_bout_pcts_per_animal', [])
        time_to_50pct = data.get('time_to_50pct_licks_per_animal', [])
        
        for i, animal_id in enumerate(animal_ids):
            # Get front-loading metrics
            first_5min_lick_val = first_5min_lick_pcts[i] if i < len(first_5min_lick_pcts) else np.nan
            first_5min_bout_val = first_5min_bout_pcts[i] if i < len(first_5min_bout_pcts) else np.nan
            time_50pct_val = time_to_50pct[i] if i < len(time_to_50pct) else np.nan
            
            long_data.append({
                'Animal': animal_id,
                'Week': week_idx,
                'Date': date,
                'first_5min_lick_pct': first_5min_lick_val,
                'first_5min_bout_pct': first_5min_bout_val,
                'time_to_50pct': time_50pct_val
            })
    
    print(f"Total data points: {len(long_data)}")
    print(f"Unique animals: {len(set(row['Animal'] for row in long_data))}")
    print("="*80 + "\n")
    
    df_long = pd.DataFrame(long_data)
    
    # Perform repeated measures ANOVA for each front-loading measure
    anova_results = {}
    measures = ['first_5min_lick_pct', 'first_5min_bout_pct', 'time_to_50pct']
    measure_names = {
        'first_5min_lick_pct': '% Licks in First 5 Minutes',
        'first_5min_bout_pct': '% Bouts in First 5 Minutes',
        'time_to_50pct': 'Time to 50% of Total Licks (min)'
    }
    
    print("\n" + "="*80)
    print("PERFORMING ONE-WAY REPEATED MEASURES ANOVA: Week Effect for Front-Loading Measures")
    print("="*80)
    
    for measure in measures:
        measure_name = measure_names[measure]
        print(f"\nAnalyzing: {measure_name}")
        print("-" * 40)
        
        # Remove NaN values for this measure
        df_measure = df_long[['Animal', 'Week', measure]].dropna()
        
        if len(df_measure) == 0:
            print(f"  ERROR: No valid data for {measure_name}")
            anova_results[measure] = {
                'measure_name': measure_name,
                'error': 'No valid data available'
            }
            continue
        
        print(f"  Valid data points: {len(df_measure)}")
        print(f"  Animals: {df_measure['Animal'].nunique()}")
        print(f"  Week levels: {sorted(df_measure['Week'].unique())}")
        
        try:
            # Repeated measures ANOVA: Week only
            aov = pg.rm_anova(
                data=df_measure,
                dv=measure,
                within='Week',
                subject='Animal',
                detailed=True
            )
            
            print("\n  Repeated Measures ANOVA Results:")
            print(aov)
            
            week_effect = aov[aov['Source'] == 'Week'].iloc[0]
            
            # Extract degrees of freedom
            df1 = week_effect['ddof1'] if 'ddof1' in week_effect.index else (week_effect['DF1'] if 'DF1' in week_effect.index else None)
            df2 = week_effect['ddof2'] if 'ddof2' in week_effect.index else (week_effect['DF2'] if 'DF2' in week_effect.index else None)
            
            # Extract sphericity information
            sphericity_met = week_effect['sphericity'] if 'sphericity' in week_effect.index else None
            w_sphericity = week_effect['W-spher'] if 'W-spher' in week_effect.index else None
            p_sphericity = week_effect['p-spher'] if 'p-spher' in week_effect.index else None
            p_gg_corrected = week_effect['p-GG-corr'] if 'p-GG-corr' in week_effect.index else None
            epsilon_gg = week_effect['eps-GG'] if 'eps-GG' in week_effect.index else None
            
            anova_results[measure] = {
                'measure_name': measure_name,
                'is_repeated_measures': True,
                'anova_table': aov,
                'f_statistic': week_effect['F'],
                'p_value': week_effect['p-unc'],
                'significant': week_effect['p-unc'] < 0.05,
                'df1': df1,
                'df2': df2,
                # Sphericity test information
                'sphericity_met': sphericity_met,
                'w_sphericity': w_sphericity,
                'p_sphericity': p_sphericity,
                'p_gg_corrected': p_gg_corrected,
                'epsilon_gg': epsilon_gg
            }
        
        except Exception as e:
            print(f"  ERROR performing ANOVA: {e}")
            anova_results[measure] = {
                'measure_name': measure_name,
                'error': str(e)
            }
    
    return anova_results


def display_frontloading_anova_results(anova_results: Dict) -> str:
    """Display front-loading ANOVA results in formatted output.
    
    Parameters:
        anova_results: Dictionary from perform_frontloading_anova
        
    Returns:
        Formatted string with ANOVA results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("FRONT-LOADING ANALYSIS: ONE-WAY REPEATED MEASURES ANOVA RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    if not anova_results:
        lines.append("No ANOVA results available (pingouin library may not be installed)")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    lines.append("Repeated Measures ANOVA: Week (within-subjects)")
    lines.append("")
    
    for measure, results in anova_results.items():
        lines.append(f"MEASURE: {results['measure_name']}")
        lines.append("-" * 80)
        
        if 'error' in results:
            lines.append(f"ERROR: {results['error']}")
            lines.append("")
            continue
        
        lines.append("")
        lines.append("Week Effect:")
        if results.get('df1') is not None and results.get('df2') is not None:
            lines.append(f"  F({results['df1']:.0f}, {results['df2']:.0f}) = {results['f_statistic']:.4f}")
        else:
            lines.append(f"  F-statistic: {results['f_statistic']:.4f}")
        lines.append(f"  p-value: {results['p_value']:.6f}")
        if results['significant']:
            lines.append(f"  *** SIGNIFICANT WEEK EFFECT (p < 0.05) ***")
        else:
            lines.append(f"  Not significant (p >= 0.05)")
        
        # Sphericity test information
        if results.get('sphericity_met') is not None:
            lines.append("")
            lines.append("Sphericity Test (Mauchly's W):")
            w_val = results.get('w_sphericity') if results.get('w_sphericity') is not None else np.nan
            p_val = results.get('p_sphericity') if results.get('p_sphericity') is not None else np.nan
            lines.append(f"  W = {w_val:.4f}, p = {p_val:.4f}")
            if results.get('sphericity_met'):
                lines.append(f"  Sphericity assumption MET (p >= 0.05)")
            else:
                lines.append(f"  Sphericity assumption VIOLATED (p < 0.05)")
                eps_gg = results.get('epsilon_gg') if results.get('epsilon_gg') is not None else np.nan
                p_gg = results.get('p_gg_corrected') if results.get('p_gg_corrected') is not None else np.nan
                lines.append(f"  Greenhouse-Geisser epsilon: {eps_gg:.4f}")
                lines.append(f"  GG-corrected p-value: {p_gg:.6f}")
                if results.get('p_gg_corrected') is not None and results.get('p_gg_corrected') < 0.05:
                    lines.append(f"  Result remains SIGNIFICANT after GG correction")
                elif results.get('p_gg_corrected') is not None:
                    lines.append(f"  Result becomes NOT SIGNIFICANT after GG correction")
        
        lines.append("")
    
    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY:")
    significant_week = [r['measure_name'] for r in anova_results.values() if r.get('significant', False)]
    lines.append(f"Significant Week effects: {', '.join(significant_week) if significant_week else 'None'}")
    
    # Note about sphericity violations
    sphericity_violated = [r['measure_name'] for r in anova_results.values()
                           if r.get('sphericity_met') is False]
    if sphericity_violated:
        lines.append(f"\nNote: Sphericity assumption violated for: {', '.join(sphericity_violated)}")
        lines.append("Greenhouse-Geisser corrected p-values are provided above.")
    
    lines.append("=" * 80)
    lines.append("")
    
    formatted_output = "\n".join(lines)
    print(formatted_output)
    
    return formatted_output


def perform_frontloading_tukey_hsd(anova_results: Dict, weekly_averages: Dict) -> Dict:
    """
    Perform Bonferroni-corrected paired t-test post-hoc tests for significant
    front-loading measures. Animal IDs are preserved across weeks, so paired
    t-tests are used to properly account for the repeated-measures structure.

    Parameters:
        anova_results: Dictionary from perform_frontloading_anova
        weekly_averages: Dictionary from compute_weekly_averages

    Returns:
        Dictionary containing post-hoc results for significant measures
    """
    import itertools
    tukey_results = {}

    for measure, anova_data in anova_results.items():
        if not anova_data.get('significant', False):
            continue

        if 'error' in anova_data:
            continue

        try:
            measure_to_key = {
                'first_5min_lick_pct': 'first_5min_lick_pcts_per_animal',
                'first_5min_bout_pct': 'first_5min_bout_pcts_per_animal',
                'time_to_50pct': 'time_to_50pct_licks_per_animal'
            }

            data_key = measure_to_key.get(measure)
            if not data_key:
                continue

            # Build {week_label: {animal_id: value}} mapping
            sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
            week_maps = {}
            for week_idx, date in enumerate(sorted_dates, 1):
                raw = weekly_averages[date].get(data_key, [])
                ids = weekly_averages[date].get('animal_ids', [])
                id_val = {aid: v for aid, v in zip(ids, raw)
                          if not (isinstance(v, float) and np.isnan(v))}
                if id_val:
                    week_maps[f"Week {week_idx}"] = id_val

            week_labels = sorted(week_maps.keys())
            pairs = list(itertools.combinations(week_labels, 2))
            k = len(pairs)
            if k == 0:
                continue

            comparisons = []
            for g1, g2 in pairs:
                map1, map2 = week_maps[g1], week_maps[g2]
                common = sorted(set(map1.keys()) & set(map2.keys()))
                if len(common) < 2:
                    continue
                v1 = np.array([map1[aid] for aid in common], dtype=float)
                v2 = np.array([map2[aid] for aid in common], dtype=float)
                diffs = v1 - v2
                mean_diff = float(np.mean(diffs))
                se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
                t_stat, p_raw = stats.ttest_rel(v1, v2)
                df_val = len(diffs) - 1
                p_adj = min(float(p_raw) * k, 1.0)
                t_crit = stats.t.ppf(0.975, df_val)
                lower_ci = mean_diff - t_crit * se_diff
                upper_ci = mean_diff + t_crit * se_diff
                comparisons.append({
                    'group1': g1,
                    'group2': g2,
                    'meandiff': mean_diff,
                    't_stat': float(t_stat),
                    'df': df_val,
                    'p_raw': float(p_raw),
                    'p_adj': p_adj,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'significant': p_adj < 0.05,
                    'n_pairs': len(common),
                })

            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'comparisons': comparisons,
            }

            print(f"\nBonferroni post-hoc completed for: {anova_data['measure_name']}")
            print(f"  Total comparisons: {len(comparisons)}")
            print(f"  Significant pairs: {sum(1 for c in comparisons if c['significant'])}")

        except Exception as e:
            tukey_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'error': f"Error performing post-hoc tests: {str(e)}"
            }

    return tukey_results


def display_frontloading_tukey_results(tukey_results: Dict) -> str:
    """
    Display Bonferroni-corrected post-hoc test results for front-loading measures.

    Parameters:
        tukey_results: Dictionary from perform_frontloading_tukey_hsd

    Returns:
        Formatted string with post-hoc results
    """
    if not tukey_results:
        return "\nNo post-hoc results to display (no significant Week effects)\n"

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BONFERRONI POST-HOC TEST RESULTS - FRONT-LOADING MEASURES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Paired t-tests with Bonferroni correction for within-subjects Week comparisons.")
    lines.append("Same animals measured each week; observations paired by animal ID.")
    lines.append("Alpha = 0.05 (Bonferroni-adjusted family-wise error rate)")
    lines.append("")

    for measure, results in tukey_results.items():
        lines.append(f"MEASURE: {results['measure_name'].upper()}")
        lines.append("-" * 80)

        if 'error' in results:
            lines.append(f"ERROR: {results['error']}")
            lines.append("")
            continue

        if 'comparisons' not in results or not results['comparisons']:
            lines.append("No pairwise comparisons available")
            lines.append("")
            continue

        lines.append(f"{'Comparison':<20} {'Mean Diff':<12} {'T':<8} {'df':<6} {'n':<5} {'p(raw)':<10} {'p(Bonf.)':<10} {'Sig':<6}")
        lines.append("-" * 80)

        for comp in results['comparisons']:
            comparison = f"{comp['group1']} vs {comp['group2']}"
            sig = "*" if comp['significant'] else "ns"
            lines.append(f"{comparison:<20} "
                         f"{comp['meandiff']:<12.3f} "
                         f"{comp.get('t_stat', float('nan')):<8.3f} "
                         f"{comp.get('df', 0):<6} "
                         f"{comp.get('n_pairs', ''):<5} "
                         f"{comp.get('p_raw', float('nan')):<10.4f} "
                         f"{comp['p_adj']:<10.4f} "
                         f"{sig:<6}")

        lines.append("")
        sig_comps = [c for c in results['comparisons'] if c['significant']]
        if sig_comps:
            lines.append("Significant pairwise differences:")
            for comp in sig_comps:
                direction = "higher" if comp['meandiff'] > 0 else "lower"
                lines.append(f"  • {comp['group1']} vs {comp['group2']}: {comp['group1']} is {direction} (p(Bonf.) = {comp['p_adj']:.4f})")
        else:
            lines.append("No significant pairwise differences found.")

        lines.append("")
        lines.append("")

    total_measures = len(tukey_results)
    measures_with_sig_pairs = len([r for r in tukey_results.values()
                                   if 'comparisons' in r and
                                   any(comp['significant'] for comp in r['comparisons'])])
    lines.append("SUMMARY OF POST-HOC RESULTS:")
    lines.append("-" * 40)
    lines.append(f"Measures tested: {total_measures}")
    lines.append(f"Measures with significant pairwise differences: {measures_with_sig_pairs}")

    if measures_with_sig_pairs > 0:
        lines.append("\nSignificant pairwise differences were found, indicating specific")
        lines.append("weeks that differ significantly from each other.")
    else:
        lines.append("\nNo significant pairwise differences found despite significant omnibus test.")

    lines.append("\n" + "=" * 80)
    lines.append("")
    formatted_output = "\n".join(lines)
    print(formatted_output)
    return formatted_output


def save_frontloading_analysis_to_file(weekly_averages: Dict, anova_output: str, save_path: Path, tukey_output: str = "") -> Path:
    """Save front-loading behavior analysis to a separate report file.
    
    This report focuses specifically on front-loading metrics:
    - % of licks in first 5 minutes
    - % of bouts in first 5 minutes
    - Time to 50% of total licks (minutes)
    
    Also includes ANOVA and Tukey HSD results for these metrics.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        anova_output: Formatted ANOVA results string
        save_path: Path where to save the text file
        tukey_output: Formatted Tukey HSD results string (optional)
        
    Returns:
        Path to the saved text file
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FRONT-LOADING BEHAVIOR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("This report analyzes front-loading behavior using three metrics:\n")
        f.write("1. % of Licks in First 5 Minutes\n")
        f.write("2. % of Bouts in First 5 Minutes\n")
        f.write("3. Time to Reach 50% of Total Licks (minutes)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        # Sort dates chronologically
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
        
        # Descriptive statistics section
        f.write("DESCRIPTIVE STATISTICS BY WEEK\n")
        f.write("=" * 80 + "\n\n")
        
        # % Licks in first 5 minutes table
        f.write("% OF LICKS IN FIRST 5 MINUTES:\n")
        f.write(f"{'Week':<8} {'Date':<12} {'Mean':<12} {'Std':<12} {'SEM':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, date in enumerate(sorted_dates, 1):
            data = weekly_averages[date]
            pcts = data.get('first_5min_lick_pcts_per_animal', np.array([]))
            mean_pct = data.get('avg_first_5min_lick_pct', np.nan)
            std_pct = data.get('std_first_5min_lick_pct', np.nan)
            sem_pct = data.get('sem_first_5min_lick_pct', np.nan)
            min_pct = np.min(pcts) if len(pcts) > 0 else np.nan
            max_pct = np.max(pcts) if len(pcts) > 0 else np.nan
            
            f.write(f"{i:<8} {date:<12} {mean_pct:<12.2f} {std_pct:<12.2f} {sem_pct:<12.2f} {min_pct:<12.2f} {max_pct:<12.2f}\n")
        
        f.write("\n\n")
        
        # % Bouts in first 5 minutes table
        f.write("% OF BOUTS IN FIRST 5 MINUTES:\n")
        f.write(f"{'Week':<8} {'Date':<12} {'Mean':<12} {'Std':<12} {'SEM':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, date in enumerate(sorted_dates, 1):
            data = weekly_averages[date]
            pcts = data.get('first_5min_bout_pcts_per_animal', np.array([]))
            mean_pct = data.get('avg_first_5min_bout_pct', np.nan)
            std_pct = data.get('std_first_5min_bout_pct', np.nan)
            sem_pct = data.get('sem_first_5min_bout_pct', np.nan)
            min_pct = np.min(pcts) if len(pcts) > 0 else np.nan
            max_pct = np.max(pcts) if len(pcts) > 0 else np.nan
            
            f.write(f"{i:<8} {date:<12} {mean_pct:<12.2f} {std_pct:<12.2f} {sem_pct:<12.2f} {min_pct:<12.2f} {max_pct:<12.2f}\n")
        
        f.write("\n\n")
        
        # Time to 50% of total licks table
        f.write("TIME TO 50% OF TOTAL LICKS (MINUTES):\n")
        f.write(f"{'Week':<8} {'Date':<12} {'Mean':<12} {'Std':<12} {'SEM':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-" * 80 + "\n")
        
        for week_idx, date in enumerate(sorted_dates, 1):
            data = weekly_averages[date]
            avg_time = data.get('avg_time_to_50pct_licks', np.nan)
            std_time = data.get('std_time_to_50pct_licks', np.nan)
            sem_time = data.get('sem_time_to_50pct_licks', np.nan)
            
            time_vals = data.get('time_to_50pct_licks_per_animal', [])
            time_vals_valid = [t for t in time_vals if not np.isnan(t)]
            min_time = np.min(time_vals_valid) if len(time_vals_valid) > 0 else np.nan
            max_time = np.max(time_vals_valid) if len(time_vals_valid) > 0 else np.nan
            
            avg_str = f"{avg_time:.2f}" if not np.isnan(avg_time) else "N/A"
            std_str = f"{std_time:.2f}" if not np.isnan(std_time) else "N/A"
            sem_str = f"{sem_time:.2f}" if not np.isnan(sem_time) else "N/A"
            min_str = f"{min_time:.2f}" if not np.isnan(min_time) else "N/A"
            max_str = f"{max_time:.2f}" if not np.isnan(max_time) else "N/A"
            
            f.write(f"{week_idx:<8} {date:<12} {avg_str:<12} {std_str:<12} {sem_str:<12} {min_str:<12} {max_str:<12}\n")
        
        f.write("\n")
        
        # Add ANOVA results
        if anova_output:
            f.write("\n\n")
            f.write(anova_output)
        
        # Add Tukey HSD results
        if tukey_output:
            f.write("\n\n")
            f.write(tukey_output)
        
        f.write("=" * 80 + "\n")
        f.write("END OF FRONT-LOADING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
    
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
            'animal_ids': result.get('animal_ids', []),  # Preserve animal IDs for unique counting
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
    
    # Collect all animal data from all weeks and track unique animal IDs
    all_data = []
    all_unique_animal_ids = set()
    bin_size_min = None
    
    sorted_dates = sort_dates_chronologically(list(lick_rate_data.keys()))
    
    for date in sorted_dates:
        data = lick_rate_data[date]
        all_data.append(data['all_animal_data'])
        
        # Track unique animal IDs
        animal_ids = data.get('animal_ids', [])
        if animal_ids:
            all_unique_animal_ids.update(animal_ids)
        
        if bin_size_min is None:
            bin_size_min = data.get('bin_size_min', 5.0)
        print(f"  Week {date}: {data['n_animals']} animals, CA {data['ca_percent']}%")
    
    # Concatenate all animal data (all weeks, all animals)
    all_animals_all_weeks = np.vstack(all_data)
    n_unique_animals = len(all_unique_animal_ids)
    print(f"\nTotal observations: {all_animals_all_weeks.shape[0]} across {len(sorted_dates)} weeks")
    print(f"Unique mice: {n_unique_animals}")
    
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
    ax.set_title(f'Comprehensive Lick Rate - All Weeks Combined (n={n_unique_animals} mice, {len(sorted_dates)} weeks)',
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
    
    plt.tight_layout()
    
    print("=" * 80 + "\n")
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Comprehensive lick rate plot saved to: {save_path}")
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_first_5min_by_week(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot showing percentage of licks in first 5 minutes for each week.
    
    In the non-ramp design, all mice stay at the same CA% across all weeks.
    This creates a single plot with one bar per week showing average percentage
    with individual mouse data points overlaid and SEM error bars.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: First 5-Minute Lick Percentage by Week")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week
    week_labels = []
    ca_percents = []
    avg_pcts = []
    sem_pcts = []
    individual_data = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_pcts.append(data['avg_first_5min_lick_pct'])
        
        # Calculate SEM
        individual_pcts = data['first_5min_lick_pcts_per_animal']
        individual_data.append(individual_pcts)
        n = len(individual_pcts)
        sem = data['std_first_5min_lick_pct'] / np.sqrt(n) if n > 0 else 0
        sem_pcts.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_pcts[-1]:.2f}%, SEM={sem:.2f}%, n={n}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'dodgerblue'
    elif ca_pct <= 1.0:
        bar_color = 'skyblue'
    else:
        bar_color = 'orangered'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_pcts, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_pcts, yerr=sem_pcts,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2)
    
    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility
    
    for i, individual_pcts in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_pcts))
        ax.scatter([x_positions[i]] * len(individual_pcts) + jitter, individual_pcts,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('% of Licks in First 5 Minutes', fontsize=12, weight='bold')
    ax.set_title(f'Percentage of Licks in First 5 Minutes Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_first_5min_bouts_by_week(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot showing percentage of bouts in first 5 minutes for each week.
    
    In the non-ramp design, all mice stay at the same CA% across all weeks.
    This creates a single plot with one bar per week showing average percentage
    with individual mouse data points overlaid and SEM error bars.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: First 5-Minute Bout Percentage by Week")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week
    week_labels = []
    ca_percents = []
    avg_bout_pcts = []
    sem_bout_pcts = []
    individual_data = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_bout_pcts.append(data['avg_first_5min_bout_pct'])
        
        # Calculate SEM
        individual_bout_pcts = data['first_5min_bout_pcts_per_animal']
        individual_data.append(individual_bout_pcts)
        n = len(individual_bout_pcts)
        sem = data['std_first_5min_bout_pct'] / np.sqrt(n) if n > 0 else 0
        sem_bout_pcts.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_bout_pcts[-1]:.2f}%, SEM={sem:.2f}%, n={n}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'seagreen'
    elif ca_pct <= 1.0:
        bar_color = 'mediumseagreen'
    else:
        bar_color = 'darkseagreen'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_bout_pcts, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_bout_pcts, yerr=sem_bout_pcts,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2)
    
    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility
    
    for i, individual_bout_pcts in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_bout_pcts))
        ax.scatter([x_positions[i]] * len(individual_bout_pcts) + jitter, individual_bout_pcts,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('% of Bouts in First 5 Minutes', fontsize=12, weight='bold')
    ax.set_title(f'Percentage of Bouts in First 5 Minutes Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_first_5min_bouts_by_week_with_lines(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot with individual mouse trajectories for bout percentages connected by lines across weeks.
    
    Shows bars for weekly averages plus lines connecting each individual mouse's data
    across weeks in the non-ramp design (constant CA%).
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: First 5-Minute Bout Percentage by Week (with Individual Trajectories)")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week and track individual animals
    week_labels = []
    ca_percents = []
    avg_bout_pcts = []
    sem_bout_pcts = []
    animal_ids_by_week = []
    individual_data_by_week = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_bout_pcts.append(data['avg_first_5min_bout_pct'])
        
        # Get individual animal data and IDs
        individual_bout_pcts = data['first_5min_bout_pcts_per_animal']
        animal_ids = data.get('animal_ids', [f"Animal_{j+1}" for j in range(len(individual_bout_pcts))])
        
        individual_data_by_week.append(individual_bout_pcts)
        animal_ids_by_week.append(animal_ids)
        
        n = len(individual_bout_pcts)
        sem = data['std_first_5min_bout_pct'] / np.sqrt(n) if n > 0 else 0
        sem_bout_pcts.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_bout_pcts[-1]:.2f}%, SEM={sem:.2f}%, n={n}")
    
    # Build a mapping of animal ID to its trajectory across weeks
    # animal_trajectories: dict mapping animal_id -> list of (week_idx, percentage) tuples
    animal_trajectories = {}
    
    for week_idx, (animal_ids, individual_bout_pcts) in enumerate(zip(animal_ids_by_week, individual_data_by_week)):
        for animal_id, bout_pct in zip(animal_ids, individual_bout_pcts):
            if animal_id not in animal_trajectories:
                animal_trajectories[animal_id] = []
            animal_trajectories[animal_id].append((week_idx, bout_pct))
    
    print(f"\nTracking {len(animal_trajectories)} individual animals across {n_weeks} weeks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'seagreen'
    elif ca_pct <= 1.0:
        bar_color = 'mediumseagreen'
    else:
        bar_color = 'darkseagreen'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_bout_pcts, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5, zorder=5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_bout_pcts, yerr=sem_bout_pcts,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2, zorder=6)
    
    # Plot individual animal trajectories with lines
    cmap = plt.cm.tab20  # Colormap with 20 distinct colors
    colors = [cmap(i % 20) for i in range(len(animal_trajectories))]
    
    for idx, (animal_id, trajectory) in enumerate(animal_trajectories.items()):
        # Sort trajectory by week index
        trajectory.sort(key=lambda x: x[0])
        
        if len(trajectory) > 1:  # Only draw lines if animal appears in multiple weeks
            weeks = [t[0] for t in trajectory]
            bout_pcts = [t[1] for t in trajectory]
            
            # Draw line connecting this animal's data points
            ax.plot(weeks, bout_pcts, color=colors[idx], alpha=0.4, linewidth=1.5, zorder=3)
            
            # Draw markers for this animal's data points
            ax.scatter(weeks, bout_pcts, color=colors[idx], s=50, alpha=0.7, 
                      edgecolors='black', linewidths=0.8, zorder=9)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('% of Bouts in First 5 Minutes', fontsize=12, weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Bout Percentage Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks and {n_animals} individual mouse trajectories")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_time_to_50pct_by_week(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot showing time to 50% of total licks for each week.
    
    In the non-ramp design, all mice stay at the same CA% across all weeks.
    This creates a single plot with one bar per week showing average time
    with individual mouse data points overlaid and SEM error bars.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: Time to 50% of Total Licks by Week")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week
    week_labels = []
    ca_percents = []
    avg_times = []
    sem_times = []
    individual_data = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_times.append(data['avg_time_to_50pct_licks'])
        
        # Calculate SEM
        individual_times = data['time_to_50pct_licks_per_animal']
        individual_data.append(individual_times)
        n = len(individual_times)
        sem = data['std_time_to_50pct_licks'] / np.sqrt(n) if n > 0 else 0
        sem_times.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_times[-1]:.2f} min, SEM={sem:.2f} min, n={n}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'mediumpurple'
    elif ca_pct <= 1.0:
        bar_color = 'plum'
    else:
        bar_color = 'darkviolet'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_times, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_times, yerr=sem_times,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2)
    
    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility
    
    for i, individual_times in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_times))
        ax.scatter([x_positions[i]] * len(individual_times) + jitter, individual_times,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('Time to 50% of Total Licks (minutes)', fontsize=12, weight='bold')
    ax.set_title(f'Time to 50% of Total Licks Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_time_to_50pct_by_week_with_lines(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot with individual mouse trajectories for time to 50% connected by lines across weeks.
    
    Shows bars for weekly averages plus lines connecting each individual mouse's data
    across weeks in the non-ramp design (constant CA%).
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: Time to 50% of Total Licks by Week (with Individual Trajectories)")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week and track individual animals
    week_labels = []
    ca_percents = []
    avg_times = []
    sem_times = []
    animal_ids_by_week = []
    individual_data_by_week = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_times.append(data['avg_time_to_50pct_licks'])
        
        # Get individual animal data and IDs
        individual_times = data['time_to_50pct_licks_per_animal']
        animal_ids = data.get('animal_ids', [f"Animal_{j+1}" for j in range(len(individual_times))])
        
        individual_data_by_week.append(individual_times)
        animal_ids_by_week.append(animal_ids)
        
        n = len(individual_times)
        sem = data['std_time_to_50pct_licks'] / np.sqrt(n) if n > 0 else 0
        sem_times.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_times[-1]:.2f} min, SEM={sem:.2f} min, n={n}")
    
    # Build a mapping of animal ID to its trajectory across weeks
    # animal_trajectories: dict mapping animal_id -> list of (week_idx, time) tuples
    animal_trajectories = {}
    
    for week_idx, (animal_ids, individual_times) in enumerate(zip(animal_ids_by_week, individual_data_by_week)):
        for animal_id, time_val in zip(animal_ids, individual_times):
            if animal_id not in animal_trajectories:
                animal_trajectories[animal_id] = []
            animal_trajectories[animal_id].append((week_idx, time_val))
    
    print(f"\nTracking {len(animal_trajectories)} individual animals across {n_weeks} weeks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'mediumpurple'
    elif ca_pct <= 1.0:
        bar_color = 'plum'
    else:
        bar_color = 'darkviolet'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_times, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5, zorder=5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_times, yerr=sem_times,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2, zorder=6)
    
    # Plot individual animal trajectories with lines
    cmap = plt.cm.tab20  # Colormap with 20 distinct colors
    colors = [cmap(i % 20) for i in range(len(animal_trajectories))]
    
    for idx, (animal_id, trajectory) in enumerate(animal_trajectories.items()):
        # Sort trajectory by week index
        trajectory.sort(key=lambda x: x[0])
        
        if len(trajectory) > 1:  # Only draw lines if animal appears in multiple weeks
            weeks = [t[0] for t in trajectory]
            times = [t[1] for t in trajectory]
            
            # Draw line connecting this animal's data points
            ax.plot(weeks, times, color=colors[idx], alpha=0.4, linewidth=1.5, zorder=3)
            
            # Draw markers for this animal's data points
            ax.scatter(weeks, times, color=colors[idx], s=50, alpha=0.7, 
                      edgecolors='black', linewidths=0.8, zorder=9)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('Time to 50% of Total Licks (minutes)', fontsize=12, weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Time to 50% Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks and {n_animals} individual mouse trajectories")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_first_5min_by_week_with_lines(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Create bar plot with individual mouse trajectories connected by lines across weeks.
    
    Shows bars for weekly averages plus lines connecting each individual mouse's data
    across weeks in the non-ramp design (constant CA%).
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    print("\n" + "=" * 80)
    print("PLOTTING: First 5-Minute Lick Percentage by Week (with Individual Trajectories)")
    print("=" * 80)
    
    # Sort dates chronologically to get proper week order
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
    n_weeks = len(sorted_dates)
    
    if n_weeks == 0:
        print("ERROR: No data found in weekly_averages.")
        return None
    
    print(f"Found {n_weeks} weeks of data")
    
    # Extract data for each week and track individual animals
    week_labels = []
    ca_percents = []
    avg_pcts = []
    sem_pcts = []
    animal_ids_by_week = []
    individual_data_by_week = []
    
    for i, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        week_labels.append(f"Week {i+1}")
        ca_percents.append(data['ca_percent'])
        avg_pcts.append(data['avg_first_5min_lick_pct'])
        
        # Get individual animal data and IDs
        individual_pcts = data['first_5min_lick_pcts_per_animal']
        animal_ids = data.get('animal_ids', [f"Animal_{j+1}" for j in range(len(individual_pcts))])
        
        individual_data_by_week.append(individual_pcts)
        animal_ids_by_week.append(animal_ids)
        
        n = len(individual_pcts)
        sem = data['std_first_5min_lick_pct'] / np.sqrt(n) if n > 0 else 0
        sem_pcts.append(sem)
        
        print(f"  Week {i+1} ({data['ca_percent']}% CA): avg={avg_pcts[-1]:.2f}%, SEM={sem:.2f}%, n={n}")
    
    # Build a mapping of animal ID to its trajectory across weeks
    # animal_trajectories: dict mapping animal_id -> list of (week_idx, percentage) tuples
    animal_trajectories = {}
    
    for week_idx, (animal_ids, individual_pcts) in enumerate(zip(animal_ids_by_week, individual_data_by_week)):
        for animal_id, pct in zip(animal_ids, individual_pcts):
            if animal_id not in animal_trajectories:
                animal_trajectories[animal_id] = []
            animal_trajectories[animal_id].append((week_idx, pct))
    
    print(f"\nTracking {len(animal_trajectories)} individual animals across {n_weeks} weeks")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # In non-ramp design, CA% should be constant, so use a single color
    ca_pct = ca_percents[0] if ca_percents else 0
    if ca_pct == 0:
        bar_color = 'dodgerblue'
    elif ca_pct <= 1.0:
        bar_color = 'skyblue'
    else:
        bar_color = 'orangered'
    
    x_positions = np.arange(n_weeks)
    bar_width = 0.6
    
    # Plot bars
    bars = ax.bar(x_positions, avg_pcts, bar_width,
                 color=bar_color, alpha=0.7, edgecolor='black', linewidth=1.5, zorder=5)
    
    # Add error bars
    ax.errorbar(x_positions, avg_pcts, yerr=sem_pcts,
               fmt='none', ecolor='black', capsize=5, linewidth=2, capthick=2, zorder=6)
    
    # Plot individual animal trajectories with lines
    cmap = plt.cm.tab20  # Colormap with 20 distinct colors
    colors = [cmap(i % 20) for i in range(len(animal_trajectories))]
    
    for idx, (animal_id, trajectory) in enumerate(animal_trajectories.items()):
        # Sort trajectory by week index
        trajectory.sort(key=lambda x: x[0])
        
        if len(trajectory) > 1:  # Only draw lines if animal appears in multiple weeks
            weeks = [t[0] for t in trajectory]
            pcts = [t[1] for t in trajectory]
            
            # Draw line connecting this animal's data points
            ax.plot(weeks, pcts, color=colors[idx], alpha=0.4, linewidth=1.5, zorder=3)
            
            # Draw markers for this animal's data points
            ax.scatter(weeks, pcts, color=colors[idx], s=50, alpha=0.7, 
                      edgecolors='black', linewidths=0.8, zorder=9)
    
    # Formatting
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('% of Licks in First 5 Minutes', fontsize=12, weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Lick Percentage Across Weeks\n({ca_pct}% CA - Non-Ramp Design)',
                fontsize=14, weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    print(f"\nPlot created with {n_weeks} weeks and {n_animals} individual mouse trajectories")
    print("=" * 80 + "\n")
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
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
    
    # Set up KDE cache file path (saves computation time on subsequent runs)
    cache_dir = capacitive_file.parent / 'kde_cache'
    cache_filename = capacitive_file.stem + '_kde_cache.csv'
    cache_file = cache_dir / cache_filename
    
    print(f"\nComputing sensor KDE baselines and raw statistics:")
    sensor_kdes = compute_sensor_KDE(df, sensor_cols, cache_file=cache_file, verbose=True)
    df = compute_KDE_normalizations(df, sensor_cols, sensor_kdes)
    
    # Use fixed threshold (same as lick_detection.py)
    print(f"\nUsing fixed threshold: {fixed_threshold}")
    thresholds = pd.Series({sensor: fixed_threshold for sensor in sensor_cols})
    events_df = detect_events_above_threshold(df, sensor_cols, thresholds)
    
    # Filter to only first 30 minutes of session (1800 seconds)
    original_length = len(events_df)
    events_df = events_df[events_df['Time_sec'] < 1800].copy()
    print(f"  Filtered to first 30 minutes: {len(events_df)}/{original_length} time points retained")
    
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
    
    # Extract sensor mappings, animal IDs, sex, and CA%
    sensor_to_weight = {}
    sensor_to_weight_loss = {}
    sensor_to_fecal = {}
    sensor_to_animal_id = {}  # Track animal IDs for repeated measures
    sensor_to_sex = {}  # NEW: Track sex for mixed ANOVA
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
        
        # Extract sex if available
        if 'sex' in row and pd.notna(row['sex']):
            sensor_to_sex[sensor_name] = str(row['sex']).upper().strip()
        else:
            sensor_to_sex[sensor_name] = "Unknown"
        
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
    first_5min_lick_pcts = []  # NEW: Percentage of licks in first 5 minutes
    first_5min_bout_pcts = []  # NEW: Percentage of bouts in first 5 minutes
    time_to_50pct_licks = []  # NEW: Time to reach 50% of total licks (minutes)
    
    print(f"\nPer-sensor data (only selected sensors):")
    for sensor in selected_sensors:
        # Lick counts
        event_col = f"{sensor}_event"
        if event_col in events_df.columns:
            sensor_licks = events_df[event_col].sum()
            total_licks += sensor_licks
            lick_counts.append(sensor_licks)
            
            # Calculate first 5 minute lick percentage
            first_5min_licks = ((events_df[event_col]) & (events_df['Time_sec'] < 300)).sum()
            first_5min_lick_pct = (first_5min_licks / sensor_licks * 100) if sensor_licks > 0 else 0
            first_5min_lick_pcts.append(first_5min_lick_pct)
            
            # Calculate time to 50% of total licks
            time_50pct = calculate_time_to_50_percent_licks(events_df, sensor)
            time_to_50pct_licks.append(time_50pct)
            
            time_50pct_str = f"{time_50pct:.2f} min" if not np.isnan(time_50pct) else "N/A"
            print(f"  {sensor}: Total licks = {sensor_licks}, First 5min licks = {first_5min_licks}, First 5min % = {first_5min_lick_pct:.1f}%, Time to 50% = {time_50pct_str}")
            
            sensor_status = f"OK - {sensor_licks} licks detected ({first_5min_lick_pct:.1f}% in first 5min)"
        else:
            lick_counts.append(0)
            first_5min_lick_pcts.append(0)
            time_to_50pct_licks.append(np.nan)
            sensor_status = f"WARNING - Event column '{event_col}' not found in data!"
        
        # Bout counts
        if sensor in bout_dict:
            sensor_bouts = bout_dict[sensor]['bout_count']
            sensor_bout_licks = bout_dict[sensor]['bout_sizes'].sum() if len(bout_dict[sensor]['bout_sizes']) > 0 else 0
            total_bouts += sensor_bouts
            total_bout_licks += sensor_bout_licks
            bout_counts.append(sensor_bouts)
            
            # Calculate first 5 minute bout percentage
            bout_start_times = bout_dict[sensor]['bout_start_times']
            first_5min_bouts = np.sum(bout_start_times < 300) if len(bout_start_times) > 0 else 0
            first_5min_bout_pct = (first_5min_bouts / sensor_bouts * 100) if sensor_bouts > 0 else 0
            first_5min_bout_pcts.append(first_5min_bout_pct)
            print(f"  {sensor}: Total bouts = {sensor_bouts}, First 5min bouts = {first_5min_bouts}, First 5min % = {first_5min_bout_pct:.1f}%")
        else:
            bout_counts.append(0)
            first_5min_bout_pcts.append(0)
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
        animal_ids.append(sensor_to_animal_id.get(sensor, f"Unknown_{sensor}"))  # Track animal ID
        
        # Additional detail line for metadata
        if lick_counts[-1] == 0:
            print(f"       Metadata: bottle_wt={bottle_wt}, total_wt={total_wt}, fecal={fecal}")
            print(f"       >>> INVESTIGATE: This sensor was selected in master CSV but detected 0 licks! <<<")
    
    # Extract sex information for each animal (in same order as other data)
    animal_sexes = [sensor_to_sex.get(sensor, "Unknown") for sensor in selected_sensors]
    
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
        'animal_ids': animal_ids,  # Include animal IDs for repeated measures tracking
        'animal_sexes': animal_sexes,  # NEW: Include sex for mixed ANOVA
        'first_5min_lick_pcts': np.array(first_5min_lick_pcts),  # NEW: First 5 min lick percentages
        'first_5min_bout_pcts': np.array(first_5min_bout_pcts),  # NEW: First 5 min bout percentages
        'time_to_50pct_licks': np.array(time_to_50pct_licks),  # NEW: Time to 50% licks per animal (minutes)
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
    
    # Perform front-loading ANOVA and Tukey HSD
    print("\nStep 6b: Performing Front-Loading ANOVA")
    print("-" * 40)
    
    frontloading_anova_results = perform_frontloading_anova(weekly_averages)
    frontloading_anova_output = display_frontloading_anova_results(frontloading_anova_results)
    
    print("\nStep 6c: Performing Tukey HSD for Front-Loading Measures")
    print("-" * 40)
    
    frontloading_tukey_results = perform_frontloading_tukey_hsd(frontloading_anova_results, weekly_averages)
    frontloading_tukey_output = display_frontloading_tukey_results(frontloading_tukey_results)
    
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
    
    # Plot first 5-minute percentage by week
    print("\nPlotting percentage of licks in first 5 minutes by week...")
    plot_first_5min_by_week(weekly_averages, show=True)
    
    # Plot first 5-minute percentage by week with individual mouse trajectories
    print("\nPlotting percentage of licks in first 5 minutes by week (with mouse trajectories)...")
    plot_first_5min_by_week_with_lines(weekly_averages, show=True)
    
    # Plot first 5-minute bout percentage by week
    print("\nPlotting percentage of bouts in first 5 minutes by week...")
    plot_first_5min_bouts_by_week(weekly_averages, show=True)
    
    # Plot first 5-minute bout percentage by week with individual mouse trajectories
    print("\nPlotting percentage of bouts in first 5 minutes by week (with mouse trajectories)...")
    plot_first_5min_bouts_by_week_with_lines(weekly_averages, show=True)
    
    # Plot time to 50% of licks by week
    print("\nPlotting time to 50% of total licks by week...")
    plot_time_to_50pct_by_week(weekly_averages, show=True)
    
    # Plot time to 50% of licks by week with individual mouse trajectories
    print("\nPlotting time to 50% of total licks by week (with mouse trajectories)...")
    plot_time_to_50pct_by_week_with_lines(weekly_averages, show=True)
    
    # Optional: Save comprehensive summary with all statistical results
    save_table = input("\nSave weekly averages, ANOVA, and Tukey HSD results as text file? (y/n): ").strip().lower()
    if save_table in ['y', 'yes']:
        table_path = master_csv.parent / "comprehensive_statistical_analysis_summary.txt"
        # Combine all outputs (excluding front-loading metrics which are in separate report)
        combined_output = formatted_output + "\n" + anova_output + "\n" + tukey_output
        save_weekly_averages_to_file(weekly_averages, combined_output, table_path)
        print(f"Comprehensive statistical analysis saved to: {table_path}")
    
    # Always save front-loading analysis to separate report
    frontloading_path = master_csv.parent / "frontloading_analysis_report.txt"
    save_frontloading_analysis_to_file(weekly_averages, frontloading_anova_output, frontloading_path, frontloading_tukey_output)
    print(f"Front-loading analysis report saved to: {frontloading_path}")
    
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
    
    # Optional: Save first 5-minute percentage plot
    save_first_5min = input("\nSave first 5-minute percentage plot as SVG? (y/n): ").strip().lower()
    if save_first_5min in ['y', 'yes']:
        save_path_5min = master_csv.parent / "first_5min_lick_percentage_by_week.svg"
        fig_5min = plot_first_5min_by_week(weekly_averages, save_path=save_path_5min, show=False)
        if fig_5min:
            plt.close(fig_5min)
            print(f"Saved first 5-minute percentage plot to: {save_path_5min}")
    
    # Optional: Save first 5-minute percentage plot with trajectories
    save_first_5min_lines = input("\nSave first 5-minute percentage plot with mouse trajectories as SVG? (y/n): ").strip().lower()
    if save_first_5min_lines in ['y', 'yes']:
        save_path_5min_lines = master_csv.parent / "first_5min_lick_percentage_by_week_with_trajectories.svg"
        fig_5min_lines = plot_first_5min_by_week_with_lines(weekly_averages, save_path=save_path_5min_lines, show=False)
        if fig_5min_lines:
            plt.close(fig_5min_lines)
            print(f"Saved first 5-minute percentage plot with trajectories to: {save_path_5min_lines}")
    
    # Optional: Save first 5-minute bout percentage plot
    save_first_5min_bouts = input("\nSave first 5-minute bout percentage plot as SVG? (y/n): ").strip().lower()
    if save_first_5min_bouts in ['y', 'yes']:
        save_path_5min_bouts = master_csv.parent / "first_5min_bout_percentage_by_week.svg"
        fig_5min_bouts = plot_first_5min_bouts_by_week(weekly_averages, save_path=save_path_5min_bouts, show=False)
        if fig_5min_bouts:
            plt.close(fig_5min_bouts)
            print(f"Saved first 5-minute bout percentage plot to: {save_path_5min_bouts}")
    
    # Optional: Save first 5-minute bout percentage plot with trajectories
    save_first_5min_bouts_lines = input("\nSave first 5-minute bout percentage plot with mouse trajectories as SVG? (y/n): ").strip().lower()
    if save_first_5min_bouts_lines in ['y', 'yes']:
        save_path_5min_bouts_lines = master_csv.parent / "first_5min_bout_percentage_by_week_with_trajectories.svg"
        fig_5min_bouts_lines = plot_first_5min_bouts_by_week_with_lines(weekly_averages, save_path=save_path_5min_bouts_lines, show=False)
        if fig_5min_bouts_lines:
            plt.close(fig_5min_bouts_lines)
            print(f"Saved first 5-minute bout percentage plot with trajectories to: {save_path_5min_bouts_lines}")
    
    # Optional: Save time to 50% plot
    save_time_50pct = input("\nSave time to 50% of licks plot as SVG? (y/n): ").strip().lower()
    if save_time_50pct in ['y', 'yes']:
        save_path_50pct = master_csv.parent / "time_to_50pct_licks_by_week.svg"
        fig_50pct = plot_time_to_50pct_by_week(weekly_averages, save_path=save_path_50pct, show=False)
        if fig_50pct:
            plt.close(fig_50pct)
            print(f"Saved time to 50% plot to: {save_path_50pct}")
    
    # Optional: Save time to 50% plot with trajectories
    save_time_50pct_lines = input("\nSave time to 50% of licks plot with mouse trajectories as SVG? (y/n): ").strip().lower()
    if save_time_50pct_lines in ['y', 'yes']:
        save_path_50pct_lines = master_csv.parent / "time_to_50pct_licks_by_week_with_trajectories.svg"
        fig_50pct_lines = plot_time_to_50pct_by_week_with_lines(weekly_averages, save_path=save_path_50pct_lines, show=False)
        if fig_50pct_lines:
            plt.close(fig_50pct_lines)
            print(f"Saved time to 50% plot with trajectories to: {save_path_50pct_lines}")
    
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