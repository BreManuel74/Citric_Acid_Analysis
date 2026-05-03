"""
Lick Analysis — Unified Ramp / Non-Ramp Script

Performs statistical comparisons across multiple weeks of capacitive sensor data.
Set EXPERIMENT_MODE (below) to control which experimental design is analysed:

  'ramp'    — CA% increases each week (within-subjects factor = CA_Percent)
  'nonramp' — CA% is constant across weeks (within-subjects factor = Week)

Usage:
    python lick_analysis.py
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

# Try importing rpy2 for R-based polynomial contrasts (lme4 / lmerTest / emmeans)
try:
    import rpy2.robjects as _rpy2_ro  # noqa: F401 – import check only
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT MODE  ← Change this line to switch between designs
#   'ramp'    → CA% increases each week; within-subjects factor = CA_Percent
#   'nonramp' → CA% is constant across weeks; within-subjects factor = Week
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_MODE = 'nonramp'  # << CHANGE THIS: 'ramp' or 'nonramp'
# ─────────────────────────────────────────────────────────────────────────────

_MODE_LABELS: dict = {
    'ramp': {
        'factor':            'CA%',
        'factor_col':        'CA_Percent',
        'factor_display':    'CA% (repeated measures)',
        'anova_heading':     'ACROSS CA% CONCENTRATIONS',
        'main_effect':       'CA% MAIN EFFECT',
        'interaction':       'SEX × CA% INTERACTION',
        'stats_header':      'CA% Statistics:',
        'n_key':             'n_ca_levels',
        'sig_across':        'significant differences across CA% concentrations',
        'no_sig':            'CA% main effect: No significant differences',
        'main_sig_in':       'CA% main effect significant in',
        'interaction_label': 'Sex × CA%',
        'post_no_sig':       'No significant CA% effects found - post-hoc tests not needed.',
        'post_header':       'within-subjects CA% comparisons.',
        'post_same':         'Same animals tested at each CA% level',
        'post_differ':       'CA% concentrations that differ',
        'fl_factor':         'CA% (within-subjects)',
        'fl_effect':         'CA% Effect:',
        'fl_sig':            'SIGNIFICANT CA% EFFECT (p < 0.05)',
        'fl_not_sig':        'significant CA% effects',
        'comp_title':        'All CA% Combined',
        'plot_suffix':       'CA% Ramp Design',
    },
    'nonramp': {
        'factor':            'Week',
        'factor_col':        'Week',
        'factor_display':    'Week (repeated measures)',
        'anova_heading':     'ACROSS WEEKS',
        'main_effect':       'WEEK MAIN EFFECT',
        'interaction':       'SEX × WEEK INTERACTION',
        'stats_header':      'Week Statistics:',
        'n_key':             'n_weeks',
        'sig_across':        'significant differences across weeks',
        'no_sig':            'Week main effect: No significant differences',
        'main_sig_in':       'Week main effect significant in',
        'interaction_label': 'Sex × Week',
        'post_no_sig':       'No significant Week effects found - post-hoc tests not needed.',
        'post_header':       'within-subjects Week comparisons.',
        'post_same':         'Same animals measured each week',
        'post_differ':       'weeks that differ',
        'fl_factor':         'Week (within-subjects)',
        'fl_effect':         'Week Effect:',
        'fl_sig':            'SIGNIFICANT WEEK EFFECT (p < 0.05)',
        'fl_not_sig':        'significant Week effects',
        'comp_title':        'All Weeks Combined',
        'plot_suffix':       'Non-Ramp Design',
    },
}
_MLB = _MODE_LABELS[EXPERIMENT_MODE]  # active mode labels shorthand

# Canonical cohort colours (match across_cohort.py)
_COLOR_0PCT  = "#1f77b4"   # 0% CA
_COLOR_2PCT  = "#f79520"   # 2% CA
_COLOR_RAMP  = "#2da048"   # Ramp
_COLOR_OTHER = "#7f3f98"   # fallback

# Active cohort colour — change alongside EXPERIMENT_MODE if needed.
# Default: ramp design → green; nonramp → blue (0% CA).
# If running on 2% CA nonramp data, set to _COLOR_2PCT.
COHORT_COLOR = _COLOR_RAMP if EXPERIMENT_MODE == 'ramp' else _COLOR_0PCT


def _detect_cohort_color(path=None, df=None) -> str:
    """Infer cohort color from the CSV file path, then from CA% values in the data."""
    if path is not None:
        p = str(path).lower()
        if '2wk' in p or '2_wk' in p or '2_week' in p or '2week' in p:
            return _COLOR_OTHER
        if 'ramp' in p:
            return _COLOR_RAMP
        if '2%' in p or '2pct' in p:
            return _COLOR_2PCT
        if '0%' in p or '0pct' in p:
            return _COLOR_0PCT
    if df is not None:
        for col in ('CA (%)', 'CA_Percent', 'ca_percent', 'CA%'):
            if col in df.columns:
                ca_vals = pd.to_numeric(df[col], errors='coerce').dropna().unique()
                if len(ca_vals) > 2:
                    return _COLOR_RAMP
                if len(ca_vals) == 1:
                    v = float(ca_vals[0])
                    if v == 0:
                        return _COLOR_0PCT
                    if v == 2:
                        return _COLOR_2PCT
                    return _COLOR_OTHER
                break
    return COHORT_COLOR  # unchanged if nothing detected

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "lines.linewidth": 0.9,
    "lines.markersize": 4,
    "figure.figsize": (3, 2.5),
    "axes.xmargin": 0,
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


def compute_lick_bouts(events_df: pd.DataFrame, sensor_cols: List[str], ili_cutoff: float = 0.5) -> dict:
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
            std_bottle_weight = np.std(bottle_weights_filtered, ddof=1) if len(bottle_weights_filtered) > 1 else 0
            print(f"Bottle weight calculation: sum={np.sum(bottle_weights_filtered)}, mean={avg_bottle_weight:.2f}")
        else:
            avg_bottle_weight = np.mean(bottle_weights)
            std_bottle_weight = np.std(bottle_weights, ddof=1) if len(bottle_weights) > 1 else 0
            print(f"\nBottle weight calculation: sum={np.sum(bottle_weights)}, mean={avg_bottle_weight:.2f}")
        
        avg_total_weight = np.mean(total_weights)
        print(f"Total weight calculation: sum={np.sum(total_weights)}, mean={avg_total_weight:.2f}")
        
        # Calculate first-5-min percentage averages
        avg_first_5min_lick_pct = np.mean(first_5min_lick_pcts)
        avg_first_5min_bout_pct = np.mean(first_5min_bout_pcts)
        
        # Calculate time to 50% statistics (excluding NaN values)
        time_to_50pct_valid = time_to_50pct_licks[~np.isnan(time_to_50pct_licks)]
        avg_time_to_50pct = np.mean(time_to_50pct_valid) if len(time_to_50pct_valid) > 0 else np.nan
        std_time_to_50pct = np.std(time_to_50pct_valid, ddof=1) if len(time_to_50pct_valid) > 1 else np.nan
        
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
        
        std_licks = np.std(lick_counts, ddof=1) if len(lick_counts) > 1 else 0
        std_bouts = np.std(bout_counts, ddof=1) if len(bout_counts) > 1 else 0
        std_fecal = np.std(fecal_counts, ddof=1) if len(fecal_counts) > 1 else 0
        std_total_weight = np.std(total_weights, ddof=1) if len(total_weights) > 1 else 0
        std_first_5min_lick_pct = np.std(first_5min_lick_pcts, ddof=1) if len(first_5min_lick_pcts) > 1 else 0
        std_first_5min_bout_pct = np.std(first_5min_bout_pcts, ddof=1) if len(first_5min_bout_pcts) > 1 else 0
        
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
    Perform MIXED ANOVA tests for each of the 5 measures.
    - Between-subjects factor: Sex (each animal has one sex)
    - Within-subjects factor: Week (nonramp) or CA_Percent (ramp)

    Uses pingouin.mixed_anova() to properly account for both factors and repeated
    measurements. Falls back to standard ANOVA with warning if pingouin is
    not available.

    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages

    Returns:
        Dictionary containing Mixed ANOVA results for each measure
    """
    if not HAS_PINGOUIN:
        print("\n" + "="*80)
        print("WARNING: Mixed ANOVA requires pingouin library")
        print("Falling back to standard ANOVA (does NOT account for repeated measures or sex)")
        print("Install pingouin with: pip install pingouin")
        print("="*80 + "\n")
        return _perform_standard_anova_fallback(weekly_averages)

    # --- Mode setup ---
    within_col = _MLB['factor_col']  # 'CA_Percent' (ramp) or 'Week' (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_weeks = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
        sorted_weeks = sort_dates_chronologically(list(weekly_averages.keys()))

    # Build long-format dataframe for mixed ANOVA
    long_data = []
    animal_sex_map: Dict[str, str] = {}

    for week_idx, date in enumerate(sorted_weeks):
        data = weekly_averages[date]
        animal_ids = data.get('animal_ids', [])
        animal_sexes = data.get('animal_sexes', [])

        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(len(data['avg_licks_per_animal']))]
        if not animal_sexes or len(animal_sexes) != len(animal_ids):
            animal_sexes = ["Unknown"] * len(animal_ids)

        # Diagnostic print (matches ramp behavior)
        print(f"\n{date} ({data['ca_percent']}% CA):")
        print(f"  Number of animals: {len(animal_ids)}")
        print(f"  Animal IDs: {animal_ids}")
        print(f"  Sexes: {animal_sexes}")

        for i, animal_id in enumerate(animal_ids):
            if animal_id not in animal_sex_map:
                animal_sex_map[animal_id] = animal_sexes[i]
            elif animal_sex_map[animal_id] != animal_sexes[i]:
                print(f"  WARNING: Animal {animal_id} has inconsistent sex: "
                      f"{animal_sex_map[animal_id]} vs {animal_sexes[i]}")

            bottle_weight = data['avg_bottle_weight_per_animal'][i]
            # Within-subjects value: actual CA% (ramp) or 1-indexed week (nonramp)
            group_val = (weekly_averages[date]['ca_percent']
                         if EXPERIMENT_MODE == 'ramp' else week_idx + 1)

            long_data.append({
                'Animal':    animal_id,
                'Sex':       animal_sex_map[animal_id],
                within_col:  group_val,
                'Date':      date,
                'licks':         data['avg_licks_per_animal'][i],
                'bouts':         data['avg_bouts_per_animal'][i],
                'fecal':         data['avg_fecal_per_animal'][i],
                'bottle_weight': bottle_weight if bottle_weight > 0 else np.nan,
                'total_weight':  data['avg_total_weight_per_animal'][i],
            })

    print("\n" + "="*80)
    print(f"Total unique animals: {len(animal_sex_map)}")
    print("="*80 + "\n")

    df_long = pd.DataFrame(long_data)

    has_sex_data = ('Sex' in df_long.columns
                    and df_long['Sex'].nunique() > 1
                    and 'Unknown' not in df_long['Sex'].values)

    if not has_sex_data:
        print("\n" + "="*80)
        print("WARNING: Sex data not found or incomplete in master CSV")
        print(f"Falling back to one-way repeated measures ANOVA ({_MLB['factor']} only)")
        print("="*80 + "\n")

    anova_results = {}
    measures = ['licks', 'bouts', 'fecal', 'bottle_weight', 'total_weight']
    measure_names = {
        'licks': 'Total Licks',
        'bouts': 'Total Bouts',
        'fecal': 'Fecal Count',
        'bottle_weight': 'Bottle Weight Loss',
        'total_weight': 'Total Weight Loss',
    }

    print("\n" + "="*80)
    if has_sex_data:
        print("PERFORMING MIXED ANOVA")
        print("Between-subjects factor: Sex")
        print(f"Within-subjects factor: {_MLB['factor']} (repeated measures)")
    else:
        print("PERFORMING REPEATED MEASURES ANOVA (ONE-WAY)")
        print(f"Within-subjects factor: {_MLB['factor']}")
    print("="*80)

    for measure in measures:
        print(f"\nAnalyzing: {measure_names[measure]}")

        if has_sex_data:
            if measure == 'bottle_weight':
                df_measure = df_long[['Animal', 'Sex', within_col, measure]].dropna()
            else:
                df_measure = df_long[['Animal', 'Sex', within_col, measure]].copy()
        else:
            if measure == 'bottle_weight':
                df_measure = df_long[['Animal', within_col, measure]].dropna()
            else:
                df_measure = df_long[['Animal', within_col, measure]].copy()

        n_animals = df_measure['Animal'].nunique()
        n_levels  = df_measure[within_col].nunique()

        animals_per_level = df_measure.groupby('Animal')[within_col].nunique()
        complete_animals  = animals_per_level[animals_per_level == n_levels]
        incomplete_animals = animals_per_level[animals_per_level < n_levels]

        if len(incomplete_animals) > 0:
            print(f"  WARNING: {len(incomplete_animals)} animals missing data at some "
                  f"{_MLB['factor']} levels:")
            for animal_id, n_present in incomplete_animals.items():
                levels_present = sorted(df_measure[df_measure['Animal'] == animal_id][within_col].unique())
                print(f"    {animal_id}: present in {n_present}/{n_levels} levels ({levels_present})")
            print(f"  Complete animals (all {n_levels} levels): {len(complete_animals)}")
            print(f"  Filtering to only animals with complete data...")
            df_measure = df_measure[df_measure['Animal'].isin(complete_animals.index)]
            n_animals = len(complete_animals)
            print(f"  After filtering: n_animals={n_animals}, n_levels={n_levels}")

        if n_levels < 2:
            anova_results[measure] = {
                'measure_name':            measure_names[measure],
                'f_statistic':             np.nan,
                'p_value':                 np.nan,
                'significant':             False,
                'f_statistic_sex':         np.nan,
                'p_value_sex':             np.nan,
                'significant_sex':         False,
                'f_statistic_interaction': np.nan,
                'p_value_interaction':     np.nan,
                'significant_interaction': False,
                'error': f'Insufficient data for ANOVA (need at least 2 {_MLB["factor"]} levels)',
                'is_repeated_measures':    True,
                'is_mixed_anova':          has_sex_data,
            }
            print(f"  ERROR: Insufficient data ({n_levels} {_MLB['factor']} levels)")
            continue

        try:
            if has_sex_data:
                result_table = pg.mixed_anova(
                    dv=measure,
                    within=within_col,
                    between='Sex',
                    subject='Animal',
                    data=df_measure,
                )

                sex_row         = result_table[result_table['Source'] == 'Sex']
                factor_row      = result_table[result_table['Source'] == within_col]
                interaction_row = result_table[result_table['Source'] == 'Interaction']

                f_stat_sex   = sex_row['F'].values[0]    if len(sex_row) > 0    else np.nan
                p_value_sex  = sex_row['p-unc'].values[0] if len(sex_row) > 0   else np.nan
                np2_sex      = (sex_row['np2'].values[0]
                                if len(sex_row) > 0 and 'np2' in sex_row.columns else np.nan)

                f_stat_factor  = factor_row['F'].values[0]     if len(factor_row) > 0     else np.nan
                p_value_factor = factor_row['p-unc'].values[0] if len(factor_row) > 0     else np.nan
                np2_factor     = (factor_row['np2'].values[0]
                                  if len(factor_row) > 0 and 'np2' in factor_row.columns else np.nan)

                if 'p-GG-corr' in factor_row.columns and len(factor_row) > 0:
                    p_gg_factor         = factor_row['p-GG-corr'].values[0]
                    sphericity_violated = (p_gg_factor != p_value_factor)
                else:
                    p_gg_factor         = p_value_factor
                    sphericity_violated = False

                f_stat_interaction  = (interaction_row['F'].values[0]
                                       if len(interaction_row) > 0 else np.nan)
                p_value_interaction = (interaction_row['p-unc'].values[0]
                                       if len(interaction_row) > 0 else np.nan)
                np2_interaction     = (interaction_row['np2'].values[0]
                                       if len(interaction_row) > 0
                                       and 'np2' in interaction_row.columns else np.nan)

                if 'p-GG-corr' in interaction_row.columns and len(interaction_row) > 0:
                    p_gg_interaction              = interaction_row['p-GG-corr'].values[0]
                    sphericity_violated_interaction = (p_gg_interaction != p_value_interaction)
                else:
                    p_gg_interaction              = p_value_interaction
                    sphericity_violated_interaction = False

                f_stat       = f_stat_factor
                p_value      = p_value_factor
                p_gg         = p_gg_factor
                effect_size  = np2_factor

            else:
                result_table = pg.rm_anova(
                    dv=measure,
                    within=within_col,
                    subject='Animal',
                    data=df_measure,
                    detailed=True,
                )

                f_stat  = result_table.loc[result_table['Source'] == within_col, 'F'].values[0]
                p_value = result_table.loc[result_table['Source'] == within_col, 'p-unc'].values[0]

                if 'p-GG-corr' in result_table.columns:
                    p_gg                = result_table.loc[result_table['Source'] == within_col,
                                                           'p-GG-corr'].values[0]
                    sphericity_violated = (p_gg != p_value)
                else:
                    p_gg                = p_value
                    sphericity_violated = False

                effect_size = (result_table.loc[result_table['Source'] == within_col, 'np2'].values[0]
                               if 'np2' in result_table.columns else np.nan)

                f_stat_sex  = np.nan;  p_value_sex  = np.nan;  np2_sex  = np.nan
                f_stat_interaction  = np.nan
                p_value_interaction = np.nan;  p_gg_interaction = np.nan
                np2_interaction = np.nan;  sphericity_violated_interaction = False

            # Descriptive statistics per within-subjects level
            group_stats: List[Dict] = []
            for level_val in sorted(df_measure[within_col].unique()):
                level_data = df_measure[df_measure[within_col] == level_val][measure]
                n    = len(level_data)
                mean = level_data.mean()
                std  = level_data.std(ddof=1)
                sem  = std / np.sqrt(n) if n > 0 else np.nan
                stat: Dict = {
                    'n':        n,
                    'mean':     mean,
                    'median':   level_data.median(),
                    'std':      std,
                    'sem':      sem,
                    'ci_lower': mean - 1.96 * sem,
                    'ci_upper': mean + 1.96 * sem,
                    'q25':      level_data.quantile(0.25),
                    'q75':      level_data.quantile(0.75),
                    'min':      level_data.min(),
                    'max':      level_data.max(),
                }
                if EXPERIMENT_MODE == 'ramp':
                    stat['ca_percent'] = level_val
                else:
                    stat['week_number'] = int(level_val)
                    stat['week']        = sorted_weeks[int(level_val) - 1]
                group_stats.append(stat)

            sex_stats: List[Dict] = []
            if has_sex_data:
                for sex_val in sorted(df_measure['Sex'].unique()):
                    sex_d = df_measure[df_measure['Sex'] == sex_val][measure]
                    sex_stats.append({'sex': sex_val, 'n': len(sex_d),
                                      'mean': sex_d.mean(), 'std': sex_d.std(ddof=1)})

            anova_results[measure] = {
                'measure_name':                      measure_names[measure],
                'f_statistic':                       f_stat,
                'p_value':                           p_value,
                'p_value_gg_corrected':              p_gg,
                'sphericity_violated':               sphericity_violated,
                'effect_size':                       effect_size,
                'significant':                       p_gg < 0.05,
                'f_statistic_sex':                   f_stat_sex,
                'p_value_sex':                       p_value_sex,
                'effect_size_sex':                   np2_sex,
                'significant_sex':                   (p_value_sex < 0.05
                                                      if not np.isnan(p_value_sex) else False),
                'f_statistic_interaction':           f_stat_interaction,
                'p_value_interaction':               p_value_interaction,
                'p_value_gg_corrected_interaction':  p_gg_interaction,
                'sphericity_violated_interaction':   sphericity_violated_interaction,
                'effect_size_interaction':           np2_interaction,
                'significant_interaction':           (p_gg_interaction < 0.05
                                                      if not np.isnan(p_gg_interaction) else False),
                'group_stats':          group_stats,
                'sex_stats':            sex_stats,
                'n_animals':            n_animals,
                _MLB['n_key']:          n_levels,
                'is_repeated_measures': True,
                'is_mixed_anova':       has_sex_data,
                'anova_table':          result_table,
            }

            print(f"  Results:")
            if has_sex_data:
                sig_s   = "***" if p_value_sex < 0.05    else ""
                sig_f   = "***" if p_gg < 0.05           else ""
                sig_i   = "***" if p_gg_interaction < 0.05 else ""
                print(f"    Sex: F = {f_stat_sex:.3f}, p = {p_value_sex:.4f} {sig_s}, "
                      f"partial η² = {np2_sex:.3f}")
                print(f"    {_MLB['factor']}: F = {f_stat:.3f}, p = {p_gg:.4f} {sig_f}, "
                      f"partial η² = {effect_size:.3f}")
                if sphericity_violated:
                    print("      (Greenhouse-Geisser corrected due to sphericity violation)")
                print(f"    {_MLB['interaction_label']}: F = {f_stat_interaction:.3f}, "
                      f"p = {p_gg_interaction:.4f} {sig_i}, "
                      f"partial η² = {np2_interaction:.3f}")
                if sphericity_violated_interaction:
                    print("      (Greenhouse-Geisser corrected due to sphericity violation)")
            else:
                sig_m = "***" if p_gg < 0.05 else ""
                print(f"  F({n_levels-1}, {(n_levels-1)*(n_animals-1)}) = "
                      f"{f_stat:.3f}, p = {p_gg:.4f} {sig_m}")
                if sphericity_violated:
                    print("  (Greenhouse-Geisser corrected p-value used due to sphericity violation)")
                print(f"  Partial η² = {effect_size:.3f}")

            print(f"  Animals: {n_animals}, Complete observations across "
                  f"{n_levels} {_MLB['factor']} levels")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            anova_results[measure] = {
                'measure_name':            measure_names[measure],
                'f_statistic':             np.nan,
                'p_value':                 np.nan,
                'significant':             False,
                'f_statistic_sex':         np.nan,
                'p_value_sex':             np.nan,
                'significant_sex':         False,
                'f_statistic_interaction': np.nan,
                'p_value_interaction':     np.nan,
                'significant_interaction': False,
                'error':                   f'ANOVA failed: {str(e)}',
                'is_repeated_measures':    True,
                'is_mixed_anova':          has_sex_data,
            }

    return anova_results


def _perform_standard_anova_fallback(weekly_averages: Dict) -> Dict:
    """Fallback to standard ANOVA when pingouin is not available.

    WARNING: This does NOT account for repeated measures and treats observations
    as independent. Results may be invalid for repeated measures designs.
    """
    measures = ['licks', 'bouts', 'fecal', 'bottle_weight', 'total_weight']
    measure_names = {
        'licks': 'Total Licks', 'bouts': 'Total Bouts',
        'fecal': 'Fecal Count', 'bottle_weight': 'Bottle Weight Loss',
        'total_weight': 'Total Weight Loss',
    }

    if EXPERIMENT_MODE == 'ramp':
        # Organise by CA% concentration
        groups: Dict = {}
        for date, data in weekly_averages.items():
            ca = data['ca_percent']
            if ca not in groups:
                groups[ca] = {m: [] for m in measures}
            for m in measures:
                if m == 'bottle_weight' and date == '11/12/25':
                    groups[ca][m].extend([w for w in data['avg_bottle_weight_per_animal'] if w > 0])
                elif m == 'bottle_weight':
                    groups[ca][m].extend(data['avg_bottle_weight_per_animal'])
                else:
                    groups[ca][m].extend(data[f'avg_{m.split("_")[0]}_per_animal']
                                         if f'avg_{m.split("_")[0]}_per_animal' in data
                                         else data.get(f'avg_{m}_per_animal', []))
        sorted_keys = sorted(groups.keys())
        key_field   = 'ca_percent'
    else:
        # Organise by date (week)
        groups = {}
        for date, data in weekly_averages.items():
            groups[date] = {m: [] for m in measures}
            groups[date]['ca_percent'] = data['ca_percent']
            for m in measures:
                if m == 'bottle_weight' and date == '11/12/25':
                    groups[date][m].extend([w for w in data['avg_bottle_weight_per_animal'] if w > 0])
                elif m == 'bottle_weight':
                    groups[date][m].extend(data['avg_bottle_weight_per_animal'])
                else:
                    groups[date][m].extend(data.get(f'avg_{m}_per_animal', []))
        sorted_keys = sort_dates_chronologically(list(groups.keys()))
        key_field   = 'week'

    anova_results = {}
    for measure in measures:
        groups_data = [groups[k][measure] for k in sorted_keys if len(groups[k][measure]) > 0]
        valid_keys  = [k for k in sorted_keys if len(groups[k][measure]) > 0]

        if len(groups_data) >= 2:
            f_stat, p_value = stats.f_oneway(*groups_data)
            group_stats = []
            for i, k in enumerate(valid_keys):
                d = groups_data[i]
                entry: Dict = {'n': len(d), 'mean': np.mean(d), 'std': np.std(d, ddof=1),
                               'min': np.min(d), 'max': np.max(d)}
                if EXPERIMENT_MODE == 'ramp':
                    entry['ca_percent'] = k
                else:
                    entry['week']       = k
                    entry['ca_percent'] = groups[k]['ca_percent']
                group_stats.append(entry)

            anova_results[measure] = {
                'measure_name':       measure_names[measure],
                'f_statistic':        f_stat,
                'p_value':            p_value,
                'significant':        p_value < 0.05,
                'group_stats':        group_stats,
                'is_repeated_measures': False,
                'warning': 'Standard ANOVA used - does NOT account for repeated measures',
            }
        else:
            anova_results[measure] = {
                'measure_name': measure_names[measure],
                'f_statistic':  np.nan,
                'p_value':      np.nan,
                'significant':  False,
                'error': 'Insufficient data for ANOVA (need at least 2 groups)',
                'is_repeated_measures': False,
            }

    return anova_results


def perform_bonferroni_posthoc(anova_results: Dict, weekly_averages: Dict) -> Dict:
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
    bonferroni_results = {}

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

            # Build {group_label: {animal_id: value}} mapping
            if EXPERIMENT_MODE == 'ramp':
                sorted_weeks = sorted(weekly_averages.keys(),
                                      key=lambda d: weekly_averages[d]['ca_percent'])
            else:
                sorted_weeks = sort_dates_chronologically(list(weekly_averages.keys()))
            group_maps = {}  # group_label -> {id: value}
            for week_idx, date in enumerate(sorted_weeks):
                raw = weekly_averages[date][data_key]
                ids = weekly_averages[date].get('animal_ids', [])
                if measure == 'bottle_weight':
                    id_val = {aid: v for aid, v in zip(ids, raw) if v > 0}
                else:
                    id_val = {aid: v for aid, v in zip(ids, raw)
                              if not (isinstance(v, float) and np.isnan(v))}
                if id_val:
                    if EXPERIMENT_MODE == 'ramp':
                        label = f"{weekly_averages[date]['ca_percent']}%"
                    else:
                        label = f"Week {week_idx + 1}"
                    group_maps[label] = id_val

            week_labels = sorted(group_maps.keys())
            pairs = list(itertools.combinations(week_labels, 2))
            k = len(pairs)
            if k == 0:
                continue

            comparisons = []
            for g1, g2 in pairs:
                map1, map2 = group_maps[g1], group_maps[g2]
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

            bonferroni_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'comparisons': comparisons,
            }

        except Exception as e:
            bonferroni_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'error': f"Error performing post-hoc tests: {str(e)}"
            }

    return bonferroni_results


def display_bonferroni_results(bonferroni_results: Dict) -> str:
    """
    Display Bonferroni-corrected post-hoc test results in a formatted table.

    Parameters:
        bonferroni_results: Dictionary from perform_bonferroni_posthoc

    Returns:
        Formatted string with post-hoc results
    """
    if not bonferroni_results:
        return "\n" + "=" * 80 + f"\n{_MLB['post_no_sig']}\n" + "=" * 80 + "\n"

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BONFERRONI POST-HOC TEST RESULTS (PAIRED T-TESTS)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Paired t-tests with Bonferroni correction for {_MLB['post_header']}")
    lines.append(f"{_MLB['post_same']}; observations paired by animal ID.")
    lines.append("Alpha = 0.05 (Bonferroni-adjusted family-wise error rate)")
    lines.append("")

    for measure, results in bonferroni_results.items():
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

    total_measures = len(bonferroni_results)
    measures_with_sig_pairs = len([r for r in bonferroni_results.values()
                                   if 'comparisons' in r and
                                   any(comp['significant'] for comp in r['comparisons'])])

    lines.append("SUMMARY OF POST-HOC RESULTS:")
    lines.append("-" * 40)
    lines.append(f"Measures tested: {total_measures}")
    lines.append(f"Measures with significant pairwise differences: {measures_with_sig_pairs}")

    if measures_with_sig_pairs > 0:
        lines.append("\nSignificant pairwise differences were found, indicating specific")
        lines.append(f"{_MLB['post_differ']} significantly from each other.")
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
        lines.append(f"MIXED ANOVA RESULTS {_MLB['anova_heading']}")
        lines.append("=" * 80)
        lines.append("Between-subjects factor: Sex")
        lines.append(f"Within-subjects factor: {_MLB['factor']} (repeated measures)")
        lines.append("Subject tracking: Animal ID")
    elif is_rm:
        lines.append(f"ONE-WAY REPEATED MEASURES ANOVA RESULTS {_MLB['anova_heading']}")
        lines.append("=" * 80)
        lines.append(f"Within-subjects factor: {_MLB['factor']}")
        lines.append("Subject tracking: Animal ID")
    else:
        lines.append(f"ONE-WAY ANOVA RESULTS {_MLB['anova_heading']}")
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
            lines.append(f"Design: Mixed ANOVA (Sex between-subjects × {_MLB['factor']} within-subjects)")
            if 'n_animals' in results and _MLB['n_key'] in results:
                lines.append(f"Animals: {results['n_animals']}, {_MLB['factor']} levels: {results[_MLB['n_key']]}")
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
            
            # Factor main effect (within-subjects, with sphericity)
            lines.append(f"{_MLB['main_effect']} (within-subjects):")
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
            
            # Interaction
            lines.append(f"{_MLB['interaction']}:")
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
            if 'n_animals' in results and _MLB['n_key'] in results:
                lines.append(f"Animals: {results['n_animals']}, {_MLB['factor']} levels: {results[_MLB['n_key']]}")
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
        lines.append(_MLB['stats_header'])
        
        # Check if we have CA% information (old format) or week_number (new format)
        has_ca_percent = 'group_stats' in results and len(results['group_stats']) > 0 and 'ca_percent' in results['group_stats'][0]
        
        if has_ca_percent:
            header = f"{'CA%':<6} {'N':<4} {'Mean':<10} {'Median':<10} {'Std':<10} {'SEM':<10}"
        else:
            header = f"{'Week':<12} {'Week#':<7} {'N':<4} {'Mean':<10} {'Median':<10} {'Std':<10} {'SEM':<10}"
        
        lines.append(header)
        lines.append("-" * 72)
        
        for group_stat in results['group_stats']:
            if has_ca_percent:
                row = (f"{group_stat['ca_percent']:<6} "
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
            header2 = f"{'CA%':<6} {'95% CI Lower':<14} {'95% CI Upper':<14} {'Q25':<10} {'Q75':<10} {'Min':<10} {'Max':<10}"
        else:
            header2 = f"{'Week':<12} {'Week#':<7} {'95% CI Lower':<14} {'95% CI Upper':<14} {'Q25':<10} {'Q75':<10}"
        lines.append(header2)
        lines.append("-" * 80)
        
        for group_stat in results['group_stats']:
            if has_ca_percent:
                row2 = (f"{group_stat['ca_percent']:<6} "
                        f"{group_stat.get('ci_lower', np.nan):<14.2f} "
                        f"{group_stat.get('ci_upper', np.nan):<14.2f} "
                        f"{group_stat.get('q25', np.nan):<10.2f} "
                        f"{group_stat.get('q75', np.nan):<10.2f} "
                        f"{group_stat['min']:<10.2f} "
                        f"{group_stat['max']:<10.2f}")
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
            lines.append(f"{_MLB['main_sig_in']}: {', '.join(significant_measures_week)}")
        else:
            lines.append(_MLB['no_sig'])

        if significant_measures_interaction:
            lines.append(f"{_MLB['interaction']} interaction significant in: {', '.join(significant_measures_interaction)}")
            lines.append(f"  ({_MLB['factor']} effect differs by sex - consider simple effects analysis)")
        else:
            lines.append(f"{_MLB['interaction']}: Not significant ({_MLB['factor']} effect similar across sexes)")
    else:
        if significant_measures_week:
            lines.append(f"Significant differences found in: {', '.join(significant_measures_week)}")
            lines.append(f"\nThese measures show statistically {_MLB['sig_across']}.")
            lines.append("Consider post-hoc tests (e.g., Bonferroni-corrected pairwise) for pairwise comparisons.")
        else:
            lines.append(f"No {_MLB['sig_across']} in any measure.")
    
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
                
                # Column names vary by pingouin version; support both
                p_col = 'p-unc' if 'p-unc' in rm_aov.columns else 'p_unc'
                df_col = 'ddof1' if 'ddof1' in rm_aov.columns else 'DF'
                df2_col = 'ddof2' if 'ddof2' in rm_aov.columns else None
                simple_effects.append({
                    between: between_level,
                    'F': rm_aov.loc[0, 'F'],
                    'df1': rm_aov.loc[0, df_col],
                    'df2': rm_aov.loc[1, df_col] if df2_col is None else rm_aov.loc[0, df2_col],
                    'p-unc': rm_aov.loc[0, p_col],
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
                    f"Std: {np.std(all_avg_licks, ddof=1):.1f}, "
                    f"Range: {np.min(all_avg_licks):.1f} - {np.max(all_avg_licks):.1f}")
        lines.append(f"Bouts - Mean: {np.mean(all_avg_bouts):.1f}, "
                    f"Std: {np.std(all_avg_bouts, ddof=1):.1f}, "
                    f"Range: {np.min(all_avg_bouts):.1f} - {np.max(all_avg_bouts):.1f}")
        lines.append(f"First 5-min Lick % - Mean: {np.mean(all_avg_first_5min_lick):.1f}%, "
                    f"Std: {np.std(all_avg_first_5min_lick, ddof=1):.1f}%, "
                    f"Range: {np.min(all_avg_first_5min_lick):.1f}% - {np.max(all_avg_first_5min_lick):.1f}%")
        lines.append(f"First 5-min Bout % - Mean: {np.mean(all_avg_first_5min_bout):.1f}%, "
                    f"Std: {np.std(all_avg_first_5min_bout, ddof=1):.1f}%, "
                    f"Range: {np.min(all_avg_first_5min_bout):.1f}% - {np.max(all_avg_first_5min_bout):.1f}%")
        lines.append(f"Fecal - Mean: {np.mean(all_avg_fecal):.1f}, "
                    f"Std: {np.std(all_avg_fecal, ddof=1):.1f}, "
                    f"Range: {np.min(all_avg_fecal):.1f} - {np.max(all_avg_fecal):.1f}")
        lines.append(f"Bottle Weight - Mean: {np.mean(all_avg_bottle):.1f}, "
                    f"Std: {np.std(all_avg_bottle, ddof=1):.1f}, "
                    f"Range: {np.min(all_avg_bottle):.1f} - {np.max(all_avg_bottle):.1f}")
        lines.append(f"Total Weight - Mean: {np.mean(all_avg_total):.1f}, "
                    f"Std: {np.std(all_avg_total, ddof=1):.1f}, "
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
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    x_pos = np.arange(len(dates))
    
    # Plot 1: Average Licks
    ax1.errorbar(x_pos, avg_licks, yerr=std_licks, 
                marker='o', capsize=5, 
                color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue')
    ax1.set_xlabel('Week', weight='bold')
    ax1.set_ylabel('Average Licks per Animal', weight='bold')
    ax1.set_title('Average Licks Across Weeks (±SEM)', weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    ax1.tick_params(direction='in', which='both', length=5)
    
    # Plot 2: Average Bouts
    ax2.errorbar(x_pos, avg_bouts, yerr=std_bouts,
                marker='s', capsize=5,
                color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    ax2.set_xlabel('Week', weight='bold')
    ax2.set_ylabel('Average Bouts per Animal', weight='bold')
    ax2.set_title('Average Lick Bouts Across Weeks (±SEM)', weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax2.set_ylim(bottom=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(False)
    ax2.tick_params(direction='in', which='both', length=5)
    
    fig1.suptitle('Behavioral Metrics Across Weeks', weight='bold', y=0.96)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig1.subplots_adjust(hspace=0.4)
    
    # ===== FIGURE 2: PHYSIOLOGICAL METRICS =====
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(5, 9))
    
    # Plot 3: Average Fecal Count
    ax3.errorbar(x_pos, avg_fecal, yerr=std_fecal,
                marker='^', capsize=5,
                color='saddlebrown', markerfacecolor='tan', markeredgecolor='saddlebrown')
    ax3.set_xlabel('Week', weight='bold')
    ax3.set_ylabel('Average Fecal Count per Animal', weight='bold')
    ax3.set_title('Average Fecal Count Across Weeks (±SEM)', weight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax3.set_ylim(bottom=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(False)
    ax3.tick_params(direction='in', which='both', length=5)
    
    # Plot 4: Average Bottle Weight Loss
    ax4.errorbar(x_pos, avg_bottle_weight, yerr=std_bottle_weight,
                marker='D', capsize=5,
                color='purple', markerfacecolor='plum', markeredgecolor='purple')
    ax4.set_xlabel('Week', weight='bold')
    ax4.set_ylabel('Average Bottle Weight Loss per Animal (g)', weight='bold')
    ax4.set_title('Average Bottle Weight Loss Across Weeks (±SEM)', weight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax4.set_ylim(bottom=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(False)
    ax4.tick_params(direction='in', which='both', length=5)
    
    # Plot 5: Average Total Weight Loss
    ax5.errorbar(x_pos, avg_total_weight, yerr=std_total_weight,
                marker='v', capsize=5,
                color='darkorange', markerfacecolor='orange', markeredgecolor='darkorange')
    ax5.set_xlabel('Week', weight='bold')
    ax5.set_ylabel('Average Total Weight Loss per Animal (g)', weight='bold')
    ax5.set_title('Average Total Weight Loss Across Weeks (±SEM)', weight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"{i+1}" for i in range(len(dates))])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(False)
    ax5.tick_params(direction='in', which='both', length=5)
    
    fig2.suptitle('Physiological Metrics Across Weeks', weight='bold', y=0.97)
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

    # Sort by CA% (ramp) or date (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    within_col = _MLB['factor_col']  # 'CA_Percent' or 'Week'
    
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

            group_val = (weekly_averages[date]['ca_percent']
                         if EXPERIMENT_MODE == 'ramp' else week_idx)
            long_data.append({
                'Animal': animal_id,
                within_col: group_val,
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
    print(f"PERFORMING ONE-WAY REPEATED MEASURES ANOVA: {_MLB['fl_factor']} Effect for Front-Loading Measures")
    print("="*80)
    
    for measure in measures:
        measure_name = measure_names[measure]
        print(f"\nAnalyzing: {measure_name}")
        print("-" * 40)
        
        # Remove NaN values for this measure
        df_measure = df_long[['Animal', within_col, measure]].dropna()

        if len(df_measure) == 0:
            print(f"  ERROR: No valid data for {measure_name}")
            anova_results[measure] = {
                'measure_name': measure_name,
                'error': 'No valid data available'
            }
            continue

        print(f"  Valid data points: {len(df_measure)}")
        print(f"  Animals: {df_measure['Animal'].nunique()}")
        print(f"  {_MLB['factor']} levels: {sorted(df_measure[within_col].unique())}")

        try:
            # Repeated measures ANOVA
            aov = pg.rm_anova(
                data=df_measure,
                dv=measure,
                within=within_col,
                subject='Animal',
                detailed=True
            )

            print("\n  Repeated Measures ANOVA Results:")
            print(aov)

            week_effect = aov[aov['Source'] == within_col].iloc[0]
            
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
    
    lines.append(f"Repeated Measures ANOVA: {_MLB['fl_factor']}")
    lines.append("")

    for measure, results in anova_results.items():
        lines.append(f"MEASURE: {results['measure_name']}")
        lines.append("-" * 80)

        if 'error' in results:
            lines.append(f"ERROR: {results['error']}")
            lines.append("")
            continue

        lines.append("")
        lines.append(_MLB['fl_effect'])
        if results.get('df1') is not None and results.get('df2') is not None:
            lines.append(f"  F({results['df1']:.0f}, {results['df2']:.0f}) = {results['f_statistic']:.4f}")
        else:
            lines.append(f"  F-statistic: {results['f_statistic']:.4f}")
        lines.append(f"  p-value: {results['p_value']:.6f}")
        if results['significant']:
            lines.append(f"  *** {_MLB['fl_sig']} ***")
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
    significant_factor = [r['measure_name'] for r in anova_results.values() if r.get('significant', False)]
    lines.append(f"Significant {_MLB['factor']} effects: {', '.join(significant_factor) if significant_factor else 'None'}")
    
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


def perform_frontloading_bonferroni_posthoc(anova_results: Dict, weekly_averages: Dict) -> Dict:
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
    bonferroni_results = {}

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

            # Build {group_label: {animal_id: value}} mapping
            if EXPERIMENT_MODE == 'ramp':
                sorted_dates = sorted(weekly_averages.keys(),
                                      key=lambda d: weekly_averages[d]['ca_percent'])
            else:
                sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))
            group_maps = {}
            for week_idx, date in enumerate(sorted_dates, 1):
                raw = weekly_averages[date].get(data_key, [])
                ids = weekly_averages[date].get('animal_ids', [])
                id_val = {aid: v for aid, v in zip(ids, raw)
                          if not (isinstance(v, float) and np.isnan(v))}
                if id_val:
                    if EXPERIMENT_MODE == 'ramp':
                        label = f"{weekly_averages[date]['ca_percent']}%"
                    else:
                        label = f"Week {week_idx}"
                    group_maps[label] = id_val

            week_labels = sorted(group_maps.keys())
            pairs = list(itertools.combinations(week_labels, 2))
            k = len(pairs)
            if k == 0:
                continue

            comparisons = []
            for g1, g2 in pairs:
                map1, map2 = group_maps[g1], group_maps[g2]
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

            bonferroni_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'comparisons': comparisons,
            }

            print(f"\nBonferroni post-hoc completed for: {anova_data['measure_name']}")
            print(f"  Total comparisons: {len(comparisons)}")
            print(f"  Significant pairs: {sum(1 for c in comparisons if c['significant'])}")

        except Exception as e:
            bonferroni_results[measure] = {
                'measure_name': anova_data['measure_name'],
                'error': f"Error performing post-hoc tests: {str(e)}"
            }

    return bonferroni_results


def display_frontloading_bonferroni_results(bonferroni_results: Dict) -> str:
    """
    Display Bonferroni-corrected post-hoc test results for front-loading measures.

    Parameters:
        bonferroni_results: Dictionary from perform_frontloading_bonferroni_posthoc

    Returns:
        Formatted string with post-hoc results
    """
    if not bonferroni_results:
        return f"\nNo post-hoc results to display (no {_MLB['fl_not_sig']})\n"

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BONFERRONI POST-HOC TEST RESULTS - FRONT-LOADING MEASURES")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Paired t-tests with Bonferroni correction for {_MLB['post_header']}")
    lines.append(f"{_MLB['post_same']}; observations paired by animal ID.")
    lines.append("Alpha = 0.05 (Bonferroni-adjusted family-wise error rate)")
    lines.append("")

    for measure, results in bonferroni_results.items():
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

    total_measures = len(bonferroni_results)
    measures_with_sig_pairs = len([r for r in bonferroni_results.values()
                                   if 'comparisons' in r and
                                   any(comp['significant'] for comp in r['comparisons'])])
    lines.append("SUMMARY OF POST-HOC RESULTS:")
    lines.append("-" * 40)
    lines.append(f"Measures tested: {total_measures}")
    lines.append(f"Measures with significant pairwise differences: {measures_with_sig_pairs}")

    if measures_with_sig_pairs > 0:
        lines.append("\nSignificant pairwise differences were found, indicating specific")
        lines.append(f"{_MLB['post_differ']} significantly from each other.")
    else:
        lines.append("\nNo significant pairwise differences found despite significant omnibus test.")

    lines.append("\n" + "=" * 80)
    lines.append("")
    formatted_output = "\n".join(lines)
    print(formatted_output)
    return formatted_output


def perform_frontloading_mixed_anova(weekly_averages: Dict) -> Dict:
    """
    Perform a TWO-WAY MIXED ANOVA (Sex × Week/CA%) for each front-loading measure:
    - % of licks in first 5 minutes
    - % of bouts in first 5 minutes
    - Time to 50% of total licks (minutes)

    Between-subjects factor: Sex
    Within-subjects factor:  Week (nonramp) or CA_Percent (ramp)

    If sex data are unavailable or only one sex is present, the function
    logs a warning and returns an empty dict (the existing one-way RM-ANOVA
    already covers that case).

    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages

    Returns:
        Dictionary of mixed ANOVA results keyed by measure name, or {} if
        sex data are insufficient.
    """
    if not HAS_PINGOUIN:
        print("\nWARNING: Mixed ANOVA requires pingouin. Skipping frontloading mixed ANOVA.")
        return {}

    within_col = _MLB['factor_col']

    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    # Build long-format data (same as one-way version but include Sex)
    long_data = []
    animal_sex_map: Dict[str, str] = {}

    for week_idx, date in enumerate(sorted_dates, 1):
        data = weekly_averages[date]
        animal_ids   = data.get('animal_ids', [])
        animal_sexes = data.get('animal_sexes', [])

        if not animal_ids:
            animal_ids = [f"Animal_{i+1}" for i in range(
                len(data.get('first_5min_lick_pcts_per_animal', [])))]
        if not animal_sexes or len(animal_sexes) != len(animal_ids):
            animal_sexes = ['Unknown'] * len(animal_ids)

        for i, animal_id in enumerate(animal_ids):
            if animal_id not in animal_sex_map:
                animal_sex_map[animal_id] = animal_sexes[i]

        first_5min_lick_pcts = data.get('first_5min_lick_pcts_per_animal', [])
        first_5min_bout_pcts = data.get('first_5min_bout_pcts_per_animal', [])
        time_to_50pct        = data.get('time_to_50pct_licks_per_animal', [])
        group_val = (weekly_averages[date]['ca_percent']
                     if EXPERIMENT_MODE == 'ramp' else week_idx)

        for i, animal_id in enumerate(animal_ids):
            long_data.append({
                'Animal':    animal_id,
                'Sex':       animal_sex_map[animal_id],
                within_col:  group_val,
                'first_5min_lick_pct': first_5min_lick_pcts[i] if i < len(first_5min_lick_pcts) else np.nan,
                'first_5min_bout_pct': first_5min_bout_pcts[i] if i < len(first_5min_bout_pcts) else np.nan,
                'time_to_50pct':       time_to_50pct[i]        if i < len(time_to_50pct)        else np.nan,
            })

    df_long = pd.DataFrame(long_data)

    has_sex_data = (
        'Sex' in df_long.columns
        and df_long['Sex'].nunique() > 1
        and 'Unknown' not in df_long['Sex'].values
    )

    if not has_sex_data:
        print("\n" + "=" * 80)
        print("WARNING: Sex data not found or only one sex present — "
              "skipping frontloading mixed ANOVA.")
        print("=" * 80 + "\n")
        return {}

    measures = ['first_5min_lick_pct', 'first_5min_bout_pct', 'time_to_50pct']
    measure_names = {
        'first_5min_lick_pct': '% Licks in First 5 Minutes',
        'first_5min_bout_pct': '% Bouts in First 5 Minutes',
        'time_to_50pct':       'Time to 50% of Total Licks (min)',
    }

    print("\n" + "=" * 80)
    print(f"PERFORMING TWO-WAY MIXED ANOVA: Sex × {_MLB['factor']} — Front-Loading Measures")
    print("Between-subjects factor: Sex")
    print(f"Within-subjects factor:  {_MLB['factor']} (repeated measures)")
    print("=" * 80)

    mixed_results: Dict = {}

    for measure in measures:
        measure_name = measure_names[measure]
        print(f"\nAnalyzing: {measure_name}")
        print("-" * 40)

        df_measure = df_long[['Animal', 'Sex', within_col, measure]].dropna()

        if len(df_measure) == 0:
            print(f"  ERROR: No valid data for {measure_name}")
            mixed_results[measure] = {'measure_name': measure_name,
                                      'error': 'No valid data available'}
            continue

        # Keep only animals with complete observations across all within-factor levels
        n_levels = df_measure[within_col].nunique()
        animals_per_level = df_measure.groupby('Animal')[within_col].nunique()
        complete_animals  = animals_per_level[animals_per_level == n_levels]

        if len(complete_animals) < len(animals_per_level):
            missing = len(animals_per_level) - len(complete_animals)
            print(f"  WARNING: {missing} animal(s) missing data at some levels — excluded.")
            df_measure = df_measure[df_measure['Animal'].isin(complete_animals.index)]

        n_animals = df_measure['Animal'].nunique()
        print(f"  Valid animals: {n_animals}, {_MLB['factor']} levels: {n_levels}")

        if n_animals < 2 or n_levels < 2:
            print(f"  ERROR: Insufficient data for mixed ANOVA")
            mixed_results[measure] = {'measure_name': measure_name,
                                      'error': 'Insufficient data'}
            continue

        try:
            result_table = pg.mixed_anova(
                dv=measure,
                within=within_col,
                between='Sex',
                subject='Animal',
                data=df_measure,
            )

            sex_row         = result_table[result_table['Source'] == 'Sex']
            factor_row      = result_table[result_table['Source'] == within_col]
            interaction_row = result_table[result_table['Source'] == 'Interaction']

            # Sex (between-subjects)
            f_sex   = sex_row['F'].values[0]     if len(sex_row) > 0     else np.nan
            p_sex   = sex_row['p-unc'].values[0] if len(sex_row) > 0     else np.nan
            np2_sex = (sex_row['np2'].values[0]
                       if len(sex_row) > 0 and 'np2' in sex_row.columns else np.nan)

            # Within-factor (with optional GG correction)
            f_factor    = factor_row['F'].values[0]     if len(factor_row) > 0     else np.nan
            p_factor    = factor_row['p-unc'].values[0] if len(factor_row) > 0     else np.nan
            np2_factor  = (factor_row['np2'].values[0]
                           if len(factor_row) > 0 and 'np2' in factor_row.columns else np.nan)
            if 'p-GG-corr' in factor_row.columns and len(factor_row) > 0:
                p_gg_factor         = factor_row['p-GG-corr'].values[0]
                sphericity_violated = (p_gg_factor != p_factor)
            else:
                p_gg_factor         = p_factor
                sphericity_violated = False

            # Interaction
            f_inter   = (interaction_row['F'].values[0]
                         if len(interaction_row) > 0 else np.nan)
            p_inter   = (interaction_row['p-unc'].values[0]
                         if len(interaction_row) > 0 else np.nan)
            np2_inter = (interaction_row['np2'].values[0]
                         if len(interaction_row) > 0
                         and 'np2' in interaction_row.columns else np.nan)
            if 'p-GG-corr' in interaction_row.columns and len(interaction_row) > 0:
                p_gg_inter              = interaction_row['p-GG-corr'].values[0]
                sphericity_violated_int = (p_gg_inter != p_inter)
            else:
                p_gg_inter              = p_inter
                sphericity_violated_int = False

            mixed_results[measure] = {
                'measure_name':                      measure_name,
                'anova_table':                       result_table,
                'df_long':                           df_measure,
                # Within-factor
                'f_statistic':                       f_factor,
                'p_value':                           p_factor,
                'p_value_gg_corrected':              p_gg_factor,
                'sphericity_violated':               sphericity_violated,
                'effect_size':                       np2_factor,
                'significant':                       p_gg_factor < 0.05,
                # Sex (between)
                'f_statistic_sex':                   f_sex,
                'p_value_sex':                       p_sex,
                'effect_size_sex':                   np2_sex,
                'significant_sex':                   (p_sex < 0.05 if not np.isnan(p_sex) else False),
                # Interaction
                'f_statistic_interaction':           f_inter,
                'p_value_interaction':               p_inter,
                'p_value_gg_corrected_interaction':  p_gg_inter,
                'sphericity_violated_interaction':   sphericity_violated_int,
                'effect_size_interaction':           np2_inter,
                'significant_interaction':           (p_gg_inter < 0.05
                                                      if not np.isnan(p_gg_inter) else False),
                'n_animals':    n_animals,
                'n_levels':     n_levels,
                'is_mixed_anova': True,
            }

            sig_s = "***" if p_sex < 0.05            else ""
            sig_f = "***" if p_gg_factor < 0.05      else ""
            sig_i = "***" if p_gg_inter < 0.05       else ""
            print(f"  Sex: F = {f_sex:.3f}, p = {p_sex:.4f} {sig_s}, partial η² = {np2_sex:.3f}")
            print(f"  {_MLB['factor']}: F = {f_factor:.3f}, p = {p_gg_factor:.4f} {sig_f}, "
                  f"partial η² = {np2_factor:.3f}")
            if sphericity_violated:
                print("    (Greenhouse-Geisser corrected)")
            print(f"  {_MLB['interaction_label']}: F = {f_inter:.3f}, "
                  f"p = {p_gg_inter:.4f} {sig_i}, partial η² = {np2_inter:.3f}")
            if sphericity_violated_int:
                print("    (Greenhouse-Geisser corrected)")

        except Exception as e:
            print(f"  ERROR: {e}")
            mixed_results[measure] = {
                'measure_name': measure_name,
                'error': str(e),
            }

    return mixed_results


def display_frontloading_mixed_anova_results(mixed_results: Dict) -> str:
    """Format two-way mixed ANOVA (Sex × Week/CA%) results for front-loading measures.

    Parameters:
        mixed_results: Dictionary from perform_frontloading_mixed_anova

    Returns:
        Formatted string with results
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("FRONT-LOADING ANALYSIS: TWO-WAY MIXED ANOVA (Sex × "
                 f"{_MLB['factor']}) RESULTS")
    lines.append("=" * 80)
    lines.append("")

    if not mixed_results:
        lines.append("No mixed ANOVA results available (insufficient sex data or "
                     "pingouin not installed).")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append(f"Between-subjects factor: Sex")
    lines.append(f"Within-subjects factor:  {_MLB['factor_display']}")
    lines.append("")

    for measure, results in mixed_results.items():
        lines.append(f"MEASURE: {results['measure_name']}")
        lines.append("-" * 80)

        if 'error' in results:
            lines.append(f"  ERROR: {results['error']}")
            lines.append("")
            continue

        n_str = (f"  N = {results['n_animals']} animals × "
                 f"{results['n_levels']} {_MLB['factor']} levels")
        lines.append(n_str)
        lines.append("")

        # Sex main effect
        lines.append(f"  {_MLB['main_effect'].replace('WEEK', 'FACTOR').split()[0]} SEX MAIN EFFECT:")
        lines.append(f"    F = {results['f_statistic_sex']:.4f}, "
                     f"p = {results['p_value_sex']:.6f}, "
                     f"partial η² = {results['effect_size_sex']:.4f}")
        if results['significant_sex']:
            lines.append("    *** SIGNIFICANT ***")
        else:
            lines.append("    Not significant (p >= 0.05)")
        lines.append("")

        # Within-factor main effect
        lines.append(f"  {_MLB['fl_effect'].rstrip(':').upper()} MAIN EFFECT:")
        lines.append(f"    F = {results['f_statistic']:.4f}, "
                     f"p = {results['p_value_gg_corrected']:.6f}, "
                     f"partial η² = {results['effect_size']:.4f}")
        if results.get('sphericity_violated'):
            lines.append(f"    (Greenhouse-Geisser corrected; uncorrected p = "
                         f"{results['p_value']:.6f})")
        if results['significant']:
            lines.append(f"    *** {_MLB['fl_sig']} ***")
        else:
            lines.append("    Not significant (p >= 0.05)")
        lines.append("")

        # Interaction
        lines.append(f"  {_MLB['interaction']} INTERACTION:")
        lines.append(f"    F = {results['f_statistic_interaction']:.4f}, "
                     f"p = {results['p_value_gg_corrected_interaction']:.6f}, "
                     f"partial η² = {results['effect_size_interaction']:.4f}")
        if results.get('sphericity_violated_interaction'):
            lines.append(f"    (Greenhouse-Geisser corrected; uncorrected p = "
                         f"{results['p_value_interaction']:.6f})")
        if results['significant_interaction']:
            lines.append("    *** SIGNIFICANT INTERACTION ***")
        else:
            lines.append("    Not significant (p >= 0.05)")
        lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY:")
    sig_within = [r['measure_name'] for r in mixed_results.values()
                  if r.get('significant', False) and 'error' not in r]
    sig_sex    = [r['measure_name'] for r in mixed_results.values()
                  if r.get('significant_sex', False) and 'error' not in r]
    sig_inter  = [r['measure_name'] for r in mixed_results.values()
                  if r.get('significant_interaction', False) and 'error' not in r]
    lines.append(f"  Significant {_MLB['factor']} main effects: "
                 f"{', '.join(sig_within) if sig_within else 'None'}")
    lines.append(f"  Significant Sex main effects:             "
                 f"{', '.join(sig_sex) if sig_sex else 'None'}")
    lines.append(f"  Significant {_MLB['interaction_label']} interactions: "
                 f"{', '.join(sig_inter) if sig_inter else 'None'}")
    lines.append("=" * 80)
    lines.append("")

    formatted_output = "\n".join(lines)
    print(formatted_output)
    return formatted_output


def perform_frontloading_mixed_posthoc(mixed_results: Dict) -> Dict:
    """
    Perform post-hoc tests for significant effects in the frontloading mixed ANOVA.

    For each measure with at least one significant effect, calls
    perform_mixed_anova_posthoc() which provides:
      1. Within-subjects pairwise (Week/CA% comparisons, Bonferroni-corrected)
      2. Between-subjects pairwise (Sex comparison)
      3. Simple effects (within-factor effect at each Sex level)

    Only runs when either the within-factor, Sex, or interaction effect is
    significant (p < 0.05).

    Parameters:
        mixed_results: Dictionary from perform_frontloading_mixed_anova

    Returns:
        Dictionary keyed by measure with posthoc result sub-dicts
    """
    if not mixed_results:
        return {}

    within_col = _MLB['factor_col']
    posthoc_all: Dict = {}

    for measure, results in mixed_results.items():
        if 'error' in results:
            continue

        any_sig = (results.get('significant', False)
                   or results.get('significant_sex', False)
                   or results.get('significant_interaction', False))

        if not any_sig:
            print(f"\n  {results['measure_name']}: No significant effects — "
                  "post-hoc tests not performed.")
            continue

        df_measure = results.get('df_long')
        if df_measure is None or df_measure.empty:
            continue

        print(f"\nPost-hoc tests for: {results['measure_name']}")
        try:
            ph = perform_mixed_anova_posthoc(
                data=df_measure,
                dv=measure,
                within=within_col,
                between='Sex',
                subject='Animal',
                alpha=0.05,
                correction='bonferroni',
            )
            posthoc_all[measure] = {
                'measure_name': results['measure_name'],
                'posthoc':      ph,
            }
        except Exception as e:
            print(f"  ERROR in post-hoc for {results['measure_name']}: {e}")
            posthoc_all[measure] = {
                'measure_name': results['measure_name'],
                'error':        str(e),
            }

    return posthoc_all


def display_frontloading_mixed_posthoc_results(posthoc_all: Dict) -> str:
    """Format mixed-ANOVA post-hoc results for all significant frontloading measures.

    Parameters:
        posthoc_all: Dictionary from perform_frontloading_mixed_posthoc

    Returns:
        Formatted string
    """
    if not posthoc_all:
        return (f"\nNo mixed-ANOVA post-hoc results to display "
                f"(no significant effects found).\n")

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("POST-HOC TESTS — FRONTLOADING MIXED ANOVA (Sex × "
                 f"{_MLB['factor']})")
    lines.append("Correction: Bonferroni")
    lines.append("=" * 80)

    all_output_parts: list = []
    for measure, entry in posthoc_all.items():
        if 'error' in entry:
            lines.append(f"\n{entry['measure_name']}: ERROR — {entry['error']}")
            continue
        # display_posthoc_results prints + returns; collect the returned string
        ph_str = display_posthoc_results(entry['posthoc'], entry['measure_name'])
        all_output_parts.append(ph_str)

    formatted_output = "\n".join(lines) + "\n" + "\n".join(all_output_parts)
    return formatted_output


def save_frontloading_analysis_to_file(weekly_averages: Dict, anova_output: str, save_path: Path,
                                       bonferroni_output: str = "",
                                       mixed_anova_output: str = "",
                                       mixed_posthoc_output: str = "") -> Path:
    """Save front-loading behavior analysis to a separate report file.
    
    This report focuses specifically on front-loading metrics:
    - % of licks in first 5 minutes
    - % of bouts in first 5 minutes
    - Time to 50% of total licks (minutes)
    
    Also includes one-way RM-ANOVA, Bonferroni post-hoc, two-way mixed ANOVA,
    and mixed ANOVA post-hoc results for these metrics.
    
    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        anova_output: Formatted one-way RM-ANOVA results string
        save_path: Path where to save the text file
        bonferroni_output: Formatted Bonferroni post-hoc results string (optional)
        mixed_anova_output: Formatted two-way mixed ANOVA results string (optional)
        mixed_posthoc_output: Formatted mixed ANOVA post-hoc results string (optional)
        
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
        
        # Add Bonferroni post-hoc results
        if bonferroni_output:
            f.write("\n\n")
            f.write(bonferroni_output)

        # Add two-way mixed ANOVA results
        if mixed_anova_output:
            f.write("\n\n")
            f.write(mixed_anova_output)

        # Add mixed ANOVA post-hoc results
        if mixed_posthoc_output:
            f.write("\n\n")
            f.write(mixed_posthoc_output)
        
        f.write("=" * 80 + "\n")
        f.write("END OF FRONT-LOADING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
    
    return save_path


def generate_lick_normality_report(weekly_averages: Dict, save_path: Optional[Path] = None) -> str:
    """Shapiro-Wilk normality and Levene's equal-variance tests on all lick measures.

    Builds a per-animal long-format DataFrame from weekly_averages and tests each
    measure stratified by Week (nonramp) or CA_Percent (ramp).

    Parameters
    ----------
    weekly_averages : output of compute_weekly_averages()
    save_path       : if provided, write the report text to this file

    Returns
    -------
    Report as a string
    """
    sorted_keys = sort_dates_chronologically(list(weekly_averages.keys()))
    rows = []
    for week_idx, date in enumerate(sorted_keys):
        data = weekly_averages[date]
        n = len(data.get('avg_licks_per_animal', []))
        animal_ids   = data.get('animal_ids',   [f'Animal_{i+1}' for i in range(n)])
        animal_sexes = data.get('animal_sexes', ['Unknown'] * n)
        ca_pct       = data.get('ca_percent', 0.0)
        licks        = np.asarray(data.get('avg_licks_per_animal',         np.zeros(n)), dtype=float)
        bouts        = np.asarray(data.get('avg_bouts_per_animal',         np.zeros(n)), dtype=float)
        fecal        = np.asarray(data.get('avg_fecal_per_animal',         np.zeros(n)), dtype=float)
        bottle_wt    = np.asarray(data.get('avg_bottle_weight_per_animal', np.zeros(n)), dtype=float)
        total_wt     = np.asarray(data.get('avg_total_weight_per_animal',  np.zeros(n)), dtype=float)
        fl_lick_pct  = np.asarray(data.get('first_5min_lick_pcts_per_animal', np.zeros(n)), dtype=float)
        fl_bout_pct  = np.asarray(data.get('first_5min_bout_pcts_per_animal', np.zeros(n)), dtype=float)
        t50          = np.asarray(data.get('time_to_50pct_licks_per_animal',  np.full(n, np.nan)), dtype=float)

        for i in range(len(animal_ids)):
            sex = animal_sexes[i] if i < len(animal_sexes) else 'Unknown'
            bw  = float(bottle_wt[i]) if bottle_wt[i] > 0 else np.nan
            rows.append({
                'Animal':          animal_ids[i],
                'Sex':             sex,
                'CA_Percent':      ca_pct,
                'Week':            week_idx + 1,
                'Total_Licks':     float(licks[i]),
                'Total_Bouts':     float(bouts[i]),
                'Fecal_Count':     float(fecal[i]),
                'Bottle_Weight':   bw,
                'Total_Weight':    float(total_wt[i]),
                'First5min_Lick%': float(fl_lick_pct[i]),
                'First5min_Bout%': float(fl_bout_pct[i]),
                'Time_to_50pct':   float(t50[i]),
            })

    if not rows:
        return '  [WARNING] No per-animal data found in weekly_averages.\n'

    df_long = pd.DataFrame(rows)

    MEASURES = [
        ('Total_Licks',     'Total Licks'),
        ('Total_Bouts',     'Total Bouts'),
        ('Fecal_Count',     'Fecal Count'),
        ('Bottle_Weight',   'Bottle Weight Loss (g)'),
        ('Total_Weight',    'Total Weight Loss (g)'),
        ('First5min_Lick%', '% Licks in First 5 min'),
        ('First5min_Bout%', '% Bouts in First 5 min'),
        ('Time_to_50pct',   'Time to 50% Licks (min)'),
    ]
    available = [(col, lbl) for col, lbl in MEASURES if col in df_long.columns]

    within_col   = 'CA_Percent' if EXPERIMENT_MODE == 'ramp' else 'Week'
    within_label = 'CA%'        if EXPERIMENT_MODE == 'ramp' else 'Week'

    def _sw_row(values: pd.Series, label: str, alpha: float = 0.05):
        """Shapiro-Wilk (or D'Agostino-Pearson for n>5000). Returns (lines, passed)."""
        values = pd.Series(values).dropna()
        n = len(values)
        if n < 3:
            return [f'  {label}: n={n} \u2014 too few observations, skipping.'], True
        if n > 5000:
            stat, p = stats.normaltest(values)
            test_name = "D'Agostino-Pearson k\u00b2"
        else:
            stat, p = stats.shapiro(values)
            test_name = 'Shapiro-Wilk W'
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        passed = p >= alpha
        conclusion = 'NORMAL (p \u2265 0.05)' if passed else 'NON-NORMAL (p < 0.05)'
        return [
            f'  {label} (n={n}):',
            f'    {test_name} = {stat:.4f},  p = {p:.4f}  [{sig}]  \u2192  {conclusion}',
        ], passed

    def _rm_residuals(df: pd.DataFrame, col: str) -> pd.Series:
        """Compute within-subject residuals: y_ij - animal_mean_i - condition_mean_j + grand_mean."""
        sub = df[['Animal', within_col, col]].dropna().copy()
        grand = sub[col].mean()
        sub['_animal_mean'] = sub.groupby('Animal')[col].transform('mean')
        sub['_cond_mean']   = sub.groupby(within_col)[col].transform('mean')
        return sub[col] - sub['_animal_mean'] - sub['_cond_mean'] + grand

    lines: List[str] = [
        '=' * 80,
        "LICK MEASURES \u2014 SHAPIRO-WILK NORMALITY (RESIDUALS) & MAUCHLY'S SPHERICITY TEST",
        '=' * 80,
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Mode      : {EXPERIMENT_MODE.upper()} (within-subjects factor = {within_label})',
        '',
        'Normality is tested on MODEL RESIDUALS, not raw data.',
        '  Residual = y_ij - animal_mean_i - condition_mean_j + grand_mean',
        '  This isolates the pure error term to which the rm-ANOVA normality',
        '  assumption applies (removes between-subject and condition effects).',
        "Mauchly's test: tests the sphericity assumption for repeated-measures ANOVA.",
        '  Sphericity violated -> use Greenhouse-Geisser or Huynh-Feldt correction.',
        "Shapiro-Wilk is used for n \u2264 5000; D'Agostino-Pearson for larger samples.",
        '',
    ]

    summary_rows: List[Tuple[str, str, str, str]] = []

    for col, lbl in available:
        lines += ['\u2500' * 80, f'MEASURE: {lbl}', '\u2500' * 80, '']

        # Build within_groups dict (still needed for Mauchly's test below)
        within_groups: Dict = {}
        for wval in sorted(df_long[within_col].dropna().unique()):
            within_groups[wval] = df_long[df_long[within_col] == wval][col]

        # ── Shapiro-Wilk on residuals ─────────────────────────────────────
        lines += ['  SHAPIRO-WILK ON MODEL RESIDUALS (pooled)', '  ' + '-' * 40]

        residuals = _rm_residuals(df_long, col)
        r_lines, passed = _sw_row(residuals, 'Pooled residuals')
        lines.extend(r_lines)
        summary_rows.append((lbl, 'Pooled residuals', 'Shapiro-Wilk',
                              'NORMAL' if passed else 'NON-NORMAL *'))

        lines.append('')

        # ── Mauchly's test of sphericity ──────────────────────────────────
        lines += ["  MAUCHLY'S TEST OF SPHERICITY (repeated measures)", '  ' + '-' * 40]

        n_within_levels = len(within_groups)
        if n_within_levels < 2:
            lines.append(f'  {within_label} has only {n_within_levels} level \u2014 Mauchly\'s test not applicable.')
        elif n_within_levels == 2:
            lines.append(f'  {within_label} has 2 levels \u2014 sphericity is trivially satisfied (no test needed).')
        elif not HAS_PINGOUIN:
            lines.append("  Requires pingouin \u2014 install with: pip install pingouin")
        else:
            sub_df = df_long[['Animal', within_col, col]].dropna()
            n_subj = sub_df['Animal'].nunique()
            if n_subj < 2:
                lines.append(f"  Mauchly's test requires \u2265 2 subjects (found {n_subj}) \u2014 skipping.")
            else:
                try:
                    result = pg.sphericity(sub_df, dv=col, within=within_col, subject='Animal')
                    W, chi2, dof, p = result.W, result.chi2, result.dof, result.pval
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    passed = (p >= 0.05)
                    conclusion = ('SPHERICITY MET (p \u2265 0.05)' if passed
                                  else 'SPHERICITY VIOLATED (p < 0.05)')
                    lines += [
                        f'  {within_label} conditions (n={n_subj} animals, {n_within_levels} levels):',
                        f"    Mauchly's W = {W:.4f},  chi2 = {chi2:.4f},  df = {dof},  p = {p:.4f}  [{sig}]  \u2192  {conclusion}",
                    ]
                    summary_rows.append((lbl, f'{within_label} (sphericity)', "Mauchly's",
                                          'MET' if passed else 'VIOLATED *'))
                except Exception as e:
                    lines.append(f"  Mauchly's test failed: {e}")

        lines.append('')

    # ── Summary table ──────────────────────────────────────────────────────
    lines += [
        '=' * 80,
        'SUMMARY TABLE',
        '=' * 80,
        f'  {"Measure":<28} {"Grouping":<28} {"Test":<14} {"Result"}',
        '  ' + '-' * 80,
    ]
    for row in summary_rows:
        lines.append(f'  {row[0]:<28} {row[1]:<28} {row[2]:<14} {row[3]}')

    lines += [
        '',
        'NOTE: Shapiro-Wilk has low power with small n \u2014 a non-significant result',
        '      does not guarantee normality. Visual inspection (Q-Q plots) is advised.',
        'NOTE: Residuals = y_ij - animal_mean - condition_mean + grand_mean.',
        '      This is the error term to which the rm-ANOVA normality assumption applies.',
        '',
    ]

    report = '\n'.join(lines)
    if save_path is not None:
        Path(save_path).write_text(report, encoding='utf-8')
        print(f'[OK] Lick normality report saved -> {save_path}')
    return report


def generate_lick_normality_qq_plots(
    weekly_averages: Dict,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Generate Q-Q plots of model residuals for each lick measure.

    Produces one SVG figure per measure showing a single Q-Q panel of the
    pooled within-subject residuals:

      residual_ij = y_ij - animal_mean_i - condition_mean_j + grand_mean

    This is the error term the rm-ANOVA normality assumption applies to.

    Parameters
    ----------
    weekly_averages : output of compute_weekly_averages()
    save_dir        : directory to save SVG files (created if absent).
                      If None the figures are only shown (requires show=True).
    show            : whether to call plt.show() for each figure
    """
    # ── Build long-format DataFrame (same as in generate_lick_normality_report) ──
    sorted_keys = sort_dates_chronologically(list(weekly_averages.keys()))
    rows = []
    for week_idx, date in enumerate(sorted_keys):
        data = weekly_averages[date]
        n = len(data.get('avg_licks_per_animal', []))
        animal_ids   = data.get('animal_ids',   [f'Animal_{i+1}' for i in range(n)])
        animal_sexes = data.get('animal_sexes', ['Unknown'] * n)
        ca_pct       = data.get('ca_percent', 0.0)
        licks        = np.asarray(data.get('avg_licks_per_animal',            np.zeros(n)), dtype=float)
        bouts        = np.asarray(data.get('avg_bouts_per_animal',            np.zeros(n)), dtype=float)
        fecal        = np.asarray(data.get('avg_fecal_per_animal',            np.zeros(n)), dtype=float)
        bottle_wt    = np.asarray(data.get('avg_bottle_weight_per_animal',    np.zeros(n)), dtype=float)
        total_wt     = np.asarray(data.get('avg_total_weight_per_animal',     np.zeros(n)), dtype=float)
        fl_lick_pct  = np.asarray(data.get('first_5min_lick_pcts_per_animal', np.zeros(n)), dtype=float)
        fl_bout_pct  = np.asarray(data.get('first_5min_bout_pcts_per_animal', np.zeros(n)), dtype=float)
        t50          = np.asarray(data.get('time_to_50pct_licks_per_animal',  np.full(n, np.nan)), dtype=float)

        for i in range(len(animal_ids)):
            bw = float(bottle_wt[i]) if bottle_wt[i] > 0 else np.nan
            rows.append({
                'Animal':          animal_ids[i],
                'Sex':             animal_sexes[i] if i < len(animal_sexes) else 'Unknown',
                'CA_Percent':      ca_pct,
                'Week':            week_idx + 1,
                'Total_Licks':     float(licks[i]),
                'Total_Bouts':     float(bouts[i]),
                'Fecal_Count':     float(fecal[i]),
                'Bottle_Weight':   bw,
                'Total_Weight':    float(total_wt[i]),
                'First5min_Lick%': float(fl_lick_pct[i]),
                'First5min_Bout%': float(fl_bout_pct[i]),
                'Time_to_50pct':   float(t50[i]),
            })

    if not rows:
        print('[WARNING] No per-animal data found — cannot create Q-Q plots.')
        return

    df_long = pd.DataFrame(rows)

    MEASURES = [
        ('Total_Licks',     'Total Licks'),
        ('Total_Bouts',     'Total Bouts'),
        ('Fecal_Count',     'Fecal Count'),
        ('Bottle_Weight',   'Bottle Weight Loss (g)'),
        ('Total_Weight',    'Total Weight Loss (g)'),
        ('First5min_Lick%', '% Licks in First 5 min'),
        ('First5min_Bout%', '% Bouts in First 5 min'),
        ('Time_to_50pct',   'Time to 50% Licks (min)'),
    ]
    available = [(col, lbl) for col, lbl in MEASURES if col in df_long.columns]

    within_col   = 'CA_Percent' if EXPERIMENT_MODE == 'ramp' else 'Week'

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _rm_residuals_qq(df: pd.DataFrame, col: str) -> np.ndarray:
        """Return pooled within-subject residuals for a single measure."""
        sub = df[['Animal', within_col, col]].dropna().copy()
        grand = sub[col].mean()
        sub['_animal_mean'] = sub.groupby('Animal')[col].transform('mean')
        sub['_cond_mean']   = sub.groupby(within_col)[col].transform('mean')
        return (sub[col] - sub['_animal_mean'] - sub['_cond_mean'] + grand).values

    for col, lbl in available:
        residuals = _rm_residuals_qq(df_long, col)
        n = len(residuals)

        fig, ax = plt.subplots()
        fig.suptitle(
            f'Q-Q Plot (residuals) \u2014 {lbl}',
            fontweight='bold',
        )
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Model residuals')
        ax.set_title(
            f'Pooled within-subject residuals  (n={n})',
            
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if n < 3:
            ax.text(0.5, 0.5, 'n < 3\ninsufficient data',
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray')
        else:
            (osm, osr), (slope, intercept, _r) = stats.probplot(residuals, dist='norm')

            # 95% pointwise CI via Beta distribution of order statistics
            k = np.arange(1, n + 1)
            p_lo = np.clip(stats.beta.ppf(0.025, k, n - k + 1), 1e-10, 1 - 1e-10)
            p_hi = np.clip(stats.beta.ppf(0.975, k, n - k + 1), 1e-10, 1 - 1e-10)
            ci_lo = slope * stats.norm.ppf(p_lo) + intercept
            ci_hi = slope * stats.norm.ppf(p_hi) + intercept
            ax.fill_between(osm, ci_lo, ci_hi, alpha=0.15, color='crimson', zorder=1)
            x_line = np.array([osm[0], osm[-1]])
            ax.plot(x_line, slope * x_line + intercept,
                    color='crimson', linewidth=1.3, zorder=2)
            ax.scatter(osm, osr, s=30, color='steelblue', alpha=0.85, zorder=3)

            # Annotate with Shapiro-Wilk result
            if n <= 5000:
                sw_stat, sw_p = stats.shapiro(residuals)
                sig = '***' if sw_p < 0.001 else '**' if sw_p < 0.01 else '*' if sw_p < 0.05 else 'ns'
                ax.annotate(
                    f'SW W={sw_stat:.3f}, p={sw_p:.4f} [{sig}]',
                    xy=(0.03, 0.96), xycoords='axes fraction',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                )

        fig.tight_layout()

        if save_dir is not None:
            safe_name = col.replace('%', 'pct').replace(' ', '_').lower()
            fname = f'lick_qq_resid_{safe_name}.svg'
            fig.savefig(Path(save_dir) / fname, format='svg', dpi=200, bbox_inches='tight')
            print(f'[OK] Q-Q plot saved -> {Path(save_dir) / fname}')

        if show:
            plt.show()
        else:
            plt.close(fig)


def generate_lick_descriptive_stats_report(
    weekly_averages: dict,
    *,
    save_report: bool = True,
) -> dict:
    """
    Generate per-week descriptive statistics for lick analysis DVs:
      - Lick Count, Lick Bout Count, Fecal Count: mean, median, SD, variance, 95% CI (t-dist)
      - % Licks in First 5 Min: mean, median, SD, variance, 95% CI (t-dist)

    Prints a formatted table to the console and optionally saves a text report.
    Returns a dict with keys 'dvs', 'weeks', 'report_path'.
    """
    from scipy.stats import t as _t_dist

    fl = _MLB['factor']

    print("\n" + "=" * 80)
    print(f"DESCRIPTIVE STATISTICS PER {fl.upper()}")
    print("=" * 80)

    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    def _week_label(i, d):
        if EXPERIMENT_MODE == 'ramp':
            ca = weekly_averages[d].get('ca_percent', '?')
            return f"CA {ca}%"
        return f"Week {i + 1}"

    def _ci95(arr):
        n = len(arr)
        if n < 2:
            return float('nan'), float('nan')
        se = float(np.std(arr, ddof=1)) / np.sqrt(n)
        margin = float(_t_dist.ppf(0.975, df=n - 1)) * se
        mean = float(np.mean(arr))
        return mean - margin, mean + margin

    dv_specs = [
        ('avg_licks_per_animal',          'Lick Count',              'count'),
        ('avg_bouts_per_animal',          'Lick Bout Count',         'count'),
        ('avg_fecal_per_animal',          'Fecal Count',             'count'),
        ('first_5min_lick_pcts_per_animal', '% Licks in First 5 Min', '%'),
    ]

    results: dict = {}

    for arr_key, dv_label, dv_type in dv_specs:
        results[dv_label] = {}
        note = f" ({dv_type})"
        print(f"\n  {dv_label}{note} \u2014 per {fl}:")
        print(f"  {'Level':>8}  {'n':>4}  {'Mean':>8}  {'Median':>8}  "
              f"{'SD':>8}  {'Var':>10}  {'95% CI':>22}")
        print(f"  {'-' * 76}")
        all_vals = []
        for i, d in enumerate(sorted_dates):
            entry = weekly_averages[d]
            arr = entry.get(arr_key)
            lbl = _week_label(i, d)
            if arr is None or len(arr) == 0:
                results[dv_label][lbl] = {'n': 0}
                print(f"  {lbl:>8}  {0:>4}  {chr(8212):>8}  {chr(8212):>8}  "
                      f"{chr(8212):>8}  {chr(8212):>10}  {chr(8212):>22}")
                continue
            arr = np.asarray(arr, dtype=float)
            all_vals.append(arr)
            n = len(arr)
            mean_v  = float(np.mean(arr))
            median_v = float(np.median(arr))
            sd_v    = float(np.std(arr, ddof=1))  if n >= 2 else float('nan')
            var_v   = float(np.var(arr, ddof=1))  if n >= 2 else float('nan')
            ci_lo, ci_hi = _ci95(arr)
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "N/A"
            results[dv_label][lbl] = {
                'n': n, 'mean': mean_v, 'median': median_v,
                'sd': sd_v, 'variance': var_v, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            }
            print(f"  {lbl:>8}  {n:>4}  {mean_v:>8.3f}  {median_v:>8.3f}  "
                  f"{sd_v:>8.3f}  {var_v:>10.3f}  {ci_str:>22}")
        # Collapsed "All" row — one grand mean per animal across weeks
        if all_vals:
            if len(all_vals) > 1 and all(len(a) == len(all_vals[0]) for a in all_vals):
                # Same n animals each week: average across weeks per animal position
                _all = np.mean(np.column_stack(all_vals), axis=1)
            else:
                # Unequal animal counts across weeks — fall back to concatenation
                _all = np.concatenate(all_vals)
            _an = len(_all)
            _am     = float(np.mean(_all))
            _amed   = float(np.median(_all))
            _asd    = float(np.std(_all, ddof=1))  if _an >= 2 else float('nan')
            _avar   = float(np.var(_all, ddof=1))  if _an >= 2 else float('nan')
            _aci_lo, _aci_hi = _ci95(_all)
            _aci_str = f"[{_aci_lo:.3f}, {_aci_hi:.3f}]" if not np.isnan(_aci_lo) else "N/A"
            results[dv_label]['_all'] = {
                'n': _an, 'mean': _am, 'median': _amed,
                'sd': _asd, 'variance': _avar, 'ci_lo': _aci_lo, 'ci_hi': _aci_hi,
            }
            print(f"  {'-' * 76}")
            print(f"  {'All':>8}  {_an:>4}  {_am:>8.3f}  {_amed:>8.3f}  "
                  f"{_asd:>8.3f}  {_avar:>10.3f}  {_aci_str:>22}")

    week_labels = [_week_label(i, d) for i, d in enumerate(sorted_dates)]

    rpt_path = None
    if save_report:
        lines = [
            "=" * 80,
            f"LICK DESCRIPTIVE STATISTICS REPORT \u2014 {EXPERIMENT_MODE.upper()} MODE",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Within-subjects factor: {fl}  |  Levels: {week_labels}",
            "=" * 80, "",
        ]
        for dv_label, dv_data in results.items():
            lines += [
                "\u2500" * 60,
                f"  {dv_label.upper()}",
                "\u2500" * 60,
                f"  {'Level':>8}  {'n':>4}  {'Mean':>8}  {'Median':>8}  "
                f"{'SD':>8}  {'Var':>10}  {'95% CI':>22}",
                f"  {'-' * 76}",
            ]
            for lbl, s in dv_data.items():
                if lbl == '_all':
                    continue
                if s.get('n', 0) == 0:
                    lines.append(
                        f"  {lbl:>8}  {0:>4}  {'N/A':>8}  {'N/A':>8}  "
                        f"{'N/A':>8}  {'N/A':>10}  {'N/A':>22}"
                    )
                    continue
                ci_str = (f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
                          if not np.isnan(s['ci_lo']) else "N/A")
                lines.append(
                    f"  {lbl:>8}  {s['n']:>4}  {s['mean']:>8.3f}  {s['median']:>8.3f}  "
                    f"{s['sd']:>8.3f}  {s['variance']:>10.3f}  {ci_str:>22}"
                )
            if '_all' in dv_data:
                s = dv_data['_all']
                ci_str = (f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
                          if not np.isnan(s['ci_lo']) else "N/A")
                lines.append(f"  {'-' * 76}")
                lines.append(
                    f"  {'All':>8}  {s['n']:>4}  {s['mean']:>8.3f}  {s['median']:>8.3f}  "
                    f"{s['sd']:>8.3f}  {s['variance']:>10.3f}  {ci_str:>22}"
                )
            lines.append("")
        lines += ["=" * 80, "END OF REPORT", "=" * 80, ""]
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpt_path = Path.cwd() / f"lick_descriptive_stats_report_{_ts}.txt"
        try:
            rpt_path.write_text("\n".join(lines), encoding='utf-8')
            print(f"\nLick descriptive stats report saved: {rpt_path}")
        except Exception as _e:
            print(f"\nWarning: could not save report: {_e}")

    return {
        'dvs': results,
        'weeks': week_labels,
        'report_path': rpt_path,
    }


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
    fig, ax = plt.subplots()
    
    # X-axis: bin centers (for positioning bars)
    num_bins = len(mean_licks)
    bin_edges = np.arange(0, num_bins * bin_size_min + bin_size_min, bin_size_min)
    bin_centers = bin_edges[:-1] + bin_size_min / 2
    
    # Create bar plot with error bars - width equals bin size for continuous bars
    ax.bar(bin_centers, mean_licks, width=bin_size_min, color='steelblue', alpha=0.7, 
           edgecolor='black', yerr=sem_licks, capsize=3,
           error_kw={'elinewidth': 1.5, 'capthick': 1.5}, align='center')
    
    # Labels and title
    ax.set_xlabel('Time (minutes)', weight='bold')
    ax.set_ylabel(f'Licks per {bin_size_min}-Minute Bin (Mean ± SEM)', weight='bold')
    ax.set_title(f'Lick Rate - Week: {date} (CA {ca_percent}%, n={n_animals})', 
                weight='bold')
    
    # Format x-axis to show bin edges
    ax.set_xlim(0, num_bins * bin_size_min)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(x)}' for x in bin_edges])
    
    # Format y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
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
    fig, ax = plt.subplots()
    
    # X-axis: bin centers (for positioning bars)
    num_bins = len(mean_licks)
    bin_edges = np.arange(0, num_bins * bin_size_min + bin_size_min, bin_size_min)
    bin_centers = bin_edges[:-1] + bin_size_min / 2
    
    # Create bar plot with error bars - width equals bin size for continuous bars
    ax.bar(bin_centers, mean_licks, width=bin_size_min, color='darkgreen', alpha=0.7,
           edgecolor='black', yerr=sem_licks, capsize=3,
           error_kw={'elinewidth': 1.5, 'capthick': 1.5}, align='center')
    
    # Labels and title
    ax.set_xlabel('Time (minutes)', weight='bold')
    ax.set_ylabel(f'Licks per {bin_size_min}-Minute Bin (Mean ± SEM)', weight='bold')
    ax.set_title(f'Comprehensive Lick Rate - All Weeks Combined (n={n_unique_animals} mice, {len(sorted_dates)} weeks)',
                weight='bold')
    
    # Format x-axis to show bin edges
    ax.set_xlim(0, num_bins * bin_size_min)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(x)}' for x in bin_edges])
    
    # Format y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    plt.tight_layout()
    
    print("=" * 80 + "\n")
    
    if save_path:
        fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Comprehensive lick rate plot saved to: {save_path}")
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_comprehensive_lick_rate_by_ca(
    lick_rate_data: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot comprehensive lick rate histogram collapsing across CA% concentrations (ramp mode).

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

    ca_groups: Dict = {}
    bin_size_min = None
    all_unique_animal_ids: set = set()

    for date, data in lick_rate_data.items():
        ca_percent = data['ca_percent']
        if ca_percent not in ca_groups:
            ca_groups[ca_percent] = []
        ca_groups[ca_percent].append(data['all_animal_data'])

        if 'animal_ids' in data and data['animal_ids']:
            ids = data['animal_ids']
            if len(ids) > 3:
                print(f"  {date} (CA {ca_percent}%): Adding {len(ids)} animal IDs: {ids[:3]}...")
            else:
                print(f"  {date} (CA {ca_percent}%): Adding animal IDs: {ids}")
            all_unique_animal_ids.update(ids)
        else:
            print(f"  {date} (CA {ca_percent}%): No animal IDs found in data")

        if bin_size_min is None:
            bin_size_min = data.get('bin_size_min', 5.0)

    print(f"\nCollected {len(all_unique_animal_ids)} unique animal IDs total")
    if all_unique_animal_ids:
        print(f"Unique IDs: {sorted(all_unique_animal_ids)}")

    for ca_pct in sorted(ca_groups.keys()):
        total = sum(arr.shape[0] for arr in ca_groups[ca_pct])
        print(f"  CA {ca_pct}%: {total} observations across {len(ca_groups[ca_pct])} week(s)")

    all_data = []
    for ca_pct in sorted(ca_groups.keys()):
        all_data.extend(ca_groups[ca_pct])

    all_animals_all_ca = np.vstack(all_data)
    n_unique = len(all_unique_animal_ids) if all_unique_animal_ids else all_animals_all_ca.shape[0]

    mean_licks = np.mean(all_animals_all_ca, axis=0)
    sem_licks  = np.std(all_animals_all_ca, axis=0, ddof=1) / np.sqrt(all_animals_all_ca.shape[0])

    fig, ax = plt.subplots()
    num_bins   = len(mean_licks)
    bin_edges  = np.arange(0, num_bins * bin_size_min + bin_size_min, bin_size_min)
    bin_centers = bin_edges[:-1] + bin_size_min / 2

    ax.bar(bin_centers, mean_licks, width=bin_size_min, color='darkgreen', alpha=0.7,
           edgecolor='black', yerr=sem_licks, capsize=3,
           error_kw={'elinewidth': 1.5, 'capthick': 1.5}, align='center')

    ax.set_xlabel('Time (minutes)', weight='bold')
    ax.set_ylabel(f'Licks per {bin_size_min}-Minute Bin (Mean \u00b1 SEM)', weight='bold')
    ax.set_title(
        f'Comprehensive Lick Rate - All CA% Combined '
        f'(n={n_unique} unique mice, {len(ca_groups)} CA% concentrations)',
        weight='bold')

    ax.set_xlim(0, num_bins * bin_size_min)
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(x)}' for x in bin_edges])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('dodgerblue')
            elif ca <= 1.0:    ca_colors.append('skyblue')
            elif ca <= 2.0:    ca_colors.append('gold')
            elif ca <= 3.0:    ca_colors.append('orange')
            else:              ca_colors.append('orangered')
        bar_color_arg = ca_colors
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'dodgerblue'
        elif ca_pct <= 1.0:   bar_color = 'skyblue'
        else:                  bar_color = 'orangered'
        bar_color_arg = bar_color

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_pcts, bar_width,
                 color=bar_color_arg, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars
    ax.errorbar(x_positions, avg_pcts, yerr=sem_pcts,
               fmt='none', ecolor='black', capsize=5, capthick=2)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_pcts[i] + 1,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')

    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility

    for i, individual_pcts in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_pcts))
        ax.scatter([x_positions[i]] * len(individual_pcts) + jitter, individual_pcts,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)

    # Formatting
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('% of Licks in First 5 Minutes', weight='bold')
    ax.set_title(f'Percentage of Licks in First 5 Minutes Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection (bout colors: green spectrum)
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('seagreen')
            elif ca <= 1.0:    ca_colors.append('mediumseagreen')
            elif ca <= 2.0:    ca_colors.append('lightseagreen')
            elif ca <= 3.0:    ca_colors.append('darkseagreen')
            else:              ca_colors.append('palegreen')
        bar_color_arg = ca_colors
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'seagreen'
        elif ca_pct <= 1.0:   bar_color = 'mediumseagreen'
        else:                  bar_color = 'darkseagreen'
        bar_color_arg = bar_color

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_bout_pcts, bar_width,
                 color=bar_color_arg, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars
    ax.errorbar(x_positions, avg_bout_pcts, yerr=sem_bout_pcts,
               fmt='none', ecolor='black', capsize=5, capthick=2)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_bout_pcts[i] + 1,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')

    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility

    for i, individual_bout_pcts in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_bout_pcts))
        ax.scatter([x_positions[i]] * len(individual_bout_pcts) + jitter, individual_bout_pcts,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)

    # Formatting
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('% of Bouts in First 5 Minutes', weight='bold')
    ax.set_title(f'Percentage of Bouts in First 5 Minutes Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection (bout colors: green spectrum)
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('seagreen')
            elif ca <= 1.0:    ca_colors.append('mediumseagreen')
            elif ca <= 2.0:    ca_colors.append('lightseagreen')
            elif ca <= 3.0:    ca_colors.append('darkseagreen')
            else:              ca_colors.append('palegreen')
        bar_color_arg = ca_colors
        alpha_val = 0.6
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'seagreen'
        elif ca_pct <= 1.0:   bar_color = 'mediumseagreen'
        else:                  bar_color = 'darkseagreen'
        bar_color_arg = bar_color
        alpha_val = 0.7

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_bout_pcts, bar_width,
                 color=bar_color_arg, alpha=alpha_val, edgecolor='black', linewidth=1.5, zorder=5)

    # Add error bars
    ax.errorbar(x_positions, avg_bout_pcts, yerr=sem_bout_pcts,
               fmt='none', ecolor='black', capsize=5, capthick=2, zorder=6)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_bout_pcts[i] + 1,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')
    
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
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('% of Bouts in First 5 Minutes', weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Bout Percentage Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('dodgerblue')
            elif ca <= 1.0:    ca_colors.append('skyblue')
            elif ca <= 2.0:    ca_colors.append('gold')
            elif ca <= 3.0:    ca_colors.append('orange')
            else:              ca_colors.append('orangered')
        bar_color_arg = ca_colors
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'mediumpurple'
        elif ca_pct <= 1.0:   bar_color = 'plum'
        else:                  bar_color = 'darkviolet'
        bar_color_arg = bar_color

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_times, bar_width,
                 color=bar_color_arg, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars
    ax.errorbar(x_positions, avg_times, yerr=sem_times,
               fmt='none', ecolor='black', capsize=5, capthick=2)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_times[i] + 0.5,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')

    # Overlay individual mouse data points
    jitter_amount = 0.15
    np.random.seed(42)  # For reproducibility

    for i, individual_times in enumerate(individual_data):
        jitter = np.random.uniform(-jitter_amount, jitter_amount, len(individual_times))
        ax.scatter([x_positions[i]] * len(individual_times) + jitter, individual_times,
                  color='black', s=40, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=10)

    # Formatting
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('Time to 50% of Total Licks (minutes)', weight='bold')
    ax.set_title(f'Time to 50% of Total Licks Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(individual_data[0]) if individual_data else 0
    ax.text(0.98, 0.02, f'n={n_animals} mice per week', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('dodgerblue')
            elif ca <= 1.0:    ca_colors.append('skyblue')
            elif ca <= 2.0:    ca_colors.append('gold')
            elif ca <= 3.0:    ca_colors.append('orange')
            else:              ca_colors.append('orangered')
        bar_color_arg = ca_colors
        alpha_val = 0.6
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'mediumpurple'
        elif ca_pct <= 1.0:   bar_color = 'plum'
        else:                  bar_color = 'darkviolet'
        bar_color_arg = bar_color
        alpha_val = 0.7

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_times, bar_width,
                 color=bar_color_arg, alpha=alpha_val, edgecolor='black', linewidth=1.5, zorder=5)

    # Add error bars
    ax.errorbar(x_positions, avg_times, yerr=sem_times,
               fmt='none', ecolor='black', capsize=5, capthick=2, zorder=6)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_times[i] + 0.5,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')
    
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
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('Time to 50% of Total Licks (minutes)', weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Time to 50% Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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

    # Sort dates by CA% (ramp) or chronologically (nonramp)
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
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
    fig, ax = plt.subplots()

    # Mode-aware color selection
    if EXPERIMENT_MODE == 'ramp':
        ca_colors = []
        for ca in ca_percents:
            if ca == 0:        ca_colors.append('dodgerblue')
            elif ca <= 1.0:    ca_colors.append('skyblue')
            elif ca <= 2.0:    ca_colors.append('gold')
            elif ca <= 3.0:    ca_colors.append('orange')
            else:              ca_colors.append('orangered')
        bar_color_arg = ca_colors
        alpha_val = 0.6
    else:
        ca_pct = ca_percents[0] if ca_percents else 0
        if ca_pct == 0:       bar_color = 'dodgerblue'
        elif ca_pct <= 1.0:   bar_color = 'skyblue'
        else:                  bar_color = 'orangered'
        bar_color_arg = bar_color
        alpha_val = 0.7

    x_positions = np.arange(n_weeks)
    bar_width = 0.6

    # Plot bars
    bars = ax.bar(x_positions, avg_pcts, bar_width,
                 color=bar_color_arg, alpha=alpha_val, edgecolor='black', linewidth=1.5, zorder=5)

    # Add error bars
    ax.errorbar(x_positions, avg_pcts, yerr=sem_pcts,
               fmt='none', ecolor='black', capsize=5, capthick=2, zorder=6)

    # Ramp: add CA% labels above each bar
    if EXPERIMENT_MODE == 'ramp':
        for i, (bar, ca_pct_i) in enumerate(zip(bars, ca_percents)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sem_pcts[i] + 1,
                    f'{ca_pct_i}% CA', ha='center', va='bottom', weight='bold')
    
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
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('% of Licks in First 5 Minutes', weight='bold')
    ax.set_title(f'Individual Mouse Trajectories of Lick Percentage Across Weeks\n({_MLB["plot_suffix"]})',
                weight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    # Add sample size annotation
    n_animals = len(animal_trajectories)
    ax.text(0.98, 0.02, f'n={n_animals} mice tracked across weeks', transform=ax.transAxes,
           ha='right', va='bottom', style='italic',
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


def plot_interaction_effects(
    anova_results: Dict,
    weekly_averages: Dict,
    save_dir: Optional[Path] = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """Plot interaction effects for measures with significant Sex × CA% interactions.
    
    Only applicable for ramp experiment mode.
    Creates line plots showing how the effect of CA% differs between sexes.
    
    Parameters:
        anova_results: Dictionary from perform_anova_analysis
        weekly_averages: Dictionary from compute_weekly_averages
        save_dir: Optional directory to save plots
        show: Whether to display the plots
        
    Returns:
        Dictionary mapping measure names to figure objects
    """
    if EXPERIMENT_MODE != 'ramp':
        print("\nInteraction effects plot is only available for ramp experiment mode.")
        return {}
    
    # Check if we have any significant interactions
    significant_interactions = {
        measure: results for measure, results in anova_results.items()
        if results.get('significant_interaction', False) and not results.get('error')
    }
    
    if not significant_interactions:
        print("\nNo significant Sex × CA% interactions found - skipping interaction plots.")
        return {}
    
    print("\n" + "=" * 80)
    print("CREATING INTERACTION PLOTS FOR SIGNIFICANT SEX × CA% INTERACTIONS")
    print("=" * 80)
    
    # Map measure codes to data keys
    measure_to_key = {
        'licks': 'avg_licks_per_animal',
        'bouts': 'avg_bouts_per_animal',
        'fecal': 'avg_fecal_per_animal',
        'bottle_weight': 'avg_bottle_weight_per_animal',
        'total_weight': 'avg_total_weight_per_animal'
    }
    
    measure_to_ylabel = {
        'licks': 'Total Licks',
        'bouts': 'Total Bouts',
        'fecal': 'Fecal Count',
        'bottle_weight': 'Bottle Weight Loss (g)',
        'total_weight': 'Total Weight Loss (g)'
    }
    
    figures = {}
    
    # Sort by CA% for proper x-axis ordering
    sorted_dates = sorted(weekly_averages.keys(), key=lambda d: weekly_averages[d]['ca_percent'])
    
    for measure, results in significant_interactions.items():
        measure_name = results['measure_name']
        print(f"\nCreating interaction plot for: {measure_name}")
        
        # Collect data organized by Sex and CA%
        data_by_sex_ca = {'M': {}, 'F': {}}
        
        for date in sorted_dates:
            data = weekly_averages[date]
            ca_percent = data['ca_percent']
            animal_ids = data.get('animal_ids', [])
            animal_sexes = data.get('animal_sexes', [])
            measure_values = data.get(measure_to_key[measure], [])
            
            # Skip if no animal data
            if not animal_ids or not animal_sexes:
                continue
            
            # Organize by sex
            for i, (animal_id, sex, value) in enumerate(zip(animal_ids, animal_sexes, measure_values)):
                sex = sex.upper().strip()
                if sex in ['M', 'F']:
                    if ca_percent not in data_by_sex_ca[sex]:
                        data_by_sex_ca[sex][ca_percent] = []
                    data_by_sex_ca[sex][ca_percent].append(value)
        
        # Compute means and SEMs for each Sex × CA% combination
        ca_levels = sorted(set(list(data_by_sex_ca['M'].keys()) + list(data_by_sex_ca['F'].keys())))
        
        males_means = []
        males_sems = []
        females_means = []
        females_sems = []
        
        for ca in ca_levels:
            # Males
            if ca in data_by_sex_ca['M'] and len(data_by_sex_ca['M'][ca]) > 0:
                m_values = np.array(data_by_sex_ca['M'][ca])
                males_means.append(np.mean(m_values))
                males_sems.append(np.std(m_values, ddof=1) / np.sqrt(len(m_values)) if len(m_values) > 1 else 0)
            else:
                males_means.append(np.nan)
                males_sems.append(np.nan)
            
            # Females
            if ca in data_by_sex_ca['F'] and len(data_by_sex_ca['F'][ca]) > 0:
                f_values = np.array(data_by_sex_ca['F'][ca])
                females_means.append(np.mean(f_values))
                females_sems.append(np.std(f_values, ddof=1) / np.sqrt(len(f_values)) if len(f_values) > 1 else 0)
            else:
                females_means.append(np.nan)
                females_sems.append(np.nan)
        
        # Create plot
        fig, ax = plt.subplots()
        
        # Plot lines for each sex
        ax.errorbar(ca_levels, males_means, yerr=males_sems,
                   marker='s', capsize=5,
                   color=COHORT_COLOR, markerfacecolor=COHORT_COLOR, markeredgecolor=COHORT_COLOR,
                   label='Male', linestyle='-')
        
        ax.errorbar(ca_levels, females_means, yerr=females_sems,
                   marker='o', capsize=5,
                   color=COHORT_COLOR, markerfacecolor=COHORT_COLOR, markeredgecolor=COHORT_COLOR,
                   label='Female', linestyle='--')
        
        # Labels and title
        ax.set_xlabel('Citric Acid Concentration (%)', weight='bold')
        ax.set_ylabel(measure_to_ylabel[measure], weight='bold')
        ax.set_title(f'Sex × CA% Interaction: {measure_name}\n(p = {results["p_value_interaction"]:.4f})',
                    weight='bold')
        
        # Format x-axis
        ax.set_xticks(ca_levels)
        ax.set_xticklabels([f'{int(ca)}%' for ca in ca_levels])
        
        # Add legend
        ax.legend(loc='best', frameon=True, shadow=True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(direction='in', which='both', length=5)
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            save_path = save_dir / f"interaction_plot_{measure}.svg"
            fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
        
        figures[measure] = fig
        
        if not show:
            plt.close(fig)
    
    print("=" * 80 + "\n")
    
    return figures


def load_master_csv(csv_path: Path) -> pd.DataFrame:
    """Load the master metadata CSV with all weeks of data."""
    global COHORT_COLOR
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Clean date column
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str).str.strip()
    
    COHORT_COLOR = _detect_cohort_color(csv_path, df)
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
    ili_cutoff: float = 0.5
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
        # Apply evaporation/instrument loss correction: subtract 0.11, floor at 0
        bottle_wt = max(0.0, bottle_wt - 0.11)
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


def plot_licks_vs_weight_correlation(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Scatter plot of total lick count vs total weight change (%) per animal per session.

    Each animal is assigned a unique colour. All points use circle markers.
    A Spearman rank correlation and linear trend line are overlaid.

    Parameters:
        weekly_averages: Dictionary from compute_weekly_averages
        save_path: Optional path to save the figure (SVG)
        show: Whether to display the plot interactively

    Returns:
        The matplotlib Figure object, or None if no data.
    """
    print("\n" + "=" * 80)
    print("PLOTTING: Total Licks vs Total Weight Change (%) – per-animal correlation")
    print("=" * 80)

    # ── Collect per-animal, per-week observations ──────────────────────────
    records = []

    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    for week_idx, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        lick_arr = np.asarray(data['avg_licks_per_animal'], dtype=float)
        wt_arr   = np.asarray(data['avg_total_weight_per_animal'], dtype=float)
        ids      = data.get('animal_ids', [f"Animal_{j+1}" for j in range(len(lick_arr))])
        ca_pct   = data['ca_percent']
        wk_label = f"Week {week_idx + 1}"

        for animal_id, licks, wt in zip(ids, lick_arr, wt_arr):
            records.append({
                'animal_id':  str(animal_id),
                'licks':      licks,
                'weight_pct': wt,
                'week_idx':   week_idx,
                'week_label': wk_label,
                'ca_percent': ca_pct,
            })

    if not records:
        print("ERROR: No per-animal data found in weekly_averages.")
        return None

    # ── Aggregate to per-animal means across weeks ─────────────────────────
    animal_licks_all = {}
    animal_wt_all    = {}
    for r in records:
        aid = r['animal_id']
        animal_licks_all.setdefault(aid, []).append(r['licks'])
        animal_wt_all.setdefault(aid, []).append(r['weight_pct'])

    unique_animals = list(dict.fromkeys(r['animal_id'] for r in records))
    n_animals = len(unique_animals)

    mean_licks = {aid: np.nanmean(animal_licks_all[aid]) for aid in unique_animals}
    mean_wt    = {aid: np.nanmean(animal_wt_all[aid])    for aid in unique_animals}

    cmap = plt.cm.tab20
    animal_color = {aid: cmap(i % 20) for i, aid in enumerate(unique_animals)}

    print(f"Found {n_animals} unique animals across {len(sorted_dates)} weeks "
          f"({len(records)} total observations → 1 mean point per animal)")

    # ── Spearman rank correlation on per-animal means ──────────────────────
    avg_licks_arr = np.array([mean_licks[aid] for aid in unique_animals])
    avg_wt_arr    = np.array([mean_wt[aid]    for aid in unique_animals])

    valid_mask  = np.isfinite(avg_licks_arr) & np.isfinite(avg_wt_arr)
    avg_licks_v = avg_licks_arr[valid_mask]
    avg_wt_v    = avg_wt_arr[valid_mask]

    if len(avg_licks_v) >= 3:
        rho, p_val = stats.spearmanr(avg_wt_v, avg_licks_v)
        slope, intercept, *_ = stats.linregress(avg_wt_v, avg_licks_v)
        x_line = np.linspace(avg_wt_v.min(), avg_wt_v.max(), 200)
        y_line = slope * x_line + intercept
        has_corr = True
        print(f"Spearman rho = {rho:.3f}, p = {p_val:.4f} (n={len(avg_licks_v)} animals)")
    else:
        has_corr = False
        print("Not enough valid animals for correlation.")

    design_label = _MLB['plot_suffix']
    _ms = plt.rcParams.get('lines.markersize', 6)

    # ── Figure 1: All observations (one point per animal per week) ──────────
    _all_wt    = np.array([r['weight_pct'] for r in records], dtype=float)
    _all_licks = np.array([r['licks']      for r in records], dtype=float)

    _valid_all  = np.isfinite(_all_wt) & np.isfinite(_all_licks)
    _all_wt_v   = _all_wt[_valid_all]
    _all_lick_v = _all_licks[_valid_all]

    if len(_all_wt_v) >= 3:
        _rho_all, _p_all = stats.spearmanr(_all_wt_v, _all_lick_v)
        _sl_all, _ic_all, *_ = stats.linregress(_all_wt_v, _all_lick_v)
        _xl_all = np.linspace(_all_wt_v.min(), _all_wt_v.max(), 200)
        _yl_all = _sl_all * _xl_all + _ic_all
        _has_corr_all = True
        print(f"All-obs Spearman rho = {_rho_all:.3f}, p = {_p_all:.4f} "
              f"(n={len(_all_wt_v)} observations)")
    else:
        _has_corr_all = False

    fig_all, ax_all = plt.subplots()

    if _has_corr_all:
        ax_all.plot(_xl_all, _yl_all, color='dimgray', linestyle='--', zorder=2)

    for r in records:
        if not (np.isfinite(r['weight_pct']) and np.isfinite(r['licks'])):
            continue
        ax_all.scatter(r['weight_pct'], r['licks'],
                       color=animal_color[r['animal_id']], marker='o',
                       s=_ms ** 2, edgecolors='black', linewidths=0.6, zorder=4)

    if _has_corr_all:
        _p_str_all = f"p = {_p_all:.4f}" if _p_all >= 0.0001 else "p < 0.0001"
        ax_all.text(0.03, 0.97,
                    f"Spearman \u03c1 = {_rho_all:.3f}\n{_p_str_all}\n"
                    f"n = {len(_all_wt_v)} obs ({n_animals} mice)",
                    transform=ax_all.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.7, edgecolor='gray'))

    ax_all.set_xlabel('Total Weight Change (%)', weight='bold')
    ax_all.set_ylabel('Total Lick Count', weight='bold')
    ax_all.set_title(f'Lick Count vs Weight Change (%) – All Observations\n({design_label})',
                     weight='bold')
    ax_all.spines['top'].set_visible(False)
    ax_all.spines['right'].set_visible(False)
    ax_all.tick_params(direction='in', which='both', length=5)

    plt.tight_layout()

    if save_path:
        fig_all.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"All-observations figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_all)

    # ── Figure 2: Per-animal means ─────────────────────────────────────────
    fig, ax = plt.subplots()

    if has_corr:
        ax.plot(x_line, y_line, color='dimgray',
                linestyle='--', zorder=2, label='Linear fit')

    for animal_id in unique_animals:
        color = animal_color[animal_id]
        ax.scatter(mean_wt[animal_id], mean_licks[animal_id],
                   color=color, marker='o', s=_ms ** 2,
                   edgecolors='black', linewidths=0.6, zorder=4,
                   label=animal_id)

    if has_corr:
        p_str = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
        ax.text(0.03, 0.97,
                f"Spearman \u03c1 = {rho:.3f}\n{p_str}\nn = {len(avg_licks_v)} animals",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.set_xlabel('Mean Total Weight Change (%)', weight='bold')
    ax.set_ylabel('Mean Total Lick Count', weight='bold')
    ax.set_title(f'Mean Lick Count vs Mean Weight Change (%) – Per Animal\n({design_label})',
                 weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)

    handles, labels = ax.get_legend_handles_labels()
    trend_handles  = [h for h, l in zip(handles, labels) if l == 'Linear fit']
    animal_handles = [h for h, l in zip(handles, labels) if l != 'Linear fit']
    animal_labels  = [l for l in labels if l != 'Linear fit']

    legend_kwargs = dict(framealpha=0.7, ncol=max(1, n_animals // 12 + 1))
    if trend_handles:
        ax.legend(trend_handles + animal_handles,
                  ['Linear fit'] + animal_labels,
                  loc='upper right' if n_animals <= 10 else 'lower left',
                  **legend_kwargs)
    else:
        ax.legend(animal_handles, animal_labels,
                  loc='upper right' if n_animals <= 10 else 'lower left',
                  **legend_kwargs)

    plt.tight_layout()

    print(f"Per-animal means plot created: {n_animals} animals (mean across {len(sorted_dates)} weeks).")
    print("=" * 80 + "\n")

    if save_path:
        _means_path = save_path.parent / (save_path.stem + '_animal_means' + save_path.suffix)
        fig.savefig(_means_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Animal-means figure saved to: {_means_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ── Mixed effects model — 2nd figure ──────────────────────────────────
    # Build panel dataset (shared by both estimation paths)
    _panel_rows = [
        {'mouse_id':   r['animal_id'],
         'time_idx':   r['week_idx'] + 1,
         'week_num':   float(r['week_idx'] + 1),
         'licks':      r['licks'],
         'weight_pct': r['weight_pct']}
        for r in records
    ]
    _pnl_raw = pd.DataFrame(_panel_rows).dropna(subset=['licks', 'weight_pct'])

    if _pnl_raw.shape[0] < 5:
        print("Not enough complete observations for mixed model.")
        return fig

    # ── Detect which backend is available: rpy2+nlme first, then linearmodels ──
    _use_rpy2 = False
    try:
        import os as _os
        import glob as _glob

        # ── Windows: ensure R's DLLs are findable by LoadLibrary ────────────
        # Bug fix: run regardless of whether R_HOME is already set in the env.
        # If R_HOME was set by conda/rpy2 but bin\x64 was never added to PATH,
        # Windows cannot resolve R.dll / Rblas.dll when loading stats.dll or
        # nlme — even though R itself started successfully via rpy2.
        if _os.name == 'nt':
            # Use R_HOME already in env, or discover it by searching standard paths
            _r_home = _os.environ.get('R_HOME', '')
            if not _r_home:
                _r_candidates = (
                    _glob.glob(r'C:\Program Files\R\R-*\bin\x64')
                    + _glob.glob(r'C:\Program Files\R\R-*\bin')
                    + _glob.glob(r'C:\R\R-*\bin\x64')
                )
                if _r_candidates:
                    _r_candidates.sort(reverse=True)
                    _r_cand = _r_candidates[0]
                    _r_home = str(Path(_r_cand).parent.parent
                                   if _r_cand.endswith('x64') else Path(_r_cand).parent)
                    _os.environ['R_HOME'] = _r_home
            if _r_home:
                # Always prepend both bin\x64 and bin so all R DLLs are found
                for _p in [_os.path.join(_r_home, 'bin', 'x64'),
                            _os.path.join(_r_home, 'bin')]:
                    if _os.path.isdir(_p):
                        if _p not in _os.environ.get('PATH', ''):
                            _os.environ['PATH'] = _p + _os.pathsep + _os.environ.get('PATH', '')
                        # Python 3.8+: add as a trusted DLL search directory
                        try:
                            _os.add_dll_directory(_p)
                        except (AttributeError, OSError):
                            pass

        # Suppress R's attempt to find 'sh' on Windows startup
        _os.environ.setdefault('R_PROFILE_USER', '')
        _os.environ.setdefault('R_ENVIRON_USER', '')

        import rpy2.robjects as _ro
        import rpy2.robjects.packages as _rpkgs
        _rpkgs.importr('lme4')   # will raise if R or lme4 not available
        _use_rpy2 = True
    except Exception as _rpy2_exc:
        print(f"NOTE: rpy2/nlme not available ({type(_rpy2_exc).__name__}: {_rpy2_exc})")
        print("      Falling back to linearmodels random-intercept model.")
        print("      To use the full model: install VC++ Redist 2022 from")
        print("        https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("      then restart your terminal and retry.")

    # ══════════════════════════════════════════════════════════════════════
    # PATH A: rpy2 + lme4 glmer.nb — GLMM Negative Binomial, random intercept
    # ══════════════════════════════════════════════════════════════════════
    if _use_rpy2:
        print("\n" + "=" * 80)
        print("MIXED EFFECTS MODEL: lme4 glmer.nb()  [via rpy2]")
        print("  licks ~ weight_pct + week_num + (1 | mouse_id)")
        print("  Family:  Negative Binomial (log link)")
        print("  Random:  random intercept per mouse")
        print("  Method:  ML (Laplace approximation)")
        print("=" * 80)

        # Push data into R — licks must be integer counts for NB
        _ro.globalenv['lme_mouse'] = _ro.StrVector(
            _pnl_raw['mouse_id'].astype(str).tolist())
        _ro.globalenv['lme_week']  = _ro.FloatVector(_pnl_raw['week_num'].tolist())
        _ro.globalenv['lme_licks'] = _ro.IntVector(
            _pnl_raw['licks'].round().astype(int).tolist())
        _ro.globalenv['lme_wt']    = _ro.FloatVector(_pnl_raw['weight_pct'].tolist())

        _ro.r('''
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(MASS))
.df_lme <- data.frame(
    mouse_id   = factor(lme_mouse),
    week_num   = lme_week,
    licks      = as.integer(lme_licks),
    weight_pct = lme_wt
)

# ── Strategy: glmer.nb alternates theta / PIRLS and can fail on small n.
# ── Step 1: Estimate NB dispersion theta from a marginal glm.nb (no RE).
# ── Step 2: Fit glmer() with theta fixed — avoids the unstable alternation.
# ── glmer.nb() is tried first; two-stage approach is the fallback.
.fit_nb_fixed_theta <- function(data) {
    .marg <- tryCatch(
        MASS::glm.nb(licks ~ weight_pct + week_num,
                     data    = data,
                     control = glm.control(maxit = 200)),
        error = function(e) { message("glm.nb failed: ", conditionMessage(e)); NULL }
    )
    if (is.null(.marg)) return(NULL)
    .th <- .marg$theta
    message(sprintf("Two-stage NB: glm.nb theta = %.4f; fitting glmer with fixed theta", .th))
    .m <- tryCatch(
        glmer(licks ~ weight_pct + week_num + (1 | mouse_id),
              data    = data,
              family  = MASS::negative.binomial(theta = .th),
              control = glmerControl(optimizer = "bobyqa",
                                     optCtrl   = list(maxfun = 2e5))),
        error = function(e) {
            message("glmer fixed-theta (bobyqa) failed: ", conditionMessage(e))
            tryCatch(
                glmer(licks ~ weight_pct + week_num + (1 | mouse_id),
                      data    = data,
                      family  = MASS::negative.binomial(theta = .th),
                      control = glmerControl(optimizer = "nloptwrap",
                                             optCtrl   = list(maxfun = 2e5))),
                error = function(e2) { message("glmer fixed-theta (nloptwrap) failed: ", conditionMessage(e2)); NULL }
            )
        }
    )
    if (!is.null(.m)) attr(.m, ".nb_theta_fixed") <- .th
    .m
}

.model_lme <- tryCatch(
    glmer.nb(
        licks ~ weight_pct + week_num + (1 | mouse_id),
        data    = .df_lme,
        control = glmerControl(optimizer = "bobyqa",
                               optCtrl   = list(maxfun = 2e5))),
    error = function(e) {
        message("glmer.nb failed: ", conditionMessage(e),
                " -- falling back to two-stage fixed-theta approach")
        .fit_nb_fixed_theta(.df_lme)
    }
)
if (is.null(.model_lme)) stop("All glmer.nb fitting strategies failed.")

# Retrieve theta: either from glmer.nb or from the fixed-theta attribute
.lme_theta <- tryCatch(
    getME(.model_lme, "glmer.nb.theta"),
    error = function(e) {
        th <- attr(.model_lme, ".nb_theta_fixed")
        if (is.null(th)) stop("Cannot retrieve NB theta.") else th
    }
)
.lme_coef  <- coef(summary(.model_lme))
.lme_vc_df <- as.data.frame(VarCorr(.model_lme))
''')

        # Extract fixed effects (z-test for GLMM; columns vary by lme4 version)
        _tt     = np.array(_ro.r('.lme_coef'))
        _tt_row = list(_ro.r('rownames(.lme_coef)'))
        _tt_col = list(_ro.r('colnames(.lme_coef)'))
        _fe_df  = pd.DataFrame(_tt, index=_tt_row, columns=_tt_col)
        # Normalise column names
        _col_map = {}
        for _c in _fe_df.columns:
            _cl = _c.lower().replace(' ', '_')
            if 'estimate' in _cl:   _col_map[_c] = 'Estimate'
            elif 'std' in _cl:      _col_map[_c] = 'Std.Error'
            elif _cl in ('z_value', 'z value', 'zval', 'z'):
                                    _col_map[_c] = 'z value'
            elif 'pr' in _cl or 'p' == _cl:
                                    _col_map[_c] = 'p-value'
        _fe_df = _fe_df.rename(columns=_col_map)

        print("\nFixed Effects (log-link, on log scale):")
        print(_fe_df.to_string(float_format='{:.4f}'.format))
        print("  Incidence Rate Ratios [exp(\u03b2)]:")
        for _idx in _fe_df.index:
            _est = float(_fe_df.loc[_idx, 'Estimate'])
            print(f"    {_idx:<20} exp(\u03b2) = {np.exp(_est):.4f}")

        _b0   = float(_fe_df.loc['(Intercept)', 'Estimate'])
        _b_wt = float(_fe_df.loc['weight_pct',  'Estimate'])
        _b_wk = float(_fe_df.loc['week_num',     'Estimate'])
        _p_wt = float(_fe_df.loc['weight_pct',  'p-value'])
        _theta = float(_ro.r('.lme_theta')[0])

        # Random-intercept variance from VarCorr data frame
        _vc_grp  = list(_ro.r('.lme_vc_df$grp'))
        _vc_vcov = list(_ro.r('.lme_vc_df$vcov'))
        _var_int_r = float(
            _vc_vcov[_vc_grp.index('mouse_id')]
            if 'mouse_id' in _vc_grp else np.nan
        )
        _mean_wk = float(_pnl_raw['week_num'].mean())

        # ICC for NB-GLMM (Nakagawa et al. 2017)
        _var_dist = np.log(1.0 / _theta + 1.0)   # distributional variance
        _icc1 = (_var_int_r / (_var_int_r + _var_dist)
                 if (_var_int_r + _var_dist) > 0 else np.nan)

        # Nakagawa & Schielzeth 2013/2017 R² for GLMMs
        _b_vec    = np.array([_b0, _b_wt, _b_wk])
        _X_mat    = np.column_stack([np.ones(len(_pnl_raw)),
                                     _pnl_raw['weight_pct'].values,
                                     _pnl_raw['week_num'].values])
        _sigma_f2    = float(np.var(_X_mat @ _b_vec))
        _total_var   = _sigma_f2 + _var_int_r + _var_dist
        _r2_marginal = _sigma_f2 / _total_var if _total_var > 0 else np.nan
        _r2_cond     = (_sigma_f2 + _var_int_r) / _total_var if _total_var > 0 else np.nan

        _model_label = "GLMM NB, RE intercept/mouse"

        print(f"\nNB dispersion \u03b8:       {_theta:.4f}")
        print(f"Random intercept \u03c3\u00b2:   {_var_int_r:.4f}")
        print(f"Distributional var:     {_var_dist:.4f}  [log(1/\u03b8 + 1)]")
        print(f"ICC(1)               = {_icc1:.3f}  (mouse-level / total variance)")
        print(f"Marginal  R\u00b2         = {_r2_marginal:.3f}  (fixed effects only)")
        print(f"Conditional R\u00b2       = {_r2_cond:.3f}  (fixed + random effects)")
        print("=" * 80 + "\n")

        _irr_lines = [
            f"  exp(\u03b2)  {_idx:<20} = {np.exp(float(_fe_df.loc[_idx, 'Estimate'])):.4f}"
            for _idx in _fe_df.index
        ]
        _mm_report = "\n".join([
            "=" * 80,
            "MIXED EFFECTS MODEL: lme4 glmer.nb()  [via rpy2]",
            "  licks ~ weight_pct + week_num + (1 | mouse_id)",
            "  Family:  Negative Binomial (log link)",
            "  Random:  random intercept per mouse",
            "  Method:  ML (Laplace approximation)",
            "=" * 80,
            "",
            "Fixed Effects  (log scale):",
            _fe_df.to_string(float_format='{:.4f}'.format),
            "",
            "Incidence Rate Ratios  [exp(\u03b2)]:",
        ] + _irr_lines + [
            "",
            "Variance components:",
            f"  NB dispersion \u03b8:       {_theta:.4f}",
            f"  Random intercept \u03c3\u00b2:   {_var_int_r:.4f}",
            f"  Distributional var:     {_var_dist:.4f}  [log(1/\u03b8 + 1)]",
            "",
            f"ICC(1)               = {_icc1:.3f}  (mouse-level / total variance)",
            f"Marginal  R\u00b2         = {_r2_marginal:.3f}  (fixed effects only)",
            f"Conditional R\u00b2       = {_r2_cond:.3f}  (fixed + random effects)",
            "=" * 80,
            "",
            f"n = {_pnl_raw.shape[0]} observations,  {n_animals} mice",
            f"Weeks: {len(sorted_dates)}  (mean week = {_mean_wk:.1f})",
        ])

        # ── Polynomial contrasts for the Week effect (lmerTest + emmeans) ──
        import tempfile as _tmpfile
        import os as _os_poly

        _poly_tmpdir = _tmpfile.gettempdir().replace('\\', '/')
        _poly_uid    = str(id(_pnl_raw))[-6:]
        _poly_data_p = f"{_poly_tmpdir}/la_poly_data_{_poly_uid}.csv"
        _poly_fe_p   = f"{_poly_tmpdir}/la_poly_fe_{_poly_uid}.csv"
        _poly_co_p   = f"{_poly_tmpdir}/la_poly_co_{_poly_uid}.csv"
        _poly_fs_p   = f"{_poly_tmpdir}/la_poly_fs_{_poly_uid}.csv"

        try:
            _poly_df = (_pnl_raw[['mouse_id', 'week_num', 'licks']]
                        .rename(columns={'mouse_id': 'ID',
                                         'week_num': 'Week',
                                         'licks':    'Total_Licks'})
                        .dropna()
                        .copy())
            _poly_df.to_csv(_poly_data_p, index=False)

            _ro.globalenv['r_poly_data_p'] = _poly_data_p
            _ro.globalenv['r_poly_fe_p']   = _poly_fe_p
            _ro.globalenv['r_poly_co_p']   = _poly_co_p
            _ro.globalenv['r_poly_fs_p']   = _poly_fs_p

            _ro.r("""
                suppressPackageStartupMessages(library(lme4))
                suppressPackageStartupMessages(library(lmerTest))
                suppressPackageStartupMessages(library(emmeans))

                df_poly   <- read.csv(r_poly_data_p)
                df_poly$ID <- factor(df_poly$ID)
                df_poly$Total_Licks <- as.integer(round(df_poly$Total_Licks))
                wvals      <- sort(unique(df_poly$Week))
                df_poly$Week_f <- factor(df_poly$Week, levels=wvals, ordered=TRUE)

                model_poly <- tryCatch(
                    glmer.nb(
                        Total_Licks ~ Week_f + (1|ID),
                        data    = df_poly,
                        control = glmerControl(optimizer = "bobyqa",
                                               optCtrl   = list(maxfun = 2e5))),
                    error = function(e) {
                        message("glmer.nb for poly contrasts failed: ", conditionMessage(e),
                                " — falling back to lmerTest::lmer on log1p-transformed counts")
                        lmerTest::lmer(
                            log1p(Total_Licks) ~ Week_f + (1|ID),
                            data=df_poly, REML=TRUE)
                    }
                )
                .poly_used_glmm <- inherits(model_poly, "glmerMod")

                fe_poly      <- as.data.frame(coef(summary(model_poly)))
                fe_poly$term <- rownames(fe_poly)
                write.csv(fe_poly, r_poly_fe_p, row.names=FALSE)

                fit_stats_poly <- data.frame(
                    AIC     = AIC(model_poly),
                    BIC     = BIC(model_poly),
                    logLik  = as.numeric(logLik(model_poly)),
                    nobs    = nobs(model_poly),
                    formula = deparse(formula(model_poly), width.cutoff=120L),
                    model   = if (.poly_used_glmm) 'glmer.nb' else 'lmer(log1p)'
                )
                write.csv(fit_stats_poly, r_poly_fs_p, row.names=FALSE)

                # emmeans on log scale; type='response' back-transforms to count scale
                em_week    <- emmeans(model_poly, ~ Week_f,
                                      type = if (.poly_used_glmm) 'response' else 'response')
                contr_week <- as.data.frame(contrast(
                    emmeans(model_poly, ~ Week_f),  # contrasts on link scale
                    'poly'))
                write.csv(contr_week, r_poly_co_p, row.names=FALSE)
            """)

            _poly_fe_df  = pd.read_csv(_poly_fe_p)
            _poly_fit_df = pd.read_csv(_poly_fs_p)
            _poly_co_df  = pd.read_csv(_poly_co_p)

            for _fpath in [_poly_data_p, _poly_fe_p, _poly_co_p, _poly_fs_p]:
                try:    _os_poly.unlink(_fpath)
                except Exception: pass

            _poly_aic    = float(_poly_fit_df['AIC'].iloc[0])
            _poly_bic    = float(_poly_fit_df['BIC'].iloc[0])
            _poly_loglik = float(_poly_fit_df['logLik'].iloc[0])
            _poly_nobs   = int(_poly_fit_df['nobs'].iloc[0])

            def _rget_val(row, *names):
                for n in names:
                    try:
                        v = row[n]
                        if not pd.isna(v): return float(v)
                    except (KeyError, TypeError, ValueError):
                        pass
                return float('nan')

            _W  = 80
            _fw = (42, 12, 10, 8, 8, 10)
            _fhdr_poly = (
                f"  {'Effect':<{_fw[0]}}  "
                f"{'Estimate':>{_fw[1]}}  "
                f"{'Std Error':>{_fw[2]}}  "
                f"{'DF':>{_fw[3]}}  "
                f"{'t Value':>{_fw[4]}}  "
                f"{'Pr > |t|':>{_fw[5]}}  Sig"
            )
            _poly_model_type = str(_poly_fit_df.get('model', pd.Series(['glmer.nb'])).iloc[0])
            _poly_lines = [
                "",
                "\u2550" * _W,
                "  POLYNOMIAL CONTRASTS FOR WEEK EFFECT  (lme4 glmer.nb / emmeans)",
                f"  Model: {_poly_model_type}  Total_Licks ~ Week_f + (1|ID)",
                "  Family: Negative Binomial (log link)   [Week_f = ordered factor]",
                "  Contrasts: emmeans::contrast(~ Week_f, 'poly')  [on link scale]",
                "\u2550" * _W,
                "",
                "  Fit Statistics",
                f"    {'AIC  (smaller is better)':<40} {_poly_aic:.1f}",
                f"    {'BIC  (smaller is better)':<40} {_poly_bic:.1f}",
                f"    {'-2 Log-Likelihood':<40} {-2 * _poly_loglik:.1f}",
                f"    {'N observations':<40} {_poly_nobs}",
                "",
                "  Solution for Fixed Effects",
                "",
                _fhdr_poly,
                "  " + "\u2500" * (sum(_fw) + 14),
            ]

            for _, _row in _poly_fe_df.iterrows():
                _t_disp = _la_poly_fmt_term(str(_row.get('term', '')))
                _est = _rget_val(_row, 'Estimate')
                _se  = _rget_val(_row, 'Std. Error', 'Std.Error', 'Std..Error')
                _df_ = _rget_val(_row, 'df', 'Df', 'DF')
                _t   = _rget_val(_row, 't value', 't.value', 't_value')
                _p   = _rget_val(_row, 'Pr(>|t|)', 'Pr...t..', 'p.value')
                _poly_lines.append(
                    f"  {_t_disp:<{_fw[0]}}  "
                    f"{_est:>{_fw[1]}.4f}  "
                    f"{_se:>{_fw[2]}.4f}  "
                    f"{_df_:>{_fw[3]}.1f}  "
                    f"{_t:>{_fw[4]}.3f}  "
                    f"{_la_poly_fmt_p(_p):>{_fw[5]}}  "
                    f"{_la_poly_sig_stars(_p)}"
                )

            _cw = (22, 12, 10, 8, 8, 10)
            _chdr_poly = (
                f"  {'Contrast':<{_cw[0]}}  "
                f"{'Estimate':>{_cw[1]}}  "
                f"{'Std Error':>{_cw[2]}}  "
                f"{'DF':>{_cw[3]}}  "
                f"{'t Value':>{_cw[4]}}  "
                f"{'Pr > |t|':>{_cw[5]}}  Sig"
            )
            _poly_lines += [
                "",
                "  Polynomial Contrasts  (Week_f main effect decomposition)",
                "",
                _chdr_poly,
                "  " + "\u2500" * (sum(_cw) + 14),
            ]

            for _, _row in _poly_co_df.iterrows():
                _name = _la_poly_rename_contrast(str(_row.get('contrast', '')))
                _est  = _rget_val(_row, 'estimate')
                _se   = _rget_val(_row, 'SE')
                _df_  = _rget_val(_row, 'df')
                _t    = _rget_val(_row, 't.ratio')
                _p    = _rget_val(_row, 'p.value')
                _poly_lines.append(
                    f"  {_name:<{_cw[0]}}  "
                    f"{_est:>{_cw[1]}.4f}  "
                    f"{_se:>{_cw[2]}.4f}  "
                    f"{_df_:>{_cw[3]}.1f}  "
                    f"{_t:>{_cw[4]}.3f}  "
                    f"{_la_poly_fmt_p(_p):>{_cw[5]}}  "
                    f"{_la_poly_sig_stars(_p)}"
                )

            _poly_lines += [
                "",
                "  Significance: *** p<.001  ** p<.01  * p<.05  . p<.10",
                "",
                "\u2550" * _W,
            ]

            _poly_section = "\n".join(_poly_lines)
            print(_poly_section)
            _mm_report += _poly_section

        except Exception as _poly_exc:
            import traceback as _tb_poly
            _poly_err = f"\n[ERROR] Polynomial contrasts (lmerTest/emmeans) failed: {_poly_exc}\n"
            print(_poly_err)
            _tb_poly.print_exc()
            _mm_report += _poly_err

        # ── Model diagnostics: residuals, Q-Q, scale-location, random effects ──
        try:
            import tempfile as _diag_tmp
            import os as _diag_os

            _diag_dir  = _diag_tmp.gettempdir().replace('\\', '/')
            _diag_uid  = str(id(_pnl_raw))[-6:]
            _diag_csv  = f"{_diag_dir}/la_diag_{_diag_uid}.csv"
            _diag_png  = f"{_diag_dir}/la_diag_{_diag_uid}.png"

            _ro.globalenv['r_diag_csv'] = _diag_csv
            _ro.globalenv['r_diag_png'] = _diag_png

            _ro.r("""
                # Collect residual diagnostics from the already-fitted .model_lme
                .resid_df <- data.frame(
                    fitted    = fitted(.model_lme),
                    residuals = residuals(.model_lme, type = 'pearson'),
                    mouse_id  = .df_lme$mouse_id,
                    week_num  = .df_lme$week_num
                )
                write.csv(.resid_df, r_diag_csv, row.names = FALSE)

                # Four-panel diagnostic figure
                png(r_diag_png, width = 1200, height = 1100, res = 150)
                par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3.5, 1.5), oma = c(0,0,2,0))

                # 1. Residuals vs Fitted
                plot(.resid_df$fitted, .resid_df$residuals,
                     xlab = 'Fitted values', ylab = 'Pearson residuals',
                     main = 'Residuals vs Fitted',
                     pch = 16, col = adjustcolor('steelblue', 0.6), cex = 0.9)
                abline(h = 0, lty = 2, col = 'grey40')
                lines(lowess(.resid_df$fitted, .resid_df$residuals),
                      col = 'firebrick', lwd = 1.8)

                # 2. Normal Q-Q of residuals
                qqnorm(.resid_df$residuals,
                       main = 'Normal Q-Q  (Pearson residuals)',
                       pch = 16, col = adjustcolor('steelblue', 0.6), cex = 0.9)
                qqline(.resid_df$residuals, col = 'firebrick', lwd = 1.8)

                # 3. Scale-Location (sqrt |residuals| vs fitted)
                plot(.resid_df$fitted, sqrt(abs(.resid_df$residuals)),
                     xlab = 'Fitted values',
                     ylab = expression(sqrt("|Pearson residuals|")),
                     main = 'Scale-Location',
                     pch = 16, col = adjustcolor('steelblue', 0.6), cex = 0.9)
                lines(lowess(.resid_df$fitted, sqrt(abs(.resid_df$residuals))),
                      col = 'firebrick', lwd = 1.8)
                abline(h = sqrt(mean(.resid_df$residuals^2)),
                       lty = 2, col = 'grey40')

                # 4. Q-Q of random intercepts (lme4 glmer.nb)
                re_int <- tryCatch(
                    ranef(.model_lme)[["mouse_id"]][["(Intercept)"]],
                    error = function(e) ranef(.model_lme)[[1]][[1]]
                )
                qqnorm(re_int,
                       main = 'Normal Q-Q  (random intercepts)',
                       pch = 16, col = adjustcolor('darkorchid', 0.7), cex = 0.9)
                qqline(re_int, col = 'firebrick', lwd = 1.8)

                mtext('lme4 glmer.nb() — Model Diagnostics', outer = TRUE,
                      cex = 1.1, font = 2)
                dev.off()
            """)

            # Read residuals back for Python-side Shapiro-Wilk & Levene tests
            _diag_df = pd.read_csv(_diag_csv)
            _resid   = _diag_df['residuals'].dropna().values

            # Shapiro-Wilk on residuals
            if 3 <= len(_resid) <= 5000:
                _sw_stat, _sw_p = stats.shapiro(_resid)
                _sw_str = (f"Shapiro-Wilk W = {_sw_stat:.4f},  p = {_sw_p:.4f}  "
                           f"({'normality OK' if _sw_p > 0.05 else 'NON-NORMAL'})")
            else:
                _sw_str = "Shapiro-Wilk: n outside 3-5000 range — skipped"

            # Levene's test for homoscedasticity across weeks
            _lev_groups = [
                _diag_df.loc[_diag_df['week_num'] == _wk, 'residuals'].dropna().values
                for _wk in sorted(_diag_df['week_num'].unique())
                if len(_diag_df.loc[_diag_df['week_num'] == _wk]) >= 2
            ]
            if len(_lev_groups) >= 2:
                _lev_stat, _lev_p = stats.levene(*_lev_groups)
                _lev_str = (f"Levene's test (homoscedasticity across weeks): "
                            f"W = {_lev_stat:.4f},  p = {_lev_p:.4f}  "
                            f"({'homoscedastic' if _lev_p > 0.05 else 'HETEROSCEDASTIC'})")
            else:
                _lev_str = "Levene's test: not enough groups — skipped"

            _diag_summary = "\n".join([
                "",
                "═" * 80,
                "  MODEL DIAGNOSTICS  (lme4 glmer.nb Pearson residuals)",
                "═" * 80,
                "",
                f"  {_sw_str}",
                f"  {_lev_str}",
                "",
                "  Four-panel PNG saved alongside this report:",
                "    Panel 1 — Residuals vs Fitted  (linearity / outlier check)",
                "    Panel 2 — Normal Q-Q of Pearson residuals",
                "    Panel 3 — Scale-Location  (homoscedasticity)",
                "    Panel 4 — Normal Q-Q of random intercepts",
                "  Red line = LOWESS smoother; dashed = reference line.",
                "  Good fit: points scatter randomly around 0 (panels 1 & 3),",
                "  points follow the diagonal (panels 2 & 4).",
                "",
            ])

            print(_diag_summary)
            _mm_report += _diag_summary

            # Copy PNG next to the save_path if available, else print temp location
            if save_path is not None:
                import shutil as _shutil
                _diag_out = save_path.parent / (save_path.stem + '_mixed_model_diagnostics.png')
                _shutil.copy2(_diag_png, str(_diag_out))
                print(f"[OK] Diagnostic plots saved -> {_diag_out}")
                _mm_report += f"  Diagnostic PNG: {_diag_out}\n"
            else:
                print(f"[OK] Diagnostic plots saved (temp) -> {_diag_png}")

            for _fp in [_diag_csv, _diag_png]:
                try:    _diag_os.unlink(_fp)
                except Exception: pass

        except Exception as _diag_exc:
            import traceback as _tb_diag
            print(f"[WARNING] Diagnostic plots failed: {_diag_exc}")
            _tb_diag.print_exc()

    # ══════════════════════════════════════════════════════════════════════
    # PATH B: linearmodels RandomEffects fallback (random intercept + AR(1) SE)
    # ══════════════════════════════════════════════════════════════════════
    else:
        try:
            from linearmodels.panel import RandomEffects as _REModel
        except ImportError:
            print("NOTE: Neither rpy2+R (recommended) nor linearmodels is installed.\n"
                  "  • For the full random-slopes + AR(1) model:  install R, then pip install rpy2\n"
                  "  • For the random-intercept fallback:          pip install linearmodels")
            return fig

        print("\n" + "=" * 80)
        print("MIXED EFFECTS MODEL (fallback): linearmodels RandomEffects")
        print("  Random intercept per mouse; AR(1)-robust kernel SE")
        print("  NOTE: Install R + rpy2 for random slopes + true AR(1) covariance")
        print("=" * 80)

        _pnl = (_pnl_raw
                .sort_values(['mouse_id', 'time_idx'])
                .set_index(['mouse_id', 'time_idx']))
        _mod = _REModel.from_formula('licks ~ 1 + weight_pct + week_num', data=_pnl)
        try:
            _res = _mod.fit(cov_type='kernel', bandwidth=1)
        except TypeError:
            _res = _mod.fit(cov_type='kernel')

        print(_res.summary)

        _vd          = _res.variance_decomposition
        _var_between = float(_vd['Effects'])
        _var_within  = float(_vd['Residual'])
        _icc1        = _var_between / (_var_between + _var_within)

        _b0   = float(_res.params['Intercept'])
        _b_wt = float(_res.params['weight_pct'])
        _b_wk = float(_res.params['week_num'])
        _p_wt = float(_res.pvalues['weight_pct'])
        _mean_wk = float(_pnl_raw['week_num'].mean())

        _b_vec    = np.array([_b0, _b_wt, _b_wk])
        _X_mat    = np.column_stack([np.ones(len(_pnl_raw)),
                                     _pnl_raw['weight_pct'].values,
                                     _pnl_raw['week_num'].values])
        _sigma_f2    = float(np.var(_X_mat @ _b_vec))
        _total_var   = _sigma_f2 + _var_between + _var_within
        _r2_marginal = _sigma_f2 / _total_var
        _r2_cond     = (_sigma_f2 + _var_between) / _total_var

        _model_label = "RE intercept/mouse, AR(1)-robust SE  (fallback: install R+rpy2)"

        print(f"\nICC(1)         = {_icc1:.3f}  (between-mouse / total variance)")
        print(f"Marginal  R²   = {_r2_marginal:.3f}  (fixed effects only)")
        print(f"Conditional R² = {_r2_cond:.3f}  (fixed + random effects)")
        print("=" * 80 + "\n")

        _mm_report = "\n".join([
            "=" * 80,
            "MIXED EFFECTS MODEL (fallback): linearmodels RandomEffects",
            "  licks ~ weight_pct + week_num",
            "  Random intercept per mouse; AR(1)-robust kernel SE",
            "  NOTE: Install R + rpy2 for random slopes + true AR(1) covariance",
            "=" * 80,
            "",
            str(_res.summary),
            "",
            f"ICC(1)         = {_icc1:.3f}  (between-mouse / total variance)",
            f"Marginal  R\u00b2   = {_r2_marginal:.3f}  (fixed effects only)",
            f"Conditional R\u00b2 = {_r2_cond:.3f}  (fixed + random effects)",
            "=" * 80,
            "",
            f"n = {_pnl_raw.shape[0]} observations,  {n_animals} mice",
            f"Weeks: {len(sorted_dates)}  (mean week used for R\u00b2 = {_mean_wk:.1f})",
        ])

    # ── Shared plot: population fixed-effect trend + raw observations ──────
    _wt_grid  = np.linspace(_pnl_raw['weight_pct'].min(),
                             _pnl_raw['weight_pct'].max(), 200)
    _fit_line = _b0 + _b_wt * _wt_grid + _b_wk * _mean_wk

    fig2, ax2 = plt.subplots()
    _ms2 = plt.rcParams.get('lines.markersize', 6)

    for _aid in unique_animals:
        _rows = _pnl_raw[_pnl_raw['mouse_id'] == _aid]
        if _rows.empty:
            continue
        ax2.scatter(_rows['weight_pct'], _rows['licks'],
                    color=animal_color[_aid], marker='o', s=_ms2 ** 2,
                    edgecolors='black', linewidths=0.4, alpha=0.5, zorder=3)

    ax2.plot(_wt_grid, _fit_line, color='black', zorder=5)

    _p_str = f"p = {_p_wt:.4f}" if _p_wt >= 0.0001 else "p < 0.0001"
    ax2.text(0.03, 0.97,
             f"\u03b2_weight = {_b_wt:.2f}\n{_p_str}\nn = {_pnl_raw.shape[0]} obs,  {n_animals} mice",
             transform=ax2.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax2.set_xlabel('Total Weight Change (%)', weight='bold')
    ax2.set_ylabel('Total Lick Count', weight='bold')
    ax2.set_title(f'Mixed Effects: Licks ~ Weight + Week\n'
                  f'({design_label}, {_model_label})',
                  weight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(direction='in', which='both', length=5)

    plt.tight_layout()

    if save_path:
        _mm_path = save_path.parent / (save_path.stem + '_mixed_model' + save_path.suffix)
        fig2.savefig(_mm_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"Mixed model figure saved to: {_mm_path}")
        _mm_txt = _mm_path.with_suffix('.txt')
        _mm_txt.write_text(_mm_report, encoding='utf-8')
        print(f"Mixed model report saved to: {_mm_txt}")

    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig


def plot_rmcorr_licks_vs_weight(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Repeated-measures correlation (rmcorr) of lick count vs weight change.

    Uses the R ``rmcorr`` package (Bakdash & Marusich 2017) which fits an
    ANCOVA with participant as a factor, producing one common within-subject
    slope and per-subject parallel regression lines.  The result is a true
    repeated-measures correlation coefficient (r_rm) that accounts for the
    non-independence of observations from the same mouse.

    Requires rpy2 and the R package ``rmcorr`` (install.packages('rmcorr')).
    Falls back to a plain Spearman scatter if rpy2 / rmcorr are unavailable.

    Saves:
        <save_path.stem>_rmcorr.svg   — matplotlib figure
        <save_path.stem>_rmcorr.txt   — text report with r_rm, CI, p-value
    """
    import os as _os
    import traceback as _tb
    try:
        return _plot_rmcorr_licks_vs_weight_impl(weekly_averages, save_path=save_path, show=show)
    except Exception:
        print("ERROR in plot_rmcorr_licks_vs_weight:")
        _tb.print_exc()
        return None


def _plot_rmcorr_licks_vs_weight_impl(
    weekly_averages: Dict,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    import os as _os

    # ── build flat per-animal-per-week data ──────────────────────────────
    if EXPERIMENT_MODE == 'ramp':
        sorted_dates = sorted(weekly_averages.keys(),
                              key=lambda d: weekly_averages[d]['ca_percent'])
    else:
        sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    records = []
    for week_idx, date in enumerate(sorted_dates):
        data = weekly_averages[date]
        lick_arr = np.asarray(data['avg_licks_per_animal'], dtype=float)
        wt_arr   = np.asarray(data['avg_total_weight_per_animal'], dtype=float)
        ids      = data.get('animal_ids',
                            [f"Animal_{j+1}" for j in range(len(lick_arr))])
        for aid, lk, wt in zip(ids, lick_arr, wt_arr):
            records.append({'mouse_id': str(aid), 'licks': lk, 'weight_pct': wt})

    if not records:
        print("ERROR: No per-animal data for rmcorr.")
        return None

    _df = pd.DataFrame(records).dropna(subset=['licks', 'weight_pct'])
    if _df.shape[0] < 5 or _df['mouse_id'].nunique() < 3:
        print("Not enough data for rmcorr (need ≥3 mice, ≥5 observations).")
        return None

    unique_animals = list(_df['mouse_id'].unique())
    n_animals      = len(unique_animals)
    cmap           = plt.cm.tab20
    animal_color   = {aid: cmap(i % 20) for i, aid in enumerate(unique_animals)}
    design_label   = _MLB['plot_suffix']

    # ── attempt rmcorr via rpy2 ──────────────────────────────────────────
    _r_rm  = np.nan
    _p_rm  = np.nan
    _ci_lo = np.nan
    _ci_hi = np.nan
    _slope = np.nan
    _per_subject_intercepts: dict = {}
    _rmcorr_ok = False
    _rmcorr_report = ""

    try:
        import os as _osi
        import glob as _globi

        if _osi.name == 'nt':
            _r_home = _osi.environ.get('R_HOME', '')
            if _r_home:
                for _p in [_osi.path.join(_r_home, 'bin', 'x64'),
                            _osi.path.join(_r_home, 'bin')]:
                    if _osi.path.isdir(_p):
                        if _p not in _osi.environ.get('PATH', ''):
                            _osi.environ['PATH'] = _p + _osi.pathsep + _osi.environ.get('PATH', '')
                        try:
                            _osi.add_dll_directory(_p)
                        except (AttributeError, OSError):
                            pass

        import rpy2.robjects as _ro2
        import rpy2.robjects.packages as _rpkgs2
        _rpkgs2.importr('rmcorr')

        import tempfile as _tmpi
        _uid  = str(id(_df))[-6:]
        _tdir = _tmpi.gettempdir().replace('\\', '/')
        _csv  = f"{_tdir}/rmcorr_in_{_uid}.csv"
        _out  = f"{_tdir}/rmcorr_out_{_uid}.csv"

        _df.to_csv(_csv, index=False)
        _ro2.globalenv['r_rmc_in']  = _csv
        _ro2.globalenv['r_rmc_out'] = _out

        _ro2.r("""
            suppressPackageStartupMessages(library(rmcorr))
            .d <- read.csv(r_rmc_in)
            .d$mouse_id <- factor(.d$mouse_id)
            .rmc <- rmcorr(participant = mouse_id,
                           measure1    = weight_pct,
                           measure2    = licks,
                           dataset     = .d)
            .all_coefs <- stats::coef(.rmc$model)
            .slope     <- as.numeric(.all_coefs["weight_pct"])
            .lvls <- levels(.d$mouse_id)
            .intercepts <- sapply(.lvls, function(.lv) {
                sx <- .d$weight_pct[.d$mouse_id == .lv]
                sy <- .d$licks[.d$mouse_id == .lv]
                if (length(sx) == 0 || is.na(.slope)) return(NA_real_)
                mean(sy) - .slope * mean(sx)
            })
            .out_df <- data.frame(
                mouse_id  = .lvls,
                intercept = as.numeric(.intercepts),
                slope     = .slope,
                r_rm      = .rmc$r,
                p_val     = .rmc$p,
                ci_lo     = .rmc$CI[1],
                ci_hi     = .rmc$CI[2],
                df_rm     = .rmc$df
            )
            write.csv(.out_df, r_rmc_out, row.names = FALSE)
        """)

        _res_df = pd.read_csv(_out)
        for _fp in [_csv, _out]:
            try:    _os.unlink(_fp)
            except Exception: pass

        _r_rm  = float(_res_df['r_rm'].iloc[0])
        _p_rm  = float(_res_df['p_val'].iloc[0])
        _ci_lo = float(_res_df['ci_lo'].iloc[0])
        _ci_hi = float(_res_df['ci_hi'].iloc[0])
        _df_rm = float(_res_df['df_rm'].iloc[0])
        _slope = float(_res_df['slope'].iloc[0])
        _per_subject_intercepts = dict(
            zip(_res_df['mouse_id'].astype(str), _res_df['intercept'].astype(float))
        )
        print(f"  [debug] slope={_slope:.4f}  n_intercepts={len(_per_subject_intercepts)}"
              f"  intercept_keys={list(_per_subject_intercepts.keys())[:3]}")
        # If R returned NaN for slope, compute it in Python from the rmcorr r value
        # using the pooled within-subject regression: slope = r_rm * SD(y)/SD(x)
        if np.isnan(_slope) and not np.isnan(_r_rm):
            _sx = _df.groupby('mouse_id')['weight_pct'].apply(lambda v: v - v.mean()).std()
            _sy = _df.groupby('mouse_id')['licks'].apply(lambda v: v - v.mean()).std()
            if _sx > 0:
                _slope = _r_rm * float(_sy) / float(_sx)
                print(f"  [debug] slope recomputed from r_rm: {_slope:.4f}")
            # recompute intercepts with new slope
            _per_subject_intercepts = {
                str(aid): (float(_df.loc[_df['mouse_id'] == aid, 'licks'].mean())
                           - _slope * float(_df.loc[_df['mouse_id'] == aid, 'weight_pct'].mean()))
                for aid in _df['mouse_id'].unique()
            }
        _rmcorr_ok = True
        print(f"\nrmcorr: r_rm = {_r_rm:.4f},  p = {_p_rm:.4f},  "
              f"95% CI [{_ci_lo:.4f}, {_ci_hi:.4f}],  df = {_df_rm:.0f}")

    except Exception as _rmc_exc:
        print(f"NOTE: rmcorr via rpy2 failed ({type(_rmc_exc).__name__}: {_rmc_exc})")
        print("      Falling back to Spearman scatter (install R package 'rmcorr' for full output).")

    # ── matplotlib figure ─────────────────────────────────────────────────
    fig, ax = plt.subplots()
    _ms = plt.rcParams.get('lines.markersize', 6)

    # scatter points
    for _, row in _df.iterrows():
        ax.scatter(row['weight_pct'], row['licks'],
                   color=animal_color[row['mouse_id']], marker='o',
                   s=_ms ** 2, edgecolors='black', linewidths=0.5,
                   alpha=0.8, zorder=4)

    if _rmcorr_ok:
        # draw single rmcorr line through the grand mean
        if not np.isnan(_slope):
            x_min = _df['weight_pct'].min()
            x_max = _df['weight_pct'].max()
            x_pad = (x_max - x_min) * 0.05
            xs = np.linspace(x_min - x_pad, x_max + x_pad, 200)
            _grand_intercept = (_df['licks'].mean()
                                - _slope * _df['weight_pct'].mean())
            ax.plot(xs, _grand_intercept + _slope * xs,
                    color='black', linewidth=1.2, zorder=4)

        _p_str = f"p = {_p_rm:.4f}" if _p_rm >= 0.0001 else "p < 0.0001"
        _ann = (f"$r_{{rm}}$ = {_r_rm:.3f}\n"
                f"{_p_str}\n"
                f"95% CI [{_ci_lo:.3f}, {_ci_hi:.3f}]\n"
                f"n = {n_animals} mice")
    else:
        # fallback: Spearman trend line
        _valid = _df[['weight_pct', 'licks']].dropna()
        if len(_valid) >= 3:
            _rho, _pv = stats.spearmanr(_valid['weight_pct'], _valid['licks'])
            _sl, _ic, *_ = stats.linregress(_valid['weight_pct'], _valid['licks'])
            xs = np.linspace(_valid['weight_pct'].min(), _valid['weight_pct'].max(), 200)
            ax.plot(xs, _sl * xs + _ic, color='dimgray', linestyle='--', zorder=2)
            _p_str = f"p = {_pv:.4f}" if _pv >= 0.0001 else "p < 0.0001"
            _ann = (f"Spearman \u03c1 = {_rho:.3f}\n{_p_str}\nn = {n_animals} mice\n"
                    f"(rmcorr unavailable)")
        else:
            _ann = f"n = {n_animals} mice\n(rmcorr unavailable)"

    ax.text(0.03, 0.97, _ann,
            transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='gray'))

    ax.set_xlabel('Total Weight Change (%)', weight='bold')
    ax.set_ylabel('Total Lick Count', weight='bold')
    ax.set_title(f'Repeated-Measures Correlation: Licks ~ Weight Change\n({design_label})',
                 weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────
    if save_path is not None:
        _rmc_svg = save_path.parent / (save_path.stem + '_rmcorr.svg')
        fig.savefig(_rmc_svg, format='svg', dpi=200, bbox_inches='tight')
        print(f"rmcorr figure saved -> {_rmc_svg}")

        _rmc_txt = save_path.parent / (save_path.stem + '_rmcorr.txt')
        if _rmcorr_ok:
            _p_str_txt = f"p = {_p_rm:.4f}" if _p_rm >= 0.0001 else "p < 0.0001"
            _rmc_txt.write_text("\n".join([
                "=" * 70,
                "REPEATED-MEASURES CORRELATION  (Bakdash & Marusich 2017)",
                "  rmcorr(participant=mouse_id, measure1=weight_pct, measure2=licks)",
                "  Package: R::rmcorr",
                "=" * 70,
                "",
                f"  r_rm  = {_r_rm:.4f}",
                f"  {_p_str_txt}",
                f"  95% CI  [{_ci_lo:.4f}, {_ci_hi:.4f}]",
                f"  df    = {_df_rm:.0f}",
                f"  n mice  = {n_animals}",
                f"  n obs   = {_df.shape[0]}",
                "",
                "  Interpretation: r_rm is the within-individual Pearson correlation",
                "  after removing between-subject variance (ANCOVA with participant",
                "  as factor). Parallel lines in the plot share one common slope.",
                "=" * 70,
            ]), encoding='utf-8')
        else:
            _rmc_txt.write_text("rmcorr not available — install R package 'rmcorr' and rpy2.\n",
                                encoding='utf-8')
        print(f"rmcorr report saved  -> {_rmc_txt}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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
    if EXPERIMENT_MODE == 'ramp':
        print("\nStep 5: Performing Mixed ANOVA (Sex × CA%) Analysis")
    else:
        print("\nStep 5: Performing One-Way ANOVA Analysis")
    print("-" * 40)
    
    anova_results = perform_anova_analysis(weekly_averages)
    anova_output = display_anova_results(anova_results)
    
    # Perform Bonferroni post-hoc for significant results
    print("\nStep 6: Performing Bonferroni Post-Hoc Tests")
    print("-" * 40)
    
    bonferroni_results = perform_bonferroni_posthoc(anova_results, weekly_averages)
    bonferroni_output = display_bonferroni_results(bonferroni_results)
    
    # Plot interaction effects (ramp mode only)
    interaction_figures = {}
    if EXPERIMENT_MODE == 'ramp':
        print("\nStep 6b: Plotting Interaction Effects")
        print("-" * 40)
        interaction_figures = plot_interaction_effects(anova_results, weekly_averages, show=True)
    
    # Perform front-loading ANOVA and Bonferroni post-hoc
    fl_step_a = '6c' if EXPERIMENT_MODE == 'ramp' else '6b'
    fl_step_b = '6d' if EXPERIMENT_MODE == 'ramp' else '6c'
    print(f"\nStep {fl_step_a}: Performing Front-Loading ANOVA")
    print("-" * 40)
    
    frontloading_anova_results = perform_frontloading_anova(weekly_averages)
    frontloading_anova_output = display_frontloading_anova_results(frontloading_anova_results)
    
    print(f"\nStep {fl_step_b}: Performing Bonferroni Post-Hoc for Front-Loading Measures")
    print("-" * 40)
    
    frontloading_bonferroni_results = perform_frontloading_bonferroni_posthoc(frontloading_anova_results, weekly_averages)
    frontloading_bonferroni_output = display_frontloading_bonferroni_results(frontloading_bonferroni_results)

    # Perform two-way mixed ANOVA (Sex × Week/CA%) for front-loading measures
    fl_step_c = '6e' if EXPERIMENT_MODE == 'ramp' else '6d'
    fl_step_d = '6f' if EXPERIMENT_MODE == 'ramp' else '6e'
    print(f"\nStep {fl_step_c}: Performing Front-Loading Mixed ANOVA (Sex × {_MLB['factor']})")
    print("-" * 40)

    frontloading_mixed_anova_results = perform_frontloading_mixed_anova(weekly_averages)
    frontloading_mixed_anova_output = display_frontloading_mixed_anova_results(frontloading_mixed_anova_results)

    print(f"\nStep {fl_step_d}: Performing Post-Hoc Tests for Front-Loading Mixed ANOVA")
    print("-" * 40)

    frontloading_mixed_posthoc_results = perform_frontloading_mixed_posthoc(frontloading_mixed_anova_results)
    frontloading_mixed_posthoc_output = display_frontloading_mixed_posthoc_results(frontloading_mixed_posthoc_results)

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
    
    # Plot comprehensive lick rate histogram
    if EXPERIMENT_MODE == 'ramp':
        print("\nPlotting comprehensive lick rate histogram (all CA% combined)...")
        plot_comprehensive_lick_rate_by_ca(lick_rate_data, show=True)
    else:
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
    save_table = input("\nSave weekly averages, ANOVA, and Bonferroni post-hoc results as text file? (y/n): ").strip().lower()
    if save_table in ['y', 'yes']:
        table_path = master_csv.parent / "comprehensive_statistical_analysis_summary.txt"
        # Combine all outputs (excluding front-loading metrics which are in separate report)
        combined_output = formatted_output + "\n" + anova_output + "\n" + bonferroni_output
        save_weekly_averages_to_file(weekly_averages, combined_output, table_path)
        print(f"Comprehensive statistical analysis saved to: {table_path}")
    
    # Always save front-loading analysis to separate report
    frontloading_path = master_csv.parent / "frontloading_analysis_report.txt"
    save_frontloading_analysis_to_file(
        weekly_averages,
        frontloading_anova_output,
        frontloading_path,
        frontloading_bonferroni_output,
        frontloading_mixed_anova_output,
        frontloading_mixed_posthoc_output,
    )
    print(f"Front-loading analysis report saved to: {frontloading_path}")
    
    # Always save brief lick summary
    brief_path = master_csv.parent / "weekly_lick_summary_brief.txt"
    save_brief_lick_summary(weekly_averages, brief_path)
    print(f"Brief lick summary saved to: {brief_path}")

    # Optional: Save statistical test registry
    save_registry = input("\nSave statistical test registry (methods documentation)? (y/n): ").strip().lower()
    if save_registry in ['y', 'yes']:
        registry_path = master_csv.parent / "statistical_test_registry.txt"
        registry_report = generate_test_registry_report(save_path=registry_path)
        print(registry_report)
    
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
        if EXPERIMENT_MODE == 'ramp':
            comprehensive_save_path = master_csv.parent / "lick_rate_comprehensive_all_ca.svg"
            fig_comp = plot_comprehensive_lick_rate_by_ca(lick_rate_data, save_path=comprehensive_save_path, show=False)
        else:
            comprehensive_save_path = master_csv.parent / "lick_rate_comprehensive_all_weeks.svg"
            fig_comp = plot_comprehensive_lick_rate(lick_rate_data, save_path=comprehensive_save_path, show=False)
        plt.close(fig_comp)  # Close after saving
        print(f"Saved {len(sorted_dates)} individual week plots + 1 comprehensive plot")
    
    # Optional: Save interaction plots (ramp mode only)
    if EXPERIMENT_MODE == 'ramp' and interaction_figures:
        save_interaction = input("\nSave interaction plots as SVG? (y/n): ").strip().lower()
        if save_interaction in ['y', 'yes']:
            interaction_figs = plot_interaction_effects(anova_results, weekly_averages,
                                                       save_dir=master_csv.parent, show=False)
            for fig in interaction_figs.values():
                plt.close(fig)
            print(f"Saved {len(interaction_figs)} interaction plots")
    
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

    # Optional: Save lick vs weight correlation plot
    save_corr = input("\nSave licks vs total weight change (%) correlation plot as SVG? (y/n): ").strip().lower()
    if save_corr in ['y', 'yes']:
        corr_save_path = master_csv.parent / "licks_vs_weight_correlation.svg"
        fig_corr = plot_licks_vs_weight_correlation(weekly_averages, save_path=corr_save_path, show=False)
        if fig_corr:
            plt.close(fig_corr)
            print(f"Saved licks vs weight correlation plot to: {corr_save_path}")

    # Optional: Repeated-measures correlation (rmcorr)
    save_rmc = input("\nRun repeated-measures correlation (rmcorr) of licks vs weight? (y/n): ").strip().lower()
    if save_rmc in ['y', 'yes']:
        rmc_save_path = master_csv.parent / "licks_vs_weight_correlation.svg"
        fig_rmc = plot_rmcorr_licks_vs_weight(weekly_averages, save_path=rmc_save_path, show=False)
        if fig_rmc:
            plt.close(fig_rmc)

    # Optional: Run normality tests on lick measures
    run_normality = input("\nRun Shapiro-Wilk & Levene's normality tests on lick measures? (y/n): ").strip().lower()
    if run_normality in ['y', 'yes']:
        normality_path = master_csv.parent / "lick_normality_report.txt"
        normality_report = generate_lick_normality_report(weekly_averages, save_path=normality_path)
        print(normality_report)

        # Optional: Save Q-Q plots for visual normality inspection
        run_qq = input("\nSave Q-Q plots for lick normality measures as SVG? (y/n): ").strip().lower()
        if run_qq in ['y', 'yes']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qq_dir = master_csv.parent / f"lick_normality_qq_{timestamp}"
            generate_lick_normality_qq_plots(weekly_averages, save_dir=qq_dir, show=False)
            print(f"[OK] Q-Q plots saved to: {qq_dir}")

    # Optional: Generate descriptive statistics report
    run_desc = input("\nGenerate per-week descriptive statistics report (lick/bout/fecal counts + 5-min %)? (y/n): ").strip().lower()
    if run_desc in ['y', 'yes']:
        generate_lick_descriptive_stats_report(weekly_averages, save_report=True)

    # Optional: R-based polynomial contrasts (lme4 / lmerTest / emmeans)
    if HAS_RPY2:
        _poly_q = input(
            "\nRun R-based lmer polynomial contrasts for week trends "
            "(linear/quadratic/cubic)? (y/n): "
        ).strip().lower()
    else:
        _poly_q = 'n'
        print("\n[INFO] rpy2 not installed — skipping polynomial contrasts.")
        print("       Install with:  pip install rpy2")
        print("       Also requires: install.packages(c('lme4','lmerTest','emmeans'))  in R")
    if _poly_q in ['y', 'yes']:
        _ts_poly = datetime.now().strftime("%Y%m%d_%H%M%S")
        _poly_save = master_csv.parent / f"lick_poly_contrasts_{_ts_poly}.txt"
        test_week_polynomial_contrasts_r(
            weekly_averages=weekly_averages,
            save_path=_poly_save,
        )

    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Analyzed: Licks, Bouts, Fecal Counts, Bottle Weight Loss, Total Weight Loss")
    if EXPERIMENT_MODE == 'ramp':
        print(f"Statistical Analysis: Mixed ANOVA (Sex × CA%) and Bonferroni post-hoc tests completed")
        print(f"All pairwise comparisons identified for significant measures")
        if interaction_figures:
            print(f"Interaction plots created for {len(interaction_figures)} measure(s) with significant Sex × CA% interactions")
    else:
        print(f"Statistical Analysis: One-way ANOVA across weeks and Bonferroni post-hoc tests completed")
        print(f"All pairwise week comparisons identified for significant measures")
    print(f"Lick Rate Analysis: 5-minute bins over 30 minutes for each week + comprehensive combined plot")
    
    return weekly_results, weekly_averages, anova_results, bonferroni_results


# =============================================================================
# R-BASED POLYNOMIAL CONTRASTS  (lme4 / lmerTest / emmeans)
# =============================================================================

def _la_poly_fmt_p(p) -> str:
    """Format a p-value for the PROC MIXED-style report."""
    try:
        if pd.isna(p): return 'n/a'
        p = float(p)
        if p < 0.0001: return '< .0001'
        return f'{p:.4f}'
    except (TypeError, ValueError):
        return 'n/a'


def _la_poly_sig_stars(p) -> str:
    """Return significance stars for a p-value."""
    try:
        if pd.isna(p): return ''
        p = float(p)
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        if p < 0.10:  return '.'
        return ''
    except (TypeError, ValueError):
        return ''


def _la_poly_rename_contrast(name: str) -> str:
    """Capitalise an emmeans polynomial contrast label."""
    name = str(name).strip()
    _map = {'linear': 'Linear', 'quadratic': 'Quadratic',
            'cubic': 'Cubic', 'quartic': 'Quartic', 'quintic': 'Quintic'}
    return _map.get(name.lower(), name.capitalize())


def _la_poly_fmt_term(term: str) -> str:
    """Convert lmerTest term names to readable display strings."""
    term = str(term)
    _replacements = [
        ('(Intercept)', 'Intercept'),
        ('Week_f.L',    'Week_f  [linear]'),
        ('Week_f.Q',    'Week_f  [quadratic]'),
        ('Week_f.C',    'Week_f  [cubic]'),
        ('Week_f^4',    'Week_f  [degree 4]'),
        ('Week_f^5',    'Week_f  [degree 5]'),
    ]
    for old, new in _replacements:
        if old in term:
            term = term.replace(old, new)
            break
    return term


def _build_lmer_df(weekly_averages: Dict) -> pd.DataFrame:
    """Convert weekly_averages dict to a long-format DataFrame for lmer.

    Returns one row per animal per week with columns:
        ID, Week, Sex, CA_Percent, Total_Licks, Total_Bouts,
        First_5min_Lick_Pct, Time_to_50pct_Licks
    """
    rows = []
    sorted_dates = sort_dates_chronologically(list(weekly_averages.keys()))

    for week_idx, date in enumerate(sorted_dates):
        wk = weekly_averages[date]
        animal_ids  = wk.get('animal_ids', [])
        animal_sexes = wk.get('animal_sexes', [])
        ca_pct      = float(wk.get('ca_percent', np.nan))

        licks    = np.asarray(wk.get('avg_licks_per_animal', []), dtype=float)
        bouts    = np.asarray(wk.get('avg_bouts_per_animal', []), dtype=float)
        fl_pct   = np.asarray(wk.get('first_5min_lick_pcts_per_animal', []), dtype=float)
        t50      = np.asarray(wk.get('time_to_50pct_licks_per_animal', []), dtype=float)

        n = len(licks)
        if n == 0:
            continue

        # pad metadata arrays if shorter than lick array
        ids   = list(animal_ids)  + [f'Animal_{i+1}' for i in range(len(animal_ids), n)]
        sexes = list(animal_sexes)+ ['Unknown' for _ in range(len(animal_sexes), n)]

        for i in range(n):
            rows.append({
                'ID':                  str(ids[i]),
                'Week':                week_idx + 1,   # 1-based
                'Sex':                 str(sexes[i]),
                'CA_Percent':          ca_pct,
                'Total_Licks':         licks[i] if i < len(licks) else np.nan,
                'Total_Bouts':         bouts[i] if i < len(bouts) else np.nan,
                'First_5min_Lick_Pct': fl_pct[i] if i < len(fl_pct) else np.nan,
                'Time_to_50pct_Licks': t50[i]   if i < len(t50)   else np.nan,
            })

    return pd.DataFrame(rows)


def test_week_polynomial_contrasts_r(
    weekly_averages: Dict,
    measures: Optional[List[str]] = None,
    max_degree: int = 3,
    save_path: Optional[Path] = None,
) -> Dict:
    """Test polynomial trends for the Week / CA% factor via R lme4 / lmerTest / emmeans.

    Fits the model in R:

        measure ~ Week_f + Sex + (1|ID)          [nonramp: Week is the repeated factor]
        measure ~ CA_Percent_f + Sex + (1|ID)    [ramp:    CA% is the repeated factor]

    where the within-subjects factor is coded as an ordered factor so that
    emmeans can decompose its main effect into orthogonal polynomial components
    (linear, quadratic, cubic, …).

    The report is formatted to mirror SAS PROC MIXED output:
        Effect | Estimate | Std Error | DF | t Value | Pr > |t|

    Parameters
    ----------
    weekly_averages : Dict
        Output of compute_weekly_averages().
    measures : list of str, optional
        Outcome measures to analyse. Defaults to Total_Licks, Total_Bouts,
        First_5min_Lick_Pct, Time_to_50pct_Licks (whichever are present).
    max_degree : int
        Maximum polynomial degree (default 3 → linear/quadratic/cubic).
        Capped at n_levels − 1 automatically.
    save_path : Path, optional
        If provided the full report is written to this path.

    Returns
    -------
    dict  with keys  'report' (str) and 'measures' (dict of per-measure results).
    """
    W = 80

    print("\n" + "=" * W)
    print("R-BASED LMER: POLYNOMIAL CONTRASTS FOR TIME  (PROC MIXED STYLE)")
    print("=" * W)

    if not HAS_RPY2:
        print(
            "[ERROR] rpy2 is not installed — cannot run R-based analysis.\n"
            "  pip install rpy2\n"
            "  Also requires R with:  install.packages(c('lme4','lmerTest','emmeans'))"
        )
        return {'error': 'rpy2 not installed'}

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    import tempfile
    import os

    # ── rpy2 converter ─────────────────────────────────────────────────
    try:
        from rpy2.robjects.conversion import localconverter
        def _to_r(df):
            with localconverter(ro.default_converter + pandas2ri.converter):
                return ro.conversion.py2rpy(df)
    except Exception:
        pandas2ri.activate()
        def _to_r(df):
            return pandas2ri.py2rpy(df)

    def _rget(row, *names):
        for n in names:
            try:
                v = row[n]
                if not pd.isna(v): return float(v)
            except (KeyError, TypeError, ValueError):
                pass
        return np.nan

    # ── build long-format dataframe ─────────────────────────────────────
    print("\nStep 1: Building long-format DataFrame from weekly_averages...")
    long_df = _build_lmer_df(weekly_averages)

    if len(long_df) == 0:
        print("[ERROR] No data could be extracted from weekly_averages.")
        return {'error': 'empty dataframe'}

    # Determine within-subjects factor
    within_col   = 'CA_Percent' if EXPERIMENT_MODE == 'ramp' else 'Week'
    within_label = 'CA%'        if EXPERIMENT_MODE == 'ramp' else 'Week'
    factor_vals  = sorted(long_df[within_col].dropna().unique())
    n_levels     = len(factor_vals)
    _max_deg     = min(max_degree, n_levels - 1)

    has_sex = 'Sex' in long_df.columns and long_df['Sex'].nunique() > 1

    print(f"  {within_label} levels : {factor_vals}")
    print(f"  N animals      : {long_df['ID'].nunique()}")
    print(f"  Sex factor     : {has_sex}")
    print(f"  Max poly degree: {_max_deg}")

    # ── default measures ────────────────────────────────────────────────
    if measures is None:
        _candidates = ['Total_Licks', 'Total_Bouts',
                       'First_5min_Lick_Pct', 'Time_to_50pct_Licks']
        measures = [m for m in _candidates if m in long_df.columns]
        if not measures:
            measures = ['Total_Licks']

    # ── verify R packages ───────────────────────────────────────────────
    print("\nStep 2: Verifying R packages (lme4, lmerTest, emmeans)...")
    try:
        importr('lme4')
        importr('lmerTest')
        importr('emmeans')
        print("  [OK] All required R packages available.")
    except Exception as _e:
        print(f"[ERROR] Required R packages not found: {_e}")
        print("  Install in R:  install.packages(c('lme4', 'lmerTest', 'emmeans'))")
        return {'error': f'R packages unavailable: {_e}'}

    _MEASURE_LABELS = {
        'Total_Licks':         'Total Licks',
        'Total_Bouts':         'Total Lick Bouts',
        'First_5min_Lick_Pct': '% Licks in First 5 min',
        'Time_to_50pct_Licks': 'Time to 50% Licks (min)',
    }

    results_by_measure: Dict = {}
    report_sections: List[str] = []
    _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_sections += [
        "=" * W,
        "  LMER POLYNOMIAL CONTRASTS  (R: lme4 / lmerTest / emmeans)",
        "  Output formatted to match SAS PROC MIXED",
        "=" * W,
        f"  Generated         : {_ts}",
        f"  Experiment mode   : {EXPERIMENT_MODE}",
        f"  Within-subj factor: {within_label}  (levels: {factor_vals})",
        f"  N animals         : {long_df['ID'].nunique()}",
        f"  Max poly degree   : {_max_deg}",
        "",
        "  Fixed-effects model:",
        f"    measure ~ {within_col}_f + Sex + (1|ID)   [Sex omitted if single-sex]",
        f"    {within_col}_f = ordered factor",
        "    (1|ID)  = random intercept per animal   Method = REML",
        "    DF      = Satterthwaite approximation  (lmerTest)",
        "    Contrasts via emmeans::contrast(..., 'poly')",
        "=" * W,
        "",
        "  Significance: *** p<.001  ** p<.01  * p<.05  . p<.10",
        "",
    ]

    for measure in measures:
        if measure not in long_df.columns:
            print(f"\n  [SKIP] '{measure}' not in dataframe.")
            continue

        print(f"\nStep 3 [{measure}]: Fitting lmer model in R...")

        req_cols = ['ID', within_col, measure]
        if has_sex:
            req_cols.append('Sex')
        adf = long_df[req_cols].dropna().copy()
        if 'Sex' not in adf.columns:
            adf['Sex'] = 'Unknown'
        adf['_factor_val'] = adf[within_col].astype(str)

        if len(adf) < 6:
            print(f"  [WARNING] Only {len(adf)} observations — skipping.")
            continue

        m_has_sex     = has_sex and adf['Sex'].nunique() > 1
        measure_label = _MEASURE_LABELS.get(measure, measure.replace('_', ' '))
        factor_col_r  = 'within_f'
        formula_disp  = (
            f"{measure} ~ {within_col}_f + Sex + (1|ID)"
            if m_has_sex else
            f"{measure} ~ {within_col}_f + (1|ID)"
        )

        _tmpdir = tempfile.gettempdir().replace('\\', '/')
        _uid    = str(id(adf))[-6:]
        _fe_p   = f"{_tmpdir}/la_lmer_fe_{measure}_{_uid}.csv"
        _co_p   = f"{_tmpdir}/la_lmer_co_{measure}_{_uid}.csv"
        _fs_p   = f"{_tmpdir}/la_lmer_fs_{measure}_{_uid}.csv"

        try:
            ro.globalenv['r_df']        = _to_r(adf)
            ro.globalenv['r_measure']   = measure
            ro.globalenv['r_within']    = within_col
            ro.globalenv['r_has_sex']   = ro.BoolVector([m_has_sex])
            ro.globalenv['r_fe_p']      = _fe_p
            ro.globalenv['r_co_p']      = _co_p
            ro.globalenv['r_fs_p']      = _fs_p

            ro.r("""
                suppressPackageStartupMessages(library(lmerTest))
                suppressPackageStartupMessages(library(emmeans))

                df_r <- r_df
                # Build ordered factor from the within-subjects column
                wvals        <- sort(unique(df_r[[r_within]]))
                df_r$within_f <- factor(df_r[[r_within]], levels = wvals, ordered = TRUE)
                df_r$Sex      <- factor(df_r$Sex)

                if (r_has_sex[1]) {
                    form <- as.formula(paste0(r_measure,
                        ' ~ within_f + Sex + (1|ID)'))
                } else {
                    form <- as.formula(paste0(r_measure,
                        ' ~ within_f + (1|ID)'))
                }

                model <- lmerTest::lmer(form, data = df_r, REML = TRUE)

                # Fixed effects (Satterthwaite p-values)
                fe      <- as.data.frame(coef(summary(model)))
                fe$term <- rownames(fe)
                write.csv(fe, r_fe_p, row.names = FALSE)

                # Fit statistics
                n_grps <- tryCatch(lme4::ngrps(model)[['ID']], error = function(e) NA_integer_)
                fit_stats <- data.frame(
                    AIC     = AIC(model),
                    BIC     = BIC(model),
                    logLik  = as.numeric(logLik(model)),
                    nobs    = nobs(model),
                    ngroups = n_grps,
                    formula = deparse(formula(model), width.cutoff = 120L)
                )
                write.csv(fit_stats, r_fs_p, row.names = FALSE)

                # Polynomial contrasts for the within-subjects factor
                em_overall    <- emmeans(model, ~ within_f)
                contr_overall <- as.data.frame(contrast(em_overall, 'poly'))
                write.csv(contr_overall, r_co_p, row.names = FALSE)
            """)

            fe_df       = pd.read_csv(_fe_p)
            fit_df      = pd.read_csv(_fs_p)
            contr_ov_df = pd.read_csv(_co_p)

            for _p in [_fe_p, _co_p, _fs_p]:
                try:    os.unlink(_p)
                except Exception: pass

            fit_aic    = float(fit_df['AIC'].iloc[0])
            fit_bic    = float(fit_df['BIC'].iloc[0])
            fit_loglik = float(fit_df['logLik'].iloc[0])
            fit_nobs   = int(fit_df['nobs'].iloc[0])
            fit_ngrps  = (int(fit_df['ngroups'].iloc[0])
                          if 'ngroups' in fit_df.columns and
                             not pd.isna(fit_df['ngroups'].iloc[0])
                          else '?')
            fit_formula = (str(fit_df['formula'].iloc[0])
                           if 'formula' in fit_df.columns else formula_disp)

            print(f"  [OK] AIC={fit_aic:.1f}  BIC={fit_bic:.1f}  "
                  f"n={fit_nobs} obs  {fit_ngrps} animals")

            # ── format report section ─────────────────────────────────
            sec: List[str] = []
            sec += [
                "",
                "\u2550" * W,
                f"  DEPENDENT VARIABLE:  {measure_label}  ({measure})",
                "\u2550" * W,
                "",
                "  THE MIXED PROCEDURE",
                "",
                "  Model Information",
                f"    {'Dependent Variable':<32} {measure_label}",
                f"    {'Covariance Structure':<32} Variance Components",
                f"    {'Subject Effect':<32} ID  (random intercept)",
                f"    {'Estimation Method':<32} REML",
                f"    {'Degrees of Freedom Method':<32} Satterthwaite  (lmerTest)",
                f"    {'Within-subjects Factor':<32} {within_label}  ({within_col}_f, ordered factor)",
                f"    {'Formula':<32} {fit_formula}",
                "",
                f"  Number of Observations Read    {fit_nobs}",
                f"  Number of Observations Used    {fit_nobs}",
                f"  Number of Subjects (ID)        {fit_ngrps}",
                "",
                "  Fit Statistics",
                f"    {'AIC  (smaller is better)':<40} {fit_aic:.1f}",
                f"    {'BIC  (smaller is better)':<40} {fit_bic:.1f}",
                f"    {'-2 Log-Likelihood':<40} {-2 * fit_loglik:.1f}",
                "",
            ]

            # Solution for Fixed Effects
            _fw = (42, 12, 10, 8, 8, 10)
            _fhdr = (
                f"  {'Effect':<{_fw[0]}}  "
                f"{'Estimate':>{_fw[1]}}  "
                f"{'Std Error':>{_fw[2]}}  "
                f"{'DF':>{_fw[3]}}  "
                f"{'t Value':>{_fw[4]}}  "
                f"{'Pr > |t|':>{_fw[5]}}  Sig"
            )
            sec += [
                "  Solution for Fixed Effects",
                "",
                _fhdr,
                "  " + "\u2500" * (sum(_fw) + 14),
            ]

            for _, row in fe_df.iterrows():
                term_raw = str(row.get('term', ''))
                # Replace generic 'within_f' placeholder with the real factor name
                term_disp = _la_poly_fmt_term(
                    term_raw.replace('within_f', f'{within_col}_f')
                )
                est = _rget(row, 'Estimate')
                se  = _rget(row, 'Std. Error', 'Std.Error', 'Std..Error')
                df_ = _rget(row, 'df', 'Df', 'DF')
                t   = _rget(row, 't value', 't.value', 't_value')
                p   = _rget(row, 'Pr(>|t|)', 'Pr...t..', 'p.value')
                sec.append(
                    f"  {term_disp:<{_fw[0]}}  "
                    f"{est:>{_fw[1]}.4f}  "
                    f"{se:>{_fw[2]}.4f}  "
                    f"{df_:>{_fw[3]}.1f}  "
                    f"{t:>{_fw[4]}.3f}  "
                    f"{_la_poly_fmt_p(p):>{_fw[5]}}  "
                    f"{_la_poly_sig_stars(p)}"
                )

            sec.append("")

            # Polynomial contrasts — OVERALL
            _cw = (22, 12, 10, 8, 8, 10)
            _chdr = (
                f"  {'Contrast':<{_cw[0]}}  "
                f"{'Estimate':>{_cw[1]}}  "
                f"{'Std Error':>{_cw[2]}}  "
                f"{'DF':>{_cw[3]}}  "
                f"{'t Value':>{_cw[4]}}  "
                f"{'Pr > |t|':>{_cw[5]}}  Sig"
            )
            sec += [
                f"  Polynomial Contrasts for {within_label}  (OVERALL)",
                "",
                _chdr,
                "  " + "\u2500" * (sum(_cw) + 14),
            ]

            for _, row in contr_ov_df.iterrows():
                name = _la_poly_rename_contrast(str(row.get('contrast', '')))
                est  = _rget(row, 'estimate')
                se   = _rget(row, 'SE')
                df_  = _rget(row, 'df')
                t    = _rget(row, 't.ratio')
                p    = _rget(row, 'p.value')
                sec.append(
                    f"  {name:<{_cw[0]}}  "
                    f"{est:>{_cw[1]}.4f}  "
                    f"{se:>{_cw[2]}.4f}  "
                    f"{df_:>{_cw[3]}.1f}  "
                    f"{t:>{_cw[4]}.3f}  "
                    f"{_la_poly_fmt_p(p):>{_cw[5]}}  "
                    f"{_la_poly_sig_stars(p)}"
                )

            sec += [
                "",
                "  Significance: *** p<.001  ** p<.01  * p<.05  . p<.10",
                "",
            ]

            report_sections.extend(sec)
            results_by_measure[measure] = {
                'fixed_effects'    : fe_df,
                'fit_stats'        : fit_df,
                'contrasts_overall': contr_ov_df,
                'formula'          : formula_disp,
                'report_section'   : '\n'.join(sec),
            }

        except Exception as _exc:
            import traceback
            print(f"  [ERROR] R analysis failed for '{measure}': {_exc}")
            traceback.print_exc()
            report_sections.append(
                f"\n  [ERROR] R analysis failed for '{measure}': {_exc}\n"
            )

    report_sections += [
        "=" * W,
        "  End of LMER Polynomial Contrasts Report",
        "=" * W,
    ]

    full_report = "\n".join(report_sections)
    print("\n" + full_report)

    if save_path is not None:
        Path(save_path).write_text(full_report, encoding='utf-8')
        print(f"\n[OK] Report saved -> {save_path}")

    return {'measures': results_by_measure, 'report': full_report}


def generate_test_registry_report(save_path=None) -> str:
    """Generate a formatted plain-text report documenting every statistical
    test used in lick_analysis.py: data/variables consumed, library source,
    and every parameter with its meaning."""
    mode_label = (
        "RAMP mode  (within-factor = CA%, repeated across citric-acid concentrations)"
        if EXPERIMENT_MODE == 'ramp'
        else "NONRAMP mode  (within-factor = Week, repeated across calendar weeks)"
    )
    within_factor = "CA_Percent" if EXPERIMENT_MODE == 'ramp' else "Week"
    wf            = "CA%"        if EXPERIMENT_MODE == 'ramp' else "Week"

    W = 80  # report line width

    def _h1(text):
        return ["=" * W, f"  {text}", "=" * W]

    def _h2(num, title):
        return [f"\n{'─' * W}", f"  TEST {num}  │  {title}", f"{'─' * W}", ""]

    def _sub(label):
        pad = W - 4 - len(label) - 2
        return [f"  {label}  {'·' * max(pad, 4)}", ""]

    def _tbl(rows, w1=12, w2=44, w3=20):
        hdr  = f"    {'Variable':<{w1}}  {'Description':<{w2}}  Data Type"
        sep  = f"    {'─'*w1}  {'─'*w2}  {'─'*w3}"
        body = [f"    {r[0]:<{w1}}  {r[1]:<{w2}}  {r[2]}" for r in rows]
        return [hdr, sep] + body

    def _out(rows, w1=14, w2=62):
        hdr  = f"    {'Column':<{w1}}  Meaning"
        sep  = f"    {'─'*w1}  {'─'*w2}"
        body = [f"    {r[0]:<{w1}}  {r[1]}" for r in rows]
        return [hdr, sep] + body

    # ── Header + Quick Reference ──────────────────────────────────────────── #
    lines = _h1("STATISTICAL TEST REGISTRY  —  lick_analysis.py")
    lines += [
        "",
        f"  Mode    : {mode_label}",
        f"  Within  : {within_factor}  |  Between: Sex  |  α = 0.05",
        "",
        f"  QUICK REFERENCE  {'·' * 57}",
        "",
        f"    {'#':<3}  {'Test':<40}  Library / Function",
        f"    {'─'*3}  {'─'*40}  {'─'*30}",
        f"    1    {'Mixed ANOVA  (primary)':<40}  pingouin / pg.mixed_anova()",
        f"    2    {f'RM-ANOVA  (no Sex data)':<40}  pingouin / pg.rm_anova()",
        f"    3    {'One-way ANOVA  (no pingouin)':<40}  scipy.stats / f_oneway()",
        f"    4    {f'Paired t-tests + Bonferroni  ({wf} post-hoc)':<40}  scipy.stats / ttest_rel()",
        f"    5    {f'Pairwise within  ({wf} post-hoc)':<40}  pingouin / pg.pairwise_tests()",
        f"    6    {'Pairwise between  (Sex post-hoc)':<40}  scipy / pingouin / ttest_ind()",
        f"    7    {'Simple-effects RM-ANOVA  (per Sex stratum)':<40}  pingouin / pg.rm_anova()",
        "",
        "    Multiple comparisons:",
        "      Bonferroni — Tests 4 (manual t-test loop) and 7 (across Sex strata)",
        "      BH-FDR     — Tests 5–6  (padjust=fdr_bh inside pg.pairwise_tests)",
        "    Sphericity : Greenhouse-Geisser auto-correction  (all pingouin tests;",
        "                 triggered when Mauchly's p-spher < 0.05)",
        "=" * W,
    ]

    # ── TEST 1 ───────────────────────────────────────────────────────────── #
    lines += _h2("1", f"Mixed (Split-Plot) ANOVA  —  Sex × {wf}")
    lines += _sub("PURPOSE")
    lines += [
        f"    Tests whether lick measures differ across {wf} levels (within-subjects),",
        f"    between sexes (between-subjects), and whether a Sex × {wf} interaction exists.",
        "    Primary test when Sex data are available; falls back to Test 2/3 without it.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.mixed_anova()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    f"Long-format DataFrame  (one row per Animal × {wf})", "pd.DataFrame"),
        ("dv",      "Lick measure  e.g. 'Total_Licks', 'Total_Bouts'",    "pd.Series[float64]"),
        ("within",  f"'{within_factor}'  — repeated-measures factor",     "pd.Series[int/str]"),
        ("between", "'Sex'  →  'Male' | 'Female'",                        "pd.Series[str]"),
        ("subject", "'Animal'  — unique per-animal identifier",            "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    correction    Not explicitly set  (pingouin default = no forced GG).",
        "                  Mauchly's sphericity test runs automatically.",
        "                  When p-spher < 0.05, GG ε adjusts df downward → more conservative",
        "                  F-test.  Cite p-GG-corr instead of p-unc in those rows.",
        "                  Note: across_cohort.py uses correction=True  (GG always applied).",
        "",
        "    ss_type       Type III SS  (pingouin default; not set explicitly).",
        "                  Each effect is adjusted for all others including interactions.",
        "                  Standard choice for unbalanced designs.",
        "",
    ]
    lines += _sub("OUTPUT  (key columns)")
    lines += _out([
        ("Source",    f"Effect label: '{within_factor}', 'Sex', '{within_factor} * Sex'"),
        ("F",         "F-statistic"),
        ("p-unc",     "Unadjusted p  (use p-GG-corr when sphericity is violated)"),
        ("np2",       "Partial η²  —  small ≥ 0.01  |  medium ≥ 0.06  |  large ≥ 0.14"),
        ("p-GG-corr", "Greenhouse-Geisser corrected p  (within-subjects / interaction rows)"),
        ("W-spher",   "Mauchly's W  (1.0 = perfect sphericity)"),
        ("p-spher",   "Mauchly's test p  (< 0.05 → sphericity violated → cite GG p)"),
        ("eps",       "GG ε  (1.0 = no correction needed; lower → larger df reduction)"),
    ])
    lines += [
        "",
        "    Measures: Total_Licks, Total_Bouts, Avg_ILI, Avg_Bout_Duration,",
        "              Bottle_Weight_Loss, First_5min_Lick_Pct, Time_to_50pct_Licks,",
        "              First_5min_Bout_Pct",
        "    Threshold: α = 0.05  (each measure is an independent testing family)",
    ]

    # ── TEST 2 ───────────────────────────────────────────────────────────── #
    lines += _h2("2", f"One-Way RM-ANOVA  —  {wf} only  (no Sex data)")
    lines += _sub("PURPOSE")
    lines += [
        f"    Fallback when Sex data are absent.  Tests the {wf} within-factor only,",
        "    treating all animals as one group.  Results carry a 'no Sex data' warning.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.rm_anova()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    f"Long-format DataFrame (one row per Animal × {wf})", "pd.DataFrame"),
        ("dv",      "Lick measure column",                                "pd.Series[float64]"),
        ("within",  f"'{within_factor}'",                                 "pd.Series[int/str]"),
        ("subject", "'Animal'",                                           "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    detailed   True  — returns full SS table including within- and between-subjects",
        "               error rows, enabling sphericity assessment and η² reporting.",
        "",
        "    All other params use pingouin defaults.",
        "    Output columns: Source, F, p-unc, np2, p-GG-corr, eps, W-spher, p-spher",
        "    Threshold: α = 0.05",
    ]

    # ── TEST 3 ───────────────────────────────────────────────────────────── #
    lines += _h2("3", "One-Way Independent ANOVA  (pingouin not installed)")
    lines += _sub("PURPOSE")
    lines += [
        "    Emergency fallback when pingouin cannot be imported.  Subjects treated as",
        "    independent observations — does NOT account for repeated measures or Sex.",
        "    Results are flagged with a warning in all reports.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    scipy.stats.f_oneway()     from scipy import stats", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("*groups", f"One 1-D array per {wf} level (all measurements at that level)",
         "np.ndarray[float64]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS  &  OUTPUT")
    lines += [
        "    *groups  — positional; pass each level's values as a separate array",
        "    (no additional keyword parameters)",
        "    Returns  — F-statistic (float),  p-value (float)",
        "    Threshold: α = 0.05",
    ]

    # ── TEST 4 ───────────────────────────────────────────────────────────── #
    lines += _h2("4", f"Bonferroni Paired t-Tests  (post-hoc for significant {wf} effect)")
    lines += _sub("PURPOSE")
    lines += [
        f"    Pairwise follow-up after a significant {wf} ANOVA.  Each pair of {wf} levels",
        "    is compared using a paired t-test  (same animals experienced both levels).",
        "    Bonferroni multiplication controls the family-wise error rate across all pairs.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    scipy.stats.ttest_rel()     from scipy import stats", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("v1", f"Values at {wf} level A  (ordered by Animal ID)", "np.ndarray[float64]"),
        ("v2", f"Values at {wf} level B  (same order as v1)",     "np.ndarray[float64]"),
    ], w1=5, w2=52, w3=20)
    lines += [
        "    Note: only animals present at BOTH levels are included  (inner join on ID).",
        "",
    ]
    lines += _sub("PARAMETERS  &  ADDITIONAL QUANTITIES")
    lines += [
        "    v1, v2     — equal-length paired 1-D arrays  (two-tailed test by default)",
        "",
        f"    Bonferroni:  p_adj = min(p_raw × k, 1.0)   where k = C(n_{wf}_levels, 2)",
        "",
        "    df         = n_pairs − 1",
        "    t_crit     = stats.t.ppf(0.975, df)  (two-tailed critical value, α = 0.05)",
        "    CI_lower   = mean_diff − t_crit × SE_diff",
        "    CI_upper   = mean_diff + t_crit × SE_diff",
        "    SE_diff    = SD(differences) / sqrt(n_pairs)",
        "",
        "    Threshold: α = 0.05 applied to Bonferroni-adjusted p",
    ]

    # ── TEST 5 ───────────────────────────────────────────────────────────── #
    lines += _h2("5", f"Within-Subjects Pairwise Tests  (post-hoc for {wf} effect)")
    lines += _sub("PURPOSE")
    lines += [
        f"    Called from perform_mixed_anova_posthoc().  All pairwise {wf} comparisons",
        "    collapsed across Sex, using pingouin's built-in FDR correction and Hedges' g.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.pairwise_tests()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "Long-format DataFrame",               "pd.DataFrame"),
        ("dv",      "Lick measure column",                 "pd.Series[float64]"),
        ("within",  f"'{within_factor}'",                  "pd.Series[int/str]"),
        ("subject", "Subject identifier column",           "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    parametric   True  — paired Student's t-test  (not Wilcoxon rank-sum)",
        "",
        "    padjust      'fdr_bh'  — Benjamini-Hochberg FDR applied across all pairwise",
        "                 comparisons for this measure.  Controls the expected proportion of",
        "                 false discoveries at 5%  (less conservative than Bonferroni).",
        "",
        "    effsize      'hedges'  — Hedges' g  =  Cohen's d × (1 − 3 / (4·df − 1)).",
        "                 Bias-corrected standardised mean difference; preferred for n < 20.",
        "",
        "    Threshold: α = 0.05 applied to BH-FDR-corrected p  (padjust column)",
    ]

    # ── TEST 6 ───────────────────────────────────────────────────────────── #
    lines += _h2("6", "Between-Subjects Pairwise Tests  (post-hoc for Sex effect)")
    lines += _sub("PURPOSE")
    lines += [
        "    Also from perform_mixed_anova_posthoc().  Compares Sex groups on the lick",
        f"    measure averaged across all {wf} levels.  Uses ttest_ind for 2 groups,",
        "    pg.pairwise_tests for 3 or more.",
        "",
    ]
    lines += _sub("LIBRARY  (exactly 2 Sex groups — the common case)")
    lines += ["    scipy.stats.ttest_ind()     from scipy import stats", ""]
    lines += _sub("INPUTS  (2-group case)")
    lines += _tbl([
        ("group1", "Per-animal means for Sex group 1",   "pd.Series[float64]"),
        ("group2", "Per-animal means for Sex group 2",   "pd.Series[float64]"),
    ], w1=8, w2=50, w3=20)
    lines.append("")
    lines += _sub("PARAMETERS  (2-group case)")
    lines += [
        "    equal_var   True  (default)  —  Student's independent-samples t-test.",
        "                Assumes equal population variances  (not Welch's correction).",
        "",
        "    Cohen's d   = (mean₁ − mean₂) / pooled_SD",
        "    pooled_SD   = sqrt( ((n₁−1)·SD₁² + (n₂−1)·SD₂²) / (n₁+n₂−2) )",
        "",
    ]
    lines += _sub("LIBRARY  (≥ 3 Sex groups — uncommon fallback)")
    lines += [
        "    pingouin.pairwise_tests(data, dv, between='Sex',",
        "                            parametric=True, padjust='fdr_bh', effsize='cohen')",
        "",
        "    Threshold: α = 0.05  (no correction for 2 groups  |  BH-FDR for ≥ 3 groups)",
    ]

    # ── TEST 7 ───────────────────────────────────────────────────────────── #
    lines += _h2("7", "Simple-Effects RM-ANOVA  (within each Sex stratum separately)")
    lines += _sub("PURPOSE")
    lines += [
        f"    From perform_mixed_anova_posthoc().  For each Sex level independently,",
        f"    a one-way RM-ANOVA tests whether the {wf} effect holds within that sex.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.rm_anova()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    f"Long-format DataFrame  filtered to one Sex stratum",  "pd.DataFrame"),
        ("dv",      "Lick measure column",                                  "pd.Series[float64]"),
        ("within",  f"'{within_factor}'",                                   "pd.Series[int/str]"),
        ("subject", "Subject identifier column",                            "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS  &  CORRECTION")
    lines += [
        "    detailed   False  (condensed output — no separate error-SS rows)",
        "    All other params use pingouin defaults.",
        "",
        "    Bonferroni across strata:  α_adj = 0.05 / n_sex_levels",
        "    Sphericity GG-correction applied automatically when p-spher < 0.05.",
        "",
        "    Output: Source, F, p-unc, np2  (+ p-GG-corr when sphericity violated)",
        "    Threshold: α_adj = 0.05 / n_sex_levels",
    ]

    # ── SUMMARY ──────────────────────────────────────────────────────────── #
    lines += [
        "",
        f"\n{'─' * W}",
        "  CORRECTION METHODS SUMMARY",
        f"  {'·' * (W - 4)}",
        "",
        f"    {'Context':<34}  {'Method':<18}  Detail",
        f"    {'─'*34}  {'─'*18}  {'─'*22}",
        f"    {'Omnibus within effect (all tests)':<34}  {'GG auto':<18}  Mauchly p < 0.05 triggers GG",
        f"    {'Paired t-test loop  (Test 4)':<34}  {'Bonferroni':<18}  p × C(k,2)  |  capped at 1.0",
        f"    {'Pairwise_tests  (Tests 5–6)':<34}  {'BH-FDR':<18}  padjust=fdr_bh in pingouin",
        f"    {'Simple-effects  (Test 7)':<34}  {'Bonferroni/Sex':<18}  α_adj = 0.05 / n_sex_levels",
        "",
        f"{'─' * W}",
        "  MEASURES ANALYSED",
        f"  {'·' * (W - 4)}",
        "",
        "    Primary lick measures",
        "      1.  Total_Licks          total lick count per session               int → float",
        "      2.  Total_Bouts          total lick bouts per session               int → float",
        "      3.  Avg_ILI              mean inter-lick interval  (ms)             float",
        "      4.  Avg_Bout_Duration    mean bout duration  (s)                    float",
        "      5.  Bottle_Weight_Loss   liquid consumed  (g; negatives removed)    float",
        "",
        "    Frontloading / temporal distribution",
        "      6.  First_5min_Lick_Pct  % of session licks in the first 5 min     float  0–100",
        "      7.  Time_to_50pct_Licks  minutes to reach 50% of session licks     float",
        "      8.  First_5min_Bout_Pct  % of session bouts in the first 5 min     float  0–100",
        "",
        "=" * W,
    ]

    report = "\n".join(lines)
    if save_path is not None:
        Path(save_path).write_text(report, encoding="utf-8")
        print(f"[OK] Test registry saved -> {save_path}")
    return report



if __name__ == "__main__":
    main()