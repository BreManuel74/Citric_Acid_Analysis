"""
Cross-Cohort Lick Analysis Module

This module provides functionality to load and compare lick data across multiple cohorts.
Each cohort consists of capacitive sensor data files and associated metadata.

Main features:
- Load multiple cohorts of lick data from capacitive sensor CSV files
- Process lick events and bouts for each cohort
- Compare lick metrics (total licks, bouts, ILI, bout duration) across cohorts
- Perform statistical analyses: Mixed ANOVA (CA% × Time × Sex)
- Stratified analyses holding sex or CA% constant
- Generate comprehensive reports and visualizations

Usage:
    from across_cohort_lick import load_lick_cohorts, perform_cross_cohort_lick_anova
    
    # Load cohorts with capacitive sensor data
    cohort_paths = {
        "0% CA": {
            "master_csv": Path("0%_files/0%_lick_data.csv"),
            "capacitive_logs": [Path("0%_files/capacitive_log_2026-1-21.csv"), ...]
        },
        "2% CA": {
            "master_csv": Path("2%_files/2%_lick_data.csv"),
            "capacitive_logs": [Path("2%_files/capacitive_log_2026-1-28.csv"), ...]
        }
    }
    cohorts = load_lick_cohorts(cohort_paths)
    
    # Perform cross-cohort analysis
    results = perform_cross_cohort_lick_anova(cohorts, measure="Total Licks")
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Try to import pingouin for mixed ANOVA
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Mixed ANOVA will not be available.")
    print("Install with: pip install pingouin")

# Try to import tkinter for GUI file selection
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available. GUI file selection will not work.")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MATPLOTLIB = True
    
    # Configure matplotlib for publication-quality plots
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
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will not be available.")

# Cache for loaded cohorts
_LICK_COHORT_DATA: Dict[str, pd.DataFrame] = {}


# =============================================================================
# LICK DATA PROCESSING FUNCTIONS (from lick_nonramp.py)
# =============================================================================

def load_capacitive_csv(csv_path: Path) -> pd.DataFrame:
    """Load and clean a capacitive CSV file.
    
    Returns DataFrame with Time_sec column and sensor readings.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Capacitive CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if "Arduino_Timestamp" not in df.columns:
        raise ValueError(f"Arduino_Timestamp column not found in {csv_path}")

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
            return 999
    
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

    event_times = events_df[events_df[event_col]]['Time_sec'].values
    if len(event_times) == 0:
        return np.nan

    event_times = np.sort(event_times)
    total_licks = len(event_times)
    idx_50 = int(np.ceil(total_licks / 2.0)) - 1  # 0-indexed midpoint
    idx_50 = min(idx_50, total_licks - 1)
    return float(event_times[idx_50]) / 60.0


def process_capacitive_file(
    capacitive_file: Path,
    fixed_threshold: float = 0.01,
    ili_cutoff: float = 0.3,
    verbose: bool = False
) -> Dict:
    """Process a single capacitive sensor file and extract lick metrics.
    
    Uses EXACT lick detection algorithm from lick_detection.py with fixed threshold.
    
    Parameters:
        capacitive_file: Path to capacitive CSV file
        fixed_threshold: Fixed threshold for event detection (default 0.01, same as lick_detection.py)
        ili_cutoff: Inter-lick interval cutoff in seconds
        verbose: Whether to print detailed processing info
        
    Returns:
        Dictionary with processed lick data and metrics
    """
    if verbose:
        print(f"\nProcessing: {capacitive_file.name}")
    
    # Load capacitive data
    df = load_capacitive_csv(capacitive_file)
    sensor_cols = get_sensor_columns(df)
    
    if len(sensor_cols) == 0:
        print(f"[WARNING] No sensor columns found in {capacitive_file.name}")
        return {'sensor_cols': [], 'bout_results': {}, 'events_df': pd.DataFrame()}
    
    # Set up KDE cache file path (saves computation time on subsequent runs)
    cache_dir = capacitive_file.parent / 'kde_cache'
    cache_filename = capacitive_file.stem + '_kde_cache.csv'
    cache_file = cache_dir / cache_filename
    
    # Compute KDE peaks
    sensor_kdes = compute_sensor_KDE(df, sensor_cols, cache_file=cache_file, verbose=verbose)
    
    # Compute deviations
    df = compute_KDE_normalizations(df, sensor_cols, sensor_kdes)
    
    # Use FIXED threshold (same as lick_detection.py)
    thresholds = compute_fixed_thresholds(sensor_cols, fixed_threshold=fixed_threshold)
    
    # Detect events
    events_df = detect_events_above_threshold(df, sensor_cols, thresholds)
    
    # Filter to only first 30 minutes of session (1800 seconds)
    original_length = len(events_df)
    events_df = events_df[events_df['Time_sec'] < 1800].copy()
    if verbose:
        print(f"    Filtered to first 30 minutes: {len(events_df)}/{original_length} time points retained")
    
    # Compute bouts
    bout_results = compute_lick_bouts(events_df, sensor_cols, ili_cutoff=ili_cutoff)
    
    return {
        'sensor_cols': sensor_cols,
        'bout_results': bout_results,
        'events_df': events_df,
        'thresholds': thresholds,
        'kdes': sensor_kdes
    }


# =============================================================================
# COHORT LOADING FUNCTIONS
# =============================================================================

def load_lick_master_csv(csv_path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load lick data master CSV with animal metadata and weekly lick summaries.
    
    Expected columns: ID, Date, Week, Sex, CA%, Total_Licks, Total_Bouts, etc.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding=encoding)
    
    # Convert Date to datetime
    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure numeric columns
    numeric_cols = ['Total_Licks', 'Total_Bouts', 'Avg_ILI', 'Avg_Bout_Duration', 
                   'Licks_Per_Bout', 'CA%']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def process_cohort_capacitive_files(
    master_csv: Path,
    capacitive_files: List[Path],
    ca_percent: float,
    cohort_label: str,
    fixed_threshold: float = 0.01,
    ili_cutoff: float = 0.3,
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """Process capacitive sensor files for a single cohort with EXACT per-animal logic from lick_nonramp.py.
    
    This function replicates the exact alignment logic where:
    1. Each animal is assigned to a specific sensor via 'selected_sensors' column
    2. Each animal gets the lick count from THEIR assigned sensor only
    3. One row per animal per week with their individual sensor's data
    
    Uses EXACT lick detection algorithm from lick_detection.py with fixed threshold of 0.01.
    
    Parameters:
        master_csv: Path to master CSV with animal metadata and sensor assignments
        capacitive_files: List of paths to capacitive sensor CSV files
        ca_percent: CA percentage for this cohort
        cohort_label: Label for this cohort
        fixed_threshold: Fixed threshold for lick detection (default 0.01, same as lick_detection.py)
        ili_cutoff: Inter-lick interval cutoff in seconds
        encoding: Optional encoding for CSV files
        
    Returns:
        DataFrame with processed lick data including metadata (one row per animal per week)
    """
    print(f"\n  Processing {len(capacitive_files)} capacitive files...")
    
    # Load master CSV for metadata
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    
    master_df = pd.read_csv(master_csv, encoding=encoding)
    print(f"  Loaded master CSV with {len(master_df)} rows")
    
    # Standardize column names (case-insensitive)
    master_df.columns = master_df.columns.str.strip().str.lower()

    # Quick sanity check: detect if user accidentally selected the behavioral master
    # (master_data_*.csv) instead of the lick master (e.g. 0%_lick_data.csv)
    _behavioral_cols = {'daily change', 'total change', 'nest made?', 'lethargy?', 'anxious behaviors?'}
    _found = set(master_df.columns)
    if _behavioral_cols & _found:
        raise ValueError(
            f"Wrong file selected for '{cohort_label}'.\n"
            f"  You chose: {master_csv.name}\n"
            f"  This looks like a behavioral master CSV (has columns: "
            f"{sorted(_behavioral_cols & _found)}).\n"
            f"  Please select the LICK data CSV instead "
            f"(e.g. '0%_lick_data.csv', '2%_lick_data.csv', '5_week_lick_data.csv')."
        )

    # Check for required columns
    required_cols = ['date', 'animal_id', 'selected_sensors']
    missing_cols = [col for col in required_cols if col not in master_df.columns]
    if missing_cols:
        raise ValueError(
            f"Master CSV missing required columns: {missing_cols}.\n"
            f"  File: {master_csv.name}\n"
            f"  Found columns: {list(master_df.columns)}\n"
            f"  Make sure you selected the LICK data CSV "
            f"(e.g. '0%_lick_data.csv'), not the behavioral master."
        )
    
    # Check if sex column exists
    has_sex = 'sex' in master_df.columns
    
    # Process each capacitive file
    all_animal_records = []
    
    for cap_file in sorted(capacitive_files):
        if not cap_file.exists():
            print(f"  [WARNING] Capacitive file not found: {cap_file.name}")
            continue
        
        print(f"\n  Processing: {cap_file.name}")
        
        # Extract date from filename (format: capacitive_log_YYYY-M-D.csv)
        date_str = None
        filename = cap_file.stem
        if 'capacitive_log_' in filename:
            date_part = filename.split('capacitive_log_')[1]
            try:
                parts = date_part.split('-')
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    date_str = f"{month}/{day}/{str(year)[2:]}"  # Convert to M/D/YY format
                    print(f"    Date extracted: {date_str}")
            except:
                print(f"    [WARNING] Could not parse date from filename")
                continue
        
        if date_str is None:
            print(f"    [WARNING] No valid date found for {cap_file.name}, skipping")
            continue
        
        # Process capacitive file to get per-sensor lick counts
        try:
            result = process_capacitive_file(cap_file, fixed_threshold=fixed_threshold, 
                                            ili_cutoff=ili_cutoff, verbose=False)
            
            # Build per-sensor lick and bout counts
            sensor_lick_counts = {}
            sensor_bout_counts = {}
            sensor_bout_durations = {}
            sensor_ilis = {}
            
            for sensor_col in result['sensor_cols']:
                # Count licks from events
                event_col = f"{sensor_col}_event"
                if event_col in result['events_df'].columns:
                    sensor_licks = int(result['events_df'][event_col].sum())
                    sensor_lick_counts[sensor_col] = sensor_licks
                else:
                    sensor_lick_counts[sensor_col] = 0
                
                # Get bout data
                bout_data = result['bout_results'].get(sensor_col, {})
                sensor_bout_counts[sensor_col] = bout_data.get('bout_count', 0)
                
                # Calculate average bout duration for this sensor
                bout_durs = bout_data.get('bout_durations', np.array([]))
                if len(bout_durs) > 0:
                    sensor_bout_durations[sensor_col] = float(np.mean(bout_durs))
                else:
                    sensor_bout_durations[sensor_col] = 0.0
                
                # Calculate average ILI for this sensor (inter-bout intervals)
                bout_starts = bout_data.get('bout_start_times', np.array([]))
                if len(bout_starts) > 1:
                    ilis = np.diff(bout_starts)
                    sensor_ilis[sensor_col] = float(np.mean(ilis))
                else:
                    sensor_ilis[sensor_col] = np.nan
            
            # Get metadata for animals tested on this date
            # Normalize date in master_df for matching
            master_df['date_normalized'] = master_df['date'].astype(str).str.strip()
            date_metadata = master_df[master_df['date_normalized'] == date_str].copy()
            
            if len(date_metadata) == 0:
                print(f"    [WARNING] No animals found in master CSV for date {date_str}")
                continue
            
            print(f"    Found {len(date_metadata)} animals for this date")
            
            # For each animal, extract their assigned sensor's data
            for _, animal_row in date_metadata.iterrows():
                animal_id = str(animal_row['animal_id']).strip()
                sensor_num = int(animal_row['selected_sensors'])
                sensor_col = f"Sensor_{sensor_num}"
                
                # Get sex if available
                if has_sex:
                    sex = str(animal_row['sex']).strip().upper()
                else:
                    # Try to infer from animal ID (A1-A6 are females in 2% cohort based on master_data_2%.csv)
                    animal_id_upper = animal_id.upper()
                    if animal_id_upper.startswith('A'):
                        # For 2% cohort: A1, A2 are females; A4, A5, A6 are males based on master_data_2%.csv
                        if any(animal_id_upper.startswith(f'A{i}') for i in [1, 2, 3]):
                            sex = 'F'
                        else:
                            sex = 'M'
                    else:
                        sex = 'Unknown'
                
                # Get this animal's lick/bout counts from their assigned sensor
                animal_licks = sensor_lick_counts.get(sensor_col, 0)
                animal_bouts = sensor_bout_counts.get(sensor_col, 0)
                animal_bout_dur = sensor_bout_durations.get(sensor_col, 0.0)
                animal_ili = sensor_ilis.get(sensor_col, np.nan)
                
                # Calculate licks per bout
                licks_per_bout = animal_licks / animal_bouts if animal_bouts > 0 else 0.0

                # Compute first-5-minute lick percentage
                _evdf = result['events_df']
                _ecol = f"{sensor_col}_event"
                if _ecol in _evdf.columns and animal_licks > 0:
                    _f5 = int((_evdf[_ecol] & (_evdf['Time_sec'] < 300)).sum())
                    first_5min_pct = _f5 / animal_licks * 100.0
                else:
                    first_5min_pct = 0.0

                # Compute time (minutes) to reach 50% of total licks
                time_50pct = calculate_time_to_50_percent_licks(_evdf, sensor_col)

                # Extract optional weight/fecal data if available
                bottle_weight = animal_row.get('bottle_weight_change', np.nan)
                total_weight = animal_row.get('total_weight_change', np.nan)
                fecal_count = animal_row.get('fecal_count', np.nan)
                
                # Create animal record
                animal_record = {
                    'ID': animal_id,
                    'Date': date_str,
                    'Sex': sex,
                    'CA%': ca_percent,
                    'Cohort': cohort_label,
                    'Sensor': sensor_num,
                    'Total_Licks': animal_licks,
                    'Total_Bouts': animal_bouts,
                    'Licks_Per_Bout': licks_per_bout,
                    'Avg_Bout_Duration': animal_bout_dur,
                    'Avg_ILI': animal_ili,
                    'First_5min_Lick_Pct': first_5min_pct,
                    'Time_to_50pct_Licks': time_50pct,
                    'Bottle_Weight_Change': bottle_weight,
                    'Total_Weight_Change': total_weight,
                    'Fecal_Count': fecal_count
                }
                
                all_animal_records.append(animal_record)

                _t50_str = f"{time_50pct:.1f} min" if np.isfinite(time_50pct) else "N/A"
                print(f"      {animal_id} (Sensor {sensor_num}): {animal_licks} licks, "
                      f"{animal_bouts} bouts, First5min={first_5min_pct:.1f}%, T50%={_t50_str}")
            
        except Exception as e:
            print(f"    [ERROR] Failed to process {cap_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame from all animal records
    if len(all_animal_records) == 0:
        print(f"  [WARNING] No animal records created")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(all_animal_records)
    
    print(f"\n  [OK] Processed {len(capacitive_files)} files, created {len(result_df)} animal records")
    print(f"       Unique animals: {result_df['ID'].nunique()}")
    print(f"       Unique dates: {result_df['Date'].nunique()}")
    
    return result_df


def load_lick_cohorts(cohort_specs: Dict[str, Dict], encoding: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load multiple lick cohorts from master CSV files and capacitive sensor files.
    
    Parameters:
        cohort_specs: Dictionary mapping cohort labels to specs:
            {
                "0% CA": {
                    "master_csv": Path("0%_files/0%_lick_data.csv"),
                    "capacitive_logs": [Path("0%_files/capacitive_log_2026-1-21.csv"), ...],
                    "ca_percent": 0.0
                },
                "2% CA": {
                    "master_csv": Path("2%_files/2%_lick_data.csv"),
                    "capacitive_logs": [Path("2%_files/capacitive_log_2026-1-28.csv"), ...],
                    "ca_percent": 2.0
                }
            }
        encoding: Optional encoding for CSV files
        
    Returns:
        Dictionary mapping cohort labels to loaded DataFrames with sequential week numbers
    """
    cohort_dfs = {}
    
    print("="*80)
    print("LOADING LICK COHORTS WITH CAPACITIVE DATA")
    print("="*80)
    
    for label, specs in cohort_specs.items():
        master_path = specs.get('master_csv')
        capacitive_files = specs.get('capacitive_logs', [])
        ca_percent = specs.get('ca_percent')
        fixed_threshold = specs.get('fixed_threshold', 0.01)
        ili_cutoff = specs.get('ili_cutoff', 0.3)
        
        print(f"\nLoading cohort: {label}")
        
        # Infer CA% if not provided
        if ca_percent is None:
            if '%' in label:
                try:
                    ca_percent = float(label.split('%')[0].split()[-1])
                    print(f"  [INFO] Inferred CA% = {ca_percent} from label")
                except:
                    print(f"  [WARNING] Could not infer CA% from label, setting to 0.0")
                    ca_percent = 0.0
            else:
                ca_percent = 0.0
        
        try:
            # If capacitive files are provided, process them
            if len(capacitive_files) > 0 and master_path is not None:
                print(f"  Master CSV: {master_path}")
                print(f"  Capacitive files: {len(capacitive_files)} files")
                
                df = process_cohort_capacitive_files(
                    master_csv=master_path,
                    capacitive_files=capacitive_files,
                    ca_percent=ca_percent,
                    cohort_label=label,
                    fixed_threshold=fixed_threshold,
                    ili_cutoff=ili_cutoff,
                    encoding=encoding
                )
            
            # Otherwise, just load master CSV
            elif master_path is not None:
                print(f"  Master CSV: {master_path}")
                df = load_lick_master_csv(master_path, encoding=encoding)
                
                # Add CA% if not in data
                if 'CA%' not in df.columns:
                    df['CA%'] = ca_percent
                
                # Add cohort label
                df['Cohort'] = label
            
            else:
                print(f"  [ERROR] No master_csv or capacitive_logs specified for cohort '{label}'")
                continue
            
            if df.empty:
                print(f"  [WARNING] No data loaded for cohort '{label}'")
                continue
            
            # Convert Date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Sort by date to ensure sequential week assignment
            if 'Date' in df.columns and 'ID' in df.columns:
                df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
            elif 'Date' in df.columns:
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Assign sequential week numbers based on unique dates
            if 'Date' in df.columns:
                unique_dates = sorted(df['Date'].dropna().unique())
                date_to_week = {date: idx for idx, date in enumerate(unique_dates)}
                df['Week'] = df['Date'].map(date_to_week)
                print(f"  Assigned weeks 0-{len(unique_dates)-1} based on {len(unique_dates)} unique dates")
            
            cohort_dfs[label] = df
            
            print(f"  [OK] Loaded {len(df)} rows")
            if 'ID' in df.columns:
                print(f"  Unique animals: {df['ID'].nunique()}")
            if 'Date' in df.columns:
                print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            if 'Week' in df.columns:
                print(f"  Weeks: {df['Week'].min()} to {df['Week'].max()}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load cohort '{label}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cache for later use
    global _LICK_COHORT_DATA
    _LICK_COHORT_DATA = cohort_dfs.copy()
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded {len(cohort_dfs)} lick cohort(s)")
    print("="*80)
    
    return cohort_dfs


def combine_lick_cohorts(cohort_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple lick cohort dataframes into a single dataframe for analysis.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        
    Returns:
        Combined DataFrame with all cohorts
    """
    combined_dfs = []
    
    for label, df in cohort_dfs.items():
        df_copy = df.copy()
        
        # Ensure cohort and CA% columns exist
        if 'Cohort' not in df_copy.columns:
            df_copy['Cohort'] = label
        
        if 'CA%' not in df_copy.columns:
            # Try to infer from label
            if '%' in label:
                try:
                    inferred_ca = float(label.split('%')[0].split()[-1])
                    df_copy['CA%'] = inferred_ca
                except:
                    df_copy['CA%'] = np.nan
            else:
                df_copy['CA%'] = np.nan
        
        combined_dfs.append(df_copy)
    
    combined = pd.concat(combined_dfs, ignore_index=True)
    
    # Ensure CA% is numeric
    if 'CA%' in combined.columns:
        combined['CA%'] = pd.to_numeric(combined['CA%'], errors='coerce')
    
    print(f"\n[OK] Combined {len(cohort_dfs)} cohorts into single dataframe")
    print(f"  Total rows: {len(combined)}")
    print(f"  CA% levels: {sorted(combined['CA%'].dropna().unique())}")
    print(f"  Unique animals: {combined['ID'].nunique() if 'ID' in combined.columns else 'Unknown'}")
    
    return combined


def add_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Week' column if not present, computing from Date per animal."""
    if 'Week' in df.columns:
        return df
    
    if 'Date' not in df.columns or 'ID' not in df.columns:
        print("[WARNING] Cannot compute Week: missing Date or ID column")
        return df
    
    df = df.copy()
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    # Group by ID and assign week numbers (0-indexed from first date)
    def assign_weeks(group):
        group = group.sort_values('Date')
        # Get unique dates and assign week numbers
        unique_dates = group['Date'].dt.date.unique()
        date_to_week = {date: i for i, date in enumerate(sorted(unique_dates))}
        group['Week'] = group['Date'].dt.date.map(date_to_week)
        return group
    
    df = df.groupby('ID', group_keys=False).apply(assign_weeks)
    
    print(f"[OK] Added 'Week' column (range: {df['Week'].min()} to {df['Week'].max()})")
    
    return df


# =============================================================================
# CROSS-COHORT STATISTICAL ANALYSIS
# =============================================================================

def perform_cross_cohort_lick_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total_Licks",
    time_points: Optional[List[int]] = None,
    ss_type: int = 3
) -> Dict:
    """Perform 3-way Mixed ANOVA for lick data: CA% (between) × Time (within) × Sex (between).
    
    This analyzes how lick measures change over time, comparing different CA% concentrations
    and sexes.
    
    Design:
        - CA%: Between-subjects factor (each animal assigned to one cohort)
        - Time (Week): Within-subjects factor (repeated measures over weeks)
        - Sex: Between-subjects factor (M or F)
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Lick measure to analyze ('Total_Licks', 'Total_Bouts', 'Avg_ILI', etc.)
        time_points: Optional list of specific weeks to include (None = all weeks)
        ss_type: Sum of squares type (2 or 3). Use 3 for unbalanced designs.
        
    Returns:
        Dictionary with ANOVA results including main effects and interactions
    """
    print("\n" + "="*80)
    print("CROSS-COHORT LICK MIXED ANOVA: CA% × TIME × SEX")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed. Cannot perform mixed ANOVA.")
        return {'error': 'pingouin not installed'}
    
    # Combine cohorts
    print("\nStep 1: Combining cohort dataframes...")
    combined_df = combine_lick_cohorts(cohort_dfs)
    
    # Add Week column if not present
    if 'Week' not in combined_df.columns:
        print("\nStep 2: Computing Week column...")
        combined_df = add_week_column(combined_df)
    
    # Prepare data for ANOVA
    print("\nStep 3: Preparing data for ANOVA...")
    required_cols = ['ID', 'Week', 'Sex', 'CA%', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(time_points)]
        time_label = f"weeks {time_points}"
    else:
        time_points = sorted(analysis_df['Week'].unique())
        time_label = f"all {len(time_points)} weeks"
    
    print(f"\nAnalyzing: {measure}")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA%'].unique())}")
    print(f"  Weeks: {sorted(analysis_df['Week'].unique())}")
    print(f"  Sex levels: {sorted(analysis_df['Sex'].unique())}")
    
    # Check data completeness
    subjects_per_week = analysis_df.groupby('ID')['Week'].nunique()
    total_weeks = analysis_df['Week'].nunique()
    complete_subjects = (subjects_per_week == total_weeks).sum()
    incomplete_subjects = (subjects_per_week < total_weeks).sum()
    
    if incomplete_subjects > 0:
        print(f"\n[WARNING] {incomplete_subjects} animals have incomplete data (not all weeks present)")
        print(f"  Complete subjects: {complete_subjects}")
        print(f"  Incomplete subjects: {incomplete_subjects}")
    
    # Enhanced descriptive statistics
    print(f"\nDescriptive Statistics by Group:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        median = group.median()
        std = group.std()
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem if not np.isnan(sem) else np.nan
        ci_upper = mean + 1.96 * sem if not np.isnan(sem) else np.nan
        
        return pd.Series({
            'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min': group.min(),
            'q25': group.quantile(0.25),
            'q75': group.quantile(0.75),
            'max': group.max()
        })
    
    # Collect statistics for each group
    stats_data = []
    for (ca_val, sex_val), group_data in analysis_df.groupby(['CA%', 'Sex'])[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA%'] = ca_val
        stats['Sex'] = sex_val
        stats_data.append(stats)
    
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA% {row['CA%']:.1f}, Sex {row['Sex']}: "
              f"n={int(row['n'])}, mean={row['mean']:.2f}±{row['sem']:.2f}, "
              f"median={row['median']:.2f}, 95% CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
    
    # Perform mixed ANOVA
    print(f"\nStep 4: Running mixed ANOVA (Type {ss_type} SS)...")
    print("  Design: CA% (between) × Week (within) × Sex (between)")
    
    try:
        # Check if we have multiple between-subjects factors
        has_sex_variation = analysis_df['Sex'].nunique() > 1
        has_ca_variation = analysis_df['CA%'].nunique() > 1
        
        if has_sex_variation and has_ca_variation:
            # Full 3-way design with 2 between factors
            print("  Running 3-way mixed ANOVA with 2 between-subjects factors...")
            
            # Create combined between-subjects factor for pingouin
            analysis_df['CA_Sex'] = analysis_df['CA%'].astype(str) + '_' + analysis_df['Sex'].astype(str)
            
            anova_result = pg.mixed_anova(
                data=analysis_df,
                dv=measure,
                within='Week',
                subject='ID',
                between='CA_Sex',
                ss_type=ss_type
            )
            
            # Also run separate analyses to decompose effects
            # CA% effect (collapsed across Sex)
            anova_ca = pg.mixed_anova(
                data=analysis_df,
                dv=measure,
                within='Week',
                subject='ID',
                between='CA%',
                ss_type=ss_type
            )
            
            # Sex effect (collapsed across CA%)
            anova_sex = pg.mixed_anova(
                data=analysis_df,
                dv=measure,
                within='Week',
                subject='ID',
                between='Sex',
                ss_type=ss_type
            )
            
            print("\n[OK] Mixed ANOVA completed successfully")
            print("\nANOVA Results (combined CA_Sex factor):")
            print(anova_result.to_string())
            
            print("\n\nANOVA Results (CA% factor, collapsed across Sex):")
            print(anova_ca.to_string())
            
            print("\n\nANOVA Results (Sex factor, collapsed across CA%):")
            print(anova_sex.to_string())
            
            # Extract effects of interest
            results = {
                'measure': measure,
                'type': '3way_mixed_anova',
                'design': 'CA% (between) × Week (within) × Sex (between)',
                'n_observations': len(analysis_df),
                'n_subjects': analysis_df['ID'].nunique(),
                'n_weeks': total_weeks,
                'weeks': time_points,
                'ca_levels': sorted(analysis_df['CA%'].unique()),
                'sex_levels': sorted(analysis_df['Sex'].unique()),
                'ss_type': ss_type,
                'anova_table_combined': anova_result,
                'anova_table_ca': anova_ca,
                'anova_table_sex': anova_sex,
                'descriptive_stats': group_stats,
                'complete_subjects': complete_subjects,
                'incomplete_subjects': incomplete_subjects
            }
            
            # Extract specific effects from decomposed tables
            # Week main effect
            week_effect = anova_ca[anova_ca['Source'] == 'Week']
            if not week_effect.empty:
                results['week_F'] = float(week_effect['F'].iloc[0])
                results['week_p'] = float(week_effect['p-unc'].iloc[0])
                results['week_significant'] = results['week_p'] < 0.05
            
            # CA% main effect
            ca_effect = anova_ca[anova_ca['Source'] == 'CA%']
            if not ca_effect.empty:
                results['ca_F'] = float(ca_effect['F'].iloc[0])
                results['ca_p'] = float(ca_effect['p-unc'].iloc[0])
                results['ca_significant'] = results['ca_p'] < 0.05
            
            # Sex main effect
            sex_effect = anova_sex[anova_sex['Source'] == 'Sex']
            if not sex_effect.empty:
                results['sex_F'] = float(sex_effect['F'].iloc[0])
                results['sex_p'] = float(sex_effect['p-unc'].iloc[0])
                results['sex_significant'] = results['sex_p'] < 0.05
            
            # Interactions
            week_ca_int = anova_ca[anova_ca['Source'] == 'Interaction']
            if not week_ca_int.empty:
                results['week_ca_interaction_F'] = float(week_ca_int['F'].iloc[0])
                results['week_ca_interaction_p'] = float(week_ca_int['p-unc'].iloc[0])
                results['week_ca_interaction_significant'] = results['week_ca_interaction_p'] < 0.05
            
            week_sex_int = anova_sex[anova_sex['Source'] == 'Interaction']
            if not week_sex_int.empty:
                results['week_sex_interaction_F'] = float(week_sex_int['F'].iloc[0])
                results['week_sex_interaction_p'] = float(week_sex_int['p-unc'].iloc[0])
                results['week_sex_interaction_significant'] = results['week_sex_interaction_p'] < 0.05
            
        elif has_ca_variation:
            # 2-way mixed: CA% × Week (no sex variation)
            print("  Running 2-way mixed ANOVA: CA% × Week...")
            anova_result = pg.mixed_anova(
                data=analysis_df,
                dv=measure,
                within='Week',
                subject='ID',
                between='CA%',
                ss_type=ss_type
            )
            
            print("\n[OK] Mixed ANOVA completed successfully")
            print("\nANOVA Results:")
            print(anova_result.to_string())
            
            results = {
                'measure': measure,
                'type': '2way_mixed_anova',
                'design': 'CA% (between) × Week (within)',
                'n_observations': len(analysis_df),
                'n_subjects': analysis_df['ID'].nunique(),
                'n_weeks': total_weeks,
                'weeks': time_points,
                'ca_levels': sorted(analysis_df['CA%'].unique()),
                'ss_type': ss_type,
                'anova_table': anova_result,
                'descriptive_stats': group_stats
            }
            
            # Extract effects
            week_effect = anova_result[anova_result['Source'] == 'Week']
            if not week_effect.empty:
                results['week_F'] = float(week_effect['F'].iloc[0])
                results['week_p'] = float(week_effect['p-unc'].iloc[0])
                results['week_significant'] = results['week_p'] < 0.05
            
            ca_effect = anova_result[anova_result['Source'] == 'CA%']
            if not ca_effect.empty:
                results['ca_F'] = float(ca_effect['F'].iloc[0])
                results['ca_p'] = float(ca_effect['p-unc'].iloc[0])
                results['ca_significant'] = results['ca_p'] < 0.05
            
            interaction = anova_result[anova_result['Source'] == 'Interaction']
            if not interaction.empty:
                results['interaction_F'] = float(interaction['F'].iloc[0])
                results['interaction_p'] = float(interaction['p-unc'].iloc[0])
                results['interaction_significant'] = results['interaction_p'] < 0.05
        
        else:
            print("[WARNING] Insufficient variation for mixed ANOVA (need multiple CA% levels)")
            return {'error': 'Insufficient variation in factors'}
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def perform_between_subjects_lick_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total_Licks",
    time_point: Optional[int] = None,
    average_over_weeks: bool = False,
    ss_type: int = 3
) -> Dict:
    """Perform 2-Way Between-Subjects ANOVA: CA% × Sex (holding time constant).
    
    This analyzes the effect of CA% and Sex on lick measures at:
    1. A specific time point (e.g., final week)
    2. Averaged across all weeks per animal
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Lick measure to analyze ('Total_Licks', 'Total_Bouts', etc.)
        time_point: Specific Week to analyze, None with average_over_weeks=True
        average_over_weeks: If True, average measure across all weeks per animal
        ss_type: Sum of squares type (2 or 3)
        
    Returns:
        Dictionary with ANOVA results
    """
    print("\n" + "="*80)
    print("BETWEEN-SUBJECTS LICK ANOVA: CA% × SEX")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed. Cannot perform ANOVA.")
        return {'error': 'pingouin not installed'}
    
    # Combine cohorts
    print("\nStep 1: Combining cohort dataframes...")
    combined_df = combine_lick_cohorts(cohort_dfs)
    
    # Add Week column if not present
    if 'Week' not in combined_df.columns:
        print("\nStep 2: Computing Week column...")
        combined_df = add_week_column(combined_df)
    
    # Prepare data
    print("\nStep 3: Preparing data for ANOVA...")
    required_cols = ['ID', 'Sex', 'CA%', measure]
    if time_point is not None:
        required_cols.append('Week')
    
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    # Filter to specific time point or average across weeks
    if time_point is not None and 'Week' in analysis_df.columns:
        analysis_df = analysis_df[analysis_df['Week'] == time_point]
        analysis_type = f"Week {time_point}"
        print(f"  Filtering to Week {time_point}")
    elif average_over_weeks:
        print(f"  Averaging across all weeks per animal...")
        analysis_df = analysis_df.groupby(['ID', 'Sex', 'CA%'])[measure].mean().reset_index()
        analysis_type = "averaged across weeks"
    else:
        raise ValueError("Must specify either time_point or average_over_weeks=True")
    
    print(f"\nAnalyzing: {measure} ({analysis_type})")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA%'].unique())}")
    print(f"  Sex levels: {sorted(analysis_df['Sex'].unique())}")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by Group:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        median = group.median()
        std = group.std()
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem if not np.isnan(sem) else np.nan
        ci_upper = mean + 1.96 * sem if not np.isnan(sem) else np.nan
        
        return pd.Series({
            'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
    
    stats_data = []
    for (ca_val, sex_val), group_data in analysis_df.groupby(['CA%', 'Sex'])[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA%'] = ca_val
        stats['Sex'] = sex_val
        stats_data.append(stats)
    
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA% {row['CA%']:.1f}, Sex {row['Sex']}: "
              f"n={int(row['n'])}, mean={row['mean']:.2f}±{row['sem']:.2f}, "
              f"95% CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
    
    print(f"\nStep 4: Running 2-way between-subjects ANOVA (Type {ss_type} SS)...")
    print("  Design: CA% (between) × Sex (between)")
    
    try:
        # Check factor variation
        has_sex_variation = analysis_df['Sex'].nunique() > 1
        has_ca_variation = analysis_df['CA%'].nunique() > 1
        
        if has_sex_variation and has_ca_variation:
            # Full 2-way ANOVA
            anova_result = pg.anova(
                data=analysis_df,
                dv=measure,
                between=['CA%', 'Sex'],
                ss_type=ss_type
            )
            
            print("\n[OK] ANOVA completed successfully")
            print("\nANOVA Results:")
            print(anova_result.to_string())
            
            results = {
                'measure': measure,
                'type': '2way_between_subjects',
                'analysis_type': analysis_type,
                'design': 'CA% (between) × Sex (between)',
                'n_subjects': analysis_df['ID'].nunique(),
                'ca_levels': sorted(analysis_df['CA%'].unique()),
                'sex_levels': sorted(analysis_df['Sex'].unique()),
                'ss_type': ss_type,
                'anova_table': anova_result,
                'descriptive_stats': group_stats
            }
            
            # Extract specific effects
            ca_effect = anova_result[anova_result['Source'] == 'CA%']
            if not ca_effect.empty:
                results['ca_F'] = float(ca_effect['F'].iloc[0])
                results['ca_p'] = float(ca_effect['p-unc'].iloc[0])
                results['ca_significant'] = results['ca_p'] < 0.05
            
            sex_effect = anova_result[anova_result['Source'] == 'Sex']
            if not sex_effect.empty:
                results['sex_F'] = float(sex_effect['F'].iloc[0])
                results['sex_p'] = float(sex_effect['p-unc'].iloc[0])
                results['sex_significant'] = results['sex_p'] < 0.05
            
            interaction = anova_result[anova_result['Source'] == 'CA% * Sex']
            if not interaction.empty:
                results['interaction_F'] = float(interaction['F'].iloc[0])
                results['interaction_p'] = float(interaction['p-unc'].iloc[0])
                results['interaction_significant'] = results['interaction_p'] < 0.05
            
            # Perform post-hoc tests if main effects are significant
            if results.get('ca_significant', False):
                print("\n[INFO] Performing post-hoc tests for CA% (Tukey HSD)...")
                posthoc_ca = pg.pairwise_tukey(data=analysis_df, dv=measure, between='CA%')
                results['posthoc_ca'] = posthoc_ca
                print(posthoc_ca.to_string())
            
            if results.get('sex_significant', False):
                print("\n[INFO] Performing post-hoc tests for Sex (Tukey HSD)...")
                posthoc_sex = pg.pairwise_tukey(data=analysis_df, dv=measure, between='Sex')
                results['posthoc_sex'] = posthoc_sex
                print(posthoc_sex.to_string())
            
            return results
            
        else:
            print("[WARNING] Insufficient variation for 2-way ANOVA")
            return {'error': 'Insufficient variation in factors'}
        
    except Exception as e:
        print(f"\n[ERROR] ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


# =============================================================================
# STRATIFIED ANALYSES
# =============================================================================

def perform_lick_anova_sex_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    sex: str,
    measure: str = "Total_Licks",
    time_points: Optional[List[int]] = None
) -> Dict:
    """Perform 2-Way Mixed ANOVA: Week (within) × CA% (between), holding Sex constant.
    
    This analyzes longitudinal lick changes for ONE sex at a time.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        sex: Sex to analyze ("M" or "F")
        measure: Lick measure to analyze
        time_points: Optional list of specific weeks to include
        
    Returns:
        Dictionary with ANOVA results for the specified sex
    """
    print("\n" + "="*80)
    print(f"SEX-STRATIFIED LICK MIXED ANOVA: WEEK (WITHIN) × CA% (BETWEEN)")
    print(f"Analyzing: {'MALES' if sex == 'M' else 'FEMALES'} only")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed.")
        return {'error': 'pingouin not installed'}
    
    if sex not in ["M", "F"]:
        raise ValueError("sex must be 'M' or 'F'")
    
    # Combine cohorts
    print("\nStep 1: Combining and filtering data...")
    combined_df = combine_lick_cohorts(cohort_dfs)
    
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    
    # Filter to specified sex
    required_cols = ['ID', 'Week', 'Sex', 'CA%', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[combined_df['Sex'] == sex][required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) == 0:
        print(f"[ERROR] No data found for sex = {sex}")
        return {'error': f'No data for sex {sex}'}
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(time_points)]
    else:
        time_points = sorted(analysis_df['Week'].unique())
    
    print(f"\nAnalyzing: {measure} ({'Males' if sex == 'M' else 'Females'})")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA%'].unique())}")
    print(f"  Weeks: {len(time_points)} weeks")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by CA% Group:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std()
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem if not np.isnan(sem) else np.nan
        ci_upper = mean + 1.96 * sem if not np.isnan(sem) else np.nan
        return pd.Series({
            'n': n, 'mean': mean, 'std': std, 'sem': sem,
            'ci_lower': ci_lower, 'ci_upper': ci_upper
        })
    
    stats_data = []
    for ca_val, group_data in analysis_df.groupby('CA%')[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA%'] = ca_val
        stats_data.append(stats)
    
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA% {row['CA%']:.1f}: n={int(row['n'])}, "
              f"mean={row['mean']:.2f}±{row['sem']:.2f}, "
              f"95% CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
    
    # Perform mixed ANOVA
    print(f"\nStep 2: Running 2-way mixed ANOVA...")
    print("  Design: Week (within) × CA% (between)")
    
    try:
        anova_result = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='CA%',
            ss_type=3
        )
        
        print("\n[OK] Mixed ANOVA completed successfully")
        print("\nANOVA Results:")
        print(anova_result.to_string())
        
        results = {
            'measure': measure,
            'sex': sex,
            'type': 'sex_stratified_mixed_anova',
            'design': 'Week (within) × CA% (between)',
            'n_observations': len(analysis_df),
            'n_subjects': analysis_df['ID'].nunique(),
            'n_weeks': len(time_points),
            'weeks': time_points,
            'ca_levels': sorted(analysis_df['CA%'].unique()),
            'anova_table': anova_result,
            'descriptive_stats': group_stats
        }
        
        # Extract effects
        week_effect = anova_result[anova_result['Source'] == 'Week']
        if not week_effect.empty:
            results['week_F'] = float(week_effect['F'].iloc[0])
            results['week_p'] = float(week_effect['p-unc'].iloc[0])
            results['week_significant'] = results['week_p'] < 0.05
        
        ca_effect = anova_result[anova_result['Source'] == 'CA%']
        if not ca_effect.empty:
            results['ca_F'] = float(ca_effect['F'].iloc[0])
            results['ca_p'] = float(ca_effect['p-unc'].iloc[0])
            results['ca_significant'] = results['ca_p'] < 0.05
        
        interaction = anova_result[anova_result['Source'] == 'Interaction']
        if not interaction.empty:
            results['interaction_F'] = float(interaction['F'].iloc[0])
            results['interaction_p'] = float(interaction['p-unc'].iloc[0])
            results['interaction_significant'] = results['interaction_p'] < 0.05
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        return {'error': str(e)}


def perform_lick_anova_ca_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    ca_percent: float,
    measure: str = "Total_Licks",
    time_points: Optional[List[int]] = None
) -> Dict:
    """Perform 2-Way Mixed ANOVA: Week (within) × Sex (between), holding CA% constant.
    
    This analyzes longitudinal lick changes for ONE CA% level at a time.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        ca_percent: CA% level to analyze (e.g., 0.0 or 2.0)
        measure: Lick measure to analyze
        time_points: Optional list of specific weeks to include
        
    Returns:
        Dictionary with ANOVA results for the specified CA% level
    """
    print("\n" + "="*80)
    print(f"CA%-STRATIFIED LICK MIXED ANOVA: WEEK (WITHIN) × SEX (BETWEEN)")
    print(f"Analyzing: {ca_percent}% CA only")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed.")
        return {'error': 'pingouin not installed'}
    
    # Combine cohorts
    print("\nStep 1: Combining and filtering data...")
    combined_df = combine_lick_cohorts(cohort_dfs)
    
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    
    # Filter to specified CA%
    required_cols = ['ID', 'Week', 'Sex', 'CA%', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[combined_df['CA%'] == ca_percent][required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) == 0:
        print(f"[ERROR] No data found for CA% = {ca_percent}")
        return {'error': f'No data for CA% {ca_percent}'}
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(time_points)]
    else:
        time_points = sorted(analysis_df['Week'].unique())
    
    print(f"\nAnalyzing: {measure} ({ca_percent}% CA)")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  Sex levels: {sorted(analysis_df['Sex'].unique())}")
    print(f"  Weeks: {len(time_points)} weeks")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by Sex:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std()
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem if not np.isnan(sem) else np.nan
        ci_upper = mean + 1.96 * sem if not np.isnan(sem) else np.nan
        return pd.Series({
            'n': n, 'mean': mean, 'std': std, 'sem': sem,
            'ci_lower': ci_lower, 'ci_upper': ci_upper
        })
    
    stats_data = []
    for sex_val, group_data in analysis_df.groupby('Sex')[measure]:
        stats = compute_desc_stats(group_data)
        stats['Sex'] = sex_val
        stats_data.append(stats)
    
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  Sex {row['Sex']}: n={int(row['n'])}, "
              f"mean={row['mean']:.2f}±{row['sem']:.2f}, "
              f"95% CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
    
    # Perform mixed ANOVA
    print(f"\nStep 2: Running 2-way mixed ANOVA...")
    print("  Design: Week (within) × Sex (between)")
    
    try:
        anova_result = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='Sex',
            ss_type=3
        )
        
        print("\n[OK] Mixed ANOVA completed successfully")
        print("\nANOVA Results:")
        print(anova_result.to_string())
        
        results = {
            'measure': measure,
            'ca_percent': ca_percent,
            'type': 'ca_stratified_mixed_anova',
            'design': 'Week (within) × Sex (between)',
            'n_observations': len(analysis_df),
            'n_subjects': analysis_df['ID'].nunique(),
            'n_weeks': len(time_points),
            'weeks': time_points,
            'sex_levels': sorted(analysis_df['Sex'].unique()),
            'anova_table': anova_result,
            'descriptive_stats': group_stats
        }
        
        # Extract effects
        week_effect = anova_result[anova_result['Source'] == 'Week']
        if not week_effect.empty:
            results['week_F'] = float(week_effect['F'].iloc[0])
            results['week_p'] = float(week_effect['p-unc'].iloc[0])
            results['week_significant'] = results['week_p'] < 0.05
        
        sex_effect = anova_result[anova_result['Source'] == 'Sex']
        if not sex_effect.empty:
            results['sex_F'] = float(sex_effect['F'].iloc[0])
            results['sex_p'] = float(sex_effect['p-unc'].iloc[0])
            results['sex_significant'] = results['sex_p'] < 0.05
        
        interaction = anova_result[anova_result['Source'] == 'Interaction']
        if not interaction.empty:
            results['interaction_F'] = float(interaction['F'].iloc[0])
            results['interaction_p'] = float(interaction['p-unc'].iloc[0])
            results['interaction_significant'] = results['interaction_p'] < 0.05
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        return {'error': str(e)}


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_lick_cohort_report(
    mixed_results: Optional[Dict] = None,
    between_results: Optional[Dict] = None,
    results_males: Optional[Dict] = None,
    results_females: Optional[Dict] = None,
    results_ca0: Optional[Dict] = None,
    results_ca2: Optional[Dict] = None,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> str:
    """Generate comprehensive cross-cohort lick analysis report.
    
    Parameters:
        mixed_results: Results from perform_cross_cohort_lick_anova
        between_results: Results from perform_between_subjects_lick_anova
        results_males: Results from perform_lick_anova_sex_stratified (males)
        results_females: Results from perform_lick_anova_sex_stratified (females)
        results_ca0: Results from perform_lick_anova_ca_stratified (0% CA)
        results_ca2: Results from perform_lick_anova_ca_stratified (2% CA)
        cohort_dfs: Original cohort dataframes
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-COHORT LICK ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Cohort summary
    if cohort_dfs is not None:
        lines.append("COHORT SUMMARY")
        lines.append("-" * 80)
        for label, df in cohort_dfs.items():
            n_animals = df['ID'].nunique() if 'ID' in df.columns else len(df)
            n_records = len(df)
            ca_pct = df['CA%'].iloc[0] if 'CA%' in df.columns else 'Unknown'
            lines.append(f"  {label}: {n_animals} animals, {n_records} records, CA% = {ca_pct}")
        lines.append("")
    
    # Mixed ANOVA results
    if mixed_results is not None and 'error' not in mixed_results:
        lines.append("=" * 80)
        lines.append("MIXED ANOVA: CA% × WEEK × SEX")
        lines.append("=" * 80)
        lines.append(f"Measure: {mixed_results['measure']}")
        lines.append(f"Design: {mixed_results['design']}")
        lines.append(f"N subjects: {mixed_results['n_subjects']}")
        lines.append(f"N observations: {mixed_results['n_observations']}")
        lines.append(f"Weeks analyzed: {mixed_results['weeks']}")
        lines.append("")
        
        # Main effects
        lines.append("MAIN EFFECTS:")
        lines.append("-" * 40)
        
        if 'week_F' in mixed_results:
            sig = "***" if mixed_results.get('week_significant', False) else "ns"
            lines.append(f"  Week: F = {mixed_results['week_F']:.3f}, "
                        f"p = {mixed_results['week_p']:.4f} {sig}")
        
        if 'ca_F' in mixed_results:
            sig = "***" if mixed_results.get('ca_significant', False) else "ns"
            lines.append(f"  CA%: F = {mixed_results['ca_F']:.3f}, "
                        f"p = {mixed_results['ca_p']:.4f} {sig}")
        
        if 'sex_F' in mixed_results:
            sig = "***" if mixed_results.get('sex_significant', False) else "ns"
            lines.append(f"  Sex: F = {mixed_results['sex_F']:.3f}, "
                        f"p = {mixed_results['sex_p']:.4f} {sig}")
        
        lines.append("")
        
        # Interactions
        lines.append("INTERACTIONS:")
        lines.append("-" * 40)
        
        if 'week_ca_interaction_F' in mixed_results:
            sig = "***" if mixed_results.get('week_ca_interaction_significant', False) else "ns"
            lines.append(f"  Week × CA%: F = {mixed_results['week_ca_interaction_F']:.3f}, "
                        f"p = {mixed_results['week_ca_interaction_p']:.4f} {sig}")
        
        if 'week_sex_interaction_F' in mixed_results:
            sig = "***" if mixed_results.get('week_sex_interaction_significant', False) else "ns"
            lines.append(f"  Week × Sex: F = {mixed_results['week_sex_interaction_F']:.3f}, "
                        f"p = {mixed_results['week_sex_interaction_p']:.4f} {sig}")
        
        lines.append("")
        
        # Descriptive statistics
        if 'descriptive_stats' in mixed_results:
            lines.append("DESCRIPTIVE STATISTICS BY GROUP:")
            lines.append("-" * 80)
            stats_df = mixed_results['descriptive_stats']
            for _, row in stats_df.iterrows():
                lines.append(f"  CA% {row['CA%']:.1f}, Sex {row['Sex']}: "
                           f"n={int(row['n'])}, mean={row['mean']:.2f}±{row['sem']:.2f}, "
                           f"95% CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
            lines.append("")
    
    # Between-subjects ANOVA results
    if between_results is not None and 'error' not in between_results:
        lines.append("=" * 80)
        lines.append("BETWEEN-SUBJECTS ANOVA: CA% × SEX")
        lines.append("=" * 80)
        lines.append(f"Measure: {between_results['measure']}")
        lines.append(f"Analysis: {between_results['analysis_type']}")
        lines.append(f"N subjects: {between_results['n_subjects']}")
        lines.append("")
        
        lines.append("MAIN EFFECTS:")
        lines.append("-" * 40)
        
        if 'ca_F' in between_results:
            sig = "***" if between_results.get('ca_significant', False) else "ns"
            lines.append(f"  CA%: F = {between_results['ca_F']:.3f}, "
                        f"p = {between_results['ca_p']:.4f} {sig}")
        
        if 'sex_F' in between_results:
            sig = "***" if between_results.get('sex_significant', False) else "ns"
            lines.append(f"  Sex: F = {between_results['sex_F']:.3f}, "
                        f"p = {between_results['sex_p']:.4f} {sig}")
        
        if 'interaction_F' in between_results:
            sig = "***" if between_results.get('interaction_significant', False) else "ns"
            lines.append(f"  CA% × Sex: F = {between_results['interaction_F']:.3f}, "
                        f"p = {between_results['interaction_p']:.4f} {sig}")
        
        lines.append("")
    
    # Sex-stratified results
    if results_males is not None or results_females is not None:
        lines.append("=" * 80)
        lines.append("SEX-STRATIFIED ANALYSES: WEEK × CA%")
        lines.append("=" * 80)
        
        for sex_label, sex_results in [("MALES", results_males), ("FEMALES", results_females)]:
            if sex_results is not None and 'error' not in sex_results:
                lines.append("")
                lines.append(f"{sex_label}:")
                lines.append("-" * 40)
                lines.append(f"  N subjects: {sex_results['n_subjects']}")
                lines.append(f"  N weeks: {sex_results['n_weeks']}")
                
                if 'week_F' in sex_results:
                    sig = "***" if sex_results.get('week_significant', False) else "ns"
                    lines.append(f"  Week: F = {sex_results['week_F']:.3f}, "
                               f"p = {sex_results['week_p']:.4f} {sig}")
                
                if 'ca_F' in sex_results:
                    sig = "***" if sex_results.get('ca_significant', False) else "ns"
                    lines.append(f"  CA%: F = {sex_results['ca_F']:.3f}, "
                               f"p = {sex_results['ca_p']:.4f} {sig}")
                
                if 'interaction_F' in sex_results:
                    sig = "***" if sex_results.get('interaction_significant', False) else "ns"
                    lines.append(f"  Week × CA%: F = {sex_results['interaction_F']:.3f}, "
                               f"p = {sex_results['interaction_p']:.4f} {sig}")
        
        lines.append("")
    
    # CA%-stratified results
    if results_ca0 is not None or results_ca2 is not None:
        lines.append("=" * 80)
        lines.append("CA%-STRATIFIED ANALYSES: WEEK × SEX")
        lines.append("=" * 80)
        
        for ca_label, ca_results in [("0% CA", results_ca0), ("2% CA", results_ca2)]:
            if ca_results is not None and 'error' not in ca_results:
                lines.append("")
                lines.append(f"{ca_label}:")
                lines.append("-" * 40)
                lines.append(f"  N subjects: {ca_results['n_subjects']}")
                lines.append(f"  N weeks: {ca_results['n_weeks']}")
                
                if 'week_F' in ca_results:
                    sig = "***" if ca_results.get('week_significant', False) else "ns"
                    lines.append(f"  Week: F = {ca_results['week_F']:.3f}, "
                               f"p = {ca_results['week_p']:.4f} {sig}")
                
                if 'sex_F' in ca_results:
                    sig = "***" if ca_results.get('sex_significant', False) else "ns"
                    lines.append(f"  Sex: F = {ca_results['sex_F']:.3f}, "
                               f"p = {ca_results['sex_p']:.4f} {sig}")
                
                if 'interaction_F' in ca_results:
                    sig = "***" if ca_results.get('interaction_significant', False) else "ns"
                    lines.append(f"  Week × Sex: F = {ca_results['interaction_F']:.3f}, "
                               f"p = {ca_results['interaction_p']:.4f} {sig}")
        
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    print("\n" + report)
    return report


# =============================================================================
# FRONTLOADING / TEMPORAL DISTRIBUTION REPORTS
# =============================================================================

def generate_lick_frontloading_descriptives_report(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> str:
    """Generate a plain-text descriptive statistics report for:

    - % of licks in the first 5 minutes (First_5min_Lick_Pct)
    - Time to reach 50 % of total licks in minutes (Time_to_50pct_Licks)

    Statistics are broken down by Cohort × Week, then by Cohort × Week × Sex.
    No statistical tests are performed — this is purely descriptive so you can
    verify the calculations before running inferential analyses.

    Parameters
    ----------
    cohort_dfs : dict from load_lick_cohorts()
    save_path  : if given, write the report text to this file

    Returns
    -------
    str  the full report text
    """
    METRICS = [
        ('First_5min_Lick_Pct',   '% of Licks in First 5 Minutes'),
        ('Time_to_50pct_Licks',   'Time to 50% of Total Licks (minutes)'),
    ]

    lines = []
    lines.append("=" * 80)
    lines.append("LICK FRONTLOADING — DESCRIPTIVE STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Metrics")
    lines.append("  First_5min_Lick_Pct  : % of licks detected in the first 300 s of the session")
    lines.append("  Time_to_50pct_Licks  : Time (min) at which cumulative lick count reaches 50% of")
    lines.append("                         the session total (a measure of frontloading)")
    lines.append("")
    lines.append("Note: data are from the loaded cohorts; re-run the script to refresh.")
    lines.append("")

    # Combine and check
    combined_dfs = []
    for label, df in cohort_dfs.items():
        d = df.copy()
        if 'Cohort' not in d.columns:
            d['Cohort'] = label
        combined_dfs.append(d)

    if not combined_dfs:
        lines.append("[ERROR] No cohort data available.")
        report = "\n".join(lines)
        if save_path:
            Path(save_path).write_text(report, encoding='utf-8')
        return report

    combined = pd.concat(combined_dfs, ignore_index=True)

    missing_cols = [m for m, _ in METRICS if m not in combined.columns]
    if missing_cols:
        lines.append(f"[ERROR] The following metric columns are not present in the loaded data:")
        for col in missing_cols:
            lines.append(f"  - {col}")
        lines.append("")
        lines.append("  These columns are computed during data loading.  Please re-run the script")
        lines.append("  so that the new metrics are calculated when the files are processed.")
        report = "\n".join(lines)
        if save_path:
            Path(save_path).write_text(report, encoding='utf-8')
        return report

    def _desc_stats(series: 'pd.Series') -> dict:
        s = series.dropna()
        n = len(s)
        if n == 0:
            return dict(n=0, mean=np.nan, sem=np.nan, sd=np.nan, median=np.nan,
                        min=np.nan, max=np.nan)
        mean = s.mean()
        sd   = s.std(ddof=1) if n > 1 else 0.0
        sem  = sd / np.sqrt(n)
        return dict(n=n, mean=mean, sem=sem, sd=sd,
                    median=s.median(), min=s.min(), max=s.max())

    def _fmt(v) -> str:
        return 'N/A' if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"

    cohort_labels = list(cohort_dfs.keys())
    has_week = 'Week' in combined.columns
    has_sex  = 'Sex' in combined.columns

    for (col, label_long) in METRICS:
        lines.append("=" * 80)
        lines.append(f"METRIC: {label_long}")
        lines.append("=" * 80)
        lines.append("")

        # ---- Overall cohort summary (collapsed over weeks & sex) ----
        lines.append("-" * 60)
        lines.append("Overall summary by cohort (all weeks combined)")
        lines.append("-" * 60)
        lines.append(f"  {'Cohort':<20} {'N':>5} {'Mean':>8} {'SEM':>8} {'SD':>8} "
                     f"{'Median':>8} {'Min':>8} {'Max':>8}")
        lines.append("  " + "-" * 75)
        for coh in cohort_labels:
            s = _desc_stats(combined.loc[combined['Cohort'] == coh, col])
            lines.append(f"  {coh:<20} {s['n']:>5} {_fmt(s['mean']):>8} "
                         f"{_fmt(s['sem']):>8} {_fmt(s['sd']):>8} "
                         f"{_fmt(s['median']):>8} {_fmt(s['min']):>8} {_fmt(s['max']):>8}")
        lines.append("")

        # ---- By cohort × week ----
        if has_week:
            all_weeks = sorted(combined['Week'].dropna().unique())
            lines.append("-" * 60)
            lines.append("By cohort × week")
            lines.append("-" * 60)
            lines.append(f"  {'Cohort':<20} {'Week':>5} {'N':>5} {'Mean':>8} "
                         f"{'SEM':>8} {'SD':>8} {'Median':>8}")
            lines.append("  " + "-" * 65)
            for coh in cohort_labels:
                sub = combined[combined['Cohort'] == coh]
                for wk in all_weeks:
                    s = _desc_stats(sub.loc[sub['Week'] == wk, col])
                    wk_label = f"Week {int(wk) + 1}"  # 0-indexed → display 1-indexed
                    lines.append(f"  {coh:<20} {wk_label:>5} {s['n']:>5} "
                                 f"{_fmt(s['mean']):>8} {_fmt(s['sem']):>8} "
                                 f"{_fmt(s['sd']):>8} {_fmt(s['median']):>8}")
            lines.append("")

        # ---- By cohort × sex (if available) ----
        if has_sex:
            sex_vals = [v for v in combined['Sex'].dropna().unique()
                        if str(v).upper() not in ('UNKNOWN', 'NAN', '')]
            if sex_vals:
                lines.append("-" * 60)
                lines.append("By cohort × sex (all weeks combined)")
                lines.append("-" * 60)
                lines.append(f"  {'Cohort':<20} {'Sex':>5} {'N':>5} {'Mean':>8} "
                             f"{'SEM':>8} {'SD':>8} {'Median':>8}")
                lines.append("  " + "-" * 65)
                for coh in cohort_labels:
                    sub = combined[combined['Cohort'] == coh]
                    for sx in sorted(sex_vals):
                        s = _desc_stats(sub.loc[sub['Sex'] == sx, col])
                        lines.append(f"  {coh:<20} {sx:>5} {s['n']:>5} "
                                     f"{_fmt(s['mean']):>8} {_fmt(s['sem']):>8} "
                                     f"{_fmt(s['sd']):>8} {_fmt(s['median']):>8}")
                lines.append("")

        # ---- By cohort × week × sex ----
        if has_week and has_sex and sex_vals:
            lines.append("-" * 60)
            lines.append("By cohort × week × sex")
            lines.append("-" * 60)
            lines.append(f"  {'Cohort':<20} {'Week':>5} {'Sex':>5} {'N':>5} "
                         f"{'Mean':>8} {'SEM':>8} {'SD':>8}")
            lines.append("  " + "-" * 65)
            for coh in cohort_labels:
                sub = combined[combined['Cohort'] == coh]
                for wk in all_weeks:
                    for sx in sorted(sex_vals):
                        s = _desc_stats(
                            sub.loc[(sub['Week'] == wk) & (sub['Sex'] == sx), col])
                        wk_label = f"Week {int(wk) + 1}"
                        lines.append(f"  {coh:<20} {wk_label:>5} {sx:>5} {s['n']:>5} "
                                     f"{_fmt(s['mean']):>8} {_fmt(s['sem']):>8} "
                                     f"{_fmt(s['sd']):>8}")
            lines.append("")

        # ---- Per-animal raw values ----
        lines.append("-" * 60)
        lines.append("Per-animal values (all sessions)")
        lines.append("-" * 60)
        id_col  = 'ID'   if 'ID'   in combined.columns else None
        wk_col  = 'Week' if has_week else None
        sex_col = 'Sex'  if has_sex  else None
        header_parts = ['Cohort', 'ID', 'Week', 'Sex']
        lines.append("  " + "  ".join(f"{p:<10}" for p in header_parts) + f"  {col}")
        lines.append("  " + "-" * 65)
        for coh in cohort_labels:
            sub = combined[combined['Cohort'] == coh].sort_values(
                [c for c in ['ID', 'Week'] if c in combined.columns]
            )
            for _, row in sub.iterrows():
                val = row.get(col, np.nan)
                _id  = str(row[id_col])  if id_col  else '-'
                _wk  = f"Week {int(row[wk_col]) + 1}" if wk_col else '-'
                _sx  = str(row[sex_col]) if sex_col  else '-'
                lines.append(f"  {coh:<10}  {_id:<10}  {_wk:<10}  {_sx:<10}  {_fmt(val)}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF FRONTLOADING DESCRIPTIVES REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if save_path is not None:
        Path(save_path).write_text(report, encoding='utf-8')
        print(f"[OK] Frontloading descriptives report saved -> {save_path}")

    return report


# =============================================================================
# OMNIBUS LICK ANOVA (4 MEASURES, BH-FDR CORRECTED)
# =============================================================================

# Measures included in the omnibus analysis
_OMNIBUS_MEASURES = ["Total_Licks", "Total_Bouts", "First_5min_Lick_Pct", "Time_to_50pct_Licks",
                    "Fecal_Count_Sqrt"]
_OMNIBUS_MEASURE_LABELS = {
    "Total_Licks":         "Total Licks",
    "Total_Bouts":         "Total Lick Bouts",
    "First_5min_Lick_Pct": "% Licks in First 5 min",
    "Time_to_50pct_Licks": "Time to 50% Licks (min)",
    "Fecal_Count_Sqrt":    "Fecal Count (\u221a-transformed)",
}


def _add_sqrt_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add square-root transformed columns for count measures.

    sqrt(Fecal_Count) stabilises variance for count data (0-18 range).
    clip(lower=0) guards against any rogue negative values before sqrt.
    """
    df = df.copy()
    if 'Fecal_Count' in df.columns:
        df['Fecal_Count_Sqrt'] = np.sqrt(df['Fecal_Count'].clip(lower=0))
    return df


def _poisson_gof_section(counts: pd.Series, label: str) -> List[str]:
    """Chi-square GoF test (Poisson) for a single group of fecal counts.

    Bins with expected frequency < 5 are merged from the tails inward.
    df = n_bins - 2  (one for the normalisation constraint, one for estimated λ).
    Returns a list of text lines suitable for embedding in a report.
    """
    counts = counts.dropna().astype(int)
    n = len(counts)
    if n < 5:
        return [f'  {label}: insufficient data (n={n}), skipping.']

    lam = counts.mean()

    # Build observed frequency table for every integer 0 … max_count
    max_count = int(counts.max())
    obs_series = counts.value_counts().sort_index()
    full_index = range(0, max_count + 1)
    observed = np.array([obs_series.get(k, 0) for k in full_index], dtype=float)
    expected = np.array([stats.poisson.pmf(k, lam) * n for k in full_index], dtype=float)

    # Merge right-tail bins with expected < 5
    while len(expected) > 2 and expected[-1] < 5:
        observed[-2] += observed[-1]
        expected[-2] += expected[-1]
        observed = observed[:-1]
        expected = expected[:-1]

    # Merge left-tail bins with expected < 5
    while len(expected) > 2 and expected[0] < 5:
        observed[1] += observed[0]
        expected[1] += expected[0]
        observed = observed[1:]
        expected = expected[1:]

    n_bins = len(observed)
    if n_bins < 2:
        return [f'  {label}: too few bins after merging (n={n}, λ={lam:.2f}), skipping.']

    # Compute chi-square statistic and p-value with df = n_bins - 2
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    df = max(n_bins - 2, 1)  # df = bins - 1 (normalisation) - 1 (estimated lambda)
    p_val = float(stats.chi2.sf(chi2_stat, df))

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    conclusion = ('SIGNIFICANT departure from Poisson' if p_val < 0.05
                  else 'No significant departure from Poisson')
    return [
        f'  {label} (n={n}, λ_est={lam:.3f}):',
        f'    χ²({df}) = {chi2_stat:.3f},  p = {p_val:.4f}  [{sig}]  →  {conclusion}',
        f'    ({n_bins} bins after merging)',
    ]


def generate_fecal_poisson_gof_report(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> str:
    """Chi-square goodness-of-fit test: does Fecal_Count follow a Poisson distribution?

    Tests are run overall, by cohort (CA%), by sex, and by cohort × sex.
    Bins with expected frequency < 5 are merged before testing.
    One degree of freedom is subtracted for the estimated Poisson mean (λ).

    Parameters
    ----------
    cohort_dfs : dict of label → per-animal weekly DataFrame (from combine_lick_cohorts)
    save_path  : if provided, write the report text to this file

    Returns
    -------
    Report as a string
    """
    combined = combine_lick_cohorts(cohort_dfs)

    if 'Fecal_Count' not in combined.columns:
        return '  [WARNING] Fecal_Count column not present in combined data.\n'

    lines: List[str] = [
        '=' * 80,
        'FECAL COUNT — CHI-SQUARE GOODNESS-OF-FIT TEST FOR POISSON DISTRIBUTION',
        '=' * 80,
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'Null hypothesis : Fecal_Count values follow a Poisson distribution.',
        'Method          : Frequencies tabulated per integer count value.',
        '                  Bins with expected freq < 5 are merged into adjacent bins.',
        '                  df = n_bins – 2  (normalisation constraint + estimated λ).',
        '',
    ]

    # Overall
    lines.append('OVERALL')
    lines.append('-' * 40)
    lines += _poisson_gof_section(combined['Fecal_Count'], 'All data combined')
    lines.append('')

    # By cohort (CA%)
    lines.append('BY COHORT (CA%)')
    lines.append('-' * 40)
    if 'CA%' in combined.columns:
        for ca_val in sorted(combined['CA%'].dropna().unique()):
            sub = combined[combined['CA%'] == ca_val]['Fecal_Count']
            lines += _poisson_gof_section(sub, f'CA% = {ca_val}')
    else:
        lines.append('  CA% column not available.')
    lines.append('')

    # By sex
    lines.append('BY SEX')
    lines.append('-' * 40)
    if 'Sex' in combined.columns:
        for sex_val in sorted(combined['Sex'].dropna().unique()):
            sub = combined[combined['Sex'] == sex_val]['Fecal_Count']
            lines += _poisson_gof_section(sub, f'Sex = {sex_val}')
    else:
        lines.append('  Sex column not available.')
    lines.append('')

    # By cohort x sex
    lines.append('BY COHORT × SEX')
    lines.append('-' * 40)
    if 'CA%' in combined.columns and 'Sex' in combined.columns:
        for ca_val in sorted(combined['CA%'].dropna().unique()):
            for sex_val in sorted(combined['Sex'].dropna().unique()):
                sub = combined[
                    (combined['CA%'] == ca_val) & (combined['Sex'] == sex_val)
                ]['Fecal_Count']
                lines += _poisson_gof_section(sub, f'CA% = {ca_val}, Sex = {sex_val}')
    else:
        lines.append('  CA% or Sex column not available.')
    lines.append('')

    lines += [
        'INTERPRETATION NOTES',
        '-' * 40,
        '  * A non-significant result means Poisson is plausible (does not confirm it).',
        '  * A significant result indicates the observed distribution departs from Poisson',
        '    (e.g. overdispersion: variance > mean, or excess zeros).',
        '  * The square-root transform used in the omnibus ANOVAs is the standard',
        '    variance-stabilising transform for Poisson-like count data and remains',
        '    appropriate regardless of this test’s outcome.',
        '',
    ]

    report = '\n'.join(lines)
    if save_path is not None:
        Path(save_path).write_text(report, encoding='utf-8')
        print(f'[OK] Fecal Poisson GoF report saved -> {save_path}')
    return report


def _shapiro_row(values: pd.Series, label: str, alpha: float = 0.05) -> Tuple[List[str], bool]:
    """Run Shapiro-Wilk (or D'Agostino-Pearson for n>5000) on *values*.

    Returns (text_lines, passed_normality).
    """
    values = values.dropna()
    n = len(values)
    if n < 3:
        return [f'  {label}: n={n} — too few observations, skipping.'], True  # conservative
    if n > 5000:
        stat, p = stats.normaltest(values)
        test_name = "D'Agostino-Pearson k²"
    else:
        stat, p = stats.shapiro(values)
        test_name = 'Shapiro-Wilk W'
    sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    passed = (p >= alpha)
    conclusion = 'NORMAL (p ≥ 0.05)' if passed else 'NON-NORMAL (p < 0.05)'
    lines = [
        f'  {label} (n={n}):',
        f'    {test_name} = {stat:.4f},  p = {p:.4f}  [{sig}]  →  {conclusion}',
    ]
    return lines, passed


def generate_fecal_normality_report(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> str:
    """Shapiro-Wilk normality tests on raw and √-transformed Fecal_Count data.

    Tests are stratified by Week, Cohort (CA%), Sex, and Cohort × Week to
    help decide whether parametric mixed ANOVA or the Friedman test is more
    appropriate for the repeated-measures (Week) factor.

    Parameters
    ----------
    cohort_dfs : dict of label → per-animal weekly DataFrame
    save_path  : if provided, write the report text to this file

    Returns
    -------
    Report as a string
    """
    combined = combine_lick_cohorts(cohort_dfs)
    if 'Fecal_Count' not in combined.columns:
        return '  [WARNING] Fecal_Count column not present in combined data.\n'

    combined = _add_sqrt_transforms(combined)

    lines: List[str] = [
        '=' * 80,
        'FECAL COUNT — SHAPIRO-WILK NORMALITY TESTS',
        '(Evaluating whether the Friedman test is appropriate)',
        '=' * 80,
        f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'Tests are run on both raw Fecal_Count and √(Fecal_Count).',
        'Shapiro-Wilk is used for n ≤ 5000; D\'Agostino-Pearson for larger samples.',
        'The Friedman test is the non-parametric repeated-measures alternative',
        'to one-way within-subjects ANOVA, appropriate when normality is violated.',
        '',
    ]

    # Track pass/fail counts for the recommendation
    raw_fails = 0
    raw_total = 0
    sqrt_fails = 0
    sqrt_total = 0

    def _run_pair(subset: pd.Series, label: str) -> None:
        nonlocal raw_fails, raw_total, sqrt_fails, sqrt_total
        r_lines, r_passed = _shapiro_row(subset, f'RAW   {label}')
        lines.extend(r_lines)
        if subset.dropna().shape[0] >= 3:
            raw_total  += 1
            if not r_passed:
                raw_fails += 1

        sqrt_vals = np.sqrt(subset.clip(lower=0))
        s_lines, s_passed = _shapiro_row(sqrt_vals, f'SQRT  {label}')
        lines.extend(s_lines)
        if sqrt_vals.dropna().shape[0] >= 3:
            sqrt_total  += 1
            if not s_passed:
                sqrt_fails += 1

    # ── Overall ──────────────────────────────────────────────────────────────
    lines += ['OVERALL', '-' * 40]
    _run_pair(combined['Fecal_Count'], 'All data combined')
    lines.append('')

    # ── By Week (most relevant for Friedman decision) ─────────────────────
    lines += ['BY WEEK  ← key for Friedman decision', '-' * 40]
    if 'Week' in combined.columns:
        for wk in sorted(combined['Week'].dropna().unique()):
            sub = combined[combined['Week'] == wk]['Fecal_Count']
            _run_pair(sub, f'Week {int(wk) + 1}')
    else:
        lines.append('  Week column not available.')
    lines.append('')

    # ── By Cohort (CA%) ───────────────────────────────────────────────────
    lines += ['BY COHORT (CA%)', '-' * 40]
    if 'CA%' in combined.columns:
        for ca_val in sorted(combined['CA%'].dropna().unique()):
            sub = combined[combined['CA%'] == ca_val]['Fecal_Count']
            _run_pair(sub, f'CA% = {ca_val}')
    else:
        lines.append('  CA% column not available.')
    lines.append('')

    # ── By Sex ────────────────────────────────────────────────────────────
    lines += ['BY SEX', '-' * 40]
    if 'Sex' in combined.columns:
        for sex_val in sorted(combined['Sex'].dropna().unique()):
            sub = combined[combined['Sex'] == sex_val]['Fecal_Count']
            _run_pair(sub, f'Sex = {sex_val}')
    else:
        lines.append('  Sex column not available.')
    lines.append('')

    # ── By Cohort × Week ──────────────────────────────────────────────────
    lines += ['BY COHORT × WEEK', '-' * 40]
    if 'CA%' in combined.columns and 'Week' in combined.columns:
        for ca_val in sorted(combined['CA%'].dropna().unique()):
            for wk in sorted(combined['Week'].dropna().unique()):
                sub = combined[
                    (combined['CA%'] == ca_val) & (combined['Week'] == wk)
                ]['Fecal_Count']
                _run_pair(sub, f'CA% = {ca_val}, Week {int(wk) + 1}')
    else:
        lines.append('  CA% or Week column not available.')
    lines.append('')

    # ── Recommendation ────────────────────────────────────────────────────
    raw_pct  = 100 * raw_fails  / raw_total  if raw_total  else 0
    sqrt_pct = 100 * sqrt_fails / sqrt_total if sqrt_total else 0

    lines += [
        '=' * 80,
        'RECOMMENDATION',
        '=' * 80,
        f'  Groups tested             : {raw_total}',
        f'  Raw  Fecal_Count — non-normal : {raw_fails}/{raw_total} ({raw_pct:.0f}%)',
        f'  Sqrt Fecal_Count — non-normal : {sqrt_fails}/{sqrt_total} ({sqrt_pct:.0f}%)',
        '',
    ]

    if raw_pct >= 50 and sqrt_pct >= 50:
        lines += [
            '  CONCLUSION: The majority of groups fail normality even after √-transform.',
            '  → The Friedman test (non-parametric repeated-measures) is RECOMMENDED',
            '    for the Week factor within each cohort/sex stratum.',
            '  → Consider the Aligned Ranks Transform (ART) ANOVA for a full factorial',
            '    non-parametric approach, or use the Friedman test per stratum.',
        ]
    elif raw_pct >= 50 and sqrt_pct < 50:
        lines += [
            '  CONCLUSION: Raw counts violate normality but √-transform fixes most groups.',
            '  → Parametric mixed ANOVA on √(Fecal_Count) is DEFENSIBLE.',
            '  → The Friedman test is still a conservative valid alternative if preferred.',
        ]
    else:
        lines += [
            '  CONCLUSION: Most groups do not significantly depart from normality.',
            '  → Parametric mixed ANOVA (raw or √-transformed) appears APPROPRIATE.',
            '  → The Friedman test would be overly conservative here, but remains valid.',
        ]

    lines += [
        '',
        'NOTE: Shapiro-Wilk has low power with very small n — a non-significant result',
        '      does not guarantee normality. Visual inspection (Q-Q plots) is also advised.',
        '',
    ]

    report = '\n'.join(lines)
    if save_path is not None:
        Path(save_path).write_text(report, encoding='utf-8')
        print(f'[OK] Fecal normality report saved -> {save_path}')
    return report


def _bonferroni_paired_ttests(data: pd.DataFrame, measure: str, within: str, subject: str) -> pd.DataFrame:
    """All-pairwise Bonferroni-corrected paired t-tests across levels of a within-subjects factor.

    Parameters
    ----------
    data     : long-format DataFrame (one row per subject × within-level)
    measure  : dependent variable column name
    within   : within-subjects factor column (e.g. 'Week')
    subject  : subject ID column

    Returns
    -------
    DataFrame with columns: A, B, t, df, p_raw, p_bonferroni, significant
    """
    levels = sorted(data[within].unique())
    pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i+1:]]
    n_comparisons = len(pairs)
    rows = []
    for a, b in pairs:
        vals_a = data[data[within] == a].set_index(subject)[measure]
        vals_b = data[data[within] == b].set_index(subject)[measure]
        common = vals_a.index.intersection(vals_b.index)
        if len(common) < 2:
            rows.append({within+'_A': a, within+'_B': b, 't': np.nan, 'df': np.nan,
                         'p_raw': np.nan, 'p_bonferroni': np.nan, 'significant': False})
            continue
        d_a = vals_a.loc[common].values
        d_b = vals_b.loc[common].values
        t_stat, p_raw = stats.ttest_rel(d_a, d_b)
        df_val = len(common) - 1
        p_bonf = min(p_raw * n_comparisons, 1.0)
        rows.append({within+'_A': a, within+'_B': b,
                     't': round(t_stat, 4), 'df': df_val,
                     'p_raw': round(p_raw, 6), 'p_bonferroni': round(p_bonf, 6),
                     'significant': p_bonf < 0.05})
    return pd.DataFrame(rows)


def _tukey_between(data: pd.DataFrame, measure: str, factor: str) -> Optional[object]:
    """Tukey HSD post-hoc for a between-subjects factor.

    Returns statsmodels TukeyHSDResults object, or None on failure.
    """
    try:
        groups = data[factor].astype(str)
        result = pairwise_tukeyhsd(endog=data[measure].values, groups=groups.values, alpha=0.05)
        return result
    except Exception as e:
        print(f"  [WARNING] Tukey HSD failed for {factor}: {e}")
        return None


def _bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in original order."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = np.array(p_values) * n / ranked
    # Enforce monotonicity from the right
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return [float(min(v, 1.0)) for v in adjusted]


def perform_omnibus_lick_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measures: Optional[List[str]] = None,
    time_points: Optional[List[int]] = None,
) -> Dict:
    """Omnibus 3-way mixed ANOVA across lick measures with BH-FDR correction.

    Design for each measure:
      - CA% (between-subjects)
      - Sex (between-subjects)
      - Week (within-subjects / repeated measures)

    Post-hoc procedure:
      - Significant CA% or Sex main effect → Tukey HSD on animal-level means
      - Significant Week main effect → Bonferroni all-pairwise paired t-tests across weeks
      - Significant interaction involving Week → Bonferroni pairwise paired t-tests
        within each between-group cell

    BH-FDR is applied across the omnibus Week F-tests of the 4 measures (matching the
    within-measures multiple-comparisons family). Between-factor F-tests are corrected
    within their own family of 4.

    Returns
    -------
    dict keyed by measure name, each containing:
      'anova'       : raw results dict from perform_cross_cohort_lick_anova
      'posthoc_week': DataFrame of Bonferroni paired t-tests across weeks (if Week sig)
      'posthoc_ca'  : Tukey results for CA% (if CA% sig)
      'posthoc_sex' : Tukey results for Sex (if Sex sig)
      'fdr_week_p'  : BH-corrected p for the Week F-test
      'fdr_ca_p'    : BH-corrected p for the CA% F-test
      'fdr_sex_p'   : BH-corrected p for the Sex F-test
      'analysis_df' : the prepared long-format DataFrame used for this measure
    """
    if measures is None:
        measures = _OMNIBUS_MEASURES

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed. Cannot run omnibus ANOVA.")
        return {}

    print("\n" + "=" * 80)
    print("OMNIBUS LICK ANOVA — 4 MEASURES, BH-FDR CORRECTED")
    print("=" * 80)
    print(f"Measures: {measures}")
    print("Design  : CA% (between) × Sex (between) × Week (within)")

    combined_df = combine_lick_cohorts(cohort_dfs)
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    combined_df = _add_sqrt_transforms(combined_df)

    all_results: Dict = {}
    week_p_values: List[float] = []
    ca_p_values:   List[float] = []
    sex_p_values:  List[float] = []

    # ------------------------------------------------------------------ #
    # Pass 1: run ANOVAs and collect omnibus p-values for FDR
    # ------------------------------------------------------------------ #
    for measure in measures:
        print(f"\n{'─'*60}")
        print(f"  ANOVA: {_OMNIBUS_MEASURE_LABELS.get(measure, measure)}")
        print(f"{'─'*60}")

        if measure not in combined_df.columns:
            print(f"  [WARNING] Column '{measure}' not found — skipping.")
            all_results[measure] = {'error': f'Column {measure} not found'}
            week_p_values.append(np.nan)
            ca_p_values.append(np.nan)
            sex_p_values.append(np.nan)
            continue

        # Subset to non-NaN rows (handles Time_to_50pct_Licks exclusions)
        req_cols = ['ID', 'Week', 'Sex', 'CA%', measure]
        adf = combined_df[req_cols].dropna().copy()

        if time_points is not None:
            adf = adf[adf['Week'].isin(time_points)]

        tp = sorted(adf['Week'].unique())

        try:
            # Pingouin requires complete cases for mixed ANOVA; drop animals with
            # missing weeks for this measure
            counts = adf.groupby('ID')['Week'].nunique()
            n_weeks = len(tp)
            complete_ids = counts[counts == n_weeks].index
            adf_complete = adf[adf['ID'].isin(complete_ids)].copy()

            n_dropped = adf['ID'].nunique() - len(complete_ids)
            if n_dropped > 0:
                print(f"  [INFO] {n_dropped} animal(s) excluded (incomplete weeks for {measure})")

            if adf_complete['CA%'].nunique() < 2 or adf_complete['ID'].nunique() < 4:
                raise ValueError("Insufficient subjects for mixed ANOVA")

            # Combined between factor for true 3-way (pingouin only supports 1 between)
            adf_complete['CA_Sex'] = (adf_complete['CA%'].astype(str) + '_'
                                      + adf_complete['Sex'].astype(str))

            tbl_combined = pg.mixed_anova(data=adf_complete, dv=measure,
                                          within='Week', subject='ID',
                                          between='CA_Sex')

            tbl_ca  = pg.mixed_anova(data=adf_complete, dv=measure,
                                     within='Week', subject='ID',
                                     between='CA%')

            tbl_sex = pg.mixed_anova(data=adf_complete, dv=measure,
                                     within='Week', subject='ID',
                                     between='Sex')

            def _get(tbl, src):
                row = tbl[tbl['Source'] == src]
                if row.empty:
                    return np.nan, np.nan, np.nan
                r = row.iloc[0]
                return float(r.get('F', np.nan)), float(r.get('p-unc', np.nan)), float(r.get('np2', np.nan))

            week_F,    week_p,    week_np2    = _get(tbl_ca,  'Week')
            ca_F,      ca_p,      ca_np2      = _get(tbl_ca,  'CA%')
            sex_F,     sex_p,     sex_np2     = _get(tbl_sex, 'Sex')
            wk_ca_F,   wk_ca_p,   wk_ca_np2  = _get(tbl_ca,  'Interaction')
            wk_sex_F,  wk_sex_p,  wk_sex_np2 = _get(tbl_sex, 'Interaction')
            grp_F,     grp_p,     grp_np2    = _get(tbl_combined, 'CA_Sex')
            wk_grp_F,  wk_grp_p,  wk_grp_np2 = _get(tbl_combined, 'Interaction')

            week_p_values.append(week_p)
            ca_p_values.append(ca_p)
            sex_p_values.append(sex_p)

            all_results[measure] = {
                'measure': measure,
                'analysis_df': adf_complete,
                'weeks': tp,
                'n_subjects': adf_complete['ID'].nunique(),
                'n_dropped': n_dropped,
                'tbl_combined': tbl_combined,
                'tbl_ca': tbl_ca,
                'tbl_sex': tbl_sex,
                'week_F': week_F,  'week_p': week_p,   'week_np2': week_np2,
                'ca_F':   ca_F,    'ca_p':   ca_p,     'ca_np2':   ca_np2,
                'sex_F':  sex_F,   'sex_p':  sex_p,    'sex_np2':  sex_np2,
                'wk_ca_F': wk_ca_F, 'wk_ca_p': wk_ca_p, 'wk_ca_np2': wk_ca_np2,
                'wk_sex_F': wk_sex_F, 'wk_sex_p': wk_sex_p, 'wk_sex_np2': wk_sex_np2,
                'grp_F': grp_F, 'grp_p': grp_p, 'grp_np2': grp_np2,
                'wk_grp_F': wk_grp_F, 'wk_grp_p': wk_grp_p, 'wk_grp_np2': wk_grp_np2,
            }

        except Exception as e:
            print(f"  [ERROR] ANOVA failed for {measure}: {e}")
            import traceback; traceback.print_exc()
            all_results[measure] = {'error': str(e)}
            week_p_values.append(np.nan)
            ca_p_values.append(np.nan)
            sex_p_values.append(np.nan)

    # ------------------------------------------------------------------ #
    # BH-FDR correction across the 4 omnibus F-test families
    # ------------------------------------------------------------------ #
    valid_week_p = [p if not np.isnan(p) else 1.0 for p in week_p_values]
    valid_ca_p   = [p if not np.isnan(p) else 1.0 for p in ca_p_values]
    valid_sex_p  = [p if not np.isnan(p) else 1.0 for p in sex_p_values]

    fdr_week = _bh_fdr(valid_week_p)
    fdr_ca   = _bh_fdr(valid_ca_p)
    fdr_sex  = _bh_fdr(valid_sex_p)

    for i, measure in enumerate(measures):
        if measure not in all_results or 'error' in all_results[measure]:
            continue
        all_results[measure]['fdr_week_p'] = fdr_week[i]
        all_results[measure]['fdr_ca_p']   = fdr_ca[i]
        all_results[measure]['fdr_sex_p']  = fdr_sex[i]

    # ------------------------------------------------------------------ #
    # Pass 2: post-hoc tests (driven by FDR-corrected omnibus significance)
    # ------------------------------------------------------------------ #
    for measure in measures:
        r = all_results.get(measure, {})
        if 'error' in r:
            continue

        adf = r['analysis_df']
        tp  = r['weeks']

        # --- Week post-hoc: Bonferroni all-pairwise paired t-tests ----------
        week_sig = r.get('fdr_week_p', 1.0) < 0.05 or r.get('wk_ca_p', 1.0) < 0.05 or r.get('wk_sex_p', 1.0) < 0.05
        if week_sig and len(tp) > 1:
            print(f"\n  Post-hoc (Week, Bonferroni): {_OMNIBUS_MEASURE_LABELS.get(measure, measure)}")
            ph = _bonferroni_paired_ttests(adf, measure, 'Week', 'ID')
            r['posthoc_week'] = ph

        # --- CA% post-hoc: Tukey on animal-level means ----------------------
        if r.get('fdr_ca_p', 1.0) < 0.05:
            print(f"  Post-hoc (CA%, Tukey): {_OMNIBUS_MEASURE_LABELS.get(measure, measure)}")
            animal_means = adf.groupby(['ID', 'CA%'])[measure].mean().reset_index()
            r['posthoc_ca'] = _tukey_between(animal_means, measure, 'CA%')

        # --- Sex post-hoc: Tukey on animal-level means -----------------------
        if r.get('fdr_sex_p', 1.0) < 0.05:
            print(f"  Post-hoc (Sex, Tukey): {_OMNIBUS_MEASURE_LABELS.get(measure, measure)}")
            animal_means = adf.groupby(['ID', 'Sex'])[measure].mean().reset_index()
            r['posthoc_sex'] = _tukey_between(animal_means, measure, 'Sex')

        # --- Interaction cell post-hocs: Bonferroni within each cell --------
        cell_posthocs = {}
        if r.get('wk_ca_p', 1.0) < 0.05:
            for ca_val in sorted(adf['CA%'].unique()):
                cell_df = adf[adf['CA%'] == ca_val]
                ph = _bonferroni_paired_ttests(cell_df, measure, 'Week', 'ID')
                cell_posthocs[f'CA%={ca_val}'] = ph
        if r.get('wk_sex_p', 1.0) < 0.05:
            for sex_val in sorted(adf['Sex'].unique()):
                cell_df = adf[adf['Sex'] == sex_val]
                ph = _bonferroni_paired_ttests(cell_df, measure, 'Week', 'ID')
                cell_posthocs[f'Sex={sex_val}'] = ph
        if cell_posthocs:
            r['posthoc_week_cells'] = cell_posthocs

    # Store ordered measure list for report use
    all_results['_measures'] = measures
    return all_results


# =============================================================================
# OMNIBUS LICK REPORT
# =============================================================================

def generate_omnibus_lick_report(
    omnibus_results: Dict,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    save_path: Optional[Path] = None,
) -> str:
    """Generate a comprehensive omnibus lick ANOVA report with post-hocs.

    Parameters
    ----------
    omnibus_results : output of perform_omnibus_lick_anova
    cohort_dfs      : original cohort DataFrames (for cohort summary header)
    save_path       : if provided, write the report to this file

    Returns
    -------
    Report as a string
    """
    measures = omnibus_results.get('_measures', _OMNIBUS_MEASURES)

    def _sig(p: float) -> str:
        if np.isnan(p): return 'n/a'
        if p < 0.001:   return '***'
        if p < 0.01:    return '**'
        if p < 0.05:    return '*'
        return 'ns'

    def _fmt_p(p: float) -> str:
        if np.isnan(p): return 'n/a'
        if p < 0.0001:  return '< 0.0001'
        return f'{p:.4f}'

    def _dec(p, fdr_p=None):
        chk = fdr_p if (fdr_p is not None and not np.isnan(fdr_p)) else p
        if np.isnan(chk): return 'n/a'
        return 'SIGNIFICANT' if chk < 0.05 else 'NOT SIGNIFICANT'

    def _ef(np2):
        if np.isnan(np2): return ''
        return f', ηp²={np2:.3f}'

    def _df_from_tbl(tbl, src):
        if tbl is None: return '?', '?'
        row = tbl[tbl['Source'] == src]
        if row.empty: return '?', '?'
        rv = row.iloc[0]
        return int(rv.get('DF1', 0)), int(rv.get('DF2', 0))

    lines = []
    lines += ['=' * 80,
              'OMNIBUS LICK ANOVA REPORT',
              '=' * 80,
              f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
              f'Measures  : {", ".join(_OMNIBUS_MEASURE_LABELS.get(m, m) for m in measures)}',
              f'Design    : CA% (between) × Sex (between) × Week (within)',
              f'Post-hocs : Tukey HSD for between-subjects; Bonferroni paired t-tests for within',
              f'Correction: BH-FDR across omnibus F-tests within each factor family (n={len(measures)})',
              '']

    # Cohort summary
    if cohort_dfs is not None:
        lines += ['COHORT SUMMARY', '-' * 40]
        for label, df in cohort_dfs.items():
            n_a = df['ID'].nunique() if 'ID' in df.columns else len(df)
            ca  = df['CA%'].iloc[0] if 'CA%' in df.columns else '?'
            sex_counts = (df.drop_duplicates('ID')['Sex'].value_counts().to_dict()
                          if 'Sex' in df.columns and 'ID' in df.columns else {})
            sex_str = ', '.join(f'{s}: {n}' for s, n in sorted(sex_counts.items()))
            lines.append(f'  {label}: {n_a} animals (CA% = {ca}){(", " + sex_str) if sex_str else ""}')
        lines.append('')

    # ------------------------------------------------------------------ #
    # BH-FDR summary table
    # ------------------------------------------------------------------ #
    lines += ['=' * 80,
              'BH-FDR CORRECTED OMNIBUS P-VALUES (α = 0.05)',
              '=' * 80,
              f'{"Measure":<32} {"Week (raw)":>12} {"Week (FDR)":>12} {"CA% (raw)":>12} '
              f'{"CA% (FDR)":>12} {"Sex (raw)":>12} {"Sex (FDR)":>12}']
    lines.append('-' * 96)
    for m in measures:
        r = omnibus_results.get(m, {})
        if 'error' in r:
            lines.append(f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<30} ERROR: {r["error"]}')
            continue
        label = _OMNIBUS_MEASURE_LABELS.get(m, m)
        lines.append(
            f'  {label:<30} '
            f'{_fmt_p(r.get("week_p", np.nan)):>12} '
            f'{_fmt_p(r.get("fdr_week_p", np.nan)):>12} '
            f'{_fmt_p(r.get("ca_p", np.nan)):>12} '
            f'{_fmt_p(r.get("fdr_ca_p", np.nan)):>12} '
            f'{_fmt_p(r.get("sex_p", np.nan)):>12} '
            f'{_fmt_p(r.get("fdr_sex_p", np.nan)):>12} '
        )
    lines.append('')

    # ------------------------------------------------------------------ #
    # Per-measure detailed sections
    # ------------------------------------------------------------------ #
    for m in measures:
        r = omnibus_results.get(m, {})
        label = _OMNIBUS_MEASURE_LABELS.get(m, m)
        divider = '━' * 80
        lines += ['', divider, f'  MEASURE: {label}', divider]

        if 'error' in r:
            lines.append(f'  ERROR: {r["error"]}')
            continue

        lines += [f'  N subjects (complete cases): {r.get("n_subjects", "?")}',
                  f'  Animals dropped (incomplete weeks): {r.get("n_dropped", 0)}',
                  f'  Weeks analyzed: {r.get("weeks", "?")}',
                  '']

        # Descriptive stats — per group (CA% × Sex), collapsed across weeks
        adf = r.get('analysis_df')
        if m == 'Fecal_Count_Sqrt' and adf is not None:
            lines.append('  NOTE: Values shown are √(Fecal_Count) transformed. '
                         'Back-transform by squaring to recover original counts.')
        if adf is not None:
            lines += ['  DESCRIPTIVE STATISTICS (by CA% × Sex, collapsed across weeks)',
                      '  ' + '-' * 70]
            for (ca_val, sex_val), grp in adf.groupby(['CA%', 'Sex'])[m]:
                n    = len(grp)
                mn   = grp.mean()
                sd   = grp.std(ddof=1) if n > 1 else np.nan
                se   = sd / np.sqrt(n) if n > 1 else np.nan
                ci_lo = mn - 1.96 * se if not np.isnan(se) else np.nan
                ci_hi = mn + 1.96 * se if not np.isnan(se) else np.nan
                ci_str = f'[{ci_lo:.2f}, {ci_hi:.2f}]' if not np.isnan(ci_lo) else 'n/a'
                lines.append(f'    CA%={ca_val:.0f}%, Sex={sex_val}: '
                              f'n={n}, M={mn:.2f}, SD={sd:.2f}, SEM={se:.2f}, 95% CI {ci_str}')
            lines.append('')

        # Print the pingouin ANOVA tables
        tbl_ca  = r.get('tbl_ca')
        tbl_sex = r.get('tbl_sex')
        for tbl_label, tbl in [('CA% × Week', tbl_ca), ('Sex × Week', tbl_sex)]:
            if tbl is not None:
                lines += [f'  ANOVA TABLE ({tbl_label} mixed ANOVA)',
                          '  ' + '-' * 70]
                disp_cols = [c for c in ['Source', 'DF1', 'DF2', 'F', 'p-unc',
                                          'p-GG-corr', 'np2', 'eps'] if c in tbl.columns]
                for ln in tbl[disp_cols].to_string(index=False).split('\n'):
                    lines.append('  ' + ln)
                lines.append('')

        # Interpretation block
        fdr_w    = r.get('fdr_week_p', np.nan)
        fdr_c    = r.get('fdr_ca_p',   np.nan)
        fdr_s    = r.get('fdr_sex_p',  np.nan)
        wk_ca_p  = r.get('wk_ca_p',   np.nan)
        wk_sex_p = r.get('wk_sex_p',  np.nan)
        wk_df1,   wk_df2   = _df_from_tbl(tbl_ca,  'Week')
        ca_df1,   ca_df2   = _df_from_tbl(tbl_ca,  'CA%')
        sex_df1,  sex_df2  = _df_from_tbl(tbl_sex, 'Sex')
        wkca_df1, wkca_df2 = _df_from_tbl(tbl_ca,  'Interaction')
        wksx_df1, wksx_df2 = _df_from_tbl(tbl_sex, 'Interaction')

        # Check whether GG correction was applied to Week (within-subjects)
        _wk_gg_p = np.nan
        if tbl_ca is not None and 'p-GG-corr' in tbl_ca.columns:
            _wk_row = tbl_ca[tbl_ca['Source'] == 'Week']
            if not _wk_row.empty:
                _wk_gg_p = float(_wk_row.iloc[0].get('p-GG-corr', np.nan))
        _wk_p_show = _wk_gg_p if not np.isnan(_wk_gg_p) else r.get('week_p', np.nan)
        _wk_p_lbl  = 'p (GG-corr)' if not np.isnan(_wk_gg_p) else 'p'

        lines += ['  INTERPRETATION', '  ' + '-' * 70]
        lines.append(
            f'  1. Week: {_dec(r.get("week_p", np.nan), fdr_w)}\n'
            f'     F({wk_df1}, {wk_df2}) = {r.get("week_F", np.nan):.2f}, '
            f'{_wk_p_lbl} = {_fmt_p(_wk_p_show)}, '
            f'p_FDR = {_fmt_p(fdr_w)} {_sig(fdr_w)}'
            f'{_ef(r.get("week_np2", np.nan))}'
        )
        lines.append(
            f'  2. CA%: {_dec(r.get("ca_p", np.nan), fdr_c)}\n'
            f'     F({ca_df1}, {ca_df2}) = {r.get("ca_F", np.nan):.2f}, '
            f'p = {_fmt_p(r.get("ca_p", np.nan))}, '
            f'p_FDR = {_fmt_p(fdr_c)} {_sig(fdr_c)}'
            f'{_ef(r.get("ca_np2", np.nan))}'
        )
        lines.append(
            f'  3. Sex: {_dec(r.get("sex_p", np.nan), fdr_s)}\n'
            f'     F({sex_df1}, {sex_df2}) = {r.get("sex_F", np.nan):.2f}, '
            f'p = {_fmt_p(r.get("sex_p", np.nan))}, '
            f'p_FDR = {_fmt_p(fdr_s)} {_sig(fdr_s)}'
            f'{_ef(r.get("sex_np2", np.nan))}'
        )
        lines.append(
            f'  4. Week × CA%: {"SIGNIFICANT" if (not np.isnan(wk_ca_p) and wk_ca_p < 0.05) else "NOT SIGNIFICANT"}\n'
            f'     F({wkca_df1}, {wkca_df2}) = {r.get("wk_ca_F", np.nan):.2f}, '
            f'p = {_fmt_p(wk_ca_p)} {_sig(wk_ca_p)}'
            f'{_ef(r.get("wk_ca_np2", np.nan))}'
        )
        lines.append(
            f'  5. Week × Sex: {"SIGNIFICANT" if (not np.isnan(wk_sex_p) and wk_sex_p < 0.05) else "NOT SIGNIFICANT"}\n'
            f'     F({wksx_df1}, {wksx_df2}) = {r.get("wk_sex_F", np.nan):.2f}, '
            f'p = {_fmt_p(wk_sex_p)} {_sig(wk_sex_p)}'
            f'{_ef(r.get("wk_sex_np2", np.nan))}'
        )
        lines.append('')

        # Post-hoc: Week (overall Bonferroni)
        if 'posthoc_week' in r:
            ph = r['posthoc_week']
            wcol_a = 'Week_A' if 'Week_A' in ph.columns else ph.columns[0]
            wcol_b = 'Week_B' if 'Week_B' in ph.columns else ph.columns[1]
            lines += ['  POST-HOC: Week (all-pairwise Bonferroni paired t-tests)',
                      '  ' + '-' * 60,
                      f'  {"Week A":>8} {"Week B":>8} {"t":>8} {"df":>6} '
                      f'{"p_raw":>12} {"p_bonf":>12} {"sig":>5}']
            for _, row in ph.iterrows():
                sig_marker = '*' if row['significant'] else ''
                lines.append(
                    f'  {str(int(row[wcol_a])):>8} {str(int(row[wcol_b])):>8} '
                    f'{row["t"]:>8.3f} {int(row["df"]):>6} '
                    f'{_fmt_p(row["p_raw"]):>12} {_fmt_p(row["p_bonferroni"]):>12} {sig_marker:>5}')
            lines.append('')

        _tukey_note = '  (reject=True means the groups differ significantly at FWER=0.05)'

        # Post-hoc: CA% (Tukey)
        if 'posthoc_ca' in r and r['posthoc_ca'] is not None:
            lines += ['  POST-HOC: CA% (Tukey HSD on animal means)',
                      '  ' + '-' * 60,
                      str(r['posthoc_ca']),
                      _tukey_note, '']

        # Post-hoc: Sex (Tukey)
        if 'posthoc_sex' in r and r['posthoc_sex'] is not None:
            lines += ['  POST-HOC: Sex (Tukey HSD on animal means)',
                      '  ' + '-' * 60,
                      str(r['posthoc_sex']),
                      _tukey_note, '']

        # Cell post-hocs (interaction decomposition)
        if 'posthoc_week_cells' in r:
            for cell_label, ph in r['posthoc_week_cells'].items():
                wcol_a = [c for c in ph.columns if c.endswith('_A')][0]
                wcol_b = [c for c in ph.columns if c.endswith('_B')][0]
                lines += [f'  POST-HOC INTERACTION CELL: {cell_label} — '
                           f'Week pairwise (Bonferroni paired t-tests)',
                           '  ' + '-' * 60,
                           f'  {"Week A":>8} {"Week B":>8} {"t":>8} {"df":>6} '
                           f'{"p_raw":>12} {"p_bonf":>12} {"sig":>5}']
                for _, row in ph.iterrows():
                    sig_marker = '*' if row['significant'] else ''
                    lines.append(
                        f'  {str(int(row[wcol_a])):>8} {str(int(row[wcol_b])):>8} '
                        f'{row["t"]:>8.3f} {int(row["df"]):>6} '
                        f'{_fmt_p(row["p_raw"]):>12} {_fmt_p(row["p_bonferroni"]):>12} {sig_marker:>5}')
                lines.append('')

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    def _star(p): return '*' if (not np.isnan(p) and p < 0.05) else ' '
    lines += ['', '=' * 80,
              'SUMMARY TABLE — BH-FDR CORRECTED OMNIBUS RESULTS',
              '=' * 80,
              '  * = significant at α = 0.05 after BH-FDR correction (main effects);'
              ' raw p for interactions',
              '',
              f'  {"Measure":<32} {"Week(FDR)":>12} {"CA%(FDR)":>12} {"Sex(FDR)":>12}'
              f' {"Wk×CA%":>10} {"Wk×Sex":>10}  Decision']
    lines.append('  ' + '-' * 95)
    for m in measures:
        r = omnibus_results.get(m, {})
        if 'error' in r:
            lines.append(f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<32} ERROR')
            continue
        fw  = r.get('fdr_week_p', np.nan)
        fc  = r.get('fdr_ca_p',   np.nan)
        fs  = r.get('fdr_sex_p',  np.nan)
        wca = r.get('wk_ca_p',    np.nan)
        wsx = r.get('wk_sex_p',   np.nan)
        sigs = []
        if not np.isnan(fw)  and fw  < 0.05: sigs.append('Week')
        if not np.isnan(fc)  and fc  < 0.05: sigs.append('CA%')
        if not np.isnan(fs)  and fs  < 0.05: sigs.append('Sex')
        if not np.isnan(wca) and wca < 0.05: sigs.append('Wk×CA%')
        if not np.isnan(wsx) and wsx < 0.05: sigs.append('Wk×Sex')
        decision = ', '.join(sigs) if sigs else 'no significant effects'
        lines.append(
            f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<32} '
            f'{_fmt_p(fw):>12}{_star(fw)} '
            f'{_fmt_p(fc):>12}{_star(fc)} '
            f'{_fmt_p(fs):>12}{_star(fs)} '
            f'{_fmt_p(wca):>10}{_star(wca)} '
            f'{_fmt_p(wsx):>10}{_star(wsx)}  {decision}'
        )
    lines += ['', '=' * 80, 'END OF OMNIBUS REPORT', '=' * 80]
    report = '\n'.join(lines)

    if save_path is not None:
        try:
            save_path.write_text(report, encoding='utf-8')
            print(f'\n[OK] Omnibus report saved -> {save_path}')
        except Exception as e:
            print(f'[WARNING] Could not save report: {e}')

    return report


# =============================================================================
# STRATIFIED OMNIBUS ANALYSES
# =============================================================================

def perform_omnibus_lick_anova_sex_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    sex: str,
    measures: Optional[List[str]] = None,
    time_points: Optional[List[int]] = None,
) -> Dict:
    """Omnibus sex-stratified lick ANOVA: Week (within) × CA% (between) for one sex.

    Runs all 4 measures, applies BH-FDR across the 4 Week and 4 CA% omnibus F-tests.
    Post-hocs: Bonferroni paired t-tests for Week; Tukey for CA%.
    """
    if measures is None:
        measures = _OMNIBUS_MEASURES
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed.")
        return {}

    sex_label = 'Males' if sex == 'M' else 'Females'
    print(f"\n{'='*80}")
    print(f"SEX-STRATIFIED OMNIBUS LICK ANOVA ({sex_label.upper()})")
    print(f"{'='*80}")

    combined_df = combine_lick_cohorts(cohort_dfs)
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    combined_df = _add_sqrt_transforms(combined_df)
    combined_df = combined_df[combined_df['Sex'] == sex]

    all_results: Dict = {}
    week_ps: List[float] = []
    ca_ps: List[float] = []

    for measure in measures:
        if measure not in combined_df.columns:
            all_results[measure] = {'error': f'Column {measure} not found'}
            week_ps.append(np.nan); ca_ps.append(np.nan)
            continue

        adf = combined_df[['ID', 'Week', 'Sex', 'CA%', measure]].dropna().copy()
        if time_points is not None:
            adf = adf[adf['Week'].isin(time_points)]
        tp = sorted(adf['Week'].unique())

        n_weeks = len(tp)
        counts = adf.groupby('ID')['Week'].nunique()
        complete_ids = counts[counts == n_weeks].index
        adf = adf[adf['ID'].isin(complete_ids)].copy()
        n_dropped = combined_df['ID'].nunique() - len(complete_ids) if len(complete_ids) else 0

        try:
            tbl = pg.mixed_anova(data=adf, dv=measure, within='Week',
                                 subject='ID', between='CA%')

            def _get(src):
                row = tbl[tbl['Source'] == src]
                if row.empty: return np.nan, np.nan, np.nan
                r = row.iloc[0]
                return float(r.get('F', np.nan)), float(r.get('p-unc', np.nan)), float(r.get('np2', np.nan))

            wF, wP, wNP2 = _get('Week')
            cF, cP, cNP2 = _get('CA%')
            iF, iP, iNP2 = _get('Interaction')

            week_ps.append(wP); ca_ps.append(cP)
            all_results[measure] = {
                'measure': measure, 'sex': sex, 'analysis_df': adf, 'weeks': tp,
                'n_subjects': len(complete_ids), 'n_dropped': n_dropped,
                'anova_table': tbl,
                'week_F': wF, 'week_p': wP, 'week_np2': wNP2,
                'ca_F': cF,   'ca_p':   cP, 'ca_np2':   cNP2,
                'int_F': iF,  'int_p':  iP, 'int_np2':  iNP2,
            }
        except Exception as e:
            print(f"  [ERROR] ANOVA failed for {measure} ({sex_label}): {e}")
            all_results[measure] = {'error': str(e)}
            week_ps.append(np.nan); ca_ps.append(np.nan)

    # BH-FDR
    fdr_w = _bh_fdr([p if not np.isnan(p) else 1.0 for p in week_ps])
    fdr_c = _bh_fdr([p if not np.isnan(p) else 1.0 for p in ca_ps])

    for i, m in enumerate(measures):
        if 'error' not in all_results.get(m, {'error': ''}):
            all_results[m]['fdr_week_p'] = fdr_w[i]
            all_results[m]['fdr_ca_p']   = fdr_c[i]

    # Post-hocs
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r: continue
        adf = r['analysis_df']
        tp  = r['weeks']
        if (r.get('fdr_week_p', 1.0) < 0.05 or r.get('int_p', 1.0) < 0.05) and len(tp) > 1:
            r['posthoc_week'] = _bonferroni_paired_ttests(adf, m, 'Week', 'ID')
        if r.get('fdr_ca_p', 1.0) < 0.05:
            am = adf.groupby(['ID', 'CA%'])[m].mean().reset_index()
            r['posthoc_ca'] = _tukey_between(am, m, 'CA%')
        if r.get('int_p', 1.0) < 0.05:
            cell_ph = {}
            for ca_val in sorted(adf['CA%'].unique()):
                ph = _bonferroni_paired_ttests(adf[adf['CA%'] == ca_val], m, 'Week', 'ID')
                cell_ph[f'CA%={ca_val}'] = ph
            r['posthoc_week_cells'] = cell_ph

    all_results['_measures'] = measures
    all_results['_sex'] = sex
    return all_results


def perform_omnibus_lick_anova_ca_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    ca_percent: float,
    measures: Optional[List[str]] = None,
    time_points: Optional[List[int]] = None,
) -> Dict:
    """Omnibus CA%-stratified lick ANOVA: Week (within) × Sex (between) for one CA% level.

    Runs all 4 measures, applies BH-FDR across the 4 Week and 4 Sex omnibus F-tests.
    Post-hocs: Bonferroni paired t-tests for Week; Tukey for Sex.
    """
    if measures is None:
        measures = _OMNIBUS_MEASURES
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed.")
        return {}

    print(f"\n{'='*80}")
    print(f"CA%-STRATIFIED OMNIBUS LICK ANOVA ({ca_percent}% CA)")
    print(f"{'='*80}")

    combined_df = combine_lick_cohorts(cohort_dfs)
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    combined_df = _add_sqrt_transforms(combined_df)
    combined_df = combined_df[combined_df['CA%'] == ca_percent]

    all_results: Dict = {}
    week_ps: List[float] = []
    sex_ps: List[float] = []

    for measure in measures:
        if measure not in combined_df.columns:
            all_results[measure] = {'error': f'Column {measure} not found'}
            week_ps.append(np.nan); sex_ps.append(np.nan)
            continue

        adf = combined_df[['ID', 'Week', 'Sex', 'CA%', measure]].dropna().copy()
        if time_points is not None:
            adf = adf[adf['Week'].isin(time_points)]
        tp = sorted(adf['Week'].unique())

        n_weeks = len(tp)
        counts = adf.groupby('ID')['Week'].nunique()
        complete_ids = counts[counts == n_weeks].index
        adf = adf[adf['ID'].isin(complete_ids)].copy()
        n_dropped = combined_df['ID'].nunique() - len(complete_ids) if len(complete_ids) else 0

        try:
            tbl = pg.mixed_anova(data=adf, dv=measure, within='Week',
                                 subject='ID', between='Sex')

            def _get(src):
                row = tbl[tbl['Source'] == src]
                if row.empty: return np.nan, np.nan, np.nan
                r = row.iloc[0]
                return float(r.get('F', np.nan)), float(r.get('p-unc', np.nan)), float(r.get('np2', np.nan))

            wF, wP, wNP2 = _get('Week')
            sF, sP, sNP2 = _get('Sex')
            iF, iP, iNP2 = _get('Interaction')

            week_ps.append(wP); sex_ps.append(sP)
            all_results[measure] = {
                'measure': measure, 'ca_percent': ca_percent, 'analysis_df': adf, 'weeks': tp,
                'n_subjects': len(complete_ids), 'n_dropped': n_dropped,
                'anova_table': tbl,
                'week_F': wF, 'week_p': wP, 'week_np2': wNP2,
                'sex_F': sF,  'sex_p':  sP, 'sex_np2':  sNP2,
                'int_F': iF,  'int_p':  iP, 'int_np2':  iNP2,
            }
        except Exception as e:
            print(f"  [ERROR] ANOVA failed for {measure} ({ca_percent}% CA): {e}")
            all_results[measure] = {'error': str(e)}
            week_ps.append(np.nan); sex_ps.append(np.nan)

    # BH-FDR
    fdr_w = _bh_fdr([p if not np.isnan(p) else 1.0 for p in week_ps])
    fdr_s = _bh_fdr([p if not np.isnan(p) else 1.0 for p in sex_ps])

    for i, m in enumerate(measures):
        if 'error' not in all_results.get(m, {'error': ''}):
            all_results[m]['fdr_week_p'] = fdr_w[i]
            all_results[m]['fdr_sex_p']  = fdr_s[i]

    # Post-hocs
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r: continue
        adf = r['analysis_df']
        tp  = r['weeks']
        if (r.get('fdr_week_p', 1.0) < 0.05 or r.get('int_p', 1.0) < 0.05) and len(tp) > 1:
            r['posthoc_week'] = _bonferroni_paired_ttests(adf, m, 'Week', 'ID')
        if r.get('fdr_sex_p', 1.0) < 0.05:
            am = adf.groupby(['ID', 'Sex'])[m].mean().reset_index()
            r['posthoc_sex'] = _tukey_between(am, m, 'Sex')
        if r.get('int_p', 1.0) < 0.05:
            cell_ph = {}
            for sex_val in sorted(adf['Sex'].unique()):
                ph = _bonferroni_paired_ttests(adf[adf['Sex'] == sex_val], m, 'Week', 'ID')
                cell_ph[f'Sex={sex_val}'] = ph
            r['posthoc_week_cells'] = cell_ph

    all_results['_measures'] = measures
    all_results['_ca_percent'] = ca_percent
    return all_results


def perform_omnibus_lick_anova_2way(
    cohort_dfs: Dict[str, pd.DataFrame],
    measures: Optional[List[str]] = None,
    time_points: Optional[List[int]] = None,
) -> Dict:
    """Omnibus 2-way mixed ANOVA across lick measures: CA% (between) × Week (within).

    Uses all subjects regardless of sex. BH-FDR is applied separately across the
    4 Week omnibus F-tests and across the 4 CA% omnibus F-tests.

    Post-hocs:
      - Significant CA% main effect  → Tukey HSD on animal-level means
      - Significant Week main effect → Bonferroni all-pairwise paired t-tests
      - Significant Week × CA%       → Bonferroni pairwise paired t-tests within
                                        each CA% cell
    """
    if measures is None:
        measures = _OMNIBUS_MEASURES
    if not HAS_PINGOUIN:
        print("[ERROR] pingouin not installed.")
        return {}

    print(f"\n{'='*80}")
    print("2-WAY OMNIBUS LICK ANOVA — CA% × WEEK (ALL SUBJECTS)")
    print(f"{'='*80}")

    combined_df = combine_lick_cohorts(cohort_dfs)
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    combined_df = _add_sqrt_transforms(combined_df)

    all_results: Dict = {}
    week_ps: List[float] = []
    ca_ps:   List[float] = []

    for measure in measures:
        print(f"\n{'─'*60}")
        print(f"  ANOVA: {_OMNIBUS_MEASURE_LABELS.get(measure, measure)}")
        print(f"{'─'*60}")

        if measure not in combined_df.columns:
            print(f"  [WARNING] Column '{measure}' not found — skipping.")
            all_results[measure] = {'error': f'Column {measure} not found'}
            week_ps.append(np.nan); ca_ps.append(np.nan)
            continue

        adf = combined_df[['ID', 'Week', 'CA%', measure]].dropna().copy()
        if time_points is not None:
            adf = adf[adf['Week'].isin(time_points)]
        tp = sorted(adf['Week'].unique())

        n_weeks = len(tp)
        counts = adf.groupby('ID')['Week'].nunique()
        complete_ids = counts[counts == n_weeks].index
        n_dropped = adf['ID'].nunique() - len(complete_ids)
        adf = adf[adf['ID'].isin(complete_ids)].copy()

        if n_dropped > 0:
            print(f"  [INFO] {n_dropped} animal(s) excluded (incomplete weeks for {measure})")

        try:
            if adf['CA%'].nunique() < 2 or adf['ID'].nunique() < 4:
                raise ValueError("Insufficient subjects for mixed ANOVA")

            tbl = pg.mixed_anova(data=adf, dv=measure, within='Week',
                                 subject='ID', between='CA%')

            def _get(src):
                row = tbl[tbl['Source'] == src]
                if row.empty: return np.nan, np.nan, np.nan
                rv = row.iloc[0]
                return float(rv.get('F', np.nan)), float(rv.get('p-unc', np.nan)), float(rv.get('np2', np.nan))

            wF, wP, wNP2 = _get('Week')
            cF, cP, cNP2 = _get('CA%')
            iF, iP, iNP2 = _get('Interaction')

            week_ps.append(wP); ca_ps.append(cP)
            all_results[measure] = {
                'measure':    measure,
                'analysis_df': adf,
                'weeks':       tp,
                'n_subjects':  adf['ID'].nunique(),
                'n_dropped':   n_dropped,
                'anova_table': tbl,
                'week_F':  wF, 'week_p':  wP, 'week_np2':  wNP2,
                'ca_F':    cF, 'ca_p':    cP, 'ca_np2':    cNP2,
                'int_F':   iF, 'int_p':   iP, 'int_np2':   iNP2,
            }
        except Exception as e:
            print(f"  [ERROR] ANOVA failed for {measure}: {e}")
            import traceback; traceback.print_exc()
            all_results[measure] = {'error': str(e)}
            week_ps.append(np.nan); ca_ps.append(np.nan)

    # BH-FDR correction
    fdr_w = _bh_fdr([p if not np.isnan(p) else 1.0 for p in week_ps])
    fdr_c = _bh_fdr([p if not np.isnan(p) else 1.0 for p in ca_ps])
    for i, m in enumerate(measures):
        if 'error' not in all_results.get(m, {'error': ''}):
            all_results[m]['fdr_week_p'] = fdr_w[i]
            all_results[m]['fdr_ca_p']   = fdr_c[i]

    # Post-hoc tests
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r: continue
        adf = r['analysis_df']
        tp  = r['weeks']

        if (r.get('fdr_week_p', 1.0) < 0.05 or r.get('int_p', 1.0) < 0.05) and len(tp) > 1:
            print(f"  Post-hoc (Week, Bonferroni): {_OMNIBUS_MEASURE_LABELS.get(m, m)}")
            r['posthoc_week'] = _bonferroni_paired_ttests(adf, m, 'Week', 'ID')

        if r.get('fdr_ca_p', 1.0) < 0.05:
            print(f"  Post-hoc (CA%, Tukey): {_OMNIBUS_MEASURE_LABELS.get(m, m)}")
            am = adf.groupby(['ID', 'CA%'])[m].mean().reset_index()
            r['posthoc_ca'] = _tukey_between(am, m, 'CA%')

        if r.get('int_p', 1.0) < 0.05:
            cell_ph = {}
            for ca_val in sorted(adf['CA%'].unique()):
                ph = _bonferroni_paired_ttests(adf[adf['CA%'] == ca_val], m, 'Week', 'ID')
                cell_ph[f'CA%={ca_val}'] = ph
            r['posthoc_week_cells'] = cell_ph

    all_results['_measures'] = measures
    return all_results


def _format_stratified_omnibus_report(
    all_results: Dict,
    header: str,
    between_factor: str,
    fdr_between_key: str,
    posthoc_between_key: str,
    save_path: Optional[Path] = None,
) -> str:
    """Shared formatter for sex-stratified, CA%-stratified, and 2-way omnibus reports."""
    measures = all_results.get('_measures', _OMNIBUS_MEASURES)

    def _sig(p):
        if np.isnan(p): return 'n/a'
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    def _fmt_p(p):
        if np.isnan(p): return 'n/a'
        if p < 0.0001:  return '< 0.0001'
        return f'{p:.4f}'

    def _dec(p, fdr_p=None):
        chk = fdr_p if (fdr_p is not None and not np.isnan(fdr_p)) else p
        if np.isnan(chk): return 'n/a'
        return 'SIGNIFICANT' if chk < 0.05 else 'NOT SIGNIFICANT'

    def _ef(np2):
        if np.isnan(np2): return ''
        return f', ηp²={np2:.3f}'

    def _df_from_tbl(tbl, src):
        if tbl is None: return '?', '?'
        row = tbl[tbl['Source'] == src]
        if row.empty: return '?', '?'
        rv = row.iloc[0]
        return int(rv.get('DF1', 0)), int(rv.get('DF2', 0))

    between_raw_key = 'ca_p'   if between_factor == 'CA%' else 'sex_p'
    between_F_key   = 'ca_F'   if between_factor == 'CA%' else 'sex_F'
    between_np2_key = 'ca_np2' if between_factor == 'CA%' else 'sex_np2'
    between_src     = 'CA%'    if between_factor == 'CA%' else 'Sex'

    lines = ['=' * 80, header, '=' * 80,
             f'Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'Design    : {between_factor} (between) × Week (within)',
             f'Post-hocs : Tukey HSD for {between_factor}; Bonferroni paired t-tests for Week',
             f'Correction: BH-FDR across {len(measures)} measures within each factor family',
             '']

    # FDR summary table
    lines += ['=' * 80, 'BH-FDR CORRECTED OMNIBUS P-VALUES', '=' * 80,
              f'{"Measure":<32} {"Week(raw)":>12} {"Week(FDR)":>12} '
              f'{between_factor+"(raw)":>14} {between_factor+"(FDR)":>14}']
    lines.append('-' * 88)
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r:
            lines.append(f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<30} ERROR')
            continue
        lines.append(
            f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<30} '
            f'{_fmt_p(r.get("week_p",np.nan)):>12} '
            f'{_fmt_p(r.get("fdr_week_p",np.nan)):>12} '
            f'{_fmt_p(r.get(between_raw_key,np.nan)):>14} '
            f'{_fmt_p(r.get(fdr_between_key,np.nan)):>14} ')
    lines.append('')

    # Per-measure sections
    for m in measures:
        r = all_results.get(m, {})
        label = _OMNIBUS_MEASURE_LABELS.get(m, m)
        divider = '━' * 80
        lines += ['', divider, f'  MEASURE: {label}', divider]

        if 'error' in r:
            lines.append(f'  ERROR: {r["error"]}')
            continue

        lines += [f'  N subjects: {r.get("n_subjects","?")}',
                  f'  Animals dropped: {r.get("n_dropped",0)}',
                  f'  Weeks: {r.get("weeks","?")}', '']

        # Descriptive stats — per group, collapsed across weeks
        adf = r.get('analysis_df')
        grp_col = 'CA%' if between_factor == 'CA%' else 'Sex'
        if adf is not None:
            lines += [f'  DESCRIPTIVE STATISTICS (by {grp_col}, collapsed across weeks)',
                      '  ' + '-' * 60]
            for gv, grp in adf.groupby(grp_col)[m]:
                n    = len(grp)
                mn   = grp.mean()
                sd   = grp.std(ddof=1) if n > 1 else np.nan
                se   = sd / np.sqrt(n) if n > 1 else np.nan
                ci_lo = mn - 1.96 * se if not np.isnan(se) else np.nan
                ci_hi = mn + 1.96 * se if not np.isnan(se) else np.nan
                ci_str = f'[{ci_lo:.2f}, {ci_hi:.2f}]' if not np.isnan(ci_lo) else 'n/a'
                lines.append(f'    {grp_col}={gv}: n={n}, M={mn:.2f}, SD={sd:.2f}, '
                              f'SEM={se:.2f}, 95% CI {ci_str}')
            lines.append('')

        # ANOVA table from pingouin
        tbl = r.get('anova_table')
        if tbl is not None:
            lines += [f'  ANOVA TABLE ({between_factor} × Week mixed ANOVA)',
                      '  ' + '-' * 60]
            disp_cols = [c for c in ['Source', 'DF1', 'DF2', 'F', 'p-unc',
                                      'p-GG-corr', 'np2', 'eps'] if c in tbl.columns]
            for ln in tbl[disp_cols].to_string(index=False).split('\n'):
                lines.append('  ' + ln)
            lines.append('')

        # Interpretation block
        fdr_w = r.get('fdr_week_p', np.nan)
        fdr_b = r.get(fdr_between_key, np.nan)
        int_p = r.get('int_p', np.nan)
        wk_df1, wk_df2 = _df_from_tbl(tbl, 'Week')
        b_df1,  b_df2  = _df_from_tbl(tbl, between_src)
        in_df1, in_df2 = _df_from_tbl(tbl, 'Interaction')

        # Check whether GG correction was applied to Week (within-subjects)
        _wk_gg_p = np.nan
        if tbl is not None and 'p-GG-corr' in tbl.columns:
            _wk_row = tbl[tbl['Source'] == 'Week']
            if not _wk_row.empty:
                _wk_gg_p = float(_wk_row.iloc[0].get('p-GG-corr', np.nan))
        _wk_p_show = _wk_gg_p if not np.isnan(_wk_gg_p) else r.get('week_p', np.nan)
        _wk_p_lbl  = 'p (GG-corr)' if not np.isnan(_wk_gg_p) else 'p'

        lines += ['  INTERPRETATION', '  ' + '-' * 60]
        lines.append(
            f'  1. Week: {_dec(r.get("week_p", np.nan), fdr_w)}\n'
            f'     F({wk_df1}, {wk_df2}) = {r.get("week_F", np.nan):.2f}, '
            f'{_wk_p_lbl} = {_fmt_p(_wk_p_show)}, '
            f'p_FDR = {_fmt_p(fdr_w)} {_sig(fdr_w)}'
            f'{_ef(r.get("week_np2", np.nan))}'
        )
        lines.append(
            f'  2. {between_factor}: {_dec(r.get(between_raw_key, np.nan), fdr_b)}\n'
            f'     F({b_df1}, {b_df2}) = {r.get(between_F_key, np.nan):.2f}, '
            f'p = {_fmt_p(r.get(between_raw_key, np.nan))}, '
            f'p_FDR = {_fmt_p(fdr_b)} {_sig(fdr_b)}'
            f'{_ef(r.get(between_np2_key, np.nan))}'
        )
        lines.append(
            f'  3. Week × {between_factor}: '
            f'{"SIGNIFICANT" if (not np.isnan(int_p) and int_p < 0.05) else "NOT SIGNIFICANT"}\n'
            f'     F({in_df1}, {in_df2}) = {r.get("int_F", np.nan):.2f}, '
            f'p = {_fmt_p(int_p)} {_sig(int_p)}'
            f'{_ef(r.get("int_np2", np.nan))}'
        )
        lines.append('')

        # Post-hoc: Week
        if 'posthoc_week' in r:
            ph = r['posthoc_week']
            wcol_a = [c for c in ph.columns if c.endswith('_A')][0]
            wcol_b = [c for c in ph.columns if c.endswith('_B')][0]
            lines += ['  POST-HOC: Week (Bonferroni paired t-tests)', '  ' + '-' * 60,
                      f'  {"Week A":>8} {"Week B":>8} {"t":>8} {"df":>6} '
                      f'{"p_raw":>12} {"p_bonf":>12} {"sig":>5}']
            for _, row in ph.iterrows():
                sm = '*' if row['significant'] else ''
                lines.append(f'  {str(int(row[wcol_a])):>8} {str(int(row[wcol_b])):>8} '
                              f'{row["t"]:>8.3f} {int(row["df"]):>6} '
                              f'{_fmt_p(row["p_raw"]):>12} {_fmt_p(row["p_bonferroni"]):>12} {sm:>5}')
            lines.append('')

        _tukey_note = '  (reject=True means the groups differ significantly at FWER=0.05)'
        if posthoc_between_key in r and r[posthoc_between_key] is not None:
            lines += [f'  POST-HOC: {between_factor} (Tukey HSD on animal means)',
                      '  ' + '-' * 60, str(r[posthoc_between_key]),
                      _tukey_note, '']

        if 'posthoc_week_cells' in r:
            for cell_label, ph in r['posthoc_week_cells'].items():
                wcol_a = [c for c in ph.columns if c.endswith('_A')][0]
                wcol_b = [c for c in ph.columns if c.endswith('_B')][0]
                lines += [f'  POST-HOC INTERACTION CELL: {cell_label} (Bonferroni paired t-tests)',
                           '  ' + '-' * 60,
                           f'  {"Week A":>8} {"Week B":>8} {"t":>8} {"df":>6} '
                           f'{"p_raw":>12} {"p_bonf":>12} {"sig":>5}']
                for _, row in ph.iterrows():
                    sm = '*' if row['significant'] else ''
                    lines.append(f'  {str(int(row[wcol_a])):>8} {str(int(row[wcol_b])):>8} '
                                  f'{row["t"]:>8.3f} {int(row["df"]):>6} '
                                  f'{_fmt_p(row["p_raw"]):>12} {_fmt_p(row["p_bonferroni"]):>12} {sm:>5}')
                lines.append('')

    # Summary table
    def _star(p): return '*' if (not np.isnan(p) and p < 0.05) else ' '
    lines += ['', '=' * 80,
              f'SUMMARY TABLE — ALL KEY P-VALUES (α = 0.05)',
              '=' * 80,
              f'  * = significant after BH-FDR correction (between-subjects); raw p for interaction',
              '',
              f'  {"Measure":<32} {"Week(FDR)":>12} {between_factor+"(FDR)":>14}'
              f' {"Wk×"+between_factor:>16}  Decision']
    lines.append('  ' + '-' * 82)
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r:
            lines.append(f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<32} ERROR')
            continue
        fw = r.get('fdr_week_p', np.nan)
        fb = r.get(fdr_between_key, np.nan)
        fi = r.get('int_p', np.nan)
        sigs = []
        if not np.isnan(fw) and fw < 0.05: sigs.append('Week')
        if not np.isnan(fb) and fb < 0.05: sigs.append(between_factor)
        if not np.isnan(fi) and fi < 0.05: sigs.append(f'Wk×{between_factor}')
        decision = ', '.join(sigs) + ' significant' if sigs else 'no significant effects'
        lines.append(
            f'  {_OMNIBUS_MEASURE_LABELS.get(m,m):<32} '
            f'{_fmt_p(fw):>12}{_star(fw)} '
            f'{_fmt_p(fb):>14}{_star(fb)} '
            f'{_fmt_p(fi):>16}{_star(fi)}  {decision}'
        )
    lines += ['', '=' * 80, 'END OF REPORT', '=' * 80]
    report = '\n'.join(lines)

    if save_path is not None:
        try:
            save_path.write_text(report, encoding='utf-8')
            print(f'[OK] Report saved -> {save_path}')
        except Exception as e:
            print(f'[WARNING] Could not save: {e}')

    return report


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_fecal_qq(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Q-Q plots for raw and √-transformed Fecal_Count.

    Produces two multi-panel figures:
      1. Raw Fecal_Count — one panel per Week × Cohort (CA%) cell
      2. √(Fecal_Count)  — same layout

    Each panel shows the Q-Q scatter against the theoretical normal quantiles
    plus the 45-degree reference line.

    Parameters
    ----------
    cohort_dfs : dict of label → per-animal weekly DataFrame
    save_dir   : directory to save SVGs (created if absent); None = don't save
    show       : whether to call plt.show()
    """
    if not HAS_MATPLOTLIB:
        print('[WARNING] matplotlib not available — cannot create Q-Q plots.')
        return

    combined = combine_lick_cohorts(cohort_dfs)
    if 'Fecal_Count' not in combined.columns:
        print('[WARNING] Fecal_Count column not present — skipping Q-Q plots.')
        return
    combined = _add_sqrt_transforms(combined)
    if 'Week' not in combined.columns:
        combined = add_week_column(combined)

    ca_levels = sorted(combined['CA%'].dropna().unique())
    weeks     = sorted(combined['Week'].dropna().unique())
    n_cols    = len(ca_levels)
    n_rows    = len(weeks)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for transform_label, col in [('Raw Fecal_Count', 'Fecal_Count'),
                                  ('√(Fecal_Count)',  'Fecal_Count_Sqrt')]:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f'Q-Q Plot — {transform_label}  (normal reference)',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for row_idx, wk in enumerate(weeks):
            for col_idx, ca_val in enumerate(ca_levels):
                ax = axes[row_idx][col_idx]
                sub = combined[
                    (combined['Week'] == wk) & (combined['CA%'] == ca_val)
                ][col].dropna().values

                ax.set_title(
                    f'CA% = {ca_val},  Week {int(wk) + 1}  (n={len(sub)})',
                    fontsize=9,
                )
                ax.set_xlabel('Theoretical quantiles', fontsize=8)
                ax.set_ylabel('Sample quantiles', fontsize=8)
                ax.tick_params(labelsize=7)

                if len(sub) < 3:
                    ax.text(0.5, 0.5, 'n < 3\ninsufficient data',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=8, color='gray')
                    continue

                # Compute Q-Q coordinates via scipy.stats.probplot
                (osm, osr), (slope, intercept, _r) = stats.probplot(sub, dist='norm')
                ax.scatter(osm, osr, s=20, color='steelblue', alpha=0.8, zorder=3)

                # Reference line fitted to 1st–3rd quartile range
                x_line = np.array([osm[0], osm[-1]])
                ax.plot(x_line, slope * x_line + intercept,
                        color='crimson', linewidth=1.2, zorder=2)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        fig.tight_layout()

        if save_dir is not None:
            fname = f"fecal_qq_{'raw' if col == 'Fecal_Count' else 'sqrt'}.svg"
            fig.savefig(Path(save_dir) / fname, format='svg', dpi=200, bbox_inches='tight')
            print(f'[OK] Q-Q plot saved -> {Path(save_dir) / fname}')

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_fecal_counts_by_week(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Line plot of mean (±SEM) Fecal_Count across weeks, one line per cohort (CA%).

    Matches the visual style of plot_lick_measure_by_cohort.

    Parameters
    ----------
    cohort_dfs : dict of label → per-animal weekly DataFrame
    save_path  : optional SVG save path
    show       : whether to call plt.show()
    """
    if not HAS_MATPLOTLIB:
        print('[WARNING] matplotlib not available — cannot create fecal count plot.')
        return None

    combined = combine_lick_cohorts(cohort_dfs)
    if 'Fecal_Count' not in combined.columns:
        print('[WARNING] Fecal_Count column not present — skipping fecal count plot.')
        return None
    if 'Week' not in combined.columns:
        combined = add_week_column(combined)

    _FL_COLORS = {
        0.0: {'line': 'steelblue',  'face': 'lightblue',  'edge': 'steelblue'},
        2.0: {'line': 'darkorange', 'face': 'moccasin',   'edge': 'darkorange'},
    }
    _DEFAULT_COLORS = [
        {'line': 'steelblue',  'face': 'lightblue',  'edge': 'steelblue'},
        {'line': 'darkorange', 'face': 'moccasin',   'edge': 'darkorange'},
        {'line': 'darkgreen',  'face': 'lightgreen', 'edge': 'darkgreen'},
        {'line': 'purple',     'face': 'plum',       'edge': 'purple'},
    ]

    ca_levels = sorted(combined['CA%'].dropna().unique())
    weeks     = sorted(combined['Week'].dropna().unique())

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, ca_val in enumerate(ca_levels):
        grp = combined[combined['CA%'] == ca_val]
        wk_stats = (
            grp.groupby('Week')['Fecal_Count']
            .agg(['mean', 'sem', 'count'])
            .reset_index()
        )
        n_per_wk = int(wk_stats['count'].iloc[0]) if len(wk_stats) > 0 else 0
        c = _FL_COLORS.get(ca_val, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])
        ax.errorbar(
            wk_stats['Week'], wk_stats['mean'],
            yerr=wk_stats['sem'],
            label=f'{ca_val:.0f}% CA (n={n_per_wk}/week)',
            marker='o', markersize=8, linewidth=2, capsize=5,
            color=c['line'],
            markerfacecolor=c['face'],
            markeredgecolor=c['edge'],
        )

    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel('Fecal Count (mean ±SEM)', fontsize=12, weight='bold')
    ax.set_title('Fecal Count Across Weeks by Cohort (mean ±SEM)',
                 fontsize=13, weight='bold')
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(int(w) + 1) for w in weeks])
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    fig.tight_layout()

    if save_path is not None:
        svg_path = Path(save_path).with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=200, bbox_inches='tight')
        print(f'[OK] Fecal count plot saved -> {svg_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_lick_measure_by_cohort(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total_Licks",
    group_by_sex: bool = False,
    use_std: bool = False,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot lick measure over time for each cohort (EXACT style from lick_nonramp.py).
    
    This replicates the exact plotting style from lick_nonramp.py but with multiple cohorts
    as separate lines on the same plot.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Measure to plot ('Total_Licks', 'Total_Bouts', 'Avg_ILI', 'Avg_Bout_Duration', etc.)
        group_by_sex: If True, plot separate lines for each cohort × sex combination
        use_std: If True, use standard deviation (like lick_nonramp.py). If False, use SEM.
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib not installed. Cannot create plots.")
        return None
    
    combined_df = combine_lick_cohorts(cohort_dfs)
    
    if 'Week' not in combined_df.columns:
        combined_df = add_week_column(combined_df)
    
    # Color scheme for cohorts (matching lick_nonramp.py style)
    colors = [
        {'line': 'steelblue', 'marker': 'lightblue', 'edge': 'steelblue'},
        {'line': 'darkgreen', 'marker': 'lightgreen', 'edge': 'darkgreen'},
        {'line': 'darkorange', 'marker': 'orange', 'edge': 'darkorange'},
        {'line': 'purple', 'marker': 'plum', 'edge': 'purple'}
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if group_by_sex:
        # Plot separate lines for each cohort × sex combination
        color_idx = 0
        for cohort_label in sorted(cohort_dfs.keys()):
            cohort_data = combined_df[combined_df['Cohort'] == cohort_label]
            
            for sex in sorted(cohort_data['Sex'].unique()):
                sex_data = cohort_data[cohort_data['Sex'] == sex]
                
                # Compute mean and std/sem per week
                if use_std:
                    weekly_stats = sex_data.groupby('Week')[measure].agg(['mean', 'std']).reset_index()
                    error_col = 'std'
                else:
                    weekly_stats = sex_data.groupby('Week')[measure].agg(['mean', 'sem']).reset_index()
                    error_col = 'sem'
                
                # Get color
                color_scheme = colors[color_idx % len(colors)]
                
                label = f"{cohort_label} - {sex}"
                ax.errorbar(weekly_stats['Week'], weekly_stats['mean'], 
                           yerr=weekly_stats[error_col], label=label,
                           marker='o', markersize=8, linewidth=2, capsize=5,
                           color=color_scheme['line'], 
                           markerfacecolor=color_scheme['marker'],
                           markeredgecolor=color_scheme['edge'])
                
                color_idx += 1
    else:
        # Plot one line per cohort (collapsed across sex) - EXACT lick_nonramp.py style
        color_idx = 0
        for cohort_label in sorted(cohort_dfs.keys()):
            cohort_data = combined_df[combined_df['Cohort'] == cohort_label]
            
            # Compute mean and std/sem per week
            if use_std:
                weekly_stats = cohort_data.groupby('Week')[measure].agg(['mean', 'std', 'count']).reset_index()
                error_col = 'std'
                error_label = '±SD'
            else:
                weekly_stats = cohort_data.groupby('Week')[measure].agg(['mean', 'sem', 'count']).reset_index()
                error_col = 'sem'
                error_label = '±SEM'
            
            # Get color
            color_scheme = colors[color_idx % len(colors)]
            
            # Create x positions (0-indexed weeks)
            x_pos = weekly_stats['Week'].values
            
            ax.errorbar(x_pos, weekly_stats['mean'], yerr=weekly_stats[error_col],
                       label=f"{cohort_label} (n={int(weekly_stats['count'].iloc[0])} per week)",
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       color=color_scheme['line'],
                       markerfacecolor=color_scheme['marker'],
                       markeredgecolor=color_scheme['edge'])
            
            color_idx += 1
    
    # Format axes (EXACT lick_nonramp.py style)
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    
    # Create nice label from measure name
    measure_label = measure.replace('_', ' ')
    if 'Total' in measure:
        measure_label = measure_label + ' per Animal'
    elif 'Avg' in measure:
        measure_label = measure_label
    
    ax.set_ylabel(measure_label, fontsize=12, weight='bold')
    
    error_type = '±SD' if use_std else '±SEM'
    ax.set_title(f'{measure_label} Across Weeks by Cohort ({error_type})', 
                fontsize=13, weight='bold')
    
    # Set x-ticks to show Week 1, 2, 3, etc.
    if len(combined_df['Week'].unique()) > 0:
        all_weeks = sorted(combined_df['Week'].unique())
        ax.set_xticks(all_weeks)
        ax.set_xticklabels([f"{int(w)+1}" for w in all_weeks])
    
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='in', which='both', length=5)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path is not None:
        save_svg = save_path.with_suffix('.svg')
        fig.savefig(save_svg, format='svg', dpi=200, bbox_inches='tight')
        print(f"[OK] Saved plot to {save_svg}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_omnibus_interaction(
    adf: pd.DataFrame,
    measure: str,
    interaction_factor: str,
    title_suffix: str = '',
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot a significant Week × factor interaction: mean \u00b1 SEM per group over weeks.

    Parameters
    ----------
    adf              : long-format DataFrame with [ID, Week, measure, interaction_factor]
    measure          : column to plot (e.g. 'Total_Licks')
    interaction_factor : 'CA%' or 'Sex'
    title_suffix     : appended to figure title (e.g. 'Males only')
    save_path        : optional; always saved as SVG
    show             : call plt.show() if True, otherwise close the figure
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib not installed. Cannot create plots.")
        return None

    measure_label = _OMNIBUS_MEASURE_LABELS.get(measure, measure.replace('_', ' '))
    groups = sorted(adf[interaction_factor].unique())

    if interaction_factor == 'CA%':
        palette = ['steelblue', 'darkorange', 'darkgreen', 'purple']
    else:  # Sex
        palette = ['steelblue', 'deeppink', 'darkgreen', 'darkorange']

    fig, ax = plt.subplots(figsize=(9, 6))
    weeks = sorted(adf['Week'].unique())

    for idx, grp_val in enumerate(groups):
        grp_df = adf[adf[interaction_factor] == grp_val]
        stats_df = grp_df.groupby('Week')[measure].agg(['mean', 'sem']).reset_index()
        color = palette[idx % len(palette)]
        lbl = f"{grp_val:.0f}% CA" if interaction_factor == 'CA%' else str(grp_val)
        ax.errorbar(stats_df['Week'], stats_df['mean'], yerr=stats_df['sem'],
                    label=lbl, marker='o', markersize=8, linewidth=2,
                    capsize=5, color=color,
                    markerfacecolor=color, markeredgecolor=color)

    factor_label = 'CA%' if interaction_factor == 'CA%' else 'Sex'
    title = f'{measure_label}: Week \u00d7 {factor_label} Interaction'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xlabel('Week', fontsize=12, weight='bold')
    ax.set_ylabel(f'{measure_label} (mean \u00b1 SEM)', fontsize=12, weight='bold')
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(int(w) + 1) for w in weeks])
    ax.set_ylim(bottom=0)
    ax.legend(title=factor_label, loc='best', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', length=5)
    fig.tight_layout()

    if save_path is not None:
        svg_path = Path(save_path).with_suffix('.svg')
        fig.savefig(svg_path, format='svg', dpi=200, bbox_inches='tight')
        print(f"[OK] Saved interaction plot -> {svg_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def _detect_lick_comparison_type(cohorts: Dict[str, pd.DataFrame]) -> str:
    """
    Inspect loaded cohort labels and return a comparison-type tag:
      '0v2'    -- 0% vs 2% nonramp only
      '0vramp' -- 0% nonramp vs ramp
      '2vramp' -- 2% nonramp vs ramp
      'all3'   -- all three cohorts
      'unknown'-- cannot determine
    """
    labels_lower = [lbl.lower() for lbl in cohorts.keys()]

    has_zero = any('0%' in l and 'ramp' not in l for l in labels_lower)
    has_two  = any('2%' in l and 'ramp' not in l for l in labels_lower)
    has_ramp = any('ramp' in l for l in labels_lower)

    n = len(cohorts)
    if n == 3 and has_zero and has_two and has_ramp:
        return 'all3'
    if n == 2 and has_zero and has_two and not has_ramp:
        return '0v2'
    if n == 2 and has_zero and has_ramp and not has_two:
        return '0vramp'
    if n == 2 and has_two and has_ramp and not has_zero:
        return '2vramp'
    return 'unknown'


def generate_test_registry_report(save_path=None) -> str:
    """Generate a formatted plain-text report documenting every statistical
    test used in across_cohort_lick.py: data/variables consumed, library
    source, and every parameter with its meaning."""

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
    lines = _h1("STATISTICAL TEST REGISTRY  —  across_cohort_lick.py")
    lines += [
        "",
        "  Script: Cross-cohort omnibus lick analysis  (0% CA vs 2% CA cohorts)",
        "  Within-factor: Week  |  Between-factors: CA% (0 vs 2), Sex  |  α = 0.05",
        "",
        f"  QUICK REFERENCE  {'·' * 57}",
        "",
        f"    {'#':<3}  {'Test':<44}  Library / Function",
        f"    {'─'*3}  {'─'*44}  {'─'*26}",
        "    1    3-Way Mixed ANOVA  (CA% × Week × Sex)          pingouin / pg.mixed_anova()",
        "    2    2-Way Mixed ANOVA  (CA% × Week, no Sex)         pingouin / pg.mixed_anova()",
        "    3    Tukey HSD post-hoc  (CA% pairwise)              statsmodels / pairwise_tukeyhsd()",
        "    4    BH-FDR correction  (across 5 omnibus measures)  internal _bh_fdr()",
        "    5    Pairwise within  (Week post-hoc)                pingouin / pg.pairwise_tests()",
        "    6    Pairwise between  (CA% or Sex post-hoc)         scipy / pingouin / ttest_ind()",
        "    7    Shapiro-Wilk normality  (fecal count)           scipy.stats / shapiro()",
        "    8    D'Agostino-Pearson normality  (n > 5000)        scipy.stats / normaltest()",
        "    9    Chi-square Poisson GoF  (fecal count)           scipy.stats / chi2.sf()",
        "    VIZ  Q-Q Plot  (normality visualisation)             scipy.stats / probplot()",
        "",
        "    Multiple comparisons:",
        "      BH-FDR  — _bh_fdr() across 5 omnibus measures (Test 4)",
        "      BH-FDR  — padjust=fdr_bh inside pg.pairwise_tests (Test 5)",
        "      Tukey HSD — statsmodels, α = 0.05 (Test 3)",
        "    Sphericity : Greenhouse-Geisser auto-correction  (all pingouin tests;",
        "                 triggered when Mauchly's p-spher < 0.05)",
        "=" * W,
    ]

    # ── TEST 1 ───────────────────────────────────────────────────────────── #
    lines += _h2("1", "3-Way Mixed ANOVA  —  CA% × Week × Sex")
    lines += _sub("PURPOSE")
    lines += [
        "    Primary omnibus test.  CA% and Sex are between-subjects factors; Week is",
        "    within-subjects.  Implemented as three separate 2-way mixed ANOVAs because",
        "    pingouin does not support two simultaneous between-subjects factors:",
        "      (a) CA% × Week  —  cohort main effect + time interaction",
        "      (b) Sex × Week  —  sex main effect + time interaction",
        "      (c) Group × Week  where Group = 'CA%_Sex'  —  captures CA% × Sex interaction",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.mixed_anova()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "Long-format DataFrame  (one row per ID × Week)",         "pd.DataFrame"),
        ("dv",      "Lick measure  e.g. 'Total_Licks', 'Fecal_Count_Sqrt'",  "pd.Series[float64]"),
        ("within",  "'Week'  — repeated-measures factor",                     "pd.Series[int]"),
        ("between", "'CA%' | 'Sex' | 'Group'  (one call per factor)",         "pd.Series[str/float]"),
        ("subject", "'ID'  — unique animal identifier",                       "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    correction    Not set  (pingouin default).  Mauchly's sphericity test runs",
        "                  automatically.  GG ε applied when p-spher < 0.05.",
        "                  Cite p-GG-corr, not p-unc, when sphericity is violated.",
        "",
        "    ss_type       Type III SS  (pingouin default; not set explicitly).",
        "",
    ]
    lines += _sub("OUTPUT  (key columns)")
    lines += _out([
        ("Source",    "Effect label: 'CA%', 'Sex', 'Group', 'Week', 'Week * CA%', …"),
        ("F",         "F-statistic"),
        ("p-unc",     "Unadjusted p  (use p-GG-corr when sphericity is violated)"),
        ("np2",       "Partial η²  —  small ≥ 0.01  |  medium ≥ 0.06  |  large ≥ 0.14"),
        ("p-GG-corr", "Greenhouse-Geisser corrected p  (within / interaction rows)"),
        ("eps",       "GG ε  (1.0 = no correction; lower → larger df reduction)"),
    ])
    lines += [
        "",
        "    BH-FDR correction via _bh_fdr() is applied ACROSS the 5 omnibus measures",
        "    separately for each effect  (Week p-values FDR-corrected, CA% p-values, etc.)",
        "    Measures: First_5min_Lick_Pct, Time_to_50pct_Licks, First_5min_Bout_Pct,",
        "              Avg_ILI, Fecal_Count_Sqrt",
        "    Threshold: BH-FDR q = 0.05  across measures  |  α = 0.05 within measure",
    ]

    # ── TEST 2 ───────────────────────────────────────────────────────────── #
    lines += _h2("2", "2-Way Mixed ANOVA  —  CA% × Week  (no Sex factor)")
    lines += _sub("PURPOSE")
    lines += [
        "    Simpler omnibus collapsing across Sex.  Maximises statistical power to detect",
        "    the CA% main effect and its interaction with Week.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.mixed_anova()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "Long-format DataFrame  (one row per ID × Week)",    "pd.DataFrame"),
        ("dv",      "Lick measure column",                               "pd.Series[float64]"),
        ("within",  "'Week'",                                            "pd.Series[int]"),
        ("between", "'CA%'  (0.0 or 2.0)",                              "pd.Series[float64]"),
        ("subject", "'ID'",                                              "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    Same structure as Test 1 but without Sex / Group factor.",
        "    BH-FDR (_bh_fdr) applied across 5 measures for Week, CA%, and Week×CA% p-values.",
        "    Threshold: BH-FDR q = 0.05  across measures  |  α = 0.05 within measure",
    ]

    # ── TEST 3 ───────────────────────────────────────────────────────────── #
    lines += _h2("3", "Tukey HSD  —  CA% pairwise post-hoc")
    lines += _sub("PURPOSE")
    lines += [
        "    Pairwise follow-up for a significant CA% main effect.  Compares all pairs of",
        "    CA% levels on the lick measure (per-animal means across all weeks).",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += [
        "    statsmodels.stats.multicomp.pairwise_tukeyhsd()",
        "    from statsmodels.stats.multicomp import pairwise_tukeyhsd",
        "",
    ]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("endog",  "Lick measure values for all subjects",  "np.ndarray[float64]"),
        ("groups", "CA% label per observation",             "np.ndarray[str/float]"),
    ], w1=8, w2=50, w3=20)
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    endog   — 1-D array of dependent-variable values  (all subjects combined)",
        "    groups  — parallel array of CA% group labels for each observation",
        "    alpha   — 0.05  (familywise error rate; controls simultaneous CI width)",
        "",
        "    Tukey HSD uses the Studentized range distribution (q) to compute a single",
        "    critical difference applying simultaneously to all pairwise contrasts at α.",
        "    Maintains FWER without requiring a fixed number of comparisons k.",
        "    Threshold: familywise α = 0.05",
    ]

    # ── TEST 4 ───────────────────────────────────────────────────────────── #
    lines += _h2("4", "Benjamini-Hochberg FDR  —  correction across omnibus measures")
    lines += _sub("PURPOSE")
    lines += [
        "    Controls the false discovery rate when testing 5 lick measures simultaneously",
        "    in the omnibus ANOVA.  Applied per effect (Week, CA%, Sex) across measures.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += [
        "    Internal helper: _bh_fdr(p_values)  defined in this script",
        "    Internally uses: scipy.stats.rankdata + manual BH formula",
        "",
    ]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("p_values", "One p-value per omnibus measure for the effect being corrected",
         "List[float]"),
    ])
    lines.append("")
    lines += _sub("ALGORITHM  &  PARAMETERS")
    lines += [
        "    1. Sort p-values ascending; assign ranks 1 … m  (m = number of measures).",
        "    2. BH threshold for rank i  =  (i / m) × q   where q = 0.05.",
        "    3. Largest rank i where p(i) ≤ threshold is significant; all smaller ranks too.",
        "    q = 0.05  (hard-coded at all _bh_fdr call sites)",
    ]

    # ── TEST 5 ───────────────────────────────────────────────────────────── #
    lines += _h2("5", "Within-Subjects Pairwise Tests  (Week post-hoc)")
    lines += _sub("PURPOSE")
    lines += [
        "    Follow-up for a significant Week main effect.  All pairwise Week comparisons",
        "    collapsed across between-subjects groups.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.pairwise_tests()     import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "Long-format DataFrame  (all subjects × weeks)",    "pd.DataFrame"),
        ("dv",      "Lick measure column",                              "pd.Series[float64]"),
        ("within",  "'Week'",                                           "pd.Series[int]"),
        ("subject", "'ID'",                                             "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    parametric   True  — paired Student's t-test",
        "    padjust      'fdr_bh'  — BH-FDR across all Week pairs for this measure",
        "    effsize      'hedges'  — Hedges' g  (bias-corrected Cohen's d)",
        "    Threshold: α = 0.05 applied to BH-FDR-corrected p  (padjust column)",
    ]

    # ── TEST 6 ───────────────────────────────────────────────────────────── #
    lines += _h2("6", "Between-Subjects Pairwise Tests  (CA% or Sex post-hoc)")
    lines += _sub("PURPOSE")
    lines += [
        "    Follow-up for a significant CA% or Sex main effect.  Per-animal lick means",
        "    are computed across all weeks first, then groups are compared.",
        "",
    ]
    lines += _sub("LIBRARY  (2 groups)")
    lines += ["    scipy.stats.ttest_ind()     from scipy import stats", ""]
    lines += _sub("INPUTS  (2-group case)")
    lines += _tbl([
        ("group1", "Per-animal means for group 1",  "pd.Series[float64]"),
        ("group2", "Per-animal means for group 2",  "pd.Series[float64]"),
    ], w1=8, w2=52, w3=20)
    lines.append("")
    lines += _sub("PARAMETERS  (2-group case)")
    lines += [
        "    equal_var   True  (default)  —  Student's independent-samples t-test.",
        "    Cohen's d   = (mean₁ − mean₂) / pooled_SD",
        "",
    ]
    lines += _sub("LIBRARY  (> 2 groups — fallback)")
    lines += [
        "    pingouin.pairwise_tests(data, dv, between='CA%' or 'Sex',",
        "                            parametric=True, padjust='fdr_bh', effsize='cohen')",
        "",
        "    Threshold: α = 0.05  (no correction for 2 groups  |  BH-FDR for ≥ 3 groups)",
    ]

    # ── TEST 7 ───────────────────────────────────────────────────────────── #
    lines += _h2("7", "Shapiro-Wilk Normality Test  (fecal count)")
    lines += _sub("PURPOSE")
    lines += [
        "    Assesses whether raw Fecal_Count and sqrt-transformed Fecal_Count_Sqrt",
        "    are approximately normally distributed within each subgroup  (Week, CA%,",
        "    Sex, CA%×Week), informing whether parametric ANOVA is appropriate.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    scipy.stats.shapiro()     from scipy import stats  (n ≤ 5000)", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("values", "Fecal_Count  or  Fecal_Count_Sqrt = sqrt(max(Fecal_Count, 0))",
         "pd.Series[float64]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS  &  OUTPUT")
    lines += [
        "    values  — 1-D array of observations  (no additional keyword params)",
        "    W       — Shapiro-Wilk statistic  (float, 0–1; closer to 1 = more normal)",
        "    p       — p-value for H0: data are from a normal distribution",
        "    Threshold: p < 0.05 → reject normality",
    ]

    # ── TEST 8 ───────────────────────────────────────────────────────────── #
    lines += _h2("8", "D'Agostino-Pearson Normality Test  (n > 5000 fallback)")
    lines += _sub("PURPOSE")
    lines += [
        "    Shapiro-Wilk is only reliable for n ≤ 5000.  D'Agostino-Pearson's omnibus",
        "    test  (skewness + kurtosis components)  substitutes automatically for n > 5000.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    scipy.stats.normaltest()     from scipy import stats", ""]
    lines += _sub("INPUTS  &  OUTPUT")
    lines += _tbl([
        ("values", "1-D array of observations  (same as Test 7)", "pd.Series[float64]"),
    ])
    lines += [
        "",
        "    axis        default 0  (operates on the 1-D input array)",
        "    statistic   combined χ²  (skewness² + kurtosis² components)",
        "    p           p-value for H0: data are from a normal distribution",
        "    Threshold: p < 0.05 → reject normality",
    ]

    # ── TEST 9 ───────────────────────────────────────────────────────────── #
    lines += _h2("9", "Chi-Square Goodness-of-Fit  —  Poisson distribution  (fecal count)")
    lines += _sub("PURPOSE")
    lines += [
        "    Tests whether the discrete distribution of Fecal_Count follows a Poisson",
        "    distribution with mean λ estimated from data.  Informs the choice of ANOVA",
        "    vs. Poisson regression for count outcomes.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += [
        "    scipy.stats.poisson.pmf(k, lam)    expected Poisson probabilities",
        "    scipy.stats.chi2.sf(chi2_stat, df) p-value  (survival function = 1 − CDF)",
        "    from scipy import stats",
        "",
    ]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("counts", "Fecal_Count values for the group being tested",  "pd.Series[int/float]"),
        ("lam",    "Poisson mean  =  sample mean of counts  (MLE)", "float"),
    ], w1=8, w2=50, w3=20)
    lines.append("")
    lines += _sub("ALGORITHM  &  PARAMETERS")
    lines += [
        "    1.  Estimate λ = mean(counts).",
        "    2.  Compute expected frequency for each integer k from 0 to max(counts):",
        "        E_k = n × Poisson.pmf(k, λ)",
        "    3.  Merge tail bins where E_k < 5  (Cochran's rule for χ² validity).",
        "    4.  χ² = Σ (O_k − E_k)² / E_k   where O_k = observed frequency of value k.",
        "    5.  df = (n_bins after merging) − 1 − 1",
        "        −1 for degrees of freedom of bins; −1 for the estimated λ parameter.",
        "    6.  p = chi2.sf(χ², df)",
        "",
        "    Threshold: p < 0.05 → data depart significantly from a Poisson distribution",
    ]

    # ── VISUALISATION ─────────────────────────────────────────────────────── #
    lines += [f"\n{'─' * W}", "  VISUALISATION  │  Q-Q Plot  (Quantile-Quantile)", f"{'─' * W}", ""]
    lines += _sub("PURPOSE")
    lines += [
        "    Graphical (not inferential) assessment of normality.  Grid of Q-Q plots for",
        "    Fecal_Count (raw) and Fecal_Count_Sqrt, panelled by Week × CA%.  Saved as SVG.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    scipy.stats.probplot()     from scipy import stats", ""]
    lines += _sub("INPUTS  &  PARAMETERS")
    lines += _tbl([
        ("values", "Fecal_Count  or  Fecal_Count_Sqrt", "pd.Series[float64]"),
    ])
    lines += [
        "",
        "    dist   'norm'  — compare against standard normal theoretical quantiles",
        "    fit    True    — fit a straight reference line through the Q-Q points",
        "    plot   matplotlib Axes  — draw directly onto the subplot axis",
        "",
        "    Interpretation: points near the diagonal → approximate normality;",
        "    S-curve → excess kurtosis;  concave / convex curve → skewness.",
    ]

    # ── SUMMARY ──────────────────────────────────────────────────────────── #
    lines += [
        "",
        f"\n{'─' * W}",
        "  CORRECTION METHODS SUMMARY",
        f"  {'·' * (W - 4)}",
        "",
        f"    {'Context':<38}  {'Method':<18}  Detail",
        f"    {'─'*38}  {'─'*18}  {'─'*18}",
        f"    {'Omnibus across 5 measures  (Test 4)':<38}  {'BH-FDR':<18}  q = 0.05  _bh_fdr()",
        f"    {'Within-measure pairwise  (Test 5)':<38}  {'BH-FDR':<18}  padjust=fdr_bh",
        f"    {'Between-subjects pairwise  (Test 3)':<38}  {'Tukey HSD':<18}  α = 0.05",
        f"    {'Sphericity  (all pingouin tests)':<38}  {'GG auto':<18}  Mauchly p < 0.05",
        "",
        f"{'─' * W}",
        "  OMNIBUS MEASURES  (5 frontloading / temporal metrics)",
        f"  {'·' * (W - 4)}",
        "",
        "      1.  First_5min_Lick_Pct   % session licks in first 5 min       float  0–100",
        "      2.  Time_to_50pct_Licks   min to accumulate 50% of licks        float",
        "      3.  First_5min_Bout_Pct   % session bouts in first 5 min        float  0–100",
        "      4.  Avg_ILI               mean inter-lick interval  (ms)         float",
        "      5.  Fecal_Count_Sqrt      sqrt(max(Fecal_Count, 0))              float",
        "          ↳  raw Fecal_Count  = integer count of fecal boli per session",
        "             sqrt transform applied for variance-stabilisation before ANOVA",
        "",
        "=" * W,
    ]

    report = "\n".join(lines)
    if save_path is not None:
        Path(save_path).write_text(report, encoding="utf-8")
        print(f"[OK] Test registry saved -> {save_path}")
    return report



def _run_lick_0v2_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for 0% vs 2% nonramp lick comparison.
    Reports are auto-saved with timestamps.
    """
    from datetime import datetime

    MEASURES = ["Total_Licks", "Total_Bouts", "Avg_ILI", "Avg_Bout_Duration", "Licks_Per_Bout"]
    combined_temp = combine_lick_cohorts(cohorts)
    available_measures = [m for m in MEASURES if m in combined_temp.columns]
    if not available_measures:
        available_measures = MEASURES  # fall back; ANOVA will handle missing

    print("\n" + "=" * 80)
    print("0% vs 2% NONRAMP — LICK ANALYSIS MENU")
    print("=" * 80)
    print("\nAll analyses use Week as the time axis (Week 1 = first measurement week).")
    print("Reports are automatically saved to the current directory.")
    print(f"\nAvailable measures : {available_measures}")
    print()
    print("  1. Full mixed ANOVA     -- CA%  x  Week  x  Sex (all lick measures)")
    print("  2. Sex-stratified       -- CA%  x  Week separately for Males and Females")
    print("  3. CA%-stratified       -- Week  x  Sex separately for 0% and 2%")
    print("  4. Lick plots           -- Mean lick measure by cohort over weeks")
    print("  5. Sex-split plots      -- Mean lick measure split by cohort x sex")
    print("  6. Frontloading report  -- Descriptive stats for % licks in first 5 min & time to 50%")
    print("  7. Omnibus ANOVA        -- 4 frontloading measures, BH-FDR corrected omnibus + post-hocs")
    print("  8. 2-Way Omnibus ANOVA  -- CA% x Week only (all subjects, no Sex factor)")
    print("  9. Frontloading plots   -- Line plots of % licks in first 5 min & time to 50% licks by cohort")
    print(" 10. Fecal normality      -- Shapiro-Wilk tests on fecal counts (raw & sqrt); Friedman recommendation")
    print(" 11. Statistical registry -- Print/save documentation of every test: variables, library, parameters")
    print(" 12. Run all (1-11)")
    print()

    user_input = input("Select option (1-12) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all = (user_input == '12')

    # ------------------------------------------------------------------ #
    # Option 1: Full mixed ANOVA (CA% x Week x Sex)
    # ------------------------------------------------------------------ #
    if user_input == '1' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Full mixed ANOVA — CA%  x  Week  x  Sex (all lick measures)")
        print("=" * 80)

        all_sections = []
        for i, measure in enumerate(available_measures):
            result = None
            try:
                result = perform_cross_cohort_lick_anova(cohorts, measure=measure)
            except Exception as e:
                print(f"  [WARNING] Full ANOVA failed for {measure}: {e}")

            section = generate_lick_cohort_report(
                mixed_results=result,
                cohort_dfs=cohorts if i == 0 else None,
            )
            all_sections.append(section)

        combined_report = "\n\n".join(all_sections)
        rpt_path = Path(f"0v2_lick_full_anova_{timestamp}.txt")
        rpt_path.write_text(combined_report, encoding='utf-8')
        print(f"\n[OK] Report saved -> {rpt_path}")

    # ------------------------------------------------------------------ #
    # Option 2: Sex-stratified
    # ------------------------------------------------------------------ #
    if user_input == '2' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Sex-stratified lick ANOVA — CA%  x  Week (Males and Females separately)")
        print("=" * 80)

        all_sections = []
        for i, measure in enumerate(available_measures):
            r_males = r_females = None
            try:
                r_males = perform_lick_anova_sex_stratified(cohorts, sex="M", measure=measure)
            except Exception as e:
                print(f"  [WARNING] Sex-stratified (Males) failed for {measure}: {e}")
            try:
                r_females = perform_lick_anova_sex_stratified(cohorts, sex="F", measure=measure)
            except Exception as e:
                print(f"  [WARNING] Sex-stratified (Females) failed for {measure}: {e}")

            section = generate_lick_cohort_report(
                results_males=r_males,
                results_females=r_females,
                cohort_dfs=cohorts if i == 0 else None,
            )
            all_sections.append(section)

        combined_report = "\n\n".join(all_sections)
        rpt_path = Path(f"0v2_lick_sex_stratified_{timestamp}.txt")
        rpt_path.write_text(combined_report, encoding='utf-8')
        print(f"\n[OK] Report saved -> {rpt_path}")

    # ------------------------------------------------------------------ #
    # Option 3: CA%-stratified
    # ------------------------------------------------------------------ #
    if user_input == '3' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: CA%-stratified lick ANOVA — Week  x  Sex (0% and 2% separately)")
        print("=" * 80)

        all_sections = []
        for i, measure in enumerate(available_measures):
            r_ca0 = r_ca2 = None
            try:
                r_ca0 = perform_lick_anova_ca_stratified(cohorts, ca_percent=0.0, measure=measure)
            except Exception as e:
                print(f"  [WARNING] CA%-stratified (0%) failed for {measure}: {e}")
            try:
                r_ca2 = perform_lick_anova_ca_stratified(cohorts, ca_percent=2.0, measure=measure)
            except Exception as e:
                print(f"  [WARNING] CA%-stratified (2%) failed for {measure}: {e}")

            section = generate_lick_cohort_report(
                results_ca0=r_ca0,
                results_ca2=r_ca2,
                cohort_dfs=cohorts if i == 0 else None,
            )
            all_sections.append(section)

        combined_report = "\n\n".join(all_sections)
        rpt_path = Path(f"0v2_lick_ca_stratified_{timestamp}.txt")
        rpt_path.write_text(combined_report, encoding='utf-8')
        print(f"\n[OK] Report saved -> {rpt_path}")

    # ------------------------------------------------------------------ #
    # Option 4: Lick plots (cohort lines)
    # ------------------------------------------------------------------ #
    if user_input == '4' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Lick measure plots by cohort (collapsed across sex)")
            print("=" * 80)

            plot_dir = Path(f"0v2_lick_plots_{timestamp}")
            plot_dir.mkdir(exist_ok=True)

            for measure in available_measures:
                try:
                    fig = plot_lick_measure_by_cohort(
                        cohorts,
                        measure=measure,
                        group_by_sex=False,
                        save_path=plot_dir / f"lick_{measure.lower()}_by_cohort.svg",
                        show=False,
                    )
                    if fig:
                        import matplotlib.pyplot as _plt
                        _plt.close(fig)
                except Exception as e:
                    print(f"  [WARNING] Plot for {measure} failed: {e}")

            print(f"\n[OK] Plots saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                import matplotlib.pyplot as _plt
                _plt.show()
            else:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close('all')
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Option 5: Sex-split plots
    # ------------------------------------------------------------------ #
    if user_input == '5' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Lick measure plots by cohort × sex")
            print("=" * 80)

            plot_dir = Path(f"0v2_lick_plots_{timestamp}")
            plot_dir.mkdir(exist_ok=True)

            for measure in available_measures:
                try:
                    fig = plot_lick_measure_by_cohort(
                        cohorts,
                        measure=measure,
                        group_by_sex=True,
                        save_path=plot_dir / f"lick_{measure.lower()}_by_cohort_sex.svg",
                        show=False,
                    )
                    if fig:
                        import matplotlib.pyplot as _plt
                        _plt.close(fig)
                except Exception as e:
                    print(f"  [WARNING] Sex-split plot for {measure} failed: {e}")

            print(f"\n[OK] Sex-split plots saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                import matplotlib.pyplot as _plt
                _plt.show()
            else:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close('all')
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Option 6: Frontloading descriptives report
    # ------------------------------------------------------------------ #
    if user_input == '6' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Frontloading descriptives — % licks in first 5 min & time to 50%")
        print("=" * 80)

        rpt_path = Path(f"0v2_lick_frontloading_descriptives_{timestamp}.txt")
        try:
            report_text = generate_lick_frontloading_descriptives_report(
                cohorts,
                save_path=rpt_path,
            )
            print(report_text)
        except Exception as e:
            print(f"  [WARNING] Frontloading descriptives report failed: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 7: Omnibus ANOVA (4 frontloading measures, BH-FDR corrected)
    # ------------------------------------------------------------------ #
    if user_input == '7' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Omnibus ANOVA — 4 measures, BH-FDR corrected, with post-hocs")
        print("=" * 80)

        combined_report_sections = []

        # Full omnibus (CA% × Sex × Week)
        try:
            omnibus_res = perform_omnibus_lick_anova(cohorts)
            section = generate_omnibus_lick_report(omnibus_res, cohort_dfs=cohorts, save_path=None)
            combined_report_sections.append(section)
            # Plot significant interactions
            if HAS_MATPLOTLIB:
                int_plot_dir = Path(f"0v2_lick_interaction_plots_{timestamp}")
                n_int_plots = 0
                for _m in omnibus_res.get('_measures', _OMNIBUS_MEASURES):
                    _r = omnibus_res.get(_m, {})
                    if 'error' in _r or 'analysis_df' not in _r:
                        continue
                    int_plot_dir.mkdir(exist_ok=True)
                    if _r.get('wk_ca_p', 1.0) < 0.05:
                        plot_omnibus_interaction(
                            _r['analysis_df'], _m, 'CA%',
                            save_path=int_plot_dir / f"omnibus3way_{_m}_week_x_ca.svg",
                            show=False)
                        n_int_plots += 1
                    if _r.get('wk_sex_p', 1.0) < 0.05:
                        plot_omnibus_interaction(
                            _r['analysis_df'], _m, 'Sex',
                            save_path=int_plot_dir / f"omnibus3way_{_m}_week_x_sex.svg",
                            show=False)
                        n_int_plots += 1
                if n_int_plots:
                    print(f"[OK] {n_int_plots} interaction plot(s) saved -> {int_plot_dir}")
                else:
                    print("  (No significant Week interactions to plot from 3-way omnibus)")
        except Exception as e:
            print(f"  [WARNING] Omnibus ANOVA failed: {e}")
            import traceback; traceback.print_exc()
            combined_report_sections.append(f"[ERROR] Omnibus ANOVA failed: {e}\n")

        # Sex-stratified omnibus
        for sex_val in ('M', 'F'):
            sex_label = 'males' if sex_val == 'M' else 'females'
            try:
                strat_res = perform_omnibus_lick_anova_sex_stratified(cohorts, sex=sex_val)
                header = f"SEX-STRATIFIED OMNIBUS LICK ANOVA ({'MALES' if sex_val=='M' else 'FEMALES'})"
                section = _format_stratified_omnibus_report(
                    strat_res, header=header,
                    between_factor='CA%', fdr_between_key='fdr_ca_p',
                    posthoc_between_key='posthoc_ca', save_path=None)
                combined_report_sections.append(section)
            except Exception as e:
                print(f"  [WARNING] Sex-stratified omnibus ({sex_label}) failed: {e}")
                import traceback; traceback.print_exc()
                combined_report_sections.append(f"[ERROR] Sex-stratified omnibus ({sex_label}) failed: {e}\n")

        # CA%-stratified omnibus
        combined_for_ca = combine_lick_cohorts(cohorts)
        ca_levels = sorted(combined_for_ca['CA%'].dropna().unique()) if 'CA%' in combined_for_ca.columns else []
        for ca_val in ca_levels:
            try:
                strat_res = perform_omnibus_lick_anova_ca_stratified(cohorts, ca_percent=ca_val)
                header = f"CA%-STRATIFIED OMNIBUS LICK ANOVA ({ca_val}% CA)"
                section = _format_stratified_omnibus_report(
                    strat_res, header=header,
                    between_factor='Sex', fdr_between_key='fdr_sex_p',
                    posthoc_between_key='posthoc_sex', save_path=None)
                combined_report_sections.append(section)
            except Exception as e:
                print(f"  [WARNING] CA%-stratified omnibus ({ca_val}%) failed: {e}")
                import traceback; traceback.print_exc()
                combined_report_sections.append(f"[ERROR] CA%-stratified omnibus ({ca_val}%) failed: {e}\n")

        # Fecal count Poisson GoF — appended to combined report
        try:
            gof_section = generate_fecal_poisson_gof_report(cohorts)
            combined_report_sections.append(gof_section)
        except Exception as e:
            print(f"  [WARNING] Fecal Poisson GoF failed: {e}")
            import traceback; traceback.print_exc()

        # Combine and save as one file
        divider = "\n\n" + "=" * 80 + "\n" + "=" * 80 + "\n\n"
        full_report = divider.join(combined_report_sections)
        combined_rpt_path = Path(f"0v2_lick_omnibus_combined_{timestamp}.txt")
        try:
            combined_rpt_path.write_text(full_report, encoding='utf-8')
            print(f"\n[OK] Combined omnibus report saved -> {combined_rpt_path}")
        except Exception as e:
            print(f"  [WARNING] Could not save combined report: {e}")
        print(full_report)

    # ------------------------------------------------------------------ #
    # Option 8: 2-way omnibus ANOVA (CA% × Week, no Sex factor)
    # ------------------------------------------------------------------ #
    if user_input == '8' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: 2-Way Omnibus ANOVA — CA% × Week (all subjects, BH-FDR corrected)")
        print("=" * 80)

        twoway_res = None
        try:
            twoway_res = perform_omnibus_lick_anova_2way(cohorts)
            header_2way = "2-WAY OMNIBUS LICK ANOVA — CA% × WEEK (ALL SUBJECTS)"
            twoway_section = _format_stratified_omnibus_report(
                twoway_res,
                header=header_2way,
                between_factor='CA%',
                fdr_between_key='fdr_ca_p',
                posthoc_between_key='posthoc_ca',
                save_path=None,
            )
            twoway_rpt_path = Path(f"0v2_lick_2way_omnibus_{timestamp}.txt")
            twoway_rpt_path.write_text(twoway_section, encoding='utf-8')
            print(f"\n[OK] 2-Way omnibus report saved -> {twoway_rpt_path}")
            print(twoway_section)
        except Exception as e:
            print(f"  [WARNING] 2-Way omnibus ANOVA/report failed: {e}")
            import traceback; traceback.print_exc()

        # Fecal count Poisson GoF (saved separately)
        try:
            gof_rpt_path = Path(f"0v2_lick_fecal_poisson_gof_{timestamp}.txt")
            gof_report = generate_fecal_poisson_gof_report(cohorts, save_path=gof_rpt_path)
            print(gof_report)
        except Exception as e:
            print(f"  [WARNING] Fecal Poisson GoF failed: {e}")
            import traceback; traceback.print_exc()

        # Plot significant Week × CA% interactions (separate try so report success is independent)
        if twoway_res is not None and HAS_MATPLOTLIB:
            try:
                int_plot_dir = Path(f"0v2_lick_2way_interaction_plots_{timestamp}")
                n_int_plots = 0
                for _m in twoway_res.get('_measures', _OMNIBUS_MEASURES):
                    _r = twoway_res.get(_m, {})
                    if 'error' in _r or 'analysis_df' not in _r:
                        print(f"  [DEBUG] Skipping {_m}: error={_r.get('error')}, has_adf={'analysis_df' in _r}")
                        continue
                    int_p_val = _r.get('int_p', 1.0)
                    print(f"  [DEBUG] {_m}: int_p={int_p_val:.4f}")
                    if int_p_val < 0.05:
                        int_plot_dir.mkdir(exist_ok=True)
                        plot_omnibus_interaction(
                            _r['analysis_df'], _m, 'CA%',
                            title_suffix='all subjects',
                            save_path=int_plot_dir / f"2way_{_m}_week_x_ca.svg",
                            show=False)
                        n_int_plots += 1
                if n_int_plots:
                    print(f"[OK] {n_int_plots} interaction plot(s) saved -> {int_plot_dir}")
                else:
                    print("  (No significant Week × CA% interactions to plot from 2-way omnibus)")
            except Exception as e:
                print(f"  [WARNING] Interaction plot failed: {e}")
                import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 9: Frontloading line plots (% First 5 min & Time to 50%)
    # ------------------------------------------------------------------ #
    if user_input == '9' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Frontloading line plots — % licks in first 5 min & time to 50%")
            print("=" * 80)

            combined_fl = combine_lick_cohorts(cohorts)
            if 'Week' not in combined_fl.columns:
                combined_fl = add_week_column(combined_fl)

            fl_plot_dir = Path(f"0v2_lick_frontloading_plots_{timestamp}")
            fl_plot_dir.mkdir(exist_ok=True)

            _FL_MEASURES = [
                ("First_5min_Lick_Pct",  "% Licks in First 5 min",   (0, 100)),
                ("Time_to_50pct_Licks",  "Time to 50% Licks (min)",  (0, None)),
            ]

            _FL_COLORS = {
                0.0: {'line': 'steelblue',  'face': 'lightblue',  'edge': 'steelblue'},
                2.0: {'line': 'darkorange', 'face': 'moccasin',   'edge': 'darkorange'},
            }
            _DEFAULT_COLORS = [
                {'line': 'steelblue',  'face': 'lightblue',  'edge': 'steelblue'},
                {'line': 'darkorange', 'face': 'moccasin',   'edge': 'darkorange'},
                {'line': 'darkgreen',  'face': 'lightgreen', 'edge': 'darkgreen'},
                {'line': 'purple',     'face': 'plum',       'edge': 'purple'},
            ]

            n_fl_plots = 0
            for col_name, y_label, (y_min, y_max) in _FL_MEASURES:
                if col_name not in combined_fl.columns:
                    print(f"  [WARNING] Column '{col_name}' not found — skipping plot.")
                    continue
                try:
                    fig_fl, ax_fl = plt.subplots(figsize=(9, 6))

                    ca_levels_fl = sorted(combined_fl['CA%'].dropna().unique())
                    weeks_fl = sorted(combined_fl['Week'].dropna().unique())

                    for idx, ca_val in enumerate(ca_levels_fl):
                        grp = combined_fl[combined_fl['CA%'] == ca_val]
                        wk_stats = (
                            grp.groupby('Week')[col_name]
                            .agg(['mean', 'sem', 'count'])
                            .reset_index()
                        )
                        n_per_wk = int(wk_stats['count'].iloc[0]) if len(wk_stats) > 0 else 0
                        c = _FL_COLORS.get(ca_val, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])
                        lbl = f"{ca_val:.0f}% CA (n={n_per_wk}/week)"
                        ax_fl.errorbar(
                            wk_stats['Week'], wk_stats['mean'],
                            yerr=wk_stats['sem'],
                            label=lbl, marker='o', markersize=8,
                            linewidth=2, capsize=5,
                            color=c['line'],
                            markerfacecolor=c['face'],
                            markeredgecolor=c['edge'],
                        )

                    ax_fl.set_xlabel('Week', fontsize=12, weight='bold')
                    ax_fl.set_ylabel(f'{y_label} (mean \u00b1 SEM)', fontsize=12, weight='bold')
                    ax_fl.set_title(
                        f'{y_label} Across Weeks by Cohort (mean \u00b1 SEM)',
                        fontsize=13, weight='bold'
                    )
                    ax_fl.set_xticks(weeks_fl)
                    ax_fl.set_xticklabels([str(int(w) + 1) for w in weeks_fl])
                    ax_fl.set_ylim(bottom=y_min if y_min is not None else ax_fl.get_ylim()[0])
                    if y_max is not None:
                        ax_fl.set_ylim(top=y_max)
                    ax_fl.legend(loc='best', fontsize=10)
                    ax_fl.spines['top'].set_visible(False)
                    ax_fl.spines['right'].set_visible(False)
                    ax_fl.tick_params(direction='in', which='both', length=5)
                    fig_fl.tight_layout()

                    svg_path = fl_plot_dir / f"frontloading_{col_name}.svg"
                    fig_fl.savefig(svg_path, format='svg', dpi=200, bbox_inches='tight')
                    plt.close(fig_fl)
                    print(f"[OK] Saved -> {svg_path}")
                    n_fl_plots += 1
                except Exception as e:
                    print(f"  [WARNING] Plot for {col_name} failed: {e}")
                    import traceback; traceback.print_exc()

            if n_fl_plots:
                print(f"\n[OK] {n_fl_plots} frontloading plot(s) saved -> {fl_plot_dir}")

    # ------------------------------------------------------------------ #
    # Option 10: Fecal count normality tests + Friedman recommendation
    # ------------------------------------------------------------------ #
    if user_input == '10' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Fecal Count — Shapiro-Wilk normality tests (raw & √-transformed)")
        print("=" * 80)
        try:
            norm_rpt_path = Path(f"0v2_lick_fecal_normality_{timestamp}.txt")
            norm_report = generate_fecal_normality_report(cohorts, save_path=norm_rpt_path)
            print(norm_report)
        except Exception as e:
            print(f"  [WARNING] Fecal normality report failed: {e}")
            import traceback; traceback.print_exc()

        if HAS_MATPLOTLIB:
            try:
                qq_dir = Path(f"0v2_lick_fecal_qq_{timestamp}")
                plot_fecal_qq(cohorts, save_dir=qq_dir, show=False)
            except Exception as e:
                print(f"  [WARNING] Q-Q plot generation failed: {e}")
                import traceback; traceback.print_exc()
            try:
                fc_plot_path = Path(f"0v2_lick_fecal_counts_{timestamp}.svg")
                plot_fecal_counts_by_week(cohorts, save_path=fc_plot_path, show=False)
            except Exception as e:
                print(f"  [WARNING] Fecal count line plot failed: {e}")
                import traceback; traceback.print_exc()
        else:
            print("  [INFO] matplotlib not available — skipping Q-Q plots")

    # ------------------------------------------------------------------ #
    # Option 11: Statistical test registry
    # ------------------------------------------------------------------ #
    if user_input == '11' or run_all:
        print("\n" + "=" * 80)
        print("STATISTICAL TEST REGISTRY — across_cohort_lick.py")
        print("=" * 80)
        try:
            registry_path = Path(f"0v2_lick_test_registry_{timestamp}.txt")
            registry_report = generate_test_registry_report(save_path=registry_path)
            print(registry_report)
        except Exception as e:
            print(f"  [WARNING] Registry generation failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("0% vs 2% lick analysis complete.")
    print("=" * 80)


def _run_lick_unknown_menu(cohorts: Dict[str, pd.DataFrame], comparison: str) -> None:
    """Placeholder menu for comparison types beyond 0v2 (ramp comparisons)."""
    label_map = {
        '0vramp': '0% nonramp vs Ramp',
        '2vramp': '2% nonramp vs Ramp',
        'all3':   '0% nonramp vs 2% nonramp vs Ramp',
        'unknown': 'Unknown cohort combination',
    }
    cohort_labels = list(cohorts.keys())
    print("\n" + "=" * 80)
    print(f"COMPARISON TYPE: {label_map.get(comparison, comparison)}")
    print("=" * 80)
    print(f"\nLoaded cohorts: {cohort_labels}")
    print("\nAvailable functions for this comparison:")
    print("  combine_lick_cohorts(cohorts)          -- combine into one DataFrame")
    print("  perform_cross_cohort_lick_anova(cohorts, measure)")
    print("  perform_lick_anova_sex_stratified(cohorts, sex, measure)")
    print("  perform_lick_anova_ca_stratified(cohorts, ca_percent, measure)")
    print("  plot_lick_measure_by_cohort(cohorts, measure, group_by_sex=True/False)")
    print()
    print("Running basic lick plots for all cohorts...")

    if HAS_MATPLOTLIB:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = Path(f"lick_plots_{timestamp}")
        plot_dir.mkdir(exist_ok=True)
        combined_temp = combine_lick_cohorts(cohorts)
        available_measures = [m for m in
                               ["Total_Licks", "Total_Bouts", "Avg_ILI", "Avg_Bout_Duration"]
                               if m in combined_temp.columns]
        for measure in available_measures:
            try:
                fig = plot_lick_measure_by_cohort(
                    cohorts,
                    measure=measure,
                    save_path=plot_dir / f"lick_{measure.lower()}_by_cohort.svg",
                    show=False,
                )
                if fig:
                    import matplotlib.pyplot as _plt
                    _plt.close(fig)
            except Exception as e:
                print(f"  [WARNING] Plot for {measure} failed: {e}")
        print(f"[OK] Plots saved -> {plot_dir}")
    print()


def _select_lick_cohorts_interactively(n_cohorts: int) -> Dict[str, pd.DataFrame]:
    """
    Prompt user to select master CSV + capacitive log files for each cohort via GUI.

    For each cohort the user will:
      1. Optionally enter a label (or accept the auto-generated one)
      2. Select the master lick CSV (e.g. 0%_lick_data.csv)
      3. Select one or more capacitive log CSVs for that cohort

    Returns a dict of {label: DataFrame} ready for analysis.
    """
    if not HAS_TKINTER:
        print("[ERROR] tkinter not available — cannot use GUI file picker.")
        print("         Build cohort_specs manually and call load_lick_cohorts() directly.")
        return {}

    import tkinter as tk
    from tkinter import filedialog, simpledialog

    root = tk.Tk()
    root.withdraw()

    cohort_specs: Dict[str, dict] = {}

    default_labels = ["0% CA", "2% CA", "Ramp"]

    for i in range(n_cohorts):
        default_label = default_labels[i] if i < len(default_labels) else f"Cohort {i + 1}"
        label = simpledialog.askstring(
            "Cohort Label",
            f"Enter a label for cohort {i + 1} (default: {default_label}):",
            initialvalue=default_label,
            parent=root,
        )
        if label is None:
            label = default_label
        label = label.strip() or default_label

        print(f"\n  Cohort '{label}': select LICK master CSV...")
        print(f"         (choose the lick_data CSV, e.g. '0%_lick_data.csv' — NOT master_data_*.csv)")
        master_path = filedialog.askopenfilename(
            title=f"Select LICK master CSV for '{label}' (e.g. 0%_lick_data.csv)",
            filetypes=[("Lick data CSV", "*lick_data*.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
            parent=root,
        )
        if not master_path:
            print(f"  [WARNING] No master CSV selected for '{label}' — skipping cohort.")
            continue

        print(f"  Cohort '{label}': select capacitive log CSVs (hold Ctrl/Shift to pick multiple)...")
        cap_paths = filedialog.askopenfilenames(
            title=f"Select capacitive log CSV(s) for '{label}'",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=root,
        )
        if not cap_paths:
            print(f"  [WARNING] No capacitive logs selected for '{label}' — skipping cohort.")
            continue

        # Try to infer ca_percent from the label
        ca_pct = 0.0
        import re as _re
        _ca_match = _re.search(r'(\d+\.?\d*)\s*%', label)
        if _ca_match:
            try:
                ca_pct = float(_ca_match.group(1))
            except ValueError:
                ca_pct = 0.0

        cohort_specs[label] = {
            "master_csv":       Path(master_path),
            "capacitive_logs":  [Path(p) for p in cap_paths],
            "ca_percent":       ca_pct,
        }

    root.destroy()

    if not cohort_specs:
        print("[ERROR] No cohorts were configured.")
        return {}

    return load_lick_cohorts(cohort_specs)


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Step 1 -- choose number of cohorts and load files via GUI
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("CROSS-COHORT LICK ANALYSIS")
    print("=" * 80)
    print("\nHow many cohorts would you like to compare?")
    print("  2 -- compare two cohorts  (e.g. 0% vs 2%, 0% vs Ramp, 2% vs Ramp)")
    print("  3 -- compare three cohorts (0% vs 2% vs Ramp)")

    n_input = input("\nEnter 2 or 3: ").strip()
    try:
        n_cohorts = int(n_input)
        if n_cohorts not in (2, 3):
            raise ValueError
    except ValueError:
        print("[ERROR] Please enter 2 or 3. Exiting.")
        raise SystemExit(1)

    cohorts = _select_lick_cohorts_interactively(n_cohorts=n_cohorts)

    if not cohorts or len(cohorts) < n_cohorts:
        print("[ERROR] Not enough cohorts loaded. Exiting.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # Step 2 -- auto-detect comparison type and route to the right menu
    # ------------------------------------------------------------------ #
    comparison = _detect_lick_comparison_type(cohorts)

    label_map = {
        '0v2':    '0% nonramp  vs  2% nonramp',
        '0vramp': '0% nonramp  vs  Ramp',
        '2vramp': '2% nonramp  vs  Ramp',
        'all3':   '0% nonramp  vs  2% nonramp  vs  Ramp',
        'unknown': 'Unknown combination',
    }
    print("\n" + "=" * 80)
    print(f"Detected comparison type: {label_map.get(comparison, comparison)}")
    print("=" * 80)

    if comparison == 'unknown':
        print("\n[WARNING] Could not auto-detect comparison type from cohort labels.")
        print("  Loaded labels:", list(cohorts.keys()))
        print("\nManually select comparison type:")
        print("  1. 0% nonramp vs 2% nonramp")
        print("  2. 0% nonramp vs Ramp")
        print("  3. 2% nonramp vs Ramp")
        print("  4. All three cohorts")
        manual = input("\nSelect (1-4): ").strip()
        comparison = {'1': '0v2', '2': '0vramp', '3': '2vramp', '4': 'all3'}.get(manual, 'unknown')

    # ------------------------------------------------------------------ #
    # Step 3 -- dispatch to comparison-specific menu
    # ------------------------------------------------------------------ #
    if comparison == '0v2':
        _run_lick_0v2_menu(cohorts)
    else:
        _run_lick_unknown_menu(cohorts, comparison)

