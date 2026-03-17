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
    
    # Compute KDE peaks
    sensor_kdes = compute_sensor_KDE(df, sensor_cols, verbose=verbose)
    
    # Compute deviations
    df = compute_KDE_normalizations(df, sensor_cols, sensor_kdes)
    
    # Use FIXED threshold (same as lick_detection.py)
    thresholds = compute_fixed_thresholds(sensor_cols, fixed_threshold=fixed_threshold)
    
    # Detect events
    events_df = detect_events_above_threshold(df, sensor_cols, thresholds)
    
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
    
    # Check for required columns
    required_cols = ['date', 'animal_id', 'selected_sensors']
    missing_cols = [col for col in required_cols if col not in master_df.columns]
    if missing_cols:
        raise ValueError(f"Master CSV missing required columns: {missing_cols}. Found: {list(master_df.columns)}")
    
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
                    'Bottle_Weight_Change': bottle_weight,
                    'Total_Weight_Change': total_weight,
                    'Fecal_Count': fecal_count
                }
                
                all_animal_records.append(animal_record)
                
                print(f"      {animal_id} (Sensor {sensor_num}): {animal_licks} licks, {animal_bouts} bouts")
            
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
# PLOTTING FUNCTIONS
# =============================================================================

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
    ax.set_xlabel('Week', fontsize=12)
    
    # Create nice label from measure name
    measure_label = measure.replace('_', ' ')
    if 'Total' in measure:
        measure_label = measure_label + ' per Animal'
    elif 'Avg' in measure:
        measure_label = measure_label
    
    ax.set_ylabel(measure_label, fontsize=12)
    
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


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CROSS-COHORT LICK ANALYSIS")
    print("="*80)
    print("\nThis module analyzes lick data across multiple cohorts (e.g., 0% vs 2% CA).")
    print("\nUsage:")
    print("  1. Load cohorts: cohorts = load_lick_cohorts(cohort_specs)")
    print("  2. Run analyses: results = perform_cross_cohort_lick_anova(cohorts)")
    print("  3. Generate report: report = generate_lick_cohort_report(results)")
    print("\nExample cohort_specs structure (with capacitive files):")
    print("""
    from pathlib import Path
    
    cohort_specs = {
        "0% CA": {
            "master_csv": Path("0%_files/0%_lick_data.csv"),
            "capacitive_logs": [
                Path("0%_files/capacitive_log_2026-1-21.csv"),
                Path("0%_files/capacitive_log_2026-1-28.csv"),
                Path("0%_files/capacitive_log_2026-2-4.csv"),
                Path("0%_files/capacitive_log_2026-2-11.csv"),
                Path("0%_files/capacitive_log_2026-2-18.csv")
            ],
            "ca_percent": 0.0,
            "fixed_threshold": 0.01,  # Optional (default 0.01, same as lick_detection.py)
            "ili_cutoff": 0.3         # Optional
        },
        "2% CA": {
            "master_csv": Path("2%_files/2%_lick_data.csv"),
            "capacitive_logs": [
                Path("2%_files/capacitive_log_2026-1-28.csv"),
                Path("2%_files/capacitive_log_2026-2-4.csv"),
                Path("2%_files/capacitive_log_2026-2-11.csv"),
                Path("2%_files/capacitive_log_2026-2-18.csv"),
                Path("2%_files/capacitive_log_2026-2-25.csv")
            ],
            "ca_percent": 2.0
        }
    }
    
    # Load cohorts (automatically assigns sequential week numbers 0-4)
    cohorts = load_lick_cohorts(cohort_specs)
    
    # Plot licks across weeks (EXACT style from lick_nonramp.py)
    fig = plot_lick_measure_by_cohort(cohorts, measure="Total_Licks", use_std=False, show=True)
    
    # Now Week 0-4 for each cohort represents sequential time order
    # Week 0 of 0% CA aligns with Week 0 of 2% CA (first week for each)
    """)
    print("\n" + "="*80)
