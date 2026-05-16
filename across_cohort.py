"""
Cross-Cohort Weight & Behavioral Analysis Module

Compares body-weight trajectories and behavioral outcomes across 2–3 cohorts
(0% CA nonramp, 2% CA nonramp, CA-ramp, and/or 2-week ramp).  Each cohort is
loaded from its own master_data_*.csv file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python across_cohort.py

  A text prompt asks for the number of cohorts (2 or 3).  GUI file-pickers
  open for each cohort's master_data_*.csv.  The script auto-detects the
  comparison type (0v2, 0vramp, 2vramp, all3, rampramp) and routes to the
  appropriate analysis menu.

  To use programmatically:
    from across_cohort import load_cohorts, perform_cross_cohort_mixed_anova
    cohorts = load_cohorts({"0% CA": Path("..."), "2% CA": Path("...")})

  Required input: master_data_*.csv (one row per animal per day; columns
    include ID, Sex, CA (%), Date, Total Change, Daily Change, and
    behavioral Yes/No columns).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STATISTICAL ANALYSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Weight analyses
    - 3-way mixed ANOVA: CA% (between) × Day (within) × Sex (between)
    - Weekly version of the above (days averaged into weeks)
    - Between-subjects ANOVA at a fixed time point or averaged over days
    - Daily between-subjects ANOVA (all days held constant, CA% × Sex)
    - Sex-stratified mixed ANOVA: Day × CA% (per sex, daily & weekly)
    - CA%-stratified mixed ANOVA: Day × Sex (per CA%, daily & weekly)
    - CA% × Week two-way mixed ANOVA (sex collapsed)
    - Omnibus BH-FDR mixed ANOVA across multiple weight measures
    - OLS assumption diagnostics (normality, homoscedasticity, leverage)
    - Per-animal linear regression slopes (Total Change or Daily Change
      vs. Day/Week) with between-group Mann-Whitney U comparisons
      (Holm-Bonferroni corrected) and within-group slope significance
    - Complete slope analysis with reports and plots
    - Distribution diagnostics (via R if rpy2 available)
  Behavioral analyses
    - Mixed Cochran's Q / GEE / McNemar for binary behavioral outcomes
      (Nest Made, Lethargic, CA-Spot Digging, Anxious Behaviors)
    - Between-subjects chi-square for behavioral outcomes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PLOTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Weight trajectories
    - Total / Daily Change by individual animal (lines colored by cohort)
    - Total / Daily Change grouped by sex (mean ± SEM per cohort)
    - Total / Daily Change grouped by CA% (mean ± SEM)
    - Total / Daily Change grouped by cohort (mean ± SEM)
    - Weekly means by cohort over time
    - Slope comparison box/scatter plots (between CA% groups)
  Interaction effects
    - CA% × Sex interaction (bar chart, grand-average)
    - Time × CA% interaction (line plot)
    - Time × Sex interaction (line plot)
    - Three-way CA% × Sex × Time interaction (faceted)
    - All significant interactions in a single call
  Behavioral metrics
    - Bar charts of behavioral measures per cohort
    - Behavioral interaction-effect plots

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Required : pandas, numpy, matplotlib, scipy
  Optional : pingouin  (mixed ANOVA)
             statsmodels (Cochran's Q, McNemar, GEE)
             rpy2 + R packages lme4/lmerTest/emmeans (distribution diagnostics,
               polynomial contrasts)
             tkinter  (GUI file pickers — standard library on most platforms)
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import re
import math
import subprocess
import shutil
import tempfile
from datetime import datetime
from scipy import stats

# Try to import pingouin for mixed ANOVA
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Mixed ANOVA will not be available.")
    print("Install with: pip install pingouin")

# Try to import statsmodels for Cochran's Q, McNemar, and GEE tests
try:
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Behavioral repeated-measures tests will not be available.")
    print("Install with: pip install statsmodels")

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
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
    # Global Matplotlib defaults (match non_ramp_analysis style)
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
        "lines.markersize": 3,
        "figure.figsize": (4.5, 2.5),
    })
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will not be available.")

# Try to import rpy2 for R-based distribution diagnostics and assumption tests
try:
    import rpy2.robjects as _rpy2_probe
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

# Canonical cohort colours
_COLOR_0PCT  = "#1f77b4"   # 0% CA
_COLOR_2PCT  = "#f79520"   # 2% CA
_COLOR_RAMP  = "#2da048"   # Ramp
_COLOR_OTHER = "#7f3f98"   # fallback


def _cohort_label_to_color(label: str) -> str:
    """Map a cohort label to its canonical hex colour."""
    lo = str(label).lower()
    if "0%" in lo:
        return _COLOR_0PCT
    if "2%" in lo:
        return _COLOR_2PCT
    if "ramp" in lo:
        return _COLOR_RAMP
    return _COLOR_OTHER


def _ensure_r_path() -> None:
    """Ensure R's bin/x64 directory is in PATH before any rpy2 call (Windows only)."""
    import os, sys
    if sys.platform != 'win32':
        return
    r_home = os.environ.get('R_HOME', '')
    if not r_home:
        try:
            import rpy2.situation as _sit
            r_home = _sit.get_r_home() or ''
        except Exception:
            pass
    if not r_home:
        return
    for _sub in (os.path.join(r_home, 'bin', 'x64'),
                 os.path.join(r_home, 'bin')):
        if os.path.isdir(_sub) and _sub.lower() not in os.environ.get('PATH', '').lower():
            os.environ['PATH'] = _sub + os.pathsep + os.environ['PATH']


# Cache for loaded cohorts
_COHORT_DFS: Dict[str, pd.DataFrame] = {}
_COHORT_PATHS: Dict[str, Path] = {}


# =============================================================================
# COHORT LOADING FUNCTIONS
# =============================================================================

def load_cohorts(cohort_paths: Dict[str, Path], encoding: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple cohort CSV files into dataframes.
    
    Parameters:
        cohort_paths: Dictionary mapping cohort labels to CSV file paths
            Example: {"0%": Path("master_data_0%.csv"), "2%": Path("master_data_2%.csv")}
        encoding: Optional encoding for CSV files (default: None, uses pandas default)
        
    Returns:
        Dictionary mapping cohort labels to loaded DataFrames
        
    Example:
        >>> cohort_paths = {
        ...     "0% CA": Path("0%_files/master_data_0%.csv"),
        ...     "2% CA (6 animals)": Path("2%_6_animals_files/master_data_2%_6_animals.csv")
        ... }
        >>> cohorts = load_cohorts(cohort_paths)
        >>> print(cohorts["0% CA"].shape)
    """
    cohort_dfs = {}
    
    print("="*80)
    print("LOADING MULTIPLE COHORTS")
    print("="*80)
    
    for label, path in cohort_paths.items():
        if not path.exists():
            print(f"\n[WARNING] {label}: File not found: {path}")
            continue
            
        print(f"\nLoading cohort: {label}")
        print(f"  Path: {path}")
        
        try:
            df = pd.read_csv(path, encoding=encoding)
            cohort_dfs[label] = df
            print(f"  [OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue
    
    # Cache for later use
    global _COHORT_DFS, _COHORT_PATHS
    _COHORT_DFS = cohort_dfs.copy()
    _COHORT_PATHS = cohort_paths.copy()
    
    print(f"\n{'='*80}")
    print(f"Successfully loaded {len(cohort_dfs)} cohort(s)")
    print("="*80)
    
    return cohort_dfs


def select_cohort_files(n_cohorts: int = 2, initialdir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Open GUI dialogs to select multiple cohort CSV files.
    
    Parameters:
        n_cohorts: Number of cohorts to select
        initialdir: Starting directory for file picker
        
    Returns:
        Dictionary mapping auto-generated labels to selected file paths
    """
    if not HAS_TKINTER:
        raise RuntimeError("tkinter is required for GUI file selection. Install tkinter or provide paths directly.")
    
    cohort_paths = {}
    start_dir = str(initialdir or Path.cwd())
    
    print("\n" + "="*80)
    print(f"SELECT {n_cohorts} COHORT CSV FILES")
    print("="*80)
    
    for i in range(1, n_cohorts + 1):
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        
        file_path = filedialog.askopenfilename(
            title=f"Select CSV for Cohort {i}",
            initialdir=start_dir,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        
        root.destroy()
        
        if not file_path:
            print(f"  [INFO] Cohort {i} selection cancelled")
            break
        
        path = Path(file_path)
        # Auto-generate label from filename
        label = path.stem.replace("master_data_", "").replace("_", " ")
        cohort_paths[label] = path
        print(f"  [OK] Selected cohort {i}: {label}")
        
        # Update start directory to the selected file's parent
        start_dir = str(path.parent)
    
    return cohort_paths


def select_and_load_cohorts(n_cohorts: int = 2, initialdir: Optional[Path] = None, 
                            encoding: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function: Select cohort files via GUI and load them.
    
    Parameters:
        n_cohorts: Number of cohorts to select
        initialdir: Starting directory for file picker
        encoding: Optional encoding for CSV files
        
    Returns:
        Dictionary mapping cohort labels to loaded DataFrames
    """
    cohort_paths = select_cohort_files(n_cohorts=n_cohorts, initialdir=initialdir)
    
    if not cohort_paths:
        print("\n[WARNING] No cohorts selected")
        return {}
    
    return load_cohorts(cohort_paths, encoding=encoding)


# =============================================================================
# COHORT PREVIEW AND SUMMARY
# =============================================================================

def preview_cohorts(cohort_dfs: Dict[str, pd.DataFrame], n_rows: int = 5) -> None:
    """
    Display summary information for all loaded cohorts.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        n_rows: Number of sample rows to display per cohort
    """
    print("\n" + "="*80)
    print("COHORT PREVIEW")
    print("="*80)
    
    for label, df in cohort_dfs.items():
        print(f"\n{'-'*80}")
        print(f"COHORT: {label}")
        print(f"{'-'*80}")
        print(f"  Shape: {df.shape[0]} rows  x  {df.shape[1]} columns")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        
        # Show unique IDs if ID column exists
        if "ID" in df.columns:
            unique_ids = df["ID"].dropna().unique()
            print(f"  Unique IDs ({len(unique_ids)}): {', '.join(map(str, sorted(unique_ids)))}")
        
        # Show sex distribution if Sex column exists
        if "Sex" in df.columns:
            sex_counts = df["Sex"].value_counts()
            print(f"  Sex distribution: {dict(sex_counts)}")
        
        # Show date range if Date column exists
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
            if len(dates) > 0:
                print(f"  Date range: {dates.min().date()} to {dates.max().date()}")
                n_days = (dates.max() - dates.min()).days + 1
                print(f"  Duration: {n_days} days")
        
        # Show first few rows
        if n_rows > 0:
            print(f"\n  Sample data (first {n_rows} rows):")
            with pd.option_context('display.max_columns', 12, 'display.width', 160):
                print(df.head(n_rows).to_string(index=False))


def summarize_cohorts(cohort_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table comparing basic statistics across cohorts.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        
    Returns:
        DataFrame with summary statistics for each cohort
    """
    summary_data = []
    
    for label, df in cohort_dfs.items():
        summary = {
            'Cohort': label,
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
        }
        
        # Count unique IDs
        if 'ID' in df.columns:
            summary['N Animals'] = df['ID'].nunique()
        
        # Sex counts
        if 'Sex' in df.columns:
            summary['N Male'] = (df['Sex'] == 'M').sum()
            summary['N Female'] = (df['Sex'] == 'F').sum()
        
        # Date range
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'], errors='coerce')
            if dates.notna().any():
                summary['Start Date'] = dates.min().date()
                summary['End Date'] = dates.max().date()
                summary['N Days'] = (dates.max() - dates.min()).days + 1
        
        # Check for key columns
        key_columns = ['Weight', 'Daily Change', 'Total Change', 'Fecal Count']
        for col in key_columns:
            summary[f'Has {col}'] = 'Yes' if col in df.columns else 'No'
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("COHORT SUMMARY TABLE")
    print("="*80)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df


# =============================================================================
# COHORT DATA CLEANING
# =============================================================================

def clean_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single cohort dataframe with standardized transformations.
    
    This applies the same cleaning logic used in single-cohort analysis:
    - Convert Date to datetime
    - Convert numeric columns to appropriate types
    - Handle missing values
    
    Parameters:
        df: Raw cohort DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Convert Date column to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Convert DOB to datetime
    if "DOB" in df.columns:
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    
    # Numeric columns to convert
    numeric_cols = ["Weight", "Daily Change", "Total Change", "Fecal Count",
                   "Initial Bottle Weight", "Final Bottle Weight", "Bottle Weight Change"]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Categorical columns
    categorical_cols = ["Sex", "Strain", "Nest Made?", "Lethargy?", 
                       "Anxious Behaviors?", "CA Spot Digging?"]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": None, "": None})
    
    return df


def clean_all_cohorts(cohort_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean all cohort dataframes using standardized cleaning.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to raw DataFrames
        
    Returns:
        Dictionary mapping cohort labels to cleaned DataFrames
    """
    cleaned_cohorts = {}
    
    print("\n" + "="*80)
    print("CLEANING COHORT DATAFRAMES")
    print("="*80)
    
    for label, df in cohort_dfs.items():
        print(f"\n  Cleaning cohort: {label}")
        cleaned_df = clean_cohort(df)
        cleaned_cohorts[label] = cleaned_df
        print(f"    [OK] Cleaned {len(cleaned_df)} rows")
    
    # Update cache
    global _COHORT_DFS
    _COHORT_DFS = cleaned_cohorts.copy()
    
    print("\n" + "="*80)
    print(f"All {len(cleaned_cohorts)} cohort(s) cleaned successfully")
    print("="*80)
    
    return cleaned_cohorts


# =============================================================================
# COHORT CACHE MANAGEMENT
# =============================================================================

def get_cached_cohorts() -> Dict[str, pd.DataFrame]:
    """
    Retrieve cached cohort dataframes.
    
    Returns:
        Dictionary of cached cohort DataFrames (empty dict if none loaded)
    """
    return _COHORT_DFS.copy()


def get_cached_paths() -> Dict[str, Path]:
    """
    Retrieve cached cohort file paths.
    
    Returns:
        Dictionary of cached cohort file paths (empty dict if none loaded)
    """
    return _COHORT_PATHS.copy()


def clear_cache() -> None:
    """Clear all cached cohort data."""
    global _COHORT_DFS, _COHORT_PATHS
    _COHORT_DFS.clear()
    _COHORT_PATHS.clear()
    print("[OK] Cohort cache cleared")


# =============================================================================
# CROSS-COHORT STATISTICAL ANALYSIS
# =============================================================================

def combine_cohorts_for_analysis(cohort_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple cohort dataframes into a single dataframe for cross-cohort analysis.
    
    Each cohort is assigned a CA% label based on the cohort name.
    Adds 'CA (%)' and 'Cohort' columns to identify the source cohort.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        
    Returns:
        Combined DataFrame with all cohorts, including 'CA (%)' and 'Cohort' columns
    """
    combined_dfs = []
    
    for label, df in cohort_dfs.items():
        df_copy = df.copy()
        
        # Add cohort identifier
        df_copy['Cohort'] = label
        
        # Extract CA% from cohort label if not already present
        if 'CA (%)' not in df_copy.columns:
            # Try to extract percentage from label (e.g., "0%", "2%", "0% CA", "2% (6 animals)")
            match = re.search(r'(\d+(?:\.\d+)?)\s*%', label)
            if match:
                ca_percent = float(match.group(1))
                df_copy['CA (%)'] = ca_percent
            else:
                print(f"[WARNING] Could not extract CA% from cohort label: {label}")
                df_copy['CA (%)'] = None
        
        combined_dfs.append(df_copy)
    
    combined = pd.concat(combined_dfs, ignore_index=True)
    
    # Ensure CA% is numeric (convert any strings to float)
    if 'CA (%)' in combined.columns:
        combined['CA (%)'] = pd.to_numeric(combined['CA (%)'], errors='coerce')
    
    print(f"\n[OK] Combined {len(cohort_dfs)} cohorts into single dataframe")
    print(f"  Total rows: {len(combined)}")
    print(f"  CA% levels: {sorted(combined['CA (%)'].dropna().unique())}")
    
    return combined


def add_day_column_across_cohorts(combined_df: pd.DataFrame, drop_ramp_baseline: bool = True) -> pd.DataFrame:
    """
    Add a cohort-aligned 'Day' column to combined data.
    
    Parameters:
        combined_df: Combined DataFrame with ID and Date columns
        drop_ramp_baseline: If True (default), drop ramp cohort Day 1 rows (baseline
            with Total Change = 0). Set False when preparing data for plotting so that
            the first ramp data point is visible.
        
    Returns:
        DataFrame with added 'Day' column
    """
    if 'Day' in combined_df.columns:
        print("[INFO] 'Day' column already exists")
        return combined_df
    
    if not {'ID', 'Date'}.issubset(combined_df.columns):
        raise ValueError("DataFrame must have 'ID' and 'Date' columns")
    
    df = combined_df.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Sort by ID and Date
    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    
    # Compute 1-indexed Day per ID:
    #   nonramp: first date = Day 0 (baseline, excluded from plots); Day 1+ plotted
    #   ramp:    first date = Day 1 (no baseline day skipped)
    first_dates = df.groupby('ID')['Date'].transform('min')
    if 'Cohort' in df.columns:
        # Apply per-cohort offset based on whether the cohort label contains 'ramp'
        is_ramp = df['Cohort'].str.lower().str.contains('ramp', na=False)
        df['Day'] = (df['Date'] - first_dates).dt.days + is_ramp.astype(int)
    else:
        # Default to nonramp logic (Day 0 = baseline)
        df['Day'] = (df['Date'] - first_dates).dt.days

    print(f"[OK] Added 'Day' column (range: {df['Day'].min()} to {df['Day'].max()})")
    
    # Drop baseline rows:
    #   nonramp: Day 0 = baseline → keep Day >= 1
    #   ramp:    Day 1 = baseline (Total Change = 0) → only drop for analyses
    df = df[df['Day'] >= 1].copy()
    if drop_ramp_baseline and 'Cohort' in df.columns:
        is_ramp = df['Cohort'].str.lower().str.contains('ramp', na=False)
        df = df[~(is_ramp & (df['Day'] == 1))].copy()
        print(f"[OK] Excluded ramp cohort baseline (Day 1) rows")
    print(f"[OK] Filtered baseline rows (range: {df['Day'].min()} to {df['Day'].max()})")

    return df


def perform_cross_cohort_mixed_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_points: Optional[List[int]] = None,
    ss_type: int = 3
) -> Dict:
    """
    Perform 3-way Mixed ANOVA across cohorts: CA% (between)  x  Time (within)  x  Sex (between).
    
    This analyzes how weight measures change over time, comparing different CA% concentrations
    (represented by different cohorts) and sexes.
    
    Design:
        - CA%: Between-subjects factor (each cohort = one CA% level)
        - Time (Day): Within-subjects factor (repeated measures over days)
        - Sex: Between-subjects factor (M or F)
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_points: Optional list of specific days to include (None = all days)
        ss_type: Sum of squares type (2 or 3). Use 3 for unbalanced designs.
        
    Returns:
        Dictionary with ANOVA results including main effects and interactions
        
    Example:
        >>> cohorts = load_cohorts(paths)
        >>> results = perform_cross_cohort_mixed_anova(cohorts, measure="Total Change")
    """
    print("\n" + "="*80)
    print("CROSS-COHORT MIXED ANOVA: CA%  x  TIME  x  SEX")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("\n[ERROR] pingouin is required for mixed ANOVA")
        print("Install with: pip install pingouin")
        return {}
    
    # Combine cohorts into single dataframe
    print("\nStep 1: Combining cohort dataframes...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    
    # Clean the combined dataframe
    print("\nStep 2: Cleaning combined data...")
    combined_df = clean_cohort(combined_df)
    
    # Add Day column if not present
    if 'Day' not in combined_df.columns:
        print("\nStep 3: Adding Day column...")
        combined_df = add_day_column_across_cohorts(combined_df)
    
    # Prepare data for ANOVA
    print("\nStep 4: Preparing data for ANOVA...")
    required_cols = ['ID', 'Day', 'Sex', 'CA (%)', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Day'].isin(time_points)]
        print(f"  Filtered to days: {sorted(time_points)}")
    else:
        time_points = sorted(analysis_df['Day'].unique())
        print(f"  Using all available days: {len(time_points)} days")
    
    print(f"\nAnalyzing: {measure}")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA (%)'].unique())}")
    print(f"  Days: {sorted(analysis_df['Day'].unique())}")
    
    # Show IDs by cohort for verification
    print(f"\nAnimal IDs by Cohort:")
    for ca_val in sorted(analysis_df['CA (%)'].unique()):
        ids_in_cohort = sorted(analysis_df[analysis_df['CA (%)'] == ca_val]['ID'].unique())
        print(f"  CA% = {ca_val}: {len(ids_in_cohort)} animals")
        print(f"    {', '.join(ids_in_cohort)}")
    
    # Descriptive statistics by group
    print(f"\nDescriptive Statistics by Group:")
    
    # Enhanced descriptive statistics
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1)
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        return pd.Series({
            'count': n,
            'mean': mean,
            'median': group.median(),
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'q25': group.quantile(0.25),
            'q75': group.quantile(0.75),
            'min': group.min(),
            'max': group.max()
        })
    
    # Collect statistics for each group
    stats_data = []
    for (ca_val, sex_val), group_data in analysis_df.groupby(['CA (%)', 'Sex'])[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA (%)'] = ca_val
        stats['Sex'] = sex_val
        stats_data.append(stats)
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
              f"n_obs={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
              f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
    
    # Check data completeness
    subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
    total_days = analysis_df['Day'].nunique()
    complete_subjects = (subjects_per_day == total_days).sum()
    incomplete_subjects = (subjects_per_day < total_days).sum()
    
    if incomplete_subjects > 0:
        print(f"\n[WARNING] {incomplete_subjects} animals have incomplete time series")
        print(f"  Complete: {complete_subjects}, Incomplete: {incomplete_subjects}")
        
        # Show which animals are incomplete
        incomplete_ids = subjects_per_day[subjects_per_day < total_days]
        print(f"\n  Incomplete animals:")
        for animal_id, n_days in incomplete_ids.items():
            print(f"    {animal_id}: {n_days}/{total_days} days")

    
    # Perform mixed ANOVA
    print(f"\nStep 5: Running mixed ANOVA (Type {ss_type} SS)...")
    print("  Design: CA% (between)  x  Time (within)  x  Sex (between)")
    
    try:
        # Use pingouin's mixed_anova for within-between design
        # We need to specify which factor is within-subjects (Day) and which are between (CA%, Sex)
        
        # Create a combined between-subjects factor for CA%  x  Sex
        analysis_df['Group'] = (analysis_df['CA (%)'].astype(str) + '%_' + 
                               analysis_df['Sex'].astype(str))
        
        # Run mixed ANOVA: Day (within)  x  Group (between)
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Day',
            subject='ID',
            between='Group',
            correction=True  # Greenhouse-Geisser correction for sphericity
        )
        
        print("\nMixed ANOVA Results (Day  x  Group):")
        print(aov.to_string())
        
        # Now run separate ANOVAs to decompose the Group effect into CA%, Sex, and CA% x Sex
        print("\n" + "="*80)
        print("DECOMPOSING GROUP EFFECTS: CA%  x  SEX")
        print("="*80)
        
        # Average across days for between-subjects effects
        subject_means = analysis_df.groupby(['ID', 'CA (%)', 'Sex'])[measure].mean().reset_index()
        
        # Rename 'CA (%)' to avoid special characters for pingouin
        subject_means_clean = subject_means.rename(columns={'CA (%)': 'CA_percent'})
        
        # Two-way ANOVA: CA%  x  Sex on averaged data
        between_aov = pg.anova(
            data=subject_means_clean,
            dv=measure,
            between=['CA_percent', 'Sex'],
            ss_type=ss_type
        )
        
        # Rename back in results for clarity
        between_aov['Source'] = between_aov['Source'].replace({
            'CA_percent': 'CA (%)',
            'CA_percent * Sex': 'CA (%) * Sex'
        })
        
        print("\nBetween-Subjects Effects (averaged across time):")
        print(between_aov.to_string())
        
        # Format results
        p_col = 'p-unc' if 'p-unc' in between_aov.columns else 'p_unc'
        
        # Post-hoc pairwise comparisons
        print("\n" + "="*80)
        print("POST-HOC PAIRWISE COMPARISONS")
        print("="*80)
        
        posthoc_results = {}
        
        # Check for significant main effects and interactions
        for source in ['CA (%)', 'Sex', 'CA (%) * Sex']:
            if source in between_aov['Source'].values:
                row = between_aov[between_aov['Source'] == source].iloc[0]
                p_val = row[p_col]
                
                if p_val < 0.05:
                    print(f"\n{source} is significant (p = {p_val:.4f}). Running pairwise comparisons...")
                    
                    try:
                        if source == 'CA (%)':
                            # Pairwise comparisons for CA%
                            pw_ca = pg.pairwise_tests(
                                data=subject_means,
                                dv=measure,
                                between='CA (%)',
                                padjust='bonf'
                            )
                            print("\nPairwise comparisons for CA%:")
                            print(pw_ca.to_string())
                            posthoc_results['CA (%)'] = pw_ca
                            
                        elif source == 'Sex':
                            # Pairwise comparisons for Sex (if more than 2 levels)
                            pw_sex = pg.pairwise_tests(
                                data=subject_means,
                                dv=measure,
                                between='Sex',
                                padjust='bonf'
                            )
                            print("\nPairwise comparisons for Sex:")
                            print(pw_sex.to_string())
                            posthoc_results['Sex'] = pw_sex
                            
                        elif source == 'CA (%) * Sex':
                            # Simple main effects for interaction
                            print("\nSimple main effects of CA% at each Sex level:")
                            for sex_level in subject_means['Sex'].unique():
                                sex_data = subject_means[subject_means['Sex'] == sex_level]
                                if sex_data['CA (%)'].nunique() > 1:
                                    pw_ca_at_sex = pg.pairwise_tests(
                                        data=sex_data,
                                        dv=measure,
                                        between='CA (%)',
                                        padjust='bonf'
                                    )
                                    print(f"\n  CA% effect at Sex={sex_level}:")
                                    print(pw_ca_at_sex.to_string())
                                    posthoc_results[f'CA (%) at Sex={sex_level}'] = pw_ca_at_sex
                            
                            print("\nSimple main effects of Sex at each CA% level:")
                            for ca_level in subject_means['CA (%)'].unique():
                                ca_data = subject_means[subject_means['CA (%)'] == ca_level]
                                if ca_data['Sex'].nunique() > 1:
                                    pw_sex_at_ca = pg.pairwise_tests(
                                        data=ca_data,
                                        dv=measure,
                                        between='Sex',
                                        padjust='bonf'
                                    )
                                    print(f"\n  Sex effect at CA%={ca_level}:")
                                    print(pw_sex_at_ca.to_string())
                                    posthoc_results[f'Sex at CA%={ca_level}'] = pw_sex_at_ca
                    
                    except Exception as e:
                        print(f"  [WARNING] Post-hoc test failed: {e}")
                else:
                    print(f"\n{source} is not significant (p = {p_val:.4f}). Skipping pairwise comparisons.")
        
        print("\n" + "="*80)
        print("FORMATTED RESULTS")
        print("="*80)
        
        # Main effects and interactions from between-subjects ANOVA
        for source in ['CA (%)', 'Sex', 'CA (%) * Sex']:
            if source in between_aov['Source'].values:
                row = between_aov[between_aov['Source'] == source].iloc[0]
                F_val = row['F']
                p_val = row[p_col]
                df1 = row['DF']
                
                # Get residual DF for reporting
                resid_row = between_aov[between_aov['Source'] == 'Residual']
                if len(resid_row) > 0:
                    df2 = resid_row.iloc[0]['DF']
                else:
                    df2 = np.nan
                
                sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"\n{source}:")
                print(f"  F({df1:.0f},{df2:.0f}) = {F_val:.3f}, p = {p_val:.4f} {sig_str}")
        
        # Time effects from mixed ANOVA
        if 'Day' in aov['Source'].values:
            day_row = aov[aov['Source'] == 'Day'].iloc[0]
            time_p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            time_p = day_row[time_p_col]
            time_F = day_row['F']
            time_df1 = day_row.get('DF', np.nan)
            time_sig = "***" if time_p < 0.001 else "**" if time_p < 0.01 else "*" if time_p < 0.05 else "ns"
            
            print(f"\nTime (Day) effect:")
            print(f"  F({time_df1:.0f},?) = {time_F:.3f}, p = {time_p:.4f} {time_sig}")
        
        # Interaction with time
        if 'Interaction' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'Interaction'].iloc[0]
            int_p = int_row[time_p_col]
            int_F = int_row['F']
            int_df1 = int_row.get('DF', np.nan)
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\nDay  x  Group interaction:")
            print(f"  F({int_df1:.0f},?) = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
        
        results = {
            'measure': measure,
            'type': 'mixed_anova_cross_cohort',
            'ss_type': ss_type,
            'n_observations': len(analysis_df),
            'n_subjects': analysis_df['ID'].nunique(),
            'n_days': len(time_points),
            'mixed_anova_table': aov,
            'between_anova_table': between_aov,
            'posthoc_tests': posthoc_results,
            'descriptive_stats': group_stats,
            'data': analysis_df
        }
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# BETWEEN-SUBJECTS ANOVA (HOLDING TIME CONSTANT)
# =============================================================================

def perform_between_subjects_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_point: Optional[int] = None,
    average_over_days: bool = False,
    ss_type: int = 3
) -> Dict:
    """
    Perform 2-Way Between-Subjects ANOVA: CA%  x  Sex (holding time constant)
    
    This analyzes the effect of CA% and Sex on weight measures at:
    1. A specific time point (e.g., final day)
    2. Averaged across all days per animal
    
    Design:
        - CA%: Between-subjects factor (each animal assigned to one cohort)
        - Sex: Between-subjects factor (M or F)
        - Time: Held constant (single day or averaged)
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_point: Specific Day to analyze (e.g., 35 for final day), None with average_over_days=True
        average_over_days: If True, average measure across all days per animal
        ss_type: Sum of squares type (2 or 3). Use 3 for unbalanced designs.
        
    Returns:
        Dictionary with ANOVA results
        
    Example:
        >>> # Analyze final day (Day 35)
        >>> results_final = perform_between_subjects_anova(cohorts, measure="Total Change", time_point=35)
        >>> 
        >>> # Analyze average across all days
        >>> results_avg = perform_between_subjects_anova(cohorts, measure="Total Change", average_over_days=True)
    """
    print("\n" + "="*80)
    print("BETWEEN-SUBJECTS ANOVA: CA%  x  SEX")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("\n[ERROR] pingouin is required for ANOVA")
        print("Install with: pip install pingouin")
        return {}
    
    # Combine cohorts
    print("\nStep 1: Combining cohort dataframes...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    
    # Clean the combined dataframe
    print("\nStep 2: Cleaning combined data...")
    combined_df = clean_cohort(combined_df)
    
    # Add Day column if not present
    if 'Day' not in combined_df.columns:
        print("\nStep 3: Adding Day column...")
        combined_df = add_day_column_across_cohorts(combined_df)
    
    # Prepare data
    print("\nStep 4: Preparing data for ANOVA...")
    required_cols = ['ID', 'Sex', 'CA (%)', measure]
    if time_point is not None:
        required_cols.append('Day')
    
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    # Filter to specific time point or average across days
    if time_point is not None and 'Day' in analysis_df.columns:
        analysis_df = analysis_df[analysis_df['Day'] == time_point]
        print(f"  Filtered to Day {time_point}")
        print(f"  Total observations: {len(analysis_df)}")
        analysis_type = f"Day {time_point}"
    elif average_over_days:
        # Average the measure across all days for each animal
        print("  Averaging measure across all days per animal...")
        analysis_df = analysis_df.groupby(['ID', 'Sex', 'CA (%)'], as_index=False)[measure].mean()
        print(f"  Total subjects: {len(analysis_df)}")
        analysis_type = "Averaged across all days"
    else:
        raise ValueError("Must specify either time_point or set average_over_days=True")
    
    print(f"\nAnalyzing: {measure} ({analysis_type})")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA (%)'].unique())}")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by Group:")
    
    # Enhanced descriptive statistics
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1)
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        return pd.Series({
            'count': n,
            'mean': mean,
            'median': group.median(),
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'q25': group.quantile(0.25),
            'q75': group.quantile(0.75),
            'min': group.min(),
            'max': group.max()
        })
    
    # Collect statistics for each group
    stats_data = []
    for (ca_val, sex_val), group_data in analysis_df.groupby(['CA (%)', 'Sex'])[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA (%)'] = ca_val
        stats['Sex'] = sex_val
        stats_data.append(stats)
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
              f"n={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
              f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
    
    print(f"\nStep 5: Running 2-way between-subjects ANOVA (Type {ss_type} SS)...")
    print("  Design: CA% (between)  x  Sex (between)")
    
    try:
        # Rename 'CA (%)' to avoid special characters
        analysis_df_clean = analysis_df.rename(columns={'CA (%)': 'CA_percent'})
        
        # Perform 2-way ANOVA
        aov = pg.anova(
            data=analysis_df_clean,
            dv=measure,
            between=['CA_percent', 'Sex'],
            ss_type=ss_type
        )
        
        # Rename back for clarity
        aov['Source'] = aov['Source'].replace({
            'CA_percent': 'CA (%)',
            'CA_percent * Sex': 'CA (%) * Sex'
        })
        
        print("\nANOVA Table:")
        print(aov.to_string())
        
        # Format results
        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        
        # Post-hoc pairwise comparisons
        print("\n" + "="*80)
        print("POST-HOC PAIRWISE COMPARISONS")
        print("="*80)
        
        posthoc_results = {}
        
        # Check for significant main effects and interactions
        for source in ['CA (%)', 'Sex', 'CA (%) * Sex']:
            if source in aov['Source'].values:
                row = aov[aov['Source'] == source].iloc[0]
                p_val = row[p_col]
                
                if p_val < 0.05:
                    print(f"\n{source} is significant (p = {p_val:.4f}). Running pairwise comparisons...")
                    
                    try:
                        if source == 'CA (%)':
                            # Pairwise comparisons for CA%
                            pw_ca = pg.pairwise_tests(
                                data=analysis_df_clean,
                                dv=measure,
                                between='CA_percent',
                                padjust='bonf'
                            )
                            # Rename back for clarity
                            print("\nPairwise comparisons for CA%:")
                            print(pw_ca.to_string())
                            posthoc_results['CA (%)'] = pw_ca
                            
                        elif source == 'Sex':
                            # Pairwise comparisons for Sex
                            pw_sex = pg.pairwise_tests(
                                data=analysis_df_clean,
                                dv=measure,
                                between='Sex',
                                padjust='bonf'
                            )
                            print("\nPairwise comparisons for Sex:")
                            print(pw_sex.to_string())
                            posthoc_results['Sex'] = pw_sex
                            
                        elif source == 'CA (%) * Sex':
                            # Simple main effects for interaction
                            print("\nSimple main effects of CA% at each Sex level:")
                            for sex_level in analysis_df_clean['Sex'].unique():
                                sex_data = analysis_df_clean[analysis_df_clean['Sex'] == sex_level]
                                if sex_data['CA_percent'].nunique() > 1:
                                    pw_ca_at_sex = pg.pairwise_tests(
                                        data=sex_data,
                                        dv=measure,
                                        between='CA_percent',
                                        padjust='bonf'
                                    )
                                    print(f"\n  CA% effect at Sex={sex_level}:")
                                    print(pw_ca_at_sex.to_string())
                                    posthoc_results[f'CA (%) at Sex={sex_level}'] = pw_ca_at_sex
                            
                            print("\nSimple main effects of Sex at each CA% level:")
                            for ca_level in analysis_df['CA (%)'].unique():
                                ca_data = analysis_df_clean[analysis_df_clean['CA_percent'] == ca_level]
                                if ca_data['Sex'].nunique() > 1:
                                    pw_sex_at_ca = pg.pairwise_tests(
                                        data=ca_data,
                                        dv=measure,
                                        between='Sex',
                                        padjust='bonf'
                                    )
                                    print(f"\n  Sex effect at CA%={ca_level}:")
                                    print(pw_sex_at_ca.to_string())
                                    posthoc_results[f'Sex at CA%={ca_level}'] = pw_sex_at_ca
                    
                    except Exception as e:
                        print(f"  [WARNING] Post-hoc test failed: {e}")
                else:
                    print(f"\n{source} is not significant (p = {p_val:.4f}). Skipping pairwise comparisons.")
        
        print("\n" + "="*80)
        print("FORMATTED RESULTS")
        print("="*80)
        
        # Extract and display each effect
        for source in ['CA (%)', 'Sex', 'CA (%) * Sex']:
            if source in aov['Source'].values:
                row = aov[aov['Source'] == source].iloc[0]
                F_val = row['F']
                p_val = row[p_col]
                df1 = row['DF']
                
                # Get residual DF
                resid_row = aov[aov['Source'] == 'Residual']
                if len(resid_row) > 0:
                    df2 = resid_row.iloc[0]['DF']
                else:
                    df2 = np.nan
                
                sig_str = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"\n{source}:")
                print(f"  F({df1:.0f},{df2:.0f}) = {F_val:.3f}, p = {p_val:.4f} {sig_str}")
        
        # Package results
        results = {
            'measure': measure,
            'analysis_type': analysis_type,
            'time_point': time_point,
            'averaged': average_over_days,
            'type': 'between_subjects',
            'ss_type': ss_type,
            'n_subjects': analysis_df['ID'].nunique(),
            'anova_table': aov,
            'posthoc_tests': posthoc_results,
            'descriptive_stats': group_stats,
            'data': analysis_df
        }
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Between-subjects ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# DAILY BETWEEN-SUBJECTS ANALYSIS (ALL DAYS HELD CONSTANT)
# =============================================================================

def perform_daily_between_subjects_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    ss_type: int = 3
) -> Dict:
    """
    Perform Between-Subjects ANOVA (CA%  x  Sex) for EACH day separately.
    
    This provides a day-by-day analysis showing how the CA% and Sex effects
    change over time, similar to the CAH weight analysis approach.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        ss_type: Sum of squares type (2 or 3). Use 3 for unbalanced designs.
        
    Returns:
        Dictionary with daily results and summary tables
    """
    print("\n" + "="*80)
    print("DAILY BETWEEN-SUBJECTS ANOVA: CA%  x  SEX FOR EACH DAY")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("\n[ERROR] pingouin is required for ANOVA")
        return {}
    
    # Combine cohorts
    print("\nStep 1: Preparing data...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    
    # Get list of all days
    all_days = sorted(combined_df['Day'].unique())
    print(f"  Analyzing {len(all_days)} days: {all_days[0]} to {all_days[-1]}")
    
    # Store results for each day
    daily_results = []
    
    print("\nStep 2: Running ANOVA for each day...")
    
    for day in all_days:
        # Filter to this day
        day_data = combined_df[combined_df['Day'] == day][['ID', 'Sex', 'CA (%)', measure]].copy()
        day_data = day_data.dropna()
        
        if len(day_data) < 4:  # Need at least 4 observations for 2x2 design
            print(f"  Day {day}: Skipped (insufficient data)")
            continue
        
        try:
            # Rename column for pingouin
            day_data_clean = day_data.rename(columns={'CA (%)': 'CA_percent'})
            
            # Run 2-way ANOVA
            aov = pg.anova(
                data=day_data_clean,
                dv=measure,
                between=['CA_percent', 'Sex'],
                ss_type=ss_type
            )
            
            # Extract key statistics
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            result = {
                'Day': day,
                'N': len(day_data)
            }
            
            # CA% effect
            if 'CA_percent' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA_percent'].iloc[0]
                result['CA_F'] = ca_row['F']
                result['CA_p'] = ca_row[p_col]
                result['CA_sig'] = '*' if ca_row[p_col] < 0.05 else 'ns'
            
            # Sex effect
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                result['Sex_F'] = sex_row['F']
                result['Sex_p'] = sex_row[p_col]
                result['Sex_sig'] = '*' if sex_row[p_col] < 0.05 else 'ns'
            
            # Interaction
            if 'CA_percent * Sex' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA_percent * Sex'].iloc[0]
                result['Interaction_F'] = int_row['F']
                result['Interaction_p'] = int_row[p_col]
                result['Interaction_sig'] = '*' if int_row[p_col] < 0.05 else 'ns'
            
            daily_results.append(result)
            
        except Exception as e:
            print(f"  Day {day}: Error - {e}")
            continue
    
    # Convert to DataFrame for easy display
    results_df = pd.DataFrame(daily_results)
    
    print(f"\n[OK] Completed analysis for {len(results_df)} days")
    
    # Create summary table
    print("\n" + "="*80)
    print("DAILY ANOVA RESULTS SUMMARY")
    print("="*80)
    
    if len(results_df) > 0:
        # Format the display
        display_cols = ['Day', 'N', 'CA_F', 'CA_p', 'CA_sig', 'Sex_F', 'Sex_p', 'Sex_sig', 
                       'Interaction_F', 'Interaction_p', 'Interaction_sig']
        available_cols = [col for col in display_cols if col in results_df.columns]
        
        print("\nCA% Effect by Day:")
        print("-" * 80)
        if 'CA_F' in results_df.columns:
            ca_cols = ['Day', 'N', 'CA_F', 'CA_p', 'CA_sig']
            print(results_df[ca_cols].to_string(index=False))
            n_sig = (results_df['CA_p'] < 0.05).sum()
            print(f"\nSignificant on {n_sig}/{len(results_df)} days")
        
        print("\nSex Effect by Day:")
        print("-" * 80)
        if 'Sex_F' in results_df.columns:
            sex_cols = ['Day', 'N', 'Sex_F', 'Sex_p', 'Sex_sig']
            print(results_df[sex_cols].to_string(index=False))
            n_sig = (results_df['Sex_p'] < 0.05).sum()
            print(f"\nSignificant on {n_sig}/{len(results_df)} days")
        
        print("\nCA%  x  Sex Interaction by Day:")
        print("-" * 80)
        if 'Interaction_F' in results_df.columns:
            int_cols = ['Day', 'N', 'Interaction_F', 'Interaction_p', 'Interaction_sig']
            print(results_df[int_cols].to_string(index=False))
            n_sig = (results_df['Interaction_p'] < 0.05).sum()
            print(f"\nSignificant on {n_sig}/{len(results_df)} days")
    
    return {
        'measure': measure,
        'type': 'daily_between_subjects',
        'ss_type': ss_type,
        'n_days': len(results_df),
        'results_table': results_df,
        'all_days': all_days
    }


# =============================================================================
# STRATIFIED MIXED ANOVA (HOLDING SEX OR CA% CONSTANT)
# =============================================================================

def perform_mixed_anova_sex_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    sex: str,
    measure: str = "Total Change",
    time_points: Optional[List[int]] = None
) -> Dict:
    """
    Perform 2-Way Mixed ANOVA: Time (within)  x  CA% (between), holding Sex constant.
    
    This analyzes longitudinal weight changes for ONE sex at a time, testing:
    - Time (Day): Within-subjects factor (repeated measures over days)
    - CA%: Between-subjects factor (different cohorts)
    
    By stratifying by sex, this reveals whether the Time  x  CA% interaction differs
    between males and females.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        sex: Sex to analyze ("M" or "F")
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_points: Optional list of specific days to include (None = all days)
        
    Returns:
        Dictionary with ANOVA results for the specified sex
    """
    print("\n" + "="*80)
    print(f"SEX-STRATIFIED MIXED ANOVA: TIME (WITHIN)  x  CA% (BETWEEN)")
    print(f"Analyzing: {'MALES' if sex == 'M' else 'FEMALES'} only")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("\n[ERROR] pingouin is required for mixed ANOVA")
        return {}
    
    # Validate sex parameter
    if sex not in ["M", "F"]:
        raise ValueError(f"sex must be 'M' or 'F', got '{sex}'")
    
    # Combine cohorts
    print("\nStep 1: Combining and filtering data...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    
    # Filter to specified sex
    required_cols = ['ID', 'Day', 'Sex', 'CA (%)', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[combined_df['Sex'] == sex][required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) == 0:
        print(f"\n[ERROR] No data available for Sex={sex}")
        return {}
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Day'].isin(time_points)]
        print(f"  Filtered to days: {sorted(time_points)}")
    else:
        time_points = sorted(analysis_df['Day'].unique())
    
    print(f"\nAnalyzing: {measure} ({'Males' if sex == 'M' else 'Females'})")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA (%)'].unique())}")
    print(f"  Days: {len(time_points)} days")
    
    # Check completeness
    subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
    total_days = analysis_df['Day'].nunique()
    complete_subjects = (subjects_per_day == total_days).sum()
    incomplete_subjects = (subjects_per_day < total_days).sum()
    
    if incomplete_subjects > 0:
        print(f"\n[WARNING] {incomplete_subjects} animals have incomplete time series")
        print(f"  Complete: {complete_subjects}, Incomplete: {incomplete_subjects}")
    
    # Check CA% group sizes
    animals_per_ca = analysis_df.groupby('CA (%)')['ID'].nunique()
    print(f"\nAnimals per CA% group:")
    for ca, n in animals_per_ca.items():
        print(f"  {ca}%: {n} animals")
    
    if any(animals_per_ca < 2):
        print(f"\n[WARNING] Some CA% groups have fewer than 2 animals")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by CA% Group:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1)
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        return pd.Series({
            'count': n,
            'mean': mean,
            'median': group.median(),
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'q25': group.quantile(0.25),
            'q75': group.quantile(0.75),
            'min': group.min(),
            'max': group.max()
        })
    
    # Collect statistics for each CA% group
    stats_data = []
    for ca_val, group_data in analysis_df.groupby('CA (%)')[measure]:
        stats = compute_desc_stats(group_data)
        stats['CA (%)'] = ca_val
        stats_data.append(stats)
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  CA%={row['CA (%)']}: "
              f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
              f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
    
    # Perform mixed ANOVA
    print(f"\nStep 2: Running mixed ANOVA (Time within, CA% between)...")
    
    try:
        # Rename CA (%) column
        analysis_df_clean = analysis_df.rename(columns={'CA (%)': 'CA_percent'})
        
        # Run mixed ANOVA: Day (within)  x  CA% (between)
        aov = pg.mixed_anova(
            data=analysis_df_clean,
            dv=measure,
            within='Day',
            subject='ID',
            between='CA_percent',
            correction=True
        )
        
        # Rename back for clarity
        aov['Source'] = aov['Source'].replace({
            'CA_percent': 'CA (%)',
            'CA_percent * Day': 'CA (%) * Day',
            'Day * CA_percent': 'CA (%) * Day',
            'Interaction': 'CA (%) * Day'
        })
        
        print("\nMixed ANOVA Results:")
        print(aov.to_string())
        
        # Format results
        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        
        print("\n" + "="*80)
        print("FORMATTED RESULTS")
        print("="*80)
        
        # Time (Day) effect
        if 'Day' in aov['Source'].values:
            day_row = aov[aov['Source'] == 'Day'].iloc[0]
            day_p = day_row[p_col]
            day_F = day_row['F']
            day_sig = "***" if day_p < 0.001 else "**" if day_p < 0.01 else "*" if day_p < 0.05 else "ns"
            
            print(f"\n1. Time (Day) Effect: {'SIGNIFICANT' if day_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {day_F:.3f}, p = {day_p:.4f} {day_sig}")
            if day_p < 0.05:
                print(f"   -> Weight changes significantly over time for {sex}s")
            else:
                print(f"   -> No significant change over time for {sex}s")
        
        # CA% main effect
        if 'CA (%)' in aov['Source'].values:
            ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
            ca_p = ca_row[p_col]
            ca_F = ca_row['F']
            ca_sig = "***" if ca_p < 0.001 else "**" if ca_p < 0.01 else "*" if ca_p < 0.05 else "ns"
            
            print(f"\n2. CA% Main Effect: {'SIGNIFICANT' if ca_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {ca_F:.3f}, p = {ca_p:.4f} {ca_sig}")
            if ca_p < 0.05:
                print(f"   -> CA% concentrations differ for {sex}s")
            else:
                print(f"   -> No CA% difference for {sex}s")
        
        # Interaction effect
        if 'CA (%) * Day' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time  x  CA% Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   -> The time course differs between CA% levels for {sex}s")
            else:
                print(f"   -> Similar time course across CA% levels for {sex}s")
        
        results = {
            'measure': measure,
            'sex': sex,
            'type': 'mixed_anova_sex_stratified',
            'n_observations': len(analysis_df),
            'n_subjects': analysis_df['ID'].nunique(),
            'n_days': len(time_points),
            'time_points': time_points,
            'anova_table': aov,
            'descriptive_stats': group_stats,
            'data': analysis_df
        }
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def perform_mixed_anova_ca_stratified(
    cohort_dfs: Dict[str, pd.DataFrame],
    ca_percent: float,
    measure: str = "Total Change",
    time_points: Optional[List[int]] = None
) -> Dict:
    """
    Perform 2-Way Mixed ANOVA: Time (within)  x  Sex (between), holding CA% constant.
    
    This analyzes longitudinal weight changes for ONE CA% level at a time, testing:
    - Time (Day): Within-subjects factor (repeated measures over days)
    - Sex: Between-subjects factor (M vs F)
    
    By stratifying by CA%, this reveals whether the Time  x  Sex interaction differs
    between CA% conditions.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        ca_percent: CA% level to analyze (e.g., 0 or 2)
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_points: Optional list of specific days to include (None = all days)
        
    Returns:
        Dictionary with ANOVA results for the specified CA% level
    """
    print("\n" + "="*80)
    print(f"CA%-STRATIFIED MIXED ANOVA: TIME (WITHIN)  x  SEX (BETWEEN)")
    print(f"Analyzing: {ca_percent}% CA only")
    print("="*80)
    
    if not HAS_PINGOUIN:
        print("\n[ERROR] pingouin is required for mixed ANOVA")
        return {}
    
    # Combine cohorts
    print("\nStep 1: Combining and filtering data...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    
    # Filter to specified CA%
    required_cols = ['ID', 'Day', 'Sex', 'CA (%)', measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    analysis_df = combined_df[combined_df['CA (%)'] == ca_percent][required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) == 0:
        print(f"\n[ERROR] No data available for CA%={ca_percent}")
        return {}
    
    # Filter to specific time points if requested
    if time_points is not None:
        analysis_df = analysis_df[analysis_df['Day'].isin(time_points)]
        print(f"  Filtered to days: {sorted(time_points)}")
    else:
        time_points = sorted(analysis_df['Day'].unique())
    
    print(f"\nAnalyzing: {measure} ({ca_percent}% CA)")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  Sex groups: {sorted(analysis_df['Sex'].unique())}")
    print(f"  Days: {len(time_points)} days")
    
    # Check completeness
    subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
    total_days = analysis_df['Day'].nunique()
    complete_subjects = (subjects_per_day == total_days).sum()
    incomplete_subjects = (subjects_per_day < total_days).sum()
    
    if incomplete_subjects > 0:
        print(f"\n[WARNING] {incomplete_subjects} animals have incomplete time series")
        print(f"  Complete: {complete_subjects}, Incomplete: {incomplete_subjects}")
    
    # Check Sex group sizes
    animals_per_sex = analysis_df.groupby('Sex')['ID'].nunique()
    print(f"\nAnimals per Sex group:")
    for sex, n in animals_per_sex.items():
        print(f"  {sex}: {n} animals")
    
    if any(animals_per_sex < 2):
        print(f"\n[WARNING] Some Sex groups have fewer than 2 animals")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics by Sex Group:")
    
    def compute_desc_stats(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1)
        sem = std / np.sqrt(n) if n > 0 else np.nan
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        return pd.Series({
            'count': n,
            'mean': mean,
            'median': group.median(),
            'std': std,
            'sem': sem,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'q25': group.quantile(0.25),
            'q75': group.quantile(0.75),
            'min': group.min(),
            'max': group.max()
        })
    
    # Collect statistics for each Sex group
    stats_data = []
    for sex_val, group_data in analysis_df.groupby('Sex')[measure]:
        stats = compute_desc_stats(group_data)
        stats['Sex'] = sex_val
        stats_data.append(stats)
    group_stats = pd.DataFrame(stats_data)
    
    for _, row in group_stats.iterrows():
        print(f"  Sex={row['Sex']}: "
              f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
              f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
    
    # Perform mixed ANOVA
    print(f"\nStep 2: Running mixed ANOVA (Time within, Sex between)...")
    
    try:
        # Run mixed ANOVA: Day (within)  x  Sex (between)
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Day',
            subject='ID',
            between='Sex',
            correction=True
        )
        
        print("\nMixed ANOVA Results:")
        print(aov.to_string())
        
        # Rename interaction term for clarity
        aov['Source'] = aov['Source'].replace({
            'Interaction': 'Sex * Day',
            'Day * Sex': 'Sex * Day'
        })
        
        # Format results
        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        
        print("\n" + "="*80)
        print("FORMATTED RESULTS")
        print("="*80)
        
        # Time (Day) effect
        if 'Day' in aov['Source'].values:
            day_row = aov[aov['Source'] == 'Day'].iloc[0]
            day_p = day_row[p_col]
            day_F = day_row['F']
            day_sig = "***" if day_p < 0.001 else "**" if day_p < 0.01 else "*" if day_p < 0.05 else "ns"
            
            print(f"\n1. Time (Day) Effect: {'SIGNIFICANT' if day_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {day_F:.3f}, p = {day_p:.4f} {day_sig}")
            if day_p < 0.05:
                print(f"   -> Weight changes significantly over time at {ca_percent}% CA")
            else:
                print(f"   -> No significant change over time at {ca_percent}% CA")
        
        # Sex main effect
        if 'Sex' in aov['Source'].values:
            sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
            sex_p = sex_row[p_col]
            sex_F = sex_row['F']
            sex_sig = "***" if sex_p < 0.001 else "**" if sex_p < 0.01 else "*" if sex_p < 0.05 else "ns"
            
            print(f"\n2. Sex Main Effect: {'SIGNIFICANT' if sex_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {sex_F:.3f}, p = {sex_p:.4f} {sex_sig}")
            if sex_p < 0.05:
                print(f"   -> Males and females differ at {ca_percent}% CA")
            else:
                print(f"   -> No sex difference at {ca_percent}% CA")
        
        # Interaction effect
        if 'Sex * Day' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'Sex * Day'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time  x  Sex Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   -> The time course differs between sexes at {ca_percent}% CA")
            else:
                print(f"   -> Similar time course for both sexes at {ca_percent}% CA")
        elif 'Day * Sex' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'Day * Sex'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time  x  Sex Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   -> The time course differs between sexes at {ca_percent}% CA")
            else:
                print(f"   -> Similar time course for both sexes at {ca_percent}% CA")
        
        results = {
            'measure': measure,
            'ca_percent': ca_percent,
            'type': 'mixed_anova_ca_stratified',
            'n_observations': len(analysis_df),
            'n_subjects': analysis_df['ID'].nunique(),
            'n_days': len(time_points),
            'time_points': time_points,
            'anova_table': aov,
            'descriptive_stats': group_stats,
            'data': analysis_df
        }
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# PLOTTING HELPER FUNCTIONS
# =============================================================================

def _get_id_sex_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from ID -> Sex ("M" or "F") using the first non-null value per ID.
    """
    cdf = clean_cohort(df)
    if "ID" not in cdf.columns or "Sex" not in cdf.columns:
        return {}

    def _norm_sex(x: pd.Series) -> Optional[str]:
        valid = x.dropna()
        if valid.empty:
            return None
        val = str(valid.iloc[0]).strip().upper()
        if val in ["M", "MALE"]:
            return "M"
        if val in ["F", "FEMALE"]:
            return "F"
        return None

    sex_map = cdf.groupby("ID")["Sex"].apply(_norm_sex).to_dict()
    return {str(k): v for k, v in sex_map.items()}


def _get_id_ca_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from ID -> CA% using the first non-null value per ID.
    """
    cdf = clean_cohort(df)
    if "ID" not in cdf.columns or "CA (%)" not in cdf.columns:
        return {}

    ca_map = cdf.groupby("ID")["CA (%)"].first().to_dict()
    return {str(k): v for k, v in ca_map.items()}


def _get_id_cohort_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from ID -> Cohort label using the first non-null value per ID.
    """
    if "ID" not in df.columns or "Cohort" not in df.columns:
        return {}
    cdf = clean_cohort(df)
    cohort_map = cdf.groupby("ID")["Cohort"].first().to_dict()
    return {str(k): v for k, v in cohort_map.items()}


def _sex_to_style(sex: Optional[str]) -> Tuple[str, str]:
    """Return (color, marker) based on sex: M=green/square, F=purple/circle."""
    if sex == "M":
        return ("green", "s")
    if sex == "F":
        return ("purple", "o")
    return ("gray", "^")


def _ca_to_style(ca_pct: Optional[float]) -> Tuple[str, str]:
    """Return (color, marker) based on CA%: 0=#1f77b4/triangle, 2=#f79520/circle."""
    if ca_pct == 0.0:
        return (_COLOR_0PCT, "^")
    if ca_pct == 2.0:
        return (_COLOR_2PCT, "o")
    return ("gray", "d")  # diamond for unknown


def build_daily_change_series_by_id(df: pd.DataFrame) -> dict:
    """
    For each ID, return a pandas Series of 'Daily Change' indexed by Day number.
    """
    required = {"ID", "Day", "Daily Change"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cdf = clean_cohort(df)
    series_by_id: dict = {}
    
    for gid, g in cdf.groupby("ID", dropna=True):
        subset = g[["Day", "Daily Change"]].dropna()
        if subset.empty:
            continue
        subset = subset.sort_values("Day")
        series = subset.set_index("Day")["Daily Change"]
        series_by_id[str(gid)] = series

    return series_by_id


def build_total_change_series_by_id(df: pd.DataFrame) -> dict:
    """
    For each ID, return a pandas Series of 'Total Change' indexed by Day number.
    """
    required = {"ID", "Day", "Total Change"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cdf = clean_cohort(df)
    series_by_id: dict = {}
    
    for gid, g in cdf.groupby("ID", dropna=True):
        subset = g[["Day", "Total Change"]].dropna()
        if subset.empty:
            continue
        subset = subset.sort_values("Day")
        series = subset.set_index("Day")["Total Change"]
        series_by_id[str(gid)] = series

    return series_by_id


def apply_common_plot_style(
    ax: plt.Axes,
    start_x_at_zero: bool = False,
    remove_top_right: bool = True,
    remove_x_margins: bool = True,
    remove_y_margins: bool = True,
    ticks_in: bool = True,
    draw_zero_dotted_line: bool = True,
) -> plt.Axes:
    """Apply common styling to plots: remove spines, set tick directions, adjust margins."""
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if ticks_in:
        ax.tick_params(direction='in', which='both', length=5)

    if remove_x_margins:
        ax.margins(x=0)

    if remove_y_margins:
        ax.margins(y=0)
        ax.autoscale(axis='y', tight=True)

    if start_x_at_zero:
        left, right = ax.get_xlim()
        ax.set_xlim(left=0, right=right)

    if draw_zero_dotted_line:
        try:
            ax.axhline(0, linestyle='-', color='0', linewidth=1.0, alpha=0.8, zorder=1)
        except Exception:
            pass

    return ax


def _auto_integer_step(
    vmin: float,
    vmax: float,
    target_ticks: int = 7,
    allow_sub5: bool = False,
) -> int:
    """Choose a 'nice' integer step so about target_ticks cover the range."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return 1
    range_int = int(abs(math.ceil(vmax) - math.floor(vmin)))
    if range_int <= 0:
        return 1
    approx = max(1.0, range_int / max(1, target_ticks))
    pow10 = 10 ** int(math.floor(math.log10(approx)))
    multipliers = (1, 2, 2.5, 3, 4, 5) if allow_sub5 else (1, 2, 5)
    for m in multipliers:
        step = int(max(1, math.ceil(m * pow10)))
        if range_int / step <= target_ticks:
            return step
    return int(max(1, 10 * pow10))


def _apply_integer_axis(
    ax: plt.Axes,
    *,
    axis: str,
    data_min: float,
    data_max: float,
    step: int,
    clamp_min: Optional[int] = None,
    left_pad_steps: int = 0,
    right_pad_steps: int = 1,
) -> None:
    """Apply integer ticks and limits to the chosen axis with one extra step beyond data."""
    step = int(max(1, step))
    base_start = int(math.floor(data_min / step) * step)
    base_end_tick = int(math.ceil(data_max / step) * step)
    tick_start = base_start - left_pad_steps * step
    tick_end = base_end_tick + right_pad_steps * step
    start = tick_start
    if clamp_min is not None and start < clamp_min:
        start = clamp_min
    end = int(data_max) + right_pad_steps * step
    if end <= start:
        end = start + step
    all_ticks = list(range(tick_start, tick_end + 1, step))
    ticks = [t for t in all_ticks if start <= t <= end]
    if axis == 'x':
        ax.set_xlim(start, end)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    elif axis == 'y':
        ax.set_ylim(start, end)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_total_change_by_id(
    df: pd.DataFrame,
    ids: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a plot showing Total Change over Day for each animal (ID),
    with colors and markers indicating sex (M=green/square, F=purple/circle).
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_total_change_series_by_id(df)
    sex_map = _get_id_sex_map(df)

    # Filter to requested IDs if provided
    if ids is not None:
        series_by_id = {k: v for k, v in series_by_id.items() if k in ids}

    if not series_by_id:
        print("[WARNING] No data to plot")
        return None

    fig, ax = plt.subplots()

    # Plot each ID as a separate line
    for mid, s in series_by_id.items():
        color, marker = _sex_to_style(sex_map.get(mid))
        ax.plot(
            s.index,
            s.values,
            label=str(mid),
            marker=marker,
            linewidth=1.0,
            alpha=0.9,
            color=color,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change")
    ax.grid(False)

    if title is None:
        title = "Total Change by Animal ID"
    ax.set_title(title)

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
    )

    # Integer axis ticks
    all_y = np.concatenate([s.values for s in series_by_id.values() if len(s) > 0])
    y_data_min = float(np.nanmin(all_y)) if all_y.size else 0.0
    y_data_max = float(np.nanmax(all_y)) if all_y.size else 1.0
    all_x = np.concatenate([np.asarray(s.index, dtype=float) for s in series_by_id.values() if len(s) > 0])
    x_data_min = int(np.nanmin(all_x)) if all_x.size else 0
    x_data_max = int(np.nanmax(all_x)) if all_x.size else 1
    x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
    _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                        step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    # Legend placement
    if len(series_by_id) > 6:
        ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax.legend(title="ID", loc="best")
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_daily_change_by_id(
    df: pd.DataFrame,
    ids: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a plot showing Daily Change over Day for each animal (ID),
    with colors and markers indicating sex (M=green/square, F=purple/circle).
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_daily_change_series_by_id(df)
    sex_map = _get_id_sex_map(df)

    # Filter to requested IDs if provided
    if ids is not None:
        series_by_id = {k: v for k, v in series_by_id.items() if k in ids}

    if not series_by_id:
        print("[WARNING] No data to plot")
        return None

    fig, ax = plt.subplots()

    # Plot each ID as a separate line
    for mid, s in series_by_id.items():
        color, marker = _sex_to_style(sex_map.get(mid))
        ax.plot(
            s.index,
            s.values,
            label=str(mid),
            marker=marker,
            linewidth=1.0,
            alpha=0.9,
            color=color,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change")
    ax.grid(False)

    if title is None:
        title = "Daily Change by Animal ID"
    ax.set_title(title)

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
    )

    # Integer axis ticks
    all_y = np.concatenate([s.values for s in series_by_id.values() if len(s) > 0])
    y_data_min = float(np.nanmin(all_y)) if all_y.size else 0.0
    y_data_max = float(np.nanmax(all_y)) if all_y.size else 1.0
    all_x = np.concatenate([np.asarray(s.index, dtype=float) for s in series_by_id.values() if len(s) > 0])
    x_data_min = int(np.nanmin(all_x)) if all_x.size else 0
    x_data_max = int(np.nanmax(all_x)) if all_x.size else 1
    x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
    _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                        step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    # Legend placement
    if len(series_by_id) > 6:
        ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        ax.legend(title="ID", loc="best")
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_total_change_by_sex(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sex-averaged Total Change with SEM error bars.
    For each sex (M/F), computes mean and SEM across all animals at each day.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_total_change_series_by_id(df)
    sex_map = _get_id_sex_map(df)
    
    # Separate series by sex
    male_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "M"}
    female_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "F"}
    
    # Compute mean and SEM for each sex
    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        mean = df_temp.mean(axis=1)
        sem = df_temp.sem(axis=1)
        return mean, sem
    
    male_mean, male_sem = _compute_mean_sem(male_series)
    female_mean, female_sem = _compute_mean_sem(female_series)
    
    fig, ax = plt.subplots()
    
    # Plot male data
    if not male_mean.empty:
        ax.plot(male_mean.index, male_mean.values, label="Male", color="green", marker="s",
linewidth=0.9, alpha=0.9)
        ax.fill_between(male_mean.index,
                        male_mean - male_sem,
                        male_mean + male_sem,
                        color="green", alpha=0.2)

    # Plot female data
    if not female_mean.empty:
        ax.plot(female_mean.index, female_mean.values, label="Female", color="purple", marker="o",
linewidth=0.9, alpha=0.9)
        ax.fill_between(female_mean.index,
                        female_mean - female_sem,
                        female_mean + female_sem,
                        color="purple", alpha=0.2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change (Mean +/- SEM)")
    ax.grid(False)

    if title is None:
        title = "Total Change by Sex (Mean +/- SEM)"
    ax.set_title(title)

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    # Integer axis ticks
    all_means = pd.concat([male_mean, female_mean], ignore_index=False)
    if not all_means.empty:
        y_data_min = float(all_means.min())
        y_data_max = float(all_means.max())
    else:
        y_data_min, y_data_max = 0.0, 1.0
    all_indices = pd.concat([male_mean.index.to_series(), female_mean.index.to_series()])
    if not all_indices.empty:
        x_data_min = int(all_indices.min())
        x_data_max = int(all_indices.max())
    else:
        x_data_min, x_data_max = 0, 1
    x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
    _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                        step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="Sex", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_daily_change_by_sex(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sex-averaged Daily Change with SEM error bars.
    For each sex (M/F), computes mean and SEM across all animals at each day.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_daily_change_series_by_id(df)
    sex_map = _get_id_sex_map(df)
    
    # Separate series by sex
    male_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "M"}
    female_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "F"}
    
    # Compute mean and SEM for each sex
    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        mean = df_temp.mean(axis=1)
        sem = df_temp.sem(axis=1)
        return mean, sem
    
    male_mean, male_sem = _compute_mean_sem(male_series)
    female_mean, female_sem = _compute_mean_sem(female_series)
    
    fig, ax = plt.subplots()
    
    # Plot male data
    if not male_mean.empty:
        ax.plot(male_mean.index, male_mean.values, label="Male", color="green", marker="s",
linewidth=0.9, alpha=0.9)
        ax.fill_between(male_mean.index,
                        male_mean - male_sem,
                        male_mean + male_sem,
                        color="green", alpha=0.2)

    # Plot female data
    if not female_mean.empty:
        ax.plot(female_mean.index, female_mean.values, label="Female", color="purple", marker="o",
linewidth=0.9, alpha=0.9)
        ax.fill_between(female_mean.index,
                        female_mean - female_sem,
                        female_mean + female_sem,
                        color="purple", alpha=0.2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change (Mean +/- SEM)")
    ax.grid(False)

    if title is None:
        title = "Daily Change by Sex (Mean +/- SEM)"
    ax.set_title(title)

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    # Integer axis ticks
    all_means = pd.concat([male_mean, female_mean], ignore_index=False)
    if not all_means.empty:
        y_data_min = float(all_means.min())
        y_data_max = float(all_means.max())
    else:
        y_data_min, y_data_max = 0.0, 1.0
    all_indices = pd.concat([male_mean.index.to_series(), female_mean.index.to_series()])
    if not all_indices.empty:
        x_data_min = int(all_indices.min())
        x_data_max = int(all_indices.max())
    else:
        x_data_min, x_data_max = 0, 1
    x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
    _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                        step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="Sex", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_total_change_by_ca(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot CA%-averaged Total Change with SEM error bars.
    For each CA% level (0%, 2%), computes mean and SEM across all animals at each day.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_total_change_series_by_id(df)
    ca_map = _get_id_ca_map(df)
    
    # Separate series by CA%
    ca_groups = {}
    for mid, s in series_by_id.items():
        ca_val = ca_map.get(mid)
        if ca_val is not None:
            if ca_val not in ca_groups:
                ca_groups[ca_val] = {}
            ca_groups[ca_val][mid] = s
    
    # Compute mean and SEM for each CA% group
    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        mean = df_temp.mean(axis=1)
        sem = df_temp.sem(axis=1)
        return mean, sem
    
    fig, ax = plt.subplots()
    
    # Plot each CA% group
    all_group_means = []
    all_group_indices = []
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=f'{ca_val:.0f}% CA',
                    marker=marker, linewidth=0.9, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change (Mean +/- SEM)")
    ax.grid(False)

    if title is None:
        title = "Total Change by CA%"
    ax.set_title(title)

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    # Integer axis ticks
    if all_group_means:
        all_means_concat = pd.concat(all_group_means, ignore_index=False)
        y_data_min = float(all_means_concat.min())
        y_data_max = float(all_means_concat.max())
        all_idx_concat = pd.concat(all_group_indices)
        x_data_min = int(all_idx_concat.min())
        x_data_max = int(all_idx_concat.max())
        x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
        _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                            step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
        _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="CA%", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_daily_change_by_ca(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot CA%-averaged Daily Change with SEM error bars.
    For each CA% level (0%, 2%), computes mean and SEM across all animals at each day.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    series_by_id = build_daily_change_series_by_id(df)
    ca_map = _get_id_ca_map(df)
    
    # Separate series by CA%
    ca_groups = {}
    for mid, s in series_by_id.items():
        ca_val = ca_map.get(mid)
        if ca_val is not None:
            if ca_val not in ca_groups:
                ca_groups[ca_val] = {}
            ca_groups[ca_val][mid] = s
    
    # Compute mean and SEM for each CA% group
    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        mean = df_temp.mean(axis=1)
        sem = df_temp.sem(axis=1)
        return mean, sem
    
    fig, ax = plt.subplots()
    
    # Plot each CA% group
    all_group_means = []
    all_group_indices = []
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=f'{ca_val:.0f}% CA',
                    marker=marker, linewidth=0.9, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change (Mean +/- SEM)")
    ax.grid(False)

    if title is None:
        title = "Daily Change by CA%"
    ax.set_title(title)

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    # Integer axis ticks
    if all_group_means:
        all_means_concat = pd.concat(all_group_means, ignore_index=False)
        y_data_min = float(all_means_concat.min())
        y_data_max = float(all_means_concat.max())
        all_idx_concat = pd.concat(all_group_indices)
        x_data_min = int(all_idx_concat.min())
        x_data_max = int(all_idx_concat.max())
        x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
        _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                            step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
        _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="CA%", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_total_change_by_cohort(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Cohort-averaged Total Change with SEM error bands.
    One line per cohort label, grouped by the 'Cohort' column.
    CA%-agnostic � suitable for comparing 0% nonramp vs ramp where CA% varies over time.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None

    series_by_id = build_total_change_series_by_id(df)
    cohort_map = _get_id_cohort_map(df)

    cohort_groups: dict = {}
    for mid, s in series_by_id.items():
        cohort_label = cohort_map.get(mid)
        if cohort_label is not None:
            cohort_groups.setdefault(cohort_label, {})[mid] = s

    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        return df_temp.mean(axis=1), df_temp.sem(axis=1)

    _MARKERS = ['o', 's', '^', 'D', 'v']
    sorted_cohorts = sorted(cohort_groups.keys())
    color_map  = {c: _cohort_label_to_color(c) for c in sorted_cohorts}
    marker_map = {c: _MARKERS[i % len(_MARKERS)] for i, c in enumerate(sorted_cohorts)}

    fig, ax = plt.subplots()

    all_group_means = []
    all_group_indices = []
    for cohort_label in sorted_cohorts:
        color  = color_map[cohort_label]
        marker = marker_map[cohort_label]
        mean, sem = _compute_mean_sem(cohort_groups[cohort_label])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=cohort_label,
                    marker=marker, linewidth=0.9, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change (Mean +/- SEM)")
    ax.grid(False)
    ax.set_title(title or "Total Change by Cohort")

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    if all_group_means:
        all_means_concat = pd.concat(all_group_means, ignore_index=False)
        y_data_min = float(all_means_concat.min())
        y_data_max = float(all_means_concat.max())
        all_idx_concat = pd.concat(all_group_indices)
        x_data_min = int(all_idx_concat.min())
        x_data_max = int(all_idx_concat.max())
        x_step = _auto_integer_step(x_data_min, x_data_max, target_ticks=10, allow_sub5=True)
        _apply_integer_axis(ax, axis='x', data_min=x_data_min, data_max=x_data_max,
                            step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
        _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="Cohort", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_daily_change_by_cohort(
    df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Cohort-averaged Daily Change with SEM error bands.
    One line per cohort label, grouped by the 'Cohort' column.
    CA%-agnostic � suitable for comparing 0% nonramp vs ramp where CA% varies over time.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None

    series_by_id = build_daily_change_series_by_id(df)
    cohort_map = _get_id_cohort_map(df)

    cohort_groups: dict = {}
    for mid, s in series_by_id.items():
        cohort_label = cohort_map.get(mid)
        if cohort_label is not None:
            cohort_groups.setdefault(cohort_label, {})[mid] = s

    def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
        if not series_dict:
            return pd.Series(), pd.Series()
        df_temp = pd.DataFrame(series_dict)
        return df_temp.mean(axis=1), df_temp.sem(axis=1)

    _MARKERS = ['o', 's', '^', 'D', 'v']
    sorted_cohorts = sorted(cohort_groups.keys())
    color_map  = {c: _cohort_label_to_color(c) for c in sorted_cohorts}
    marker_map = {c: _MARKERS[i % len(_MARKERS)] for i, c in enumerate(sorted_cohorts)}

    fig, ax = plt.subplots()

    all_group_means = []
    all_group_indices = []
    for cohort_label in sorted_cohorts:
        color  = color_map[cohort_label]
        marker = marker_map[cohort_label]
        mean, sem = _compute_mean_sem(cohort_groups[cohort_label])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=cohort_label,
                    marker=marker, linewidth=0.9, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change (Mean +/- SEM)")
    ax.grid(False)
    ax.set_title(title or "Daily Change by Cohort")

    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)

    if all_group_means:
        all_means_concat = pd.concat(all_group_means, ignore_index=False)
        y_data_min = float(all_means_concat.min())
        y_data_max = float(all_means_concat.max())
        all_idx_concat = pd.concat(all_group_indices)
        x_data_min = int(all_idx_concat.min())
        x_data_max = int(all_idx_concat.max())
        x_step = _auto_integer_step(1, 36, target_ticks=10, allow_sub5=True)
        _apply_integer_axis(ax, axis='x', data_min=1, data_max=36,
                            step=x_step, clamp_min=1, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(-15, 5, target_ticks=6)
        _apply_integer_axis(ax, axis='y', data_min=-15, data_max=5,
                            step=y_step, left_pad_steps=0, right_pad_steps=0)

    ax.legend(title="Cohort", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# INTERACTION PLOTTING FUNCTIONS
# =============================================================================

def plot_interaction_ca_sex(
    anova_results: Dict,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot CA%  x  Sex interaction from between-subjects ANOVA.
    
    Shows mean values for each CA%  x  Sex combination with error bars (SEM).
    Only creates plot if the interaction is significant (p < 0.05).
    
    Parameters:
        anova_results: Results dict from perform_between_subjects_anova()
        title: Optional custom title
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object, or None if interaction not significant
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    # Check if CA%  x  Sex interaction is significant
    anova_table = anova_results.get('anova_table')
    if anova_table is None:
        print("[WARNING] No ANOVA table found in results")
        return None
    
    p_col = 'p-unc' if 'p-unc' in anova_table.columns else 'p_unc'
    interaction_row = anova_table[anova_table['Source'] == 'CA (%) * Sex']
    
    if len(interaction_row) == 0:
        print(f"[WARNING] CA%  x  Sex interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] CA%  x  Sex interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting CA%  x  Sex interaction (p = {p_value:.4f})")
    
    # Get descriptive statistics
    group_stats = anova_results.get('descriptive_stats')
    if group_stats is None or len(group_stats) == 0:
        print("[WARNING] No descriptive statistics found")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Get unique CA% levels and Sex levels
    ca_levels = sorted(group_stats['CA (%)'].unique())
    sex_levels = sorted(group_stats['Sex'].unique())
    
    # Plot separate line for each sex
    for sex in sex_levels:
        sex_data = group_stats[group_stats['Sex'] == sex].sort_values('CA (%)')
        
        color = 'green' if sex == 'M' else 'purple'
        marker = 'o' if sex == 'M' else 's'
        label = 'Males' if sex == 'M' else 'Females'
        
        ax.errorbar(
            sex_data['CA (%)'], 
            sex_data['mean'],
            yerr=sex_data['sem'],
            marker=marker, linewidth=0.9, capsize=5,
            color=color, label=label, alpha=0.85, markeredgewidth=1.5,
            markeredgecolor='black'
        )
    
    ax.set_xlabel('CA% Concentration')
    ax.set_ylabel(f'{measure} (Mean +/- SEM)')
    
    if title is None:
        analysis_type = anova_results.get('analysis_type', '')
        title = f'CA%  x  Sex Interaction: {measure}\n({analysis_type})'
    
    ax.set_title(title, pad=15)
    
    ax.grid(False)
    apply_common_plot_style(ax, remove_top_right=True, ticks_in=True, remove_x_margins=False, remove_y_margins=True)
    ax.legend(title='Sex', loc='best')
    
    # Add significance annotation
    ax.text(0.02, 0.98, f'Interaction: p = {p_value:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"[OK] Saved interaction plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_interaction_time_ca(
    anova_results: Dict,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot Time  x  CA% interaction from sex-stratified mixed ANOVA.
    
    Shows trajectories over time for each CA% level with error bars (SEM).
    Only creates plot if the interaction is significant (p < 0.05).
    
    Parameters:
        anova_results: Results dict from perform_mixed_anova_sex_stratified()
        title: Optional custom title
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object, or None if interaction not significant
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    # Check if Time  x  CA% interaction is significant
    anova_table = anova_results.get('anova_table')
    if anova_table is None:
        print("[WARNING] No ANOVA table found in results")
        return None
    
    p_col = 'p-unc' if 'p-unc' in anova_table.columns else 'p_unc'
    interaction_row = anova_table[anova_table['Source'] == 'CA (%) * Day']
    
    if len(interaction_row) == 0:
        print(f"[WARNING] CA%  x  Day interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] Time  x  CA% interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting Time  x  CA% interaction (p = {p_value:.4f})")
    
    # Get data
    data_df = anova_results.get('data')
    if data_df is None or len(data_df) == 0:
        print("[WARNING] No data found in results")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    sex = anova_results.get('sex', 'Unknown')
    sex_label = 'Males' if sex == 'M' else 'Females'
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Compute mean and SEM for each Day  x  CA% combination
    summary = data_df.groupby(['Day', 'CA (%)'])[measure].agg(['mean', 'sem', 'count']).reset_index()
    
    # Get unique CA% levels
    ca_levels = sorted(summary['CA (%)'].unique())
    
    # Plot trajectory for each CA% level
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ca_levels)))
    
    for i, ca_val in enumerate(ca_levels):
        ca_data = summary[summary['CA (%)'] == ca_val].sort_values('Day')
        
        ax.plot(
            ca_data['Day'], ca_data['mean'],
            marker='o', linewidth=0.9, alpha=0.9,
            color=colors[i], label=f'{ca_val:.0f}% CA'
        )
        ax.fill_between(
            ca_data['Day'],
            ca_data['mean'] - ca_data['sem'],
            ca_data['mean'] + ca_data['sem'],
            color=colors[i], alpha=0.2
        )
    
    ax.set_xlabel('Day')
    ax.set_ylabel(f'{measure} (Mean +/- SEM)')
    
    if title is None:
        title = f'Time  x  CA% Interaction: {measure}\n({sex_label} Only)'
    
    ax.set_title(title, pad=15)
    
    ax.grid(False)
    apply_common_plot_style(ax, remove_top_right=True, ticks_in=True, remove_x_margins=False, remove_y_margins=True)
    ax.legend(title='CA% Level', loc='best')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Add significance annotation
    ax.text(0.02, 0.98, f'Interaction: p = {p_value:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"[OK] Saved interaction plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_interaction_time_sex(
    anova_results: Dict,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot Time  x  Sex interaction from CA%-stratified mixed ANOVA.
    
    Shows trajectories over time for Males vs Females with error bars (SEM).
    Only creates plot if the interaction is significant (p < 0.05).
    
    Parameters:
        anova_results: Results dict from perform_mixed_anova_ca_stratified()
        title: Optional custom title
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object, or None if interaction not significant
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    # Check if Time  x  Sex interaction is significant
    anova_table = anova_results.get('anova_table')
    if anova_table is None:
        print("[WARNING] No ANOVA table found in results")
        return None
    
    p_col = 'p-unc' if 'p-unc' in anova_table.columns else 'p_unc'
    interaction_row = anova_table[anova_table['Source'] == 'Sex * Day']
    
    if len(interaction_row) == 0:
        # Try alternate name
        interaction_row = anova_table[anova_table['Source'] == 'Day * Sex']
    
    if len(interaction_row) == 0:
        print(f"[WARNING] Sex  x  Day interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] Time  x  Sex interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting Time  x  Sex interaction (p = {p_value:.4f})")
    
    # Get data
    data_df = anova_results.get('data')
    if data_df is None or len(data_df) == 0:
        print("[WARNING] No data found in results")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    ca_percent = anova_results.get('ca_percent', 'Unknown')
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Compute mean and SEM for each Day  x  Sex combination
    summary = data_df.groupby(['Day', 'Sex'])[measure].agg(['mean', 'sem', 'count']).reset_index()
    
    # Get unique Sex levels
    sex_levels = sorted(summary['Sex'].unique())
    
    # Plot trajectory for each sex
    for sex in sex_levels:
        sex_data = summary[summary['Sex'] == sex].sort_values('Day')
        
        color = 'green' if sex == 'M' else 'purple'
        label = 'Males' if sex == 'M' else 'Females'
        marker = 'o' if sex == 'M' else 's'
        
        ax.plot(
            sex_data['Day'], sex_data['mean'],
            marker=marker, linewidth=0.9, alpha=0.9,
            color=color, label=label
        )
        ax.fill_between(
            sex_data['Day'],
            sex_data['mean'] - sex_data['sem'],
            sex_data['mean'] + sex_data['sem'],
            color=color, alpha=0.2
        )
    
    ax.set_xlabel('Day')
    ax.set_ylabel(f'{measure} (Mean +/- SEM)')
    
    if title is None:
        title = f'Time  x  Sex Interaction: {measure}\n({ca_percent}% CA Only)'
    
    ax.set_title(title, pad=15)
    
    ax.grid(False)
    apply_common_plot_style(ax, remove_top_right=True, ticks_in=True, remove_x_margins=False, remove_y_margins=True)
    ax.legend(title='Sex', loc='best')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Add significance annotation
    ax.text(0.02, 0.98, f'Interaction: p = {p_value:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"[OK] Saved interaction plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_three_way_interaction(
    anova_results: Dict,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot CA%  x  Sex  x  Time three-way interaction from full mixed ANOVA.
    
    Creates a 1 x 2 panel plot showing Time  x  CA% trajectories separately for
    Males (left) and Females (right) with error bars (SEM).
    
    Only creates plot if the three-way interaction is significant OR if both
    two-way interactions (Time  x  CA% and Time  x  Sex) are significant.
    
    Parameters:
        anova_results: Results dict from perform_cross_cohort_mixed_anova()
        title: Optional custom title
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object, or None if interaction not significant
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None
    
    # Get data
    data_df = anova_results.get('data')
    if data_df is None or len(data_df) == 0:
        print("[WARNING] No data found in results")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    
    # Check for significant interactions
    mixed_anova = anova_results.get('mixed_anova_table')
    between_anova = anova_results.get('between_anova_table')
    
    plot_three_way = False
    
    # Check if Day  x  Group interaction is significant (proxy for three-way)
    if mixed_anova is not None:
        p_col = 'p-unc' if 'p-unc' in mixed_anova.columns else 'p_unc'
        interaction_row = mixed_anova[mixed_anova['Source'] == 'Interaction']
        
        if len(interaction_row) > 0:
            p_value_time_group = interaction_row.iloc[0][p_col]
            if p_value_time_group < 0.05:
                print(f"\n[INFO] Time  x  Group interaction significant (p = {p_value_time_group:.4f})")
                plot_three_way = True
    
    # Check if CA%  x  Sex interaction is significant
    if between_anova is not None and not plot_three_way:
        p_col = 'p-unc' if 'p-unc' in between_anova.columns else 'p_unc'
        ca_sex_row = between_anova[between_anova['Source'] == 'CA (%) * Sex']
        
        if len(ca_sex_row) > 0:
            p_value_ca_sex = ca_sex_row.iloc[0][p_col]
            if p_value_ca_sex < 0.05:
                print(f"\n[INFO] CA%  x  Sex interaction significant (p = {p_value_ca_sex:.4f})")
                plot_three_way = True
    
    if not plot_three_way:
        print("[INFO] No significant three-way or relevant two-way interactions. Skipping plot.")
        return None
    
    print("\n[INFO] Creating three-way interaction plot (Time  x  CA%  x  Sex)")
    
    # Create 1 x 2 panel figure
    fig, axes = plt.subplots(1, 2, sharey=True)
    
    # Get unique CA% levels
    ca_levels = sorted(data_df['CA (%)'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ca_levels)))
    
    for sex_idx, sex in enumerate(['M', 'F']):
        ax = axes[sex_idx]
        sex_data = data_df[data_df['Sex'] == sex]
        
        if len(sex_data) == 0:
            continue
        
        # Compute mean and SEM for each Day  x  CA% combination
        summary = sex_data.groupby(['Day', 'CA (%)'])[measure].agg(['mean', 'sem', 'count']).reset_index()
        
        # Plot trajectory for each CA% level
        for i, ca_val in enumerate(ca_levels):
            ca_data = summary[summary['CA (%)'] == ca_val].sort_values('Day')
            
            ax.plot(
                ca_data['Day'], ca_data['mean'],
                marker='o', linewidth=0.9, alpha=0.9,
                color=colors[i], label=f'{ca_val:.0f}% CA'
            )
            ax.fill_between(
                ca_data['Day'],
                ca_data['mean'] - ca_data['sem'],
                ca_data['mean'] + ca_data['sem'],
                color=colors[i], alpha=0.2
            )
        
        ax.set_xlabel('Day')
        
        if sex_idx == 0:
            ax.set_ylabel(f'{measure} (Mean +/- SEM)')
        
        sex_label = 'Males' if sex == 'M' else 'Females'
        ax.set_title(sex_label, pad=10)
        
        ax.grid(False)
        apply_common_plot_style(ax, remove_top_right=True, ticks_in=True, remove_x_margins=False, remove_y_margins=True)
        ax.legend(title='CA% Level', loc='best')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = f'Three-Way Interaction: Time  x  CA%  x  Sex\n{measure}'
    
    fig.suptitle(title, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"[OK] Saved three-way interaction plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_all_significant_interactions(
    between_results: Optional[Dict] = None,
    mixed_results: Optional[Dict] = None,
    sex_stratified_results: Optional[Dict] = None,
    ca_stratified_results: Optional[Dict] = None,
    save_dir: Optional[Path] = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Convenience function to plot all significant interactions from various analyses.
    
    Automatically detects which interactions are significant and creates
    appropriate plots for each.
    
    Parameters:
        between_results: Results from perform_between_subjects_anova()
        mixed_results: Results from perform_cross_cohort_mixed_anova()
        sex_stratified_results: Dict with 'males' and 'females' keys from 
                               perform_mixed_anova_sex_stratified()
        ca_stratified_results: Dict with CA% keys from perform_mixed_anova_ca_stratified()
        save_dir: Optional directory to save all plots
        show: Whether to display plots
        
    Returns:
        Dictionary mapping plot names to Figure objects
    """
    figures = {}
    
    print("\n" + "="*80)
    print("PLOTTING ALL SIGNIFICANT INTERACTIONS")
    print("="*80)
    
    # 1. CA%  x  Sex interaction (between-subjects)
    if between_results is not None:
        print("\n[DEBUG] Checking CA%  x  Sex interaction...")
        measure = between_results.get('measure', 'measure')
        analysis_type = between_results.get('analysis_type', 'analysis')
        
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"interaction_CA_Sex_{measure}_{analysis_type}.svg"
        
        fig = plot_interaction_ca_sex(between_results, save_path=save_path, show=show)
        if fig is not None:
            figures['CA_Sex'] = fig
            print(f"  OK CA%  x  Sex plot created")
        else:
            print(f"  x CA%  x  Sex interaction not significant or missing data")
    
    # 2. Three-way interaction (full mixed ANOVA)
    if mixed_results is not None:
        print("\n[DEBUG] Checking three-way interaction...")
        measure = mixed_results.get('measure', 'measure')
        
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"interaction_Time_CA_Sex_{measure}.svg"
        
        fig = plot_three_way_interaction(mixed_results, save_path=save_path, show=show)
        if fig is not None:
            figures['Time_CA_Sex'] = fig
            print(f"  OK Three-way interaction plot created")
        else:
            print(f"  x Three-way interaction not significant or missing data")
    
    # 3. Time  x  CA% interactions (sex-stratified)
    if sex_stratified_results is not None:
        print("\n[DEBUG] Checking sex-stratified Time  x  CA% interactions...")
        for sex_key in ['males', 'females']:
            sex_results = sex_stratified_results.get(sex_key)
            if sex_results is not None:
                measure = sex_results.get('measure', 'measure')
                sex = sex_results.get('sex', sex_key[0].upper())
                sex_label = 'Males' if sex == 'M' else 'Females'
                
                print(f"  Checking {sex_label}...")
                save_path = None
                if save_dir is not None:
                    save_path = Path(save_dir) / f"interaction_Time_CA_{sex_label}_{measure}.svg"
                
                fig = plot_interaction_time_ca(sex_results, save_path=save_path, show=show)
                if fig is not None:
                    figures[f'Time_CA_{sex_label}'] = fig
                    print(f"    OK Time  x  CA% ({sex_label}) plot created")
                else:
                    print(f"    x Time  x  CA% ({sex_label}) interaction not significant or missing data")
            else:
                print(f"  x {sex_key} results are None")
    
    # 4. Time  x  Sex interactions (CA%-stratified)
    if ca_stratified_results is not None:
        print("\n[DEBUG] Checking CA%-stratified Time  x  Sex interactions...")
        for ca_key, ca_results in ca_stratified_results.items():
            if isinstance(ca_results, dict):
                measure = ca_results.get('measure', 'measure')
                ca_percent = ca_results.get('ca_percent', ca_key)
                
                print(f"  Checking CA {ca_percent}%...")
                save_path = None
                if save_dir is not None:
                    save_path = Path(save_dir) / f"interaction_Time_Sex_CA{ca_percent}_{measure}.svg"
                
                fig = plot_interaction_time_sex(ca_results, save_path=save_path, show=show)
                if fig is not None:
                    figures[f'Time_Sex_CA{ca_percent}'] = fig
                    print(f"    OK Time  x  Sex (CA {ca_percent}%) plot created")
                else:
                    print(f"    x Time  x  Sex (CA {ca_percent}%) interaction not significant or missing data")
            else:
                print(f"  x Results for CA {ca_key}% are not a dict: {type(ca_results)}")
    
    # Summary
    print("\n" + "="*80)
    print(f"INTERACTION PLOT SUMMARY: Created {len(figures)} plot(s)")
    print("="*80)
    
    if len(figures) > 0:
        for plot_name in figures.keys():
            print(f"  OK {plot_name}")
    else:
        print("  No significant interactions found to plot")
    
    return figures


def plot_behavioral_metrics_by_cohort(
    cohort_dfs: Dict[str, pd.DataFrame],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    3-panel line plot comparing behavioral metric prevalence across cohorts over weeks.

    One panel per metric: No Nest (Nest Made? == No), Anxious (Anxious Behaviors? == Yes),
    Lethargy (Lethargy? == Yes).  Each cohort is a separate colored line.
    Y-axis: % of observations (0-100%).  X-axis: Week (1-indexed).
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available -- cannot generate behavioral plot")
        return None

    _COLORS = ['#1f77b4', '#f79520', '#2da048', '#984EA3', '#FF7F00']

    # (column, aberrant_value, panel_title)
    BEHAVIORS = [
        ('Nest Made?',         False, 'No Nest'),
        ('Anxious Behaviors?', True,  'Anxious'),
        ('Lethargy?',          True,  'Lethargy'),
    ]

    def _to_bool(series: pd.Series) -> pd.Series:
        """Coerce Yes/No/True/False strings and booleans to bool, else None."""
        _T = {'yes', 'true', '1'}
        _F = {'no', 'false', '0'}
        def _cv(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                ls = v.strip().lower()
                if ls in _T:
                    return True
                if ls in _F:
                    return False
            return None
        return series.map(_cv)

    # Build per-cohort weekly percentages
    cohort_data: Dict[str, Dict] = {}
    all_weeks_set: set = set()

    for label, df in cohort_dfs.items():
        cdf = clean_cohort(df.copy())
        if 'Date' in cdf.columns:
            cdf['Date'] = pd.to_datetime(cdf['Date'], errors='coerce')
        cdf = cdf.sort_values(['ID', 'Date']).reset_index(drop=True)
        first_dates = cdf.groupby('ID')['Date'].transform('min')
        _day_offset = 1 if 'ramp' in label.lower() else 0
        cdf['_Day'] = (cdf['Date'] - first_dates).dt.days + _day_offset
        cdf = cdf[cdf['_Day'] >= 1].copy()
        cdf['_Week'] = (cdf['_Day'] - 1) // 7 + 1

        weeks = sorted(cdf['_Week'].dropna().unique().astype(int))
        all_weeks_set.update(weeks)

        pcts_by_col: Dict[str, List[float]] = {}
        sems_by_col: Dict[str, List[float]] = {}
        for col, aberrant_val, _ in BEHAVIORS:
            pcts = []
            sems = []
            for w in weeks:
                grp = cdf[cdf['_Week'] == w]
                if col not in grp.columns or len(grp) == 0:
                    pcts.append(float('nan'))
                    sems.append(float('nan'))
                    continue
                # Per-animal proportion, then mean across animals
                animal_pcts = []
                for _, animal_data in grp.groupby('ID'):
                    valid = _to_bool(animal_data[col]).dropna()
                    if len(valid) > 0:
                        animal_pcts.append(100.0 * (valid == aberrant_val).sum() / len(valid))
                if animal_pcts:
                    arr = np.array(animal_pcts)
                    pcts.append(float(np.mean(arr)))
                    sems.append(float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
                else:
                    pcts.append(float('nan'))
                    sems.append(float('nan'))
            pcts_by_col[col] = pcts
            sems_by_col[col] = sems

        cohort_data[label] = {'weeks': weeks, 'pcts': pcts_by_col, 'sems': sems_by_col}

    all_weeks_sorted = sorted(all_weeks_set)
    x_pos = {w: i for i, w in enumerate(all_weeks_sorted)}
    x_labels = [str(w) for w in all_weeks_sorted]

    # Determine shared y-axis upper limit from the highest pct+sem across all behaviors/cohorts
    _all_tops: List[float] = []
    for _data in cohort_data.values():
        for col, _, _ in BEHAVIORS:
            _ps = _data['pcts'].get(col, [])
            _ss = _data['sems'].get(col, [])
            for _p, _s in zip(_ps, _ss):
                if not (np.isnan(_p) or np.isnan(_s)):
                    _all_tops.append(_p + _s)
    _y_max = (max(_all_tops) * 1.1) if _all_tops else 100.0

    fig, axes = plt.subplots(1, 3, sharey=True)
    plot_title = title or "Behavioral Metrics by Cohort � Across Weeks"
    fig.suptitle(plot_title, weight='bold', y=0.98)

    for ax, (col, _, panel_title) in zip(axes, BEHAVIORS):
        for i, (label, data) in enumerate(cohort_data.items()):
            color = _cohort_label_to_color(label)
            xs = [x_pos[w] for w in data['weeks']]
            ys = data['pcts'][col]
            sems = data['sems'][col]
            ax.plot(
                xs, ys,
                color=color, linewidth=0.9,
                marker='o',
                markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2,
                label=label,
            )
            ax.errorbar(
                xs, ys,
                yerr=sems,
                fmt='none',
                color=color,
                capsize=4, capthick=0.8, linewidth=0.9,
                alpha=0.7,
            )
        ax.set_xticks(range(len(all_weeks_sorted)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Week')
        ax.set_ylim(0, 50)
        ax.set_title(panel_title, pad=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both', length=5)

    axes[0].set_ylabel('% of Observations', weight='bold')
    axes[-1].legend(loc='upper right', frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path))
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_behavioral_interaction_effects(
    results: Dict,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, "plt.Figure"]:
    """
    For each behavioral metric in `results` where GEE found at least one
    significant effect (Week, Cohort, or Cohort�Week; non-degenerate fit),
    produce a Cohort � Week interaction line plot.

    Parameters
    ----------
    results   : return value of perform_behavioral_mixed_analysis()
    save_dir  : if given, SVG files are saved here
    show      : whether to call plt.show()

    Returns
    -------
    dict of metric_label -> Figure for each plot generated
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available \u2014 cannot generate interaction plots")
        return {}

    _COLORS = ['#2166AC', '#D6604D', '#4DAC26', '#984EA3', '#FF7F00']

    def _stars(p: float) -> str:
        if np.isnan(p):
            return 'n/a'
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    def _fmt_p(p: float) -> str:
        if np.isnan(p):
            return 'N/A'
        if p < 0.001:
            return '< 0.001'
        return f'{p:.3f}'

    figs: Dict[str, plt.Figure] = {}

    for metric_label in ['No Nest', 'Lethargy', 'Anxious']:
        res = results.get(metric_label)
        if res is None:
            continue

        gee = res.get('gee', {})
        if gee.get('degenerate', False):
            print(f"  [{metric_label}] Skipping interaction plot \u2014 GEE fit degenerate")
            continue

        week_p   = gee.get('week',        {}).get('p', float('nan'))
        cohort_p = gee.get('cohort',      {}).get('p', float('nan'))
        inter_p  = gee.get('interaction', {}).get('p', float('nan'))

        if not any(not np.isnan(p) and p < 0.05 for p in [week_p, cohort_p, inter_p]):
            print(f"  [{metric_label}] No significant GEE effects \u2014 skipping plot")
            continue

        desc          = res.get('descriptives', {})
        all_weeks     = res.get('all_weeks', sorted({w for cd in desc.values() for w in cd}))
        cohort_labels = res.get('cohort_labels', list(desc.keys()))

        fig, ax = plt.subplots()

        for i, label in enumerate(cohort_labels):
            color = _COLORS[i % len(_COLORS)]
            pcts_by_week = desc.get(label, {})
            xs = list(range(len(all_weeks)))
            ys = [pcts_by_week.get(w, float('nan')) for w in all_weeks]
            ax.plot(
                xs, ys,
                color=color, linewidth=0.9,
                marker='o',
                markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2,
                label=label,
            )

        ax.set_xticks(range(len(all_weeks)))
        ax.set_xticklabels([f"Week {w}" for w in all_weeks])
        ax.set_xlabel('Week', weight='bold')
        ax.set_ylabel('% Observations', weight='bold')
        ax.set_ylim(bottom=0)
        ax.set_title(f"Cohort \u00d7 Week Interaction \u2014 {metric_label}",
                     weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both', length=5)
        ax.legend(frameon=False)

        ann = (
            f"GEE (n={gee.get('n_subjects', '?')} subjects)\n"
            f"Week: p {_fmt_p(week_p)} {_stars(week_p)}\n"
            f"Cohort: p {_fmt_p(cohort_p)} {_stars(cohort_p)}\n"
            f"Cohort\u00d7Week: p {_fmt_p(inter_p)} {_stars(inter_p)}"
        )
        ax.text(
            0.98, 0.97, ann,
            transform=ax.transAxes,
                        verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='#cccccc'),
        )

        fig.tight_layout()

        if save_dir is not None:
            _sdir = Path(save_dir)
            _sdir.mkdir(parents=True, exist_ok=True)
            fname = f"behavioral_interaction_{metric_label.lower().replace(' ', '_')}.svg"
            fig.savefig(str(_sdir / fname))
            print(f"  Saved interaction plot: {_sdir / fname}")

        if show:
            plt.show()

        figs[metric_label] = fig

    if not figs:
        print("  No significant non-degenerate GEE effects found \u2014 no interaction plots generated.")

    return figs


def plot_weight_interaction_effects(
    cohort_dfs: Dict[str, pd.DataFrame],
    measures: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, "plt.Figure"]:
    """
    Cohort � Week interaction line plots for continuous weight measures
    (Daily Change and/or Total Change).

    One plot per measure.  Each cohort becomes one line; x-axis = Week;
    y-axis = mean � SEM across animals in that cohort that week.

    Parameters
    ----------
    cohort_dfs : return value of load_cohorts() / select_and_load_cohorts()
    measures   : list of column names to plot (default: ['Total Change', 'Daily Change'])
    save_dir   : if given, SVG files are saved here
    show       : whether to call plt.show()

    Returns
    -------
    dict of measure_name -> Figure
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available � cannot generate weight interaction plots")
        return {}

    if measures is None:
        measures = ["Total Change", "Daily Change"]

    _COLORS = ['#2166AC', '#D6604D', '#4DAC26', '#984EA3', '#FF7F00']

    def _fmt_p(p: float) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 'N/A'
        return '< 0.001' if p < 0.001 else f'{p:.3f}'

    def _stars(p: float) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 'n/a'
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    # Build a week-aggregated combined dataframe
    try:
        combined = combine_cohorts_for_analysis(cohort_dfs)
        combined = clean_cohort(combined)
        if 'Day' not in combined.columns:
            combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
        combined = _add_week_column_across_cohorts(combined)
        combined = combined[combined['Day'] >= 1]
    except Exception as e:
        print(f"[ERROR] Could not prepare combined data for weight interaction plots: {e}")
        return {}

    cohort_labels = list(cohort_dfs.keys())

    figs: Dict[str, "plt.Figure"] = {}

    for measure in measures:
        if measure not in combined.columns:
            print(f"  [{measure}] Column not found in combined data � skipping")
            continue

        fig, ax = plt.subplots()

        all_weeks = sorted(combined['Week'].dropna().unique())

        for i, label in enumerate(cohort_labels):
            label_df = combined[combined['Cohort'] == label] if 'Cohort' in combined.columns else None

            # Fall back to CA% matching if no 'Cohort' column
            if label_df is None or len(label_df) == 0:
                # Try to extract CA% from the label
                ca_val = None
                for part in label.replace('%', '').split():
                    try:
                        ca_val = float(part)
                        break
                    except ValueError:
                        continue
                if ca_val is not None and 'CA (%)' in combined.columns:
                    label_df = combined[combined['CA (%)'] == ca_val]
                else:
                    print(f"  [{measure}] Cannot identify cohort '{label}' in combined data � skipping line")
                    continue

            color = _COLORS[i % len(_COLORS)]

            week_means = (
                label_df.groupby(['ID', 'Week'])[measure]
                .mean()
                .reset_index()
                .groupby('Week')[measure]
                .agg(['mean', 'sem'])
            )
            week_means = week_means.reindex(all_weeks)

            xs = list(range(len(all_weeks)))
            ys = week_means['mean'].values
            sems = week_means['sem'].values

            ax.plot(
                xs, ys,
                color=color, linewidth=0.9,
                marker='o',
                markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2,
                label=label,
            )
            finite_mask = np.isfinite(ys) & np.isfinite(sems)
            if finite_mask.any():
                xs_arr = np.array(xs)
                ax.fill_between(
                    xs_arr[finite_mask],
                    (ys - sems)[finite_mask],
                    (ys + sems)[finite_mask],
                    color=color, alpha=0.18,
                )

        ax.set_xticks(range(len(all_weeks)))
        ax.set_xticklabels([f"Week {int(w)}" for w in all_weeks])
        ax.set_xlabel('Week', weight='bold')
        ax.set_ylabel(f'{measure} (Mean � SEM)', weight='bold')
        ax.set_title(f"Cohort � Week � {measure}", weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in', which='both', length=5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.legend(frameon=False)

        fig.tight_layout()

        if save_dir is not None:
            _sdir = Path(save_dir)
            _sdir.mkdir(parents=True, exist_ok=True)
            fname = f"weight_interaction_{measure.lower().replace(' ', '_')}.svg"
            fig.savefig(str(_sdir / fname))
            print(f"  Saved weight interaction plot: {_sdir / fname}")

        if show:
            plt.show()

        figs[measure] = fig

    if not figs:
        print("  No weight interaction plots generated.")

    return figs


# =============================================================================
# BEHAVIORAL BINARY OUTCOME ANALYSIS
# Two-way repeated-measures analysis for binary behavioral outcomes:
#   Cohort (between) x Week (within)
# Uses Cochran's Q for within-subjects (Week) and Cohort effects, with pairwise
# McNemar post-hoc tests (Bonferroni-corrected) for any significant effects.
# =============================================================================

def _coerce_to_binary(series: pd.Series) -> pd.Series:
    """Convert Yes/No/True/False strings or booleans to 1/0, else NaN."""
    _TRUE  = {'yes', 'true', '1'}
    _FALSE = {'no', 'false', '0'}
    def _cv(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            if v == 1:
                return 1
            if v == 0:
                return 0
        if isinstance(v, str):
            ls = v.strip().lower()
            if ls in _TRUE:
                return 1
            if ls in _FALSE:
                return 0
        return float('nan')
    return series.map(_cv)


def perform_behavioral_mixed_analysis(
    cohort_dfs: Dict[str, pd.DataFrame],
) -> Dict:
    """
    Two-way repeated-measures analysis for binary behavioral outcomes.

    Design:
        - Cohort  (between-subjects): e.g. 0% vs 2%
        - Week    (within-subjects):  weeks 1-N, same animals measured each week

    For each behavioral metric (No Nest, Lethargy, Anxious Behaviors):

      PRIMARY � GEE (Generalized Estimating Equations):
        Single model: Response ~ C(Week) + C(Cohort) + C(Week):C(Cohort)
        groups = animal ID, family = Binomial(), cov_struct = Exchangeable()
        Provides formal p-values for Week, Cohort, and Cohort�Week interaction
        from one joint model, properly accounting for within-subject correlation.
        Reports Wald p-values and Odds Ratios with 95% CIs.

      SENSITIVITY � Cochran's Q (non-parametric, complete-cases only):        Non-parametric analog of repeated-measures ANOVA for binary data.
        Requires complete data across all weeks; included as a robustness check.

      POST-HOC (if GEE Week or interaction p < 0.05):
        Pairwise McNemar tests within each cohort, Bonferroni-corrected.
        Reports chi2, adjusted p, and phi coefficient (effect size).

    Parameters:
        cohort_dfs: dict mapping cohort label -> DataFrame with columns
                    ID, Date, Nest Made?, Lethargy?, Anxious Behaviors?

    Returns:
        Nested dict keyed by metric name, then sub-keys:
          'gee', 'cochrans_q', 'posthoc', 'descriptives', 'n_weeks'
    """
    print("\n" + "=" * 80)
    print("BEHAVIORAL BINARY OUTCOME ANALYSIS: COHORT x WEEK")
    print("=" * 80)

    if not HAS_STATSMODELS:
        print("\n[ERROR] statsmodels is required for this analysis.")
        print("  Install with: pip install statsmodels")
        return {}

    BEHAVIORS = [
        ('Nest Made?',         False, 'No Nest',  'no_nest'),
        ('Anxious Behaviors?', True,  'Anxious',  'anxious'),
        ('Lethargy?',          True,  'Lethargy', 'lethargy'),
    ]

    # ------------------------------------------------------------------
    # Step 1: clean + add Week per cohort, build combined long-form df
    # ------------------------------------------------------------------
    long_frames: List[pd.DataFrame] = []
    for label, df in cohort_dfs.items():
        cdf = clean_cohort(df.copy())
        if 'Date' in cdf.columns:
            cdf['Date'] = pd.to_datetime(cdf['Date'], errors='coerce')
        cdf = cdf.sort_values(['ID', 'Date']).reset_index(drop=True)
        first_dates = cdf.groupby('ID')['Date'].transform('min')
        _day_offset = 1 if 'ramp' in label.lower() else 0
        cdf['_Day'] = (cdf['Date'] - first_dates).dt.days + _day_offset
        cdf = cdf[cdf['_Day'] >= 1].copy()
        cdf['_Week'] = (cdf['_Day'] - 1) // 7 + 1
        cdf['_Cohort'] = label
        for col, _, _, _ in BEHAVIORS:
            if col in cdf.columns:
                cdf[col] = _coerce_to_binary(cdf[col])
        long_frames.append(cdf)

    combined = pd.concat(long_frames, ignore_index=True)
    all_weeks = sorted(combined['_Week'].dropna().unique().astype(int))
    n_weeks = len(all_weeks)
    cohort_labels = list(cohort_dfs.keys())

    print(f"\n  Cohorts : {cohort_labels}")
    print(f"  Weeks   : {all_weeks}")
    print(f"  Subjects: {combined['ID'].nunique()} total")

    all_results: Dict = {}

    for col, aberrant_val, metric_label, metric_key in BEHAVIORS:
        print(f"\n{'=' * 60}")
        print(f"Metric: {metric_label}  (column: '{col}', aberrant = {aberrant_val})")
        print(f"{'=' * 60}")

        if col not in combined.columns:
            print(f"  [SKIP] Column not found in data.")
            all_results[metric_label] = {'error': f"Column '{col}' not found"}
            continue

        # Aggregate to one row per (ID, Week): take first non-null (one behavioral
        # observation per animal per week is expected)
        agg = (
            combined[['ID', '_Cohort', '_Week', col]]
            .dropna(subset=[col])
            .groupby(['ID', '_Cohort', '_Week'], as_index=False)[col]
            .first()
        )
        agg['Response'] = (agg[col] == int(aberrant_val)).astype(int)

        # ------------------------------------------------------------------
        # Descriptive statistics � raw daily observations (consistent with plot)
        # ------------------------------------------------------------------
        # Descriptive statistics � per-animal proportions averaged across animals
        # For each animal: (# yes in week) / (# obs in week)  ? mean across animals
        # n reported = number of animals contributing data that week
        # ------------------------------------------------------------------
        desc: Dict = {}
        desc_n: Dict = {}
        print(f"\n  Descriptive statistics (mean % aberrant per animal per week):")
        print(f"  {'Cohort':<30} " + "  ".join(f"Wk{w}" for w in all_weeks))
        for label in cohort_labels:
            row_pcts = []
            row_ns = []
            cohort_sub = combined[combined['_Cohort'] == label]
            for w in all_weeks:
                w_sub = cohort_sub[cohort_sub['_Week'] == w].dropna(subset=[col])
                animal_pcts = []
                for _, animal_data in w_sub.groupby('ID'):
                    n_obs = len(animal_data)
                    n_yes = (animal_data[col] == int(aberrant_val)).sum()
                    animal_pcts.append(100.0 * n_yes / n_obs)
                n_animals = len(animal_pcts)
                pct = float(np.mean(animal_pcts)) if n_animals > 0 else float('nan')
                row_pcts.append(pct)
                row_ns.append(n_animals)
            desc[label] = {w: p for w, p in zip(all_weeks, row_pcts)}
            desc_n[label] = {w: n for w, n in zip(all_weeks, row_ns)}
            pct_str = "  ".join(f"{p:5.1f}%" if not np.isnan(p) else "   N/A" for p in row_pcts)
            print(f"  {label:<30} {pct_str}")

        # ------------------------------------------------------------------
        # PRIMARY [A] � GEE: joint Cohort + Week + Cohort�Week model
        # ------------------------------------------------------------------
        print(f"\n  [A] PRIMARY: GEE (Binomial, Exchangeable) � joint Cohort � Week model")
        gee_result: Dict = {}
        try:
            gee_df = agg[['ID', '_Cohort', '_Week', 'Response']].copy()
            gee_df['_Week'] = gee_df['_Week'].astype('category')
            gee_df['_Cohort'] = gee_df['_Cohort'].astype('category')
            gee_df = gee_df.sort_values('ID').reset_index(drop=True)

            model = GEE.from_formula(
                "Response ~ C(_Week) + C(_Cohort) + C(_Week):C(_Cohort)",
                groups=gee_df['ID'],
                data=gee_df,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            _n_clusters = int(gee_df['ID'].nunique())
            try:
                fit = model.fit(cov_type='bias_reduced', maxiter=100, ddof_scale=None)
                _se_label = f'Mancl-DeRouen bias-reduced (BC; n={_n_clusters} clusters)'
            except Exception:
                fit = model.fit(cov_type='robust', maxiter=100, ddof_scale=None)
                _se_label = 'robust (BC correction failed)'

            pvals = fit.pvalues
            params = fit.params
            bse   = fit.bse

            # Extract p-values by scanning parameter names for key terms
            def _gee_term_p(keyword: str) -> float:
                """Return the smallest p-value for parameters matching keyword."""
                matches = [p for k, p in pvals.items()
                           if keyword.lower() in k.lower()]
                return float(np.nanmin(matches)) if matches else float('nan')

            def _gee_term_or(keyword: str):
                """Return (OR, CI_lo, CI_hi) for the first param matching keyword."""
                for k in params.index:
                    if keyword.lower() in k.lower():
                        coef = params[k]
                        se   = bse[k]
                        return (float(np.exp(coef)),
                                float(np.exp(coef - 1.96 * se)),
                                float(np.exp(coef + 1.96 * se)))
                return (float('nan'), float('nan'), float('nan'))

            p_week    = _gee_term_p('_Week')
            p_cohort  = _gee_term_p('_Cohort')
            p_inter   = _gee_term_p(':')

            or_cohort, or_lo, or_hi = _gee_term_or('_Cohort')

            def _stars(p):
                return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            _gee_degenerate = all(
                not np.isnan(p) and p >= 0.99
                for p in [p_week, p_cohort, p_inter]
                if not np.isnan(p)
            )
            print(f"    SE correction: {_se_label}")
            if _gee_degenerate:
                print(f"    ??  DEGENERATE FIT: all Wald p � 1.0 � likely near-complete separation")
                print(f"        (outcome near-uniform; GEE p-values unreliable � use Cochran's Q [B])")
            print(f"    Week effect  : Wald p = {p_week:.4f} {_stars(p_week)}")
            print(f"    Cohort effect: Wald p = {p_cohort:.4f} {_stars(p_cohort)}  "
                  f"(OR = {or_cohort:.3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}])")
            print(f"    Interaction  : Wald p = {p_inter:.4f} {_stars(p_inter)}")
            print(f"    n = {len(gee_df['ID'].unique())} subjects, {len(gee_df)} observations")

            gee_result = {
                'model': 'GEE Binomial Exchangeable',
                'formula': 'Response ~ C(Week) + C(Cohort) + C(Week):C(Cohort)',
                'n_subjects': int(gee_df['ID'].nunique()),
                'n_obs': int(len(gee_df)),
                'se_correction': _se_label,
                'degenerate': _gee_degenerate,
                'week': {
                    'p': float(p_week),
                    'significant': bool(p_week < 0.05),
                },
                'cohort': {
                    'p': float(p_cohort),
                    'significant': bool(p_cohort < 0.05),
                    'odds_ratio': or_cohort,
                    'or_ci_lower': or_lo,
                    'or_ci_upper': or_hi,
                },
                'interaction': {
                    'p': float(p_inter),
                    'significant': bool(p_inter < 0.05),
                },
                'converged': bool(fit.converged),
                'params': {k: float(v) for k, v in params.items()},
                'pvalues': {k: float(v) for k, v in pvals.items()},
            }

        except Exception as e:
            print(f"    [WARNING] GEE failed: {type(e).__name__}: {e}")
            gee_result = {
                'model': 'GEE Binomial Exchangeable',
                'error': f"{type(e).__name__}: {e}",
                'week': {'p': float('nan'), 'significant': False},
                'cohort': {'p': float('nan'), 'significant': False},
                'interaction': {'p': float('nan'), 'significant': False},
            }

        # ------------------------------------------------------------------
        # SENSITIVITY [B] � Cochran's Q (complete cases only)
        # ------------------------------------------------------------------
        print(f"\n  [B] SENSITIVITY: Cochran's Q (non-parametric, complete subjects only)")
        subjects_nweeks = agg.groupby('ID')['_Week'].nunique()
        complete_ids = subjects_nweeks[subjects_nweeks == n_weeks].index
        complete_agg = agg[agg['ID'].isin(complete_ids)].copy()

        cochran_result: Dict = {}
        if len(complete_ids) < 3:
            print(f"    [WARNING] Only {len(complete_ids)} subjects with all {n_weeks} weeks � need =3")
            cochran_result = {'test': "Cochran's Q", 'statistic': np.nan, 'p': np.nan,
                              'significant': False, 'n': int(len(complete_ids)),
                              'note': 'Insufficient complete subjects'}
        else:
            week_agg = complete_agg.groupby(['ID', '_Week'], as_index=False)['Response'].first()
            wide = week_agg.pivot(index='ID', columns='_Week', values='Response').dropna()
            try:
                q_res = cochrans_q(wide)
                q_stat = float(q_res.statistic)
                p_q    = float(q_res.pvalue)
                df_q   = n_weeks - 1
                if np.isnan(q_stat):
                    # Happens when all responses are uniform (0/0 in Q formula)
                    print(f"    Not computable � all responses uniform across weeks (no variation)")
                    cochran_result = {
                        'test': "Cochran's Q", 'statistic': np.nan, 'p': np.nan,
                        'significant': False, 'n': int(len(wide)),
                        'note': 'Not computable � responses uniform across weeks (no variation to test)',
                    }
                else:
                    stars  = '***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'ns'
                    print(f"    Q({df_q}) = {q_stat:.3f}, p = {p_q:.4f} {stars}  "
                          f"(n = {len(wide)} complete subjects)")
                    cochran_result = {
                        'test': "Cochran's Q", 'statistic': q_stat, 'df': df_q,
                        'p': p_q, 'significant': bool(p_q < 0.05), 'n': int(len(wide)),
                    }
                    # Permutation test: 5000 within-subject shuffles
                    try:
                        _n_perm = 5000
                        _rng = np.random.default_rng(42)
                        _wide_arr = wide.values.astype(float)
                        _perm_qs = []
                        for _ in range(_n_perm):
                            _shuffled = _wide_arr.copy()
                            for _row_i in range(len(_shuffled)):
                                _rng.shuffle(_shuffled[_row_i])
                            _perm_q = float(cochrans_q(pd.DataFrame(_shuffled, columns=wide.columns)).statistic)
                            _perm_qs.append(_perm_q)
                        _p_perm = float(np.mean([q >= q_stat for q in _perm_qs]))
                        cochran_result['p_permutation'] = _p_perm
                        cochran_result['significant_permutation'] = bool(_p_perm < 0.05)
                        _perm_stars = ('***' if _p_perm < 0.001 else '**' if _p_perm < 0.01
                                       else '*' if _p_perm < 0.05 else 'ns')
                        print(f"    Permutation p (5000 shuffles): {_p_perm:.4f} {_perm_stars}")
                    except Exception as _pe:
                        print(f"    [WARNING] Permutation test failed: {_pe}")
                        cochran_result['p_permutation'] = float('nan')
                        cochran_result['significant_permutation'] = False
            except Exception as e:
                print(f"    [WARNING] Cochran's Q failed: {e}")
                cochran_result = {'test': "Cochran's Q", 'statistic': np.nan, 'p': np.nan,
                                  'significant': False, 'n': int(len(complete_ids)),
                                  'error': str(e)}

        # ------------------------------------------------------------------
        # POST-HOC [C] � pairwise McNemar with effect size (if GEE Week/interaction sig)
        # ------------------------------------------------------------------
        posthoc: Dict = {}
        gee_week_sig   = gee_result.get('week', {}).get('significant', False)
        gee_inter_sig  = gee_result.get('interaction', {}).get('significant', False)

        if gee_week_sig or gee_inter_sig:
            print(f"\n  [C] Post-hoc: pairwise McNemar (Bonferroni-corrected, phi effect size)")
            from itertools import combinations

            for label in cohort_labels:
                c_agg = agg[agg['_Cohort'] == label].copy()
                c_nweeks = c_agg.groupby('ID')['_Week'].nunique()
                c_complete = c_nweeks[c_nweeks >= 2].index
                c_sub = c_agg[c_agg['ID'].isin(c_complete)].groupby(
                    ['ID', '_Week'], as_index=False)['Response'].first()
                weeks_present = sorted(c_sub['_Week'].unique().astype(int))
                pairs = list(combinations(weeks_present, 2))
                if not pairs:
                    continue

                comparisons = []
                for w1, w2 in pairs:
                    df_w1 = c_sub[c_sub['_Week'] == w1][['ID', 'Response']].rename(columns={'Response': 'R1'})
                    df_w2 = c_sub[c_sub['_Week'] == w2][['ID', 'Response']].rename(columns={'Response': 'R2'})
                    paired = pd.merge(df_w1, df_w2, on='ID')
                    if len(paired) < 5:
                        continue
                    ct = pd.crosstab(paired['R1'], paired['R2'])
                    for v in [0, 1]:
                        if v not in ct.index:
                            ct.loc[v] = 0
                        if v not in ct.columns:
                            ct[v] = 0
                    ct = ct.sort_index().sort_index(axis=1)
                    try:
                        mcn = mcnemar(ct, exact=False, correction=True)
                        # Phi coefficient: sign from direction of change
                        b = int(ct.loc[0, 1]) if (0 in ct.index and 1 in ct.columns) else 0
                        c_ = int(ct.loc[1, 0]) if (1 in ct.index and 0 in ct.columns) else 0
                        n_disc = b + c_
                        phi = float(np.sqrt(mcn.statistic / len(paired))) if len(paired) > 0 else float('nan')
                        comparisons.append({
                            'week1': int(w1), 'week2': int(w2),
                            'n_paired': int(len(paired)),
                            'prop_w1': float(paired['R1'].mean()),
                            'prop_w2': float(paired['R2'].mean()),
                            'statistic': float(mcn.statistic),
                            'p_raw': float(mcn.pvalue),
                            'phi': phi,
                        })
                    except Exception as e:
                        print(f"      Week {w1} vs {w2}: [WARNING] {e}")

                actual_n = len(comparisons)
                for comp in comparisons:
                    comp['p_adj'] = min(comp['p_raw'] * actual_n, 1.0)
                    comp['significant'] = comp['p_adj'] < 0.05

                posthoc[label] = {
                    'comparisons': comparisons,
                    'n_comparisons': actual_n,
                    'bonferroni_alpha': 0.05 / actual_n if actual_n > 0 else 0.05,
                }

                print(f"\n    {label} ({actual_n} pairs, Bonferroni a = {0.05/actual_n:.4f}):")
                for comp in comparisons:
                    sig = '*' if comp['significant'] else 'ns'
                    phi_s = f"f={comp['phi']:.2f}" if not np.isnan(comp['phi']) else ""
                    print(f"      Week {comp['week1']} vs Week {comp['week2']}: "
                          f"chi2 = {comp['statistic']:.3f}, "
                          f"p_adj = {comp['p_adj']:.4f} {sig}  "
                          f"({100*comp['prop_w1']:.0f}% ? {100*comp['prop_w2']:.0f}%)  {phi_s}")
        else:
            print(f"\n  [C] Post-hoc: Not needed (GEE Week and Interaction both ns)")

        all_results[metric_label] = {
            'metric': metric_label,
            'column': col,
            'aberrant_value': aberrant_val,
            'n_weeks': n_weeks,
            'all_weeks': all_weeks,
            'cohort_labels': cohort_labels,
            'descriptives': desc,
            'descriptives_n': desc_n,
            'gee': gee_result,
            'cochrans_q': cochran_result,
            'posthoc': posthoc,
        }

    # BH-FDR correction on GEE week p-values across the 3 primary behavioral measures
    _primary_labels = ['No Nest', 'Lethargy', 'Anxious']
    _gee_week_ps = [
        all_results.get(_ml, {}).get('gee', {}).get('week', {}).get('p', float('nan'))
        for _ml in _primary_labels
    ]
    _fdr_adj = [float('nan')] * len(_gee_week_ps)
    _valid_idx = [i for i, p in enumerate(_gee_week_ps) if not np.isnan(p)]
    if _valid_idx:
        _valid_ps = [_gee_week_ps[i] for i in _valid_idx]
        _m_fdr = len(_valid_ps)
        _order = sorted(range(_m_fdr), key=lambda i: _valid_ps[i])
        _sorted_ps = [_valid_ps[i] for i in _order]
        _bh_sorted = [min(1.0, _sorted_ps[k] * _m_fdr / (k + 1)) for k in range(_m_fdr)]
        for _k in range(_m_fdr - 2, -1, -1):
            _bh_sorted[_k] = min(_bh_sorted[_k], _bh_sorted[_k + 1])
        _bh_orig = [0.0] * _m_fdr
        for _spos, _opos in enumerate(_order):
            _bh_orig[_opos] = _bh_sorted[_spos]
        for _vpos, _orig_idx in enumerate(_valid_idx):
            _fdr_adj[_orig_idx] = _bh_orig[_vpos]
    for _i, _ml in enumerate(_primary_labels):
        if _ml in all_results:
            all_results[_ml]['gee_week_fdr'] = _fdr_adj[_i]

    print("\n" + "=" * 80)
    return all_results


def generate_behavioral_report(
    results: Dict,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    save_path: Optional[Path] = None,
) -> str:
    """
    Generate a formatted text report from perform_behavioral_mixed_analysis results.

    Parameters:
        results: Output of perform_behavioral_mixed_analysis()
        cohort_dfs: Original cohort dict (used for preamble counts only)
        save_path: Optional path to write the report

    Returns:
        Report as a string.
    """
    def _fmt_p(p: float) -> str:
        if np.isnan(p):
            return "N/A"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    def _stars(p: float) -> str:
        if np.isnan(p):
            return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    lines: List[str] = []
    W = 80

    lines.append("=" * W)
    lines.append("BEHAVIORAL BINARY OUTCOME ANALYSIS: COHORT x WEEK")
    lines.append("=" * W)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("DESIGN")
    lines.append("-" * W)
    lines.append("  Outcome variables : No Nest (Nest Made? = No), Anxious Behaviors? = Yes,")
    lines.append("                      Lethargy? = Yes")
    lines.append("  Between factor    : Cohort (0% CA vs 2% CA)")
    lines.append("  Within factor     : Week (repeated measures within subjects)")
    lines.append("")
    lines.append("  PRIMARY ANALYSIS  : GEE (Generalized Estimating Equations)")
    lines.append("    Model: Response ~ C(Week) + C(Cohort) + C(Week):C(Cohort)")
    lines.append("    Family: Binomial (logit link)    Corr. structure: Exchangeable")
    lines.append("    Groups: Animal ID")
    lines.append("    SE: Mancl-DeRouen bias-reduced (BC) correction when available (n_clusters<30)")
    lines.append("    Provides: Wald p-values for Week, Cohort, and Cohort�Week interaction")
    lines.append("    Effect size: Odds Ratios with 95% Wald CIs for Cohort terms")
    lines.append("")
    lines.append("  SENSITIVITY CHECK : Cochran's Q (non-parametric; complete subjects only)")
    lines.append("    Non-parametric analog of RM-ANOVA for binary data.")
    lines.append("    Subjects missing any week excluded; use as robustness check.")
    lines.append("    + permutation p-value (5000 within-subject shuffles; no asymptotic assumptions)")
    lines.append("")
    lines.append("  POST-HOC          : Pairwise McNemar tests (if GEE Week or interaction")
    lines.append("    p < 0.05), Bonferroni-corrected. Effect size: phi coefficient.")
    lines.append("  FDR NOTE          : BH-FDR corrected GEE Week p-values across the 3")
    lines.append("    primary behavioral measures shown in table below.")
    lines.append("")

    if cohort_dfs:
        lines.append("COHORT SUMMARY")
        lines.append("-" * W)
        for label, df in cohort_dfs.items():
            n_sub = df['ID'].nunique() if 'ID' in df.columns else '?'
            lines.append(f"  {label:<35} n = {n_sub} subjects")
        lines.append("")

    # BH-FDR summary table
    if results:
        _primary = ['No Nest', 'Lethargy', 'Anxious']
        _any_fdr = any(
            _ml in results and 'gee_week_fdr' in results[_ml]
            for _ml in _primary
        )
        if _any_fdr:
            lines.append("BH-FDR CORRECTED GEE WEEK P-VALUES (across primary behavioral measures)")
            lines.append("-" * W)
            for _ml in _primary:
                if _ml in results:
                    _raw_p = results[_ml].get('gee', {}).get('week', {}).get('p', float('nan'))
                    _fdr_p = results[_ml].get('gee_week_fdr', float('nan'))
                    _fdr_s = _stars(_fdr_p) if not np.isnan(_fdr_p) else 'na'
                    _raw_s = f"{_raw_p:.4f}" if not np.isnan(_raw_p) else 'N/A'
                    _fdr_s2 = f"{_fdr_p:.4f}" if not np.isnan(_fdr_p) else 'N/A'
                    lines.append(f"  {_ml:<25}: raw p = {_raw_s}  \u2192  BH-FDR p = {_fdr_s2} {_fdr_s}")
            lines.append("")

    if not results:
        lines.append("[No results available]")
        report = "\n".join(lines)
        if save_path is not None:
            Path(save_path).write_text(report, encoding='utf-8')
        return report

    for metric_label, res in results.items():
        if 'error' in res:
            lines.append(f"{'=' * W}")
            lines.append(f"METRIC: {metric_label}")
            lines.append(f"  [ERROR] {res['error']}")
            lines.append("")
            continue

        lines.append("=" * W)
        lines.append(f"METRIC: {metric_label.upper()}")
        lines.append("=" * W)
        lines.append("")

        cohort_labels = res.get('cohort_labels', [])
        all_weeks     = res.get('all_weeks', [])

        # ---- Descriptive table ----
        lines.append("Descriptive Statistics (mean % yes per animal per week; n = animals):")
        lines.append("-" * W)
        hdr = f"  {'Cohort':<32}" + "".join(f"  Week {w:>2}" for w in all_weeks)
        lines.append(hdr)
        desc   = res.get('descriptives', {})
        desc_n = res.get('descriptives_n', {})
        for label in cohort_labels:
            row = f"  {label:<32}"
            for w in all_weeks:
                pct = desc.get(label, {}).get(w, float('nan'))
                n_w = desc_n.get(label, {}).get(w, '?')
                if not np.isnan(pct):
                    row += f"  {pct:>5.1f}% (n={n_w})"
                else:
                    row += "         N/A"
            lines.append(row)
        lines.append("")

        # ---- PRIMARY: GEE ----
        gee = res.get('gee', {})
        lines.append("[A] PRIMARY � GEE (Binomial, Exchangeable working correlation)")
        lines.append("-" * W)
        if 'error' in gee:
            lines.append(f"  [MODEL FAILED] {gee['error']}")
            lines.append(f"  See sensitivity analysis (Cochran's Q) below.")
        else:
            n_s = gee.get('n_subjects', '?')
            n_o = gee.get('n_obs', '?')
            conv = "yes" if gee.get('converged', False) else "no"
            lines.append(f"  Model: {gee.get('formula', 'Response ~ C(Week) + C(Cohort) + C(Week):C(Cohort)')}")
            lines.append(f"  n = {n_s} subjects, {n_o} observations   Converged: {conv}")
            if gee.get('se_correction'):
                lines.append(f"    SE correction: {gee['se_correction']}")
            if gee.get('degenerate', False):
                lines.append(f"  ??  DEGENERATE FIT: all Wald p � 1.0 � likely near-complete separation")
                lines.append(f"      (outcome is near-uniform; GEE p-values unreliable � use Cochran's Q [B] instead)")
            lines.append("")

            # Week
            wg = gee.get('week', {})
            p_w = wg.get('p', float('nan'))
            lines.append(f"  Week main effect  : Wald {_fmt_p(p_w)}  {_stars(p_w)}")
            lines.append(f"    Interpretation  : "
                         + ("Significant change in prevalence across weeks."
                            if wg.get('significant') else
                            "No significant change in prevalence across weeks."))
            lines.append("")

            # Cohort
            cg = gee.get('cohort', {})
            p_c  = cg.get('p', float('nan'))
            or_c = cg.get('odds_ratio', float('nan'))
            or_lo = cg.get('or_ci_lower', float('nan'))
            or_hi = cg.get('or_ci_upper', float('nan'))
            lines.append(f"  Cohort main effect: Wald {_fmt_p(p_c)}  {_stars(p_c)}")
            if not np.isnan(or_c):
                lines.append(f"    Odds Ratio ({cohort_labels[1] if len(cohort_labels) > 1 else 'Coh2'} "
                              f"vs {cohort_labels[0]}): "
                              f"OR = {or_c:.3f}  (95% CI: {or_lo:.3f}�{or_hi:.3f})")
            lines.append(f"    Interpretation  : "
                         + ("Significant cohort difference in overall prevalence."
                            if cg.get('significant') else
                            "No significant overall difference between cohorts."))
            lines.append("")

            # Interaction
            ig = gee.get('interaction', {})
            p_i = ig.get('p', float('nan'))
            lines.append(f"  Cohort � Week     : Wald {_fmt_p(p_i)}  {_stars(p_i)}")
            lines.append(f"    Interpretation  : "
                         + ("Significant interaction � cohort differences vary across weeks."
                            if ig.get('significant') else
                            "No significant interaction � cohort differences stable across weeks."))
        lines.append("")

        # ---- SENSITIVITY: Cochran's Q ----
        cq = res.get('cochrans_q', {})
        lines.append("[B] SENSITIVITY � Cochran's Q (non-parametric, complete subjects only)")
        lines.append("-" * W)
        q_stat = cq.get('statistic', float('nan'))
        p_q    = cq.get('p', float('nan'))
        df_q   = cq.get('df', float('nan'))
        n_q    = cq.get('n', '?')
        if 'error' in cq:
            lines.append(f"  [FAILED] {cq['error']}")
        elif 'note' in cq:
            lines.append(f"  {cq['note']}")
        elif not np.isnan(q_stat):
            lines.append(f"  Q({int(df_q) if not np.isnan(df_q) else '?'}) = {q_stat:.3f},  "
                         f"{_fmt_p(p_q)}  {_stars(p_q)}  (n = {n_q} complete subjects)")
            _p_perm = cq.get('p_permutation', float('nan'))
            if not np.isnan(_p_perm):
                lines.append(f"  Permutation p (5000 shuffles): {_p_perm:.4f}  {_stars(_p_perm)}")
            _gee_week_p = res.get('gee', {}).get('week', {}).get('p', float('nan'))
            _gee_week_sig = not np.isnan(_gee_week_p) and _gee_week_p < 0.05
            _cq_sig = cq.get('significant', False)
            if _cq_sig and _gee_week_sig:
                _interp = "Corroborates GEE week effect."
            elif _cq_sig and not _gee_week_sig:
                _interp = "Diverges from GEE � Q significant but GEE week ns; interpret with caution."
            elif not _cq_sig and _gee_week_sig:
                _interp = ("Diverges from GEE � GEE week p < 0.05 but Q ns. "
                           "Likely reflects complete-case restriction, low power, or "
                           "separation inflating GEE significance.")
            else:
                _interp = "Consistent with GEE � no significant week effect in either test."
            lines.append(f"  Interpretation: {_interp}")
        else:
            lines.append("  Not computed.")
        lines.append("")

        # ---- POST-HOC ----
        ph = res.get('posthoc', {})
        lines.append("[C] POST-HOC � Pairwise McNemar (Bonferroni-corrected, phi effect size)")
        lines.append("-" * W)
        if not ph:
            lines.append("  Not performed � GEE Week and Interaction both non-significant.")
        else:
            for label in cohort_labels:
                cohort_ph = ph.get(label)
                if not cohort_ph:
                    continue
                comparisons = cohort_ph.get('comparisons', [])
                n_comp  = cohort_ph.get('n_comparisons', 0)
                b_alpha = cohort_ph.get('bonferroni_alpha', 0.05)
                lines.append(f"  {label}  ({n_comp} pair(s), Bonferroni a = {b_alpha:.4f}):")
                if not comparisons:
                    lines.append("    No valid pairs found.")
                    continue
                lines.append(f"    {'Comparison':<22} {'chi2':>8}  {'p (adj)':>10}  {'Sig':>4}  "
                              f"{'%Wk1':>7}  {'%Wk2':>7}  {'phi':>6}")
                lines.append(f"    {'-'*22}  {'-'*8}  {'-'*10}  {'-'*4}  "
                              f"{'-'*7}  {'-'*7}  {'-'*6}")
                for comp in comparisons:
                    comp_lbl = f"Week {comp['week1']} vs Week {comp['week2']}"
                    sig = '*' if comp['significant'] else 'ns'
                    phi_s = f"{comp.get('phi', float('nan')):.2f}" if not np.isnan(comp.get('phi', float('nan'))) else " N/A"
                    lines.append(
                        f"    {comp_lbl:<22}  {comp['statistic']:8.3f}  {_fmt_p(comp['p_adj']):>10}  "
                        f"{sig:>4}  {100*comp['prop_w1']:>6.1f}%  {100*comp['prop_w2']:>6.1f}%  {phi_s:>6}"
                    )
        lines.append("")

    # Footer
    lines.append("=" * W)
    lines.append("STATISTICAL NOTES")
    lines.append("=" * W)
    lines.append("  GEE: Generalized Estimating Equations with Binomial family (logit link) and")
    lines.append("    Exchangeable working correlation structure. Accounts for within-subject")
    lines.append("    correlation across repeated weekly observations. Population-averaged")
    lines.append("    interpretation. Wald z-tests for each term.")
    lines.append("    SE: Mancl-DeRouen bias-reduced (BC) correction applied when n_clusters<30;")
    lines.append("    falls back to robust SE if BC fails. Recommended for small animal studies.")
    lines.append("  Cochran's Q: Non-parametric test for within-subjects binary proportions.")
    lines.append("    Complete-case only; subjects missing any week excluded. Use as")
    lines.append("    robustness check alongside GEE.")
    lines.append("    Permutation p: 5000 within-subject label shuffles; no asymptotic assumptions.")
    lines.append("  BH-FDR: Benjamini-Hochberg corrected GEE Week p-values across the 3 primary")
    lines.append("    behavioral measures (No Nest, Lethargy, Anxious). Correct sort-in-sorted-space")
    lines.append("    step-up procedure used.")
    lines.append("  McNemar: Paired chi-square with continuity correction (exact=False).")
    lines.append("    P-values multiplied by number of pairs (Bonferroni). Phi coefficient")
    lines.append("    = sqrt(chi2 / n) provides standardised effect size.")
    lines.append("  Alpha = 0.05.  * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    lines.append("=" * W)

    report = "\n".join(lines)

    if save_path is not None:
        Path(save_path).write_text(report, encoding='utf-8')
        print(f"\n[OK] Behavioral report saved -> {save_path}")

    return report


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_cross_cohort_report(
    between_results: Optional[Dict] = None,
    mixed_results: Optional[Dict] = None,
    daily_results: Optional[Dict] = None,
    results_males: Optional[Dict] = None,
    results_females: Optional[Dict] = None,
    results_ca0: Optional[Dict] = None,
    results_ca2: Optional[Dict] = None,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None
) -> str:
    """
    Generate a comprehensive formatted report of cross-cohort analyses.
    
    Parameters:
        between_results: Results from perform_between_subjects_anova()
        mixed_results: Results from perform_cross_cohort_mixed_anova()
        daily_results: Results from perform_daily_between_subjects_anova()
        results_males: Results from perform_mixed_anova_sex_stratified() for males
        results_females: Results from perform_mixed_anova_sex_stratified() for females
        results_ca0: Results from perform_mixed_anova_ca_stratified() for 0% CA
        results_ca2: Results from perform_mixed_anova_ca_stratified() for 2% CA
        cohort_dfs: Original cohort DataFrames for context
        
    Returns:
        Formatted string report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("CROSS-COHORT STATISTICAL ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Study design summary
    if cohort_dfs is not None:
        report_lines.append("STUDY DESIGN")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        total_animals = 0
        for label, df in cohort_dfs.items():
            n_animals = df['ID'].nunique() if 'ID' in df.columns else 0
            total_animals += n_animals
            report_lines.append(f"  {label}: {n_animals} animals")
        
        report_lines.append(f"\n  Total animals: {total_animals}")
        
        # Get combined dataframe for design info
        combined = combine_cohorts_for_analysis(cohort_dfs)
        combined = clean_cohort(combined)
        
        if 'Sex' in combined.columns:
            sex_counts = combined.groupby('Sex')['ID'].nunique()
            report_lines.append(f"\n  Sex distribution:")
            for sex, count in sex_counts.items():
                report_lines.append(f"    {sex}: {count} animals")
        
        if 'CA (%)' in combined.columns:
            ca_counts = combined.groupby('CA (%)')['ID'].nunique()
            report_lines.append(f"\n  CA% distribution:")
            for ca, count in ca_counts.items():
                report_lines.append(f"    {ca}%: {count} animals")
        
        if 'Day' in combined.columns or 'Date' in combined.columns:
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined)
            day_range = (combined['Day'].min(), combined['Day'].max())
            report_lines.append(f"\n  Time period: Day {day_range[0]} to Day {day_range[1]}")
        
        report_lines.append("\n  Design: Between-subjects (CA% and Sex)")
        report_lines.append("          Within-subjects (Time/Day)")
        report_lines.append("")
    
    # Between-subjects ANOVA results (time held constant)
    if between_results:
        report_lines.append("=" * 80)
        report_lines.append("BETWEEN-SUBJECTS ANOVA: CA%  x  SEX")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Analysis Type: {between_results.get('analysis_type', 'Unknown')}")
        report_lines.append(f"Measure: {between_results.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {between_results.get('n_subjects', 'Unknown')}")
        report_lines.append("")
        
        # Descriptive Statistics
        desc_stats = between_results.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("Descriptive Statistics:")
            report_lines.append("-" * 80)
            report_lines.append("Basic Statistics:")
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
                                  f"n={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
            report_lines.append("\nConfidence Intervals & Range:")
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
                                  f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
                                  f"IQR [{row['q25']:.3f}, {row['q75']:.3f}], "
                                  f"Range [{row['min']:.3f}, {row['max']:.3f}]")
            report_lines.append("")
        
        aov = between_results.get('anova_table')
        if aov is not None:
            report_lines.append("ANOVA Table:")
            report_lines.append("-" * 80)
            report_lines.append(aov.to_string())
            report_lines.append("")
            
            # Interpret results
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            # CA% main effect
            if 'CA (%)' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                ca_df = ca_row['DF']
                resid_df = aov[aov['Source'] == 'Residual'].iloc[0]['DF']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. CA% Main Effect: {sig_str}")
                report_lines.append(f"   F({ca_df:.0f},{resid_df:.0f}) = {ca_F:.3f}, p = {ca_p:.4f}")
                
                if ca_p < 0.05:
                    report_lines.append(f"   -> Weight measures differ significantly between CA% concentrations")
                else:
                    report_lines.append(f"   -> No significant difference between CA% concentrations")
            
            # Sex main effect
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                sex_df = sex_row['DF']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Main Effect: {sig_str}")
                report_lines.append(f"   F({sex_df:.0f},{resid_df:.0f}) = {sex_F:.3f}, p = {sex_p:.4f}")
                
                if sex_p < 0.05:
                    report_lines.append(f"   -> Males and females show significantly different weight measures")
                else:
                    report_lines.append(f"   -> No significant sex difference")
            
            # Interaction effect
            if 'CA (%) * Sex' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                int_df = int_row['DF']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. CA%  x  Sex Interaction: {sig_str}")
                report_lines.append(f"   F({int_df:.0f},{resid_df:.0f}) = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   -> The effect of CA% differs between males and females")
                    report_lines.append(f"   -> (or equivalently: the sex difference depends on CA% level)")
                else:
                    report_lines.append(f"   -> The effect of CA% is similar for both sexes")
        
        # Post-hoc tests
        posthoc = between_results.get('posthoc_tests')
        if posthoc and len(posthoc) > 0:
            report_lines.append("")
            report_lines.append("Post-Hoc Pairwise Comparisons:")
            report_lines.append("-" * 80)
            
            for test_name, test_df in posthoc.items():
                report_lines.append(f"\n{test_name}:")
                if test_df is not None and len(test_df) > 0:
                    report_lines.append(test_df.to_string())
                else:
                    report_lines.append("  No pairwise comparisons available")
        
        report_lines.append("")
    
    # Mixed ANOVA results (including time)
    if mixed_results:
        report_lines.append("=" * 80)
        report_lines.append("MIXED ANOVA: CA%  x  TIME  x  SEX")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {mixed_results.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {mixed_results.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {mixed_results.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {mixed_results.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        # Mixed ANOVA table (Time within  x  Group between)
        mixed_aov = mixed_results.get('mixed_anova_table')
        if mixed_aov is not None:
            report_lines.append("Mixed ANOVA Table (Day within  x  Group between):")
            report_lines.append("-" * 80)
            report_lines.append(mixed_aov.to_string())
            report_lines.append("")
        
        # Between-subjects decomposition table
        between_aov = mixed_results.get('between_anova_table')
        if between_aov is not None:
            report_lines.append("Between-Subjects Effects (CA%  x  Sex):")
            report_lines.append("-" * 80)
            report_lines.append(between_aov.to_string())
            report_lines.append("")
            
            # Interpret results
            p_col = 'p-unc' if 'p-unc' in between_aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            # CA% effect
            if 'CA (%)' in between_aov['Source'].values:
                ca_row = between_aov[between_aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                ca_df = ca_row['DF']
                resid_df = between_aov[between_aov['Source'] == 'Residual'].iloc[0]['DF']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. CA% Main Effect: {sig_str}")
                report_lines.append(f"   F({ca_df:.0f},{resid_df:.0f}) = {ca_F:.3f}, p = {ca_p:.4f}")
                
                if ca_p < 0.001:
                    report_lines.append(f"   -> HIGHLY SIGNIFICANT difference between CA% concentrations")
                elif ca_p < 0.05:
                    report_lines.append(f"   -> Significant difference between CA% concentrations")
                else:
                    report_lines.append(f"   -> No significant CA% effect")
            
            # Sex effect
            if 'Sex' in between_aov['Source'].values:
                sex_row = between_aov[between_aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                sex_df = sex_row['DF']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Main Effect: {sig_str}")
                report_lines.append(f"   F({sex_df:.0f},{resid_df:.0f}) = {sex_F:.3f}, p = {sex_p:.4f}")
                
                if sex_p < 0.05:
                    report_lines.append(f"   -> Males and females differ in weight measures")
                else:
                    report_lines.append(f"   -> No significant sex difference")
            
            # CA%  x  Sex interaction
            if 'CA (%) * Sex' in between_aov['Source'].values:
                int_row = between_aov[between_aov['Source'] == 'CA (%) * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                int_df = int_row['DF']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. CA%  x  Sex Interaction: {sig_str}")
                report_lines.append(f"   F({int_df:.0f},{resid_df:.0f}) = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   -> The CA% effect differs between sexes")
                else:
                    report_lines.append(f"   -> Similar CA% effect for both sexes")
            
            # Time effect
            if mixed_aov is not None and 'Day' in mixed_aov['Source'].values:
                day_row = mixed_aov[mixed_aov['Source'] == 'Day'].iloc[0]
                time_p_col = 'p-unc' if 'p-unc' in mixed_aov.columns else 'p_unc'
                day_p = day_row[time_p_col]
                day_F = day_row['F']
                
                sig_str = "SIGNIFICANT" if day_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n4. Time (Day) Effect: {sig_str}")
                report_lines.append(f"   F = {day_F:.3f}, p = {day_p:.4f}")
                
                if day_p < 0.001:
                    report_lines.append(f"   -> HIGHLY SIGNIFICANT change over time")
                elif day_p < 0.05:
                    report_lines.append(f"   -> Significant change over time")
                else:
                    report_lines.append(f"   -> No significant change over time")
            
            # Day  x  Group interaction
            if mixed_aov is not None and 'Interaction' in mixed_aov['Source'].values:
                int_row = mixed_aov[mixed_aov['Source'] == 'Interaction'].iloc[0]
                int_p = int_row[time_p_col]
                int_F = int_row['F']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n5. Day  x  Group Interaction: {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   -> The pattern of change over time differs between groups")
                    report_lines.append(f"   -> Different trajectories for different CA%/Sex combinations")
                else:
                    report_lines.append(f"   -> Similar time course for all groups")
        
        # Descriptive Statistics
        desc_stats = mixed_results.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("")
            report_lines.append("Descriptive Statistics:")
            report_lines.append("-" * 80)
            report_lines.append("Basic Statistics:")
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
                                  f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
            report_lines.append("\nConfidence Intervals & Range:")
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
                                  f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
                                  f"IQR [{row['q25']:.3f}, {row['q75']:.3f}], "
                                  f"Range [{row['min']:.3f}, {row['max']:.3f}]")
        
        # Post-hoc tests
        posthoc = mixed_results.get('posthoc_tests')
        if posthoc and len(posthoc) > 0:
            report_lines.append("")
            report_lines.append("Post-Hoc Pairwise Comparisons:")
            report_lines.append("-" * 80)
            
            for test_name, test_df in posthoc.items():
                report_lines.append(f"\n{test_name}:")
                if test_df is not None and len(test_df) > 0:
                    report_lines.append(test_df.to_string())
                else:
                    report_lines.append("  No pairwise comparisons available")
        
        report_lines.append("")
    
    # Sex-stratified mixed ANOVA results
    # Bonferroni note: two tests in the sex-stratified family (k=2)
    if results_males or results_females:
        report_lines.append("=" * 80)
        report_lines.append("SEX-STRATIFIED MIXED ANOVA: NOTE ON MULTIPLE COMPARISONS")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Two sex-stratified tests are run (males and females), forming a family of k=2.")
        report_lines.append("Bonferroni correction is applied: adjusted alpha = 0.05 / 2 = 0.025.")
        report_lines.append("Both raw and Bonferroni-corrected p-values are reported for the key interaction.")
        report_lines.append("")
    if results_males:
        report_lines.append("=" * 80)
        report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME  x  CA% (MALES ONLY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {results_males.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {results_males.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {results_males.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {results_males.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        aov = results_males.get('anova_table')
        if aov is not None:
            report_lines.append("Mixed ANOVA Table (Males):")
            report_lines.append("-" * 80)
            report_lines.append(aov.to_string())
            report_lines.append("")
            
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            if 'Day' in aov['Source'].values:
                day_row = aov[aov['Source'] == 'Day'].iloc[0]
                day_p = day_row[p_col]
                day_F = day_row['F']
                
                sig_str = "SIGNIFICANT" if day_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. Time Effect (Males): {sig_str}")
                report_lines.append(f"   F = {day_F:.3f}, p = {day_p:.4f}")
                if day_p < 0.05:
                    report_lines.append(f"   -> Males show significant weight changes over time")
                else:
                    report_lines.append(f"   -> No significant time effect in males")
            
            if 'CA (%)' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. CA% Effect (Males): {sig_str}")
                report_lines.append(f"   F = {ca_F:.3f}, p = {ca_p:.4f}")
                if ca_p < 0.05:
                    report_lines.append(f"   -> CA% levels differ in males")
                else:
                    report_lines.append(f"   -> No CA% difference in males")
            
            if 'CA (%) * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time  x  CA% Interaction (Males): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   -> The time course significantly differs between CA% levels in males (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   -> Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   -> Similar time course across CA% levels in males")
        
        # Descriptive statistics
        desc_stats = results_males.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("")
            report_lines.append("Descriptive Statistics (Males):")
            report_lines.append("-" * 80)
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}: "
                                  f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        
        report_lines.append("")
    
    if results_females:
        report_lines.append("=" * 80)
        report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME  x  CA% (FEMALES ONLY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {results_females.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {results_females.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {results_females.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {results_females.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        aov = results_females.get('anova_table')
        if aov is not None:
            report_lines.append("Mixed ANOVA Table (Females):")
            report_lines.append("-" * 80)
            report_lines.append(aov.to_string())
            report_lines.append("")
            
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            if 'Day' in aov['Source'].values:
                day_row = aov[aov['Source'] == 'Day'].iloc[0]
                day_p = day_row[p_col]
                day_F = day_row['F']
                
                sig_str = "SIGNIFICANT" if day_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. Time Effect (Females): {sig_str}")
                report_lines.append(f"   F = {day_F:.3f}, p = {day_p:.4f}")
                if day_p < 0.05:
                    report_lines.append(f"   -> Females show significant weight changes over time")
                else:
                    report_lines.append(f"   -> No significant time effect in females")
            
            if 'CA (%)' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. CA% Effect (Females): {sig_str}")
                report_lines.append(f"   F = {ca_F:.3f}, p = {ca_p:.4f}")
                if ca_p < 0.05:
                    report_lines.append(f"   -> CA% levels differ in females")
                else:
                    report_lines.append(f"   -> No CA% difference in females")
            
            if 'CA (%) * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time  x  CA% Interaction (Females): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   -> The time course significantly differs between CA% levels in females (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   -> Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   -> Similar time course across CA% levels in females")
        
        # Descriptive statistics
        desc_stats = results_females.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("")
            report_lines.append("Descriptive Statistics (Females):")
            report_lines.append("-" * 80)
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  CA%={row['CA (%)']}: "
                                  f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        
        report_lines.append("")
    
    # Comparison of sex-stratified results
    if results_males and results_females:
        report_lines.append("=" * 80)
        report_lines.append("SEX-STRATIFIED COMPARISON")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("Comparing Time  x  CA% interactions between males and females:")
        report_lines.append("-" * 80)
        
        # Get interaction p-values
        males_aov = results_males.get('anova_table')
        females_aov = results_females.get('anova_table')
        
        if males_aov is not None and females_aov is not None:
            p_col = 'p-unc' if 'p-unc' in males_aov.columns else 'p_unc'
            
            # Males interaction
            males_int_p = None
            males_int_p_adj = None
            if 'CA (%) * Day' in males_aov['Source'].values:
                males_int_p = males_aov[males_aov['Source'] == 'CA (%) * Day'].iloc[0][p_col]
                males_int_p_adj = min(1.0, males_int_p * 2)
                males_int_sig = "SIGNIFICANT" if males_int_p < 0.05 else "NOT significant"
                males_int_sig_adj = "SIGNIFICANT" if males_int_p_adj < 0.05 else "NOT significant"
                report_lines.append(f"\nMales Time \u00d7 CA% interaction:")
                report_lines.append(f"  p (raw) = {males_int_p:.4f} [{males_int_sig}]  |  p (Bonf, k=2) = {males_int_p_adj:.4f} [{males_int_sig_adj}]")
            
            # Females interaction
            females_int_p = None
            females_int_p_adj = None
            if 'CA (%) * Day' in females_aov['Source'].values:
                females_int_p = females_aov[females_aov['Source'] == 'CA (%) * Day'].iloc[0][p_col]
                females_int_p_adj = min(1.0, females_int_p * 2)
                females_int_sig = "SIGNIFICANT" if females_int_p < 0.05 else "NOT significant"
                females_int_sig_adj = "SIGNIFICANT" if females_int_p_adj < 0.05 else "NOT significant"
                report_lines.append(f"Females Time \u00d7 CA% interaction:")
                report_lines.append(f"  p (raw) = {females_int_p:.4f} [{females_int_sig}]  |  p (Bonf, k=2) = {females_int_p_adj:.4f} [{females_int_sig_adj}]")
            
            # Interpretation using Bonferroni-corrected p
            if males_int_p is not None and females_int_p is not None:
                report_lines.append("")
                report_lines.append("Interpretation (Bonferroni-corrected, alpha = 0.025):")
                males_sig = males_int_p_adj < 0.05
                females_sig = females_int_p_adj < 0.05
                if males_sig and females_sig:
                    report_lines.append("\u2192 Both sexes show significant Time \u00d7 CA% interaction (after correction)")
                    report_lines.append("\u2192 CA% effects on weight trajectory differ by sex")
                elif males_sig and not females_sig:
                    report_lines.append("\u2192 Time \u00d7 CA% interaction survives correction for MALES only")
                    report_lines.append("\u2192 CA% effects on weight trajectory may be sex-specific")
                elif not males_sig and females_sig:
                    report_lines.append("\u2192 Time \u00d7 CA% interaction survives correction for FEMALES only")
                    report_lines.append("\u2192 CA% effects on weight trajectory may be sex-specific")
                else:
                    report_lines.append("\u2192 Neither sex shows significant Time \u00d7 CA% interaction after correction")
        
        report_lines.append("")
    
    # CA%-stratified mixed ANOVA results
    # Bonferroni note: two tests in the CA%-stratified family (k=2)
    if results_ca0 or results_ca2:
        report_lines.append("=" * 80)
        report_lines.append("CA%-STRATIFIED MIXED ANOVA: NOTE ON MULTIPLE COMPARISONS")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Two CA%-stratified tests are run (0% CA and 2% CA), forming a family of k=2.")
        report_lines.append("Bonferroni correction is applied: adjusted alpha = 0.05 / 2 = 0.025.")
        report_lines.append("Both raw and Bonferroni-corrected p-values are reported for the key interaction.")
        report_lines.append("")
    if results_ca0:
        report_lines.append("=" * 80)
        report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME  x  SEX (0% CA ONLY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {results_ca0.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {results_ca0.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {results_ca0.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {results_ca0.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        aov = results_ca0.get('anova_table')
        if aov is not None:
            report_lines.append("Mixed ANOVA Table (0% CA):")
            report_lines.append("-" * 80)
            report_lines.append(aov.to_string())
            report_lines.append("")
            
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            if 'Day' in aov['Source'].values:
                day_row = aov[aov['Source'] == 'Day'].iloc[0]
                day_p = day_row[p_col]
                day_F = day_row['F']
                
                sig_str = "SIGNIFICANT" if day_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. Time Effect (0% CA): {sig_str}")
                report_lines.append(f"   F = {day_F:.3f}, p = {day_p:.4f}")
                if day_p < 0.05:
                    report_lines.append(f"   -> Weight changes significantly over time at 0% CA")
                else:
                    report_lines.append(f"   -> No significant time effect at 0% CA")
            
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Effect (0% CA): {sig_str}")
                report_lines.append(f"   F = {sex_F:.3f}, p = {sex_p:.4f}")
                if sex_p < 0.05:
                    report_lines.append(f"   -> Males and females differ at 0% CA")
                else:
                    report_lines.append(f"   -> No sex difference at 0% CA")
            
            # Check for both possible interaction names
            int_p = None
            int_F = None
            if 'Sex * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'Sex * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
            elif 'Day * Sex' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'Day * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
            
            if int_p is not None:
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time  x  Sex Interaction (0% CA): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   -> The time course significantly differs between sexes at 0% CA (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   -> Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   -> Similar time course for both sexes at 0% CA")
        
        # Descriptive statistics
        desc_stats = results_ca0.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("")
            report_lines.append("Descriptive Statistics (0% CA):")
            report_lines.append("-" * 80)
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  Sex={row['Sex']}: "
                                  f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        
        report_lines.append("")
    
    if results_ca2:
        report_lines.append("=" * 80)
        report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME  x  SEX (2% CA ONLY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {results_ca2.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {results_ca2.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {results_ca2.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {results_ca2.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        aov = results_ca2.get('anova_table')
        if aov is not None:
            report_lines.append("Mixed ANOVA Table (2% CA):")
            report_lines.append("-" * 80)
            report_lines.append(aov.to_string())
            report_lines.append("")
            
            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            if 'Day' in aov['Source'].values:
                day_row = aov[aov['Source'] == 'Day'].iloc[0]
                day_p = day_row[p_col]
                day_F = day_row['F']
                
                sig_str = "SIGNIFICANT" if day_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n1. Time Effect (2% CA): {sig_str}")
                report_lines.append(f"   F = {day_F:.3f}, p = {day_p:.4f}")
                if day_p < 0.05:
                    report_lines.append(f"   -> Weight changes significantly over time at 2% CA")
                else:
                    report_lines.append(f"   -> No significant time effect at 2% CA")
            
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Effect (2% CA): {sig_str}")
                report_lines.append(f"   F = {sex_F:.3f}, p = {sex_p:.4f}")
                if sex_p < 0.05:
                    report_lines.append(f"   -> Males and females differ at 2% CA")
                else:
                    report_lines.append(f"   -> No sex difference at 2% CA")
            
            # Check for both possible interaction names
            int_p = None
            int_F = None
            if 'Sex * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'Sex * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
            elif 'Day * Sex' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'Day * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
            
            if int_p is not None:
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time  x  Sex Interaction (2% CA): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   -> The time course significantly differs between sexes at 2% CA (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   -> Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   -> Similar time course for both sexes at 2% CA")
        
        # Descriptive statistics
        desc_stats = results_ca2.get('descriptive_stats')
        if desc_stats is not None and len(desc_stats) > 0:
            report_lines.append("")
            report_lines.append("Descriptive Statistics (2% CA):")
            report_lines.append("-" * 80)
            for _, row in desc_stats.iterrows():
                report_lines.append(f"  Sex={row['Sex']}: "
                                  f"n_obs={int(row['count'])}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
                                  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
        
        report_lines.append("")
    
    # Comparison of CA%-stratified results
    if results_ca0 and results_ca2:
        report_lines.append("=" * 80)
        report_lines.append("CA%-STRATIFIED COMPARISON")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("Comparing Time  x  Sex interactions between CA% levels:")
        report_lines.append("-" * 80)
        
        # Get interaction p-values
        ca0_aov = results_ca0.get('anova_table')
        ca2_aov = results_ca2.get('anova_table')
        
        if ca0_aov is not None and ca2_aov is not None:
            p_col = 'p-unc' if 'p-unc' in ca0_aov.columns else 'p_unc'
            
            # 0% CA interaction
            ca0_int_p = None
            if 'Sex * Day' in ca0_aov['Source'].values:
                ca0_int_p = ca0_aov[ca0_aov['Source'] == 'Sex * Day'].iloc[0][p_col]
            elif 'Day * Sex' in ca0_aov['Source'].values:
                ca0_int_p = ca0_aov[ca0_aov['Source'] == 'Day * Sex'].iloc[0][p_col]
            
            if ca0_int_p is not None:
                ca0_int_p_adj = min(1.0, ca0_int_p * 2)
                ca0_int_sig = "SIGNIFICANT" if ca0_int_p < 0.05 else "NOT significant"
                ca0_int_sig_adj = "SIGNIFICANT" if ca0_int_p_adj < 0.05 else "NOT significant"
                report_lines.append(f"\n0% CA Time  x  Sex interaction:")
                report_lines.append(f"  p (raw) = {ca0_int_p:.4f} [{ca0_int_sig}]  |  p (Bonf, k=2) = {ca0_int_p_adj:.4f} [{ca0_int_sig_adj}]")
            else:
                ca0_int_p_adj = None
            
            # 2% CA interaction
            ca2_int_p = None
            ca2_int_p_adj = None
            if 'Sex * Day' in ca2_aov['Source'].values:
                ca2_int_p = ca2_aov[ca2_aov['Source'] == 'Sex * Day'].iloc[0][p_col]
            elif 'Day * Sex' in ca2_aov['Source'].values:
                ca2_int_p = ca2_aov[ca2_aov['Source'] == 'Day * Sex'].iloc[0][p_col]
            
            if ca2_int_p is not None:
                ca2_int_p_adj = min(1.0, ca2_int_p * 2)
                ca2_int_sig = "SIGNIFICANT" if ca2_int_p < 0.05 else "NOT significant"
                ca2_int_sig_adj = "SIGNIFICANT" if ca2_int_p_adj < 0.05 else "NOT significant"
                report_lines.append(f"2% CA Time  x  Sex interaction:")
                report_lines.append(f"  p (raw) = {ca2_int_p:.4f} [{ca2_int_sig}]  |  p (Bonf, k=2) = {ca2_int_p_adj:.4f} [{ca2_int_sig_adj}]")
            
            # Interpretation using Bonferroni-corrected p
            if ca0_int_p is not None and ca2_int_p is not None:
                report_lines.append("")
                report_lines.append("Interpretation (Bonferroni-corrected, alpha = 0.025):")
                ca0_sig = (ca0_int_p_adj < 0.05) if ca0_int_p_adj is not None else False
                ca2_sig = (ca2_int_p_adj < 0.05) if ca2_int_p_adj is not None else False
                if ca0_sig and ca2_sig:
                    report_lines.append("-> Both CA% levels show significant Time  x  Sex interaction (after correction)")
                    report_lines.append("-> Sex differences in weight trajectory are consistent across CA% levels")
                elif ca0_sig and not ca2_sig:
                    report_lines.append("-> Time  x  Sex interaction survives correction at 0% CA but NOT at 2% CA")
                    report_lines.append("-> CA% may modulate sex differences in weight trajectory")
                elif not ca0_sig and ca2_sig:
                    report_lines.append("-> Time  x  Sex interaction survives correction at 2% CA but NOT at 0% CA")
                    report_lines.append("-> CA% may modulate sex differences in weight trajectory")
                else:
                    report_lines.append("-> Neither CA% level shows significant Time  x  Sex interaction after correction")
        
        report_lines.append("")
    
    # Daily between-subjects results
    if daily_results:
        report_lines.append("=" * 80)
        report_lines.append("DAILY BETWEEN-SUBJECTS ANOVA: CA%  x  SEX (EACH DAY ANALYZED SEPARATELY)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {daily_results.get('measure', 'Unknown')}")
        report_lines.append(f"Number of days analyzed: {daily_results.get('n_days', 'Unknown')}")
        report_lines.append("")
        
        results_table = daily_results.get('results_table')
        if results_table is not None and len(results_table) > 0:
            # CA% Effect Summary
            report_lines.append("CA% Effect by Day:")
            report_lines.append("-" * 80)
            if 'CA_F' in results_table.columns:
                ca_cols = ['Day', 'N', 'CA_F', 'CA_p', 'CA_sig']
                report_lines.append(results_table[ca_cols].to_string(index=False))
                n_sig = (results_table['CA_p'] < 0.05).sum()
                report_lines.append(f"\nSignificant CA% effect on {n_sig}/{len(results_table)} days")
                
                if n_sig > 0:
                    sig_days = results_table[results_table['CA_p'] < 0.05]['Day'].tolist()
                    report_lines.append(f"Significant days: {', '.join(map(str, sig_days))}")
            
            report_lines.append("")
            
            # Sex Effect Summary
            report_lines.append("Sex Effect by Day:")
            report_lines.append("-" * 80)
            if 'Sex_F' in results_table.columns:
                sex_cols = ['Day', 'N', 'Sex_F', 'Sex_p', 'Sex_sig']
                report_lines.append(results_table[sex_cols].to_string(index=False))
                n_sig = (results_table['Sex_p'] < 0.05).sum()
                report_lines.append(f"\nSignificant Sex effect on {n_sig}/{len(results_table)} days")
                
                if n_sig > 0:
                    sig_days = results_table[results_table['Sex_p'] < 0.05]['Day'].tolist()
                    report_lines.append(f"Significant days: {', '.join(map(str, sig_days))}")
            
            report_lines.append("")
            
            # Interaction Summary
            report_lines.append("CA%  x  Sex Interaction by Day:")
            report_lines.append("-" * 80)
            if 'Interaction_F' in results_table.columns:
                int_cols = ['Day', 'N', 'Interaction_F', 'Interaction_p', 'Interaction_sig']
                report_lines.append(results_table[int_cols].to_string(index=False))
                n_sig = (results_table['Interaction_p'] < 0.05).sum()
                report_lines.append(f"\nSignificant Interaction on {n_sig}/{len(results_table)} days")
                
                if n_sig > 0:
                    sig_days = results_table[results_table['Interaction_p'] < 0.05]['Day'].tolist()
                    report_lines.append(f"Significant days: {', '.join(map(str, sig_days))}")
            
            report_lines.append("")
            
            # Interpretation
            report_lines.append("Interpretation:")
            report_lines.append("-" * 80)
            
            if 'CA_p' in results_table.columns:
                ca_sig_count = (results_table['CA_p'] < 0.05).sum()
                if ca_sig_count > len(results_table) * 0.5:
                    report_lines.append("\n- CA% effect is CONSISTENTLY significant across most days")
                elif ca_sig_count > 0:
                    report_lines.append(f"\n- CA% effect is significant on some days ({ca_sig_count}/{len(results_table)})")
                else:
                    report_lines.append("\n- CA% effect is not significant on any individual day")
            
            if 'Sex_p' in results_table.columns:
                sex_sig_count = (results_table['Sex_p'] < 0.05).sum()
                if sex_sig_count > len(results_table) * 0.5:
                    report_lines.append("- Sex effect is CONSISTENTLY significant across most days")
                elif sex_sig_count > 0:
                    report_lines.append(f"- Sex effect is significant on some days ({sex_sig_count}/{len(results_table)})")
                else:
                    report_lines.append("- Sex effect is not significant on any individual day")
            
            if 'Interaction_p' in results_table.columns:
                int_sig_count = (results_table['Interaction_p'] < 0.05).sum()
                if int_sig_count > len(results_table) * 0.5:
                    report_lines.append("- CA%  x  Sex interaction is CONSISTENTLY significant across most days")
                elif int_sig_count > 0:
                    report_lines.append(f"- CA%  x  Sex interaction is significant on some days ({int_sig_count}/{len(results_table)})")
                else:
                    report_lines.append("- CA%  x  Sex interaction is not significant on any individual day")
        
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


# =============================================================================
# OLS ASSUMPTION DIAGNOSTICS FOR CROSS-COHORT ANALYSES
# =============================================================================

def check_ols_assumptions_cross_cohort(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
    *,
    save_dir: Optional[Path] = None,
    show: bool = True,
    save_svg: bool = False,
    save_report: bool = False,
) -> dict:
    """
    Assess whether a weight measure meets OLS assumptions for the two-way
    mixed ANOVA: Week (within-subjects) x Cohort (between-subjects).

    Model: Value ~ C(Week) + C(Cohort) + C(ID)
      - C(Week) captures systematic change over time (within-subjects).
      - C(Cohort) captures the mean difference between cohorts (between-subjects).
      - C(ID) partials out between-subject differences (repeated measures).
      - Residuals should be IID: mean � 0, normally distributed, constant variance,
        and uncorrelated across subjects.

    Returns a dict with keys:
      'residual_mean', 'normality', 'homoscedasticity', 'variance_ratio',
      'independence', 'outliers', 'influential', 'sensitivity_analysis',
      'figure', 'figure_reduced', 'n', 'n_subjects', 'week_levels', 'cohort_levels'
    """
    if not HAS_STATSMODELS:
        print("check_ols_assumptions_cross_cohort: statsmodels not available; skipping.")
        return {'error': 'statsmodels not available'}

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm as _anova_lm
    except ImportError:
        return {'error': 'statsmodels not available'}

    from scipy import stats as _scipy_stats
    import numpy as _np

    print("\n" + "=" * 80)
    print(f"OLS ASSUMPTION DIAGNOSTICS: {measure}")
    print("Model: Value ~ C(Week) + C(Cohort) + C(ID)")
    print("=" * 80)

    # -- Data preparation ------------------------------------------------------
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    required_cols = ['ID', 'Week', 'Cohort', measure]
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return {'error': f"Missing columns: {missing}"}

    analysis_df = combined_df[required_cols].dropna().copy()

    if weeks is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(weeks)]
        print(f"  Filtered to weeks: {sorted(weeks)}")

    # One value per ID � Week (average within week per animal)
    agg = (
        analysis_df
        .groupby(['ID', 'Week', 'Cohort'], as_index=False)[measure]
        .mean()
        .rename(columns={measure: 'Value'})
    )
    agg = _filter_complete_subjects_weekly(agg, 'ID', 'Week')

    agg['Week_str'] = agg['Week'].astype(str)
    agg['ID'] = agg['ID'].astype(str)
    agg['Cohort'] = agg['Cohort'].astype(str)

    n = len(agg)
    n_subjects = agg['ID'].nunique()
    week_levels = sorted(agg['Week'].unique())
    cohort_levels = sorted(agg['Cohort'].unique())

    print(f"\nData: {n} observations, {n_subjects} subjects, "
          f"{len(week_levels)} weeks: {week_levels}, "
          f"{len(cohort_levels)} cohorts: {cohort_levels}")

    if n < 10:
        print("WARNING: Very small sample � OLS diagnostics may be unreliable.")

    # -- OLS fit ---------------------------------------------------------------
    try:
        model = smf.ols('Value ~ C(Week_str) + C(Cohort) + C(ID)', data=agg).fit()
    except Exception as e:
        print(f"ERROR fitting OLS model: {e}")
        return {'error': str(e)}

    residuals = model.resid.values
    fitted = model.fittedvalues.values
    resid_mean = float(_np.mean(residuals))

    print(f"\nResidual mean: {resid_mean:.6f}  (should be � 0)")

    # -- Normality --------------------------------------------------------------
    normality = {}
    if len(residuals) >= 3:
        sw_stat, sw_p = _scipy_stats.shapiro(residuals)
        normality = {
            'test': "Shapiro-Wilk",
            'statistic': float(sw_stat),
            'p': float(sw_p),
            'passed': bool(sw_p >= 0.05),
        }
        print(f"\nNormality (Shapiro-Wilk): W = {sw_stat:.4f}, p = {sw_p:.4f} "
              f"({'PASSED ?' if sw_p >= 0.05 else 'FAILED ? � residuals may not be normal'})")
    else:
        normality = {'test': 'Shapiro-Wilk', 'error': 'Too few observations'}
        print("\nNormality: too few observations for Shapiro-Wilk")

    # -- Homoscedasticity (Levene's by Week) -----------------------------------
    homoscedasticity = {}
    week_str_levels = [str(w) for w in week_levels]
    level_groups = [residuals[agg['Week_str'].values == lv] for lv in week_str_levels
                    if (agg['Week_str'].values == lv).any()]
    level_groups = [g for g in level_groups if len(g) >= 2]
    if len(level_groups) >= 2:
        lev_stat, lev_p = _scipy_stats.levene(*level_groups, center='median')
        homoscedasticity = {
            'test': "Levene (median)",
            'statistic': float(lev_stat),
            'p': float(lev_p),
            'passed': bool(lev_p >= 0.05),
        }
        print(f"\nHomoscedasticity (Levene's test by Week): W = {lev_stat:.4f}, p = {lev_p:.4f} "
              f"({'PASSED ?' if lev_p >= 0.05 else 'FAILED ? � variance may not be constant across weeks'})")
    else:
        homoscedasticity = {'test': 'Levene', 'error': 'Need =2 week groups with =2 obs each'}
        print("\nHomoscedasticity: insufficient groups for Levene's test")

    # -- Variance ratio (max/min residual variance by Week) -------------------
    variance_ratio_info = {}
    level_variances = {
        str(w): float(_np.var(residuals[agg['Week_str'].values == str(w)], ddof=1))
        for w in week_levels
        if (agg['Week_str'].values == str(w)).sum() >= 2
    }
    if len(level_variances) >= 2:
        _vars = list(level_variances.values())
        _max_var = max(_vars)
        _min_var = min(_vars)
        _max_lvl = max(level_variances, key=level_variances.get)
        _min_lvl = min(level_variances, key=level_variances.get)
        var_ratio = _max_var / _min_var if _min_var > 0 else float('inf')
        var_ratio_failed = var_ratio > 4.0
        variance_ratio_info = {
            'ratio': float(var_ratio),
            'max_var': float(_max_var),
            'max_level': _max_lvl,
            'min_var': float(_min_var),
            'min_level': _min_lvl,
            'level_variances': {k: float(v) for k, v in level_variances.items()},
            'passed': not var_ratio_failed,
        }
        print(f"\nVariance Ratio (max/min residual variance by Week): {var_ratio:.3f} "
              f"({'PASSED ?' if not var_ratio_failed else 'FAILED ? � ratio > 4, mixed ANOVA may be inappropriate'})")
        print(f"  Per-week residual variances:")
        for lv in week_str_levels:
            if lv in level_variances:
                print(f"    Week {lv}: residual var = {level_variances[lv]:.4f}")
        if var_ratio_failed:
            print(f"  WARNING: Largest variance (Week {_max_lvl}, var={_max_var:.4f}) is "
                  f"{var_ratio:.1f}� the smallest (Week {_min_lvl}, var={_min_var:.4f}).")
            print(f"  Consider a variance-stabilizing transformation (e.g., log, sqrt).")
    else:
        variance_ratio_info = {'error': 'Insufficient groups for variance ratio'}
        print("\nVariance ratio: insufficient groups")

    # -- Outlier detection -----------------------------------------------------
    _influence_ok = False
    stud_resid = _np.full(n, _np.nan)
    cooks_d = _np.zeros(n)
    try:
        _infl = model.get_influence()
        stud_resid = _infl.resid_studentized_external
        cooks_d, _ = _infl.cooks_distance
        _influence_ok = True
    except Exception as _ie:
        print(f"\nNote: Influence statistics unavailable ({_ie}); outlier/Cook's D checks skipped.")

    _q1, _q3 = _np.percentile(residuals, [25, 75])
    _iqr = _q3 - _q1
    _iqr_lo, _iqr_hi = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr
    _iqr_mask = (residuals < _iqr_lo) | (residuals > _iqr_hi)
    _iqr_idx = _np.where(_iqr_mask)[0]

    _stud_mask = _np.abs(stud_resid) > 3.0 if _influence_ok else _np.zeros(n, dtype=bool)
    _stud_idx = _np.where(_stud_mask)[0]

    outlier_info = {
        'iqr': {
            'lower_bound': float(_iqr_lo),
            'upper_bound': float(_iqr_hi),
            'n_outliers': int(_iqr_mask.sum()),
            'outlier_indices': _iqr_idx.tolist(),
            'outlier_subjects': [str(agg['ID'].iloc[i]) for i in _iqr_idx],
            'outlier_weeks': [str(agg['Week'].iloc[i]) for i in _iqr_idx],
            'outlier_cohorts': [str(agg['Cohort'].iloc[i]) for i in _iqr_idx],
            'outlier_residuals': [float(residuals[i]) for i in _iqr_idx],
        },
        'studentized': {
            'threshold': 3.0,
            'available': _influence_ok,
            'n_outliers': int(_stud_mask.sum()),
            'outlier_indices': _stud_idx.tolist(),
            'outlier_subjects': [str(agg['ID'].iloc[i]) for i in _stud_idx],
            'outlier_values': [float(stud_resid[i]) for i in _stud_idx],
            'max_abs': float(_np.nanmax(_np.abs(stud_resid))) if _influence_ok else float('nan'),
        },
    }

    print(f"\nOutlier Detection:")
    print(f"  IQR method (1.5�IQR fence):  [{_iqr_lo:.3f}, {_iqr_hi:.3f}]")
    if _iqr_mask.sum() == 0:
        print("    No IQR outliers detected ?")
    else:
        print(f"    {_iqr_mask.sum()} outlier(s) detected ?")
        for i in _iqr_idx:
            print(f"      Obs {i}: ID={agg['ID'].iloc[i]}, Week={agg['Week'].iloc[i]}, "
                  f"Cohort={agg['Cohort'].iloc[i]}, residual = {residuals[i]:.3f}")
    if _influence_ok:
        print(f"  Studentized residuals (|t*|>3): max |t*| = {outlier_info['studentized']['max_abs']:.3f}")
        if _stud_mask.sum() == 0:
            print("    No studentized outliers detected ?")
        else:
            print(f"    {_stud_mask.sum()} outlier(s) detected ?")
            for i in _stud_idx:
                print(f"      Obs {i}: ID={agg['ID'].iloc[i]}, Week={agg['Week'].iloc[i]}, "
                      f"Cohort={agg['Cohort'].iloc[i]}, t* = {stud_resid[i]:.3f}")

    # -- Influential observations (Cook's D) -----------------------------------
    _cooks_thresh = 4.0 / n
    _inf_mask = cooks_d > _cooks_thresh if _influence_ok else _np.zeros(n, dtype=bool)
    _inf_idx = _np.where(_inf_mask)[0]

    influential_info = {
        'test': "Cook's Distance",
        'available': _influence_ok,
        'threshold': float(_cooks_thresh),
        'n_influential': int(_inf_mask.sum()),
        'max_cooks_d': float(_np.max(cooks_d)) if _influence_ok else float('nan'),
        'influential_indices': _inf_idx.tolist(),
        'influential_subjects': [str(agg['ID'].iloc[i]) for i in _inf_idx],
        'cooks_d_values': [float(cooks_d[i]) for i in _inf_idx],
        'passed': bool(_inf_mask.sum() == 0),
    }

    if _influence_ok:
        print(f"\nInfluential Observations (Cook's D, threshold = 4/n = {_cooks_thresh:.4f}):")
        print(f"  Max Cook's D = {_np.max(cooks_d):.4f}")
        if _inf_mask.sum() == 0:
            print("  No influential observations detected ?")
        else:
            print(f"  {_inf_mask.sum()} influential observation(s) detected ?")
            for i in _inf_idx:
                print(f"    Obs {i}: ID={agg['ID'].iloc[i]}, Week={agg['Week'].iloc[i]}, "
                      f"Cohort={agg['Cohort'].iloc[i]}, Cook's D = {cooks_d[i]:.4f}")

    # -- Cross-subject independence --------------------------------------------
    def _cross_subj_independence(agg_df, resid_arr):
        """Build per-subject time-ordered residuals and compute cross-subject correlations."""
        _sr = {}
        for _subj in agg_df['ID'].unique():
            _mask = agg_df['ID'].values == _subj
            _idx = _np.where(_mask)[0]
            try:
                _order = _np.argsort(agg_df['Week'].iloc[_idx].astype(float).values)
            except (ValueError, TypeError):
                _order = _np.argsort(agg_df['Week'].iloc[_idx].values)
            _sr[str(_subj)] = resid_arr[_idx[_order]]
        _subj_ids = list(_sr.keys())
        _n_subj = len(_subj_ids)
        _subj_lens = [len(v) for v in _sr.values()]
        _balanced = len(set(_subj_lens)) == 1 and _subj_lens[0] >= 3
        _cross_corr_mean = float('nan')
        _cross_corr_max = float('nan')
        _corr_matrix = None
        if _balanced and _n_subj >= 2:
            _mat = _np.stack([_sr[s] for s in _subj_ids], axis=0)
            _corr_matrix = _np.corrcoef(_mat)
            _off_diag = [abs(_corr_matrix[i, j])
                         for i in range(_n_subj) for j in range(i + 1, _n_subj)]
            _cross_corr_mean = float(_np.mean(_off_diag)) if _off_diag else float('nan')
            _cross_corr_max = float(_np.max(_off_diag)) if _off_diag else float('nan')
        return _sr, _cross_corr_mean, _cross_corr_max, _corr_matrix, _subj_ids, _balanced

    subj_resid, _cross_corr, _cross_corr_max, _corr_mat, _subj_ids_list, _balanced = \
        _cross_subj_independence(agg, residuals)
    _indep_pass = bool(_cross_corr < 0.3) if not _np.isnan(_cross_corr) else True
    independence = {
        'test': 'Cross-subject residual correlation',
        'mean_abs_cross_corr': _cross_corr,
        'max_abs_cross_corr': _cross_corr_max,
        'correlation_matrix': _corr_mat,
        'subject_ids': _subj_ids_list,
        'per_subject_residuals': subj_resid,
        'balanced': _balanced,
        'note': ('Mean absolute cross-subject residual correlation. '
                 'Near 0 = subjects are independent. '
                 'Threshold 0.3 used as soft indicator.'),
        'passed': _indep_pass,
    }
    print(f"\nIndependence (cross-subject residual correlation):")
    if _balanced and not _np.isnan(_cross_corr):
        print(f"  Mean |r| across subject pairs = {_cross_corr:.4f}  (max = {_cross_corr_max:.4f})")
        print(f"  {'PASSED ?' if _indep_pass else 'CONCERN ? � residuals may be correlated across subjects'} "
              f"(threshold: mean |r| < 0.3)")
        print(f"  (n={len(_subj_ids_list)} subjects � treat as indicative only)")
    else:
        print("  Unbalanced design or fewer than 2 subjects � cross-subject correlation not computed.")

    if subj_resid:
        _resid_means = _np.array([_np.mean(v) for v in subj_resid.values()])
        mean_subj_resid = float(_np.mean(_resid_means))
        print(f"\nMean within-subject residual (should be � 0 if ID partialled out): {mean_subj_resid:.6f}")
        independence['mean_subject_residual'] = mean_subj_resid

    # -- Sensitivity analysis (refit excluding studentized outliers) -----------
    sensitivity_info: dict = {'available': False,
                               'reason': 'No studentized residual outliers (|t*| > 3)'}
    if len(_stud_idx) > 0:
        agg_sens = agg.drop(index=_stud_idx.tolist()).reset_index(drop=True)
        n_sens = len(agg_sens)
        print(f"\n{'-'*60}")
        print(f"SENSITIVITY ANALYSIS: Refit excluding {len(_stud_idx)} studentized outlier(s) (|t*| > 3)")
        print(f"  Full model: n={n}  |  Reduced model: n={n_sens}")
        try:
            model_sens = smf.ols('Value ~ C(Week_str) + C(Cohort) + C(ID)', data=agg_sens).fit()
            resid_sens = model_sens.resid.values
            fitted_sens = model_sens.fittedvalues.values

            sw_s_stat, sw_s_p = (_scipy_stats.shapiro(resid_sens)
                                  if len(resid_sens) >= 3 else (float('nan'), float('nan')))
            _wl_sens = sorted(agg_sens['Week_str'].unique())
            _lg_sens = [resid_sens[agg_sens['Week_str'].values == _lv]
                        for _lv in _wl_sens
                        if (agg_sens['Week_str'].values == _lv).sum() >= 2]
            lev_s_stat, lev_s_p = (_scipy_stats.levene(*_lg_sens, center='median')
                                    if len(_lg_sens) >= 2 else (float('nan'), float('nan')))
            _subj_resid_s, _cross_corr_s, _cross_corr_max_s, _, _, _balanced_s = \
                _cross_subj_independence(agg_sens, resid_sens)
            _indep_pass_s = bool(_cross_corr_s < 0.3) if not _np.isnan(_cross_corr_s) else True
            _resid_mean_s = float(_np.mean(resid_sens))

            # Variance ratio for reduced
            _lvars_s = {
                lv: float(_np.var(resid_sens[agg_sens['Week_str'].values == lv], ddof=1))
                for lv in _wl_sens
                if (agg_sens['Week_str'].values == lv).sum() >= 2
            }
            if len(_lvars_s) >= 2:
                _vr_s = (max(_lvars_s.values()) / min(_lvars_s.values())
                         if min(_lvars_s.values()) > 0 else float('inf'))
                _vr_s_pass = bool(_vr_s <= 4.0)
            else:
                _vr_s, _vr_s_pass = float('nan'), True

            # IQR / stud / cooks for reduced
            _q1_s, _q3_s = float(_np.percentile(resid_sens, 25)), float(_np.percentile(resid_sens, 75))
            _iqr_s_val = _q3_s - _q1_s
            _n_iqr_s = int(
                ((resid_sens < _q1_s - 1.5 * _iqr_s_val) |
                 (resid_sens > _q3_s + 1.5 * _iqr_s_val)).sum()
            )
            _n_stud_s = 0; _max_stud_s = float('nan')
            _n_cooks_s = 0; _max_cooks_s = float('nan')
            _cooks_thresh_s = 4.0 / n_sens if n_sens > 0 else float('nan')
            try:
                _infl_s = model_sens.get_influence()
                _stud_s_vals = _infl_s.resid_studentized_external
                _cooks_s_vals, _ = _infl_s.cooks_distance
                _n_stud_s = int((_np.abs(_stud_s_vals) > 3.0).sum())
                _max_stud_s = float(_np.nanmax(_np.abs(_stud_s_vals)))
                _n_cooks_s = int((_cooks_s_vals > _cooks_thresh_s).sum())
                _max_cooks_s = float(_np.max(_cooks_s_vals))
            except Exception:
                pass

            r2_full = float(model.rsquared)
            r2_sens = float(model_sens.rsquared)
            rmse_full = float(_np.sqrt(_np.mean(residuals**2)))
            rmse_sens = float(_np.sqrt(_np.mean(resid_sens**2)))

            # F for Week effect from Type-I ANOVA table
            f_week_full = float('nan')
            f_week_sens = float('nan')
            try:
                _at_full = _anova_lm(model, typ=1)
                _at_sens = _anova_lm(model_sens, typ=1)
                if 'C(Week_str)' in _at_full.index:
                    f_week_full = float(_at_full.loc['C(Week_str)', 'F'])
                if 'C(Week_str)' in _at_sens.index:
                    f_week_sens = float(_at_sens.loc['C(Week_str)', 'F'])
            except Exception:
                pass

            _sw_p_full = normality.get('p', float('nan'))
            _lev_p_full = homoscedasticity.get('p', float('nan'))

            print(f"\n  {'Metric':<26} {'Full model':>12}  {'Reduced model':>14}  {'?':>10}")
            print(f"  {'-'*66}")
            print(f"  {'n':<26} {n:>12d}  {n_sens:>14d}  {n_sens - n:>+10d}")
            print(f"  {'R�':<26} {r2_full:>12.4f}  {r2_sens:>14.4f}  {r2_sens - r2_full:>+10.4f}")
            print(f"  {'Residual RMSE':<26} {rmse_full:>12.4f}  {rmse_sens:>14.4f}  {rmse_sens - rmse_full:>+10.4f}")
            if not _np.isnan(f_week_full):
                print(f"  {'F (Week)':<26} {f_week_full:>12.3f}  {f_week_sens:>14.3f}  {f_week_sens - f_week_full:>+10.3f}")
            print(f"  {'SW normality p':<26} {_sw_p_full:>12.4f}  {sw_s_p:>14.4f}  {sw_s_p - _sw_p_full:>+10.4f}")
            print(f"  {'Levene p':<26} {_lev_p_full:>12.4f}  {lev_s_p:>14.4f}  {lev_s_p - _lev_p_full:>+10.4f}")
            if not (_np.isnan(_cross_corr) or _np.isnan(_cross_corr_s)):
                print(f"  {'Cross-subj mean |r|':<26} {_cross_corr:>12.4f}  {_cross_corr_s:>14.4f}  {_cross_corr_s - _cross_corr:>+10.4f}")
            _delta_r2 = r2_sens - r2_full
            if abs(_delta_r2) > 0.05:
                print(f"\n  WARNING: R� changed by {_delta_r2:+.4f} � influential points substantially affect model fit.")
            else:
                print(f"\n  Model is robust to removal of influential observations (?R� = {_delta_r2:+.4f}).")

            sensitivity_info = {
                'available': True,
                'n_removed': len(_stud_idx),
                'removed_subjects': [str(agg['ID'].iloc[i]) for i in _stud_idx],
                'n_full': n, 'n_reduced': n_sens,
                'r2_full': r2_full, 'r2_reduced': r2_sens, 'delta_r2': float(_delta_r2),
                'rmse_full': rmse_full, 'rmse_reduced': rmse_sens,
                'delta_rmse': float(rmse_sens - rmse_full),
                'f_week_full': f_week_full, 'f_week_reduced': f_week_sens,
                'sw_p_full': float(_sw_p_full), 'sw_p_reduced': float(sw_s_p),
                'levene_p_full': float(_lev_p_full), 'levene_p_reduced': float(lev_s_p),
                'cross_corr_full': _cross_corr, 'cross_corr_reduced': _cross_corr_s,
                'cross_corr_max_full': _cross_corr_max, 'cross_corr_max_reduced': _cross_corr_max_s,
                'per_subj_resid_reduced': _subj_resid_s,
                'resid_mean_reduced': _resid_mean_s,
                'vr_reduced': _vr_s, 'vr_pass_reduced': _vr_s_pass,
                'sw_pass_reduced': bool(sw_s_p > 0.05),
                'lev_pass_reduced': bool(lev_s_p > 0.05),
                'indep_pass_reduced': _indep_pass_s,
                'n_iqr_reduced': _n_iqr_s, 'n_stud_reduced': _n_stud_s,
                'max_stud_reduced': _max_stud_s, 'n_cooks_reduced': _n_cooks_s,
                'max_cooks_reduced': _max_cooks_s, 'cooks_thresh_reduced': _cooks_thresh_s,
                'model_reduced': model_sens,
                'residuals_reduced': resid_sens,
                'fitted_reduced': fitted_sens,
                'agg_reduced': agg_sens,
            }
        except Exception as _se:
            print(f"  Sensitivity refit failed: {_se}")
            sensitivity_info = {'available': False, 'reason': str(_se)}
    else:
        print(f"\nSensitivity analysis: no studentized outliers (|t*| > 3) � full model is the final model.")

    # -- Diagnostic plots ------------------------------------------------------
    import matplotlib.pyplot as _plt

    def _make_diag_fig(
        fig_title, n_obs, n_subj, resid_arr, fitted_arr,
        sw_p_val, lev_p_val, indep_info, wk_levels, agg_df,
        stud_arr, cooks_arr, iqr_mask_arr, iqr_lo_val, iqr_hi_val,
        stud_mask_arr, inf_mask_arr, cooks_thresh_val, infl_ok,
    ):
        _fig, _axes = _plt.subplots(3, 3)
        _fig.suptitle(fig_title, fontsize=12, fontweight='bold')

        # 1. Residuals vs Fitted
        _ax = _axes[0, 0]
        _ax.scatter(fitted_arr, resid_arr, alpha=0.6, edgecolors='k', linewidths=0.5)
        _ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        _ax.set_xlabel("Fitted Values")
        _ax.set_ylabel("Residuals")
        _ax.set_title("Residuals vs Fitted\n(should scatter around 0 uniformly)")

        # 2. Q-Q plot
        _ax = _axes[0, 1]
        (osm, osr), (slope, intercept, _) = _scipy_stats.probplot(resid_arr, dist="norm")
        _ax.scatter(osm, osr, alpha=0.6, edgecolors='k', linewidths=0.5)
        _xlim = _np.array([min(osm), max(osm)])
        _ax.plot(_xlim, slope * _xlim + intercept, color='red', linestyle='--', linewidth=1.5)
        _ax.set_xlabel("Theoretical Quantiles")
        _ax.set_ylabel("Sample Quantiles")
        _ax.set_title(f"Normal Q-Q Plot\nShapiro-Wilk p = {sw_p_val:.4f}")

        # 3. Scale-Location
        _ax = _axes[0, 2]
        _ax.scatter(fitted_arr, _np.sqrt(_np.abs(resid_arr)), alpha=0.6, edgecolors='k', linewidths=0.5)
        _ax.set_xlabel("Fitted Values")
        _ax.set_ylabel("v|Residuals|")
        _ax.set_title(f"Scale-Location\nLevene p = {lev_p_val:.4f}")

        # 4. IQR Outlier Detection
        _ax = _axes[1, 0]
        _iqr_cols = ['red' if m else 'steelblue' for m in iqr_mask_arr]
        _iqr_flagged = _np.where(iqr_mask_arr)[0]
        _ax.scatter(range(n_obs), resid_arr, c=_iqr_cols, alpha=0.7,
                    edgecolors='k', linewidths=0.4, zorder=3)
        _ax.axhline(iqr_hi_val, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Upper fence ({iqr_hi_val:.2f})')
        _ax.axhline(iqr_lo_val, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Lower fence ({iqr_lo_val:.2f})')
        _ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        for _i in _iqr_flagged:
            _ax.annotate(f"{agg_df['ID'].iloc[_i]}\nWk{agg_df['Week'].iloc[_i]}",
                         xy=(_i, resid_arr[_i]), xytext=(4, 4),
                         textcoords='offset points', fontsize=7, color='red')
        _ax.set_xlabel("Observation Index")
        _ax.set_ylabel("Residuals")
        _ax.set_title(f"IQR Outlier Detection (1.5�IQR fence)\n"
                      f"{iqr_mask_arr.sum()} outlier(s) � red=flagged, blue=normal")
        _ax.legend(fontsize=7)

        # 5. Residuals by Week
        _ax = _axes[1, 1]
        _lvl_data = [resid_arr[agg_df['Week_str'].values == str(w)] for w in wk_levels]
        _ax.boxplot(_lvl_data, tick_labels=[str(w) for w in wk_levels], patch_artist=True)
        _ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        _ax.set_xlabel("Week")
        _ax.set_ylabel("Residuals")
        _ax.set_title("Residuals by Week\n(boxes should be centered near 0 with similar spread)")

        # 6. Residual Histogram
        _ax = _axes[1, 2]
        _ax.hist(resid_arr, bins=max(8, int(_np.sqrt(n_obs))), edgecolor='black', alpha=0.7)
        _mu, _std = float(_np.mean(resid_arr)), float(_np.std(resid_arr))
        _xn = _np.linspace(_mu - 3.5 * _std, _mu + 3.5 * _std, 200)
        _ax.plot(_xn, _scipy_stats.norm.pdf(_xn, _mu, _std) * n_obs * (_xn[1] - _xn[0]),
                 color='red', linestyle='--', linewidth=1.5, label='Normal fit')
        _ax.set_xlabel("Residuals")
        _ax.set_ylabel("Count")
        _ax.set_title(f"Residual Histogram\nmean = {_mu:.3f}")
        _ax.legend(fontsize=8)

        # 7. Per-subject residuals (cross-subject independence)
        _ax = _axes[2, 0]
        _cr = indep_info.get('mean_abs_cross_corr', float('nan'))
        _ps = indep_info.get('per_subject_residuals', {})
        _prop_cycle = _plt.rcParams['axes.prop_cycle'].by_key()['color']
        for _ci, (_sid, _sv) in enumerate(_ps.items()):
            _col = _prop_cycle[_ci % len(_prop_cycle)]
            _ax.plot(range(len(_sv)), _sv, 'o-', alpha=0.7,
                     markersize=4, color=_col, label=str(_sid))
        _ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        _ax.set_xlabel("Week order within subject")
        _ax.set_ylabel("Residuals")
        _cr_str = f"mean |r| = {_cr:.4f}" if not _np.isnan(_cr) else "unbalanced"
        _ax.set_title(f"Residuals per Subject\n({_cr_str}; parallel lines ? concern)")
        _ax.legend(fontsize=7, ncol=max(1, len(_ps) // 4))

        # 8. Cook's Distance
        _ax = _axes[2, 1]
        if infl_ok:
            _stems = _ax.stem(range(n_obs), cooks_arr, markerfmt='o', linefmt='grey', basefmt=' ')
            _plt.setp(_stems.markerline, markersize=4)
            _ax.axhline(cooks_thresh_val, color='red', linestyle='--', linewidth=1.5,
                        label=f"Threshold 4/n={cooks_thresh_val:.3f}")
            for _i in _np.where(inf_mask_arr)[0]:
                _ax.annotate(f"{agg_df['ID'].iloc[_i]}",
                             xy=(_i, cooks_arr[_i]), xytext=(4, 4),
                             textcoords='offset points', fontsize=7, color='red')
            _ax.set_title(f"Cook's Distance\n({inf_mask_arr.sum()} influential, "
                          f"threshold={cooks_thresh_val:.3f})")
            _ax.legend(fontsize=8)
        else:
            _ax.text(0.5, 0.5, "Unavailable", ha='center', va='center',
                     transform=_ax.transAxes)
            _ax.set_title("Cook's Distance")
        _ax.set_xlabel("Observation Index")
        _ax.set_ylabel("Cook's D")

        # 9. Externally Studentized Residuals
        _ax = _axes[2, 2]
        if infl_ok:
            _stud_flag = _np.where(stud_mask_arr)[0]
            _ax.scatter(range(n_obs), stud_arr, alpha=0.6, edgecolors='k', linewidths=0.5)
            _ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            _ax.axhline(3, color='red', linestyle='--', linewidth=1.5, label='|t*| = 3')
            _ax.axhline(-3, color='red', linestyle='--', linewidth=1.5)
            for _i in _stud_flag:
                _ax.annotate(f"{agg_df['ID'].iloc[_i]}",
                             xy=(_i, stud_arr[_i]), xytext=(4, 4),
                             textcoords='offset points', fontsize=7, color='red')
            _ax.set_title(f"Studentized Residuals\n({stud_mask_arr.sum()} outliers, |t*| > 3)")
            _ax.legend(fontsize=8)
        else:
            _ax.text(0.5, 0.5, "Unavailable", ha='center', va='center',
                     transform=_ax.transAxes)
            _ax.set_title("Studentized Residuals")
        _ax.set_xlabel("Observation Index")
        _ax.set_ylabel("Studentized Residual (t*)")

        try:
            _plt.tight_layout()
        except Exception:
            pass
        return _fig

    # -- Build full model figure -----------------------------------------------
    _full_title = (f"OLS Residual Diagnostics � {measure} (Full Model)\n"
                   f"Model: Value ~ C(Week) + C(Cohort) + C(ID)   "
                   f"[n={n}, subjects={n_subjects}, cohorts={cohort_levels}]")
    fig = _make_diag_fig(
        fig_title=_full_title,
        n_obs=n, n_subj=n_subjects,
        resid_arr=residuals, fitted_arr=fitted,
        sw_p_val=normality.get('p', float('nan')),
        lev_p_val=homoscedasticity.get('p', float('nan')),
        indep_info=independence,
        wk_levels=week_levels, agg_df=agg,
        stud_arr=stud_resid, cooks_arr=cooks_d,
        iqr_mask_arr=_iqr_mask, iqr_lo_val=_iqr_lo, iqr_hi_val=_iqr_hi,
        stud_mask_arr=_stud_mask, inf_mask_arr=_inf_mask,
        cooks_thresh_val=_cooks_thresh, infl_ok=_influence_ok,
    )

    # -- Reduced model figure (sensitivity) -----------------------------------
    fig_reduced = None
    if sensitivity_info.get('available'):
        _si = sensitivity_info
        _red_model = _si['model_reduced']
        _red_resid = _si['residuals_reduced']
        _red_fitted = _si['fitted_reduced']
        _red_agg = _si['agg_reduced']
        _red_n = _si['n_reduced']
        _red_n_subj = _red_agg['ID'].nunique()
        _red_wlevels = sorted(_red_agg['Week'].unique())

        _q1r, _q3r = _np.percentile(_red_resid, [25, 75])
        _iqr_r = _q3r - _q1r
        _iqr_lo_r, _iqr_hi_r = _q1r - 1.5 * _iqr_r, _q3r + 1.5 * _iqr_r
        _iqr_mask_r = (_red_resid < _iqr_lo_r) | (_red_resid > _iqr_hi_r)

        _stud_r_arr = _np.full(_red_n, _np.nan)
        _cooks_r_arr = _np.zeros(_red_n)
        _infl_ok_r = False
        try:
            _infl_r = _red_model.get_influence()
            _stud_r_arr = _infl_r.resid_studentized_external
            _cooks_r_arr, _ = _infl_r.cooks_distance
            _infl_ok_r = True
        except Exception:
            pass
        _stud_mask_r = _np.abs(_stud_r_arr) > 3.0 if _infl_ok_r else _np.zeros(_red_n, dtype=bool)
        _cooks_thresh_r = 4.0 / _red_n if _red_n > 0 else float('nan')
        _inf_mask_r = _cooks_r_arr > _cooks_thresh_r if _infl_ok_r else _np.zeros(_red_n, dtype=bool)

        _red_title = (
            f"OLS Residual Diagnostics � {measure} (Reduced Model)\n"
            f"Model: Value ~ C(Week) + C(Cohort) + C(ID)   "
            f"[n={_red_n}, subjects={_red_n_subj}, {_si['n_removed']} outlier(s) removed]"
        )
        fig_reduced = _make_diag_fig(
            fig_title=_red_title,
            n_obs=_red_n, n_subj=_red_n_subj,
            resid_arr=_red_resid, fitted_arr=_red_fitted,
            sw_p_val=_si['sw_p_reduced'],
            lev_p_val=_si['levene_p_reduced'],
            indep_info={
                'mean_abs_cross_corr': _si.get('cross_corr_reduced', float('nan')),
                'per_subject_residuals': _si.get('per_subj_resid_reduced', {}),
            },
            wk_levels=_red_wlevels, agg_df=_red_agg,
            stud_arr=_stud_r_arr, cooks_arr=_cooks_r_arr,
            iqr_mask_arr=_iqr_mask_r, iqr_lo_val=_iqr_lo_r, iqr_hi_val=_iqr_hi_r,
            stud_mask_arr=_stud_mask_r, inf_mask_arr=_inf_mask_r,
            cooks_thresh_val=_cooks_thresh_r, infl_ok=_infl_ok_r,
        )

    _fig_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _fig_pairs = [(fig, 'full'), (fig_reduced, 'reduced')]

    if save_dir is not None:
        _save_dir = Path(save_dir)
        _save_dir.mkdir(parents=True, exist_ok=True)
        for _save_fig, _suffix in _fig_pairs:
            if _save_fig is None:
                continue
            _png = _save_dir / f"ols_cross_cohort_{measure.replace(' ', '_')}_{_suffix}_{_fig_ts}.png"
            _save_fig.savefig(_png, dpi=150)
            print(f"\nSaved OLS diagnostics ({_suffix} model) to: {_png}")
            if save_svg:
                _svg = _save_dir / f"ols_cross_cohort_{measure.replace(' ', '_')}_{_suffix}_{_fig_ts}.svg"
                _save_fig.savefig(_svg, format='svg')
    elif save_report:
        for _save_fig, _suffix in _fig_pairs:
            if _save_fig is None:
                continue
            _png = Path.cwd() / f"ols_cross_cohort_{measure.replace(' ', '_')}_{_suffix}_{_fig_ts}.png"
            try:
                _save_fig.savefig(_png, dpi=150)
                print(f"\nSaved OLS diagnostics ({_suffix} model) to: {_png}")
            except Exception as _fe:
                print(f"\nWarning: could not save OLS figure ({_suffix}): {_fe}")

    if show:
        _plt.show()

    # -- Summary ---------------------------------------------------------------
    print("\n" + "-" * 50)
    print("ASSUMPTION CHECK SUMMARY:")
    print(f"  Residual mean � 0:  {resid_mean:.6f}")
    print(f"  Normality:          {'PASSED' if normality.get('passed') else 'FAILED'} (SW p={normality.get('p', float('nan')):.4f})")
    print(f"  Homoscedasticity:   {'PASSED' if homoscedasticity.get('passed') else 'FAILED/N/A'} (Levene p={homoscedasticity.get('p', float('nan')):.4f})")
    _vr = variance_ratio_info.get('ratio', float('nan'))
    _vr_pass = variance_ratio_info.get('passed', True)
    print(f"  Variance ratio:     {'PASSED' if _vr_pass else 'FAILED'} (max/min residual var = {_vr:.3f}{'  ? > 4: mixed ANOVA may be inappropriate' if not _vr_pass else ''})")
    _cr_display = (f"mean |r| = {_cross_corr:.4f}" if not _np.isnan(_cross_corr)
                   else "unbalanced � not computed")
    print(f"  Independence:       {'PASSED' if independence.get('passed') else 'CONCERN'} ({_cr_display})")
    print(f"  Outliers (IQR):     {outlier_info['iqr']['n_outliers']} flagged "
          f"{'?' if outlier_info['iqr']['n_outliers'] == 0 else '?'}")
    if outlier_info['studentized']['available']:
        print(f"  Outliers (|t*|>3):  {outlier_info['studentized']['n_outliers']} flagged "
              f"{'?' if outlier_info['studentized']['n_outliers'] == 0 else '?'} "
              f"(max |t*|={outlier_info['studentized']['max_abs']:.3f})")
    if influential_info['available']:
        print(f"  Cook's D (>4/n):    {influential_info['n_influential']} influential "
              f"{'?' if influential_info['passed'] else '?'} "
              f"(max D={influential_info['max_cooks_d']:.4f})")
    all_pass = (normality.get('passed', False) and homoscedasticity.get('passed', False)
                and independence.get('passed', False) and _vr_pass)
    if all_pass:
        print(f"\nOverall: All OLS assumptions met � mixed ANOVA is appropriate.")
    elif not _vr_pass:
        print(f"\nOverall: Variance ratio > 4 � mixed ANOVA results should be interpreted with caution.")
        print(f"  Consider a transformation (log, sqrt) or a non-parametric alternative.")
    else:
        print(f"\nOverall: One or more assumptions FAILED � interpret mixed ANOVA results cautiously.")
    print("=" * 80)

    # -- Text report -----------------------------------------------------------
    if save_report:
        def _fmt_sum(label, n_obs, resid_mean_val, sw_p, sw_pass, lev_p, lev_pass,
                     vr, vr_pass, cross_corr, cross_corr_pass, n_iqr, n_stud, max_stud,
                     n_cooks_d, max_cooks_d, cooks_thresh, r2, rmse):
            L = ["-" * 50,
                 f"ASSUMPTION SUMMARY � {label}", f"  n observations: {n_obs}",
                 f"  R�: {r2:.4f}  |  RMSE: {rmse:.4f}",
                 f"  Residual mean � 0:  {resid_mean_val:.6f}",
                 f"  Normality:          {'PASSED' if sw_pass else 'FAILED'} (SW p={sw_p:.4f})",
                 f"  Homoscedasticity:   {'PASSED' if lev_pass else 'FAILED/N/A'} (Levene p={lev_p:.4f})"]
            _vr_str = f"{vr:.3f}" if not _np.isnan(vr) else "N/A"
            L.append(f"  Variance ratio:     {'PASSED' if vr_pass else 'FAILED'} "
                     f"(max/min = {_vr_str}{'  ? > 4: mixed ANOVA may be inappropriate' if not vr_pass else ''})")
            _cc_str = f"{cross_corr:.4f}" if not _np.isnan(cross_corr) else "N/A (unbalanced)"
            L.append(f"  Independence:       {'PASSED' if cross_corr_pass else 'CONCERN'} (mean |r| = {_cc_str})")
            L.append(f"  Outliers (IQR):     {n_iqr} flagged {'?' if n_iqr == 0 else '?'}")
            _ms = f"{max_stud:.3f}" if not _np.isnan(max_stud) else "N/A"
            L.append(f"  Outliers (|t*|>3):  {n_stud} flagged {'?' if n_stud == 0 else '?'} (max |t*|={_ms})")
            _mc = f"{max_cooks_d:.4f}" if not _np.isnan(max_cooks_d) else "N/A"
            _ct = f"{cooks_thresh:.4f}" if not _np.isnan(cooks_thresh) else "4/n"
            L.append(f"  Cook's D (>{_ct}):  {n_cooks_d} influential "
                     f"{'?' if n_cooks_d == 0 else '?'} (max D={_mc})")
            _all = sw_pass and lev_pass and cross_corr_pass and vr_pass
            if _all:
                L.append("Overall: All OLS assumptions met � mixed ANOVA is appropriate.")
            elif not vr_pass:
                L.append("Overall: Variance ratio > 4 � mixed ANOVA results should be interpreted with caution.")
            else:
                L.append("Overall: One or more assumptions FAILED � interpret mixed ANOVA results cautiously.")
            return "\n".join(L)

        _r2_full = float(model.rsquared)
        _rmse_full = float(_np.sqrt(_np.mean(residuals**2)))
        _full_sum = _fmt_sum(
            label="FULL MODEL", n_obs=n, resid_mean_val=resid_mean,
            sw_p=normality.get('p', float('nan')), sw_pass=normality.get('passed', False),
            lev_p=homoscedasticity.get('p', float('nan')), lev_pass=homoscedasticity.get('passed', False),
            vr=variance_ratio_info.get('ratio', float('nan')), vr_pass=variance_ratio_info.get('passed', True),
            cross_corr=_cross_corr, cross_corr_pass=independence.get('passed', False),
            n_iqr=outlier_info['iqr']['n_outliers'],
            n_stud=outlier_info['studentized'].get('n_outliers', 0),
            max_stud=outlier_info['studentized'].get('max_abs', float('nan')),
            n_cooks_d=influential_info.get('n_influential', 0),
            max_cooks_d=influential_info.get('max_cooks_d', float('nan')),
            cooks_thresh=influential_info.get('threshold', 4.0 / n if n > 0 else float('nan')),
            r2=_r2_full, rmse=_rmse_full,
        )

        _rpt = [
            "=" * 80,
            "OLS ASSUMPTION CHECK REPORT � CROSS-COHORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Measure: {measure}",
            f"Model: Value ~ C(Week) + C(Cohort) + C(ID)   [n={n}, subjects={n_subjects}]",
            f"Cohorts: {cohort_levels}",
            f"Weeks: {week_levels}",
            "=" * 80, "",
            _full_sum,
        ]

        _si = sensitivity_info
        if _si.get('available'):
            _red_sum = _fmt_sum(
                label=f"REDUCED MODEL (n={_si['n_reduced']}, {_si['n_removed']} outlier(s) removed)",
                n_obs=_si['n_reduced'], resid_mean_val=_si.get('resid_mean_reduced', float('nan')),
                sw_p=_si['sw_p_reduced'], sw_pass=_si.get('sw_pass_reduced', _si['sw_p_reduced'] > 0.05),
                lev_p=_si['levene_p_reduced'], lev_pass=_si.get('lev_pass_reduced', _si['levene_p_reduced'] > 0.05),
                vr=_si.get('vr_reduced', float('nan')), vr_pass=_si.get('vr_pass_reduced', True),
                cross_corr=_si.get('cross_corr_reduced', float('nan')),
                cross_corr_pass=_si.get('indep_pass_reduced', False),
                n_iqr=_si.get('n_iqr_reduced', 0),
                n_stud=_si.get('n_stud_reduced', 0), max_stud=_si.get('max_stud_reduced', float('nan')),
                n_cooks_d=_si.get('n_cooks_reduced', 0), max_cooks_d=_si.get('max_cooks_reduced', float('nan')),
                cooks_thresh=_si.get('cooks_thresh_reduced',
                                     4.0 / _si['n_reduced'] if _si['n_reduced'] > 0 else float('nan')),
                r2=_si['r2_reduced'], rmse=_si['rmse_reduced'],
            )
            _rpt.extend([
                "", "=" * 80, "SENSITIVITY ANALYSIS � REDUCED MODEL",
                f"Removed {_si['n_removed']} studentized outlier(s) (|t*| > 3)",
                f"Removed subjects: {', '.join(_si['removed_subjects'])}",
                "=" * 80, "",
                _red_sum, "",
                "-" * 60, "COMPARISON TABLE (Full vs Reduced)",
                f"  {'Metric':<26} {'Full model':>12}  {'Reduced model':>14}  {'?':>10}",
                "  " + "-" * 66,
                f"  {'n':<26} {n:>12d}  {_si['n_reduced']:>14d}  {_si['n_reduced']-n:>+10d}",
                f"  {'R�':<26} {_si['r2_full']:>12.4f}  {_si['r2_reduced']:>14.4f}  {_si['delta_r2']:>+10.4f}",
                f"  {'RMSE':<26} {_si['rmse_full']:>12.4f}  {_si['rmse_reduced']:>14.4f}  {_si['delta_rmse']:>+10.4f}",
                f"  {'SW normality p':<26} {_si['sw_p_full']:>12.4f}  {_si['sw_p_reduced']:>14.4f}  {_si['sw_p_reduced']-_si['sw_p_full']:>+10.4f}",
                f"  {'Levene p':<26} {_si['levene_p_full']:>12.4f}  {_si['levene_p_reduced']:>14.4f}  {_si['levene_p_reduced']-_si['levene_p_full']:>+10.4f}",
            ])
            if not (_np.isnan(_si.get('cross_corr_full', float('nan'))) or
                    _np.isnan(_si.get('cross_corr_reduced', float('nan')))):
                _rpt.append(
                    f"  {'Cross-subj mean |r|':<26} {_si['cross_corr_full']:>12.4f}"
                    f"  {_si['cross_corr_reduced']:>14.4f}  {_si['cross_corr_reduced']-_si['cross_corr_full']:>+10.4f}"
                )
            if not _np.isnan(_si.get('f_week_full', float('nan'))):
                _rpt.append(
                    f"  {'F (Week)':<26} {_si['f_week_full']:>12.3f}"
                    f"  {_si['f_week_reduced']:>14.3f}  {_si['f_week_reduced']-_si['f_week_full']:>+10.3f}"
                )
            _dr2 = _si['delta_r2']
            if abs(_dr2) > 0.05:
                _rpt.append(f"\nWARNING: R� changed by {_dr2:+.4f} � outliers substantially affect model fit.")
            else:
                _rpt.append(f"\nModel is robust to removal of outliers (?R� = {_dr2:+.4f}).")
        else:
            _rpt.extend(["", f"SENSITIVITY ANALYSIS: {_si.get('reason', 'not run')}"])

        _rpt.extend(["", "=" * 80, "END OF REPORT", "=" * 80, ""])
        _rpt_text = "\n".join(_rpt)
        _rpt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _rpt_path = Path.cwd() / f"ols_cross_cohort_{measure.replace(' ', '_')}_{_rpt_ts}.txt"
        try:
            _rpt_path.write_text(_rpt_text, encoding='utf-8')
            print(f"\nOLS assumption report saved to: {_rpt_path}")
        except Exception as _re:
            print(f"\nWarning: could not save OLS report: {_re}")

    return {
        'n': n,
        'n_subjects': n_subjects,
        'week_levels': week_levels,
        'cohort_levels': cohort_levels,
        'residual_mean': resid_mean,
        'normality': normality,
        'homoscedasticity': homoscedasticity,
        'variance_ratio': variance_ratio_info,
        'independence': independence,
        'cross_corr': _cross_corr,
        'outliers': outlier_info,
        'influential': influential_info,
        'sensitivity_analysis': sensitivity_info,
        'figure': fig,
        'figure_reduced': fig_reduced,
    }


# =============================================================================
# WEEK-LEVEL MIXED ANOVA FUNCTIONS (CONSERVATIVE, BONFERRONI-CORRECTED)
# =============================================================================

def _add_week_column_across_cohorts(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Week' column (1-indexed) derived from 'Day'.
    Week 1 = Days 1-7, Week 2 = Days 8-14, etc.
    Requires 'Day' column (calls add_day_column_across_cohorts if absent).
    """
    df = combined_df.copy()
    if 'Day' not in df.columns:
        df = add_day_column_across_cohorts(df)
    df['Week'] = (df['Day'] - 1) // 7 + 1
    return df


def _filter_complete_subjects_weekly(df: pd.DataFrame, subject_col: str, time_col: str) -> pd.DataFrame:
    """
    Keep only subjects that have at least one observation in every level of time_col.
    Prints a summary of how many subjects were dropped.
    """
    counts = df.groupby(subject_col)[time_col].nunique()
    total = df[time_col].nunique()
    complete_ids = counts[counts == total].index
    n_before = df[subject_col].nunique()
    n_dropped = n_before - len(complete_ids)
    if n_dropped > 0:
        print(f"  [NOTE] Excluded {n_dropped} subject(s) with incomplete records; "
              f"{len(complete_ids)} of {n_before} subjects retained")
    else:
        print(f"  [OK] All {n_before} subjects have complete records across all weeks")
    return df[df[subject_col].isin(complete_ids)].copy()


def perform_cross_cohort_mixed_anova_weekly(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
) -> Dict:
    """
    Week-level Mixed ANOVA: Week (within-subjects)  x  Group (between-subjects).

    This is the week-level equivalent of perform_cross_cohort_mixed_anova().
    'Group' is a combined CA%/Sex label (e.g. '0% / M', '2% / F').
    Using Week (1-5) instead of Day is more conservative: it averages out
    day-to-day noise, reduces the within-subjects degrees of freedom, and
    exactly mirrors the design used in non_ramp_analysis.py.

    Only subjects with a complete record across ALL retained weeks are included.

    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        weeks: Optional list of week numbers to include (default: all)

    Returns:
        Dictionary with ANOVA results, descriptive stats, and the analysis DataFrame
    """
    print("\n" + "="*80)
    print("WEEK-LEVEL MIXED ANOVA: Week (WITHIN)  x  Group/CA%+Sex (BETWEEN)")
    print("="*80)

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required for mixed ANOVA")
        return {}

    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    required_cols = ['ID', 'Week', 'Sex', 'CA (%)', measure]
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    analysis_df = combined_df[required_cols].dropna().copy()

    if weeks is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(weeks)]
        print(f"  Filtered to weeks: {sorted(weeks)}")

    # Average within Week per animal (one value per ID  x  Week)
    analysis_df = (
        analysis_df.groupby(['ID', 'Week', 'Sex', 'CA (%)'])[measure]
        .mean()
        .reset_index()
    )

    # Create combined Group label
    analysis_df['Group'] = analysis_df['CA (%)'].astype(str) + '% / ' + analysis_df['Sex']

    print(f"\n  Measure: {measure}")
    print(f"  Animals before completeness filter: {analysis_df['ID'].nunique()}")
    print(f"  Weeks present: {sorted(analysis_df['Week'].unique())}")

    analysis_df = _filter_complete_subjects_weekly(analysis_df, 'ID', 'Week')

    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Groups: {sorted(analysis_df['Group'].unique())}")

    def _desc(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1) if n > 1 else np.nan
        sem = std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan
        return pd.Series({
            'count': n, 'mean': mean, 'median': group.median(),
            'std': std, 'sem': sem,
            'ci_lower': mean - 1.96 * sem if np.isfinite(sem) else np.nan,
            'ci_upper': mean + 1.96 * sem if np.isfinite(sem) else np.nan,
            'min': group.min(), 'max': group.max(),
        })

    stats_data = []
    for (ca_val, sex_val), grp in analysis_df.groupby(['CA (%)', 'Sex'])[measure]:
        s = _desc(grp)
        s['CA (%)'] = ca_val
        s['Sex'] = sex_val
        stats_data.append(s)
    group_stats = pd.DataFrame(stats_data)

    print("\nDescriptive Statistics:")
    for _, row in group_stats.iterrows():
        print(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
              f"n={int(row['count'])}, M={row['mean']:.3f}, SD={row['std']:.3f}, SEM={row['sem']:.3f}")

    print("\nRunning mixed ANOVA (Week within  x  Group between)...")
    try:
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='Group',
            correction='auto',
        )
        aov['Source'] = aov['Source'].replace({'Interaction': 'Week * Group'})

        print("\nMixed ANOVA Results:")
        print(aov.to_string())

        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        print("\nFormatted Results:")
        for _, row in aov.iterrows():
            p = row[p_col]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {row['Source']}: F = {row['F']:.3f}, p = {p:.4f} {sig}")

        return {
            'measure': measure,
            'type': 'mixed_anova_weekly',
            'n_subjects': analysis_df['ID'].nunique(),
            'n_weeks': analysis_df['Week'].nunique(),
            'n_observations': len(analysis_df),
            'weeks': sorted(analysis_df['Week'].unique()),
            'anova_table': aov,
            'descriptive_stats': group_stats,
            'data': analysis_df,
        }
    except Exception as e:
        print(f"[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def perform_mixed_anova_sex_stratified_weekly(
    cohort_dfs: Dict[str, pd.DataFrame],
    sex: str,
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
) -> Dict:
    """
    Week-level Mixed ANOVA: Week (within-subjects)  x  CA% (between-subjects),
    stratified by sex.

    Week-level equivalent of perform_mixed_anova_sex_stratified().
    Only subjects with a complete record across all retained weeks are included.

    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        sex: Sex to analyze ('M' or 'F')
        measure: Weight measure to analyze
        weeks: Optional list of week numbers to include (default: all)

    Returns:
        Dictionary with ANOVA results
    """
    sex_label = 'MALES' if sex == 'M' else 'FEMALES'
    print("\n" + "="*80)
    print(f"WEEK-LEVEL SEX-STRATIFIED MIXED ANOVA ({sex_label}): Week (WITHIN)  x  CA% (BETWEEN)")
    print("="*80)

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required for mixed ANOVA")
        return {}

    if sex not in ["M", "F"]:
        raise ValueError("sex must be 'M' or 'F'")

    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    required_cols = ['ID', 'Week', 'Sex', 'CA (%)', measure]
    analysis_df = combined_df[combined_df['Sex'] == sex][required_cols].dropna().copy()

    if len(analysis_df) == 0:
        print(f"[ERROR] No data for sex={sex}")
        return {}

    if weeks is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(weeks)]

    # Average within Week per animal
    analysis_df = (
        analysis_df.groupby(['ID', 'Week', 'CA (%)'])[measure]
        .mean()
        .reset_index()
    )

    print(f"\n  Measure: {measure} ({'Males' if sex == 'M' else 'Females'})")
    print(f"  Animals before completeness filter: {analysis_df['ID'].nunique()}")

    analysis_df = _filter_complete_subjects_weekly(analysis_df, 'ID', 'Week')

    print(f"  CA% levels: {sorted(analysis_df['CA (%)'].unique())}")
    print(f"  Weeks: {sorted(analysis_df['Week'].unique())}")
    print(f"  Total observations: {len(analysis_df)}")

    def _desc(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1) if n > 1 else np.nan
        sem = std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan
        return pd.Series({
            'count': n, 'mean': mean, 'median': group.median(),
            'std': std, 'sem': sem,
            'ci_lower': mean - 1.96 * sem if np.isfinite(sem) else np.nan,
            'ci_upper': mean + 1.96 * sem if np.isfinite(sem) else np.nan,
            'min': group.min(), 'max': group.max(),
        })

    stats_data = []
    for ca_val, grp in analysis_df.groupby('CA (%)')[measure]:
        s = _desc(grp)
        s['CA (%)'] = ca_val
        stats_data.append(s)
    group_stats = pd.DataFrame(stats_data)

    print("\nDescriptive Statistics:")
    for _, row in group_stats.iterrows():
        print(f"  CA%={row['CA (%)']}: "
              f"n={int(row['count'])}, M={row['mean']:.3f}, SD={row['std']:.3f}, SEM={row['sem']:.3f}")

    print(f"\nRunning mixed ANOVA (Week within  x  CA% between)...")
    try:
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='CA (%)',
            correction='auto',
        )
        aov['Source'] = aov['Source'].replace({'Interaction': 'CA (%) * Week'})

        print("\nMixed ANOVA Results:")
        print(aov.to_string())

        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        print("\nFormatted Results:")
        for _, row in aov.iterrows():
            p = row[p_col]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {row['Source']}: F = {row['F']:.3f}, p = {p:.4f} {sig}")

        return {
            'measure': measure,
            'sex': sex,
            'type': 'mixed_anova_sex_stratified_weekly',
            'n_subjects': analysis_df['ID'].nunique(),
            'n_weeks': analysis_df['Week'].nunique(),
            'n_observations': len(analysis_df),
            'weeks': sorted(analysis_df['Week'].unique()),
            'anova_table': aov,
            'descriptive_stats': group_stats,
            'data': analysis_df,
        }
    except Exception as e:
        print(f"[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def perform_mixed_anova_ca_stratified_weekly(
    cohort_dfs: Dict[str, pd.DataFrame],
    ca_percent: float,
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
) -> Dict:
    """
    Week-level Mixed ANOVA: Week (within-subjects)  x  Sex (between-subjects),
    stratified by CA%.

    Week-level equivalent of perform_mixed_anova_ca_stratified().
    Only subjects with a complete record across all retained weeks are included.

    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        ca_percent: CA% level to analyze (e.g., 0 or 2)
        measure: Weight measure to analyze
        weeks: Optional list of week numbers to include (default: all)

    Returns:
        Dictionary with ANOVA results
    """
    print("\n" + "="*80)
    print(f"WEEK-LEVEL CA%-STRATIFIED MIXED ANOVA ({ca_percent}% CA): Week (WITHIN)  x  Sex (BETWEEN)")
    print("="*80)

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required for mixed ANOVA")
        return {}

    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    required_cols = ['ID', 'Week', 'Sex', 'CA (%)', measure]
    analysis_df = combined_df[combined_df['CA (%)'] == ca_percent][required_cols].dropna().copy()

    if len(analysis_df) == 0:
        print(f"[ERROR] No data for CA%={ca_percent}")
        return {}

    if weeks is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(weeks)]

    # Average within Week per animal
    analysis_df = (
        analysis_df.groupby(['ID', 'Week', 'Sex'])[measure]
        .mean()
        .reset_index()
    )

    print(f"\n  Measure: {measure} ({ca_percent}% CA)")
    print(f"  Animals before completeness filter: {analysis_df['ID'].nunique()}")

    analysis_df = _filter_complete_subjects_weekly(analysis_df, 'ID', 'Week')

    print(f"  Sex groups: {sorted(analysis_df['Sex'].unique())}")
    print(f"  Weeks: {sorted(analysis_df['Week'].unique())}")
    print(f"  Total observations: {len(analysis_df)}")

    def _desc(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1) if n > 1 else np.nan
        sem = std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan
        return pd.Series({
            'count': n, 'mean': mean, 'median': group.median(),
            'std': std, 'sem': sem,
            'ci_lower': mean - 1.96 * sem if np.isfinite(sem) else np.nan,
            'ci_upper': mean + 1.96 * sem if np.isfinite(sem) else np.nan,
            'min': group.min(), 'max': group.max(),
        })

    stats_data = []
    for sex_val, grp in analysis_df.groupby('Sex')[measure]:
        s = _desc(grp)
        s['Sex'] = sex_val
        stats_data.append(s)
    group_stats = pd.DataFrame(stats_data)

    print("\nDescriptive Statistics:")
    for _, row in group_stats.iterrows():
        print(f"  Sex={row['Sex']}: "
              f"n={int(row['count'])}, M={row['mean']:.3f}, SD={row['std']:.3f}, SEM={row['sem']:.3f}")

    print(f"\nRunning mixed ANOVA (Week within  x  Sex between)...")
    try:
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='Sex',
            correction='auto',
        )
        aov['Source'] = aov['Source'].replace({'Interaction': 'Sex * Week'})

        print("\nMixed ANOVA Results:")
        print(aov.to_string())

        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
        print("\nFormatted Results:")
        for _, row in aov.iterrows():
            p = row[p_col]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {row['Source']}: F = {row['F']:.3f}, p = {p:.4f} {sig}")

        return {
            'measure': measure,
            'ca_percent': ca_percent,
            'type': 'mixed_anova_ca_stratified_weekly',
            'n_subjects': analysis_df['ID'].nunique(),
            'n_weeks': analysis_df['Week'].nunique(),
            'n_observations': len(analysis_df),
            'weeks': sorted(analysis_df['Week'].unique()),
            'anova_table': aov,
            'descriptive_stats': group_stats,
            'data': analysis_df,
        }
    except Exception as e:
        print(f"[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_cross_cohort_report_weekly(
    mixed_results: Optional[Dict] = None,
    results_males: Optional[Dict] = None,
    results_females: Optional[Dict] = None,
    results_ca0: Optional[Dict] = None,
    results_ca2: Optional[Dict] = None,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    include_preamble: bool = True,
    include_footer: bool = True,
) -> str:
    """
    Generate a report for week-level cross-cohort analyses with Bonferroni corrections.

    Uses Week (1-5) as the within-subjects factor rather than individual days.
    This matches the design in non_ramp_analysis.py and is more conservative:
    - Daily autocorrelation is averaged out within each animal-week
    - Within-subjects degrees of freedom are greatly reduced (4 vs ~34)
    - Only complete-record subjects are included

    Bonferroni corrections applied across stratified test families:
      - Sex-stratified (males + females, k=2): alpha_adj = 0.025
        Corrects the Week  x  CA% interaction p-values
      - CA%-stratified (0% + 2% CA, k=2): alpha_adj = 0.025
        Corrects the Week  x  Sex interaction p-values

    Parameters:
        mixed_results: Results from perform_cross_cohort_mixed_anova_weekly()
        results_males: Results from perform_mixed_anova_sex_stratified_weekly() for males
        results_females: Results from perform_mixed_anova_sex_stratified_weekly() for females
        results_ca0: Results from perform_mixed_anova_ca_stratified_weekly() for 0% CA
        results_ca2: Results from perform_mixed_anova_ca_stratified_weekly() for 2% CA
        cohort_dfs: Original cohort DataFrames for context

    Returns:
        Formatted string report
    """
    lines = []

    # Determine the measure label for the per-measure section banner
    _measure_label = "Unknown"
    for _r in (mixed_results, results_males, results_females, results_ca0, results_ca2):
        if _r and _r.get('measure'):
            _measure_label = _r['measure']
            break

    def h1(text):
        lines.append("=" * 80)
        lines.append(text)
        lines.append("=" * 80)

    def h2(text):
        lines.append(text)
        lines.append("-" * 80)

    def sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def bonferroni(p, k):
        return min(1.0, p * k)

    # =========================================================================
    # Preamble (title, methodology, study design) -- printed once per report
    # =========================================================================
    if include_preamble:
        h1("CROSS-COHORT STATISTICAL ANALYSIS REPORT -- WEEK-LEVEL (CONSERVATIVE)")
        lines.append("")
    else:
        # Inter-measure separator when preamble is suppressed
        lines.append("\n" + "#" * 80)
        lines.append("# NEW MEASURE SECTION (continued in same report)")
        lines.append("#" * 80 + "\n")

    # Per-measure banner (always shown)
    lines.append("=" * 80)
    lines.append(f"MEASURE: {_measure_label}")
    lines.append("=" * 80)
    lines.append("")

    if include_preamble:
        lines.append("METHODOLOGICAL NOTES")
        lines.append("-" * 80)
        lines.append("")
        lines.append("This report uses a WEEK-LEVEL design to address two limitations of the")
        lines.append("day-level (daily) analyses:")
        lines.append("")
        lines.append("  1. Inflated power from day-level resolution: With ~35 daily measurements")
        lines.append("     as the within-subjects factor, the F-test for Time  x  Sex or Time  x  CA%")
        lines.append("     interactions has very high degrees of freedom, detecting even transient")
        lines.append("     day-to-day noise as 'significant'. Using Week (5 levels) drastically")
        lines.append("     reduces within-subjects df and averages out day-to-day variability,")
        lines.append("     mirroring the approach in non_ramp_analysis.py.")
        lines.append("")
        lines.append("  2. Multiple testing across stratified models: Running separate ANOVAs for")
        lines.append("     each sex (k=2) and each CA% level (k=2) inflates the family-wise error")
        lines.append("     rate. Bonferroni corrections are applied within each family:")
        lines.append("       \u2022 Sex-stratified family (Males + Females, k=2): \u03b1_adj = 0.025")
        lines.append("         Corrected p = min(1.0, raw_p \u00d7 2)")
        lines.append("       \u2022 CA%-stratified family (0% + 2% CA, k=2): \u03b1_adj = 0.025")
        lines.append("         Corrected p = min(1.0, raw_p \u00d7 2)")
        lines.append("")
        lines.append("  3. Complete-subject design: Animals missing data in any week are excluded,")
        lines.append("     ensuring a balanced repeated-measures design.")
        lines.append("")

    # -------------------------------------------------------------------------
    # Study design summary (only with preamble)
    if include_preamble and cohort_dfs is not None:
        h1("STUDY DESIGN")
        lines.append("")
        total_animals = 0
        for label, df in cohort_dfs.items():
            n = df['ID'].nunique() if 'ID' in df.columns else 0
            total_animals += n
            lines.append(f"  {label}: {n} animals")
        lines.append(f"\n  Total animals: {total_animals}")

        try:
            combined = combine_cohorts_for_analysis(cohort_dfs)
            combined = clean_cohort(combined)
            if 'Sex' in combined.columns:
                sex_counts = combined.groupby('Sex')['ID'].nunique()
                lines.append("\n  Sex distribution:")
                for sex, count in sex_counts.items():
                    lines.append(f"    {sex}: {count} animals")
            if 'CA (%)' in combined.columns:
                ca_counts = combined.groupby('CA (%)')['ID'].nunique()
                lines.append("\n  CA% distribution:")
                for ca, count in ca_counts.items():
                    lines.append(f"    {ca}%: {count} animals")
        except Exception:
            pass

        lines.append("\n  Time factor: Week (1-5; one averaged value per animal per week)")
        lines.append("  Between-subjects factors: CA% and Sex")
        lines.append("  Incomplete subjects excluded (complete-subject design)")
        lines.append("")

    # -------------------------------------------------------------------------
    # Omnibus mixed ANOVA
    if mixed_results is not None:
        h1("OMNIBUS WEEK-LEVEL MIXED ANOVA: Week (WITHIN)  x  Group (BETWEEN)")
        lines.append("")
        lines.append(f"  Measure         : {mixed_results.get('measure', 'Unknown')}")
        lines.append(f"  Subjects        : {mixed_results.get('n_subjects', 'Unknown')}")
        lines.append(f"  Weeks included  : {mixed_results.get('weeks', 'Unknown')}")
        lines.append(f"  Observations    : {mixed_results.get('n_observations', 'Unknown')}")
        lines.append(f"  Note: Group = CA%  x  Sex combined label")
        lines.append("")

        aov = mixed_results.get('anova_table')
        if aov is not None:
            h2("ANOVA Table")
            lines.append(aov.to_string())
            lines.append("")

            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            h2("Interpretation")
            lines.append("")
            for i, (_, row) in enumerate(aov.iterrows(), start=1):
                p = row[p_col]
                sig = sig_stars(p)
                sig_str = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
                lines.append(f"  {i}. {row['Source']}: {sig_str}")
                lines.append(f"     F = {row['F']:.3f}, p = {p:.4f} {sig}")
                if row['Source'] == 'Week':
                    lines.append("     -> Weight measure changes significantly across weeks" if p < 0.05
                                 else "     -> No significant week-to-week change overall")
                elif row['Source'] == 'Group':
                    lines.append("     -> Groups (CA%/Sex combinations) differ overall" if p < 0.05
                                 else "     -> No significant difference between groups overall")
                elif 'Week' in row['Source'] and 'Group' in row['Source']:
                    lines.append("     -> Trajectories differ between groups (different week-to-week patterns)" if p < 0.05
                                 else "     -> All groups follow similar week-to-week patterns")
                lines.append("")

        # Descriptive stats
        desc = mixed_results.get('descriptive_stats')
        if desc is not None and len(desc) > 0:
            h2("Descriptive Statistics by Group")
            for _, row in desc.iterrows():
                lines.append(f"  CA%={row['CA (%)']}, Sex={row['Sex']}: "
                             f"n={int(row['count'])}, M={row['mean']:.3f}, "
                             f"SD={row['std']:.3f}, SEM={row['sem']:.3f}, "
                             f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            lines.append("")

    # -------------------------------------------------------------------------
    # Helper: format a single stratified result block
    def _format_stratified_result(result_dict, stratum_label, within_label,
                                  between_label, interaction_source,
                                  bonferroni_k, bonferroni_family_label):
        if not result_dict:
            lines.append(f"  [No results available for {stratum_label}]")
            lines.append("")
            return None  # return interaction p for summary table

        aov = result_dict.get('anova_table')
        lines.append(f"  Measure         : {result_dict.get('measure', 'Unknown')}")
        lines.append(f"  Subjects        : {result_dict.get('n_subjects', 'Unknown')}")
        lines.append(f"  Weeks included  : {result_dict.get('weeks', 'Unknown')}")
        lines.append(f"  Observations    : {result_dict.get('n_observations', 'Unknown')}")
        lines.append("")

        if aov is not None:
            h2("ANOVA Table")
            lines.append(aov.to_string())
            lines.append("")

            p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
            h2("Interpretation (with Bonferroni correction)")
            lines.append(f"  Bonferroni family: {bonferroni_family_label} (k={bonferroni_k}, alpha_adj = {0.05/bonferroni_k:.3f})")
            lines.append("")

            int_p_raw = None
            for i, (_, row) in enumerate(aov.iterrows(), start=1):
                p_raw = row[p_col]
                sig_raw = sig_stars(p_raw)
                src = row['Source']

                # Apply Bonferroni only to the interaction term
                if src == interaction_source:
                    p_adj = bonferroni(p_raw, bonferroni_k)
                    int_p_raw = p_raw
                    sig_adj = sig_stars(p_adj)
                    sig_str_adj = "SIGNIFICANT" if p_adj < 0.05 else "NOT SIGNIFICANT"
                    lines.append(f"  {i}. {src}:")
                    lines.append(f"     Raw p = {p_raw:.4f} ({sig_raw})")
                    lines.append(f"     Bonferroni-corrected p = {p_adj:.4f} ({sig_adj})  <- USE THIS")
                    lines.append(f"     Decision: {sig_str_adj} after correction")
                    if src.startswith('CA') and 'Week' in src:
                        lines.append("     -> CA% levels follow different weekly trajectories" if p_adj < 0.05
                                     else "     -> Similar weekly trajectories across CA% levels")
                    elif src.startswith('Sex') and 'Week' in src:
                        lines.append("     -> Males and females follow different weekly trajectories" if p_adj < 0.05
                                     else "     -> Similar weekly trajectories for both sexes")
                else:
                    sig_str = "SIGNIFICANT" if p_raw < 0.05 else "NOT SIGNIFICANT"
                    lines.append(f"  {i}. {src}: {sig_str}")
                    lines.append(f"     F = {row['F']:.3f}, p = {p_raw:.4f} {sig_raw}")
                    if src == 'Week':
                        lines.append("     -> Weight changes significantly across weeks" if p_raw < 0.05
                                     else "     -> No significant week-to-week change")
                    elif src in ('Sex', 'CA (%)'):
                        lines.append("     -> Groups differ overall (averaged across weeks)" if p_raw < 0.05
                                     else "     -> No significant group difference overall")
                lines.append("")

        # Descriptive stats
        desc = result_dict.get('descriptive_stats')
        if desc is not None and len(desc) > 0:
            h2("Descriptive Statistics")
            for _, row in desc.iterrows():
                grp_cols = [c for c in ['CA (%)', 'Sex'] if c in row.index]
                grp_str = ', '.join(f"{c}={row[c]}" for c in grp_cols)
                lines.append(f"  {grp_str}: "
                             f"n={int(row['count'])}, M={row['mean']:.3f}, "
                             f"SD={row['std']:.3f}, SEM={row['sem']:.3f}, "
                             f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            lines.append("")

        return int_p_raw

    # -------------------------------------------------------------------------
    # Sex-stratified results (k=2 family)
    h1("SEX-STRATIFIED WEEK-LEVEL MIXED ANOVA: Week (WITHIN)  x  CA% (BETWEEN)")
    lines.append("")
    lines.append("These two models form a Bonferroni family (k=2).")
    lines.append("The Week  x  CA% interaction p-values are multiplied by 2 before")
    lines.append("comparing to alpha = 0.05 (equivalent to requiring p_raw < 0.025).")
    lines.append("")

    lines.append("|" * 80)
    lines.append("Males Only")
    lines.append("|" * 80)
    lines.append("")
    males_int_p_raw = _format_stratified_result(
        results_males, "Males", "Week", "CA (%)", "CA (%) * Week",
        bonferroni_k=2, bonferroni_family_label="Males + Females"
    )

    lines.append("|" * 80)
    lines.append("Females Only")
    lines.append("|" * 80)
    lines.append("")
    females_int_p_raw = _format_stratified_result(
        results_females, "Females", "Week", "CA (%)", "CA (%) * Week",
        bonferroni_k=2, bonferroni_family_label="Males + Females"
    )

    # Sex-stratified comparison summary
    if males_int_p_raw is not None and females_int_p_raw is not None:
        h2("Sex-Stratified Comparison Summary (Week  x  CA% Interaction)")
        lines.append("")
        m_adj = bonferroni(males_int_p_raw, 2)
        f_adj = bonferroni(females_int_p_raw, 2)
        lines.append(f"  Males   : raw p = {males_int_p_raw:.4f},  Bonferroni p = {m_adj:.4f}"
                     f"  {'<- SIGNIFICANT' if m_adj < 0.05 else '<- not significant'}")
        lines.append(f"  Females : raw p = {females_int_p_raw:.4f},  Bonferroni p = {f_adj:.4f}"
                     f"  {'<- SIGNIFICANT' if f_adj < 0.05 else '<- not significant'}")
        lines.append("")
        m_sig = m_adj < 0.05
        f_sig = f_adj < 0.05
        if m_sig and f_sig:
            lines.append("  -> BOTH sexes: week  x  CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in both males and females")
        elif m_sig and not f_sig:
            lines.append("  -> MALES ONLY: week  x  CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in males but not females")
        elif not m_sig and f_sig:
            lines.append("  -> FEMALES ONLY: week  x  CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in females but not males")
        else:
            lines.append("  -> NEITHER sex: week  x  CA% interaction significant after correction")
            lines.append("    No differential CA% effect on weekly trajectories in either sex")
        lines.append("")

    # -------------------------------------------------------------------------
    # CA%-stratified results (k=2 family)
    h1("CA%-STRATIFIED WEEK-LEVEL MIXED ANOVA: Week (WITHIN)  x  Sex (BETWEEN)")
    lines.append("")
    lines.append("These two models form a Bonferroni family (k=2).")
    lines.append("The Week  x  Sex interaction p-values are multiplied by 2 before")
    lines.append("comparing to alpha = 0.05 (equivalent to requiring p_raw < 0.025).")
    lines.append("")

    lines.append("|" * 80)
    lines.append("0% CA Only")
    lines.append("|" * 80)
    lines.append("")
    ca0_int_p_raw = _format_stratified_result(
        results_ca0, "0% CA", "Week", "Sex", "Sex * Week",
        bonferroni_k=2, bonferroni_family_label="0% CA + 2% CA"
    )

    lines.append("|" * 80)
    lines.append("2% CA Only")
    lines.append("|" * 80)
    lines.append("")
    ca2_int_p_raw = _format_stratified_result(
        results_ca2, "2% CA", "Week", "Sex", "Sex * Week",
        bonferroni_k=2, bonferroni_family_label="0% CA + 2% CA"
    )

    # CA%-stratified comparison summary
    if ca0_int_p_raw is not None and ca2_int_p_raw is not None:
        h2("CA%-Stratified Comparison Summary (Week  x  Sex Interaction)")
        lines.append("")
        c0_adj = bonferroni(ca0_int_p_raw, 2)
        c2_adj = bonferroni(ca2_int_p_raw, 2)
        lines.append(f"  0% CA : raw p = {ca0_int_p_raw:.4f},  Bonferroni p = {c0_adj:.4f}"
                     f"  {'<- SIGNIFICANT' if c0_adj < 0.05 else '<- not significant'}")
        lines.append(f"  2% CA : raw p = {ca2_int_p_raw:.4f},  Bonferroni p = {c2_adj:.4f}"
                     f"  {'<- SIGNIFICANT' if c2_adj < 0.05 else '<- not significant'}")
        lines.append("")
        c0_sig = c0_adj < 0.05
        c2_sig = c2_adj < 0.05
        if c0_sig and c2_sig:
            lines.append("  -> BOTH CA% levels: week  x  sex interaction significant after correction")
            lines.append("    Males and females follow different weekly trajectories at both CA% levels")
        elif c0_sig and not c2_sig:
            lines.append("  -> 0% CA ONLY: week  x  sex interaction significant after correction")
            lines.append("    Sex differences in weekly trajectory exist only in 0% CA animals")
        elif not c0_sig and c2_sig:
            lines.append("  -> 2% CA ONLY: week  x  sex interaction significant after correction")
            lines.append("    Sex differences in weekly trajectory exist only in 2% CA animals")
        else:
            lines.append("  -> NEITHER CA% level: week  x  sex interaction significant after correction")
            lines.append("    The day-level significance in 0% CA does not survive at the week level")
            lines.append("    after Bonferroni correction, suggesting it was driven by day-to-day noise")
        lines.append("")

    # -------------------------------------------------------------------------
    # Master summary table
    h1("SUMMARY TABLE -- ALL KEY INTERACTION P-VALUES")
    lines.append("")
    lines.append("  All interaction terms from the stratified week-level models.")
    lines.append("  Bonferroni k=2 within each family; * = significant at alpha=0.05 after correction.")
    lines.append("")
    lines.append(f"  {'Analysis':<45} {'Raw p':>9}  {'Bonf. p':>9}  {'Decision'}")
    lines.append("  " + "-" * 76)

    def _summary_row(label, p_raw, k):
        if p_raw is None:
            lines.append(f"  {label:<45} {'N/A':>9}  {'N/A':>9}  N/A")
            return
        p_adj = bonferroni(p_raw, k)
        decision = "SIGNIFICANT *" if p_adj < 0.05 else "not significant"
        lines.append(f"  {label:<45} {p_raw:>9.4f}  {p_adj:>9.4f}  {decision}")

    _summary_row("Week  x  CA% (Males)",      males_int_p_raw,   2)
    _summary_row("Week  x  CA% (Females)",    females_int_p_raw, 2)
    _summary_row("Week  x  Sex  (0% CA)",     ca0_int_p_raw,     2)
    _summary_row("Week  x  Sex  (2% CA)",     ca2_int_p_raw,     2)
    lines.append("")

    # -------------------------------------------------------------------------
    if include_footer:
        h1("END OF WEEK-LEVEL REPORT")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# 2-WAY MIXED ANOVA: CA%  x  WEEK (SEX COLLAPSED) WITH POST-HOC AND SPHERICITY
# =============================================================================

def perform_mixed_anova_ca_x_week(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
) -> Dict:
    """
    2-Way Mixed ANOVA: CA% (between-subjects)  x  Week (within-subjects), sex collapsed.

    This collapses across sex to give the clearest view of the CA%  x  time interaction
    with maximum power. Includes:
      - Mauchly's sphericity test for the Week within-subjects factor      - Greenhouse-Geisser epsilon and corrected p-values when sphericity is violated
      - Post-hoc pairwise comparisons (Bonferroni-adjusted paired t-tests) for Week
      - Simple-effects post-hoc when CA%  x  Week interaction is significant:
          * CA% comparison at each Week (independent samples, Bonferroni across weeks)
          * Week pairwise comparisons within each CA% group (paired, Bonferroni)

    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change')
        weeks: Optional list of week numbers to include (default: all)

    Returns:
        Dictionary with ANOVA table, sphericity info, post-hoc results, and descriptives
    """
    print("\n" + "="*80)
    print("2-WAY MIXED ANOVA: CA% (BETWEEN)  x  WEEK (WITHIN) -- SEX COLLAPSED")
    print("="*80)

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required for mixed ANOVA")
        return {}

    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    required_cols = ['ID', 'Week', 'CA (%)', measure]
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    analysis_df = combined_df[required_cols].dropna().copy()

    if weeks is not None:
        analysis_df = analysis_df[analysis_df['Week'].isin(weeks)]
        print(f"  Filtered to weeks: {sorted(weeks)}")

    # Average within Week per animal (one value per ID  x  Week, sex ignored)
    analysis_df = (
        analysis_df.groupby(['ID', 'Week', 'CA (%)'])[measure]
        .mean()
        .reset_index()
    )

    print(f"\n  Measure: {measure}")
    print(f"  Animals before completeness filter: {analysis_df['ID'].nunique()}")
    print(f"  Weeks present: {sorted(analysis_df['Week'].unique())}")

    analysis_df = _filter_complete_subjects_weekly(analysis_df, 'ID', 'Week')

    n_subjects = analysis_df['ID'].nunique()
    ca_levels = sorted(analysis_df['CA (%)'].unique())
    week_levels = sorted(analysis_df['Week'].unique())
    print(f"  Total subjects: {n_subjects}")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  CA% groups: {ca_levels}")

    # -------------------------------------------------------------------------
    # Descriptive statistics
    def _desc(group):
        n = len(group)
        mean = group.mean()
        std = group.std(ddof=1) if n > 1 else np.nan
        sem = std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan
        return pd.Series({
            'count': n, 'mean': mean, 'std': std, 'sem': sem,
            'ci_lower': mean - 1.96 * sem if np.isfinite(sem) else np.nan,
            'ci_upper': mean + 1.96 * sem if np.isfinite(sem) else np.nan,
        })

    ca_stats_rows = []
    for ca_val, grp in analysis_df.groupby('CA (%)')[measure]:
        s = _desc(grp); s['CA (%)'] = ca_val; ca_stats_rows.append(s)
    ca_stats = pd.DataFrame(ca_stats_rows)

    week_stats_rows = []
    for wk, grp in analysis_df.groupby('Week')[measure]:
        s = _desc(grp); s['Week'] = wk; week_stats_rows.append(s)
    week_stats = pd.DataFrame(week_stats_rows)

    cell_stats_rows = []
    for (ca_val, wk), grp in analysis_df.groupby(['CA (%)', 'Week'])[measure]:
        s = _desc(grp); s['CA (%)'] = ca_val; s['Week'] = wk; cell_stats_rows.append(s)
    cell_stats = pd.DataFrame(cell_stats_rows)

    print("\nDescriptive Statistics by CA%:")
    for _, row in ca_stats.iterrows():
        print(f"  CA%={row['CA (%)']}: n={int(row['count'])}, M={row['mean']:.3f}, "
              f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")

    print("\nDescriptive Statistics by Week:")
    for _, row in week_stats.iterrows():
        print(f"  Week {int(row['Week'])}: n={int(row['count'])}, M={row['mean']:.3f}, "
              f"SD={row['std']:.3f}")

    # -------------------------------------------------------------------------
    # Mixed ANOVA
    print("\nRunning 2-way mixed ANOVA (CA% between  x  Week within)...")
    try:
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            between='CA (%)',
            correction='auto',
        )
        aov['Source'] = aov['Source'].replace({'Interaction': 'CA (%) * Week'})

        print("\nMixed ANOVA Results:")
        print(aov.to_string())

        p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'

        # Extract sphericity info from Week row
        week_row_df = aov[aov['Source'] == 'Week']
        sphericity_info = {}
        if len(week_row_df) > 0:
            wr = week_row_df.iloc[0]
            sph_violated = (wr.get('sphericity', True) is False) or (wr.get('sphericity', True) == False)
            sphericity_info = {
                'W': wr.get('W-spher', np.nan),
                'p_spher': wr.get('p-spher', np.nan),
                'sphericity_ok': not sph_violated,
                'eps': wr.get('eps', np.nan),
                'p_gg': wr.get('p-GG-corr', np.nan),
            }
            if sph_violated:
                print(f"\n  [WARNING] Sphericity VIOLATED (Mauchly's W = {sphericity_info['W']:.4f}, "
                      f"p = {sphericity_info['p_spher']:.4f})")
                print(f"  Greenhouse-Geisser epsilon = {sphericity_info['eps']:.4f}")
                print(f"  Using GG-corrected p = {sphericity_info['p_gg']:.4f} for Week effect")
            else:
                w_val = sphericity_info.get('W', float('nan'))
                p_val = sphericity_info.get('p_spher', float('nan'))
                if np.isfinite(w_val):
                    print(f"\n  Sphericity OK (Mauchly's W = {w_val:.4f}, p = {p_val:.4f})")
                else:
                    print("\n  Sphericity: not computed (too few levels or missing values)")

        # Determine significance using GG-corrected p for Week when violated
        def _p_for_effect(src_name):
            row_df = aov[aov['Source'] == src_name]
            if len(row_df) == 0:
                return np.nan
            row = row_df.iloc[0]
            if src_name == 'Week' and not sphericity_info.get('sphericity_ok', True):
                gg = row.get('p-GG-corr', np.nan)
                if np.isfinite(gg):
                    return float(gg)
            return float(row[p_col])

        p_week = _p_for_effect('Week')
        p_ca = _p_for_effect('CA (%)')
        p_int = _p_for_effect('CA (%) * Week')
        sig_week = np.isfinite(p_week) and p_week < 0.05
        sig_ca = np.isfinite(p_ca) and p_ca < 0.05
        sig_interaction = np.isfinite(p_int) and p_int < 0.05

        # -------------------------------------------------------------------------
        # Post-hoc tests
        posthoc_week = None
        posthoc_ca_per_week = {}
        posthoc_week_per_ca = {}

        # Week main effect post-hoc (within-subjects pairwise, Bonferroni-adjusted)
        if sig_week or sig_interaction:
            print("\nRunning Week pairwise post-hoc (within-subjects, Bonferroni-adjusted)...")
            try:
                posthoc_week = pg.pairwise_tests(
                    data=analysis_df,
                    dv=measure,
                    within='Week',
                    subject='ID',
                    padjust='bonf',
                )
                print(f"  {len(posthoc_week)} pairwise comparisons computed")
            except Exception as e:
                print(f"  [WARNING] Week post-hoc failed: {e}")

        # Interaction post-hoc: CA% comparison at each Week
        if sig_interaction:
            n_weeks_for_bonf = len(week_levels)
            print(f"\nRunning interaction post-hoc: CA% comparison at each Week "
                  f"(Bonferroni k={n_weeks_for_bonf} across weeks)...")
            if len(ca_levels) == 2:
                for wk in week_levels:
                    wk_data = analysis_df[analysis_df['Week'] == wk]
                    g1 = wk_data[wk_data['CA (%)'] == ca_levels[0]][measure]
                    g2 = wk_data[wk_data['CA (%)'] == ca_levels[1]][measure]
                    try:
                        t_res = pg.ttest(g1, g2)
                        raw_p = float(t_res['p-val'].iloc[0])
                        t_res = t_res.copy()
                        t_res['p-bonf'] = min(1.0, raw_p * n_weeks_for_bonf)
                        t_res['Week'] = wk
                        t_res['CA_comparison'] = f"{ca_levels[0]}% vs {ca_levels[1]}%"
                        posthoc_ca_per_week[wk] = t_res
                        sig_str = "SIGNIFICANT" if t_res['p-bonf'].iloc[0] < 0.05 else "ns"
                        print(f"  Week {wk}: t = {float(t_res['T'].iloc[0]):.3f}, "
                              f"raw p = {raw_p:.4f}, Bonf p = {t_res['p-bonf'].iloc[0]:.4f} ({sig_str})")
                    except Exception as e:
                        print(f"  [WARNING] CA% comparison at Week {wk} failed: {e}")

            # Interaction post-hoc: Week pairwise within each CA% group
            print("\nRunning interaction post-hoc: Week pairwise within each CA% group...")
            for ca_val in ca_levels:
                subset = analysis_df[analysis_df['CA (%)'] == ca_val].copy()
                try:
                    ph = pg.pairwise_tests(
                        data=subset,
                        dv=measure,
                        within='Week',
                        subject='ID',
                        padjust='bonf',
                    )
                    posthoc_week_per_ca[ca_val] = ph
                    n_sig = (ph['p-corr'] < 0.05).sum() if 'p-corr' in ph.columns else 0
                    print(f"  CA%={ca_val}: {len(ph)} comparisons, {n_sig} significant after correction")
                except Exception as e:
                    print(f"  [WARNING] Week post-hoc for CA%={ca_val} failed: {e}")

        return {
            'measure': measure,
            'type': 'mixed_anova_ca_x_week',
            'n_subjects': n_subjects,
            'n_observations': len(analysis_df),
            'weeks': week_levels,
            'ca_levels': ca_levels,
            'anova_table': aov,
            'sphericity_info': sphericity_info,
            'sig_week': sig_week,
            'sig_ca': sig_ca,
            'sig_interaction': sig_interaction,
            'p_week': p_week,
            'p_ca': p_ca,
            'p_int': p_int,
            'ca_stats': ca_stats,
            'week_stats': week_stats,
            'cell_stats': cell_stats,
            'data': analysis_df,
            'posthoc_week': posthoc_week,
            'posthoc_ca_per_week': posthoc_ca_per_week,
            'posthoc_week_per_ca': posthoc_week_per_ca,
        }
    except Exception as e:
        print(f"[ERROR] Mixed ANOVA failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_ca_x_week_report(
    results: Optional[Dict] = None,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    include_preamble: bool = True,
    include_footer: bool = True,
) -> str:
    """
    Generate a report for the 2-way CA%  x  Week mixed ANOVA with post-hoc tests.

    Parameters:
        results: Output from perform_mixed_anova_ca_x_week()
        cohort_dfs: Cohort DataFrames for study design section
        include_preamble: Whether to include title and methodology header
        include_footer: Whether to include end-of-report footer
    """
    lines = []

    def h1(text):
        lines.append("=" * 80)
        lines.append(text)
        lines.append("=" * 80)

    def h2(text):
        lines.append(text)
        lines.append("-" * 80)

    def sig_stars(p):
        if not np.isfinite(p):
            return "?"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    # Detect measure label
    _measure_label = results.get('measure', 'Unknown') if results else 'Unknown'

    # Per-measure banner (always shown)
    lines.append("=" * 80)
    lines.append(f"MEASURE: {_measure_label}")
    lines.append("=" * 80)
    lines.append("")

    if include_preamble:
        h1("CROSS-COHORT -- 2-WAY MIXED ANOVA: CA%  x  WEEK (SEX COLLAPSED)")
        lines.append("")
        h2("METHODOLOGICAL NOTES")
        lines.append("")
        lines.append("  Design: CA% (between-subjects)  x  Week (within-subjects).")
        lines.append("  Sex is collapsed (all animals in each CA% group combined).")
        lines.append("  One value per animal per week (daily observations averaged within each week).")
        lines.append("  Complete-subject design: animals missing any week are excluded.")
        lines.append("")
        lines.append("  Sphericity is tested using Mauchly's W test.")
        lines.append("  When sphericity is violated (p < 0.05), the Greenhouse-Geisser (GG)")
        lines.append("  corrected p-value and epsilon (epsilon) are reported for the Week effect.")
        lines.append("")
        lines.append("  Post-hoc tests:")
        lines.append("    - Week main effect: pairwise paired t-tests, Bonferroni-adjusted")
        lines.append("      (within-subjects; equivalent to Tukey HSD for the within factor)")
        lines.append("    - CA%  x  Week interaction -- simple effects:")
        lines.append("        - CA% comparison (0% vs 2%) at each Week, Bonferroni across weeks")
        lines.append("        - Week pairwise comparisons within each CA% group, Bonferroni-adjusted")
        lines.append("")

    # -------------------------------------------------------------------------
    # Study design
    if include_preamble and cohort_dfs is not None:
        h1("STUDY DESIGN")
        lines.append("")
        for label, df in cohort_dfs.items():
            n_animals = df['ID'].nunique() if 'ID' in df.columns else len(df)
            lines.append(f"  {label}: {n_animals} animals")
        if results:
            lines.append(f"\n  Total subjects in analysis: {results.get('n_subjects', 'Unknown')}")
            lines.append(f"  Total observations: {results.get('n_observations', 'Unknown')}")
            lines.append(f"  Weeks: {results.get('weeks', 'Unknown')}")
            lines.append(f"  CA% levels: {results.get('ca_levels', 'Unknown')}")
        lines.append("")

    if results is None or not results:
        lines.append("  [No results available]")
        lines.append("")
        if include_footer:
            h1("END OF REPORT")
            lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Descriptive statistics
    h1("DESCRIPTIVE STATISTICS")
    lines.append("")

    ca_stats = results.get('ca_stats')
    if ca_stats is not None and len(ca_stats) > 0:
        h2("By CA% Group (averaged across all weeks)")
        for _, row in ca_stats.iterrows():
            lines.append(f"  CA%={row['CA (%)']}: n={int(row['count'])}, "
                         f"M={row['mean']:.3f}, SD={row['std']:.3f}, SEM={row['sem']:.3f}, "
                         f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        lines.append("")

    week_stats = results.get('week_stats')
    if week_stats is not None and len(week_stats) > 0:
        h2("By Week (averaged across all CA% groups)")
        for _, row in week_stats.iterrows():
            lines.append(f"  Week {int(row['Week'])}: n={int(row['count'])}, "
                         f"M={row['mean']:.3f}, SD={row['std']:.3f}, SEM={row['sem']:.3f}, "
                         f"95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        lines.append("")

    cell_stats = results.get('cell_stats')
    if cell_stats is not None and len(cell_stats) > 0:
        h2("Cell Means: CA%  x  Week")
        ca_levels = sorted(cell_stats['CA (%)'].unique())
        week_levels = sorted(cell_stats['Week'].unique())
        # Header row
        header = f"  {'CA%':<8}" + "".join([f"  Week {int(w):<6}" for w in week_levels])
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for ca_val in ca_levels:
            row_vals = []
            for wk in week_levels:
                cell = cell_stats[(cell_stats['CA (%)'] == ca_val) & (cell_stats['Week'] == wk)]
                if len(cell) > 0:
                    row_vals.append(f"  {cell.iloc[0]['mean']:>8.3f}")
                else:
                    row_vals.append(f"  {'-':>8}")
            lines.append(f"  {ca_val}%{'':<5}" + "".join(row_vals))
        lines.append("")

    # -------------------------------------------------------------------------
    # ANOVA table + sphericity
    h1("MIXED ANOVA RESULTS")
    lines.append("")

    aov = results.get('anova_table')
    sph = results.get('sphericity_info', {})
    p_col = 'p-unc'

    # Sphericity section
    h2("Sphericity Assessment (Mauchly's Test for Week factor)")
    w_val = sph.get('W', float('nan'))
    p_spher = sph.get('p_spher', float('nan'))
    eps = sph.get('eps', float('nan'))
    sph_ok = sph.get('sphericity_ok', True)

    if np.isfinite(w_val):
        lines.append(f"  Mauchly's W = {w_val:.4f}")
        lines.append(f"  p (Mauchly) = {p_spher:.4f}")
        if sph_ok:
            lines.append("  Decision: Sphericity SATISFIED -- use uncorrected p-values")
        else:
            lines.append("  Decision: Sphericity VIOLATED -- Greenhouse-Geisser correction applied")
            lines.append(f"  Greenhouse-Geisser epsilon = {eps:.4f}")
            lines.append(f"  (epsilon < 0.75 indicates substantial non-sphericity; epsilon = 1.0 = perfect sphericity)")
    else:
        lines.append("  Mauchly's W: not available (2-level within factor or missing)")
        if np.isfinite(eps):
            lines.append(f"  Greenhouse-Geisser epsilon = {eps:.4f}")
    lines.append("")

    # ANOVA table
    if aov is not None:
        h2("ANOVA Table")
        lines.append(aov.to_string())
        lines.append("")

        # Interpretation
        h2("Interpretation")
        lines.append("")

        for _, row in aov.iterrows():
            src = row['Source']
            p_raw = row[p_col]
            # Use GG-corrected p for Week if sphericity violated
            if src == 'Week' and not sph_ok:
                p_use = row.get('p-GG-corr', p_raw)
                if not np.isfinite(p_use):
                    p_use = p_raw
                p_note = f"GG-corrected p = {p_use:.4f}" if np.isfinite(p_use) else f"p = {p_raw:.4f}"
            else:
                p_use = p_raw
                p_note = f"p = {p_use:.4f}"
            sig = sig_stars(p_use)
            sig_str = "SIGNIFICANT" if p_use < 0.05 else "NOT SIGNIFICANT"
            lines.append(f"  {src}: {sig_str}")
            lines.append(f"     F({int(row['DF1'])}, {int(row['DF2'])}) = {row['F']:.3f}, {p_note} {sig}")
            # Effect size
            np2 = row.get('np2', float('nan'))
            if np.isfinite(np2):
                lines.append(f"     eta^2p = {np2:.4f}")
            # Narrative
            if src == 'CA (%)':
                lines.append("     -> 0% and 2% CA groups differ overall in " + _measure_label
                              if p_use < 0.05 else
                              "     -> No significant overall difference between CA% groups")
            elif src == 'Week':
                if not sph_ok:
                    lines.append(f"     (Sphericity violated -- GG correction applied, epsilon = {eps:.4f})")
                lines.append("     -> " + _measure_label + " changes significantly across weeks"
                              if p_use < 0.05 else
                              "     -> No significant week-to-week change overall")
            elif src == 'CA (%) * Week':
                lines.append("     -> CA% groups follow different weekly trajectories (interaction)"
                              if p_use < 0.05 else
                              "     -> Both CA% groups follow similar weekly trajectories")
            lines.append("")

    # -------------------------------------------------------------------------
    # Post-hoc: Week main effect
    posthoc_week = results.get('posthoc_week')
    p_col_ph = 'p-corr' if (posthoc_week is not None and 'p-corr' in posthoc_week.columns) else 'p-unc'

    if posthoc_week is not None and len(posthoc_week) > 0:
        h1("POST-HOC: WEEK PAIRWISE COMPARISONS")
        lines.append("")
        lines.append("  Within-subjects pairwise comparisons (Bonferroni-adjusted).")
        lines.append("  All pairs of adjacent and non-adjacent weeks tested.")
        lines.append("")
        h2("Pairwise Week Comparisons")
        p_corr_col = 'p-corr' if 'p-corr' in posthoc_week.columns else 'p-unc'
        lines.append(f"  {'Comparison':<20} {'T':>8} {'df':>6} {'p (raw)':>10} {'p (Bonf.)':>10} {'Sig':>5}")
        lines.append("  " + "-" * 60)
        for _, row in posthoc_week.iterrows():
            a = row.get('A', row.get('Week1', '?'))
            b = row.get('B', row.get('Week2', '?'))
            comp_label = f"Week {int(a)} vs {int(b)}"
            t_val = row.get('T', row.get('t', float('nan')))
            df_val = row.get('dof', row.get('df', float('nan')))
            p_raw_val = row.get('p-unc', float('nan'))
            p_corr_val = row.get(p_corr_col, float('nan'))
            sig = sig_stars(p_corr_val)
            t_str = f"{t_val:.3f}" if np.isfinite(t_val) else "-"
            df_str = f"{df_val:.0f}" if np.isfinite(df_val) else "-"
            p_raw_str = f"{p_raw_val:.4f}" if np.isfinite(p_raw_val) else "-"
            p_corr_str = f"{p_corr_val:.4f}" if np.isfinite(p_corr_val) else "-"
            lines.append(f"  {comp_label:<20} {t_str:>8} {df_str:>6} {p_raw_str:>10} {p_corr_str:>10} {sig:>5}")
        lines.append("")

    # -------------------------------------------------------------------------
    # Post-hoc: Interaction simple effects
    posthoc_ca_per_week = results.get('posthoc_ca_per_week', {})
    posthoc_week_per_ca = results.get('posthoc_week_per_ca', {})
    ca_levels_res = results.get('ca_levels', [])

    if posthoc_ca_per_week or posthoc_week_per_ca:
        h1("POST-HOC: CA%  x  WEEK INTERACTION SIMPLE EFFECTS")
        lines.append("")

    if posthoc_ca_per_week:
        h2(f"CA% Comparison ({ca_levels_res[0]}% vs {ca_levels_res[-1]}%) at Each Week")
        lines.append("")
        lines.append("  Bonferroni correction applied across all weeks (k = number of weeks).")
        lines.append("")
        n_weeks_k = len(posthoc_ca_per_week)
        lines.append(f"  {'Week':<8} {'T':>8} {'df':>6} {'p (raw)':>10} "
                     f"{'p (Bonf.)':>12} {'Sig':>5} {'Cohen d':>9}")
        lines.append("  " + "-" * 60)
        for wk in sorted(posthoc_ca_per_week.keys()):
            t_res = posthoc_ca_per_week[wk]
            t_val = float(t_res['T'].iloc[0])
            df_val = float(t_res['dof'].iloc[0])
            p_raw_val = float(t_res['p-val'].iloc[0])
            p_bonf_val = float(t_res['p-bonf'].iloc[0])
            d_val = float(t_res['cohen-d'].iloc[0]) if 'cohen-d' in t_res.columns else float('nan')
            sig = sig_stars(p_bonf_val)
            d_str = f"{d_val:.3f}" if np.isfinite(d_val) else "-"
            lines.append(f"  Week {int(wk):<3}  {t_val:>8.3f} {df_val:>6.0f} {p_raw_val:>10.4f} "
                         f"{p_bonf_val:>12.4f} {sig:>5} {d_str:>9}")
        lines.append("")

    if posthoc_week_per_ca:
        h2("Week Pairwise Comparisons Within Each CA% Group")
        lines.append("")
        lines.append("  Within-subjects paired comparisons, Bonferroni-adjusted.")
        lines.append("")
        for ca_val in sorted(posthoc_week_per_ca.keys()):
            ph = posthoc_week_per_ca[ca_val]
            lines.append(f"  -- CA% = {ca_val}% --")
            if ph is None or len(ph) == 0:
                lines.append("    No post-hoc data available")
                continue
            p_c_col = 'p-corr' if 'p-corr' in ph.columns else 'p-unc'
            lines.append(f"  {'Comparison':<20} {'T':>8} {'df':>6} {'p (raw)':>10} {'p (Bonf.)':>10} {'Sig':>5}")
            lines.append("  " + "-" * 60)
            for _, row in ph.iterrows():
                a = row.get('A', row.get('Week1', '?'))
                b = row.get('B', row.get('Week2', '?'))
                comp_label = f"Week {int(a)} vs {int(b)}"
                t_val = row.get('T', row.get('t', float('nan')))
                df_val = row.get('dof', row.get('df', float('nan')))
                p_raw_val = row.get('p-unc', float('nan'))
                p_c_val = row.get(p_c_col, float('nan'))
                sig = sig_stars(p_c_val)
                t_str = f"{t_val:.3f}" if np.isfinite(t_val) else "-"
                df_str = f"{df_val:.0f}" if np.isfinite(df_val) else "-"
                p_raw_str = f"{p_raw_val:.4f}" if np.isfinite(p_raw_val) else "-"
                p_c_str = f"{p_c_val:.4f}" if np.isfinite(p_c_val) else "-"
                lines.append(f"  {comp_label:<20} {t_str:>8} {df_str:>6} {p_raw_str:>10} {p_c_str:>10} {sig:>5}")
            lines.append("")

    # -------------------------------------------------------------------------
    if include_footer:
        h1("END OF CA%  x  WEEK REPORT")
        lines.append("")

    return "\n".join(lines)


# =============================================================================


# =============================================================================
# 2-WAY OMNIBUS WEIGHT ANOVA (BH-FDR CORRECTED ACROSS MEASURES)
# =============================================================================

def _bh_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in original order."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = np.array(p_values) * n / ranked
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return [float(min(v, 1.0)) for v in adjusted]


def perform_omnibus_weight_anova_2way(
    cohort_dfs: Dict[str, pd.DataFrame],
    measures: Optional[List[str]] = None,
    weeks: Optional[List[int]] = None,
) -> Dict:
    """Omnibus 2-way mixed ANOVA for weight measures: CA% (between) � Week (within).

    Runs the 2-way mixed ANOVA for all measures simultaneously and applies
    BH-FDR correction separately across measures within each factor family
    (Week p-values corrected together; CA% p-values corrected together).

    Post-hocs per measure:
      - Significant CA% main effect  ? Tukey HSD on animal-level means
      - Significant Week main effect ? Bonferroni all-pairwise paired t-tests
      - Significant Week � CA%       ? Bonferroni pairwise within each CA% cell

    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measures: Weight measures to analyze (default: ['Total Change', 'Daily Change'])
        weeks: Optional list of week numbers to include (default: all)

    Returns:
        Dict keyed by measure name with ANOVA results; also includes '_measures' list.
    """
    if measures is None:
        measures = ["Total Change", "Daily Change"]

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required for mixed ANOVA")
        return {}

    print(f"\n{'='*80}")
    print("2-WAY OMNIBUS WEIGHT ANOVA � CA% � WEEK (ALL MEASURES, BH-FDR CORRECTED)")
    print(f"{'='*80}")

    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    combined_df = _add_week_column_across_cohorts(combined_df)
    combined_df = combined_df[combined_df['Day'] >= 0]

    all_results: Dict = {}
    week_ps: List[float] = []
    ca_ps: List[float] = []

    for measure in measures:
        print(f"\n{'-'*60}")
        print(f"  ANOVA: {measure}")
        print(f"{'-'*60}")

        if measure not in combined_df.columns:
            print(f"  [WARNING] Column '{measure}' not found � skipping.")
            all_results[measure] = {'error': f'Column {measure} not found'}
            week_ps.append(np.nan)
            ca_ps.append(np.nan)
            continue

        adf = combined_df[['ID', 'Week', 'CA (%)', measure]].dropna().copy()
        if weeks is not None:
            adf = adf[adf['Week'].isin(weeks)]

        # Average within Week � ID (collapse daily observations to one per animal per week)
        adf = adf.groupby(['ID', 'Week', 'CA (%)'])[measure].mean().reset_index()

        tp = sorted(adf['Week'].unique())
        n_weeks = len(tp)

        # Complete-subject design: drop animals missing any week
        counts = adf.groupby('ID')['Week'].nunique()
        complete_ids = counts[counts == n_weeks].index
        n_dropped = adf['ID'].nunique() - len(complete_ids)
        adf = adf[adf['ID'].isin(complete_ids)].copy()

        if n_dropped > 0:
            print(f"  [INFO] {n_dropped} animal(s) excluded (incomplete weeks for {measure})")

        try:
            if adf['CA (%)'].nunique() < 2 or adf['ID'].nunique() < 4:
                raise ValueError("Insufficient subjects for mixed ANOVA")

            tbl = pg.mixed_anova(
                data=adf, dv=measure, within='Week',
                subject='ID', between='CA (%)',
                correction='auto',
            )

            def _get(src):
                row = tbl[tbl['Source'] == src]
                if row.empty:
                    return np.nan, np.nan, np.nan, np.nan
                rv = row.iloc[0]
                F = float(rv.get('F', np.nan))
                p_unc = float(rv.get('p-unc', np.nan))
                np2 = float(rv.get('np2', np.nan))
                eps = float(rv.get('eps', np.nan))
                return F, p_unc, np2, eps

            wF, wP, wNP2, wEps = _get('Week')
            cF, cP, cNP2, _   = _get('CA (%)')
            iF, iP, iNP2, _   = _get('Interaction')

            # Use GG-corrected p for Week if sphericity violated
            week_row = tbl[tbl['Source'] == 'Week']
            wP_gg = np.nan
            sphericity_ok = True
            if len(week_row) > 0:
                wr = week_row.iloc[0]
                sph_val = wr.get('sphericity', True)
                sphericity_ok = not (sph_val is False or sph_val == False)
                wP_gg = float(wr.get('p-GG-corr', np.nan))

            # p used for reporting and FDR (GG-corrected when violated)
            wP_use = wP_gg if (not sphericity_ok and np.isfinite(wP_gg)) else wP

            week_ps.append(wP_use)
            ca_ps.append(cP)

            # Descriptive stats by CA% and by CA%�Week
            desc_rows = []
            for ca_val, grp in adf.groupby('CA (%)')[measure]:
                n = len(grp)
                m = grp.mean()
                s = grp.std(ddof=1) if n > 1 else np.nan
                sem = s / np.sqrt(n) if (n > 0 and np.isfinite(s)) else np.nan
                desc_rows.append({
                    'CA (%)': ca_val, 'n': n, 'mean': m, 'sd': s,
                    'sem': sem,
                    'ci_lower': m - 1.96 * sem if np.isfinite(sem) else np.nan,
                    'ci_upper': m + 1.96 * sem if np.isfinite(sem) else np.nan,
                })
            desc_df = pd.DataFrame(desc_rows)

            all_results[measure] = {
                'measure':       measure,
                'analysis_df':   adf,
                'weeks':         tp,
                'n_subjects':    adf['ID'].nunique(),
                'n_dropped':     n_dropped,
                'anova_table':   tbl,
                'desc_df':       desc_df,
                'week_F':   wF,  'week_p':   wP_use, 'week_p_raw': wP,
                'week_np2': wNP2, 'week_eps': wEps,
                'sphericity_ok': sphericity_ok,
                'ca_F':   cF,  'ca_p':   cP,  'ca_np2':   cNP2,
                'int_F':  iF,  'int_p':  iP,  'int_np2':  iNP2,
            }

        except Exception as e:
            print(f"  [ERROR] ANOVA failed for {measure}: {e}")
            import traceback; traceback.print_exc()
            all_results[measure] = {'error': str(e)}
            week_ps.append(np.nan)
            ca_ps.append(np.nan)

    # --- BH-FDR correction across measures ---
    fdr_w = _bh_fdr([p if np.isfinite(p) else 1.0 for p in week_ps])
    fdr_c = _bh_fdr([p if np.isfinite(p) else 1.0 for p in ca_ps])
    for i, m in enumerate(measures):
        r = all_results.get(m, {})
        if 'error' not in r:
            r['fdr_week_p'] = fdr_w[i]
            r['fdr_ca_p']   = fdr_c[i]

    # --- Post-hoc tests ---
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r:
            continue
        adf = r['analysis_df']
        tp  = r['weeks']
        ca_levels = sorted(adf['CA (%)'].unique())

        # Week main effect or interaction ? Bonferroni paired t-tests across all week pairs
        if (r.get('fdr_week_p', 1.0) < 0.05 or r.get('int_p', 1.0) < 0.05) and len(tp) > 1:
            print(f"  Post-hoc (Week, Bonferroni): {m}")
            pairs = [(a, b) for i, a in enumerate(tp) for b in tp[i+1:]]
            ph_rows = []
            for wA, wB in pairs:
                dA = adf[adf['Week'] == wA][['ID', m]].rename(columns={m: 'vA'})
                dB = adf[adf['Week'] == wB][['ID', m]].rename(columns={m: 'vB'})
                merged = dA.merge(dB, on='ID').dropna()
                if len(merged) < 3:
                    continue
                t_stat, p_raw = stats.ttest_rel(merged['vA'], merged['vB'])
                df_val = len(merged) - 1
                p_bonf = float(min(1.0, p_raw * len(pairs)))
                ph_rows.append({
                    'Week A': wA, 'Week B': wB,
                    't': t_stat, 'df': df_val,
                    'p_raw': p_raw, 'p_bonf': p_bonf,
                    'sig': '*' if p_bonf < 0.05 else '',
                })
            r['posthoc_week'] = pd.DataFrame(ph_rows)

        # CA% main effect ? Tukey HSD on per-animal means (collapsed across weeks)
        if r.get('fdr_ca_p', 1.0) < 0.05 and len(ca_levels) >= 2:
            print(f"  Post-hoc (CA%, Tukey): {m}")
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                am = adf.groupby(['ID', 'CA (%)'])[m].mean().reset_index()
                tukey = pairwise_tukeyhsd(am[m], am['CA (%)'])
                r['posthoc_ca'] = tukey
            except Exception as ex:
                print(f"  [WARNING] Tukey failed for {m}: {ex}")

        # Significant interaction ? per-cell Bonferroni paired t-tests
        if r.get('int_p', 1.0) < 0.05:
            cell_ph = {}
            for ca_val in ca_levels:
                sub = adf[adf['CA (%)'] == ca_val].copy()
                pairs = [(a, b) for i, a in enumerate(tp) for b in tp[i+1:]]
                ph_rows = []
                for wA, wB in pairs:
                    dA = sub[sub['Week'] == wA][['ID', m]].rename(columns={m: 'vA'})
                    dB = sub[sub['Week'] == wB][['ID', m]].rename(columns={m: 'vB'})
                    merged = dA.merge(dB, on='ID').dropna()
                    if len(merged) < 3:
                        continue
                    t_stat, p_raw = stats.ttest_rel(merged['vA'], merged['vB'])
                    df_val = len(merged) - 1
                    p_bonf = float(min(1.0, p_raw * len(pairs)))
                    ph_rows.append({
                        'Week A': wA, 'Week B': wB,
                        't': t_stat, 'df': df_val,
                        'p_raw': p_raw, 'p_bonf': p_bonf,
                        'sig': '*' if p_bonf < 0.05 else '',
                    })
                cell_ph[ca_val] = pd.DataFrame(ph_rows)
            r['posthoc_week_cells'] = cell_ph

    all_results['_measures'] = measures
    return all_results


def generate_omnibus_weight_report_2way(
    all_results: Dict,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """Generate a report for the 2-way omnibus BH-FDR corrected weight ANOVA."""
    lines = []
    measures = all_results.get('_measures', ["Total Change", "Daily Change"])

    def _sig(p):
        if not np.isfinite(p): return '?'
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    def _fp(p):
        if not np.isfinite(p): return '-'
        if p < 0.0001: return '< 0.0001'
        return f'{p:.4f}'

    lines.append("=" * 80)
    lines.append("2-WAY OMNIBUS WEIGHT ANOVA � CA% � WEEK (ALL MEASURES)")
    lines.append("=" * 80)
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Design    : CA% (between) � Week (within), sex collapsed")
    lines.append("Post-hocs : Tukey HSD for CA%; Bonferroni paired t-tests for Week")
    lines.append("Correction: BH-FDR across measures within each factor family")
    lines.append("")

    # Study design
    if cohort_dfs is not None:
        lines.append("=" * 80)
        lines.append("STUDY DESIGN")
        lines.append("=" * 80)
        for label, df in cohort_dfs.items():
            n_animals = df['ID'].nunique() if 'ID' in df.columns else len(df)
            lines.append(f"  {label}: {n_animals} animals")
        lines.append("")

    # BH-FDR summary table
    lines.append("=" * 80)
    lines.append("BH-FDR CORRECTED OMNIBUS P-VALUES")
    lines.append("=" * 80)
    col_w = 38
    lines.append(
        f"  {'Measure':<{col_w}} {'Week(raw)':>12} {'Week(FDR)':>12} "
        f"{'CA%(raw)':>12} {'CA%(FDR)':>12}"
    )
    lines.append("  " + "-" * 90)
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r:
            lines.append(f"  {m:<{col_w}} [ERROR]")
            continue
        wraw = _fp(r.get('week_p', np.nan))
        wfdr = _fp(r.get('fdr_week_p', np.nan))
        craw = _fp(r.get('ca_p', np.nan))
        cfdr = _fp(r.get('fdr_ca_p', np.nan))
        # add star to FDR value if significant
        wfdr_s = f"{wfdr} {_sig(r.get('fdr_week_p', np.nan))}" if r.get('fdr_week_p', np.nan) < 0.05 else f"{wfdr} "
        cfdr_s = f"{cfdr} {_sig(r.get('fdr_ca_p', np.nan))}" if r.get('fdr_ca_p', np.nan) < 0.05 else f"{cfdr} "
        lines.append(
            f"  {m:<{col_w}} {wraw:>12} {wfdr_s:>14} {craw:>12} {cfdr_s:>14}"
        )
    lines.append("")
    lines.append("")

    # Per-measure sections
    for m in measures:
        r = all_results.get(m, {})
        lines.append("?" * 80)
        lines.append(f"  MEASURE: {m}")
        lines.append("?" * 80)

        if 'error' in r:
            lines.append(f"  [ERROR] {r['error']}")
            lines.append("")
            continue

        adf = r.get('analysis_df')
        tp  = r.get('weeks', [])
        desc_df = r.get('desc_df')

        lines.append(f"  N subjects: {r.get('n_subjects', '?')}")
        lines.append(f"  Animals dropped: {r.get('n_dropped', 0)}")
        lines.append(f"  Weeks: {tp}")
        lines.append("")

        # Descriptives
        if desc_df is not None and len(desc_df) > 0:
            lines.append("  DESCRIPTIVE STATISTICS (by CA%, collapsed across weeks)")
            lines.append("  " + "-" * 60)
            for _, row in desc_df.iterrows():
                lines.append(
                    f"    CA%={row['CA (%)']:.1f}: n={int(row['n'])}, "
                    f"M={row['mean']:.3f}, SD={row['sd']:.3f}, "
                    f"SEM={row['sem']:.3f}, 95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                )
            lines.append("")

        # ANOVA table
        tbl = r.get('anova_table')
        if tbl is not None:
            lines.append("  ANOVA TABLE (CA% � Week mixed ANOVA)")
            lines.append("  " + "-" * 60)
            lines.append("  " + tbl.to_string().replace('\n', '\n  '))
            lines.append("")

        # Interpretation
        lines.append("  INTERPRETATION")
        lines.append("  " + "-" * 60)

        wP = r.get('week_p', np.nan)
        wFDR = r.get('fdr_week_p', np.nan)
        wF = r.get('week_F', np.nan)
        wNP2 = r.get('week_np2', np.nan)
        wEps = r.get('week_eps', np.nan)
        sph_ok = r.get('sphericity_ok', True)
        cP = r.get('ca_p', np.nan)
        cFDR = r.get('fdr_ca_p', np.nan)
        cF = r.get('ca_F', np.nan)
        cNP2 = r.get('ca_np2', np.nan)
        iP = r.get('int_p', np.nan)
        iF = r.get('int_F', np.nan)
        iNP2 = r.get('int_np2', np.nan)
        n_subj = r.get('n_subjects', '?')
        # determine DF2 for within factor from ANOVA table
        try:
            wDF1 = int(tbl[tbl['Source'] == 'Week'].iloc[0]['DF1'])
            wDF2 = int(tbl[tbl['Source'] == 'Week'].iloc[0]['DF2'])
            cDF2 = int(tbl[tbl['Source'] == 'CA (%)'].iloc[0]['DF2'])
            iDF1 = int(tbl[tbl['Source'] == 'Interaction'].iloc[0]['DF1'])
            iDF2 = int(tbl[tbl['Source'] == 'Interaction'].iloc[0]['DF2'])
        except Exception:
            wDF1 = wDF2 = cDF2 = iDF1 = iDF2 = '?'

        wk_label = "p (GG-corr)" if not sph_ok else "p"
        wSig = "SIGNIFICANT" if wFDR < 0.05 else "NOT SIGNIFICANT"
        cSig = "SIGNIFICANT" if cFDR < 0.05 else "NOT SIGNIFICANT"
        iSig = "SIGNIFICANT" if (np.isfinite(iP) and iP < 0.05) else "NOT SIGNIFICANT"

        lines.append(f"  1. Week: {wSig}")
        lines.append(
            f"     F({wDF1}, {wDF2}) = {wF:.2f}, {wk_label} = {_fp(wP)}, "
            f"p_FDR = {_fp(wFDR)} {_sig(wFDR)}, ?p�={wNP2:.3f}"
        )
        if not sph_ok and np.isfinite(wEps):
            lines.append(f"     (GG epsilon = {wEps:.3f})")

        lines.append(f"  2. CA%: {cSig}")
        lines.append(
            f"     F(1, {cDF2}) = {cF:.2f}, p = {_fp(cP)}, "
            f"p_FDR = {_fp(cFDR)} {_sig(cFDR)}, ?p�={cNP2:.3f}"
        )

        lines.append(f"  3. Week � CA%: {iSig}")
        lines.append(
            f"     F({iDF1}, {iDF2}) = {iF:.2f}, p = {_fp(iP)} {_sig(iP)}, ?p�={iNP2:.3f}"
        )
        lines.append("")

        # Post-hoc: Week
        ph_week = r.get('posthoc_week')
        if ph_week is not None and len(ph_week) > 0:
            lines.append("  POST-HOC: Week (Bonferroni paired t-tests)")
            lines.append("  " + "-" * 60)
            lines.append(f"    {'Week A':>7} {'Week B':>7} {'t':>9} {'df':>5} {'p_raw':>12} {'p_bonf':>12} {'sig':>4}")
            for _, row in ph_week.iterrows():
                sig_s = row.get('sig', '')
                lines.append(
                    f"    {int(row['Week A']):>7} {int(row['Week B']):>7} "
                    f"{row['t']:>9.3f} {int(row['df']):>5} "
                    f"{row['p_raw']:>12.4f} {row['p_bonf']:>12.4f} {sig_s:>4}"
                )
            lines.append("")

        # Post-hoc: CA% (Tukey)
        ph_ca = r.get('posthoc_ca')
        if ph_ca is not None:
            lines.append("  POST-HOC: CA% (Tukey HSD on animal means)")
            lines.append("  " + "-" * 60)
            lines.append(str(ph_ca.summary()))
            lines.append("  (reject=True means the groups differ significantly at FWER=0.05)")
            lines.append("")

        # Post-hoc: interaction cells
        cell_ph = r.get('posthoc_week_cells', {})
        if cell_ph:
            for ca_val, ph in cell_ph.items():
                if ph is None or len(ph) == 0:
                    continue
                lines.append(f"  POST-HOC INTERACTION CELL: CA%={ca_val} (Bonferroni paired t-tests)")
                lines.append("  " + "-" * 60)
                lines.append(f"    {'Week A':>7} {'Week B':>7} {'t':>9} {'df':>5} {'p_raw':>12} {'p_bonf':>12} {'sig':>4}")
                for _, row in ph.iterrows():
                    sig_s = row.get('sig', '')
                    lines.append(
                        f"    {int(row['Week A']):>7} {int(row['Week B']):>7} "
                        f"{row['t']:>9.3f} {int(row['df']):>5} "
                        f"{row['p_raw']:>12.4f} {row['p_bonf']:>12.4f} {sig_s:>4}"
                    )
                lines.append("")

    # Summary table
    lines.append("=" * 80)
    lines.append("SUMMARY TABLE � ALL KEY P-VALUES (a = 0.05)")
    lines.append("=" * 80)
    lines.append("  * = significant after BH-FDR correction; raw p for interaction")
    lines.append("")
    lines.append(
        f"  {'Measure':<30} {'Week(FDR)':>12} {'CA%(FDR)':>12} {'Wk�CA%':>10}  Decision"
    )
    lines.append("  " + "-" * 90)
    for m in measures:
        r = all_results.get(m, {})
        if 'error' in r:
            continue
        wfdr = _fp(r.get('fdr_week_p', np.nan))
        cfdr = _fp(r.get('fdr_ca_p', np.nan))
        ip   = _fp(r.get('int_p', np.nan))
        wstar = '*' if r.get('fdr_week_p', np.nan) < 0.05 else ''
        cstar = '*' if r.get('fdr_ca_p', np.nan) < 0.05 else ''
        istar = '*' if (np.isfinite(r.get('int_p', np.nan)) and r.get('int_p') < 0.05) else ''
        decision_parts = []
        if wstar: decision_parts.append("Week")
        if cstar: decision_parts.append("CA%")
        if istar: decision_parts.append("Wk�CA%")
        decision = ", ".join(decision_parts) + " significant" if decision_parts else "None significant"
        lines.append(
            f"  {m:<30} {wfdr+wstar:>12} {cfdr+cstar:>12} {ip+istar:>10}   {decision}"
        )
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# DISTRIBUTION DIAGNOSTICS + MIXED-MODEL ASSUMPTION CHECKS (R-BASED)
# =============================================================================

def diagnose_weight_distributions(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> Dict:
    """
    Distribution diagnostics and mixed-model assumption checks for a continuous
    weight-change DV in a Cohort (between-subjects) × Week (within-subjects) design.

    All inference is run in R via rpy2.  Seven checks are reported:

      1. Overall distribution: N, range, mean, SD, skewness, kurtosis, Shapiro-Wilk
      2. Per-cell normality  : Shapiro-Wilk within each Cohort × Week cell
      3. Homogeneity of variance: Levene's test (car pkg) across cohorts; Bartlett fallback
      4. Sphericity          : Mauchly's W on the subject × week covariance matrix
      5. LMM residual normality: Shapiro-Wilk on lme4::lmer residuals (if lme4 available)
      6. Outlier detection   : IQR-based flagging (> Q3 + 1.5·IQR or < Q1 - 1.5·IQR)
      7. Cullen-Frey diagram : saved as PNG if fitdistrplus is installed in R

    Input data are aggregated to weekly means per animal before all tests.

    Parameters
    ----------
    cohort_dfs : dict
        Loaded cohort DataFrames (from load_cohorts).
    measure : str
        Column name for the weight-change DV. Default: 'Total Change'.
    weeks : list of int, optional
        Restrict to these Week indices (1-based). None = all weeks.
    save_path : Path, optional
        Write the full plain-text report here.

    Returns
    -------
    dict  with keys 'text' (full report string) and optional 'error'.
    """
    W = 80
    if not HAS_RPY2:
        print("[ERROR] rpy2 is not installed — cannot run R-based diagnostics.\n"
              "  pip install rpy2\n  Also requires R with: install.packages('car')")
        return {'error': 'rpy2 not installed'}

    _ensure_r_path()
    import rpy2.robjects as ro
    import tempfile, os

    print("\n" + "=" * W)
    print("DISTRIBUTION DIAGNOSTICS + MIXED-MODEL ASSUMPTION CHECKS  (R-based)")
    print("=" * W)

    # ── Build weekly-mean aggregate dataframe ───────────────────────────
    try:
        combined_df = combine_cohorts_for_analysis(cohort_dfs)
        combined_df = clean_cohort(combined_df)
        if 'Day' not in combined_df.columns:
            combined_df = add_day_column_across_cohorts(combined_df)
        combined_df = combined_df[combined_df['Day'] >= 1].copy()
        combined_df = _add_week_column_across_cohorts(combined_df)
    except Exception as _e:
        print(f"[ERROR] Could not prepare data: {_e}")
        return {'error': str(_e)}

    if weeks is not None:
        combined_df = combined_df[combined_df['Week'].isin(weeks)].copy()

    if measure not in combined_df.columns:
        print(f"[ERROR] Measure '{measure}' not found in combined data.")
        return {'error': f"measure '{measure}' not found"}

    req_cols = ['ID', 'Cohort', 'Week', measure]
    agg_df = (
        combined_df
        .dropna(subset=req_cols)
        .groupby(['ID', 'Cohort', 'Week'], as_index=False)[measure]
        .mean()
        .rename(columns={measure: 'DV'})
    )
    agg_df = _filter_complete_subjects_weekly(agg_df, 'ID', 'Week')
    agg_df['Week']   = agg_df['Week'].astype(int)
    agg_df['ID']     = agg_df['ID'].astype(str)
    agg_df['Cohort'] = agg_df['Cohort'].astype(str)

    cohort_labels = sorted(agg_df['Cohort'].unique())
    week_vals     = sorted(agg_df['Week'].unique())
    n_subj        = agg_df['ID'].nunique()
    n_obs         = len(agg_df)

    print(f"  Cohorts : {cohort_labels}")
    print(f"  Weeks   : {week_vals}")
    print(f"  Subjects: {n_subj}  |  Observations: {n_obs}")

    _tmpdir  = tempfile.gettempdir().replace('\\', '/')
    _uid     = str(abs(hash(measure + str(week_vals))))[-6:]
    _data_fp = f"{_tmpdir}/wt_diag_data_{_uid}.csv"
    _txt_fp  = f"{_tmpdir}/wt_diag_txt_{_uid}.txt"
    _cf_base = str(save_path).replace('.txt', '') if save_path else \
               f"wt_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _cf_fp   = (_cf_base + f"__{measure.replace(' ', '_')}_cullen_frey.png").replace('\\', '/')

    agg_df.to_csv(_data_fp.replace('/', os.sep), index=False)

    ro.globalenv['r_data_fp']    = _data_fp
    ro.globalenv['r_txt_fp']     = _txt_fp
    ro.globalenv['r_cf_fp']      = _cf_fp
    ro.globalenv['r_meas_name']  = measure

    try:
        ro.r(r"""
            suppressPackageStartupMessages(library(MASS))

            df_r        <- read.csv(r_data_fp, stringsAsFactors = FALSE)
            df_r$DV     <- as.numeric(df_r$DV)
            df_r$Cohort <- as.factor(df_r$Cohort)
            df_r$Week   <- as.integer(df_r$Week)
            df_r$ID     <- as.factor(df_r$ID)

            vals    <- na.omit(df_r$DV)
            n       <- length(vals)
            mn      <- mean(vals)
            vr      <- var(vals)
            sdv     <- sd(vals)
            sk      <- if (sdv > 0) mean(((vals - mn)/sdv)^3) else NA
            ku      <- if (sdv > 0) mean(((vals - mn)/sdv)^4) else NA
            cohorts <- levels(df_r$Cohort)
            weeks   <- sort(unique(df_r$Week))

            out <- character(0)
            hr  <- paste0("  ", paste(rep("\u2500", 76), collapse = ""))

            # ── 1. Overall distribution ────────────────────────────────
            out <- c(out,
                "  1. OVERALL DISTRIBUTION SUMMARY",
                hr,
                paste0("  Measure                       : ", r_meas_name),
                paste0("  N observations (weekly means) : ", n),
                paste0("  Cohorts                       : ", paste(cohorts, collapse = ", ")),
                paste0("  Weeks                         : ", paste(weeks, collapse = ", ")),
                paste0("  Range                         : [",
                       round(min(vals), 3), ",  ", round(max(vals), 3), "]"),
                paste0("  Mean  (pooled)                : ", format(mn,  digits = 5, nsmall = 2)),
                paste0("  SD    (pooled)                : ", format(sdv, digits = 5, nsmall = 2)),
                paste0("  Skewness                      : ", format(sk,  digits = 4),
                       "  (0 = symmetric)"),
                paste0("  Kurtosis (raw)                : ", format(ku,  digits = 4),
                       "  (3 = normal tails)")
            )

            sw_all <- tryCatch(shapiro.test(vals), error = function(e) NULL)
            if (!is.null(sw_all)) {
                sw_interp <- if (sw_all$p.value < 0.001) "strongly non-normal (p < .001)" else
                             if (sw_all$p.value < 0.050) "significantly non-normal (p < .05)" else
                             "no significant departure from normality"
                out <- c(out,
                    "",
                    "  Shapiro-Wilk (pooled data):",
                    paste0("    W = ", format(sw_all$statistic, digits = 6),
                           "   p = ", format(sw_all$p.value, digits = 4)),
                    paste0("    Interpretation : ", sw_interp)
                )
            }
            out <- c(out, "")

            # ── 2. Per-cell normality ──────────────────────────────────
            out <- c(out,
                "  2. PER-CELL NORMALITY  (Shapiro-Wilk per Cohort \u00d7 Week cell)",
                hr,
                sprintf("  %-28s  %5s  %8s  %8s  %9s  %8s  %s",
                        "Cell", "n", "Mean", "SD", "W", "p", "Normal?"),
                paste0("  ", paste(rep("-", 78), collapse = ""))
            )
            for (coh in cohorts) {
                for (wk in weeks) {
                    sub <- df_r$DV[df_r$Cohort == coh & df_r$Week == wk]
                    sub <- na.omit(sub)
                    cell_lbl <- paste0(coh, " W", wk)
                    if (length(sub) >= 3) {
                        sw_c <- tryCatch(shapiro.test(sub), error = function(e) NULL)
                        if (!is.null(sw_c)) {
                            norm_tag <- if (sw_c$p.value >= 0.05) "yes" else
                                        if (sw_c$p.value >= 0.01) "marginal" else "NO"
                            out <- c(out, sprintf(
                                "  %-28s  %5d  %8.3f  %8.3f  %9.5f  %8.4f  %s",
                                cell_lbl, length(sub),
                                mean(sub), sd(sub),
                                sw_c$statistic, sw_c$p.value, norm_tag))
                        }
                    } else {
                        out <- c(out, sprintf(
                            "  %-28s  %5d  [too few for S-W test]", cell_lbl, length(sub)))
                    }
                }
            }
            out <- c(out, "")

            # ── 3. Homogeneity of variance ─────────────────────────────
            out <- c(out,
                "  3. HOMOGENEITY OF VARIANCE ACROSS COHORTS",
                hr
            )
            # Levene's test (car) if available; Bartlett fallback
            has_car <- requireNamespace("car", quietly = TRUE)
            if (has_car) {
                lev_res <- tryCatch(
                    car::leveneTest(DV ~ Cohort, data = df_r, center = median),
                    error = function(e) NULL)
                if (!is.null(lev_res)) {
                    lev_p  <- lev_res[1, "Pr(>F)"]
                    lev_f  <- lev_res[1, "F value"]
                    lev_df1 <- lev_res[1, "Df"]
                    lev_df2 <- lev_res[2, "Df"]
                    hom_tag <- if (lev_p >= 0.05) "homogeneous (p >= .05)" else
                               if (lev_p >= 0.01) "marginally heterogeneous (p < .05)" else
                               "HETEROGENEOUS (p < .01)"
                    out <- c(out,
                        paste0("  Levene's test  (car::leveneTest, center = median)"),
                        paste0("    F(", lev_df1, ", ", lev_df2, ") = ",
                               format(lev_f, digits = 5),
                               "   p = ", format(lev_p, digits = 4)),
                        paste0("    Interpretation : ", hom_tag)
                    )
                } else {
                    out <- c(out, "  Levene's test failed; trying Bartlett's test.")
                    has_car <- FALSE
                }
            }
            if (!has_car) {
                bart_res <- tryCatch(
                    bartlett.test(DV ~ Cohort, data = df_r),
                    error = function(e) NULL)
                if (!is.null(bart_res)) {
                    hom_tag <- if (bart_res$p.value >= 0.05) "homogeneous (p >= .05)" else "HETEROGENEOUS"
                    out <- c(out,
                        paste0("  Bartlett's test:"),
                        paste0("    K-sq = ", format(bart_res$statistic, digits = 5),
                               "   df = ", bart_res$parameter,
                               "   p = ", format(bart_res$p.value, digits = 4)),
                        paste0("    Interpretation : ", hom_tag)
                    )
                } else {
                    out <- c(out, "  [Homogeneity test failed to run]")
                }
            }
            out <- c(out, "",
                "  Note: if HETEROGENEOUS, use a heteroscedastic LMM",
                "        (glmmTMB with dispformula, or lme with weights = varIdent(~1|Cohort)).",
                ""
            )

            # ── 4. Sphericity (Mauchly's W) ────────────────────────────
            out <- c(out,
                "  4. SPHERICITY  (Mauchly's W test)",
                hr
            )
            if (length(weeks) < 3) {
                out <- c(out, "  [Sphericity test requires >= 3 time points; skipped]", "")
            } else {
                # Need wide format with complete subjects
                wide_df <- reshape(df_r[, c("ID", "Week", "DV")],
                                   idvar   = "ID",
                                   timevar = "Week",
                                   direction = "wide")
                dv_cols <- grep("^DV\\.", names(wide_df), value = TRUE)
                Y <- as.matrix(wide_df[, dv_cols])
                Y <- Y[complete.cases(Y), ]
                if (nrow(Y) >= ncol(Y) + 1) {
                    idata <- data.frame(Week = factor(seq_len(ncol(Y))))
                    mauch <- tryCatch({
                        fit_mlm <- lm(Y ~ 1)
                        mauchly.test(fit_mlm, idata = idata,
                                     X = ~ 1)
                    }, error = function(e) NULL)
                    if (!is.null(mauch)) {
                        sph_tag <- if (mauch$p.value >= 0.05) "sphericity not violated (p >= .05)" else
                                   if (mauch$p.value >= 0.01) "marginal violation (p < .05) -- use GG/HF correction" else
                                   "VIOLATED (p < .01) -- Greenhouse-Geisser correction required"
                        out <- c(out,
                            paste0("  Mauchly's W = ", format(mauch$statistic, digits = 5),
                                   "   p = ", format(mauch$p.value, digits = 4),
                                   "   df = ", mauch$parameter),
                            paste0("  Interpretation : ", sph_tag),
                            ""
                        )
                    } else {
                        out <- c(out, "  [Mauchly's test failed to run]", "")
                    }
                } else {
                    out <- c(out, "  [Insufficient complete subjects for Mauchly's test]", "")
                }
            }

            # ── 5. LMM residual normality ──────────────────────────────
            out <- c(out,
                "  5. LMM RESIDUAL NORMALITY  (lme4::lmer, if available)",
                hr
            )
            has_lme4 <- requireNamespace("lme4", quietly = TRUE)
            if (has_lme4) {
                lmm_res <- tryCatch(
                    lme4::lmer(DV ~ Cohort + Week + (1 | ID),
                               data = df_r, REML = TRUE,
                               control = lme4::lmerControl(optimizer = "bobyqa")),
                    error   = function(e) NULL,
                    warning = function(w) {
                        tryCatch(
                            lme4::lmer(DV ~ Cohort + Week + (1 | ID),
                                       data = df_r, REML = TRUE),
                            error = function(e2) NULL)
                    }
                )
                if (!is.null(lmm_res)) {
                    res_vals <- residuals(lmm_res)
                    sw_res   <- tryCatch(shapiro.test(res_vals), error = function(e) NULL)
                    if (!is.null(sw_res)) {
                        res_interp <- if (sw_res$p.value >= 0.05) "residuals appear normally distributed" else
                                      if (sw_res$p.value >= 0.01) "marginal non-normality of residuals" else
                                      "NON-NORMAL residuals -- consider transformation"
                        out <- c(out,
                            paste0("  Model: DV ~ Cohort + Week + (1 | ID)  [main-effects LMM]"),
                            paste0("  Residual Shapiro-Wilk:"),
                            paste0("    W = ", format(sw_res$statistic, digits = 6),
                                   "   p = ", format(sw_res$p.value, digits = 4),
                                   "   n_res = ", length(res_vals)),
                            paste0("    Interpretation : ", res_interp)
                        )
                    } else {
                        out <- c(out, "  [Residual S-W test failed]")
                    }
                } else {
                    out <- c(out, "  [lme4::lmer failed to converge; residuals not evaluated]")
                }
            } else {
                out <- c(out,
                    "  [lme4 not installed; install with: install.packages('lme4')]",
                    "  Skipping LMM residual check."
                )
            }
            out <- c(out, "")

            # ── 6. Outlier detection (IQR-based, per cell) ─────────────
            out <- c(out,
                "  6. OUTLIER DETECTION  (IQR-based, per Cohort \u00d7 Week cell)",
                hr,
                sprintf("  %-28s  %-10s  %8s  %8s  %8s  %s",
                        "Cell", "ID", "Value", "Q1", "Q3", "Flag")
            )
            n_outliers <- 0L
            for (coh in cohorts) {
                for (wk in weeks) {
                    mask <- df_r$Cohort == coh & df_r$Week == wk
                    sub  <- df_r[mask, ]
                    sub  <- sub[!is.na(sub$DV), ]
                    if (nrow(sub) < 4) next
                    q1  <- quantile(sub$DV, 0.25)
                    q3  <- quantile(sub$DV, 0.75)
                    iqr <- q3 - q1
                    lo  <- q1 - 1.5 * iqr
                    hi  <- q3 + 1.5 * iqr
                    hits <- sub[sub$DV < lo | sub$DV > hi, ]
                    if (nrow(hits) > 0) {
                        cell_lbl <- paste0(coh, " W", wk)
                        for (i in seq_len(nrow(hits))) {
                            flag <- if (hits$DV[i] > hi) "HIGH" else "LOW"
                            out <- c(out, sprintf(
                                "  %-28s  %-10s  %8.3f  %8.3f  %8.3f  %s",
                                cell_lbl, as.character(hits$ID[i]),
                                hits$DV[i], q1, q3, flag))
                            n_outliers <- n_outliers + 1L
                        }
                    }
                }
            }
            if (n_outliers == 0L) {
                out <- c(out, "  No IQR-based outliers detected in any cell.")
            } else {
                out <- c(out, "",
                    paste0("  Total outlier observations: ", n_outliers),
                    "  Consider reviewing these points before fitting the model."
                )
            }
            out <- c(out, "")

            # ── 7. Cullen-Frey diagram ──────────────────────────────────
            out <- c(out,
                "  7. CULLEN-FREY DIAGRAM  (fitdistrplus optional)",
                hr
            )
            has_fd <- requireNamespace("fitdistrplus", quietly = TRUE)
            cf_status <- if (has_fd) {
                tryCatch({
                    grDevices::png(r_cf_fp, width = 900, height = 700, res = 110)
                    fitdistrplus::descdist(vals, discrete = FALSE, boot = 500)
                    graphics::title(main = paste0("Cullen-Frey: ", r_meas_name,
                                                  "  (weekly means, pooled across cohorts)"))
                    grDevices::dev.off()
                    paste0("  Cullen-Frey plot saved : ", r_cf_fp)
                }, error = function(e) {
                    tryCatch(grDevices::dev.off(), error = function(e2) NULL)
                    paste0("  [Cullen-Frey failed: ", conditionMessage(e), "]")
                })
            } else {
                "  [fitdistrplus not installed -- install with: install.packages('fitdistrplus')]"
            }
            out <- c(out, cf_status, "")

            # ── Overall recommendation ─────────────────────────────────
            sw_ok  <- is.null(sw_all) || sw_all$p.value >= 0.05
            res_ok <- !has_lme4 || TRUE   # assessed above individually
            out <- c(out,
                "  RECOMMENDATION",
                hr,
                if (sw_ok)
                    "  Pooled data appear normally distributed."
                else
                    "  Pooled data depart from normality -- inspect per-cell results.",
                "  For a Cohort x Week repeated-measures design with continuous unbounded DV:",
                "    Primary model : gaussian LMM  (lme4::lmer or nlme::lme)",
                "                    DV ~ Cohort * Week + (1 | ID)",
                "    If residuals non-normal : consider log(DV + offset) or sqrt transformation",
                "    If heteroscedastic      : add dispformula or varIdent weights by Cohort",
                "    For sphericity violation: Greenhouse-Geisser or Huynh-Feldt correction",
                "                             (pingouin / ezANOVA automatically apply GG)",
                ""
            )

            con <- file(r_txt_fp, encoding = "UTF-8")
            writeLines(out, con)
            close(con)
        """)

        txt_content = (
            Path(_txt_fp.replace('/', os.sep)).read_text(encoding='utf-8', errors='replace')
            if Path(_txt_fp.replace('/', os.sep)).exists()
            else "[R output file not found]"
        )
        result = {'text': txt_content}

    except Exception as _e:
        txt_content = f"  [R error: {_e}]"
        result = {'error': str(_e)}

    finally:
        for _fp in [_data_fp, _txt_fp]:
            try:
                real = _fp.replace('/', os.sep)
                if os.path.exists(real):
                    os.unlink(real)
            except Exception:
                pass

    W2 = 80
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header_lines = [
        "=" * W2,
        "  DISTRIBUTION DIAGNOSTICS + MIXED-MODEL ASSUMPTION CHECKS",
        f"  Measure : {measure}",
        "  Design  : Cohort (between-subjects) \u00d7 Week (within-subjects)",
        "=" * W2,
        f"  Generated  : {now_str}",
        f"  Cohorts    : {', '.join(cohort_labels)}",
        f"  Weeks      : {', '.join(str(w) for w in week_vals)}",
        f"  N subjects : {n_subj}  |  N obs (weekly means) : {n_obs}",
        "=" * W2,
        "",
    ]
    full_report = "\n".join(header_lines) + "\n" + txt_content

    print("\n" + full_report)

    if save_path is not None:
        Path(save_path).write_text(full_report, encoding='utf-8')
        print(f"\n[OK] Diagnostics report saved -> {save_path}")

    return result


# =============================================================================

def calculate_animal_slopes(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_unit: str = "Week",
    combined_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate linear regression slope for each animal's weight change over time.    
    For each animal, fits a linear regression: measure ~ time
    Returns slopes, intercepts, R^2, and metadata for all animals.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_unit: Time variable to use ('Week' or 'Day')
        combined_df: Optional pre-built, properly aligned combined dataframe. When
            provided the internal combine/clean/align steps are skipped. For
            time_unit='Week' the dataframe must already have a 'Week' column.
        
    Returns:
        DataFrame with columns: ID, Sex, CA (%), Cohort, Slope, Intercept, R2, N_points
        
    Example:
        >>> slopes_df = calculate_animal_slopes(cohorts, measure="Total Change", time_unit="Week")
        >>> print(slopes_df)
    """
    from scipy import stats
    
    print("\n" + "="*80)
    print(f"CALCULATING INDIVIDUAL ANIMAL SLOPES: {measure} vs {time_unit}")
    print("="*80)
    
    if combined_df is not None:
        # Use pre-built, properly aligned dataframe (e.g. with ramp day offset applied)
        print("\nStep 1: Using pre-built combined dataframe...")
        working_df = combined_df.copy()
    else:
        # Build from raw cohort dicts
        print("\nStep 1: Combining cohort dataframes...")
        working_df = combine_cohorts_for_analysis(cohort_dfs)
        working_df = clean_cohort(working_df)
        if 'Day' not in working_df.columns:
            working_df = add_day_column_across_cohorts(working_df)
        if time_unit == "Week" and 'Week' not in working_df.columns:
            working_df = _add_week_column_across_cohorts(working_df)
    
    required_cols = ['ID', 'Sex', 'Cohort', time_unit, measure]
    missing_cols = [col for col in required_cols if col not in working_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # When using Week as the time axis, aggregate to one representative value per
    # animal per week (mean across days) so each animal contributes exactly one
    # data point per week to the regression rather than 7 repeated x-values.
    if time_unit == "Week":
        print("\nAggregating to weekly means (one value per animal per week)...")
        agg_dict: dict = {measure: 'mean', 'Sex': 'first', 'Cohort': 'first'}
        if 'CA (%)' in working_df.columns:
            agg_dict['CA (%)'] = 'first'
        analysis_df = (
            working_df.groupby(['ID', 'Week'])
            .agg(agg_dict)
            .reset_index()
            .dropna(subset=[measure])
        )
    else:
        keep_cols = [c for c in ['ID', 'Sex', 'CA (%)', 'Cohort', time_unit, measure]
                     if c in working_df.columns]
        analysis_df = working_df[keep_cols].copy().dropna(subset=[measure])
    
    print(f"\nAnalyzing: {measure} vs {time_unit}")
    print(f"  Total observations (rows): {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  Cohorts: {sorted(analysis_df['Cohort'].unique())}")
    
    # Calculate slopes for each animal
    print(f"\nStep 2: Fitting linear regression for each animal...")
    
    slope_results = []
    
    for animal_id in analysis_df['ID'].unique():
        animal_data = analysis_df[analysis_df['ID'] == animal_id].copy()
        
        # Get metadata
        sex = animal_data['Sex'].iloc[0]
        ca_pct = animal_data['CA (%)'].iloc[0]
        cohort = animal_data['Cohort'].iloc[0]
        
        # Get time and measure values
        x = animal_data[time_unit].values
        y = animal_data[measure].values
        
        # Skip if insufficient data points
        if len(x) < 2:
            print(f"  Warning: Skipping {animal_id} - insufficient data points (n={len(x)})")
            continue
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        slope_results.append({
            'ID': animal_id,
            'Sex': sex,
            'CA (%)': ca_pct,
            'Cohort': cohort,
            'Slope': slope,
            'Intercept': intercept,
            'R2': r_value**2,
            'P_value': p_value,
            'Std_Error': std_err,
            'N_points': len(x)
        })
    
    slopes_df = pd.DataFrame(slope_results)
    
    print(f"\n[OK] Calculated slopes for {len(slopes_df)} animals")
    
    # Display summary statistics by cohort
    print("\n" + "="*80)
    print("SLOPE SUMMARY BY COHORT")
    print("="*80)
    
    for cohort_label in sorted(slopes_df['Cohort'].unique()):
        cohort_slopes = slopes_df[slopes_df['Cohort'] == cohort_label]['Slope']
        print(f"\n{cohort_label} (n={len(cohort_slopes)} animals):")
        print(f"  Mean Slope:   {cohort_slopes.mean():.4f} {measure} per {time_unit}")
        print(f"  Median Slope: {cohort_slopes.median():.4f} {measure} per {time_unit}")
        print(f"  SD:           {cohort_slopes.std():.4f}")
        print(f"  Range:        [{cohort_slopes.min():.4f}, {cohort_slopes.max():.4f}]")
        print(f"  Mean R^2:      {slopes_df[slopes_df['Cohort'] == cohort_label]['R2'].mean():.4f}")
    
    return slopes_df


def compare_slopes_within_cohorts(slopes_df: pd.DataFrame) -> Dict:
    """
    Compare slopes within each cohort using descriptive statistics and variance tests.
    
    This analyzes the variability of slopes within each CA% group.
    
    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()
        
    Returns:
        Dictionary with within-cohort statistics and Levene's test results    """
    from scipy import stats
    
    print("\n" + "="*80)
    print("WITHIN-COHORT SLOPE VARIABILITY ANALYSIS")
    print("="*80)
    
    results = {
        'cohort_stats': [],
        'levene_test': None
    }
    
    # Descriptive statistics + one-sample t-test vs zero by cohort
    for cohort_label in sorted(slopes_df['Cohort'].unique()):
        cohort_slopes = slopes_df[slopes_df['Cohort'] == cohort_label]['Slope'].values
        
        # One-sample t-test: is the mean slope different from zero?
        t_stat_zero, p_val_zero = stats.ttest_1samp(cohort_slopes, popmean=0)
        df_zero = len(cohort_slopes) - 1
        
        cohort_stat = {
            'Cohort': cohort_label,
            'N': len(cohort_slopes),
            'Mean': cohort_slopes.mean(),
            'Median': np.median(cohort_slopes),
            'SD': cohort_slopes.std(),
            'SEM': cohort_slopes.std() / np.sqrt(len(cohort_slopes)),
            'Min': cohort_slopes.min(),
            'Max': cohort_slopes.max(),
            'IQR': np.percentile(cohort_slopes, 75) - np.percentile(cohort_slopes, 25),
            'CV': (cohort_slopes.std() / cohort_slopes.mean() * 100) if cohort_slopes.mean() != 0 else np.nan,
            'ttest_vs_zero': {
                'statistic': t_stat_zero,
                'p_value': p_val_zero,
                'df': df_zero,
            },
        }
        
        results['cohort_stats'].append(cohort_stat)
        
        print(f"\n{cohort_label} (n={cohort_stat['N']}):")
        print(f"  Mean +/- SEM:         {cohort_stat['Mean']:.4f} +/- {cohort_stat['SEM']:.4f}")
        print(f"  Median (IQR):       {cohort_stat['Median']:.4f} ({cohort_stat['IQR']:.4f})")
        print(f"  SD:                 {cohort_stat['SD']:.4f}")
        print(f"  Coefficient of Var: {cohort_stat['CV']:.2f}%")
        print(f"  Range:              [{cohort_stat['Min']:.4f}, {cohort_stat['Max']:.4f}]")
        print(f"  One-sample t-test (slope vs 0): t({df_zero}) = {t_stat_zero:.4f}, p = {p_val_zero:.4f}")
        direction = 'gaining' if cohort_slopes.mean() > 0 else 'losing'
        if p_val_zero < 0.05:
            print(f"    -> Cohort is significantly {direction} weight over time (p < 0.05)")
        else:
            print(f"    -> Slope does not significantly differ from zero (p >= 0.05)")
    
    # Levene's test for equality of variances between cohorts
    cohort_groups = sorted(slopes_df['Cohort'].unique())
    
    if len(cohort_groups) >= 2:
        print("\n" + "-"*80)
        print("LEVENE'S TEST: Equality of Variances Between Cohorts")
        print("-"*80)
        
        group_slopes = [slopes_df[slopes_df['Cohort'] == c]['Slope'].values
                       for c in cohort_groups]
        
        levene_stat, levene_p = stats.levene(*group_slopes)
        
        results['levene_test'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'cohort_groups': cohort_groups
        }
        
        print(f"Levene's statistic: W = {levene_stat:.4f}")
        print(f"P-value: p = {levene_p:.4f}")
        
        if levene_p < 0.05:
            print("Result: Variances are significantly different between cohorts (p < 0.05)")
        else:
            print("Result: No significant difference in variances between cohorts (p >= 0.05)")
    
    return results


def compare_slopes_between_cohorts(slopes_df: pd.DataFrame) -> Dict:
    """
    Statistically compare average slopes between cohorts (2 or more cohorts).

    For 2 cohorts:
      - Welch's t-test, Mann-Whitney U, Cohen's d, 95% CI for mean difference

    For 3+ cohorts:
      - Kruskal-Wallis omnibus test (non-parametric)
      - One-way ANOVA with Welch correction (Brown-Forsythe via scipy)
      - All pairwise MWU tests with Holm-Bonferroni correction
      - Per-pair Cohen's d and Hodges-Lehmann shift with bootstrap 95% CI

    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()

    Returns:
        Dictionary with test results and effect sizes
    """
    from scipy import stats
    from itertools import combinations as _comb

    print("\n" + "="*80)
    print("BETWEEN-COHORT SLOPE COMPARISON")
    print("="*80)

    ca_groups = sorted(slopes_df['Cohort'].unique())
    n_groups = len(ca_groups)

    if n_groups < 2:
        print(f"Warning: Need at least 2 cohorts, found {n_groups}. Returning empty results.")
        return {}

    group_slopes = [slopes_df[slopes_df['Cohort'] == g]['Slope'].values for g in ca_groups]

    print(f"\nCohorts: {', '.join(ca_groups)}")
    for g, s in zip(ca_groups, group_slopes):
        print(f"  {g}: n={len(s)}, Mean={s.mean():.4f}, SD={s.std():.4f}")

    results = {
        'ca_groups': ca_groups,
        'group_stats': [
            {'cohort': g, 'n': len(s), 'mean': float(s.mean()), 'sd': float(s.std())}
            for g, s in zip(ca_groups, group_slopes)
        ],
    }

    # -------------------------------------------------------------------------
    # 1. Omnibus tests
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("1. OMNIBUS TESTS")
    print("-"*80)

    # Kruskal-Wallis (non-parametric)
    kw_stat, kw_p = stats.kruskal(*group_slopes)
    print(f"Kruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.4f}")
    if kw_p < 0.05:
        print("  -> At least one cohort differs significantly (KW, p < 0.05)")
    else:
        print("  -> No significant omnibus difference (KW, p >= 0.05)")
    results['kruskal_wallis'] = {'statistic': kw_stat, 'p_value': kw_p}

    # One-way ANOVA (standard F-test via scipy f_oneway)
    f_stat, f_p = stats.f_oneway(*group_slopes)
    print(f"One-way ANOVA   F = {f_stat:.4f}, p = {f_p:.4f}")
    if f_p < 0.05:
        print("  -> At least one cohort differs significantly (ANOVA, p < 0.05)")
    else:
        print("  -> No significant omnibus difference (ANOVA, p >= 0.05)")
    results['anova'] = {'statistic': f_stat, 'p_value': f_p}

    # -------------------------------------------------------------------------
    # 2. Pairwise MWU with Holm-Bonferroni correction + Cohen's d + HL CI
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("2. PAIRWISE COMPARISONS (Holm-Bonferroni corrected MWU)")
    print("-"*80)

    pairs = list(_comb(range(n_groups), 2))
    _rng = np.random.default_rng(0)
    pair_data = []
    for seed_i, (i, j) in enumerate(pairs):
        a, b = group_slopes[i], group_slopes[j]
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            continue
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
        r_rb = 1.0 - 2.0 * float(u_stat) / (n_a * n_b)
        _diffs = (np.asarray(a)[:, None] - np.asarray(b)[None, :]).ravel()
        hl_est = float(np.median(_diffs))
        _rng_pair = np.random.default_rng(seed_i)
        _boot = [
            float(np.median(
                (_rng_pair.choice(a, n_a, replace=True)[:, None]
                 - _rng_pair.choice(b, n_b, replace=True)[None, :]).ravel()
            ))
            for _ in range(2000)
        ]
        hl_ci_lo = float(np.percentile(_boot, 2.5))
        hl_ci_hi = float(np.percentile(_boot, 97.5))
        # Cohen's d (pooled SD)
        pooled_sd = np.sqrt(((n_a - 1) * a.std()**2 + (n_b - 1) * b.std()**2)
                            / (n_a + n_b - 2)) if (n_a + n_b - 2) > 0 else np.nan
        cohens_d = float((b.mean() - a.mean()) / pooled_sd) if pooled_sd and pooled_sd > 0 else float('nan')
        if abs(cohens_d) < 0.2:
            d_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            d_interp = "small"
        elif abs(cohens_d) < 0.8:
            d_interp = "medium"
        else:
            d_interp = "large"
        pair_data.append({
            'idx_a': i, 'idx_b': j,
            'label_a': ca_groups[i], 'label_b': ca_groups[j],
            'n_a': n_a, 'n_b': n_b,
            'U': float(u_stat),
            'p_raw': float(u_p),
            'p_corrected': 0.0,   # filled after Holm
            'rank_biserial_r': r_rb,
            'hodges_lehmann': hl_est,
            'ci_95_lo': hl_ci_lo,
            'ci_95_hi': hl_ci_hi,
            'cohens_d': cohens_d,
            'd_interpretation': d_interp,
            'mean_diff': float(b.mean() - a.mean()),
        })

    # Apply Holm-Bonferroni
    if pair_data:
        _p_raw = [r['p_raw'] for r in pair_data]
        _n = len(_p_raw)
        _indexed = sorted(enumerate(_p_raw), key=lambda x: x[1])
        _running_max = 0.0
        _p_adj = [0.0] * _n
        for _rank, (_orig, _p) in enumerate(_indexed):
            _a_val = min(1.0, float(_p) * (_n - _rank))
            _a_val = max(_a_val, _running_max)
            _p_adj[_orig] = _a_val
            _running_max = _a_val
        for r, pa in zip(pair_data, _p_adj):
            r['p_corrected'] = pa

    for r in pair_data:
        sig = '***' if r['p_corrected'] < 0.001 else ('**' if r['p_corrected'] < 0.01
              else ('*' if r['p_corrected'] < 0.05 else 'ns'))
        print(f"  {r['label_a']} vs {r['label_b']}: "
              f"U={r['U']:.2f}, p_raw={r['p_raw']:.4f}, p_adj={r['p_corrected']:.4f} {sig}, "
              f"r_rb={r['rank_biserial_r']:.3f}, d={r['cohens_d']:.3f} ({r['d_interpretation']})")

    results['pairwise'] = pair_data

    # For backward-compatibility: if exactly 2 cohorts, keep legacy keys
    if n_groups == 2 and pair_data:
        r0 = pair_data[0]
        a, b = group_slopes[0], group_slopes[1]
        results['n_0'] = r0['n_a']
        results['n_1'] = r0['n_b']
        results['mean_0'] = float(a.mean())
        results['mean_1'] = float(b.mean())
        results['sd_0'] = float(a.std())
        results['sd_1'] = float(b.std())
        results['mean_diff'] = r0['mean_diff']
        results['mann_whitney'] = {
            'statistic': r0['U'], 'p_value': r0['p_corrected'],
            'rank_biserial_r': r0['rank_biserial_r'],
            'hodges_lehmann': r0['hodges_lehmann'],
            'ci_95_lo': r0['ci_95_lo'], 'ci_95_hi': r0['ci_95_hi'],
        }
        # Welch's t-test (only meaningful for 2 groups)
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        s1_sq, s2_sq = a.var(), b.var()
        n1, n2 = len(a), len(b)
        _df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
        results['t_test'] = {'statistic': t_stat, 'p_value': t_p, 'df': _df}
        results['effect_size'] = {
            'cohens_d': r0['cohens_d'],
            'interpretation': r0['d_interpretation'],
        }
        se_diff = np.sqrt(s1_sq/n1 + s2_sq/n2)
        t_crit = stats.t.ppf(0.975, _df)
        results['confidence_interval'] = {
            'mean_diff': r0['mean_diff'],
            'se_diff': se_diff,
            'ci_95_lower': r0['mean_diff'] - t_crit * se_diff,
            'ci_95_upper': r0['mean_diff'] + t_crit * se_diff,
        }

    return results


def _format_mwu_slope_report(
    pair_results: list,
    measure: str,
    time_unit: str,
) -> str:
    """Format MWU bracket-test results for slope comparison as a text report."""
    n_pairs = len(pair_results)
    corr_note = (
        f"Holm-Bonferroni step-down corrected across {n_pairs} pairs"
        if n_pairs > 1 else "no correction (single pair)"
    )
    lines = [
        "=" * 90,
        "MANN-WHITNEY U TEST RESULTS  --  Slope Comparison Brackets",
        "=" * 90,
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Measure    : {measure}",
        f"Time unit  : {time_unit}",
        f"Correction : {corr_note}",
        "",
        "Field definitions:",
        "  U        : Mann-Whitney U statistic",
        "  p(raw)   : Two-sided p-value (uncorrected)",
        "  p(adj)   : Holm-Bonferroni step-down corrected p-value",
        "  r_rb     : Rank-biserial correlation (effect size r = 1 - 2U/(nA*nB))",
        "             |r| < 0.3 = small, 0.3-0.5 = medium, > 0.5 = large",
        "  HL_est   : Hodges-Lehmann location shift (median of all pairwise diffs A-B)",
        "  95% CI   : Bootstrap CI (n=2 000 resamples) on Hodges-Lehmann estimator",
        "",
    ]
    hdr = (
        f"{'Comparison':<38} {'nA':>4} {'nB':>4}  "
        f"{'U':>9}  {'p(raw)':>9}  {'p(adj)':>9}  "
        f"{'r_rb':>7}  {'HL_est':>9}  {'95% CI'}"
    )
    lines.append(hdr)
    lines.append("-" * 100)

    def _fp(p: float) -> str:
        return "< 0.001" if p < 0.001 else f"{p:.4f}"

    for r in pair_results:
        comp = f"{r['label_a']} vs {r['label_b']}"
        ci_s = f"[{r['ci_95_lo']:.4f}, {r['ci_95_hi']:.4f}]"
        lines.append(
            f"{comp:<38} {r['n_a']:>4} {r['n_b']:>4}  "
            f"{r['U']:>9.2f}  {_fp(r['p_raw']):>9}  {_fp(r['p_corrected']):>9}  "
            f"{r['rank_biserial_r']:>7.4f}  {r['hodges_lehmann']:>9.4f}  {ci_s}"
        )
    lines += [
        "",
        "Significance (after correction): * p<0.05   ** p<0.01   *** p<0.001",
        "=" * 90,
    ]
    return "\n".join(lines)


def plot_slopes_comparison(
    slopes_df: pd.DataFrame,
    measure: str = "Total Change",
    time_unit: str = "Week",
    title: Optional[Path] = None,
    save_path: Optional[Path] = None,
    save_mwu_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create visualization comparing slope distributions between cohorts.
    
    Creates a figure with multiple subplots:
    1. Box plot comparing slopes between cohorts
    2. Individual data points with means
    3. Histogram/density plot of slope distributions
    
    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()
        measure: Measure name for axis labels
        time_unit: Time unit for axis labels
        title: Optional custom title
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    ca_groups = sorted(slopes_df['Cohort'].unique())
    colors = [_cohort_label_to_color(c) for c in ca_groups]
    positions = list(range(len(ca_groups)))
    box_data = [slopes_df[slopes_df['Cohort'] == c]['Slope'].values for c in ca_groups]

    # Clean tick labels: strip parenthetical suffixes like "(6 animals)"
    import re as _re
    tick_labels = [_re.sub(r'\s*\(.*?\)', '', g).strip() for g in ca_groups]

    fig, ax = plt.subplots()

    # Bar plot: mean ± SEM per cohort
    bar_means = [np.mean(d) if len(d) > 0 else 0.0 for d in box_data]
    bar_sems  = [stats.sem(d) if len(d) > 1 else 0.0 for d in box_data]
    ax.bar(positions, bar_means, width=0.65, color=colors, alpha=0.7,
           yerr=bar_sems, error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='black'),
           zorder=2)

    # Overlay individual points
    for i, ca_val in enumerate(ca_groups):
        cohort_slopes = slopes_df[slopes_df['Cohort'] == ca_val]['Slope'].values
        x_jitter = np.random.normal(i, 0.04, size=len(cohort_slopes))
        ax.scatter(x_jitter, cohort_slopes, alpha=0.6, s=10,
                   color=colors[i], edgecolors='black', linewidths=0.5, zorder=3)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.7, len(ca_groups) - 1 + 0.7)
    ax.set_ylim(-6, 6)
    ax.set_yticks(range(-6, 7, 2))
    ax.set_xlabel('Cohort')
    ax.set_ylabel(f'Slope ({measure} per {time_unit})')
    ax.tick_params(direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pairwise Mann-Whitney U tests with significance brackets
    from itertools import combinations
    pairs = list(combinations(range(len(ca_groups)), 2))
    n_pairs = len(pairs)
    step = 0.35
    tick_h = step * 0.15
    _data_max = max((float(np.max(d)) for d in box_data if len(d) > 0), default=0.0)
    y_top = min(_data_max + 0.3, 5.9 - step * n_pairs - tick_h)

    _mwu_bracket_results = []
    for level, (i, j) in enumerate(pairs):
        a, b = box_data[i], box_data[j]
        if len(a) < 2 or len(b) < 2:
            continue
        u_stat, p_raw = stats.mannwhitneyu(a, b, alternative='two-sided')
        n_a, n_b = len(a), len(b)
        r_rb = 1.0 - 2.0 * float(u_stat) / (n_a * n_b)
        _diffs = (np.asarray(a)[:, None] - np.asarray(b)[None, :]).ravel()
        hl_est = float(np.median(_diffs))
        _rng_br = np.random.default_rng(level)
        _boot_br = [
            float(np.median(
                (_rng_br.choice(a, n_a, replace=True)[:, None]
                 - _rng_br.choice(b, n_b, replace=True)[None, :]).ravel()
            ))
            for _ in range(2000)
        ]
        _mwu_bracket_results.append({
            'idx_a': i, 'idx_b': j,
            'label_a': ca_groups[i], 'label_b': ca_groups[j],
            'n_a': n_a, 'n_b': n_b,
            'U': float(u_stat),
            'p_raw': float(p_raw),
            'p_corrected': 0.0,    # filled in below after Holm
            'rank_biserial_r': r_rb,
            'hodges_lehmann': hl_est,
            'ci_95_lo': float(np.percentile(_boot_br, 2.5)),
            'ci_95_hi': float(np.percentile(_boot_br, 97.5)),
        })

    # Apply Holm-Bonferroni step-down correction across all valid pairs
    if _mwu_bracket_results:
        _p_raw_all = [r['p_raw'] for r in _mwu_bracket_results]
        _n = len(_p_raw_all)
        _indexed = sorted(enumerate(_p_raw_all), key=lambda x: x[1])
        _running_max = 0.0
        _p_adj = [0.0] * _n
        for _rank, (_orig, _p) in enumerate(_indexed):
            _a = min(1.0, float(_p) * (_n - _rank))
            _a = max(_a, _running_max)
            _p_adj[_orig] = _a
            _running_max = _a
        for _r, _pa in zip(_mwu_bracket_results, _p_adj):
            _r['p_corrected'] = _pa

    for level, (i, j) in enumerate(pairs):
        # Find matching result (skip pairs that were filtered out)
        _match = next(
            (r for r in _mwu_bracket_results if r['idx_a'] == i and r['idx_b'] == j),
            None,
        )
        if _match is None:
            continue
        p_corr = _match['p_corrected']
        if p_corr < 0.001:
            label = '***'
        elif p_corr < 0.01:
            label = '**'
        elif p_corr < 0.05:
            label = '*'
        else:
            label = 'ns'

        y_bracket = y_top + step * (level + 1)
        x1, x2 = positions[i], positions[j]
        ax.plot([x1, x1, x2, x2], [y_bracket - tick_h, y_bracket, y_bracket, y_bracket - tick_h],
                color='black', linewidth=0.5)
        ax.text((x1 + x2) / 2, y_bracket + tick_h * 0.5, label,
                ha='center', va='bottom', fontsize=7)

    if save_mwu_path is not None and _mwu_bracket_results:
        _mwu_lines = _format_mwu_slope_report(
            _mwu_bracket_results, measure, time_unit
        )
        Path(save_mwu_path).write_text(_mwu_lines, encoding='utf-8')
        print(f"  [OK] MWU report saved -> {save_mwu_path}")

    if title is None:
        title = f'Slope Analysis: {measure} vs {time_unit}'
    ax.set_title(title)

    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"\n[OK] Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def generate_slope_analysis_report(
    slopes_df: pd.DataFrame,
    within_results: Dict,
    between_results: Dict,
    measure: str = "Total Change",
    time_unit: str = "Week"
) -> str:
    """
    Generate a comprehensive text report of slope analysis results.
    
    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()
        within_results: Results from compare_slopes_within_cohorts()
        between_results: Results from compare_slopes_between_cohorts()
        measure: Measure name
        time_unit: Time unit name
        
    Returns:
        Formatted text report as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SLOPE ANALYSIS REPORT: RATE OF WEIGHT CHANGE ACROSS COHORTS")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Measure: {measure}")
    lines.append(f"Time Unit: {time_unit}")
    lines.append(f"Total Animals: {len(slopes_df)}")
    
    # Individual animal slopes
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 1: INDIVIDUAL ANIMAL SLOPES")
    lines.append("=" * 80)
    lines.append(f"\nEach animal's {measure} was fit with a linear regression model:")
    lines.append(f"    {measure} = Slope  x  {time_unit} + Intercept")
    lines.append(f"\nThe slope represents the rate of change in {measure} per {time_unit}.")
    
    lines.append("\n" + "-" * 80)
    lines.append("Individual Animal Results:")
    lines.append("-" * 80)
    
    for ca_val in sorted(slopes_df['Cohort'].unique()):
        cohort_data = slopes_df[slopes_df['Cohort'] == ca_val].sort_values('Slope', ascending=False)
        lines.append(f"\n{ca_val} (n={len(cohort_data)}):")
        
        for _, row in cohort_data.iterrows():
            lines.append(f"  {row['ID']:8s} ({row['Sex']}): "
                        f"Slope = {row['Slope']:7.4f}, "
                        f"R^2 = {row['R2']:.3f}, "
                        f"p = {row['P_value']:.4f}")
    
    # Within-cohort variability
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 2: WITHIN-COHORT VARIABILITY")
    lines.append("=" * 80)
    lines.append("\nThis section examines the variability of slopes within each cohort.")
    
    for cohort_stat in within_results['cohort_stats']:
        lines.append(f"\n{cohort_stat['Cohort']}:")
        lines.append(f"  Sample Size:           n = {cohort_stat['N']}")
        lines.append(f"  Mean Slope:            {cohort_stat['Mean']:.4f} {measure} per {time_unit}")
        lines.append(f"  Standard Error (SEM):  {cohort_stat['SEM']:.4f}")
        lines.append(f"  Standard Deviation:    {cohort_stat['SD']:.4f}")
        lines.append(f"  Median Slope:          {cohort_stat['Median']:.4f}")
        lines.append(f"  Interquartile Range:   {cohort_stat['IQR']:.4f}")
        lines.append(f"  Range:                 [{cohort_stat['Min']:.4f}, {cohort_stat['Max']:.4f}]")
        lines.append(f"  Coefficient of Var:    {cohort_stat['CV']:.2f}%")
        if 'ttest_vs_zero' in cohort_stat:
            t0 = cohort_stat['ttest_vs_zero']
            lines.append(f"  One-Sample t-test (slope vs 0):")
            lines.append(f"    t({t0['df']}) = {t0['statistic']:.4f}, p = {t0['p_value']:.4f}")
            direction = 'gaining' if cohort_stat['Mean'] > 0 else 'losing'
            if t0['p_value'] < 0.001:
                lines.append(f"    Result: Cohort is significantly {direction} weight over time (p < 0.001)")
            elif t0['p_value'] < 0.01:
                lines.append(f"    Result: Cohort is significantly {direction} weight over time (p < 0.01)")
            elif t0['p_value'] < 0.05:
                lines.append(f"    Result: Cohort is significantly {direction} weight over time (p < 0.05)")
            else:
                lines.append(f"    Result: Slope does not significantly differ from zero (p >= 0.05)")
    
    if within_results['levene_test']:
        levene = within_results['levene_test']
        lines.append("\n" + "-" * 80)
        lines.append("Levene's Test for Equality of Variances:")
        lines.append("-" * 80)
        lines.append(f"  Statistic: W = {levene['statistic']:.4f}")
        lines.append(f"  P-value:   p = {levene['p_value']:.4f}")
        if levene['p_value'] < 0.05:
            lines.append("  Result: Variances are significantly different between cohorts (p < 0.05)")
        else:
            lines.append("  Result: No significant difference in variances (p >= 0.05)")
    
    # Between-cohort comparison
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 3: BETWEEN-COHORT COMPARISON")
    lines.append("=" * 80)

    ca_groups = sorted(slopes_df['Cohort'].unique())
    n_groups = len(ca_groups)

    lines.append(f"\nThis section compares the average slopes across {n_groups} cohort(s).")

    if not between_results:
        lines.append("\n[WARNING] Between-cohort comparison could not be performed.")
        lines.append("Please ensure you have loaded at least 2 cohort CSV files.")
    else:
        # Group summary table
        lines.append("\nGroup Means:")
        lines.append("-" * 80)
        for gs in between_results.get('group_stats', []):
            lines.append(f"  {gs['cohort']:<30} n={gs['n']:>3},  Mean={gs['mean']:>8.4f},  SD={gs['sd']:>7.4f}")

        # Omnibus tests
        lines.append("\n" + "-" * 80)
        lines.append("Omnibus Tests:")
        lines.append("-" * 80)
        if 'kruskal_wallis' in between_results:
            kw = between_results['kruskal_wallis']
            kw_sig = "p < 0.001" if kw['p_value'] < 0.001 else (
                     "p < 0.01" if kw['p_value'] < 0.01 else (
                     "p < 0.05" if kw['p_value'] < 0.05 else "p >= 0.05 (ns)"))
            lines.append(f"  Kruskal-Wallis: H = {kw['statistic']:.4f}, p = {kw['p_value']:.4f}  [{kw_sig}]")
        if 'anova' in between_results:
            an = between_results['anova']
            an_sig = "p < 0.001" if an['p_value'] < 0.001 else (
                     "p < 0.01" if an['p_value'] < 0.01 else (
                     "p < 0.05" if an['p_value'] < 0.05 else "p >= 0.05 (ns)"))
            lines.append(f"  One-way ANOVA:  F = {an['statistic']:.4f}, p = {an['p_value']:.4f}  [{an_sig}]")
        # Legacy 2-cohort Welch t-test
        if 't_test' in between_results:
            tt = between_results['t_test']
            tt_sig = "p < 0.001" if tt['p_value'] < 0.001 else (
                     "p < 0.01" if tt['p_value'] < 0.01 else (
                     "p < 0.05" if tt['p_value'] < 0.05 else "p >= 0.05 (ns)"))
            lines.append(f"  Welch's t-test: t({tt['df']:.2f}) = {tt['statistic']:.4f}, p = {tt['p_value']:.4f}  [{tt_sig}]")

        # Pairwise comparisons
        pairwise = between_results.get('pairwise', [])
        if pairwise:
            lines.append("\n" + "-" * 80)
            lines.append("Pairwise Comparisons (Holm-Bonferroni corrected MWU):")
            lines.append("-" * 80)

            def _fp(p: float) -> str:
                return "< 0.001" if p < 0.001 else f"{p:.4f}"

            hdr = (f"  {'Comparison':<38} {'nA':>4} {'nB':>4}  "
                   f"{'U':>9}  {'p(raw)':>9}  {'p(adj)':>9}  "
                   f"{'r_rb':>7}  {'d':>7}  {'HL_est':>9}  {'95% CI'}")
            lines.append(hdr)
            lines.append("  " + "-" * 98)
            for r in pairwise:
                sig = ('***' if r['p_corrected'] < 0.001 else
                       ('**' if r['p_corrected'] < 0.01 else
                        ('*' if r['p_corrected'] < 0.05 else 'ns')))
                comp = f"{r['label_a']} vs {r['label_b']}"
                ci_s = f"[{r['ci_95_lo']:.4f}, {r['ci_95_hi']:.4f}]"
                lines.append(
                    f"  {comp:<38} {r['n_a']:>4} {r['n_b']:>4}  "
                    f"{r['U']:>9.2f}  {_fp(r['p_raw']):>9}  {_fp(r['p_corrected']):>9}  "
                    f"{r['rank_biserial_r']:>7.4f}  {r['cohens_d']:>7.4f}  "
                    f"{r['hodges_lehmann']:>9.4f}  {ci_s}  {sig}"
                )
            lines.append("")
            lines.append("  Significance (after Holm-Bonferroni correction):")
            lines.append("  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant")
            lines.append("  r_rb = rank-biserial r  |  d = Cohen's d  |  HL_est = Hodges-Lehmann shift (A-B)")
            lines.append("  95% CI = bootstrap CI (n=2 000 resamples) on Hodges-Lehmann estimator")

        # Legacy 2-cohort CI block
        if 'confidence_interval' in between_results:
            ci = between_results['confidence_interval']
            lines.append("\n" + "-" * 80)
            lines.append("95% Confidence Interval for Mean Difference (Welch):")
            lines.append("-" * 80)
            lines.append(f"  Mean Difference: {ci['mean_diff']:.4f}")
            lines.append(f"  95% CI: [{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}]")
            if ci['ci_95_lower'] * ci['ci_95_upper'] > 0:
                lines.append("  Interpretation: CI does not include zero (significant difference)")
            else:
                lines.append("  Interpretation: CI includes zero (no significant difference)")

    # Interpretation and conclusion
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 4: INTERPRETATION AND CONCLUSIONS")
    lines.append("=" * 80)

    if not between_results:
        lines.append(f"\n[WARNING] Between-cohort comparison could not be completed.")
        lines.append("Please run the analysis again with at least 2 cohort CSV files selected.")
    else:
        pairwise = between_results.get('pairwise', [])
        # Pick the most relevant omnibus p-value
        omnibus_p = between_results.get('kruskal_wallis', {}).get('p_value', 1.0)
        sig_pairs = [r for r in pairwise if r['p_corrected'] < 0.05]

        if omnibus_p < 0.05:
            lines.append(
                f"\nThe omnibus test detected a significant difference in slopes across cohorts "
                f"(Kruskal-Wallis p = {omnibus_p:.4f})."
            )
            if sig_pairs:
                lines.append(
                    f"Pairwise Holm-Bonferroni-corrected MWU identified {len(sig_pairs)} "
                    f"significant pair(s):"
                )
                for r in sig_pairs:
                    direction = "higher" if r['mean_diff'] > 0 else "lower"
                    lines.append(
                        f"  {r['label_a']} vs {r['label_b']}: "
                        f"{r['label_b']} slope is {abs(r['mean_diff']):.4f} {measure}/{time_unit} "
                        f"{direction} than {r['label_a']} "
                        f"(p_adj = {r['p_corrected']:.4f}, d = {r['cohens_d']:.3f} [{r['d_interpretation']}])"
                    )
            else:
                lines.append(
                    "No individual pairwise contrast survived Holm-Bonferroni correction "
                    "(omnibus significance may reflect modest effect spread)."
                )
        else:
            lines.append(
                f"\nNo significant omnibus difference in slopes across cohorts "
                f"(Kruskal-Wallis p = {omnibus_p:.4f})."
            )
            lines.append(
                "All cohorts appear to change weight at a similar rate."
            )
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def perform_complete_slope_analysis(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_unit: str = "Week",
    save_plot: bool = True,
    save_report: bool = True,
    output_dir: Optional[Path] = None,
    combined_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Perform complete slope analysis: calculate slopes, compare within/between cohorts,
    generate plots and report.
    
    This is a convenience function that runs the entire analysis pipeline.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_unit: Time variable to use ('Week' or 'Day')
        save_plot: Whether to save the visualization
        save_report: Whether to save the text report
        output_dir: Directory for saving outputs (default: current directory)
        combined_df: Optional pre-built combined dataframe (see calculate_animal_slopes).
        
    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "="*80)
    print("COMPLETE SLOPE ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Calculate individual slopes
    slopes_df = calculate_animal_slopes(
        cohort_dfs, measure=measure, time_unit=time_unit, combined_df=combined_df
    )
    
    # Step 2: Compare within cohorts
    within_results = compare_slopes_within_cohorts(slopes_df)
    
    # Step 3: Compare between cohorts
    between_results = compare_slopes_between_cohorts(slopes_df)
    
    # Step 4: Generate report
    report_text = generate_slope_analysis_report(
        slopes_df, within_results, between_results, 
        measure=measure, time_unit=time_unit
    )
    
    print("\n" + "="*80)
    print("REPORT PREVIEW")
    print("="*80)
    print(report_text)
    
    # Prepare output directory and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: Save report to file
    if save_report:
        report_path = output_dir / f"slope_analysis_report_{measure.replace(' ', '_')}_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n[OK] Report saved to: {report_path}")
    
    # Step 6: Create visualization
    if HAS_MATPLOTLIB:
        plot_path = output_dir / f"slope_analysis_{measure.replace(' ', '_')}_{timestamp}.svg" if save_plot else None
        mwu_path  = output_dir / f"slope_analysis_mwu_{measure.replace(' ', '_')}_{timestamp}.txt" if save_report else None
        
        fig = plot_slopes_comparison(
            slopes_df, 
            measure=measure, 
            time_unit=time_unit,
            save_path=plot_path,
            save_mwu_path=mwu_path,
            show=True
        )
    
    return {
        'slopes_df': slopes_df,
        'within_results': within_results,
        'between_results': between_results,
        'report_text': report_text,
        'measure': measure,
        'time_unit': time_unit
    }


# =============================================================================


# =============================================================================
# TEST/DEMO FUNCTIONS
# =============================================================================

def test_load_two_cohorts():
    """
    Interactive test: Use GUI to select two cohort files and verify they load successfully.
    """
    print("\n" + "="*80)
    print("INTERACTIVE COHORT LOADER - SELECT 2 COHORT FILES")
    print("="*80)
    print("\nInstructions:")
    print("  1. File dialog will open - select your first cohort CSV file")
    print("  2. File dialog will open again - select your second cohort CSV file")
    print("  3. DataFrames will be loaded and validated")
    print("\n")
    
    # Use GUI to select and load cohorts
    cohorts = select_and_load_cohorts(n_cohorts=2)
    
    if not cohorts:
        print("\n[ERROR] No cohorts were loaded. Please try again.")
        return None
    
    # Verify dataframes were created
    print("\n" + "="*80)
    print("DATAFRAME VERIFICATION")
    print("="*80)
    
    all_valid = True
    for label, df in cohorts.items():
        print(f"\nCohort: {label}")
        
        # Check if it's a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"  [ERROR] Not a valid DataFrame!")
            all_valid = False
            continue
        
        # Check if it has data
        if df.empty:
            print(f"  [WARNING] DataFrame is empty!")
            all_valid = False
            continue
        
        # Success - show basic info
        print(f"  [OK] Valid DataFrame created")
        print(f"  [OK] Shape: {df.shape[0]} rows  x  {df.shape[1]} columns")
        
        if "ID" in df.columns:
            n_animals = df["ID"].nunique()
            print(f"  [OK] Contains {n_animals} unique animals")
        
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"], errors="coerce")
            n_dates = dates.notna().sum()
            print(f"  [OK] Contains {n_dates} date entries")
    
    # Final summary
    print("\n" + "="*80)
    if all_valid:
        print("[SUCCESS] All cohorts loaded successfully!")
        print(f"Total cohorts: {len(cohorts)}")
        print("="*80)
        
        # Show detailed preview
        preview_cohorts(cohorts, n_rows=3)
        
        # Show summary table
        summary = summarize_cohorts(cohorts)
        
        return cohorts
    else:
        print("[WARNING] Some cohorts had issues. Check messages above.")
        print("="*80)
        return cohorts


def test_slope_analysis():
    """
    Interactive test: Select two cohort files and perform complete slope analysis.
    
    This will:
    1. Load two cohort CSV files via GUI
    2. Calculate linear slopes for each animal's Total Change vs Week    3. Compare slopes within each cohort
    4. Compare slopes between cohorts
    5. Generate plots and report
    """
    print("\n" + "="*80)
    print("INTERACTIVE SLOPE ANALYSIS TEST")
    print("="*80)
    print("\nThis analysis will:")
    print("  1. Load two cohort CSV files")
    print("  2. Calculate rate of weight change (slope) for each animal")
    print("  3. Compare slopes within and between cohorts")
    print("  4. Generate plots and statistical report")
    print("\n")
    
    # Load cohorts
    cohorts = test_load_two_cohorts()
    
    if not cohorts or len(cohorts) < 2:
        print("\n[ERROR] Need at least 2 cohorts for slope analysis.")
        return None
    
    # Ask user for parameters
    print("\n" + "="*80)
    print("ANALYSIS PARAMETERS")
    print("="*80)
    
    # Measure selection
    print("\nSelect measure to analyze:")
    print("  1. Total Change (default)")
    print("  2. Daily Change")
    print("  3. Weight")
    
    measure_choice = input("\nEnter choice (1-3) [default: 1]: ").strip()
    
    if measure_choice == "2":
        measure = "Daily Change"
    elif measure_choice == "3":
        measure = "Weight"
    else:
        measure = "Total Change"
    
    # Time unit selection
    print("\nSelect time unit:")
    print("  1. Week (default)")
    print("  2. Day")
    
    time_choice = input("\nEnter choice (1-2) [default: 1]: ").strip()
    
    if time_choice == "2":
        time_unit = "Day"
    else:
        time_unit = "Week"
    
    print(f"\n[OK] Analysis parameters:")
    print(f"  Measure:   {measure}")
    print(f"  Time Unit: {time_unit}")
    
    # Perform complete analysis
    print("\n" + "="*80)
    print("RUNNING SLOPE ANALYSIS...")
    print("="*80)
    
    results = perform_complete_slope_analysis(
        cohorts,
        measure=measure,
        time_unit=time_unit,
        save_plot=True,
        save_report=True
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] Slope analysis completed!")
    print("="*80)
    print("\nResults saved to current directory:")
    print("  - Plot: slope_analysis_*.svg")
    print("  - Report: slope_analysis_report_*.txt")
    
    return results


# =============================================================================
# WEEK-ALIGNED CROSS-COHORT FRAMEWORK
# Supports comparing 2-3 cohorts regardless of experiment type (ramp or nonramp).
# Uses 'Cohort' label as the between-subjects factor and 'Week' (1-indexed
# chronological measurement order) as the within-subjects factor, so ramp and
# nonramp cohorts are aligned on the same Week 1-N axis.
#
# Usage example:
#
#   cohort_metadata = {
#       "0% nonramp": {"ca_schedule": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}},
#       "2% nonramp": {"ca_schedule": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}},
#       "ramp":       {"ca_schedule": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}},
#   }
#   combined = combine_cohorts_with_weeks(cohort_dfs, cohort_metadata)
#   results  = perform_cohort_week_mixed_anova(cohort_dfs, "Total Change",
#                                               cohort_metadata)
#   report   = generate_cohort_week_report(results, save_path="report.txt")
#   fig      = plot_weekly_means_by_cohort(combined, "Total Change",
#                                          cohort_metadata)
# =============================================================================

def combine_cohorts_with_weeks(
    cohort_dfs: Dict[str, pd.DataFrame],
    cohort_metadata: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Combine cohort DataFrames into a single DataFrame aligned on Week number.

    Works for both ramp and nonramp cohorts.  Week numbers are derived from each
    animal's own measurement timeline (Week 1 = first measurement week, Week 2 =    second, ...), so cohorts starting on different calendar dates still line up.

    Adds a 'CA_at_week' column from either:
      - the raw 'CA (%)' column in the data  (for ramp cohorts that store it per row), or
      - 'ca_schedule' in cohort_metadata  (for nonramp cohorts that don't have CA per row).
    Parameters
    ----------
    cohort_dfs : dict
        Mapping of cohort label -> DataFrame (as returned by load_cohorts()).
    cohort_metadata : dict, optional
        Mapping of cohort label -> dict with optional keys:
          'ca_schedule' : {week_int: ca_float}  e.g. {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 'Cohort', 'Week', and 'CA_at_week' columns
        (plus all original measurement columns).
    """
    print("\n" + "=" * 80)
    print("COMBINING COHORTS -- WEEK-ALIGNED FRAMEWORK")
    print("=" * 80)

    combined = combine_cohorts_for_analysis(cohort_dfs)
    combined = clean_cohort(combined)

    if 'Day' not in combined.columns:
        combined = add_day_column_across_cohorts(combined)
    combined = _add_week_column_across_cohorts(combined)

    combined = combined[combined['Day'] >= 1].copy()

    # ------------------------------------------------------------------ #
    # Build CA_at_week annotation
    # ------------------------------------------------------------------ #
    # For cohorts that carry CA(%) in the raw data (e.g. ramp), convert
    # the string column "0%"/"1%"/... to a float and call it CA_at_week.
    if 'CA (%)' in combined.columns:
        def _parse_ca_str(val):
            if isinstance(val, str):
                return float(val.strip('%'))
            try:
                return float(val)
            except (TypeError, ValueError):
                return np.nan
        combined['CA_at_week'] = combined['CA (%)'].apply(_parse_ca_str)
    else:
        combined['CA_at_week'] = np.nan

    # For cohorts where CA_at_week is still NaN, fill from cohort_metadata
    if cohort_metadata:
        for label, meta in cohort_metadata.items():
            sched = meta.get('ca_schedule', {})
            if not sched:
                continue
            mask = (combined['Cohort'] == label) & combined['CA_at_week'].isna()
            if mask.any():
                combined.loc[mask, 'CA_at_week'] = (
                    combined.loc[mask, 'Week'].map(
                        {int(k): float(v) for k, v in sched.items()}
                    )
                )

    # Summary
    for label in combined['Cohort'].unique():
        sub = combined[combined['Cohort'] == label]
        n_animals = sub['ID'].nunique()
        weeks = sorted(sub['Week'].dropna().unique().astype(int))
        print(f"\n  {label}: {n_animals} animals, weeks {weeks}")
        if 'CA_at_week' in sub.columns and sub['CA_at_week'].notna().any():
            sched_found = (
                sub.groupby('Week')['CA_at_week'].first()
                .sort_index()
                .dropna()
                .to_dict()
            )
            print(f"    CA schedule (week->%): { {int(k): v for k, v in sched_found.items()} }")

    print(f"\n[OK] Combined DataFrame: {len(combined)} rows, "
          f"{combined['ID'].nunique()} animals, "
          f"{combined['Cohort'].nunique()} cohorts")
    print("=" * 80)
    return combined


def perform_cohort_week_mixed_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    cohort_metadata: Optional[Dict[str, Dict]] = None,
    include_sex: bool = True,
) -> Dict:
    """
    2- or 3-way Mixed ANOVA: Week (within)  x  Cohort [ x  Sex] (between).

    Works for any combination of ramp and nonramp cohorts because it uses
    'Cohort' label -- not CA(%) -- as the between-subjects factor.

    Design
    ------
    - Within-subjects:   Week  (1 ... N, chronological measurement order)
    - Between-subjects:  Cohort [ x  Sex]  (combined into a 'Group' label so
                         pingouin's single-between-factor API can handle it)    - Post-hoc:
        - Week pairwise (paired Bonferroni, within-subjects)
        - Cohort pairwise on per-animal means (independent Bonferroni)
        - Simple effects: Week RM-ANOVA within each Cohort
        - Simple effects: one-way ANOVA + pairwise for Cohort at each Week

    Only animals with complete records across every retained week are kept
    (list-wise deletion).

    Parameters
    ----------
    cohort_dfs : dict
        Mapping of cohort label -> DataFrame.
    measure : str
        Numeric column to analyse (e.g. 'Total Change', 'Daily Change', 'Weight').
    cohort_metadata : dict, optional
        Passed through to combine_cohorts_with_weeks() for CA annotation.
    include_sex : bool
        Whether to include Sex as a between-subjects factor.  Automatically
        set to False if sex data are absent or single-valued.

    Returns
    -------
    dict with keys:
        'measure', 'include_sex', 'n_cohorts', 'n_animals', 'n_weeks',
        'mixed_anova_table', 'between_anova_table',
        'posthoc'  (dict with 'week_pairwise', 'cohort_pairwise',
                    'simple_effects_week_by_cohort',
                    'simple_effects_cohort_by_week'),
        'descriptive_stats', 'data', 'cohort_metadata'
    """
    print("\n" + "=" * 80)
    print("COHORT  x  WEEK MIXED ANOVA  (week-aligned cross-cohort framework)")
    print(f"  Measure: {measure}")
    print("=" * 80)

    if not HAS_PINGOUIN:
        print("[ERROR] pingouin is required. Install with: pip install pingouin")
        return {}

    # ------------------------------------------------------------------ #
    # Prepare combined DataFrame
    # ------------------------------------------------------------------ #
    combined = combine_cohorts_with_weeks(cohort_dfs, cohort_metadata)

    required = ['ID', 'Week', 'Cohort', measure]
    if include_sex:
        required.append('Sex')
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns in combined data: {missing}")

    cols = list(required)
    analysis_df = combined[cols].dropna(subset=['ID', 'Week', 'Cohort', measure]).copy()

    # Check whether sex data are usable
    if include_sex and 'Sex' in analysis_df.columns:
        valid_sexes = analysis_df['Sex'].dropna().unique()
        valid_sexes = [s for s in valid_sexes if s not in ('nan', 'None', '')]
        if len(valid_sexes) < 2:
            print("\n[WARNING] Sex data absent or single-valued -- running Cohort  x  Week only")
            include_sex = False
            analysis_df = analysis_df.drop(columns=['Sex'], errors='ignore')

    # ------------------------------------------------------------------ #
    # Aggregate to one value per (animal, week) -- average over days
    # ------------------------------------------------------------------ #
    group_cols = ['ID', 'Week', 'Cohort']
    if include_sex:
        group_cols.append('Sex')
    analysis_df = (
        analysis_df.groupby(group_cols, dropna=False)[measure]
        .mean()
        .reset_index()
    )

    # Complete-case filter
    print(f"\n  Animals before completeness filter: {analysis_df['ID'].nunique()}")
    analysis_df = _filter_complete_subjects_weekly(analysis_df, 'ID', 'Week')
    n_animals = analysis_df['ID'].nunique()
    n_weeks = int(analysis_df['Week'].nunique())

    print(f"  Cohorts:  {sorted(analysis_df['Cohort'].unique())}")
    print(f"  Animals:  {n_animals}")
    print(f"  Weeks:    {sorted(analysis_df['Week'].dropna().unique().astype(int))}")

    if n_animals < 3:
        print("[WARNING] Too few animals for reliable ANOVA results.")

    # ------------------------------------------------------------------ #
    # Descriptive statistics
    # ------------------------------------------------------------------ #
    desc_cols = ['Cohort', 'Week'] + (['Sex'] if include_sex else [])
    desc_rows = []
    for keys, grp in analysis_df.groupby(desc_cols)[measure]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(grp)
        mean = grp.mean()
        std = grp.std(ddof=1) if n > 1 else np.nan
        sem = std / np.sqrt(n) if (n > 0 and np.isfinite(std)) else np.nan
        row = dict(zip(desc_cols, keys))
        row.update({'n': n, 'mean': mean, 'std': std, 'sem': sem,
                    'ci_lower': mean - 1.96 * sem if np.isfinite(sem) else np.nan,
                    'ci_upper': mean + 1.96 * sem if np.isfinite(sem) else np.nan})
        desc_rows.append(row)
    desc_df = pd.DataFrame(desc_rows)

    # ------------------------------------------------------------------ #
    # Build Group label (Cohort [+ Sex])
    # ------------------------------------------------------------------ #
    analysis_df = analysis_df.copy()
    if include_sex:
        analysis_df['Group'] = analysis_df['Cohort'] + ' | ' + analysis_df['Sex'].astype(str)
    else:
        analysis_df['Group'] = analysis_df['Cohort']

    # ------------------------------------------------------------------ #
    # Primary mixed ANOVA: Week (within)  x  Group (between)
    # ------------------------------------------------------------------ #
    print("\nRunning mixed ANOVA (Week within  x  Group between)...")
    try:
        aov_main = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Week',
            between='Group',
            subject='ID',
            correction='auto',
        )
        aov_main['Source'] = aov_main['Source'].replace({'Interaction': 'Week * Group'})
        p_col = 'p-unc' if 'p-unc' in aov_main.columns else 'p_unc'
        print(aov_main[['Source', 'F', p_col]].to_string(index=False))
    except Exception as e:
        print(f"[ERROR] Mixed ANOVA failed: {e}")
        import traceback; traceback.print_exc()
        return {}

    # ------------------------------------------------------------------ #
    # Between-subjects decomposition: Cohort [ x  Sex] on per-animal means
    # ------------------------------------------------------------------ #
    subj_mean_cols = ['ID', 'Cohort'] + (['Sex'] if include_sex else [])
    subject_means = (
        analysis_df.groupby(subj_mean_cols)[measure]
        .mean()
        .reset_index()
    )
    between_cols = ['Cohort'] + (['Sex'] if include_sex else [])
    try:
        aov_between = pg.anova(
            data=subject_means,
            dv=measure,
            between=between_cols,
            ss_type=3,
        )
        _bp_col = 'p-unc' if 'p-unc' in aov_between.columns else 'p_unc'
        print("\nBetween-subjects decomposition (Cohort [ x  Sex]):")
        print(aov_between[['Source', 'F', _bp_col]].to_string(index=False))
    except Exception as e:
        print(f"[WARNING] Between-subjects decomposition failed: {e}")
        aov_between = None

    # ------------------------------------------------------------------ #
    # Post-hoc tests
    # ------------------------------------------------------------------ #
    posthoc: Dict = {}

    # 1. Week pairwise -- within-subjects, Bonferroni
    try:
        pw_week = pg.pairwise_tests(
            data=analysis_df,
            dv=measure,
            within='Week',
            subject='ID',
            parametric=True,
            padjust='bonferroni',
            effsize='hedges',
        )
        posthoc['week_pairwise'] = pw_week
    except Exception as e:
        posthoc['week_pairwise'] = f"Error: {e}"

    # 2. Cohort pairwise -- between-subjects on subject means, Bonferroni
    try:
        if subject_means['Cohort'].nunique() >= 2:
            pw_cohort = pg.pairwise_tests(
                data=subject_means,
                dv=measure,
                between='Cohort',
                parametric=True,
                padjust='bonferroni',
                effsize='cohen',
            )
            posthoc['cohort_pairwise'] = pw_cohort
    except Exception as e:
        posthoc['cohort_pairwise'] = f"Error: {e}"

    # 3. Simple effects: Week RM-ANOVA within each Cohort
    simple_week_by_cohort: Dict = {}
    for cohort_label in sorted(analysis_df['Cohort'].unique()):
        coh_data = analysis_df[analysis_df['Cohort'] == cohort_label]
        if coh_data['Week'].nunique() < 2:
            continue
        try:
            rm = pg.rm_anova(
                data=coh_data,
                dv=measure,
                within='Week',
                subject='ID',
                detailed=True,
                correction=True,
            )
            simple_week_by_cohort[cohort_label] = rm
        except Exception as e:
            simple_week_by_cohort[cohort_label] = f"Error: {e}"
    posthoc['simple_effects_week_by_cohort'] = simple_week_by_cohort

    # 4. Simple effects: Cohort effect at each Week
    simple_cohort_by_week: Dict = {}
    for week_num in sorted(analysis_df['Week'].dropna().unique()):
        wk_data = analysis_df[analysis_df['Week'] == week_num]
        if wk_data['Cohort'].nunique() < 2:
            continue
        try:
            groups = [grp[measure].dropna().values
                      for _, grp in wk_data.groupby('Cohort')]
            f_stat, p_val = stats.f_oneway(*groups)
            pw = pg.pairwise_tests(
                data=wk_data,
                dv=measure,
                between='Cohort',
                parametric=True,
                padjust='bonferroni',
            )
            simple_cohort_by_week[int(week_num)] = {
                'f_stat': f_stat, 'p_val': p_val, 'pairwise': pw,
            }
        except Exception as e:
            simple_cohort_by_week[int(week_num)] = {'error': str(e)}
    posthoc['simple_effects_cohort_by_week'] = simple_cohort_by_week

    print("\n[OK] Analysis complete")
    return {
        'measure': measure,
        'include_sex': include_sex,
        'n_cohorts': analysis_df['Cohort'].nunique(),
        'n_animals': n_animals,
        'n_weeks': n_weeks,
        'mixed_anova_table': aov_main,
        'between_anova_table': aov_between,
        'posthoc': posthoc,
        'descriptive_stats': desc_df,
        'data': analysis_df,
        'cohort_metadata': cohort_metadata or {},
    }


def generate_cohort_week_report(
    results: Dict,
    save_path: Optional[Path] = None,
) -> str:
    """
    Format and optionally save a cross-cohort week-aligned ANOVA report.

    Parameters
    ----------
    results : dict
        Output of perform_cohort_week_mixed_anova().
    save_path : Path or str, optional
        If provided, the report is written to this file.

    Returns
    -------
    str
        Formatted report text (also printed to stdout).
    """
    lines: List[str] = []

    def _sig(p: float) -> str:
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return "ns"

    lines.append("=" * 80)
    lines.append("CROSS-COHORT WEEK-ALIGNED MIXED ANOVA REPORT")
    lines.append("=" * 80)

    if not results:
        lines.append("No results available.")
        out = "\n".join(lines)
        print(out)
        return out

    lines.append(f"\nMeasure : {results.get('measure', '?')}")
    lines.append(f"Cohorts : {results.get('n_cohorts', '?')}")
    lines.append(f"Animals : {results.get('n_animals', '?')}")
    lines.append(f"Weeks   : {results.get('n_weeks', '?')}")
    if results.get('include_sex'):
        lines.append("Sex     : included as between-subjects factor")
    else:
        lines.append("Sex     : not included (absent or single-valued)")

    # ------------------------------------------------------------------ #
    # Descriptive statistics
    # ------------------------------------------------------------------ #
    desc = results.get('descriptive_stats')
    if desc is not None and len(desc) > 0:
        lines.append("\n" + "-" * 80)
        lines.append("DESCRIPTIVE STATISTICS")
        lines.append("-" * 80)
        lines.append(desc.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Mixed ANOVA table
    # ------------------------------------------------------------------ #
    aov_main = results.get('mixed_anova_table')
    if aov_main is not None:
        lines.append("\n" + "-" * 80)
        lines.append("MIXED ANOVA: Week (within)  x  Group (between)")
        lines.append("-" * 80)
        lines.append(aov_main.to_string())
        p_col = 'p-unc' if 'p-unc' in aov_main.columns else 'p_unc'
        gg_col = 'p-GG-corr' if 'p-GG-corr' in aov_main.columns else None
        lines.append("\nInterpretation:")
        for _, row in aov_main.iterrows():
            p = row.get(p_col, np.nan)
            p_gg = row.get(gg_col, np.nan) if gg_col else np.nan
            p_report = p_gg if (gg_col and np.isfinite(p_gg)) else p
            sig = _sig(p_report) if np.isfinite(p_report) else "?"
            f = row.get('F', np.nan)
            lines.append(f"  {row['Source']}: F = {f:.3f}, p = {p_report:.4f} {sig}")

    # ------------------------------------------------------------------ #
    # Between-subjects decomposition
    # ------------------------------------------------------------------ #
    aov_between = results.get('between_anova_table')
    if aov_between is not None:
        lines.append("\n" + "-" * 80)
        lines.append("BETWEEN-SUBJECTS DECOMPOSITION (Cohort [ x  Sex])")
        lines.append("-" * 80)
        lines.append(aov_between.to_string())
        p_col = 'p-unc' if 'p-unc' in aov_between.columns else 'p_unc'
        lines.append("\nInterpretation:")
        for _, row in aov_between.iterrows():
            p = row.get(p_col, np.nan)
            sig = _sig(p) if np.isfinite(p) else "?"
            f = row.get('F', np.nan)
            lines.append(f"  {row['Source']}: F = {f:.3f}, p = {p:.4f} {sig}")

    # ------------------------------------------------------------------ #
    # Post-hoc: Week pairwise
    # ------------------------------------------------------------------ #
    posthoc = results.get('posthoc', {})
    pw_week = posthoc.get('week_pairwise')
    if isinstance(pw_week, pd.DataFrame) and len(pw_week) > 0:
        lines.append("\n" + "-" * 80)
        lines.append("POST-HOC: Week Pairwise (Bonferroni, within-subjects)")
        lines.append("-" * 80)
        lines.append(pw_week.to_string())

    # ------------------------------------------------------------------ #
    # Post-hoc: Cohort pairwise
    # ------------------------------------------------------------------ #
    pw_cohort = posthoc.get('cohort_pairwise')
    if isinstance(pw_cohort, pd.DataFrame) and len(pw_cohort) > 0:
        lines.append("\n" + "-" * 80)
        lines.append("POST-HOC: Cohort Pairwise (Bonferroni, between-subjects)")
        lines.append("-" * 80)
        lines.append(pw_cohort.to_string())

    # ------------------------------------------------------------------ #
    # Simple effects: Week by Cohort
    # ------------------------------------------------------------------ #
    simple_wk = posthoc.get('simple_effects_week_by_cohort', {})
    if isinstance(simple_wk, dict) and simple_wk:
        lines.append("\n" + "-" * 80)
        lines.append("SIMPLE EFFECTS: Week effect within each Cohort")
        lines.append("-" * 80)
        p_col_rm = 'p-unc'
        for cohort_label, rm in simple_wk.items():
            if isinstance(rm, pd.DataFrame):
                wk_rows = rm[rm['Source'].str.upper() == 'WEEK']
                if len(wk_rows) > 0:
                    row = wk_rows.iloc[0]
                    f = row.get('F', np.nan)
                    p = row.get(p_col_rm, np.nan)
                    gg_p = row.get('p-GG-corr', np.nan)
                    p_rep = gg_p if np.isfinite(gg_p) else p
                    sig = _sig(p_rep)
                    lines.append(f"  {cohort_label}: F = {f:.3f}, p = {p_rep:.4f} {sig}")
                else:
                    lines.append(f"  {cohort_label}: (no Week row in RM-ANOVA table)")
            else:
                lines.append(f"  {cohort_label}: {rm}")

    # ------------------------------------------------------------------ #
    # Simple effects: Cohort by Week
    # ------------------------------------------------------------------ #
    simple_coh = posthoc.get('simple_effects_cohort_by_week', {})
    if isinstance(simple_coh, dict) and simple_coh:
        lines.append("\n" + "-" * 80)
        lines.append("SIMPLE EFFECTS: Cohort effect at each Week")
        lines.append("-" * 80)
        for week_num in sorted(simple_coh.keys()):
            res = simple_coh[week_num]
            if isinstance(res, dict) and 'p_val' in res:
                f = res['f_stat']
                p = res['p_val']
                sig = _sig(p)
                lines.append(f"\n  Week {week_num}: F = {f:.3f}, p = {p:.4f} {sig}")
                pw = res.get('pairwise')
                if isinstance(pw, pd.DataFrame) and len(pw) > 0:
                    lines.append("  Pairwise comparisons:")
                    lines.append("  " + pw.to_string().replace("\n", "\n  "))
            else:
                err = res.get('error', str(res)) if isinstance(res, dict) else str(res)
                lines.append(f"  Week {week_num}: Error -- {err}")

    lines.append("\n" + "=" * 80)
    out = "\n".join(lines)
    print(out)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(out, encoding='utf-8')
        print(f"\n[OK] Report saved -> {save_path}")

    return out


def plot_weekly_means_by_cohort(
    combined_df: pd.DataFrame,
    measure: str = "Total Change",
    cohort_metadata: Optional[Dict[str, Dict]] = None,
    by_sex: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Plot mean +/- SEM of *measure* for each cohort across weeks.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Output of combine_cohorts_with_weeks().
    measure : str
        Column to plot.
    cohort_metadata : dict, optional
        If provided and a cohort has 'ca_schedule', the x-axis ticks are
        annotated with the CA% for that cohort (e.g. "Week 1\\n(0% CA)").
        The first cohort with a 'ca_schedule' in metadata is used for annotation.
    by_sex : bool
        If True, plot males and females as separate lines (dashed = females).
    title : str, optional
        Plot title.  Defaults to "<measure> by Cohort Across Weeks".
    save_path : Path or str, optional
        If provided, save figure here (PNG, SVG, etc.).
    show : bool
        Whether to call plt.show() after plotting.

    Returns
    -------
    matplotlib Figure or None if matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        return None

    if 'Cohort' not in combined_df.columns or 'Week' not in combined_df.columns:
        raise ValueError(
            "combined_df must have 'Cohort' and 'Week' columns. "
            "Use combine_cohorts_with_weeks() to build it."
        )

    cohorts = sorted(combined_df['Cohort'].dropna().unique())
    n_cohorts = len(cohorts)

    # Colour / marker palette -- supports up to 5 cohorts
    _MARKERS = ['o', 's', '^', 'D', 'v']
    color_map  = {c: _cohort_label_to_color(c) for c in cohorts}
    marker_map = {c: _MARKERS[i % len(_MARKERS)] for i, c in enumerate(cohorts)}

    fig, ax = plt.subplots()

    group_cols = ['Cohort', 'Week']
    if by_sex and 'Sex' in combined_df.columns:
        group_cols.append('Sex')

    for cohort_label in cohorts:
        coh_data = combined_df[combined_df['Cohort'] == cohort_label].copy()
        color  = color_map[cohort_label]
        marker = marker_map[cohort_label]

        if by_sex and 'Sex' in coh_data.columns:
            for sex, sdf in coh_data.groupby('Sex'):
                wk_stats = (
                    sdf.groupby('Week')[measure]
                    .agg(['mean', 'sem'])
                    .rename(columns={'mean': 'mean', 'sem': 'sem'})
                    .reset_index()
                    .sort_values('Week')
                )
                linestyle = '-' if str(sex) == 'M' else '--'
                sex_label = 'Males' if str(sex) == 'M' else 'Females'
                ax.errorbar(
                    wk_stats['Week'], wk_stats['mean'], yerr=wk_stats['sem'],
                    label=f"{cohort_label} ({sex_label})",
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=0.9, capsize=4, alpha=0.85,
                )
        else:
            wk_stats = (
                coh_data.groupby('Week')[measure]
                .agg(['mean', 'sem'])
                .reset_index()
                .sort_values('Week')
            )
            ax.errorbar(
                wk_stats['Week'], wk_stats['mean'], yerr=wk_stats['sem'],
                label=cohort_label,
                color=color, marker=marker,
                linewidth=0.9, capsize=4, alpha=0.9,
            )

    # X-axis annotation (CA% per week) from the first cohort with a schedule
    week_nums = sorted(combined_df['Week'].dropna().unique().astype(int))
    annotated = False
    if cohort_metadata:
        for lbl, meta in cohort_metadata.items():
            sched = meta.get('ca_schedule', {})
            if sched:
                tick_labels = []
                for wk in week_nums:
                    ca = sched.get(int(wk))
                    ca_str = f"\n({int(ca) if ca == int(ca) else ca}% CA)" if ca is not None else ""
                    tick_labels.append(f"Week {wk}{ca_str}")
                ax.set_xticks(week_nums)
                ax.set_xticklabels(tick_labels, fontsize=9)
                annotated = True
                break  # Only annotate once

    if not annotated:
        ax.set_xticks(week_nums)
        ax.set_xticklabels([f"Week {w}" for w in week_nums])

    ax.set_xlabel("Week")
    ax.set_ylabel(f"{measure} (Mean +/- SEM)")
    ax.set_title(title or f"{measure} by Cohort Across Weeks")

    if by_sex:
        # Add line-style key entries so the legend explains solid = Males, dashed = Females
        from matplotlib.lines import Line2D
        key_handles = [
            Line2D([0], [0], color='black', linewidth=1.0, linestyle='-',  label='-- Males'),
            Line2D([0], [0], color='black', linewidth=1.0, linestyle='--', label='- - Females'),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + key_handles,
            labels=labels + ['-- Males', '- - Females'],
            title="Cohort",
            loc="best",
            framealpha=0.9,
        )
    else:
        ax.legend(title="Cohort", loc="best", framealpha=0.9)

    apply_common_plot_style(ax)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[OK] Plot saved -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def _detect_comparison_type(cohorts: Dict[str, pd.DataFrame]) -> str:
    """
    Inspect loaded cohort labels and return a comparison-type tag:
      '0v2'    -- 0% vs 2% nonramp only
      '0vramp' -- 0% nonramp vs ramp
      '2vramp' -- 2% nonramp vs ramp
      'all3'   -- all three cohorts
      'unknown'-- cannot determine
    """
    labels_lower = [lbl.lower() for lbl in cohorts.keys()]

    has_zero  = any('0%' in l and 'ramp' not in l for l in labels_lower)
    has_two   = any('2%' in l and 'ramp' not in l for l in labels_lower)
    has_ramp  = any('ramp' in l for l in labels_lower)

    _TWOWK_KW = ('2wk', '2_wk', '2week', '2 wk', '2-wk', '2-week', '2 week')
    # A "2wk" label may not contain the word "ramp" (e.g. label == '2wk')
    has_twowk = any(any(kw in l for kw in _TWOWK_KW) for l in labels_lower)

    n = len(cohorts)
    # rampramp: one regular ramp label and one 2-week-ramp label (no CA% nonramp cohorts)
    if n == 2 and has_twowk and (has_ramp or has_twowk) and not has_zero and not has_two:
        return 'rampramp'
    if n == 3 and has_zero and has_two and has_ramp:
        return 'all3'
    if n == 2 and has_zero and has_two and not has_ramp:
        return '0v2'
    if n == 2 and has_zero and has_ramp and not has_two:
        return '0vramp'
    if n == 2 and has_two and has_ramp and not has_zero:
        return '2vramp'
    return 'unknown'


# ---------------------------------------------------------------------------
# Statistical Test Registry
# ---------------------------------------------------------------------------

def generate_test_registry_report(save_path=None) -> str:
    """Generate a formatted plain-text report documenting every statistical
    test used in across_cohort.py: data/variables consumed, library source,
    and every parameter with its meaning."""

    W = 80  # report line width

    def _h1(text):
        return ["=" * W, f"  {text}", "=" * W]

    def _h2(num, title):
        return [f"\n{'-' * W}", f"  TEST {num}  �  {title}", f"{'-' * W}", ""]

    def _sub(label):
        pad = W - 4 - len(label) - 2
        return [f"  {label}  {'�' * max(pad, 4)}", ""]

    def _tbl(rows, w1=12, w2=44, w3=20):
        hdr  = f"    {'Variable':<{w1}}  {'Description':<{w2}}  Data Type"
        sep  = f"    {'-'*w1}  {'-'*w2}  {'-'*w3}"
        body = [f"    {r[0]:<{w1}}  {r[1]:<{w2}}  {r[2]}" for r in rows]
        return [hdr, sep] + body

    def _out(rows, w1=14, w2=62):
        hdr  = f"    {'Column':<{w1}}  Meaning"
        sep  = f"    {'-'*w1}  {'-'*w2}"
        body = [f"    {r[0]:<{w1}}  {r[1]}" for r in rows]
        return [hdr, sep] + body

    # -- Header + Quick Reference -------------------------------------------- #
    lines = _h1("STATISTICAL TEST REGISTRY  �  across_cohort.py")
    lines += [
        "",
        "  Script: 0% vs 2% nonramp cross-cohort weight and behavioral analysis",
        "  Within-factor: Week  |  Between-factors: CA% (0 vs 2), Sex  |  a = 0.05",
        "",
        f"  QUICK REFERENCE  {'�' * 57}",
        "",
        f"    {'#':<3}  {'Test':<44}  Library / Function",
        f"    {'-'*3}  {'-'*44}  {'-'*26}",
        "    1    Mixed ANOVA  (CA% � Week � Sex)        pingouin / pg.mixed_anova()",
        "    2    Between-subjects ANOVA  (CA% � Sex)        pingouin / pg.anova()",
        "    3    Pairwise post-hoc  (Bonferroni)        pingouin / pg.pairwise_tests()",
        "    4    Cochran's Q  (binary outcomes across weeks)        statsmodels / cochrans_q()",
        "    5    McNemar  (pairwise week post-hoc, binary)        statsmodels / mcnemar()",
        "",
        "    Multiple comparisons:",
        "      Bonferroni  � Test 3  (padjust='bonf')  and  Test 5  (p � n_pairs)",
        "      No cross-measure correction  (each weight measure is a separate family)",
        "    Sphericity : correction=True  (GG always applied, not just when violated)",
        "=" * W,
    ]

    # -- TEST 1 ------------------------------------------------------------- #
    lines += _h2("1", "Mixed (Split-Plot) ANOVA  �  CA% � Week � Sex")
    lines += _sub("PURPOSE")
    lines += [
        "    Tests whether weight-change measures differ across weeks (within-subjects),",
        "    between CA% cohorts and sexes (between-subjects), and whether any two-way or",
        "    three-way interactions exist.  Primary longitudinal test for weight outcomes.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.mixed_anova()    import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "Long-format DataFrame  (one row per ID � Week)",       "pd.DataFrame"),
        ("dv",      "Measure  e.g. 'Total Change' or 'Daily Change'  (g)", "pd.Series[float64]"),
        ("within",  "'Week'  � repeated-measures factor  (integer index)",  "pd.Series[int]"),
        ("subject", "'ID'  � unique animal identifier",                     "pd.Series[str]"),
        ("between", "'Group'  (e.g. '0% Male') | 'Sex' | 'CA (%)'",        "pd.Series[str]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    correction   True  � Greenhouse-Geisser e correction is ALWAYS applied to",
        "                 within-subjects and interaction effects, regardless of whether",
        "                 Mauchly's sphericity test is violated.",
        "                 GG adjusts both numerator and denominator df downward, producing",
        "                 a conservative F-test even when sphericity holds.",
        "                 contrast: behavioral_analysis.py uses correction='auto'",
        "                 (GG only when p-spher < 0.05).",
        "",
        "    ss_type      Type III SS  (pingouin default; not set explicitly).",
        "",
    ]
    lines += _sub("OUTPUT  (key columns)")
    lines += _out([
        ("Source",  "Effect label: 'Week', 'Group', 'Sex', 'CA (%)', 'Week * Group', �"),
        ("ddof1",   "Numerator df  (GG-corrected because correction=True)"),
        ("ddof2",   "Denominator df  (GG-corrected)"),
        ("F",       "F-statistic"),
        ("p-unc",   "Unadjusted p  (note: GG df already used; compare to p-GG-corr)"),
        ("np2",     "Partial ?�  �  small = 0.01  |  medium = 0.06  |  large = 0.14"),
        ("eps",     "GG e  (1.0 = no sphericity violation; lower ? more df reduction)"),
    ])
    lines += ["", "    Threshold: a = 0.05"]

    # -- TEST 2 ------------------------------------------------------------- #
    lines += _h2("2", "Two-Way Between-Subjects ANOVA  �  CA% � Sex")
    lines += _sub("PURPOSE")
    lines += [
        "    Examines the cross-sectional between-subjects effects using per-animal means",
        "    averaged across all weeks.  Runs after or alongside the mixed ANOVA.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.anova()    import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "subject_means_clean  � one row per animal  (week-averaged)",  "pd.DataFrame"),
        ("dv",      "Measure column  (float, grams)",                              "pd.Series[float64]"),
        ("between", "['CA_percent', 'Sex']",                                       "list[str]"),
        ("",        "  CA_percent � CA concentration  (0 or 2)",                   "float/int"),
        ("",        "  Sex        � 'Male' | 'Female'",                            "str"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    ss_type   int � Sum-of-Squares type used to partition variance:",
        "              Type 1  sequential SS; each factor adjusted for entries before it;",
        "                      order-dependent � sensitive to factor entry order.",
        "              Type 2  hierarchical SS; main effects adjusted for each other",
        "                      but not interactions.",
        "              Type 3  marginal / orthogonal SS; each effect adjusted for all",
        "              (default) others including interactions.  Standard for unbalanced",
        "                      designs � the recommended choice here.",
        "",
    ]
    lines += _sub("OUTPUT  (key columns)")
    lines += _out([
        ("Source",  "'CA_percent', 'Sex', 'CA_percent * Sex', 'Residual'"),
        ("SS",      "Sum of squares for the effect"),
        ("DF",      "Degrees of freedom"),
        ("MS",      "Mean square  =  SS / DF"),
        ("F",       "F-statistic"),
        ("p-unc",   "Unadjusted p-value"),
        ("np2",     "Partial ?�"),
    ])
    lines += ["", "    Threshold: a = 0.05"]

    # -- TEST 3 ------------------------------------------------------------- #
    lines += _h2("3", "Pairwise Post-Hoc Tests  �  Bonferroni-corrected  (CA% or Sex)")
    lines += _sub("PURPOSE")
    lines += [
        "    Follow-up for a significant CA% or Sex main effect.  Compares all pairs",
        "    of groups on the weight measure (per-animal means across weeks).",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += ["    pingouin.pairwise_tests()    import pingouin as pg", ""]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("data",    "subject_means  � one row per animal",               "pd.DataFrame"),
        ("dv",      "Measure column  (float)",                           "pd.Series[float64]"),
        ("between", "'CA (%)'  � categorical grouping factor",           "pd.Series[str/float]"),
    ])
    lines.append("")
    lines += _sub("PARAMETERS")
    lines += [
        "    padjust   'bonf'  � Bonferroni correction: multiply each raw p by the number",
        "              of comparisons k; cap at 1.0.  Controls FWER at a = 0.05.",
        "              More conservative than BH-FDR used in lick scripts; chosen here",
        "              because the number of weight-measure comparisons is small.",
        "",
    ]
    lines += _sub("OUTPUT  (key columns)")
    lines += _out([
        ("A, B",    "The two groups being compared"),
        ("T",       "t-statistic"),
        ("dof",     "Degrees of freedom"),
        ("p-unc",   "Uncorrected p-value"),
        ("p-corr",  "Bonferroni-corrected p-value"),
        ("hedges",  "Hedges' g effect size  (bias-corrected Cohen's d)"),
    ])
    lines += ["", "    Threshold: a = 0.05 applied to Bonferroni-corrected p"]

    # -- TEST 4 ------------------------------------------------------------- #
    lines += _h2("4", "Cochran's Q Test  �  binary behavioral outcomes across weeks")
    lines += _sub("PURPOSE")
    lines += [
        "    Non-parametric repeated-measures test for binary (0/1) outcomes across weeks.",
        "    Analogous to a RM-ANOVA for proportions.  Used for nesting, lethargy,",
        "    anxious behaviors, and CA-spot digging.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += [
        "    statsmodels.stats.contingency_tables.cochrans_q()",
        "    from statsmodels.stats.contingency_tables import cochrans_q",
        "",
    ]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("wide", "Wide-format binary matrix:  rows = subjects,  columns = weeks",
         "pd.DataFrame[float64]"),
    ])
    lines += [
        "    Cell values: 0 (absent) or 1 (present).",
        "    Constructed by pivoting the behavioral DataFrame on Week; NaN ? 0.",
        "",
    ]
    lines += _sub("PARAMETERS  &  OUTPUT")
    lines += [
        "    wide  � positional DataFrame argument  (no additional keyword params)",
        "    Uses chi-squared approximation internally.",
        "",
        "    statistic   Cochran's Q  (float; chi-square distributed with df = k - 1)",
        "                where k = number of weeks",
        "    pvalue      asymptotic p-value  (H0: binary outcome probability equals",
        "                across all weeks)",
        "",
    ]
    lines += _sub("SUPPLEMENTARY PERMUTATION TEST")
    lines += [
        "    n_permutations : _n_perm = 5000",
        "    RNG seed       : np.random.default_rng(seed=42)  (for reproducibility)",
        "    Method: within each subject (row), shuffle binary responses across weeks;",
        "            re-compute Q each iteration.",
        "    Empirical p  =  proportion of permuted Q = observed Q",
        "    Threshold: a = 0.05",
    ]

    # -- TEST 5 ------------------------------------------------------------- #
    lines += _h2("5", "McNemar's Test  �  pairwise week post-hoc  (binary outcomes)")
    lines += _sub("PURPOSE")
    lines += [
        "    Performed after a significant Cochran's Q (Test 4).  Compares each pair of",
        "    weeks on the binary outcome for subjects observed at both weeks.",
        "",
    ]
    lines += _sub("LIBRARY")
    lines += [
        "    statsmodels.stats.contingency_tables.mcnemar()",
        "    from statsmodels.stats.contingency_tables import mcnemar",
        "",
    ]
    lines += _sub("INPUTS")
    lines += _tbl([
        ("table", "2�2 contingency table:  rows = response at week A (0/1),",
         "np.ndarray[int]"),
        ("",      "cols = response at week B (0/1)", ""),
    ])
    lines += [
        "    Constructed by cross-tabulating the binary outcome column between two weeks",
        "    for the subjects observed at both.",
        "",
    ]
    lines += _sub("PARAMETERS")
    lines += [
        "    exact        False  � chi-square approximation  (used when discordant pairs = 25;",
        "                 chi-square approximation is more powerful and common in practice).",
        "",
        "    correction   True   � Yates' continuity correction:  subtract 0.5 from",
        "                 |b - c| before squaring.  Reduces Type-I error for small",
        "                 numbers of discordant pairs.",
        "",
    ]
    lines += _sub("EFFECT SIZE  &  CORRECTION")
    lines += [
        "    Bonferroni: p_adj = min(p_raw � actual_n, 1.0)",
        "    where actual_n = number of week-pairs that produced a valid test result",
        "    (some pairs skipped when too few discordant pairs exist).",
        "",
        "    statistic   chi-square value  (or exact stat when exact=True)",
        "    pvalue      raw p-value; Bonferroni-adjusted p also reported",
        "    Threshold: a = 0.05 applied to Bonferroni-adjusted p",
    ]

    # -- SUMMARY ------------------------------------------------------------ #
    lines += [
        "",
        f"\n{'-' * W}",
        "  CORRECTION METHODS SUMMARY",
        f"  {'�' * (W - 4)}",
        "",
        f"    {'Context':<38}  {'Method':<18}  Detail",
        f"    {'-'*38}  {'-'*18}  {'-'*18}",
        f"    {'Mixed ANOVA within/interaction':<38}  {'GG always':<18}  correction=True",
        f"    {'Between-subjects pairwise  (Test 3)':<38}  {'Bonferroni':<18}  padjust=bonf",
        f"    {'McNemar post-hoc  (Test 5)':<38}  {'Bonferroni':<18}  p � actual_n pairs",
        f"    {'Cochrans Q  (Test 4)':<38}  {'None':<18}  each behavior separate",
        "",
        f"{'-' * W}",
        "  MEASURES ANALYSED",
        f"  {'�' * (W - 4)}",
        "",
        "    Continuous weight measures  (Mixed ANOVA + Two-way ANOVA)",
        "      Total Change    cumulative weight change across the experiment   float  (g)",
        "      Daily Change    mean daily weight change per session             float  (g/day)",
        "",
        "    Binary behavioral observations  (Cochran's Q + McNemar)",
        "      Nesting       nest present (1) or absent (0) per week",
        "      Lethargy      observed (1) or not (0) per week",
        "      Anxious Behaviors   observed (1) or not (0) per week",
        "      CA-spot Digging     observed (1) or not (0) per week",
        "",
        "=" * W,
    ]

    report = "\n".join(lines)
    if save_path is not None:
        try:
            from pathlib import Path as _Path
            _Path(save_path).write_text(report, encoding="utf-8")
            print(f"[OK] Test registry saved -> {save_path}")
        except Exception as _e:
            print(f"[WARNING] Could not save registry: {_e}")
    return report



def _run_0v2_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for the 0% vs 2% nonramp comparison.
    All analyses use Week as the time axis (not Day).
    Reports are auto-saved with timestamps.
    Sex-based analyses removed; OLS assumption diagnostics added.
    """
    from datetime import datetime

    MEASURES = ["Total Change", "Daily Change"]

    combined_temp = combine_cohorts_for_analysis(cohorts)
    available_measures = [m for m in MEASURES if m in combined_temp.columns]
    ca_levels = sorted(combined_temp['CA (%)'].dropna().unique())

    print("\n" + "=" * 80)
    print("0% vs 2% NONRAMP ANALYSIS MENU")
    print("=" * 80)
    print("\nAll analyses use Week as the time axis (Week 1 = first measurement week).")
    print("Reports are automatically saved to the current directory.")
    print(f"\nAvailable measures : {available_measures}")
    print(f"CA% levels present : {[int(c) if c == int(c) else c for c in ca_levels]}")
    print()
    print("  1. OLS assumption diagnostics -- 2-way model: Value ~ C(Week) + C(Cohort) + C(ID)")
    print("  2. 2-way Cohort x Week mixed ANOVA + sphericity/GG + post-hoc (all measures)")
    print("  3. Slope analysis       -- Compare rate of weight change between cohorts")
    print("  4. Weight plots         -- Total/Daily Change by ID and CA%")
    print("  5. Behavioral plots     -- Nesting, lethargy, anxiety prevalence across weeks")
    print("  6. Behavioral stats     -- Cohort x Week analysis of binary behavioral metrics")
    print("  7. BH-FDR 2-way omnibus -- CA% x Week for all measures, BH-FDR corrected across measures")
    print("  8. Statistical registry -- Print/save methods documentation for all tests used")
    print("  9. Distribution + assumption checks -- R-based: normality, homogeneity, sphericity, LMM residuals")
    print(" 10. R: nparLD Cohort x Week -- nonparametric two-way ANOVA (Cohort between, Week within) on weekly means")
    print(" 11. R: ART Cohort x Week  -- behavioral metrics (No Nest, Anxious, Lethargy) via Aligned Ranks Transformation")
    print(" 12. R: nparLD behavioral  -- nparLD F1-LD-F1 for No Nest / Anxious / Lethargy (comparison to ART)")
    print(" 13. Run all (1-12)")
    print()

    user_input = input("Select option (1-13) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all = (user_input == '13')

    # ------------------------------------------------------------------ #
    # Option 1: OLS assumption diagnostics
    # ------------------------------------------------------------------ #
    if user_input == '1' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: OLS assumption diagnostics -- Cohort x Week model")
        print("=" * 80)
        diag_dir = Path(f"0v2_ols_diagnostics_{timestamp}")
        for measure in available_measures:
            try:
                check_ols_assumptions_cross_cohort(
                    cohorts, measure=measure,
                    save_dir=diag_dir, show=True, save_report=True,
                )
            except Exception as e:
                print(f"  [WARNING] OLS diagnostics failed for {measure}: {e}")

    # ------------------------------------------------------------------ #
    # Option 2: 2-way Cohort x Week mixed ANOVA (sex collapsed)
    # ------------------------------------------------------------------ #
    if user_input == '2' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: 2-way CA%  x  Week mixed ANOVA (sex collapsed) -- all measures")
        print("=" * 80)

        all_sections = []
        for i, measure in enumerate(available_measures):
            ca_week_results = None
            try:
                ca_week_results = perform_mixed_anova_ca_x_week(cohorts, measure=measure)
            except Exception as e:
                print(f"  [WARNING] CA%  x  Week ANOVA failed for {measure}: {e}")

            section = generate_ca_x_week_report(
                results=ca_week_results,
                cohort_dfs=cohorts if i == 0 else None,
                include_preamble=(i == 0),
                include_footer=(i == len(available_measures) - 1),
            )
            all_sections.append(section)

        combined_report = "\n\n".join(all_sections)
        rpt_path = Path(f"0v2_CA_x_week_{timestamp}.txt")
        rpt_path.write_text(combined_report, encoding='utf-8')
        print(f"\n[OK] Report saved -> {rpt_path}")

    # ------------------------------------------------------------------ #
    # Option 3: Slope analysis
    # ------------------------------------------------------------------ #
    if user_input == '3' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Slope analysis")
        print("=" * 80)

        for measure in available_measures:
            try:
                perform_complete_slope_analysis(
                    cohorts, measure=measure, time_unit='Week',
                    save_plot=True, save_report=True
                )
            except Exception as e:
                print(f"  [WARNING] Slope analysis failed for {measure}: {e}")

    # ------------------------------------------------------------------ #
    # Option 4: Weight plots (by ID and CA%)
    # ------------------------------------------------------------------ #
    if user_input == '4' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Weight plots (by ID and CA%)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)

            plot_dir = Path(f"0v2_plots_{timestamp}")
            plot_dir.mkdir(exist_ok=True)

            figs = {}
            for fname, fn, arg in [
                ("total_change_by_id",    plot_total_change_by_id,    combined),
                ("daily_change_by_id",    plot_daily_change_by_id,    combined),
                ("total_change_by_ca",    plot_total_change_by_ca,    combined),
                ("daily_change_by_ca",    plot_daily_change_by_ca,    combined),
            ]:
                try:
                    fig = fn(arg, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            # Cohort x Week weekly mean plots (+/- SEM)
            print("\n  Generating Cohort x Week weekly mean plots (+/- SEM)...")
            try:
                combined_wk = combined[combined['Day'] >= 1].copy()
                combined_wk = _add_week_column_across_cohorts(combined_wk)

                for measure in available_measures:
                    fname_wk = "cohort_week_{}.svg".format(
                        measure.lower().replace(' ', '_')
                    )
                    try:
                        fig_wk = plot_weekly_means_by_cohort(
                            combined_wk,
                            measure=measure,
                            cohort_metadata=None,
                            by_sex=False,
                            title="{} by Cohort Across Weeks (0% vs 2%)".format(measure),
                            save_path=plot_dir / fname_wk,
                            show=False,
                        )
                        if fig_wk:
                            figs[fname_wk] = fig_wk
                    except Exception as e:
                        print(f"  [WARNING] Cohort x Week plot for {measure} failed: {e}")
            except Exception as e:
                print(f"  [WARNING] Cohort x Week plots failed: {e}")

            print(f"\n[OK] {len(figs)} plots saved -> {plot_dir}")

            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 5: Behavioral metric plots
    # ------------------------------------------------------------------ #
    if user_input == '5' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Behavioral plots (Nesting, Lethargy, Anxiety by week)")
            print("=" * 80)

            plot_dir = Path(f"0v2_plots_{timestamp}")
            plot_dir.mkdir(exist_ok=True)

            try:
                fig = plot_behavioral_metrics_by_cohort(
                    cohorts,
                    title="Behavioral Metrics: 0% vs 2% CA \u2014 Across Weeks",
                    save_path=plot_dir / "behavioral_metrics_by_cohort.svg",
                    show=False,
                )
                if fig:
                    print(f"\n[OK] Behavioral plot saved -> {plot_dir / 'behavioral_metrics_by_cohort.svg'}")
            except Exception as e:
                print(f"  [WARNING] Behavioral plot failed: {e}")

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 6: Behavioral stats (Cohort x Week)
    # ------------------------------------------------------------------ #
    if user_input == '6' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Behavioral statistics (Cohort x Week for binary metrics)")
        print("=" * 80)

        beh_results = {}
        try:
            beh_results = perform_behavioral_mixed_analysis(cohorts)
        except Exception as e:
            print(f"  [WARNING] Behavioral analysis failed: {e}")

        try:
            rpt_path = Path(f"0v2_behavioral_stats_{timestamp}.txt")
            generate_behavioral_report(
                beh_results,
                cohort_dfs=cohorts,
                save_path=rpt_path,
            )
        except Exception as e:
            print(f"  [WARNING] Behavioral report generation failed: {e}")

        if beh_results and HAS_MATPLOTLIB:
            print("\nGenerating interaction plots for significant behavioral effects...")
            beh_plot_dir = Path(f"0v2_behavioral_plots_{timestamp}")
            try:
                beh_figs = plot_behavioral_interaction_effects(
                    beh_results,
                    save_dir=beh_plot_dir,
                    show=False,
                )
                if beh_figs:
                    print(f"[OK] {len(beh_figs)} interaction plot(s) saved -> {beh_plot_dir}")
                    show_now = input("\nDisplay interaction plots now? (y/n): ").strip().lower()
                    if show_now == 'y':
                        plt.show()
                    else:
                        plt.close('all')
            except Exception as e:
                print(f"  [WARNING] Interaction plot generation failed: {e}")

    # ------------------------------------------------------------------ #
    # Option 7: BH-FDR 2-way omnibus (CA% � Week, all measures)
    # ------------------------------------------------------------------ #
    if user_input == '7' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: BH-FDR 2-Way Omnibus � CA% � Week (all measures)")
        print("=" * 80)

        omnibus2_res = None
        try:
            omnibus2_res = perform_omnibus_weight_anova_2way(cohorts)
            report_2way = generate_omnibus_weight_report_2way(
                omnibus2_res, cohort_dfs=cohorts
            )
            rpt_path = Path(f"0v2_omnibus_2way_fdr_{timestamp}.txt")
            rpt_path.write_text(report_2way, encoding='utf-8')
            print(f"\n[OK] BH-FDR 2-way omnibus report saved -> {rpt_path}")
            print(report_2way)
        except Exception as e:
            print(f"  [WARNING] BH-FDR 2-way omnibus failed: {e}")
            import traceback; traceback.print_exc()

        # Plot significant Week � CA% interactions (separate try)
        if omnibus2_res is not None and HAS_MATPLOTLIB:
            try:
                measures_res = omnibus2_res.get('_measures', available_measures)
                int_plot_dir = Path(f"0v2_omnibus_2way_interaction_plots_{timestamp}")
                n_plots = 0
                for _m in measures_res:
                    _r = omnibus2_res.get(_m, {})
                    if 'error' in _r or 'analysis_df' not in _r:
                        continue
                    int_p = _r.get('int_p', 1.0)
                    if np.isfinite(int_p) and int_p < 0.05:
                        int_plot_dir.mkdir(exist_ok=True)
                        adf = _r['analysis_df']
                        ca_levels_plot = sorted(adf['CA (%)'].unique())
                        weeks_plot = sorted(adf['Week'].unique())
                        palette = ['#1f77b4', '#f79520', '#2da048', '#9467bd']
                        fig, ax = plt.subplots()
                        for idx, ca_val in enumerate(ca_levels_plot):
                            grp = adf[adf['CA (%)'] == ca_val]
                            stats_df = grp.groupby('Week')[_m].agg(['mean', 'sem']).reset_index()
                            color = palette[idx % len(palette)]
                            lbl = f"{ca_val:.0f}% CA"
                            ax.errorbar(stats_df['Week'], stats_df['mean'],
                                        yerr=stats_df['sem'],
                                        label=lbl, marker='o',
                                        linewidth=0.9, capsize=5, color=color,
                                        markerfacecolor=color, markeredgecolor=color)
                        ax.set_title(f'{_m}: Week � CA% Interaction', weight='bold')
                        ax.set_xlabel('Week', weight='bold')
                        ax.set_ylabel(f'{_m} (mean � SEM)', weight='bold')
                        ax.set_xticks(weeks_plot)
                        ax.set_xticklabels([str(int(w)) for w in weeks_plot])
                        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
                        ax.legend(title='CA%', loc='best')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        fig.tight_layout()
                        save_path = int_plot_dir / f"2way_{_m.replace(' ', '_')}_week_x_ca.svg"
                        fig.savefig(save_path, format='svg', dpi=200)
                        plt.close(fig)
                        print(f"[OK] Saved interaction plot -> {save_path}")
                        n_plots += 1
                if n_plots:
                    print(f"[OK] {n_plots} interaction plot(s) saved -> {int_plot_dir}")
                else:
                    print("  (No significant Week � CA% interactions to plot)")
            except Exception as e:
                print(f"  [WARNING] Interaction plot failed: {e}")
                import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 8: Statistical test registry
    # ------------------------------------------------------------------ #
    if user_input == '8' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Statistical test registry (methods documentation)")
        print("=" * 80)
        try:
            registry_path = Path(f"0v2_test_registry_{timestamp}.txt")
            registry_report = generate_test_registry_report(save_path=registry_path)
            print(registry_report)
        except Exception as e:
            print(f"  [WARNING] Registry generation failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 9: Distribution + assumption checks
    # ------------------------------------------------------------------ #
    if user_input == '9' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Distribution diagnostics + mixed-model assumption checks (R)")
        print("=" * 80)
        try:
            diag_path = Path(f"0v2_weight_dist_diag_{timestamp}.txt")
            diagnose_weight_distributions(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_path=diag_path,
            )
        except Exception as e:
            print(f"  [WARNING] Distribution diagnostics failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 10: R nparLD -- Cohort x Week nonparametric two-way ANOVA
    # ------------------------------------------------------------------ #
    if user_input == '10' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Cohort x Week nonparametric two-way ANOVA")
        print("=" * 80)
        try:
            run_nparld_cohort_week_r(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_report=True,
                prefix="0v2",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 11: R ART -- Cohort x Week ART ANOVA for behavioral metrics
    # ------------------------------------------------------------------ #
    if user_input == '11' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R ART -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_art_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="0v2",
            )
        except Exception as e:
            print(f"  [WARNING] ART analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 12: R nparLD -- Behavioral metrics (comparison to ART)
    # ------------------------------------------------------------------ #
    if user_input == '12' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_nparld_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="0v2",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD behavior analysis failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("0% vs 2% analysis complete.")
    print("=" * 80)


def _run_0vramp_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for the 0% nonramp vs ramp comparison.
    All analyses use Week as the time axis.
    Sex-based analyses removed; OLS assumption diagnostics added.
    """
    from datetime import datetime

    MEASURES = ["Total Change", "Daily Change"]

    combined_temp = combine_cohorts_for_analysis(cohorts)
    available_measures = [m for m in MEASURES if m in combined_temp.columns]

    # Identify which label is ramp vs 0% nonramp
    ramp_label = next((lbl for lbl in cohorts if 'ramp' in lbl.lower()), None)
    zero_label = next(
        (lbl for lbl in cohorts if '0%' in lbl.lower() and 'ramp' not in lbl.lower()),
        None,
    )

    # Detect number of experiment weeks from data
    try:
        _tmp = clean_cohort(combined_temp.copy())
        if 'Day' not in _tmp.columns:
            _tmp = add_day_column_across_cohorts(_tmp)
        _tmp = _add_week_column_across_cohorts(_tmp)
        n_weeks = int(_tmp['Week'].dropna().max()) if 'Week' in _tmp.columns else 5
    except Exception:
        n_weeks = 5

    # Build cohort_metadata with CA schedules for x-axis tick annotation
    # Ramp schedule: Week 1 = 0% CA, Week 2 = 1% CA, Week 3 = 2% CA, ...
    ramp_schedule = {w: w - 1 for w in range(1, n_weeks + 1)}
    zero_schedule  = {w: 0     for w in range(1, n_weeks + 1)}
    cohort_metadata: Dict[str, Dict] = {}
    if zero_label:
        cohort_metadata[zero_label] = {"ca_schedule": zero_schedule}
    if ramp_label:
        cohort_metadata[ramp_label] = {"ca_schedule": ramp_schedule}

    print("\n" + "=" * 80)
    print("0% NONRAMP vs RAMP � ANALYSIS MENU")
    print("=" * 80)
    print("\nAll analyses use Week as the time axis (Week 1 = first measurement week).")
    print("Outputs are saved to a timestamped directory.")
    print(f"\nAvailable measures : {available_measures}")
    print(f"Cohorts present    : {list(cohorts.keys())}")
    print(f"Weeks detected     : {n_weeks}")
    if ramp_label and ramp_schedule:
        print(f"Ramp CA schedule   : {ramp_schedule}")
    print()
    print("  1. OLS assumption diagnostics -- 2-way model: Value ~ C(Week) + C(Cohort) + C(ID)")
    print("  2. Weight plots by ID     -- Total/Daily Change per animal across time")
    print("  3. Cohort x Week plots    -- Weekly group means (+/- SEM) per cohort")
    print("  4. Behavioral plots       -- Nesting, Lethargy, Anxiety prevalence across weeks")
    print("  5. Cohort-avg plots       -- Total/Daily Change averaged by cohort (CA%-agnostic)")
    print("  6. Slope analysis         -- Per-animal fitted slopes within cohorts + between-cohort comparison")
    print("  7. Distribution + assumption checks -- R-based: normality, homogeneity, sphericity, LMM residuals")
    print("  8. R: nparLD Cohort x Week -- nonparametric two-way ANOVA (Cohort between, Week within) on weekly means")
    print("  9. R: ART Cohort x Week  -- behavioral metrics (No Nest, Anxious, Lethargy) via Aligned Ranks Transformation")
    print(" 10. R: nparLD behavioral  -- nparLD F1-LD-F1 for No Nest / Anxious / Lethargy (comparison to ART)")
    print(" 11. Run all (1-10)")
    print()

    user_input = input("Select option (1-11) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all = (user_input == '11')

    plot_dir = Path(f"0vramp_plots_{timestamp}")

    # ------------------------------------------------------------------ #
    # Option 1: OLS assumption diagnostics
    # ------------------------------------------------------------------ #
    if user_input == '1' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: OLS assumption diagnostics -- Cohort x Week model")
        print("=" * 80)
        diag_dir = Path(f"0vramp_ols_diagnostics_{timestamp}")
        for measure in available_measures:
            try:
                check_ols_assumptions_cross_cohort(
                    cohorts, measure=measure,
                    save_dir=diag_dir, show=True, save_report=True,
                )
            except Exception as e:
                print(f"  [WARNING] OLS diagnostics failed for {measure}: {e}")

    # ------------------------------------------------------------------ #
    # Option 2: Weight plots by ID
    # ------------------------------------------------------------------ #
    if user_input == '2' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Weight plots by ID (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_id", plot_total_change_by_id),
                ("daily_change_by_id", plot_daily_change_by_id),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 3: Cohort x Week interaction plots
    # ------------------------------------------------------------------ #
    if user_input == '3' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort x Week weekly mean plots (+/- SEM)")
            print("=" * 80)

            try:
                combined_wk = combine_cohorts_for_analysis(cohorts)
                combined_wk = clean_cohort(combined_wk)
                if 'Day' not in combined_wk.columns:
                    combined_wk = add_day_column_across_cohorts(combined_wk, drop_ramp_baseline=False)
                combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
                combined_wk = _add_week_column_across_cohorts(combined_wk)
            except Exception as e:
                print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
                combined_wk = None

            if combined_wk is not None:
                plot_dir.mkdir(exist_ok=True)
                figs = {}
                for measure in available_measures:
                    fname = "cohort_week_{}.svg".format(
                        measure.lower().replace(' ', '_')
                    )
                    try:
                        fig = plot_weekly_means_by_cohort(
                            combined_wk,
                            measure=measure,
                            cohort_metadata=None,
                            by_sex=False,
                            title="{} by Cohort Across Weeks (0% vs Ramp)".format(measure),
                            save_path=plot_dir / fname,
                            show=False,
                        )
                        if fig:
                            figs[fname] = fig
                    except Exception as e:
                        print(f"  [WARNING] Cohort x Week plot for {measure} failed: {e}")

                print(f"\n[OK] {len(figs)} Cohort x Week plot(s) saved -> {plot_dir}")
                show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
                if show_now == 'y':
                    plt.show()
                else:
                    plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 4: Behavioral plots
    # ------------------------------------------------------------------ #
    if user_input == '4' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Behavioral plots (Nesting, Lethargy, Anxiety by week)")
            print("=" * 80)

            plot_dir.mkdir(exist_ok=True)
            try:
                fig = plot_behavioral_metrics_by_cohort(
                    cohorts,
                    title="Behavioral Metrics: 0% Nonramp vs Ramp \u2014 Across Weeks",
                    save_path=plot_dir / "behavioral_metrics_by_cohort.svg",
                    show=False,
                )
                if fig:
                    print(
                        f"\n[OK] Behavioral plot saved -> "
                        f"{plot_dir / 'behavioral_metrics_by_cohort.svg'}"
                    )
            except Exception as e:
                print(f"  [WARNING] Behavioral plot failed: {e}")

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 5: Cohort-averaged weight plots (CA%-agnostic)
    # ------------------------------------------------------------------ #
    if user_input == '5' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort-averaged weight plots (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_cohort", plot_total_change_by_cohort),
                ("daily_change_by_cohort", plot_daily_change_by_cohort),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 6: Slope analysis (per-animal OLS within cohorts + between-cohort)
    # ------------------------------------------------------------------ #
    if user_input == '6' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Slope analysis (per-animal fitted slopes, within- and between-cohort)")
        print("=" * 80)
        print("  Fits OLS regression (Total Change ~ Week) per animal using weekly means.")
        print("  Within each cohort: variability of slopes + Levene's test.")
        print("  Between cohorts   : Welch's t-test, Mann-Whitney U, Cohen's d.")

        try:
            combined_wk = combine_cohorts_for_analysis(cohorts)
            combined_wk = clean_cohort(combined_wk)
            if 'Day' not in combined_wk.columns:
                combined_wk = add_day_column_across_cohorts(combined_wk)
            combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
            combined_wk = _add_week_column_across_cohorts(combined_wk)
        except Exception as e:
            print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
            import traceback; traceback.print_exc()
            combined_wk = None

        if combined_wk is not None and 'Total Change' in combined_wk.columns:
            plot_dir.mkdir(exist_ok=True)
            try:
                perform_complete_slope_analysis(
                    cohorts,
                    measure='Total Change',
                    time_unit='Week',
                    save_plot=True,
                    save_report=True,
                    output_dir=plot_dir,
                    combined_df=combined_wk,
                )
            except Exception as e:
                print(f"  [WARNING] Slope analysis failed: {e}")
                import traceback; traceback.print_exc()

            print(f"\n[OK] Slope analysis outputs saved -> {plot_dir}")

    # ------------------------------------------------------------------ #
    # Option 7: Distribution + assumption checks
    # ------------------------------------------------------------------ #
    if user_input == '7' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Distribution diagnostics + mixed-model assumption checks (R)")
        print("=" * 80)
        try:
            diag_path = Path(f"0vramp_weight_dist_diag_{timestamp}.txt")
            diagnose_weight_distributions(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_path=diag_path,
            )
        except Exception as e:
            print(f"  [WARNING] Distribution diagnostics failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 8: R nparLD -- Cohort x Week nonparametric two-way ANOVA
    # ------------------------------------------------------------------ #
    if user_input == '8' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Cohort x Week nonparametric two-way ANOVA")
        print("=" * 80)
        try:
            run_nparld_cohort_week_r(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_report=True,
                prefix="0vramp",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 9: R ART -- Cohort x Week ART ANOVA for behavioral metrics
    # ------------------------------------------------------------------ #
    if user_input == '9' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R ART -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_art_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="0vramp",
            )
        except Exception as e:
            print(f"  [WARNING] ART analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 10: R nparLD -- Behavioral metrics (comparison to ART)
    # ------------------------------------------------------------------ #
    if user_input == '10' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_nparld_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="0vramp",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD behavior analysis failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("0% vs Ramp analysis complete.")
    print("=" * 80)


def _run_2vramp_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for the 2% nonramp vs ramp comparison.
    Sex-based analyses removed; OLS assumption diagnostics added.
        Ramp and nonramp cohorts are aligned directly on Day 1-35.
    """
    from datetime import datetime

    MEASURES = ["Total Change", "Daily Change"]

    combined_temp = combine_cohorts_for_analysis(cohorts)
    available_measures = [m for m in MEASURES if m in combined_temp.columns]

    # Identify which label is ramp vs 2% nonramp
    ramp_label = next((lbl for lbl in cohorts if 'ramp' in lbl.lower()), None)
    two_label = next(
        (lbl for lbl in cohorts if '2%' in lbl.lower() and 'ramp' not in lbl.lower()),
        None,
    )

    # Detect number of experiment weeks from data
    try:
        _tmp = clean_cohort(combined_temp.copy())
        if 'Day' not in _tmp.columns:
            _tmp = add_day_column_across_cohorts(_tmp)
        _tmp = _add_week_column_across_cohorts(_tmp)
        n_weeks = int(_tmp['Week'].dropna().max()) if 'Week' in _tmp.columns else 5
    except Exception:
        n_weeks = 5

    # Build cohort_metadata with CA schedules for x-axis tick annotation.
    # Ramp: Week 1 = 0%, Week 2 = 1%, Week 3 = 2%, ...
    # 2% nonramp: constant 2% throughout.
    ramp_schedule = {w: w - 1 for w in range(1, n_weeks + 1)}
    two_schedule  = {w: 2     for w in range(1, n_weeks + 1)}
    cohort_metadata: Dict[str, Dict] = {}
    if two_label:
        cohort_metadata[two_label] = {"ca_schedule": two_schedule}
    if ramp_label:
        cohort_metadata[ramp_label] = {"ca_schedule": ramp_schedule}

    print("\n" + "=" * 80)
    print("2% NONRAMP vs RAMP \u2014 ANALYSIS MENU")
    print("=" * 80)
    print("\nAll analyses use Week as the time axis (Week 1 = first measurement week).")
    print("Outputs are saved to a timestamped directory.")
    print(f"\nAvailable measures : {available_measures}")
    print(f"Cohorts present    : {list(cohorts.keys())}")
    print(f"Weeks detected     : {n_weeks}")
    if ramp_label and ramp_schedule:
        print(f"Ramp CA schedule   : {ramp_schedule}")
    print()
    print("  1. OLS assumption diagnostics -- 2-way model: Value ~ C(Week) + C(Cohort) + C(ID)")
    print("  2. Weight plots by ID     -- Total/Daily Change per animal across time")
    print("  3. Cohort x Week plots    -- Weekly group means (+/- SEM) per cohort")
    print("  4. Behavioral plots       -- Nesting, Lethargy, Anxiety prevalence across weeks")
    print("  5. Cohort-avg plots       -- Total/Daily Change averaged by cohort (CA%-agnostic)")
    print("  6. Slope analysis         -- Per-animal fitted slopes within cohorts + between-cohort comparison")
    print("  7. Week 3 weight bar      -- Bar plot of average Total Change at Week 3 per cohort (mean \u00b1 SEM + individual points)")
    print("  8. Week 3 behavioral bars -- Bar plots of % observations for No Nest / Anxious / Lethargy at Week 3")
    print("  9. Distribution + assumption checks -- R-based: normality, homogeneity, sphericity, LMM residuals")
    print(" 10. R: nparLD Cohort x Week -- nonparametric two-way ANOVA (Cohort between, Week within) on weekly means")
    print(" 11. R: ART Cohort x Week  -- behavioral metrics (No Nest, Anxious, Lethargy) via Aligned Ranks Transformation")
    print(" 12. R: nparLD behavioral  -- nparLD F1-LD-F1 for No Nest / Anxious / Lethargy (comparison to ART)")
    print(" 13. Run all (1-12)")
    print()

    user_input = input("Select option (1-13) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all = (user_input == '13')

    plot_dir = Path(f"2vramp_plots_{timestamp}")

    # ------------------------------------------------------------------ #
    # Option 1: OLS assumption diagnostics
    # ------------------------------------------------------------------ #
    if user_input == '1' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: OLS assumption diagnostics -- Cohort x Week model")
        print("=" * 80)
        diag_dir = Path(f"2vramp_ols_diagnostics_{timestamp}")
        for measure in available_measures:
            try:
                check_ols_assumptions_cross_cohort(
                    cohorts, measure=measure,
                    save_dir=diag_dir, show=True, save_report=True,
                )
            except Exception as e:
                print(f"  [WARNING] OLS diagnostics failed for {measure}: {e}")

    # ------------------------------------------------------------------ #
    # Option 2: Weight plots by ID
    # ------------------------------------------------------------------ #
    if user_input == '2' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Weight plots by ID (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_id", plot_total_change_by_id),
                ("daily_change_by_id", plot_daily_change_by_id),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 3: Cohort x Week interaction plots
    # ------------------------------------------------------------------ #
    if user_input == '3' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort x Week weekly mean plots (+/- SEM)")
            print("=" * 80)

            try:
                combined_wk = combine_cohorts_for_analysis(cohorts)
                combined_wk = clean_cohort(combined_wk)
                if 'Day' not in combined_wk.columns:
                    combined_wk = add_day_column_across_cohorts(combined_wk, drop_ramp_baseline=False)
                combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
                combined_wk = _add_week_column_across_cohorts(combined_wk)
            except Exception as e:
                print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
                combined_wk = None

            if combined_wk is not None:
                plot_dir.mkdir(exist_ok=True)
                figs = {}
                for measure in available_measures:
                    fname = "cohort_week_{}.svg".format(
                        measure.lower().replace(' ', '_')
                    )
                    try:
                        fig = plot_weekly_means_by_cohort(
                            combined_wk,
                            measure=measure,
                            cohort_metadata=None,
                            by_sex=False,
                            title="{} by Cohort Across Weeks (2% vs Ramp)".format(measure),
                            save_path=plot_dir / fname,
                            show=False,
                        )
                        if fig:
                            figs[fname] = fig
                    except Exception as e:
                        print(f"  [WARNING] Cohort x Week plot for {measure} failed: {e}")

                print(f"\n[OK] {len(figs)} Cohort x Week plot(s) saved -> {plot_dir}")
                show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
                if show_now == 'y':
                    plt.show()
                else:
                    plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 4: Behavioral plots
    # ------------------------------------------------------------------ #
    if user_input == '4' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Behavioral plots (Nesting, Lethargy, Anxiety by week)")
            print("=" * 80)

            plot_dir.mkdir(exist_ok=True)
            try:
                fig = plot_behavioral_metrics_by_cohort(
                    cohorts,
                    title="Behavioral Metrics: 2% Nonramp vs Ramp \u2014 Across Weeks",
                    save_path=plot_dir / "behavioral_metrics_by_cohort.svg",
                    show=False,
                )
                if fig:
                    print(
                        f"\n[OK] Behavioral plot saved -> "
                        f"{plot_dir / 'behavioral_metrics_by_cohort.svg'}"
                    )
            except Exception as e:
                print(f"  [WARNING] Behavioral plot failed: {e}")

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 5: Cohort-averaged weight plots (CA%-agnostic)
    # ------------------------------------------------------------------ #
    if user_input == '5' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort-averaged weight plots (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_cohort", plot_total_change_by_cohort),
                ("daily_change_by_cohort", plot_daily_change_by_cohort),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 6: Slope analysis (per-animal OLS within cohorts + between-cohort)
    # ------------------------------------------------------------------ #
    if user_input == '6' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Slope analysis (per-animal fitted slopes, within- and between-cohort)")
        print("=" * 80)
        print("  Fits OLS regression (Total Change ~ Week) per animal using weekly means.")
        print("  Within each cohort: variability of slopes + Levene's test.")
        print("  Between cohorts   : Welch's t-test, Mann-Whitney U, Cohen's d.")

        try:
            combined_wk = combine_cohorts_for_analysis(cohorts)
            combined_wk = clean_cohort(combined_wk)
            if 'Day' not in combined_wk.columns:
                combined_wk = add_day_column_across_cohorts(combined_wk)
            combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
            combined_wk = _add_week_column_across_cohorts(combined_wk)
        except Exception as e:
            print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
            import traceback; traceback.print_exc()
            combined_wk = None

        if combined_wk is not None and 'Total Change' in combined_wk.columns:
            plot_dir.mkdir(exist_ok=True)
            try:
                perform_complete_slope_analysis(
                    cohorts,
                    measure='Total Change',
                    time_unit='Week',
                    save_plot=True,
                    save_report=True,
                    output_dir=plot_dir,
                    combined_df=combined_wk,
                )
            except Exception as e:
                print(f"  [WARNING] Slope analysis failed: {e}")
                import traceback; traceback.print_exc()

            print(f"\n[OK] Slope analysis outputs saved -> {plot_dir}")

    # ------------------------------------------------------------------ #
    # Option 7: Week 3 Total Change bar plot with individual points + SEM
    # ------------------------------------------------------------------ #
    if user_input == '7' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Week 3 Total Change bar plot (mean \u00b1 SEM + individual points)")
            print("=" * 80)
            try:
                import numpy as _np

                combined_wk3 = combine_cohorts_for_analysis(cohorts)
                combined_wk3 = clean_cohort(combined_wk3)
                if 'Day' not in combined_wk3.columns:
                    combined_wk3 = add_day_column_across_cohorts(combined_wk3, drop_ramp_baseline=False)
                combined_wk3 = combined_wk3[combined_wk3['Day'] >= 1].copy()
                combined_wk3 = _add_week_column_across_cohorts(combined_wk3)

                WEEK3 = 3  # 1-indexed
                week3_df = combined_wk3[combined_wk3['Week'] == WEEK3]

                if 'Total Change' not in week3_df.columns:
                    print("  [WARNING] 'Total Change' column not found -- skipping.")
                elif week3_df.empty:
                    print("  [WARNING] No data found for Week 3 -- skipping.")
                else:
                    cohort_labels_w3 = sorted(week3_df['Cohort'].dropna().unique()) if 'Cohort' in week3_df.columns else list(cohorts.keys())

                    _BAR_COLORS = [
                        {'face': '#1f77b4', 'edge': '#0d3d5c'},
                        {'face': '#f79520', 'edge': '#8a5200'},
                        {'face': '#2da048', 'edge': '#155224'},
                        {'face': '#9467bd', 'edge': '#4a3560'},
                    ]

                    # Compute per-animal weekly means first, then take group average
                    animal_means = (
                        week3_df.groupby(['ID', 'Cohort'])['Total Change'].mean().reset_index()
                        if 'Cohort' in week3_df.columns
                        else week3_df.groupby('ID')['Total Change'].mean().reset_index()
                    )

                    fig_w3, ax_w3 = plt.subplots()
                    x_positions = _np.arange(len(cohort_labels_w3))
                    bar_width = 0.5
                    rng_w3 = _np.random.default_rng(42)

                    for i, cohort_lbl in enumerate(cohort_labels_w3):
                        if 'Cohort' in animal_means.columns:
                            grp = animal_means[animal_means['Cohort'] == cohort_lbl]
                        else:
                            grp = animal_means
                        vals = grp['Total Change'].dropna().values

                        mean_val = _np.mean(vals) if len(vals) > 0 else 0.0
                        sem_val  = (_np.std(vals, ddof=1) / _np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        c = _BAR_COLORS[i % len(_BAR_COLORS)]

                        ax_w3.bar(
                            x_positions[i], mean_val,
                            width=bar_width,
                            color=c['face'], edgecolor=c['edge'], linewidth=0.9,
                            zorder=2, label=cohort_lbl,
                        )
                        ax_w3.errorbar(
                            x_positions[i], mean_val,
                            yerr=sem_val,
                            fmt='none', color='black',
                            capsize=6, capthick=0.8, linewidth=1.0,
                            zorder=3,
                        )
                        jitter = rng_w3.uniform(-0.12, 0.12, size=len(vals))
                        ax_w3.scatter(
                            x_positions[i] + jitter, vals,
                            color='black', s=30, alpha=0.7,
                            zorder=4, linewidths=0,
                        )

                    ax_w3.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)
                    ax_w3.set_xticks(x_positions)
                    ax_w3.set_xticklabels(cohort_labels_w3)
                    ax_w3.set_ylabel('Total Weight Change (g) (mean \u00b1 SEM)', weight='bold')
                    ax_w3.set_title('Total Weight Change at Week 3 by Cohort', weight='bold')
                    ax_w3.spines['top'].set_visible(False)
                    ax_w3.spines['right'].set_visible(False)
                    ax_w3.tick_params(direction='in', which='both', length=5)
                    fig_w3.tight_layout()

                    plot_dir.mkdir(exist_ok=True)
                    save_path_w3 = plot_dir / "total_change_week3_bar.svg"
                    fig_w3.savefig(save_path_w3, format='svg', dpi=200)
                    plt.close(fig_w3)
                    print(f"[OK] Saved -> {save_path_w3}")

            except Exception as e:
                print(f"  [WARNING] Week 3 weight bar plot failed: {e}")
                import traceback; traceback.print_exc()

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 8: Week 3 behavioral bar plots (No Nest / Anxious / Lethargy)
    # ------------------------------------------------------------------ #
    if user_input == '8' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Week 3 behavioral bar plots (% observations per cohort)")
            print("=" * 80)
            try:
                import numpy as _np

                BEHAVIORS_BAR = [
                    ('Nest Made?',         False, 'No Nest'),
                    ('Anxious Behaviors?', True,  'Anxious'),
                    ('Lethargy?',          True,  'Lethargy'),
                ]

                def _to_bool_bar(series):
                    _T = {'yes', 'true', '1'}
                    _F = {'no', 'false', '0'}
                    def _cv(v):
                        if isinstance(v, bool): return v
                        if isinstance(v, str):
                            ls = v.strip().lower()
                            if ls in _T: return True
                            if ls in _F: return False
                        return None
                    return series.map(_cv)

                WEEK3_B = 3  # 1-indexed
                _BAR_COLORS_B = [
                    {'face': '#1f77b4', 'edge': '#0d3d5c'},
                    {'face': '#f79520', 'edge': '#8a5200'},
                    {'face': '#2da048', 'edge': '#155224'},
                    {'face': '#9467bd', 'edge': '#4a3560'},
                ]

                # Build per-cohort per-animal % for each behavioral metric at Week 3
                cohort_labels_b = list(cohorts.keys())
                # animal_pcts[col][cohort_label] = list of per-animal %
                animal_pcts_all = {col: {lbl: [] for lbl in cohort_labels_b} for col, _, _ in BEHAVIORS_BAR}

                for lbl, df in cohorts.items():
                    cdf = clean_cohort(df.copy())
                    if 'Date' in cdf.columns:
                        cdf['Date'] = pd.to_datetime(cdf['Date'], errors='coerce')
                    cdf = cdf.sort_values(['ID', 'Date']).reset_index(drop=True)
                    first_dates = cdf.groupby('ID')['Date'].transform('min')
                    _day_offset = 1 if 'ramp' in lbl.lower() else 0
                    cdf['_Day'] = (cdf['Date'] - first_dates).dt.days + _day_offset
                    cdf = cdf[cdf['_Day'] >= 1].copy()
                    cdf['_Week'] = (cdf['_Day'] - 1) // 7 + 1
                    week3_cdf = cdf[cdf['_Week'] == WEEK3_B]

                    for col, aberrant_val, _ in BEHAVIORS_BAR:
                        if col not in week3_cdf.columns:
                            continue
                        for _, animal_data in week3_cdf.groupby('ID'):
                            valid = _to_bool_bar(animal_data[col]).dropna()
                            if len(valid) > 0:
                                animal_pcts_all[col][lbl].append(
                                    100.0 * (valid == aberrant_val).sum() / len(valid)
                                )

                fig_b, axes_b = plt.subplots(1, 3, sharey=False)
                fig_b.suptitle('Behavioral Metrics at Week 3 by Cohort', weight='bold', y=1.01)

                rng_b = _np.random.default_rng(42)
                x_positions_b = _np.arange(len(cohort_labels_b))
                bar_width_b = 0.5

                for ax_b, (col, _, panel_title) in zip(axes_b, BEHAVIORS_BAR):
                    for i, lbl in enumerate(cohort_labels_b):
                        vals = _np.array(animal_pcts_all[col][lbl], dtype=float)
                        mean_val = _np.mean(vals) if len(vals) > 0 else 0.0
                        sem_val  = (_np.std(vals, ddof=1) / _np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        c = _BAR_COLORS_B[i % len(_BAR_COLORS_B)]

                        ax_b.bar(
                            x_positions_b[i], mean_val,
                            width=bar_width_b,
                            color=c['face'], edgecolor=c['edge'], linewidth=0.9,
                            zorder=2,
                        )
                        ax_b.errorbar(
                            x_positions_b[i], mean_val,
                            yerr=sem_val,
                            fmt='none', color='black',
                            capsize=6, capthick=0.8, linewidth=1.0,
                            zorder=3,
                        )
                        jitter = rng_b.uniform(-0.12, 0.12, size=len(vals))
                        ax_b.scatter(
                            x_positions_b[i] + jitter, vals,
                            color='black', s=30, alpha=0.7,
                            zorder=4, linewidths=0,
                        )

                    ax_b.set_xticks(x_positions_b)
                    ax_b.set_xticklabels(cohort_labels_b, rotation=15, ha='right')
                    ax_b.set_ylim(0, 110)
                    ax_b.set_ylabel('% of Observations (mean \u00b1 SEM)', weight='bold')
                    ax_b.set_title(panel_title, weight='bold')
                    ax_b.spines['top'].set_visible(False)
                    ax_b.spines['right'].set_visible(False)
                    ax_b.tick_params(direction='in', which='both', length=5)

                fig_b.tight_layout()
                plot_dir.mkdir(exist_ok=True)
                save_path_b = plot_dir / "behavioral_week3_bar.svg"
                fig_b.savefig(save_path_b, format='svg', dpi=200)
                plt.close(fig_b)
                print(f"[OK] Saved -> {save_path_b}")

            except Exception as e:
                print(f"  [WARNING] Week 3 behavioral bar plot failed: {e}")
                import traceback; traceback.print_exc()

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 9: Distribution + assumption checks
    # ------------------------------------------------------------------ #
    if user_input == '9' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Distribution diagnostics + mixed-model assumption checks (R)")
        print("=" * 80)
        try:
            diag_path = Path(f"2vramp_weight_dist_diag_{timestamp}.txt")
            diagnose_weight_distributions(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_path=diag_path,
            )
        except Exception as e:
            print(f"  [WARNING] Distribution diagnostics failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 10: R nparLD -- Cohort x Week nonparametric two-way ANOVA
    # ------------------------------------------------------------------ #
    if user_input == '10' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Cohort x Week nonparametric two-way ANOVA")
        print("=" * 80)
        try:
            run_nparld_cohort_week_r(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_report=True,
                prefix="2vramp",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 11: R ART -- Cohort x Week ART ANOVA for behavioral metrics
    # ------------------------------------------------------------------ #
    if user_input == '11' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R ART -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_art_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="2vramp",
            )
        except Exception as e:
            print(f"  [WARNING] ART analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 12: R nparLD -- Behavioral metrics (comparison to ART)
    # ------------------------------------------------------------------ #
    if user_input == '12' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_nparld_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="2vramp",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD behavior analysis failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("2% vs Ramp analysis complete.")
    print("=" * 80)


def _run_all3_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for the 3-cohort comparison:
    0% nonramp vs 2% nonramp vs Ramp.
    All analyses use Week as the time axis.
        All cohorts are aligned directly on Day 1-35.
    """
    from datetime import datetime

    MEASURES = ["Total Change", "Daily Change"]

    combined_temp = combine_cohorts_for_analysis(cohorts)
    available_measures = [m for m in MEASURES if m in combined_temp.columns]

    # Identify labels for each cohort
    ramp_label = next((lbl for lbl in cohorts if 'ramp' in lbl.lower()), None)
    zero_label = next(
        (lbl for lbl in cohorts if '0%' in lbl.lower() and 'ramp' not in lbl.lower()),
        None,
    )
    two_label = next(
        (lbl for lbl in cohorts if '2%' in lbl.lower() and 'ramp' not in lbl.lower()),
        None,
    )

    # Detect number of experiment weeks from data
    try:
        _tmp = clean_cohort(combined_temp.copy())
        if 'Day' not in _tmp.columns:
            _tmp = add_day_column_across_cohorts(_tmp)
        _tmp = _add_week_column_across_cohorts(_tmp)
        n_weeks = int(_tmp['Week'].dropna().max()) if 'Week' in _tmp.columns else 5
    except Exception:
        n_weeks = 5

    # Build cohort_metadata with CA schedules for x-axis tick annotation.
    # Ramp: Week 1 = 0%, Week 2 = 1%, Week 3 = 2%, ...
    ramp_schedule = {w: w - 1 for w in range(1, n_weeks + 1)}
    zero_schedule = {w: 0     for w in range(1, n_weeks + 1)}
    two_schedule  = {w: 2     for w in range(1, n_weeks + 1)}
    cohort_metadata: Dict[str, Dict] = {}
    if zero_label:
        cohort_metadata[zero_label] = {"ca_schedule": zero_schedule}
    if two_label:
        cohort_metadata[two_label] = {"ca_schedule": two_schedule}
    if ramp_label:
        cohort_metadata[ramp_label] = {"ca_schedule": ramp_schedule}

    print("\n" + "=" * 80)
    print("0% NONRAMP vs 2% NONRAMP vs RAMP \u2014 ANALYSIS MENU")
    print("=" * 80)
    print("\nAll analyses use Week as the time axis (Week 1 = first measurement week).")
    print("Outputs are saved to a timestamped directory.")
    print(f"\nAvailable measures : {available_measures}")
    print(f"Cohorts present    : {list(cohorts.keys())}")
    print(f"Weeks detected     : {n_weeks}")
    if ramp_label and ramp_schedule:
        print(f"Ramp CA schedule   : {ramp_schedule}")
    print()
    print("  1. Weight plots by ID     -- Total/Daily Change per animal across time")
    print("  2. Weight plots by Sex    -- Total/Daily Change averaged by sex across time")
    print("  3. Cohort x Week plots    -- Weekly group means (+/- SEM) per cohort")
    print("  4. Behavioral plots       -- Nesting, Lethargy, Anxiety prevalence across weeks")
    print("  5. Cohort-avg plots       -- Total/Daily Change averaged by cohort (CA%-agnostic)")
    print("  6. Slope analysis         -- Per-animal fitted slopes within cohorts + between-cohort comparison")
    print("  7. Distribution + assumption checks -- R-based: normality, homogeneity, sphericity, LMM residuals")
    print("  8. R: nparLD Cohort x Week -- nonparametric two-way ANOVA (Cohort between, Week within) on weekly means")
    print("  9. R: ART Cohort x Week  -- behavioral metrics (No Nest, Anxious, Lethargy) via Aligned Ranks Transformation")
    print(" 10. R: nparLD behavioral  -- nparLD F1-LD-F1 for No Nest / Anxious / Lethargy (comparison to ART)")
    print(" 11. Run all (1-10)")
    print()

    user_input = input("Select option (1-11) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_all = (user_input == '11')

    plot_dir = Path(f"all3_plots_{timestamp}")

    # ------------------------------------------------------------------ #
    # Option 1: Weight plots by ID
    # ------------------------------------------------------------------ #
    if user_input == '1' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Weight plots by ID (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_id", plot_total_change_by_id),
                ("daily_change_by_id", plot_daily_change_by_id),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 2: Weight plots by Sex
    # ------------------------------------------------------------------ #
    if user_input == '2' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Weight plots by Sex (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_sex", plot_total_change_by_sex),
                ("daily_change_by_sex", plot_daily_change_by_sex),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 3: Cohort x Week interaction plots
    # ------------------------------------------------------------------ #
    if user_input == '3' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort x Week weekly mean plots (+/- SEM)")
            print("=" * 80)

            try:
                combined_wk = combine_cohorts_for_analysis(cohorts)
                combined_wk = clean_cohort(combined_wk)
                if 'Day' not in combined_wk.columns:
                    combined_wk = add_day_column_across_cohorts(combined_wk, drop_ramp_baseline=False)
                combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
                combined_wk = _add_week_column_across_cohorts(combined_wk)
            except Exception as e:
                print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
                combined_wk = None

            if combined_wk is not None:
                plot_dir.mkdir(exist_ok=True)
                figs = {}
                for measure in available_measures:
                    fname = "cohort_week_{}_all.svg".format(
                        measure.lower().replace(' ', '_')
                    )
                    try:
                        fig = plot_weekly_means_by_cohort(
                            combined_wk,
                            measure=measure,
                            cohort_metadata=None,
                            by_sex=False,
                            title="{} by Cohort Across Weeks (0% vs 2% vs Ramp)".format(measure),
                            save_path=plot_dir / fname,
                            show=False,
                        )
                        if fig:
                            figs[fname] = fig
                    except Exception as e:
                        print(f"  [WARNING] Cohort x Week plot (all) for {measure} failed: {e}")

                    fname_sex = "cohort_week_{}_by_sex.svg".format(
                        measure.lower().replace(' ', '_')
                    )
                    try:
                        fig_sex = plot_weekly_means_by_cohort(
                            combined_wk,
                            measure=measure,
                            cohort_metadata=None,
                            by_sex=True,
                            title="{} by Cohort and Sex Across Weeks (0% vs 2% vs Ramp)".format(measure),
                            save_path=plot_dir / fname_sex,
                            show=False,
                        )
                        if fig_sex:
                            figs[fname_sex] = fig_sex
                    except Exception as e:
                        print(f"  [WARNING] Cohort x Week plot (by sex) for {measure} failed: {e}")

                print(f"\n[OK] {len(figs)} Cohort x Week plot(s) saved -> {plot_dir}")
                show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
                if show_now == 'y':
                    plt.show()
                else:
                    plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 4: Behavioral plots
    # ------------------------------------------------------------------ #
    if user_input == '4' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Behavioral plots (Nesting, Lethargy, Anxiety by week)")
            print("=" * 80)

            plot_dir.mkdir(exist_ok=True)
            try:
                fig = plot_behavioral_metrics_by_cohort(
                    cohorts,
                    title="Behavioral Metrics: 0% vs 2% Nonramp vs Ramp \u2014 Across Weeks",
                    save_path=plot_dir / "behavioral_metrics_by_cohort.svg",
                    show=False,
                )
                if fig:
                    print(
                        f"\n[OK] Behavioral plot saved -> "
                        f"{plot_dir / 'behavioral_metrics_by_cohort.svg'}"
                    )
            except Exception as e:
                print(f"  [WARNING] Behavioral plot failed: {e}")

            show_now = input("\nDisplay plot now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 5: Cohort-averaged weight plots (CA%-agnostic)
    # ------------------------------------------------------------------ #
    if user_input == '5' or run_all:
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available -- cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Cohort-averaged weight plots (Total Change and Daily Change)")
            print("=" * 80)

            combined = combine_cohorts_for_analysis(cohorts)
            combined = clean_cohort(combined)
            if 'Day' not in combined.columns:
                combined = add_day_column_across_cohorts(combined, drop_ramp_baseline=False)
            combined = combined[combined['Day'] >= 1].copy()

            plot_dir.mkdir(exist_ok=True)
            figs = {}
            for fname, fn in [
                ("total_change_by_cohort", plot_total_change_by_cohort),
                ("daily_change_by_cohort", plot_daily_change_by_cohort),
            ]:
                try:
                    fig = fn(combined, save_path=plot_dir / f"{fname}.svg", show=False)
                    if fig:
                        figs[fname] = fig
                except Exception as e:
                    print(f"  [WARNING] Plot {fname} failed: {e}")

            print(f"\n[OK] {len(figs)} plot(s) saved -> {plot_dir}")
            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    # ------------------------------------------------------------------ #
    # Option 6: Slope analysis (per-animal OLS within cohorts + between-cohort)
    # ------------------------------------------------------------------ #
    if user_input == '6' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Slope analysis (per-animal fitted slopes, within- and between-cohort)")
        print("=" * 80)
        print("  Fits OLS regression (Total Change ~ Week) per animal using weekly means.")
        print("  Within each cohort: variability of slopes + Levene's test.")
        print("  Between cohorts   : pairwise Welch's t-test, Mann-Whitney U, Cohen's d.")

        try:
            combined_wk = combine_cohorts_for_analysis(cohorts)
            combined_wk = clean_cohort(combined_wk)
            if 'Day' not in combined_wk.columns:
                combined_wk = add_day_column_across_cohorts(combined_wk)
            combined_wk = combined_wk[combined_wk['Day'] >= 1].copy()
            combined_wk = _add_week_column_across_cohorts(combined_wk)
        except Exception as e:
            print(f"  [ERROR] Could not build cohort-week dataframe: {e}")
            import traceback; traceback.print_exc()
            combined_wk = None

        if combined_wk is not None and 'Total Change' in combined_wk.columns:
            plot_dir.mkdir(exist_ok=True)
            try:
                perform_complete_slope_analysis(
                    cohorts,
                    measure='Total Change',
                    time_unit='Week',
                    save_plot=True,
                    save_report=True,
                    output_dir=plot_dir,
                    combined_df=combined_wk,
                )
            except Exception as e:
                print(f"  [WARNING] Slope analysis failed: {e}")
                import traceback; traceback.print_exc()

            print(f"\n[OK] Slope analysis outputs saved -> {plot_dir}")

    # ------------------------------------------------------------------ #
    # Option 7: Distribution + assumption checks
    # ------------------------------------------------------------------ #
    if user_input == '7' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: Distribution diagnostics + mixed-model assumption checks (R)")
        print("=" * 80)
        try:
            diag_path = Path(f"all3_weight_dist_diag_{timestamp}.txt")
            diagnose_weight_distributions(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_path=diag_path,
            )
        except Exception as e:
            print(f"  [WARNING] Distribution diagnostics failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 8: R nparLD -- Cohort x Week nonparametric two-way ANOVA
    # ------------------------------------------------------------------ #
    if user_input == '8' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Cohort x Week nonparametric two-way ANOVA")
        print("=" * 80)
        try:
            run_nparld_cohort_week_r(
                cohort_dfs=cohorts,
                measure="Total Change",
                save_report=True,
                prefix="all3",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 9: R ART -- Cohort x Week ART ANOVA for behavioral metrics
    # ------------------------------------------------------------------ #
    if user_input == '9' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R ART -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_art_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="all3",
            )
        except Exception as e:
            print(f"  [WARNING] ART analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Option 10: R nparLD -- Behavioral metrics (comparison to ART)
    # ------------------------------------------------------------------ #
    if user_input == '10' or run_all:
        print("\n" + "=" * 80)
        print("RUNNING: R nparLD -- Behavioral metrics Cohort x Week (No Nest, Anxious, Lethargy)")
        print("=" * 80)
        try:
            run_nparld_behavior_r(
                cohort_dfs=cohorts,
                save_report=True,
                prefix="all3",
            )
        except Exception as e:
            print(f"  [WARNING] nparLD behavior analysis failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("0% vs 2% vs Ramp analysis complete.")
    print("=" * 80)


def _run_rampramp_menu(cohorts: Dict[str, pd.DataFrame]) -> None:
    """
    Interactive analysis menu for the Ramp vs 2-Week Ramp comparison.
    Compares Daily Change at the CA% transition days:
      - Ramp cohort   : Day 8 (0%->1%) and Day 15 (1%->2%)
      - 2-Week Ramp   : Day 4 (0%->1%) and Day  7 (1%->2%)
    Plots: bar (mean +/- SEM) + individual animal points per cohort.
    Stats: Mann-Whitney U, Holm-Bonferroni corrected across 2 comparisons.
    """
    from datetime import datetime
    import numpy as _np
    from scipy import stats as _stats

    _TWOWK_KW = ('2wk', '2_wk', '2week', '2 wk', '2-wk', '2-week', '2 week')

    # Identify which label is the regular ramp and which is the 2wk ramp
    twowk_label = next(
        (lbl for lbl in cohorts if any(kw in lbl.lower() for kw in _TWOWK_KW)),
        None,
    )
    ramp_label = next(
        (lbl for lbl in cohorts if lbl != twowk_label),
        None,
    )
    if twowk_label is None or ramp_label is None:
        labels = list(cohorts.keys())
        ramp_label, twowk_label = labels[0], labels[1]

    # CA% transition days for each cohort (Day 1 = first measurement day)
    TRANSITION_DAYS = {
        ramp_label:  {'0%->1%': 8,  '1%->2%': 15},
        twowk_label: {'0%->1%': 4,  '1%->2%': 7},
    }
    TRANSITIONS = ['0%->1%', '1%->2%']

    RAMP_COLOR  = _COLOR_RAMP   # "#2da048" green
    TWOWK_COLOR = _COLOR_OTHER  # "#7f3f98" purple

    COLOR_MAP = {
        ramp_label:  {'face': RAMP_COLOR,  'edge': '#155224'},
        twowk_label: {'face': TWOWK_COLOR, 'edge': '#3f1f6e'},
    }

    print("\n" + "=" * 80)
    print("RAMP vs 2-WEEK RAMP \u2014 ANALYSIS MENU")
    print("=" * 80)
    print(f"\n  Ramp cohort   : {ramp_label}")
    print(f"  2-Week Ramp   : {twowk_label}")
    print(f"\n  Transition days (Ramp)       : Day 8 (0%\u21921%), Day 15 (1%\u21922%)")
    print(f"  Transition days (2-Wk Ramp)  : Day 4 (0%\u21921%), Day  7 (1%\u21922%)")
    print()
    print("  1. Transition-day Daily Change bar plots + Mann-Whitney U (Holm-Bonferroni)")
    print()

    user_input = input("Select option (1) or 'n' to skip: ").strip()
    if user_input.lower() == 'n':
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path(f"rampramp_plots_{timestamp}")

    if user_input == '1':
        if not HAS_MATPLOTLIB:
            print("\n[WARNING] matplotlib not available \u2014 cannot generate plots")
        else:
            print("\n" + "=" * 80)
            print("GENERATING: Transition-day Daily Change bar plots")
            print("=" * 80)
            try:
                # --- Per-animal values at each transition day per cohort ---
                # per_cohort_data[transition][label] = 1-D array of per-animal means
                per_cohort_data = {t: {} for t in TRANSITIONS}

                for label, df in cohorts.items():
                    cdf = clean_cohort(df.copy())
                    if 'Date' in cdf.columns:
                        cdf['Date'] = pd.to_datetime(cdf['Date'], errors='coerce')
                    if 'Day' not in cdf.columns:
                        cdf = cdf.sort_values(['ID', 'Date']).reset_index(drop=True)
                        first_dates = cdf.groupby('ID')['Date'].transform('min')
                        # Both cohorts in this menu are ramp-type (no Day 0 baseline);
                        # always offset by 1 so the first measurement is Day 1.
                        cdf['Day'] = (cdf['Date'] - first_dates).dt.days + 1
                    cdf = cdf[cdf['Day'] >= 1].copy()

                    if 'Daily Change' not in cdf.columns:
                        print(f"  [WARNING] 'Daily Change' not in {label} \u2014 skipping")
                        for t in TRANSITIONS:
                            per_cohort_data[t][label] = _np.array([])
                        continue

                    for transition, tday in TRANSITION_DAYS[label].items():
                        day_df = cdf[cdf['Day'] == tday]
                        if day_df.empty:
                            print(
                                f"  [WARNING] No data for {label} Day {tday} "
                                f"({transition}) \u2014 skipping"
                            )
                            per_cohort_data[transition][label] = _np.array([])
                            continue
                        animal_vals = day_df.groupby('ID')['Daily Change'].mean().values
                        per_cohort_data[transition][label] = animal_vals

                # --- Helper functions ---
                def _fmt_p(p):
                    if _np.isnan(p):
                        return 'N/A'
                    if p < 0.001:
                        return '< 0.001'
                    return f'{p:.4f}'

                def _sig(p):
                    if _np.isnan(p):
                        return ''
                    if p < 0.001:
                        return '***'
                    if p < 0.01:
                        return '**'
                    if p < 0.05:
                        return '*'
                    return 'ns'

                # --- Pre-compute MWU + Holm-Bonferroni before drawing ---
                stat_results = []
                for transition in TRANSITIONS:
                    ramp_vals  = per_cohort_data[transition].get(ramp_label,  _np.array([]))
                    twowk_vals = per_cohort_data[transition].get(twowk_label, _np.array([]))
                    sr = {
                        'transition':  transition,
                        'ramp_label':  ramp_label,
                        'twowk_label': twowk_label,
                        'ramp_day':    TRANSITION_DAYS[ramp_label][transition],
                        'twowk_day':   TRANSITION_DAYS[twowk_label][transition],
                        'ramp_n':      len(ramp_vals),
                        'twowk_n':     len(twowk_vals),
                        'ramp_mean':   float(_np.mean(ramp_vals))  if len(ramp_vals)  > 0 else float('nan'),
                        'ramp_sem':    float(_np.std(ramp_vals,  ddof=1) / _np.sqrt(len(ramp_vals)))  if len(ramp_vals)  > 1 else float('nan'),
                        'twowk_mean':  float(_np.mean(twowk_vals)) if len(twowk_vals) > 0 else float('nan'),
                        'twowk_sem':   float(_np.std(twowk_vals, ddof=1) / _np.sqrt(len(twowk_vals))) if len(twowk_vals) > 1 else float('nan'),
                        'ramp_vals':   ramp_vals,
                        'twowk_vals':  twowk_vals,
                        'U':           float('nan'),
                        'p_raw':       float('nan'),
                        'p_adj':       float('nan'),
                    }
                    if len(ramp_vals) >= 2 and len(twowk_vals) >= 2:
                        U, p_raw = _stats.mannwhitneyu(ramp_vals, twowk_vals, alternative='two-sided')
                        sr['U']     = float(U)
                        sr['p_raw'] = float(p_raw)
                    stat_results.append(sr)

                # Holm-Bonferroni
                valid = [sr for sr in stat_results if not _np.isnan(sr['p_raw'])]
                k = len(valid)
                if k > 0:
                    valid_sorted = sorted(valid, key=lambda x: x['p_raw'])
                    for rank, sr in enumerate(valid_sorted):
                        sr['p_adj'] = min(sr['p_raw'] * (k - rank), 1.0)
                    for i in range(1, len(valid_sorted)):
                        valid_sorted[i]['p_adj'] = max(
                            valid_sorted[i]['p_adj'],
                            valid_sorted[i - 1]['p_adj'],
                        )

                # --- Bar plot: 2 panels, 2 bars each, with sig brackets ---
                fig, axes = plt.subplots(1, 2, sharey=True)
                fig.suptitle(
                    'Daily Weight Change at CA% Transition Days\n'
                    'Ramp (Day 8, 15) vs 2-Week Ramp (Day 4, 7)',
                )
                rng = _np.random.default_rng(42)
                bar_width = 0.35
                cohort_order = [ramp_label, twowk_label]
                x_pos = _np.array([0.2, 0.8])

                # Pre-compute global y-range across both panels for shared axis
                _global_plotted = [0.0]
                for sr in stat_results:
                    for vals in [sr['ramp_vals'], sr['twowk_vals']]:
                        if len(vals) == 0:
                            continue
                        mean_val = float(_np.mean(vals))
                        sem_val  = float(_np.std(vals, ddof=1) / _np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        _global_plotted.extend(vals.tolist())
                        _global_plotted.append(mean_val + sem_val)
                        _global_plotted.append(mean_val - sem_val)

                _ylim_bot = -12.5
                _ylim_top = 2.5
                _g_span  = _ylim_top - _ylim_bot
                _tick_h  = 0.04 * _g_span
                _bkt_y   = _ylim_top - _tick_h * 3.5

                for ax_idx, (ax, sr) in enumerate(zip(axes, stat_results)):
                    transition = sr['transition']
                    vals_list  = [sr['ramp_vals'], sr['twowk_vals']]

                    for i, (lbl, vals) in enumerate(zip(cohort_order, vals_list)):
                        if len(vals) == 0:
                            continue
                        mean_val = float(_np.mean(vals))
                        sem_val  = float(_np.std(vals, ddof=1) / _np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        c = COLOR_MAP[lbl]
                        ax.bar(x_pos[i], mean_val, width=bar_width,
                               color=c['face'], edgecolor=c['edge'], linewidth=0.9, zorder=2)
                        ax.errorbar(x_pos[i], mean_val, yerr=sem_val,
                                    fmt='none', color='black',
                                    capsize=6, capthick=0.8, linewidth=1.0, zorder=3)
                        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                        ax.scatter(x_pos[i] + jitter, vals,
                                   color=c['face'], s=10, alpha=0.6,
                                   edgecolors='black', linewidths=0.5, zorder=4)

                    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', zorder=1)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([
                        f"Ramp\n(Day {TRANSITION_DAYS[ramp_label][transition]})",
                        f"2-Wk Ramp\n(Day {TRANSITION_DAYS[twowk_label][transition]})",
                    ])
                    # Only the left panel gets a y-axis label
                    if ax_idx == 0:
                        ax.set_ylabel('Daily Change (%) (mean \u00b1 SEM)')
                    ax.set_title(f'Transition {transition}')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(direction='in', which='both', length=5)
                    ax.set_xlim(-0.1, 1.1)

                    # significance bracket (globally positioned, same height both panels)
                    ax.plot([x_pos[0], x_pos[1]], [_bkt_y, _bkt_y],
                            color='black', lw=0.8, zorder=5)
                    ax.plot([x_pos[0], x_pos[0]], [_bkt_y - _tick_h, _bkt_y],
                            color='black', lw=0.8, zorder=5)
                    ax.plot([x_pos[1], x_pos[1]], [_bkt_y - _tick_h, _bkt_y],
                            color='black', lw=0.8, zorder=5)
                    ax.text(0.5 * (x_pos[0] + x_pos[1]), _bkt_y + _tick_h * 0.3,
                            _sig(sr['p_adj']), ha='center', va='bottom', fontsize=8, zorder=6)

                # Apply shared y limits once (sharey=True means this covers both panels)
                axes[0].set_ylim(bottom=_ylim_bot, top=_ylim_top)
                fig.subplots_adjust(wspace=0.05)
                fig.tight_layout()
                plot_dir.mkdir(exist_ok=True)
                bar_svg = plot_dir / "transition_day_daily_change_bar.svg"
                fig.savefig(bar_svg, format='svg', dpi=200)
                plt.close(fig)
                print(f"[OK] Saved bar plot -> {bar_svg}")

                # --- Print and save stats report ---
                print("\n" + "=" * 80)
                print("MANN-WHITNEY U TESTS \u2014 Transition Day Daily Change")
                print("Correction: Holm-Bonferroni (family = 2 comparisons)")
                print("=" * 80)

                W = 74
                report_lines = [
                    "=" * W,
                    "RAMP vs 2-WEEK RAMP \u2014 CA% TRANSITION DAY DAILY CHANGE",
                    f"Cohorts: {ramp_label}  vs  {twowk_label}",
                    "",
                    f"Transition days  \u2014  Ramp    : Day  8 (0%\u21921%),  Day 15 (1%\u21922%)",
                    f"                   2-Wk Ramp: Day  4 (0%\u21921%),  Day  7 (1%\u21922%)",
                    "",
                    "Statistical test  : Mann-Whitney U (two-sided)",
                    "Correction        : Holm-Bonferroni (k=2 comparisons)",
                    "",
                    f"{'Transition':<10}  {'n_Ramp':>6}  {'n_2Wk':>6}  {'U':>8}  {'p_raw':>9}  {'p_adj':>9}  sig",
                    "-" * W,
                ]
                for sr in stat_results:
                    U_str = f"{sr['U']:.1f}" if not _np.isnan(sr['U']) else 'N/A'
                    report_lines.append(
                        f"{sr['transition']:<10}  {sr['ramp_n']:>6}  {sr['twowk_n']:>6}  "
                        f"{U_str:>8}  {_fmt_p(sr['p_raw']):>9}  {_fmt_p(sr['p_adj']):>9}  {_sig(sr['p_adj'])}"
                    )
                    print(
                        f"\n  Transition {sr['transition']}:"
                        f"\n    {ramp_label}: n={sr['ramp_n']}, "
                        f"mean={sr['ramp_mean']:.4f} \u00b1 {sr['ramp_sem']:.4f} SEM"
                        f"\n    {twowk_label}: n={sr['twowk_n']}, "
                        f"mean={sr['twowk_mean']:.4f} \u00b1 {sr['twowk_sem']:.4f} SEM"
                        f"\n    U={U_str}, p_raw={_fmt_p(sr['p_raw'])}, "
                        f"p_adj (Holm)={_fmt_p(sr['p_adj'])}  {_sig(sr['p_adj'])}"
                    )

                report_lines += [
                    "-" * W,
                    "",
                    "Descriptive statistics:",
                    "",
                    f"{'Transition':<10}  {'Cohort':<28}  {'n':>4}  {'Mean (%/day)':>13}  {'SEM':>8}",
                    "-" * W,
                ]
                for sr in stat_results:
                    report_lines.append(
                        f"{sr['transition']:<10}  {sr['ramp_label']:<28}  {sr['ramp_n']:>4}  "
                        f"{sr['ramp_mean']:>13.4f}  {sr['ramp_sem']:>8.4f}"
                    )
                    report_lines.append(
                        f"{'':10}  {sr['twowk_label']:<28}  {sr['twowk_n']:>4}  "
                        f"{sr['twowk_mean']:>13.4f}  {sr['twowk_sem']:>8.4f}"
                    )
                report_lines.append("=" * W)

                report_text = "\n".join(report_lines)
                print("\n" + report_text)
                rpt_path = plot_dir / f"transition_day_mwu_{timestamp}.txt"
                rpt_path.write_text(report_text, encoding='utf-8')
                print(f"\n[OK] Saved report -> {rpt_path}")

            except Exception as e:
                print(f"  [WARNING] Transition-day analysis failed: {e}")
                import traceback; traceback.print_exc()

            show_now = input("\nDisplay plots now? (y/n): ").strip().lower()
            if show_now == 'y':
                plt.show()
            else:
                plt.close('all')

    print("\n" + "=" * 80)
    print("Ramp vs 2-Week Ramp analysis complete.")
    print("=" * 80)


# =============================================================================
# R-BASED nparLD: NONPARAMETRIC TWO-WAY REPEATED-MEASURES ANOVA
# Design: F1-LD-F1 (Cohort = between-subjects, Week = within-subjects)
# =============================================================================

def run_nparld_cohort_week_r(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
    save_report: bool = True,
    prefix: str = "nparld",
) -> dict:
    """
    Nonparametric equivalent of a two-way repeated-measures ANOVA via R nparLD.

    Design  : F1-LD-F1 (one between-subjects factor x one within-subjects factor)
    Between : Cohort  (CA% schedule label)
    Within  : Week    (1-indexed chronological week)
    Response: per-animal per-week mean of ``measure``

    Requires R with the nparLD package installed.

    Returns
    -------
    dict with keys: 'r_output', 'report_path', 'n_subjects', 'cohorts', 'weeks'
    """
    # ── Build combined weekly-mean dataset ────────────────────────────────────
    frames = []
    for label, df in cohort_dfs.items():
        d = df.copy()
        d = clean_cohort(d)
        if 'Day' not in d.columns:
            d['Cohort'] = label  # ensure ramp day-numbering and Day 1 exclusion are applied correctly
            d = add_day_column_across_cohorts(d)
        d = d[d['Day'] >= 1].copy()
        d = _add_week_column_across_cohorts(d)
        if measure not in d.columns:
            print(f"  [WARNING] '{measure}' not found in cohort '{label}' — skipping")
            continue
        d['_Cohort'] = label
        sub = d[['ID', '_Cohort', 'Week', measure]].dropna(subset=[measure])
        frames.append(sub)

    if not frames:
        print(f"  [ERROR] No data available for measure '{measure}'.")
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={measure: 'TotalChange', '_Cohort': 'Cohort'})

    # Per-animal per-week mean
    weekly = (
        combined
        .groupby(['ID', 'Cohort', 'Week'], as_index=False)['TotalChange']
        .mean()
    )

    if weeks is not None:
        weekly = weekly[weekly['Week'].isin(weeks)].copy()

    # ── Diagnostic: per-cohort per-week means (for cross-verification) ───
    print("\n  [DIAG] Per-cohort per-week mean TotalChange (before complete-case filter):")
    diag = (
        weekly
        .groupby(['Cohort', 'Week'])['TotalChange']
        .agg(n='count', mean='mean', sd='std')
        .reset_index()
    )
    for cohort, grp in diag.groupby('Cohort'):
        print(f"    Cohort: {cohort}")
        for _, row in grp.iterrows():
            print(f"      Week {int(row['Week'])}: n={int(row['n'])}, mean={row['mean']:.4f}, SD={row['sd']:.4f}")
    print()

    # Complete-case filter: keep only animals present in every Week
    week_counts = weekly.groupby('ID')['Week'].nunique()
    all_weeks_n = weekly['Week'].nunique()
    complete_ids = week_counts[week_counts == all_weeks_n].index
    n_dropped = weekly['ID'].nunique() - len(complete_ids)
    if n_dropped:
        print(f"  [NOTE] Dropped {n_dropped} animal(s) with incomplete weekly records; "
              f"{len(complete_ids)} retained")
    weekly = weekly[weekly['ID'].isin(complete_ids)].copy()

    if weekly.empty or len(complete_ids) < 3:
        print(f"  [ERROR] Insufficient complete-case animals ({len(complete_ids)}) for nparLD.")
        return {}

    week_levels = sorted(weekly['Week'].unique())
    cohort_levels = sorted(weekly['Cohort'].unique())
    n_subjects = int(weekly['ID'].nunique())

    print(f"\nnparLD: {n_subjects} subjects x {len(week_levels)} weeks x {len(cohort_levels)} cohorts")
    print(f"  Cohorts : {cohort_levels}")
    print(f"  Weeks   : {week_levels}")

    # ── Write temp CSV ──────────────────────────────────────────────────────
    tmp_csv = Path(tempfile.mktemp(suffix='.csv'))
    weekly[['ID', 'Cohort', 'Week', 'TotalChange']].to_csv(str(tmp_csv), index=False)

    _csv_r     = str(tmp_csv).replace('\\', '/')
    _wk_levels = ', '.join(str(w) for w in week_levels)
    _co_levels = ', '.join(f'"{c}"' for c in cohort_levels)

    r_script = (
        'options(warn=1, scipen=999)\n'
        'if (!require("nparLD", quietly=TRUE, warn.conflicts=FALSE)) {\n'
        '  install.packages("nparLD", repos="https://cran.r-project.org", quiet=TRUE)\n'
        '  library(nparLD)\n'
        '}\n'
        '\n'
        f'data <- read.csv("{_csv_r}")\n'
        f'data$Week   <- factor(data$Week,   levels=c({_wk_levels}))\n'
        f'data$Cohort <- factor(data$Cohort, levels=c({_co_levels}))\n'
        'data$ID     <- factor(data$ID)\n'
        '\n'
        'cat("\\n================================================================\\n")\n'
        f'cat("nparLD: {measure} -- Cohort x Week (F1-LD-F1 design)\\n")\n'
        'cat("================================================================\\n")\n'
        'cat("N subjects :", nlevels(data$ID), "\\n")\n'
        'cat("Cohorts    :", paste(levels(data$Cohort), collapse=", "), "\\n")\n'
        'cat("Weeks      :", paste(levels(data$Week),  collapse=", "), "\\n\\n")\n'
        '\n'
        'result <- f1.ld.f1(\n'
        '  y          = data$TotalChange,\n'
        '  time       = data$Week,\n'
        '  group      = data$Cohort,\n'
        '  subject    = data$ID,\n'
        '  time.name  = "Week",\n'
        '  group.name = "Cohort",\n'
        '  description = FALSE\n'
        ')\n'
        '\n'
        'cat("\\n--- ANOVA-Type Statistic (ATS) ---\\n")\n'
        'print(result$ANOVA.test)\n'
        '\n'
        'cat("\\n--- ATS with Box approximation ---\\n")\n'
        'print(result$ANOVA.test.mod.Box)\n'
        '\n'
        'cat("\\n--- Wald-Type Statistic (WTS) ---\\n")\n'
        'print(result$Wald.test)\n'
        '\n'
        'cat("\\n--- Relative Treatment Effects (RTE) ---\\n")\n'
        'print(result$RTE)\n'
        '\n'
        'cat("\\n================================================================\\n")\n'
        'cat("POST-HOC: Between-cohort (Mann-Whitney U, Holm corrected)\\n")\n'
        'cat("  (per-animal mean across all weeks)\\n")\n'
        'cat("================================================================\\n")\n'
        'per_animal <- aggregate(TotalChange ~ ID + Cohort, data=data, FUN=mean)\n'
        'ph_grp <- pairwise.wilcox.test(\n'
        '  per_animal$TotalChange, per_animal$Cohort,\n'
        '  p.adjust.method="holm", exact=FALSE\n'
        ')\n'
        'print(ph_grp)\n'
        '\n'
        'cat("\\n================================================================\\n")\n'
        'cat("POST-HOC: Between-cohort comparisons at each week\\n")\n'
        'cat("  (Mann-Whitney U, Holm corrected within each week)\\n")\n'
        'cat("================================================================\\n")\n'
        'for (wk in levels(data$Week)) {\n'
        '  sub <- data[data$Week == wk, ]\n'
        '  cat("\\nWeek:", wk, "\\n")\n'
        '  cohs <- levels(sub$Cohort)\n'
        '  if (length(cohs) < 2) { cat("  (only 1 cohort -- no pairwise test)\\n"); next }\n'
        '  pairs_idx <- combn(length(cohs), 2)\n'
        '  p_raw <- apply(pairs_idx, 2, function(idx) {\n'
        '    x <- sub$TotalChange[sub$Cohort == cohs[idx[1]]]\n'
        '    y <- sub$TotalChange[sub$Cohort == cohs[idx[2]]]\n'
        '    if (length(x) < 1 || length(y) < 1) return(NA_real_)\n'
        '    tryCatch(\n'
        '      wilcox.test(x, y, exact=FALSE)$p.value,\n'
        '      error = function(e) NA_real_\n'
        '    )\n'
        '  })\n'
        '  p_holm <- p.adjust(p_raw, method="holm")\n'
        '  n_pairs <- ncol(pairs_idx)\n'
        '  for (k in seq_len(n_pairs)) {\n'
        '    ca <- cohs[pairs_idx[1, k]]\n'
        '    cb <- cohs[pairs_idx[2, k]]\n'
        '    na_ca <- sum(!is.na(sub$TotalChange[sub$Cohort == ca]))\n'
        '    nb_cb <- sum(!is.na(sub$TotalChange[sub$Cohort == cb]))\n'
        '    if (is.na(p_raw[k])) {\n'
        '      cat(sprintf("  %s (n=%d) vs %s (n=%d) : NA\\n", ca, na_ca, cb, nb_cb))\n'
        '    } else {\n'
        '      sig <- ifelse(p_holm[k] < 0.001, "***",\n'
        '               ifelse(p_holm[k] < 0.01, "**",\n'
        '                 ifelse(p_holm[k] < 0.05, "*", "ns")))\n'
        '      cat(sprintf("  %s (n=%d) vs %s (n=%d) : p_raw=%.4f  p_holm=%.4f  %s\\n",\n'
        '                  ca, na_ca, cb, nb_cb, p_raw[k], p_holm[k], sig))\n'
        '    }\n'
        '  }\n'
        '}\n'
        '\n'
        'cat("\\n================================================================\\n")\n'
        'cat("OMNIBUS SUMMARY -- ATS (single DV; BH-FDR across DVs not applicable)\\n")\n'
        'cat("================================================================\\n")\n'
        'sig_fn3 <- function(p) { if(is.na(p)) return(""); if(p<0.001) return("***"); if(p<0.01) return("**"); if(p<0.05) return("*"); if(p<0.1) return("."); return("") }\n'
        'fmt_p3  <- function(p) { if(is.na(p)) return("     NA"); if(p<0.0001) return("< .0001"); sprintf("%.4f", p) }\n'
        'cat(sprintf("  %-20s  %10s  %8s  %12s  %s\\n", "Effect", "ATS", "df", "p-value", "Sig"))\n'
        'cat("  ", strrep("-", 57), "\\n", sep="")\n'
        'ats_df3 <- result$ANOVA.test\n'
        'for (i3 in seq_len(nrow(ats_df3))) {\n'
        '  eff3 <- rownames(ats_df3)[i3]\n'
        '  pv3  <- ats_df3[i3, ncol(ats_df3)]\n'
        '  cat(sprintf("  %-20s  %10.4f  %8.4f  %12s  %s\\n",\n'
        '    eff3, ats_df3[i3,1], ats_df3[i3,2], fmt_p3(pv3), sig_fn3(pv3)))\n'
        '}\n'
        '\n'
        'cat("\\n================================================================\\n")\n'
        'cat("END\\n")\n'
    )

    tmp_r = Path(tempfile.mktemp(suffix='.R'))
    tmp_r.write_text(r_script, encoding='utf-8')

    # ── Locate Rscript ─────────────────────────────────────────────────────
    import glob as _glob
    rscript = shutil.which('Rscript') or shutil.which('Rscript.exe')
    if rscript is None:
        for _pat in (
            r'C:\Program Files\R\R-*\bin\Rscript.exe',
            r'C:\Program Files\R\R-*\bin\x64\Rscript.exe',
        ):
            _m = sorted(_glob.glob(_pat))
            if _m:
                rscript = _m[-1]
                break

    if rscript is None:
        print("ERROR: 'Rscript' not found. Install R and add to PATH.")
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)
        return {}

    r_output = ''
    try:
        proc = subprocess.run(
            [rscript, '--vanilla', str(tmp_r)],
            capture_output=True, text=True, timeout=300,
        )
        r_output = proc.stdout
        r_stderr = proc.stderr.strip()
        if proc.returncode != 0:
            print(f"R exited with code {proc.returncode}.")
        if r_stderr:
            non_trivial = [
                ln for ln in r_stderr.splitlines()
                if not ln.startswith('Loading') and ln.strip()
            ]
            if non_trivial:
                print("R messages:\n" + '\n'.join(non_trivial))
    except FileNotFoundError:
        print("ERROR: 'Rscript' not found. Install R and add to the system PATH.")
        return {}
    except subprocess.TimeoutExpired:
        print("ERROR: R script timed out after 300 s.")
        return {}
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)

    # Clean up R output: rename Wilcoxon label, strip trailing whitespace and leading tabs
    r_output = r_output.replace(
        'Pairwise comparisons using Wilcoxon rank sum test with continuity correction',
        'Pairwise comparisons using Mann-Whitney U test with continuity correction'
    )
    r_output = '\n'.join(ln.rstrip() for ln in r_output.replace('\t', '  ').splitlines())

    print(r_output)

    # ── Save report ────────────────────────────────────────────────────────
    report_path: Optional[Path] = None
    if save_report and r_output.strip():
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path.cwd() / f"{prefix}_nparld_cohort_week_{_ts}.txt"
        header_lines = [
            "=" * 72,
            f"nparLD: {measure} -- Cohort x Week (F1-LD-F1 nonparametric ANOVA)",
            "=" * 72,
            f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Measure     : {measure}",
            f"Cohorts     : {', '.join(cohort_levels)}",
            f"Weeks       : {week_levels}",
            f"N subjects  : {n_subjects} (complete cases across all weeks)",
            "",
            "Design      : F1-LD-F1 (one between-subjects x one within-subjects factor)",
            "Between     : Cohort  (CA% schedule)",
            "Within      : Week    (1-indexed chronological week)",
            "Response    : Per-animal per-week mean of measure",
            "",
            "Statistics  : ANOVA-Type Statistic (ATS, chi-sq approx.) -- primary",
            "              Box approximation of ATS -- secondary",
            "              Wald-Type Statistic (WTS) -- reference only",
            "Post-hoc    : Between-cohort: pairwise Mann-Whitney U, Holm corrected",
            "              Per-week     : pairwise Mann-Whitney U between cohorts at each week, Holm corrected",
            "",
            "R output:",
            "-" * 72,
            "",
        ]
        report_path.write_text('\n'.join(header_lines) + r_output, encoding='utf-8')
        print(f"\n[OK] Report saved -> {report_path}")

    return {
        'r_output': r_output,
        'report_path': report_path,
        'n_subjects': n_subjects,
        'cohorts': cohort_levels,
        'weeks': week_levels,
    }


# =============================================================================
# R-BASED ART: NONPARAMETRIC REPEATED-MEASURES ANOVA FOR BEHAVIORAL METRICS
# Design: Cohort (between-subjects) x Week (within-subjects)
# Response: per-animal per-week % of aberrant binary observations
# Metrics: No Nest, Anxious Behaviors, Lethargy
# =============================================================================

def run_art_behavior_r(
    cohort_dfs: Dict[str, pd.DataFrame],
    weeks: Optional[List[int]] = None,
    save_report: bool = True,
    prefix: str = "art",
) -> dict:
    """
    Aligned Ranks Transformation (ART) nonparametric two-way ANOVA for
    behavioral metrics: No Nest, Anxious Behaviors, Lethargy.

    Design  : Cohort (between-subjects) x Week (within-subjects)
    Response: per-animal per-week % of aberrant binary observations:
                No Nest   = % days Nest Made? == No
                Anxious   = % days Anxious Behaviors? == Yes
                Lethargy  = % days Lethargy? == Yes
    Model   : art(Value ~ Cohort * Week + (1|ID), data = ...)
    Post-hoc: art.con() pairwise contrasts, Holm corrected
                -- Cohort main effect
                -- Cohort x Week interaction (cohort comparisons at each week)

    Requires R with ARTool and emmeans packages installed.

    Returns
    -------
    dict with keys: 'r_output', 'report_path', 'n_subjects', 'cohorts', 'weeks'
    """
    BEHAVIORS = [
        ('Nest Made?',         False, 'No Nest',  'NoNest'),
        ('Anxious Behaviors?', True,  'Anxious Behaviors', 'Anxious'),
        ('Lethargy?',          True,  'Lethargy', 'Lethargy'),
    ]

    def _to_bool_series(s: pd.Series) -> pd.Series:
        T = {'yes', 'true', '1', 'y'}
        F = {'no', 'false', '0', 'n'}
        def _cv(v):
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)) and not pd.isna(v): return bool(v)
            if isinstance(v, str):
                ls = v.strip().lower()
                if ls in T: return True
                if ls in F: return False
            return None
        return s.map(_cv)

    # ── Build per-animal per-week records ────────────────────────────────
    frames = []
    for label, df in cohort_dfs.items():
        cdf = clean_cohort(df.copy())
        if 'Day' not in cdf.columns:
            cdf['Cohort'] = label  # ensure ramp day-numbering and Day 1 exclusion are applied correctly
            cdf = add_day_column_across_cohorts(cdf)
        cdf = cdf[cdf['Day'] >= 1].copy()
        cdf = _add_week_column_across_cohorts(cdf)

        for col, aberrant_val, _, col_key in BEHAVIORS:
            if col in cdf.columns:
                cdf[f'_bin_{col_key}'] = _to_bool_series(cdf[col])

        for (animal_id, week_num), grp in cdf.groupby(['ID', 'Week']):
            row: dict = {'ID': str(animal_id), 'Cohort': label, 'Week': int(week_num)}
            for col, aberrant_val, _, col_key in BEHAVIORS:
                bcol = f'_bin_{col_key}'
                if bcol not in grp.columns:
                    row[col_key] = float('nan')
                    continue
                valid = grp[bcol].dropna()
                if len(valid) == 0:
                    row[col_key] = float('nan')
                else:
                    row[col_key] = 100.0 * (valid == aberrant_val).sum() / len(valid)
            frames.append(row)

    if not frames:
        print("  [ERROR] No behavioral data found in any cohort.")
        return {}

    weekly = pd.DataFrame(frames)

    if weeks is not None:
        weekly = weekly[weekly['Week'].isin(weeks)].copy()

    # Complete-case filter: animals observed in every week
    week_counts = weekly.groupby('ID')['Week'].nunique()
    all_weeks_n = weekly['Week'].nunique()
    complete_ids = week_counts[week_counts == all_weeks_n].index
    n_dropped = weekly['ID'].nunique() - len(complete_ids)
    if n_dropped:
        print(f"  [NOTE] Dropped {n_dropped} animal(s) with incomplete weekly records; "
              f"{len(complete_ids)} retained")
    weekly = weekly[weekly['ID'].isin(complete_ids)].copy()

    if weekly.empty or len(complete_ids) < 3:
        print(f"  [ERROR] Insufficient complete-case animals ({len(complete_ids)}) for ART.")
        return {}

    week_levels   = sorted(weekly['Week'].unique())
    cohort_levels = sorted(weekly['Cohort'].unique())
    n_subjects    = int(weekly['ID'].nunique())

    print(f"\nART behavior: {n_subjects} subjects x {len(week_levels)} weeks "
          f"x {len(cohort_levels)} cohorts")
    print(f"  Cohorts : {cohort_levels}")
    print(f"  Weeks   : {week_levels}")

    # ── Write temp CSV ───────────────────────────────────────────────────
    tmp_csv = Path(tempfile.mktemp(suffix='.csv'))
    col_keys = [ck for _, _, _, ck in BEHAVIORS]
    weekly[['ID', 'Cohort', 'Week'] + col_keys].to_csv(str(tmp_csv), index=False)

    _csv_r     = str(tmp_csv).replace('\\', '/')
    _wk_levels = ', '.join(str(w) for w in week_levels)
    _co_levels = ', '.join(f'"{c}"' for c in cohort_levels)

    r_script = (
        'suppressPackageStartupMessages({\n'
        '  for (pkg in c("ARTool", "emmeans")) {\n'
        '    if (!require(pkg, quietly=TRUE, warn.conflicts=FALSE, character.only=TRUE)) {\n'
        '      install.packages(pkg, repos="https://cran.r-project.org", quiet=TRUE)\n'
        '      library(pkg, character.only=TRUE)\n'
        '    }\n'
        '  }\n'
        '})\n'
        '\n'
        f'data <- read.csv("{_csv_r}")\n'
        f'data$Week   <- factor(data$Week,   levels=c({_wk_levels}))\n'
        f'data$Cohort <- factor(data$Cohort, levels=c({_co_levels}))\n'
        'data$ID     <- factor(data$ID)\n'
        '\n'
        'metrics <- list(\n'
        '  list(col="NoNest",   label="No Nest (% days nest absent)"),\n'
        '  list(col="Anxious",  label="Anxious Behaviors (% days)"),\n'
        '  list(col="Lethargy", label="Lethargy (% days)")\n'
        ')\n'
        '\n'
        'for (m in metrics) {\n'
        '  cat("\\n", strrep("=", 72), "\\n", sep="")\n'
        '  cat("ART: ", m$label, "\\n", sep="")\n'
        '  cat(strrep("=", 72), "\\n", sep="")\n'
        '\n'
        '  sub <- data[!is.na(data[[m$col]]), ]\n'
        '  sub$Cohort <- droplevels(sub$Cohort)\n'
        '  sub$Week   <- droplevels(sub$Week)\n'
        '  sub$ID     <- droplevels(sub$ID)\n'
        '  cat("N subjects :", nlevels(sub$ID), "\\n")\n'
        '  cat("Cohorts    :", paste(levels(sub$Cohort), collapse=", "), "\\n")\n'
        '  cat("Weeks      :", paste(levels(sub$Week),   collapse=", "), "\\n\\n")\n'
        '\n'
        '  form <- as.formula(paste(m$col, "~ Cohort * Week + (1|ID)"))\n'
        '  fit <- tryCatch(\n'
        '    art(form, data=sub),\n'
        '    error = function(e) {\n'
        '      cat("  [ERROR] ART fit failed:", conditionMessage(e), "\\n"); NULL\n'
        '    }\n'
        '  )\n'
        '  if (is.null(fit)) next\n'
        '\n'
        '  cat("--- ART ANOVA (omnibus) ---\\n")\n'
        '  print(anova(fit))\n'
        '\n'
        '  cat("\\n--- Post-hoc: Cohort main effect (pairwise, Holm) ---\\n")\n'
        '  tryCatch({\n'
        '    con_coh <- art.con(fit, "Cohort", adjust="holm")\n'
        '    print(summary(con_coh))\n'
        '  }, error = function(e) cat("  [SKIP cohort post-hoc]", conditionMessage(e), "\\n"))\n'
        '\n'
        '  cat("\\n--- Post-hoc: Cohort x Week interaction (pairwise cohorts at each week, Holm) ---\\n")\n'
        '  tryCatch({\n'
        '    con_int <- art.con(fit, "Cohort:Week", adjust="holm")\n'
        '    print(summary(con_int))\n'
        '  }, error = function(e) cat("  [SKIP interaction post-hoc]", conditionMessage(e), "\\n"))\n'
        '}\n'
        '\n'
        'cat("\\n", strrep("=", 72), "\\n", sep="")\n'
        'cat("END\\n")\n'
    )

    tmp_r = Path(tempfile.mktemp(suffix='.R'))
    tmp_r.write_text(r_script, encoding='utf-8')

    # ── Locate Rscript ───────────────────────────────────────────────────
    import glob as _glob
    rscript = shutil.which('Rscript') or shutil.which('Rscript.exe')
    if rscript is None:
        for _pat in (
            r'C:\Program Files\R\R-*\bin\Rscript.exe',
            r'C:\Program Files\R\R-*\bin\x64\Rscript.exe',
        ):
            _m = sorted(_glob.glob(_pat))
            if _m:
                rscript = _m[-1]
                break

    if rscript is None:
        print("ERROR: 'Rscript' not found. Install R and add to PATH.")
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)
        return {}

    r_output = ''
    try:
        proc = subprocess.run(
            [rscript, '--vanilla', str(tmp_r)],
            capture_output=True, text=True, timeout=300,
        )
        r_output = proc.stdout
        r_stderr = proc.stderr.strip()
        if proc.returncode != 0:
            print(f"R exited with code {proc.returncode}.")
        if r_stderr:
            non_trivial = [
                ln for ln in r_stderr.splitlines()
                if not ln.startswith('Loading') and ln.strip()
            ]
            if non_trivial:
                print("R messages:\n" + '\n'.join(non_trivial))
    except FileNotFoundError:
        print("ERROR: 'Rscript' not found. Install R and add to the system PATH.")
        return {}
    except subprocess.TimeoutExpired:
        print("ERROR: R script timed out after 300 s.")
        return {}
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)

    print(r_output)

    # ── Save report ──────────────────────────────────────────────────────
    report_path: Optional[Path] = None
    if save_report and r_output.strip():
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path.cwd() / f"{prefix}_art_behavior_{_ts}.txt"
        header_lines = [
            "=" * 72,
            "ART: Behavioral Metrics -- Cohort x Week",
            "     Aligned Ranks Transformation (ARTool package)",
            "=" * 72,
            f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Metrics     : No Nest, Anxious Behaviors, Lethargy",
            f"Cohorts     : {', '.join(cohort_levels)}",
            f"Weeks       : {week_levels}",
            f"N subjects  : {n_subjects} (complete cases across all weeks)",
            "",
            "Design      : Cohort (between-subjects) x Week (within-subjects)",
            "Response    : Per-animal per-week % of aberrant binary observations",
            "              No Nest   = % days Nest Made? == No",
            "              Anxious   = % days Anxious Behaviors? == Yes",
            "              Lethargy  = % days Lethargy? == Yes",
            "",
            "Model       : art(Value ~ Cohort * Week + (1|ID))",
            "Omnibus     : ART ANOVA (F-test on aligned-and-ranked data)",
            "Post-hoc    : art.con() pairwise contrasts, Holm corrected",
            "              -- Cohort main effect (overall)",
            "              -- Cohort x Week interaction (cohort pairs at each week)",
            "",
            "R output:",
            "-" * 72,
            "",
        ]
        report_path.write_text('\n'.join(header_lines) + r_output, encoding='utf-8')
        print(f"\n[OK] Report saved -> {report_path}")

    return {
        'r_output': r_output,
        'report_path': report_path,
        'n_subjects': n_subjects,
        'cohorts': cohort_levels,
        'weeks': week_levels,
    }


# =============================================================================
# R-BASED nparLD FOR BEHAVIORAL METRICS
# Same F1-LD-F1 design as the weight nparLD, applied to per-animal per-week
# % of aberrant binary observations for No Nest, Anxious, Lethargy.
# Provides ATS / Box / WTS / RTE and between-cohort per-week MWU post-hoc.
# =============================================================================

def run_nparld_behavior_r(
    cohort_dfs: Dict[str, pd.DataFrame],
    weeks: Optional[List[int]] = None,
    save_report: bool = True,
    prefix: str = "nparld_behavior",
) -> dict:
    """
    nparLD F1-LD-F1 nonparametric two-way ANOVA for behavioral metrics.

    Design  : Cohort (between-subjects) x Week (within-subjects)
    Response: per-animal per-week % of aberrant binary observations:
                No Nest   = % days Nest Made? == No
                Anxious   = % days Anxious Behaviors? == Yes
                Lethargy  = % days Lethargy? == Yes
    Model   : f1.ld.f1() run separately for each metric

    Returns
    -------
    dict with keys: 'r_output', 'report_path', 'n_subjects', 'cohorts', 'weeks'
    """
    BEHAVIORS = [
        ('Nest Made?',         False, 'No Nest',            'NoNest'),
        ('Anxious Behaviors?', True,  'Anxious Behaviors',  'Anxious'),
        ('Lethargy?',          True,  'Lethargy',           'Lethargy'),
    ]

    def _to_bool_series(s: pd.Series) -> pd.Series:
        T = {'yes', 'true', '1', 'y'}
        F = {'no', 'false', '0', 'n'}
        def _cv(v):
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)) and not pd.isna(v): return bool(v)
            if isinstance(v, str):
                ls = v.strip().lower()
                if ls in T: return True
                if ls in F: return False
            return None
        return s.map(_cv)

    # ── Build per-animal per-week records ────────────────────────────────
    frames = []
    for label, df in cohort_dfs.items():
        cdf = clean_cohort(df.copy())
        if 'Day' not in cdf.columns:
            cdf['Cohort'] = label  # ensure ramp day-numbering and Day 1 exclusion are applied correctly
            cdf = add_day_column_across_cohorts(cdf)
        cdf = cdf[cdf['Day'] >= 1].copy()
        cdf = _add_week_column_across_cohorts(cdf)

        for col, aberrant_val, _, col_key in BEHAVIORS:
            if col in cdf.columns:
                cdf[f'_bin_{col_key}'] = _to_bool_series(cdf[col])

        for (animal_id, week_num), grp in cdf.groupby(['ID', 'Week']):
            row: dict = {'ID': str(animal_id), 'Cohort': label, 'Week': int(week_num)}
            for col, aberrant_val, _, col_key in BEHAVIORS:
                bcol = f'_bin_{col_key}'
                if bcol not in grp.columns:
                    row[col_key] = float('nan')
                    continue
                valid = grp[bcol].dropna()
                if len(valid) == 0:
                    row[col_key] = float('nan')
                else:
                    row[col_key] = 100.0 * (valid == aberrant_val).sum() / len(valid)
            frames.append(row)

    if not frames:
        print("  [ERROR] No behavioral data found in any cohort.")
        return {}

    weekly = pd.DataFrame(frames)

    if weeks is not None:
        weekly = weekly[weekly['Week'].isin(weeks)].copy()

    # Complete-case filter
    week_counts = weekly.groupby('ID')['Week'].nunique()
    all_weeks_n = weekly['Week'].nunique()
    complete_ids = week_counts[week_counts == all_weeks_n].index
    n_dropped = weekly['ID'].nunique() - len(complete_ids)
    if n_dropped:
        print(f"  [NOTE] Dropped {n_dropped} animal(s) with incomplete records; "
              f"{len(complete_ids)} retained")
    weekly = weekly[weekly['ID'].isin(complete_ids)].copy()

    if weekly.empty or len(complete_ids) < 3:
        print(f"  [ERROR] Insufficient complete-case animals ({len(complete_ids)}) for nparLD.")
        return {}

    week_levels   = sorted(weekly['Week'].unique())
    cohort_levels = sorted(weekly['Cohort'].unique())
    n_subjects    = int(weekly['ID'].nunique())

    print(f"\nnparLD behavior: {n_subjects} subjects x {len(week_levels)} weeks "
          f"x {len(cohort_levels)} cohorts")
    print(f"  Cohorts : {cohort_levels}")
    print(f"  Weeks   : {week_levels}")

    # ── Write temp CSV ───────────────────────────────────────────────────
    col_keys = [ck for _, _, _, ck in BEHAVIORS]
    tmp_csv = Path(tempfile.mktemp(suffix='.csv'))
    weekly[['ID', 'Cohort', 'Week'] + col_keys].to_csv(str(tmp_csv), index=False)

    _csv_r     = str(tmp_csv).replace('\\', '/')
    _wk_levels = ', '.join(str(w) for w in week_levels)
    _co_levels = ', '.join(f'"{c}"' for c in cohort_levels)

    # Build post-hoc block identical to weight nparLD (between-cohort per week, MWU Holm)
    _posthoc = (
        'cat("\\n--- POST-HOC: Between-cohort at each week (Mann-Whitney U, Holm) ---\\n")\n'
        'for (wk in levels(sub$Week)) {\n'
        '  sw <- sub[sub$Week == wk, ]\n'
        '  cat("\\nWeek:", wk, "\\n")\n'
        '  cohs <- levels(sw$Cohort)\n'
        '  if (length(cohs) < 2) { cat("  (only 1 cohort)\\n"); next }\n'
        '  pidx <- combn(length(cohs), 2)\n'
        '  p_raw <- apply(pidx, 2, function(idx) {\n'
        '    x <- sw$Value[sw$Cohort == cohs[idx[1]]]\n'
        '    y <- sw$Value[sw$Cohort == cohs[idx[2]]]\n'
        '    if (length(x) < 1 || length(y) < 1) return(NA_real_)\n'
        '    tryCatch(wilcox.test(x, y, exact=FALSE)$p.value, error=function(e) NA_real_)\n'
        '  })\n'
        '  p_holm <- p.adjust(p_raw, method="holm")\n'
        '  for (k in seq_len(ncol(pidx))) {\n'
        '    ca <- cohs[pidx[1, k]]; cb <- cohs[pidx[2, k]]\n'
        '    na <- sum(!is.na(sw$Value[sw$Cohort == ca]))\n'
        '    nb <- sum(!is.na(sw$Value[sw$Cohort == cb]))\n'
        '    if (is.na(p_raw[k])) {\n'
        '      cat(sprintf("  %s (n=%d) vs %s (n=%d) : NA\\n", ca, na, cb, nb))\n'
        '    } else {\n'
        '      sig <- ifelse(p_holm[k]<0.001,"***",ifelse(p_holm[k]<0.01,"**",ifelse(p_holm[k]<0.05,"*","ns")))\n'
        '      cat(sprintf("  %s (n=%d) vs %s (n=%d) : p_raw=%.4f  p_holm=%.4f  %s\\n",\n'
        '                  ca, na, cb, nb, p_raw[k], p_holm[k], sig))\n'
        '    }\n'
        '  }\n'
        '}\n'
    )

    r_script = (
        'options(warn=1, scipen=999)\n'
        'if (!require("nparLD", quietly=TRUE, warn.conflicts=FALSE)) {\n'
        '  install.packages("nparLD", repos="https://cran.r-project.org", quiet=TRUE)\n'
        '  library(nparLD)\n'
        '}\n'
        '\n'
        f'data <- read.csv("{_csv_r}")\n'
        f'data$Week   <- factor(data$Week,   levels=c({_wk_levels}))\n'
        f'data$Cohort <- factor(data$Cohort, levels=c({_co_levels}))\n'
        'data$ID     <- factor(data$ID)\n'
        '\n'
        'metrics <- list(\n'
        '  list(col="NoNest",   label="No Nest (% days nest absent)"),\n'
        '  list(col="Anxious",  label="Anxious Behaviors (% days)"),\n'
        '  list(col="Lethargy", label="Lethargy (% days)")\n'
        ')\n'
        '\n'
        'dv_labels <- character(0)\n'
        'p_cohort  <- numeric(0)\n'
        'p_week    <- numeric(0)\n'
        'p_int     <- numeric(0)\n'
        '\n'
        'for (m in metrics) {\n'
        '  cat("\\n", strrep("=", 72), "\\n", sep="")\n'
        '  cat("nparLD: ", m$label, "\\n", sep="")\n'
        '  cat(strrep("=", 72), "\\n", sep="")\n'
        '\n'
        '  sub <- data[!is.na(data[[m$col]]), ]\n'
        '  sub$Cohort <- droplevels(sub$Cohort)\n'
        '  sub$Week   <- droplevels(sub$Week)\n'
        '  sub$ID     <- droplevels(sub$ID)\n'
        '  sub$Value  <- sub[[m$col]]\n'
        '  cat("N subjects :", nlevels(sub$ID), "\\n")\n'
        '  cat("Cohorts    :", paste(levels(sub$Cohort), collapse=", "), "\\n")\n'
        '  cat("Weeks      :", paste(levels(sub$Week),   collapse=", "), "\\n\\n")\n'
        '\n'
        '  result <- tryCatch(\n'
        '    f1.ld.f1(\n'
        '      y          = sub$Value,\n'
        '      time       = sub$Week,\n'
        '      group      = sub$Cohort,\n'
        '      subject    = sub$ID,\n'
        '      time.name  = "Week",\n'
        '      group.name = "Cohort",\n'
        '      description = FALSE\n'
        '    ),\n'
        '    error = function(e) {\n'
        '      cat("  [ERROR] nparLD fit failed:", conditionMessage(e), "\\n"); NULL\n'
        '    }\n'
        '  )\n'
        '  if (is.null(result)) next\n'
        '\n'
        '  cat("\\n--- ANOVA-Type Statistic (ATS) ---\\n")\n'
        '  print(result$ANOVA.test)\n'
        '  cat("\\n--- ATS with Box approximation ---\\n")\n'
        '  print(result$ANOVA.test.mod.Box)\n'
        '  cat("\\n--- Wald-Type Statistic (WTS) ---\\n")\n'
        '  print(result$Wald.test)\n'
        '  cat("\\n--- Relative Treatment Effects (RTE) ---\\n")\n'
        '  print(result$RTE)\n'
        '  dv_labels <- c(dv_labels, m$label)\n'
        '  ats_p <- result$ANOVA.test[, ncol(result$ANOVA.test)]\n'
        '  p_cohort <- c(p_cohort, ats_p[1])\n'
        '  p_week   <- c(p_week,   ats_p[2])\n'
        '  p_int    <- c(p_int,    ats_p[3])\n'
        + _posthoc +
        '}\n'
        '\n'
        'cat("\\n", strrep("=", 72), "\\n", sep="")\n'
        'cat("BH-FDR CORRECTION ACROSS BEHAVIORAL METRICS\\n")\n'
        'cat(strrep("=", 72), "\\n", sep="")\n'
        'cat("  ATS p-values corrected using Benjamini-Hochberg FDR.\\n")\n'
        'cat("  Correction applied separately per effect row\\n")\n'
        'cat("  (one adjusted set for Cohort, Week, and Cohort:Week).\\n\\n")\n'
        'sig_fn <- function(p) { if(is.na(p)) return(""); if(p<0.001) return("***"); if(p<0.01) return("**"); if(p<0.05) return("*"); return("") }\n'
        'fmt_p  <- function(p) { if(is.na(p)) return("     NA"); if(p<0.0001) return("< .0001"); sprintf("%.4f", p) }\n'
        'effects_smry <- list(\n'
        '  list(name="Cohort",      p_raw=p_cohort),\n'
        '  list(name="Week",        p_raw=p_week),\n'
        '  list(name="Cohort:Week", p_raw=p_int)\n'
        ')\n'
        'for (eff in effects_smry) {\n'
        '  cat(sprintf("  Effect: %s\\n", eff$name))\n'
        '  cat(sprintf("  %-38s  %10s  %10s  %s\\n", "DV", "p (raw)", "p (BH-adj)", "Sig (adj)"))\n'
        '  cat("  ", strrep("-", 65), "\\n", sep="")\n'
        '  adj <- p.adjust(eff$p_raw, method="BH")\n'
        '  for (i in seq_along(dv_labels)) {\n'
        '    cat(sprintf("  %-38s  %10s  %10s  %s\\n",\n'
        '      dv_labels[i], fmt_p(eff$p_raw[i]), fmt_p(adj[i]), sig_fn(adj[i])))\n'
        '  }\n'
        '  cat("\\n")\n'
        '}\n'
        '\n'
        'cat("\\n", strrep("=", 72), "\\n", sep="")\n'
        'cat("END\\n")\n'
    )

    tmp_r = Path(tempfile.mktemp(suffix='.R'))
    tmp_r.write_text(r_script, encoding='utf-8')

    # ── Locate Rscript ───────────────────────────────────────────────────
    import glob as _glob
    rscript = shutil.which('Rscript') or shutil.which('Rscript.exe')
    if rscript is None:
        for _pat in (
            r'C:\Program Files\R\R-*\bin\Rscript.exe',
            r'C:\Program Files\R\R-*\bin\x64\Rscript.exe',
        ):
            _m = sorted(_glob.glob(_pat))
            if _m:
                rscript = _m[-1]
                break

    if rscript is None:
        print("ERROR: 'Rscript' not found. Install R and add to PATH.")
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)
        return {}

    r_output = ''
    try:
        proc = subprocess.run(
            [rscript, '--vanilla', str(tmp_r)],
            capture_output=True, text=True, timeout=300,
        )
        r_output = proc.stdout
        r_stderr = proc.stderr.strip()
        if proc.returncode != 0:
            print(f"R exited with code {proc.returncode}.")
        if r_stderr:
            non_trivial = [
                ln for ln in r_stderr.splitlines()
                if not ln.startswith('Loading') and ln.strip()
            ]
            if non_trivial:
                print("R messages:\n" + '\n'.join(non_trivial))
    except FileNotFoundError:
        print("ERROR: 'Rscript' not found. Install R and add to the system PATH.")
        return {}
    except subprocess.TimeoutExpired:
        print("ERROR: R script timed out after 300 s.")
        return {}
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)

    print(r_output)

    # ── Save report ──────────────────────────────────────────────────────
    report_path: Optional[Path] = None
    if save_report and r_output.strip():
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path.cwd() / f"{prefix}_nparld_behavior_{_ts}.txt"
        header_lines = [
            "=" * 72,
            "nparLD: Behavioral Metrics -- Cohort x Week (F1-LD-F1)",
            "=" * 72,
            f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Metrics     : No Nest, Anxious Behaviors, Lethargy",
            f"Cohorts     : {', '.join(cohort_levels)}",
            f"Weeks       : {week_levels}",
            f"N subjects  : {n_subjects} (complete cases across all weeks)",
            "",
            "Design      : F1-LD-F1 (one between-subjects x one within-subjects factor)",
            "Between     : Cohort  (CA% schedule)",
            "Within      : Week    (1-indexed chronological week)",
            "Response    : Per-animal per-week % of aberrant binary observations",
            "              No Nest   = % days Nest Made? == No",
            "              Anxious   = % days Anxious Behaviors? == Yes",
            "              Lethargy  = % days Lethargy? == Yes",
            "",
            "Statistics  : ATS (chi-sq approx.) -- primary",
            "              Box approx. of ATS   -- secondary (preferred for small N)",
            "              WTS                  -- reference only",
            "              RTE                  -- relative treatment effects",
            "Post-hoc    : Between-cohort at each week: pairwise Mann-Whitney U, Holm corrected",
            "",
            "R output:",
            "-" * 72,
            "",
        ]
        report_path.write_text('\n'.join(header_lines) + r_output, encoding='utf-8')
        print(f"\n[OK] Report saved -> {report_path}")

    return {
        'r_output': r_output,
        'report_path': report_path,
        'n_subjects': n_subjects,
        'cohorts': cohort_levels,
        'weeks': week_levels,
    }


def _run_unknown_menu(cohorts: Dict[str, pd.DataFrame], comparison: str) -> None:
    """Placeholder menu for comparison types not yet implemented."""
    label_map = {
        '0vramp': '0% nonramp vs Ramp',
        '2vramp': '2% nonramp vs Ramp',
        'all3':   '0% nonramp vs 2% nonramp vs Ramp',
        'unknown': 'Unknown cohort combination',
    }
    print("\n" + "=" * 80)
    print(f"COMPARISON TYPE: {label_map.get(comparison, comparison)}")
    print("=" * 80)
    print("\n[INFO] Analysis menu for this comparison type is coming soon.")
    print("       The cohort-week framework functions are available:")
    print("         combine_cohorts_with_weeks(cohorts, cohort_metadata)")
    print("         perform_cohort_week_mixed_anova(cohorts, measure, cohort_metadata)")
    print("         generate_cohort_week_report(results, save_path=...)")
    print("         plot_weekly_means_by_cohort(combined, measure, cohort_metadata)")
    print()


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Step 1 -- choose number of cohorts and load files
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("CROSS-COHORT ANALYSIS")
    print("=" * 80)
    print("\nHow many cohorts would you like to compare?")
    print("  2 -- compare two cohorts")
    print("  3 -- compare three cohorts")
    n_input = input("\nEnter 2 or 3: ").strip()
    try:
        n_cohorts = int(n_input)
        if n_cohorts not in (2, 3):
            raise ValueError
    except ValueError:
        print("[ERROR] Please enter 2 or 3. Exiting.")
        raise SystemExit(1)

    cohorts = select_and_load_cohorts(n_cohorts=n_cohorts)

    if not cohorts or len(cohorts) < n_cohorts:
        print("[ERROR] Not enough cohorts loaded. Exiting.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # Step 2 -- auto-detect comparison type and route to the right menu
    # ------------------------------------------------------------------ #
    comparison = _detect_comparison_type(cohorts)

    label_map = {
        '0v2':      '0% nonramp  vs  2% nonramp',
        '0vramp':   '0% nonramp  vs  Ramp',
        '2vramp':   '2% nonramp  vs  Ramp',
        'all3':     '0% nonramp  vs  2% nonramp  vs  Ramp',
        'rampramp': 'Ramp  vs  2-Week Ramp',
        'unknown':  'Unknown combination',
    }
    print("\n" + "=" * 80)
    print(f"Detected comparison type: {label_map.get(comparison, comparison)}")
    print("=" * 80)

    # If detection is ambiguous, ask the user to confirm or override
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
        _run_0v2_menu(cohorts)
    elif comparison == '0vramp':
        _run_0vramp_menu(cohorts)
    elif comparison == '2vramp':
        _run_2vramp_menu(cohorts)
    elif comparison == 'all3':
        _run_all3_menu(cohorts)
    elif comparison == 'rampramp':
        _run_rampramp_menu(cohorts)
    else:
        _run_unknown_menu(cohorts, comparison)
