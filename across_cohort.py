"""
Cross-Cohort Analysis Module

This module provides functionality to load and compare data across multiple cohorts.
Each cohort is loaded from a separate master CSV file (e.g., master_data_0%.csv, master_data_2%.csv).

Main features:
- Load multiple cohort CSV files into separate dataframes
- Clean and standardize data across cohorts
- Compare metrics (weights, behavioral measures, fecal counts) across cohorts
- Aggregate and summarize cross-cohort data

Usage:
    from across_cohort import load_cohorts, preview_cohorts
    
    # Option 1: Specify paths directly
    cohort_paths = {
        "0% CA": Path("0%_files/master_data_0%.csv"),
        "2% CA": Path("2%_files/master_data_2%.csv")
    }
    cohorts = load_cohorts(cohort_paths)
    
    # Option 2: Use GUI to select files
    cohorts = select_and_load_cohorts(n_cohorts=2)
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import re
import math
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
        print(f"\n{'─'*80}")
        print(f"COHORT: {label}")
        print(f"{'─'*80}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
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


def add_day_column_across_cohorts(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Day' column to combined cohort data where Day 0 is the first date for each animal.
    
    Parameters:
        combined_df: Combined DataFrame with ID and Date columns
        
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
    
    # Compute day number per ID (Day 0 = first measurement day, subtracting 1
    # so that the first actual data day is Day 0 rather than Day 1)
    first_dates = df.groupby('ID')['Date'].transform('min')
    df['Day'] = (df['Date'] - first_dates).dt.days - 1
    
    print(f"[OK] Added 'Day' column (range: {df['Day'].min()} to {df['Day'].max()})")
    
    return df


def perform_cross_cohort_mixed_anova(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_points: Optional[List[int]] = None,
    ss_type: int = 3
) -> Dict:
    """
    Perform 3-way Mixed ANOVA across cohorts: CA% (between) × Time (within) × Sex (between).
    
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
    print("CROSS-COHORT MIXED ANOVA: CA% × TIME × SEX")
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
    print("  Design: CA% (between) × Time (within) × Sex (between)")
    
    try:
        # Use pingouin's mixed_anova for within-between design
        # We need to specify which factor is within-subjects (Day) and which are between (CA%, Sex)
        
        # Create a combined between-subjects factor for CA% × Sex
        analysis_df['Group'] = (analysis_df['CA (%)'].astype(str) + '%_' + 
                               analysis_df['Sex'].astype(str))
        
        # Run mixed ANOVA: Day (within) × Group (between)
        aov = pg.mixed_anova(
            data=analysis_df,
            dv=measure,
            within='Day',
            subject='ID',
            between='Group',
            correction=True  # Greenhouse-Geisser correction for sphericity
        )
        
        print("\nMixed ANOVA Results (Day × Group):")
        print(aov.to_string())
        
        # Now run separate ANOVAs to decompose the Group effect into CA%, Sex, and CA%×Sex
        print("\n" + "="*80)
        print("DECOMPOSING GROUP EFFECTS: CA% × SEX")
        print("="*80)
        
        # Average across days for between-subjects effects
        subject_means = analysis_df.groupby(['ID', 'CA (%)', 'Sex'])[measure].mean().reset_index()
        
        # Rename 'CA (%)' to avoid special characters for pingouin
        subject_means_clean = subject_means.rename(columns={'CA (%)': 'CA_percent'})
        
        # Two-way ANOVA: CA% × Sex on averaged data
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
            
            print(f"\nDay × Group interaction:")
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
    Perform 2-Way Between-Subjects ANOVA: CA% × Sex (holding time constant)
    
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
    print("BETWEEN-SUBJECTS ANOVA: CA% × SEX")
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
    print("  Design: CA% (between) × Sex (between)")
    
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
    Perform Between-Subjects ANOVA (CA% × Sex) for EACH day separately.
    
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
    print("DAILY BETWEEN-SUBJECTS ANOVA: CA% × SEX FOR EACH DAY")
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
        
        print("\nCA% × Sex Interaction by Day:")
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
    Perform 2-Way Mixed ANOVA: Time (within) × CA% (between), holding Sex constant.
    
    This analyzes longitudinal weight changes for ONE sex at a time, testing:
    - Time (Day): Within-subjects factor (repeated measures over days)
    - CA%: Between-subjects factor (different cohorts)
    
    By stratifying by sex, this reveals whether the Time × CA% interaction differs
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
    print(f"SEX-STRATIFIED MIXED ANOVA: TIME (WITHIN) × CA% (BETWEEN)")
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
        
        # Run mixed ANOVA: Day (within) × CA% (between)
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
                print(f"   → Weight changes significantly over time for {sex}s")
            else:
                print(f"   → No significant change over time for {sex}s")
        
        # CA% main effect
        if 'CA (%)' in aov['Source'].values:
            ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
            ca_p = ca_row[p_col]
            ca_F = ca_row['F']
            ca_sig = "***" if ca_p < 0.001 else "**" if ca_p < 0.01 else "*" if ca_p < 0.05 else "ns"
            
            print(f"\n2. CA% Main Effect: {'SIGNIFICANT' if ca_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {ca_F:.3f}, p = {ca_p:.4f} {ca_sig}")
            if ca_p < 0.05:
                print(f"   → CA% concentrations differ for {sex}s")
            else:
                print(f"   → No CA% difference for {sex}s")
        
        # Interaction effect
        if 'CA (%) * Day' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time × CA% Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   → The time course differs between CA% levels for {sex}s")
            else:
                print(f"   → Similar time course across CA% levels for {sex}s")
        
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
    Perform 2-Way Mixed ANOVA: Time (within) × Sex (between), holding CA% constant.
    
    This analyzes longitudinal weight changes for ONE CA% level at a time, testing:
    - Time (Day): Within-subjects factor (repeated measures over days)
    - Sex: Between-subjects factor (M vs F)
    
    By stratifying by CA%, this reveals whether the Time × Sex interaction differs
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
    print(f"CA%-STRATIFIED MIXED ANOVA: TIME (WITHIN) × SEX (BETWEEN)")
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
        # Run mixed ANOVA: Day (within) × Sex (between)
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
                print(f"   → Weight changes significantly over time at {ca_percent}% CA")
            else:
                print(f"   → No significant change over time at {ca_percent}% CA")
        
        # Sex main effect
        if 'Sex' in aov['Source'].values:
            sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
            sex_p = sex_row[p_col]
            sex_F = sex_row['F']
            sex_sig = "***" if sex_p < 0.001 else "**" if sex_p < 0.01 else "*" if sex_p < 0.05 else "ns"
            
            print(f"\n2. Sex Main Effect: {'SIGNIFICANT' if sex_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {sex_F:.3f}, p = {sex_p:.4f} {sex_sig}")
            if sex_p < 0.05:
                print(f"   → Males and females differ at {ca_percent}% CA")
            else:
                print(f"   → No sex difference at {ca_percent}% CA")
        
        # Interaction effect
        if 'Sex * Day' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'Sex * Day'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time × Sex Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   → The time course differs between sexes at {ca_percent}% CA")
            else:
                print(f"   → Similar time course for both sexes at {ca_percent}% CA")
        elif 'Day * Sex' in aov['Source'].values:
            int_row = aov[aov['Source'] == 'Day * Sex'].iloc[0]
            int_p = int_row[p_col]
            int_F = int_row['F']
            int_sig = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "ns"
            
            print(f"\n3. Time × Sex Interaction: {'SIGNIFICANT' if int_p < 0.05 else 'NOT SIGNIFICANT'}")
            print(f"   F = {int_F:.3f}, p = {int_p:.4f} {int_sig}")
            if int_p < 0.05:
                print(f"   → The time course differs between sexes at {ca_percent}% CA")
            else:
                print(f"   → Similar time course for both sexes at {ca_percent}% CA")
        
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


def _sex_to_style(sex: Optional[str]) -> Tuple[str, str]:
    """Return (color, marker) based on sex: M=green/square, F=purple/circle."""
    if sex == "M":
        return ("green", "s")
    if sex == "F":
        return ("purple", "o")
    return ("gray", "^")


def _ca_to_style(ca_pct: Optional[float]) -> Tuple[str, str]:
    """Return (color, marker) based on CA%: 0=dodgerblue/triangle, 2=orangered/circle."""
    if ca_pct == 0.0:
        return ("dodgerblue", "^")
    if ca_pct == 2.0:
        return ("orangered", "o")
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
            ax.axhline(0, linestyle='-', color='0', linewidth=1.5, alpha=0.8, zorder=1)
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

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot each ID as a separate line
    for mid, s in series_by_id.items():
        color, marker = _sex_to_style(sex_map.get(mid))
        ax.plot(
            s.index,
            s.values,
            label=str(mid),
            marker=marker,
            markersize=3,
            linewidth=1.5,
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
                        step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
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
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot each ID as a separate line
    for mid, s in series_by_id.items():
        color, marker = _sex_to_style(sex_map.get(mid))
        ax.plot(
            s.index,
            s.values,
            label=str(mid),
            marker=marker,
            markersize=3,
            linewidth=1.5,
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
                        step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
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
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot male data
    if not male_mean.empty:
        ax.plot(male_mean.index, male_mean.values, label="Male", color="green", marker="s",
                markersize=4, linewidth=2, alpha=0.9)
        ax.fill_between(male_mean.index,
                        male_mean - male_sem,
                        male_mean + male_sem,
                        color="green", alpha=0.2)

    # Plot female data
    if not female_mean.empty:
        ax.plot(female_mean.index, female_mean.values, label="Female", color="purple", marker="o",
                markersize=4, linewidth=2, alpha=0.9)
        ax.fill_between(female_mean.index,
                        female_mean - female_sem,
                        female_mean + female_sem,
                        color="purple", alpha=0.2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change (Mean ± SEM)")
    ax.grid(False)

    if title is None:
        title = "Total Change by Sex (Mean ± SEM)"
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
                        step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="Sex", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot male data
    if not male_mean.empty:
        ax.plot(male_mean.index, male_mean.values, label="Male", color="green", marker="s",
                markersize=4, linewidth=2, alpha=0.9)
        ax.fill_between(male_mean.index,
                        male_mean - male_sem,
                        male_mean + male_sem,
                        color="green", alpha=0.2)

    # Plot female data
    if not female_mean.empty:
        ax.plot(female_mean.index, female_mean.values, label="Female", color="purple", marker="o",
                markersize=4, linewidth=2, alpha=0.9)
        ax.fill_between(female_mean.index,
                        female_mean - female_sem,
                        female_mean + female_sem,
                        color="purple", alpha=0.2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change (Mean ± SEM)")
    ax.grid(False)

    if title is None:
        title = "Daily Change by Sex (Mean ± SEM)"
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
                        step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
    y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
    _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="Sex", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot each CA% group
    all_group_means = []
    all_group_indices = []
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=f'{ca_val:.0f}% CA',
                    marker=marker, markersize=4, linewidth=2, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Change (Mean ± SEM)")
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
                            step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
        _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="CA%", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Plot each CA% group
    all_group_means = []
    all_group_indices = []
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        if not mean.empty:
            ax.plot(mean.index, mean.values, label=f'{ca_val:.0f}% CA',
                    marker=marker, markersize=4, linewidth=2, alpha=0.9, color=color)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
            all_group_means.append(mean)
            all_group_indices.append(mean.index.to_series())

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Change (Mean ± SEM)")
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
                            step=x_step, clamp_min=0, left_pad_steps=0, right_pad_steps=0)
        y_step = _auto_integer_step(y_data_min, y_data_max, target_ticks=7)
        _apply_integer_axis(ax, axis='y', data_min=y_data_min, data_max=y_data_max,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(title="CA%", loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
    Plot CA% × Sex interaction from between-subjects ANOVA.
    
    Shows mean values for each CA% × Sex combination with error bars (SEM).
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
    
    # Check if CA% × Sex interaction is significant
    anova_table = anova_results.get('anova_table')
    if anova_table is None:
        print("[WARNING] No ANOVA table found in results")
        return None
    
    p_col = 'p-unc' if 'p-unc' in anova_table.columns else 'p_unc'
    interaction_row = anova_table[anova_table['Source'] == 'CA (%) * Sex']
    
    if len(interaction_row) == 0:
        print(f"[WARNING] CA% × Sex interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] CA% × Sex interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting CA% × Sex interaction (p = {p_value:.4f})")
    
    # Get descriptive statistics
    group_stats = anova_results.get('descriptive_stats')
    if group_stats is None or len(group_stats) == 0:
        print("[WARNING] No descriptive statistics found")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    
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
            marker=marker, markersize=8, linewidth=2.5, capsize=5,
            color=color, label=label, alpha=0.85, markeredgewidth=1.5,
            markeredgecolor='black'
        )
    
    ax.set_xlabel('CA% Concentration')
    ax.set_ylabel(f'{measure} (Mean ± SEM)')
    
    if title is None:
        analysis_type = anova_results.get('analysis_type', '')
        title = f'CA% × Sex Interaction: {measure}\n({analysis_type})'
    
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
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    Plot Time × CA% interaction from sex-stratified mixed ANOVA.
    
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
    
    # Check if Time × CA% interaction is significant
    anova_table = anova_results.get('anova_table')
    if anova_table is None:
        print("[WARNING] No ANOVA table found in results")
        return None
    
    p_col = 'p-unc' if 'p-unc' in anova_table.columns else 'p_unc'
    interaction_row = anova_table[anova_table['Source'] == 'CA (%) * Day']
    
    if len(interaction_row) == 0:
        print(f"[WARNING] CA% × Day interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] Time × CA% interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting Time × CA% interaction (p = {p_value:.4f})")
    
    # Get data
    data_df = anova_results.get('data')
    if data_df is None or len(data_df) == 0:
        print("[WARNING] No data found in results")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    sex = anova_results.get('sex', 'Unknown')
    sex_label = 'Males' if sex == 'M' else 'Females'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Compute mean and SEM for each Day × CA% combination
    summary = data_df.groupby(['Day', 'CA (%)'])[measure].agg(['mean', 'sem', 'count']).reset_index()
    
    # Get unique CA% levels
    ca_levels = sorted(summary['CA (%)'].unique())
    
    # Plot trajectory for each CA% level
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ca_levels)))
    
    for i, ca_val in enumerate(ca_levels):
        ca_data = summary[summary['CA (%)'] == ca_val].sort_values('Day')
        
        ax.plot(
            ca_data['Day'], ca_data['mean'],
            marker='o', markersize=4, linewidth=2, alpha=0.9,
            color=colors[i], label=f'{ca_val:.0f}% CA'
        )
        ax.fill_between(
            ca_data['Day'],
            ca_data['mean'] - ca_data['sem'],
            ca_data['mean'] + ca_data['sem'],
            color=colors[i], alpha=0.2
        )
    
    ax.set_xlabel('Day')
    ax.set_ylabel(f'{measure} (Mean ± SEM)')
    
    if title is None:
        title = f'Time × CA% Interaction: {measure}\n({sex_label} Only)'
    
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
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    Plot Time × Sex interaction from CA%-stratified mixed ANOVA.
    
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
    
    # Check if Time × Sex interaction is significant
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
        print(f"[WARNING] Sex × Day interaction not found in ANOVA table. Available sources: {list(anova_table['Source'])}")
        return None
    
    p_value = interaction_row.iloc[0][p_col]
    
    if p_value >= 0.05:
        print(f"[INFO] Time × Sex interaction not significant (p = {p_value:.4f}). Skipping plot.")
        return None
    
    print(f"\n[INFO] Plotting Time × Sex interaction (p = {p_value:.4f})")
    
    # Get data
    data_df = anova_results.get('data')
    if data_df is None or len(data_df) == 0:
        print("[WARNING] No data found in results")
        return None
    
    measure = anova_results.get('measure', 'Measure')
    ca_percent = anova_results.get('ca_percent', 'Unknown')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Compute mean and SEM for each Day × Sex combination
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
            marker=marker, markersize=4, linewidth=2, alpha=0.9,
            color=color, label=label
        )
        ax.fill_between(
            sex_data['Day'],
            sex_data['mean'] - sex_data['sem'],
            sex_data['mean'] + sex_data['sem'],
            color=color, alpha=0.2
        )
    
    ax.set_xlabel('Day')
    ax.set_ylabel(f'{measure} (Mean ± SEM)')
    
    if title is None:
        title = f'Time × Sex Interaction: {measure}\n({ca_percent}% CA Only)'
    
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
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    Plot CA% × Sex × Time three-way interaction from full mixed ANOVA.
    
    Creates a 1×2 panel plot showing Time × CA% trajectories separately for
    Males (left) and Females (right) with error bars (SEM).
    
    Only creates plot if the three-way interaction is significant OR if both
    two-way interactions (Time × CA% and Time × Sex) are significant.
    
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
    
    # Check if Day × Group interaction is significant (proxy for three-way)
    if mixed_anova is not None:
        p_col = 'p-unc' if 'p-unc' in mixed_anova.columns else 'p_unc'
        interaction_row = mixed_anova[mixed_anova['Source'] == 'Interaction']
        
        if len(interaction_row) > 0:
            p_value_time_group = interaction_row.iloc[0][p_col]
            if p_value_time_group < 0.05:
                print(f"\n[INFO] Time × Group interaction significant (p = {p_value_time_group:.4f})")
                plot_three_way = True
    
    # Check if CA% × Sex interaction is significant
    if between_anova is not None and not plot_three_way:
        p_col = 'p-unc' if 'p-unc' in between_anova.columns else 'p_unc'
        ca_sex_row = between_anova[between_anova['Source'] == 'CA (%) * Sex']
        
        if len(ca_sex_row) > 0:
            p_value_ca_sex = ca_sex_row.iloc[0][p_col]
            if p_value_ca_sex < 0.05:
                print(f"\n[INFO] CA% × Sex interaction significant (p = {p_value_ca_sex:.4f})")
                plot_three_way = True
    
    if not plot_three_way:
        print("[INFO] No significant three-way or relevant two-way interactions. Skipping plot.")
        return None
    
    print("\n[INFO] Creating three-way interaction plot (Time × CA% × Sex)")
    
    # Create 1×2 panel figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Get unique CA% levels
    ca_levels = sorted(data_df['CA (%)'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ca_levels)))
    
    for sex_idx, sex in enumerate(['M', 'F']):
        ax = axes[sex_idx]
        sex_data = data_df[data_df['Sex'] == sex]
        
        if len(sex_data) == 0:
            continue
        
        # Compute mean and SEM for each Day × CA% combination
        summary = sex_data.groupby(['Day', 'CA (%)'])[measure].agg(['mean', 'sem', 'count']).reset_index()
        
        # Plot trajectory for each CA% level
        for i, ca_val in enumerate(ca_levels):
            ca_data = summary[summary['CA (%)'] == ca_val].sort_values('Day')
            
            ax.plot(
                ca_data['Day'], ca_data['mean'],
                marker='o', markersize=4, linewidth=2, alpha=0.9,
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
            ax.set_ylabel(f'{measure} (Mean ± SEM)')
        
        sex_label = 'Males' if sex == 'M' else 'Females'
        ax.set_title(sex_label, pad=10)
        
        ax.grid(False)
        apply_common_plot_style(ax, remove_top_right=True, ticks_in=True, remove_x_margins=False, remove_y_margins=True)
        ax.legend(title='CA% Level', loc='best')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = f'Three-Way Interaction: Time × CA% × Sex\n{measure}'
    
    fig.suptitle(title, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    
    # 1. CA% × Sex interaction (between-subjects)
    if between_results is not None:
        print("\n[DEBUG] Checking CA% × Sex interaction...")
        measure = between_results.get('measure', 'measure')
        analysis_type = between_results.get('analysis_type', 'analysis')
        
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"interaction_CA_Sex_{measure}_{analysis_type}.png"
        
        fig = plot_interaction_ca_sex(between_results, save_path=save_path, show=show)
        if fig is not None:
            figures['CA_Sex'] = fig
            print(f"  ✓ CA% × Sex plot created")
        else:
            print(f"  ✗ CA% × Sex interaction not significant or missing data")
    
    # 2. Three-way interaction (full mixed ANOVA)
    if mixed_results is not None:
        print("\n[DEBUG] Checking three-way interaction...")
        measure = mixed_results.get('measure', 'measure')
        
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"interaction_Time_CA_Sex_{measure}.png"
        
        fig = plot_three_way_interaction(mixed_results, save_path=save_path, show=show)
        if fig is not None:
            figures['Time_CA_Sex'] = fig
            print(f"  ✓ Three-way interaction plot created")
        else:
            print(f"  ✗ Three-way interaction not significant or missing data")
    
    # 3. Time × CA% interactions (sex-stratified)
    if sex_stratified_results is not None:
        print("\n[DEBUG] Checking sex-stratified Time × CA% interactions...")
        for sex_key in ['males', 'females']:
            sex_results = sex_stratified_results.get(sex_key)
            if sex_results is not None:
                measure = sex_results.get('measure', 'measure')
                sex = sex_results.get('sex', sex_key[0].upper())
                sex_label = 'Males' if sex == 'M' else 'Females'
                
                print(f"  Checking {sex_label}...")
                save_path = None
                if save_dir is not None:
                    save_path = Path(save_dir) / f"interaction_Time_CA_{sex_label}_{measure}.png"
                
                fig = plot_interaction_time_ca(sex_results, save_path=save_path, show=show)
                if fig is not None:
                    figures[f'Time_CA_{sex_label}'] = fig
                    print(f"    ✓ Time × CA% ({sex_label}) plot created")
                else:
                    print(f"    ✗ Time × CA% ({sex_label}) interaction not significant or missing data")
            else:
                print(f"  ✗ {sex_key} results are None")
    
    # 4. Time × Sex interactions (CA%-stratified)
    if ca_stratified_results is not None:
        print("\n[DEBUG] Checking CA%-stratified Time × Sex interactions...")
        for ca_key, ca_results in ca_stratified_results.items():
            if isinstance(ca_results, dict):
                measure = ca_results.get('measure', 'measure')
                ca_percent = ca_results.get('ca_percent', ca_key)
                
                print(f"  Checking CA {ca_percent}%...")
                save_path = None
                if save_dir is not None:
                    save_path = Path(save_dir) / f"interaction_Time_Sex_CA{ca_percent}_{measure}.png"
                
                fig = plot_interaction_time_sex(ca_results, save_path=save_path, show=show)
                if fig is not None:
                    figures[f'Time_Sex_CA{ca_percent}'] = fig
                    print(f"    ✓ Time × Sex (CA {ca_percent}%) plot created")
                else:
                    print(f"    ✗ Time × Sex (CA {ca_percent}%) interaction not significant or missing data")
            else:
                print(f"  ✗ Results for CA {ca_key}% are not a dict: {type(ca_results)}")
    
    # Summary
    print("\n" + "="*80)
    print(f"INTERACTION PLOT SUMMARY: Created {len(figures)} plot(s)")
    print("="*80)
    
    if len(figures) > 0:
        for plot_name in figures.keys():
            print(f"  ✓ {plot_name}")
    else:
        print("  No significant interactions found to plot")
    
    return figures


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
        report_lines.append("BETWEEN-SUBJECTS ANOVA: CA% × SEX")
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
                    report_lines.append(f"   → Weight measures differ significantly between CA% concentrations")
                else:
                    report_lines.append(f"   → No significant difference between CA% concentrations")
            
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
                    report_lines.append(f"   → Males and females show significantly different weight measures")
                else:
                    report_lines.append(f"   → No significant sex difference")
            
            # Interaction effect
            if 'CA (%) * Sex' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                int_df = int_row['DF']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. CA% × Sex Interaction: {sig_str}")
                report_lines.append(f"   F({int_df:.0f},{resid_df:.0f}) = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   → The effect of CA% differs between males and females")
                    report_lines.append(f"   → (or equivalently: the sex difference depends on CA% level)")
                else:
                    report_lines.append(f"   → The effect of CA% is similar for both sexes")
        
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
        report_lines.append("MIXED ANOVA: CA% × TIME × SEX")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Measure: {mixed_results.get('measure', 'Unknown')}")
        report_lines.append(f"Number of subjects: {mixed_results.get('n_subjects', 'Unknown')}")
        report_lines.append(f"Number of time points: {mixed_results.get('n_days', 'Unknown')}")
        report_lines.append(f"Total observations: {mixed_results.get('n_observations', 'Unknown')}")
        report_lines.append("")
        
        # Mixed ANOVA table (Time within × Group between)
        mixed_aov = mixed_results.get('mixed_anova_table')
        if mixed_aov is not None:
            report_lines.append("Mixed ANOVA Table (Day within × Group between):")
            report_lines.append("-" * 80)
            report_lines.append(mixed_aov.to_string())
            report_lines.append("")
        
        # Between-subjects decomposition table
        between_aov = mixed_results.get('between_anova_table')
        if between_aov is not None:
            report_lines.append("Between-Subjects Effects (CA% × Sex):")
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
                    report_lines.append(f"   → HIGHLY SIGNIFICANT difference between CA% concentrations")
                elif ca_p < 0.05:
                    report_lines.append(f"   → Significant difference between CA% concentrations")
                else:
                    report_lines.append(f"   → No significant CA% effect")
            
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
                    report_lines.append(f"   → Males and females differ in weight measures")
                else:
                    report_lines.append(f"   → No significant sex difference")
            
            # CA% × Sex interaction
            if 'CA (%) * Sex' in between_aov['Source'].values:
                int_row = between_aov[between_aov['Source'] == 'CA (%) * Sex'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                int_df = int_row['DF']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. CA% × Sex Interaction: {sig_str}")
                report_lines.append(f"   F({int_df:.0f},{resid_df:.0f}) = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   → The CA% effect differs between sexes")
                else:
                    report_lines.append(f"   → Similar CA% effect for both sexes")
            
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
                    report_lines.append(f"   → HIGHLY SIGNIFICANT change over time")
                elif day_p < 0.05:
                    report_lines.append(f"   → Significant change over time")
                else:
                    report_lines.append(f"   → No significant change over time")
            
            # Day × Group interaction
            if mixed_aov is not None and 'Interaction' in mixed_aov['Source'].values:
                int_row = mixed_aov[mixed_aov['Source'] == 'Interaction'].iloc[0]
                int_p = int_row[time_p_col]
                int_F = int_row['F']
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n5. Day × Group Interaction: {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                
                if int_p < 0.05:
                    report_lines.append(f"   → The pattern of change over time differs between groups")
                    report_lines.append(f"   → Different trajectories for different CA%/Sex combinations")
                else:
                    report_lines.append(f"   → Similar time course for all groups")
        
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
        report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME × CA% (MALES ONLY)")
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
                    report_lines.append(f"   → Males show significant weight changes over time")
                else:
                    report_lines.append(f"   → No significant time effect in males")
            
            if 'CA (%)' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. CA% Effect (Males): {sig_str}")
                report_lines.append(f"   F = {ca_F:.3f}, p = {ca_p:.4f}")
                if ca_p < 0.05:
                    report_lines.append(f"   → CA% levels differ in males")
                else:
                    report_lines.append(f"   → No CA% difference in males")
            
            if 'CA (%) * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × CA% Interaction (Males): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   → The time course significantly differs between CA% levels in males (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   → Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   → Similar time course across CA% levels in males")
        
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
        report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME × CA% (FEMALES ONLY)")
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
                    report_lines.append(f"   → Females show significant weight changes over time")
                else:
                    report_lines.append(f"   → No significant time effect in females")
            
            if 'CA (%)' in aov['Source'].values:
                ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0]
                ca_p = ca_row[p_col]
                ca_F = ca_row['F']
                
                sig_str = "SIGNIFICANT" if ca_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. CA% Effect (Females): {sig_str}")
                report_lines.append(f"   F = {ca_F:.3f}, p = {ca_p:.4f}")
                if ca_p < 0.05:
                    report_lines.append(f"   → CA% levels differ in females")
                else:
                    report_lines.append(f"   → No CA% difference in females")
            
            if 'CA (%) * Day' in aov['Source'].values:
                int_row = aov[aov['Source'] == 'CA (%) * Day'].iloc[0]
                int_p = int_row[p_col]
                int_F = int_row['F']
                
                int_p_adj = min(1.0, int_p * 2)
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                sig_str_adj = "SIGNIFICANT" if int_p_adj < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × CA% Interaction (Females): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   → The time course significantly differs between CA% levels in females (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   → Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   → Similar time course across CA% levels in females")
        
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
        
        report_lines.append("Comparing Time × CA% interactions between males and females:")
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
        report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME × SEX (0% CA ONLY)")
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
                    report_lines.append(f"   → Weight changes significantly over time at 0% CA")
                else:
                    report_lines.append(f"   → No significant time effect at 0% CA")
            
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Effect (0% CA): {sig_str}")
                report_lines.append(f"   F = {sex_F:.3f}, p = {sex_p:.4f}")
                if sex_p < 0.05:
                    report_lines.append(f"   → Males and females differ at 0% CA")
                else:
                    report_lines.append(f"   → No sex difference at 0% CA")
            
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
                report_lines.append(f"\n3. Time × Sex Interaction (0% CA): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   → The time course significantly differs between sexes at 0% CA (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   → Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   → Similar time course for both sexes at 0% CA")
        
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
        report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME × SEX (2% CA ONLY)")
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
                    report_lines.append(f"   → Weight changes significantly over time at 2% CA")
                else:
                    report_lines.append(f"   → No significant time effect at 2% CA")
            
            if 'Sex' in aov['Source'].values:
                sex_row = aov[aov['Source'] == 'Sex'].iloc[0]
                sex_p = sex_row[p_col]
                sex_F = sex_row['F']
                
                sig_str = "SIGNIFICANT" if sex_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n2. Sex Effect (2% CA): {sig_str}")
                report_lines.append(f"   F = {sex_F:.3f}, p = {sex_p:.4f}")
                if sex_p < 0.05:
                    report_lines.append(f"   → Males and females differ at 2% CA")
                else:
                    report_lines.append(f"   → No sex difference at 2% CA")
            
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
                report_lines.append(f"\n3. Time × Sex Interaction (2% CA): {sig_str} (raw) / {sig_str_adj} (Bonferroni)")
                report_lines.append(f"   F = {int_F:.3f}, p (raw) = {int_p:.4f},  p (Bonf, k=2) = {int_p_adj:.4f}")
                if int_p_adj < 0.05:
                    report_lines.append(f"   → The time course significantly differs between sexes at 2% CA (survives correction)")
                elif int_p < 0.05:
                    report_lines.append(f"   → Nominally significant but does NOT survive Bonferroni correction")
                else:
                    report_lines.append(f"   → Similar time course for both sexes at 2% CA")
        
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
        
        report_lines.append("Comparing Time × Sex interactions between CA% levels:")
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
                report_lines.append(f"\n0% CA Time × Sex interaction:")
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
                report_lines.append(f"2% CA Time × Sex interaction:")
                report_lines.append(f"  p (raw) = {ca2_int_p:.4f} [{ca2_int_sig}]  |  p (Bonf, k=2) = {ca2_int_p_adj:.4f} [{ca2_int_sig_adj}]")
            
            # Interpretation using Bonferroni-corrected p
            if ca0_int_p is not None and ca2_int_p is not None:
                report_lines.append("")
                report_lines.append("Interpretation (Bonferroni-corrected, alpha = 0.025):")
                ca0_sig = (ca0_int_p_adj < 0.05) if ca0_int_p_adj is not None else False
                ca2_sig = (ca2_int_p_adj < 0.05) if ca2_int_p_adj is not None else False
                if ca0_sig and ca2_sig:
                    report_lines.append("→ Both CA% levels show significant Time × Sex interaction (after correction)")
                    report_lines.append("→ Sex differences in weight trajectory are consistent across CA% levels")
                elif ca0_sig and not ca2_sig:
                    report_lines.append("→ Time × Sex interaction survives correction at 0% CA but NOT at 2% CA")
                    report_lines.append("→ CA% may modulate sex differences in weight trajectory")
                elif not ca0_sig and ca2_sig:
                    report_lines.append("→ Time × Sex interaction survives correction at 2% CA but NOT at 0% CA")
                    report_lines.append("→ CA% may modulate sex differences in weight trajectory")
                else:
                    report_lines.append("→ Neither CA% level shows significant Time × Sex interaction after correction")
        
        report_lines.append("")
    
    # Daily between-subjects results
    if daily_results:
        report_lines.append("=" * 80)
        report_lines.append("DAILY BETWEEN-SUBJECTS ANOVA: CA% × SEX (EACH DAY ANALYZED SEPARATELY)")
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
            report_lines.append("CA% × Sex Interaction by Day:")
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
                    report_lines.append("- CA% × Sex interaction is CONSISTENTLY significant across most days")
                elif int_sig_count > 0:
                    report_lines.append(f"- CA% × Sex interaction is significant on some days ({int_sig_count}/{len(results_table)})")
                else:
                    report_lines.append("- CA% × Sex interaction is not significant on any individual day")
        
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


# =============================================================================
# WEEK-LEVEL MIXED ANOVA FUNCTIONS (CONSERVATIVE, BONFERRONI-CORRECTED)
# =============================================================================

def _add_week_column_across_cohorts(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Week' column (1-indexed) derived from 'Day'.
    Week 1 = Days 0–6, Week 2 = Days 7–13, etc.
    Requires 'Day' column (calls add_day_column_across_cohorts if absent).
    """
    df = combined_df.copy()
    if 'Day' not in df.columns:
        df = add_day_column_across_cohorts(df)
    df['Week'] = (df['Day'] // 7) + 1
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
    Week-level Mixed ANOVA: Week (within-subjects) × Group (between-subjects).

    This is the week-level equivalent of perform_cross_cohort_mixed_anova().
    'Group' is a combined CA%/Sex label (e.g. '0% / M', '2% / F').
    Using Week (1–5) instead of Day is more conservative: it averages out
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
    print("WEEK-LEVEL MIXED ANOVA: Week (WITHIN) × Group/CA%+Sex (BETWEEN)")
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

    # Average within Week per animal (one value per ID × Week)
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

    print("\nRunning mixed ANOVA (Week within × Group between)...")
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
    Week-level Mixed ANOVA: Week (within-subjects) × CA% (between-subjects),
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
    print(f"WEEK-LEVEL SEX-STRATIFIED MIXED ANOVA ({sex_label}): Week (WITHIN) × CA% (BETWEEN)")
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

    print(f"\nRunning mixed ANOVA (Week within × CA% between)...")
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
    Week-level Mixed ANOVA: Week (within-subjects) × Sex (between-subjects),
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
    print(f"WEEK-LEVEL CA%-STRATIFIED MIXED ANOVA ({ca_percent}% CA): Week (WITHIN) × Sex (BETWEEN)")
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

    print(f"\nRunning mixed ANOVA (Week within × Sex between)...")
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

    Uses Week (1–5) as the within-subjects factor rather than individual days.
    This matches the design in non_ramp_analysis.py and is more conservative:
    - Daily autocorrelation is averaged out within each animal-week
    - Within-subjects degrees of freedom are greatly reduced (4 vs ~34)
    - Only complete-record subjects are included

    Bonferroni corrections applied across stratified test families:
      - Sex-stratified (males + females, k=2): α_adj = 0.025
        Corrects the Week × CA% interaction p-values
      - CA%-stratified (0% + 2% CA, k=2): α_adj = 0.025
        Corrects the Week × Sex interaction p-values

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
    # Preamble (title, methodology, study design) — printed once per report
    # =========================================================================
    if include_preamble:
        h1("CROSS-COHORT STATISTICAL ANALYSIS REPORT — WEEK-LEVEL (CONSERVATIVE)")
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
        lines.append("     as the within-subjects factor, the F-test for Time × Sex or Time × CA%")
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

        lines.append("\n  Time factor: Week (1–5; one averaged value per animal per week)")
        lines.append("  Between-subjects factors: CA% and Sex")
        lines.append("  Incomplete subjects excluded (complete-subject design)")
        lines.append("")

    # -------------------------------------------------------------------------
    # Omnibus mixed ANOVA
    if mixed_results is not None:
        h1("OMNIBUS WEEK-LEVEL MIXED ANOVA: Week (WITHIN) × Group (BETWEEN)")
        lines.append("")
        lines.append(f"  Measure         : {mixed_results.get('measure', 'Unknown')}")
        lines.append(f"  Subjects        : {mixed_results.get('n_subjects', 'Unknown')}")
        lines.append(f"  Weeks included  : {mixed_results.get('weeks', 'Unknown')}")
        lines.append(f"  Observations    : {mixed_results.get('n_observations', 'Unknown')}")
        lines.append(f"  Note: Group = CA% × Sex combined label")
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
                    lines.append("     → Weight measure changes significantly across weeks" if p < 0.05
                                 else "     → No significant week-to-week change overall")
                elif row['Source'] == 'Group':
                    lines.append("     → Groups (CA%/Sex combinations) differ overall" if p < 0.05
                                 else "     → No significant difference between groups overall")
                elif 'Week' in row['Source'] and 'Group' in row['Source']:
                    lines.append("     → Trajectories differ between groups (different week-to-week patterns)" if p < 0.05
                                 else "     → All groups follow similar week-to-week patterns")
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
            lines.append(f"  Bonferroni family: {bonferroni_family_label} (k={bonferroni_k}, α_adj = {0.05/bonferroni_k:.3f})")
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
                    lines.append(f"     Bonferroni-corrected p = {p_adj:.4f} ({sig_adj})  ← USE THIS")
                    lines.append(f"     Decision: {sig_str_adj} after correction")
                    if src.startswith('CA') and 'Week' in src:
                        lines.append("     → CA% levels follow different weekly trajectories" if p_adj < 0.05
                                     else "     → Similar weekly trajectories across CA% levels")
                    elif src.startswith('Sex') and 'Week' in src:
                        lines.append("     → Males and females follow different weekly trajectories" if p_adj < 0.05
                                     else "     → Similar weekly trajectories for both sexes")
                else:
                    sig_str = "SIGNIFICANT" if p_raw < 0.05 else "NOT SIGNIFICANT"
                    lines.append(f"  {i}. {src}: {sig_str}")
                    lines.append(f"     F = {row['F']:.3f}, p = {p_raw:.4f} {sig_raw}")
                    if src == 'Week':
                        lines.append("     → Weight changes significantly across weeks" if p_raw < 0.05
                                     else "     → No significant week-to-week change")
                    elif src in ('Sex', 'CA (%)'):
                        lines.append("     → Groups differ overall (averaged across weeks)" if p_raw < 0.05
                                     else "     → No significant group difference overall")
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
    h1("SEX-STRATIFIED WEEK-LEVEL MIXED ANOVA: Week (WITHIN) × CA% (BETWEEN)")
    lines.append("")
    lines.append("These two models form a Bonferroni family (k=2).")
    lines.append("The Week × CA% interaction p-values are multiplied by 2 before")
    lines.append("comparing to α = 0.05 (equivalent to requiring p_raw < 0.025).")
    lines.append("")

    lines.append("━" * 80)
    lines.append("Males Only")
    lines.append("━" * 80)
    lines.append("")
    males_int_p_raw = _format_stratified_result(
        results_males, "Males", "Week", "CA (%)", "CA (%) * Week",
        bonferroni_k=2, bonferroni_family_label="Males + Females"
    )

    lines.append("━" * 80)
    lines.append("Females Only")
    lines.append("━" * 80)
    lines.append("")
    females_int_p_raw = _format_stratified_result(
        results_females, "Females", "Week", "CA (%)", "CA (%) * Week",
        bonferroni_k=2, bonferroni_family_label="Males + Females"
    )

    # Sex-stratified comparison summary
    if males_int_p_raw is not None and females_int_p_raw is not None:
        h2("Sex-Stratified Comparison Summary (Week × CA% Interaction)")
        lines.append("")
        m_adj = bonferroni(males_int_p_raw, 2)
        f_adj = bonferroni(females_int_p_raw, 2)
        lines.append(f"  Males   : raw p = {males_int_p_raw:.4f},  Bonferroni p = {m_adj:.4f}"
                     f"  {'← SIGNIFICANT' if m_adj < 0.05 else '← not significant'}")
        lines.append(f"  Females : raw p = {females_int_p_raw:.4f},  Bonferroni p = {f_adj:.4f}"
                     f"  {'← SIGNIFICANT' if f_adj < 0.05 else '← not significant'}")
        lines.append("")
        m_sig = m_adj < 0.05
        f_sig = f_adj < 0.05
        if m_sig and f_sig:
            lines.append("  → BOTH sexes: week × CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in both males and females")
        elif m_sig and not f_sig:
            lines.append("  → MALES ONLY: week × CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in males but not females")
        elif not m_sig and f_sig:
            lines.append("  → FEMALES ONLY: week × CA% interaction significant after correction")
            lines.append("    CA% shapes the weekly weight trajectory in females but not males")
        else:
            lines.append("  → NEITHER sex: week × CA% interaction significant after correction")
            lines.append("    No differential CA% effect on weekly trajectories in either sex")
        lines.append("")

    # -------------------------------------------------------------------------
    # CA%-stratified results (k=2 family)
    h1("CA%-STRATIFIED WEEK-LEVEL MIXED ANOVA: Week (WITHIN) × Sex (BETWEEN)")
    lines.append("")
    lines.append("These two models form a Bonferroni family (k=2).")
    lines.append("The Week × Sex interaction p-values are multiplied by 2 before")
    lines.append("comparing to α = 0.05 (equivalent to requiring p_raw < 0.025).")
    lines.append("")

    lines.append("━" * 80)
    lines.append("0% CA Only")
    lines.append("━" * 80)
    lines.append("")
    ca0_int_p_raw = _format_stratified_result(
        results_ca0, "0% CA", "Week", "Sex", "Sex * Week",
        bonferroni_k=2, bonferroni_family_label="0% CA + 2% CA"
    )

    lines.append("━" * 80)
    lines.append("2% CA Only")
    lines.append("━" * 80)
    lines.append("")
    ca2_int_p_raw = _format_stratified_result(
        results_ca2, "2% CA", "Week", "Sex", "Sex * Week",
        bonferroni_k=2, bonferroni_family_label="0% CA + 2% CA"
    )

    # CA%-stratified comparison summary
    if ca0_int_p_raw is not None and ca2_int_p_raw is not None:
        h2("CA%-Stratified Comparison Summary (Week × Sex Interaction)")
        lines.append("")
        c0_adj = bonferroni(ca0_int_p_raw, 2)
        c2_adj = bonferroni(ca2_int_p_raw, 2)
        lines.append(f"  0% CA : raw p = {ca0_int_p_raw:.4f},  Bonferroni p = {c0_adj:.4f}"
                     f"  {'← SIGNIFICANT' if c0_adj < 0.05 else '← not significant'}")
        lines.append(f"  2% CA : raw p = {ca2_int_p_raw:.4f},  Bonferroni p = {c2_adj:.4f}"
                     f"  {'← SIGNIFICANT' if c2_adj < 0.05 else '← not significant'}")
        lines.append("")
        c0_sig = c0_adj < 0.05
        c2_sig = c2_adj < 0.05
        if c0_sig and c2_sig:
            lines.append("  → BOTH CA% levels: week × sex interaction significant after correction")
            lines.append("    Males and females follow different weekly trajectories at both CA% levels")
        elif c0_sig and not c2_sig:
            lines.append("  → 0% CA ONLY: week × sex interaction significant after correction")
            lines.append("    Sex differences in weekly trajectory exist only in 0% CA animals")
        elif not c0_sig and c2_sig:
            lines.append("  → 2% CA ONLY: week × sex interaction significant after correction")
            lines.append("    Sex differences in weekly trajectory exist only in 2% CA animals")
        else:
            lines.append("  → NEITHER CA% level: week × sex interaction significant after correction")
            lines.append("    The day-level significance in 0% CA does not survive at the week level")
            lines.append("    after Bonferroni correction, suggesting it was driven by day-to-day noise")
        lines.append("")

    # -------------------------------------------------------------------------
    # Master summary table
    h1("SUMMARY TABLE — ALL KEY INTERACTION P-VALUES")
    lines.append("")
    lines.append("  All interaction terms from the stratified week-level models.")
    lines.append("  Bonferroni k=2 within each family; * = significant at α=0.05 after correction.")
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

    _summary_row("Week × CA% (Males)",      males_int_p_raw,   2)
    _summary_row("Week × CA% (Females)",    females_int_p_raw, 2)
    _summary_row("Week × Sex  (0% CA)",     ca0_int_p_raw,     2)
    _summary_row("Week × Sex  (2% CA)",     ca2_int_p_raw,     2)
    lines.append("")

    # -------------------------------------------------------------------------
    if include_footer:
        h1("END OF WEEK-LEVEL REPORT")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# 2-WAY MIXED ANOVA: CA% × WEEK (SEX COLLAPSED) WITH POST-HOC AND SPHERICITY
# =============================================================================

def perform_mixed_anova_ca_x_week(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    weeks: Optional[List[int]] = None,
) -> Dict:
    """
    2-Way Mixed ANOVA: CA% (between-subjects) × Week (within-subjects), sex collapsed.

    This collapses across sex to give the clearest view of the CA% × time interaction
    with maximum power. Includes:
      - Mauchly's sphericity test for the Week within-subjects factor
      - Greenhouse-Geisser epsilon and corrected p-values when sphericity is violated
      - Post-hoc pairwise comparisons (Bonferroni-adjusted paired t-tests) for Week
      - Simple-effects post-hoc when CA% × Week interaction is significant:
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
    print("2-WAY MIXED ANOVA: CA% (BETWEEN) × WEEK (WITHIN) — SEX COLLAPSED")
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

    # Average within Week per animal (one value per ID × Week, sex ignored)
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
    print("\nRunning 2-way mixed ANOVA (CA% between × Week within)...")
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
                print(f"  Greenhouse-Geisser ε = {sphericity_info['eps']:.4f}")
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
    Generate a report for the 2-way CA% × Week mixed ANOVA with post-hoc tests.

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
        h1("CROSS-COHORT — 2-WAY MIXED ANOVA: CA% × WEEK (SEX COLLAPSED)")
        lines.append("")
        h2("METHODOLOGICAL NOTES")
        lines.append("")
        lines.append("  Design: CA% (between-subjects) × Week (within-subjects).")
        lines.append("  Sex is collapsed (all animals in each CA% group combined).")
        lines.append("  One value per animal per week (daily observations averaged within each week).")
        lines.append("  Complete-subject design: animals missing any week are excluded.")
        lines.append("")
        lines.append("  Sphericity is tested using Mauchly's W test.")
        lines.append("  When sphericity is violated (p < 0.05), the Greenhouse-Geisser (GG)")
        lines.append("  corrected p-value and epsilon (ε) are reported for the Week effect.")
        lines.append("")
        lines.append("  Post-hoc tests:")
        lines.append("    • Week main effect: pairwise paired t-tests, Bonferroni-adjusted")
        lines.append("      (within-subjects; equivalent to Tukey HSD for the within factor)")
        lines.append("    • CA% × Week interaction — simple effects:")
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
        h2("Cell Means: CA% × Week")
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
                    row_vals.append(f"  {'–':>8}")
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
            lines.append("  Decision: Sphericity SATISFIED — use uncorrected p-values")
        else:
            lines.append("  Decision: Sphericity VIOLATED — Greenhouse-Geisser correction applied")
            lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
            lines.append(f"  (ε < 0.75 indicates substantial non-sphericity; ε = 1.0 = perfect sphericity)")
    else:
        lines.append("  Mauchly's W: not available (2-level within factor or missing)")
        if np.isfinite(eps):
            lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
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
                lines.append(f"     η²p = {np2:.4f}")
            # Narrative
            if src == 'CA (%)':
                lines.append("     → 0% and 2% CA groups differ overall in " + _measure_label
                              if p_use < 0.05 else
                              "     → No significant overall difference between CA% groups")
            elif src == 'Week':
                if not sph_ok:
                    lines.append(f"     (Sphericity violated — GG correction applied, ε = {eps:.4f})")
                lines.append("     → " + _measure_label + " changes significantly across weeks"
                              if p_use < 0.05 else
                              "     → No significant week-to-week change overall")
            elif src == 'CA (%) * Week':
                lines.append("     → CA% groups follow different weekly trajectories (interaction)"
                              if p_use < 0.05 else
                              "     → Both CA% groups follow similar weekly trajectories")
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
            t_str = f"{t_val:.3f}" if np.isfinite(t_val) else "–"
            df_str = f"{df_val:.0f}" if np.isfinite(df_val) else "–"
            p_raw_str = f"{p_raw_val:.4f}" if np.isfinite(p_raw_val) else "–"
            p_corr_str = f"{p_corr_val:.4f}" if np.isfinite(p_corr_val) else "–"
            lines.append(f"  {comp_label:<20} {t_str:>8} {df_str:>6} {p_raw_str:>10} {p_corr_str:>10} {sig:>5}")
        lines.append("")

    # -------------------------------------------------------------------------
    # Post-hoc: Interaction simple effects
    posthoc_ca_per_week = results.get('posthoc_ca_per_week', {})
    posthoc_week_per_ca = results.get('posthoc_week_per_ca', {})
    ca_levels_res = results.get('ca_levels', [])

    if posthoc_ca_per_week or posthoc_week_per_ca:
        h1("POST-HOC: CA% × WEEK INTERACTION SIMPLE EFFECTS")
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
            d_str = f"{d_val:.3f}" if np.isfinite(d_val) else "–"
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
            lines.append(f"  ── CA% = {ca_val}% ──")
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
                t_str = f"{t_val:.3f}" if np.isfinite(t_val) else "–"
                df_str = f"{df_val:.0f}" if np.isfinite(df_val) else "–"
                p_raw_str = f"{p_raw_val:.4f}" if np.isfinite(p_raw_val) else "–"
                p_c_str = f"{p_c_val:.4f}" if np.isfinite(p_c_val) else "–"
                lines.append(f"  {comp_label:<20} {t_str:>8} {df_str:>6} {p_raw_str:>10} {p_c_str:>10} {sig:>5}")
            lines.append("")

    # -------------------------------------------------------------------------
    if include_footer:
        h1("END OF CA% × WEEK REPORT")
        lines.append("")

    return "\n".join(lines)


# =============================================================================


# =============================================================================
# SLOPE ANALYSIS: COMPARING RATE OF WEIGHT CHANGE ACROSS COHORTS
# =============================================================================

def calculate_animal_slopes(
    cohort_dfs: Dict[str, pd.DataFrame],
    measure: str = "Total Change",
    time_unit: str = "Week"
) -> pd.DataFrame:
    """
    Calculate linear regression slope for each animal's weight change over time.
    
    For each animal, fits a linear regression: measure ~ time
    Returns slopes, intercepts, R², and metadata for all animals.
    
    Parameters:
        cohort_dfs: Dictionary mapping cohort labels to DataFrames
        measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
        time_unit: Time variable to use ('Week' or 'Day')
        
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
    
    # Combine cohorts
    print("\nStep 1: Combining cohort dataframes...")
    combined_df = combine_cohorts_for_analysis(cohort_dfs)
    combined_df = clean_cohort(combined_df)
    
    # Add time columns if needed
    if 'Day' not in combined_df.columns:
        combined_df = add_day_column_across_cohorts(combined_df)
    
    if time_unit == "Week" and 'Week' not in combined_df.columns:
        combined_df = _add_week_column_across_cohorts(combined_df)
    
    # Check required columns
    required_cols = ['ID', 'Sex', 'CA (%)', 'Cohort', time_unit, measure]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare data
    analysis_df = combined_df[required_cols].copy()
    analysis_df = analysis_df.dropna()
    
    print(f"\nAnalyzing: {measure} vs {time_unit}")
    print(f"  Total observations: {len(analysis_df)}")
    print(f"  Unique animals: {analysis_df['ID'].nunique()}")
    print(f"  CA% levels: {sorted(analysis_df['CA (%)'].unique())}")
    
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
    
    for ca_val in sorted(slopes_df['CA (%)'].unique()):
        cohort_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope']
        print(f"\n{ca_val}% CA (n={len(cohort_slopes)} animals):")
        print(f"  Mean Slope:   {cohort_slopes.mean():.4f} {measure} per {time_unit}")
        print(f"  Median Slope: {cohort_slopes.median():.4f} {measure} per {time_unit}")
        print(f"  SD:           {cohort_slopes.std():.4f}")
        print(f"  Range:        [{cohort_slopes.min():.4f}, {cohort_slopes.max():.4f}]")
        print(f"  Mean R²:      {slopes_df[slopes_df['CA (%)'] == ca_val]['R2'].mean():.4f}")
    
    return slopes_df


def compare_slopes_within_cohorts(slopes_df: pd.DataFrame) -> Dict:
    """
    Compare slopes within each cohort using descriptive statistics and variance tests.
    
    This analyzes the variability of slopes within each CA% group.
    
    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()
        
    Returns:
        Dictionary with within-cohort statistics and Levene's test results
    """
    from scipy import stats
    
    print("\n" + "="*80)
    print("WITHIN-COHORT SLOPE VARIABILITY ANALYSIS")
    print("="*80)
    
    results = {
        'cohort_stats': [],
        'levene_test': None
    }
    
    # Descriptive statistics by cohort
    for ca_val in sorted(slopes_df['CA (%)'].unique()):
        cohort_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope'].values
        
        cohort_stat = {
            'CA (%)': ca_val,
            'N': len(cohort_slopes),
            'Mean': cohort_slopes.mean(),
            'Median': np.median(cohort_slopes),
            'SD': cohort_slopes.std(),
            'SEM': cohort_slopes.std() / np.sqrt(len(cohort_slopes)),
            'Min': cohort_slopes.min(),
            'Max': cohort_slopes.max(),
            'IQR': np.percentile(cohort_slopes, 75) - np.percentile(cohort_slopes, 25),
            'CV': (cohort_slopes.std() / cohort_slopes.mean() * 100) if cohort_slopes.mean() != 0 else np.nan
        }
        
        results['cohort_stats'].append(cohort_stat)
        
        print(f"\n{ca_val}% CA Cohort (n={cohort_stat['N']}):")
        print(f"  Mean ± SEM:         {cohort_stat['Mean']:.4f} ± {cohort_stat['SEM']:.4f}")
        print(f"  Median (IQR):       {cohort_stat['Median']:.4f} ({cohort_stat['IQR']:.4f})")
        print(f"  SD:                 {cohort_stat['SD']:.4f}")
        print(f"  Coefficient of Var: {cohort_stat['CV']:.2f}%")
        print(f"  Range:              [{cohort_stat['Min']:.4f}, {cohort_stat['Max']:.4f}]")
    
    # Levene's test for equality of variances between cohorts
    ca_groups = sorted(slopes_df['CA (%)'].unique())
    
    if len(ca_groups) >= 2:
        print("\n" + "-"*80)
        print("LEVENE'S TEST: Equality of Variances Between Cohorts")
        print("-"*80)
        
        group_slopes = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values 
                       for ca in ca_groups]
        
        levene_stat, levene_p = stats.levene(*group_slopes)
        
        results['levene_test'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'ca_groups': ca_groups
        }
        
        print(f"Levene's statistic: W = {levene_stat:.4f}")
        print(f"P-value: p = {levene_p:.4f}")
        
        if levene_p < 0.05:
            print("Result: Variances are significantly different between cohorts (p < 0.05)")
        else:
            print("Result: No significant difference in variances between cohorts (p ≥ 0.05)")
    
    return results


def compare_slopes_between_cohorts(slopes_df: pd.DataFrame) -> Dict:
    """
    Statistically compare average slopes between cohorts.
    
    Performs:
    1. Independent samples t-test (or Welch's t-test if variances unequal)
    2. Mann-Whitney U test (non-parametric alternative)
    3. Effect size calculation (Cohen's d)
    
    Parameters:
        slopes_df: DataFrame from calculate_animal_slopes()
        
    Returns:
        Dictionary with test results and effect sizes
    """
    from scipy import stats
    
    print("\n" + "="*80)
    print("BETWEEN-COHORT SLOPE COMPARISON")
    print("="*80)
    
    ca_groups = sorted(slopes_df['CA (%)'].unique())
    
    if len(ca_groups) != 2:
        print(f"Warning: Expected 2 cohorts, found {len(ca_groups)}. Returning empty results.")
        return {}
    
    ca_0 = ca_groups[0]
    ca_1 = ca_groups[1]
    
    slopes_0 = slopes_df[slopes_df['CA (%)'] == ca_0]['Slope'].values
    slopes_1 = slopes_df[slopes_df['CA (%)'] == ca_1]['Slope'].values
    
    print(f"\nComparing: {ca_0}% CA (n={len(slopes_0)}) vs {ca_1}% CA (n={len(slopes_1)})")
    print(f"  {ca_0}% CA: Mean = {slopes_0.mean():.4f}, SD = {slopes_0.std():.4f}")
    print(f"  {ca_1}% CA: Mean = {slopes_1.mean():.4f}, SD = {slopes_1.std():.4f}")
    print(f"  Difference in means: {slopes_1.mean() - slopes_0.mean():.4f}")
    
    results = {
        'ca_groups': ca_groups,
        'n_0': len(slopes_0),
        'n_1': len(slopes_1),
        'mean_0': slopes_0.mean(),
        'mean_1': slopes_1.mean(),
        'sd_0': slopes_0.std(),
        'sd_1': slopes_1.std(),
        'mean_diff': slopes_1.mean() - slopes_0.mean()
    }
    
    # 1. Welch's t-test (unequal variances assumed)
    print("\n" + "-"*80)
    print("1. WELCH'S T-TEST (Independent samples, unequal variances)")
    print("-"*80)
    
    t_stat, t_p = stats.ttest_ind(slopes_0, slopes_1, equal_var=False)
    
    # Calculate Welch-Satterthwaite degrees of freedom
    s1_sq = slopes_0.var()
    s2_sq = slopes_1.var()
    n1 = len(slopes_0)
    n2 = len(slopes_1)
    df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
    
    print(f"t-statistic: t = {t_stat:.4f}")
    print(f"P-value: p = {t_p:.4f}")
    print(f"Degrees of freedom: df = {df:.2f} (Welch-Satterthwaite)")
    
    if t_p < 0.05:
        print(f"Result: Slopes are significantly different between cohorts (p < 0.05)")
    else:
        print(f"Result: No significant difference in slopes between cohorts (p ≥ 0.05)")
    
    results['t_test'] = {
        'statistic': t_stat,
        'p_value': t_p,
        'df': df
    }
    
    # 2. Mann-Whitney U test (non-parametric)
    print("\n" + "-"*80)
    print("2. MANN-WHITNEY U TEST (Non-parametric)")
    print("-"*80)
    
    u_stat, u_p = stats.mannwhitneyu(slopes_0, slopes_1, alternative='two-sided')
    
    print(f"U-statistic: U = {u_stat:.4f}")
    print(f"P-value: p = {u_p:.4f}")
    
    if u_p < 0.05:
        print(f"Result: Slopes are significantly different between cohorts (p < 0.05)")
    else:
        print(f"Result: No significant difference in slopes between cohorts (p ≥ 0.05)")
    
    results['mann_whitney'] = {
        'statistic': u_stat,
        'p_value': u_p
    }
    
    # 3. Effect size (Cohen's d)
    print("\n" + "-"*80)
    print("3. EFFECT SIZE (Cohen's d)")
    print("-"*80)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((len(slopes_0) - 1) * slopes_0.std()**2 + 
                         (len(slopes_1) - 1) * slopes_1.std()**2) / 
                        (len(slopes_0) + len(slopes_1) - 2))
    
    cohens_d = (slopes_1.mean() - slopes_0.mean()) / pooled_sd
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Interpretation: {interpretation} effect size")
    
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'pooled_sd': pooled_sd,
        'interpretation': interpretation
    }
    
    # 4. Confidence interval for mean difference (using Welch df)
    print("\n" + "-"*80)
    print("4. 95% CONFIDENCE INTERVAL FOR MEAN DIFFERENCE")
    print("-"*80)
    
    # Standard error of the difference (for unequal variances)
    se_diff = np.sqrt((slopes_0.var() / len(slopes_0)) + 
                      (slopes_1.var() / len(slopes_1)))
    
    # Critical value for 95% CI using Welch df
    t_crit = stats.t.ppf(0.975, df)
    
    ci_lower = results['mean_diff'] - t_crit * se_diff
    ci_upper = results['mean_diff'] + t_crit * se_diff
    
    print(f"Mean difference: {results['mean_diff']:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if ci_lower * ci_upper > 0:
        print("Result: CI does not include zero (significant difference)")
    else:
        print("Result: CI includes zero (no significant difference)")
    
    results['confidence_interval'] = {
        'mean_diff': results['mean_diff'],
        'se_diff': se_diff,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper
    }
    
    return results


def plot_slopes_comparison(
    slopes_df: pd.DataFrame,
    measure: str = "Total Change",
    time_unit: str = "Week",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
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
    
    import matplotlib.ticker as mticker
    
    ca_groups = sorted(slopes_df['CA (%)'].unique())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Subplot 1: Box plot with individual points
    ax1 = axes[0]
    
    positions = list(range(len(ca_groups)))
    box_data = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values for ca in ca_groups]
    
    bp = ax1.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Overlay individual points
    for i, ca_val in enumerate(ca_groups):
        cohort_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope'].values
        x_jitter = np.random.normal(i, 0.04, size=len(cohort_slopes))
        ax1.scatter(x_jitter, cohort_slopes, alpha=0.6, s=50, color='darkblue', edgecolors='black', linewidths=0.5)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'{ca}%' for ca in ca_groups])
    ax1.set_xlim(-0.7, len(ca_groups) - 1 + 0.7)
    ax1.set_xlabel('CA% Concentration')
    ax1.set_ylabel(f'Slope ({measure} per {time_unit})')
    ax1.set_title('Slope Distribution by Cohort')
    ax1.grid(False)
    ax1.tick_params(direction='in', length=6)
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Subplot 2: Bar plot with error bars (Mean ± SEM)
    ax2 = axes[1]
    
    means = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].mean() for ca in ca_groups]
    sems = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].std() / 
            np.sqrt(len(slopes_df[slopes_df['CA (%)'] == ca])) for ca in ca_groups]
    
    colors = ['dodgerblue', 'orangered']
    bars = ax2.bar(positions, means, yerr=sems, capsize=8, width=0.6, 
                   color=colors[:len(ca_groups)], alpha=0.7, 
                   edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'{ca}%' for ca in ca_groups])
    ax2.set_xlim(-0.7, len(ca_groups) - 1 + 0.7)
    ax2.set_xlabel('CA% Concentration')
    ax2.set_ylabel(f'Mean Slope ± SEM ({measure} per {time_unit})')
    ax2.set_title('Mean Slopes by Cohort')
    ax2.grid(False)
    ax2.tick_params(direction='in', length=6)
    ax2.set_ylim(bottom=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Subplot 3: Histogram/density comparison
    ax3 = axes[2]
    
    for i, ca_val in enumerate(ca_groups):
        cohort_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope'].values
        ax3.hist(cohort_slopes, bins=8, alpha=0.6, label=f'{ca_val}% CA (n={len(cohort_slopes)})',
                color=colors[i], edgecolor='black', linewidth=1)
    
    ax3.set_xlabel(f'Slope ({measure} per {time_unit})')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Slope Distribution Comparison')
    ax3.legend(loc='best')
    ax3.grid(False)
    ax3.tick_params(direction='in', length=6)
    ax3.set_xlim(left=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Overall title
    if title is None:
        title = f'Slope Analysis: {measure} vs {time_unit}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    lines.append(f"    {measure} = Slope × {time_unit} + Intercept")
    lines.append("\nThe slope represents the rate of change in {measure} per {time_unit}.")
    
    lines.append("\n" + "-" * 80)
    lines.append("Individual Animal Results:")
    lines.append("-" * 80)
    
    for ca_val in sorted(slopes_df['CA (%)'].unique()):
        cohort_data = slopes_df[slopes_df['CA (%)'] == ca_val].sort_values('Slope', ascending=False)
        lines.append(f"\n{ca_val}% CA Cohort (n={len(cohort_data)}):")
        
        for _, row in cohort_data.iterrows():
            lines.append(f"  {row['ID']:8s} ({row['Sex']}): "
                        f"Slope = {row['Slope']:7.4f}, "
                        f"R² = {row['R2']:.3f}, "
                        f"p = {row['P_value']:.4f}")
    
    # Within-cohort variability
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 2: WITHIN-COHORT VARIABILITY")
    lines.append("=" * 80)
    lines.append("\nThis section examines the variability of slopes within each cohort.")
    
    for cohort_stat in within_results['cohort_stats']:
        lines.append(f"\n{cohort_stat['CA (%)']}% CA Cohort:")
        lines.append(f"  Sample Size:           n = {cohort_stat['N']}")
        lines.append(f"  Mean Slope:            {cohort_stat['Mean']:.4f} {measure} per {time_unit}")
        lines.append(f"  Standard Error (SEM):  {cohort_stat['SEM']:.4f}")
        lines.append(f"  Standard Deviation:    {cohort_stat['SD']:.4f}")
        lines.append(f"  Median Slope:          {cohort_stat['Median']:.4f}")
        lines.append(f"  Interquartile Range:   {cohort_stat['IQR']:.4f}")
        lines.append(f"  Range:                 [{cohort_stat['Min']:.4f}, {cohort_stat['Max']:.4f}]")
        lines.append(f"  Coefficient of Var:    {cohort_stat['CV']:.2f}%")
    
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
            lines.append("  Result: No significant difference in variances (p ≥ 0.05)")
    
    # Between-cohort comparison
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 3: BETWEEN-COHORT COMPARISON")
    lines.append("=" * 80)
    lines.append("\nThis section compares the average slopes between the two cohorts.")
    
    if between_results:
        ca_0 = between_results['ca_groups'][0]
        ca_1 = between_results['ca_groups'][1]
        
        lines.append(f"\nCohort Comparison: {ca_0}% CA vs {ca_1}% CA")
        lines.append("-" * 80)
        lines.append(f"  {ca_0}% CA: Mean = {between_results['mean_0']:.4f}, SD = {between_results['sd_0']:.4f} (n={between_results['n_0']})")
        lines.append(f"  {ca_1}% CA: Mean = {between_results['mean_1']:.4f}, SD = {between_results['sd_1']:.4f} (n={between_results['n_1']})")
        lines.append(f"  Difference in means: {between_results['mean_diff']:.4f}")
        
        # T-test results
        lines.append("\n" + "-" * 80)
        lines.append("Welch's T-Test (unequal variances):")
        lines.append("-" * 80)
        t_test = between_results['t_test']
        lines.append(f"  t-statistic: t({t_test['df']:.2f}) = {t_test['statistic']:.4f}")
        lines.append(f"  P-value:     p = {t_test['p_value']:.4f}")
        
        if t_test['p_value'] < 0.001:
            sig_str = "p < 0.001 (highly significant)"
        elif t_test['p_value'] < 0.01:
            sig_str = "p < 0.01 (very significant)"
        elif t_test['p_value'] < 0.05:
            sig_str = "p < 0.05 (significant)"
        else:
            sig_str = "p ≥ 0.05 (not significant)"
        
        lines.append(f"  Result: {sig_str}")
        
        # Mann-Whitney U test
        lines.append("\n" + "-" * 80)
        lines.append("Mann-Whitney U Test (Non-parametric):")
        lines.append("-" * 80)
        mw = between_results['mann_whitney']
        lines.append(f"  U-statistic: U = {mw['statistic']:.4f}")
        lines.append(f"  P-value:     p = {mw['p_value']:.4f}")
        
        if mw['p_value'] < 0.05:
            lines.append(f"  Result: Significant difference (p < 0.05)")
        else:
            lines.append(f"  Result: No significant difference (p ≥ 0.05)")
        
        # Effect size
        lines.append("\n" + "-" * 80)
        lines.append("Effect Size (Cohen's d):")
        lines.append("-" * 80)
        es = between_results['effect_size']
        lines.append(f"  Cohen's d:   {es['cohens_d']:.4f}")
        lines.append(f"  Interpretation: {es['interpretation'].capitalize()} effect size")
        
        # Confidence interval
        lines.append("\n" + "-" * 80)
        lines.append("95% Confidence Interval for Mean Difference:")
        lines.append("-" * 80)
        ci = between_results['confidence_interval']
        lines.append(f"  Mean Difference: {ci['mean_diff']:.4f}")
        lines.append(f"  95% CI: [{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}]")
        
        if ci['ci_95_lower'] * ci['ci_95_upper'] > 0:
            lines.append("  Interpretation: CI does not include zero (significant difference)")
        else:
            lines.append("  Interpretation: CI includes zero (no significant difference)")
    else:
        # No between_results (less than 2 cohorts)
        lines.append("\n[WARNING] Between-cohort comparison could not be performed.")
        lines.append("Expected 2 cohorts but found a different number.")
        lines.append("Please ensure you have loaded exactly 2 cohort CSV files.")
    
    # Interpretation and conclusion
    lines.append("\n\n" + "=" * 80)
    lines.append("SECTION 4: INTERPRETATION AND CONCLUSIONS")
    lines.append("=" * 80)
    
    if between_results and 't_test' in between_results and between_results['t_test']['p_value'] < 0.05:
        lines.append(f"\nThe two cohorts show SIGNIFICANTLY DIFFERENT rates of weight change.")
        lines.append(f"The {between_results['ca_groups'][1]}% CA group has a mean slope that is")
        lines.append(f"{abs(between_results['mean_diff']):.4f} {measure} per {time_unit} {'higher' if between_results['mean_diff'] > 0 else 'lower'}")
        lines.append(f"than the {between_results['ca_groups'][0]}% CA group (p = {between_results['t_test']['p_value']:.4f}).")
    elif between_results and 't_test' in between_results:
        lines.append(f"\nThe two cohorts show NO SIGNIFICANT DIFFERENCE in rates of weight change.")
        lines.append(f"Both groups put on weight at approximately the same rate (p = {between_results['t_test']['p_value']:.4f}).")
    else:
        lines.append(f"\n[WARNING] Between-cohort comparison could not be completed.")
        lines.append("This analysis requires exactly 2 cohorts to be loaded.")
        lines.append("Please run the analysis again with 2 cohort CSV files selected.")
    
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
    output_dir: Optional[Path] = None
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
        
    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "="*80)
    print("COMPLETE SLOPE ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Calculate individual slopes
    slopes_df = calculate_animal_slopes(cohort_dfs, measure=measure, time_unit=time_unit)
    
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
        
        fig = plot_slopes_comparison(
            slopes_df, 
            measure=measure, 
            time_unit=time_unit,
            save_path=plot_path,
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
        print(f"  [OK] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
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
    2. Calculate linear slopes for each animal's Total Change vs Week
    3. Compare slopes within each cohort
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
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run interactive test to select and load two cohort files, then perform mixed ANOVA.
    """
    cohorts = test_load_two_cohorts()
    
    if cohorts and len(cohorts) >= 2:
        print("\n" + "="*80)
        print("RUNNING CROSS-COHORT MIXED ANOVA")
        print("="*80)
        
        print("\nYou can now run mixed ANOVA to compare cohorts.")
        print("Example analyses:")
        print("\n1. Total Change analysis:")
        print("   results_total = perform_cross_cohort_mixed_anova(")
        print("       cohorts, measure='Total Change')")
        
        print("\n2. Daily Change analysis:")
        print("   results_daily = perform_cross_cohort_mixed_anova(")
        print("       cohorts, measure='Daily Change')")
        
        # Run analyses automatically or interactively
        print("\nAnalysis Options:")
        print("  1. Mixed ANOVA (CA% × Time × Sex)")
        print("  2. Between-Subjects ANOVA at specific day")
        print("  3. Between-Subjects ANOVA averaged across all days")
        print("  4. Daily Between-Subjects ANOVA (all days held constant)")
        print("  5. All analyses + comprehensive report")
        print("  6. Generate weight plots (6 plots: Total/Daily Change by ID, Sex, CA%)")
        print("  7. All analyses + report + plots")
        print("  8. Generate INTERACTION PLOTS for significant interactions")
        print("  9. All analyses + report + all plots (weight + interactions)")
        print(" 10. Week-level conservative analyses + auto-save reports (all measures, Bonferroni-corrected)")
        print(" 11. 2-way CA% × Week mixed ANOVA (sex collapsed) + post-hoc Tukey + sphericity/GG")
        print(" 12. SLOPE ANALYSIS - Compare rate of weight change between cohorts")
        
        user_input = input("\nSelect option (1-12) or 'n' to skip: ")
        
        mixed_results = None
        between_final_results = None
        between_avg_results = None
        daily_results = None
        results_males = None
        results_females = None
        results_ca0 = None
        results_ca2 = None
        plot_figures = {}
        interaction_figures = {}
        
        if user_input == '1' or user_input == '5' or user_input == '7' or user_input == '8' or user_input == '9':
            # Run mixed ANOVA
            mixed_results = perform_cross_cohort_mixed_anova(
                cohorts,
                measure="Total Change",
                ss_type=3
            )
        
        if user_input == '2':
            # Run between-subjects ANOVA at final day
            day_input = input("\nEnter day number for analysis (e.g., 35 for final day): ")
            try:
                day_num = int(day_input)
                between_final_results = perform_between_subjects_anova(
                    cohorts,
                    measure="Total Change",
                    time_point=day_num,
                    ss_type=3
                )
            except ValueError:
                print("Invalid day number, skipping...")
        
        if user_input == '3' or user_input == '5' or user_input == '7' or user_input == '8' or user_input == '9':
            # Run between-subjects ANOVA averaged across days
            between_avg_results = perform_between_subjects_anova(
                cohorts,
                measure="Total Change",
                average_over_days=True,
                ss_type=3
            )
        
        if user_input == '4' or user_input == '5' or user_input == '7' or user_input == '9':
            # Run daily between-subjects ANOVA
            daily_results = perform_daily_between_subjects_anova(
                cohorts,
                measure="Total Change",
                ss_type=3
            )
        
        # Run stratified mixed ANOVAs for comprehensive analysis
        if user_input == '5' or user_input == '7' or user_input == '8' or user_input == '9':
            # Initialize results variables
            results_males = None
            results_females = None
            results_ca0 = None
            results_ca2 = None
            
            # Sex-stratified analyses
            print("\n" + "="*80)
            print("RUNNING SEX-STRATIFIED ANALYSES")
            print("="*80)
            
            results_males = perform_mixed_anova_sex_stratified(
                cohorts,
                sex="M",
                measure="Total Change"
            )
            
            results_females = perform_mixed_anova_sex_stratified(
                cohorts,
                sex="F",
                measure="Total Change"
            )
            
            # CA%-stratified analyses
            print("\n" + "="*80)
            print("RUNNING CA%-STRATIFIED ANALYSES")
            print("="*80)
            
            # Determine available CA% levels
            combined_temp = combine_cohorts_for_analysis(cohorts)
            ca_levels = sorted(combined_temp['CA (%)'].dropna().unique())
            print(f"  Available CA% levels: {ca_levels}")
            
            if 0.0 in ca_levels or 0 in ca_levels:
                results_ca0 = perform_mixed_anova_ca_stratified(
                    cohorts,
                    ca_percent=0,
                    measure="Total Change"
                )
            
            if 2.0 in ca_levels or 2 in ca_levels:
                results_ca2 = perform_mixed_anova_ca_stratified(
                    cohorts,
                    ca_percent=2,
                    measure="Total Change"
                )
        
        # Generate interaction plots for significant interactions
        if user_input == '8' or user_input == '9':
            if not HAS_MATPLOTLIB:
                print("\n[WARNING] matplotlib not available - cannot generate interaction plots")
            else:
                print("\n" + "="*80)
                print("GENERATING INTERACTION PLOTS")
                print("="*80)
                
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_dir = Path(f"interaction_plots_{timestamp}")
                plot_dir.mkdir(exist_ok=True)
                
                # Automatically generate all significant interaction plots
                interaction_figures = plot_all_significant_interactions(
                    between_results=between_avg_results,
                    mixed_results=mixed_results,
                    sex_stratified_results={'males': results_males, 'females': results_females},
                    ca_stratified_results={0.0: results_ca0, 2.0: results_ca2},
                    save_dir=plot_dir,
                    show=False  # Don't show immediately, will prompt later
                )
                
                if len(interaction_figures) > 0:
                    print(f"\n[OK] Generated {len(interaction_figures)} interaction plot(s)")
                    print(f"[OK] Plots saved to: {plot_dir}")
                    
                    # Show plots if user wants
                    show_plots = input("\nDisplay interaction plots now? (y/n): ")
                    if show_plots.lower() == 'y':
                        plt.show()  # Show all open figures
                    else:
                        # Close all figures
                        plt.close('all')
                else:
                    print("\n[INFO] No significant interactions found to plot")
                    # Only remove directory if it's empty
                    if plot_dir.exists() and not any(plot_dir.iterdir()):
                        plot_dir.rmdir()
        
        # Generate weight plots
        if user_input == '6' or user_input == '7' or user_input == '9':
            if not HAS_MATPLOTLIB:
                print("\n[WARNING] matplotlib not available - cannot generate plots")
            else:
                print("\n" + "="*80)
                print("GENERATING WEIGHT PLOTS")
                print("="*80)
                
                # Combine cohorts for plotting
                combined = combine_cohorts_for_analysis(cohorts)
                combined = clean_cohort(combined)
                if 'Day' not in combined.columns:
                    combined = add_day_column_across_cohorts(combined)
                
                # Create plots
                print("\n1. Total Change by Individual ID...")
                fig1 = plot_total_change_by_id(combined, show=False)
                if fig1:
                    plot_figures['total_change_by_id'] = fig1
                
                print("\n2. Daily Change by Individual ID...")
                fig2 = plot_daily_change_by_id(combined, show=False)
                if fig2:
                    plot_figures['daily_change_by_id'] = fig2
                
                print("\n3. Total Change by Sex (averaged)...")
                fig3 = plot_total_change_by_sex(combined, show=False)
                if fig3:
                    plot_figures['total_change_by_sex'] = fig3
                
                print("\n4. Daily Change by Sex (averaged)...")
                fig4 = plot_daily_change_by_sex(combined, show=False)
                if fig4:
                    plot_figures['daily_change_by_sex'] = fig4
                
                print("\n5. Total Change by CA% (averaged)...")
                fig5 = plot_total_change_by_ca(combined, show=False)
                if fig5:
                    plot_figures['total_change_by_ca'] = fig5
                
                print("\n6. Daily Change by CA% (averaged)...")
                fig6 = plot_daily_change_by_ca(combined, show=False)
                if fig6:
                    plot_figures['daily_change_by_ca'] = fig6
                
                print(f"\n[OK] Generated {len(plot_figures)} plots")
                
                # Offer to save plots
                save_plots = input("\nSave plots to files? (y/n): ")
                if save_plots.lower() == 'y':
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_dir = Path(f"cross_cohort_plots_{timestamp}")
                    plot_dir.mkdir(exist_ok=True)
                    
                    for plot_name, fig in plot_figures.items():
                        plot_path = plot_dir / f"{plot_name}.png"
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        print(f"  Saved: {plot_path}")
                    
                    print(f"\n[OK] All plots saved to: {plot_dir}")
                
                # Show plots
                show_plots = input("\nDisplay plots now? (y/n): ")
                if show_plots.lower() == 'y':
                    plt.show()
                else:
                    # Close all figures
                    for fig in plot_figures.values():
                        plt.close(fig)
        
        # Generate comprehensive report if any results exist
        if user_input == '5' or user_input == '7' or user_input == '9':
            if mixed_results or between_avg_results or daily_results:
                print("\n" + "="*80)
                print("GENERATING COMPREHENSIVE REPORT")
                print("="*80)
                
                report = generate_cross_cohort_report(
                    between_results=between_avg_results,
                    mixed_results=mixed_results,
                    daily_results=daily_results,
                    results_males=results_males,
                    results_females=results_females,
                    results_ca0=results_ca0,
                    results_ca2=results_ca2,
                    cohort_dfs=cohorts
                )
                
                print("\n" + report)
                
                # Offer to save report
                save_input = input("\nSave report to file? (y/n): ")
                if save_input.lower() == 'y':
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = Path(f"cross_cohort_analysis_report_{timestamp}.txt")
                    report_file.write_text(report, encoding='utf-8')
                    print(f"\n[OK] Report saved to: {report_file}")
            else:
                print("\n[WARNING] No analysis results to include in report")
        
        # --- Option 10: Week-level conservative analyses (all measures, Bonferroni-corrected) ---
        if user_input == '10':
            print("\n" + "="*80)
            print("RUNNING WEEK-LEVEL ANALYSES (ALL MEASURES, BONFERRONI-CORRECTED)")
            print("="*80)
            
            measures_to_run = ["Total Change", "Daily Change"]
            
            # Check which measures are available in the data
            combined_temp = combine_cohorts_for_analysis(cohorts)
            available_measures = [m for m in measures_to_run if m in combined_temp.columns]
            print(f"\n  Available measures: {available_measures}")
            
            ca_levels_avail = sorted(combined_temp['CA (%)'].dropna().unique())
            print(f"  Available CA% levels: {ca_levels_avail}")
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            all_measure_sections = []
            for i, measure in enumerate(available_measures):
                is_first = (i == 0)
                is_last  = (i == len(available_measures) - 1)

                print(f"\n{'='*80}")
                print(f"MEASURE: {measure}")
                print(f"{'='*80}")
                
                w_mixed = None
                w_males = None
                w_females = None
                w_ca0 = None
                w_ca2 = None
                
                try:
                    w_mixed = perform_cross_cohort_mixed_anova_weekly(cohorts, measure=measure)
                except Exception as e:
                    print(f"  [WARNING] Omnibus week-level ANOVA failed for {measure}: {e}")
                
                try:
                    w_males = perform_mixed_anova_sex_stratified_weekly(cohorts, sex="M", measure=measure)
                except Exception as e:
                    print(f"  [WARNING] Sex-stratified (Males) week ANOVA failed for {measure}: {e}")
                
                try:
                    w_females = perform_mixed_anova_sex_stratified_weekly(cohorts, sex="F", measure=measure)
                except Exception as e:
                    print(f"  [WARNING] Sex-stratified (Females) week ANOVA failed for {measure}: {e}")
                
                if 0.0 in ca_levels_avail or 0 in ca_levels_avail:
                    try:
                        w_ca0 = perform_mixed_anova_ca_stratified_weekly(cohorts, ca_percent=0, measure=measure)
                    except Exception as e:
                        print(f"  [WARNING] CA%-stratified (0%) week ANOVA failed for {measure}: {e}")
                
                if 2.0 in ca_levels_avail or 2 in ca_levels_avail:
                    try:
                        w_ca2 = perform_mixed_anova_ca_stratified_weekly(cohorts, ca_percent=2, measure=measure)
                    except Exception as e:
                        print(f"  [WARNING] CA%-stratified (2%) week ANOVA failed for {measure}: {e}")
                
                section = generate_cross_cohort_report_weekly(
                    mixed_results=w_mixed,
                    results_males=w_males,
                    results_females=w_females,
                    results_ca0=w_ca0,
                    results_ca2=w_ca2,
                    cohort_dfs=cohorts if is_first else None,
                    include_preamble=is_first,
                    include_footer=is_last,
                )
                all_measure_sections.append(section)
                print("\n" + section)
            
            # Combine all measures into one report and save a single file
            combined_report = "\n\n".join(all_measure_sections)
            report_file = Path(f"cross_cohort_WEEKLY_all_measures_{timestamp}.txt")
            report_file.write_text(combined_report, encoding='utf-8')
            print(f"\n{'='*80}")
            print(f"[OK] Combined weekly report saved to: {report_file}")
            print(f"{'='*80}")

        # --- Option 11: 2-way CA% × Week mixed ANOVA (sex collapsed) with post-hoc ---
        if user_input == '11':
            print("\n" + "="*80)
            print("2-WAY CA% × WEEK MIXED ANOVA (SEX COLLAPSED) — WITH POST-HOC & SPHERICITY")
            print("="*80)

            measures_to_run = ["Total Change", "Daily Change"]

            combined_temp = combine_cohorts_for_analysis(cohorts)
            available_measures = [m for m in measures_to_run if m in combined_temp.columns]
            print(f"\n  Available measures: {available_measures}")

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            all_sections = []
            for i, measure in enumerate(available_measures):
                is_first = (i == 0)
                is_last  = (i == len(available_measures) - 1)

                print(f"\n{'='*80}")
                print(f"MEASURE: {measure}")
                print(f"{'='*80}")

                ca_week_results = None
                try:
                    ca_week_results = perform_mixed_anova_ca_x_week(cohorts, measure=measure)
                except Exception as e:
                    print(f"  [WARNING] CA% × Week ANOVA failed for {measure}: {e}")

                section = generate_ca_x_week_report(
                    results=ca_week_results,
                    cohort_dfs=cohorts if is_first else None,
                    include_preamble=is_first,
                    include_footer=is_last,
                )
                all_sections.append(section)
                print("\n" + section)

            combined_report = "\n\n".join(all_sections)
            report_file = Path(f"cross_cohort_CA_x_WEEK_{timestamp}.txt")
            report_file.write_text(combined_report, encoding='utf-8')
            print(f"\n{'='*80}")
            print(f"[OK] CA% × Week report saved to: {report_file}")
            print(f"{'='*80}")

        # --- Option 12: SLOPE ANALYSIS ---
        if user_input == '12':
            print("\n" + "="*80)
            print("SLOPE ANALYSIS: RATE OF WEIGHT CHANGE COMPARISON")
            print("="*80)
            
            # Ask user for parameters
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
            slope_results = perform_complete_slope_analysis(
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

        # Summary of what was run
        if mixed_results or between_final_results or between_avg_results or daily_results or results_males or results_females or results_ca0 or results_ca2 or plot_figures or interaction_figures:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)
            
            if mixed_results:
                print(f"\n[✓] Mixed ANOVA (Time × CA% × Sex)")
                print(f"    Observations: {mixed_results.get('n_observations')}")
                print(f"    Subjects: {mixed_results.get('n_subjects')}")
                print(f"    Days: {mixed_results.get('n_days')}")
            
            if between_final_results:
                print(f"\n[✓] Between-Subjects ANOVA ({between_final_results.get('analysis_type')})")
                print(f"    Subjects: {between_final_results.get('n_subjects')}")
            
            if between_avg_results:
                print(f"\n[✓] Between-Subjects ANOVA ({between_avg_results.get('analysis_type')})")
                print(f"    Subjects: {between_avg_results.get('n_subjects')}")
            
            if daily_results:
                print(f"\n[✓] Daily Between-Subjects ANOVA (Each day held constant)")
                print(f"    Days analyzed: {daily_results.get('n_days')}")
            
            if results_males:
                print(f"\n[✓] Sex-Stratified Mixed ANOVA (Males)")
                print(f"    Subjects: {results_males.get('n_subjects')}")
                print(f"    Days: {results_males.get('n_days')}")
            
            if results_females:
                print(f"\n[✓] Sex-Stratified Mixed ANOVA (Females)")
                print(f"    Subjects: {results_females.get('n_subjects')}")
                print(f"    Days: {results_females.get('n_days')}")
            
            if results_ca0:
                print(f"\n[✓] CA%-Stratified Mixed ANOVA (0% CA)")
                print(f"    Subjects: {results_ca0.get('n_subjects')}")
                print(f"    Days: {results_ca0.get('n_days')}")
            
            if results_ca2:
                print(f"\n[✓] CA%-Stratified Mixed ANOVA (2% CA)")
                print(f"    Subjects: {results_ca2.get('n_subjects')}")
                print(f"    Days: {results_ca2.get('n_days')}")
            
            if plot_figures:
                print(f"\n[✓] Weight Plots Generated")
                print(f"    Number of plots: {len(plot_figures)}")
                print(f"    Plots: {', '.join(plot_figures.keys())}")
            
            if interaction_figures:
                print(f"\n[✓] Interaction Plots Generated")
                print(f"    Number of plots: {len(interaction_figures)}")
                print(f"    Plots: {', '.join(interaction_figures.keys())}")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\nYour cohorts are loaded and ready for analysis!")
        print("\nAvailable analyses:")
        print("  1. Mixed ANOVA (Time × CA% × Sex):")
        print("     results_mixed = perform_cross_cohort_mixed_anova(cohorts, measure='Total Change')")
        print("\n  2. Between-Subjects ANOVA at specific day:")
        print("     results_day35 = perform_between_subjects_anova(cohorts, measure='Total Change', time_point=35)")
        print("\n  3. Between-Subjects ANOVA averaged across days:")
        print("     results_avg = perform_between_subjects_anova(cohorts, measure='Total Change', average_over_days=True)")
        print("\n  4. Daily Between-Subjects ANOVA (all days held constant):")
        print("     results_daily = perform_daily_between_subjects_anova(cohorts, measure='Total Change')")
        print("\n  5. Sex-Stratified Mixed ANOVA (Time × CA%, holding Sex constant):")
        print("     results_males = perform_mixed_anova_sex_stratified(cohorts, sex='M', measure='Total Change')")
        print("     results_females = perform_mixed_anova_sex_stratified(cohorts, sex='F', measure='Total Change')")
        print("\n  6. CA%-Stratified Mixed ANOVA (Time × Sex, holding CA% constant):")
        print("     results_ca0 = perform_mixed_anova_ca_stratified(cohorts, ca_percent=0, measure='Total Change')")
        print("     results_ca2 = perform_mixed_anova_ca_stratified(cohorts, ca_percent=2, measure='Total Change')")
        print("\n  7. Generate INTERACTION PLOTS automatically:")
        print("     interaction_figs = plot_all_significant_interactions(")
        print("         between_results=results_avg, mixed_results=results_mixed,")
        print("         sex_stratified_results={'males': results_males, 'females': results_females},")
        print("         ca_stratified_results={0.0: results_ca0, 2.0: results_ca2},")
        print("         save_dir=Path('interaction_plots'), show=True)")
        print("\n  8. Generate comprehensive report:")
        print("     report = generate_cross_cohort_report(")
        print("         between_results=results_avg, mixed_results=results_mixed,")
        print("         daily_results=results_daily,")
        print("         results_males=results_males, results_females=results_females,")
        print("         results_ca0=results_ca0, results_ca2=results_ca2,")
        print("         cohort_dfs=cohorts)")
        print("     print(report)")
        print("\n  9. Generate weight plots:")
        print("     # First combine cohorts")
        print("     combined = combine_cohorts_for_analysis(cohorts)")
        print("     combined = clean_cohort(combined)")
        print("     combined = add_day_column_across_cohorts(combined)")
        print("     ")
        print("     # Then create plots")
        print("     fig1 = plot_total_change_by_id(combined)")
        print("     fig2 = plot_daily_change_by_id(combined)")
        print("     fig3 = plot_total_change_by_sex(combined)")
        print("     fig4 = plot_daily_change_by_sex(combined)")
        print("     fig5 = plot_total_change_by_ca(combined)")
        print("     fig6 = plot_daily_change_by_ca(combined)")
        print("\n 10. Week-level conservative analyses (all measures, Bonferroni-corrected):")
        print("     # Runs for each of: Total Change, Daily Change, Weight")
        print("     # Two separate report files: cross_cohort_WEEKLY_<measure>_<timestamp>.txt")
        print("     w_males   = perform_mixed_anova_sex_stratified_weekly(cohorts, sex='M', measure='Total Change')")
        print("     w_females = perform_mixed_anova_sex_stratified_weekly(cohorts, sex='F', measure='Total Change')")
        print("     w_ca0     = perform_mixed_anova_ca_stratified_weekly(cohorts, ca_percent=0, measure='Total Change')")
        print("     w_ca2     = perform_mixed_anova_ca_stratified_weekly(cohorts, ca_percent=2, measure='Total Change')")
        print("     report_w  = generate_cross_cohort_report_weekly(")
        print("         mixed_results=perform_cross_cohort_mixed_anova_weekly(cohorts),")
        print("         results_males=w_males, results_females=w_females,")
        print("         results_ca0=w_ca0, results_ca2=w_ca2, cohort_dfs=cohorts)")
        print("\n 11. SLOPE ANALYSIS - Compare rate of weight change between cohorts:")
        print("     # Calculate individual slopes and compare between cohorts")
        print("     slope_results = perform_complete_slope_analysis(")
        print("         cohorts, measure='Total Change', time_unit='Week',")
        print("         save_plot=True, save_report=True)")
        print("     # Or step by step:")
        print("     slopes_df = calculate_animal_slopes(cohorts, measure='Total Change', time_unit='Week')")
        print("     within = compare_slopes_within_cohorts(slopes_df)")
        print("     between = compare_slopes_between_cohorts(slopes_df)")
        print("     fig = plot_slopes_comparison(slopes_df, measure='Total Change', time_unit='Week')")
        print("="*80)
