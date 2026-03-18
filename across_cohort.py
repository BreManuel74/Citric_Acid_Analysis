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
from datetime import datetime

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
    
    # Compute day number per ID (Day 0 = first date for that animal)
    first_dates = df.groupby('ID')['Date'].transform('min')
    df['Day'] = (df['Date'] - first_dates).dt.days
    
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
            'Day * CA_percent': 'CA (%) * Day'
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
) -> None:
    """Apply common styling to plots: remove spines, set tick directions, adjust margins."""
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if ticks_in:
        ax.tick_params(direction='in')
    
    if remove_x_margins:
        ax.margins(x=0)
    
    if remove_y_margins:
        ax.margins(y=0)


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
        sex = sex_map.get(mid)
        color, marker = _sex_to_style(sex)
        
        ax.plot(
            s.index,
            s.values,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            color=color,
            label=f"{mid} ({sex or 'Unknown'})",
            alpha=0.7
        )

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Total Change (g)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if title is None:
        title = "Total Change by Animal ID"
    ax.set_title(title, fontsize=14, weight='bold')

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
    )

    # Legend placement
    if len(series_by_id) > 6:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
        sex = sex_map.get(mid)
        color, marker = _sex_to_style(sex)
        
        ax.plot(
            s.index,
            s.values,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            color=color,
            label=f"{mid} ({sex or 'Unknown'})",
            alpha=0.7
        )

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Daily Change (g)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if title is None:
        title = "Daily Change by Animal ID"
    ax.set_title(title, fontsize=14, weight='bold')

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
    )

    # Legend placement
    if len(series_by_id) > 6:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
        ax.errorbar(
            male_mean.index, male_mean.values, yerr=male_sem.values,
            marker='s', markersize=6, linewidth=2, capsize=4,
            color='green', label='Male', alpha=0.8
        )
    
    # Plot female data
    if not female_mean.empty:
        ax.errorbar(
            female_mean.index, female_mean.values, yerr=female_sem.values,
            marker='o', markersize=6, linewidth=2, capsize=4,
            color='purple', label='Female', alpha=0.8
        )
    
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Total Change (g, Mean ± SEM)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = "Total Change by Sex"
    ax.set_title(title, fontsize=14, weight='bold')
    
    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)
    
    ax.legend(title="Sex", loc="best", fontsize=11)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
        ax.errorbar(
            male_mean.index, male_mean.values, yerr=male_sem.values,
            marker='s', markersize=6, linewidth=2, capsize=4,
            color='green', label='Male', alpha=0.8
        )
    
    # Plot female data
    if not female_mean.empty:
        ax.errorbar(
            female_mean.index, female_mean.values, yerr=female_sem.values,
            marker='o', markersize=6, linewidth=2, capsize=4,
            color='purple', label='Female', alpha=0.8
        )
    
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Daily Change (g, Mean ± SEM)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = "Daily Change by Sex"
    ax.set_title(title, fontsize=14, weight='bold')
    
    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)
    
    ax.legend(title="Sex", loc="best", fontsize=11)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        
        if not mean.empty:
            ax.errorbar(
                mean.index, mean.values, yerr=sem.values,
                marker=marker, markersize=6, linewidth=2, capsize=4,
                color=color, label=f'{ca_val:.0f}% CA', alpha=0.8
            )
    
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Total Change (g, Mean ± SEM)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = "Total Change by CA%"
    ax.set_title(title, fontsize=14, weight='bold')
    
    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)
    
    ax.legend(title="CA%", loc="best", fontsize=11)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
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
    for ca_val in sorted(ca_groups.keys()):
        color, marker = _ca_to_style(ca_val)
        mean, sem = _compute_mean_sem(ca_groups[ca_val])
        
        if not mean.empty:
            ax.errorbar(
                mean.index, mean.values, yerr=sem.values,
                marker=marker, markersize=6, linewidth=2, capsize=4,
                color=color, label=f'{ca_val:.0f}% CA', alpha=0.8
            )
    
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Daily Change (g, Mean ± SEM)", fontsize=12)
    ax.grid(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    if title is None:
        title = "Daily Change by CA%"
    ax.set_title(title, fontsize=14, weight='bold')
    
    apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
                            remove_x_margins=True, remove_y_margins=True, ticks_in=True)
    
    ax.legend(title="CA%", loc="best", fontsize=11)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


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
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × CA% Interaction (Males): {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                if int_p < 0.05:
                    report_lines.append(f"   → The time course differs between CA% levels in males")
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
                
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × CA% Interaction (Females): {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                if int_p < 0.05:
                    report_lines.append(f"   → The time course differs between CA% levels in females")
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
            if 'CA (%) * Day' in males_aov['Source'].values:
                males_int_p = males_aov[males_aov['Source'] == 'CA (%) * Day'].iloc[0][p_col]
                males_int_sig = "SIGNIFICANT" if males_int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\nMales Time × CA% interaction: {males_int_sig} (p = {males_int_p:.4f})")
            
            # Females interaction
            females_int_p = None
            if 'CA (%) * Day' in females_aov['Source'].values:
                females_int_p = females_aov[females_aov['Source'] == 'CA (%) * Day'].iloc[0][p_col]
                females_int_sig = "SIGNIFICANT" if females_int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"Females Time × CA% interaction: {females_int_sig} (p = {females_int_p:.4f})")
            
            # Interpretation
            if males_int_p is not None and females_int_p is not None:
                report_lines.append("")
                if males_int_p < 0.05 and females_int_p < 0.05:
                    report_lines.append("→ Both sexes show significant Time × CA% interaction")
                    report_lines.append("→ CA% effects on weight trajectory differ by sex")
                elif males_int_p < 0.05 and females_int_p >= 0.05:
                    report_lines.append("→ Time × CA% interaction is SIGNIFICANT in males but NOT in females")
                    report_lines.append("→ CA% effects on weight trajectory are sex-specific")
                elif males_int_p >= 0.05 and females_int_p < 0.05:
                    report_lines.append("→ Time × CA% interaction is SIGNIFICANT in females but NOT in males")
                    report_lines.append("→ CA% effects on weight trajectory are sex-specific")
                else:
                    report_lines.append("→ Neither sex shows significant Time × CA% interaction")
        
        report_lines.append("")
    
    # CA%-stratified mixed ANOVA results
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
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × Sex Interaction (0% CA): {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                if int_p < 0.05:
                    report_lines.append(f"   → The time course differs between sexes at 0% CA")
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
                sig_str = "SIGNIFICANT" if int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n3. Time × Sex Interaction (2% CA): {sig_str}")
                report_lines.append(f"   F = {int_F:.3f}, p = {int_p:.4f}")
                if int_p < 0.05:
                    report_lines.append(f"   → The time course differs between sexes at 2% CA")
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
                ca0_int_sig = "SIGNIFICANT" if ca0_int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"\n0% CA Time × Sex interaction: {ca0_int_sig} (p = {ca0_int_p:.4f})")
            
            # 2% CA interaction
            ca2_int_p = None
            if 'Sex * Day' in ca2_aov['Source'].values:
                ca2_int_p = ca2_aov[ca2_aov['Source'] == 'Sex * Day'].iloc[0][p_col]
            elif 'Day * Sex' in ca2_aov['Source'].values:
                ca2_int_p = ca2_aov[ca2_aov['Source'] == 'Day * Sex'].iloc[0][p_col]
            
            if ca2_int_p is not None:
                ca2_int_sig = "SIGNIFICANT" if ca2_int_p < 0.05 else "NOT SIGNIFICANT"
                report_lines.append(f"2% CA Time × Sex interaction: {ca2_int_sig} (p = {ca2_int_p:.4f})")
            
            # Interpretation
            if ca0_int_p is not None and ca2_int_p is not None:
                report_lines.append("")
                if ca0_int_p < 0.05 and ca2_int_p < 0.05:
                    report_lines.append("→ Both CA% levels show significant Time × Sex interaction")
                    report_lines.append("→ Sex differences in weight trajectory are consistent across CA% levels")
                elif ca0_int_p < 0.05 and ca2_int_p >= 0.05:
                    report_lines.append("→ Time × Sex interaction is SIGNIFICANT at 0% CA but NOT at 2% CA")
                    report_lines.append("→ CA% modulates sex differences in weight trajectory")
                elif ca0_int_p >= 0.05 and ca2_int_p < 0.05:
                    report_lines.append("→ Time × Sex interaction is SIGNIFICANT at 2% CA but NOT at 0% CA")
                    report_lines.append("→ CA% modulates sex differences in weight trajectory")
                else:
                    report_lines.append("→ Neither CA% level shows significant Time × Sex interaction")
        
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
        
        user_input = input("\nSelect option (1-7) or 'n' to skip: ")
        
        mixed_results = None
        between_final_results = None
        between_avg_results = None
        daily_results = None
        results_males = None
        results_females = None
        results_ca0 = None
        results_ca2 = None
        plot_figures = {}
        
        if user_input == '1' or user_input == '5' or user_input == '7':
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
        
        if user_input == '3' or user_input == '5' or user_input == '7':
            # Run between-subjects ANOVA averaged across days
            between_avg_results = perform_between_subjects_anova(
                cohorts,
                measure="Total Change",
                average_over_days=True,
                ss_type=3
            )
        
        if user_input == '4' or user_input == '5' or user_input == '7':
            # Run daily between-subjects ANOVA
            daily_results = perform_daily_between_subjects_anova(
                cohorts,
                measure="Total Change",
                ss_type=3
            )
        
        # Run stratified mixed ANOVAs for comprehensive analysis
        if user_input == '5' or user_input == '7':
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
        
        # Generate plots
        if user_input == '6' or user_input == '7':
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
        if user_input == '5' or user_input == '7':
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
        
        # Summary of what was run
        if mixed_results or between_final_results or between_avg_results or daily_results or results_males or results_females or results_ca0 or results_ca2 or plot_figures:
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
        print("\n  7. Generate comprehensive report:")
        print("     report = generate_cross_cohort_report(")
        print("         between_results=results_avg, mixed_results=results_mixed,")
        print("         daily_results=results_daily,")
        print("         results_males=results_males, results_females=results_females,")
        print("         results_ca0=results_ca0, results_ca2=results_ca2,")
        print("         cohort_dfs=cohorts)")
        print("     print(report)")
        print("\n  8. Generate weight plots:")
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
        print("="*80)
