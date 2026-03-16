"""
CAH Cohort Weight Analysis
Analyzes weight changes across CA% concentrations and sex for the CAH cohort.
Adapted from ramp_analysis.py structure.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re

# Try to import pingouin for mixed ANOVA (repeated measures)
try:
	import pingouin as pg
	HAS_PINGOUIN = True
except ImportError:
	HAS_PINGOUIN = False
	print("Warning: pingouin not installed. Mixed ANOVA will not be available.")
	print("Install with: pip install pingouin")

# Try to import statsmodels for post-hoc tests
try:
	from statsmodels.stats.multicomp import pairwise_tukeyhsd
	HAS_STATSMODELS = True
except ImportError:
	HAS_STATSMODELS = False
	print("Warning: statsmodels not installed. Tukey HSD post-hoc tests will not be available.")
	print("Install with: pip install statsmodels")


# ==============================================================================
# PLOTTING HELPER FUNCTIONS (adapted from ramp_analysis.py)
# ==============================================================================

def _get_id_sex_map(df: pd.DataFrame) -> dict:
	"""
	Build a mapping from ID -> Sex ("M" or "F") using the first non-null value per ID.
	If Sex isn't available, returns an empty mapping.
	"""
	cdf = clean_cah_dataframe(df)
	if "ID" not in cdf.columns or "Sex" not in cdf.columns:
		return {}

	def _norm_sex(x: pd.Series) -> Optional[str]:
		vals = x.dropna().astype(str).str.strip().str.upper()
		if vals.empty:
			return None
		v = vals.iloc[0]
		if v.startswith("M"):
			return "M"
		if v.startswith("F"):
			return "F"
		return None

	sex_map = cdf.groupby("ID")["Sex"].apply(_norm_sex).to_dict()
	return {str(k): v for k, v in sex_map.items()}


def _sex_to_style(sex: Optional[str]) -> Tuple[str, str]:
	"""Return (color, marker) based on sex: M=green/square, F=purple/circle, unknown=gray/triangle."""
	if sex == "M":
		return ("green", "s")
	if sex == "F":
		return ("purple", "o")
	return ("gray", "^")


def _ca_to_style(ca_pct: Optional[int]) -> Tuple[str, str]:
	"""Return (color, marker) based on CA%: 0=dodgerblue/triangle-up, 2=orangered/triangle-down."""
	if ca_pct == 0:
		return ("dodgerblue", "x")
	if ca_pct == 2:
		return ("orangered", "v")
	return ("gray", "d")  # diamond for unknown


def build_daily_change_series_by_id(df: pd.DataFrame, index: str = "day") -> dict:
	"""
	For each ID, return a pandas Series of 'Daily Change' indexed by Day number.
	
	Parameters:
		df: DataFrame with ID, Day, Daily Change columns
		index: "day" or "date" (CAH cohort uses day numbers)
		
	Returns:
		dict[str, pd.Series]: Mapping from ID to Series of daily changes
	"""
	required = {"ID", "Day", "Daily Change"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

	cdf = clean_cah_dataframe(df)
	if "Day" not in cdf.columns:
		cdf = add_day_number_column(cdf)

	series_by_id: dict = {}
	for gid, g in cdf.groupby("ID", dropna=True):
		g = g.dropna(subset=["Daily Change"])
		if g.empty:
			continue
		
		g["Day"] = g["Day"].astype("Int64")
		ser = g.set_index("Day")["Daily Change"].sort_index()
		s = ser.groupby(level=0).last()
		s.name = str(gid)
		series_by_id[str(gid)] = s

	return series_by_id


def build_total_change_series_by_id(df: pd.DataFrame, index: str = "day") -> dict:
	"""
	For each ID, return a pandas Series of 'Total Change' indexed by Day number.
	
	Parameters:
		df: DataFrame with ID, Day, Total Change columns
		index: "day" or "date" (CAH cohort uses day numbers)
		
	Returns:
		dict[str, pd.Series]: Mapping from ID to Series of total changes
	"""
	required = {"ID", "Day", "Total Change"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

	cdf = clean_cah_dataframe(df)
	if "Day" not in cdf.columns:
		cdf = add_day_number_column(cdf)

	series_by_id: dict = {}
	for gid, g in cdf.groupby("ID", dropna=True):
		g = g.dropna(subset=["Total Change"])
		if g.empty:
			continue
		
		g["Day"] = g["Day"].astype("Int64")
		ser = g.set_index("Day")["Total Change"].sort_index()
		s = ser.groupby(level=0).last()
		s.name = str(gid)
		series_by_id[str(gid)] = s

	return series_by_id


def apply_common_plot_style(
	ax: plt.Axes,
	start_x_at_zero: bool = False,
	remove_top_right: bool = True,
	remove_x_margins: bool = True,
	remove_y_margins: bool = True,
	ticks_in: bool = True,
) -> None:
	"""
	Apply common styling to plots: remove spines, set tick directions, adjust margins.
	"""
	if remove_top_right:
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
	
	if ticks_in:
		ax.tick_params(axis='both', direction='in', which='both')
	
	if remove_x_margins:
		ax.margins(x=0)
	
	if remove_y_margins:
		ax.margins(y=0)


# ==============================================================================
# DATA LOADING AND CLEANING
# ==============================================================================

def load_cah_data(csv_path: Union[str, Path]) -> pd.DataFrame:
	"""
	Load the CAH master CSV file.
	
	Parameters:
		csv_path: Path to master_data_CAH.csv
		
	Returns:
		Raw DataFrame
	"""
	csv_path = Path(csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")
	
	df = pd.read_csv(csv_path)
	print(f"\nLoaded: {csv_path.name}")
	print(f"  Rows: {len(df):,}")
	print(f"  Columns: {list(df.columns)}")
	
	return df


def clean_cah_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Clean and standardize column types for CAH cohort data.
	
	Processing steps:
	- Parse dates in 'DOB' and 'Date'
	- Coerce 'Daily Change', 'Total Change', 'Weight' to numeric
	- Extract CA (%) from 'Condition' column (e.g., '0%' -> 0, '2%' -> 2)
	- Normalize Sex to 'M' or 'F'
	- Drop rows with missing ID or Date
	- Sort by ID then Date
	
	Parameters:
		df: Raw DataFrame from CSV
		
	Returns:
		Cleaned and standardized DataFrame
	"""
	df = df.copy()
	
	# Standardize column names (case-insensitive matching)
	rename_map = {}
	for col in list(df.columns):
		canon = col.strip()
		if canon.lower() == "daily change":
			rename_map[col] = "Daily Change"
		elif canon.lower() == "total change":
			rename_map[col] = "Total Change"
		elif canon.lower() == "date":
			rename_map[col] = "Date"
		elif canon.lower() == "dob":
			rename_map[col] = "DOB"
		elif canon.lower() == "weight":
			rename_map[col] = "Weight"
		elif canon.lower() == "id":
			rename_map[col] = "ID"
		elif canon.lower() == "sex":
			rename_map[col] = "Sex"
		elif canon.lower() == "strain":
			rename_map[col] = "Strain"
		elif canon.lower() == "condition":
			rename_map[col] = "Condition"
		elif canon.lower() == "notes":
			rename_map[col] = "Notes"
	
	if rename_map:
		df = df.rename(columns=rename_map)
	
	# Parse dates
	for dcol in ["DOB", "Date"]:
		if dcol in df.columns:
			df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
	
	# Extract CA (%) from Condition column
	# Examples: '0%' -> 0, '2%' -> 2
	if "Condition" in df.columns:
		df["CA (%)"] = (
			df["Condition"]
			.astype(str)
			.str.strip()
			.str.replace("%", "", regex=False)
		)
		df["CA (%)"] = pd.to_numeric(df["CA (%)"], errors="coerce")
	
	# Numeric coercions for weight measures
	for ncol in ["Daily Change", "Total Change", "Weight"]:
		if ncol in df.columns:
			df[ncol] = pd.to_numeric(df[ncol], errors="coerce")
	
	# Normalize Sex to 'M' or 'F'
	if "Sex" in df.columns:
		df["Sex"] = (
			df["Sex"]
			.astype(str)
			.str.strip()
			.str.upper()
			.map(lambda x: "M" if x.startswith("M") else ("F" if x.startswith("F") else None))
		)
	
	# Remove rows without essential keys
	if "ID" in df.columns and "Date" in df.columns:
		df = df.dropna(subset=["ID", "Date"]).copy()
	
	# Sort for stable downstream grouping
	sort_cols = [c for c in ["ID", "Date"] if c in df.columns]
	if sort_cols:
		df = df.sort_values(sort_cols).reset_index(drop=True)
	
	return df


def add_day_number_column(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add a per-ID 'Day' column where the first date for each ID is day 0.
	
	Parameters:
		df: DataFrame with 'ID' and 'Date' columns (Date as datetime)
		
	Returns:
		DataFrame with added 'Day' column
	"""
	if not {"ID", "Date"}.issubset(df.columns):
		return df
	
	df = df.copy()
	
	# Ensure sorted by ID and Date
	df = df.sort_values(["ID", "Date"]).reset_index(drop=True)
	
	# Compute day number per ID (Day 0 = first date for that animal)
	first_dates = df.groupby("ID")["Date"].transform("min")
	df["Day"] = (df["Date"] - first_dates).dt.days
	
	return df


def summarize_dataframe(df: pd.DataFrame) -> None:
	"""
	Print a comprehensive summary of the DataFrame structure and data.
	
	Parameters:
		df: Cleaned DataFrame
	"""
	print("\n" + "="*80)
	print("DATAFRAME SUMMARY")
	print("="*80)
	
	print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
	print(f"\nColumns: {list(df.columns)}")
	
	# Key grouping variables
	if "ID" in df.columns:
		print(f"\nUnique IDs (animals): {df['ID'].nunique()}")
		print(f"  IDs: {sorted(df['ID'].unique())}")
	
	if "Sex" in df.columns:
		print(f"\nSex distribution:")
		sex_counts = df.groupby("Sex")["ID"].nunique()
		for sex, count in sex_counts.items():
			print(f"  {sex}: {count} animals")
	
	if "Strain" in df.columns:
		print(f"\nStrain distribution:")
		strain_counts = df.groupby("Strain")["ID"].nunique()
		for strain, count in strain_counts.items():
			print(f"  {strain}: {count} animals")
	
	if "CA (%)" in df.columns:
		print(f"\nCA (%) levels: {sorted(df['CA (%)'].dropna().unique())}")
		ca_counts = df.groupby("CA (%)")["ID"].nunique()
		for ca, count in ca_counts.items():
			print(f"  {ca}%: {count} animals")
	
	if "Condition" in df.columns:
		print(f"\nCondition distribution:")
		cond_counts = df.groupby("Condition")["ID"].nunique()
		for cond, count in cond_counts.items():
			print(f"  {cond}: {count} animals")
	
	# Check for Sex × CA% design
	if {"Sex", "CA (%)"}.issubset(df.columns):
		print(f"\nSex × CA (%) design:")
		design = df.groupby(["Sex", "CA (%)"])["ID"].nunique().reset_index()
		design.columns = ["Sex", "CA (%)", "n_animals"]
		print(design.to_string(index=False))
	
	# Date range
	if "Date" in df.columns:
		print(f"\nDate range:")
		print(f"  First: {df['Date'].min().strftime('%Y-%m-%d')}" if pd.notna(df['Date'].min()) else "  First: N/A")
		print(f"  Last: {df['Date'].max().strftime('%Y-%m-%d')}" if pd.notna(df['Date'].max()) else "  Last: N/A")
		print(f"  Days: {(df['Date'].max() - df['Date'].min()).days}" if pd.notna(df['Date'].max()) and pd.notna(df['Date'].min()) else "  Days: N/A")
	
	# Day range (if available)
	if "Day" in df.columns:
		print(f"\nDay range (per-animal day numbers):")
		print(f"  Min: {df['Day'].min()}")
		print(f"  Max: {df['Day'].max()}")
		
		# Check for Day < 0 (baseline measurements)
		baseline_count = (df['Day'] < 0).sum()
		if baseline_count > 0:
			print(f"  Baseline measurements (Day < 0): {baseline_count} rows")
	
	# Weight measures summary
	print(f"\nWeight measures:")
	for col in ["Weight", "Daily Change", "Total Change"]:
		if col in df.columns:
			non_null = df[col].notna().sum()
			print(f"  {col}: {non_null} non-null values")
			if non_null > 0:
				print(f"    Mean: {df[col].mean():.2f}, SD: {df[col].std():.2f}")
				print(f"    Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
	
	# Check experimental design: Between-subjects or within-subjects for CA%?
	if {"ID", "CA (%)"}.issubset(df.columns):
		subjects_per_ca = df.groupby('ID')['CA (%)'].nunique()
		total_ca_levels = df['CA (%)'].nunique()
		
		# If all animals have only 1 CA% level, it's between-subjects
		if all(subjects_per_ca == 1):
			print(f"\n[OK] Experimental Design: CA% is BETWEEN-SUBJECTS")
			print(f"  Each animal assigned to one CA% level: {sorted(df['CA (%)'].unique())}")
			
			# Check for balance across groups
			group_sizes = df.groupby(['Sex', 'CA (%)'])['ID'].nunique()
			if group_sizes.nunique() == 1:
				print(f"  [OK] Balanced design: {group_sizes.iloc[0]} animals per group")
			else:
				print(f"  ⚠ Unbalanced design:")
				for (sex, ca), n in group_sizes.items():
					print(f"    Sex={sex}, CA%={ca}: {n} animals")
			
			# Check completeness across time points (if Day column exists)
			if "Day" in df.columns:
				print(f"\n  Time series completeness:")
				total_days = df['Day'].nunique()
				subjects_per_day = df.groupby('ID')['Day'].nunique()
				complete_time_subjects = subjects_per_day[subjects_per_day == total_days].index.tolist()
				incomplete_time_subjects = subjects_per_day[subjects_per_day < total_days].index.tolist()
				
				print(f"    Total time points: {total_days} days")
				print(f"    Animals with complete time series: {len(complete_time_subjects)} / {df['ID'].nunique()}")
				
				if incomplete_time_subjects:
					print(f"    ⚠ Animals with missing time points: {len(incomplete_time_subjects)}")
					for subj in incomplete_time_subjects[:3]:
						days_present = df[df['ID'] == subj]['Day'].nunique()
						print(f"      {subj}: {days_present}/{total_days} days")
					if len(incomplete_time_subjects) > 3:
						print(f"      ... and {len(incomplete_time_subjects) - 3} more")
		else:
			# Within-subjects design for CA%
			complete_subjects = subjects_per_ca[subjects_per_ca == total_ca_levels].index.tolist()
			incomplete_subjects = subjects_per_ca[subjects_per_ca < total_ca_levels].index.tolist()
			
			print(f"\n[OK] Experimental Design: CA% is WITHIN-SUBJECTS")
			print(f"  Total CA% levels: {total_ca_levels}")
			print(f"  Animals with complete data (all CA% levels): {len(complete_subjects)}")
			if incomplete_subjects:
				print(f"  ⚠ Animals with incomplete data: {len(incomplete_subjects)}")
				for subj in incomplete_subjects[:5]:
					ca_present = sorted(df[df['ID'] == subj]['CA (%)'].dropna().unique())
					print(f"    {subj}: present in CA% {ca_present}")
				if len(incomplete_subjects) > 5:
					print(f"    ... and {len(incomplete_subjects) - 5} more")
	
	print("\n" + "="*80)


def get_animals_by_group(df: pd.DataFrame, sex: Optional[str] = None, 
						 ca_percent: Optional[float] = None,
						 strain: Optional[str] = None) -> pd.DataFrame:
	"""
	Filter DataFrame to specific groups.
	
	Parameters:
		df: Cleaned DataFrame
		sex: Filter by sex ('M' or 'F'), None = all
		ca_percent: Filter by CA%, None = all
		strain: Filter by strain, None = all
		
	Returns:
		Filtered DataFrame
	"""
	result = df.copy()
	
	if sex is not None and "Sex" in df.columns:
		result = result[result["Sex"] == sex]
	
	if ca_percent is not None and "CA (%)" in df.columns:
		result = result[result["CA (%)"] == ca_percent]
	
	if strain is not None and "Strain" in df.columns:
		result = result[result["Strain"] == strain]
	
	return result


def perform_two_way_between_anova(df: pd.DataFrame, measure: str = "Total Change",
								  time_point: Optional[int] = None,
								  average_over_days: bool = False) -> dict:
	"""
	Perform 2×2 Between-Subjects ANOVA: Sex × CA%
	
	This analyzes the effect of Sex and CA% on weight measures where:
	- Sex: Between-subjects factor (each animal is M or F)
	- CA%: Between-subjects factor (each animal assigned to 0% or 2%)
	
	Can analyze:
	1. Single time point (specify time_point=Day)
	2. Average across all days (set average_over_days=True)
	
	Parameters:
		df: Cleaned DataFrame with Day column
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_point: Specific Day to analyze (e.g., 27 for final day), None = all days
		average_over_days: If True, average measure across all days per animal
		
	Returns:
		Dictionary with ANOVA results
	"""
	print("\n" + "="*80)
	print(f"TWO-WAY BETWEEN-SUBJECTS ANOVA: SEX × CA%")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Prepare data
	required_cols = ["ID", "Sex", "CA (%)", measure]
	if not all(col in df.columns for col in required_cols):
		print(f"\nERROR: Missing required columns. Need: {required_cols}")
		return {}
	
	analysis_df = df[required_cols].copy()
	analysis_df = analysis_df.dropna()
	
	# Filter to specific time point if requested
	if time_point is not None and "Day" in df.columns:
		day_df = df[df["Day"] == time_point][required_cols].copy()
		day_df = day_df.dropna()
		analysis_df = day_df
		print(f"\nAnalyzing: {measure} at Day {time_point}")
		print(f"  Data points: {len(analysis_df)}")
	elif average_over_days:
		# Average measure across all days for each animal
		avg_df = df.groupby(["ID", "Sex", "CA (%)"])[measure].mean().reset_index()
		avg_df.columns = ["ID", "Sex", "CA (%)", measure]
		analysis_df = avg_df
		print(f"\nAnalyzing: {measure} averaged across all days")
		print(f"  Animals: {len(analysis_df)}")
	else:
		print(f"\nAnalyzing: {measure} across all time points")
		print(f"  Total observations: {len(analysis_df)}")
		print(f"  Note: This pools all days. Consider using time_point or average_over_days.")
	
	# Descriptive statistics
	print(f"\nDescriptive Statistics:")
	
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
	for (sex, ca), group_data in analysis_df.groupby(["Sex", "CA (%)"])[measure]:
		stats = compute_desc_stats(group_data)
		stats['Sex'] = sex
		stats['CA (%)'] = ca
		stats_data.append(stats)
	group_stats = pd.DataFrame(stats_data)
	
	for _, row in group_stats.iterrows():
		print(f"  Sex={row['Sex']}, CA%={row['CA (%)']}: "
			  f"n={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
			  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
		print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
			  f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
	
	# Perform 2-way ANOVA using pingouin
	print(f"\nRunning 2-way between-subjects ANOVA...")
	
	aov = pg.anova(
		data=analysis_df,
		dv=measure,
		between=['Sex', 'CA (%)']
	)
	
	print(f"\nANOVA Table:")
	print(aov.to_string())
	
	# Determine p-value column name (different pingouin versions use different names)
	p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
	
	# Extract results
	sex_row = aov[aov['Source'] == 'Sex'].iloc[0] if 'Sex' in aov['Source'].values else None
	ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0] if 'CA (%)' in aov['Source'].values else None
	interaction_row = aov[aov['Source'] == 'Sex * CA (%)'].iloc[0] if 'Sex * CA (%)' in aov['Source'].values else None
	
	print(f"\nFormatted Results:")
	if sex_row is not None:
		print(f"  Sex: F({sex_row['DF']:.0f},{aov[aov['Source']=='Residual'].iloc[0]['DF']:.0f}) = "
			  f"{sex_row['F']:.3f}, p = {sex_row[p_col]:.4f} "
			  f"{'***' if sex_row[p_col] < 0.001 else '**' if sex_row[p_col] < 0.01 else '*' if sex_row[p_col] < 0.05 else 'ns'}")
	
	if ca_row is not None:
		print(f"  CA%: F({ca_row['DF']:.0f},{aov[aov['Source']=='Residual'].iloc[0]['DF']:.0f}) = "
			  f"{ca_row['F']:.3f}, p = {ca_row[p_col]:.4f} "
			  f"{'***' if ca_row[p_col] < 0.001 else '**' if ca_row[p_col] < 0.01 else '*' if ca_row[p_col] < 0.05 else 'ns'}")
	
	if interaction_row is not None:
		print(f"  Sex × CA%: F({interaction_row['DF']:.0f},{aov[aov['Source']=='Residual'].iloc[0]['DF']:.0f}) = "
			  f"{interaction_row['F']:.3f}, p = {interaction_row[p_col]:.4f} "
			  f"{'***' if interaction_row[p_col] < 0.001 else '**' if interaction_row[p_col] < 0.01 else '*' if interaction_row[p_col] < 0.05 else 'ns'}")
	
	results = {
		'measure': measure,
		'type': 'between_subjects',
		'anova_table': aov,
		'sex': {
			'F': sex_row['F'] if sex_row is not None else np.nan,
			'p': sex_row[p_col] if sex_row is not None else np.nan,
			'df1': sex_row['DF'] if sex_row is not None else np.nan,
			'df2': aov[aov['Source']=='Residual'].iloc[0]['DF'] if sex_row is not None else np.nan,
			'significant': sex_row[p_col] < 0.05 if sex_row is not None else False
		},
		'ca_percent': {
			'F': ca_row['F'] if ca_row is not None else np.nan,
			'p': ca_row[p_col] if ca_row is not None else np.nan,
			'df1': ca_row['DF'] if ca_row is not None else np.nan,
			'df2': aov[aov['Source']=='Residual'].iloc[0]['DF'] if ca_row is not None else np.nan,
			'significant': ca_row[p_col] < 0.05 if ca_row is not None else False
		},
		'interaction': {
			'F': interaction_row['F'] if interaction_row is not None else np.nan,
			'p': interaction_row[p_col] if interaction_row is not None else np.nan,
			'df1': interaction_row['DF'] if interaction_row is not None else np.nan,
			'df2': aov[aov['Source']=='Residual'].iloc[0]['DF'] if interaction_row is not None else np.nan,
			'significant': interaction_row[p_col] < 0.05 if interaction_row is not None else False
		}
	}
	
	return results


def perform_mixed_anova_time(df: pd.DataFrame, measure: str = "Total Change",
							 time_points: Optional[list] = None) -> dict:
	"""
	Perform 3-Way Mixed ANOVA: Time (within) × Sex (between) × CA% (between)
	
	This analyzes longitudinal weight changes where:
	- Time (Day): Within-subjects factor (repeated measures over days)
	- Sex: Between-subjects factor (each animal is M or F)
	- CA%: Between-subjects factor (each animal assigned to 0% or 2%)
	
	Parameters:
		df: Cleaned DataFrame with Day column
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific Days to include, None = all days
		
	Returns:
		Dictionary with ANOVA results
	"""
	print("\n" + "="*80)
	print(f"THREE-WAY MIXED ANOVA: TIME (WITHIN) × SEX (BETWEEN) × CA% (BETWEEN)")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Prepare data
	required_cols = ["ID", "Day", "Sex", "CA (%)", measure]
	if not all(col in df.columns for col in required_cols):
		print(f"\nERROR: Missing required columns. Need: {required_cols}")
		return {}
	
	analysis_df = df[required_cols].copy()
	analysis_df = analysis_df.dropna()
	
	# Filter to specific time points if requested
	if time_points is not None:
		analysis_df = analysis_df[analysis_df["Day"].isin(time_points)]
		print(f"\nAnalyzing: {measure} at Days {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all days")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  Days: {sorted(analysis_df['Day'].unique())}")
	
	# Check completeness across time points
	subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
	total_days = analysis_df['Day'].nunique()
	complete_subjects = subjects_per_day[subjects_per_day == total_days].index.tolist()
	incomplete_subjects = subjects_per_day[subjects_per_day < total_days].index.tolist()
	
	if incomplete_subjects:
		print(f"\nWarning: {len(incomplete_subjects)} animals missing data at some time points")
		print(f"Complete animals (all time points): {len(complete_subjects)}")
		print(f"Note: Filtering to only animals with complete data...")
		analysis_df = analysis_df[analysis_df['ID'].isin(complete_subjects)].copy()
		print(f"After filtering: {len(analysis_df['ID'].unique())} animals, {len(analysis_df)} observations")
	
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
	for (sex, ca), group_data in analysis_df.groupby(["Sex", "CA (%)"])[measure]:
		stats = compute_desc_stats(group_data)
		stats['Sex'] = sex
		stats['CA (%)'] = ca
		stats_data.append(stats)
	group_stats = pd.DataFrame(stats_data)
	
	for _, row in group_stats.iterrows():
		print(f"  Sex={row['Sex']}, CA%={row['CA (%)']}: "
			  f"n_obs={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
			  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
		print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
			  f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
	
	# Create a grouping variable for Sex × CA%
	analysis_df['Group'] = analysis_df['Sex'].astype(str) + '_' + analysis_df['CA (%)'].astype(str) + '%'
	
	# Perform mixed ANOVA with Time as within-subjects, Group as between-subjects
	print(f"\nRunning mixed ANOVA (Time within, Sex×CA% between)...")
	
	try:
		aov = pg.mixed_anova(
			data=analysis_df,
			dv=measure,
			within='Day',
			between='Group',
			subject='ID',
			correction='auto'
		)
		
		print(f"\nMixed ANOVA Table:")
		print(aov.to_string())
		
		# Determine p-value column name (different pingouin versions use different names)
		p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
		
		# For 3-way interaction, need to run separate models
		# First, test Sex × CA% at each time point
		print(f"\n" + "="*60)
		print("Testing Sex × CA% Interaction at Each Time Point")
		print("="*60)
		
		time_point_results = {}
		for day in sorted(analysis_df['Day'].unique()):
			day_df = analysis_df[analysis_df['Day'] == day]
			try:
				day_aov = pg.anova(data=day_df, dv=measure, between=['Sex', 'CA (%)'])
				# Determine p-value column for this ANOVA table
				day_p_col = 'p-unc' if 'p-unc' in day_aov.columns else 'p_unc'
				interaction_row = day_aov[day_aov['Source'] == 'Sex * CA (%)'].iloc[0]
				time_point_results[day] = {
					'F': interaction_row['F'],
					'p': interaction_row[day_p_col],
					'significant': interaction_row[day_p_col] < 0.05
				}
				sig_marker = '***' if interaction_row[day_p_col] < 0.001 else '**' if interaction_row[day_p_col] < 0.01 else '*' if interaction_row[day_p_col] < 0.05 else 'ns'
				print(f"  Day {day}: F = {interaction_row['F']:.3f}, p = {interaction_row[day_p_col]:.4f} {sig_marker}")
			except Exception as e:
				print(f"  Day {day}: Unable to compute ({e})")
				time_point_results[day] = {'error': str(e)}
		
		results = {
			'measure': measure,
			'type': 'mixed_anova',
			'anova_table': aov,
			'time_point_interactions': time_point_results
		}
		
		return results
		
	except Exception as e:
		print(f"\nERROR running mixed ANOVA: {e}")
		import traceback
		traceback.print_exc()
		return {}


def perform_mixed_anova_sex_stratified(df: pd.DataFrame, sex: str, measure: str = "Total Change",
										time_points: Optional[list] = None) -> dict:
	"""
	Perform 2-Way Mixed ANOVA: Time (within) × CA% (between), holding Sex constant
	
	This analyzes longitudinal weight changes for ONE sex at a time, testing:
	- Time (Day): Within-subjects factor (repeated measures over days)
	- CA%: Between-subjects factor (0% vs 2%)
	
	By stratifying by sex, this reveals whether the Time × CA% interaction differs
	between males and females.
	
	Parameters:
		df: Cleaned DataFrame with Day column
		sex: Sex to analyze ("M" or "F")
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific Days to include, None = all days
		
	Returns:
		Dictionary with ANOVA results for the specified sex
	"""
	print("\n" + "="*80)
	print(f"TWO-WAY MIXED ANOVA (SEX-STRATIFIED): TIME (WITHIN) × CA% (BETWEEN)")
	print(f"Analyzing: {'MALES' if sex == 'M' else 'FEMALES'} only")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Validate sex parameter
	if sex not in ["M", "F"]:
		print(f"\nERROR: Sex must be 'M' or 'F', got '{sex}'")
		return {}
	
	# Prepare data
	required_cols = ["ID", "Day", "Sex", "CA (%)", measure]
	if not all(col in df.columns for col in required_cols):
		print(f"\nERROR: Missing required columns. Need: {required_cols}")
		return {}
	
	# Filter to specified sex
	analysis_df = df[df["Sex"] == sex][required_cols].copy()
	analysis_df = analysis_df.dropna()
	
	if len(analysis_df) == 0:
		print(f"\nERROR: No data found for Sex={sex}")
		return {}
	
	# Filter to specific time points if requested
	if time_points is not None:
		analysis_df = analysis_df[analysis_df["Day"].isin(time_points)]
		print(f"\nAnalyzing: {measure} at Days {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all days")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  Days: {sorted(analysis_df['Day'].unique())}")
	print(f"  CA% groups: {sorted(analysis_df['CA (%)'].unique())}")
	
	# Check completeness across time points
	subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
	total_days = analysis_df['Day'].nunique()
	complete_subjects = subjects_per_day[subjects_per_day == total_days].index.tolist()
	incomplete_subjects = subjects_per_day[subjects_per_day < total_days].index.tolist()
	
	if incomplete_subjects:
		print(f"\nWarning: {len(incomplete_subjects)} animals missing data at some time points")
		print(f"Complete animals (all time points): {len(complete_subjects)}")
		print(f"Note: Filtering to only animals with complete data...")
		analysis_df = analysis_df[analysis_df['ID'].isin(complete_subjects)].copy()
		print(f"After filtering: {len(analysis_df['ID'].unique())} animals, {len(analysis_df)} observations")
	
	# Check that we have at least 2 animals per CA% group
	animals_per_ca = analysis_df.groupby('CA (%)')['ID'].nunique()
	print(f"\nAnimals per CA% group:")
	for ca, n in animals_per_ca.items():
		print(f"  {ca}%: {n} animals")
	
	if any(animals_per_ca < 2):
		print(f"\nWARNING: Some CA% groups have < 2 animals. Results may be unreliable.")
	
	# Descriptive statistics
	print(f"\nDescriptive Statistics by CA% Group:")
	
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
	for ca, group_data in analysis_df.groupby("CA (%)")[measure]:
		stats = compute_desc_stats(group_data)
		stats['CA (%)'] = ca
		stats_data.append(stats)
	group_stats = pd.DataFrame(stats_data)
	
	for _, row in group_stats.iterrows():
		print(f"  CA%={row['CA (%)']}: "
			  f"n_obs={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
			  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
		print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
			  f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
	
	# Perform mixed ANOVA with Time as within-subjects, CA% as between-subjects
	print(f"\nRunning mixed ANOVA (Time within, CA% between)...")
	
	try:
		aov = pg.mixed_anova(
			data=analysis_df,
			dv=measure,
			within='Day',
			between='CA (%)',
			subject='ID',
			correction='auto'
		)
		
		print(f"\nMixed ANOVA Table:")
		print(aov.to_string())
		
		# Determine p-value column name (different pingouin versions use different names)
		p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
		
		# Extract key results
		time_row = aov[aov['Source'] == 'Day'].iloc[0] if 'Day' in aov['Source'].values else None
		ca_row = aov[aov['Source'] == 'CA (%)'].iloc[0] if 'CA (%)' in aov['Source'].values else None
		interaction_row = aov[aov['Source'] == 'Interaction'].iloc[0] if 'Interaction' in aov['Source'].values else None
		
		# Alternative interaction label check
		if interaction_row is None:
			interaction_row = aov[aov['Source'] == 'Day * CA (%)'].iloc[0] if 'Day * CA (%)' in aov['Source'].values else None
		
		print(f"\nFormatted Results:")
		if time_row is not None:
			sig = '***' if time_row[p_col] < 0.001 else '**' if time_row[p_col] < 0.01 else '*' if time_row[p_col] < 0.05 else 'ns'
			print(f"  Time: F({time_row['DF1']:.0f},{time_row['DF2']:.0f}) = "
				  f"{time_row['F']:.3f}, p = {time_row[p_col]:.4f} {sig}")
		
		if ca_row is not None:
			sig = '***' if ca_row[p_col] < 0.001 else '**' if ca_row[p_col] < 0.01 else '*' if ca_row[p_col] < 0.05 else 'ns'
			print(f"  CA%: F({ca_row['DF1']:.0f},{ca_row['DF2']:.0f}) = "
				  f"{ca_row['F']:.3f}, p = {ca_row[p_col]:.4f} {sig}")
		
		if interaction_row is not None:
			sig = '***' if interaction_row[p_col] < 0.001 else '**' if interaction_row[p_col] < 0.01 else '*' if interaction_row[p_col] < 0.05 else 'ns'
			print(f"  Time × CA%: F({interaction_row['DF1']:.0f},{interaction_row['DF2']:.0f}) = "
				  f"{interaction_row['F']:.3f}, p = {interaction_row[p_col]:.4f} {sig}")
		
		results = {
			'measure': measure,
			'sex': sex,
			'type': 'mixed_anova_sex_stratified',
			'anova_table': aov,
			'time': {
				'F': time_row['F'] if time_row is not None else np.nan,
				'p': time_row[p_col] if time_row is not None else np.nan,
				'df1': time_row['DF1'] if time_row is not None else np.nan,
				'df2': time_row['DF2'] if time_row is not None else np.nan,
				'significant': time_row[p_col] < 0.05 if time_row is not None else False
			},
			'ca_percent': {
				'F': ca_row['F'] if ca_row is not None else np.nan,
				'p': ca_row[p_col] if ca_row is not None else np.nan,
				'df1': ca_row['DF1'] if ca_row is not None else np.nan,
				'df2': ca_row['DF2'] if ca_row is not None else np.nan,
				'significant': ca_row[p_col] < 0.05 if ca_row is not None else False
			},
			'interaction': {
				'F': interaction_row['F'] if interaction_row is not None else np.nan,
				'p': interaction_row[p_col] if interaction_row is not None else np.nan,
				'df1': interaction_row['DF1'] if interaction_row is not None else np.nan,
				'df2': interaction_row['DF2'] if interaction_row is not None else np.nan,
				'significant': interaction_row[p_col] < 0.05 if interaction_row is not None else False
			}
		}
		
		return results
		
	except Exception as e:
		print(f"\nERROR running mixed ANOVA: {e}")
		import traceback
		traceback.print_exc()
		return {}


def perform_mixed_anova_ca_stratified(df: pd.DataFrame, ca_percent: int, measure: str = "Total Change",
									   time_points: Optional[list] = None) -> dict:
	"""
	Perform 2-Way Mixed ANOVA: Time (within) × Sex (between), holding CA% constant
	
	This analyzes longitudinal weight changes for ONE CA% level at a time, testing:
	- Time (Day): Within-subjects factor (repeated measures over days)
	- Sex: Between-subjects factor (M vs F)
	
	By stratifying by CA%, this reveals whether the Time × Sex interaction differs
	between CA% conditions.
	
	Parameters:
		df: Cleaned DataFrame with Day column
		ca_percent: CA% level to analyze (e.g., 0 or 2)
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific Days to include, None = all days
		
	Returns:
		Dictionary with ANOVA results for the specified CA% level
	"""
	print("\n" + "="*80)
	print(f"TWO-WAY MIXED ANOVA (CA%-STRATIFIED): TIME (WITHIN) × SEX (BETWEEN)")
	print(f"Analyzing: {ca_percent}% CA only")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Prepare data
	required_cols = ["ID", "Day", "Sex", "CA (%)", measure]
	if not all(col in df.columns for col in required_cols):
		print(f"\nERROR: Missing required columns. Need: {required_cols}")
		return {}
	
	# Filter to specified CA%
	analysis_df = df[df["CA (%)"] == ca_percent][required_cols].copy()
	analysis_df = analysis_df.dropna()
	
	if len(analysis_df) == 0:
		print(f"\nERROR: No data found for CA%={ca_percent}")
		return {}
	
	# Filter to specific time points if requested
	if time_points is not None:
		analysis_df = analysis_df[analysis_df["Day"].isin(time_points)]
		print(f"\nAnalyzing: {measure} at Days {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all days")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  Days: {sorted(analysis_df['Day'].unique())}")
	print(f"  Sex groups: {sorted(analysis_df['Sex'].unique())}")
	
	# Check completeness across time points
	subjects_per_day = analysis_df.groupby('ID')['Day'].nunique()
	total_days = analysis_df['Day'].nunique()
	complete_subjects = subjects_per_day[subjects_per_day == total_days].index.tolist()
	incomplete_subjects = subjects_per_day[subjects_per_day < total_days].index.tolist()
	
	if incomplete_subjects:
		print(f"\nWarning: {len(incomplete_subjects)} animals missing data at some time points")
		print(f"Complete animals (all time points): {len(complete_subjects)}")
		print(f"Note: Filtering to only animals with complete data...")
		analysis_df = analysis_df[analysis_df['ID'].isin(complete_subjects)].copy()
		print(f"After filtering: {len(analysis_df['ID'].unique())} animals, {len(analysis_df)} observations")
	
	# Check that we have at least 2 animals per Sex group
	animals_per_sex = analysis_df.groupby('Sex')['ID'].nunique()
	print(f"\nAnimals per Sex group:")
	for sex, n in animals_per_sex.items():
		print(f"  {sex}: {n} animals")
	
	if any(animals_per_sex < 2):
		print(f"\nWARNING: Some Sex groups have < 2 animals. Results may be unreliable.")
	
	# Descriptive statistics
	print(f"\nDescriptive Statistics by Sex Group:")
	
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
	for sex, group_data in analysis_df.groupby("Sex")[measure]:
		stats = compute_desc_stats(group_data)
		stats['Sex'] = sex
		stats_data.append(stats)
	group_stats = pd.DataFrame(stats_data)
	
	for _, row in group_stats.iterrows():
		print(f"  Sex={row['Sex']}: "
			  f"n_obs={row['count']:.0f}, M={row['mean']:.3f}, Mdn={row['median']:.3f}, "
			  f"SD={row['std']:.3f}, SEM={row['sem']:.3f}")
		print(f"    95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
			  f"IQR: [{row['q25']:.3f}, {row['q75']:.3f}]")
	
	# Perform mixed ANOVA with Time as within-subjects, Sex as between-subjects
	print(f"\nRunning mixed ANOVA (Time within, Sex between)...")
	
	try:
		aov = pg.mixed_anova(
			data=analysis_df,
			dv=measure,
			within='Day',
			between='Sex',
			subject='ID',
			correction='auto'
		)
		
		print(f"\nMixed ANOVA Table:")
		print(aov.to_string())
		
		# Determine p-value column name (different pingouin versions use different names)
		p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
		
		# Extract key results
		time_row = aov[aov['Source'] == 'Day'].iloc[0] if 'Day' in aov['Source'].values else None
		sex_row = aov[aov['Source'] == 'Sex'].iloc[0] if 'Sex' in aov['Source'].values else None
		interaction_row = aov[aov['Source'] == 'Interaction'].iloc[0] if 'Interaction' in aov['Source'].values else None
		
		# Alternative interaction label check
		if interaction_row is None:
			interaction_row = aov[aov['Source'] == 'Day * Sex'].iloc[0] if 'Day * Sex' in aov['Source'].values else None
		
		print(f"\nFormatted Results:")
		if time_row is not None:
			sig = '***' if time_row[p_col] < 0.001 else '**' if time_row[p_col] < 0.01 else '*' if time_row[p_col] < 0.05 else 'ns'
			print(f"  Time: F({time_row['DF1']:.0f},{time_row['DF2']:.0f}) = "
				  f"{time_row['F']:.3f}, p = {time_row[p_col]:.4f} {sig}")
		
		if sex_row is not None:
			sig = '***' if sex_row[p_col] < 0.001 else '**' if sex_row[p_col] < 0.01 else '*' if sex_row[p_col] < 0.05 else 'ns'
			print(f"  Sex: F({sex_row['DF1']:.0f},{sex_row['DF2']:.0f}) = "
				  f"{sex_row['F']:.3f}, p = {sex_row[p_col]:.4f} {sig}")
		
		if interaction_row is not None:
			sig = '***' if interaction_row[p_col] < 0.001 else '**' if interaction_row[p_col] < 0.01 else '*' if interaction_row[p_col] < 0.05 else 'ns'
			print(f"  Time × Sex: F({interaction_row['DF1']:.0f},{interaction_row['DF2']:.0f}) = "
				  f"{interaction_row['F']:.3f}, p = {interaction_row[p_col]:.4f} {sig}")
		
		results = {
			'measure': measure,
			'ca_percent': ca_percent,
			'type': 'mixed_anova_ca_stratified',
			'anova_table': aov,
			'time': {
				'F': time_row['F'] if time_row is not None else np.nan,
				'p': time_row[p_col] if time_row is not None else np.nan,
				'df1': time_row['DF1'] if time_row is not None else np.nan,
				'df2': time_row['DF2'] if time_row is not None else np.nan,
				'significant': time_row[p_col] < 0.05 if time_row is not None else False
			},
			'sex': {
				'F': sex_row['F'] if sex_row is not None else np.nan,
				'p': sex_row[p_col] if sex_row is not None else np.nan,
				'df1': sex_row['DF1'] if sex_row is not None else np.nan,
				'df2': sex_row['DF2'] if sex_row is not None else np.nan,
				'significant': sex_row[p_col] < 0.05 if sex_row is not None else False
			},
			'interaction': {
				'F': interaction_row['F'] if interaction_row is not None else np.nan,
				'p': interaction_row[p_col] if interaction_row is not None else np.nan,
				'df1': interaction_row['DF1'] if interaction_row is not None else np.nan,
				'df2': interaction_row['DF2'] if interaction_row is not None else np.nan,
				'significant': interaction_row[p_col] < 0.05 if interaction_row is not None else False
			}
		}
		
		return results
		
	except Exception as e:
		print(f"\nERROR running mixed ANOVA: {e}")
		import traceback
		traceback.print_exc()
		return {}


def generate_analysis_report(
	between_results: Optional[dict] = None,
	mixed_results: Optional[dict] = None,
	results_males: Optional[dict] = None,
	results_females: Optional[dict] = None,
	results_ca0: Optional[dict] = None,
	results_ca2: Optional[dict] = None,
	tukey_results: Optional[dict] = None,
	mixed_posthoc_males: Optional[dict] = None,
	mixed_posthoc_females: Optional[dict] = None,
	mixed_posthoc_ca0: Optional[dict] = None,
	mixed_posthoc_ca2: Optional[dict] = None,
	df: Optional[pd.DataFrame] = None
) -> str:
	"""
	Generate a comprehensive formatted report of all CAH cohort analyses.
	
	Similar to ramp_analysis.py's display_two_way_anova_results, but adapted
	for the CAH cohort's between-subjects design.
	
	Parameters:
		between_results: Results from perform_two_way_between_anova()
		mixed_results: Results from perform_mixed_anova_time()
		results_males: Results from perform_mixed_anova_sex_stratified() for males
		results_females: Results from perform_mixed_anova_sex_stratified() for females
		results_ca0: Results from perform_mixed_anova_ca_stratified() for 0% CA
		results_ca2: Results from perform_mixed_anova_ca_stratified() for 2% CA
		tukey_results: Results from perform_tukey_hsd()
		mixed_posthoc_males: Post-hoc results for males stratified analysis
		mixed_posthoc_females: Post-hoc results for females stratified analysis
		mixed_posthoc_ca0: Post-hoc results for 0% CA stratified analysis
		mixed_posthoc_ca2: Post-hoc results for 2% CA stratified analysis
		df: Original DataFrame for additional context
		
	Returns:
		Formatted string report
	"""
	report_lines = []
	
	# Header
	report_lines.append("=" * 80)
	report_lines.append("CAH COHORT STATISTICAL ANALYSIS REPORT")
	report_lines.append("=" * 80)
	report_lines.append("")
	
	# Study design summary
	if df is not None:
		cdf = clean_cah_dataframe(df)
		n_animals = cdf['ID'].nunique()
		n_obs = len(cdf)
		
		report_lines.append("STUDY DESIGN")
		report_lines.append("-" * 80)
		report_lines.append(f"Total animals: {n_animals}")
		report_lines.append(f"Total observations: {n_obs}")
		
		if 'Sex' in cdf.columns and 'CA (%)' in cdf.columns:
			design = cdf.groupby(['Sex', 'CA (%)'])['ID'].nunique()
			report_lines.append("\nExperimental Design: 2×2 Between-Subjects (Sex × CA%)")
			for (sex, ca), n in design.items():
				report_lines.append(f"  {sex}, {ca}%: n = {n}")
		
		if 'Day' in cdf.columns:
			days = cdf['Day'].nunique()
			day_range = f"{cdf['Day'].min()}-{cdf['Day'].max()}"
			report_lines.append(f"\nLongitudinal measurement: {days} time points (Days {day_range})")
		
		report_lines.append("")
		report_lines.append("")
	
	# Between-subjects ANOVA results
	if between_results:
		report_lines.append("=" * 80)
		report_lines.append("BETWEEN-SUBJECTS ANOVA: SEX × CA%")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = between_results.get('measure', 'Unknown')
		analysis_type = between_results.get('type', 'Unknown')
		
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Analysis Type: {analysis_type}")
		report_lines.append("")
		
		# Descriptive Statistics by Group (show direction of effects)
		if df is not None:
			cdf = clean_cah_dataframe(df)
			if all(col in cdf.columns for col in ['Sex', 'CA (%)', measure]):
				report_lines.append("Descriptive Statistics by Group:")
				report_lines.append("-" * 80)
				
				# Helper function for enhanced descriptive statistics
				def compute_desc_stats_report(group):
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
				
				# Overall means by Sex
				stats_data = []
				for sex, group_data in cdf.groupby('Sex')[measure]:
					stats = compute_desc_stats_report(group_data)
					stats['Sex'] = sex
					stats_data.append(stats)
				sex_stats = pd.DataFrame(stats_data)
				
				report_lines.append("By Sex:")
				report_lines.append("  Basic Statistics:")
				for _, row in sex_stats.iterrows():
					report_lines.append(f"    {row['Sex']}: n = {int(row['count'])}, M = {row['mean']:.3f}, "
									  f"Mdn = {row['median']:.3f}, SD = {row['std']:.3f}, SEM = {row['sem']:.3f}")
				report_lines.append("  Confidence Intervals & Range:")
				for _, row in sex_stats.iterrows():
					report_lines.append(f"    {row['Sex']}: 95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
									  f"IQR [{row['q25']:.3f}, {row['q75']:.3f}], Range [{row['min']:.3f}, {row['max']:.3f}]")
				
				report_lines.append("")
				
				# Overall means by CA%
				stats_data = []
				for ca, group_data in cdf.groupby('CA (%)')[measure]:
					stats = compute_desc_stats_report(group_data)
					stats['CA (%)'] = ca
					stats_data.append(stats)
				ca_stats = pd.DataFrame(stats_data)
				
				report_lines.append("By CA%:")
				report_lines.append("  Basic Statistics:")
				for _, row in ca_stats.iterrows():
					report_lines.append(f"    {row['CA (%)']}%: n = {int(row['count'])}, M = {row['mean']:.3f}, "
									  f"Mdn = {row['median']:.3f}, SD = {row['std']:.3f}, SEM = {row['sem']:.3f}")
				report_lines.append("  Confidence Intervals & Range:")
				for _, row in ca_stats.iterrows():
					report_lines.append(f"    {row['CA (%)']}%: 95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
									  f"IQR [{row['q25']:.3f}, {row['q75']:.3f}], Range [{row['min']:.3f}, {row['max']:.3f}]")
				
				report_lines.append("")
				
				# Interaction: means by both factors
				stats_data = []
				for (sex, ca), group_data in cdf.groupby(['Sex', 'CA (%)'])[measure]:
					stats = compute_desc_stats_report(group_data)
					stats['Sex'] = sex
					stats['CA (%)'] = ca
					stats_data.append(stats)
				group_stats = pd.DataFrame(stats_data)
				
				report_lines.append("By Sex × CA% Combination:")
				report_lines.append("  Basic Statistics:")
				for _, row in group_stats.iterrows():
					report_lines.append(f"    {row['Sex']}, {row['CA (%)']}%: n = {int(row['count'])}, M = {row['mean']:.3f}, "
									  f"Mdn = {row['median']:.3f}, SD = {row['std']:.3f}, SEM = {row['sem']:.3f}")
				report_lines.append("  Confidence Intervals & Range:")
				for _, row in group_stats.iterrows():
					report_lines.append(f"    {row['Sex']}, {row['CA (%)']}%: 95% CI [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
									  f"IQR [{row['q25']:.3f}, {row['q75']:.3f}], Range [{row['min']:.3f}, {row['max']:.3f}]")
				
				report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in between_results:
			report_lines.append("ANOVA Table:")
			report_lines.append("-" * 80)
			aov_str = between_results['anova_table'].to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
		
		# Main effects and interaction
		report_lines.append("Main Effects and Interaction:")
		report_lines.append("-" * 80)
		
		for effect_name, effect_key in [('Sex', 'sex'), ('CA%', 'ca_percent'), ('Sex × CA%', 'interaction')]:
			if effect_key in between_results:
				effect = between_results[effect_key]
				F = effect.get('F', np.nan)
				p = effect.get('p', np.nan)
				df1 = effect.get('df1', np.nan)
				df2 = effect.get('df2', np.nan)
				sig = effect.get('significant', False)
				
				if not np.isnan(F):
					sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
					report_lines.append(f"{effect_name}:")
					report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
					
					# Effect size if available
					if 'anova_table' in between_results:
						aov = between_results['anova_table']
						# Check for eta-squared or partial eta-squared
						if 'np2' in aov.columns:
							source_name = 'Sex' if effect_key == 'sex' else 'CA (%)' if effect_key == 'ca_percent' else 'Sex * CA (%)'
							if source_name in aov['Source'].values:
								eta_sq = aov[aov['Source'] == source_name]['np2'].iloc[0]
								if not np.isnan(eta_sq):
									report_lines.append(f"  Partial η² = {eta_sq:.3f}")
					report_lines.append("")
		
		report_lines.append("")
	
	# Mixed ANOVA results
	if mixed_results:
		report_lines.append("=" * 80)
		report_lines.append("MIXED ANOVA: TIME (WITHIN) × SEX × CA% (BETWEEN)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = mixed_results.get('measure', 'Unknown')
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Design: Time (within-subjects) × Sex × CA% (between-subjects)")
		report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in mixed_results:
			aov = mixed_results['anova_table']
			report_lines.append("Mixed ANOVA Table:")
			report_lines.append("-" * 80)
			aov_str = aov.to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
			
			# Sphericity violations (Mauchly's test)
			report_lines.append("Sphericity Assessment:")
			report_lines.append("-" * 80)
			
			# Check if sphericity columns are present
			if 'sphericity' in aov.columns or 'W-spher' in aov.columns:
				day_row = aov[aov['Source'] == 'Day']
				if not day_row.empty:
					if 'sphericity' in aov.columns:
						sphericity = day_row['sphericity'].iloc[0]
						report_lines.append(f"Mauchly's Test of Sphericity: W = {sphericity:.4f}")
					
					# Check epsilon values for corrections
					if 'eps' in aov.columns and not pd.isna(day_row['eps'].iloc[0]):
						epsilon = day_row['eps'].iloc[0]
						report_lines.append(f"Greenhouse-Geisser ε = {epsilon:.4f}")
						
						if epsilon < 0.75:
							report_lines.append("⚠ WARNING: Sphericity assumption violated (ε < 0.75)")
							report_lines.append("  → Greenhouse-Geisser correction applied")
						else:
							report_lines.append("[OK] Sphericity assumption met (ε ≥ 0.75)")
					else:
						report_lines.append("Note: Sphericity test not available")
				else:
					report_lines.append("Note: Sphericity information not found in ANOVA table")
			else:
				report_lines.append("Note: Sphericity statistics not available in this analysis")
			
			report_lines.append("")
			
			# Main effects and interactions
			report_lines.append("Main Effects and Interactions:")
			report_lines.append("-" * 80)
			
			for idx, row in aov.iterrows():
				source = row['Source']
				F = row.get('F', np.nan)
				df1 = row.get('DF1', row.get('df1', np.nan))
				df2 = row.get('DF2', row.get('df2', np.nan))
				
				# Get p-value (handle different column names)
				p = row.get('p-unc', row.get('p_unc', np.nan))
				
				if not np.isnan(F) and source not in ['Residual']:
					sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
					report_lines.append(f"{source}:")
					report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
					
					# Effect size
					if 'np2' in row and not np.isnan(row['np2']):
						report_lines.append(f"  Partial η² = {row['np2']:.3f}")
					
					report_lines.append("")
			
			report_lines.append("")
		
		# Time-point specific interactions
		if 'time_point_interactions' in mixed_results:
			report_lines.append("Time-Point Specific Sex × CA% Interactions:")
			report_lines.append("-" * 80)
			
			time_results = mixed_results['time_point_interactions']
			for day in sorted(time_results.keys()):
				result = time_results[day]
				if 'error' in result:
					report_lines.append(f"Day {day}: {result['error']}")
				else:
					F = result.get('F', np.nan)
					p = result.get('p', np.nan)
					sig = result.get('significant', False)
					sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
					report_lines.append(f"Day {day}: F = {F:.3f}, p = {p:.4f} {sig_marker}")
			
			report_lines.append("")
			report_lines.append("")
	
	# Sex-stratified mixed ANOVA results
	if results_males:
		report_lines.append("=" * 80)
		report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME × CA% (MALES ONLY)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = results_males.get('measure', 'Unknown')
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Design: Time (within-subjects) × CA% (between-subjects), Males only")
		report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in results_males:
			aov = results_males['anova_table']
			report_lines.append("Mixed ANOVA Table (Males):")
			report_lines.append("-" * 80)
			aov_str = aov.to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
			
			# Main effects and interaction
			report_lines.append("Main Effects and Interaction:")
			report_lines.append("-" * 80)
			
			for effect_name, effect_key in [('Time', 'time'), ('CA%', 'ca_percent'), ('Time × CA%', 'interaction')]:
				if effect_key in results_males:
					effect = results_males[effect_key]
					F = effect.get('F', np.nan)
					p = effect.get('p', np.nan)
					df1 = effect.get('df1', np.nan)
					df2 = effect.get('df2', np.nan)
					
					if not np.isnan(F):
						sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
						report_lines.append("")
		
		report_lines.append("")
	
	if results_females:
		report_lines.append("=" * 80)
		report_lines.append("SEX-STRATIFIED MIXED ANOVA: TIME × CA% (FEMALES ONLY)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = results_females.get('measure', 'Unknown')
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Design: Time (within-subjects) × CA% (between-subjects), Females only")
		report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in results_females:
			aov = results_females['anova_table']
			report_lines.append("Mixed ANOVA Table (Females):")
			report_lines.append("-" * 80)
			aov_str = aov.to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
			
			# Main effects and interaction
			report_lines.append("Main Effects and Interaction:")
			report_lines.append("-" * 80)
			
			for effect_name, effect_key in [('Time', 'time'), ('CA%', 'ca_percent'), ('Time × CA%', 'interaction')]:
				if effect_key in results_females:
					effect = results_females[effect_key]
					F = effect.get('F', np.nan)
					p = effect.get('p', np.nan)
					df1 = effect.get('df1', np.nan)
					df2 = effect.get('df2', np.nan)
					
					if not np.isnan(F):
						sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
						report_lines.append("")
		
		report_lines.append("")
	
	# Comparison of sex-stratified results
	if results_males and results_females:
		report_lines.append("=" * 80)
		report_lines.append("COMPARISON: TIME × CA% INTERACTION BY SEX")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		male_int = results_males.get('interaction', {})
		female_int = results_females.get('interaction', {})
		
		report_lines.append("Time × CA% Interaction:")
		report_lines.append("-" * 80)
		
		male_sig = male_int.get('significant', False)
		female_sig = female_int.get('significant', False)
		
		male_marker = '***' if male_int.get('p', 1) < 0.001 else '**' if male_int.get('p', 1) < 0.01 else '*' if male_int.get('p', 1) < 0.05 else 'ns'
		female_marker = '***' if female_int.get('p', 1) < 0.001 else '**' if female_int.get('p', 1) < 0.01 else '*' if female_int.get('p', 1) < 0.05 else 'ns'
		
		report_lines.append(f"Males:   F({male_int.get('df1', 0):.0f}, {male_int.get('df2', 0):.0f}) = "
						  f"{male_int.get('F', np.nan):.3f}, p = {male_int.get('p', np.nan):.4f} {male_marker}")
		report_lines.append(f"Females: F({female_int.get('df1', 0):.0f}, {female_int.get('df2', 0):.0f}) = "
						  f"{female_int.get('F', np.nan):.3f}, p = {female_int.get('p', np.nan):.4f} {female_marker}")
		report_lines.append("")
		
		if male_sig and not female_sig:
			report_lines.append("→ Time × CA% interaction is significant in MALES but not FEMALES")
			report_lines.append("  This suggests sex differences in how CA% affects weight over time.")
		elif female_sig and not male_sig:
			report_lines.append("→ Time × CA% interaction is significant in FEMALES but not MALES")
			report_lines.append("  This suggests sex differences in how CA% affects weight over time.")
		elif male_sig and female_sig:
			report_lines.append("→ Time × CA% interaction is significant in BOTH sexes")
			report_lines.append("  Both males and females show CA%-dependent weight trajectories.")
		else:
			report_lines.append("→ Time × CA% interaction is NOT significant in either sex")
			report_lines.append("  CA% does not differentially affect weight trajectories over time.")
		
		report_lines.append("")
	
	# CA%-stratified mixed ANOVA results
	if results_ca0:
		report_lines.append("=" * 80)
		report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME × SEX (0% CA ONLY)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = results_ca0.get('measure', 'Unknown')
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Design: Time (within-subjects) × Sex (between-subjects), 0% CA only")
		report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in results_ca0:
			aov = results_ca0['anova_table']
			report_lines.append("Mixed ANOVA Table (0% CA):")
			report_lines.append("-" * 80)
			aov_str = aov.to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
			
			# Main effects and interaction
			report_lines.append("Main Effects and Interaction:")
			report_lines.append("-" * 80)
			
			for effect_name, effect_key in [('Time', 'time'), ('Sex', 'sex'), ('Time × Sex', 'interaction')]:
				if effect_key in results_ca0:
					effect = results_ca0[effect_key]
					F = effect.get('F', np.nan)
					p = effect.get('p', np.nan)
					df1 = effect.get('df1', np.nan)
					df2 = effect.get('df2', np.nan)
					
					if not np.isnan(F):
						sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
						report_lines.append("")
		
		report_lines.append("")
	
	if results_ca2:
		report_lines.append("=" * 80)
		report_lines.append("CA%-STRATIFIED MIXED ANOVA: TIME × SEX (2% CA ONLY)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		measure = results_ca2.get('measure', 'Unknown')
		report_lines.append(f"Dependent Variable: {measure}")
		report_lines.append(f"Design: Time (within-subjects) × Sex (between-subjects), 2% CA only")
		report_lines.append("")
		
		# ANOVA Table
		if 'anova_table' in results_ca2:
			aov = results_ca2['anova_table']
			report_lines.append("Mixed ANOVA Table (2% CA):")
			report_lines.append("-" * 80)
			aov_str = aov.to_string(index=False)
			report_lines.append(aov_str)
			report_lines.append("")
			
			# Main effects and interaction
			report_lines.append("Main Effects and Interaction:")
			report_lines.append("-" * 80)
			
			for effect_name, effect_key in [('Time', 'time'), ('Sex', 'sex'), ('Time × Sex', 'interaction')]:
				if effect_key in results_ca2:
					effect = results_ca2[effect_key]
					F = effect.get('F', np.nan)
					p = effect.get('p', np.nan)
					df1 = effect.get('df1', np.nan)
					df2 = effect.get('df2', np.nan)
					
					if not np.isnan(F):
						sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}")
						report_lines.append("")
		
		report_lines.append("")
	
	# Comparison of CA%-stratified results
	if results_ca0 and results_ca2:
		report_lines.append("=" * 80)
		report_lines.append("COMPARISON: TIME × SEX INTERACTION BY CA%")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		ca0_int = results_ca0.get('interaction', {})
		ca2_int = results_ca2.get('interaction', {})
		
		report_lines.append("Time × Sex Interaction:")
		report_lines.append("-" * 80)
		
		ca0_sig = ca0_int.get('significant', False)
		ca2_sig = ca2_int.get('significant', False)
		
		ca0_marker = '***' if ca0_int.get('p', 1) < 0.001 else '**' if ca0_int.get('p', 1) < 0.01 else '*' if ca0_int.get('p', 1) < 0.05 else 'ns'
		ca2_marker = '***' if ca2_int.get('p', 1) < 0.001 else '**' if ca2_int.get('p', 1) < 0.01 else '*' if ca2_int.get('p', 1) < 0.05 else 'ns'
		
		report_lines.append(f"0% CA:  F({ca0_int.get('df1', 0):.0f}, {ca0_int.get('df2', 0):.0f}) = "
						  f"{ca0_int.get('F', np.nan):.3f}, p = {ca0_int.get('p', np.nan):.4f} {ca0_marker}")
		report_lines.append(f"2% CA:  F({ca2_int.get('df1', 0):.0f}, {ca2_int.get('df2', 0):.0f}) = "
						  f"{ca2_int.get('F', np.nan):.3f}, p = {ca2_int.get('p', np.nan):.4f} {ca2_marker}")
		report_lines.append("")
		
		if ca0_sig and not ca2_sig:
			report_lines.append("→ Time × Sex interaction is significant at 0% CA but NOT at 2% CA")
			report_lines.append("  Sex differences in weight trajectories are present without CA but eliminated with CA.")
		elif ca2_sig and not ca0_sig:
			report_lines.append("→ Time × Sex interaction is significant at 2% CA but NOT at 0% CA")
			report_lines.append("  CA exposure reveals sex differences in weight trajectories that aren't present in controls.")
		elif ca0_sig and ca2_sig:
			report_lines.append("→ Time × Sex interaction is significant at BOTH CA% levels")
			report_lines.append("  Sex differences in weight trajectories persist across CA% conditions.")
		else:
			report_lines.append("→ Time × Sex interaction is NOT significant at either CA% level")
			report_lines.append("  Males and females have similar weight trajectories regardless of CA% exposure.")
		
		report_lines.append("")
	
	# Mixed ANOVA post-hoc results
	posthoc_list = [
		("Males (Time × CA%)", results_males, mixed_posthoc_males),
		("Females (Time × CA%)", results_females, mixed_posthoc_females),
		("0% CA (Time × Sex)", results_ca0, mixed_posthoc_ca0),
		("2% CA (Time × Sex)", results_ca2, mixed_posthoc_ca2)
	]
	
	has_any_posthoc = any(ph for _, _, ph in posthoc_list if ph)
	
	if has_any_posthoc:
		report_lines.append("=" * 80)
		report_lines.append("MIXED ANOVA POST-HOC TESTS (REPEATED MEASURES)")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		for title, anova_results, mixed_posthoc in posthoc_list:
			if not mixed_posthoc:
				continue
			
			report_lines.append("-" * 80)
			report_lines.append(f"POST-HOC: {title}")
			report_lines.append("-" * 80)
			report_lines.append("")
			
			within = mixed_posthoc.get('within', 'Unknown')
			between = mixed_posthoc.get('between', 'Unknown')
			padjust = mixed_posthoc.get('padjust', 'Unknown')
			
			report_lines.append(f"Within-subjects factor: {within}")
			report_lines.append(f"Between-subjects factor: {between}")
			report_lines.append(f"Multiple comparison correction: {padjust}")
			report_lines.append("")
			
			# Simple effects
			if 'simple_effects' in mixed_posthoc and not mixed_posthoc['simple_effects'].empty:
				se_df = mixed_posthoc['simple_effects']
				report_lines.append("Simple Effects Analysis:")
				report_lines.append(f"Effect of {between} at each {within} level")
				report_lines.append("-" * 60)
				
				for _, row in se_df.iterrows():
					level = row['within_level']
					
					if 't' in row:
						# t-test results
						t_val = row['t']
						p_val = row['p']
						p_adj = row.get('p_adjusted', p_val)
						sig = row.get('significant_adjusted', row['significant'])
						sig_marker = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
						
						report_lines.append(f"{within} = {level}:")
						report_lines.append(f"  {row['comparison']}")
						report_lines.append(f"  t = {t_val:.3f}, p = {p_val:.4f}, p-adj = {p_adj:.4f} {sig_marker}")
						
					elif 'F' in row:
						# ANOVA results
						f_val = row['F']
						p_val = row['p']
						p_adj = row.get('p_adjusted', p_val)
						sig = row.get('significant_adjusted', row['significant'])
						sig_marker = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
						
						report_lines.append(f"{within} = {level}:")
						report_lines.append(f"  F = {f_val:.3f}, p = {p_val:.4f}, p-adj = {p_adj:.4f} {sig_marker}")
					
					report_lines.append("")
				
				# Summary
				if 'significant_adjusted' in se_df.columns:
					n_sig = se_df['significant_adjusted'].sum()
					n_total = len(se_df)
					report_lines.append(f"Summary: {n_sig}/{n_total} time points show significant {between} effects (after correction)")
				else:
					n_sig = se_df['significant'].sum()
					n_total = len(se_df)
					report_lines.append(f"Summary: {n_sig}/{n_total} time points show significant {between} effects")
				
				report_lines.append("")
			
			# Within-subjects pairwise comparisons (if any significant)
			if 'within_pairwise' in mixed_posthoc and not mixed_posthoc['within_pairwise'].empty:
				within_pw = mixed_posthoc['within_pairwise']
				
				# Check for adjusted p-value column
				p_col = None
				for col in ['p-corr', f'p-{padjust}', 'p-adj']:
					if col in within_pw.columns:
						p_col = col
						break
				
				if p_col:
					sig_within = within_pw[within_pw[p_col] < 0.05]
					
					if len(sig_within) > 0:
						report_lines.append(f"Significant Pairwise {within} Comparisons (p-adj < 0.05):")
						report_lines.append("-" * 60)
						report_lines.append(f"Total: {len(sig_within)} significant out of {len(within_pw)} comparisons")
						
						# Show top 10
						if len(sig_within) > 10:
							report_lines.append("(Showing top 10 strongest effects)")
						
						for _, row in sig_within.head(10).iterrows():
							a = row['A']
							b = row['B']
							p_adj = row[p_col]
							sig_marker = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
							report_lines.append(f"  {within} {a} vs {b}: p-adj = {p_adj:.4f} {sig_marker}")
						
						report_lines.append("")
			
			report_lines.append("")
	
	# Tukey HSD post-hoc results
	if tukey_results:
		report_lines.append("=" * 80)
		report_lines.append("POST-HOC TESTS: TUKEY HSD")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		factor = tukey_results.get('factor', 'Unknown')
		report_lines.append(f"Factor: {factor}")
		report_lines.append("")
		
		if 'tukey_table' in tukey_results:
			tukey_df = tukey_results['tukey_table']
			report_lines.append("Pairwise Comparisons:")
			report_lines.append("-" * 80)
			
			# Format the table
			for idx, row in tukey_df.iterrows():
				g1 = row['group1']
				g2 = row['group2']
				diff = float(row['meandiff'])
				p_adj = float(row['p-adj'])
				lower = float(row['lower'])
				upper = float(row['upper'])
				reject = row['reject']
				
				sig_marker = '*' if reject else 'ns'
				report_lines.append(f"{g1} vs {g2}:")
				report_lines.append(f"  Mean Difference = {diff:.3f}, 95% CI [{lower:.3f}, {upper:.3f}]")
				report_lines.append(f"  p-adj = {p_adj:.4f} {sig_marker}")
				report_lines.append("")
		
		# Summary of significant comparisons
		if 'significant_comparisons' in tukey_results:
			sig_comps = tukey_results['significant_comparisons']
			if len(sig_comps) > 0:
				report_lines.append("Summary of Significant Pairwise Differences (α = 0.05):")
				report_lines.append("-" * 80)
				for idx, row in sig_comps.iterrows():
					g1 = row['group1']
					g2 = row['group2']
					p_adj = float(row['p-adj'])
					report_lines.append(f"  {g1} ≠ {g2} (p = {p_adj:.4f})")
				report_lines.append("")
			else:
				report_lines.append("No significant pairwise differences found.")
				report_lines.append("")
		
		report_lines.append("")
	
	# Footer
	report_lines.append("=" * 80)
	report_lines.append("END OF REPORT")
	report_lines.append("=" * 80)
	
	return "\n".join(report_lines)


def perform_tukey_hsd(df: pd.DataFrame, measure: str, factor: str) -> dict:
	"""
	Perform Tukey HSD post-hoc test for pairwise comparisons.
	
	Parameters:
		df: DataFrame with data
		measure: Dependent variable column name
		factor: Factor column name for grouping
		
	Returns:
		Dictionary with Tukey HSD results
	"""
	if not HAS_STATSMODELS:
		print("\nERROR: statsmodels required for Tukey HSD.")
		print("Install with: pip install statsmodels")
		return {}
	
	print(f"\n" + "="*60)
	print(f"TUKEY HSD POST-HOC TEST: {factor}")
	print("="*60)
	
	# Prepare data
	test_df = df[[measure, factor]].dropna()
	
	# Perform Tukey HSD
	tukey = pairwise_tukeyhsd(endog=test_df[measure], groups=test_df[factor], alpha=0.05)
	
	print(f"\n{tukey}")
	
	# Extract significant comparisons
	tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
	significant = tukey_df[tukey_df['reject'] == True]
	
	if len(significant) > 0:
		print(f"\nSignificant pairwise differences (p < 0.05):")
		for _, row in significant.iterrows():
			print(f"  {row['group1']} vs {row['group2']}: p = {float(row['p-adj']):.4f}")
	else:
		print(f"\nNo significant pairwise differences found.")
	
	return {
		'factor': factor,
		'tukey_table': tukey_df,
		'significant_comparisons': significant
	}


def perform_mixed_anova_posthoc(df: pd.DataFrame, measure: str, 
								within: str, between: str, subject: str,
								padjust: str = 'fdr_bh') -> dict:
	"""
	Perform post-hoc pairwise comparisons for mixed ANOVA using pingouin.
	
	This function properly handles repeated measures by using pingouin.pairwise_tests(),
	which applies appropriate corrections for within-subjects comparisons.
	
	Parameters:
		df: DataFrame with complete data
		measure: Dependent variable column name
		within: Within-subjects factor (e.g., 'Day')
		between: Between-subjects factor (e.g., 'Sex' or 'CA (%)')
		subject: Subject identifier column (e.g., 'ID')
		padjust: Multiple comparison correction method:
			- 'fdr_bh': Benjamini-Hochberg FDR (recommended for many comparisons)
			- 'bonf': Bonferroni (very conservative)
			- 'holm': Holm-Bonferroni (less conservative than Bonferroni)
			- 'none': No correction (not recommended)
			
	Returns:
		Dictionary with post-hoc results including:
		- within_pairwise: Pairwise comparisons across time points
		- between_pairwise: Pairwise comparisons between groups (if >2 levels)
		- interaction_simple_effects: Simple effects at each level
	"""
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin required for repeated measures post-hoc tests.")
		print("Install with: pip install pingouin")
		return {}
	
	print(f"\n" + "="*80)
	print(f"MIXED ANOVA POST-HOC TESTS")
	print(f"Within: {within}, Between: {between}, Subject: {subject}")
	print("="*80)
	
	results = {
		'measure': measure,
		'within': within,
		'between': between,
		'padjust': padjust
	}
	
	# Prepare data
	required_cols = [subject, within, between, measure]
	test_df = df[required_cols].copy().dropna()
	
	print(f"\nData: {len(test_df)} observations, {test_df[subject].nunique()} subjects")
	print(f"Correction method: {padjust}")
	
	# 1. PAIRWISE COMPARISONS ACROSS WITHIN-SUBJECTS FACTOR (e.g., Time points)
	print(f"\n" + "-"*80)
	print(f"1. PAIRWISE COMPARISONS: {within} (within-subjects)")
	print("-"*80)
	
	try:
		# Use pairwise_tests with parametric=True for paired t-tests
		within_pw = pg.pairwise_tests(
			data=test_df,
			dv=measure,
			within=within,
			subject=subject,
			parametric=True,
			padjust=padjust,
			return_desc=True
		)
		
		print(f"\nPairwise comparisons across {within}:")
		# Filter to just the within-subject comparisons
		within_only = within_pw[within_pw['Contrast'] == within].copy()
		
		if len(within_only) > 0:
			print(f"\nFound {len(within_only)} pairwise comparisons")
			
			# Show significant comparisons
			p_col = 'p-unc' if 'p-unc' in within_only.columns else 'pval'
			padj_col = 'p-corr' if 'p-corr' in within_only.columns else f'p-{padjust}'
			
			if padj_col in within_only.columns:
				sig_within = within_only[within_only[padj_col] < 0.05]
				print(f"Significant comparisons (p-adj < 0.05): {len(sig_within)}")
				
				if len(sig_within) > 0:
					print(f"\nTop 10 significant {within} comparisons:")
					for idx, row in sig_within.head(10).iterrows():
						a = row['A']
						b = row['B']
						p_adj = row[padj_col]
						print(f"  {within} {a} vs {b}: p-adj = {p_adj:.4f}")
			
			results['within_pairwise'] = within_only
		else:
			print(f"No within-subjects comparisons found.")
			results['within_pairwise'] = pd.DataFrame()
			
	except Exception as e:
		print(f"\nERROR computing within-subjects pairwise comparisons: {e}")
		results['within_pairwise'] = pd.DataFrame()
	
	# 2. PAIRWISE COMPARISONS FOR BETWEEN-SUBJECTS FACTOR (if applicable)
	between_levels = test_df[between].nunique()
	
	if between_levels > 2:
		print(f"\n" + "-"*80)
		print(f"2. PAIRWISE COMPARISONS: {between} (between-subjects)")
		print("-"*80)
		
		try:
			# Average across within-subjects factor for between-subjects comparison
			avg_df = test_df.groupby([subject, between])[measure].mean().reset_index()
			
			# Use regular pairwise_tests for between-subjects
			between_pw = pg.pairwise_tests(
				data=avg_df,
				dv=measure,
				between=between,
				parametric=True,
				padjust=padjust
			)
			
			print(f"\nPairwise comparisons for {between} (averaged across {within}):")
			print(between_pw.to_string())
			
			results['between_pairwise'] = between_pw
			
		except Exception as e:
			print(f"\nERROR computing between-subjects pairwise comparisons: {e}")
			results['between_pairwise'] = pd.DataFrame()
	else:
		print(f"\n{between} has only {between_levels} levels - no post-hoc needed (use main effect p-value)")
		results['between_pairwise'] = pd.DataFrame()
	
	# 3. SIMPLE EFFECTS: Compare between-subjects factor at each level of within-subjects factor
	print(f"\n" + "-"*80)
	print(f"3. SIMPLE EFFECTS: {between} at each {within} level")
	print("-"*80)
	
	simple_effects = []
	within_levels = sorted(test_df[within].unique())
	
	print(f"\nTesting {between} differences at each {within} level...")
	
	for level in within_levels:
		level_data = test_df[test_df[within] == level].copy()
		
		try:
			# For 2-level between factor, use t-test; for >2 levels, use ANOVA
			if between_levels == 2:
				groups = level_data.groupby(between)[measure].apply(list)
				group_names = list(groups.index)
				
				if len(group_names) == 2:
					from scipy.stats import ttest_ind
					t_stat, p_val = ttest_ind(groups.iloc[0], groups.iloc[1])
					
					simple_effects.append({
						'within_level': level,
						'comparison': f"{group_names[0]} vs {group_names[1]}",
						'test': 't-test',
						't': t_stat,
						'p': p_val,
						'significant': p_val < 0.05
					})
					
					sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
					print(f"  {within}={level}: t = {t_stat:.3f}, p = {p_val:.4f} {sig_marker}")
					
			else:
				# Use one-way ANOVA for >2 groups
				aov = pg.anova(data=level_data, dv=measure, between=between)
				p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
				
				f_val = aov.iloc[0]['F']
				p_val = aov.iloc[0][p_col]
				
				simple_effects.append({
					'within_level': level,
					'test': 'ANOVA',
					'F': f_val,
					'p': p_val,
					'significant': p_val < 0.05
				})
				
				sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
				print(f"  {within}={level}: F = {f_val:.3f}, p = {p_val:.4f} {sig_marker}")
				
		except Exception as e:
			print(f"  {within}={level}: Error - {e}")
	
	if simple_effects:
		results['simple_effects'] = pd.DataFrame(simple_effects)
		
		# Apply correction for multiple comparisons across simple effects
		if len(simple_effects) > 1 and padjust != 'none':
			from statsmodels.stats.multitest import multipletests
			p_values = [se['p'] for se in simple_effects]
			rejected, p_adjusted, _, _ = multipletests(p_values, method=padjust)
			
			results['simple_effects']['p_adjusted'] = p_adjusted
			results['simple_effects']['significant_adjusted'] = rejected
			
			print(f"\nAfter {padjust} correction for {len(simple_effects)} comparisons:")
			sig_count = sum(rejected)
			print(f"  Significant simple effects: {sig_count}/{len(simple_effects)}")
	else:
		results['simple_effects'] = pd.DataFrame()
	
	print("\n" + "="*80)
	
	return results


def plot_interaction_effects(
	between_results: Optional[dict] = None,
	mixed_results: Optional[dict] = None,
	df: Optional[pd.DataFrame] = None,
	save_dir: Optional[Path] = None,
	show: bool = True
) -> Dict[str, plt.Figure]:
	"""
	Plot Sex × CA% interaction effects for CAH cohort (between-subjects design).
	
	Creates line plots showing how the effect of CA% differs between sexes.
	Displays mean weight measures (±SEM) for each Sex × CA% combination.
	
	IMPORTANT: Assumes BETWEEN-SUBJECTS design where:
	- Each animal is assigned to ONE sex (M or F)
	- Each animal is assigned to ONE CA% level (not repeated measures on CA%)
	- Error bars represent between-subjects variability (SEM across animals in each group)
	
	The plot shows 4 data points representing the 2×2 factorial design:
	- Male-0%, Male-2%, Female-0%, Female-2%
	
	Parameters:
		between_results: Results from perform_two_way_between_anova()
		mixed_results: Results from perform_mixed_anova_time()
		df: Original DataFrame with weight data
		save_dir: Optional directory to save plots
		show: Whether to display the plots
		
	Returns:
		Dictionary mapping measure names to figure objects
	"""
	print("\n" + "=" * 80)
	print("CREATING INTERACTION PLOTS: SEX × CA%")
	print("=" * 80)
	
	figures = {}
	
	if df is None:
		print("\nNo DataFrame provided - skipping interaction plots")
		return figures
	
	# Clean and prepare dataframe
	cdf = clean_cah_dataframe(df)
	if 'Day' not in cdf.columns:
		cdf = add_day_number_column(cdf)
	
	# Get the measure being analyzed
	measure = None
	if between_results:
		measure = between_results.get('measure', 'Total Change')
	elif mixed_results:
		measure = mixed_results.get('measure', 'Total Change')
	else:
		print("\nNo ANOVA results provided - skipping interaction plots")
		return figures
	
	# Check if measure column exists
	if measure not in cdf.columns:
		print(f"\nColumn '{measure}' not found in dataframe - skipping interaction plots")
		return figures
	
	# Check for significant interaction
	interaction_significant = False
	interaction_p = np.nan
	
	if between_results and 'interaction' in between_results:
		interaction_significant = between_results['interaction'].get('significant', False)
		interaction_p = between_results['interaction'].get('p', np.nan)
	
	# Even if not significant, we can still plot to show the pattern
	print(f"\nCreating interaction plot for: {measure}")
	if not np.isnan(interaction_p):
		sig_marker = '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else 'ns'
		print(f"  Sex × CA% interaction: p = {interaction_p:.4f} {sig_marker}")
	
	# Organize data by Sex and CA% - BETWEEN-SUBJECTS DESIGN
	# Each animal has ONE sex and ONE CA% assignment (not repeated measures on CA%)
	measure_df = cdf[["ID", "Sex", "CA (%)", measure]].copy()
	measure_df = measure_df.dropna()
	
	# Verify between-subjects design: each animal should have only one CA% level
	ca_per_subject = measure_df.groupby('ID')['CA (%)'].nunique()
	if any(ca_per_subject > 1):
		print(f"\n  WARNING: Some animals have multiple CA% levels (within-subjects design detected)")
		print(f"  This plot assumes between-subjects design (one CA% per animal)")
	
	# Compute mean for each subject (averaging over days/timepoints)
	# Since CA% is between-subjects, each ID appears in only ONE CA% group
	subject_means = measure_df.groupby(['ID', 'Sex', 'CA (%)'])[measure].mean().reset_index()
	
	# Get unique CA% levels
	ca_levels = sorted(subject_means['CA (%)'].unique())
	
	print(f"  CA% levels: {ca_levels}")
	print(f"  Unique subjects: {subject_means['ID'].nunique()}")
	print(f"  Between-subjects design: Each animal assigned to one Sex × CA% group")
	
	# Compute means and SEMs for each Sex × CA% combination
	# Error bars represent between-subjects variability (SD across animals in each group)
	males_means = []
	males_sems = []
	females_means = []
	females_sems = []
	
	for ca in ca_levels:
		# Males at this CA% level (independent group of animals)
		m_vals = subject_means[(subject_means['Sex'] == 'M') & (subject_means['CA (%)'] == ca)][measure]
		if len(m_vals) > 0:
			males_means.append(m_vals.mean())
			males_sems.append(m_vals.std(ddof=1) / np.sqrt(len(m_vals)) if len(m_vals) > 1 else 0)
			print(f"  Males at {ca}%: n={len(m_vals)}, M={m_vals.mean():.3f}, SEM={males_sems[-1]:.3f}")
		else:
			males_means.append(np.nan)
			males_sems.append(np.nan)
			print(f"  Males at {ca}%: no data")
		
		# Females at this CA% level (independent group of animals)
		f_vals = subject_means[(subject_means['Sex'] == 'F') & (subject_means['CA (%)'] == ca)][measure]
		if len(f_vals) > 0:
			females_means.append(f_vals.mean())
			females_sems.append(f_vals.std(ddof=1) / np.sqrt(len(f_vals)) if len(f_vals) > 1 else 0)
			print(f"  Females at {ca}%: n={len(f_vals)}, M={f_vals.mean():.3f}, SEM={females_sems[-1]:.3f}")
		else:
			females_means.append(np.nan)
			females_sems.append(np.nan)
			print(f"  Females at {ca}%: no data")
	
	# Create plot with larger size for better readability
	fig, ax = plt.subplots(figsize=(12, 8))
	
	ax.errorbar(ca_levels, males_means, yerr=males_sems,
			   marker='o', markersize=10, linewidth=2.5, capsize=6, capthick=2,
			   color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue',
			   label='Male', linestyle='-', markeredgewidth=2)
	
	ax.errorbar(ca_levels, females_means, yerr=females_sems,
			   marker='s', markersize=10, linewidth=2.5, capsize=6, capthick=2,
			   color='coral', markerfacecolor='lightcoral', markeredgecolor='coral',
			   label='Female', linestyle='--', markeredgewidth=2)
	
	ax.set_xlabel('Citric Acid Concentration (%)', fontsize=14, weight='bold')
	ax.set_ylabel(f'{measure} (g)', fontsize=14, weight='bold')
	
	# Title with p-value if available
	if not np.isnan(interaction_p):
		sig_marker = '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else 'ns'
		ax.set_title(f'Sex × CA% Interaction: {measure}\n(p = {interaction_p:.4f} {sig_marker})',
					fontsize=16, weight='bold', pad=20)
	else:
		ax.set_title(f'Sex × CA% Interaction: {measure}',
					fontsize=16, weight='bold', pad=20)
	
	ax.set_xticks(ca_levels)
	ax.set_xticklabels([f'{ca:.0f}%' for ca in ca_levels], fontsize=12)
	ax.tick_params(axis='y', labelsize=12)
	ax.legend(loc='best', fontsize=13, frameon=True, shadow=True, fancybox=True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=1)
	
	plt.tight_layout(pad=1.5)
	
	if save_dir:
		save_dir = Path(save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		save_path = save_dir / f"CAH_interaction_plot_{measure.replace(' ', '_')}.svg"
		fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
		print(f"\n  [OK] Saved to: {save_path}")
	
	figures[measure] = fig
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	print("=" * 80 + "\n")
	
	return figures


def plot_total_change_by_id(
	df: pd.DataFrame,
	ids: Optional[list] = None,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Create a plot showing Total Change over Day for each animal (ID),
	with colors and markers indicating sex (M=green/square, F=purple/circle).
	
	Adapted from ramp_analysis.py for CAH cohort (no CA% ramping background).
	
	Parameters:
		df: DataFrame with ID, Day, Sex, Total Change columns
		ids: Optional list of IDs to plot (None = all)
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_total_change_series_by_id(df, index="day")
	sex_map = _get_id_sex_map(df)

	# Filter to requested IDs if provided
	if ids is not None:
		series_by_id = {k: v for k, v in series_by_id.items() if k in set(ids)}

	if not series_by_id:
		raise ValueError("No series available to plot. Check input DataFrame and 'ids' filter.")

	fig, ax = plt.subplots(figsize=(11, 6))

	# Plot each ID as a separate line
	for mid, s in series_by_id.items():
		if s.empty:
			continue
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

	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Total Change (g)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Total Weight Change by Day per Animal"
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
		ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=9)
		fig.tight_layout(rect=[0, 0, 0.85, 1])
	else:
		ax.legend(title="ID", loc="best", fontsize=10)
		fig.tight_layout()

	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")

	if save_svg:
		base = svg_filename or (title or "total_change_by_id")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")

	if show:
		plt.show()
	else:
		plt.close(fig)

	return fig


def plot_daily_change_by_id(
	df: pd.DataFrame,
	ids: Optional[list] = None,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Create a plot showing Daily Change over Day for each animal (ID),
	with colors and markers indicating sex (M=green/square, F=purple/circle).
	
	Adapted from ramp_analysis.py for CAH cohort (no CA% ramping background).
	
	Parameters:
		df: DataFrame with ID, Day, Sex, Daily Change columns
		ids: Optional list of IDs to plot (None = all)
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_daily_change_series_by_id(df, index="day")
	sex_map = _get_id_sex_map(df)

	# Filter to requested IDs if provided
	if ids is not None:
		series_by_id = {k: v for k, v in series_by_id.items() if k in set(ids)}

	if not series_by_id:
		raise ValueError("No series available to plot. Check input DataFrame and 'ids' filter.")

	fig, ax = plt.subplots(figsize=(11, 6))

	# Plot each ID as a separate line
	for mid, s in series_by_id.items():
		if s.empty:
			continue
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

	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Daily Change (g)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Daily Weight Change by Day per Animal"
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
		ax.legend(title="ID", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=9)
		fig.tight_layout(rect=[0, 0, 0.85, 1])
	else:
		ax.legend(title="ID", loc="best", fontsize=10)
		fig.tight_layout()

	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")

	if save_svg:
		base = svg_filename or (title or "daily_change_by_id")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")

	if show:
		plt.show()
	else:
		plt.close(fig)

	return fig


def plot_total_change_by_sex(
	df: pd.DataFrame,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Plot sex-averaged Total Change with SEM error bars.


	For each sex (M/F), computes mean and SEM across all animals at each day.
	Uses green for males and purple for females matching individual plots.
	
	Parameters:
		df: DataFrame with ID, Day, Sex, Total Change columns
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_total_change_series_by_id(df, index="day")
	sex_map = _get_id_sex_map(df)
	
	# Separate series by sex
	male_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "M"}
	female_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "F"}
	
	# Compute mean and SEM for each sex
	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		mean = combined.mean(axis=1)
		sem = combined.sem(axis=1)
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
	
	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Total Change (g, Mean ± SEM)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	if title is None:
		title = "Total Weight Change by Sex (Mean ± SEM)"
	ax.set_title(title, fontsize=14, weight='bold')
	
	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)
	
	ax.legend(title="Sex", loc="best", fontsize=11)
	fig.tight_layout()
	
	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
	
	if save_svg:
		base = svg_filename or (title or "total_change_by_sex")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_daily_change_by_sex(
	df: pd.DataFrame,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Plot sex-averaged Daily Change with SEM error bars.
	
	For each sex (M/F), computes mean and SEM across all animals at each day.
	Uses green for males and purple for females matching individual plots.
	
	Parameters:
		df: DataFrame with ID, Day, Sex, Daily Change columns
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_daily_change_series_by_id(df, index="day")
	sex_map = _get_id_sex_map(df)
	
	# Separate series by sex
	male_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "M"}
	female_series = {mid: s for mid, s in series_by_id.items() if sex_map.get(mid) == "F"}
	
	# Compute mean and SEM for each sex
	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		mean = combined.mean(axis=1)
		sem = combined.sem(axis=1)
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
	
	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Daily Change (g, Mean ± SEM)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	if title is None:
		title = "Daily Weight Change by Sex (Mean ± SEM)"
	ax.set_title(title, fontsize=14, weight='bold')
	
	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)
	
	ax.legend(title="Sex", loc="best", fontsize=11)
	fig.tight_layout()
	
	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
	
	if save_svg:
		base = svg_filename or (title or "daily_change_by_sex")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_total_change_by_ca(
	df: pd.DataFrame,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Plot CA%-averaged Total Change with SEM error bars.
	
	For each CA% group (0%, 2%), computes mean and SEM across all animals at each day.
	Uses dodgerblue for 0% and orangered for 2% with triangle markers.
	
	Parameters:
		df: DataFrame with ID, Day, CA (%), Total Change columns
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_total_change_series_by_id(df, index="day")
	
	# Get CA% for each animal
	ca_map = {}
	for animal_id in df["ID"].unique():
		animal_data = df[df["ID"] == animal_id]
		if len(animal_data) > 0 and "CA (%)" in animal_data.columns:
			ca_values = animal_data["CA (%)"].dropna().unique()
			if len(ca_values) > 0:
				ca_map[animal_id] = int(ca_values[0])
	
	# Separate series by CA%
	ca0_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 0}
	ca2_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 2}
	
	# Compute mean and SEM for each CA% group
	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		mean = combined.mean(axis=1)
		sem = combined.sem(axis=1)
		return mean, sem
	
	ca0_mean, ca0_sem = _compute_mean_sem(ca0_series)
	ca2_mean, ca2_sem = _compute_mean_sem(ca2_series)
	
	# Get colors and markers from helper
	ca0_color, ca0_marker = _ca_to_style(0)
	ca2_color, ca2_marker = _ca_to_style(2)
	
	fig, ax = plt.subplots(figsize=(11, 6))
	
	# Plot 0% CA data
	if not ca0_mean.empty:
		ax.plot(ca0_mean.index, ca0_mean.values, label="0% CA", color=ca0_color, marker=ca0_marker,
				markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca0_mean.index,
						ca0_mean - ca0_sem,
						ca0_mean + ca0_sem,
						color=ca0_color, alpha=0.2)
	
	# Plot 2% CA data
	if not ca2_mean.empty:
		ax.plot(ca2_mean.index, ca2_mean.values, label="2% CA", color=ca2_color, marker=ca2_marker,
				markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca2_mean.index,
						ca2_mean - ca2_sem,
						ca2_mean + ca2_sem,
						color=ca2_color, alpha=0.2)
	
	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Total Change (g, Mean ± SEM)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	if title is None:
		title = "Total Weight Change by CA% (Mean ± SEM)"
	ax.set_title(title, fontsize=14, weight='bold')
	
	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)
	
	ax.legend(title="CA%", loc="best", fontsize=11)
	fig.tight_layout()
	
	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
	
	if save_svg:
		base = svg_filename or (title or "total_change_by_ca")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_daily_change_by_ca(
	df: pd.DataFrame,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Plot CA%-averaged Daily Change with SEM error bars.
	
	For each CA% group (0%, 2%), computes mean and SEM across all animals at each day.
	Uses dodgerblue for 0% and orangered for 2% with triangle markers.
	
	Parameters:
		df: DataFrame with ID, Day, CA (%), Daily Change columns
		title: Plot title
		save_path: Optional path to save figure
		show: Whether to display the plot
		save_svg: Whether to save as SVG
		svg_filename: Custom SVG filename
		
	Returns:
		matplotlib Figure object
	"""
	series_by_id = build_daily_change_series_by_id(df, index="day")
	
	# Get CA% for each animal
	ca_map = {}
	for animal_id in df["ID"].unique():
		animal_data = df[df["ID"] == animal_id]
		if len(animal_data) > 0 and "CA (%)" in animal_data.columns:
			ca_values = animal_data["CA (%)"].dropna().unique()
			if len(ca_values) > 0:
				ca_map[animal_id] = int(ca_values[0])
	
	# Separate series by CA%
	ca0_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 0}
	ca2_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 2}
	
	# Compute mean and SEM for each CA% group
	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		mean = combined.mean(axis=1)
		sem = combined.sem(axis=1)
		return mean, sem
	
	ca0_mean, ca0_sem = _compute_mean_sem(ca0_series)
	ca2_mean, ca2_sem = _compute_mean_sem(ca2_series)
	
	# Get colors and markers from helper
	ca0_color, ca0_marker = _ca_to_style(0)
	ca2_color, ca2_marker = _ca_to_style(2)
	
	fig, ax = plt.subplots(figsize=(11, 6))
	
	# Plot 0% CA data
	if not ca0_mean.empty:
		ax.plot(ca0_mean.index, ca0_mean.values, label="0% CA", color=ca0_color, marker=ca0_marker,
				markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca0_mean.index,
						ca0_mean - ca0_sem,
						ca0_mean + ca0_sem,
						color=ca0_color, alpha=0.2)
	
	# Plot 2% CA data
	if not ca2_mean.empty:
		ax.plot(ca2_mean.index, ca2_mean.values, label="2% CA", color=ca2_color, marker=ca2_marker,
				markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca2_mean.index,
						ca2_mean - ca2_sem,
						ca2_mean + ca2_sem,
						color=ca2_color, alpha=0.2)
	
	ax.set_xlabel("Day", fontsize=12)
	ax.set_ylabel("Daily Change (g, Mean ± SEM)", fontsize=12)
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	if title is None:
		title = "Daily Weight Change by CA% (Mean ± SEM)"
	ax.set_title(title, fontsize=14, weight='bold')
	
	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)
	
	ax.legend(title="CA%", loc="best", fontsize=11)
	fig.tight_layout()
	
	if save_path is not None:
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
	
	if save_svg:
		base = svg_filename or (title or "daily_change_by_ca")
		safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(base)).strip("-_.") or "plot"
		if not safe.lower().endswith(".svg"):
			safe += ".svg"
		out_path = Path.cwd() / safe
		fig.savefig(str(out_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {out_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def main():
	"""
	Main function: Load, clean, and summarize CAH cohort data.
	"""
	# Define path to master CSV
	# Adjust this path as needed
	csv_path = Path(__file__).parent.parent / "CAH_cohort" / "master_data_CAH.csv"
	
	print("="*80)
	print("CAH COHORT WEIGHT ANALYSIS")
	print("="*80)
	
	# Load data
	df_raw = load_cah_data(csv_path)
	
	# Clean and process
	print("\nCleaning data...")
	df = clean_cah_dataframe(df_raw)
	
	# Add Day column
	print("Adding per-animal day numbering...")
	df = add_day_number_column(df)
	
	# Summarize
	summarize_dataframe(df)
	
	# Show sample data
	print("\n" + "="*80)
	print("SAMPLE DATA (first 10 rows)")
	print("="*80)
	with pd.option_context('display.max_columns', None, 
						   'display.width', 200,
						   'display.max_colwidth', 30):
		print(df.head(10))
	
	# Example: Show data for one animal
	if "ID" in df.columns and len(df) > 0:
		sample_id = df["ID"].iloc[0]
		print("\n" + "="*80)
		print(f"EXAMPLE: All measurements for {sample_id}")
		print("="*80)
		sample_data = df[df["ID"] == sample_id][["Date", "Day", "Sex", "Strain", "CA (%)", 
												  "Weight", "Daily Change", "Total Change"]]
		with pd.option_context('display.max_rows', None):
			print(sample_data.to_string(index=False))
	
	# Check ANOVA readiness
	print("\n" + "="*80)
	print("ANOVA READINESS CHECK")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\n⚠ WARNING: pingouin not installed. ANOVA will not be available.")
		print("Install with: pip install pingouin")
	else:
		print("\n[OK] pingouin is installed and ready")
	
	# Check for required columns
	required_cols = ["ID", "Sex", "CA (%)", "Day", "Daily Change", "Total Change"]
	missing_cols = [col for col in required_cols if col not in df.columns]
	
	if missing_cols:
		print(f"\n⚠ WARNING: Missing required columns for ANOVA: {missing_cols}")
	else:
		print(f"\n[OK] All required columns present")
	
	# Check for complete cases
	if all(col in df.columns for col in required_cols):
		complete_df = df.dropna(subset=required_cols)
		print(f"\n[OK] Complete cases (non-null): {len(complete_df)} / {len(df)} rows")
		
		# Check experimental design
		subjects_per_ca = complete_df.groupby('ID')['CA (%)'].nunique()
		
		if all(subjects_per_ca == 1):
			print(f"\n[OK] BETWEEN-SUBJECTS DESIGN (CA% is between-subjects)")
			print(f"  Available analyses:")
			print(f"    1. Between-subjects ANOVA: Sex × CA% (at single time point or averaged)")
			print(f"    2. Mixed ANOVA: Time (within) × Sex (between) × CA% (between)")
			print(f"    3. Sex-stratified Mixed ANOVA: Time (within) × CA% (between), holding sex constant")
			print(f"    4. CA%-stratified Mixed ANOVA: Time (within) × Sex (between), holding CA% constant")
			
			# Check time series completeness
			total_days = complete_df['Day'].nunique()
			subjects_per_day = complete_df.groupby('ID')['Day'].nunique()
			complete_time = (subjects_per_day == total_days).sum()
			
			print(f"\n  Time series completeness: {complete_time} / {complete_df['ID'].nunique()} animals")
			if complete_time < complete_df['ID'].nunique():
				print(f"  ⚠ {complete_df['ID'].nunique() - complete_time} animals have missing time points")
				print(f"    (will be excluded from mixed ANOVA)")
		else:
			print(f"\n[OK] WITHIN-SUBJECTS DESIGN (CA% is within-subjects)")
			print(f"  Available analyses:")
			print(f"    1. Mixed ANOVA with CA% as within-subjects factor")
	
	# ========================================================================
	# DEMONSTRATE ANOVA ANALYSES
	# ========================================================================
	
	print("\n" + "="*80)
	print("EXAMPLE ANALYSES")
	print("="*80)
	
	user_input = input("\nWould you like to run example ANOVA analyses? (y/n): ").strip().lower()
	
	if user_input == 'y':
		# Example 1: Between-subjects ANOVA at final time point
		print("\n\n" + "="*80)
		print("EXAMPLE 1: Between-Subjects ANOVA at Final Time Point (Day 27)")
		print("="*80)
		results_final = perform_two_way_between_anova(
			df, 
			measure="Total Change",
			time_point=27,
			average_over_days=False
		)
		
		# Example 2: Between-subjects ANOVA with averaged data
		print("\n\n" + "="*80)
		print("EXAMPLE 2: Between-Subjects ANOVA with Day-Averaged Data")
		print("="*80)
		results_avg = perform_two_way_between_anova(
			df,
			measure="Total Change",
			average_over_days=True
		)
		
		# Example 3: Mixed ANOVA with Time
		print("\n\n" + "="*80)
		print("EXAMPLE 3: Mixed ANOVA - Time × Sex × CA%")
		print("="*80)
		print("Note: Using all available time points for complete analysis")
		results_mixed = perform_mixed_anova_time(
			df,
			measure="Total Change",
			time_points=None  # Use all available days
		)
		
		# Example 4: Sex-stratified Mixed ANOVA - Males only
		print("\n\n" + "="*80)
		print("EXAMPLE 4: Sex-Stratified Mixed ANOVA - Time × CA% (MALES ONLY)")
		print("="*80)
		results_males = perform_mixed_anova_sex_stratified(
			df,
			sex="M",
			measure="Total Change",
			time_points=None
		)
		
		# Example 5: Sex-stratified Mixed ANOVA - Females only
		print("\n\n" + "="*80)
		print("EXAMPLE 5: Sex-Stratified Mixed ANOVA - Time × CA% (FEMALES ONLY)")
		print("="*80)
		results_females = perform_mixed_anova_sex_stratified(
			df,
			sex="F",
			measure="Total Change",
			time_points=None
		)
		
		# Example 6: CA%-stratified Mixed ANOVA - 0% CA only
		print("\n\n" + "="*80)
		print("EXAMPLE 6: CA%-Stratified Mixed ANOVA - Time × Sex (0% CA ONLY)")
		print("="*80)
		results_ca0 = perform_mixed_anova_ca_stratified(
			df,
			ca_percent=0,
			measure="Total Change",
			time_points=None
		)
		
		# Example 7: CA%-stratified Mixed ANOVA - 2% CA only
		print("\n\n" + "="*80)
		print("EXAMPLE 7: CA%-Stratified Mixed ANOVA - Time × Sex (2% CA ONLY)")
		print("="*80)
		results_ca2 = perform_mixed_anova_ca_stratified(
			df,
			ca_percent=2,
			measure="Total Change",
			time_points=None
		)
		
		# Post-hoc tests if main effects are significant
		tukey_results = None
		
		if results_avg and results_avg.get('sex', {}).get('significant'):
			print("\n\nSex effect is significant. Running Tukey HSD...")
			avg_df = df.groupby(["ID", "Sex", "CA (%)"])["Total Change"].mean().reset_index()
			tukey_sex = perform_tukey_hsd(avg_df, "Total Change", "Sex")
		
		if results_avg and results_avg.get('ca_percent', {}).get('significant'):
			print("\n\nCA% effect is significant. Running Tukey HSD...")
			avg_df = df.groupby(["ID", "Sex", "CA (%)"])["Total Change"].mean().reset_index()
			avg_df["CA (%) Group"] = avg_df["CA (%)"].astype(str) + "%"
			tukey_results = perform_tukey_hsd(avg_df, "Total Change", "CA (%) Group")
		
		# Run post-hoc tests for sex-stratified mixed ANOVAs if significant interaction
		mixed_posthoc_males = None
		mixed_posthoc_females = None
		mixed_posthoc_ca0 = None
		mixed_posthoc_ca2 = None
		
		# Check if any of the stratified analyses had significant interactions
		run_posthoc = input("\nWould you like to run post-hoc tests for stratified mixed ANOVAs? (y/n): ").strip().lower()
		
		if run_posthoc == 'y':
			# Post-hoc for males (Time x CA%)
			if results_males and results_males.get('interaction', {}).get('significant'):
				print("\n\nTime × CA% interaction significant in males. Running post-hoc tests...")
				male_df = df[df['Sex'] == 'M'].copy()
				mixed_posthoc_males = perform_mixed_anova_posthoc(
					male_df,
					measure="Total Change",
					within="Day",
					between="CA (%)",
					subject="ID",
					padjust='fdr_bh'
				)
			
			# Post-hoc for females (Time x CA%)
			if results_females and results_females.get('interaction', {}).get('significant'):
				print("\n\nTime × CA% interaction significant in females. Running post-hoc tests...")
				female_df = df[df['Sex'] == 'F'].copy()
				mixed_posthoc_females = perform_mixed_anova_posthoc(
					female_df,
					measure="Total Change",
					within="Day",
					between="CA (%)",
					subject="ID",
					padjust='fdr_bh'
				)
			
			# Post-hoc for 0% CA (Time x Sex)
			if results_ca0 and results_ca0.get('interaction', {}).get('significant'):
				print("\n\nTime × Sex interaction significant at 0% CA. Running post-hoc tests...")
				ca0_df = df[df['CA (%)'] == 0].copy()
				mixed_posthoc_ca0 = perform_mixed_anova_posthoc(
					ca0_df,
					measure="Total Change",
					within="Day",
					between="Sex",
					subject="ID",
					padjust='fdr_bh'
				)
			
			# Post-hoc for 2% CA (Time x Sex)
			if results_ca2 and results_ca2.get('interaction', {}).get('significant'):
				print("\n\nTime × Sex interaction significant at 2% CA. Running post-hoc tests...")
				ca2_df = df[df['CA (%)'] == 2].copy()
				mixed_posthoc_ca2 = perform_mixed_anova_posthoc(
					ca2_df,
					measure="Total Change",
					within="Day",
					between="Sex",
					subject="ID",
					padjust='fdr_bh'
				)
		
		print("\n" + "="*80)
		print("ANALYSIS COMPLETE")
		print("="*80)
		
		# Generate comprehensive report
		print("\n\n" + "="*80)
		print("GENERATING COMPREHENSIVE STATISTICAL REPORT")
		print("="*80)
		
		report = generate_analysis_report(
			between_results=results_avg,
			mixed_results=results_mixed,
			results_males=results_males,
			results_females=results_females,
			results_ca0=results_ca0,
			results_ca2=results_ca2,
			tukey_results=tukey_results,
			mixed_posthoc_males=mixed_posthoc_males,
			mixed_posthoc_females=mixed_posthoc_females,
			mixed_posthoc_ca0=mixed_posthoc_ca0,
			mixed_posthoc_ca2=mixed_posthoc_ca2,
			df=df
		)
		
		print("\n" + report)
		
		# Offer to save report
		save_report = input("\nWould you like to save this report to a file? (y/n): ").strip().lower()
		if save_report == 'y':
			from datetime import datetime
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			report_filename = f"CAH_statistical_report_{timestamp}.txt"
			report_path = Path(__file__).parent / report_filename
			
			with open(report_path, 'w', encoding='utf-8') as f:
				f.write(report)
			
			print(f"\n[OK] Report saved to: {report_path}")
		
		# Generate interaction plots
		print("\n\n" + "="*80)
		print("GENERATING INTERACTION PLOTS")
		print("="*80)
		
		plot_interaction = input("\nWould you like to generate Sex × CA% interaction plots? (y/n): ").strip().lower()
		if plot_interaction == 'y':
			# Determine save directory (same as report if saved)
			save_dir = None
			if save_report == 'y':
				save_dir = Path(__file__).parent
			
			# Generate plots
			interaction_figs = plot_interaction_effects(
				between_results=results_avg,
				mixed_results=results_mixed,
				df=df,
				save_dir=save_dir,
				show=True
			)
			
			if interaction_figs:
				print(f"\n[OK] Generated {len(interaction_figs)} interaction plot(s)")
	
	return df


if __name__ == "__main__":
	df = main()
