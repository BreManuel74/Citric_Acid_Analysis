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
from datetime import datetime

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


def add_week_column(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add a 'Week' column based on Day number, excluding Day 0.
	
	Week assignment:
	- Day 0: Excluded (baseline, not in any week)
	- Days 1-7: Week 1
	- Days 8-14: Week 2
	- Days 15-21: Week 3
	- Days 22-28: Week 4
	- etc.
	
	Parameters:
		df: DataFrame with 'Day' column
		
	Returns:
		DataFrame with added 'Week' column (NaN for Day 0)
	"""
	if "Day" not in df.columns:
		print("[WARNING] 'Day' column not found. Cannot add 'Week' column.")
		return df
	
	df = df.copy()
	
	# Week = 0 for Day 0 (will be filtered out), then Week 1 starts at Day 1
	df["Week"] = np.where(df["Day"] == 0, np.nan, ((df["Day"] - 1) // 7) + 1)
	
	return df


def average_by_week(df: pd.DataFrame, measure: str = "Total Change") -> pd.DataFrame:
	"""
	Average weight measures within each week per animal, excluding Day 0.
	
	Parameters:
		df: DataFrame with ID, Week, and measure columns
		measure: Weight measure to average
		
	Returns:
		DataFrame with one row per ID × Week combination
	"""
	if not {"ID", "Week"}.issubset(df.columns):
		raise ValueError("DataFrame must have 'ID' and 'Week' columns")
	
	if measure not in df.columns:
		raise ValueError(f"Measure '{measure}' not found in DataFrame")
	
	# Exclude Day 0 (Week = NaN)
	df = df[df["Week"].notna()].copy()
	
	# Determine grouping columns (keep metadata like Sex, CA%, Strain)
	meta_cols = ["ID", "Week"]
	for col in ["Sex", "CA (%)", "Strain", "Condition"]:
		if col in df.columns:
			meta_cols.append(col)
	
	# Average the measure within each Week per ID
	weekly_df = (
		df.groupby(meta_cols, dropna=False)[measure]
		.mean()
		.reset_index()
	)
	
	return weekly_df


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
							 time_points: Optional[list] = None,
							 time_unit: str = "day") -> dict:
	"""
	Perform 3-Way Mixed ANOVA: Time (within) × Sex (between) × CA% (between)
	
	This analyzes longitudinal weight changes where:
	- Time (Day or Week): Within-subjects factor (repeated measures over time)
	- Sex: Between-subjects factor (each animal is M or F)
	- CA%: Between-subjects factor (each animal assigned to 0% or 2%)
	
	Parameters:
		df: Cleaned DataFrame with Day or Week column
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific time points to include (Days or Weeks), None = all
		time_unit: 'day' or 'week' - determines which column to use for time
		
	Returns:
		Dictionary with ANOVA results
	"""
	time_col = "Week" if time_unit.lower() == "week" else "Day"
	time_label = "WEEK" if time_unit.lower() == "week" else "DAY"
	
	print("\n" + "="*80)
	print(f"THREE-WAY MIXED ANOVA: TIME ({time_label}, WITHIN) × SEX (BETWEEN) × CA% (BETWEEN)")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Prepare data
	required_cols = ["ID", time_col, "Sex", "CA (%)", measure]
	if not all(col in df.columns for col in required_cols):
		print(f"\nERROR: Missing required columns. Need: {required_cols}")
		return {}
	
	analysis_df = df[required_cols].copy()
	analysis_df = analysis_df.dropna()
	
	# Filter to specific time points if requested
	if time_points is not None:
		analysis_df = analysis_df[analysis_df[time_col].isin(time_points)]
		print(f"\nAnalyzing: {measure} at {time_label}s {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all {time_label.lower()}s")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  {time_label}s: {sorted(analysis_df[time_col].unique())}")
	
	# Check completeness across time points
	subjects_per_time = analysis_df.groupby('ID')[time_col].nunique()
	total_times = analysis_df[time_col].nunique()
	complete_subjects = subjects_per_time[subjects_per_time == total_times].index.tolist()
	incomplete_subjects = subjects_per_time[subjects_per_time < total_times].index.tolist()
	
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
			within=time_col,
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
		print(f"Testing Sex × CA% Interaction at Each {time_label}")
		print("="*60)
		
		time_point_results = {}
		for timepoint in sorted(analysis_df[time_col].unique()):
			tp_df = analysis_df[analysis_df[time_col] == timepoint]
			try:
				tp_aov = pg.anova(data=tp_df, dv=measure, between=['Sex', 'CA (%)'])
				# Determine p-value column for this ANOVA table
				tp_p_col = 'p-unc' if 'p-unc' in tp_aov.columns else 'p_unc'
				interaction_row = tp_aov[tp_aov['Source'] == 'Sex * CA (%)'].iloc[0]
				time_point_results[timepoint] = {
					'F': interaction_row['F'],
					'p': interaction_row[tp_p_col],
					'significant': interaction_row[tp_p_col] < 0.05
				}
				sig_marker = '***' if interaction_row[tp_p_col] < 0.001 else '**' if interaction_row[tp_p_col] < 0.01 else '*' if interaction_row[tp_p_col] < 0.05 else 'ns'
				print(f"  {time_label} {int(timepoint)}: F = {interaction_row['F']:.3f}, p = {interaction_row[tp_p_col]:.4f} {sig_marker}")
			except Exception as e:
				print(f"  {time_label} {int(timepoint)}: Unable to compute ({e})")
				time_point_results[timepoint] = {'error': str(e)}
		
		results = {
			'measure': measure,
			'type': 'mixed_anova',
			'time_unit': time_unit,
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
										time_points: Optional[list] = None,
										time_unit: str = "day") -> dict:
	"""
	Perform 2-Way Mixed ANOVA: Time (within) × CA% (between), holding Sex constant
	
	This analyzes longitudinal weight changes for ONE sex at a time, testing:
	- Time (Day or Week): Within-subjects factor (repeated measures over time)
	- CA%: Between-subjects factor (0% vs 2%)
	
	By stratifying by sex, this reveals whether the Time × CA% interaction differs
	between males and females.
	
	Parameters:
		df: Cleaned DataFrame with Day or Week column
		sex: Sex to analyze ("M" or "F")
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific time points to include (Days or Weeks), None = all
		time_unit: 'day' or 'week' - determines which column to use for time
		
	Returns:
		Dictionary with ANOVA results for the specified sex
	"""
	time_col = "Week" if time_unit.lower() == "week" else "Day"
	time_label = "WEEK" if time_unit.lower() == "week" else "DAY"
	
	print("\n" + "="*80)
	print(f"TWO-WAY MIXED ANOVA (SEX-STRATIFIED): TIME ({time_label}, WITHIN) × CA% (BETWEEN)")
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
	required_cols = ["ID", time_col, "Sex", "CA (%)", measure]
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
		analysis_df = analysis_df[analysis_df[time_col].isin(time_points)]
		print(f"\nAnalyzing: {measure} at {time_label}s {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all {time_label.lower()}s")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  {time_label}s: {sorted(analysis_df[time_col].unique())}")
	print(f"  CA% groups: {sorted(analysis_df['CA (%)'].unique())}")
	
	# Check completeness across time points
	subjects_per_time = analysis_df.groupby('ID')[time_col].nunique()
	total_times = analysis_df[time_col].nunique()
	complete_subjects = subjects_per_time[subjects_per_time == total_times].index.tolist()
	incomplete_subjects = subjects_per_time[subjects_per_time < total_times].index.tolist()
	
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
			within=time_col,
			between='CA (%)',
			subject='ID',
			correction=True  # force GG correction; 'auto' lacks power with n=3
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
		
		# Greenhouse-Geisser correction for within-subjects effects
		def _gg_p(row):
			if row is None:
				return np.nan, False
			p_gg = row.get('p-GG-corr', np.nan)
			if not pd.isna(p_gg):
				return float(p_gg), True
			return float(row[p_col]), False

		time_p, time_gg = _gg_p(time_row)
		ca_p = float(ca_row[p_col]) if ca_row is not None else np.nan
		int_p, int_gg = _gg_p(interaction_row)

		eps_val = np.nan
		if time_row is not None:
			eps_val = time_row.get('eps', np.nan)
		if not pd.isna(eps_val):
			print(f"\nSphericity: Greenhouse-Geisser ε = {float(eps_val):.4f}")
			if float(eps_val) < 0.75:
				print("  ⚠ Sphericity violated — GG correction applied to within-subjects p-values")

		print(f"\nFormatted Results:")
		if time_row is not None:
			sig = '***' if time_p < 0.001 else '**' if time_p < 0.01 else '*' if time_p < 0.05 else 'ns'
			corr_note = " (GG-corrected)" if time_gg else ""
			print(f"  Time: F({time_row['DF1']:.0f},{time_row['DF2']:.0f}) = "
				  f"{time_row['F']:.3f}, p = {time_p:.4f} {sig}{corr_note}")

		if ca_row is not None:
			sig = '***' if ca_p < 0.001 else '**' if ca_p < 0.01 else '*' if ca_p < 0.05 else 'ns'
			print(f"  CA%: F({ca_row['DF1']:.0f},{ca_row['DF2']:.0f}) = "
				  f"{ca_row['F']:.3f}, p = {ca_p:.4f} {sig}")

		if interaction_row is not None:
			sig = '***' if int_p < 0.001 else '**' if int_p < 0.01 else '*' if int_p < 0.05 else 'ns'
			corr_note = " (GG-corrected)" if int_gg else ""
			print(f"  Time × CA%: F({interaction_row['DF1']:.0f},{interaction_row['DF2']:.0f}) = "
				  f"{interaction_row['F']:.3f}, p = {int_p:.4f} {sig}{corr_note}")

		results = {
			'measure': measure,
			'sex': sex,
			'type': 'mixed_anova_sex_stratified',
			'anova_table': aov,
			'time': {
				'F': time_row['F'] if time_row is not None else np.nan,
				'p': time_p,
				'p_unc': float(time_row[p_col]) if time_row is not None else np.nan,
				'gg_corrected': time_gg,
				'eps': float(eps_val) if not pd.isna(eps_val) else np.nan,
				'df1': time_row['DF1'] if time_row is not None else np.nan,
				'df2': time_row['DF2'] if time_row is not None else np.nan,
				'significant': time_p < 0.05 if not np.isnan(time_p) else False
			},
			'ca_percent': {
				'F': ca_row['F'] if ca_row is not None else np.nan,
				'p': ca_p,
				'df1': ca_row['DF1'] if ca_row is not None else np.nan,
				'df2': ca_row['DF2'] if ca_row is not None else np.nan,
				'significant': ca_p < 0.05 if not np.isnan(ca_p) else False
			},
			'interaction': {
				'F': interaction_row['F'] if interaction_row is not None else np.nan,
				'p': int_p,
				'p_unc': float(interaction_row[p_col]) if interaction_row is not None else np.nan,
				'gg_corrected': int_gg,
				'df1': interaction_row['DF1'] if interaction_row is not None else np.nan,
				'df2': interaction_row['DF2'] if interaction_row is not None else np.nan,
				'significant': int_p < 0.05 if not np.isnan(int_p) else False
			}
		}

		return results

	except Exception as e:
		print(f"\nERROR running mixed ANOVA: {e}")
		import traceback
		traceback.print_exc()
		return {}


def perform_mixed_anova_ca_stratified(df: pd.DataFrame, ca_percent: int, measure: str = "Total Change",
									   time_points: Optional[list] = None,
									   time_unit: str = "day") -> dict:
	"""
	Perform 2-Way Mixed ANOVA: Time (within) × Sex (between), holding CA% constant
	
	This analyzes longitudinal weight changes for ONE CA% level at a time, testing:
	- Time (Day or Week): Within-subjects factor (repeated measures over time)
	- Sex: Between-subjects factor (M vs F)
	
	By stratifying by CA%, this reveals whether the Time × Sex interaction differs
	between CA% conditions.
	
	Parameters:
		df: Cleaned DataFrame with Day or Week column
		ca_percent: CA% level to analyze (e.g., 0 or 2)
		measure: Weight measure to analyze ('Weight', 'Daily Change', 'Total Change')
		time_points: List of specific time points to include (Days or Weeks), None = all
		time_unit: 'day' or 'week' - determines which column to use for time
		
	Returns:
		Dictionary with ANOVA results for the specified CA% level
	"""
	time_col = "Week" if time_unit.lower() == "week" else "Day"
	time_label = "WEEK" if time_unit.lower() == "week" else "DAY"
	
	print("\n" + "="*80)
	print(f"TWO-WAY MIXED ANOVA (CA%-STRATIFIED): TIME ({time_label}, WITHIN) × SEX (BETWEEN)")
	print(f"Analyzing: {ca_percent}% CA only")
	print("="*80)
	
	if not HAS_PINGOUIN:
		print("\nERROR: pingouin library required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}
	
	# Prepare data
	required_cols = ["ID", time_col, "Sex", "CA (%)", measure]
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
		analysis_df = analysis_df[analysis_df[time_col].isin(time_points)]
		print(f"\nAnalyzing: {measure} at {time_label}s {time_points}")
	else:
		print(f"\nAnalyzing: {measure} across all {time_label.lower()}s")
	
	print(f"  Total observations: {len(analysis_df)}")
	print(f"  Unique animals: {analysis_df['ID'].nunique()}")
	print(f"  {time_label}s: {sorted(analysis_df[time_col].unique())}")
	print(f"  Sex groups: {sorted(analysis_df['Sex'].unique())}")
	
	# Check completeness across time points
	subjects_per_time = analysis_df.groupby('ID')[time_col].nunique()
	total_times = analysis_df[time_col].nunique()
	complete_subjects = subjects_per_time[subjects_per_time == total_times].index.tolist()
	incomplete_subjects = subjects_per_time[subjects_per_time < total_times].index.tolist()
	
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
			within=time_col,
			between='Sex',
			subject='ID',
			correction=True  # force GG correction; 'auto' lacks power with n=3
		)
		
		print(f"\nMixed ANOVA Table:")
		print(aov.to_string())
		
		# Determine p-value column name (different pingouin versions use different names)
		p_col = 'p-unc' if 'p-unc' in aov.columns else 'p_unc'
		
		# Extract key results
		time_row = aov[aov['Source'] == time_col].iloc[0] if time_col in aov['Source'].values else None
		sex_row = aov[aov['Source'] == 'Sex'].iloc[0] if 'Sex' in aov['Source'].values else None
		interaction_row = aov[aov['Source'] == 'Interaction'].iloc[0] if 'Interaction' in aov['Source'].values else None
		
		# Alternative interaction label check
		if interaction_row is None:
			interaction_label = f'{time_col} * Sex'
			interaction_row = aov[aov['Source'] == interaction_label].iloc[0] if interaction_label in aov['Source'].values else None
		
		# Greenhouse-Geisser correction for within-subjects effects
		def _gg_p(row):
			if row is None:
				return np.nan, False
			p_gg = row.get('p-GG-corr', np.nan)
			if not pd.isna(p_gg):
				return float(p_gg), True
			return float(row[p_col]), False

		time_p, time_gg = _gg_p(time_row)
		sex_p = float(sex_row[p_col]) if sex_row is not None else np.nan
		int_p, int_gg = _gg_p(interaction_row)

		eps_val = np.nan
		if time_row is not None:
			eps_val = time_row.get('eps', np.nan)
		if not pd.isna(eps_val):
			print(f"\nSphericity: Greenhouse-Geisser ε = {float(eps_val):.4f}")
			if float(eps_val) < 0.75:
				print("  ⚠ Sphericity violated — GG correction applied to within-subjects p-values")

		print(f"\nFormatted Results:")
		if time_row is not None:
			sig = '***' if time_p < 0.001 else '**' if time_p < 0.01 else '*' if time_p < 0.05 else 'ns'
			corr_note = " (GG-corrected)" if time_gg else ""
			print(f"  Time: F({time_row['DF1']:.0f},{time_row['DF2']:.0f}) = "
				  f"{time_row['F']:.3f}, p = {time_p:.4f} {sig}{corr_note}")

		if sex_row is not None:
			sig = '***' if sex_p < 0.001 else '**' if sex_p < 0.01 else '*' if sex_p < 0.05 else 'ns'
			print(f"  Sex: F({sex_row['DF1']:.0f},{sex_row['DF2']:.0f}) = "
				  f"{sex_row['F']:.3f}, p = {sex_p:.4f} {sig}")

		if interaction_row is not None:
			sig = '***' if int_p < 0.001 else '**' if int_p < 0.01 else '*' if int_p < 0.05 else 'ns'
			corr_note = " (GG-corrected)" if int_gg else ""
			print(f"  Time × Sex: F({interaction_row['DF1']:.0f},{interaction_row['DF2']:.0f}) = "
				  f"{interaction_row['F']:.3f}, p = {int_p:.4f} {sig}{corr_note}")

		results = {
			'measure': measure,
			'ca_percent': ca_percent,
			'type': 'mixed_anova_ca_stratified',
			'anova_table': aov,
			'time': {
				'F': time_row['F'] if time_row is not None else np.nan,
				'p': time_p,
				'p_unc': float(time_row[p_col]) if time_row is not None else np.nan,
				'gg_corrected': time_gg,
				'eps': float(eps_val) if not pd.isna(eps_val) else np.nan,
				'df1': time_row['DF1'] if time_row is not None else np.nan,
				'df2': time_row['DF2'] if time_row is not None else np.nan,
				'significant': time_p < 0.05 if not np.isnan(time_p) else False
			},
			'sex': {
				'F': sex_row['F'] if sex_row is not None else np.nan,
				'p': sex_p,
				'df1': sex_row['DF1'] if sex_row is not None else np.nan,
				'df2': sex_row['DF2'] if sex_row is not None else np.nan,
				'significant': sex_p < 0.05 if not np.isnan(sex_p) else False
			},
			'interaction': {
				'F': interaction_row['F'] if interaction_row is not None else np.nan,
				'p': int_p,
				'p_unc': float(interaction_row[p_col]) if interaction_row is not None else np.nan,
				'gg_corrected': int_gg,
				'df1': interaction_row['DF1'] if interaction_row is not None else np.nan,
				'df2': interaction_row['DF2'] if interaction_row is not None else np.nan,
				'significant': int_p < 0.05 if not np.isnan(int_p) else False
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
			report_lines.append("Sphericity assumption: Variances of differences between all time points are equal.")
			report_lines.append("When violated (ε < 0.75), Greenhouse-Geisser (GG) correction is applied to p-values.")
			report_lines.append("")
			
			# Check sphericity and Greenhouse-Geisser epsilon
			# Determine time column name
			time_col = 'Week' if mixed_results.get('time_unit', 'day') == 'week' else 'Day'
			eps_rows = aov[aov['Source'] == time_col] if time_col in aov['Source'].values else pd.DataFrame()
			
			if 'sphericity' in aov.columns or 'W-spher' in aov.columns:
				if not eps_rows.empty:
					if 'W-spher' in aov.columns and not pd.isna(eps_rows['W-spher'].iloc[0]):
						w_spher = float(eps_rows['W-spher'].iloc[0])
						p_spher = float(eps_rows['p-spher'].iloc[0]) if 'p-spher' in eps_rows.columns else np.nan
						report_lines.append(f"Mauchly's Test: W = {w_spher:.6f}, p = {p_spher:.4f}")
						if not np.isnan(p_spher):
							if p_spher < 0.05:
								report_lines.append("  → Sphericity VIOLATED (p < 0.05)")
							else:
								report_lines.append("  → Sphericity met (p ≥ 0.05)")
					
					if 'eps' in aov.columns and not pd.isna(eps_rows['eps'].iloc[0]):
						epsilon = float(eps_rows['eps'].iloc[0])
						report_lines.append(f"Greenhouse-Geisser ε = {epsilon:.4f}")
						if epsilon < 0.75:
							report_lines.append(f"  → ε < 0.75: Using GG-corrected p-values for {time_col} and Interaction")
						else:
							report_lines.append(f"  → ε ≥ 0.75: Uncorrected p-values acceptable")
				else:
					report_lines.append("Note: Sphericity information not found in ANOVA table")
			elif 'eps' in aov.columns and not eps_rows.empty and not pd.isna(eps_rows['eps'].iloc[0]):
				epsilon = float(eps_rows['eps'].iloc[0])
				report_lines.append(f"Greenhouse-Geisser ε = {epsilon:.4f}")
				if epsilon < 0.75:
					report_lines.append(f"  → ε < 0.75: Using GG-corrected p-values for {time_col} and Interaction")
				else:
					report_lines.append(f"  → ε ≥ 0.75: Uncorrected p-values acceptable")
			else:
				report_lines.append("Note: Sphericity statistics not available in this analysis")
			
			report_lines.append("")
			
			# Main effects and interactions
			report_lines.append("Main Effects and Interactions:")
			report_lines.append("-" * 80)
			
			# Within-subjects sources where GG correction applies (both Day and Week possible)
			within_sources = {'Day', 'Week', 'Interaction'}
			
			for idx, row in aov.iterrows():
				source = row['Source']
				F = row.get('F', np.nan)
				df1 = row.get('DF1', row.get('df1', np.nan))
				df2 = row.get('DF2', row.get('df2', np.nan))
				
				# Use GG-corrected p for within-subjects effects when available
				p_unc = row.get('p-unc', row.get('p_unc', np.nan))
				p_gg = row.get('p-GG-corr', np.nan)
				use_gg = source in within_sources and not pd.isna(p_gg)
				p = float(p_gg) if use_gg else (float(p_unc) if not pd.isna(p_unc) else np.nan)
				
				if not np.isnan(F) and source not in ['Residual']:
					sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
					corr_note = " (GG-corrected)" if use_gg else ""
					report_lines.append(f"{source}:")
					report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}{corr_note}")
					
					# Effect size
					if 'np2' in row and not pd.isna(row.get('np2', np.nan)):
						report_lines.append(f"  Partial η² = {row['np2']:.3f}")
					
					# Show epsilon for within-subjects effects
					if source in within_sources and 'eps' in row and not pd.isna(row.get('eps', np.nan)):
						report_lines.append(f"  Greenhouse-Geisser ε = {row['eps']:.4f}")
					
					report_lines.append("")
			
			report_lines.append("")
		
		# Time-point specific interactions
		if 'time_point_interactions' in mixed_results:
			time_label = "Week" if mixed_results.get('time_unit', 'day') == 'week' else "Day"
			n_tests = len(mixed_results['time_point_interactions'])
			bonf_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
			
			report_lines.append(f"Time-Point Specific Sex × CA% Interactions:")
			report_lines.append("-" * 80)
			report_lines.append(f"Note: Testing Sex × CA% at each {time_label} separately (k = {n_tests} tests)")
			report_lines.append(f"Bonferroni-corrected α = 0.05 / {n_tests} = {bonf_alpha:.4f}")
			report_lines.append(f"Showing both uncorrected p and Bonferroni significance")
			report_lines.append("")
			
			time_results = mixed_results['time_point_interactions']
			sig_count_raw = 0
			sig_count_bonf = 0
			
			for timepoint in sorted(time_results.keys()):
				result = time_results[timepoint]
				if 'error' in result:
					report_lines.append(f"{time_label} {int(timepoint)}: {result['error']}")
				else:
					F = result.get('F', np.nan)
					p = result.get('p', np.nan)
					sig_raw = p < 0.05
					sig_bonf = p < bonf_alpha
					
					if sig_raw:
						sig_count_raw += 1
					if sig_bonf:
						sig_count_bonf += 1
					
					# Significance markers
					sig_marker_raw = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
					sig_marker_bonf = '***' if p < 0.001 / n_tests else '**' if p < 0.01 / n_tests else '*' if p < bonf_alpha else 'ns'
					
					bonf_note = " (survives Bonferroni)" if sig_bonf else " (does not survive Bonferroni)" if sig_raw else ""
					
					report_lines.append(f"{time_label} {int(timepoint)}: F = {F:.3f}, p = {p:.4f} {sig_marker_raw}{bonf_note}")
			
			report_lines.append("")
			report_lines.append(f"Summary: {sig_count_raw}/{n_tests} significant at α=0.05 (uncorrected)")
			report_lines.append(f"         {sig_count_bonf}/{n_tests} significant at α={bonf_alpha:.4f} (Bonferroni-corrected)")
			
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
						corr_note = " (GG-corrected)" if effect.get('gg_corrected', False) else ""
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}{corr_note}")
						eps = effect.get('eps', np.nan)
						if effect.get('gg_corrected', False) and not pd.isna(eps):
							report_lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
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
						corr_note = " (GG-corrected)" if effect.get('gg_corrected', False) else ""
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}{corr_note}")
						eps = effect.get('eps', np.nan)
						if effect.get('gg_corrected', False) and not pd.isna(eps):
							report_lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
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
		
		# Bonferroni correction for k=2 stratified tests
		n_tests = 2
		bonf_alpha = 0.05 / n_tests
		
		report_lines.append("Time × CA% Interaction:")
		report_lines.append("-" * 80)
		report_lines.append(f"Note: Comparing interaction in k = {n_tests} sex strata")
		report_lines.append(f"Bonferroni-corrected α = 0.05 / {n_tests} = {bonf_alpha:.4f}")
		report_lines.append("")
		
		male_p = male_int.get('p', np.nan)
		female_p = female_int.get('p', np.nan)
		
		male_sig_raw = male_p < 0.05
		female_sig_raw = female_p < 0.05
		male_sig_bonf = male_p < bonf_alpha
		female_sig_bonf = female_p < bonf_alpha
		
		male_marker = '***' if male_p < 0.001 else '**' if male_p < 0.01 else '*' if male_p < 0.05 else 'ns'
		female_marker = '***' if female_p < 0.001 else '**' if female_p < 0.01 else '*' if female_p < 0.05 else 'ns'
		
		male_bonf_note = " (survives Bonferroni)" if male_sig_bonf else " (does not survive Bonferroni)" if male_sig_raw else ""
		female_bonf_note = " (survives Bonferroni)" if female_sig_bonf else " (does not survive Bonferroni)" if female_sig_raw else ""
		
		report_lines.append(f"Males:   F({male_int.get('df1', 0):.0f}, {male_int.get('df2', 0):.0f}) = "
					  f"{male_int.get('F', np.nan):.3f}, p = {male_p:.4f} {male_marker}{male_bonf_note}")
		report_lines.append(f"Females: F({female_int.get('df1', 0):.0f}, {female_int.get('df2', 0):.0f}) = "
					  f"{female_int.get('F', np.nan):.3f}, p = {female_p:.4f} {female_marker}{female_bonf_note}")
		report_lines.append("")
		
		if male_sig_bonf and not female_sig_bonf:
			report_lines.append("→ Time × CA% interaction is significant in MALES but not FEMALES (Bonferroni-corrected)")
			report_lines.append("  This suggests sex differences in how CA% affects weight over time.")
		elif female_sig_bonf and not male_sig_bonf:
			report_lines.append("→ Time × CA% interaction is significant in FEMALES but not MALES (Bonferroni-corrected)")
			report_lines.append("  This suggests sex differences in how CA% affects weight over time.")
		elif male_sig_bonf and female_sig_bonf:
			report_lines.append("→ Time × CA% interaction is significant in BOTH sexes (Bonferroni-corrected)")
			report_lines.append("  Both males and females show CA%-dependent weight trajectories.")
		else:
			report_lines.append("→ Time × CA% interaction is NOT significant in either sex (Bonferroni-corrected)")
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
						corr_note = " (GG-corrected)" if effect.get('gg_corrected', False) else ""
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}{corr_note}")
						eps = effect.get('eps', np.nan)
						if effect.get('gg_corrected', False) and not pd.isna(eps):
							report_lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
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
						corr_note = " (GG-corrected)" if effect.get('gg_corrected', False) else ""
						report_lines.append(f"{effect_name}:")
						report_lines.append(f"  F({df1:.0f}, {df2:.0f}) = {F:.3f}, p = {p:.4f} {sig_marker}{corr_note}")
						eps = effect.get('eps', np.nan)
						if effect.get('gg_corrected', False) and not pd.isna(eps):
							report_lines.append(f"  Greenhouse-Geisser ε = {eps:.4f}")
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
		
		# Bonferroni correction for k=2 stratified tests
		n_tests = 2
		bonf_alpha = 0.05 / n_tests
		
		report_lines.append("Time × Sex Interaction:")
		report_lines.append("-" * 80)
		report_lines.append(f"Note: Comparing interaction in k = {n_tests} CA% strata")
		report_lines.append(f"Bonferroni-corrected α = 0.05 / {n_tests} = {bonf_alpha:.4f}")
		report_lines.append("")
		
		ca0_p = ca0_int.get('p', np.nan)
		ca2_p = ca2_int.get('p', np.nan)
		
		ca0_sig_raw = ca0_p < 0.05
		ca2_sig_raw = ca2_p < 0.05
		ca0_sig_bonf = ca0_p < bonf_alpha
		ca2_sig_bonf = ca2_p < bonf_alpha
		
		ca0_marker = '***' if ca0_p < 0.001 else '**' if ca0_p < 0.01 else '*' if ca0_p < 0.05 else 'ns'
		ca2_marker = '***' if ca2_p < 0.001 else '**' if ca2_p < 0.01 else '*' if ca2_p < 0.05 else 'ns'
		
		ca0_bonf_note = " (survives Bonferroni)" if ca0_sig_bonf else " (does not survive Bonferroni)" if ca0_sig_raw else ""
		ca2_bonf_note = " (survives Bonferroni)" if ca2_sig_bonf else " (does not survive Bonferroni)" if ca2_sig_raw else ""
		
		report_lines.append(f"0% CA:  F({ca0_int.get('df1', 0):.0f}, {ca0_int.get('df2', 0):.0f}) = "
					  f"{ca0_int.get('F', np.nan):.3f}, p = {ca0_p:.4f} {ca0_marker}{ca0_bonf_note}")
		report_lines.append(f"2% CA:  F({ca2_int.get('df1', 0):.0f}, {ca2_int.get('df2', 0):.0f}) = "
					  f"{ca2_int.get('F', np.nan):.3f}, p = {ca2_p:.4f} {ca2_marker}{ca2_bonf_note}")
		report_lines.append("")
		
		if ca0_sig_bonf and not ca2_sig_bonf:
			report_lines.append("→ Time × Sex interaction is significant at 0% CA but NOT at 2% CA (Bonferroni-corrected)")
			report_lines.append("  Sex differences in weight trajectories are present without CA but eliminated with CA.")
		elif ca2_sig_bonf and not ca0_sig_bonf:
			report_lines.append("→ Time × Sex interaction is significant at 2% CA but NOT at 0% CA (Bonferroni-corrected)")
			report_lines.append("  CA exposure reveals sex differences in weight trajectories that aren't present in controls.")
		elif ca0_sig_bonf and ca2_sig_bonf:
			report_lines.append("→ Time × Sex interaction is significant at BOTH CA% levels (Bonferroni-corrected)")
			report_lines.append("  Sex differences in weight trajectories persist across CA% conditions.")
		else:
			report_lines.append("→ Time × Sex interaction is NOT significant at either CA% level (Bonferroni-corrected)")
	
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


def plot_time_by_sex_interaction(
	df: pd.DataFrame,
	measure: str,
	time_col: str = 'Day',
	results: Optional[dict] = None,
	save_dir: Optional[Path] = None,
	show: bool = True
) -> plt.Figure:
	"""
	Plot Time × Sex interaction showing how measure changes over time for each sex.
	
	Parameters:
		df: DataFrame with time_col, Sex, and measure columns
		measure: Name of the measure column to plot (e.g., "Total Change")
		time_col: Name of the time column ('Day' or 'Week')
		results: Optional results dict with interaction p-value
		save_dir: Optional directory to save plot
		show: Whether to display plot
		
	Returns:
		matplotlib Figure object
	"""
	# Group by sex and time, compute mean and SEM
	grouped = df.groupby(['Sex', time_col])[measure].agg(['mean', 'sem']).reset_index()
	
	male_data = grouped[grouped['Sex'] == 'M']
	female_data = grouped[grouped['Sex'] == 'F']
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Plot males
	ax.errorbar(male_data[time_col], male_data['mean'], yerr=male_data['sem'],
			   marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue',
			   label='Male', linestyle='-', markeredgewidth=2, alpha=0.8)
	
	# Plot females
	ax.errorbar(female_data[time_col], female_data['mean'], yerr=female_data['sem'],
			   marker='s', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='coral', markerfacecolor='lightcoral', markeredgecolor='coral',
			   label='Female', linestyle='--', markeredgewidth=2, alpha=0.8)
	
	ax.set_xlabel(time_col, fontsize=14, weight='bold')
	ax.set_ylabel(f'{measure} (%, Mean ± SEM)', fontsize=14, weight='bold')
	
	# Title with p-value if available
	if results and 'interaction' in results:
		p_val = results['interaction'].get('p', np.nan)
		if not np.isnan(p_val):
			sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
			ax.set_title(f'{time_col} × Sex Interaction: {measure}\n(p = {p_val:.4f} {sig_marker})',
						fontsize=16, weight='bold', pad=20)
		else:
			ax.set_title(f'{time_col} × Sex Interaction: {measure}', fontsize=16, weight='bold', pad=20)
	else:
		ax.set_title(f'{time_col} × Sex Interaction: {measure}', fontsize=16, weight='bold', pad=20)
	
	ax.tick_params(axis='both', labelsize=12, direction='in', length=6)
	ax.legend(loc='best', fontsize=13, frameon=True, shadow=True, fancybox=True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.grid(False)
	
	# Ensure x-axis shows whole numbers only
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	# Set y-axis to start on a tick mark
	ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins='auto', prune=None))
	ymin, ymax = ax.get_ylim()
	ticks = ax.yaxis.get_major_locator().tick_values(ymin, ymax)
	if len(ticks) > 0:
		ax.set_ylim(bottom=ticks[0])
	
	plt.tight_layout(pad=1.5)
	
	if save_dir:
		save_dir = Path(save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		safe_measure = measure.replace(' ', '_')
		save_path = save_dir / f"CAH_{time_col}_by_Sex_{safe_measure}.svg"
		fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
		print(f"  [OK] Saved to: {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_time_by_ca_interaction(
	df: pd.DataFrame,
	measure: str,
	time_col: str = 'Day',
	results: Optional[dict] = None,
	save_dir: Optional[Path] = None,
	show: bool = True
) -> plt.Figure:
	"""
	Plot Time × CA% interaction showing how measure changes over time for each CA% level.
	
	Parameters:
		df: DataFrame with time_col, CA (%), and measure columns
		measure: Name of the measure column to plot (e.g., "Total Change")
		time_col: Name of the time column ('Day' or 'Week')
		results: Optional results dict with interaction p-value
		save_dir: Optional directory to save plot
		show: Whether to display plot
		
	Returns:
		matplotlib Figure object
	"""
	# Group by CA% and time, compute mean and SEM
	grouped = df.groupby(['CA (%)', time_col])[measure].agg(['mean', 'sem']).reset_index()
	
	ca0_data = grouped[grouped['CA (%)'] == 0]
	ca2_data = grouped[grouped['CA (%)'] == 2]
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Plot 0% CA
	ax.errorbar(ca0_data[time_col], ca0_data['mean'], yerr=ca0_data['sem'],
			   marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='forestgreen', markerfacecolor='lightgreen', markeredgecolor='forestgreen',
			   label='0% CA', linestyle='-', markeredgewidth=2, alpha=0.8)
	
	# Plot 2% CA
	ax.errorbar(ca2_data[time_col], ca2_data['mean'], yerr=ca2_data['sem'],
			   marker='s', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='crimson', markerfacecolor='lightcoral', markeredgecolor='crimson',
			   label='2% CA', linestyle='--', markeredgewidth=2, alpha=0.8)
	
	ax.set_xlabel(time_col, fontsize=14, weight='bold')
	ax.set_ylabel(f'{measure} (%, Mean ± SEM)', fontsize=14, weight='bold')
	
	# Title with p-value if available
	if results and 'interaction' in results:
		p_val = results['interaction'].get('p', np.nan)
		if not np.isnan(p_val):
			sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
			ax.set_title(f'{time_col} × CA% Interaction: {measure}\n(p = {p_val:.4f} {sig_marker})',
						fontsize=16, weight='bold', pad=20)
		else:
			ax.set_title(f'{time_col} × CA% Interaction: {measure}', fontsize=16, weight='bold', pad=20)
	else:
		ax.set_title(f'{time_col} × CA% Interaction: {measure}', fontsize=16, weight='bold', pad=20)
	
	ax.tick_params(axis='both', labelsize=12, direction='in', length=6)
	ax.legend(loc='best', fontsize=13, frameon=True, shadow=True, fancybox=True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.grid(False)
	
	# Ensure x-axis shows whole numbers only
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	# Set y-axis to start on a tick mark
	ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins='auto', prune=None))
	ymin, ymax = ax.get_ylim()
	ticks = ax.yaxis.get_major_locator().tick_values(ymin, ymax)
	if len(ticks) > 0:
		ax.set_ylim(bottom=ticks[0])
	
	plt.tight_layout(pad=1.5)
	
	if save_dir:
		save_dir = Path(save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		safe_measure = measure.replace(' ', '_')
		save_path = save_dir / f"CAH_{time_col}_by_CA_{safe_measure}.svg"
		fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
		print(f"  [OK] Saved to: {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_time_by_ca_stratified(
	df: pd.DataFrame,
	measure: str,
	sex: str,
	time_col: str = 'Day',
	results: Optional[dict] = None,
	save_dir: Optional[Path] = None,
	show: bool = True
) -> plt.Figure:
	"""
	Plot Time × CA% interaction for a specific sex (sex-stratified analysis).
	
	Parameters:
		df: DataFrame with time_col, CA (%), and measure columns (already filtered by sex)
		measure: Name of the measure column to plot
		sex: Sex being analyzed ('M' or 'F')
		time_col: Name of the time column ('Day' or 'Week')
		results: Optional results dict with interaction p-value
		save_dir: Optional directory to save plot
		show: Whether to display plot
		
	Returns:
		matplotlib Figure object
	"""
	# Group by CA% and time, compute mean and SEM
	grouped = df.groupby(['CA (%)', time_col])[measure].agg(['mean', 'sem']).reset_index()
	
	ca0_data = grouped[grouped['CA (%)'] == 0]
	ca2_data = grouped[grouped['CA (%)'] == 2]
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Plot 0% CA
	ax.errorbar(ca0_data[time_col], ca0_data['mean'], yerr=ca0_data['sem'],
			   marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='forestgreen', markerfacecolor='lightgreen', markeredgecolor='forestgreen',
			   label='0% CA', linestyle='-', markeredgewidth=2, alpha=0.8)
	
	# Plot 2% CA
	ax.errorbar(ca2_data[time_col], ca2_data['mean'], yerr=ca2_data['sem'],
			   marker='s', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='crimson', markerfacecolor='lightcoral', markeredgecolor='crimson',
			   label='2% CA', linestyle='--', markeredgewidth=2, alpha=0.8)
	
	sex_label = 'Males' if sex == 'M' else 'Females'
	ax.set_xlabel(time_col, fontsize=14, weight='bold')
	ax.set_ylabel(f'{measure} (%, Mean ± SEM)', fontsize=14, weight='bold')
	
	# Title with p-value if available
	if results and 'interaction' in results:
		p_val = results['interaction'].get('p', np.nan)
		if not np.isnan(p_val):
			sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
			ax.set_title(f'{time_col} × CA% Interaction in {sex_label}: {measure}\n(p = {p_val:.4f} {sig_marker})',
						fontsize=16, weight='bold', pad=20)
		else:
			ax.set_title(f'{time_col} × CA% Interaction in {sex_label}: {measure}', fontsize=16, weight='bold', pad=20)
	else:
		ax.set_title(f'{time_col} × CA% Interaction in {sex_label}: {measure}', fontsize=16, weight='bold', pad=20)
	
	ax.tick_params(axis='both', labelsize=12, direction='in', length=6)
	ax.legend(loc='best', fontsize=13, frameon=True, shadow=True, fancybox=True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.grid(False)
	
	# Ensure x-axis shows whole numbers only
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	# Set y-axis to start on a tick mark
	ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins='auto', prune=None))
	ymin, ymax = ax.get_ylim()
	ticks = ax.yaxis.get_major_locator().tick_values(ymin, ymax)
	if len(ticks) > 0:
		ax.set_ylim(bottom=ticks[0])
	
	plt.tight_layout(pad=1.5)
	
	if save_dir:
		save_dir = Path(save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		safe_measure = measure.replace(' ', '_')
		save_path = save_dir / f"CAH_{time_col}_by_CA_{sex_label}_{safe_measure}.svg"
		fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
		print(f"  [OK] Saved to: {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def plot_time_by_sex_stratified(
	df: pd.DataFrame,
	measure: str,
	ca_percent: int,
	time_col: str = 'Day',
	results: Optional[dict] = None,
	save_dir: Optional[Path] = None,
	show: bool = True
) -> plt.Figure:
	"""
	Plot Time × Sex interaction for a specific CA% level (CA%-stratified analysis).
	
	Parameters:
		df: DataFrame with time_col, Sex, and measure columns (already filtered by CA%)
		measure: Name of the measure column to plot
		ca_percent: CA% level being analyzed (0 or 2)
		time_col: Name of the time column ('Day' or 'Week')
		results: Optional results dict with interaction p-value
		save_dir: Optional directory to save plot
		show: Whether to display plot
		
	Returns:
		matplotlib Figure object
	"""
	# Group by sex and time, compute mean and SEM
	grouped = df.groupby(['Sex', time_col])[measure].agg(['mean', 'sem']).reset_index()
	
	male_data = grouped[grouped['Sex'] == 'M']
	female_data = grouped[grouped['Sex'] == 'F']
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Plot males
	ax.errorbar(male_data[time_col], male_data['mean'], yerr=male_data['sem'],
			   marker='o', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue',
			   label='Male', linestyle='-', markeredgewidth=2, alpha=0.8)
	
	# Plot females
	ax.errorbar(female_data[time_col], female_data['mean'], yerr=female_data['sem'],
			   marker='s', markersize=8, linewidth=2.5, capsize=5, capthick=2,
			   color='coral', markerfacecolor='lightcoral', markeredgecolor='coral',
			   label='Female', linestyle='--', markeredgewidth=2, alpha=0.8)
	
	ax.set_xlabel(time_col, fontsize=14, weight='bold')
	ax.set_ylabel(f'{measure} (%, Mean ± SEM)', fontsize=14, weight='bold')
	
	# Title with p-value if available
	if results and 'interaction' in results:
		p_val = results['interaction'].get('p', np.nan)
		if not np.isnan(p_val):
			sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
			ax.set_title(f'{time_col} × Sex Interaction at {ca_percent}% CA: {measure}\n(p = {p_val:.4f} {sig_marker})',
						fontsize=16, weight='bold', pad=20)
		else:
			ax.set_title(f'{time_col} × Sex Interaction at {ca_percent}% CA: {measure}', fontsize=16, weight='bold', pad=20)
	else:
		ax.set_title(f'{time_col} × Sex Interaction at {ca_percent}% CA: {measure}', fontsize=16, weight='bold', pad=20)
	
	ax.tick_params(axis='both', labelsize=12, direction='in', length=6)
	ax.legend(loc='best', fontsize=13, frameon=True, shadow=True, fancybox=True)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.grid(False)
	
	# Ensure x-axis shows whole numbers only
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
	
	# Set y-axis to start on a tick mark
	ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins='auto', prune=None))
	ymin, ymax = ax.get_ylim()
	ticks = ax.yaxis.get_major_locator().tick_values(ymin, ymax)
	if len(ticks) > 0:
		ax.set_ylim(bottom=ticks[0])
	
	plt.tight_layout(pad=1.5)
	
	if save_dir:
		save_dir = Path(save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		safe_measure = measure.replace(' ', '_')
		save_path = save_dir / f"CAH_{time_col}_by_Sex_at_{ca_percent}pctCA_{safe_measure}.svg"
		fig.savefig(save_path, format='svg', dpi=200, bbox_inches='tight')
		print(f"  [OK] Saved to: {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


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
	ax.set_ylabel(f'{measure} (%)', fontsize=14, weight='bold')
	
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
	ax.set_ylabel("Total Change (%)", fontsize=12)
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
	ax.set_ylabel("Daily Change (%)", fontsize=12)
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
	ax.set_ylabel("Total Change (%, Mean ± SEM)", fontsize=12)
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
	ax.set_ylabel("Daily Change (%, Mean ± SEM)", fontsize=12)
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
	ax.set_ylabel("Total Change (%, Mean ± SEM)", fontsize=12)
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
	ax.set_ylabel("Daily Change (%, Mean ± SEM)", fontsize=12)
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


# ==============================================================================
# SLOPE ANALYSIS: COMPARING RATE OF WEIGHT CHANGE BETWEEN CA% GROUPS
# ==============================================================================

def calculate_animal_slopes_cah(
	df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week"
) -> pd.DataFrame:
	"""
	Calculate linear regression slopes for each animal's weight change over time.
	
	For each animal, fits: measure ~ time_unit (e.g., Total Change ~ Week)
	Returns slope (rate of change), R², p-value, etc.
	
	Parameters:
		df: Cleaned DataFrame with ID, Sex, CA (%), Day/Week, and measure columns
		measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
		time_unit: Time variable to use ('Week' or 'Day')
		
	Returns:
		DataFrame with one row per animal containing slope statistics
	"""
	from scipy import stats
	
	print("\n" + "="*80)
	print(f"CALCULATING ANIMAL SLOPES: {measure} ~ {time_unit}")
	print("="*80)
	
	# Ensure required columns exist
	required_cols = ['ID', 'Sex', 'CA (%)', measure]
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	
	# Ensure time column exists
	if time_unit not in df.columns:
		if time_unit == "Week" and "Day" in df.columns:
			print(f"[INFO] Week column not found. Adding Week column...")
			df = add_week_column(df)
		elif time_unit == "Day" and "Day" not in df.columns:
			print(f"[INFO] Day column not found. Adding Day column...")
			df = add_day_number_column(df)
		else:
			raise ValueError(f"Time column '{time_unit}' not found and cannot be created")
	
	# Filter to rows with valid time and measure values
	analysis_df = df[[col for col in ['ID', 'Sex', 'CA (%)', time_unit, measure] if col in df.columns]].copy()
	analysis_df = analysis_df.dropna()
	
	# If using weeks, exclude Week 0 (baseline)
	if time_unit == "Week":
		analysis_df = analysis_df[analysis_df[time_unit] > 0].copy()
	
	print(f"  Animals: {analysis_df['ID'].nunique()}")
	print(f"  CA% groups: {sorted(analysis_df['CA (%)'].unique())}")
	print(f"  {time_unit} range: {analysis_df[time_unit].min():.0f} to {analysis_df[time_unit].max():.0f}")
	
	# Calculate slope for each animal
	slopes_data = []
	
	for animal_id in analysis_df['ID'].unique():
		animal_data = analysis_df[analysis_df['ID'] == animal_id].copy()
		
		# Get metadata
		sex = animal_data['Sex'].iloc[0]
		ca_pct = animal_data['CA (%)'].iloc[0]
		
		# Get time and measure arrays
		x = animal_data[time_unit].values
		y = animal_data[measure].values
		
		if len(x) < 2:
			# Need at least 2 points for regression
			continue
		
		# Perform linear regression
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
		
		slopes_data.append({
			'ID': animal_id,
			'Sex': sex,
			'CA (%)': ca_pct,
			'Slope': slope,
			'Intercept': intercept,
			'R2': r_value**2,
			'P_value': p_value,
			'Std_Error': std_err,
			'N_points': len(x)
		})
	
	slopes_df = pd.DataFrame(slopes_data)
	
	# Print summary by CA% group
	print(f"\nSlope Summary by CA% Group:")
	for ca_val in sorted(slopes_df['CA (%)'].unique()):
		ca_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope']
		print(f"  {ca_val}% CA (n={len(ca_slopes)}):")
		print(f"    Mean slope:   {ca_slopes.mean():.4f} {measure} per {time_unit}")
		print(f"    Median slope: {ca_slopes.median():.4f} {measure} per {time_unit}")
		print(f"    SD:           {ca_slopes.std():.4f}")
		print(f"    Range:        [{ca_slopes.min():.4f}, {ca_slopes.max():.4f}]")
		r2_values = slopes_df[slopes_df['CA (%)'] == ca_val]['R2']
		print(f"    Mean R²:      {r2_values.mean():.4f}")
	
	return slopes_df


def compare_slopes_within_ca_groups(slopes_df: pd.DataFrame) -> Dict:
	"""
	Analyze slope variability within each CA% group using descriptive statistics.
	
	This analyzes the variability of slopes within each CA% group separately,
	providing detailed descriptive statistics.
	
	Parameters:
		slopes_df: DataFrame from calculate_animal_slopes_cah()
		
	Returns:
		Dictionary with within-group statistics
	"""
	print("\n" + "="*80)
	print("WITHIN-GROUP SLOPE VARIABILITY ANALYSIS")
	print("="*80)
	
	results = {
		'group_stats': []
	}
	
	# Descriptive statistics by CA% group
	for ca_val in sorted(slopes_df['CA (%)'].unique()):
		group_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope'].values
		
		group_stat = {
			'CA (%)': ca_val,
			'N': len(group_slopes),
			'Mean': group_slopes.mean(),
			'Median': np.median(group_slopes),
			'SD': group_slopes.std(),
			'SEM': group_slopes.std() / np.sqrt(len(group_slopes)),
			'Min': group_slopes.min(),
			'Max': group_slopes.max(),
			'IQR': np.percentile(group_slopes, 75) - np.percentile(group_slopes, 25),
			'CV': (group_slopes.std() / group_slopes.mean() * 100) if group_slopes.mean() != 0 else np.nan
		}
		
		results['group_stats'].append(group_stat)
		
		print(f"\n{ca_val}% CA Group (n={group_stat['N']}):")
		print(f"  Mean ± SEM:         {group_stat['Mean']:.4f} ± {group_stat['SEM']:.4f}")
		print(f"  Median (IQR):       {group_stat['Median']:.4f} ({group_stat['IQR']:.4f})")
		print(f"  SD:                 {group_stat['SD']:.4f}")
		print(f"  Coefficient of Var: {group_stat['CV']:.2f}%")
		print(f"  Range:              [{group_stat['Min']:.4f}, {group_stat['Max']:.4f}]")
	
	return results


def compare_slopes_between_ca_groups(slopes_df: pd.DataFrame) -> Dict:
	"""
	Statistically compare average slopes between CA% groups.
	
	Performs:
	1. Independent samples t-test (or Welch's t-test if variances unequal)
	2. Mann-Whitney U test (non-parametric alternative)
	3. Effect size calculation (Cohen's d)
	
	Parameters:
		slopes_df: DataFrame from calculate_animal_slopes_cah()
		
	Returns:
		Dictionary with test results and effect sizes
	"""
	from scipy import stats
	
	print("\n" + "="*80)
	print("BETWEEN-GROUP SLOPE COMPARISON: 0% CA vs 2% CA")
	print("="*80)
	
	ca_groups = sorted(slopes_df['CA (%)'].unique())
	
	if len(ca_groups) != 2:
		print(f"Warning: Expected 2 CA% groups, found {len(ca_groups)}. Returning empty results.")
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
	t_stat, t_p = stats.ttest_ind(slopes_0, slopes_1, equal_var=False)
	
	print(f"\nWelch's T-Test (unequal variances):")
	print(f"  t = {t_stat:.4f}, p = {t_p:.4f}")
	
	if t_p < 0.001:
		sig_str = "p < 0.001 (highly significant)"
	elif t_p < 0.01:
		sig_str = "p < 0.01 (very significant)"
	elif t_p < 0.05:
		sig_str = "p < 0.05 (significant)"
	else:
		sig_str = "p ≥ 0.05 (not significant)"
	
	print(f"  Result: {sig_str}")
	
	# Calculate Welch-Satterthwaite degrees of freedom
	s1_sq = slopes_0.var()
	s2_sq = slopes_1.var()
	n1 = len(slopes_0)
	n2 = len(slopes_1)
	df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
	
	results['t_test'] = {
		'statistic': t_stat,
		'p_value': t_p,
		'df': df,
		'significant': t_p < 0.05
	}
	
	# 2. Mann-Whitney U test (non-parametric)
	u_stat, u_p = stats.mannwhitneyu(slopes_0, slopes_1, alternative='two-sided')
	
	print(f"\nMann-Whitney U Test (non-parametric):")
	print(f"  U = {u_stat:.4f}, p = {u_p:.4f}")
	print(f"  Result: {'Significant' if u_p < 0.05 else 'Not significant'} (α = 0.05)")
	
	results['mann_whitney'] = {
		'statistic': u_stat,
		'p_value': u_p,
		'significant': u_p < 0.05
	}
	
	# 3. Effect size (Cohen's d)
	# Pooled standard deviation
	n1 = len(slopes_0)
	n2 = len(slopes_1)
	s1 = slopes_0.std(ddof=1)
	s2 = slopes_1.std(ddof=1)
	pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
	
	cohens_d = (slopes_1.mean() - slopes_0.mean()) / pooled_sd
	
	# Interpret effect size
	abs_d = abs(cohens_d)
	if abs_d < 0.2:
		interpretation = "negligible"
	elif abs_d < 0.5:
		interpretation = "small"
	elif abs_d < 0.8:
		interpretation = "medium"
	else:
		interpretation = "large"
	
	print(f"\nEffect Size (Cohen's d):")
	print(f"  d = {cohens_d:.4f}")
	print(f"  Interpretation: {interpretation.capitalize()} effect size")
	
	results['effect_size'] = {
		'cohens_d': cohens_d,
		'pooled_sd': pooled_sd,
		'interpretation': interpretation
	}
	
	# 4. 95% Confidence Interval for mean difference (using Welch df)
	# Use unequal variance formula for SE
	se_diff = np.sqrt(slopes_0.var()/n1 + slopes_1.var()/n2)
	t_crit = stats.t.ppf(0.975, df)
	ci_lower = results['mean_diff'] - t_crit * se_diff
	ci_upper = results['mean_diff'] + t_crit * se_diff
	
	print(f"\n95% Confidence Interval for Mean Difference:")
	print(f"  Mean Difference: {results['mean_diff']:.4f}")
	print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
	
	if ci_lower * ci_upper > 0:
		print(f"  Interpretation: CI does not include zero (significant difference)")
	else:
		print(f"  Interpretation: CI includes zero (no significant difference)")
	
	results['confidence_interval'] = {
		'mean_diff': results['mean_diff'],
		'se_diff': se_diff,
		't_critical': t_crit,
		'ci_95_lower': ci_lower,
		'ci_95_upper': ci_upper
	}
	
	return results


def plot_slopes_comparison_cah(
	slopes_df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True
) -> plt.Figure:
	"""
	Create visualization comparing slopes between CA% groups.
	
	Creates a 3-panel figure:
	1. Boxplot with individual points
	2. Bar chart with error bars (Mean ± SEM)
	3. Histogram overlay
	
	Parameters:
		slopes_df: DataFrame from calculate_animal_slopes_cah()
		measure: Weight measure analyzed
		time_unit: Time unit used
		title: Optional custom title
		save_path: Optional path to save figure
		show: Whether to display the plot
		
	Returns:
		matplotlib Figure object
	"""
	print("\n" + "="*80)
	print("CREATING SLOPE COMPARISON PLOTS")
	print("="*80)
	
	fig, axes = plt.subplots(1, 3, figsize=(16, 5))
	
	ca_groups = sorted(slopes_df['CA (%)'].unique())
	colors = ['dodgerblue', 'orangered']
	
	# Panel 1: Boxplot with individual points
	ax1 = axes[0]
	
	# Create boxplot
	box_data = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values for ca in ca_groups]
	bp = ax1.boxplot(box_data, positions=range(len(ca_groups)), widths=0.6,
					 patch_artist=True, showfliers=False)
	
	# Color boxes
	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.6)
	
	# Overlay individual points (jittered)
	for i, ca in enumerate(ca_groups):
		slopes = slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values
		x = np.random.normal(i, 0.04, size=len(slopes))
		ax1.scatter(x, slopes, alpha=0.5, color=colors[i], s=50, edgecolors='black', linewidth=0.5)
	
	ax1.set_xticks(range(len(ca_groups)))
	ax1.set_xticklabels([f'{ca}% CA' for ca in ca_groups])
	ax1.set_ylabel(f'Slope ({measure} per {time_unit})')
	ax1.set_title('Slope Distribution by Group')
	ax1.grid(False)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	
	# Panel 2: Bar chart with error bars
	ax2 = axes[1]
	
	means = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].mean() for ca in ca_groups]
	sems = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].sem() for ca in ca_groups]
	
	bars = ax2.bar(range(len(ca_groups)), means, yerr=sems, capsize=5,
				   color=colors, alpha=0.6, edgecolor='black', linewidth=1)
	
	ax2.set_xticks(range(len(ca_groups)))
	ax2.set_xticklabels([f'{ca}% CA' for ca in ca_groups])
	ax2.set_ylabel(f'Mean Slope ± SEM ({measure} per {time_unit})')
	ax2.set_title('Mean Slopes with Error Bars')
	ax2.grid(False)
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	
	# Panel 3: Histogram overlay
	ax3 = axes[2]
	
	for i, ca in enumerate(ca_groups):
		slopes = slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values
		ax3.hist(slopes, bins=10, alpha=0.5, color=colors[i], label=f'{ca}% CA', edgecolor='black')
	
	ax3.set_xlabel(f'Slope ({measure} per {time_unit})')
	ax3.set_ylabel('Frequency')
	ax3.set_title('Slope Distribution Histogram')
	ax3.legend()
	ax3.grid(False)
	ax3.spines['top'].set_visible(False)
	ax3.spines['right'].set_visible(False)
	
	# Overall title
	if title is None:
		title = f"Slope Comparison: {measure} ~ {time_unit}"
	fig.suptitle(title, fontsize=14, fontweight='bold')
	
	plt.tight_layout()
	
	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"\n[OK] Plot saved to: {save_path}")
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return fig


def generate_slope_analysis_report_cah(
	slopes_df: pd.DataFrame,
	within_results: Dict,
	between_results: Dict,
	measure: str = "Total Change",
	time_unit: str = "Week"
) -> str:
	"""
	Generate comprehensive text report of slope analysis results.
	
	Parameters:
		slopes_df: DataFrame from calculate_animal_slopes_cah()
		within_results: Dictionary from compare_slopes_within_ca_groups()
		between_results: Dictionary from compare_slopes_between_ca_groups()
		measure: Weight measure analyzed
		time_unit: Time unit used
		
	Returns:
		Formatted text report
	"""
	lines = []
	
	lines.append("="*80)
	lines.append("SLOPE ANALYSIS REPORT: CAH COHORT")
	lines.append("="*80)
	lines.append(f"\nMeasure: {measure}")
	lines.append(f"Time Unit: {time_unit}")
	lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
	
	# Section 1: Individual animal slopes
	lines.append("\n\n" + "="*80)
	lines.append("SECTION 1: INDIVIDUAL ANIMAL SLOPES")
	lines.append("="*80)
	lines.append(f"\nLinear regression: {measure} ~ {time_unit}")
	
	for ca_val in sorted(slopes_df['CA (%)'].unique()):
		ca_data = slopes_df[slopes_df['CA (%)'] == ca_val].sort_values('Slope', ascending=False)
		
		lines.append(f"\n{ca_val}% CA Group (n={len(ca_data)}):")
		lines.append("-"*80)
		lines.append(f"{'ID':<15} {'Sex':<6} {'Slope':>10} {'R²':>8} {'P-value':>10} {'N Points':>10}")
		lines.append("-"*80)
		
		for _, row in ca_data.iterrows():
			lines.append(f"{str(row['ID']):<15} {row['Sex']:<6} {row['Slope']:>10.4f} {row['R2']:>8.4f} {row['P_value']:>10.4f} {row['N_points']:>10.0f}")
	
	# Section 2: Within-group variability
	lines.append("\n\n" + "="*80)
	lines.append("SECTION 2: WITHIN-GROUP VARIABILITY")
	lines.append("="*80)
	lines.append("\nThis section analyzes the variability of slopes within each CA% group.")
	
	for group_stat in within_results['group_stats']:
		lines.append(f"\n{group_stat['CA (%)']}% CA Group (n={group_stat['N']}):")
		lines.append("-"*80)
		lines.append(f"  Mean:               {group_stat['Mean']:.4f}")
		lines.append(f"  Median:             {group_stat['Median']:.4f}")
		lines.append(f"  Standard Deviation: {group_stat['SD']:.4f}")
		lines.append(f"  SEM:                {group_stat['SEM']:.4f}")
		lines.append(f"  Min:                {group_stat['Min']:.4f}")
		lines.append(f"  Max:                {group_stat['Max']:.4f}")
		lines.append(f"  IQR:                {group_stat['IQR']:.4f}")
		lines.append(f"  Coefficient of Var: {group_stat['CV']:.2f}%")
	
	# Section 3: Between-group comparison
	lines.append("\n\n" + "="*80)
	lines.append("SECTION 3: BETWEEN-GROUP COMPARISON")
	lines.append("="*80)
	lines.append("\nThis section compares the average slopes between 0% CA and 2% CA groups.")
	
	if between_results and 'ca_groups' in between_results:
		ca_0 = between_results['ca_groups'][0]
		ca_1 = between_results['ca_groups'][1]
		
		lines.append(f"\nGroup Comparison: {ca_0}% CA vs {ca_1}% CA")
		lines.append("-"*80)
		lines.append(f"  {ca_0}% CA: Mean = {between_results['mean_0']:.4f}, SD = {between_results['sd_0']:.4f} (n={between_results['n_0']})")
		lines.append(f"  {ca_1}% CA: Mean = {between_results['mean_1']:.4f}, SD = {between_results['sd_1']:.4f} (n={between_results['n_1']})")
		lines.append(f"  Difference in means: {between_results['mean_diff']:.4f}")
		
		# T-test results
		lines.append("\n" + "-"*80)
		lines.append("Welch's T-Test (unequal variances):")
		lines.append("-"*80)
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
		lines.append("\n" + "-"*80)
		lines.append("Mann-Whitney U Test (Non-parametric):")
		lines.append("-"*80)
		mw = between_results['mann_whitney']
		lines.append(f"  U-statistic: U = {mw['statistic']:.4f}")
		lines.append(f"  P-value:     p = {mw['p_value']:.4f}")
		
		if mw['p_value'] < 0.05:
			lines.append(f"  Result: Significant difference (p < 0.05)")
		else:
			lines.append(f"  Result: No significant difference (p ≥ 0.05)")
		
		# Effect size
		lines.append("\n" + "-"*80)
		lines.append("Effect Size (Cohen's d):")
		lines.append("-"*80)
		es = between_results['effect_size']
		lines.append(f"  Cohen's d:   {es['cohens_d']:.4f}")
		lines.append(f"  Interpretation: {es['interpretation'].capitalize()} effect size")
		
		# Confidence interval
		lines.append("\n" + "-"*80)
		lines.append("95% Confidence Interval for Mean Difference:")
		lines.append("-"*80)
		ci = between_results['confidence_interval']
		lines.append(f"  Mean Difference: {ci['mean_diff']:.4f}")
		lines.append(f"  95% CI: [{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}]")
		
		if ci['ci_95_lower'] * ci['ci_95_upper'] > 0:
			lines.append("  Interpretation: CI does not include zero (significant difference)")
		else:
			lines.append("  Interpretation: CI includes zero (no significant difference)")
	else:
		lines.append("\n[WARNING] Between-group comparison could not be performed.")
		lines.append("Expected 2 CA% groups but found a different number.")
	
	# Interpretation and conclusion
	lines.append("\n\n" + "="*80)
	lines.append("SECTION 4: INTERPRETATION AND CONCLUSIONS")
	lines.append("="*80)
	
	if between_results and 't_test' in between_results and between_results['t_test']['p_value'] < 0.05:
		lines.append(f"\nThe two groups show SIGNIFICANTLY DIFFERENT rates of weight change.")
		lines.append(f"The {between_results['ca_groups'][1]}% CA group has a mean slope that is")
		lines.append(f"{abs(between_results['mean_diff']):.4f} {measure} per {time_unit} {'higher' if between_results['mean_diff'] > 0 else 'lower'}")
		lines.append(f"than the {between_results['ca_groups'][0]}% CA group (p = {between_results['t_test']['p_value']:.4f}).")
	elif between_results and 't_test' in between_results:
		lines.append(f"\nThe two groups show NO SIGNIFICANT DIFFERENCE in rates of weight change.")
		lines.append(f"Both groups put on weight at approximately the same rate (p = {between_results['t_test']['p_value']:.4f}).")
	else:
		lines.append(f"\n[WARNING] Between-group comparison could not be completed.")
		lines.append("This analysis requires exactly 2 CA% groups.")
	
	lines.append("\n" + "="*80)
	lines.append("END OF REPORT")
	lines.append("="*80)
	
	return "\n".join(lines)


def perform_complete_slope_analysis_cah(
	df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
	save_plot: bool = True,
	save_report: bool = True,
	output_dir: Optional[Path] = None
) -> Dict:
	"""
	Complete pipeline for slope analysis: calculate slopes, compare groups, plot, report.
	
	Parameters:
		df: Cleaned DataFrame with all required columns
		measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
		time_unit: Time unit to use ('Week' or 'Day')
		save_plot: Whether to save the plot to a file
		save_report: Whether to save the report to a file
		output_dir: Directory to save outputs (None = current directory)
		
	Returns:
		Dictionary with all analysis results
	"""
	print("\n" + "="*80)
	print("COMPLETE SLOPE ANALYSIS PIPELINE - CAH COHORT")
	print("="*80)
	print(f"\nMeasure: {measure}")
	print(f"Time Unit: {time_unit}")
	
	# Step 1: Calculate slopes for each animal
	slopes_df = calculate_animal_slopes_cah(df, measure=measure, time_unit=time_unit)
	
	# Step 2: Analyze within-group variability
	within_results = compare_slopes_within_ca_groups(slopes_df)
	
	# Step 3: Compare slopes between groups
	between_results = compare_slopes_between_ca_groups(slopes_df)
	
	# Step 4: Generate report
	report_text = generate_slope_analysis_report_cah(
		slopes_df,
		within_results,
		between_results,
		measure=measure,
		time_unit=time_unit
	)
	
	# Print preview
	print("\n" + "="*80)
	print("REPORT PREVIEW")
	print("="*80)
	print(report_text)
	
	# Prepare output directory and timestamp
	if output_dir is None:
		output_dir = Path.cwd()
	else:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Save report to file
	if save_report:
		report_path = output_dir / f"CAH_slope_analysis_report_{measure.replace(' ', '_')}_{timestamp}.txt"
		with open(report_path, 'w', encoding='utf-8') as f:
			f.write(report_text)
		print(f"\n[OK] Report saved to: {report_path}")
	
	# Create visualization
	if save_plot:
		plot_path = output_dir / f"CAH_slope_analysis_{measure.replace(' ', '_')}_{timestamp}.svg"
		plot_slopes_comparison_cah(
			slopes_df,
			measure=measure,
			time_unit=time_unit,
			save_path=plot_path,
			show=False
		)
	
	return {
		'slopes_df': slopes_df,
		'within_results': within_results,
		'between_results': between_results,
		'report_text': report_text,
		'measure': measure,
		'time_unit': time_unit
	}


def perform_mixed_anova_ca_x_week(
	df: pd.DataFrame,
	measure: str = "Total Change",
	padjust: str = 'fdr_bh',
) -> dict:
	"""
	2-Way Mixed ANOVA: CA% (between) × Week (within), all animals combined.

	Post-hoc tests (correction method = padjust):
	  1. Pairwise Week comparisons (main effect of Week, all CA% pooled)
	  2. Simple effects: 0% vs 2% CA% at each individual Week
	     (independent-samples Welch t-tests, corrected across all weeks)
	  3. Pairwise Week comparisons within each CA% group separately

	Parameters
	----------
	df      : DataFrame with 'ID', 'Week', 'CA (%)', and *measure* columns.
	          If 'Week' is absent, add_week_column() is called first.
	measure : Dependent variable column name.
	padjust : Multiple-comparison correction method ('fdr_bh', 'bonf', 'holm').

	Returns
	-------
	dict with keys: anova_table, time, ca_percent, interaction,
	                posthoc_week_pairwise, posthoc_ca_per_week,
	                posthoc_within_ca, analysis_df
	"""
	print("\n" + "="*80)
	print(f"2-WAY MIXED ANOVA: CA% (BETWEEN) × WEEK (WITHIN) — {measure}")
	print(f"All animals combined  |  Correction: {padjust.upper()}")
	print("="*80)

	if not HAS_PINGOUIN:
		print("\n[ERROR] pingouin is required for mixed ANOVA.")
		print("Install with: pip install pingouin")
		return {}

	df = df.copy()
	if "Week" not in df.columns:
		df = add_week_column(df)

	required = {"ID", "Week", "CA (%)", measure}
	missing = required - set(df.columns)
	if missing:
		print(f"\n[ERROR] Missing columns: {missing}")
		return {}

	analysis_df = df[["ID", "Week", "CA (%)", measure]].copy().dropna()
	analysis_df["CA (%)"] = pd.to_numeric(analysis_df["CA (%)"], errors="coerce")
	analysis_df = analysis_df.dropna(subset=["CA (%)"])

	weeks     = sorted(analysis_df["Week"].unique())
	ca_levels = sorted(analysis_df["CA (%)"].unique())

	print(f"\n  Weeks      : {[int(w) for w in weeks]}")
	print(f"  CA% levels : {[int(c) for c in ca_levels]}")
	print(f"  Animals    : {analysis_df['ID'].nunique()}")

	# ── Descriptive statistics ────────────────────────────────────────────────
	print(f"\nDESCRIPTIVE STATISTICS  ({measure}  mean ± SEM  by CA% × Week)")
	print("-"*72)
	for ca in ca_levels:
		for wk in weeks:
			grp = analysis_df[
				(analysis_df["CA (%)"] == ca) & (analysis_df["Week"] == wk)
			][measure]
			n   = len(grp)
			mn  = grp.mean()
			sem = grp.sem()
			print(f"  CA% {int(ca):2d},  Week {int(wk):2d} : n={n},  M={mn:.3f},  SEM={sem:.3f}")

	# ── Mixed ANOVA ───────────────────────────────────────────────────────────
	print(f"\nMIXED ANOVA TABLE")
	print("-"*72)
	try:
		aov = pg.mixed_anova(
			data=analysis_df,
			dv=measure,
			within="Week",
			between="CA (%)",
			subject="ID",
			correction=True,   # force Greenhouse-Geisser correction
		)
		print(aov.to_string())
	except Exception as e:
		print(f"\n[ERROR] Mixed ANOVA failed: {e}")
		import traceback; traceback.print_exc()
		return {}

	p_col = "p-unc" if "p-unc" in aov.columns else "p_unc"

	def _find_row(label):
		for lbl in [label, f"Week * {label}", f"{label} * Week"]:
			rows = aov[aov["Source"].str.strip() == lbl]
			if not rows.empty:
				return rows.iloc[0]
		return None

	week_row  = _find_row("Week")
	ca_row    = _find_row("CA (%)")
	inter_row = None
	for src in aov["Source"]:
		s = str(src).strip()
		if "Interaction" in s or ("Week" in s and "CA" in s):
			inter_row = aov[aov["Source"] == src].iloc[0]
			break

	def _gg_p(row):
		if row is None:
			return np.nan, False
		p_gg = row.get("p-GG-corr", np.nan)
		if not pd.isna(p_gg):
			return float(p_gg), True
		return float(row[p_col]), False

	week_p,  week_gg  = _gg_p(week_row)
	ca_p              = float(ca_row[p_col]) if ca_row is not None else np.nan
	inter_p, inter_gg = _gg_p(inter_row)

	eps_val = np.nan
	if week_row is not None:
		eps_val = week_row.get("eps", np.nan)

	print(f"\nFORMATTED RESULTS")
	print("-"*72)
	if week_row is not None:
		sig  = "***" if week_p < 0.001 else "**" if week_p < 0.01 else "*" if week_p < 0.05 else "ns"
		note = " (GG-corrected)" if week_gg else ""
		print(f"  Week       : F({week_row['DF1']:.0f},{week_row['DF2']:.0f}) = {week_row['F']:.3f},  p = {week_p:.4f} {sig}{note}")
	if ca_row is not None:
		sig  = "***" if ca_p < 0.001 else "**" if ca_p < 0.01 else "*" if ca_p < 0.05 else "ns"
		print(f"  CA%        : F({ca_row['DF1']:.0f},{ca_row['DF2']:.0f}) = {ca_row['F']:.3f},  p = {ca_p:.4f} {sig}")
	if inter_row is not None:
		sig  = "***" if inter_p < 0.001 else "**" if inter_p < 0.01 else "*" if inter_p < 0.05 else "ns"
		note = " (GG-corrected)" if inter_gg else ""
		print(f"  Week × CA% : F({inter_row['DF1']:.0f},{inter_row['DF2']:.0f}) = {inter_row['F']:.3f},  p = {inter_p:.4f} {sig}{note}")
	if not pd.isna(eps_val):
		print(f"\n  Sphericity : Greenhouse-Geisser ε = {float(eps_val):.4f}", end="")
		if float(eps_val) < 0.75:
			print("  ← violated, GG correction applied")
		else:
			print("  ← acceptable")

	results = {
		"measure": measure,
		"type": "mixed_anova_ca_x_week",
		"padjust": padjust,
		"anova_table": aov,
		"analysis_df": analysis_df,
		"time": {
			"F":            week_row["F"]   if week_row  is not None else np.nan,
			"p":            week_p,
			"gg_corrected": week_gg,
			"eps":          float(eps_val) if not pd.isna(eps_val) else np.nan,
			"df1":          float(week_row["DF1"]) if week_row  is not None else np.nan,
			"df2":          float(week_row["DF2"]) if week_row  is not None else np.nan,
			"significant":  week_p < 0.05 if not np.isnan(week_p) else False,
		},
		"ca_percent": {
			"F":           ca_row["F"]   if ca_row is not None else np.nan,
			"p":           ca_p,
			"df1":         float(ca_row["DF1"]) if ca_row is not None else np.nan,
			"df2":         float(ca_row["DF2"]) if ca_row is not None else np.nan,
			"significant": ca_p < 0.05 if not np.isnan(ca_p) else False,
		},
		"interaction": {
			"F":            inter_row["F"]   if inter_row is not None else np.nan,
			"p":            inter_p,
			"gg_corrected": inter_gg,
			"df1":          float(inter_row["DF1"]) if inter_row is not None else np.nan,
			"df2":          float(inter_row["DF2"]) if inter_row is not None else np.nan,
			"significant":  inter_p < 0.05 if not np.isnan(inter_p) else False,
		},
	}

	# ── POST-HOC 1: Pairwise Week comparisons (main effect of Week) ───────────
	print(f"\n{'='*80}")
	print(f"POST-HOC 1: PAIRWISE WEEK COMPARISONS")
	print(f"(Main effect of Week — all CA% pooled — {padjust.upper()} corrected)")
	print("="*80)
	try:
		week_pw = pg.pairwise_tests(
			data=analysis_df,
			dv=measure,
			within="Week",
			subject="ID",
			parametric=True,
			padjust=padjust,
			return_desc=True,
		)
		week_pw   = week_pw[week_pw["Contrast"] == "Week"].copy()
		padj_col  = "p-corr" if "p-corr" in week_pw.columns else padjust
		p_raw_col = "p-unc"  if "p-unc"  in week_pw.columns else "p"
		sig_n     = (week_pw[padj_col] < 0.05).sum() if padj_col in week_pw.columns else 0
		print(f"\n  {len(week_pw)} pairs — {sig_n} significant after correction\n")
		for _, row in week_pw.iterrows():
			a, b   = int(row["A"]), int(row["B"])
			p_adj  = row.get(padj_col, np.nan)
			p_raw_val = row.get(p_raw_col, np.nan)
			sig    = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
			print(f"  Week {a:2d} vs Week {b:2d}: p-raw = {p_raw_val:.4f},  p-adj = {p_adj:.4f}  [{sig}]")
		results["posthoc_week_pairwise"] = week_pw
	except Exception as e:
		print(f"\n  [ERROR] Pairwise Week post-hoc failed: {e}")
		results["posthoc_week_pairwise"] = pd.DataFrame()

	# ── POST-HOC 2: Simple effects — CA% at each Week ────────────────────────
	print(f"\n{'='*80}")
	print(f"POST-HOC 2: SIMPLE EFFECTS — CA% AT EACH WEEK")
	print(f"(Welch t-test: 0% vs 2% per week — {padjust.upper()} corrected across weeks)")
	print("="*80)
	simple_rows = []
	for wk in weeks:
		wk_data = analysis_df[analysis_df["Week"] == wk]
		if len(ca_levels) == 2:
			g1 = wk_data[wk_data["CA (%)"] == ca_levels[0]][measure].dropna().values
			g2 = wk_data[wk_data["CA (%)"] == ca_levels[1]][measure].dropna().values
			if len(g1) < 2 or len(g2) < 2:
				print(f"  Week {int(wk):2d}: insufficient data — skipping")
				continue
			from scipy.stats import ttest_ind
			t_stat, p_raw = ttest_ind(g1, g2, equal_var=False)
			simple_rows.append({
				"Week":   int(wk),
				"CA_A":   int(ca_levels[0]),
				"CA_B":   int(ca_levels[1]),
				"n_A":    len(g1),
				"n_B":    len(g2),
				"mean_A": g1.mean(),
				"mean_B": g2.mean(),
				"t":      t_stat,
				"p_raw":  p_raw,
			})

	if simple_rows:
		from statsmodels.stats.multitest import multipletests
		p_raws = [r["p_raw"] for r in simple_rows]
		reject, p_adj_arr, _, _ = multipletests(p_raws, method=padjust)
		for i, row in enumerate(simple_rows):
			row["p_adj"]      = p_adj_arr[i]
			row["significant"] = bool(reject[i])
		ca_per_week_df = pd.DataFrame(simple_rows)
		results["posthoc_ca_per_week"] = ca_per_week_df
		n_sig = ca_per_week_df["significant"].sum()
		print(f"\n  {len(simple_rows)} tests — {n_sig} significant after correction\n")
		for _, row in ca_per_week_df.iterrows():
			sig = "***" if row["p_adj"] < 0.001 else "**" if row["p_adj"] < 0.01 else "*" if row["p_adj"] < 0.05 else "ns"
			print(
				f"  Week {int(row['Week']):2d}: "
				f"CA% {int(row['CA_A'])}% (n={int(row['n_A'])}, M={row['mean_A']:.3f}) vs "
				f"{int(row['CA_B'])}% (n={int(row['n_B'])}, M={row['mean_B']:.3f}):  "
				f"t = {row['t']:.3f},  p-raw = {row['p_raw']:.4f},  p-adj = {row['p_adj']:.4f}  [{sig}]"
			)
	else:
		print("  [WARNING] No valid week-level comparisons could be performed.")
		results["posthoc_ca_per_week"] = pd.DataFrame()

	# ── POST-HOC 3: Pairwise Week comparisons within each CA% group ───────────
	print(f"\n{'='*80}")
	print(f"POST-HOC 3: WEEK PAIRWISE WITHIN EACH CA% GROUP")
	print(f"(Simple effects of Time — {padjust.upper()} corrected)")
	print("="*80)
	within_ca_posthoc = {}
	for ca in ca_levels:
		ca_data = analysis_df[analysis_df["CA (%)"] == ca].copy()
		print(f"\n  CA% = {int(ca)}%:")
		try:
			ca_pw = pg.pairwise_tests(
				data=ca_data,
				dv=measure,
				within="Week",
				subject="ID",
				parametric=True,
				padjust=padjust,
				return_desc=True,
			)
			ca_pw    = ca_pw[ca_pw["Contrast"] == "Week"].copy()
			padj_col = "p-corr" if "p-corr" in ca_pw.columns else padjust
			p_raw_col = "p-unc" if "p-unc" in ca_pw.columns else "p"
			sig_n    = (ca_pw[padj_col] < 0.05).sum() if padj_col in ca_pw.columns else 0
			print(f"  {len(ca_pw)} pairs — {sig_n} significant after correction")
			for _, row in ca_pw.iterrows():
				a, b  = int(row["A"]), int(row["B"])
				p_adj = row.get(padj_col, np.nan)
				p_unc = row.get(p_raw_col, np.nan)
				sig   = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
				print(f"    Week {a:2d} vs Week {b:2d}: p-raw = {p_unc:.4f},  p-adj = {p_adj:.4f}  [{sig}]")
			within_ca_posthoc[int(ca)] = ca_pw
		except Exception as e:
			print(f"    [ERROR] Failed for CA% {int(ca)}%: {e}")
			within_ca_posthoc[int(ca)] = pd.DataFrame()
	results["posthoc_within_ca"] = within_ca_posthoc

	print(f"\n{'='*80}")
	print(f"CA% × WEEK MIXED ANOVA COMPLETE")
	print(f"{'='*80}")
	return results


def _save_report(report_text: str, filename: str) -> Path:
	"""Save a report string to the analysis output directory and return the path."""
	report_path = Path(__file__).parent / filename
	with open(report_path, 'w', encoding='utf-8') as f:
		f.write(report_text)
	print(f"\n[OK] Report saved to: {report_path}")
	return report_path


# ── Omnibus-style report helpers ─────────────────────────────────────────────

def _fmt_p(p: float) -> str:
	"""Format a p-value: show '< 0.0001' for very small values."""
	if np.isnan(p):
		return "n/a"
	if p < 0.0001:
		return "< 0.0001"
	return f"{p:.4f}"


def _sig_stars(p: float) -> str:
	if np.isnan(p):
		return ""
	if p < 0.001:
		return "***"
	if p < 0.01:
		return "**"
	if p < 0.05:
		return "*"
	return "ns"


def _omnibus_header(title: str, design: str, correction_name: str,
                    timestamp: str, measures: list) -> str:
	"""Return a formatted report header block."""
	W = 80
	lines = [
		"=" * W,
		f"CAH COHORT — {title}",
		"=" * W,
		f"Generated  : {timestamp}",
		f"Design     : {design}",
		f"Correction : {correction_name}",
		f"Measures   : {', '.join(measures)}",
		"",
	]
	return "\n".join(lines)


def _omnibus_measure_header(measure: str) -> str:
	"""Return thick-separator section header for a measure."""
	W = 80
	thick = "━" * W
	return f"\n{thick}\n  MEASURE: {measure}\n{thick}\n"


def _omnibus_desc_stats_block(dm: pd.DataFrame, measure: str,
                               group_cols: list) -> str:
	"""Build a descriptive-stats table for the given grouping columns."""
	lines = []
	header_parts = group_cols + ["n", "Mean", "SD", "SEM", "95% CI"]
	col_widths = [max(12, len(c) + 2) for c in header_parts]
	# header row
	header = "  " + "".join(f"{h:<{w}}" for h, w in zip(header_parts, col_widths))
	lines.append(header)
	lines.append("  " + "-" * (sum(col_widths)))
	for keys, grp in dm.groupby(group_cols)[measure]:
		if not isinstance(keys, tuple):
			keys = (keys,)
		vals = grp.dropna()
		n = len(vals)
		if n == 0:
			continue
		m = vals.mean()
		sd = vals.std(ddof=1)
		sem = sd / np.sqrt(n)
		ci_lo = m - 1.96 * sem
		ci_hi = m + 1.96 * sem
		key_strs = [str(k) for k in keys]
		row_vals = key_strs + [str(n), f"{m:.3f}", f"{sd:.3f}", f"{sem:.3f}",
		                        f"[{ci_lo:.3f}, {ci_hi:.3f}]"]
		row = "  " + "".join(f"{v:<{w}}" for v, w in zip(row_vals, col_widths))
		lines.append(row)
	return "\n".join(lines)


def _omnibus_anova_block(results: dict, effect_map: list) -> str:
	"""
	Return a formatted ANOVA effects block.
	effect_map: list of (label, key) pairs — key is the results dict key.
	"""
	lines = ["  ANOVA Results:", "  " + "-" * 74]
	for label, key in effect_map:
		r = results.get(key, {})
		if not r:
			continue
		F = r.get('F', np.nan)
		p = r.get('p', np.nan)
		df1 = r.get('df1', np.nan)
		df2 = r.get('df2', np.nan)
		gg = r.get('gg_corrected', False)
		stars = _sig_stars(p)
		gg_tag = " (GG-corrected)" if gg else ""
		eta2 = r.get('np2', np.nan)
		eta_str = f", ηp² = {eta2:.3f}" if not np.isnan(eta2) else ""
		lines.append(f"  {label}:")
		lines.append(f"    F({df1:.0f}, {df2:.0f}) = {F:.3f},  p = {_fmt_p(p)}  {stars}{gg_tag}{eta_str}")
	return "\n".join(lines)


def _omnibus_interpretation(results: dict, effect_map: list, analysis_desc: str) -> str:
	"""Return a plain-English INTERPRETATION block."""
	lines = ["", "  INTERPRETATION", "  " + "-" * 74]
	for i, (label, key) in enumerate(effect_map, start=1):
		r = results.get(key, {})
		if not r:
			continue
		p = r.get('p', np.nan)
		stars = _sig_stars(p)
		sig_word = "significant" if p < 0.05 else "not significant"
		lines.append(f"  {i}. {label}: p = {_fmt_p(p)} {stars} — {sig_word}.")
	lines.append("")
	lines.append(f"  Analysis: {analysis_desc}")
	return "\n".join(lines)


def _omnibus_posthoc_block(title: str, ph_df) -> str:
	"""Format a post-hoc DataFrame as a clean aligned table (not raw to_string)."""
	if ph_df is None or len(ph_df) == 0:
		return f"  {title}\n  (no pairs to report)\n"
	lines = [f"  {title}", "  " + "-" * 74]
	# Determine which columns to show based on what's available
	want_cols = ['A', 'B', 'Contrast', 'mean(A)', 'mean(B)', 'T', 't', 'dof',
	             'p-unc', 'p-corr', 'p_raw', 'p_adj', 'significant',
	             'Week', 'CA_A', 'CA_B', 'mean_A', 'mean_B', 'hedges']
	show_cols = [c for c in want_cols if c in ph_df.columns]
	if not show_cols:
		show_cols = list(ph_df.columns)
	sub = ph_df[show_cols].copy()

	# Add a significance-stars column based on the adjusted p-value
	adj_col = next((c for c in ('p-corr', 'p_adj') if c in sub.columns), None)
	if adj_col is None:
		adj_col = next((c for c in ('p-unc', 'p_raw') if c in sub.columns), None)
	if adj_col is not None:
		def _stars(val):
			try:
				v = float(val)
			except (TypeError, ValueError):
				return ""
			if v < 0.001:
				return "***"
			if v < 0.01:
				return "**"
			if v < 0.05:
				return "*"
			return "ns"
		sub.insert(show_cols.index(adj_col) + 1, 'sig', sub[adj_col].apply(_stars))
		show_cols = list(sub.columns)

	# Format float columns to 4 decimal places
	for col in sub.select_dtypes(include=[float]).columns:
		sub[col] = sub[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "n/a")
	# Build fixed-width table
	col_widths = [max(len(str(c)), sub[c].astype(str).str.len().max()) + 2
	              for c in show_cols]
	header = "  " + "".join(f"{c:<{w}}" for c, w in zip(show_cols, col_widths))
	lines.append(header)
	lines.append("  " + "-" * sum(col_widths))
	for _, row in sub.iterrows():
		r_str = "  " + "".join(f"{str(row[c]):<{w}}" for c, w in zip(show_cols, col_widths))
		lines.append(r_str)
	return "\n".join(lines) + "\n"


def _omnibus_summary_table(measures: list, all_results: list, effect_labels: list) -> str:
	"""
	Build a summary table showing key p-values for each measure.
	all_results: list of result dicts (one per measure, same order as measures).
	effect_labels: list of (column_header, key) pairs.
	"""
	W = 80
	lines = [
		"",
		"=" * W,
		"  SUMMARY TABLE — KEY p-VALUES (α = 0.05)",
		"=" * W,
	]
	# Determine column widths
	mw = max(len("Measure"), max(len(m) for m in measures)) + 2
	ew = [max(len(lbl), 10) + 2 for lbl, _ in effect_labels]
	dw = max(len("Decision"), 20) + 2
	# Header row
	hdr = f"  {'Measure':<{mw}}" + "".join(f"{lbl:<{w}}" for (lbl, _), w in zip(effect_labels, ew)) + f"{'Decision':<{dw}}"
	lines.append(hdr)
	lines.append("  " + "-" * (mw + sum(ew) + dw))
	for measure, res in zip(measures, all_results):
		if not res:
			row = f"  {measure:<{mw}}" + "".join(f"{'n/a':<{w}}" for w in ew) + f"{'no results':<{dw}}"
			lines.append(row)
			continue
		sig_effects = []
		p_strs = []
		for (lbl, key), w in zip(effect_labels, ew):
			r = res.get(key, {})
			p = r.get('p', np.nan) if isinstance(r, dict) else np.nan
			stars = _sig_stars(p)
			cell = f"{_fmt_p(p)} {stars}"
			p_strs.append(f"{cell:<{w}}")
			if p < 0.05:
				sig_effects.append(lbl)
		if sig_effects:
			decision = "Sig: " + ", ".join(sig_effects)
		else:
			decision = "No significant effects"
		row = f"  {measure:<{mw}}" + "".join(p_strs) + f"{decision:<{dw}}"
		lines.append(row)
	lines.append("=" * W)
	return "\n".join(lines)


def main():
	"""
	Main function: Load, clean, and summarize CAH cohort data.
	Presents a menu to run individual analyses, each saving its own report.
	"""
	csv_path = Path(__file__).parent.parent / "CAH_cohort" / "master_data_CAH.csv"

	print("="*80)
	print("CAH COHORT WEIGHT ANALYSIS")
	print("="*80)

	# ── Load and clean ────────────────────────────────────────────────────────
	df_raw = load_cah_data(csv_path)
	print("\nCleaning data...")
	df = clean_cah_dataframe(df_raw)
	print("Adding per-animal day numbering...")
	df = add_day_number_column(df)
	summarize_dataframe(df)

	# ── Global settings ───────────────────────────────────────────────────────
	print("\n" + "="*80)
	print("GLOBAL SETTINGS")
	print("="*80)

	print("\nChoose time unit for longitudinal analyses:")
	print("  [1] Days (most granular)")
	print("  [2] Weeks (averaged by week, Day 0 excluded)")
	time_unit_choice = input("Enter choice (1 or 2, default=2): ").strip()
	use_weeks  = (time_unit_choice != '1')
	time_unit  = 'week' if use_weeks else 'day'
	time_label = 'Week'  if use_weeks else 'Day'

	if use_weeks:
		df = add_week_column(df)
		week_counts = df.groupby('Week')['Day'].apply(lambda x: (x.min(), x.max()))
		print(f"\nWeek mapping:")
		for week, (d_min, d_max) in week_counts.items():
			if pd.notna(week):
				print(f"  Week {int(week)}: Days {int(d_min)}-{int(d_max)}")

	print("\nChoose multiple-comparison correction method:")
	print("  [1] FDR-BH — Benjamini-Hochberg (recommended)")
	print("  [2] Bonferroni — conservative, controls family-wise error rate")
	correction_choice = input("Enter choice (1 or 2, default=1): ").strip()
	padjust_method  = 'bonf' if correction_choice == '2' else 'fdr_bh'
	correction_name = 'Bonferroni' if padjust_method == 'bonf' else 'FDR-BH'
	print(f"  → Using {correction_name} correction")

	measures_to_analyze = ["Total Change", "Daily Change"]

	time_col  = "Week" if use_weeks else "Day"
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	time_sfx  = "weekly" if use_weeks else "daily"
	corr_sfx  = "bonf"   if padjust_method == 'bonf' else "fdr"
	out_dir   = Path(__file__).parent

	# ── Menu loop ─────────────────────────────────────────────────────────────
	MENU = """
================================================================================
  CAH ANALYSIS MENU
================================================================================
  [1]  Between-subjects ANOVA — Sex × CA% (grand average across time)
  [2]  3-Way Mixed ANOVA      — Time × Sex × CA%
  [3]  Sex-stratified Mixed ANOVA — Time × CA% (Males)
  [4]  Sex-stratified Mixed ANOVA — Time × CA% (Females)
  [5]  CA%-stratified Mixed ANOVA — Time × Sex (0% CA)
  [6]  CA%-stratified Mixed ANOVA — Time × Sex (2% CA)
  [7]  2-Way Mixed ANOVA — CA% × {time} (all animals, sex collapsed){week_only}
  [8]  Slope analysis — rate of weight change between CA% groups
  [9]  Interaction plots for significant effects
  [A]  Run ALL analyses (1-8) and save all reports
  [Q]  Quit
================================================================================"""

	week_only_note = "  ← weeks only" if not use_weeks else ""

	while True:
		print(MENU.format(time=time_label, week_only=week_only_note))
		choice = input("Select option: ").strip().upper()

		if choice == 'Q':
			print("\nExiting.")
			break

		# ── Helper: prepare df_measure for each measure ───────────────────────
		def _prep(measure):
			dm = df.copy()
			if use_weeks:
				dm = average_by_week(dm, measure=measure)
			return dm

		# ── Helper: build a timestamped filename ──────────────────────────────
		def _fname(tag):
			return f"CAH_{tag}_{time_sfx}_{corr_sfx}_{timestamp}.txt"

		# ── Option 1: Between-subjects ANOVA ─────────────────────────────────
		if choice in ('1', 'A'):
			effect_map_1 = [('Sex', 'sex'), ('CA%', 'ca_percent'), ('Sex × CA%', 'interaction')]
			all_res_1 = []
			report_1  = _omnibus_header(
				"BETWEEN-SUBJECTS ANOVA — Sex × CA% (grand average)",
				"Sex (between) × CA% (between), grand mean across " + time_label.lower() + "s",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] BETWEEN-SUBJECTS ANOVA — Sex × CA% (grand average)")
				print("="*80)
				results_avg = perform_two_way_between_anova(dm, measure=measure, average_over_days=True)
				all_res_1.append(results_avg)

				tukey_results = None
				if results_avg and results_avg.get('ca_percent', {}).get('significant'):
					avg_df = dm.groupby(["ID", "Sex", "CA (%)"])[measure].mean().reset_index()
					avg_df["CA (%) Group"] = avg_df["CA (%)"].astype(str) + "%"
					tukey_results = perform_tukey_hsd(avg_df, measure, "CA (%) Group")

				report_1 += _omnibus_measure_header(measure)
				if results_avg:
					# Descriptive stats
					avg_dm = dm.groupby(["ID", "Sex", "CA (%)"])[measure].mean().reset_index()
					report_1 += "  Descriptive Statistics by Sex:\n"
					report_1 += _omnibus_desc_stats_block(avg_dm, measure, ["Sex"]) + "\n"
					report_1 += "  Descriptive Statistics by CA%:\n"
					report_1 += _omnibus_desc_stats_block(avg_dm, measure, ["CA (%)"]) + "\n"
					report_1 += "  Descriptive Statistics by Sex × CA%:\n"
					report_1 += _omnibus_desc_stats_block(avg_dm, measure, ["Sex", "CA (%)"]) + "\n\n"
					report_1 += _omnibus_anova_block(results_avg, effect_map_1) + "\n"
					sig_eff = [l for l, k in effect_map_1 if results_avg.get(k, {}).get('significant')]
					desc = ("Significant between-subjects effects: " + ", ".join(sig_eff))\
					       if sig_eff else "No significant between-subjects effects."
					report_1 += _omnibus_interpretation(results_avg, effect_map_1, desc) + "\n"
				else:
					report_1 += "  Analysis did not complete.\n"
				if tukey_results:
					ph_df = tukey_results.get('results')
					report_1 += _omnibus_posthoc_block("Post-hoc: Tukey HSD — CA% pairwise  [Tukey's HSD]", ph_df)
			report_1 += _omnibus_summary_table(measures_to_analyze, all_res_1, effect_map_1)
			report_1 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_1, _fname("between_anova"))
			if choice != 'A':
				continue

		# ── Option 2: 3-Way Mixed ANOVA ───────────────────────────────────────
		if choice in ('2', 'A'):
			# 3-way ANOVA returns time_point_interactions dict; use raw anova_table
			all_res_2 = []
			report_2  = _omnibus_header(
				f"3-WAY MIXED ANOVA — {time_label} × Sex × CA%",
				f"{time_label} (within) × Sex (between) × CA% (between)",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] 3-WAY MIXED ANOVA — Time × Sex × CA%")
				print("="*80)
				results_mixed = perform_mixed_anova_time(dm, measure=measure,
				                                         time_points=None, time_unit=time_unit)
				all_res_2.append(results_mixed)
				report_2 += _omnibus_measure_header(measure)
				if results_mixed:
					report_2 += f"  N animals: {dm['ID'].nunique()}\n\n"
					report_2 += "  Descriptive Statistics by Sex:\n"
					report_2 += _omnibus_desc_stats_block(dm, measure, ["Sex"]) + "\n"
					report_2 += "  Descriptive Statistics by CA%:\n"
					report_2 += _omnibus_desc_stats_block(dm, measure, ["CA (%)"]) + "\n\n"
					aov = results_mixed.get('anova_table')
					if aov is not None:
						report_2 += "  ANOVA Table:\n"
						for line in aov.to_string().splitlines():
							report_2 += "    " + line + "\n"
						report_2 += "\n"
					# Interaction summary
					time_sex_p = results_mixed.get('time_sex', {}).get('p', np.nan)
					time_ca_p  = results_mixed.get('time_ca',  {}).get('p', np.nan)
					time_ca_sex_p = results_mixed.get('time_ca_sex', {}).get('p', np.nan)
					report_2 += "  INTERPRETATION\n  " + "-"*74 + "\n"
					report_2 += f"  Time × Sex:       p = {_fmt_p(time_sex_p)}  {_sig_stars(time_sex_p)}\n"
					report_2 += f"  Time × CA%:       p = {_fmt_p(time_ca_p)}  {_sig_stars(time_ca_p)}\n"
					report_2 += f"  Time × Sex × CA%: p = {_fmt_p(time_ca_sex_p)}  {_sig_stars(time_ca_sex_p)}\n"
				else:
					report_2 += "  Analysis did not complete.\n"
			# No clean summary table for 3-way (non-standard effect keys)
			report_2 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_2, _fname("3way_mixed_anova"))
			if choice != 'A':
				continue

		# ── Option 3: Sex-stratified — Males ──────────────────────────────────
		if choice in ('3', 'A'):
			effect_map_3  = [('Time', 'time'), ('CA%', 'ca_percent'), ('Time × CA%', 'interaction')]
			all_res_3     = []
			report_3 = _omnibus_header(
				f"SEX-STRATIFIED MIXED ANOVA — {time_label} × CA% (Males)",
				f"{time_label} (within) × CA% (between), Males only",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] SEX-STRATIFIED MIXED ANOVA — Time × CA% (Males)")
				print("="*80)
				results_males = perform_mixed_anova_sex_stratified(dm, sex="M", measure=measure,
				                                                    time_points=None, time_unit=time_unit)
				all_res_3.append(results_males)
				posthoc_males = None
				if results_males and results_males.get('interaction', {}).get('significant'):
					male_df = dm[dm['Sex'] == 'M'].copy()
					posthoc_males = perform_mixed_anova_posthoc(male_df, measure=measure,
					                    within=time_col, between="CA (%)", subject="ID",
					                    padjust=padjust_method)
				report_3 += _omnibus_measure_header(measure)
				if results_males:
					male_dm = dm[dm['Sex'] == 'M'].copy()
					report_3 += f"  N animals (M): {male_dm['ID'].nunique()}\n\n"
					report_3 += "  Descriptive Statistics by CA%:\n"
					report_3 += _omnibus_desc_stats_block(male_dm, measure, ["CA (%)"]) + "\n\n"
					report_3 += _omnibus_anova_block(results_males, effect_map_3) + "\n"
					sig_eff = [l for l, k in effect_map_3 if results_males.get(k, {}).get('significant')]
					desc = ("Significant effects (Males): " + ", ".join(sig_eff)) if sig_eff else "No significant effects in males."
					report_3 += _omnibus_interpretation(results_males, effect_map_3, desc) + "\n"
				else:
					report_3 += "  Analysis did not complete.\n"
				if posthoc_males:
					ph_df = posthoc_males.get('results') if isinstance(posthoc_males, dict) else posthoc_males
					report_3 += _omnibus_posthoc_block(f"Post-hoc: {time_label} × CA% pairwise (Males)  [paired t-test, {correction_name}]", ph_df)
			report_3 += _omnibus_summary_table(measures_to_analyze, all_res_3, effect_map_3)
			report_3 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_3, _fname("sex_strat_males"))
			if choice != 'A':
				continue

		# ── Option 4: Sex-stratified — Females ────────────────────────────────
		if choice in ('4', 'A'):
			effect_map_4  = [('Time', 'time'), ('CA%', 'ca_percent'), ('Time × CA%', 'interaction')]
			all_res_4     = []
			report_4 = _omnibus_header(
				f"SEX-STRATIFIED MIXED ANOVA — {time_label} × CA% (Females)",
				f"{time_label} (within) × CA% (between), Females only",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] SEX-STRATIFIED MIXED ANOVA — Time × CA% (Females)")
				print("="*80)
				results_females = perform_mixed_anova_sex_stratified(dm, sex="F", measure=measure,
				                                                      time_points=None, time_unit=time_unit)
				all_res_4.append(results_females)
				posthoc_females = None
				if results_females and results_females.get('interaction', {}).get('significant'):
					female_df = dm[dm['Sex'] == 'F'].copy()
					posthoc_females = perform_mixed_anova_posthoc(female_df, measure=measure,
					                      within=time_col, between="CA (%)", subject="ID",
					                      padjust=padjust_method)
				report_4 += _omnibus_measure_header(measure)
				if results_females:
					female_dm = dm[dm['Sex'] == 'F'].copy()
					report_4 += f"  N animals (F): {female_dm['ID'].nunique()}\n\n"
					report_4 += "  Descriptive Statistics by CA%:\n"
					report_4 += _omnibus_desc_stats_block(female_dm, measure, ["CA (%)"]) + "\n\n"
					report_4 += _omnibus_anova_block(results_females, effect_map_4) + "\n"
					sig_eff = [l for l, k in effect_map_4 if results_females.get(k, {}).get('significant')]
					desc = ("Significant effects (Females): " + ", ".join(sig_eff)) if sig_eff else "No significant effects in females."
					report_4 += _omnibus_interpretation(results_females, effect_map_4, desc) + "\n"
				else:
					report_4 += "  Analysis did not complete.\n"
				if posthoc_females:
					ph_df = posthoc_females.get('results') if isinstance(posthoc_females, dict) else posthoc_females
					report_4 += _omnibus_posthoc_block(f"Post-hoc: {time_label} × CA% pairwise (Females)  [paired t-test, {correction_name}]", ph_df)
			report_4 += _omnibus_summary_table(measures_to_analyze, all_res_4, effect_map_4)
			report_4 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_4, _fname("sex_strat_females"))
			if choice != 'A':
				continue

		# ── Option 5: CA%-stratified — 0% CA ──────────────────────────────────
		if choice in ('5', 'A'):
			effect_map_5 = [('Time', 'time'), ('Sex', 'sex'), ('Time × Sex', 'interaction')]
			all_res_5    = []
			report_5 = _omnibus_header(
				f"CA%-STRATIFIED MIXED ANOVA — {time_label} × Sex (0% CA)",
				f"{time_label} (within) × Sex (between), 0% CA only",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] CA%-STRATIFIED MIXED ANOVA — Time × Sex (0% CA)")
				print("="*80)
				results_ca0 = perform_mixed_anova_ca_stratified(dm, ca_percent=0, measure=measure,
				                                                 time_points=None, time_unit=time_unit)
				all_res_5.append(results_ca0)
				posthoc_ca0 = None
				if results_ca0 and results_ca0.get('interaction', {}).get('significant'):
					ca0_df = dm[dm['CA (%)'] == 0].copy()
					posthoc_ca0 = perform_mixed_anova_posthoc(ca0_df, measure=measure,
					                  within=time_col, between="Sex", subject="ID",
					                  padjust=padjust_method)
				report_5 += _omnibus_measure_header(measure)
				if results_ca0:
					ca0_dm = dm[dm['CA (%)'] == 0].copy()
					report_5 += f"  N animals (0% CA): {ca0_dm['ID'].nunique()}\n\n"
					report_5 += "  Descriptive Statistics by Sex:\n"
					report_5 += _omnibus_desc_stats_block(ca0_dm, measure, ["Sex"]) + "\n\n"
					report_5 += _omnibus_anova_block(results_ca0, effect_map_5) + "\n"
					sig_eff = [l for l, k in effect_map_5 if results_ca0.get(k, {}).get('significant')]
					desc = ("Significant effects (0% CA): " + ", ".join(sig_eff)) if sig_eff else "No significant effects in 0% CA group."
					report_5 += _omnibus_interpretation(results_ca0, effect_map_5, desc) + "\n"
				else:
					report_5 += "  Analysis did not complete.\n"
				if posthoc_ca0:
					ph_df = posthoc_ca0.get('results') if isinstance(posthoc_ca0, dict) else posthoc_ca0
					report_5 += _omnibus_posthoc_block(f"Post-hoc: {time_label} × Sex pairwise (0% CA)  [paired t-test, {correction_name}]", ph_df)
			report_5 += _omnibus_summary_table(measures_to_analyze, all_res_5, effect_map_5)
			report_5 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_5, _fname("ca_strat_0pct"))
			if choice != 'A':
				continue

		# ── Option 6: CA%-stratified — 2% CA ──────────────────────────────────
		if choice in ('6', 'A'):
			effect_map_6 = [('Time', 'time'), ('Sex', 'sex'), ('Time × Sex', 'interaction')]
			all_res_6    = []
			report_6 = _omnibus_header(
				f"CA%-STRATIFIED MIXED ANOVA — {time_label} × Sex (2% CA)",
				f"{time_label} (within) × Sex (between), 2% CA only",
				correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				measures_to_analyze)
			for measure in measures_to_analyze:
				dm = _prep(measure)
				print(f"\n{'='*80}")
				print(f"[{measure}] CA%-STRATIFIED MIXED ANOVA — Time × Sex (2% CA)")
				print("="*80)
				results_ca2 = perform_mixed_anova_ca_stratified(dm, ca_percent=2, measure=measure,
				                                                 time_points=None, time_unit=time_unit)
				all_res_6.append(results_ca2)
				posthoc_ca2 = None
				if results_ca2 and results_ca2.get('interaction', {}).get('significant'):
					ca2_df = dm[dm['CA (%)'] == 2].copy()
					posthoc_ca2 = perform_mixed_anova_posthoc(ca2_df, measure=measure,
					                  within=time_col, between="Sex", subject="ID",
					                  padjust=padjust_method)
				report_6 += _omnibus_measure_header(measure)
				if results_ca2:
					ca2_dm = dm[dm['CA (%)'] == 2].copy()
					report_6 += f"  N animals (2% CA): {ca2_dm['ID'].nunique()}\n\n"
					report_6 += "  Descriptive Statistics by Sex:\n"
					report_6 += _omnibus_desc_stats_block(ca2_dm, measure, ["Sex"]) + "\n\n"
					report_6 += _omnibus_anova_block(results_ca2, effect_map_6) + "\n"
					sig_eff = [l for l, k in effect_map_6 if results_ca2.get(k, {}).get('significant')]
					desc = ("Significant effects (2% CA): " + ", ".join(sig_eff)) if sig_eff else "No significant effects in 2% CA group."
					report_6 += _omnibus_interpretation(results_ca2, effect_map_6, desc) + "\n"
				else:
					report_6 += "  Analysis did not complete.\n"
				if posthoc_ca2:
					ph_df = posthoc_ca2.get('results') if isinstance(posthoc_ca2, dict) else posthoc_ca2
					report_6 += _omnibus_posthoc_block(f"Post-hoc: {time_label} × Sex pairwise (2% CA)  [paired t-test, {correction_name}]", ph_df)
			report_6 += _omnibus_summary_table(measures_to_analyze, all_res_6, effect_map_6)
			report_6 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
			_save_report(report_6, _fname("ca_strat_2pct"))
			if choice != 'A':
				continue

		# ── Option 7: 2-Way Mixed ANOVA — CA% × Week ─────────────────────────
		if choice in ('7', 'A'):
			if not use_weeks:
				print("\n[INFO] Option 7 requires weekly data. Re-run with time unit = Weeks.")
			else:
				effect_map_7 = [('Week (within)', 'time'), ('CA%', 'ca_percent'),
				                ('Week × CA%', 'interaction')]
				all_res_7    = []
				report_7 = _omnibus_header(
					"2-WAY MIXED ANOVA — CA% × Week (all animals, sex collapsed)",
					"Week (within) × CA% (between), all animals",
					correction_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
					measures_to_analyze)
				for measure in measures_to_analyze:
					dm = _prep(measure)
					print(f"\n{'='*80}")
					print(f"[{measure}] 2-WAY MIXED ANOVA — CA% × Week (all animals, sex collapsed)")
					print("="*80)
					results_caw = perform_mixed_anova_ca_x_week(dm, measure=measure,
					                                             padjust=padjust_method)
					all_res_7.append(results_caw)
					report_7 += _omnibus_measure_header(measure)
					if results_caw:
						n_animals = dm['ID'].nunique()
						report_7 += f"  N animals: {n_animals}\n\n"
						# Descriptive stats by CA% and by Week
						report_7 += "  Descriptive Statistics by CA%:\n"
						report_7 += _omnibus_desc_stats_block(dm, measure, ["CA (%)"]) + "\n"
						report_7 += "  Descriptive Statistics by Week:\n"
						report_7 += _omnibus_desc_stats_block(dm, measure, ["Week"]) + "\n\n"
						# ANOVA effects block
						report_7 += _omnibus_anova_block(results_caw, effect_map_7) + "\n"
						# Interpretation
						sig_eff = [l for l, k in effect_map_7 if results_caw.get(k, {}).get('significant')]
						desc = ("Significant effects: " + ", ".join(sig_eff)) if sig_eff else "No significant effects."
						report_7 += _omnibus_interpretation(results_caw, effect_map_7, desc) + "\n"
						# Post-hoc 1: Week pairwise
						ph1 = results_caw.get("posthoc_week_pairwise")
						if ph1 is not None and len(ph1):
							report_7 += _omnibus_posthoc_block(
								f"Post-hoc 1 — Pairwise Week comparisons (all CA% pooled)  [paired t-test, {correction_name}]", ph1)
						# Post-hoc 2: CA% at each week
						ph2 = results_caw.get("posthoc_ca_per_week")
						if ph2 is not None and len(ph2):
							report_7 += _omnibus_posthoc_block(
								f"Post-hoc 2 — CA% simple effects at each Week  [independent samples t-test, {correction_name}]", ph2)
						# Post-hoc 3: Week pairwise within each CA%
						ph3 = results_caw.get("posthoc_within_ca", {})
						for ca_val, df_ph in ph3.items():
							if df_ph is not None and len(df_ph):
								report_7 += _omnibus_posthoc_block(
									f"Post-hoc 3 — Week pairwise within CA% = {ca_val}%  [paired t-test, {correction_name}]", df_ph)
					else:
						report_7 += "  Analysis did not complete.\n"
				report_7 += _omnibus_summary_table(measures_to_analyze, all_res_7, effect_map_7)
				report_7 += "\n" + "=" * 80 + "\nEND OF REPORT\n" + "=" * 80 + "\n"
				_save_report(report_7, _fname("ca_x_week_mixed_anova"))
			if choice != 'A':
				continue

		# ── Option 8: Slope analysis ──────────────────────────────────────────
		if choice in ('8', 'A'):
			print("\nSelect measure for slope analysis:")
			print("  [1] Total Change (default)")
			print("  [2] Daily Change")
			print("  [3] Weight")
			if choice == 'A':
				slope_measure = "Total Change"
				print("  (Running with Total Change for 'Run All')")
			else:
				mc = input("Enter choice (1-3, default=1): ").strip()
				slope_measure = {"2": "Daily Change", "3": "Weight"}.get(mc, "Total Change")

			slope_time_unit = time_unit.capitalize()
			df_slope = df.copy()
			if use_weeks and "Week" not in df_slope.columns:
				df_slope = add_week_column(df_slope)

			perform_complete_slope_analysis_cah(
				df_slope,
				measure=slope_measure,
				time_unit=slope_time_unit,
				save_plot=True,
				save_report=True,
				output_dir=out_dir,
			)
			if choice != 'A':
				continue

		# ── Option 9: Interaction plots ───────────────────────────────────────
		if choice == '9':
			print("\nThis option requires analyses to have been run first.")
			print("Running between-subjects ANOVA and 3-way mixed ANOVA to obtain results...")

			save_plots  = input("Save plots to files? (y/n, default=y): ").strip().lower() != 'n'
			show_plots  = input("Display plots interactively? (y/n, default=n): ").strip().lower() == 'y'
			save_dir_p  = out_dir if save_plots else None
			total_plots = 0

			for measure in measures_to_analyze:
				dm = _prep(measure)
				r_between = perform_two_way_between_anova(dm, measure=measure, average_over_days=True)
				r_mixed   = perform_mixed_anova_time(dm, measure=measure,
				                                     time_points=None, time_unit=time_unit)

				if r_between and r_between.get('interaction', {}).get('significant'):
					plot_interaction_effects(between_results=r_between, df=df,
					                         save_dir=save_dir_p, show=show_plots)
					total_plots += 1

				if r_mixed and r_mixed.get('time_sex', {}).get('significant'):
					plot_time_by_sex_interaction(df=dm, measure=measure, time_col=time_col,
					                             results={'interaction': r_mixed['time_sex']},
					                             save_dir=save_dir_p, show=show_plots)
					total_plots += 1

				if r_mixed and r_mixed.get('time_ca', {}).get('significant'):
					plot_time_by_ca_interaction(df=dm, measure=measure, time_col=time_col,
					                            results={'interaction': r_mixed['time_ca']},
					                            save_dir=save_dir_p, show=show_plots)
					total_plots += 1

			print(f"\nTotal plots generated: {total_plots}")
			if save_dir_p:
				print(f"Plots saved to: {out_dir.resolve()}")
			continue

		# ── 'A' — clean-up message ────────────────────────────────────────────
		if choice == 'A':
			print("\n" + "="*80)
			print("RUN ALL COMPLETE")
			print(f"All reports saved to: {out_dir.resolve()}")
			print("="*80)
			continue

		if choice not in ('1','2','3','4','5','6','7','8','9','A','Q'):
			print(f"  [!] Unknown option '{choice}'. Enter a number 1-9, A, or Q.")

	return df


if __name__ == "__main__":
	df = main()
