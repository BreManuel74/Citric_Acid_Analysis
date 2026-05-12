"""
RV Cohort Weight Analysis

Analyzes body-weight changes across CA% concentrations (0% and 2%) for the
RV cohort.  Does not include sex as a factor (single-sex or sex-collapsed
cohort).  Designed as a standalone menu-driven script.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python RV_weight_analysis.py

  The script automatically loads  RV_cohort/RV_master_data.csv  from the
  same directory as this file (no file picker needed).  On startup you choose
  a time unit (Days or Weeks), then an interactive text menu is presented.

  Required input: RV_master_data.csv  (one row per animal per day;
    columns must include ID, CA (%), Date, Total Change, Daily Change)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MENU OPTIONS / ANALYSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [1]  Total Change by ID           — individual lines, colored by CA%
  [2]  Daily Change by ID           — individual lines, colored by CA%
  [3]  Total Change by CA%          — group mean ± SEM, colored by CA%
  [4]  Daily Change by CA%          — group mean ± SEM, colored by CA%
  [P]  All plots (1–4) at once
  [5]  Slope analysis: Total Change ~ Day/Week between CA% groups
         Per-animal linear regression, Mann-Whitney U between CA% groups
         (Holm-Bonferroni corrected), within-group slope significance;
         scatter/box comparison plot + text report saved to disk
  [6]  Slope analysis: Daily Change ~ Day/Week between CA% groups
  [A]  Run all analyses (plots + both slope analyses) and save all outputs
  [Q]  Quit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT FILES (saved to the RV_cohort directory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RV_total_change_by_id.png / .svg
  RV_daily_change_by_id.png / .svg
  RV_total_change_by_ca.png / .svg
  RV_daily_change_by_ca.png / .svg
  RV_slope_analysis_Total_Change_*.png / .svg
  RV_slope_analysis_Daily_Change_*.png / .svg
  RV_slope_report_*.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Required : pandas, numpy, matplotlib, scipy
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

# Configure matplotlib for publication-quality plots
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
    "figure.figsize": (3, 2.5),
})

# ==============================================================================
# PLOTTING HELPER FUNCTIONS
# ==============================================================================

def _ca_to_style(ca_pct: Optional[int]) -> Tuple[str, str]:
	"""Return (color, marker) based on CA%: 0=black/triangle, 2=red/square, unknown=gray/triangle."""
	if ca_pct == 0:
		return ("black", "^")
	if ca_pct == 2:
		return ("#e8262a", "s")
	return ("gray", "^")


def _get_id_ca_map(df: pd.DataFrame) -> dict:
	"""Build a mapping from ID -> CA (%) using the first non-null value per ID."""
	cdf = clean_rv_dataframe(df)
	if "ID" not in cdf.columns or "CA (%)" not in cdf.columns:
		return {}

	def _norm_ca(x: pd.Series) -> Optional[int]:
		valid = x.dropna().unique()
		if len(valid) > 0:
			return int(valid[0])
		return None

	ca_map = cdf.groupby("ID")["CA (%)"].apply(_norm_ca).to_dict()
	return {str(k): v for k, v in ca_map.items()}


def apply_common_plot_style(
	ax: plt.Axes,
	start_x_at_zero: bool = False,
	remove_top_right: bool = True,
	remove_x_margins: bool = True,
	remove_y_margins: bool = True,
	ticks_in: bool = True,
) -> None:
	"""Apply common styling: remove spines, set tick directions, adjust margins."""
	if remove_top_right:
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

	if ticks_in:
		ax.tick_params(axis='both', direction='in', which='both')

	if remove_x_margins:
		ax.margins(x=0)

	if remove_y_margins:
		ax.margins(y=0)


def build_daily_change_series_by_id(df: pd.DataFrame) -> dict:
	"""
	For each ID, return a pandas Series of 'Daily Change' indexed by Day number.

	Returns:
		dict[str, pd.Series]: Mapping from ID to Series of daily changes
	"""
	required = {"ID", "Day", "Daily Change"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

	cdf = clean_rv_dataframe(df)
	if "Day" not in cdf.columns:
		cdf = add_day_number_column(cdf)

	series_by_id: dict = {}
	for gid, g in cdf.groupby("ID", dropna=True):
		g = g.dropna(subset=["Daily Change"])
		if g.empty:
			continue
		g = g.copy()
		g["Day"] = g["Day"].astype("Int64")
		ser = g.set_index("Day")["Daily Change"].sort_index()
		s = ser.groupby(level=0).last()
		s.name = str(gid)
		series_by_id[str(gid)] = s

	return series_by_id


def build_total_change_series_by_id(df: pd.DataFrame) -> dict:
	"""
	For each ID, return a pandas Series of 'Total Change' indexed by Day number.

	Returns:
		dict[str, pd.Series]: Mapping from ID to Series of total changes
	"""
	required = {"ID", "Day", "Total Change"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

	cdf = clean_rv_dataframe(df)
	if "Day" not in cdf.columns:
		cdf = add_day_number_column(cdf)

	series_by_id: dict = {}
	for gid, g in cdf.groupby("ID", dropna=True):
		g = g.dropna(subset=["Total Change"])
		if g.empty:
			continue
		g = g.copy()
		g["Day"] = g["Day"].astype("Int64")
		ser = g.set_index("Day")["Total Change"].sort_index()
		s = ser.groupby(level=0).last()
		s.name = str(gid)
		series_by_id[str(gid)] = s

	return series_by_id


# ==============================================================================
# DATA LOADING AND CLEANING
# ==============================================================================

def load_rv_data(csv_path: Union[str, Path]) -> pd.DataFrame:
	"""
	Load the RV master CSV file.

	Parameters:
		csv_path: Path to RV_master_data.csv

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


def clean_rv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Clean and standardize column types for RV cohort data.

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

	# Standardize column names (case-insensitive)
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
	if "Condition" in df.columns:
		df["CA (%)"] = (
			df["Condition"]
			.astype(str)
			.str.strip()
			.str.replace("%", "", regex=False)
		)
		df["CA (%)"] = pd.to_numeric(df["CA (%)"], errors="coerce")

	# Numeric coercions
	for ncol in ["Daily Change", "Total Change", "Weight"]:
		if ncol in df.columns:
			df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

	# Normalize Sex
	if "Sex" in df.columns:
		df["Sex"] = (
			df["Sex"]
			.astype(str)
			.str.strip()
			.str.upper()
			.map(lambda x: "M" if x.startswith("M") else ("F" if x.startswith("F") else None))
		)

	# Drop rows without essential keys
	if "ID" in df.columns and "Date" in df.columns:
		df = df.dropna(subset=["ID", "Date"]).copy()

	# Sort for stable grouping
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
	df = df.sort_values(["ID", "Date"]).reset_index(drop=True)
	first_dates = df.groupby("ID")["Date"].transform("min")
	df["Day"] = (df["Date"] - first_dates).dt.days

	return df


def add_week_column(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add a 'Week' column based on Day number, excluding Day 0.

	Week assignment:
	- Day 0: NaN (baseline)
	- Days 1-7: Week 1, Days 8-14: Week 2, etc.
	"""
	if "Day" not in df.columns:
		print("[WARNING] 'Day' column not found. Cannot add 'Week' column.")
		return df

	df = df.copy()
	df["Week"] = np.where(df["Day"] == 0, np.nan, ((df["Day"] - 1) // 7) + 1)

	return df


def average_by_week(df: pd.DataFrame, measure: str = "Total Change") -> pd.DataFrame:
	"""Average weight measures within each week per animal, excluding Day 0."""
	if not {"ID", "Week"}.issubset(df.columns):
		raise ValueError("DataFrame must have 'ID' and 'Week' columns")
	if measure not in df.columns:
		raise ValueError(f"Measure '{measure}' not found in DataFrame")

	df = df[df["Week"].notna()].copy()

	meta_cols = ["ID", "Week"]
	for col in ["Sex", "CA (%)", "Strain", "Condition"]:
		if col in df.columns:
			meta_cols.append(col)

	weekly_df = (
		df.groupby(meta_cols, dropna=False)[measure]
		.mean()
		.reset_index()
	)

	return weekly_df


def summarize_dataframe(df: pd.DataFrame) -> None:
	"""Print a summary of the DataFrame structure."""
	print("\n" + "=" * 80)
	print("DATAFRAME SUMMARY")
	print("=" * 80)
	print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
	print(f"\nColumns: {list(df.columns)}")

	if "ID" in df.columns:
		print(f"\nUnique IDs: {df['ID'].nunique()}")
		print(f"  IDs: {sorted(df['ID'].unique())}")

	if "Sex" in df.columns:
		print(f"\nSex distribution:")
		for sex, n in df.groupby("Sex")["ID"].nunique().items():
			print(f"  {sex}: {n} animals")

	if "CA (%)" in df.columns:
		print(f"\nCA% distribution:")
		for ca, n in df.groupby("CA (%)")["ID"].nunique().items():
			print(f"  {ca}%: {n} animals")

	if {"Sex", "CA (%)"}.issubset(df.columns):
		print(f"\n2x2 Design (Sex x CA%):")
		for (sex, ca), n in df.groupby(["Sex", "CA (%)"])["ID"].nunique().items():
			print(f"  {sex}, {ca}%: {n} animals")

	if "Date" in df.columns:
		print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")

	if "Day" in df.columns:
		day_range = f"{int(df['Day'].min())}-{int(df['Day'].max())}"
		print(f"\nDay range: {day_range} ({df['Day'].nunique()} unique days)")

	print("\n" + "=" * 80)


# ==============================================================================
# INDIVIDUAL & GROUP PLOTS
# ==============================================================================

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
	Plot Total Change over Day for each animal, colored by CA%.
	0% CA = black/triangle, 2% CA = red/square.
	"""
	series_by_id = build_total_change_series_by_id(df)
	ca_map = _get_id_ca_map(df)

	if ids is not None:
		series_by_id = {k: v for k, v in series_by_id.items() if k in set(ids)}

	if not series_by_id:
		raise ValueError("No series available to plot. Check input DataFrame and 'ids' filter.")

	fig, ax = plt.subplots()

	ca_groups_plotted = {}

	for mid, s in series_by_id.items():
		if s.empty:
			continue
		ca_pct = ca_map.get(mid)
		color, marker = _ca_to_style(ca_pct)

		if ca_pct not in ca_groups_plotted:
			label = f"{ca_pct}% CA"
			ca_groups_plotted[ca_pct] = True
		else:
			label = None

		ax.plot(
			s.index,
			s.values,
			label=label,
			marker=marker,
			markersize=3,
			linewidth=1.5,
			alpha=0.9,
			color=color,
		)

	ax.set_xlabel("Day")
	ax.set_ylabel("Total Change (%)")
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Total Weight Change by Day per Animal"
	ax.set_title(title, weight='bold')

	apply_common_plot_style(
		ax,
		start_x_at_zero=False,
		remove_top_right=True,
		remove_x_margins=True,
		remove_y_margins=True,
		ticks_in=True,
	)

	ax.legend(title="CA%", loc="best")
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
		svg_path = save_path.with_suffix(".svg")
		fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {svg_path}")
	elif save_svg:
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
	Plot Daily Change over Day for each animal, colored by CA%.
	0% CA = black/triangle, 2% CA = red/square.
	"""
	series_by_id = build_daily_change_series_by_id(df)
	ca_map = _get_id_ca_map(df)

	if ids is not None:
		series_by_id = {k: v for k, v in series_by_id.items() if k in set(ids)}

	if not series_by_id:
		raise ValueError("No series available to plot. Check input DataFrame and 'ids' filter.")

	fig, ax = plt.subplots()

	ca_groups_plotted = {}

	for mid, s in series_by_id.items():
		if s.empty:
			continue
		ca_pct = ca_map.get(mid)
		color, marker = _ca_to_style(ca_pct)

		if ca_pct not in ca_groups_plotted:
			label = f"{ca_pct}% CA"
			ca_groups_plotted[ca_pct] = True
		else:
			label = None

		ax.plot(
			s.index,
			s.values,
			label=label,
			marker=marker,
			markersize=3,
			linewidth=1.5,
			alpha=0.9,
			color=color,
		)

	ax.set_xlabel("Day")
	ax.set_ylabel("Daily Change (%)")
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Daily Weight Change by Day per Animal"
	ax.set_title(title, weight='bold')

	apply_common_plot_style(
		ax,
		start_x_at_zero=False,
		remove_top_right=True,
		remove_x_margins=True,
		remove_y_margins=True,
		ticks_in=True,
	)

	ax.legend(title="CA%", loc="best")
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
		svg_path = save_path.with_suffix(".svg")
		fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {svg_path}")
	elif save_svg:
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


def plot_total_change_by_ca(
	df: pd.DataFrame,
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
	save_svg: bool = False,
	svg_filename: Optional[str] = None,
) -> plt.Figure:
	"""
	Plot CA%-averaged Total Change with SEM shading.
	0% CA = black/triangle, 2% CA = red/square.
	"""
	series_by_id = build_total_change_series_by_id(df)
	ca_map = _get_id_ca_map(df)

	ca0_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 0}
	ca2_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 2}

	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		return combined.mean(axis=1), combined.sem(axis=1)

	ca0_mean, ca0_sem = _compute_mean_sem(ca0_series)
	ca2_mean, ca2_sem = _compute_mean_sem(ca2_series)

	ca0_color, ca0_marker = _ca_to_style(0)
	ca2_color, ca2_marker = _ca_to_style(2)

	fig, ax = plt.subplots()

	if not ca0_mean.empty:
		ax.plot(ca0_mean.index, ca0_mean.values, label="0% CA", color=ca0_color,
				marker=ca0_marker, markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca0_mean.index, ca0_mean - ca0_sem, ca0_mean + ca0_sem,
						color=ca0_color, alpha=0.2)

	if not ca2_mean.empty:
		ax.plot(ca2_mean.index, ca2_mean.values, label="2% CA", color=ca2_color,
				marker=ca2_marker, markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca2_mean.index, ca2_mean - ca2_sem, ca2_mean + ca2_sem,
						color=ca2_color, alpha=0.2)

	ax.set_xlabel("Day")
	ax.set_ylabel("Total Change (%, Mean \u00b1 SEM)")
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Total Weight Change by CA% (Mean \u00b1 SEM)"
	ax.set_title(title, weight='bold')

	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)

	ax.legend(title="CA%", loc="best")
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
		svg_path = save_path.with_suffix(".svg")
		fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {svg_path}")
	elif save_svg:
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
	Plot CA%-averaged Daily Change with SEM shading.
	0% CA = black/triangle, 2% CA = red/square.
	"""
	series_by_id = build_daily_change_series_by_id(df)
	ca_map = _get_id_ca_map(df)

	ca0_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 0}
	ca2_series = {mid: s for mid, s in series_by_id.items() if ca_map.get(mid) == 2}

	def _compute_mean_sem(series_dict: dict) -> Tuple[pd.Series, pd.Series]:
		if not series_dict:
			return pd.Series(dtype=float), pd.Series(dtype=float)
		combined = pd.DataFrame(series_dict)
		return combined.mean(axis=1), combined.sem(axis=1)

	ca0_mean, ca0_sem = _compute_mean_sem(ca0_series)
	ca2_mean, ca2_sem = _compute_mean_sem(ca2_series)

	ca0_color, ca0_marker = _ca_to_style(0)
	ca2_color, ca2_marker = _ca_to_style(2)

	fig, ax = plt.subplots()

	if not ca0_mean.empty:
		ax.plot(ca0_mean.index, ca0_mean.values, label="0% CA", color=ca0_color,
				marker=ca0_marker, markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca0_mean.index, ca0_mean - ca0_sem, ca0_mean + ca0_sem,
						color=ca0_color, alpha=0.2)

	if not ca2_mean.empty:
		ax.plot(ca2_mean.index, ca2_mean.values, label="2% CA", color=ca2_color,
				marker=ca2_marker, markersize=5, linewidth=2, alpha=0.9)
		ax.fill_between(ca2_mean.index, ca2_mean - ca2_sem, ca2_mean + ca2_sem,
						color=ca2_color, alpha=0.2)

	ax.set_xlabel("Day")
	ax.set_ylabel("Daily Change (%, Mean \u00b1 SEM)")
	ax.grid(False)
	ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

	if title is None:
		title = "Daily Weight Change by CA% (Mean \u00b1 SEM)"
	ax.set_title(title, weight='bold')

	apply_common_plot_style(ax, start_x_at_zero=False, remove_top_right=True,
							remove_x_margins=True, remove_y_margins=True, ticks_in=True)

	ax.legend(title="CA%", loc="best")
	fig.tight_layout()

	if save_path is not None:
		save_path = Path(save_path)
		fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
		svg_path = save_path.with_suffix(".svg")
		fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
		print(f"  [OK] Saved SVG: {svg_path}")
	elif save_svg:
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

def calculate_animal_slopes_rv(
	df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
) -> pd.DataFrame:
	"""
	Calculate linear regression slopes for each animal's weight change over time.

	For each animal, fits: measure ~ time_unit (e.g., Total Change ~ Week).

	Parameters:
		df: Cleaned DataFrame with ID, Sex, CA (%), Day/Week, and measure columns
		measure: Weight measure to analyze ('Total Change', 'Daily Change', 'Weight')
		time_unit: Time variable to use ('Week' or 'Day')

	Returns:
		DataFrame with one row per animal containing slope statistics
	"""
	print("\n" + "=" * 80)
	print(f"CALCULATING ANIMAL SLOPES: {measure} ~ {time_unit}")
	print("=" * 80)

	required_cols = ['ID', 'CA (%)']
	if 'Sex' in df.columns:
		required_cols.append('Sex')
	required_cols.append(measure)

	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	if time_unit not in df.columns:
		if time_unit == "Week" and "Day" in df.columns:
			print(f"[INFO] Adding Week column...")
			df = add_week_column(df)
		elif time_unit == "Day" and "Day" not in df.columns:
			print(f"[INFO] Adding Day column...")
			df = add_day_number_column(df)
		else:
			raise ValueError(f"Time column '{time_unit}' not found and cannot be created")

	available_cols = [col for col in ['ID', 'Sex', 'CA (%)', time_unit, measure] if col in df.columns]
	analysis_df = df[available_cols].copy().dropna()

	if time_unit == "Week":
		analysis_df = analysis_df[analysis_df[time_unit] > 0].copy()

	print(f"  Animals: {analysis_df['ID'].nunique()}")
	print(f"  CA% groups: {sorted(analysis_df['CA (%)'].unique())}")
	print(f"  {time_unit} range: {analysis_df[time_unit].min():.0f} to {analysis_df[time_unit].max():.0f}")

	slopes_data = []

	for animal_id in analysis_df['ID'].unique():
		animal_data = analysis_df[analysis_df['ID'] == animal_id].copy()

		ca_pct = animal_data['CA (%)'].iloc[0]
		sex = animal_data['Sex'].iloc[0] if 'Sex' in animal_data.columns else None

		x = animal_data[time_unit].values
		y = animal_data[measure].values

		if len(x) < 2:
			continue

		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

		row = {
			'ID': animal_id,
			'CA (%)': ca_pct,
			'Slope': slope,
			'Intercept': intercept,
			'R2': r_value ** 2,
			'P_value': p_value,
			'Std_Error': std_err,
			'N_points': len(x),
		}
		if sex is not None:
			row['Sex'] = sex

		slopes_data.append(row)

	slopes_df = pd.DataFrame(slopes_data)

	print(f"\nSlope Summary by CA% Group:")
	for ca_val in sorted(slopes_df['CA (%)'].unique()):
		ca_slopes = slopes_df[slopes_df['CA (%)'] == ca_val]['Slope']
		print(f"  {ca_val}% CA (n={len(ca_slopes)}):")
		print(f"    Mean slope:   {ca_slopes.mean():.4f} {measure} per {time_unit}")
		print(f"    Median slope: {ca_slopes.median():.4f} {measure} per {time_unit}")
		print(f"    SD:           {ca_slopes.std():.4f}")
		print(f"    Range:        [{ca_slopes.min():.4f}, {ca_slopes.max():.4f}]")
		r2_vals = slopes_df[slopes_df['CA (%)'] == ca_val]['R2']
		print(f"    Mean R\u00b2:      {r2_vals.mean():.4f}")

	return slopes_df


def compare_slopes_within_ca_groups(slopes_df: pd.DataFrame) -> Dict:
	"""
	Analyze slope variability within each CA% group using descriptive statistics.

	Returns:
		Dictionary with within-group statistics
	"""
	print("\n" + "=" * 80)
	print("WITHIN-GROUP SLOPE VARIABILITY ANALYSIS")
	print("=" * 80)

	results = {'group_stats': []}

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
			'CV': (group_slopes.std() / group_slopes.mean() * 100) if group_slopes.mean() != 0 else np.nan,
		}

		results['group_stats'].append(group_stat)

		print(f"\n{ca_val}% CA Group (n={group_stat['N']}):")
		print(f"  Mean \u00b1 SEM:          {group_stat['Mean']:.4f} \u00b1 {group_stat['SEM']:.4f}")
		print(f"  Median (IQR):       {group_stat['Median']:.4f} ({group_stat['IQR']:.4f})")
		print(f"  SD:                 {group_stat['SD']:.4f}")
		print(f"  Coefficient of Var: {group_stat['CV']:.2f}%")
		print(f"  Range:              [{group_stat['Min']:.4f}, {group_stat['Max']:.4f}]")

	return results


def compare_slopes_between_ca_groups(slopes_df: pd.DataFrame) -> Dict:
	"""
	Statistically compare slopes between 0% and 2% CA groups.

	Performs:
	1. Welch's t-test (unequal variances)
	2. Mann-Whitney U test (non-parametric alternative)
	3. Cohen's d effect size
	4. 95% CI for mean difference

	Returns:
		Dictionary with test results and effect sizes
	"""
	print("\n" + "=" * 80)
	print("BETWEEN-GROUP SLOPE COMPARISON: 0% CA vs 2% CA")
	print("=" * 80)

	ca_groups = sorted(slopes_df['CA (%)'].unique())

	if len(ca_groups) != 2:
		print(f"Warning: Expected 2 CA% groups, found {len(ca_groups)}. Returning empty results.")
		return {}

	ca_0, ca_1 = ca_groups
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
		'mean_diff': slopes_1.mean() - slopes_0.mean(),
	}

	# Welch's t-test
	t_stat, t_p = stats.ttest_ind(slopes_0, slopes_1, equal_var=False)
	print(f"\nWelch's T-Test (unequal variances):")
	print(f"  t = {t_stat:.4f}, p = {t_p:.4f}")
	print(f"  Result: {'Significant' if t_p < 0.05 else 'Not significant'} (alpha = 0.05)")

	n1, n2 = len(slopes_0), len(slopes_1)
	s1_sq, s2_sq = slopes_0.var(), slopes_1.var()
	df_welch = (s1_sq / n1 + s2_sq / n2) ** 2 / (
		(s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
	)

	results['t_test'] = {
		'statistic': t_stat,
		'p_value': t_p,
		'df': df_welch,
		'significant': t_p < 0.05,
	}

	# Mann-Whitney U test
	u_stat, u_p = stats.mannwhitneyu(slopes_0, slopes_1, alternative='two-sided')
	print(f"\nMann-Whitney U Test (non-parametric):")
	print(f"  U = {u_stat:.4f}, p = {u_p:.4f}")
	print(f"  Result: {'Significant' if u_p < 0.05 else 'Not significant'} (alpha = 0.05)")

	results['mann_whitney'] = {
		'statistic': u_stat,
		'p_value': u_p,
		'significant': u_p < 0.05,
	}

	# Cohen's d
	s1 = slopes_0.std(ddof=1)
	s2 = slopes_1.std(ddof=1)
	pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
	cohens_d = (slopes_1.mean() - slopes_0.mean()) / pooled_sd

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
	print(f"  d = {cohens_d:.4f}  ({interpretation})")

	results['effect_size'] = {
		'cohens_d': cohens_d,
		'pooled_sd': pooled_sd,
		'interpretation': interpretation,
	}

	# 95% CI for mean difference
	se_diff = np.sqrt(slopes_0.var() / n1 + slopes_1.var() / n2)
	t_crit = stats.t.ppf(0.975, df_welch)
	ci_lower = results['mean_diff'] - t_crit * se_diff
	ci_upper = results['mean_diff'] + t_crit * se_diff

	print(f"\n95% CI for Mean Difference:")
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
		'ci_95_upper': ci_upper,
	}

	return results


def plot_slopes_comparison_rv(
	slopes_df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
	title: Optional[str] = None,
	save_path: Optional[Path] = None,
	show: bool = True,
) -> Dict[str, plt.Figure]:
	"""
	Create visualizations comparing slopes between CA% groups.

	Creates 3 individual figures:
	1. Boxplot with individual points
	2. Bar chart (Mean ± SEM)
	3. Histogram overlay

	Returns:
		Dictionary with 'boxplot', 'barplot', 'histogram' Figure objects
	"""
	print("\n" + "=" * 80)
	print("CREATING SLOPE COMPARISON PLOTS")
	print("=" * 80)

	ca_groups = sorted(slopes_df['CA (%)'].unique())
	# Derive colors and markers from _ca_to_style() to enforce CA% color scheme
	colors = [_ca_to_style(ca)[0] for ca in ca_groups]
	markers = [_ca_to_style(ca)[1] for ca in ca_groups]
	_ms = plt.rcParams.get('lines.markersize', 3)
	figures = {}

	# -- Figure 1: Boxplot with individual points --
	fig1, ax1 = plt.subplots()

	box_data = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values for ca in ca_groups]
	bp = ax1.boxplot(box_data, positions=range(len(ca_groups)), widths=0.6,
					 patch_artist=True, showfliers=False)

	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.7)
	for element in ('whiskers', 'caps', 'medians'):
		for line in bp[element]:
			line.set_linewidth(plt.rcParams.get('lines.linewidth', 0.9))

	for i, ca in enumerate(ca_groups):
		slopes = slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values
		x_jitter = np.random.default_rng(42).normal(i, 0.04, size=len(slopes))
		ax1.scatter(x_jitter, slopes, alpha=0.9, color=colors[i], s=_ms ** 2,
					marker='o', edgecolors='black', zorder=3,
					linewidth=plt.rcParams.get('lines.linewidth', 0.9) * 0.5)

	# -- Mann-Whitney U test annotation --
	if len(ca_groups) == 2:
		_s0 = slopes_df[slopes_df['CA (%)'] == ca_groups[0]]['Slope'].values
		_s1 = slopes_df[slopes_df['CA (%)'] == ca_groups[1]]['Slope'].values
		_u_stat, _mw_p = stats.mannwhitneyu(_s0, _s1, alternative='two-sided')
		if _mw_p < 0.001:
			_p_str = '*** p < 0.001'
		elif _mw_p < 0.01:
			_p_str = f'** p = {_mw_p:.3f}'
		elif _mw_p < 0.05:
			_p_str = f'* p = {_mw_p:.3f}'
		else:
			_p_str = f'ns p = {_mw_p:.3f}'
		_all = np.concatenate([_s0, _s1])
		_d_max = np.max(_all)
		_d_range = np.max(_all) - np.min(_all) if np.max(_all) != np.min(_all) else 1.0
		_brk_y = _d_max + _d_range * 0.12
		_tick_h = _d_range * 0.04
		_p_lw = plt.rcParams.get('lines.linewidth', 0.9)
		_x0, _x1 = 0, len(ca_groups) - 1
		ax1.plot([_x0, _x0, _x1, _x1],
				 [_brk_y - _tick_h, _brk_y, _brk_y, _brk_y - _tick_h],
				 lw=_p_lw, color='black', clip_on=False)
		ax1.text((_x0 + _x1) / 2, _brk_y + _d_range * 0.02, _p_str,
				 ha='center', va='bottom',
				 fontsize=plt.rcParams.get('font.size', 8))

	ax1.set_xticks(range(len(ca_groups)))
	ax1.set_xticklabels([f'{ca}% CA' for ca in ca_groups])
	ax1.set_ylabel(f'Slope ({measure} per {time_unit})')
	ax1.grid(False)
	plot_title = 'Slope Distribution by Group'
	if title:
		plot_title = f'{title}: {plot_title}'
	ax1.set_title(plot_title)
	apply_common_plot_style(ax1, remove_top_right=True, ticks_in=True,
							remove_x_margins=False, remove_y_margins=False)
	fig1.tight_layout()
	figures['boxplot'] = fig1

	if save_path is not None:
		bp_path = Path(str(save_path).replace('.png', '_boxplot.png'))
		fig1.savefig(str(bp_path), dpi=300, bbox_inches='tight')
		fig1.savefig(str(bp_path.with_suffix('.svg')), format='svg', bbox_inches='tight')
		print(f"  [OK] Saved boxplot: {bp_path.with_suffix('.svg')}")

	if show:
		plt.show()
	else:
		plt.close(fig1)

	# -- Figure 2: Bar chart with SEM error bars --
	fig2, ax2 = plt.subplots()

	means = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].mean() for ca in ca_groups]
	sems = [slopes_df[slopes_df['CA (%)'] == ca]['Slope'].sem() for ca in ca_groups]

	ax2.bar(range(len(ca_groups)), means, yerr=sems, capsize=4,
			color=colors, alpha=0.6, edgecolor='black',
			linewidth=plt.rcParams.get('lines.linewidth', 0.9))

	ax2.set_xticks(range(len(ca_groups)))
	ax2.set_xticklabels([f'{ca}% CA' for ca in ca_groups])
	ax2.set_ylabel(f'Mean Slope \u00b1 SEM ({measure} per {time_unit})')
	ax2.grid(False)
	plot_title = 'Mean Slopes with Error Bars'
	if title:
		plot_title = f'{title}: {plot_title}'
	ax2.set_title(plot_title)
	apply_common_plot_style(ax2, remove_top_right=True, ticks_in=True,
							remove_x_margins=False, remove_y_margins=True)
	fig2.tight_layout()
	figures['barplot'] = fig2

	if save_path is not None:
		bar_path = Path(str(save_path).replace('.png', '_barplot.png'))
		fig2.savefig(str(bar_path), dpi=300, bbox_inches='tight')
		fig2.savefig(str(bar_path.with_suffix('.svg')), format='svg', bbox_inches='tight')
		print(f"  [OK] Saved barplot: {bar_path.with_suffix('.svg')}")

	if show:
		plt.show()
	else:
		plt.close(fig2)

	# -- Figure 3: Histogram overlay --
	fig3, ax3 = plt.subplots()

	for i, ca in enumerate(ca_groups):
		slopes = slopes_df[slopes_df['CA (%)'] == ca]['Slope'].values
		ax3.hist(slopes, bins=8, alpha=0.5, color=colors[i], label=f'{ca}% CA',
				 edgecolor='black',
				 linewidth=plt.rcParams.get('lines.linewidth', 0.9) * 0.5)

	ax3.set_xlabel(f'Slope ({measure} per {time_unit})')
	ax3.set_ylabel('Frequency')
	ax3.grid(False)
	plot_title = 'Slope Distribution Histogram'
	if title:
		plot_title = f'{title}: {plot_title}'
	ax3.set_title(plot_title)
	ax3.legend()
	apply_common_plot_style(ax3, remove_top_right=True, ticks_in=True,
							remove_x_margins=True, remove_y_margins=True)
	fig3.tight_layout()
	figures['histogram'] = fig3

	if save_path is not None:
		hist_path = Path(str(save_path).replace('.png', '_histogram.png'))
		fig3.savefig(str(hist_path), dpi=300, bbox_inches='tight')
		fig3.savefig(str(hist_path.with_suffix('.svg')), format='svg', bbox_inches='tight')
		print(f"  [OK] Saved histogram: {hist_path.with_suffix('.svg')}")

	if show:
		plt.show()
	else:
		plt.close(fig3)

	return figures


def generate_slope_analysis_report_rv(
	slopes_df: pd.DataFrame,
	within_results: Dict,
	between_results: Dict,
	measure: str = "Total Change",
	time_unit: str = "Week",
) -> str:
	"""
	Generate a comprehensive text report of slope analysis results.

	Parameters:
		slopes_df: DataFrame from calculate_animal_slopes_rv()
		within_results: Dictionary from compare_slopes_within_ca_groups()
		between_results: Dictionary from compare_slopes_between_ca_groups()
		measure: Weight measure analyzed
		time_unit: Time unit used

	Returns:
		Formatted text report
	"""
	lines = []

	lines.append("=" * 80)
	lines.append("SLOPE ANALYSIS REPORT: RV COHORT")
	lines.append("=" * 80)
	lines.append(f"\nMeasure:   {measure}")
	lines.append(f"Time Unit: {time_unit}")
	lines.append(f"Date:      {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

	# Section 1: Individual animal slopes
	lines.append("\n\n" + "=" * 80)
	lines.append("SECTION 1: INDIVIDUAL ANIMAL SLOPES")
	lines.append("=" * 80)
	lines.append(f"\nLinear regression: {measure} ~ {time_unit}")

	has_sex = 'Sex' in slopes_df.columns

	for ca_val in sorted(slopes_df['CA (%)'].unique()):
		ca_data = slopes_df[slopes_df['CA (%)'] == ca_val].sort_values('Slope', ascending=False)
		lines.append(f"\n{ca_val}% CA Group (n={len(ca_data)}):")
		lines.append("-" * 80)
		if has_sex:
			lines.append(f"{'ID':<15} {'Sex':<6} {'Slope':>10} {'R\u00b2':>8} {'P-value':>10} {'N Points':>10}")
		else:
			lines.append(f"{'ID':<15} {'Slope':>10} {'R\u00b2':>8} {'P-value':>10} {'N Points':>10}")
		lines.append("-" * 80)

		for _, row in ca_data.iterrows():
			if has_sex:
				lines.append(f"{str(row['ID']):<15} {row['Sex']:<6} {row['Slope']:>10.4f} "
							 f"{row['R2']:>8.4f} {row['P_value']:>10.4f} {row['N_points']:>10.0f}")
			else:
				lines.append(f"{str(row['ID']):<15} {row['Slope']:>10.4f} "
							 f"{row['R2']:>8.4f} {row['P_value']:>10.4f} {row['N_points']:>10.0f}")

	# Section 2: Within-group variability
	lines.append("\n\n" + "=" * 80)
	lines.append("SECTION 2: WITHIN-GROUP VARIABILITY")
	lines.append("=" * 80)

	for group_stat in within_results['group_stats']:
		lines.append(f"\n{group_stat['CA (%)']}% CA Group (n={group_stat['N']}):")
		lines.append("-" * 80)
		lines.append(f"  Mean:               {group_stat['Mean']:.4f}")
		lines.append(f"  Median:             {group_stat['Median']:.4f}")
		lines.append(f"  Standard Deviation: {group_stat['SD']:.4f}")
		lines.append(f"  SEM:                {group_stat['SEM']:.4f}")
		lines.append(f"  Min:                {group_stat['Min']:.4f}")
		lines.append(f"  Max:                {group_stat['Max']:.4f}")
		lines.append(f"  IQR:                {group_stat['IQR']:.4f}")
		lines.append(f"  Coefficient of Var: {group_stat['CV']:.2f}%")

	# Section 3: Between-group comparison
	lines.append("\n\n" + "=" * 80)
	lines.append("SECTION 3: BETWEEN-GROUP COMPARISON")
	lines.append("=" * 80)

	if between_results and 'ca_groups' in between_results:
		ca_0, ca_1 = between_results['ca_groups']

		lines.append(f"\nGroup Comparison: {ca_0}% CA vs {ca_1}% CA")
		lines.append("-" * 80)
		lines.append(f"  {ca_0}% CA: Mean = {between_results['mean_0']:.4f}, "
					 f"SD = {between_results['sd_0']:.4f} (n={between_results['n_0']})")
		lines.append(f"  {ca_1}% CA: Mean = {between_results['mean_1']:.4f}, "
					 f"SD = {between_results['sd_1']:.4f} (n={between_results['n_1']})")
		lines.append(f"  Difference in means: {between_results['mean_diff']:.4f}")

		# Welch's t-test
		t = between_results['t_test']
		lines.append("\nWelch's T-Test (unequal variances):")
		lines.append("-" * 80)
		lines.append(f"  t({t['df']:.2f}) = {t['statistic']:.4f},  p = {t['p_value']:.4f}")
		if t['p_value'] < 0.001:
			sig_str = "p < 0.001 (highly significant)"
		elif t['p_value'] < 0.01:
			sig_str = "p < 0.01 (very significant)"
		elif t['p_value'] < 0.05:
			sig_str = "p < 0.05 (significant)"
		else:
			sig_str = "p >= 0.05 (not significant)"
		lines.append(f"  Result: {sig_str}")

		# Mann-Whitney
		mw = between_results['mann_whitney']
		lines.append("\nMann-Whitney U Test (non-parametric):")
		lines.append("-" * 80)
		lines.append(f"  U = {mw['statistic']:.4f},  p = {mw['p_value']:.4f}")
		lines.append(f"  Result: {'Significant' if mw['significant'] else 'Not significant'} (alpha = 0.05)")

		# Effect size
		es = between_results['effect_size']
		lines.append("\nEffect Size (Cohen's d):")
		lines.append("-" * 80)
		lines.append(f"  Cohen's d = {es['cohens_d']:.4f}  ({es['interpretation'].capitalize()} effect)")

		# 95% CI
		ci = between_results['confidence_interval']
		lines.append("\n95% CI for Mean Difference:")
		lines.append("-" * 80)
		lines.append(f"  Mean Difference: {ci['mean_diff']:.4f}")
		lines.append(f"  95% CI: [{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}]")
		if ci['ci_95_lower'] * ci['ci_95_upper'] > 0:
			lines.append("  Interpretation: CI does not include zero (significant difference)")
		else:
			lines.append("  Interpretation: CI includes zero (no significant difference)")
	else:
		lines.append("\n[WARNING] Between-group comparison could not be performed.")

	# Section 4: Conclusions
	lines.append("\n\n" + "=" * 80)
	lines.append("SECTION 4: INTERPRETATION AND CONCLUSIONS")
	lines.append("=" * 80)

	if between_results and 't_test' in between_results:
		t = between_results['t_test']
		ca_0, ca_1 = between_results['ca_groups']
		direction = 'higher' if between_results['mean_diff'] > 0 else 'lower'
		if t['significant']:
			lines.append(f"\nThe two groups show SIGNIFICANTLY DIFFERENT rates of weight change.")
			lines.append(f"The {ca_1}% CA group has a mean slope that is "
						 f"{abs(between_results['mean_diff']):.4f} {measure} per {time_unit} {direction}")
			lines.append(f"than the {ca_0}% CA group (p = {t['p_value']:.4f}).")
		else:
			lines.append(f"\nThe two groups show NO SIGNIFICANT DIFFERENCE in rates of weight change.")
			lines.append(f"Both groups change weight at approximately the same rate (p = {t['p_value']:.4f}).")

	lines.append("\n" + "=" * 80)
	lines.append("END OF REPORT")
	lines.append("=" * 80)

	return "\n".join(lines)


def generate_slope_mwu_report_rv(
	slopes_df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
) -> str:
	"""Generate a Mann-Whitney U test report table for slope comparison brackets."""
	SEP = "=" * 90
	lines = [
		SEP,
		"MANN-WHITNEY U TEST RESULTS  --  Slope Comparison Brackets",
		SEP,
		f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
		f"Measure    : {measure}",
		f"Time unit  : {time_unit}",
		"Correction : None (single pairwise comparison)",
		"",
		"Field definitions:",
		"  U        : Mann-Whitney U statistic",
		"  p(raw)   : Two-sided p-value (uncorrected)",
		"  p(adj)   : Adjusted p-value (= p(raw); no correction for single comparison)",
		"  r_rb     : Rank-biserial correlation (effect size r = 1 - 2U/(nA*nB))",
		"             |r| < 0.3 = small, 0.3-0.5 = medium, > 0.5 = large",
		"  HL_est   : Hodges-Lehmann location shift (median of all pairwise diffs A-B)",
		"  95% CI   : Bootstrap CI (n=2 000 resamples) on Hodges-Lehmann estimator",
		"",
		f"{'Comparison':<40} {'nA':>4} {'nB':>4} {'U':>10} {'p(raw)':>10} {'p(adj)':>10} {'r_rb':>8} {'HL_est':>10}  95% CI",
		"-" * 100,
	]

	ca_groups = sorted(slopes_df['CA (%)'].unique())
	for i, ca_a in enumerate(ca_groups):
		for ca_b in ca_groups[i + 1:]:
			s_a = slopes_df[slopes_df['CA (%)'] == ca_a]['Slope'].values
			s_b = slopes_df[slopes_df['CA (%)'] == ca_b]['Slope'].values
			nA, nB = len(s_a), len(s_b)
			u_stat, p_raw = stats.mannwhitneyu(s_a, s_b, alternative='two-sided')
			p_adj = p_raw  # no correction for single comparison
			r_rb = 1.0 - 2.0 * u_stat / (nA * nB)
			# Hodges-Lehmann estimator: median of all pairwise differences
			hl_est = float(np.median(np.subtract.outer(s_a, s_b).ravel()))
			# Bootstrap 95% CI on HL
			rng = np.random.default_rng(42)
			hl_boot = np.array([
				np.median(np.subtract.outer(
					rng.choice(s_a, size=nA, replace=True),
					rng.choice(s_b, size=nB, replace=True)
				).ravel())
				for _ in range(2000)
			])
			ci_lo = float(np.quantile(hl_boot, 0.025))
			ci_hi = float(np.quantile(hl_boot, 0.975))
			p_raw_str = "< 0.001" if p_raw < 0.001 else f"{p_raw:.4f}"
			p_adj_str = "< 0.001" if p_adj < 0.001 else f"{p_adj:.4f}"
			label = f"{int(ca_a)}% CA vs {int(ca_b)}% CA"
			lines.append(
				f"{label:<40} {nA:>4} {nB:>4} {u_stat:>10.2f} {p_raw_str:>10} {p_adj_str:>10}"
				f" {r_rb:>8.4f} {hl_est:>10.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]"
			)

	lines += [
		"",
		"Significance: * p<0.05   ** p<0.01   *** p<0.001",
		SEP,
	]
	return "\n".join(lines)


def perform_complete_slope_analysis_rv(
	df: pd.DataFrame,
	measure: str = "Total Change",
	time_unit: str = "Week",
	save_plot: bool = True,
	save_report: bool = True,
	output_dir: Optional[Path] = None,
) -> Dict:
	"""
	Complete pipeline: calculate slopes, compare groups, plot, save report.

	Parameters:
		df: Cleaned DataFrame with all required columns
		measure: Weight measure to analyze
		time_unit: Time unit ('Week' or 'Day')
		save_plot: Whether to save plots to file
		save_report: Whether to save the report to file
		output_dir: Directory for output files (None = script directory)

	Returns:
		Dictionary with all analysis results
	"""
	print("\n" + "=" * 80)
	print("COMPLETE SLOPE ANALYSIS PIPELINE - RV COHORT")
	print("=" * 80)
	print(f"\nMeasure:   {measure}")
	print(f"Time Unit: {time_unit}")

	slopes_df = calculate_animal_slopes_rv(df, measure=measure, time_unit=time_unit)
	within_results = compare_slopes_within_ca_groups(slopes_df)
	between_results = compare_slopes_between_ca_groups(slopes_df)

	report_text = generate_slope_analysis_report_rv(
		slopes_df, within_results, between_results,
		measure=measure, time_unit=time_unit,
	)

	print("\n" + "=" * 80)
	print("REPORT PREVIEW")
	print("=" * 80)
	print(report_text)

	if output_dir is None:
		output_dir = Path(__file__).parent
	else:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	if save_report:
		report_path = output_dir / f"RV_slope_analysis_report_{measure.replace(' ', '_')}_{timestamp}.txt"
		report_path.write_text(report_text, encoding='utf-8')
		print(f"\n[OK] Report saved to: {report_path}")

	mwu_report = generate_slope_mwu_report_rv(slopes_df, measure=measure, time_unit=time_unit)
	if save_report:
		mwu_path = output_dir / f"RV_slope_analysis_mwu_{measure.replace(' ', '_')}.txt"
		mwu_path.write_text(mwu_report, encoding='utf-8')
		print(f"[OK] MWU report saved to: {mwu_path}")

	if save_plot:
		plot_path = output_dir / f"RV_slope_analysis_{measure.replace(' ', '_')}_{timestamp}.png"
		plot_slopes_comparison_rv(
			slopes_df,
			measure=measure,
			time_unit=time_unit,
			save_path=plot_path,
			show=False,
		)

	return {
		'slopes_df': slopes_df,
		'within_results': within_results,
		'between_results': between_results,
		'report_text': report_text,
		'mwu_report': mwu_report,
		'measure': measure,
		'time_unit': time_unit,
	}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
	"""
	Load, clean, and analyze RV cohort weight data.
	Menu-driven interface for plots and slope analysis.
	"""
	csv_path = Path(__file__).parent / "RV_master_data.csv"

	print("=" * 80)
	print("RV COHORT WEIGHT ANALYSIS")
	print("=" * 80)

	df_raw = load_rv_data(csv_path)
	print("\nCleaning data...")
	df = clean_rv_dataframe(df_raw)
	print("Adding per-animal day numbering...")
	df = add_day_number_column(df)
	summarize_dataframe(df)

	# -- Global settings -------------------------------------------------------
	print("\n" + "=" * 80)
	print("GLOBAL SETTINGS")
	print("=" * 80)

	print("\nChoose time unit for slope analysis:")
	print("  [1] Days (most granular)")
	print("  [2] Weeks (averaged by week, Day 0 excluded)")
	time_unit_choice = input("Enter choice (1 or 2, default=2): ").strip()
	use_weeks = (time_unit_choice != '1')
	time_unit = 'Week' if use_weeks else 'Day'

	if use_weeks:
		df = add_week_column(df)
		week_counts = df.groupby('Week')['Day'].apply(lambda x: (x.min(), x.max()))
		print(f"\nWeek mapping:")
		for week, (d_min, d_max) in week_counts.items():
			if pd.notna(week):
				print(f"  Week {int(week)}: Days {int(d_min)}-{int(d_max)}")

	out_dir = Path(__file__).parent

	MENU = """
================================================================================
  RV ANALYSIS MENU
================================================================================
  [1]  Total Change by ID           (individual lines, colored by CA%)
  [2]  Daily Change by ID           (individual lines, colored by CA%)
  [3]  Total Change by CA%          (group mean \u00b1 SEM, colored by CA%)
  [4]  Daily Change by CA%          (group mean \u00b1 SEM, colored by CA%)
  [P]  All plots (1-4)
  [5]  Slope analysis: Total Change ~ {time} between CA% groups
  [6]  Slope analysis: Daily Change ~ {time} between CA% groups
  [A]  All analyses (plots + both slope analyses)
  [Q]  Quit
================================================================================"""

	while True:
		print(MENU.format(time=time_unit))
		choice = input("Select option: ").strip().upper()

		if choice == 'Q':
			print("\nExiting.")
			break

		save_plots = None
		show_plots = None

		def _ask_save_show():
			nonlocal save_plots, show_plots
			if save_plots is None:
				save_plots = input("Save plots to files? (y/n, default=y): ").strip().lower() != 'n'
				show_plots = input("Display plots interactively? (y/n, default=n): ").strip().lower() == 'y'

		def _plot_save(fig: plt.Figure, stem: str) -> None:
			if save_plots:
				png_path = out_dir / f"{stem}.png"
				fig.savefig(str(png_path), dpi=200, bbox_inches='tight')
				print(f"  [OK] Saved PNG: {png_path}")
				svg_path = out_dir / f"{stem}.svg"
				fig.savefig(str(svg_path), format='svg', bbox_inches='tight')
				print(f"  [OK] Saved SVG: {svg_path}")
			if not show_plots:
				plt.close(fig)

		# -- Plots -------------------------------------------------------------
		if choice in ('1', 'P', 'A'):
			_ask_save_show()
			fig = plot_total_change_by_id(df, show=show_plots,
										  title="Total Weight Change by Day per Animal")
			_plot_save(fig, "RV_total_change_by_id")

		if choice in ('2', 'P', 'A'):
			_ask_save_show()
			fig = plot_daily_change_by_id(df, show=show_plots,
										  title="Daily Weight Change by Day per Animal")
			_plot_save(fig, "RV_daily_change_by_id")

		if choice in ('3', 'P', 'A'):
			_ask_save_show()
			fig = plot_total_change_by_ca(df, show=show_plots,
										  title="Total Weight Change by CA% (Mean \u00b1 SEM)")
			_plot_save(fig, "RV_total_change_by_ca")

		if choice in ('4', 'P', 'A'):
			_ask_save_show()
			fig = plot_daily_change_by_ca(df, show=show_plots,
										  title="Daily Weight Change by CA% (Mean \u00b1 SEM)")
			_plot_save(fig, "RV_daily_change_by_ca")

		# -- Slope analyses ----------------------------------------------------
		if choice in ('5', 'A'):
			df_slope = df.copy()
			if use_weeks and "Week" not in df_slope.columns:
				df_slope = add_week_column(df_slope)
			perform_complete_slope_analysis_rv(
				df_slope,
				measure="Total Change",
				time_unit=time_unit,
				save_plot=True,
				save_report=True,
				output_dir=out_dir,
			)

		if choice in ('6', 'A'):
			df_slope = df.copy()
			if use_weeks and "Week" not in df_slope.columns:
				df_slope = add_week_column(df_slope)
			perform_complete_slope_analysis_rv(
				df_slope,
				measure="Daily Change",
				time_unit=time_unit,
				save_plot=True,
				save_report=True,
				output_dir=out_dir,
			)

		if choice == 'A':
			print("\n" + "=" * 80)
			print("RUN ALL COMPLETE")
			print(f"All outputs saved to: {out_dir.resolve()}")
			print("=" * 80)

		if choice not in ('1', '2', '3', '4', '5', '6', 'P', 'A', 'Q'):
			print(f"  [!] Unknown option '{choice}'. Enter a number 1-6, P, A, or Q.")

	return df


if __name__ == "__main__":
	df = main()
