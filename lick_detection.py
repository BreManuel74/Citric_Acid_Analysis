"""
Lick detection plotting utility

Reads a capacitive sensor CSV log (Arduino output), converts Arduino_Timestamp from ms to minutes,
and plots all sensor columns (Sensor_1..Sensor_24) over time (x-axis in seconds).
Also creates a separate plot for a chosen sensor (x-axis in seconds).

Usage (from the workspace root):
	python lick_detection.py --input "capacitive_log_2025-09-29_15-38-39.csv" --save "lick_sensors_plot.png"

If --input is omitted, a file picker window will open to select a CSV. The plot window will display interactively on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
"""Ramp analysis utilities: load master CSV, clean types, build per-ID series, and plot."""

plt.rcParams.update({
    "font.size": 11,          # base text
    "axes.titlesize": 13,     # ax.set_title / suptitle
    "axes.labelsize": 12,     # x/y labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,   # plt.suptitle
})

def load_capacitive_csv(csv_path: Path) -> pd.DataFrame:
	"""Load the capacitive CSV and return a cleaned DataFrame.

	- Ensures Arduino_Timestamp is numeric
	- Also computes Time_min and Time_sec (x-axis uses seconds)
	- Preserves the original CSV row order so each reading is aligned to its recorded timestamp
	"""
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	# Read CSV; let pandas infer dtypes, but make sure timestamp is numeric
	df = pd.read_csv(csv_path)

	if "Arduino_Timestamp" not in df.columns:
		raise ValueError("Expected column 'Arduino_Timestamp' not found in CSV.")

	# Coerce to numeric in case of stray non-numeric values
	df["Arduino_Timestamp"] = pd.to_numeric(df["Arduino_Timestamp"], errors="coerce")
	# Drop rows without a valid timestamp
	df = df.dropna(subset=["Arduino_Timestamp"]).copy()
	df["Arduino_Timestamp"] = df["Arduino_Timestamp"].astype("int64")

	# Compute minutes and seconds (seconds used for plotting)
	df["Time_min"] = df["Arduino_Timestamp"] / 60000.0
	df["Time_sec"] = df["Arduino_Timestamp"] / 1000.0

	return df


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
	"""Return list of sensor columns like Sensor_1..Sensor_N in sorted numeric order.
	
	Excludes deviation columns (those ending with '_deviation').
	"""
	sensor_cols = [c for c in df.columns if c.startswith("Sensor_") and not c.endswith("_deviation")]
	# Sort by numeric suffix if possible
	def key(c: str) -> int:
		try:
			return int(c.split("_")[-1])
		except Exception:
			return 0

	sensor_cols.sort(key=key)
	if not sensor_cols:
		raise ValueError("No sensor columns found (expected columns like 'Sensor_1', 'Sensor_2', ...)")
	return sensor_cols


def compute_sensor_modes(df: pd.DataFrame, sensor_cols: List[str]) -> pd.Series:
	"""Compute the mode (most frequent value) for each sensor column.
	
	Returns a Series indexed by sensor column names with their mode values.
	If multiple modes exist, returns the first one.
	"""
	modes = {}
	for col in sensor_cols:
		series = pd.to_numeric(df[col], errors="coerce")
		# Drop NaN values before computing mode
		series = series.dropna()
		if len(series) > 0:
			mode_result = series.mode()
			# mode() returns a Series; take the first value if multiple modes exist
			modes[col] = mode_result.iloc[0] if len(mode_result) > 0 else None
		else:
			modes[col] = None
	return pd.Series(modes)


def compute_mode_deviations(df: pd.DataFrame, sensor_cols: List[str], sensor_modes: pd.Series) -> pd.DataFrame:
	"""Compute absolute difference between each sensor's mode and its values at each time point.
	
	For each sensor column, creates a new column with suffix '_deviation' containing
	the absolute value of (mode - sensor_value) at each time point.
	
	Returns a copy of the dataframe with the new deviation columns added.
	"""
	df_with_deviations = df.copy()
	
	for col in sensor_cols:
		mode_val = sensor_modes[col]
		if mode_val is not None and col in df.columns:
			# Compute absolute deviation: |mode - value|
			sensor_series = pd.to_numeric(df[col], errors="coerce")
			deviation_col_name = f"{col}_deviation"
			df_with_deviations[deviation_col_name] = abs(mode_val - sensor_series)
		else:
			# If mode is None or column missing, set deviation to NaN
			deviation_col_name = f"{col}_deviation"
			df_with_deviations[deviation_col_name] = pd.NA
	
	return df_with_deviations


def plot_sensors_over_time(
	df: pd.DataFrame,
	sensor_cols: List[str],
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	return_fig: bool = False,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Plot all sensor columns vs Time_sec (s) and save to PNG.

	Returns the path to the saved PNG.
	"""
	if "Time_sec" not in df.columns:
		raise ValueError("'Time_sec' column not found. Did you run load_capacitive_csv()?")

	fig, ax = plt.subplots(figsize=(14, 8))

	# Use a colormap to differentiate many lines
	cmap = plt.get_cmap("tab20")
	for idx, col in enumerate(sensor_cols):
		color = cmap(idx % cmap.N)
		ax.plot(df["Time_sec"], df[col], label=col, linewidth=1.0, color=color)

	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Capacitive reading (a.u.)")
	if title:
		ax.set_title(title)

	# Put legend outside to the right
	ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
	fig.tight_layout()

	# Global style tweaks per request: start x at 0, remove top/right spines, inward ticks
	left, right = ax.get_xlim()
	ax.set_xlim(left=0, right=right)
	ax.margins(x=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(direction="in", which="both", length=5)

	# Save only if a save_path is provided
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (save_path, fig) if return_fig else save_path


def plot_single_sensor_over_time(
	df: pd.DataFrame,
	sensor_col: str,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	return_fig: bool = False,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Plot a single sensor column vs Time_sec (s) and save to PNG.

	Returns the path to the saved PNG.
	"""
	if "Time_sec" not in df.columns:
		raise ValueError("'Time_sec' column not found. Did you run load_capacitive_csv()?")
	if sensor_col not in df.columns:
		raise ValueError(f"Sensor column '{sensor_col}' not found in DataFrame.")

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(df["Time_sec"], df[sensor_col], label=sensor_col, color="#1f77b4", linewidth=1.5)

	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Capacitive reading (a.u.)")
	if title:
		ax.set_title(title)
	ax.legend(loc="best")
	fig.tight_layout()

	# Style adjustments per request for single-plot as well
	left, right = ax.get_xlim()
	ax.set_xlim(left=0, right=right)
	ax.margins(x=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(direction="in", which="both", length=5)

	# Save only if a save_path is provided
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (save_path, fig) if return_fig else save_path


def plot_selected_sensors_grid(
	df: pd.DataFrame,
	sensor_cols: List[str],
	*,
	nrows: int = 2,
	ncols: int = 6,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	y_limits: tuple[float, float] | None = None,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Plot selected sensors in a grid of subplots (default 2x6 for 12 sensors).

	Each subplot shows one sensor vs Time_sec (s), styled similarly to the single-sensor plot.
	If y_limits is provided, all subplots use the same y-axis range.
	"""
	if "Time_sec" not in df.columns:
		raise ValueError("'Time_sec' column not found. Did you run load_capacitive_csv()?")
	needed = nrows * ncols
	if len(sensor_cols) != needed:
		raise ValueError(f"Expected exactly {needed} sensors, got {len(sensor_cols)}")
	for col in sensor_cols:
		if col not in df.columns:
			raise ValueError(f"Sensor column '{col}' not found in DataFrame.")

	# Reasonable figure size for a 2x6 grid
	fig_w = max(12, ncols * 3)
	fig_h = max(6, nrows * 3)
	fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
	axes = axes.reshape(nrows, ncols)

	# Determine common y-limits (either provided or computed from the selected sensors)
	if y_limits is None:
		y_limits = compute_y_limits(df, sensor_cols)

	# Plot each sensor in its subplot
	for i, col in enumerate(sensor_cols):
		r = i // ncols
		c = i % ncols
		ax = axes[r][c]
		ax.plot(df["Time_sec"], df[col], color="#1f77b4", linewidth=1.2)
		ax.set_title(col, fontsize=10)
		# Only label leftmost y-axes and bottom x-axes to reduce clutter
		if c == 0:
			ax.set_ylabel("Capacitive (a.u.)")
		else:
			ax.set_ylabel("")
		if r == nrows - 1:
			ax.set_xlabel("Time (s)")
		else:
			ax.set_xlabel("")

		# Apply common y-limits
		if y_limits is not None:
			ax.set_ylim(*y_limits)

		# Style adjustments per request:
		# - Start x-axis at 0
		# - Remove top and right spines (boundaries)
		# - Tick marks inward and remove extra x-margins
		left, right = ax.get_xlim()
		ax.set_xlim(left=0, right=right)
		ax.margins(x=0)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		ax.tick_params(direction="in", which="both", length=5)

	if title:
		fig.suptitle(title)
	fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)

	# Save only if a save_path is provided
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (save_path, fig) if True else save_path


def plot_deviations_grid(
	df: pd.DataFrame,
	sensor_cols: List[str],
	*,
	nrows: int = 2,
	ncols: int = 6,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	y_limits: tuple[float, float] | None = None,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Plot sensor deviations in a grid of subplots (default 2x6 for 12 sensors).

	Each subplot shows one sensor's deviation (|mode - value|) vs Time_sec (s).
	If y_limits is provided, all subplots use the same y-axis range.
	"""
	if "Time_sec" not in df.columns:
		raise ValueError("'Time_sec' column not found. Did you run load_capacitive_csv()?")
	needed = nrows * ncols
	if len(sensor_cols) != needed:
		raise ValueError(f"Expected exactly {needed} sensors, got {len(sensor_cols)}")
	
	# Build list of deviation column names
	deviation_cols = [f"{col}_deviation" for col in sensor_cols]
	for col in deviation_cols:
		if col not in df.columns:
			raise ValueError(f"Deviation column '{col}' not found in DataFrame. Did you run compute_mode_deviations()?")

	# Reasonable figure size for a 2x6 grid
	fig_w = max(12, ncols * 3)
	fig_h = max(6, nrows * 3)
	fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
	axes = axes.reshape(nrows, ncols)

	# Determine common y-limits (either provided or computed from the deviation columns)
	if y_limits is None:
		y_limits = compute_y_limits(df, deviation_cols)

	# Plot each deviation in its subplot
	for i, col in enumerate(sensor_cols):
		dev_col = f"{col}_deviation"
		r = i // ncols
		c = i % ncols
		ax = axes[r][c]
		ax.plot(df["Time_sec"], df[dev_col], color="#ff7f0e", linewidth=1.2)
		ax.set_title(f"{col} (deviation)", fontsize=10)
		# Only label leftmost y-axes and bottom x-axes to reduce clutter
		if c == 0:
			ax.set_ylabel("Deviation (a.u.)")
		else:
			ax.set_ylabel("")
		if r == nrows - 1:
			ax.set_xlabel("Time (s)")
		else:
			ax.set_xlabel("")

		# Apply common y-limits
		if y_limits is not None:
			ax.set_ylim(*y_limits)

		# Style adjustments per request:
		# - Start x-axis at 0
		# - Remove top and right spines (boundaries)
		# - Tick marks inward and remove extra x-margins
		left, right = ax.get_xlim()
		ax.set_xlim(left=0, right=right)
		ax.margins(x=0)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		ax.tick_params(direction="in", which="both", length=5)

	if title:
		fig.suptitle(title)
	fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)

	# Save only if a save_path is provided
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (save_path, fig) if True else save_path


def compute_dynamic_thresholds(
	df: pd.DataFrame,
	sensor_cols: List[str],
	z_threshold: float = 4.0
) -> pd.Series:
	"""Compute dynamic threshold for each sensor using z-score normalization.
	
	For each sensor's deviation column, the threshold is set at:
		mean + (z_threshold * std)
	
	This identifies values that are more than z_threshold standard deviations
	above the mean deviation, indicating significant contact/lick events.
	
	Parameters:
		df: DataFrame with deviation columns
		sensor_cols: List of sensor column names (e.g., ['Sensor_1', 'Sensor_2'])
		z_threshold: Number of standard deviations above mean (default 4.0)
		
	Returns:
		pd.Series indexed by sensor name with threshold values
		
	Example:
		>>> thresholds = compute_dynamic_thresholds(df, ['Sensor_1', 'Sensor_2'], z_threshold=4.0)
		>>> print(thresholds)
		Sensor_1    45.2
		Sensor_2    38.7
		dtype: float64
	"""
	import numpy as np
	
	thresholds = {}
	
	for sensor_col in sensor_cols:
		dev_col = f"{sensor_col}_deviation"
		if dev_col not in df.columns:
			print(f"Warning: Deviation column '{dev_col}' not found. Skipping {sensor_col}.")
			thresholds[sensor_col] = None
			continue
		
		# Get deviation values and compute statistics
		deviations = pd.to_numeric(df[dev_col], errors="coerce")
		deviations = deviations.dropna()
		
		if len(deviations) == 0:
			print(f"Warning: No valid deviation data for {sensor_col}. Skipping.")
			thresholds[sensor_col] = None
			continue
		
		# Compute mean and standard deviation
		mean_dev = float(deviations.mean())
		std_dev = float(deviations.std())
		
		# Threshold = mean + (z_threshold * std)
		threshold = mean_dev + (z_threshold * std_dev)
		thresholds[sensor_col] = threshold
	
	return pd.Series(thresholds)


def detect_events_above_threshold(
	df: pd.DataFrame,
	sensor_cols: List[str],
	thresholds: pd.Series
) -> pd.DataFrame:
	"""Detect time points where deviation exceeds the dynamic threshold for each sensor.
	
	Creates boolean columns indicating when each sensor's deviation exceeds its threshold.
	
	Parameters:
		df: DataFrame with Time_sec and deviation columns
		sensor_cols: List of sensor column names
		thresholds: Series of threshold values per sensor (from compute_dynamic_thresholds)
		
	Returns:
		DataFrame with columns:
			- Time_sec
			- For each sensor: {sensor}_event (boolean indicating threshold exceedance)
			- For each sensor: {sensor}_deviation (original deviation value)
	"""
	import numpy as np
	
	result = pd.DataFrame()
	result['Time_sec'] = df['Time_sec']
	
	for sensor_col in sensor_cols:
		dev_col = f"{sensor_col}_deviation"
		event_col = f"{sensor_col}_event"
		
		if dev_col not in df.columns:
			result[event_col] = False
			result[dev_col] = np.nan
			continue
		
		threshold = thresholds.get(sensor_col)
		if threshold is None or not np.isfinite(threshold):
			result[event_col] = False
			result[dev_col] = df[dev_col]
			continue
		
		# Mark events where deviation exceeds threshold
		deviations = pd.to_numeric(df[dev_col], errors="coerce")
		result[dev_col] = deviations
		result[event_col] = deviations > threshold
	
	return result


def compute_inter_lick_intervals(
	events_df: pd.DataFrame,
	sensor_cols: List[str]
) -> dict:
	"""Compute inter-lick intervals (ILI) for each sensor.
	
	For each sensor, finds all time points where events occur (event == True),
	then computes the time difference between consecutive events.
	
	Parameters:
		events_df: DataFrame from detect_events_above_threshold with Time_sec and {sensor}_event columns
		sensor_cols: List of sensor column names
		
	Returns:
		Dictionary mapping sensor names to arrays of inter-lick intervals (in seconds).
		If a sensor has fewer than 2 events, returns an empty array.
		
	Example:
		>>> ili_dict = compute_inter_lick_intervals(events_df, ['Sensor_1', 'Sensor_2'])
		>>> print(f"Sensor_1 ILIs: {ili_dict['Sensor_1'][:5]}")  # First 5 intervals
		Sensor_1 ILIs: [0.245, 0.189, 0.312, 0.276, 0.198]
	"""
	import numpy as np
	
	ili_results = {}
	
	for sensor_col in sensor_cols:
		event_col = f"{sensor_col}_event"
		
		if event_col not in events_df.columns:
			ili_results[sensor_col] = np.array([])
			continue
		
		# Get timestamps where events occurred
		event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
		
		if len(event_times) < 2:
			# Need at least 2 events to compute an interval
			ili_results[sensor_col] = np.array([])
			continue
		
		# Compute differences between consecutive event times
		intervals = np.diff(event_times)
		ili_results[sensor_col] = intervals
	
	return ili_results


def compute_lick_bouts(
	events_df: pd.DataFrame,
	sensor_cols: List[str],
	ili_cutoff: float = 0.3
) -> dict:
	"""Compute lick bouts for each sensor using an inter-lick interval cutoff.
	
	A lick bout is a sequence of consecutive lick events where the time between
	events (ILI) is less than the cutoff. When ILI >= cutoff, a new bout begins.
	
	Parameters:
		events_df: DataFrame from detect_events_above_threshold with Time_sec and {sensor}_event columns
		sensor_cols: List of sensor column names
		ili_cutoff: Maximum ILI (in seconds) to consider events part of the same bout (default 0.3s)
		
	Returns:
		Dictionary mapping sensor names to bout information dictionaries containing:
			- 'bout_count': Total number of bouts
			- 'bout_sizes': Array of lick counts per bout
			- 'bout_durations': Array of bout durations (in seconds)
			- 'bout_start_times': Array of bout start times (in seconds)
			- 'bout_end_times': Array of bout end times (in seconds)
		
	Example:
		>>> bouts = compute_lick_bouts(events_df, ['Sensor_1'], ili_cutoff=0.3)
		>>> print(f"Sensor_1 had {bouts['Sensor_1']['bout_count']} bouts")
		>>> print(f"Average bout size: {bouts['Sensor_1']['bout_sizes'].mean():.1f} licks")
	"""

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
		
		# Get timestamps where events occurred
		event_times = events_df.loc[events_df[event_col] == True, 'Time_sec'].values
		
		if len(event_times) == 0:
			bout_results[sensor_col] = {
				'bout_count': 0,
				'bout_sizes': np.array([]),
				'bout_durations': np.array([]),
				'bout_start_times': np.array([]),
				'bout_end_times': np.array([])
			}
			continue
		
		# Single event = single bout of size 1
		if len(event_times) == 1:
			bout_results[sensor_col] = {
				'bout_count': 1,
				'bout_sizes': np.array([1]),
				'bout_durations': np.array([0.0]),
				'bout_start_times': event_times,
				'bout_end_times': event_times
			}
			continue
		
		# Compute ILIs
		intervals = np.diff(event_times)
		
		# Identify bout boundaries: wherever ILI >= cutoff, a new bout starts
		bout_breaks = intervals >= ili_cutoff
		
		# Build bouts by iterating through events
		bout_sizes = []
		bout_start_times = []
		bout_end_times = []
		
		current_bout_start = event_times[0]
		current_bout_size = 1
		
		for i in range(len(intervals)):
			if bout_breaks[i]:
				# End current bout
				bout_sizes.append(current_bout_size)
				bout_start_times.append(current_bout_start)
				bout_end_times.append(event_times[i])
				
				# Start new bout
				current_bout_start = event_times[i + 1]
				current_bout_size = 1
			else:
				# Continue current bout
				current_bout_size += 1
		
		# Don't forget the last bout
		bout_sizes.append(current_bout_size)
		bout_start_times.append(current_bout_start)
		bout_end_times.append(event_times[-1])
		
		# Convert to numpy arrays
		bout_sizes = np.array(bout_sizes)
		bout_start_times = np.array(bout_start_times)
		bout_end_times = np.array(bout_end_times)
		bout_durations = bout_end_times - bout_start_times
		
		bout_results[sensor_col] = {
			'bout_count': len(bout_sizes),
			'bout_sizes': bout_sizes,
			'bout_durations': bout_durations,
			'bout_start_times': bout_start_times,
			'bout_end_times': bout_end_times
		}
	
	return bout_results


def plot_bout_weight_correlation(
	bout_dict: dict,
	sensor_to_weight: dict,
	*,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	return_fig: bool = False,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Create a scatter plot showing correlation between lick bout counts and bottle weight changes.
	
	Parameters:
		bout_dict: Dictionary from compute_lick_bouts() with bout information per sensor
		sensor_to_weight: Dictionary mapping sensor names to bottle weight changes (grams)
		title: Optional plot title
		save_path: If provided, saves the figure to this path
		show: If True, calls plt.show() at the end
		return_fig: If True, returns (save_path, figure) tuple instead of just save_path
		
	Returns:
		Path to saved file, or tuple of (Path, Figure) if return_fig=True
	"""
	# Extract data for plotting
	bout_counts = []
	weight_changes = []
	sensor_labels = []
	
	for sensor in sorted(sensor_to_weight.keys()):
		weight = sensor_to_weight[sensor]
		bout_info = bout_dict.get(sensor, {})
		bout_count = bout_info.get('bout_count', 0)
		
		bout_counts.append(bout_count)
		weight_changes.append(weight)
		sensor_labels.append(sensor)
	
	# Convert to numpy arrays for correlation calculation
	bout_counts_arr = np.array(bout_counts)
	weight_changes_arr = np.array(weight_changes)
	
	# Calculate Pearson correlation coefficient
	if len(bout_counts_arr) > 1:
		correlation = np.corrcoef(bout_counts_arr, weight_changes_arr)[0, 1]
	else:
		correlation = np.nan
	
	# Create scatter plot
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Plot points
	ax.scatter(weight_changes, bout_counts, s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
	
	# Add sensor labels next to each point
	for i, sensor in enumerate(sensor_labels):
		# Extract sensor number for cleaner label
		sensor_num = sensor.split('_')[-1]
		ax.annotate(sensor_num, (weight_changes[i], bout_counts[i]), 
				   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
	
	# Add best-fit line if we have enough points
	if len(bout_counts_arr) > 1:
		z = np.polyfit(weight_changes_arr, bout_counts_arr, 1)
		p = np.poly1d(z)
		x_line = np.linspace(min(weight_changes_arr), max(weight_changes_arr), 100)
		ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2, label=f'Best fit (r = {correlation:.3f})')
		ax.legend(loc='best', frameon=True, fontsize=11)
	
	# Labels and title
	ax.set_xlabel('Bottle Weight Change (g)', fontsize=12, weight='bold')
	ax.set_ylabel('Lick Bout Count', fontsize=12, weight='bold')
	
	if title is None:
		if not np.isnan(correlation):
			title = f'Lick Bouts vs. Bottle Weight Change\nPearson r = {correlation:.3f}'
		else:
			title = 'Lick Bouts vs. Bottle Weight Change'
	ax.set_title(title, fontsize=14, weight='bold', pad=15)
	
	# Grid for easier reading
	ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
	
	# Style adjustments
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.tick_params(direction='in', which='both', length=5)
	
	fig.tight_layout()
	
	# Save if requested
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=200, bbox_inches='tight')
	
	if show:
		plt.show()
	else:
		plt.close(fig)
	
	return (save_path, fig) if return_fig else save_path


def compute_y_limits(df: pd.DataFrame, sensor_cols: List[str], pad_ratio: float = 0.05) -> tuple[float, float]:
	"""Compute common y-axis limits across given sensors with optional padding.

	Ignores NaNs. If min==max, expands symmetrically by 1.0.
	"""
	vals = []
	for col in sensor_cols:
		if col in df.columns:
			s = pd.to_numeric(df[col], errors="coerce")
			vals.append(s.values)
	if not vals:
		return (0.0, 1.0)
	import numpy as _np
	arr = _np.concatenate(vals)
	arr = arr[_np.isfinite(arr)]
	if arr.size == 0:
		return (0.0, 1.0)
	vmin = float(arr.min())
	vmax = float(arr.max())
	if vmin == vmax:
		return (vmin - 1.0, vmax + 1.0)
	span = vmax - vmin
	pad = span * max(0.0, pad_ratio)
	return (vmin - pad, vmax + pad)


def plot_timestamps_series(
	df: pd.DataFrame,
	col_name: str = "Arduino_Timestamp",
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
	return_fig: bool = False,
) -> Path | tuple[Optional[Path], plt.Figure]:
	"""Plot the running difference between consecutive timestamps vs sample index.

	- X-axis: sample index (1..N-1) corresponding to the delta positions
	- Y-axis: delta time in seconds
	"""
	# Prefer using Time_sec if available; otherwise compute from Arduino_Timestamp
	if "Time_sec" in df.columns:
		series = pd.to_numeric(df["Time_sec"], errors="coerce")
	else:
		if col_name not in df.columns:
			raise ValueError(f"Timestamp column '{col_name}' not found in DataFrame.")
		series = pd.to_numeric(df[col_name], errors="coerce") / 1000.0

	delta = series.diff()
	delta = delta.dropna()

	fig, ax = plt.subplots(figsize=(12, 4))
	ax.plot(delta.index, delta.values, color="#2ca02c", linewidth=1.0)

	ax.set_xlabel("Sample index")
	ax.set_ylabel("Delta time (s)")
	if title:
		ax.set_title(title)
	fig.tight_layout()

	# Save only if a save_path is provided
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return (save_path, fig) if return_fig else save_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot capacitive sensor readings over time (s)")
	parser.add_argument(
		"--input",
		"-i",
		type=str,
		default=None,
		help="Path to the capacitive log CSV. If omitted, a file picker will open.",
	)
	parser.add_argument(
		"--save",
		"-o",
		type=str,
		default=None,
		help="Path to save the PNG plot (default: <input> with _plot.png suffix)",
	)
	return parser.parse_args()


def select_csv_via_tkinter(initial_dir: Optional[Path] = None) -> Optional[Path]:
	"""Open a Tkinter file dialog to select a CSV file. Returns a Path or None if canceled/unavailable."""
	try:
		import tkinter as tk
		from tkinter import filedialog
		root = tk.Tk()
		root.withdraw()
		# Ensure the dialog appears in front on some platforms
		root.update()
		path = filedialog.askopenfilename(
			initialdir=str(initial_dir or Path.cwd()),
			title="Select capacitive CSV",
			filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
		)
		try:
			root.destroy()
		except Exception:
			pass
		return Path(path) if path else None
	except Exception:
		# Tkinter may be unavailable in some environments
		return None


def normalize_sensor_name(sensor_arg: str, df: pd.DataFrame) -> str:
	"""Normalize user-provided sensor identifier to a DataFrame column name.

	Examples:
	- "19" -> "Sensor_19"
	- "Sensor_19" -> "Sensor_19"
	Raises ValueError if the resolved column isn't present.
	"""
	arg = sensor_arg.strip()
	if arg.lower().startswith("sensor_"):
		name = arg
	else:
		# Expect a number
		try:
			num = int(arg)
			name = f"Sensor_{num}"
		except ValueError:
			raise ValueError(f"Invalid sensor identifier '{sensor_arg}'. Use a number (e.g., 19) or 'Sensor_19'.")

	if name not in df.columns:
		raise ValueError(f"Sensor column '{name}' not found in CSV. Available sensors: {[c for c in df.columns if c.startswith('Sensor_')]}")
	return name


def main() -> None:
	args = parse_args()

	# Resolve CSV path: use CLI input if provided, otherwise open a file picker, then fallback to console prompt
	csv_path: Optional[Path] = Path(args.input) if args.input else None
	if csv_path is None:
		csv_path = select_csv_via_tkinter(initial_dir=Path.cwd())
	if csv_path is None:
		while True:
			entry = input("Enter path to the capacitive CSV (or leave blank to cancel): ").strip()
			if not entry:
				print("No file selected. Exiting.")
				return
			candidate = Path(entry)
			if candidate.exists():
				csv_path = candidate
				break
			print(f"File not found: {candidate}. Please try again.\n")

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")
	df = load_capacitive_csv(csv_path)
	sensor_cols = get_sensor_columns(df)

	# Compute and display the mode for each sensor
	print("\n" + "=" * 60)
	print("MODE VALUES FOR EACH SENSOR")
	print("=" * 60)
	sensor_modes = compute_sensor_modes(df, sensor_cols)
	for sensor, mode_val in sensor_modes.items():
		print(f"{sensor:12s} : {mode_val}")
	print("=" * 60 + "\n")

	# Compute mode deviations (absolute difference from mode at each time point)
	print("Computing mode deviations for each sensor...")
	df = compute_mode_deviations(df, sensor_cols, sensor_modes)
	print(f"Added deviation columns: {', '.join([f'{col}_deviation' for col in sensor_cols[:3]])}...\n")

	# Do not save PNGs; SVG saving is handled later via prompts
	save_path = None

	title = f"Capacitive Sensors over Time — {csv_path.name}"

	_ = plot_sensors_over_time(
		df=df,
		sensor_cols=sensor_cols,
		title=title,
		save_path=save_path,
		show=True,
	)

	# Plot the timestamp column (first column) vs sample index
	try:
		ts_title = f"Timestamps over Samples — {csv_path.name}"
		# show only; saving will be handled later with user-provided name
		plot_timestamps_series(
			df=df,
			col_name="Arduino_Timestamp",
			title=ts_title,
			save_path=None,
			show=True,
		)
	except ValueError as e:
		print(str(e))

	# Single-sensor interactive prompt and plot
	single_sensor_col: Optional[str] = None
	single_title: Optional[str] = None
	try:
		entry = input("Enter a single sensor to plot (e.g., 19 or Sensor_19). Leave blank to skip: ").strip()
		if entry:
			resolved = normalize_sensor_name(entry, df)
			single_sensor_col = resolved
			single_title = f"{resolved} over Time — {csv_path.name}"
			plot_single_sensor_over_time(
				df=df,
				sensor_col=resolved,
				title=single_title,
				save_path=None,
				show=True,
			)
	except Exception as e:
		print(str(e))

	# Prompt for 12 sensors to plot in a 2x6 grid
	def _prompt_for_sensor_list(df: pd.DataFrame, count: int = 12) -> List[str]:
		while True:
			entry = input(
				f"Enter {count} sensors as a comma-separated list (e.g., 1, 3, 7, Sensor_10, ...): "
			).strip()
			# Split on commas and strip whitespace; ignore empty tokens
			raw_items = [tok.strip() for tok in entry.split(",") if tok.strip()]
			if len(raw_items) != count:
				print(f"Please enter exactly {count} items. You provided {len(raw_items)}. Try again.\n")
				continue
			try:
				names = [normalize_sensor_name(tok, df) for tok in raw_items]
				# Enforce uniqueness
				if len(set(names)) != count:
					print("Each sensor must be unique. Duplicates detected. Please try again.\n")
					continue
				return names
			except ValueError as ve:
				print(f"{ve}\nPlease correct the list and try again.\n")

	try:
		selected = _prompt_for_sensor_list(df, count=12)
		
		# Prompt for bottle weight changes (12 values matching the 12 selected sensors)
		def _prompt_for_bottle_weights(sensor_list: List[str]) -> List[float]:
			while True:
				print(f"\nEnter 12 bottle weight change values (in grams) corresponding to the sensors in order:")
				print(f"Sensors: {', '.join(sensor_list)}")
				entry = input("Enter 12 values separated by commas: ").strip()
				
				# Split on commas and strip whitespace
				raw_values = [tok.strip() for tok in entry.split(",") if tok.strip()]
				
				if len(raw_values) != 12:
					print(f"Please enter exactly 12 values. You provided {len(raw_values)}. Try again.\n")
					continue
				
				try:
					# Convert to float
					weights = [float(val) for val in raw_values]
					return weights
				except ValueError as ve:
					print(f"Invalid number format: {ve}\nPlease enter numeric values only.\n")
		
		bottle_weight_change = _prompt_for_bottle_weights(selected)
		
		# Create a mapping of sensor to bottle weight change
		sensor_to_weight = {sensor: weight for sensor, weight in zip(selected, bottle_weight_change)}
		
		# Display the sensor-weight mapping
		print("\n" + "=" * 60)
		print("SENSOR TO BOTTLE WEIGHT CHANGE MAPPING")
		print("=" * 60)
		for sensor, weight in sensor_to_weight.items():
			print(f"{sensor:12s} : {weight:8.2f} g")
		print("=" * 60 + "\n")
		
		grid_title = f"Selected Sensors (12) — {csv_path.name}"
		# Compute common y-limits across all 24 sensors so both grids share the same scale
		all_sensor_cols = get_sensor_columns(df)
		common_y_limits = compute_y_limits(df, all_sensor_cols)
		
		# Use fixed threshold of 10 for all selected sensors
		fixed_threshold = 10.0
		print("\n" + "=" * 60)
		print(f"FIXED THRESHOLD: {fixed_threshold}")
		print("=" * 60)
		thresholds = pd.Series({sensor: fixed_threshold for sensor in selected})
		for sensor in selected:
			print(f"{sensor:12s} : {fixed_threshold:8.2f}")
		print("=" * 60 + "\n")
		
		# Detect events above threshold
		print("Detecting events above threshold for selected sensors...")
		events_df = detect_events_above_threshold(df, selected, thresholds)
		
		# Count events per sensor
		print("\n" + "=" * 60)
		print("EVENT COUNTS (deviation > 10)")
		print("=" * 60)
		for sensor in selected:
			event_col = f"{sensor}_event"
			if event_col in events_df.columns:
				count = events_df[event_col].sum()
				print(f"{sensor:12s} : {count:6d} events")
		print("=" * 60 + "\n")
		
		# Compute inter-lick intervals
		print("Computing inter-lick intervals for selected sensors...")
		ili_dict = compute_inter_lick_intervals(events_df, selected)
		
		# Display ILI statistics
		print("\n" + "=" * 60)
		print("INTER-LICK INTERVAL STATISTICS (seconds)")
		print("=" * 60)
		import numpy as np
		for sensor in selected:
			intervals = ili_dict.get(sensor, np.array([]))
			if len(intervals) > 0:
				mean_ili = np.mean(intervals)
				std_ili = np.std(intervals)
				min_ili = np.min(intervals)
				max_ili = np.max(intervals)
				print(f"{sensor:12s} : n={len(intervals):4d}, mean={mean_ili:6.3f}s, std={std_ili:6.3f}s, min={min_ili:6.3f}s, max={max_ili:6.3f}s")
			else:
				print(f"{sensor:12s} : No intervals (< 2 events)")
		print("=" * 60 + "\n")
		
		# Compute lick bouts using 0.3s cutoff
		ili_cutoff = 0.3
		print(f"Computing lick bouts (ILI cutoff = {ili_cutoff}s)...")
		bout_dict = compute_lick_bouts(events_df, selected, ili_cutoff=ili_cutoff)
		
		# Display bout statistics
		print("\n" + "=" * 60)
		print(f"LICK BOUT STATISTICS (ILI cutoff = {ili_cutoff}s)")
		print("=" * 60)
		for sensor in selected:
			bout_info = bout_dict.get(sensor, {})
			bout_count = bout_info.get('bout_count', 0)
			bout_sizes = bout_info.get('bout_sizes', np.array([]))
			bout_durations = bout_info.get('bout_durations', np.array([]))
			
			if bout_count > 0:
				mean_size = np.mean(bout_sizes)
				mean_duration = np.mean(bout_durations)
				total_licks = np.sum(bout_sizes)
				print(f"{sensor:12s} : {bout_count:3d} bouts, {int(total_licks):4d} total licks, "
				      f"mean size={mean_size:5.1f} licks, mean duration={mean_duration:6.3f}s")
			else:
				print(f"{sensor:12s} : No bouts detected")
		print("=" * 60 + "\n")
		
		# Generate correlation plot of bout counts vs bottle weight changes
		print("Generating correlation plot: Lick Bouts vs. Bottle Weight Change...")
		corr_title = f"Lick Bouts vs. Bottle Weight Change — {csv_path.name}"
		plot_bout_weight_correlation(
			bout_dict=bout_dict,
			sensor_to_weight=sensor_to_weight,
			title=corr_title,
			save_path=None,
			show=True,
		)
		
		# Show selected grid first (no save yet)
		plot_selected_sensors_grid(
			df=df,
			sensor_cols=selected,
			title=grid_title,
			save_path=None,
			show=True,
			y_limits=common_y_limits,
		)

		# Compute and show the remaining sensors grid
		all_sensors = get_sensor_columns(df)
		remaining = [c for c in all_sensors if c not in set(selected)]
		if len(remaining) == 12:
			rem_title = f"Remaining Sensors (12) — {csv_path.name}"
			plot_selected_sensors_grid(
				df=df,
				sensor_cols=remaining,
				title=rem_title,
				save_path=None,
				show=True,
				y_limits=common_y_limits,
			)
		else:
			print(f"Skipping remaining-sensors grid: expected 12 remaining, found {len(remaining)}.")

		# Plot deviations for selected sensors
		print("\nPlotting deviations for selected sensors...")
		# Compute common y-limits for deviation plots across all 24 deviation columns
		all_deviation_cols = [f"{col}_deviation" for col in all_sensor_cols]
		common_dev_y_limits = compute_y_limits(df, all_deviation_cols)
		
		dev_sel_title = f"Selected Sensors Deviations (12) — {csv_path.name}"
		plot_deviations_grid(
			df=df,
			sensor_cols=selected,
			title=dev_sel_title,
			save_path=None,
			show=True,
			y_limits=common_dev_y_limits,
		)

		# Plot deviations for remaining sensors
		if len(remaining) == 12:
			dev_rem_title = f"Remaining Sensors Deviations (12) — {csv_path.name}"
			plot_deviations_grid(
				df=df,
				sensor_cols=remaining,
				title=dev_rem_title,
				save_path=None,
				show=True,
				y_limits=common_dev_y_limits,
			)

		# After viewing, prompt for SVG filenames to save each figure
		def _safe_svg(name: str) -> Path:
			base = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_.") or "plot"
			if not base.lower().endswith(".svg"):
				base += ".svg"
			return Path.cwd() / base

		print("\n(Optional) Save figures as SVGs. Leave blank to skip.")
		ov_name = input("SVG filename for all-sensors overview: ").strip()
		sel_name = input("SVG filename for selected 12 grid: ").strip()
		rem_name = input("SVG filename for remaining 12 grid: ").strip()
		dev_sel_name = input("SVG filename for selected 12 deviations: ").strip()
		dev_rem_name = input("SVG filename for remaining 12 deviations: ").strip()
		corr_name = input("SVG filename for bout-weight correlation plot: ").strip()
		# Only ask for single-sensor save name if a single sensor was plotted
		sing_name = ""
		if single_sensor_col is not None:
			sing_name = input("SVG filename for single-sensor plot: ").strip()
		ts_name = input("SVG filename for timestamp deltas: ").strip()

		# Re-generate figures in memory without showing, then save with provided names
		# All-sensors overview
		if ov_name:
			_, fig_ov = plot_sensors_over_time(
				df=df,
				sensor_cols=sensor_cols,
				title=title,
				save_path=None,
				show=False,
				return_fig=True,
			)
			fig_ov.savefig(str(_safe_svg(ov_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(ov_name)}")

		# Selected grid
		if sel_name:
			_, fig_sel = plot_selected_sensors_grid(
				df=df,
				sensor_cols=selected,
				title=grid_title,
				save_path=None,
				show=False,
				y_limits=common_y_limits,
			)
			fig_sel.savefig(str(_safe_svg(sel_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(sel_name)}")

		# Remaining grid
		if rem_name and len(remaining) == 12:
			_, fig_rem = plot_selected_sensors_grid(
				df=df,
				sensor_cols=remaining,
				title=rem_title,
				save_path=None,
				show=False,
				y_limits=common_y_limits,
			)
			fig_rem.savefig(str(_safe_svg(rem_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(rem_name)}")

		# Selected deviations grid
		if dev_sel_name:
			_, fig_dev_sel = plot_deviations_grid(
				df=df,
				sensor_cols=selected,
				title=dev_sel_title,
				save_path=None,
				show=False,
				y_limits=common_dev_y_limits,
			)
			fig_dev_sel.savefig(str(_safe_svg(dev_sel_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(dev_sel_name)}")

		# Remaining deviations grid
		if dev_rem_name and len(remaining) == 12:
			_, fig_dev_rem = plot_deviations_grid(
				df=df,
				sensor_cols=remaining,
				title=dev_rem_title,
				save_path=None,
				show=False,
				y_limits=common_dev_y_limits,
			)
			fig_dev_rem.savefig(str(_safe_svg(dev_rem_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(dev_rem_name)}")

		# Correlation plot
		if corr_name:
			_, fig_corr = plot_bout_weight_correlation(
				bout_dict=bout_dict,
				sensor_to_weight=sensor_to_weight,
				title=corr_title,
				save_path=None,
				show=False,
				return_fig=True,
			)
			fig_corr.savefig(str(_safe_svg(corr_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(corr_name)}")

		# Single-sensor figure
		if sing_name and single_sensor_col is not None:
			_, fig_sing = plot_single_sensor_over_time(
				df=df,
				sensor_col=single_sensor_col,
				title=single_title or single_sensor_col,
				save_path=None,
				show=False,
				return_fig=True,
			)
			fig_sing.savefig(str(_safe_svg(sing_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(sing_name)}")

		# Timestamp deltas figure
		if ts_name:
			_, fig_ts = plot_timestamps_series(
				df=df,
				col_name="Arduino_Timestamp",
				title=ts_title,
				save_path=None,
				show=False,
				return_fig=True,
			)
			fig_ts.savefig(str(_safe_svg(ts_name)), format="svg", bbox_inches="tight")
			print(f"Saved SVG: {_safe_svg(ts_name)}")

	except Exception as e:
		print(str(e))


if __name__ == "__main__":
	main()

