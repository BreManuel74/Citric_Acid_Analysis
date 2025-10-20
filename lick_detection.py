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
	"""Return list of sensor columns like Sensor_1..Sensor_N in sorted numeric order."""
	sensor_cols = [c for c in df.columns if c.startswith("Sensor_")]
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
		grid_title = f"Selected Sensors (12) — {csv_path.name}"
		# Compute common y-limits across all 24 sensors so both grids share the same scale
		all_sensor_cols = [c for c in df.columns if c.startswith("Sensor_")]
		common_y_limits = compute_y_limits(df, all_sensor_cols)
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
		all_sensors = [c for c in df.columns if c.startswith("Sensor_")]
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

