"""
Lick detection plotting utility

Reads a capacitive sensor CSV log (Arduino output), converts Arduino_Timestamp from ms to minutes,
and plots all sensor columns (Sensor_1..Sensor_24) over time (x-axis in seconds).
Also creates a separate plot for a chosen sensor (x-axis in seconds).

Usage (from the workspace root):
	python lick_detection.py --input "capacitive_log_2025-09-29_15-38-39.csv" --save "lick_sensors_plot.png"

If --save is omitted, a PNG will be saved next to the input file with suffix _plot.png.
The plot window will display interactively on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

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
) -> Path:
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
	ax.grid(True, alpha=0.3)

	# Put legend outside to the right
	ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
	fig.tight_layout()

	# Determine save path
	if save_path is None:
		save_path = Path("lick_sensors_plot.png")

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return save_path


def plot_single_sensor_over_time(
	df: pd.DataFrame,
	sensor_col: str,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
) -> Path:
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
	ax.grid(True, alpha=0.3)
	ax.legend(loc="best")
	fig.tight_layout()

	if save_path is None:
		save_path = Path(f"{sensor_col}_plot.png")

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return save_path

def plot_selected_sensors_grid(
	df: pd.DataFrame,
	sensor_cols: List[str],
	*,
	nrows: int = 2,
	ncols: int = 6,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
) -> Path:
	"""Plot selected sensors in a grid of subplots (default 2x6 for 12 sensors).

	Each subplot shows one sensor vs Time_sec (s), styled similarly to the single-sensor plot.
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

	# Plot each sensor in its subplot
	for i, col in enumerate(sensor_cols):
		r = i // ncols
		c = i % ncols
		ax = axes[r][c]
		ax.plot(df["Time_sec"], df[col], color="#1f77b4", linewidth=1.2)
		ax.set_title(col, fontsize=10)
		ax.grid(True, alpha=0.3)
		# Only label leftmost y-axes and bottom x-axes to reduce clutter
		if c == 0:
			ax.set_ylabel("Capacitive (a.u.)")
		else:
			ax.set_ylabel("")
		if r == nrows - 1:
			ax.set_xlabel("Time (s)")
		else:
			ax.set_xlabel("")

	if title:
		fig.suptitle(title)
	fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)

	if save_path is None:
		save_path = Path("selected_12_sensors_grid.png")
    
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return save_path


def plot_timestamps_series(
	df: pd.DataFrame,
	col_name: str = "Arduino_Timestamp",
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
) -> Path:
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
	ax.grid(True, alpha=0.3)
	fig.tight_layout()

	if save_path is None:
		save_path = Path("timestamps_plot.png")

	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(save_path, dpi=150)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return save_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot capacitive sensor readings over time (s)")
	parser.add_argument(
		"--input",
		"-i",
		type=str,
		default=str(Path("capacitive_log_2025-09-29_15-38-39.csv")),
		help="Path to the capacitive log CSV (default: capacitive_log_2025-09-29_15-38-39.csv)",
	)
	parser.add_argument(
		"--save",
		"-o",
		type=str,
		default=None,
		help="Path to save the PNG plot (default: <input> with _plot.png suffix)",
	)
	return parser.parse_args()


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

	csv_path = Path(args.input)
	df = load_capacitive_csv(csv_path)
	sensor_cols = get_sensor_columns(df)

	# Build default save path next to input file
	if args.save is None:
		save_path = csv_path.with_name(csv_path.stem + "_plot.png")
	else:
		save_path = Path(args.save)

	title = f"Capacitive Sensors over Time — {csv_path.name}"

	out_path = plot_sensors_over_time(
		df=df,
		sensor_cols=sensor_cols,
		title=title,
		save_path=save_path,
		show=True,
	)

	print(f"Saved plot to: {out_path}")

	# Plot the timestamp column (first column) vs sample index
	try:
		ts_title = f"Timestamps over Samples — {csv_path.name}"
		ts_path = csv_path.with_name(csv_path.stem + "_timestamps_plot.png")
		out_ts = plot_timestamps_series(
			df=df,
			col_name="Arduino_Timestamp",
			title=ts_title,
			save_path=ts_path,
			show=True,
		)
		print(f"Saved plot to: {out_ts}")
	except ValueError as e:
		print(str(e))

	# Removed single-sensor interactive prompt and plot per user request

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
		grid_path = csv_path.with_name(csv_path.stem + "_selected12_grid.png")
		out_grid = plot_selected_sensors_grid(
			df=df,
			sensor_cols=selected,
			title=grid_title,
			save_path=grid_path,
			show=True,
		)
		print(f"Saved plot to: {out_grid}")

		# Compute and plot the remaining sensors (all sensors minus selected)
		all_sensors = [c for c in df.columns if c.startswith("Sensor_")]
		remaining = [c for c in all_sensors if c not in set(selected)]
		if len(remaining) == 12:
			rem_title = f"Remaining Sensors (12) — {csv_path.name}"
			rem_path = csv_path.with_name(csv_path.stem + "_remaining12_grid.png")
			out_grid_rem = plot_selected_sensors_grid(
				df=df,
				sensor_cols=remaining,
				title=rem_title,
				save_path=rem_path,
				show=True,
			)
			print(f"Saved plot to: {out_grid_rem}")
		else:
			print(f"Skipping remaining-sensors grid: expected 12 remaining, found {len(remaining)}.")
	except Exception as e:
		print(str(e))


if __name__ == "__main__":
	main()

