"""
figure_maker.py — Publication Figure Generation Script

Standalone script for producing all paper figures and extended-data panels.
All data paths are hardcoded in the PATH CONFIGURATION section below so that
any figure can be reproduced by running this file directly.

This file is intentionally self-contained: all analysis and plotting functions
used for each figure are copied directly into the relevant figure section below.
Readers can view the exact code that produced each figure without consulting
any other file.

NOTE: There will be minor formatting differences between the figures produced by this script
and the final published figures. The final figures were polished in Adobe Illustrator for 
font consistency, line thickness, evenly spaced x and y axis limits, panel alignment, etc.

NOTE: This script requires R to be installed and available in the system PATH, including all
neecessary R packages (e.g. nonparLD). 

Usage
-----
  python figure_maker.py

  Figures are saved as SVG to the OUTPUT directories
  configured below.  Set SHOW_PLOTS = True to also display them interactively.

Author: Brenna Manuel
"""

from __future__ import annotations

# =============================================================================
# STANDARD LIBRARY & THIRD-PARTY IMPORTS
# =============================================================================

import math
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from scipy import stats

# ── Optional packages ─────────────────────────────────────────────────────────

try:
    from scipy.signal import find_peaks
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False


# =============================================================================
# GLOBAL DISPLAY TOGGLE
# =============================================================================

#: Set to True to call plt.show() after every figure during interactive use.
#: Set to False for batch / headless runs (figures saved to disk only).
SHOW_PLOTS: bool = False


# =============================================================================
# PATH CONFIGURATION
# Hardcode every input file and output directory here.  Comment out or set to
# None any path that does not apply to the current analysis run.
# =============================================================================

# ── Root workspace ─────────────────────────────────────────────────────────────
_ROOT = Path(
    r"C:\Users\brema\OneDrive_The Pennsylvania State University"
    r"\OneDrive - The Pennsylvania State University"
    r"\Desktop\CA_paper_analysis"
)
# will eventually need to make this robust to work on other machines, but for now this is fine.

# ── 0 % CA non-ramp cohort ────────────────────────────────────────────────────
DIR_0PCT          = _ROOT / "0%_files"
MASTER_0PCT       = DIR_0PCT / "master_data_0%.csv"
LICK_MASTER_0PCT  = DIR_0PCT / "0%_lick_data.csv"
CAP_LOGS_0PCT: List[Path] = sorted(DIR_0PCT.glob("capacitive_log_*.csv"))

# ── 2 % CA non-ramp cohort (6 animals) ────────────────────────────────────────
DIR_2PCT          = _ROOT / "2%_6_animals_files"
MASTER_2PCT       = DIR_2PCT / "master_data_2%_6_animals.csv"
LICK_MASTER_2PCT  = DIR_2PCT / "2%_lick_data.csv"
CAP_LOGS_2PCT: List[Path] = sorted(DIR_2PCT.glob("capacitive_log_*.csv"))

# ── 5-week CA (slow) ramp cohort — 1-day-normalised version ─────────────────────────────────
DIR_RAMP          = _ROOT / "5_week_ramp_1_day_normalized"
MASTER_RAMP       = DIR_RAMP / "master_data_SR.csv"
LICK_MASTER_RAMP  = DIR_RAMP / "5_week_lick_data.csv"
CAP_LOGS_RAMP: List[Path] = sorted(DIR_RAMP.glob("capacitive_log_*.csv"))

# ── 2 % CA full cohort (12 animals, clean_up folder) ─────────────────────────
DIR_2PCT_FULL          = _ROOT / "clean_up" / "2%_files"
MASTER_2PCT_FULL       = DIR_2PCT_FULL / "master_data_2%.csv"
LICK_MASTER_2PCT_FULL  = DIR_2PCT_FULL / "2%_lick_data.csv"
CAP_LOGS_2PCT_FULL: List[Path] = sorted(DIR_2PCT_FULL.glob("capacitive_log_*.csv"))

# ── 2-week CA (fast) ramp cohort ─────────────────────────────────────────────────────
DIR_2WK           = _ROOT / "2_week_files"
MASTER_2WK        = DIR_2WK / "master_data_2wk.csv"
# (no lick master or capacitive logs for 2-wk ramp)

# ── CAH cohort ────────────────────────────────────────────────────────────────
DIR_CAH           = _ROOT / "CAH_cohort"
MASTER_CAH        = DIR_CAH / "master_data_CAH.csv"

# ── RV cohort ─────────────────────────────────────────────────────────────────
DIR_RV            = _ROOT / "RV_cohort"
MASTER_RV         = DIR_RV / "RV_master_data.csv"

# ── Pilot data ────────────────────────────────────────────────────────────────
DIR_PILOT         = _ROOT / "pilot_data"
PILOT_CSV_1       = DIR_PILOT / "pilot_cohort_1.csv"

# ── Output directories (one per paper figure / extended-data panel) ───────────
OUT_FIG2          = _ROOT / "Figure_2"
OUT_FIG3          = _ROOT / "Figure_3"
OUT_FIG4          = _ROOT / "Figure_4"
OUT_FIG5          = _ROOT / "Figure_5"
OUT_FIG6          = _ROOT / "Figure_6"
OUT_FIG7          = _ROOT / "Figure_7"
OUT_FIG8          = _ROOT / "Figure_8"
OUT_FIG9          = _ROOT / "Figure_9"
OUT_EXT1          = _ROOT / "Ex_data_1"
OUT_EXT2_3        = _ROOT / "Ext_data_2-3"
OUT_EXT4_5        = _ROOT / "Ext_data_4-5"

# ── Convenience: create all output directories if they don't exist ────────────
for _out in [OUT_FIG2, OUT_FIG3, OUT_FIG4, OUT_FIG5,
             OUT_FIG6, OUT_FIG7, OUT_FIG8, OUT_FIG9,
             OUT_EXT1, OUT_EXT2_3, OUT_EXT4_5]:
    _out.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CANONICAL COHORT COLORS
# These match across_cohort.py / across_cohort_lick.py exactly so that every
# figure in the paper uses the same palette.
# =============================================================================

COLOR_0PCT  = "#1f77b4"   # 0 % CA  — blue
COLOR_2PCT  = "#f79520"   # 2 % CA  — orange
COLOR_RAMP  = "#2da048"   # slow ramp — green
COLOR_4PCT  = "#424143"   # 4 % CA (pilot) — dark grey
COLOR_OTHER = "#7f3f98"   # fast ramp - purple

# Sex marker styles (cohort color is applied separately)
#   Males:   filled square  "s"
#   Females: filled circle  "o"
SEX_MARKER: Dict[str, str] = {"M": "s", "F": "o", "Unknown": "^"}


def cohort_color(label: str) -> str:
    """Return the canonical hex color for a cohort label string."""
    lo = str(label).lower()
    if "0%" in lo:
        return COLOR_0PCT
    if "4%" in lo:
        return COLOR_4PCT
    if "2%" in lo:
        return COLOR_2PCT
    # Fast / 2-week ramp must be checked before the generic "ramp" branch
    if "2-week" in lo or "2wk" in lo or "fast" in lo:
        return COLOR_OTHER
    if "ramp" in lo:
        return COLOR_RAMP
    return COLOR_OTHER


# =============================================================================
# MATPLOTLIB STYLE — publication-quality defaults
# All rcParams are set once here; individual plot functions may override locally.
# =============================================================================

plt.rcParams["font.family"]       = "sans-serif"
plt.rcParams["font.sans-serif"]   = ["Arial"]
plt.rcParams["svg.fonttype"]      = "none"   # keep text as text in SVG

plt.rcParams.update({
    # Font sizes
    "font.size":          8,
    "axes.titlesize":     10,
    "axes.labelsize":     8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "figure.titlesize":   10,
    # Line / marker defaults
    "lines.linewidth":    0.9,
    "lines.markersize":   3,
    # Default figure size — individual figures override as needed
    "figure.figsize":     (3.5, 2.5),
    # Axes margins
    "axes.xmargin":       0.0,
    "axes.ymargin":       0.05,
    # Save quality
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


# =============================================================================
# COMMON PLOT HELPER FUNCTIONS
# Ported from behavioral_analysis.py / across_cohort.py so that figure_maker
# does not depend on those modules for style utilities.
# =============================================================================

def apply_common_plot_style(
    ax: plt.Axes,
    *,
    start_x_at_zero: bool = False,
    remove_top_right: bool = True,
    remove_x_margins: bool = True,
    remove_y_margins: bool = False,
    ticks_in: bool = True,
    draw_zero_dotted_line: bool = True,
) -> plt.Axes:
    """Apply the canonical publication-style spine / tick settings to *ax*.

    Parameters
    ----------
    ax                  : target Axes
    start_x_at_zero     : force x-axis left limit to 0
    remove_top_right    : hide top and right spines
    remove_x_margins    : eliminate default x-axis padding
    remove_y_margins    : eliminate default y-axis padding
    ticks_in            : point tick marks inward
    draw_zero_dotted_line : draw a dotted horizontal line at y = 0
    """
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if ticks_in:
        ax.tick_params(direction="in", which="both", length=4)

    if remove_x_margins:
        ax.margins(x=0)

    if remove_y_margins:
        ax.margins(y=0)

    if start_x_at_zero:
        left, right = ax.get_xlim()
        ax.set_xlim(left=0, right=right)

    if draw_zero_dotted_line:
        ax.axhline(0, color="0.35", linestyle=":", linewidth=0.8, zorder=0)

    return ax


def _auto_integer_step(
    vmin: float,
    vmax: float,
    target_ticks: int = 7,
    allow_sub5: bool = False,
) -> int:
    """Choose a 'nice' integer step so ~*target_ticks* ticks span [vmin, vmax]."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return 1
    range_int = int(abs(math.ceil(vmax) - math.floor(vmin)))
    if range_int <= 0:
        return 1
    approx   = max(1.0, range_int / max(1, target_ticks))
    pow10    = 10 ** int(math.floor(math.log10(approx)))
    mults    = (1, 2, 2.5, 3, 4, 5) if allow_sub5 else (1, 2, 5)
    for m in mults:
        if m * pow10 >= approx:
            return int(m * pow10)
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
    """Set integer ticks and limits on *axis* ('x' or 'y') of *ax*."""
    step = int(max(1, step))
    base_start    = int(math.floor(data_min / step) * step)
    base_end_tick = int(math.ceil(data_max  / step) * step)
    tick_start    = base_start    - left_pad_steps  * step
    tick_end      = base_end_tick + right_pad_steps * step
    start = tick_start
    if clamp_min is not None and start < clamp_min:
        start = clamp_min
    end   = int(data_max) + right_pad_steps * step
    if end <= start:
        end = start + step
    all_ticks = list(range(tick_start, tick_end + 1, step))
    ticks     = [t for t in all_ticks if start <= t <= end]
    if axis == "x":
        ax.set_xlim(start, end)
        ax.set_xticks(ticks)
    elif axis == "y":
        ax.set_ylim(start, end)
        ax.set_yticks(ticks)


def save_fig(
    fig: plt.Figure,
    path: Path,
) -> None:
    """Save *fig* to *path* as SVG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = path.with_suffix(".svg")
    fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
    print(f"  Saved → {svg_path.relative_to(_ROOT)}")
    if not SHOW_PLOTS:
        plt.close(fig)


# =============================================================================
# DATA LOADING HELPERS
# Minimal CSV readers — analysis-specific loaders are defined inside each
# figure section alongside the functions that consume them.
# =============================================================================

def _load_master_csv(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read a master_data_*.csv and return a cleaned DataFrame.

    Converts Date to datetime, coerces numeric columns, and normalises
    yes/no behavioral columns to bool.
    """
    df = pd.read_csv(path, encoding=encoding)
    df.columns = df.columns.str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Strip trailing "%" before numeric conversion (handles "3%" → 3.0)
    if "CA (%)" in df.columns:
        df["CA (%)"] = (
            df["CA (%)"].astype(str)
            .str.replace(r"\s*%\s*$", "", regex=True)
            .str.strip()
        )

    for col in ("Weight", "Daily Change", "Total Change",
                "Bottle Weight Change", "CA (%)"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if str(col).endswith("?"):
            df[col] = df[col].map(
                {"yes": True, "no": False,
                 "Yes": True, "No": False,
                 "YES": True, "NO": False,
                 True: True, False: False}
            )

    return df


def _load_lick_master_csv(path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read a lick master CSV (e.g. 0%_lick_data.csv).

    Lowercases column names and coerces numeric fields.
    """
    df = pd.read_csv(path, encoding=encoding)
    df.columns = df.columns.str.strip().str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ("total_licks", "total_bouts", "avg_ili",
                "avg_bout_duration", "licks_per_bout", "ca%"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# SHARED DATA PROCESSING HELPERS  (Figure 2 and beyond)
# Mode-aware: pass mode='ramp' or mode='nonramp' to each function.
# Input DataFrames should be pre-loaded via _load_master_csv().
# =============================================================================

def _add_day_col(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Add per-ID integer Day column.

    mode='ramp'
        Day 1 = first recorded date per animal (no separate baseline day).
        Total Change is 0 on Day 1 by construction.
    mode='nonramp'
        Day 0 = first recorded date (baseline, Total Change = 0).
        Day 1+ = treatment days.  Plots clamp the x-axis to Day 1.
    """
    if "ID" not in df.columns or "Date" not in df.columns:
        return df.copy()
    df = df.copy()
    df = df.sort_values(["ID", "Date"]).reset_index(drop=True)
    first_dates = df.groupby("ID")["Date"].transform("min")
    delta = (df["Date"] - first_dates).dt.days
    df["Day"] = (delta + 1).astype(int) if mode == "ramp" else delta.astype(int)
    return df


def _add_week_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add Week (1-indexed) from Day column (Week 1 = Days 1–7, etc.)."""
    if "Day" not in df.columns:
        return df.copy()
    df = df.copy()
    df["Week"] = ((df["Day"] - 1) // 7) + 1
    return df


def _get_sex_map_from_df(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return {str(animal_id): 'M' | 'F' | None} from a master DataFrame."""
    if "ID" not in df.columns or "Sex" not in df.columns:
        return {}
    result: Dict[str, Optional[str]] = {}
    for mid, g in df.groupby("ID"):
        sex_val = None
        for v in g["Sex"].dropna().astype(str).str.strip().str.upper():
            if v.startswith("M"):
                sex_val = "M"; break
            elif v.startswith("F"):
                sex_val = "F"; break
        result[str(mid)] = sex_val
    return result


def _build_total_change_series(
    df: pd.DataFrame,
    mode: str,
) -> Dict[str, pd.Series]:
    """Per-animal Total Change series indexed by integer Day.

    Parameters
    ----------
    df   : DataFrame from _load_master_csv() (Date parsed, Total Change numeric).
    mode : 'ramp' or 'nonramp' — controls Day numbering convention.

    Returns
    -------
    dict  str(animal_id) → pd.Series(index=Day[int], values=Total Change[float]).
    Duplicate Day values for the same animal are resolved by keeping the last.
    """
    cdf = _add_day_col(df, mode)
    cdf = cdf.dropna(subset=["Total Change", "Day"])
    result: Dict[str, pd.Series] = {}
    for gid, g in cdf.groupby("ID", dropna=True):
        ser = g.set_index("Day")["Total Change"].sort_index()
        ser = ser.groupby(level=0).last()
        ser.name = str(gid)
        result[str(gid)] = ser
    return result


def _build_ca_series_by_day(df: pd.DataFrame) -> pd.Series:
    """Ramp mode: Series(Day → CA%) for block-boundary annotations.

    Template rows (CA% filled but no measurements recorded) are excluded
    so that unreached concentrations do not appear as phantom blocks.
    """
    if "CA (%)" not in df.columns:
        return pd.Series(dtype=float)
    cdf = _add_day_col(df, "ramp")
    meas_cols = [c for c in ("Daily Change", "Total Change", "Weight") if c in cdf.columns]
    if meas_cols:
        cdf = cdf.dropna(subset=meas_cols, how="all")
    cdf = cdf.dropna(subset=["Day", "CA (%)"])
    if cdf.empty:
        return pd.Series(dtype=float)
    return (
        cdf.groupby("Day")["CA (%)"]
        .agg(lambda x: float(x.mode().iloc[0]) if not x.empty else np.nan)
        .sort_index()
        .dropna()
    )


def _build_week_series_by_day(df: pd.DataFrame) -> pd.Series:
    """Nonramp mode: Series(Day → Week) for block-boundary annotations."""
    cdf = _add_day_col(df, "nonramp")
    cdf = _add_week_col(cdf)
    cdf = cdf.dropna(subset=["Day", "Week"])
    cdf = cdf[cdf["Day"] >= 1]
    if cdf.empty:
        return pd.Series(dtype=float)
    return (
        cdf.groupby("Day")["Week"]
        .agg(lambda x: float(x.mode().iloc[0]) if not x.empty else np.nan)
        .sort_index()
        .dropna()
    )


def _draw_block_boundaries(
    ax: plt.Axes,
    block_series: pd.Series,
    *,
    linestyle: str = "--",
    color: str = "0.25",
    linewidth: float = 0.9,
    alpha: float = 0.8,
) -> None:
    """Vertical dashed line at the last day of each contiguous block, except the last block."""
    if block_series is None or block_series.empty:
        return
    s = block_series.dropna()
    if s.empty:
        return
    seg_ids  = (s != s.shift(1)).cumsum()
    end_days = [int(seg.index[-1]) for _, seg in s.groupby(seg_ids)]
    for d in end_days[:-1]:
        ax.axvline(x=d, linestyle=linestyle, color=color,
                   linewidth=linewidth, alpha=alpha, zorder=1)


def _draw_block_labels(
    ax: plt.Axes,
    block_series: pd.Series,
    fmt_fn,
    *,
    y_frac: float = 0.97,
    fontsize: float = 7,
    color: str = "0.35",
) -> None:
    """Float a text label centred over each contiguous block.

    Parameters
    ----------
    fmt_fn : callable  e.g. ``lambda v: f"{int(v)}%"`` for ramp CA% labels,
                            ``lambda v: f"W{int(v)}"`` for nonramp week labels.
    """
    if block_series is None or block_series.empty:
        return
    s = block_series.dropna()
    if s.empty:
        return
    seg_ids = (s != s.shift(1)).cumsum()
    trans   = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for _, seg in s.groupby(seg_ids):
        x_ctr = (float(seg.index[0]) + float(seg.index[-1])) / 2.0
        ax.text(
            x_ctr, y_frac, fmt_fn(seg.iloc[0]),
            transform=trans, ha="center", va="top",
            fontsize=fontsize, color=color, zorder=5,
        )

# =============================================================================
# FIGURE 2 — Total Weight Change Over Time, Per Cohort
#
# Five separate figure panels (no subplot structure), one per cohort:
#   2a — 0 % CA non-ramp                           (nonramp mode)
#   2b — 2 % CA full cohort, 12 animals             (nonramp mode)
#   2c — 5-week slow CA ramp, 1-day-normalised      (ramp mode)
#   2d — 2-week fast CA ramp                        (ramp mode)
#   2e — 4 % CA pilot, first 3 days                 (special: pilot CSV)
#   2f - slope comparison
#
# Source column : "Total Change" (% body-weight change from baseline)
# x-axis        : Day (integer, per-animal from first recording date)
#
# Day alignment:
#   nonramp — Day 0 = baseline recording (excluded from x-axis by clamp_min=1).
#             Week-block boundaries (dashed) + "W1"…"W5" labels from Week column.
#   ramp    — Day 1 = first recording (Total Change = 0 by construction).
#             CA%-block boundaries (dashed) + "0%"…"4%" labels from CA (%) column.
#   4% pilot— Day 0 = baseline per animal; only Days 1–3 plotted; no blocks.
# =============================================================================

def _fig2_plot_cohort(
    df: pd.DataFrame,
    mode: str,
    cohort_color: str,
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Core Figure 2 panel: per-animal Total Weight Change over Day.

    Parameters
    ----------
    df           : master_data_*.csv loaded via _load_master_csv()
    mode         : 'ramp' or 'nonramp'  (controls Day numbering and block type)
    cohort_color : hex color for all animal lines; sex encoded by marker only
    title        : figure title string
    save_path    : stem path (no extension); SVG saved via save_fig()

    Animal lines
    ------------
    All animals share *cohort_color*; sex is encoded by marker shape:
      Male   → filled square  "s"
      Female → filled circle  "o"

    Axes
    ----
    x : integer Day; starts at Day 1 (clamp_min=1); auto-ticked.
    y : integer Total Change (%); auto-ranged, one extra step of padding.
    Block boundaries and floating labels added after axis limits are set.
    """
    series_by_id = _build_total_change_series(df, mode)
    sex_map      = _get_sex_map_from_df(df)

    if mode == "ramp":
        block_series = _build_ca_series_by_day(df)
        block_fmt    = lambda v: f"{int(v)}%"
    else:
        block_series = _build_week_series_by_day(df)
        block_fmt    = lambda v: f"W{int(v)}"

    fig, ax = plt.subplots()

    for mid, s in series_by_id.items():
        sex    = sex_map.get(str(mid))
        marker = SEX_MARKER.get(sex or "Unknown", "^")
        ax.plot(
            s.index, s.values,
            color=cohort_color,
            marker=marker,
            markersize=plt.rcParams["lines.markersize"],
            linewidth=plt.rcParams["lines.linewidth"],
            alpha=0.85,
            label=str(mid),
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Weight Change (%)")
    ax.set_title(title)
    ax.grid(False)

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
        draw_zero_dotted_line=True,
    )

    # x-axis: integer ticks, left edge clamped to Day 1
    all_x  = np.concatenate([np.asarray(s.index, dtype=float)
                              for s in series_by_id.values() if len(s) > 0])
    x_lo   = int(np.nanmin(all_x)) if all_x.size else 1
    x_hi   = int(np.nanmax(all_x)) if all_x.size else 35
    x_step = _auto_integer_step(x_lo, x_hi, target_ticks=10, allow_sub5=True)
    _apply_integer_axis(ax, axis="x", data_min=x_lo, data_max=x_hi,
                        step=x_step, clamp_min=1,
                        left_pad_steps=0, right_pad_steps=0)

    # y-axis: auto-ranged, integer ticks
    all_y  = np.concatenate([s.values for s in series_by_id.values() if len(s) > 0])
    y_lo   = float(np.nanmin(all_y)) if all_y.size else -20.0
    y_hi   = float(np.nanmax(all_y)) if all_y.size else  5.0
    y_step = _auto_integer_step(y_lo, y_hi, target_ticks=7)
    _apply_integer_axis(ax, axis="y", data_min=y_lo, data_max=y_hi,
                        step=y_step, left_pad_steps=0, right_pad_steps=1)

    # Block annotations (must come after axis limits are set)
    _draw_block_boundaries(ax, block_series)
    _draw_block_labels(ax, block_series, block_fmt)

    # Sex legend
    leg = [
        ax.plot([], [], color=cohort_color, marker="s",
                linestyle="-", label="Male")[0],
        ax.plot([], [], color=cohort_color, marker="o",
                linestyle="-", label="Female")[0],
    ]
    ax.legend(handles=leg, loc="best", frameon=False)

    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig2_plot_pilot_4pct(
    csv_path: Path,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 2d: Total Weight Change for 4% CA pilot animals, Days 1–3.

    Reads pilot_cohort_1.csv, filters to rows where Condition starts with
    '4%', and assigns Day numbers using the nonramp convention:
      Day 0 = first recorded date per animal (baseline).
      Days 1, 2, 3 = first three treatment recording days.

    No block annotations are drawn (3-day span is too short).
    Color: COLOR_4PCT (#424143, dark grey).
    """
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()

    # Flexible column rename (case-insensitive)
    col_lower = {c.strip().lower(): c for c in raw.columns}
    rename = {}
    for key, canonical in [("id", "ID"), ("sex", "Sex"),
                            ("condition", "Condition"), ("date", "Date"),
                            ("total change", "Total Change")]:
        if key in col_lower and col_lower[key] != canonical:
            rename[col_lower[key]] = canonical
    if rename:
        raw = raw.rename(columns=rename)

    raw["Date"]         = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Total Change"] = pd.to_numeric(raw["Total Change"], errors="coerce")
    if "Sex" in raw.columns:
        raw["Sex"] = (
            raw["Sex"].astype(str).str.strip().str.upper()
            .map(lambda x: "M" if x.startswith("M")
                           else ("F" if x.startswith("F") else np.nan))
        )

    # Filter to 4% CA condition
    if "Condition" not in raw.columns:
        raise ValueError(f"'Condition' column not found in {csv_path.name}.")
    df = raw[raw["Condition"].astype(str).str.strip().str.lower()
             .str.startswith("4%")].copy()
    df = df.dropna(subset=["ID", "Date"]).reset_index(drop=True)

    # Per-animal Day (Day 0 = first/baseline date)
    df = df.sort_values(["ID", "Date"]).reset_index(drop=True)
    first_dates = df.groupby("ID")["Date"].transform("min")
    df["Day"] = (df["Date"] - first_dates).dt.days.astype(int)

    # Keep only Days 1, 2, 3
    df = df[df["Day"].isin([1, 2, 3])].copy()
    if df.empty:
        raise ValueError(
            f"No 4% CA pilot data found for Days 1–3 in {csv_path.name}. "
            "Check that a baseline (Day 0) date is present for each animal."
        )

    # Sex map
    sex_map: Dict[str, Optional[str]] = {}
    for mid, g in df.groupby("ID"):
        sex_val = None
        for v in g["Sex"].dropna().astype(str).str.upper():
            if v.startswith("M"):
                sex_val = "M"; break
            elif v.startswith("F"):
                sex_val = "F"; break
        sex_map[str(mid)] = sex_val

    # Per-animal Total Change series
    series_by_id: Dict[str, pd.Series] = {}
    for mid, g in df.groupby("ID", dropna=True):
        g = g.dropna(subset=["Total Change"])
        if g.empty:
            continue
        ser = g.set_index("Day")["Total Change"].sort_index()
        ser = ser.groupby(level=0).last()
        ser.name = str(mid)
        series_by_id[str(mid)] = ser

    fig, ax = plt.subplots()

    for mid, s in series_by_id.items():
        sex    = sex_map.get(str(mid))
        marker = SEX_MARKER.get(sex or "Unknown", "^")
        ax.plot(
            s.index, s.values,
            color=COLOR_4PCT,
            marker=marker,
            markersize=plt.rcParams["lines.markersize"],
            linewidth=plt.rcParams["lines.linewidth"],
            alpha=0.85,
            label=str(mid),
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Total Weight Change (%)")
    ax.set_title("Total Weight Change — 4% CA Pilot (Days 1–3)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1", "2", "3"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-25, 0)
    ax.set_yticks([-25, -20, -15, -10, -5, 0])
    ax.grid(False)

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=False,   # fixed x-range; skip auto-margin
        remove_y_margins=True,
        ticks_in=True,
        draw_zero_dotted_line=True,
    )

    leg = [
        ax.plot([], [], color=COLOR_4PCT, marker="s",
                linestyle="-", label="Male")[0],
        ax.plot([], [], color=COLOR_4PCT, marker="o",
                linestyle="-", label="Female")[0],
    ]
    ax.legend(handles=leg, loc="best", frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig2_generate_descriptive_stats(
    df: pd.DataFrame,
    mode: str,
    csv_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> str:
    """Per-Week (nonramp) or per-CA% (ramp) descriptive statistics report.

    Mirrors the format produced by
    behavioral_analysis.generate_descriptive_stats_report().

    Parameters
    ----------
    df       : master_data_*.csv loaded via _load_master_csv()
    mode     : 'ramp' or 'nonramp'
    csv_path : original CSV path (used to detect 2-wk ramp; may be None)
    save_path: if provided, written as a .txt file at this stem path
    """
    from datetime import datetime as _dt
    from scipy.stats import t as _td

    # ── Configuration ──────────────────────────────────────────────────────────
    if mode == "ramp":
        wc, fl, mode_label = "CA (%)", "CA%", "RAMP"
        def _grp_lbl(v: float) -> str: return f"{int(v)}%"
    else:
        wc, fl, mode_label = "Week", "Week", "NONRAMP"
        def _grp_lbl(v: float) -> str: return f"Week {int(v)}"

    # ── Data preparation ────────────────────────────────────────────────────────
    cdf = df.copy()
    cdf = _add_day_col(cdf, mode)

    if mode == "nonramp":
        cdf = _add_week_col(cdf)
        cdf = cdf[cdf["Day"] >= 1].copy()   # exclude baseline Day 0
    else:
        # Exclude Day 1 baseline (TC = 0) for standard ramp; keep for 2-wk ramp
        _is_2wk = csv_path is not None and any(
            kw in str(csv_path).lower() for kw in ("2wk", "2_wk", "2_week", "2week")
        )
        if not _is_2wk and "Day" in cdf.columns:
            cdf = cdf[cdf["Day"] > 1].copy()

    if wc not in cdf.columns:
        return f"ERROR: Column '{wc}' not found.  Cannot generate descriptive stats.\n"

    levels = sorted(
        float(w) for w in cdf[wc].dropna().unique()
        if not (mode == "nonramp" and float(w) <= 0)
    )
    if not levels:
        return "ERROR: No within-factor levels found after filtering.\n"

    # ── Statistical helpers ─────────────────────────────────────────────────────
    def _ci95(arr: np.ndarray):
        n = len(arr)
        if n < 2: return float("nan"), float("nan")
        se  = float(np.std(arr, ddof=1)) / np.sqrt(n)
        t_c = float(_td.ppf(0.975, df=n - 1))
        mu  = float(np.mean(arr))
        return mu - t_c * se, mu + t_c * se

    def _bootstrap_ci95(arr: np.ndarray, n_boot: int = 5000, seed: int = 42):
        n = len(arr)
        if n < 2: return float("nan"), float("nan")
        rng   = np.random.default_rng(seed)
        boots = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
        return max(0.0, float(np.percentile(boots, 2.5))), min(100.0, float(np.percentile(boots, 97.5)))

    def _wilson_ci(k: int, n: int, z: float = 1.96):
        if n == 0: return float("nan"), float("nan")
        p = k / n;  d = 1 + z**2 / n
        c = (p + z**2 / (2*n)) / d
        h = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
        return max(0.0, c - h), min(1.0, c + h)

    # ── Build report ────────────────────────────────────────────────────────────
    W   = 80
    SEP = "\u2500" * 60    # ─ repeated 60
    level_labels = [_grp_lbl(w) for w in levels]

    lines: List[str] = [
        "=" * W,
        f"DESCRIPTIVE STATISTICS REPORT \u2014 {mode_label} MODE",
        "=" * W,
        f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Within-subjects factor: {fl}  |  Levels: {level_labels}",
        "=" * W,
        "",
        "CONFIDENCE INTERVAL METHODS",
        "-" * 60,
        "  Continuous DVs (Total Change, Daily Change):",
        "    95% CI = t-distribution  (x\u0305 \u00b1 t\u2080.\u2097\u2085 * SD/\u221an)",
        "    Assumes approximate normality of animal-level means.",
        "",
        "  Behavioral mean% per animal (Nest Made?, Lethargy?, etc.):",
        "    95% CI = bootstrap percentile  (n_boot=5000, seed=42)",
        "    Non-parametric; no normality assumption; clamped to [0%, 100%].",
        "    Used because t-CI produces invalid bounds near 0% or 100%.",
        "    NOTE: SD and Var are reported on the proportion scale (0\u20131), not the",
        "    percentage scale (0\u2013100).  This matches the output of behavioral_analysis.py.",
        "    To convert: SD_pct = SD_prop \u00d7 100;  Var_pct = Var_prop \u00d7 10 000.",
        "",
        "  Behavioral prevalence (proportion of animals showing \u22651 event):",
        "    95% CI = Wilson score  (binomial proportion CI)",
        "    Appropriate for k/n counts; valid at extreme proportions.",
        "",
        "  Starting weight by sex:",
        "    95% CI = t-distribution  (x\u0305 \u00b1 t\u2080.\u2097\u2085 * SD/\u221an)",
        "-" * 60,
        "",
    ]

    # ── Continuous DVs ──────────────────────────────────────────────────────────
    cont_cols = [c for c in ("Total Change", "Daily Change") if c in cdf.columns]

    for col in cont_cols:
        lines += [
            SEP,
            f"  {col.upper()}",
            SEP,
            "  Note: 95% CI = t-distribution (x\u0305 \u00b1 t\u2080.\u2097\u2085 * SD/\u221an)",
            f"  {'Level':>8}  {'n':>4}  {'Mean':>8}  {'Median':>8}  "
            f"{'SD':>8}  {'Var':>10}  {'95% CI (t)':>22}",
            "  " + "-" * 76,
        ]
        for wk in levels:
            sub = cdf[cdf[wc] == wk]
            am  = sub.groupby("ID")[col].mean().dropna().values \
                  if "ID" in sub.columns else sub[col].dropna().values
            n = len(am)
            if n == 0: continue
            mu, med = float(np.mean(am)), float(np.median(am))
            sd  = float(np.std(am, ddof=1)) if n > 1 else float("nan")
            var = sd**2 if not np.isnan(sd) else float("nan")
            lo, hi = _ci95(am)
            ci = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "N/A"
            lines.append(
                f"  {_grp_lbl(wk):>8}  {n:>4}  {mu:>8.3f}  {med:>8.3f}  "
                f"{sd:>8.3f}  {var:>10.3f}  {ci:>22}"
            )
        lines.append("  " + "-" * 76)
        am_a = cdf.groupby("ID")[col].mean().dropna().values \
               if "ID" in cdf.columns else cdf[col].dropna().values
        n_a = len(am_a)
        if n_a > 0:
            mu_a, med_a = float(np.mean(am_a)), float(np.median(am_a))
            sd_a  = float(np.std(am_a, ddof=1)) if n_a > 1 else float("nan")
            var_a = sd_a**2 if not np.isnan(sd_a) else float("nan")
            lo_a, hi_a = _ci95(am_a)
            ci_a = f"[{lo_a:.3f}, {hi_a:.3f}]" if not np.isnan(lo_a) else "N/A"
            lines.append(
                f"  {'All':>8}  {n_a:>4}  {mu_a:>8.3f}  {med_a:>8.3f}  "
                f"{sd_a:>8.3f}  {var_a:>10.3f}  {ci_a:>22}"
            )
        lines.append("")

    # ── Behavioral DVs ──────────────────────────────────────────────────────────
    beh_cands = ("Nest Made?", "Lethargy?", "Anxious Behaviors?", "CA Spot Digging?")
    beh_cols  = [c for c in beh_cands if c in cdf.columns]

    for col in beh_cols:
        lines += [
            SEP,
            f"  {col.upper()}",
            SEP,
            "  Note: 95% CI = bootstrap percentile (n_boot=5000, seed=42), clamped to [0%, 100%]",
            "  Note: SD and Var are on the proportion scale (0\u20131). Mean% and Median% are displayed",
            "        as percentages (\u00d7100) for readability; SD and Var are NOT scaled.",
            f"  {'Level':>8}  {'n':>4}  {'n\u22651':>7}  {'Mean%':>7}  "
            f"{'95% CI (boot)':>22}  {'Median%':>8}  {'SD':>8}  {'Var':>10}",
            "  " + "-" * 85,
        ]
        for wk in levels:
            sub = cdf[cdf[wc] == wk]
            if "ID" not in sub.columns: continue
            pcts: List[float] = []; n_once = 0
            for _, adf in sub.groupby("ID"):
                v = adf[col].dropna()
                if len(v) == 0: continue
                b = v.astype(bool)
                pcts.append(float(b.mean()))   # proportion scale (0–1); SD/Var match original
                if b.any(): n_once += 1
            n = len(pcts)
            if n == 0: continue
            arr = np.array(pcts)
            mu, med = float(np.mean(arr)), float(np.median(arr))
            sd  = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
            var = sd**2 if not np.isnan(sd) else float("nan")
            lo, hi = _bootstrap_ci95(arr * 100)   # bootstrap on % scale for [0,100] clamping
            ci = f"[{lo:.1f}%, {hi:.1f}%]" if not np.isnan(lo) else "N/A"
            lines.append(
                f"  {_grp_lbl(wk):>8}  {n:>4}  {n_once:>7}  {f'{mu*100:.1f}%':>7}  "
                f"{ci:>22}  {f'{med*100:.1f}%':>8}  {sd:>8.3f}  {var:>10.3f}"
            )
        lines.append("  " + "-" * 85)
        if "ID" in cdf.columns:
            pcts_a: List[float] = []; n_once_a = 0
            for _, adf in cdf.groupby("ID"):
                v = adf[col].dropna()
                if len(v) == 0: continue
                b = v.astype(bool)
                pcts_a.append(float(b.mean()))   # proportion scale (0–1)
                if b.any(): n_once_a += 1
            n_a = len(pcts_a)
            if n_a > 0:
                arr_a = np.array(pcts_a)
                mu_a, med_a = float(np.mean(arr_a)), float(np.median(arr_a))
                sd_a  = float(np.std(arr_a, ddof=1)) if n_a > 1 else float("nan")
                var_a = sd_a**2 if not np.isnan(sd_a) else float("nan")
                lo_a, hi_a = _bootstrap_ci95(arr_a * 100)   # bootstrap on % scale
                ci_a = f"[{lo_a:.1f}%, {hi_a:.1f}%]" if not np.isnan(lo_a) else "N/A"
                lines.append(
                    f"  {'All':>8}  {n_a:>4}  {n_once_a:>7}  {f'{mu_a*100:.1f}%':>7}  "
                    f"{ci_a:>22}  {f'{med_a*100:.1f}%':>8}  {sd_a:>8.3f}  {var_a:>10.3f}"
                )
        lines.append("")

    # ── Behavioral Prevalence ───────────────────────────────────────────────────
    if beh_cols:
        lines += [
            "",
            "=" * W,
            f"BEHAVIORAL PREVALENCE  (proportion of animals showing \u22651 event, per {fl})",
            "=" * W,
            "Note: prevalence = n_true / n  (binomial proportion; Wilson 95% CI)",
            "",
        ]
        for col in beh_cols:
            lines += [
                SEP,
                f"  {col.upper()}",
                SEP,
                f"  {'Level':>8}  {'n':>4}  {'n\u22651':>7}  {'Prev%':>7}  {'95% CI (Wilson)':>24}",
                "  " + "-" * 58,
            ]
            for wk in levels:
                sub = cdf[cdf[wc] == wk]
                if "ID" not in sub.columns: continue
                n = sub["ID"].nunique(); n_once = 0
                for _, adf in sub.groupby("ID"):
                    v = adf[col].dropna()
                    if len(v) > 0 and v.astype(bool).any(): n_once += 1
                if n == 0: continue
                prev = 100.0 * n_once / n
                lo, hi = _wilson_ci(n_once, n)
                ci = f"[{lo*100:.1f}%, {hi*100:.1f}%]" if not np.isnan(lo) else "N/A"
                lines.append(
                    f"  {_grp_lbl(wk):>8}  {n:>4}  {n_once:>7}  {prev:>6.1f}%  {ci:>24}"
                )
            lines.append("  " + "-" * 58)
            if "ID" in cdf.columns:
                n_a = cdf["ID"].nunique(); n_once_a = 0
                for _, adf in cdf.groupby("ID"):
                    v = adf[col].dropna()
                    if len(v) > 0 and v.astype(bool).any(): n_once_a += 1
                if n_a > 0:
                    prev_a = 100.0 * n_once_a / n_a
                    lo_a, hi_a = _wilson_ci(n_once_a, n_a)
                    ci_a = f"[{lo_a*100:.1f}%, {hi_a*100:.1f}%]" if not np.isnan(lo_a) else "N/A"
                    lines.append(
                        f"  {'All':>8}  {n_a:>4}  {n_once_a:>7}  {prev_a:>6.1f}%  {ci_a:>24}"
                    )
            lines.append("")

    # ── Starting Weight by Sex ──────────────────────────────────────────────────
    if all(c in df.columns for c in ("Weight", "ID", "Sex", "Date")):
        sw_rows: List[dict] = []
        for sid, grp in df.sort_values(["ID", "Date"]).groupby("ID"):
            first = grp.dropna(subset=["Weight"]).head(1)
            if not first.empty:
                svals = grp["Sex"].dropna().astype(str).str.strip().str.upper()
                sex = "M" if any(v.startswith("M") for v in svals) else \
                      "F" if any(v.startswith("F") for v in svals) else "Unknown"
                sw_rows.append({"ID": str(sid), "Sex": sex,
                                 "Weight": float(first["Weight"].iloc[0])})
        if sw_rows:
            sw = pd.DataFrame(sw_rows)
            lines += [
                "",
                "=" * W,
                "STARTING WEIGHT BY SEX",
                "=" * W,
                "Note: starting weight = first recorded weight per animal (earliest date)",
                "Note: 95% CI = t-distribution (x\u0305 \u00b1 t\u2080.\u2097\u2085 * SD/\u221an)",
                "",
                "Individual starting weights:",
                "-" * 44,
                f"  {'ID':<18}  {'Sex':>4}  {'Starting Weight (g)':>20}",
                "-" * 44,
            ]
            for _, row in sw.sort_values(["Sex", "ID"]).iterrows():
                lines.append(f"  {row['ID']:<18}  {row['Sex']:>4}  {row['Weight']:>20.2f}")
            lines += [
                "",
                "Summary statistics by sex:",
                "-" * 54,
                f"   {'Sex':>4}  {'n':>4}  {'Mean (g)':>10}  {'SD':>8}  {'':>22}  95% CI",
                "-" * 54,
            ]
            for sex in sorted(sw["Sex"].unique()):
                arr = sw[sw["Sex"] == sex]["Weight"].values.astype(float)
                n_s = len(arr)
                mu_s = float(np.mean(arr))
                sd_s = float(np.std(arr, ddof=1)) if n_s > 1 else float("nan")
                lo_s, hi_s = _ci95(arr)
                ci_s = f"[{lo_s:.2f}, {hi_s:.2f}]" if not np.isnan(lo_s) else "N/A"
                lines.append(f"     {sex:>4}  {n_s:>4}  {mu_s:>10.2f}  {sd_s:>8.3f}  {ci_s:>22}")
            lines.append("-" * 54)
            arr_a = sw["Weight"].values.astype(float)
            n_aa = len(arr_a); mu_aa = float(np.mean(arr_a))
            sd_aa = float(np.std(arr_a, ddof=1)) if n_aa > 1 else float("nan")
            lo_aa, hi_aa = _ci95(arr_a)
            ci_aa = f"[{lo_aa:.2f}, {hi_aa:.2f}]" if not np.isnan(lo_aa) else "N/A"
            lines.append(f"     {'All':>4}  {n_aa:>4}  {mu_aa:>10.2f}  {sd_aa:>8.3f}  {ci_aa:>22}")

    lines += ["", "=" * W, "END OF REPORT", "=" * W, ""]
    report = "\n".join(lines)

    if save_path is not None:
        sp = Path(save_path).with_suffix(".txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(report, encoding="utf-8")
        print(f"  Saved \u2192 {sp.relative_to(_ROOT)}")

    return report


# =============================================================================
# FIGURE 2f HELPERS — Slope Analysis
# Ported from across_cohort.py: calculate_animal_slopes, compare_slopes_*,
# plot_slopes_comparison, generate_slope_analysis_report.
# =============================================================================

def _fig2_prepare_combined_for_slopes(
    cohort_specs: List[Tuple[str, pd.DataFrame, str]],
) -> pd.DataFrame:
    """Combine cohorts into one DataFrame ready for per-animal slope fitting.

    Parameters
    ----------
    cohort_specs : list of (label, df, mode)
        label : cohort name stored in the 'Cohort' column
        df    : master CSV DataFrame from _load_master_csv()
        mode  : 'ramp' or 'nonramp'  (controls Day numbering)
    """
    frames = []
    for label, df, mode in cohort_specs:
        cdf = _add_day_col(df.copy(), mode)
        cdf["Cohort"] = label
        cdf["_mode"]  = mode   # temporary; used below to drop ramp Day 1
        frames.append(cdf)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["Day"] >= 1].copy()   # exclude nonramp baseline (Day 0)
    # Drop ramp Day 1 (Total Change = 0 by construction — a measurement artifact,
    # consistent with across_cohort.py add_day_column_across_cohorts drop_ramp_baseline=True)
    ramp_day1 = (combined["_mode"] == "ramp") & (combined["Day"] == 1)
    combined  = combined[~ramp_day1].copy()
    combined  = combined.drop(columns=["_mode"])
    combined["Week"] = ((combined["Day"] - 1) // 7) + 1
    return combined


def _fig2_calculate_slopes(
    combined_df: pd.DataFrame,
    measure: str = "Total Change",
    time_unit: str = "Week",
) -> pd.DataFrame:
    """Per-animal linear regression slope of *measure* vs *time_unit*.

    When time_unit='Week', daily values are averaged within each animal-week
    so every animal contributes exactly one data point per week to the fit.
    Returns DataFrame with columns: ID, Sex, Cohort, Slope, Intercept, R2, N_points.
    """
    req = ["ID", "Cohort", time_unit, measure]
    missing = [c for c in req if c not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing columns for slope calculation: {missing}")

    if time_unit == "Week":
        grp_cols = [c for c in ("ID", "Week", "Sex", "Cohort") if c in combined_df.columns]
        adf = combined_df.groupby(grp_cols, as_index=False)[measure].mean()
    else:
        adf = combined_df.copy()

    rows = []
    for aid, g in adf.groupby("ID", dropna=True):
        x = g[time_unit].values.astype(float)
        y = g[measure].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue
        slope, intercept, r_val, _, _ = stats.linregress(x, y)
        rows.append({
            "ID":        aid,
            "Sex":       g["Sex"].iloc[0] if "Sex" in g.columns else "Unknown",
            "Cohort":    g["Cohort"].iloc[0],
            "Slope":     float(slope),
            "Intercept": float(intercept),
            "R2":        float(r_val ** 2),
            "N_points":  int(len(x)),
        })
    return pd.DataFrame(rows)


def _fig2_dunn_posthoc_internal(
    group_data:   List[np.ndarray],
    group_labels: List[str],
) -> List[dict]:
    """Dunn's post-hoc test using pooled KW ranks, Holm-Bonferroni corrected.

    Consistent with the Kruskal-Wallis omnibus: uses the same pooled rank matrix.
    Returns a list of dicts with keys: label_a, label_b, na, nb, z_stat,
    p_raw, p_adj, r_rb.
    """
    from itertools import combinations as _comb

    sizes    = [len(g) for g in group_data]
    all_data = np.concatenate([np.asarray(g, dtype=float) for g in group_data])
    N        = len(all_data)
    all_rnks = stats.rankdata(all_data)

    grp_rnks, idx = [], 0
    for n_g in sizes:
        grp_rnks.append(all_rnks[idx:idx + n_g])
        idx += n_g

    _, tc       = np.unique(all_rnks, return_counts=True)
    tie_corr    = float(np.sum(tc ** 3 - tc))

    results = []
    for i, j in _comb(range(len(group_data)), 2):
        ni, nj = sizes[i], sizes[j]
        Ri, Rj = float(np.mean(grp_rnks[i])), float(np.mean(grp_rnks[j]))
        se     = np.sqrt(((N * (N + 1)) / 12 - tie_corr / (12 * (N - 1))) * (1/ni + 1/nj))
        z      = (Ri - Rj) / se if se > 0 else 0.0
        p_raw  = 2.0 * float(stats.norm.sf(abs(z)))
        results.append({
            "label_a": group_labels[i], "label_b": group_labels[j],
            "na": ni, "nb": nj, "z_stat": z, "p_raw": p_raw, "p_adj": float("nan"),
            "r_rb": z / np.sqrt(ni + nj),
        })

    # Holm-Bonferroni step-down
    valid = sorted([(k, r) for k, r in enumerate(results) if not np.isnan(r["p_raw"])],
                   key=lambda x: x[1]["p_raw"])
    running_max = 0.0
    for rank, (orig_idx, r) in enumerate(valid):
        adj         = min(r["p_raw"] * (len(valid) - rank), 1.0)
        running_max = max(running_max, adj)
        results[orig_idx]["p_adj"] = running_max
    return results


def _fig2_hl_bca_ci_internal(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed:   int = 0,
) -> Tuple[float, float, float]:
    """Hodges-Lehmann shift (A−B) + 95% BCa bootstrap CI.

    Returns (hl_estimate, ci_lo, ci_hi).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def _hl(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.median(np.subtract.outer(x, y).ravel()))

    hl_obs = _hl(a, b)
    rng    = np.random.default_rng(seed)
    boots  = np.array([
        _hl(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True))
        for _ in range(n_boot)
    ])

    # BCa: bias correction
    prop = np.clip(float(np.mean(boots < hl_obs)), 1e-6, 1 - 1e-6)
    z0   = float(stats.norm.ppf(prop))

    # BCa: acceleration via jackknife
    jack = ([_hl(np.delete(a, i), b) for i in range(len(a))] +
            [_hl(a, np.delete(b, j)) for j in range(len(b))])
    jack = np.array(jack)
    jm   = jack.mean()
    num  = np.sum((jm - jack) ** 3)
    den  = np.sum((jm - jack) ** 2)
    acc  = float(num / (6.0 * den ** 1.5)) if den > 0 else 0.0

    def _adj_pct(z_a: float) -> float:
        denom = max(1 - acc * (z0 + z_a), 1e-9)
        return float(stats.norm.cdf(z0 + (z0 + z_a) / denom))

    alpha = 0.025
    lo = float(np.percentile(boots, 100 * _adj_pct(stats.norm.ppf(alpha))))
    hi = float(np.percentile(boots, 100 * _adj_pct(stats.norm.ppf(1 - alpha))))
    return hl_obs, lo, hi


def _fig2_compare_slopes_between(slopes_df: pd.DataFrame) -> dict:
    """Kruskal-Wallis omnibus + Dunn’s post-hoc + Cohen’s d + HL shift.

    Returns dict with keys: groups, group_data, kruskal_wallis, pairwise.
    Each pairwise entry adds: cohens_d, hl_est, ci_lo, ci_hi.
    """
    groups     = sorted(slopes_df["Cohort"].unique())
    group_data = [slopes_df[slopes_df["Cohort"] == g]["Slope"].values for g in groups]

    kw_stat, kw_p = stats.kruskal(*group_data)
    dunn          = _fig2_dunn_posthoc_internal(group_data, groups)

    for r in dunn:
        a  = slopes_df[slopes_df["Cohort"] == r["label_a"]]["Slope"].values
        b  = slopes_df[slopes_df["Cohort"] == r["label_b"]]["Slope"].values
        na, nb = len(a), len(b)
        denom  = na + nb - 2
        pool_sd = (np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / denom)
                   if denom > 0 else float("nan"))
        r["cohens_d"] = (float((np.mean(a) - np.mean(b)) / pool_sd)
                         if not np.isnan(pool_sd) and pool_sd > 0 else float("nan"))
        hl, lo, hi    = _fig2_hl_bca_ci_internal(a, b)
        r["hl_est"], r["ci_lo"], r["ci_hi"] = hl, lo, hi

    return {
        "groups":         groups,
        "group_data":     group_data,
        "kruskal_wallis": {"statistic": float(kw_stat), "p_value": float(kw_p)},
        "pairwise":       dunn,
    }


def _fig2_compare_slopes_within(slopes_df: pd.DataFrame) -> dict:
    """One-sample t-test (slope ≠ 0) + descriptive stats per cohort."""
    rows = []
    for cohort in sorted(slopes_df["Cohort"].unique()):
        s   = slopes_df[slopes_df["Cohort"] == cohort]["Slope"].values
        n   = len(s)
        mu  = float(np.mean(s))
        sd  = float(np.std(s, ddof=1)) if n > 1 else float("nan")
        t_s, t_p = (stats.ttest_1samp(s, 0.0) if n >= 2
                    else (float("nan"), float("nan")))
        rows.append({"Cohort": cohort, "n": n, "mean_slope": mu, "sd_slope": sd,
                     "t_stat": float(t_s), "p_vs_zero": float(t_p)})
    return {"cohort_stats": rows}


def _fig2_plot_slopes(
    slopes_df:      pd.DataFrame,
    between_results: dict,
    measure:   str = "Total Change",
    time_unit: str = "Week",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart (mean ± SEM) of per-animal slopes, one bar per cohort, with
    overlaid individual data points and Dunn’s post-hoc significance brackets.

    Colors are the canonical cohort palette from the module-level constants.
    """
    import re as _re
    from itertools import combinations as _comb

    groups    = between_results["groups"]
    colors    = [cohort_color(g) for g in groups]
    positions = list(range(len(groups)))
    box_data  = [slopes_df[slopes_df["Cohort"] == g]["Slope"].values for g in groups]
    tick_lbl  = [_re.sub(r"\s*\(.*?\)", "", g).strip() for g in groups]

    fig, ax = plt.subplots()

    bar_means = [float(np.mean(d)) if len(d) > 0 else 0.0 for d in box_data]
    bar_sems  = [float(stats.sem(d))  if len(d) > 1 else 0.0 for d in box_data]
    ax.bar(positions, bar_means, width=0.65, color=colors, alpha=0.7,
           yerr=bar_sems,
           error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor="black"),
           zorder=2)

    rng = np.random.default_rng(42)
    for i, d in enumerate(box_data):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d,
                   color=colors[i], alpha=0.85, s=12, zorder=3, edgecolors="none")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_lbl)
    ax.set_xlim(-0.7, len(groups) - 1 + 0.7)
    ax.set_xlabel("Cohort")
    ax.set_ylabel(f"Slope  ({measure} per {time_unit})")
    ax.grid(False)
    apply_common_plot_style(ax, ticks_in=True, remove_top_right=True,
                            remove_x_margins=False, remove_y_margins=False,
                            draw_zero_dotted_line=False)

    # Significance brackets from Dunn’s post-hoc
    pairwise = between_results.get("pairwise", [])
    dunn_map = {(r["label_a"], r["label_b"]): r["p_adj"] for r in pairwise}
    dunn_map.update({(r["label_b"], r["label_a"]): r["p_adj"] for r in pairwise})
    pairs    = list(_comb(range(len(groups)), 2))
    n_pairs  = len(pairs)

    all_slopes = slopes_df["Slope"].values
    y_max_d  = float(np.nanmax(all_slopes)) if len(all_slopes) > 0 else 2.0
    y_min_d  = float(np.nanmin(all_slopes)) if len(all_slopes) > 0 else -2.0
    y_span   = max(y_max_d - y_min_d, 0.1)
    step     = y_span * 0.14
    tick_h   = step  * 0.15
    y_top    = y_max_d + y_span * 0.1

    for level, (i, j) in enumerate(pairs):
        p_adj = dunn_map.get((groups[i], groups[j]), float("nan"))
        if np.isnan(p_adj):
            continue
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        y_br = y_top + level * step
        ax.plot([i, i, j, j], [y_br - tick_h, y_br, y_br, y_br - tick_h],
                color="black", linewidth=0.8, zorder=4)
        ax.text((i + j) / 2, y_br + tick_h * 0.3, sig,
                ha="center", va="bottom", fontsize=7, zorder=5)

    ax.set_ylim(y_min_d - y_span * 0.12,
                y_top + n_pairs * step + y_span * 0.15)

    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig2_slope_analysis_report(
    slopes_df:       pd.DataFrame,
    within_results:  dict,
    between_results: dict,
    measure:   str = "Total Change",
    time_unit: str = "Week",
    save_path: Optional[Path] = None,
) -> str:
    """Comprehensive text report for the Figure 2f slope analysis.

    Includes individual slopes, within-cohort t-tests, Kruskal-Wallis omnibus,
    and Dunn’s post-hoc with effect sizes and BCa-bootstrap Hodges-Lehmann CIs.
    """
    from datetime import datetime as _dt

    W    = 80
    kw   = between_results.get("kruskal_wallis", {})
    pair = between_results.get("pairwise", [])

    def _fp(p: float) -> str:
        if np.isnan(p): return "N/A"
        return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    def _sig(p: float) -> str:
        if np.isnan(p): return "  "
        return "***" if p < 0.001 else " **" if p < 0.01 else "  *" if p < 0.05 else " ns"

    lines = [
        "=" * W,
        "SLOPE ANALYSIS REPORT: RATE OF WEIGHT CHANGE ACROSS COHORTS",
        "=" * W,
        f"Generated  : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Measure    : {measure}",
        f"Time unit  : {time_unit}",
        f"Cohorts    : {', '.join(between_results.get('groups', []))}",
        f"Animals    : {len(slopes_df)}",
        "",
        "METHOD",
        "-" * W,
        f"  Linear regression fitted per animal:  {measure} ~ {time_unit}.",
        f"  When time_unit='Week', measurements within each week are averaged",
        f"  first (one data point per animal per week).",
        "  Omnibus : Kruskal-Wallis H (non-parametric, 3 cohorts).",
        "  Post-hoc: Dunn's test (pooled KW ranks), Holm-Bonferroni corrected.",
        "  Effect size: rank-biserial r = z/√(nA+nB); Hodges-Lehmann shift (A−B);",
        "              95% BCa bootstrap CI on HL estimate (n=2 000 resamples).",
        "",
    ]

    # Individual slopes
    lines += [
        "=" * W, "INDIVIDUAL ANIMAL SLOPES", "=" * W,
        f"  {'ID':<18}  {'Sex':>4}  {'Cohort':<32}  {'Slope':>10}  {'R²':>8}  {'N':>4}",
        "  " + "-" * 80,
    ]
    for _, row in slopes_df.sort_values(["Cohort", "ID"]).iterrows():
        lines.append(
            f"  {str(row['ID']):<18}  {str(row.get('Sex','?')):>4}  "
            f"{str(row['Cohort']):<32}  {row['Slope']:>10.4f}  "
            f"{row['R2']:>8.3f}  {int(row['N_points']):>4}"
        )
    lines.append("")

    # Within-cohort t-tests
    lines += [
        "=" * W, "WITHIN-COHORT STATISTICS  (one-sample t-test: slope ≠ 0)", "=" * W,
        f"  {'Cohort':<32}  {'n':>4}  {'Mean Slope':>12}  {'SD':>8}  {'t':>8}  {'p vs 0':>10}",
        "  " + "-" * 82,
    ]
    for r in within_results.get("cohort_stats", []):
        lines.append(
            f"  {r['Cohort']:<32}  {r['n']:>4}  {r['mean_slope']:>12.4f}  "
            f"{r['sd_slope']:>8.4f}  {r['t_stat']:>8.3f}  "
            f"{_fp(r['p_vs_zero']):>10}  {_sig(r['p_vs_zero'])}"
        )
    lines.append("")

    # KW omnibus
    lines += [
        "=" * W, "BETWEEN-COHORT OMNIBUS  (Kruskal-Wallis)", "=" * W,
        f"  H = {kw.get('statistic', float('nan')):.4f},  "
        f"p = {_fp(kw.get('p_value', float('nan')))}  "
        f"{_sig(kw.get('p_value', float('nan')))}",
        "",
    ]

    # Dunn’s post-hoc
    lines += [
        "=" * W,
        "POST-HOC: DUNN'S TEST  (Holm-Bonferroni corrected, pooled KW ranks)",
        "=" * W,
        f"  {'Comparison':<42}  {'nA':>4}  {'nB':>4}  {'z':>8}  "
        f"{'p(raw)':>10}  {'p(adj)':>10}  {'r_rb':>7}  {'HL':>9}  {'95% CI'}",
        "  " + "-" * 106,
    ]
    for r in pair:
        p_adj  = r["p_adj"]
        hl_str = (f"{r.get('hl_est', float('nan')):.4f}"
                  if not np.isnan(r.get("hl_est", float("nan"))) else "N/A")
        ci_str = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                  if not np.isnan(r.get("ci_lo", float("nan"))) else "N/A")
        cmp    = f"{r['label_a']} vs {r['label_b']}"
        lines.append(
            f"  {cmp:<42}  {r['na']:>4}  {r['nb']:>4}  {r['z_stat']:>8.3f}  "
            f"{_fp(r['p_raw']):>10}  {_fp(p_adj):>10}  {_sig(p_adj):3}  "
            f"{r['r_rb']:>7.3f}  {hl_str:>9}  {ci_str}"
        )
    lines += [
        "",
        "  r_rb  : z/√(nA+nB); |r| ≥ 0.1 small, ≥ 0.3 medium, ≥ 0.5 large",
        "  HL    : Hodges-Lehmann location shift (A − B)",
        "  95% CI: BCa bootstrap on HL estimate (n=2 000 resamples, seed=0)",
        "  *p<0.05  **p<0.01  ***p<0.001  ns=not significant",
        "",
        "=" * W, "END OF REPORT", "=" * W, "",
    ]

    report = "\n".join(lines)

    if save_path is not None:
        sp = Path(save_path).with_suffix(".txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(report, encoding="utf-8")
        print(f"  Saved \u2192 {sp.relative_to(_ROOT)}")

    return report


def _fig2_run_nparld_total_change(
    cohort_specs: List[Tuple[str, pd.DataFrame, str]],
    save_path: Optional[Path] = None,
) -> dict:
    """
    Nonparametric repeated-measures analysis (nparLD F1-LD-F1) for Total Change.

    Ported verbatim from across_cohort.py run_nparld_cohort_week_r.
    Calls R via subprocess — requires R with the nparLD package installed.

    Parameters
    ----------
    cohort_specs : list of (label, df, mode)
        label : cohort label stored in the report
        df    : master CSV DataFrame from _load_master_csv()
        mode  : 'ramp' or 'nonramp'
    save_path : stem path; report written as .txt

    Returns
    -------
    dict with keys: r_output, report_str, report_path, n_subjects, cohorts, weeks, p_values
    """
    from datetime import datetime as _dt
    import os as _os
    import glob as _glob

    measure = "Total Change"

    # ── Data preparation (figure_maker convention) ─────────────────────────
    # nonramp: Day 0 = baseline (drop Day 0, keep Day >= 1)
    # ramp   : Day 1 = TC=0 construction artefact (drop Day 1, keep Day > 1)
    frames = []
    for label, df, mode in cohort_specs:
        cdf = _add_day_col(df.copy(), mode)
        if mode == "ramp":
            cdf = cdf[cdf["Day"] > 1].copy()   # exclude TC=0 baseline
        else:
            cdf = cdf[cdf["Day"] >= 1].copy()  # exclude pre-treatment baseline
        cdf = _add_week_col(cdf)
        if measure not in cdf.columns:
            print(f"  [WARNING] '{measure}' not found in cohort '{label}' — skipping")
            continue
        cdf["_Cohort"] = label
        frames.append(cdf[["ID", "_Cohort", "Week", measure]].dropna(subset=[measure]))

    if not frames:
        print(f"  [ERROR] No data available for nparLD {measure}.")
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={measure: "TotalChange", "_Cohort": "Cohort"})

    # ── Per-animal per-week mean ────────────────────────────────────────────
    weekly = (
        combined
        .groupby(["ID", "Cohort", "Week"], as_index=False)["TotalChange"]
        .mean()
    )

    # Complete-case filter
    week_counts  = weekly.groupby("ID")["Week"].nunique()
    all_weeks_n  = weekly["Week"].nunique()
    complete_ids = week_counts[week_counts == all_weeks_n].index
    n_dropped    = weekly["ID"].nunique() - len(complete_ids)
    weekly = weekly[weekly["ID"].isin(complete_ids)].copy()

    if weekly.empty or len(complete_ids) < 3:
        print(f"  [ERROR] Insufficient complete-case animals ({len(complete_ids)}) for nparLD.")
        return {}

    week_levels   = sorted(weekly["Week"].unique())
    cohort_levels = sorted(weekly["Cohort"].unique())
    n_subjects    = int(weekly["ID"].nunique())

    # ── Write temp CSV ──────────────────────────────────────────────────────
    tmp_csv = Path(tempfile.mktemp(suffix=".csv"))
    weekly[["ID", "Cohort", "Week", "TotalChange"]].to_csv(str(tmp_csv), index=False)

    _csv_r     = str(tmp_csv).replace("\\", "/")
    _wk_levels = ", ".join(str(w) for w in week_levels)
    _co_levels = ", ".join(f'"{c}"' for c in cohort_levels)

    _td  = tempfile.gettempdir().replace("\\", "/")
    _uid = str(id(weekly))[-6:]
    _ats_csv   = f"{_td}/nwt_ats_{_uid}.csv"
    _box_csv   = f"{_td}/nwt_box_{_uid}.csv"
    _wts_csv   = f"{_td}/nwt_wts_{_uid}.csv"
    _rte_csv   = f"{_td}/nwt_rte_{_uid}.csv"
    _pair_csv  = f"{_td}/nwt_pair_{_uid}.csv"
    _time_csv  = f"{_td}/nwt_time_{_uid}.csv"
    _wkmwu_csv = f"{_td}/nwt_wkmwu_{_uid}.csv"

    r_script = (
        "options(warn=1, scipen=999)\n"
        "set.seed(42)\n"
        "if (!require(\"nparLD\", quietly=TRUE, warn.conflicts=FALSE)) {\n"
        "  install.packages(\"nparLD\", repos=\"https://cran.r-project.org\", quiet=TRUE)\n"
        "  library(nparLD)\n"
        "}\n"
        f"data <- read.csv(\"{_csv_r}\")\n"
        f"data$Week   <- factor(data$Week,   levels=c({_wk_levels}))\n"
        f"data$Cohort <- factor(data$Cohort, levels=c({_co_levels}))\n"
        "data$ID <- factor(data$ID)\n"
        "result <- f1.ld.f1(\n"
        "  y=data$TotalChange, time=data$Week, group=data$Cohort, subject=data$ID,\n"
        "  time.name=\"Week\", group.name=\"Cohort\", description=FALSE\n"
        ")\n"
        f"ats <- as.data.frame(result$ANOVA.test); ats$effect <- rownames(ats)\n"
        f"write.csv(ats, \"{_ats_csv}\", row.names=FALSE)\n"
        f"if (!is.null(result$ANOVA.test.mod.Box)) {{\n"
        f"  box <- as.data.frame(result$ANOVA.test.mod.Box); box$effect <- rownames(box)\n"
        f"  write.csv(box, \"{_box_csv}\", row.names=FALSE)\n"
        f"}} else write.csv(data.frame(), \"{_box_csv}\", row.names=FALSE)\n"
        f"wts <- as.data.frame(result$Wald.test); wts$effect <- rownames(wts)\n"
        f"write.csv(wts, \"{_wts_csv}\", row.names=FALSE)\n"
        f"write.csv(as.data.frame(result$RTE), \"{_rte_csv}\", row.names=TRUE)\n"
        f"if (!is.null(result$pair.comparison)) {{\n"
        f"  write.csv(result$pair.comparison, \"{_pair_csv}\", row.names=FALSE)\n"
        f"}} else write.csv(data.frame(), \"{_pair_csv}\", row.names=FALSE)\n"
        f"if (!is.null(result$ANOVA.test.time)) {{\n"
        f"  att <- as.data.frame(result$ANOVA.test.time); att$cohort <- rownames(att)\n"
        f"  write.csv(att, \"{_time_csv}\", row.names=FALSE)\n"
        f"}} else write.csv(data.frame(), \"{_time_csv}\", row.names=FALSE)\n"
        "wk_rows <- list(); k_row <- 1\n"
        "for (wk in levels(data$Week)) {\n"
        "  sub <- data[data$Week == wk, ]\n"
        "  cohs <- as.character(levels(droplevels(sub$Cohort)))\n"
        "  if (length(cohs) < 2) next\n"
        "  pm <- combn(length(cohs), 2)\n"
        "  for (k in seq_len(ncol(pm))) {\n"
        "    g1 <- cohs[pm[1,k]]; g2 <- cohs[pm[2,k]]\n"
        "    x  <- sub$TotalChange[sub$Cohort == g1]\n"
        "    y  <- sub$TotalChange[sub$Cohort == g2]\n"
        "    x  <- x[!is.na(x)]; y <- y[!is.na(y)]\n"
        "    if (length(x)<1 || length(y)<1) next\n"
        "    tryCatch({\n"
        "      wt <- wilcox.test(x, y, exact=FALSE, conf.int=TRUE)\n"
        "      wk_rows[[k_row]] <- data.frame(\n"
        "        week=as.integer(wk), g1=g1, n1=length(x), g2=g2, n2=length(y),\n"
        "        U=as.numeric(wt$statistic), p_raw=wt$p.value,\n"
        "        hl=as.numeric(wt$estimate), ci_lo=wt$conf.int[1], ci_hi=wt$conf.int[2],\n"
        "        stringsAsFactors=FALSE)\n"
        "      k_row <- k_row + 1\n"
        "    }, error=function(e) {\n"
        "      tryCatch({\n"
        "        wt2 <- wilcox.test(x, y, exact=FALSE, conf.int=FALSE)\n"
        "        wk_rows[[k_row]] <<- data.frame(\n"
        "          week=as.integer(wk), g1=g1, n1=length(x), g2=g2, n2=length(y),\n"
        "          U=as.numeric(wt2$statistic), p_raw=wt2$p.value,\n"
        "          hl=NA_real_, ci_lo=NA_real_, ci_hi=NA_real_,\n"
        "          stringsAsFactors=FALSE)\n"
        "        k_row <<- k_row + 1\n"
        "      }, error=function(e2) NULL)\n"
        "    })\n"
        "  }\n"
        "}\n"
        f"if (length(wk_rows)>0) write.csv(do.call(rbind,wk_rows),\"{_wkmwu_csv}\",row.names=FALSE) else write.csv(data.frame(),\"{_wkmwu_csv}\",row.names=FALSE)\n"
        "cat(\"NPARLD_WT_DONE\\n\")\n"
    )

    tmp_r = Path(tempfile.mktemp(suffix=".R"))
    tmp_r.write_text(r_script, encoding="utf-8")

    # ── Locate Rscript ──────────────────────────────────────────────────────
    rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if rscript is None:
        for _pat in (
            r"C:\Program Files\R\R-*\bin\Rscript.exe",
            r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
        ):
            _m = sorted(_glob.glob(_pat))
            if _m:
                rscript = _m[-1]
                break

    if rscript is None:
        print("ERROR: 'Rscript' not found. Install R and add to PATH. nparLD skipped.")
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)
        return {}

    r_output = ""
    try:
        proc = subprocess.run(
            [rscript, "--vanilla", str(tmp_r)],
            capture_output=True, text=True, timeout=300,
        )
        r_output = proc.stdout
        r_stderr = proc.stderr.strip()
        if proc.returncode != 0:
            print(f"R exited with code {proc.returncode}.")
        if r_stderr:
            non_trivial = [ln for ln in r_stderr.splitlines()
                           if not ln.startswith("Loading") and ln.strip()]
            if non_trivial:
                print("R messages:\n" + "\n".join(non_trivial))
    except FileNotFoundError:
        print("ERROR: 'Rscript' not found. nparLD skipped.")
        return {}
    except subprocess.TimeoutExpired:
        print("ERROR: R script timed out after 300 s. nparLD skipped.")
        return {}
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_r.unlink(missing_ok=True)

    # ── Read back result CSVs ───────────────────────────────────────────────
    def _safe_read_csv(fp):
        try:    return pd.read_csv(fp) if _os.path.exists(fp) else pd.DataFrame()
        except Exception: return pd.DataFrame()

    ats_df   = _safe_read_csv(_ats_csv)
    box_df   = _safe_read_csv(_box_csv)
    wts_df   = _safe_read_csv(_wts_csv)
    rte_df   = _safe_read_csv(_rte_csv)
    pair_df  = _safe_read_csv(_pair_csv)
    time_df  = _safe_read_csv(_time_csv)
    wkmwu_df = _safe_read_csv(_wkmwu_csv)

    for _fp in [_ats_csv, _box_csv, _wts_csv, _rte_csv, _pair_csv, _time_csv, _wkmwu_csv]:
        try:    _os.unlink(_fp)
        except Exception: pass

    # ── Fix zero p-values underflowed by R ─────────────────────────────────
    def _fix_zero_pvals(df, stat_cols, df1_cols, p_col, mode="ats", df2_col=None):
        if df.empty or p_col not in df.columns: return df
        s_c  = next((c for c in stat_cols if c in df.columns), None)
        d1_c = next((c for c in df1_cols  if c in df.columns), None)
        if s_c is None or d1_c is None: return df
        df = df.copy()
        for idx in df.index:
            try:
                pv = float(df.at[idx, p_col])
            except (TypeError, ValueError): continue
            if np.isnan(pv) or pv != 0.0: continue
            try:
                st = float(df.at[idx, s_c])
                d1 = float(df.at[idx, d1_c])
                if mode == "ats" and d1 > 0:
                    df.at[idx, p_col] = float(stats.chi2.sf(st * d1, d1))
                elif mode == "wts" and d1 > 0:
                    df.at[idx, p_col] = float(stats.chi2.sf(st, d1))
                elif mode == "box" and df2_col and df2_col in df.columns:
                    d2 = float(df.at[idx, df2_col])
                    if d2 > 0:
                        df.at[idx, p_col] = float(stats.f.sf(st, d1, d2))
            except Exception: pass
        return df

    _ats_pc  = next((c for c in ["p-value", "p.value", "Pr(>F)"]     if c in ats_df.columns),  None)
    _wts_pc  = next((c for c in ["p-value", "p.value", "Pr(>Chisq)"] if c in wts_df.columns),  None)
    _box_pc  = next((c for c in ["p-value", "p.value"]                if c in box_df.columns),  None)
    _pair_pc = next((c for c in ["p-value", "p.value"]                if c in pair_df.columns), None)
    _time_pc = next((c for c in ["p-value", "p.value"]                if c in time_df.columns), None)
    if _ats_pc:  ats_df  = _fix_zero_pvals(ats_df,  ["Statistic","ATS","F"],   ["df","df1","Df"], _ats_pc,  "ats")
    if _wts_pc:  wts_df  = _fix_zero_pvals(wts_df,  ["Statistic","WTS","Chisq"],["df","Df"],       _wts_pc,  "wts")
    if _box_pc:  box_df  = _fix_zero_pvals(box_df,  ["Statistic","ATS","F"],   ["df1","df","Df"], _box_pc,  "box", df2_col="df2")
    if _pair_pc: pair_df = _fix_zero_pvals(pair_df, ["Statistic","ATS"],       ["df","df1"],       _pair_pc, "ats")
    if _time_pc: time_df = _fix_zero_pvals(time_df, ["Statistic","ATS"],       ["df","df1"],       _time_pc, "ats")

    # ── Local formatting helpers ────────────────────────────────────────────
    def _fp_nw(p: float) -> str:
        try:
            if np.isnan(p): return "n/a"
        except (TypeError, ValueError): return "n/a"
        p = float(p)
        return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    def _sig_nw(p: float) -> str:
        try:
            if np.isnan(p): return ""
        except (TypeError, ValueError): return ""
        p = float(p)
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "."
        return ""

    def _holm_nw(p_list):
        n = len(p_list)
        if n == 0: return []
        order = sorted(range(n), key=lambda i: p_list[i])
        adj = [0.0] * n; running_min = 1.0
        for rank, idx in enumerate(reversed(order)):
            k = n - rank
            adj[idx] = min(running_min, p_list[idx] * k)
            running_min = adj[idx]
        return adj

    # ── Build report ────────────────────────────────────────────────────────
    W   = 80
    _ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines: List[str] = [
        "=" * W,
        "  NONPARAMETRIC REPEATED MEASURES ANALYSIS  (R: nparLD)",
        "  Design: F1-LD-F1  (1 between-subjects \u00d7 1 within-subjects factor)",
        "=" * W,
        f"  Generated         : {_ts}",
        f"  Measure           : {measure}",
        f"  Between factor    : Cohort  ({', '.join(cohort_levels)})",
        f"  Within factor     : Week    ({', '.join(str(w) for w in week_levels)})",
        f"  N subjects        : {n_subjects} (complete cases across all weeks)",
        "",
        "  Test statistics:",
        "    ATS = ANOVA-Type Statistic (F-approximation, recommended for small n)",
        "    WTS = Wald-Type Statistic  (Chi-sq, asymptotic, shown for reference)",
        "    RTE = Relative Treatment Effect (rank-based mean, 0-1 scale)",
        "",
        "  Post-hoc (per week): Mann-Whitney U + Holm correction",
        "    HL = Hodges-Lehmann estimator (x \u2212 y); 95% CI via wilcox.test",
        "    r_rb = rank-biserial correlation = 1 \u2212 2U/(n\u2081\u00b7n\u2082)",
        "=" * W,
        "",
        "  Significance: *** p<.001  ** p<.01  * p<.05  . p<.10",
        "",
        "\u2550" * W,
        f"  DEPENDENT VARIABLE:  {measure}",
        "\u2550" * W,
        "",
        "  Model Information",
        f"    {'Model (R)':<40} F1 LD F1 Model",
        f"    {'Method':<40} Nonparametric rank-based (nparLD)",
        f"    {'Between-subjects factor':<40} Cohort  ({', '.join(cohort_levels)})",
        f"    {'Within-subjects factor':<40} Week    ({', '.join(str(w) for w in week_levels)})",
        f"    {'N subjects (complete data)':<40} {n_subjects}",
        f"    {'N subjects dropped':<40} {n_dropped}",
        "",
        "  Note: nparLD uses rank transformation; no parametric model is fit.",
        "  Residuals, R\u00b2, AIC/BIC are not applicable. Inference is based on",
        "  Relative Treatment Effects (RTEs) which are bounded in [0, 1].",
        "",
    ]

    # ATS table
    if not ats_df.empty:
        _s_c  = next((c for c in ["Statistic","ATS","F"]     if c in ats_df.columns), None)
        _d1_c = next((c for c in ["df1","df","Df"]           if c in ats_df.columns), None)
        _d2_c = next((c for c in ["df2"]                      if c in ats_df.columns), None)
        _p_c  = next((c for c in ["p-value","p.value","Pr(>F)"] if c in ats_df.columns), None)
        report_lines += [
            "  ANOVA-Type Statistics (ATS)  \u2014  F-approximation, recommended for small n",
            f"  {'Effect':<30}  {'ATS':>10}  {'df1':>8}  {'df2':>8}  {'p-value':>10}  Sig",
            "  " + "\u2500" * 74,
        ]
        for _, row in ats_df.iterrows():
            eff  = str(row.get("effect", "")).strip()
            stat = float(row[_s_c])  if _s_c  else np.nan
            df1  = float(row[_d1_c]) if _d1_c else np.nan
            df2  = float(row[_d2_c]) if _d2_c else np.nan
            pv   = float(row[_p_c])  if _p_c  else np.nan
            df2s = f"{df2:>8.2f}" if not np.isnan(df2) else f"{'\u2014':>8}"
            report_lines.append(
                f"  {eff:<30}  {stat:>10.4f}  {df1:>8.2f}  {df2s}  {_fp_nw(pv):>10}  {_sig_nw(pv)}"
            )
        report_lines.append("")

    # Box approximation
    if not box_df.empty:
        _bs_c  = next((c for c in ["Statistic","ATS","F"] if c in box_df.columns), None)
        _bd1_c = next((c for c in ["df1","df","Df"]       if c in box_df.columns), None)
        _bd2_c = next((c for c in ["df2"]                  if c in box_df.columns), None)
        _bp_c  = next((c for c in ["p-value","p.value","Pr(>F)"] if c in box_df.columns), None)
        report_lines += [
            "  ATS with Box approximation  \u2014  preferred for small between-subjects N",
            f"  {'Effect':<30}  {'Statistic':>10}  {'df1':>8}  {'df2':>8}  {'p-value':>10}  Sig",
            "  " + "\u2500" * 74,
        ]
        for _, row in box_df.iterrows():
            eff  = str(row.get("effect", "")).strip()
            stat = float(row[_bs_c])  if _bs_c  else np.nan
            df1  = float(row[_bd1_c]) if _bd1_c else np.nan
            df2  = float(row[_bd2_c]) if _bd2_c else np.nan
            pv   = float(row[_bp_c])  if _bp_c  else np.nan
            df2s = f"{df2:>8.4f}" if not np.isnan(df2) else f"{'\u2014':>8}"
            report_lines.append(
                f"  {eff:<30}  {stat:>10.4f}  {df1:>8.4f}  {df2s}  {_fp_nw(pv):>10}  {_sig_nw(pv)}"
            )
        report_lines.append("")

    # WTS table
    if not wts_df.empty:
        _ws_c = next((c for c in ["Statistic","WTS","Chisq"] if c in wts_df.columns), None)
        _wd_c = next((c for c in ["df","Df"]                  if c in wts_df.columns), None)
        _wp_c = next((c for c in ["p-value","p.value","Pr(>Chisq)"] if c in wts_df.columns), None)
        report_lines += [
            "  Wald-Type Statistics (WTS)  \u2014  Chi-square, asymptotic (shown for reference)",
            f"  {'Effect':<30}  {'WTS (chi-sq)':>12}  {'df':>6}  {'p-value':>10}  Sig",
            "  " + "\u2500" * 66,
        ]
        for _, row in wts_df.iterrows():
            eff  = str(row.get("effect", "")).strip()
            stat = float(row[_ws_c]) if _ws_c else np.nan
            df_  = float(row[_wd_c]) if _wd_c else np.nan
            pv   = float(row[_wp_c]) if _wp_c else np.nan
            report_lines.append(
                f"  {eff:<30}  {stat:>12.4f}  {df_:>6.2f}  {_fp_nw(pv):>10}  {_sig_nw(pv)}"
            )
        report_lines.append("")

    # RTE section
    if not rte_df.empty:
        if "Unnamed: 0" in rte_df.columns:
            rte_df = rte_df.rename(columns={"Unnamed: 0": "label"})
        _lbl_c  = "label" if "label" in rte_df.columns else rte_df.columns[0]
        _est_c  = next((c for c in ["RTE","Estimate","rte"] if c in rte_df.columns), None)
        _nobs_c = next((c for c in ["Nobs","nobs","N"]      if c in rte_df.columns), None)
        report_lines += [
            "  Relative Treatment Effects (RTEs)  \u2014  rank-based effect sizes (0-1 scale)",
            "  Interpretation: RTE near 0.5 = no effect; >0.5 = higher ranks in this cell",
            "",
        ]
        cohort_rte_rows, week_rte_rows, cell_rte_rows = [], [], []
        for _, row in rte_df.iterrows():
            lbl = str(row.get(_lbl_c, "")).strip()
            if ":" in lbl:          cell_rte_rows.append((lbl, row))
            elif lbl.startswith("Week"): week_rte_rows.append((lbl, row))
            else:                   cohort_rte_rows.append((lbl, row))

        def _rte_line(lbl, row):
            est  = float(row[_est_c])  if _est_c  else np.nan
            nobs = int(row[_nobs_c]) if (_nobs_c and not pd.isna(row.get(_nobs_c))) else None
            ns   = f"  n={nobs}" if nobs is not None else ""
            return f"      {lbl:<36}  {est:>6.4f}{ns}"

        if cohort_rte_rows:
            report_lines.append("  Between-subjects (Cohort marginal):")
            for lbl, row in cohort_rte_rows: report_lines.append(_rte_line(lbl, row))
            report_lines.append("")
        if week_rte_rows:
            report_lines.append("  Within-subjects (Week marginal):")
            for lbl, row in week_rte_rows: report_lines.append(_rte_line(lbl, row))
            report_lines.append("")
        if cell_rte_rows:
            report_lines.append("  Cell RTEs (Cohort \u00d7 Week):")
            for lbl, row in cell_rte_rows: report_lines.append(_rte_line(lbl, row))
            report_lines.append("")

    # Pairwise comparisons
    _cohort_rte_map: Dict[str, float] = {}
    if not rte_df.empty and _est_c:
        for _, _crrow in rte_df.iterrows():
            _crlbl = str(_crrow.get(_lbl_c, "")).strip()
            if ":" not in _crlbl and not _crlbl.startswith("Week"):
                _crgrp = _crlbl.replace("Cohort", "", 1).strip()
                try: _cohort_rte_map[_crgrp] = float(_crrow[_est_c])
                except Exception: pass

    if not pair_df.empty:
        _pc_pair = next((c for c in pair_df.columns if "pair" in c.lower()), pair_df.columns[0])
        _pc_test = next((c for c in pair_df.columns if "test" in c.lower()), None)
        _pc_stat = next((c for c in ["Statistic","ATS"]  if c in pair_df.columns), None)
        _pc_df   = next((c for c in ["df","Df"]           if c in pair_df.columns), None)
        _pc_p    = next((c for c in ["p-value","p.value"] if c in pair_df.columns), None)

        _parsed: List[dict] = []
        for _, row in pair_df.iterrows():
            try:
                plbl = str(row.get(_pc_pair, "")).replace("Cohort", "").replace(":", " vs ")
                tlbl = str(row.get(_pc_test, "")) if _pc_test else ""
                stat = float(row[_pc_stat]) if _pc_stat else np.nan
                df_v = float(row[_pc_df])   if _pc_df   else np.nan
                pv   = float(row[_pc_p])    if _pc_p    else np.nan
                _parsed.append({"pair": plbl, "test": tlbl, "stat": stat, "df": df_v, "p_raw": pv})
            except Exception:
                _parsed.append({"raw": str(row.to_dict())})

        # Holm correction within each test type
        _ttype_idx: Dict[str, list] = {}
        for _pi, _pr in enumerate(_parsed):
            if "test" in _pr and "p_raw" in _pr:
                _ttype_idx.setdefault(_pr["test"], []).append(_pi)
        for _tt, _tidxs in _ttype_idx.items():
            _adjs = _holm_nw([_parsed[i]["p_raw"] for i in _tidxs])
            for _ti, _ap in zip(_tidxs, _adjs):
                _parsed[_ti]["p_adj"] = _ap

        report_lines += [
            "  Pairwise Group Comparisons  (from nparLD $pair.comparison)",
            "  Holm-Bonferroni correction applied within each test type (Cohort / Week / Cohort:Week)",
            "  Effect size: |\u0394RTE| = absolute difference in marginal RTEs (Cohort tests only)",
            f"  {'Pair':<28}  {'Test':<12}  {'ATS':>11}  {'df':>6}  {'p (raw)':>10}  {'p (Holm)':>10}  {'|\u0394RTE|':>8}  Sig",
            "  " + "\u2500" * 102,
        ]
        for _pr in _parsed:
            if "raw" in _pr:
                report_lines.append("  " + _pr["raw"])
            else:
                _p_adj = _pr.get("p_adj", _pr["p_raw"])
                if "Cohort" in _pr["test"] and "Week" not in _pr["test"] and _cohort_rte_map:
                    _pp = [p.strip() for p in _pr["pair"].split(" vs ")]
                    if len(_pp) == 2:
                        _r1 = _cohort_rte_map.get(_pp[0], float("nan"))
                        _r2 = _cohort_rte_map.get(_pp[1], float("nan"))
                        _dv = abs(_r1 - _r2) if not (np.isnan(_r1) or np.isnan(_r2)) else float("nan")
                        _ds = f"{_dv:.3f}" if not np.isnan(_dv) else "N/A"
                    else:
                        _ds = "N/A"
                else:
                    _ds = "\u2014"
                report_lines.append(
                    f"  {_pr['pair']:<28}  {_pr['test']:<12}  ATS={_pr['stat']:>9.4f}  "
                    f"df={_pr['df']:>6.2f}  {_fp_nw(_pr['p_raw']):>10}  "
                    f"{_fp_nw(_p_adj):>10}  {_ds:>8}  {_sig_nw(_p_adj)}"
                )
        report_lines.append("")

    # Within-group time effects
    if not time_df.empty:
        _tc_coh  = "cohort" if "cohort" in time_df.columns else time_df.columns[-1]
        _tc_stat = next((c for c in ["Statistic","ATS"] if c in time_df.columns), None)
        _tc_df   = next((c for c in ["df","Df"]          if c in time_df.columns), None)
        _tc_p    = next((c for c in ["p-value","p.value"] if c in time_df.columns), None)
        report_lines += [
            "  Within-Group Time Effects  (ATS per cohort, from nparLD $ANOVA.test.time)",
            "  " + "\u2500" * 77,
            f"  {'Cohort':<18}  {'ATS':>10}  {'df':>8}  {'p-value':>12}  Sig",
            "  " + "\u2500" * 56,
        ]
        for _, row in time_df.iterrows():
            coh  = str(row.get(_tc_coh, "")).strip()
            stat = float(row[_tc_stat]) if _tc_stat else np.nan
            df_  = float(row[_tc_df])   if _tc_df   else np.nan
            pv   = float(row[_tc_p])    if _tc_p    else np.nan
            report_lines.append(
                f"  {coh:<18}  {stat:>10.4f}  {df_:>8.4f}  {_fp_nw(pv):>12}  {_sig_nw(pv)}"
            )
        report_lines.append("")

    # Per-week MWU
    if not wkmwu_df.empty:
        report_lines += [
            "  Between-Cohort Comparisons at Each Week  (Mann-Whitney U, Holm corrected within each week)",
            "  Effect size: r_rb = rank-biserial correlation = 1 \u2212 2U/(n\u2081\u00b7n\u2082); range [\u22121, 1], positive = group1 > group2",
            "  HL = Hodges-Lehmann estimator of location shift (x \u2212 y); 95% CI via wilcox.test",
            "  " + "\u2500" * 88,
        ]
        for _wk_val in sorted(wkmwu_df["week"].unique()):
            _wk_sub  = wkmwu_df[wkmwu_df["week"] == _wk_val].copy()
            _p_adjs  = _holm_nw(_wk_sub["p_raw"].tolist())
            report_lines.append(f"  Week: {int(_wk_val)}")
            for (_idx, _row), _pa in zip(_wk_sub.iterrows(), _p_adjs):
                _g1, _n1 = str(_row["g1"]), int(_row["n1"])
                _g2, _n2 = str(_row["g2"]), int(_row["n2"])
                _U_v  = float(_row["U"])
                _pr_v = float(_row["p_raw"])
                _hl   = float(_row["hl"])
                _ci_lo = float(_row["ci_lo"])
                _ci_hi = float(_row["ci_hi"])
                _rrb  = 1.0 - 2.0 * _U_v / (_n1 * _n2)
                report_lines.append(
                    f"    {_g1} (n={_n1}) vs {_g2} (n={_n2}) :  "
                    f"U={_U_v:.0f}  p_raw={_fp_nw(_pr_v)}  p_holm={_fp_nw(_pa)}  "
                    f"r_rb={_rrb:+.3f}  {_sig_nw(_pa)}"
                )
                if np.isnan(_hl):
                    report_lines.append("      HL=[N/A]  95%CI [N/A, N/A]")
                else:
                    report_lines.append(f"      HL={_hl:+.4f}  95%CI [{_ci_lo:+.4f}, {_ci_hi:+.4f}]")
            report_lines.append("")

    # Collapsed across-weeks MWU
    _coll_groups = sorted(weekly["Cohort"].unique())
    if len(_coll_groups) >= 2:
        _coll_means = weekly.groupby(["ID", "Cohort"], as_index=False)["TotalChange"].mean()
        _coll_pairs_raw, _coll_pairs_info = [], []
        for _ci in range(len(_coll_groups)):
            for _cj in range(_ci + 1, len(_coll_groups)):
                _ga, _gb = _coll_groups[_ci], _coll_groups[_cj]
                _va = _coll_means[_coll_means["Cohort"] == _ga]["TotalChange"].dropna().values
                _vb = _coll_means[_coll_means["Cohort"] == _gb]["TotalChange"].dropna().values
                if len(_va) > 0 and len(_vb) > 0:
                    _U_c, _p_c = stats.mannwhitneyu(_va, _vb, alternative="two-sided")
                    _U_c = float(_U_c)
                    _rrb_c = 1.0 - 2.0 * _U_c / (len(_va) * len(_vb))
                    _hl_c  = float(np.median(np.subtract.outer(_va, _vb).ravel()))
                    _rng_c = np.random.default_rng(42)
                    _bt_c  = np.array([
                        np.median(np.subtract.outer(
                            _rng_c.choice(_va, size=len(_va), replace=True),
                            _rng_c.choice(_vb, size=len(_vb), replace=True)
                        ).ravel()) for _ in range(2000)
                    ])
                    _ci_lo_c = float(np.percentile(_bt_c, 2.5))
                    _ci_hi_c = float(np.percentile(_bt_c, 97.5))
                else:
                    _U_c = float("nan"); _p_c = 1.0
                    _rrb_c = float("nan"); _hl_c = float("nan")
                    _ci_lo_c = float("nan"); _ci_hi_c = float("nan")
                _coll_pairs_raw.append(_p_c)
                _coll_pairs_info.append((_ga, len(_va), _gb, len(_vb), _U_c, _p_c, _rrb_c, _hl_c, _ci_lo_c, _ci_hi_c))
        _coll_adj = _holm_nw(_coll_pairs_raw)
        report_lines += [
            "  Between-Cohort Comparisons Collapsed Across Weeks  (per-animal mean; MWU, Holm corrected)",
            "  Effect size: r_rb = rank-biserial correlation = 1 \u2212 2U/(n\u2081\u00b7n\u2082); range [\u22121, 1], positive = group1 > group2",
            "  HL = Hodges-Lehmann estimator of location shift (x \u2212 y); 95% CI via bootstrap percentile (2000 resamples)",
            "  " + "\u2500" * 88,
        ]
        for (_ga, _na, _gb, _nb, _U_c, _pr_c, _rrb_c, _hl_c, _ci_lo_c, _ci_hi_c), _pa_c in zip(_coll_pairs_info, _coll_adj):
            _rrb_s = f"{_rrb_c:+.3f}" if not np.isnan(_rrb_c) else "N/A"
            _u_s   = f"{_U_c:.0f}"    if not np.isnan(_U_c)   else "N/A"
            _hl_s  = f"{_hl_c:+.4f}"  if not np.isnan(_hl_c)  else "N/A"
            _ci_s  = (f"[{_ci_lo_c:+.4f}, {_ci_hi_c:+.4f}]"
                      if not (np.isnan(_ci_lo_c) or np.isnan(_ci_hi_c)) else "[N/A]")
            report_lines.append(
                f"    {_ga} (n={_na}) vs {_gb} (n={_nb}) :  "
                f"U={_u_s}  p_raw={_fp_nw(_pr_c)}  p_holm={_fp_nw(_pa_c)}  "
                f"r_rb={_rrb_s}  {_sig_nw(_pa_c)}"
            )
            report_lines.append(f"      HL={_hl_s}  95%CI {_ci_s}")
        report_lines.append("")

    report_lines += [
        "  Significance: *** p<.001  ** p<.01  * p<.05  . p<.10", "",
        "=" * W, "  End of nparLD Report", "=" * W,
    ]
    report_str = "\n".join(report_lines)

    # ── Save report ────────────────────────────────────────────────────────
    report_path: Optional[Path] = None
    if save_path is not None:
        sp = Path(save_path).with_suffix(".txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(report_str, encoding="utf-8")
        print(f"  Saved \u2192 {sp.relative_to(_ROOT)}")
        report_path = sp

    # Build p_values convenience DataFrame
    p_values: Optional[pd.DataFrame] = None
    if not ats_df.empty:
        _p_col_ats  = next((c for c in ["p-value","p.value","Pr(>F)"] if c in ats_df.columns), None)
        _s_col_ats  = next((c for c in ["Statistic","ATS","F"]         if c in ats_df.columns), None)
        _d1_col_ats = next((c for c in ["df1","df","Df"]                if c in ats_df.columns), None)
        if _p_col_ats and "effect" in ats_df.columns:
            _pv_cols   = ["effect", _p_col_ats]
            _pv_rename = {"effect": "Effect", _p_col_ats: "p_value"}
            if _s_col_ats:
                _pv_cols.append(_s_col_ats)
                _pv_rename[_s_col_ats] = "Statistic"
            if _d1_col_ats:
                _pv_cols.append(_d1_col_ats)
                _pv_rename[_d1_col_ats] = "df"
            p_values = ats_df[_pv_cols].rename(columns=_pv_rename)

    return {
        "r_output"   : r_output,
        "report_str" : report_str,
        "report_path": report_path,
        "n_subjects" : n_subjects,
        "cohorts"    : cohort_levels,
        "weeks"      : week_levels,
        "p_values"   : p_values,
    }


def figure_2() -> None:
    """Total Weight Change over Days — six panels for Figure 2.

    SVG files saved in OUT_FIG2:
      fig2a_total_change_0pct          — 0 % CA non-ramp         (nonramp)
      fig2b_total_change_2pct_full     — 2 % CA, 12 animals      (nonramp)
      fig2c_total_change_slow_ramp     — 5-wk slow CA ramp        (ramp)
      fig2d_total_change_fast_ramp     — 2-wk fast CA ramp        (ramp)
      fig2e_total_change_4pct_pilot    — 4 % CA pilot, Days 1–3  (special)
      fig2f_slope_comparison           — per-animal slope bar chart

    Text reports in OUT_FIG2:
      descriptive_stats_0pct.txt       — 0 % CA behavioral/weight descriptives
      descriptive_stats_2pct_6animals.txt  — 2 % CA (6-animal cohort)
      descriptive_stats_slow_ramp.txt  — 5-wk slow ramp
      descriptive_stats_fast_ramp.txt  — 2-wk fast ramp
      fig2f_slope_analysis_report.txt  — slope statistics (KW + Dunn’s)

    No subplot structure — each panel is its own figure.
    """
    print("\n" + "=" * 60)
    print("FIGURE 2 — Total Weight Change Per Cohort")
    print("=" * 60)

    # ── 2a: 0 % CA non-ramp ────────────────────────────────────────────────
    if MASTER_0PCT.exists():
        print("\n[2a] 0% CA non-ramp ...")
        _fig2_plot_cohort(
            _load_master_csv(MASTER_0PCT),
            mode="nonramp",
            cohort_color=COLOR_0PCT,
            title="Total Weight Change — 0% CA",
            save_path=OUT_FIG2 / "fig2a_total_change_0pct",
        )
    else:
        print(f"[2a] SKIPPED — not found: {MASTER_0PCT}")

    # ── 2b: 2 % CA full cohort (12 animals) ────────────────────────────────
    if MASTER_2PCT_FULL.exists():
        print("\n[2b] 2% CA full cohort (12 animals) ...")
        _fig2_plot_cohort(
            _load_master_csv(MASTER_2PCT_FULL),
            mode="nonramp",
            cohort_color=COLOR_2PCT,
            title="Total Weight Change — 2% CA (12 Animals)",
            save_path=OUT_FIG2 / "fig2b_total_change_2pct_full",
        )
    else:
        print(f"[2b] SKIPPED — not found: {MASTER_2PCT_FULL}")

    # ── 2c: 5-week slow CA ramp (1-day-normalised master CSV) ──────────────
    if MASTER_RAMP.exists():
        print("\n[2c] 5-week slow CA ramp ...")
        _fig2_plot_cohort(
            _load_master_csv(MASTER_RAMP),
            mode="ramp",
            cohort_color=COLOR_RAMP,
            title="Total Weight Change — 5-Week CA Ramp",
            save_path=OUT_FIG2 / "fig2c_total_change_slow_ramp",
        )
    else:
        print(f"[2c] SKIPPED — not found: {MASTER_RAMP}")

    # ── 2e: 4 % CA pilot (Days 1–3 only) ───────────────────────────────────
    if PILOT_CSV_1.exists():
        print("\n[2e] 4% CA pilot (Days 1–3) ...")
        _fig2_plot_pilot_4pct(
            PILOT_CSV_1,
            save_path=OUT_FIG2 / "fig2e_total_change_4pct_pilot",
        )
    else:
        print(f"[2e] SKIPPED — not found: {PILOT_CSV_1}")

    # ── 2d: 2-week fast CA ramp ─────────────────────────────────────────────
    if MASTER_2WK.exists():
        print("\n[2d] 2-week fast CA ramp ...")
        _fig2_plot_cohort(
            _load_master_csv(MASTER_2WK),
            mode="ramp",
            cohort_color=COLOR_OTHER,
            title="Total Weight Change — 2-Week CA Ramp",
            save_path=OUT_FIG2 / "fig2d_total_change_fast_ramp",
        )
    else:
        print(f"[2d] SKIPPED — not found: {MASTER_2WK}")

    # ── Descriptive statistics reports (saved to OUT_FIG2) ──────────────────
    print("\n" + "-" * 60)
    print("FIGURE 2 — Descriptive Statistics Reports")
    print("-" * 60)

    if MASTER_0PCT.exists():
        print("\n[stats-0%] 0% CA non-ramp ...")
        _fig2_generate_descriptive_stats(
            _load_master_csv(MASTER_0PCT),
            mode="nonramp",
            csv_path=MASTER_0PCT,
            save_path=OUT_FIG2 / "descriptive_stats_0pct",
        )

    if MASTER_2PCT.exists():          # 6-animal cohort, NOT the 12-animal full cohort
        print("\n[stats-2%] 2% CA (6-animal cohort) ...")
        _fig2_generate_descriptive_stats(
            _load_master_csv(MASTER_2PCT),
            mode="nonramp",
            csv_path=MASTER_2PCT,
            save_path=OUT_FIG2 / "descriptive_stats_2pct_6animals",
        )

    if MASTER_RAMP.exists():
        print("\n[stats-ramp] 5-wk slow ramp ...")
        _fig2_generate_descriptive_stats(
            _load_master_csv(MASTER_RAMP),
            mode="ramp",
            csv_path=MASTER_RAMP,
            save_path=OUT_FIG2 / "descriptive_stats_slow_ramp",
        )

    if MASTER_2WK.exists():
        print("\n[stats-2wk] 2-wk fast ramp ...")
        _fig2_generate_descriptive_stats(
            _load_master_csv(MASTER_2WK),
            mode="ramp",
            csv_path=MASTER_2WK,
            save_path=OUT_FIG2 / "descriptive_stats_fast_ramp",
        )

    # ── 2f: Slope comparison — 0%, 2% (6 animals), 5-wk slow ramp ────────────
    _slope_specs: List[Tuple[str, pd.DataFrame, str]] = []
    if MASTER_0PCT.exists():
        _slope_specs.append(("0% CA",            _load_master_csv(MASTER_0PCT),  "nonramp"))
    if MASTER_2PCT.exists():
        _slope_specs.append(("2% CA (6 animals)", _load_master_csv(MASTER_2PCT),  "nonramp"))
    if MASTER_RAMP.exists():
        _slope_specs.append(("5-Week Ramp",       _load_master_csv(MASTER_RAMP),  "ramp"))

    if len(_slope_specs) >= 2:
        print(f"\n[2f] Slope comparison ({len(_slope_specs)} cohorts) ...")
        _combined_s  = _fig2_prepare_combined_for_slopes(_slope_specs)
        _slopes_df   = _fig2_calculate_slopes(_combined_s, measure="Total Change", time_unit="Week")
        _within_s    = _fig2_compare_slopes_within(_slopes_df)
        _between_s   = _fig2_compare_slopes_between(_slopes_df)
        _fig2_plot_slopes(
            _slopes_df, _between_s,
            measure="Total Change", time_unit="Week",
            save_path=OUT_FIG2 / "fig2f_slope_comparison",
        )
        _fig2_slope_analysis_report(
            _slopes_df, _within_s, _between_s,
            measure="Total Change", time_unit="Week",
            save_path=OUT_FIG2 / "fig2f_slope_analysis_report",
        )
    else:
        print("[2f] SKIPPED — fewer than 2 slope cohorts available")

    # ── 2g: nparLD Total Change — 0%, 2% (6 animals), 5-wk slow ramp ────────
    _nparld_specs: List[Tuple[str, pd.DataFrame, str]] = []
    if MASTER_0PCT.exists():
        _nparld_specs.append(("0%",           _load_master_csv(MASTER_0PCT),  "nonramp"))
    if MASTER_2PCT.exists():
        _nparld_specs.append(("2% 6 animals", _load_master_csv(MASTER_2PCT),  "nonramp"))
    if MASTER_RAMP.exists():
        _nparld_specs.append(("ramp",          _load_master_csv(MASTER_RAMP),  "ramp"))

    if len(_nparld_specs) >= 2:
        print(f"\n[2g] nparLD Total Change ({len(_nparld_specs)} cohorts) ...")
        _fig2_run_nparld_total_change(
            _nparld_specs,
            save_path=OUT_FIG2 / "nparld_total_change",
        )
    else:
        print("[2g] SKIPPED — fewer than 2 cohorts available")

    print("\n[OK] Figure 2 complete.")


# =============================================================================
# FIGURE 3 — Daily Weight Change Across Days
#
# Panel:
#   3a — Mean ± SEM Daily Change per Day, three cohorts on one plot:
#         0 % CA (nonramp), 2 % CA 6-animals (nonramp), 5-Wk Slow Ramp (ramp)
#   3b — Mean ± SEM Aberrant Behavioral Metrics per Week, three cohorts on one plot:
#        0 % CA (nonramp), 2 % CA 6-animals (nonramp), 5-Wk Slow Ramp (ramp)
#   3c - 4-Cohort Transition-Day Daily Change
#        2 % CA 12-animals (nonramp), 5-Wk Slow Ramp (ramp), 2-Wk Fast Ramp (ramp), 4 % CA Pilot (nonramp)
#   3d - 2-Cohort Transition-Day Daily Change
#        5-Wk Slow Ramp (ramp), 2-Wk Fast Ramp (ramp)
#
# Source column : "Daily Change" (% body-weight change from the previous day)
# x-axis        : Day (integer, per-animal)
#
# Day alignment (critical — ramp and nonramp use different Day-0 conventions):
#   nonramp  Day 0 = baseline recording (excluded; filter Day >= 1).
#            Day 1 = first CA-exposure day.
#   ramp     Day 1 = first recorded date (no separate baseline row).
#            Day 1 IS included for the Daily Change plot — unlike the slope
#            analysis, Daily Change on Day 1 is a real measurement, not a
#            constructed zero.
#   Both cohort types therefore start at Day 1 on the same x-axis.
# =============================================================================

def _fig3a_plot_daily_change_cohorts(
    cohort_specs: List[Tuple[str, pd.DataFrame, str]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 3a: Mean ± SEM Daily Change across Days, one line per cohort.

    Parameters
    ----------
    cohort_specs : list of (label, df, mode)
        label : cohort display name — drives legend text and color lookup
        df    : master CSV DataFrame loaded by _load_master_csv()
        mode  : 'ramp' or 'nonramp'  (controls Day numbering via _add_day_col)
    save_path : stem path (no extension); SVG saved via save_fig()

    Day alignment
    -------------
    nonramp cohorts
        _add_day_col assigns Day 0 to the first recorded date (baseline).
        Rows with Day 0 are excluded here (filter ``Day >= 1``).
        Day 1 = first CA-exposure measurement.

    ramp cohort
        _add_day_col assigns Day 1 to the first recorded date.
        Day 1 is retained — for Daily Change, the Day-1 value is a real
        observation, not the TC=0 construction artefact that is excluded
        from slope analysis.

    Both cohort types plot from Day 1 on the same x-axis.
    """
    # ── Per-cohort, per-animal Daily Change series indexed by Day ────────────
    cohort_series: Dict[str, Dict[str, pd.Series]] = {}
    for label, df, mode in cohort_specs:
        cdf = _add_day_col(df.copy(), mode)
        cdf = cdf[cdf["Day"] >= 1].copy()               # drop nonramp baseline Day 0
        cdf = cdf.dropna(subset=["Daily Change", "Day"])
        series_by_id: Dict[str, pd.Series] = {}
        for gid, g in cdf.groupby("ID", dropna=True):
            ser = g.set_index("Day")["Daily Change"].sort_index()
            ser = ser.groupby(level=0).last()            # resolve duplicate days
            ser.name = str(gid)
            series_by_id[str(gid)] = ser
        cohort_series[label] = series_by_id

    fig, ax = plt.subplots()

    y_lo_all: List[float] = []
    y_hi_all: List[float] = []
    x_all:    List[float] = []

    for label, series_dict in cohort_series.items():
        if not series_dict:
            continue
        color = cohort_color(label)

        # Aligned per-day mean and SEM across animals
        all_days = sorted(set().union(*[set(s.index) for s in series_dict.values()]))
        days_plot, means_plot, sems_plot = [], [], []
        for day in all_days:
            vals = [
                float(s.at[day])
                for s in series_dict.values()
                if day in s.index and not np.isnan(float(s.at[day]))
            ]
            if not vals:
                continue
            mu  = float(np.mean(vals))
            sem = float(stats.sem(vals)) if len(vals) > 1 else 0.0
            days_plot.append(day)
            means_plot.append(mu)
            sems_plot.append(sem)

        if not days_plot:
            continue

        d  = np.asarray(days_plot,  dtype=float)
        mu = np.asarray(means_plot, dtype=float)
        se = np.asarray(sems_plot,  dtype=float)

        ax.plot(d, mu, color=color, linewidth=plt.rcParams["lines.linewidth"],
                label=label, zorder=3)
        ax.fill_between(d, mu - se, mu + se, color=color, alpha=0.2, zorder=2)

        x_all.extend(d.tolist())
        y_lo_all.extend((mu - se).tolist())
        y_hi_all.extend((mu + se).tolist())

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Weight Change (%)")
    ax.set_title("Daily Weight Change — 0% CA, 2% CA, 5-Wk Ramp")
    ax.grid(False)

    apply_common_plot_style(
        ax,
        start_x_at_zero=False,
        remove_top_right=True,
        remove_x_margins=True,
        remove_y_margins=True,
        ticks_in=True,
        draw_zero_dotted_line=True,
    )

    # x-axis: integer ticks, left edge clamped to Day 1
    if x_all:
        x_lo   = int(min(x_all)); x_hi = int(max(x_all))
        x_step = _auto_integer_step(x_lo, x_hi, target_ticks=10, allow_sub5=True)
        _apply_integer_axis(ax, axis="x", data_min=x_lo, data_max=x_hi,
                            step=x_step, clamp_min=1,
                            left_pad_steps=0, right_pad_steps=0)

    # y-axis: based on mean ± SEM bounds so shaded bands are fully visible
    if y_lo_all and y_hi_all:
        y_lo   = float(min(y_lo_all)); y_hi = float(max(y_hi_all))
        y_step = _auto_integer_step(y_lo, y_hi, target_ticks=7)
        _apply_integer_axis(ax, axis="y", data_min=y_lo, data_max=y_hi,
                            step=y_step, left_pad_steps=0, right_pad_steps=1)

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig3b_plot_behavioral_metrics(
    cohort_specs: List[Tuple[str, pd.DataFrame, str]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 3b: Three-panel line plot of aberrant behavioral observations per week.

    Panels (left to right)
    ----------------------
    No Nest  — Nest Made? == False
    Anxious  — Anxious Behaviors? == True
    Lethargy — Lethargy? == True

    For each cohort × week the per-animal % of aberrant observations is computed
    (proportion of that animal's recordings that week showing the behaviour), then
    mean ± SEM across animals is plotted as a line with a shaded error band.

    Day / Week alignment
    --------------------
    nonramp  Day 0 = baseline (excluded; filter Day ≥ 1).  Day 1 = first
             CA-exposure recording.  Week 1 = Days 1–7.
    ramp     Day 1 = first recorded date (included).  Week 1 = Days 1–7.
    Both cohort types share the same Week x-axis.

    Parameters
    ----------
    cohort_specs : list of (label, df, mode)
        label : cohort display name (legend text and color lookup)
        df    : master CSV DataFrame loaded by _load_master_csv()
        mode  : 'ramp' or 'nonramp'
    save_path : stem path (no extension); SVG saved via save_fig()
    """
    BEHAVIORS: List[Tuple[str, bool, str]] = [
        ("Nest Made?",         False, "No Nest"),
        ("Anxious Behaviors?", True,  "Anxious"),
        ("Lethargy?",          True,  "Lethargy"),
    ]

    def _is_aberrant(series: pd.Series, aberrant_val: bool) -> pd.Series:
        """Coerce column to bool, return boolean series testing for aberrant_val."""
        coerced = series.map({
            True: True, False: False,
            "yes": True, "no": False,
            "Yes": True, "No": False,
            "YES": True, "NO": False,
        })
        return coerced == aberrant_val

    # ── Build per-cohort weekly stats ────────────────────────────────────────
    # cohort_week_data[label][col][week] = (mean_pct, sem_pct)
    cohort_week_data: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = {}
    all_weeks: set = set()

    for label, df, mode in cohort_specs:
        cdf = _add_day_col(df.copy(), mode)
        cdf = cdf[cdf["Day"] >= 1].copy()   # drop nonramp baseline (Day 0)
        cdf = _add_week_col(cdf)

        beh_results: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for col, aberrant_val, _ in BEHAVIORS:
            week_stats: Dict[int, Tuple[float, float]] = {}
            if col not in cdf.columns:
                beh_results[col] = {}
                continue
            for week_val, wdf in cdf.groupby("Week"):
                if "ID" not in wdf.columns:
                    continue
                animal_pcts: List[float] = []
                for _, adf in wdf.groupby("ID"):
                    v = adf[col].dropna()
                    if len(v) == 0:
                        continue
                    ab = _is_aberrant(v, aberrant_val)
                    animal_pcts.append(100.0 * float(ab.mean()))
                if not animal_pcts:
                    continue
                mu  = float(np.mean(animal_pcts))
                sem = float(stats.sem(animal_pcts)) if len(animal_pcts) > 1 else 0.0
                week_stats[int(week_val)] = (mu, sem)
                all_weeks.add(int(week_val))
            beh_results[col] = week_stats
        cohort_week_data[label] = beh_results

    if not all_weeks:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5))
        for ax in axes:
            ax.text(0.5, 0.5, "No behavioral data", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        if save_path is not None:
            save_fig(fig, save_path)
        return fig

    sorted_weeks = sorted(all_weeks)

    # Shared y-axis ceiling: max(mean + SEM) × 1.1, floored at 5%
    y_tops: List[float] = [
        mu + sem
        for beh_results in cohort_week_data.values()
        for col, _, _ in BEHAVIORS
        for mu, sem in beh_results.get(col, {}).values()
    ]
    y_max = max(max(y_tops) * 1.1, 5.0) if y_tops else 100.0

    # ── Plot: 1 × 3 panels with shared y-axis ────────────────────────────────
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9.0, 2.5))

    for ax_idx, (col, _, panel_title) in enumerate(BEHAVIORS):
        ax = axes[ax_idx]

        for label, beh_results in cohort_week_data.items():
            week_stats = beh_results.get(col, {})
            if not week_stats:
                continue
            color = cohort_color(label)
            weeks_avail = sorted(week_stats.keys())
            means = [week_stats[w][0] for w in weeks_avail]
            sems  = [week_stats[w][1] for w in weeks_avail]

            ax.errorbar(weeks_avail, means, yerr=sems,
                        color=color, linewidth=plt.rcParams["lines.linewidth"],
                        marker="o", markersize=plt.rcParams["lines.markersize"],
                        capsize=3, capthick=0.8, elinewidth=0.8,
                        label=label, zorder=3)

        ax.set_title(panel_title)
        ax.set_xlabel("Week")
        ax.set_xticks(sorted_weeks)
        ax.set_xticklabels([str(w) for w in sorted_weeks])
        ax.set_xlim(sorted_weeks[0] - 0.5, sorted_weeks[-1] + 0.5)
        ax.grid(False)
        apply_common_plot_style(
            ax,
            start_x_at_zero=False,
            remove_top_right=True,
            remove_x_margins=False,
            remove_y_margins=True,
            ticks_in=True,
            draw_zero_dotted_line=False,
        )

    axes[0].set_ylabel("% of Observations")
    axes[0].set_ylim(0, y_max)
    # Legend on last panel only
    axes[-1].legend(loc="best", frameon=False)

    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# FIGURE 3c HELPERS — 4-Cohort Transition-Day Daily Change
#
# Each cohort contributes one Daily Change value per animal, taken at the day
# that represents the 0 % → 1 % CA transition (or first CA exposure):
#
#   Cohort               Target day            Convention
#   ──────────────────────────────────────────────────────────────────
#   2% CA (6 animals)    Day 1  (0-based nonramp)  First CA-exposure recording
#   4% CA (Pilot)        Day 1  (0-based)           First CA-exposure recording
#   5-Week Ramp          Day 8  (1-based ramp)       0% → 1% CA transition
#   2-Week Ramp          Day 4  (1-based ramp)       0% → 1% CA transition
# =============================================================================

def _fig3c_extract_transition_values() -> Dict[str, np.ndarray]:
    """
    Extract per-animal Daily Change at the cohort-specific transition day.

    Returns
    -------
    Dict[str, np.ndarray]
        Ordered dict: cohort label → 1-D array of per-animal Daily Change values.
        Cohorts whose data file is missing are silently omitted.
    """
    result: Dict[str, np.ndarray] = {}

    # ── 2% CA (12 animals, full cohort) — nonramp Day 1 (0-based) ───────────────
    if MASTER_2PCT_FULL.exists():
        cdf = _add_day_col(_load_master_csv(MASTER_2PCT_FULL), "nonramp")
        vals = (cdf[cdf["Day"] == 1]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["2% CA"] = vals

    # ── 4% CA (Pilot) — Day 1 (0-based, same nonramp convention) ─────────────────
    if PILOT_CSV_1.exists():
        raw = pd.read_csv(PILOT_CSV_1)
        raw.columns = raw.columns.str.strip()
        col_lower = {c.strip().lower(): c for c in raw.columns}
        rename = {}
        for key, canonical in [("id", "ID"), ("condition", "Condition"),
                               ("date", "Date"), ("daily change", "Daily Change")]:
            if key in col_lower and col_lower[key] != canonical:
                rename[col_lower[key]] = canonical
        if rename:
            raw = raw.rename(columns=rename)
        raw["Date"]         = pd.to_datetime(raw["Date"], errors="coerce")
        raw["Daily Change"] = pd.to_numeric(raw["Daily Change"], errors="coerce")
        df_p = raw[raw["Condition"].astype(str).str.strip().str.lower()
                   .str.startswith("4%")].copy()
        df_p = df_p.dropna(subset=["ID", "Date"]).reset_index(drop=True)
        df_p = df_p.sort_values(["ID", "Date"]).reset_index(drop=True)
        first_dates = df_p.groupby("ID")["Date"].transform("min")
        df_p["Day"] = (df_p["Date"] - first_dates).dt.days.astype(int)
        vals = (df_p[df_p["Day"] == 1]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["4% CA (Pilot)"] = vals

    # ── 5-Week Ramp — ramp Day 8 (0% → 1% CA transition) ───────────────────────
    if MASTER_RAMP.exists():
        cdf = _add_day_col(_load_master_csv(MASTER_RAMP), "ramp")
        vals = (cdf[cdf["Day"] == 8]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["5-Week Ramp"] = vals

    # ── 2-Week Ramp — ramp Day 4 (0% → 1% CA transition) ───────────────────────
    if MASTER_2WK.exists():
        cdf = _add_day_col(_load_master_csv(MASTER_2WK), "ramp")
        vals = (cdf[cdf["Day"] == 4]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["2-Week Ramp"] = vals

    return result


def _fig3c_kw_transition_analysis(
    cohort_values: Dict[str, np.ndarray],
) -> dict:
    """
    Kruskal-Wallis omnibus + Dunn’s post-hoc + HL shift on transition-day values.

    Returns dict with keys:
        groups, group_data, kruskal_wallis, eta2_H, pairwise.
    Each pairwise entry: label_a, label_b, na, nb, z_stat, p_raw, p_adj,
                         r_rb, hl_est, ci_lo, ci_hi.
    """
    labels     = [lbl for lbl, v in cohort_values.items() if len(v) > 0]
    group_data = [cohort_values[lbl] for lbl in labels]

    if len(labels) < 2:
        return {"groups": labels, "group_data": group_data,
                "kruskal_wallis": {}, "eta2_H": float("nan"), "pairwise": []}

    kw_stat, kw_p = stats.kruskal(*group_data)
    n_total = sum(len(g) for g in group_data)
    k       = len(group_data)
    eta2_H  = float((kw_stat - k + 1) / (n_total - k)) if (n_total - k) > 0 else float("nan")

    dunn = _fig2_dunn_posthoc_internal(group_data, labels)
    for r in dunn:
        hl, lo, hi = _fig2_hl_bca_ci_internal(
            cohort_values[r["label_a"]], cohort_values[r["label_b"]]
        )
        r["hl_est"], r["ci_lo"], r["ci_hi"] = hl, lo, hi

    return {
        "groups":         labels,
        "group_data":     group_data,
        "kruskal_wallis": {"statistic": float(kw_stat), "p_value": float(kw_p)},
        "eta2_H":         eta2_H,
        "pairwise":       dunn,
    }


def _fig3c_plot_transition_comparison(
    cohort_values: Dict[str, np.ndarray],
    kw_results:    dict,
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart (mean ± SEM) with individual scatter points.

    Significance brackets are drawn only for pairs with p_adj < 0.05 to keep
    the 4-cohort, 6-pair plot compact.
    """
    import re as _re
    from itertools import combinations as _comb

    labels    = list(cohort_values.keys())
    colors    = [cohort_color(lbl) for lbl in labels]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    bar_means = [float(np.mean(v)) if len(v) > 0 else 0.0 for v in cohort_values.values()]
    bar_sems  = [float(stats.sem(v)) if len(v) > 1 else 0.0 for v in cohort_values.values()]
    ax.bar(positions, bar_means, width=0.65, color=colors, alpha=0.7,
           yerr=bar_sems,
           error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor="black"),
           zorder=2)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(cohort_values.values()):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i], alpha=0.85, s=12, zorder=3, edgecolors="none")

    # Solid baseline at y = 0 (bars extend downward from here)
    ax.axhline(0, color="black", linewidth=0.9, zorder=2)

    # Only draw brackets for significant pairs.
    # All means are negative (weight loss), so bars extend downward from 0.
    # Brackets must start above 0 — anchor y_top on 0, not on scatter max.
    pairwise = kw_results.get("pairwise", [])
    dunn_map = {(r["label_a"], r["label_b"]): r["p_adj"] for r in pairwise}
    dunn_map.update({(r["label_b"], r["label_a"]): r["p_adj"] for r in pairwise})

    all_vals = np.concatenate(list(cohort_values.values()))
    y_min_d  = float(np.nanmin(all_vals)) if len(all_vals) > 0 else -15.0
    y_span   = max(abs(y_min_d), 0.1)   # full range from bottom to 0
    step     = y_span * 0.12
    tick_h   = step   * 0.15
    y_top    = y_span * 0.05             # just above 0

    sig_pairs = [
        (i, j) for i, j in _comb(range(len(labels)), 2)
        if dunn_map.get((labels[i], labels[j]), 1.0) < 0.05
    ]
    for level, (i, j) in enumerate(sig_pairs):
        p_adj = dunn_map[(labels[i], labels[j])]
        sig   = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*"
        y_br  = y_top + level * step
        ax.plot([i, i, j, j], [y_br - tick_h, y_br, y_br, y_br - tick_h],
                color="black", linewidth=0.8, zorder=4)
        ax.text((i + j) / 2, y_br + tick_h * 0.3, sig,
                ha="center", va="bottom", fontsize=7, zorder=5)

    ax.set_ylim(y_min_d - y_span * 0.08,
                y_top + len(sig_pairs) * step + y_span * 0.25)

    tick_labels = [_re.sub(r"\s*\(.*?\)", "", lbl).strip() for lbl in labels]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_xlim(-0.7, len(labels) - 1 + 0.7)
    ax.set_ylabel("Daily Weight Change (%/day)")
    ax.set_title("Transition-Day Daily Change — 4 Cohorts")
    ax.grid(False)
    apply_common_plot_style(ax, ticks_in=True, remove_top_right=True,
                            remove_x_margins=False, remove_y_margins=True,
                            draw_zero_dotted_line=False)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig3c_transition_report(
    cohort_values: Dict[str, np.ndarray],
    kw_results:    dict,
    save_path:     Optional[Path] = None,
) -> str:
    """Text report for the 4-cohort transition-day KW analysis.

    Format mirrors day1_transition_kw_all4.txt.
    """
    from datetime import datetime as _dt
    from scipy.stats import t as _td

    def _fp(p: float) -> str:
        if np.isnan(p): return "N/A"
        return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    def _sig(p: float) -> str:
        if np.isnan(p): return "   "
        return "***" if p < 0.001 else " **" if p < 0.01 else "  *" if p < 0.05 else " ns"

    def _eta_lbl(e: float) -> str:
        if np.isnan(e): return "N/A"
        return ("large" if e >= 0.14 else "moderate" if e >= 0.06
                else "small" if e >= 0.01 else "negligible")

    def _ci95(arr: np.ndarray):
        n = len(arr)
        if n < 2: return float("nan"), float("nan")
        se = float(np.std(arr, ddof=1)) / np.sqrt(n)
        tc = float(_td.ppf(0.975, df=n - 1))
        mu = float(np.mean(arr))
        return mu - tc * se, mu + tc * se

    W    = 135
    kw   = kw_results.get("kruskal_wallis", {})
    eta  = kw_results.get("eta2_H", float("nan"))
    pair = kw_results.get("pairwise", [])
    k    = len(kw_results.get("groups", []))

    lines = [
        "=" * W,
        "4-COHORT COMPARISON — DAY-1 / TRANSITION-DAY DAILY CHANGE",
        "=" * W,
        f"Generated  : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Target days (Daily Change extracted per animal):",
        "  2% CA (12 animals) : Day 1  (0-based, nonramp) = first CA-exposure day",
        "  4% CA (Pilot)      : Day 1  (0-based)          = first CA-exposure day",
        "  5-Week Ramp        : Day 8  (1-based, ramp)    = 0% \u2192 1% CA transition",
        "  2-Week Ramp        : Day 4  (1-based, ramp)    = 0% \u2192 1% CA transition",
        "",
        "Omnibus test : Kruskal-Wallis H-test",
        "Omnibus ES   : eta2_H = (H \u2212 k + 1) / (n \u2212 k);"
        "  <0.01 negligible, 0.01\u20130.06 small, 0.06\u20130.14 moderate, >=0.14 large",
        "Post-hoc     : Dunn\u2019s test (pooled KW ranks, Holm-Bonferroni corrected)",
        "Post-hoc ES  : r_rb = z / \u221a(nA+nB);"
        "  |r| <0.1 negligible, 0.1\u20130.3 small, 0.3\u20130.5 medium, >=0.5 large",
        "HL estimate  : Hodges-Lehmann shift (A\u2212B); 95% CI = BCa bootstrap (n=2 000, seed=0)",
        "",
        "Kruskal-Wallis result:",
        f"  H({k - 1}) = {kw.get('statistic', float('nan')):.4f},  "
        f"p = {_fp(kw.get('p_value', float('nan')))}  "
        f"{_sig(kw.get('p_value', float('nan')))}",
        f"  Effect size: eta2_H = {eta:.4f}  ({_eta_lbl(eta)})",
        "",
        "Descriptive statistics  (95% CI = t-based, two-tailed):",
        f"  {'Cohort':<32}  {'n':>4}  {'Mean (%/day)':>13}  {'SEM':>8}  {'95% CI':>26}",
        "  " + "-" * 90,
    ]
    for label, vals in cohort_values.items():
        n = len(vals)
        if n == 0: continue
        mu  = float(np.mean(vals))
        sem = float(stats.sem(vals)) if n > 1 else float("nan")
        lo, hi = _ci95(vals)
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if not np.isnan(lo) else "N/A"
        lines.append(
            f"  {label:<32}  {n:>4}  {mu:>13.4f}  {sem:>8.4f}  {ci_str:>26}"
        )

    lines += [
        "",
        "Pairwise comparisons (Dunn\u2019s test, Holm-Bonferroni corrected):",
        f"  {'Pair':<48}  {'nA':>4}  {'nB':>4}  {'z(Dunn)':>9}  "
        f"{'p_raw':>10}  {'p_adj (Holm)':>13}  {'sig':>4}  "
        f"{'HL_est':>9}  {'95% CI (HL)':>24}  {'r_rb':>6}",
        "  " + "-" * 145,
    ]
    for r in pair:
        cmp    = f"{r['label_a']}  vs  {r['label_b']}"
        hl_str = (f"{r.get('hl_est', float('nan')):>9.4f}"
                  if not np.isnan(r.get("hl_est", float("nan"))) else f"{'N/A':>9}")
        ci_str = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                  if not np.isnan(r.get("ci_lo", float("nan"))) else "N/A")
        lines.append(
            f"  {cmp:<48}  {r['na']:>4}  {r['nb']:>4}  {r['z_stat']:>9.4f}  "
            f"{_fp(r['p_raw']):>10}  {_fp(r['p_adj']):>13}  "
            f"{_sig(r['p_adj']):>4}  {hl_str}  {ci_str:>24}  {r['r_rb']:>6.3f}"
        )
    lines += [
        "",
        "  r_rb : z/\u221a(nA+nB); |r| <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, \u22650.5 large",
        "  HL   : Hodges-Lehmann location shift (A \u2212 B)",
        "  95% CI: BCa bootstrap on HL estimate (n=2 000 resamples, seed=0)",
        "  *p<0.05  **p<0.01  ***p<0.001  ns=not significant",
        "=" * W,
    ]

    report = "\n".join(lines)

    if save_path is not None:
        sp = Path(save_path).with_suffix(".txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(report, encoding="utf-8")
        print(f"  Saved \u2192 {sp.relative_to(_ROOT)}")

    return report


# =============================================================================
# FIGURE 3d HELPERS — 1%→2% CA Transition-Day Ramp Comparison
#
# Compares the Daily Change at the 1%→2% CA transition day between the
# slow (5-week) and fast (2-week) CA ramp cohorts.
#
#   5-Week Ramp : Day 15  (1-based ramp) — first day on 2% CA
#   2-Week Ramp : Day  7  (1-based ramp) — first day on 2% CA
#
# Statistical test: Mann-Whitney U (two-sided), k=1 comparison,
#                   no multiple-comparison correction needed.
# Effect size     : r_rb = 1 − 2U / (nA × nB)   [range −1, +1]
# CI              : Hodges-Lehmann shift ± 95% BCa bootstrap CI
# =============================================================================

def _fig3d_extract_1pct_to_2pct_values() -> Dict[str, np.ndarray]:
    """
    Extract per-animal Daily Change at the 1%→2% CA transition day.

    Returns
    -------
    Dict[str, np.ndarray]
        Ordered dict: cohort label → 1-D array of Daily Change values.
        Cohorts whose data file is missing are silently omitted.
    """
    result: Dict[str, np.ndarray] = {}

    # ── 5-Week Slow Ramp — ramp Day 15 (1%→2% CA) ──────────────────────────
    if MASTER_RAMP.exists():
        cdf = _add_day_col(_load_master_csv(MASTER_RAMP), "ramp")
        vals = (cdf[cdf["Day"] == 15]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["5-Week Ramp"] = vals

    # ── 2-Week Fast Ramp — ramp Day 7 (1%→2% CA) ──────────────────────────
    if MASTER_2WK.exists():
        cdf = _add_day_col(_load_master_csv(MASTER_2WK), "ramp")
        vals = (cdf[cdf["Day"] == 7]
                .dropna(subset=["Daily Change"])
                .groupby("ID")["Daily Change"].mean()
                .dropna().values.astype(float))
        if len(vals) > 0:
            result["2-Week Ramp"] = vals

    return result


def _fig3d_mwu_analysis(
    cohort_values: Dict[str, np.ndarray],
) -> dict:
    """
    Mann-Whitney U test + Hodges-Lehmann shift for the 1%→2% transition.

    Returns dict with keys:
        groups, group_data, mwu (U, p_value, nA, nB), r_rb, hl_est, ci_lo, ci_hi.
    r_rb = 1 − 2U / (nA × nB) consistent with across_cohort _run_rampramp_menu.
    """
    labels = list(cohort_values.keys())
    if len(labels) < 2:
        return {"groups": labels, "group_data": [], "mwu": {},
                "r_rb": float("nan"), "hl_est": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}

    a, b = cohort_values[labels[0]], cohort_values[labels[1]]
    nA, nB = len(a), len(b)
    U, p   = stats.mannwhitneyu(a, b, alternative="two-sided")
    r_rb   = 1.0 - 2.0 * float(U) / (nA * nB)
    hl, ci_lo, ci_hi = _fig2_hl_bca_ci_internal(a, b)

    return {
        "groups":     labels,
        "group_data": [a, b],
        "mwu":        {"U": float(U), "p_value": float(p), "nA": nA, "nB": nB},
        "r_rb":       float(r_rb),
        "hl_est":     float(hl),
        "ci_lo":      float(ci_lo),
        "ci_hi":      float(ci_hi),
    }


def _fig3d_plot_1pct_to_2pct(
    cohort_values: Dict[str, np.ndarray],
    mwu_results:   dict,
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart (mean ± SEM) with individual scatter points for the
    1%→2% transition-day comparison between 5-Wk Ramp and 2-Wk Ramp.

    Y-axis fixed to [−8, 2] to match across_cohort.py _run_rampramp_menu.
    """
    labels   = list(cohort_values.keys())
    colors   = [cohort_color(lbl) for lbl in labels]
    x_pos    = np.array([0.2, 0.8])
    bar_w    = 0.35

    fig, ax = plt.subplots()

    bar_means = [float(np.mean(cohort_values[lbl])) for lbl in labels]
    bar_sems  = [
        float(stats.sem(cohort_values[lbl])) if len(cohort_values[lbl]) > 1 else 0.0
        for lbl in labels
    ]

    for i, lbl in enumerate(labels):
        ax.bar(x_pos[i], bar_means[i], width=bar_w,
               color=colors[i], alpha=0.7, zorder=2)
        ax.errorbar(x_pos[i], bar_means[i], yerr=bar_sems[i],
                    fmt="none", color="black",
                    capsize=4, capthick=0.8, elinewidth=0.8, zorder=3)

    rng = np.random.default_rng(42)
    for i, lbl in enumerate(labels):
        vals   = cohort_values[lbl]
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(x_pos[i] + jitter, vals,
                   color=colors[i], alpha=0.85, s=12, zorder=4, edgecolors="none")

    # Solid baseline
    ax.axhline(0, color="black", linewidth=0.9, zorder=2)

    # Fixed y limits (matching original)
    ylim_bot, ylim_top = -8.0, 2.0
    g_span  = ylim_top - ylim_bot
    tick_h  = 0.04 * g_span
    bkt_y   = ylim_top - tick_h * 3.5

    # Significance bracket
    p_val = mwu_results.get("mwu", {}).get("p_value", float("nan"))
    if not np.isnan(p_val):
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.plot([x_pos[0], x_pos[1]], [bkt_y, bkt_y],
                color="black", lw=0.8, zorder=5)
        ax.plot([x_pos[0], x_pos[0]], [bkt_y - tick_h, bkt_y],
                color="black", lw=0.8, zorder=5)
        ax.plot([x_pos[1], x_pos[1]], [bkt_y - tick_h, bkt_y],
                color="black", lw=0.8, zorder=5)
        ax.text(0.5 * (x_pos[0] + x_pos[1]), bkt_y + tick_h * 0.3, sig,
                ha="center", va="bottom", fontsize=7, zorder=6)

    ax.set_ylim(ylim_bot, ylim_top)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["5-Wk Ramp\n(Day 15)", "2-Wk Ramp\n(Day 7)"])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylabel("Daily Weight Change (%/day)")
    ax.set_title("1%\u21922% Transition Day\nDaily Weight Change")
    ax.grid(False)
    apply_common_plot_style(ax, ticks_in=True, remove_top_right=True,
                            remove_x_margins=False, remove_y_margins=True,
                            draw_zero_dotted_line=False)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _fig3d_transition_report(
    cohort_values: Dict[str, np.ndarray],
    mwu_results:   dict,
    save_path:     Optional[Path] = None,
) -> str:
    """Text report for the 1%→2% ramp transition MWU analysis.

    Format mirrors the RAMP vs 2-WEEK RAMP report from across_cohort.py
    _run_rampramp_menu.
    """
    from datetime import datetime as _dt
    from scipy.stats import t as _td

    def _fp(p: float) -> str:
        if np.isnan(p): return "N/A"
        return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

    def _sig(p: float) -> str:
        if np.isnan(p): return "   "
        return "***" if p < 0.001 else " **" if p < 0.01 else "  *" if p < 0.05 else " ns"

    def _ci95(arr: np.ndarray):
        n = len(arr)
        if n < 2: return float("nan"), float("nan")
        from scipy.stats import t as _td2
        se = float(np.std(arr, ddof=1)) / np.sqrt(n)
        tc = float(_td2.ppf(0.975, df=n - 1))
        mu = float(np.mean(arr))
        return mu - tc * se, mu + tc * se

    labels = mwu_results.get("groups", [])
    mwu    = mwu_results.get("mwu", {})
    W      = 108

    lines = [
        "=" * W,
        "RAMP vs 2-WEEK RAMP \u2014 CA% TRANSITION DAY DAILY CHANGE",
        "=" * W,
        f"Generated  : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cohorts    : {', '.join(labels)}",
        "",
        "Transition days  \u2014  5-Week Ramp : Day 15 (1%\u21922% CA)",
        "                   2-Week Ramp : Day  7 (1%\u21922% CA)",
        "",
        "Statistical test : Mann-Whitney U (two-sided)",
        "Correction       : None (k=1 comparison; no correction needed)",
        "Effect size      : r_rb = 1 \u2212 2U / (nA \u00d7 nB);  range [\u22121, +1], "
        "positive = 5-Wk Ramp > 2-Wk Ramp",
        "CI               : Hodges-Lehmann shift (A\u2212B) \u00b1 95% BCa bootstrap "
        "(n=2 000 resamples, seed=0)",
        "",
        "Descriptive statistics  (95% CI = t-based, two-tailed):",
        f"  {'Cohort':<22}  {'n':>4}  {'Mean (%/day)':>13}  {'SEM':>8}  {'95% CI':>26}",
        "  " + "-" * 78,
    ]
    for lbl, vals in cohort_values.items():
        n  = len(vals)
        mu = float(np.mean(vals))
        se = float(stats.sem(vals)) if n > 1 else float("nan")
        lo, hi = _ci95(vals)
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if not np.isnan(lo) else "N/A"
        lines.append(f"  {lbl:<22}  {n:>4}  {mu:>13.4f}  {se:>8.4f}  {ci_str:>26}")

    nA  = mwu.get("nA", 0); nB = mwu.get("nB", 0)
    U   = mwu.get("U",    float("nan"))
    p   = mwu.get("p_value", float("nan"))
    rrb = mwu_results.get("r_rb",    float("nan"))
    hl  = mwu_results.get("hl_est",  float("nan"))
    clo = mwu_results.get("ci_lo",   float("nan"))
    chi = mwu_results.get("ci_hi",   float("nan"))

    U_str  = f"{U:.1f}"   if not np.isnan(U)   else "N/A"
    rrb_s  = f"{rrb:.3f}" if not np.isnan(rrb) else "N/A"
    hl_s   = f"{hl:.4f}"  if not np.isnan(hl)  else "N/A"
    ci_s   = f"[{clo:.4f}, {chi:.4f}]" if not np.isnan(clo) else "N/A"

    lines += [
        "",
        "Mann-Whitney U result:",
        f"  {'Trans.':<12}  {'n_Ramp':>6}  {'n_2Wk':>6}  {'U':>8}  "
        f"{'p_raw':>10}  {'p_adj':>10}  {'r_rb':>6}  {'HL_est':>9}  {'95% CI (HL)':>22}  sig",
        "  " + "-" * W,
        f"  {'1%->2%':<12}  {nA:>6}  {nB:>6}  {U_str:>8}  "
        f"{_fp(p):>10}  {_fp(p):>10}  {rrb_s:>6}  {hl_s:>9}  {ci_s:>22}  {_sig(p)}",
        "",
        "  r_rb  : 1 \u2212 2U/(nA\u00d7nB); positive = 5-Wk Ramp daily change > 2-Wk Ramp",
        "  HL    : Hodges-Lehmann location shift (5-Wk Ramp \u2212 2-Wk Ramp)",
        "  95% CI: BCa bootstrap on HL estimate (n=2 000 resamples, seed=0)",
        "  *p<0.05  **p<0.01  ***p<0.001  ns=not significant",
        "=" * W,
    ]

    report = "\n".join(lines)

    if save_path is not None:
        sp = Path(save_path).with_suffix(".txt")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(report, encoding="utf-8")
        print(f"  Saved \u2192 {sp.relative_to(_ROOT)}")

    return report


def figure_3() -> None:
    """Daily Weight Change, Behavioral Metrics, and Transition-Day Comparisons — Figure 3.

    SVG files saved in OUT_FIG3:
      fig3a_daily_change_cohorts  — mean ± SEM daily change per day,
                                     0 % CA, 2 % CA (6 animals), 5-Wk Slow Ramp
      fig3b_behavioral_metrics    — No Nest / Anxious / Lethargy % per week,
                                     three panels (1 × 3), same three cohorts
      fig3c_transition_comparison — 0%→1% transition bar chart + KW, 4 cohorts
      fig3d_transition_1pct_2pct  — 1%→2% transition bar chart + MWU,
                                     5-Wk vs 2-Wk Ramp only

    Text reports in OUT_FIG3:
      fig3c_transition_kw_report.txt   — KW omnibus + Dunn’s + HL for 3c
      fig3d_transition_mwu_report.txt  — MWU + HL for 3d

    Day alignment
    -------------
    nonramp (0 % CA, 2 % CA) : Day 0 = baseline (excluded), Day 1 = first treatment day.
    ramp (5-Wk)               : Day 1 = first recording (included for daily-change plots).
    Both cohorts share the same Day 1 / Week 1 origin on the x-axis.
    """
    print("\n" + "=" * 60)
    print("FIGURE 3 — Daily Weight Change")
    print("=" * 60)

    # ── 3a: Daily Change — 0%, 2% (6 animals), 5-wk slow ramp ───────────────
    _cohort_specs_3a: List[Tuple[str, pd.DataFrame, str]] = []
    if MASTER_0PCT.exists():
        _cohort_specs_3a.append(("0% CA",            _load_master_csv(MASTER_0PCT),  "nonramp"))
    if MASTER_2PCT.exists():
        _cohort_specs_3a.append(("2% CA (6 animals)", _load_master_csv(MASTER_2PCT),  "nonramp"))
    if MASTER_RAMP.exists():
        _cohort_specs_3a.append(("5-Week Ramp",       _load_master_csv(MASTER_RAMP),  "ramp"))

    if _cohort_specs_3a:
        print("\n[3a] Daily Change across days — 0%, 2% (6 animals), 5-wk slow ramp ...")
        _fig3a_plot_daily_change_cohorts(
            _cohort_specs_3a,
            save_path=OUT_FIG3 / "fig3a_daily_change_cohorts",
        )
    else:
        print("[3a] SKIPPED — no cohort data found")

    # ── 3b: Behavioral metrics — No Nest, Anxious, Lethargy per week ──────────
    _cohort_specs_3b: List[Tuple[str, pd.DataFrame, str]] = []
    if MASTER_0PCT.exists():
        _cohort_specs_3b.append(("0% CA",            _load_master_csv(MASTER_0PCT),  "nonramp"))
    if MASTER_2PCT.exists():
        _cohort_specs_3b.append(("2% CA (6 animals)", _load_master_csv(MASTER_2PCT),  "nonramp"))
    if MASTER_RAMP.exists():
        _cohort_specs_3b.append(("5-Week Ramp",       _load_master_csv(MASTER_RAMP),  "ramp"))

    if _cohort_specs_3b:
        print("\n[3b] Behavioral metrics (No Nest, Anxious, Lethargy) per week ...")
        _fig3b_plot_behavioral_metrics(
            _cohort_specs_3b,
            save_path=OUT_FIG3 / "fig3b_behavioral_metrics",
        )
    else:
        print("[3b] SKIPPED — no cohort data found")

    # ── 3c: 4-cohort transition-day comparison ──────────────────────────────────
    print("\n[3c] Transition-day Daily Change — 4 cohorts ...")
    _trans_vals = _fig3c_extract_transition_values()
    if len(_trans_vals) >= 2:
        _kw_trans = _fig3c_kw_transition_analysis(_trans_vals)
        _fig3c_plot_transition_comparison(
            _trans_vals, _kw_trans,
            save_path=OUT_FIG3 / "fig3c_transition_comparison",
        )
        _fig3c_transition_report(
            _trans_vals, _kw_trans,
            save_path=OUT_FIG3 / "fig3c_transition_kw_report",
        )
    else:
        print("[3c] SKIPPED — fewer than 2 cohorts available")

    # ── 3d: 1%→2% transition — 5-Wk Ramp vs 2-Wk Ramp ──────────────────────
    print("\n[3d] 1%→2% transition — 5-Wk Ramp (Day 15) vs 2-Wk Ramp (Day 7) ...")
    _trans_1pct_2pct = _fig3d_extract_1pct_to_2pct_values()
    if len(_trans_1pct_2pct) >= 2:
        _mwu_3d = _fig3d_mwu_analysis(_trans_1pct_2pct)
        _fig3d_plot_1pct_to_2pct(
            _trans_1pct_2pct, _mwu_3d,
            save_path=OUT_FIG3 / "fig3d_transition_1pct_2pct",
        )
        _fig3d_transition_report(
            _trans_1pct_2pct, _mwu_3d,
            save_path=OUT_FIG3 / "fig3d_transition_mwu_report",
        )
    else:
        print("[3d] SKIPPED — fewer than 2 ramp cohorts available")

    print("\n[OK] Figure 3 complete.")


# =============================================================================
# LICK DETECTION PIPELINE  (shared infrastructure for Figure 4 and beyond)
#
# These functions are copied verbatim from lick_analysis.py / across_cohort_lick.py
# so that figure_maker.py remains fully self-contained.
#
# Detection pipeline (in order):
#   1. load_capacitive_csv         — load CSV, derive Time_sec column
#   2. get_sensor_columns          — sorted Sensor_N column list
#   3. compute_sensor_KDE          — KDE mode per sensor (with disk cache)
#   4. compute_KDE_normalizations  — abs((value - KDE) / KDE) per sensor
#   5. compute_fixed_thresholds    — fixed 0.01 threshold Series
#   6. detect_events_above_threshold — find_peaks-based lick event detection
# =============================================================================

def load_capacitive_csv(csv_path: Path) -> pd.DataFrame:
    """Load and clean a capacitive CSV file.

    Returns DataFrame with Time_sec column and sensor readings.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Arduino_Timestamp" not in df.columns:
        raise ValueError(f"No Arduino_Timestamp column found in {csv_path}")

    # Clean timestamp data
    df["Arduino_Timestamp"] = pd.to_numeric(df["Arduino_Timestamp"], errors="coerce")
    df = df.dropna(subset=["Arduino_Timestamp"]).copy()
    df["Arduino_Timestamp"] = df["Arduino_Timestamp"].astype("int64")

    # Compute time in seconds
    df["Time_sec"] = df["Arduino_Timestamp"] / 1000.0

    return df


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return list of sensor columns in sorted numeric order."""
    sensor_cols = [c for c in df.columns
                   if c.startswith("Sensor_") and not c.endswith("_deviation")]

    def key(c: str) -> int:
        try:
            return int(c.split("_")[1])
        except (IndexError, ValueError):
            return 0

    sensor_cols.sort(key=key)
    return sensor_cols


def compute_sensor_KDE(
    df: pd.DataFrame,
    sensor_cols: List[str],
    cache_file: Optional[Path] = None,
    verbose: bool = False,
) -> pd.Series:
    """Compute the KDE (Kernel Density Estimation) peak for each sensor column.

    Returns a Series indexed by sensor column names with their KDE peak values.
    The KDE peak represents the most probable value in the distribution.

    Parameters
    ----------
    df          : DataFrame containing sensor data
    sensor_cols : list of sensor column names
    cache_file  : optional path to save/load cached KDE values (speeds up re-runs)
    verbose     : whether to print per-sensor processing info
    """
    # Try to load from cache if available
    if cache_file and Path(cache_file).exists():
        try:
            cached_df  = pd.read_csv(cache_file, index_col=0)
            cached_kdes = cached_df["KDE_Peak"]
            if all(col in cached_kdes.index for col in sensor_cols):
                if verbose:
                    print(f"  \u2713 Loaded KDE values from cache: {Path(cache_file).name}")
                return cached_kdes[sensor_cols]
            else:
                if verbose:
                    print("  \u26a0 Cache incomplete, recomputing KDE values")
        except Exception as e:
            if verbose:
                print(f"  \u26a0 Error loading cache ({e}), recomputing KDE values")

    kdes: dict = {}
    for col in sensor_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 1:
            try:
                kde     = stats.gaussian_kde(series)
                x_eval  = np.linspace(series.min(), series.max(), 1000)
                density = kde(x_eval)
                kdes[col] = float(x_eval[np.argmax(density)])
                if verbose:
                    print(f"  {col}: KDE={kdes[col]:.2f}, mean={series.mean():.2f}, "
                          f"std={series.std():.2f}, min={series.min():.2f}, max={series.max():.2f}")
            except Exception:
                kdes[col] = float(series.mean())
                if verbose:
                    print(f"  {col}: KDE failed, using mean={kdes[col]:.2f}")
        else:
            kdes[col] = float(series.iloc[0]) if len(series) == 1 else None
            if verbose:
                print(f"  {col}: Insufficient data, KDE={kdes[col]}")

    result = pd.Series(kdes)

    if cache_file is not None:
        try:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(cache_file, header=["KDE_Peak"])
            if verbose:
                print(f"  \u2713 Saved KDE values to cache: {Path(cache_file).name}")
        except Exception as e:
            if verbose:
                print(f"  \u26a0 Could not save cache: {e}")

    return result


def compute_KDE_normalizations(
    df: pd.DataFrame,
    sensor_cols: List[str],
    sensor_kdes: pd.Series,
) -> pd.DataFrame:
    """Compute KDE normalization for each sensor: abs((value \u2212 KDE) / KDE).

    For each sensor column, creates a new column with suffix ``_deviation``
    containing the absolute normalised value: abs((capacitance \u2212 KDE) / KDE).

    Returns a copy of the dataframe with the new normalisation columns added.
    """
    df_out = df.copy()
    for col in sensor_cols:
        kde_val = sensor_kdes[col]
        dev_col = f"{col}_deviation"
        if kde_val is not None and kde_val != 0 and col in df.columns:
            sensor_series = pd.to_numeric(df[col], errors="coerce")
            df_out[dev_col] = abs((sensor_series - kde_val) / kde_val)
        else:
            df_out[dev_col] = pd.NA
    return df_out


def compute_fixed_thresholds(
    sensor_cols: List[str],
    fixed_threshold: float = 0.01,
) -> pd.Series:
    """Return a fixed-threshold Series (default 0.01) for every sensor.

    Uses the EXACT lick detection algorithm from lick_detection.py — a fixed
    threshold applied to KDE-normalised deviations (not a dynamic z-score).
    """
    return pd.Series({col: fixed_threshold for col in sensor_cols})


def detect_events_above_threshold(
    df: pd.DataFrame,
    sensor_cols: List[str],
    thresholds: pd.Series,
) -> pd.DataFrame:
    """Detect lick events where KDE-normalised deviation exceeds threshold.

    Uses ``scipy.signal.find_peaks`` for robust peak detection in the discrete
    sampled deviation signal.  For each sensor produces three columns:

      ``{sensor}_event``      bool   — True at detected peak samples
      ``{sensor}_deviation``  float  — KDE-normalised deviation value
      ``{sensor}_derivative`` float  — forward-difference derivative

    Parameters
    ----------
    df          : DataFrame with Time_sec and ``{sensor}_deviation`` columns
    sensor_cols : list of sensor column names (e.g. ``['Sensor_1', ...]``)
    thresholds  : Series of per-sensor threshold values
                  (from ``compute_fixed_thresholds``)
    """
    if not HAS_SCIPY_SIGNAL:
        raise ImportError("scipy.signal.find_peaks is required for lick detection. "
                          "Install scipy: pip install scipy")

    result = pd.DataFrame({"Time_sec": df["Time_sec"]})

    for sensor_col in sensor_cols:
        dev_col   = f"{sensor_col}_deviation"
        event_col = f"{sensor_col}_event"
        deriv_col = f"{sensor_col}_derivative"

        if dev_col not in df.columns:
            result[event_col] = False
            result[dev_col]   = np.nan
            result[deriv_col] = np.nan
            continue

        threshold = thresholds.get(sensor_col)
        if threshold is None or not np.isfinite(threshold):
            result[event_col] = False
            result[dev_col]   = df[dev_col]
            result[deriv_col] = np.nan
            continue

        deviations = pd.to_numeric(df[dev_col], errors="coerce")
        result[dev_col] = deviations

        # Forward-difference derivative
        clean_dev  = deviations.fillna(0).values
        derivative = np.zeros_like(clean_dev)
        derivative[:-1] = np.diff(clean_dev)
        derivative[-1]  = derivative[-2] if len(derivative) > 1 else 0.0
        result[deriv_col] = derivative

        # Peak detection
        peaks, _ = find_peaks(clean_dev, height=threshold, distance=1)
        peak_mask = np.zeros(len(clean_dev), dtype=bool)
        peak_mask[peaks] = True
        result[event_col] = peak_mask

    return result

# =============================================================================
# FIGURE 4 — Lick Detection
#
# Panels:
#   4B — Single sensor KDE-normalised deviation, first 5 seconds, detected events.
#        Sensor_10 (A2R, 2/25/26, 2% CA 6 animals).  Fixed threshold 0.01.
#   4C — Total Licks per animal per week, mean ± SEM.  3 cohorts.
#   4E — % of Licks in First 5 Minutes per week, mean ± SEM.  3 cohorts.
#   4G — Fecal Count per week, mean ± SEM.  3 cohorts.
#
# Cohorts for 4C / 4E / 4G:  0% CA, 2% CA (6 animals), 5-Week Ramp
#
# Data pipeline (4C / 4E / 4G)
# Ported verbatim from across_cohort_lick.py _run_lick_all3_menu
# options 1 (Total Licks), 3 (% First 5 min), 4 (Fecal Count):
#   load_capacitive_csv  → compute_sensor_KDE (disk-cached)
#   → compute_KDE_normalizations → compute_fixed_thresholds (0.01)
#   → detect_events_above_threshold (first 30 min only)
#   → per-animal: Total_Licks, First_5min_Lick_Pct, Fecal_Count
#   → assign sequential Week numbers from sorted unique recording dates
#
# Source CSVs for 4C / 4E / 4G:
#   0% CA        : LICK_MASTER_0PCT, CAP_LOGS_0PCT
#   2% CA        : LICK_MASTER_2PCT, CAP_LOGS_2PCT
#   5-Week Ramp  : LICK_MASTER_RAMP, CAP_LOGS_RAMP
# =============================================================================

def _fig4_generate_lick_descriptive_stats_report(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_report: bool = True,
) -> None:
    """Generate descriptive statistics report for lick analysis DVs across all loaded cohorts.

    For each cohort, computes per-week (or per-recording-date) descriptive statistics:
      - Total_Licks, Fecal_Count: mean, median, SD, variance, 95% CI (t-dist)
      - First_5min_Lick_Pct: mean, median, SD, variance, 95% CI (t-dist)

    Parameters
    ----------
    cohort_dfs : Dict[str, pd.DataFrame]
        Dictionary mapping cohort label to DataFrame with columns:
        ID, Date, Sex, CA%, Cohort, Sensor, Total_Licks, First_5min_Lick_Pct, Fecal_Count, Week
    save_report : bool, optional
        If True, save report to a text file. Default is True.
    """
    from datetime import datetime as _dt
    from scipy.stats import t as _t_dist

    print("\n" + "=" * 80)
    print("LICK ANALYSIS — DESCRIPTIVE STATISTICS")
    print("=" * 80)

    def _ci95(arr):
        """Compute 95% CI using t-distribution."""
        n = len(arr)
        if n < 2:
            return float('nan'), float('nan')
        se = float(np.std(arr, ddof=1)) / np.sqrt(n)
        margin = float(_t_dist.ppf(0.975, df=n - 1)) * se
        mean = float(np.mean(arr))
        return mean - margin, mean + margin

    all_reports = []

    for cohort_label, df in cohort_dfs.items():
        if df.empty:
            print(f"\n[{cohort_label}] SKIPPED — empty DataFrame")
            continue

        print(f"\n[{cohort_label}]")
        df_sorted = df.sort_values(["ID", "Date"]).copy()

        # Group by Week (already assigned by _fig4_load_lick_cohorts)
        weeks = sorted(df_sorted["Week"].dropna().unique())
        if not weeks:
            print(f"  No weeks found")
            continue

        cohort_report_lines = [
            "=" * 80,
            f"LICK DESCRIPTIVE STATISTICS — {cohort_label.upper()}",
            "=" * 80,
            f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Weeks: {len(weeks)} (0–{max(weeks)})",
            "=" * 80,
            "",
        ]

        dv_specs = [
            ("Total_Licks",           "Total Licks",              "count"),
            ("Fecal_Count",           "Fecal Count",              "count"),
            ("First_5min_Lick_Pct",   "% Licks in First 5 Min",   "%"),
        ]

        for dv_col, dv_label, dv_type in dv_specs:
            cohort_report_lines += [
                "\u2500" * 70,
                f"  {dv_label.upper()}  ({dv_type})",
                "\u2500" * 70,
                f"  {'Week':>6}  {'n':>4}  {'Mean':>10}  {'Median':>10}  "
                f"{'SD':>10}  {'Var':>10}  {'95% CI':>25}",
                f"  {'-' * 68}",
            ]

            all_week_vals = []
            for w in weeks:
                week_df = df_sorted[df_sorted["Week"] == w]
                if week_df.empty:
                    cohort_report_lines.append(
                        f"  {w:>6}  {0:>4}  {'N/A':>10}  {'N/A':>10}  "
                        f"{'N/A':>10}  {'N/A':>10}  {'N/A':>25}"
                    )
                    continue

                # Extract per-animal values (one value per animal per week)
                arr = week_df[dv_col].values
                arr = np.asarray(arr, dtype=float)
                arr_valid = arr[~np.isnan(arr)]
                all_week_vals.append(arr_valid)

                if len(arr_valid) == 0:
                    cohort_report_lines.append(
                        f"  {w:>6}  {0:>4}  {'N/A':>10}  {'N/A':>10}  "
                        f"{'N/A':>10}  {'N/A':>10}  {'N/A':>25}"
                    )
                    continue

                n = len(arr_valid)
                mean_v = float(np.mean(arr_valid))
                median_v = float(np.median(arr_valid))
                sd_v = float(np.std(arr_valid, ddof=1)) if n >= 2 else float('nan')
                var_v = float(np.var(arr_valid, ddof=1)) if n >= 2 else float('nan')
                ci_lo, ci_hi = _ci95(arr_valid)
                ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "N/A"

                cohort_report_lines.append(
                    f"  {w:>6}  {n:>4}  {mean_v:>10.3f}  {median_v:>10.3f}  "
                    f"{sd_v:>10.3f}  {var_v:>10.3f}  {ci_str:>25}"
                )

            # Collapsed "All" row — mean per animal across all weeks
            if all_week_vals:
                # Compute per-animal mean across all weeks
                animals_in_cohort = df_sorted["ID"].unique()
                per_animal_means = []
                for animal_id in animals_in_cohort:
                    animal_all_vals = df_sorted[df_sorted["ID"] == animal_id][dv_col].values
                    animal_all_vals = np.asarray(animal_all_vals, dtype=float)
                    animal_valid = animal_all_vals[~np.isnan(animal_all_vals)]
                    if len(animal_valid) > 0:
                        per_animal_means.append(float(np.mean(animal_valid)))
                
                _all_valid = np.asarray(per_animal_means, dtype=float)
                if len(_all_valid) > 0:
                    _an = len(_all_valid)
                    _am = float(np.mean(_all_valid))
                    _amed = float(np.median(_all_valid))
                    _asd = float(np.std(_all_valid, ddof=1)) if _an >= 2 else float('nan')
                    _avar = float(np.var(_all_valid, ddof=1)) if _an >= 2 else float('nan')
                    _aci_lo, _aci_hi = _ci95(_all_valid)
                    _aci_str = f"[{_aci_lo:.3f}, {_aci_hi:.3f}]" if not np.isnan(_aci_lo) else "N/A"

                    cohort_report_lines.append(f"  {'-' * 68}")
                    cohort_report_lines.append(
                        f"  {'All':>6}  {_an:>4}  {_am:>10.3f}  {_amed:>10.3f}  "
                        f"{_asd:>10.3f}  {_avar:>10.3f}  {_aci_str:>25}"
                    )

            cohort_report_lines.append("")

        cohort_report_lines += ["=" * 80, ""]
        all_reports.append("\n".join(cohort_report_lines))

    # ── Save combined report ──────────────────────────────────────────────
    if save_report and all_reports:
        lines = [
            "=" * 80,
            "FIGURE 4 — LICK DESCRIPTIVE STATISTICS REPORT",
            "=" * 80,
            f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
        ]
        lines.extend("\n".join(all_reports).split("\n"))
        lines += ["=" * 80, "END OF REPORT", "=" * 80, ""]

        rpt_path = OUT_FIG4 / f"fig4_lick_descriptive_stats.txt"
        try:
            rpt_path.parent.mkdir(parents=True, exist_ok=True)
            rpt_path.write_text("\n".join(lines), encoding='utf-8')
            print(f"\n[Saved] Descriptive stats report → {rpt_path.name}")
        except Exception as _e:
            print(f"\n[WARNING] Could not save report: {_e}")


def _fig4_process_cohort_cap_files(
    master_csv: Path,
    capacitive_files: List[Path],
    ca_percent: float,
    cohort_label: str,
    fixed_threshold: float = 0.01,
) -> pd.DataFrame:
    """Process all capacitive log files for one cohort; return per-animal records.

    Ported verbatim from across_cohort_lick.py process_cohort_capacitive_files,
    trimmed to only the columns needed for Figures 4C, 4E, 4G.

    Each capacitive log filename encodes the recording date in the form
    ``capacitive_log_YYYY-M-D.csv``.  That date is matched against the lick
    master CSV to identify which animal was attached to which sensor.  One row
    per animal per session is returned containing:
        ID, Date, Sex, CA%, Cohort, Sensor, Total_Licks,
        First_5min_Lick_Pct, Fecal_Count
    """
    master_df = pd.read_csv(master_csv)
    master_df.columns = master_df.columns.str.strip().str.lower()
    has_sex = "sex" in master_df.columns
    all_records: List[dict] = []

    for cap_file in sorted(capacitive_files):
        if not cap_file.exists():
            continue

        # ── Date extraction from filename ──────────────────────────────────
        date_str: Optional[str] = None
        stem = cap_file.stem
        if "capacitive_log_" in stem:
            date_part = stem.split("capacitive_log_")[1]
            try:
                yr, mo, dy = [int(x) for x in date_part.split("-")]
                date_str = f"{mo}/{dy}/{str(yr)[2:]}"
            except Exception:
                continue
        if date_str is None:
            continue

        # ── Lick detection pipeline ─────────────────────────────────────────
        try:
            cap_df          = load_capacitive_csv(cap_file)
            all_sensors     = get_sensor_columns(cap_df)
            _cache          = cap_file.parent / "kde_cache" / (cap_file.stem + "_kde_cache.csv")
            sensor_kdes     = compute_sensor_KDE(cap_df, all_sensors, cache_file=_cache, verbose=False)
            cap_df          = compute_KDE_normalizations(cap_df, all_sensors, sensor_kdes)
            thresh          = compute_fixed_thresholds(all_sensors, fixed_threshold)
            events_df       = detect_events_above_threshold(cap_df, all_sensors, thresh)
            # Restrict to first 30 minutes — exact behaviour from source
            events_df       = events_df[events_df["Time_sec"] < 1800].copy()
        except Exception as e:
            print(f"  [WARNING] {cap_file.name}: {e}")
            continue

        # ── Per-sensor lick counts ──────────────────────────────────────────
        s_licks = {}
        for sc in all_sensors:
            ec = f"{sc}_event"
            s_licks[sc] = int(events_df[ec].sum()) if ec in events_df.columns else 0

        # ── Match animals via master CSV ────────────────────────────────────
        master_df["_date_norm"] = master_df["date"].astype(str).str.strip()
        date_rows = master_df[master_df["_date_norm"] == date_str].copy()
        if date_rows.empty:
            continue

        for _, arow in date_rows.iterrows():
            animal_id  = str(arow["animal_id"]).strip()
            sensor_num = int(arow["selected_sensors"])
            sc         = f"Sensor_{sensor_num}"

            n_licks     = s_licks.get(sc, 0)

            ec = f"{sc}_event"
            if ec in events_df.columns and n_licks >= 2:
                f5          = int((events_df[ec] & (events_df["Time_sec"] < 300)).sum())
                first5_pct  = f5 / n_licks * 100.0
            else:
                first5_pct  = np.nan          # < 2 licks: exclude from frontloading mean

            sex = str(arow.get("sex", "Unknown")).strip().upper() if has_sex else "Unknown"
            sex = sex if sex.startswith(("M", "F")) else "Unknown"

            all_records.append({
                "ID":                  animal_id,
                "Date":                date_str,
                "Sex":                 sex,
                "CA%":                 ca_percent,
                "Cohort":              cohort_label,
                "Sensor":              sensor_num,
                "Total_Licks":         n_licks,
                "First_5min_Lick_Pct": first5_pct,
                "Fecal_Count":         pd.to_numeric(
                    arow.get("fecal_count", np.nan), errors="coerce"),
            })

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def _fig4_load_lick_cohorts(
    cohort_specs: List[Tuple[str, Path, List[Path], float]],
) -> Dict[str, pd.DataFrame]:
    """Load and process lick data for each cohort.

    Ported from across_cohort_lick.py load_lick_cohorts.
    Assigns 0-based sequential Week numbers from sorted unique recording dates.

    Parameters
    ----------
    cohort_specs : list of (label, master_csv_path, cap_log_list, ca_percent)
    """
    cohort_dfs: Dict[str, pd.DataFrame] = {}
    for label, master_csv, cap_logs, ca_pct in cohort_specs:
        if not master_csv.exists():
            print(f"  [SKIP] {label}: master CSV not found"); continue
        avail = [p for p in cap_logs if p.exists()]
        if not avail:
            print(f"  [SKIP] {label}: no capacitive logs found"); continue
        print(f"  Loading {label} ({len(avail)} logs) ...")
        df = _fig4_process_cohort_cap_files(master_csv, avail, ca_pct, label)
        if df.empty:
            print(f"  [WARN] {label}: no records"); continue
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
        df = df.sort_values(["ID", "Date"]).reset_index(drop=True)
        unique_dates = sorted(df["Date"].dropna().unique())
        df["Week"] = df["Date"].map({d: i for i, d in enumerate(unique_dates)})
        cohort_dfs[label] = df
        print(f"    → {df['ID'].nunique()} animals, {df['Week'].nunique()} weeks")
    return cohort_dfs

def _fig4b_plot_sensor_deviation_with_events(
    cap_log_path: Path,
    sensor_col: str,
    fixed_threshold: float = 0.01,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Figure 4B: Single sensor KDE-normalised deviation + detected lick events.

    Ported verbatim from lick_detection.py plot_single_sensor_deviation_with_events.
    Runs the full lick detection pipeline, then plots the first 5 seconds.
    """
    cap_df          = load_capacitive_csv(cap_log_path)
    all_sensors     = get_sensor_columns(cap_df)
    _cache          = cap_log_path.parent / "kde_cache" / (cap_log_path.stem + "_kde_cache.csv")
    sensor_kdes     = compute_sensor_KDE(cap_df, all_sensors, cache_file=_cache, verbose=False)
    cap_df          = compute_KDE_normalizations(cap_df, all_sensors, sensor_kdes)
    thresh          = compute_fixed_thresholds([sensor_col], fixed_threshold)
    events_df       = detect_events_above_threshold(cap_df, [sensor_col], thresh)

    dev_col   = f"{sensor_col}_deviation"
    event_col = f"{sensor_col}_event"
    t_min, t_max = 0.0, 5.0
    mask   = (events_df["Time_sec"] >= t_min) & (events_df["Time_sec"] <= t_max)
    df_win = events_df.loc[mask]

    fig, ax = plt.subplots()
    ax.plot(df_win["Time_sec"], df_win[dev_col],
            color="#747575", linewidth=0.8, label="Capacitance")
    if event_col in df_win.columns:
        ev = df_win[event_col].astype(bool)
        ax.scatter(df_win.loc[ev, "Time_sec"], df_win.loc[ev, dev_col],
                   color="#eb0d8c", s=15, zorder=5, label="Detected events")
    ax.axhline(y=fixed_threshold, color="#2278b5", linestyle="-", linewidth=1.0,
               label="Threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Capacitance (a.u.)")
    ax.set_title(f"{sensor_col} \u2014 2% CA (A2R, 2/25/26)")
    ax.legend(loc="best", frameon=False)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0.00, 0.03)
    ax.margins(x=0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", which="both", length=4)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _fig4c_plot_total_licks(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Figure 4C: Total Licks per animal per week, mean ± SEM, 3 cohorts.

    Ported from across_cohort_lick.py plot_lick_measure_by_cohort
    (option 1 of _run_lick_all3_menu, measure='Total_Licks', group_by_sex=False).
    """
    frames   = [df.copy() for df in cohort_dfs.values()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fig, ax = plt.subplots()
    if not combined.empty and "Total_Licks" in combined.columns:
        for lbl in sorted(cohort_dfs.keys()):
            grp  = combined[combined["Cohort"] == lbl]
            if grp.empty: continue
            wks  = grp.groupby("Week")["Total_Licks"].agg(["mean", "sem", "count"]).reset_index()
            n_pw = int(wks["count"].iloc[0]) if len(wks) > 0 else 0
            ax.errorbar(wks["Week"], wks["mean"], yerr=wks["sem"],
                        label=f"{lbl} (n={n_pw}/week)",
                        marker="o", linewidth=0.9, capsize=5,
                        color=cohort_color(lbl))

    weeks = sorted(combined["Week"].dropna().unique()) if not combined.empty else []
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(int(w) + 1) for w in weeks])
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Licks per Animal")
    ax.set_title("Total Licks per Animal")
    ax.set_ylim(bottom=0, top=2500)
    ax.set_yticks(range(0, 2501, 500))
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="in", which="both", length=4)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _fig4e_plot_first5min_pct(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Figure 4E: % of Licks in First 5 Minutes per week, mean ± SEM, 3 cohorts.

    Ported from _run_lick_all3_menu option 3 (First_5min_Lick_Pct column).
    Animals with < 2 licks are excluded from the mean (NaN in First_5min_Lick_Pct).
    """
    frames   = [df.copy() for df in cohort_dfs.values()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fig, ax = plt.subplots()
    if not combined.empty and "First_5min_Lick_Pct" in combined.columns:
        for lbl in sorted(cohort_dfs.keys()):
            grp  = combined[combined["Cohort"] == lbl]
            if grp.empty: continue
            wks  = (grp.groupby("Week")["First_5min_Lick_Pct"]
                    .agg(["mean", "sem", "count"]).reset_index())
            n_pw = int(wks["count"].iloc[0]) if len(wks) > 0 else 0
            ax.errorbar(wks["Week"], wks["mean"], yerr=wks["sem"],
                        label=f"{lbl} (n={n_pw}/week)",
                        marker="o", linewidth=0.9, capsize=5,
                        color=cohort_color(lbl))

    weeks = sorted(combined["Week"].dropna().unique()) if not combined.empty else []
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(int(w) + 1) for w in weeks])
    ax.set_xlabel("Week")
    ax.set_ylabel("% Licks in First 5 min")
    ax.set_title("% Licks in First 5 min")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="in", which="both", length=4)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _fig4g_plot_fecal_count(
    cohort_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Figure 4G: Fecal Count per week, mean ± SEM, 3 cohorts.

    Ported from across_cohort_lick.py plot_fecal_counts_by_week
    (option 4 of _run_lick_all3_menu, group_by_cohort=True).
    Groups by Cohort label (not CA%) so the ramp cohort plots as a single line.
    """
    frames   = [df.copy() for df in cohort_dfs.values()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fig, ax = plt.subplots()
    if not combined.empty and "Fecal_Count" in combined.columns:
        for lbl in sorted(cohort_dfs.keys()):
            grp  = combined[combined["Cohort"] == lbl]
            if grp.empty: continue
            wks  = (grp.groupby("Week")["Fecal_Count"]
                    .agg(["mean", "sem", "count"]).reset_index())
            n_pw = int(wks["count"].iloc[0]) if len(wks) > 0 else 0
            ax.errorbar(wks["Week"], wks["mean"], yerr=wks["sem"],
                        label=f"{lbl} (n={n_pw}/week)",
                        marker="o", linewidth=0.9, capsize=5,
                        color=cohort_color(lbl))

    weeks = sorted(combined["Week"].dropna().unique()) if not combined.empty else []
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(int(w) + 1) for w in weeks])
    ax.set_xlabel("Week")
    ax.set_ylabel("Fecal Count")
    ax.set_title("Fecal Count")
    ax.set_ylim(bottom=0, top=10)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="in", which="both", length=4)
    fig.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    elif SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return fig


def figure_4() -> None:
    """Lick Detection — Figure 4.

    SVG files saved in OUT_FIG4:
      fig4b_sensor10_deviation_events — deviation + detected events (Sensor_10, A2R, 2/25/26)
      fig4c_total_licks                — Total Licks per animal, 3 cohorts × week
      fig4e_first5min_lick_pct         — % Licks in First 5 min, 3 cohorts × week
      fig4g_fecal_count                — Fecal Count, 3 cohorts × week

    Cohorts for 4C/4E/4G: 0% CA, 2% CA (6 animals), 5-Week Ramp.
    Processing is the exact pipeline from across_cohort_lick.py _run_lick_all3_menu.
    """
    print("\n" + "=" * 60)
    print("FIGURE 4 \u2014 Lick Detection")
    print("=" * 60)

    # ── 4b: single sensor deviation trace ─────────────────────────────────
    _cap_log = DIR_2PCT / "capacitive_log_2026-2-25.csv"
    if _cap_log.exists():
        print("\n[4b] Sensor_10 deviation + events (A2R, 2/25/26, 2% 6 animals) ...")
        _fig4b_plot_sensor_deviation_with_events(
            cap_log_path=_cap_log, sensor_col="Sensor_10",
            save_path=OUT_FIG4 / "fig4b_sensor10_deviation_events",
        )
    else:
        print(f"[4b] SKIPPED \u2014 not found: {_cap_log}")

    # ── 4c / 4e / 4g: load all three lick cohorts ─────────────────────────
    print("\n[4c/4e/4g] Loading lick cohorts ...")
    _lick_specs: List[Tuple[str, Path, List[Path], float]] = [
        ("0% CA",             LICK_MASTER_0PCT,  CAP_LOGS_0PCT,  0.0),
        ("2% CA (6 animals)", LICK_MASTER_2PCT,  CAP_LOGS_2PCT,  2.0),
        ("5-Week Ramp",       LICK_MASTER_RAMP,  CAP_LOGS_RAMP,  0.0),
    ]
    _cohort_dfs = _fig4_load_lick_cohorts(_lick_specs)

    if _cohort_dfs:
        print(f"\n[4c] Total Licks per week ...")
        _fig4c_plot_total_licks(
            _cohort_dfs, save_path=OUT_FIG4 / "fig4c_total_licks")

        print(f"\n[4e] % Licks in First 5 min per week ...")
        _fig4e_plot_first5min_pct(
            _cohort_dfs, save_path=OUT_FIG4 / "fig4e_first5min_lick_pct")

        print(f"\n[4g] Fecal Count per week ...")
        _fig4g_plot_fecal_count(
            _cohort_dfs, save_path=OUT_FIG4 / "fig4g_fecal_count")

        # ── Generate descriptive statistics report ──────────────────────
        print(f"\n[4 Stats] Generating descriptive statistics report ...")
        _fig4_generate_lick_descriptive_stats_report(_cohort_dfs, save_report=True)
    else:
        print("[4c/4e/4g] SKIPPED \u2014 no cohort data loaded")

    print("\n[OK] Figure 4 complete.")


# =============================================================================
# FIGURE 5 — (add description when porting)
# =============================================================================

def figure_5() -> None:
    """Placeholder — port the Figure 5 panels here."""
    pass


# =============================================================================
# FIGURE 6 — (add description when porting)
# =============================================================================

def figure_6() -> None:
    """Placeholder — port the Figure 6 panels here."""
    pass


# =============================================================================
# FIGURE 7 — (add description when porting)
# =============================================================================

def figure_7() -> None:
    """Placeholder — port the Figure 7 panels here."""
    pass


# =============================================================================
# FIGURE 8 — (add description when porting)
# =============================================================================

def figure_8() -> None:
    """Placeholder — port the Figure 8 panels here."""
    pass


# =============================================================================
# FIGURE 9 — (add description when porting)
# =============================================================================

def figure_9() -> None:
    """Placeholder — port the Figure 9 panels here."""
    pass


# =============================================================================
# EXTENDED DATA 1 — (add description when porting)
# =============================================================================

def extended_data_1() -> None:
    """Placeholder — port the Extended Data 1 panels here."""
    pass


# =============================================================================
# EXTENDED DATA 2–3 — (add description when porting)
# =============================================================================

def extended_data_2_3() -> None:
    """Placeholder — port the Extended Data 2–3 panels here."""
    pass


# =============================================================================
# EXTENDED DATA 4–5 — (add description when porting)
# =============================================================================

def extended_data_4_5() -> None:
    """Placeholder — port the Extended Data 4–5 panels here."""
    pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    figure_2()
    figure_3()
    figure_4()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
    extended_data_1()
    extended_data_2_3()
    extended_data_4_5()
