"""
figure_maker.py — Publication Figure Generation Script

Standalone script for producing all paper figures and extended-data panels.
All data paths are hardcoded in the PATH CONFIGURATION section below so that
any figure can be reproduced by running this file directly.

This file is intentionally self-contained: all analysis and plotting functions
used for each figure are copied directly into the relevant figure section below.
Readers can view the exact code that produced each figure without consulting
any other file.

NOTE: There will be minor formatting differences between the figures produced by this script and the final published figures.  
The final figures were polished in Adobe Illustrator for font consistency, line thickness, 
evenly spaced x and y axis limits, panel alignment, etc.

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
# (no lick master or capacitive logs for 2-wk ramp — add if available)

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
# FIGURE 4 — (add description when porting)
# =============================================================================

def figure_4() -> None:
    """Placeholder — port the Figure 4 panels here."""
    pass


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
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
    extended_data_1()
    extended_data_2_3()
    extended_data_4_5()
