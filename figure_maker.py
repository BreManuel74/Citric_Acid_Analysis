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
OUT_FIG3          = _ROOT / "Figure_3_stats_reports"
OUT_FIG4          = _ROOT / "Figure_4_stats_reports"
OUT_FIG5          = _ROOT / "Figure_5_stats_reports"
OUT_FIG6          = _ROOT / "Figure_6_stats_reports"
OUT_FIG7          = _ROOT / "Figure_7_stats_reports"
OUT_FIG8          = _ROOT / "Figure_8_stats_reports"
OUT_FIG9          = _ROOT / "Figure_9_stats_reports"
OUT_EXT1          = _ROOT / "Ex_data_1_stats_reports"
OUT_EXT2_3        = _ROOT / "Ext_data_2-3_Stats_reports"
OUT_EXT4_5        = _ROOT / "Ext_data_4-5_stats_reports"

# ── Convenience: create all output directories if they don't exist ────────────
for _out in [OUT_FIG2, OUT_FIG3, OUT_FIG4, OUT_FIG5,
             OUT_FIG6, OUT_FIG7, OUT_FIG8, OUT_FIG9,
             OUT_EXT1, OUT_EXT2_3, OUT_EXT4_5]:
    _out.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CANONICAL COHORT COLOURS
# These match across_cohort.py / across_cohort_lick.py exactly so that every
# figure in the paper uses the same palette.
# =============================================================================

COLOR_0PCT  = "#1f77b4"   # 0 % CA  — blue
COLOR_2PCT  = "#f79520"   # 2 % CA  — orange
COLOR_RAMP  = "#2da048"   # slow ramp — green
COLOR_4PCT  = "#424143"   # 4 % CA (pilot) — dark grey
COLOR_OTHER = "#7f3f98"   # fast ramp - purple

# Sex marker styles (cohort colour is applied separately)
#   Males:   filled square  "s"
#   Females: filled circle  "o"
SEX_MARKER: Dict[str, str] = {"M": "s", "F": "o", "Unknown": "^"}


def cohort_color(label: str) -> str:
    """Return the canonical hex colour for a cohort label string."""
    lo = str(label).lower()
    if "0%" in lo:
        return COLOR_0PCT
    if "4%" in lo:
        return COLOR_4PCT
    if "2%" in lo:
        return COLOR_2PCT
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
    cohort_color : hex colour for all animal lines; sex encoded by marker only
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
    Colour: COLOR_4PCT (#424143, dark grey).
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


def figure_2() -> None:
    """Total Weight Change over Days — five separate cohort panels.

    Each panel saved as SVG in OUT_FIG2:
      fig2a_total_change_0pct       — 0 % CA non-ramp         (nonramp)
      fig2b_total_change_2pct_full  — 2 % CA, 12 animals      (nonramp)
      fig2c_total_change_slow_ramp  — 5-wk slow CA ramp        (ramp)
      fig2d_total_change_fast_ramp  — 2-wk fast CA ramp        (ramp)
      fig2e_total_change_4pct_pilot — 4 % CA pilot, Days 1–3  (special)

    No subplot structure — each cohort is its own plot.
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

    print("\n[OK] Figure 2 complete.")


# =============================================================================
# FIGURE 3 — (add description when porting)
# =============================================================================

def figure_3() -> None:
    """Placeholder — port the Figure 3 panels here."""
    pass


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
