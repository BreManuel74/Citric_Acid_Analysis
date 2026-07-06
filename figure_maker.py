"""
figure_maker.py — Publication Figure Generation Script

Standalone script for producing all paper figures and extended-data panels.
All data paths are hardcoded in the PATH CONFIGURATION section below so that
any figure can be reproduced by running this file directly.

This file is intentionally self-contained: all analysis and plotting functions
used for each figure are copied directly into the relevant figure section below.
Readers can view the exact code that produced each figure without consulting
any other file.

Usage
-----
  python figure_maker.py

  Figures are saved as SVG (and optionally PNG) to the OUTPUT directories
  configured below.  Set SHOW_PLOTS = True to also display them interactively.
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
OUT_FIG1          = _ROOT / "Figure_1_stats_reports"
OUT_FIG2          = _ROOT / "Figure_2_stats_reports"
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
for _out in [OUT_FIG1, OUT_FIG2, OUT_FIG3, OUT_FIG4, OUT_FIG5,
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
    *,
    also_png: bool = False,
    dpi: int = 300,
) -> None:
    """Save *fig* to *path* as SVG.  Optionally also save a PNG side-by-side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = path.with_suffix(".svg")
    fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
    print(f"  Saved → {svg_path.relative_to(_ROOT)}")
    if also_png:
        png_path = path.with_suffix(".png")
        fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {png_path.relative_to(_ROOT)}")
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
# FIGURE 2 — (add description when porting)
# =============================================================================

def figure_2() -> None:
    """Placeholder — port the Figure 2 panels here."""
    pass


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
