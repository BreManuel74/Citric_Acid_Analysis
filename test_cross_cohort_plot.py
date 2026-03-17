"""
Quick test script to plot licks across weeks for both cohorts.

This script loads both 0% and 2% CA cohorts and creates a plot showing
licks across weeks with both cohorts as separate lines (matching lick_nonramp.py style).
"""

from pathlib import Path
from across_cohort_lick import load_lick_cohorts, plot_lick_measure_by_cohort

# Define cohort specifications
base_dir = Path(__file__).parent
data_dir = base_dir.parent  # Data folders are one level up

cohort_specs = {
    "0% CA": {
        "master_csv": data_dir / "0%_files" / "0%_lick_data.csv",
        "capacitive_logs": [
            data_dir / "0%_files" / "capacitive_log_2026-1-21.csv",
            data_dir / "0%_files" / "capacitive_log_2026-1-28.csv",
            data_dir / "0%_files" / "capacitive_log_2026-2-4.csv",
            data_dir / "0%_files" / "capacitive_log_2026-2-11.csv",
            data_dir / "0%_files" / "capacitive_log_2026-2-18.csv"
        ],
        "ca_percent": 0.0,
        "fixed_threshold": 0.01,  # EXACT algorithm from lick_detection.py
        "ili_cutoff": 0.3
    },
    "2% CA (6 animals)": {
        "master_csv": data_dir / "2%_6_animals_files" / "2%_lick_data.csv",
        "capacitive_logs": [
            data_dir / "2%_6_animals_files" / "capacitive_log_2026-1-28.csv",
            data_dir / "2%_6_animals_files" / "capacitive_log_2026-2-4.csv",
            data_dir / "2%_6_animals_files" / "capacitive_log_2026-2-11.csv",
            data_dir / "2%_6_animals_files" / "capacitive_log_2026-2-18.csv",
            data_dir / "2%_6_animals_files" / "capacitive_log_2026-2-25.csv"
        ],
        "ca_percent": 2.0,
        "fixed_threshold": 0.01,  # EXACT algorithm from lick_detection.py
        "ili_cutoff": 0.3
    }
}

print("="*80)
print("TESTING CROSS-COHORT LICK PLOTTING")
print("="*80)
print()

# Load cohorts
print("Loading cohorts...")
cohorts = load_lick_cohorts(cohort_specs)

print()
print("="*80)
print("CREATING PLOTS")
print("="*80)
print()

# Plot Total Licks (with SEM)
print("Plotting Total Licks across weeks...")
fig1 = plot_lick_measure_by_cohort(
    cohorts, 
    measure="Total_Licks", 
    use_std=False,  # Use SEM
    save_path=data_dir / "cross_cohort_total_licks.svg",
    show=True
)

# Plot Total Bouts
print("\nPlotting Total Bouts across weeks...")
fig2 = plot_lick_measure_by_cohort(
    cohorts, 
    measure="Total_Bouts", 
    use_std=False,
    save_path=data_dir / "cross_cohort_total_bouts.svg",
    show=True
)

# Plot Average ILI
print("\nPlotting Average ILI across weeks...")
fig3 = plot_lick_measure_by_cohort(
    cohorts, 
    measure="Avg_ILI", 
    use_std=False,
    save_path=data_dir / "cross_cohort_avg_ili.svg",
    show=True
)

# Plot Average Bout Duration
print("\nPlotting Average Bout Duration across weeks...")
fig4 = plot_lick_measure_by_cohort(
    cohorts, 
    measure="Avg_Bout_Duration", 
    use_std=False,
    save_path=data_dir / "cross_cohort_avg_bout_duration.svg",
    show=True
)

print()
print("="*80)
print("PLOTTING COMPLETE")
print("="*80)
print()
print("Saved plots:")
print("  1. cross_cohort_total_licks.svg")
print("  2. cross_cohort_total_bouts.svg")
print("  3. cross_cohort_avg_ili.svg")
print("  4. cross_cohort_avg_bout_duration.svg")
