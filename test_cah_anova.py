"""
Test script to demonstrate CAH ANOVA analyses without user input
"""

from pathlib import Path
from CAH_weight_analysis import (
    load_cah_data, 
    clean_cah_dataframe, 
    add_day_number_column,
    perform_two_way_between_anova,
    perform_mixed_anova_time,
    perform_tukey_hsd,
    generate_analysis_report,
    plot_interaction_effects,
    plot_total_change_by_id,
    plot_daily_change_by_id,
    plot_total_change_by_sex,
    plot_daily_change_by_sex,
    plot_total_change_by_ca,
    plot_daily_change_by_ca,
)

def main():
    # Load and prepare data
    csv_path = Path(__file__).parent.parent / "CAH_cohort" / "master_data_CAH.csv"
    
    print("="*80)
    print("CAH COHORT ANOVA ANALYSIS TEST")
    print("="*80)
    
    df_raw = load_cah_data(csv_path)
    df = clean_cah_dataframe(df_raw)
    df = add_day_number_column(df)
    
    print(f"\nData prepared: {len(df)} rows, {df['ID'].nunique()} animals")
    
    # Example 1: Between-subjects ANOVA at final time point
    print("\n\n" + "="*80)
    print("ANALYSIS 1: Between-Subjects ANOVA at Final Time Point (Day 27)")
    print("="*80)
    results_final = perform_two_way_between_anova(
        df, 
        measure="Total Change",
        time_point=27,
        average_over_days=False
    )
    
    # Example 2: Between-subjects ANOVA with averaged data
    print("\n\n" + "="*80)
    print("ANALYSIS 2: Between-Subjects ANOVA with Day-Averaged Data")
    print("="*80)
    results_avg = perform_two_way_between_anova(
        df,
        measure="Total Change",
        average_over_days=True
    )
    
    # Example 3: Mixed ANOVA with Time
    print("\n\n" + "="*80)
    print("ANALYSIS 3: Mixed ANOVA - Time × Sex × CA%")
    print("="*80)
    print("Using all available time points for complete analysis")
    results_mixed = perform_mixed_anova_time(
        df,
        measure="Total Change",
        time_points=None  # Use all available days
    )
    
    # Post-hoc tests if significant
    tukey_results = None
    
    if results_avg and results_avg.get('ca_percent', {}).get('significant'):
        print("\n\nCA% effect is significant. Running Tukey HSD...")
        avg_df = df.groupby(["ID", "Sex", "CA (%)"])["Total Change"].mean().reset_index()
        avg_df["CA (%) Group"] = avg_df["CA (%)"].astype(str) + "%"
        tukey_ca = perform_tukey_hsd(avg_df, "Total Change", "CA (%) Group")
        tukey_results = tukey_ca
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    if results_avg:
        # Get p-value column name
        p_col = 'p-unc' if 'p-unc' in results_avg['anova_table'].columns else 'p_unc'
        print(f"  Sex effect: p = {results_avg['sex']['p']:.4f} {'(significant)' if results_avg['sex']['significant'] else '(ns)'}")
        print(f"  CA% effect: p = {results_avg['ca_percent']['p']:.4f} {'(significant)' if results_avg['ca_percent']['significant'] else '(ns)'}")
        print(f"  Sex × CA% interaction: p = {results_avg['interaction']['p']:.4f} {'(significant)' if results_avg['interaction']['significant'] else '(ns)'}")
    
    # Generate comprehensive report
    print("\n\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL REPORT")
    print("="*80)
    
    report = generate_analysis_report(
        between_results=results_avg,
        mixed_results=results_mixed,
        tukey_results=tukey_results,
        df=df
    )
    
    # Don't print report to console (contains Unicode that may not display in Windows console)
    # Just save to file
    print("\n[Report generated - saving to file...]")
    
    # Save report to file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"CAH_statistical_report_{timestamp}.txt"
    report_path = Path(__file__).parent / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] Report saved to: {report_path}")
    
    # Generate interaction plots
    print("\n\n" + "="*80)
    print("GENERATING INTERACTION PLOTS")
    print("="*80)
    
    interaction_figs = plot_interaction_effects(
        between_results=results_avg,
        mixed_results=results_mixed,
        df=df,
        save_dir=Path(__file__).parent,
        show=False  # Don't show plots, just save them
    )
    
    if interaction_figs:
        print(f"\n[OK] Generated {len(interaction_figs)} interaction plot(s)")
    
    # Generate individual animal plots
    print("\n\n" + "="*80)
    print("GENERATING INDIVIDUAL ANIMAL PLOTS")
    print("="*80)
    
    print("\nPlotting Total Change by Individual Animal...")
    plot_total_change_by_id(
        df,
        save_svg=True,
        svg_filename="CAH_total_change_by_id.svg",
        show=False
    )
    
    print("\nPlotting Daily Change by Individual Animal...")
    plot_daily_change_by_id(
        df,
        save_svg=True,
        svg_filename="CAH_daily_change_by_id.svg",
        show=False
    )
    
    # Generate sex-averaged plots
    print("\n\n" + "="*80)
    print("GENERATING SEX-AVERAGED PLOTS")
    print("="*80)
    
    print("\nPlotting Total Change Averaged by Sex...")
    plot_total_change_by_sex(
        df,
        save_svg=True,
        svg_filename="CAH_total_change_by_sex.svg",
        show=False
    )
    
    print("\nPlotting Daily Change Averaged by Sex...")
    plot_daily_change_by_sex(
        df,
        save_svg=True,
        svg_filename="CAH_daily_change_by_sex.svg",
        show=False
    )
    
    print("\n" + "="*80)
    print("GENERATING CA%-AVERAGED PLOTS")
    print("="*80)
    
    print("\nPlotting Total Change Averaged by CA%...")
    plot_total_change_by_ca(
        df,
        save_svg=True,
        svg_filename="CAH_total_change_by_ca.svg",
        show=False
    )
    
    print("\nPlotting Daily Change Averaged by CA%...")
    plot_daily_change_by_ca(
        df,
        save_svg=True,
        svg_filename="CAH_daily_change_by_ca.svg",
        show=False
    )
    
    print("\n" + "="*80)
    print("ALL PLOTS COMPLETE - Check directory for SVG files")
    print("="*80)

if __name__ == "__main__":
    main()
