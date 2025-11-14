"""Test Step 7 imports"""
try:
    from qb_research.analysis.statistical_analysis import (
        analyze_qb_stat_correlations_with_wins,
        pca_factor_analysis_qb_stats,
        regression_with_pc1_factors
    )
    print("✓ All 3 statistical analysis functions imported successfully!")
    
    from qb_research.analysis.era_adjustment import (
        calculate_era_adjustment_factors,
        apply_era_adjustments,
        create_era_adjusted_payment_data
    )
    print("✓ All 3 era adjustment functions imported successfully!")
    
    print("\n✓ Step 7 complete: All 6 analysis functions imported successfully!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

