"""Test Step 11 imports"""
try:
    from qb_research.comparisons import (
        year_weighting_regression,
        find_most_similar_qbs,
        find_comps_both_metrics,
        extract_year_weights_from_regression_results,
        batch_comp_analysis
    )
    print("OK: All 5 comparison functions imported successfully!")
    print(f"  - year_weighting_regression: {year_weighting_regression}")
    print(f"  - find_most_similar_qbs: {find_most_similar_qbs}")
    print(f"  - find_comps_both_metrics: {find_comps_both_metrics}")
    print(f"  - extract_year_weights_from_regression_results: {extract_year_weights_from_regression_results}")
    print(f"  - batch_comp_analysis: {batch_comp_analysis}")
    
    # Test backward compatibility
    from QB_research import (
        year_weighting_regression as qb_year_weighting,
        find_most_similar_qbs as qb_find_comps,
        find_comps_both_metrics as qb_find_both,
        extract_year_weights_from_regression_results as qb_extract,
        batch_comp_analysis as qb_batch
    )
    print("\nOK: Backward compatibility verified - functions accessible from QB_research")
    
    print("\nOK: Step 11 complete: All 5 comparison functions imported successfully!")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback
    traceback.print_exc()
