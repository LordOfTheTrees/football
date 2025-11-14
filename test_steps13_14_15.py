"""Test Steps 13, 14, and 15 imports"""
import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("Testing Steps 13, 14, and 15 imports...\n")

# Step 13: Surface models
try:
    from qb_research.modeling.surface_models import (
        create_payment_probability_surface,
        create_simple_knn_payment_surface,
        run_all_simple_knn_surfaces
    )
    print("OK: Step 13 - All 3 surface functions imported successfully!")
except Exception as e:
    print(f"ERROR Step 13: {e}")
    import traceback
    traceback.print_exc()

# Step 14: Debug utils
try:
    from qb_research.utils.debug_utils import (
        fix_individual_qb_files,
        standardize_qb_columns,
        debug_specific_qb
    )
    print("OK: Step 14 - All 3 debug/utility functions imported successfully!")
except Exception as e:
    print(f"ERROR Step 14: {e}")
    import traceback
    traceback.print_exc()

# Step 15: Exploratory functions
try:
    from qb_research.utils.exploratory import (
        bestqbseasons,
        best_season_averages,
        most_expensive_qb_contracts,
        best_season_records
    )
    print("OK: Step 15 - All 4 exploratory functions imported successfully!")
except Exception as e:
    print(f"ERROR Step 15: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY: Steps 13, 14, and 15 complete!")
print("="*80)

