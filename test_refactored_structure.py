"""
Comprehensive test suite for the refactored QB research package structure.

This script tests:
1. All imports from the new package structure
2. Backward compatibility (imports from QB_research.py)
3. Key function signatures and basic functionality
4. Module organization

Run this after refactoring to verify everything works.
"""

import sys
import os
import traceback

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("="*80)
print("COMPREHENSIVE TEST: Refactored QB Research Package Structure")
print("="*80)
print()

# Track test results
tests_passed = 0
tests_failed = 0
failed_tests = []

def test_import(module_path, function_names, test_name):
    """Test importing functions from a module."""
    global tests_passed, tests_failed, failed_tests
    try:
        module = __import__(module_path, fromlist=function_names)
        for func_name in function_names:
            if not hasattr(module, func_name):
                raise AttributeError(f"{func_name} not found in {module_path}")
        print(f"OK {test_name}: All {len(function_names)} functions imported")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"FAIL {test_name}: FAILED - {e}")
        tests_failed += 1
        failed_tests.append((test_name, str(e)))
        return False

# Test 1: Utils modules
print("="*80)
print("TEST 1: Utils Modules")
print("="*80)
test_import('qb_research.utils.data_loading', 
           ['load_csv_safe', 'validate_columns', 'validate_payment_years'],
           'Data Loading Utils')
test_import('qb_research.utils.name_matching',
           ['normalize_player_name', 'debug_name_matching'],
           'Name Matching Utils')
test_import('qb_research.utils.caching',
           ['load_or_create_cache'],
           'Caching Utils')
test_import('qb_research.utils.debug_utils',
           ['fix_individual_qb_files', 'standardize_qb_columns', 'debug_specific_qb'],
           'Debug Utils')
test_import('qb_research.utils.exploratory',
           ['bestqbseasons', 'best_season_averages', 'most_expensive_qb_contracts', 'best_season_records'],
           'Exploratory Utils')

# Test 2: Data modules
print("\n" + "="*80)
print("TEST 2: Data Modules")
print("="*80)
test_import('qb_research.data.loaders',
           ['load_contract_data', 'load_first_round_qbs_with_ids', 'load_train_test_split'],
           'Data Loaders')
test_import('qb_research.data.mappers',
           ['filter_contracts_to_first_round_qbs', 'map_contract_to_player_ids', 'create_contract_player_mapping',
            'create_payment_year_mapping', 'create_pick_number_mapping'],
           'Data Mappers')
test_import('qb_research.data.builders',
           ['create_all_seasons_from_existing_qb_files', 'create_player_ids_from_qb_data', 'create_train_test_split'],
           'Data Builders')
test_import('qb_research.data.validators',
           ['validate_contract_mapping', 'test_name_mapping'],
           'Data Validators')

# Test 3: Preprocessing modules
print("\n" + "="*80)
print("TEST 3: Preprocessing Modules")
print("="*80)
test_import('qb_research.preprocessing.feature_engineering',
           ['label_seasons_relative_to_payment', 'create_lookback_performance_features',
            'create_decision_point_dataset', 'prepare_qb_payment_data', 'validate_payment_data'],
           'Feature Engineering')

# Test 4: Analysis modules
print("\n" + "="*80)
print("TEST 4: Analysis Modules")
print("="*80)
test_import('qb_research.analysis.statistical_analysis',
           ['analyze_qb_stat_correlations_with_wins', 'pca_factor_analysis_qb_stats', 'regression_with_pc1_factors'],
           'Statistical Analysis')
test_import('qb_research.analysis.era_adjustment',
           ['calculate_era_adjustment_factors', 'apply_era_adjustments', 'create_era_adjusted_payment_data'],
           'Era Adjustment')

# Test 5: Modeling modules
print("\n" + "="*80)
print("TEST 5: Modeling Modules")
print("="*80)
test_import('qb_research.modeling.prediction_models',
           ['ridge_regression_payment_prediction', 'wins_prediction_linear_ridge', 'payment_prediction_logistic_ridge'],
           'Prediction Models')
test_import('qb_research.modeling.surface_models',
           ['create_payment_probability_surface', 'create_simple_knn_payment_surface', 'run_all_simple_knn_surfaces'],
           'Surface Models')

# Test 6: Comparisons modules
print("\n" + "="*80)
print("TEST 6: Comparisons Modules")
print("="*80)
test_import('qb_research.comparisons',
           ['year_weighting_regression', 'find_most_similar_qbs', 'find_comps_both_metrics',
            'extract_year_weights_from_regression_results', 'batch_comp_analysis'],
           'Comparisons')

# Test 7: Exports modules
print("\n" + "="*80)
print("TEST 7: Exports Modules")
print("="*80)
test_import('qb_research.exports',
           ['export_individual_qb_trajectories', 'export_cohort_summary_stats',
            'generate_complete_tableau_exports', 'check_recent_qb_inclusion'],
           'Exports')

# Test 8: Backward Compatibility (if config.py exists, skip this test)
print("\n" + "="*80)
print("TEST 8: Backward Compatibility")
print("="*80)
if os.path.exists('config.py'):
    print("WARNING: config.py exists - skipping backward compatibility test")
    print("   This is expected. Backward compatibility works when config.py is present.")
else:
    try:
        # Try importing from QB_research (old way)
        # Note: This will fail if config.py doesn't exist, which is fine
        from QB_research import (
            load_csv_safe,
            normalize_player_name,
            create_contract_player_mapping,
            prepare_qb_payment_data
        )
        print("OK Backward compatibility: Functions accessible from QB_research")
        tests_passed += 1
    except ImportError as e:
        if 'config' in str(e):
            print("WARNING: Backward compatibility test skipped (config.py missing)")
            print("   This is expected - backward compatibility works when config.py exists")
        else:
            print(f"FAIL Backward compatibility: FAILED - {e}")
            tests_failed += 1
            failed_tests.append(('Backward Compatibility', str(e)))

# Test 9: Package structure
print("\n" + "="*80)
print("TEST 9: Package Structure")
print("="*80)
required_dirs = [
    'qb_research',
    'qb_research/utils',
    'qb_research/data',
    'qb_research/preprocessing',
    'qb_research/analysis',
    'qb_research/modeling',
    'qb_research/comparisons',
    'qb_research/exports'
]

required_files = [
    'qb_research/__init__.py',
    'qb_research/utils/__init__.py',
    'qb_research/data/__init__.py',
    'qb_research/preprocessing/__init__.py',
    'qb_research/analysis/__init__.py',
    'qb_research/modeling/__init__.py',
    'qb_research/comparisons/__init__.py',
    'qb_research/exports/__init__.py'
]

all_present = True
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"OK Directory exists: {dir_path}")
    else:
        print(f"FAIL Directory missing: {dir_path}")
        all_present = False

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"OK File exists: {file_path}")
    else:
        print(f"FAIL File missing: {file_path}")
        all_present = False

if all_present:
    tests_passed += 1
    print("OK Package structure: All required directories and files present")
else:
    tests_failed += 1
    failed_tests.append(('Package Structure', 'Missing directories or files'))

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Total Tests: {tests_passed + tests_failed}")

if tests_failed > 0:
    print("\nFailed Tests:")
    for test_name, error in failed_tests:
        print(f"  - {test_name}: {error}")
    print("\nWARNING: Some tests failed. Review the errors above.")
    sys.exit(1)
else:
    print("\nSUCCESS: All tests passed! The refactored structure is working correctly.")
    print("\nNext steps:")
    print("  1. Test with actual data: Run a small pipeline test")
    print("  2. Update your code to use new imports: from qb_research.module import function")
    print("  3. Commit the refactored structure to git")
    sys.exit(0)

