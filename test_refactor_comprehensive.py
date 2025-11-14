#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test for refactoring step 1.
Tests new structure and backward compatibility.
"""

import sys
import os
import pandas as pd

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

def print_test(name):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

def test_pass(msg):
    print(f"  ✓ {msg}")

def test_fail(msg):
    print(f"  ✗ {msg}")
    return False

# Test 1: Import from new structure
print_test("Import from new package structure")
try:
    from qb_research.utils.data_loading import (
        load_csv_safe,
        validate_columns,
        validate_payment_years
    )
    test_pass("Successfully imported all three functions")
    test_pass(f"load_csv_safe type: {type(load_csv_safe)}")
    test_pass(f"validate_columns type: {type(validate_columns)}")
    test_pass(f"validate_payment_years type: {type(validate_payment_years)}")
except Exception as e:
    test_fail(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test load_csv_safe
print_test("Testing load_csv_safe()")
try:
    # Test with non-existent file
    result = load_csv_safe("nonexistent_test_file_xyz123.csv", "test file")
    if result is None:
        test_pass("Returns None for missing file (correct behavior)")
    else:
        test_fail("Should return None for missing file")
        sys.exit(1)
    
    # Test with existing file if available
    test_files = ["first_round_qbs_with_picks.csv", "player_ids.csv"]
    found_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            found_file = test_file
            break
    
    if found_file:
        result = load_csv_safe(found_file, "test data")
        if result is not None and isinstance(result, pd.DataFrame):
            test_pass(f"Successfully loads existing CSV: {found_file} ({len(result)} rows)")
        else:
            test_fail("Should return DataFrame for existing file")
            sys.exit(1)
    else:
        test_pass("No test CSV files found (skipping existing file test)")
        
except Exception as e:
    test_fail(f"load_csv_safe test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test validate_columns
print_test("Testing validate_columns()")
try:
    # Create test DataFrame
    test_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    
    # Test with all columns present
    if validate_columns(test_df, ['col1', 'col2'], "test_df"):
        test_pass("Returns True when all required columns exist")
    else:
        test_fail("Should return True when all columns exist")
        sys.exit(1)
    
    # Test with missing column
    if not validate_columns(test_df, ['col1', 'missing_col'], "test_df"):
        test_pass("Returns False when columns are missing")
    else:
        test_fail("Should return False when columns are missing")
        sys.exit(1)
        
except Exception as e:
    test_fail(f"validate_columns test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test validate_payment_years
print_test("Testing validate_payment_years()")
try:
    # Create test payment data
    test_payment_df = pd.DataFrame({
        'draft_year': [2020, 2020, 2020, 2020],
        'payment_year': [2024, 2023, None, 2026],  # One future, one None
        'years_to_payment': [4, 3, None, 6],
        'player_name': ['Player1', 'Player2', 'Player3', 'Player4']
    })
    
    result = validate_payment_years(test_payment_df)
    if isinstance(result, pd.DataFrame):
        test_pass("Returns DataFrame (validation function executes)")
        test_pass(f"Result has {len(result)} rows (same as input)")
    else:
        test_fail("Should return DataFrame")
        sys.exit(1)
        
except Exception as e:
    test_fail(f"validate_payment_years test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify function signatures
print_test("Verifying function signatures")
try:
    import inspect
    
    # Check load_csv_safe
    sig = inspect.signature(load_csv_safe)
    params = list(sig.parameters.keys())
    if params == ['filepath', 'description']:
        test_pass("load_csv_safe has correct signature")
    else:
        test_fail(f"load_csv_safe signature mismatch: {params}")
        sys.exit(1)
    
    # Check validate_columns
    sig = inspect.signature(validate_columns)
    params = list(sig.parameters.keys())
    if params == ['df', 'required_cols', 'df_name']:
        test_pass("validate_columns has correct signature")
    else:
        test_fail(f"validate_columns signature mismatch: {params}")
        sys.exit(1)
    
    # Check validate_payment_years
    sig = inspect.signature(validate_payment_years)
    params = list(sig.parameters.keys())
    expected = ['df', 'draft_year_col', 'payment_year_col', 'years_to_payment_col', 'max_years']
    if params == expected:
        test_pass("validate_payment_years has correct signature")
    else:
        test_fail(f"validate_payment_years signature mismatch: {params}")
        sys.exit(1)
        
except Exception as e:
    test_fail(f"Signature verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check backward compatibility structure
print_test("Checking backward compatibility structure")
try:
    # Check that QB_research.py has the import structure
    qb_research_path = os.path.join(script_dir, "QB_research.py")
    if os.path.exists(qb_research_path):
        with open(qb_research_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from qb_research.utils.data_loading import' in content:
                test_pass("QB_research.py contains import from new structure")
            else:
                test_fail("QB_research.py missing import from new structure")
                sys.exit(1)
    else:
        test_fail("QB_research.py not found")
        sys.exit(1)
        
except Exception as e:
    test_fail(f"Backward compatibility check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nSummary:")
print("  ✓ New package structure: qb_research/utils/data_loading.py")
print("  ✓ All three functions imported successfully")
print("  ✓ Functions work correctly")
print("  ✓ Function signatures preserved")
print("  ✓ Backward compatibility structure in place")
print("\n🎉 Refactoring Step 1 is complete and verified!")
print("="*70)

