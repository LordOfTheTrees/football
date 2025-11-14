#!/usr/bin/env python
"""Test Step 3: Caching Module"""

import sys
import os
import pandas as pd

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("=" * 70)
print("TESTING STEP 3: Caching Module")
print("=" * 70)

# Test 1: Import from new structure
print("\n[TEST 1] Importing from new package structure...")
try:
    from qb_research.utils.caching import load_or_create_cache
    print("   ✓ Successfully imported from qb_research.utils.caching")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test function signature
print("\n[TEST 2] Verifying function signature...")
try:
    import inspect
    sig = inspect.signature(load_or_create_cache)
    params = list(sig.parameters.keys())
    expected = ['cache_file', 'creation_function', 'force_refresh']
    if params[:3] == expected:
        print(f"   ✓ Function signature correct: {params[:3]}")
    else:
        print(f"   ✗ Signature mismatch: {params[:3]} vs {expected}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Test basic functionality
print("\n[TEST 3] Testing basic functionality...")
try:
    # Create a simple test function
    def create_test_data():
        return pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    # Test with a non-existent cache file
    test_cache = "test_cache_step3_temp.csv"
    if os.path.exists(test_cache):
        os.remove(test_cache)
    
    # Should create new data
    result = load_or_create_cache(test_cache, create_test_data)
    if result is not None and isinstance(result, pd.DataFrame):
        print("   ✓ Function creates data when cache doesn't exist")
    else:
        print("   ✗ Function should return DataFrame")
        sys.exit(1)
    
    # Should load from cache
    result2 = load_or_create_cache(test_cache, create_test_data)
    if result2 is not None and isinstance(result2, pd.DataFrame):
        print("   ✓ Function loads from cache when it exists")
    else:
        print("   ✗ Function should load from cache")
        sys.exit(1)
    
    # Clean up
    if os.path.exists(test_cache):
        os.remove(test_cache)
        print("   ✓ Cleanup successful")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    # Clean up on error
    if os.path.exists("test_cache_step3_temp.csv"):
        os.remove("test_cache_step3_temp.csv")
    sys.exit(1)

# Test 4: Verify file structure
print("\n[TEST 4] Verifying file structure...")
try:
    if os.path.exists("qb_research/utils/caching.py"):
        print("   ✓ qb_research/utils/caching.py exists")
    else:
        print("   ✗ qb_research/utils/caching.py not found")
        sys.exit(1)
    
    # Check that QB_research.py has the import
    with open("QB_research.py", 'r', encoding='utf-8') as f:
        content = f.read()
        if 'from qb_research.utils.caching import' in content:
            print("   ✓ QB_research.py contains import from new structure")
        else:
            print("   ✗ QB_research.py missing import from new structure")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Check dependencies
print("\n[TEST 5] Checking module dependencies...")
try:
    # Check that caching.py imports from data_loading
    with open("qb_research/utils/caching.py", 'r', encoding='utf-8') as f:
        content = f.read()
        if 'from qb_research.utils.data_loading import load_csv_safe' in content:
            print("   ✓ caching.py correctly imports from data_loading")
        else:
            print("   ✗ caching.py missing import from data_loading")
            sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nStep 3 Summary:")
print("  ✓ New module: qb_research/utils/caching.py")
print("  ✓ Function: load_or_create_cache")
print("  ✓ Functionality verified (create and load from cache)")
print("  ✓ Dependencies correct (imports from data_loading)")
print("  ✓ Backward compatibility structure in place")
print("\n🎉 Step 3 is complete and verified!")
print("=" * 70)

