"""Test Step 10 imports"""
try:
    from qb_research.data.validators import (
        validate_contract_mapping,
        test_name_mapping
    )
    print("✓ All 2 validation functions imported successfully!")
    print(f"  - validate_contract_mapping: {validate_contract_mapping}")
    print(f"  - test_name_mapping: {test_name_mapping}")
    
    # Test backward compatibility
    from QB_research import (
        validate_contract_mapping as qb_validate,
        test_name_mapping as qb_test
    )
    print("\n✓ Backward compatibility verified - functions accessible from QB_research")
    
    print("\n✓ Step 10 complete: All 2 validation functions imported successfully!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

