"""Test validators import"""
import sys
import traceback

try:
    print("Testing normalize_player_name import...")
    from qb_research.utils.name_matching import normalize_player_name
    print("✓ normalize_player_name imported")
    
    print("\nTesting validators module import...")
    import qb_research.data.validators as validators
    print("✓ validators module imported")
    print(f"Module attributes: {[x for x in dir(validators) if not x.startswith('_')]}")
    
    if hasattr(validators, 'validate_contract_mapping'):
        print("✓ validate_contract_mapping found")
    else:
        print("✗ validate_contract_mapping NOT found")
        
    if hasattr(validators, 'test_name_mapping'):
        print("✓ test_name_mapping found")
    else:
        print("✗ test_name_mapping NOT found")
        
    # Try direct import
    print("\nTesting direct import...")
    from qb_research.data.validators import validate_contract_mapping, test_name_mapping
    print("✓ Direct import successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

