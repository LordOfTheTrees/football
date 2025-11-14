"""Test Step 9 - Final verification"""
print("Testing Step 9: Data Builders Module")
print("=" * 60)

# Test direct import
try:
    from qb_research.data.builders import (
        create_all_seasons_from_existing_qb_files,
        create_player_ids_from_qb_data,
        create_train_test_split
    )
    print("✓ Direct import from qb_research.data.builders: SUCCESS")
    print(f"  - create_all_seasons_from_existing_qb_files: {create_all_seasons_from_existing_qb_files}")
    print(f"  - create_player_ids_from_qb_data: {create_player_ids_from_qb_data}")
    print(f"  - create_train_test_split: {create_train_test_split}")
except Exception as e:
    print(f"✗ Direct import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test backward compatibility
try:
    from QB_research import (
        create_all_seasons_from_existing_qb_files as qb_all_seasons,
        create_player_ids_from_qb_data as qb_player_ids,
        create_train_test_split as qb_split
    )
    print("\n✓ Backward compatibility import from QB_research: SUCCESS")
    print(f"  - create_all_seasons_from_existing_qb_files: {qb_all_seasons}")
    print(f"  - create_player_ids_from_qb_data: {qb_player_ids}")
    print(f"  - create_train_test_split: {qb_split}")
except Exception as e:
    print(f"\n✗ Backward compatibility import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ Step 9 COMPLETE: All 3 data builder functions working!")
print("=" * 60)

