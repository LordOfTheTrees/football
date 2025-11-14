"""Test Step 9 imports"""
try:
    from qb_research.data.builders import (
        create_all_seasons_from_existing_qb_files,
        create_player_ids_from_qb_data,
        create_train_test_split
    )
    print("✓ All 3 data builder functions imported successfully!")
    print(f"  - create_all_seasons_from_existing_qb_files: {create_all_seasons_from_existing_qb_files}")
    print(f"  - create_player_ids_from_qb_data: {create_player_ids_from_qb_data}")
    print(f"  - create_train_test_split: {create_train_test_split}")
    
    # Test backward compatibility
    from QB_research import (
        create_all_seasons_from_existing_qb_files as qb_all_seasons,
        create_player_ids_from_qb_data as qb_player_ids,
        create_train_test_split as qb_split
    )
    print("\n✓ Backward compatibility verified - functions accessible from QB_research")
    
    print("\n✓ Step 9 complete: All 3 data builder functions imported successfully!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

