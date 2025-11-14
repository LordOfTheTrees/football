"""Test Step 6 imports"""
try:
    from qb_research.preprocessing.feature_engineering import (
        validate_payment_data,
        label_seasons_relative_to_payment,
        create_lookback_performance_features,
        create_decision_point_dataset,
        prepare_qb_payment_data
    )
    print("✓ All 5 preprocessing functions imported successfully!")
    print(f"  - validate_payment_data: {validate_payment_data}")
    print(f"  - label_seasons_relative_to_payment: {label_seasons_relative_to_payment}")
    print(f"  - create_lookback_performance_features: {create_lookback_performance_features}")
    print(f"  - create_decision_point_dataset: {create_decision_point_dataset}")
    print(f"  - prepare_qb_payment_data: {prepare_qb_payment_data}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

