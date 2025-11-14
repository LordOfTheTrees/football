"""Test Step 8 imports"""
try:
    from qb_research.modeling.prediction_models import (
        ridge_regression_payment_prediction,
        wins_prediction_linear_ridge,
        payment_prediction_logistic_ridge
    )
    print("✓ All 3 modeling functions imported successfully!")
    print(f"  - ridge_regression_payment_prediction: {ridge_regression_payment_prediction}")
    print(f"  - wins_prediction_linear_ridge: {wins_prediction_linear_ridge}")
    print(f"  - payment_prediction_logistic_ridge: {payment_prediction_logistic_ridge}")
    
    # Test backward compatibility
    from QB_research import (
        ridge_regression_payment_prediction as qb_ridge,
        wins_prediction_linear_ridge as qb_wins,
        payment_prediction_logistic_ridge as qb_payment
    )
    print("\n✓ Backward compatibility verified - functions accessible from QB_research")
    
    print("\n✓ Step 8 complete: All 3 modeling functions imported successfully!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

