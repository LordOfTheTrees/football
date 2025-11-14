"""Test Step 13 imports"""
try:
    from qb_research.modeling.surface_models import (
        create_payment_probability_surface,
        create_simple_knn_payment_surface,
        run_all_simple_knn_surfaces
    )
    print("OK: All 3 surface functions imported successfully!")
    print(f"  - create_payment_probability_surface: {create_payment_probability_surface}")
    print(f"  - create_simple_knn_payment_surface: {create_simple_knn_payment_surface}")
    print(f"  - run_all_simple_knn_surfaces: {run_all_simple_knn_surfaces}")

    # Test backward compatibility
    from QB_research import (
        create_payment_probability_surface as qb_surface,
        create_simple_knn_payment_surface as qb_knn,
        run_all_simple_knn_surfaces as qb_run_all
    )
    print("\nOK: Backward compatibility verified - functions accessible from QB_research")

    print("\nOK: Step 13 complete: All 3 surface functions imported successfully!")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback
    traceback.print_exc()

