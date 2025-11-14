"""Test Step 12 imports"""
try:
    from qb_research.exports import (
        export_individual_qb_trajectories,
        export_cohort_summary_stats,
        generate_complete_tableau_exports,
        check_recent_qb_inclusion
    )
    print("OK: All 4 export functions imported successfully!")
    print(f"  - export_individual_qb_trajectories: {export_individual_qb_trajectories}")
    print(f"  - export_cohort_summary_stats: {export_cohort_summary_stats}")
    print(f"  - generate_complete_tableau_exports: {generate_complete_tableau_exports}")
    print(f"  - check_recent_qb_inclusion: {check_recent_qb_inclusion}")

    # Test backward compatibility
    from QB_research import (
        export_individual_qb_trajectories as qb_export_traj,
        export_cohort_summary_stats as qb_export_cohort,
        generate_complete_tableau_exports as qb_generate,
        check_recent_qb_inclusion as qb_check
    )
    print("\nOK: Backward compatibility verified - functions accessible from QB_research")

    print("\nOK: Step 12 complete: All 4 export functions imported successfully!")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback
    traceback.print_exc()

