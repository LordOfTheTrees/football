"""
Script to regenerate the era-adjusted payment data with injury projections.

This ensures the master DataFrame has both _adj and _adj_proj columns.
"""

import sys
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from qb_research.adjustments.era_adjustment import create_era_adjusted_payment_data
from qb_research.preprocessing.feature_engineering import prepare_qb_payment_data

if __name__ == "__main__":
    print("="*80)
    print("REGENERATING DATA WITH INJURY PROJECTIONS")
    print("="*80)
    print("\nThis will:")
    print("  1. Prepare payment data (ensures qb_seasons_payment_labeled.csv is current)")
    print("  2. Apply era adjustments (creates _adj columns)")
    print("  3. Apply injury projections (creates _adj_proj columns)")
    print("  4. Save to qb_seasons_payment_labeled_era_adjusted.csv")
    print("="*80)
    
    # Step 1: Prepare payment data (ensures qb_seasons_payment_labeled.csv is current)
    print("\n[Step 1/2] Preparing payment data...")
    df = prepare_qb_payment_data(
        qb_seasons_file='all_seasons_df.csv',
        save_output=True,
        output_file='qb_seasons_payment_labeled.csv',
        force_refresh_contracts=True  # Always refresh to pick up manual updates to QB_contract_data.csv
    )
    
    if df is None:
        print("✗ ERROR: Failed to prepare payment data")
        print("Make sure all_seasons_df.csv exists")
        sys.exit(1)
    
    print(f"✓ Payment data prepared: {len(df)} seasons")
    
    # Step 2: Create era-adjusted data with projections (always refresh factors for new season data)
    print("\n[Step 2/2] Creating era-adjusted data with projections...")
    df = create_era_adjusted_payment_data(force_refresh=True)
    
    if df is not None:
        print("\n" + "="*80)
        print("DATA REGENERATION COMPLETE")
        print("="*80)
        
        # Verify columns
        adj_cols = [c for c in df.columns if c.endswith('_adj') and not c.endswith('_adj_proj')]
        proj_cols = [c for c in df.columns if c.endswith('_proj')]
        
        print(f"\nEra-adjusted columns: {len(adj_cols)}")
        print(f"Projected columns: {len(proj_cols)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df)}")
        
        if proj_cols:
            print(f"\nSample projected columns: {proj_cols[:5]}")
        else:
            print("\n[WARNING] No projected columns found!")
    else:
        print("\n[ERROR] Failed to regenerate data")
