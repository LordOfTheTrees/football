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

if __name__ == "__main__":
    print("="*80)
    print("REGENERATING DATA WITH INJURY PROJECTIONS")
    print("="*80)
    print("\nThis will:")
    print("  1. Load qb_seasons_payment_labeled.csv")
    print("  2. Apply era adjustments (creates _adj columns)")
    print("  3. Apply injury projections (creates _adj_proj columns)")
    print("  4. Save to qb_seasons_payment_labeled_era_adjusted.csv")
    print("="*80)
    
    # Regenerate the data
    df = create_era_adjusted_payment_data(force_refresh=False)
    
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
