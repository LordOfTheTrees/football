"""
Unified script to update QB data for a completed season.

This script orchestrates the entire data update pipeline:
1. Force re-scrape QB data (optional)
2. Rebuild all_seasons_df.csv from individual QB files
3. Regenerate era-adjusted and injury-projected data

Can be run manually or via GitHub Actions.

Usage:
    python update_season_data.py [--year YYYY] [--overwrite] [--test] [--skip-scrape]
    
    --year: Specific year to update (default: current year - 1, most recent completed season)
    --overwrite: Force re-scrape all QBs even if data exists
    --test: Only test on 2-3 sample QBs (for validation)
    --skip-scrape: Skip scraping step, just rebuild from existing QB_Data files
"""

import sys
import io
import argparse
import os
from datetime import datetime

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from qb_research.utils.year_utils import get_current_season_year
from qb_research.adjustments.era_adjustment import create_era_adjusted_payment_data
from qb_research.preprocessing.feature_engineering import prepare_qb_payment_data
import PFR_Tools as PFR


def test_pfr_scraping(test_qbs=None):
    """
    Test PFR scraping on a few sample QBs to verify structure.
    
    Args:
        test_qbs (list): List of (qb_name, qb_id) tuples to test. 
            If None, uses default test QBs.
    
    Returns:
        bool: True if scraping works, False otherwise
    """
    print("\n" + "="*80)
    print("TESTING PFR SCRAPING LOGIC")
    print("="*80)
    
    if test_qbs is None:
        # Default test QBs - active, well-known players
        test_qbs = [
            ('Josh Allen', 'AlleJo02'),
            ('Patrick Mahomes', 'MahoPa00'),
        ]
    
    print(f"\nTesting {len(test_qbs)} sample QBs...")
    
    success_count = 0
    for qb_name, qb_id in test_qbs:
        print(f"\nTesting: {qb_name} ({qb_id})")
        try:
            qb_stats = PFR.get_qb_seasons(
                qb_name=qb_name,
                qb_id=qb_id,
                debugging=True,
                overwrite=True
            )
            
            if qb_stats is not None and not qb_stats.empty:
                max_season = qb_stats['season'].max()
                print(f"  ✓ Success: Found {len(qb_stats)} seasons, latest: {max_season}")
                success_count += 1
            else:
                print(f"  ✗ Failed: No data returned")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    if success_count == len(test_qbs):
        print(f"✓ All {success_count} test QBs scraped successfully")
        return True
    else:
        print(f"⚠️  Only {success_count}/{len(test_qbs)} test QBs succeeded")
        return False


def update_qb_data(overwrite=False, test_mode=False, target_year=None):
    """
    Update QB data by scraping from PFR.
    
    Args:
        overwrite (bool): Force re-scrape even if CSV exists
        test_mode (bool): Only test on sample QBs
        target_year (int, optional): Target year to update for
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*80)
    print("UPDATING QB DATA FROM PRO FOOTBALL REFERENCE")
    print("="*80)
    
    if test_mode:
        print("\n[TEST MODE] Only testing on sample QBs")
        return test_pfr_scraping()
    
    if not os.path.exists("1st_rd_qb_ids.csv"):
        print("✗ ERROR: 1st_rd_qb_ids.csv not found")
        print("Cannot update QB data without QB ID list")
        return False
    
    print(f"\nScraping QB data (overwrite={overwrite})...")
    success = PFR.pull_updated_QB_data(overwrite=overwrite, target_year=target_year)
    
    if success:
        print("✓ QB data update complete")
    else:
        print("✗ QB data update failed")
    
    return success


def rebuild_all_seasons():
    """
    Rebuild all_seasons_df.csv from individual QB files.
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*80)
    print("REBUILDING all_seasons_df.csv")
    print("="*80)
    
    # Import the rebuild function
    from rebuild_all_seasons_pipeline import rebuild_all_seasons_df
    
    success = rebuild_all_seasons_df()
    
    if success:
        print("✓ all_seasons_df.csv rebuild complete")
    else:
        print("✗ all_seasons_df.csv rebuild failed")
    
    return success


def prepare_payment_data():
    """
    Prepare QB payment data with contract labels and features.
    
    This must run after rebuild_all_seasons() and before regenerate_adjustments().
    Automatically refreshes contract mapping from QB_contract_data.csv on each run.
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*80)
    print("PREPARING QB PAYMENT DATA")
    print("="*80)
    
    df = prepare_qb_payment_data(
        qb_seasons_file='all_seasons_df.csv',
        save_output=True,
        output_file='qb_seasons_payment_labeled.csv',
        force_refresh_contracts=True  # Always refresh to pick up manual updates to QB_contract_data.csv
    )
    
    if df is not None:
        print("✓ Payment data preparation complete")
        print(f"  Total rows: {len(df)}")
        print(f"  Max season: {df['season'].max() if 'season' in df.columns else 'N/A'}")
        return True
    else:
        print("✗ Payment data preparation failed")
        return False


def regenerate_adjustments(force_refresh=False):
    """
    Regenerate era-adjusted and injury-projected data.
    
    Args:
        force_refresh (bool): Force recalculation of adjustment factors
    
    Returns:
        bool: True if successful
    """
    print("\n" + "="*80)
    print("REGENERATING ERA-ADJUSTED AND INJURY-PROJECTED DATA")
    print("="*80)
    
    df = create_era_adjusted_payment_data(force_refresh=force_refresh)
    
    if df is not None:
        print("✓ Data regeneration complete")
        
        # Verify columns
        adj_cols = [c for c in df.columns if c.endswith('_adj') and not c.endswith('_adj_proj')]
        proj_cols = [c for c in df.columns if c.endswith('_proj')]
        
        print(f"\n  Era-adjusted columns: {len(adj_cols)}")
        print(f"  Projected columns: {len(proj_cols)}")
        print(f"  Total rows: {len(df)}")
        
        return True
    else:
        print("✗ Data regeneration failed")
        return False


def validate_output(target_year):
    """
    Validate that the target year appears in output files.
    
    Args:
        target_year (int): Year to validate
    
    Returns:
        bool: True if validation passes
    """
    print("\n" + "="*80)
    print("VALIDATING OUTPUT")
    print("="*80)
    
    import pandas as pd
    
    # Check all_seasons_df.csv
    if os.path.exists('all_seasons_df.csv'):
        df = pd.read_csv('all_seasons_df.csv')
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
        max_season = df['season'].max()
        
        print(f"\nall_seasons_df.csv:")
        print(f"  Max season: {max_season}")
        if max_season >= target_year:
            print(f"  ✓ Contains {target_year} data")
        else:
            print(f"  ✗ Missing {target_year} data (latest is {max_season})")
            return False
    else:
        print("✗ all_seasons_df.csv not found")
        return False
    
    # Check era-adjusted file
    if os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
        max_season = df['season'].max()
        
        print(f"\nqb_seasons_payment_labeled_era_adjusted.csv:")
        print(f"  Max season: {max_season}")
        if max_season >= target_year:
            print(f"  ✓ Contains {target_year} data")
        else:
            print(f"  ✗ Missing {target_year} data (latest is {max_season})")
            return False
    else:
        print("✗ qb_seasons_payment_labeled_era_adjusted.csv not found")
        return False
    
    print("\n✓ Validation passed")
    return True


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description='Update QB season data for completed NFL season',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Specific year to update (default: most recent completed season)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Force re-scrape all QBs even if data exists'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only test on 2-3 sample QBs'
    )
    
    parser.add_argument(
        '--skip-scrape',
        action='store_true',
        help='Skip scraping step, just rebuild from existing QB_Data files'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force recalculation of era adjustment factors'
    )
    
    args = parser.parse_args()
    
    # Determine target year
    if args.year:
        target_year = args.year
    else:
        target_year = get_current_season_year()
    
    print("="*80)
    print("QB SEASON DATA UPDATE")
    print("="*80)
    print(f"\nTarget year: {target_year}")
    print(f"Current year: {datetime.now().year}")
    print(f"Overwrite existing data: {args.overwrite}")
    print(f"Test mode: {args.test}")
    print(f"Skip scraping: {args.skip_scrape}")
    print("="*80)
    
    # Step 1: Update QB data (unless skipping)
    if not args.skip_scrape:
        if not update_qb_data(
            overwrite=args.overwrite,
            test_mode=args.test,
            target_year=target_year
        ):
            if not args.test:
                print("\n⚠️  QB data update had issues, but continuing...")
            else:
                print("\n✗ Test mode failed - aborting")
                return 1
    else:
        print("\n[Skipping scraping step]")
    
    # Step 2: Rebuild all_seasons_df.csv
    if not rebuild_all_seasons():
        print("\n✗ Pipeline failed at rebuild step")
        return 1
    
    # Step 3: Prepare payment data (adds contract labels, pick numbers, lookback features)
    if not prepare_payment_data():
        print("\n✗ Pipeline failed at payment data preparation step")
        return 1
    
    # Step 4: Regenerate adjustments
    if not regenerate_adjustments(force_refresh=args.force_refresh):
        print("\n✗ Pipeline failed at adjustment step")
        return 1
    
    # Step 5: Validate output
    if not validate_output(target_year):
        print("\n⚠️  Validation found issues - check output files")
        return 1
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print(f"\nData updated for {target_year} season")
    print("\nNext steps:")
    print("  - Review output files")
    print("  - Spot-check stats against Pro Football Reference")
    print("  - Commit changes if satisfied")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
