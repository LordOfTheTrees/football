"""
Tableau export functions for QB research data.

This module contains functions for exporting QB trajectory and summary statistics
data in formats suitable for Tableau visualization.
"""

import os
import pandas as pd

from qb_research.utils.data_loading import load_csv_safe


def export_individual_qb_trajectories(
    metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
    qb_list=None,
    years_range=(0, 6),
    include_recent_drafts=True,
    recent_draft_cutoff=2024
):
    """
    Exports year-by-year performance trajectories for individual QBs.
    
    FIXED: Now includes recent draft picks (2021+) for complete trajectory visualization.
    
    Args:
        metrics (list): Performance metrics to include
        qb_list (list): Specific player_ids to export (None = all)
        years_range (tuple): Min and max years since draft to include
        include_recent_drafts (bool): Whether to include QBs drafted after 2020
        recent_draft_cutoff (int): Earliest year to include for recent drafts
    
    Returns:
        DataFrame: QB trajectories in long format
    """
    print("\n" + "="*80)
    print("EXPORTING INDIVIDUAL QB TRAJECTORIES")
    print("="*80)
    
    # Load era-adjusted payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: Run create_era_adjusted_payment_data() first")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Handle draft year filtering
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    
    if include_recent_drafts:
        # Include ALL first-round QBs for trajectory visualization
        print(f"Including ALL first-round QBs (including recent drafts {recent_draft_cutoff}+)")
        # Show breakdown of QBs by era
        established_qbs = payment_df[payment_df['draft_year'] <= 2020]['player_id'].nunique()
        recent_qbs = payment_df[payment_df['draft_year'] >= recent_draft_cutoff]['player_id'].nunique()
        print(f"  Established QBs (≤2020): {established_qbs}")
        print(f"  Recent QBs ({recent_draft_cutoff}+): {recent_qbs}")
        # Don't filter by draft year for trajectories
    else:
        # Filter to only established QBs (for payment prediction modeling)
        cutoff_year = 2020
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        print(f"Filtering to QBs drafted ≤{cutoff_year} only")
    
    # Calculate years since draft
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
    payment_df['years_since_draft'] = payment_df['season'] - payment_df['draft_year']
    
    # Filter to years range
    min_year, max_year = years_range
    payment_df = payment_df[
        (payment_df['years_since_draft'] >= min_year) & 
        (payment_df['years_since_draft'] <= max_year)
    ]
    
    # Filter to specific QBs if requested
    if qb_list is not None:
        payment_df = payment_df[payment_df['player_id'].isin(qb_list)]
        print(f"Filtering to {len(qb_list)} specific QBs")
    
    # Select columns to export
    base_cols = [
        'player_id',
        'player_name', 
        'draft_year',
        'draft_team',
        'pick_number',
        'season',
        'years_since_draft',
        'got_paid',
        'payment_year',
        'years_to_payment'
    ]
    
    export_cols = base_cols + metrics
    
    # Check which columns exist
    available_cols = [col for col in export_cols if col in payment_df.columns]
    missing_cols = [col for col in export_cols if col not in payment_df.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: {len(missing_cols)} columns not found:")
        for col in missing_cols[:5]:
            print(f"  - {col}")
    
    trajectories_df = payment_df[available_cols].copy()
    
    # Convert metrics to numeric
    for metric in metrics:
        if metric in trajectories_df.columns:
            trajectories_df[metric] = pd.to_numeric(trajectories_df[metric], errors='coerce')
    
    # Sort by player and year
    trajectories_df = trajectories_df.sort_values(['player_id', 'years_since_draft'])
    
    print(f"\nExported {len(trajectories_df)} QB-season records")
    print(f"Unique QBs: {trajectories_df['player_id'].nunique()}")
    
    # Show breakdown by draft era
    draft_year_counts = trajectories_df.groupby('draft_year')['player_id'].nunique().sort_index()
    print(f"\nQBs by draft year:")
    for year, count in draft_year_counts.tail(10).items():  # Show last 10 years
        print(f"  {int(year)}: {count} QBs")
    
    # Payment status breakdown
    if 'got_paid' in trajectories_df.columns:
        paid_qbs = trajectories_df[trajectories_df['got_paid']]['player_id'].nunique()
        unpaid_qbs = trajectories_df[~trajectories_df['got_paid']]['player_id'].nunique()
        print(f"\nPayment status:")
        print(f"  QBs who got paid: {paid_qbs}")
        print(f"  QBs who didn't get paid: {unpaid_qbs}")
    
    # Show sample recent QBs
    print(f"\nSample recent QBs (2020+):")
    recent_sample = trajectories_df[trajectories_df['draft_year'] >= 2020].groupby('player_id').agg({
        'player_name': 'first',
        'draft_year': 'first',
        'pick_number': 'first',
        'years_since_draft': 'max'
    }).sort_values('draft_year', ascending=False).head(10)
    
    for idx, row in recent_sample.iterrows():
        print(f"  {row['player_name']}: Drafted {int(row['draft_year'])} (#{int(row['pick_number'])}), {int(row['years_since_draft'])} seasons")
    
    # Show sample
    print("\nSample trajectory (first QB):")
    sample_qb = trajectories_df['player_id'].iloc[0]
    sample = trajectories_df[trajectories_df['player_id'] == sample_qb][
        ['player_name', 'years_since_draft'] + metrics + ['got_paid']
    ]
    print(sample.to_string(index=False))
    
    # Save to CSV
    output_file = 'qb_trajectories_for_tableau.csv'
    trajectories_df.to_csv(output_file, index=False, mode='w')  # 'w' = write mode (overwrite)
    print(f"\n✓ Saved to: {output_file}")
    
    return trajectories_df


def export_cohort_summary_stats(
    metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
    years_range=(0, 6),
    include_recent_drafts=True
):
    """
    Exports summary statistics for the entire first-round QB cohort by year.
    
    FIXED: Now includes recent draft picks for complete visualization.
    
    Args:
        metrics (list): Performance metrics to summarize
        years_range (tuple): Min and max years since draft to include
        include_recent_drafts (bool): Whether to include recent drafts
    
    Returns:
        DataFrame: Summary stats by year and metric
    """
    print("\n" + "="*80)
    print("EXPORTING COHORT SUMMARY STATISTICS")
    print("="*80)
    
    # Load era-adjusted payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: Run create_era_adjusted_payment_data() first")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Handle draft year filtering
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    
    if include_recent_drafts:
        print(f"Including ALL first-round QBs for cohort statistics")
        # Don't filter by draft year
    else:
        # Filter to only established QBs
        cutoff_year = 2020
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        print(f"Filtering to QBs drafted ≤{cutoff_year} only")
    
    # Calculate years since draft
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
    payment_df['years_since_draft'] = payment_df['season'] - payment_df['draft_year']
    
    # Filter to years range
    min_year, max_year = years_range
    payment_df = payment_df[
        (payment_df['years_since_draft'] >= min_year) & 
        (payment_df['years_since_draft'] <= max_year)
    ]
    
    print(f"Cohort size: {payment_df['player_id'].nunique()} QBs")
    print(f"Years range: {min_year}-{max_year}")
    print(f"Draft years: {int(payment_df['draft_year'].min())}-{int(payment_df['draft_year'].max())}")
    
    # Calculate summary stats for each metric and year
    summary_rows = []
    
    for metric in metrics:
        if metric not in payment_df.columns:
            print(f"⚠️  Warning: {metric} not found, skipping")
            continue
        
        print(f"\nProcessing {metric}...")
        
        # Convert to numeric
        payment_df[metric] = pd.to_numeric(payment_df[metric], errors='coerce')
        
        # Group by years_since_draft
        for year in sorted(payment_df['years_since_draft'].unique()):
            year_data = payment_df[payment_df['years_since_draft'] == year][metric].dropna()
            
            if len(year_data) == 0:
                continue
            
            summary_rows.append({
                'metric': metric,
                'years_since_draft': int(year),
                'count': len(year_data),
                'min': year_data.min(),
                'p10': year_data.quantile(0.10),
                'p25': year_data.quantile(0.25),
                'median': year_data.median(),
                'mean': year_data.mean(),
                'p75': year_data.quantile(0.75),
                'p90': year_data.quantile(0.90),
                'max': year_data.max(),
                'std': year_data.std()
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    print(f"\nCreated summary stats for {len(summary_df)} year-metric combinations")
    
    # Display sample
    print("\nSample statistics:")
    print(summary_df.head(10).to_string(index=False))
    
    # Save to CSV
    output_file = 'cohort_summary_stats.csv'
    summary_df.to_csv(output_file, index=False, mode='w')
    print(f"\n✓ Saved to: {output_file}")
    
    return summary_df


def generate_complete_tableau_exports():
    """
    Master function to generate all Tableau export files with recent draft picks included.
    
    This replaces the partial export that was missing 2021+ draft picks.
    """
    print("\n" + "="*80)
    print("GENERATING COMPLETE TABLEAU EXPORT FILES")
    print("(Including Recent Draft Picks 2021+)")
    print("="*80)
    
    # Define metrics to analyze
    primary_metrics = ['total_yards_adj', 'Pass_ANY/A_adj']
    years_to_include = (0, 6)  # Y0 through Y6
    
    # 1. Export individual QB trajectories (WITH recent drafts)
    print("\n\n[1/2] Exporting QB trajectories (ALL draft years)...")
    trajectories = export_individual_qb_trajectories(
        metrics=primary_metrics,
        qb_list=None,  # None = all QBs
        years_range=years_to_include,
        include_recent_drafts=True  # KEY FIX: Include 2021+ drafts
    )
    
    # 2. Export cohort summary stats (WITH recent drafts)
    print("\n\n[2/2] Exporting cohort summary statistics (ALL draft years)...")
    summary = export_cohort_summary_stats(
        metrics=primary_metrics,
        years_range=years_to_include,
        include_recent_drafts=True  # KEY FIX: Include 2021+ drafts
    )
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - qb_trajectories_for_tableau.csv (ALL QBs including 2021+ drafts)")
    print("  - cohort_summary_stats.csv (ALL QBs including 2021+ drafts)")
    print("\nThese files now include:")
    print("  - 2021: Trevor Lawrence, Zach Wilson, Trey Lance, Justin Fields, Mac Jones")
    print("  - 2022: Kenny Pickett, Malik Willis")
    print("  - 2023: Bryce Young, C.J. Stroud, Anthony Richardson")
    print("  - 2024: Caleb Williams, Jayden Daniels, Drake Maye, Michael Penix Jr., J.J. McCarthy, Bo Nix")
    
    return trajectories, summary


def check_recent_qb_inclusion():
    """
    Quick test to verify recent QBs are included in trajectory data.
    """
    print("\n" + "="*80)
    print("CHECKING RECENT QB INCLUSION")
    print("="*80)
    
    # Test with include_recent_drafts=True
    trajectories = export_individual_qb_trajectories(
        metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
        include_recent_drafts=True
    )
    
    if trajectories is not None:
        recent_qbs = trajectories[trajectories['draft_year'] >= 2021].groupby(['player_name', 'draft_year']).size().reset_index(name='seasons')
        recent_qbs = recent_qbs.sort_values('draft_year', ascending=False)
        
        print(f"\nRecent QBs found in trajectory data:")
        for _, row in recent_qbs.head(15).iterrows():
            print(f"  {row['player_name']}: {int(row['draft_year'])} ({row['seasons']} seasons)")
        
        if len(recent_qbs) == 0:
            print("⚠️  WARNING: No recent QBs (2021+) found in data!")
        else:
            print(f"\n✓ SUCCESS: {len(recent_qbs)} recent QBs included")
    
    return trajectories

