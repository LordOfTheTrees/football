#!/usr/bin/env python3
"""
QB Trajectory Comp Analysis Tool - Enhanced with Projection Mode
Interactive script to find similar QBs based on trajectory matching
Now supports projecting hypothetical future performance

Usage:
  python qb_comp_tool.py                    # Interactive mode
  python qb_comp_tool.py -p                 # Projection mode
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse

# Import functions from QB_research
try:
    from QB_research import (
        find_comps_both_metrics,
        find_most_similar_qbs
    )
except ImportError:
    print("ERROR: Could not import from QB_research.py")
    print("Make sure QB_research.py is in the same directory")
    sys.exit(1)


def load_qb_id_lookup():
    """
    Creates a lookup dictionary from player names to player IDs.
    
    Returns:
        dict: {player_name_lower: player_id}
    """
    # Try multiple possible files
    possible_files = [
        'first_round_qbs_with_picks.csv',
        'first_round_qbs.csv',
        'player_ids.csv',
        'qb_seasons_payment_labeled_era_adjusted.csv'
    ]
    
    lookup = {}
    
    for filepath in possible_files:
        if not os.path.exists(filepath):
            continue
        
        df = pd.read_csv(filepath)
        
        # Determine which columns to use
        name_col = None
        id_col = None
        
        if 'player_name' in df.columns and 'player_id' in df.columns:
            name_col = 'player_name'
            id_col = 'player_id'
        elif 'name' in df.columns and 'player_id' in df.columns:
            name_col = 'name'
            id_col = 'player_id'
        
        if name_col and id_col:
            # Create lookup entries
            for _, row in df[[name_col, id_col]].drop_duplicates().iterrows():
                name = str(row[name_col]).strip().lower()
                player_id = str(row[id_col]).strip()
                lookup[name] = player_id
    
    return lookup


def search_qb_by_name(search_term, lookup_dict):
    """
    Searches for QBs matching the search term.
    
    Args:
        search_term: Partial or full name to search for
        lookup_dict: Dictionary from load_qb_id_lookup()
    
    Returns:
        list: List of (player_name, player_id) tuples that match
    """
    search_lower = search_term.strip().lower()
    matches = []
    
    for name_lower, player_id in lookup_dict.items():
        if search_lower in name_lower:
            # Capitalize for display
            display_name = ' '.join(word.capitalize() for word in name_lower.split())
            matches.append((display_name, player_id))
    
    return matches


def inject_projected_seasons(target_qb_id, projected_data):
    """
    Injects hypothetical future seasons into QB dataset for projection analysis.
    
    Args:
        target_qb_id: Player ID to inject data for
        projected_data: dict with keys like:
            {
                'years': [2, 3, 4, 5],  # Which future years to project
                'total_yards_adj': [3500, 3800, 4000, 4200],  # Optional
                'Pass_ANY/A_adj': [5.5, 6.0, 6.5, 7.0]  # Optional
            }
    
    Returns:
        Modified DataFrame with projected seasons appended
    """
    print("\n" + "="*80)
    print("INJECTING PROJECTED SEASONS")
    print("="*80)
    
    # Load existing QB data
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Get QB's existing data to understand structure
    qb_data = df[df['player_id'] == target_qb_id].copy()
    
    if len(qb_data) == 0:
        print(f"✗ ERROR: No data found for QB ID: {target_qb_id}")
        return None
    
    # Calculate years_since_draft
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['years_since_draft'] = df['season'] - df['draft_year']
    
    qb_data['season'] = pd.to_numeric(qb_data['season'], errors='coerce')
    qb_data['draft_year'] = pd.to_numeric(qb_data['draft_year'], errors='coerce')
    qb_data['years_since_draft'] = qb_data['season'] - qb_data['draft_year']
    
    # Get latest existing year
    latest_year = qb_data['years_since_draft'].max()
    latest_season = qb_data['season'].max()
    draft_year = qb_data['draft_year'].iloc[0]
    player_name = qb_data['player_name'].iloc[0]
    
    print(f"QB: {player_name} ({target_qb_id})")
    print(f"Latest existing data: Year {int(latest_year)} (Season {int(latest_season)})")
    print(f"Draft year: {int(draft_year)}")
    
    # Create projected seasons
    projected_years = projected_data.get('years', [])
    
    if not projected_years:
        print("\n✗ ERROR: No projected years specified")
        return None
    
    print(f"\nProjected years to inject: {projected_years}")
    
    # Check that projected years are in the future
    for year in projected_years:
        if year <= latest_year:
            print(f"\n⚠️  WARNING: Year {year} is not in the future (latest year = {int(latest_year)})")
    
    # Determine which metrics are being projected
    has_yards = 'total_yards_adj' in projected_data and projected_data['total_yards_adj'] is not None
    has_anya = 'Pass_ANY/A_adj' in projected_data and projected_data['Pass_ANY/A_adj'] is not None
    
    if not has_yards and not has_anya:
        print("\n✗ ERROR: No projected metrics specified (need total_yards_adj and/or Pass_ANY/A_adj)")
        return None
    
    print(f"\nProjecting metrics:")
    if has_yards:
        print(f"  ✓ total_yards_adj: {projected_data['total_yards_adj']}")
    if has_anya:
        print(f"  ✓ Pass_ANY/A_adj: {projected_data['Pass_ANY/A_adj']}")
    
    # Create template row from latest season
    template_row = qb_data.iloc[-1].copy()
    
    # Build projected rows
    projected_rows = []
    
    for i, year in enumerate(projected_years):
        new_row = template_row.copy()
        
        # Update year/season
        new_row['years_since_draft'] = year
        new_row['season'] = draft_year + year
        
        # Update projected metrics
        if has_yards and i < len(projected_data['total_yards_adj']):
            new_row['total_yards_adj'] = projected_data['total_yards_adj'][i]
        else:
            new_row['total_yards_adj'] = np.nan  # Mark as unavailable
        
        if has_anya and i < len(projected_data['Pass_ANY/A_adj']):
            new_row['Pass_ANY/A_adj'] = projected_data['Pass_ANY/A_adj'][i]
        else:
            new_row['Pass_ANY/A_adj'] = np.nan  # Mark as unavailable
        
        # Mark as projected
        new_row['is_projected'] = True
        
        projected_rows.append(new_row)
    
    # Create projected DataFrame
    projected_df = pd.DataFrame(projected_rows)
    
    # Add is_projected flag to original data
    df['is_projected'] = False
    qb_data['is_projected'] = False
    
    # Combine: remove QB's original data, add back with projections
    df_without_qb = df[df['player_id'] != target_qb_id]
    df_with_projections = pd.concat([df_without_qb, qb_data, projected_df], ignore_index=True)
    
    # Keep in memory only - do NOT save to disk
    print(f"\n✓ Created projected dataset in memory (not saved to disk)")
    print(f"  Total seasons for {player_name}: {len(qb_data) + len(projected_df)} ({len(projected_df)} projected)")
    
    # Show projection summary
    print(f"\n📊 PROJECTION SUMMARY:")
    for i, year in enumerate(projected_years):
        season = int(draft_year + year)
        print(f"  Year {year} (Season {season}):")
        if has_yards and i < len(projected_data['total_yards_adj']):
            print(f"    total_yards_adj: {projected_data['total_yards_adj'][i]:,.1f}")
        if has_anya and i < len(projected_data['Pass_ANY/A_adj']):
            print(f"    Pass_ANY/A_adj: {projected_data['Pass_ANY/A_adj'][i]:.2f}")
    
    return df_with_projections


def run_comps_with_projection(target_qb_id, target_qb_name, projected_data, 
                               decision_year, n_comps, yards_weights, anya_weights):
    """
    Runs comp analysis using projected future data.
    
    Args:
        target_qb_id: QB to analyze
        target_qb_name: QB name for display
        projected_data: Projection data dict
        decision_year: Which year to evaluate at
        n_comps: Number of comps
        yards_weights: Year weights for yards
        anya_weights: Year weights for ANY/A
    
    Returns:
        dict with comp results
    """
    # Inject projected seasons (returns dataframe in memory)
    df_with_proj = inject_projected_seasons(target_qb_id, projected_data)
    
    if df_with_proj is None:
        return None
    
    # Save current data file and temporarily replace with projected version
    original_file = 'qb_seasons_payment_labeled_era_adjusted.csv'
    backup_file = 'qb_seasons_payment_labeled_era_adjusted_BACKUP_TEMP.csv'
    
    # Backup original
    if os.path.exists(original_file):
        os.rename(original_file, backup_file)
    
    # Write projected data to standard location temporarily
    df_with_proj.to_csv(original_file, index=False)
    
    try:
        # Determine which metrics to analyze
        has_yards = ('total_yards_adj' in projected_data and 
                     projected_data['total_yards_adj'] is not None)
        has_anya = ('Pass_ANY/A_adj' in projected_data and 
                    projected_data['Pass_ANY/A_adj'] is not None)
        
        yards_comps = None
        anya_comps = None
        
        # Run yards comp if projected
        if has_yards:
            print("\n" + "="*80)
            print("RUNNING YARDS COMP WITH PROJECTION")
            print("="*80)
            yards_comps = find_most_similar_qbs(
                target_qb_id=target_qb_id,
                decision_year=decision_year,
                metric='total_yards_adj',
                n_comps=n_comps,
                year_weights=yards_weights
            )
        
        # Run ANY/A comp if projected
        if has_anya:
            print("\n" + "="*80)
            print("RUNNING ANY/A COMP WITH PROJECTION")
            print("="*80)
            anya_comps = find_most_similar_qbs(
                target_qb_id=target_qb_id,
                decision_year=decision_year,
                metric='Pass_ANY/A_adj',
                n_comps=n_comps,
                year_weights=anya_weights
            )
        
        results = {
            'yards_comps': yards_comps,
            'anya_comps': anya_comps,
            'projected_data': projected_data
        }
        
        return results
        
    finally:
        # Restore original file and clean up
        if os.path.exists(original_file):
            os.remove(original_file)  # Remove projected version
        if os.path.exists(backup_file):
            os.rename(backup_file, original_file)  # Restore original
        
        print(f"\n✓ Restored original dataset (projected data removed)")


def export_comp_analysis_for_tableau(
    target_qb_id,
    target_qb_name,
    yards_comps,
    anya_comps,
    decision_year,
    output_dir='comp_analysis_output',
    is_projection=False
):
    """
    Exports comp analysis results in Tableau-friendly format.
    
    Creates CSV files with all data needed for visualization:
    1. comp_summary.csv - High-level summary
    2. comp_detailed_yards.csv - Full yards comp data
    3. comp_detailed_anya.csv - Full ANY/A comp data
    4. comp_trajectories.csv - Year-by-year trajectory data for target + comps
    
    Args:
        target_qb_id: Player ID of target QB
        target_qb_name: Name of target QB
        yards_comps: DataFrame from find_most_similar_qbs for yards
        anya_comps: DataFrame from find_most_similar_qbs for ANY/A
        decision_year: Which decision year was analyzed
        output_dir: Directory to save output files
        is_projection: Whether this includes projected data
    """
    print("\n" + "="*80)
    print("EXPORTING DATA FOR TABLEAU")
    print("="*80)
    
    # Create QB-specific output directory: comp_analysis_output/QBIDxx00/
    base_output_dir = 'comp_analysis_output'
    qb_output_dir = os.path.join(base_output_dir, target_qb_id)
    os.makedirs(qb_output_dir, exist_ok=True)
    
    print(f"Output directory: {qb_output_dir}/")
    
    # Load full trajectory data
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['years_since_draft'] = df['season'] - df['draft_year']
    
    # Filter to decision year
    df = df[df['years_since_draft'] < decision_year]
    
    # =========================================================================
    # FILE 1: Summary statistics
    # =========================================================================
    summary_data = []
    
    if yards_comps is not None:
        yards_payment_rate = yards_comps['got_paid'].mean() * 100
        summary_data.append({
            'target_qb_id': target_qb_id,
            'target_qb_name': target_qb_name,
            'decision_year': decision_year,
            'metric': 'total_yards_adj',
            'n_comps': len(yards_comps),
            'payment_rate_pct': yards_payment_rate,
            'comps_paid': yards_comps['got_paid'].sum(),
            'comps_not_paid': (~yards_comps['got_paid']).sum(),
            'is_projection': is_projection
        })
    
    if anya_comps is not None:
        anya_payment_rate = anya_comps['got_paid'].mean() * 100
        summary_data.append({
            'target_qb_id': target_qb_id,
            'target_qb_name': target_qb_name,
            'decision_year': decision_year,
            'metric': 'Pass_ANY/A_adj',
            'n_comps': len(anya_comps),
            'payment_rate_pct': anya_payment_rate,
            'comps_paid': anya_comps['got_paid'].sum(),
            'comps_not_paid': (~anya_comps['got_paid']).sum(),
            'is_projection': is_projection
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Build filename with 'projection' first if needed
    if is_projection:
        summary_file = os.path.join(qb_output_dir, f'projection_comp_summary_{target_qb_id}.csv')
    else:
        summary_file = os.path.join(qb_output_dir, f'comp_summary_{target_qb_id}.csv')
    
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")
    
    # =========================================================================
    # FILE 2 & 3: Detailed comp lists
    # =========================================================================
    if yards_comps is not None:
        yards_detailed = yards_comps.copy()
        yards_detailed['target_qb_id'] = target_qb_id
        yards_detailed['target_qb_name'] = target_qb_name
        yards_detailed['decision_year'] = decision_year
        yards_detailed['metric'] = 'total_yards_adj'
        yards_detailed['comp_rank'] = range(1, len(yards_detailed) + 1)
        yards_detailed['is_projection'] = is_projection
        
        if is_projection:
            yards_file = os.path.join(qb_output_dir, f'projection_yards_comps_{target_qb_id}.csv')
        else:
            yards_file = os.path.join(qb_output_dir, f'yards_comps_{target_qb_id}.csv')
        
        yards_detailed.to_csv(yards_file, index=False)
        print(f"✓ Saved yards comps: {yards_file}")
    
    if anya_comps is not None:
        anya_detailed = anya_comps.copy()
        anya_detailed['target_qb_id'] = target_qb_id
        anya_detailed['target_qb_name'] = target_qb_name
        anya_detailed['decision_year'] = decision_year
        anya_detailed['metric'] = 'Pass_ANY/A_adj'
        anya_detailed['comp_rank'] = range(1, len(anya_detailed) + 1)
        anya_detailed['is_projection'] = is_projection
        
        if is_projection:
            anya_file = os.path.join(qb_output_dir, f'projection_anya_comps_{target_qb_id}.csv')
        else:
            anya_file = os.path.join(qb_output_dir, f'anya_comps_{target_qb_id}.csv')
        
        anya_detailed.to_csv(anya_file, index=False)
        print(f"✓ Saved ANY/A comps: {anya_file}")
    
    # =========================================================================
    # FILE 4: Trajectory data (target + all comps)
    # =========================================================================
    trajectory_rows = []
    
    # Collect all unique comp IDs
    comp_ids = set()
    if yards_comps is not None:
        comp_ids.update(yards_comps['player_id'].tolist())
    if anya_comps is not None:
        comp_ids.update(anya_comps['player_id'].tolist())
    
    # Add target QB
    comp_ids.add(target_qb_id)
    
    print(f"\nExtracting trajectory data for {len(comp_ids)} QBs...")
    
    for qb_id in comp_ids:
        qb_data = df[df['player_id'] == qb_id].sort_values('years_since_draft')
        
        if len(qb_data) == 0:
            continue
        
        qb_name = qb_data.iloc[0]['player_name']
        draft_year = qb_data.iloc[0].get('draft_year', np.nan)
        got_paid = qb_data.iloc[0].get('got_paid', False)
        
        # Determine QB type
        if qb_id == target_qb_id:
            qb_type = 'target'
        else:
            qb_type = 'comp'
        
        # Get comp ranks
        yards_rank = np.nan
        anya_rank = np.nan
        
        if yards_comps is not None and qb_id in yards_comps['player_id'].values:
            yards_rank = int(yards_comps[yards_comps['player_id'] == qb_id].index[0] + 1)
        
        if anya_comps is not None and qb_id in anya_comps['player_id'].values:
            anya_rank = int(anya_comps[anya_comps['player_id'] == qb_id].index[0] + 1)
        
        # Extract metrics for each year
        for _, row in qb_data.iterrows():
            year = int(row['years_since_draft'])
            season = int(row['season'])
            
            trajectory_rows.append({
                'target_qb_id': target_qb_id,
                'target_qb_name': target_qb_name,
                'qb_id': qb_id,
                'qb_name': qb_name,
                'qb_type': qb_type,
                'yards_comp_rank': yards_rank,
                'anya_comp_rank': anya_rank,
                'decision_year': decision_year,
                'years_since_draft': year,
                'season': season,
                'draft_year': draft_year,
                'total_yards_adj': pd.to_numeric(row['total_yards_adj'], errors='coerce'),
                'Pass_ANY/A_adj': pd.to_numeric(row['Pass_ANY/A_adj'], errors='coerce'),
                'Pass_TD_adj': pd.to_numeric(row.get('Pass_TD_adj', np.nan), errors='coerce'),
                'Rush_Rushing_Succ%_adj': pd.to_numeric(row.get('Rush_Rushing_Succ%_adj', np.nan), errors='coerce'),
                'got_paid': got_paid,
                'payment_year': row.get('payment_year', np.nan),
                'is_projected': row.get('is_projected', False),
                'is_projection_analysis': is_projection
            })
    
    trajectory_df = pd.DataFrame(trajectory_rows)
    
    if is_projection:
        trajectory_file = os.path.join(qb_output_dir, f'projection_trajectories_{target_qb_id}.csv')
    else:
        trajectory_file = os.path.join(qb_output_dir, f'trajectories_{target_qb_id}.csv')
    
    trajectory_df.to_csv(trajectory_file, index=False)
    print(f"✓ Saved trajectories: {trajectory_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nFiles created in: {qb_output_dir}/")
    
    file_prefix = "projection_" if is_projection else ""
    print(f"  1. {file_prefix}comp_summary_{target_qb_id}.csv - High-level summary")
    print(f"  2. {file_prefix}yards_comps_{target_qb_id}.csv - Yards comp rankings")
    print(f"  3. {file_prefix}anya_comps_{target_qb_id}.csv - ANY/A comp rankings")
    print(f"  4. {file_prefix}trajectories_{target_qb_id}.csv - Year-by-year data for visualization")
    print("\nReady for Tableau import!")


def run_interactive_comp_analysis():
    """
    Main interactive function - prompts user for QB name and runs analysis.
    """
    print("\n" + "="*80)
    print("QB TRAJECTORY COMP ANALYSIS - INTERACTIVE MODE")
    print("="*80)
    
    # Check required files exist
    required_files = ['qb_seasons_payment_labeled_era_adjusted.csv']
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"\n✗ ERROR: Required file not found: {filepath}")
            print("Run prepare_qb_payment_data() and create_era_adjusted_payment_data() first")
            return
    
    # Load QB lookup
    print("\nLoading QB database...")
    lookup = load_qb_id_lookup()
    
    if not lookup:
        print("✗ ERROR: Could not create QB lookup table")
        print("Make sure you have one of these files:")
        print("  - first_round_qbs_with_picks.csv")
        print("  - player_ids.csv")
        print("  - qb_seasons_payment_labeled_era_adjusted.csv")
        return
    
    print(f"✓ Loaded {len(lookup)} QBs")
    
    # Get QB name from user
    print("\n" + "="*80)
    print("ENTER QB NAME")
    print("="*80)
    print("You can enter a partial name (e.g., 'fields', 'mahomes', 'allen')")
    
    search_term = input("\nQB Name: ").strip()
    
    if not search_term:
        print("No name entered. Exiting.")
        return
    
    # Search for matches
    matches = search_qb_by_name(search_term, lookup)
    
    if not matches:
        print(f"\n✗ No QBs found matching '{search_term}'")
        print("\nTry a different search term")
        return
    
    # If multiple matches, let user choose
    if len(matches) > 1:
        print(f"\n✓ Found {len(matches)} QBs matching '{search_term}':")
        print("\n" + "-"*80)
        for idx, (name, player_id) in enumerate(matches, 1):
            print(f"  {idx}. {name} ({player_id})")
        print("-"*80)
        
        choice = input(f"\nSelect QB (1-{len(matches)}): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(matches):
                print("Invalid selection. Exiting.")
                return
            
            target_name, target_id = matches[choice_idx]
        except ValueError:
            print("Invalid input. Exiting.")
            return
    else:
        target_name, target_id = matches[0]
        print(f"\n✓ Found: {target_name} ({target_id})")
    
    # Get decision year
    print("\n" + "="*80)
    print("DECISION YEAR")
    print("="*80)
    print("Which year to evaluate at?")
    print("  3 = After Year 2 (5th year option decision)")
    print("  4 = After Year 3 (typical extension)")
    print("  5 = After Year 4 (late extension)")
    print("  6 = After Year 5 (2nd contract)")
    
    decision_year_input = input("\nDecision Year (default=4): ").strip()
    
    if decision_year_input:
        try:
            decision_year = int(decision_year_input)
            if decision_year < 3 or decision_year > 6:
                print("Decision year must be 3-6. Using 4.")
                decision_year = 4
        except ValueError:
            print("Invalid input. Using 4.")
            decision_year = 4
    else:
        decision_year = 4
    
    # Get number of comps
    n_comps_input = input("\nNumber of comps to find (default=10): ").strip()
    
    if n_comps_input:
        try:
            n_comps = int(n_comps_input)
            if n_comps < 1 or n_comps > 50:
                print("Number of comps must be 1-50. Using 10.")
                n_comps = 10
        except ValueError:
            print("Invalid input. Using 10.")
            n_comps = 10
    else:
        n_comps = 10
    
    # Load year weights
    print("\n" + "="*80)
    print("LOADING YEAR WEIGHTS")
    print("="*80)
    
    yards_weights = None
    anya_weights = None
    
    yards_file = 'year_weights_total_yards_adj.csv'
    anya_file = 'year_weights_Pass_ANY_A_adj.csv'
    
    if os.path.exists(yards_file):
        yards_df = pd.read_csv(yards_file)
        yards_subset = yards_df[yards_df['Decision_Year'] == decision_year]
        if len(yards_subset) > 0:
            yards_weights = {}
            for _, row in yards_subset.iterrows():
                year_str = row['Performance_Year']
                year_num = int(year_str.split()[-1])
                yards_weights[year_num] = row['Weight_%'] / 100.0
            print(f"✓ Loaded yards weights for Decision Year {decision_year}")
        else:
            print(f"⚠️  No yards weights for Decision Year {decision_year}, using uniform")
    else:
        print(f"⚠️  {yards_file} not found, using uniform weights")
    
    if os.path.exists(anya_file):
        anya_df = pd.read_csv(anya_file)
        anya_subset = anya_df[anya_df['Decision_Year'] == decision_year]
        if len(anya_subset) > 0:
            anya_weights = {}
            for _, row in anya_subset.iterrows():
                year_str = row['Performance_Year']
                year_num = int(year_str.split()[-1])
                anya_weights[year_num] = row['Weight_%'] / 100.0
            print(f"✓ Loaded ANY/A weights for Decision Year {decision_year}")
        else:
            print(f"⚠️  No ANY/A weights for Decision Year {decision_year}, using uniform")
    else:
        print(f"⚠️  {anya_file} not found, using uniform weights")
    
    # Run analysis
    print("\n" + "="*80)
    print(f"RUNNING COMP ANALYSIS FOR {target_name}")
    print("="*80)
    
    results = find_comps_both_metrics(
        target_qb_id=target_id,
        decision_year=decision_year,
        n_comps=n_comps,
        year_weights_yards=yards_weights,
        year_weights_anya=anya_weights
    )
    
    if results is None:
        print("\n✗ ERROR: Comp analysis failed")
        return
    
    yards_comps = results['yards_comps']
    anya_comps = results['anya_comps']
    
    # Export for Tableau
    export_comp_analysis_for_tableau(
        target_qb_id=target_id,
        target_qb_name=target_name,
        yards_comps=yards_comps,
        anya_comps=anya_comps,
        decision_year=decision_year,
        output_dir='comp_analysis_output',
        is_projection=False
    )
    
    print("\n✓ Analysis complete!")


def run_projection_mode():
    """
    Projection mode - allows user to specify hypothetical future performance.
    """
    print("\n" + "="*80)
    print("QB TRAJECTORY COMP ANALYSIS - PROJECTION MODE")
    print("="*80)
    print("This mode allows you to project hypothetical future seasons and see")
    print("how they change the comp analysis and payment probability.")
    
    # Check required files exist
    required_files = ['qb_seasons_payment_labeled_era_adjusted.csv']
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"\n✗ ERROR: Required file not found: {filepath}")
            return
    
    # Load QB lookup
    print("\nLoading QB database...")
    lookup = load_qb_id_lookup()
    
    if not lookup:
        print("✗ ERROR: Could not create QB lookup table")
        return
    
    print(f"✓ Loaded {len(lookup)} QBs")
    
    # Get QB
    print("\n" + "="*80)
    print("ENTER QB NAME")
    print("="*80)
    
    search_term = input("\nQB Name: ").strip()
    
    if not search_term:
        print("No name entered. Exiting.")
        return
    
    matches = search_qb_by_name(search_term, lookup)
    
    if not matches:
        print(f"\n✗ No QBs found matching '{search_term}'")
        return
    
    if len(matches) > 1:
        print(f"\n✓ Found {len(matches)} QBs:")
        for idx, (name, player_id) in enumerate(matches, 1):
            print(f"  {idx}. {name} ({player_id})")
        
        choice = input(f"\nSelect QB (1-{len(matches)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            target_name, target_id = matches[choice_idx]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return
    else:
        target_name, target_id = matches[0]
        print(f"\n✓ Found: {target_name} ({target_id})")
    
    # Show existing data
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['years_since_draft'] = df['season'] - df['draft_year']
    
    qb_data = df[df['player_id'] == target_id].sort_values('years_since_draft')
    
    if len(qb_data) == 0:
        print(f"\n✗ ERROR: No data found for {target_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"EXISTING DATA FOR {target_name}")
    print('='*80)
    
    for _, row in qb_data.iterrows():
        year = int(row['years_since_draft'])
        season = int(row['season'])
        yards = row['total_yards_adj']
        anya = row['Pass_ANY/A_adj']
        
        yards_str = f"{yards:,.0f}" if pd.notna(yards) else "N/A"
        anya_str = f"{anya:.2f}" if pd.notna(anya) else "N/A"
        
        print(f"  Year {year} (Season {season}): Yards={yards_str}, ANY/A={anya_str}")
    
    latest_year = int(qb_data['years_since_draft'].max())
    
    # Get projection inputs
    print(f"\n{'='*80}")
    print("PROJECTION INPUTS")
    print('='*80)
    print(f"Enter projected performance for future years (after Year {latest_year})")
    print("You can project ANY/A, total yards, or both")
    print("Leave blank to skip a metric or year")
    
    # Get decision year
    decision_year_input = input(f"\nDecision year to evaluate at (default={latest_year+2}): ").strip()
    if decision_year_input:
        try:
            decision_year = int(decision_year_input)
        except ValueError:
            decision_year = latest_year + 2
    else:
        decision_year = latest_year + 2
    
    # Determine which years to project
    years_to_project = list(range(latest_year + 1, decision_year))
    
    if not years_to_project:
        print(f"\n✗ ERROR: No years to project (latest={latest_year}, decision={decision_year})")
        return
    
    print(f"\nYears to project: {years_to_project}")
    
    # Collect projections
    projected_anya = []
    projected_yards = []
    
    for year in years_to_project:
        print(f"\n--- Year {year} ---")
        
        anya_input = input(f"  ANY/A (leave blank to skip): ").strip()
        if anya_input:
            try:
                projected_anya.append(float(anya_input))
            except ValueError:
                print("  Invalid input, skipping ANY/A")
                projected_anya.append(None)
        else:
            projected_anya.append(None)
        
        yards_input = input(f"  Total Yards Adj (leave blank to skip): ").strip()
        if yards_input:
            try:
                projected_yards.append(float(yards_input))
            except ValueError:
                print("  Invalid input, skipping yards")
                projected_yards.append(None)
        else:
            projected_yards.append(None)
    
    # Build projection data dict
    projected_data = {'years': years_to_project}
    
    # Only include metrics that have at least one value
    if any(x is not None for x in projected_anya):
        projected_data['Pass_ANY/A_adj'] = projected_anya
    
    if any(x is not None for x in projected_yards):
        projected_data['total_yards_adj'] = projected_yards
    
    if 'Pass_ANY/A_adj' not in projected_data and 'total_yards_adj' not in projected_data:
        print("\n✗ ERROR: No projections entered")
        return
    
    # Get number of comps
    n_comps = 10
    n_comps_input = input(f"\nNumber of comps to find (default={n_comps}): ").strip()
    if n_comps_input:
        try:
            n_comps = int(n_comps_input)
        except ValueError:
            pass
    
    # Load weights
    yards_weights = None
    anya_weights = None
    
    yards_file = 'year_weights_total_yards_adj.csv'
    anya_file = 'year_weights_Pass_ANY_A_adj.csv'
    
    if os.path.exists(yards_file):
        yards_df = pd.read_csv(yards_file)
        yards_subset = yards_df[yards_df['Decision_Year'] == decision_year]
        if len(yards_subset) > 0:
            yards_weights = {}
            for _, row in yards_subset.iterrows():
                year_num = int(row['Performance_Year'].split()[-1])
                yards_weights[year_num] = row['Weight_%'] / 100.0
    
    if os.path.exists(anya_file):
        anya_df = pd.read_csv(anya_file)
        anya_subset = anya_df[anya_df['Decision_Year'] == decision_year]
        if len(anya_subset) > 0:
            anya_weights = {}
            for _, row in anya_subset.iterrows():
                year_num = int(row['Performance_Year'].split()[-1])
                anya_weights[year_num] = row['Weight_%'] / 100.0
    
    # Run comp analysis with projection
    print(f"\n{'='*80}")
    print(f"RUNNING PROJECTED COMP ANALYSIS")
    print('='*80)
    
    results = run_comps_with_projection(
        target_qb_id=target_id,
        target_qb_name=target_name,
        projected_data=projected_data,
        decision_year=decision_year,
        n_comps=n_comps,
        yards_weights=yards_weights,
        anya_weights=anya_weights
    )
    
    if results is None:
        print("\n✗ ERROR: Projection analysis failed")
        return
    
    # Export results
    export_comp_analysis_for_tableau(
        target_qb_id=target_id,
        target_qb_name=target_name,
        yards_comps=results['yards_comps'],
        anya_comps=results['anya_comps'],
        decision_year=decision_year,
        output_dir='comp_analysis_output',
        is_projection=True
    )
    
    print("\n✓ Projection analysis complete!")


def run_single_qb_comp(target_id, target_name, decision_year=4, n_comps=10):
    """
    Run comp analysis for a single QB and save to their QB-specific folder.
    
    Args:
        target_id: Player ID
        target_name: Player name
        decision_year: Which year to evaluate at (4 = after Year 3)
        n_comps: Number of comps to return per metric
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Load year weights if available
        yards_weights = None
        anya_weights = None
        
        yards_file = 'year_weights_total_yards_adj.csv'
        anya_file = 'year_weights_Pass_ANY_A_adj.csv'
        
        if os.path.exists(yards_file):
            yards_df = pd.read_csv(yards_file)
            yards_subset = yards_df[yards_df['Decision_Year'] == decision_year]
            if len(yards_subset) > 0:
                yards_weights = {}
                for _, row in yards_subset.iterrows():
                    year_num = int(row['Performance_Year'].split()[-1])
                    yards_weights[year_num] = row['Weight_%'] / 100.0
        
        if os.path.exists(anya_file):
            anya_df = pd.read_csv(anya_file)
            anya_subset = anya_df[anya_df['Decision_Year'] == decision_year]
            if len(anya_subset) > 0:
                anya_weights = {}
                for _, row in anya_subset.iterrows():
                    year_num = int(row['Performance_Year'].split()[-1])
                    anya_weights[year_num] = row['Weight_%'] / 100.0
        
        # Run comp analysis
        from QB_research import find_comps_both_metrics
        
        results = find_comps_both_metrics(
            target_qb_id=target_id,
            decision_year=decision_year,
            n_comps=n_comps,
            year_weights_yards=yards_weights,
            year_weights_anya=anya_weights
        )
        
        if results is None or results['yards_comps'] is None:
            print(f"  ⚠️  Insufficient data for {target_name}")
            return False
        
        # Export to QB-specific folder
        export_comp_analysis_for_tableau(
            target_qb_id=target_id,
            target_qb_name=target_name,
            yards_comps=results['yards_comps'],
            anya_comps=results['anya_comps'],
            decision_year=decision_year,
            is_projection=False
        )
        
        print(f"  ✓ Success")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def run_all_qbs_comp(decision_year=4, n_comps=10):
    """
    Run comp analysis for ALL QBs in the dataset.
    Each QB's results will be saved in: comp_analysis_output/QBIDxx00/
    
    Args:
        decision_year: Which year to evaluate at (4 = after Year 3)
        n_comps: Number of comps to return per metric
    """
    print("\n" + "="*80)
    print("BATCH COMP ANALYSIS - ALL QBs")
    print("="*80)
    print(f"Decision Year: {decision_year}")
    print(f"Comps per metric: {n_comps}")
    print(f"Output structure: comp_analysis_output/[QB_ID]/")
    
    # Load QB list
    data_file = 'qb_seasons_payment_labeled_era_adjusted.csv'
    if not os.path.exists(data_file):
        print(f"\n✗ ERROR: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    
    # Calculate years_since_draft
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['years_since_draft'] = df['season'] - df['draft_year']
    
    # Only include QBs who have at least decision_year-1 years of data
    qb_max_years = df.groupby('player_id')['years_since_draft'].max()
    eligible_qbs = qb_max_years[qb_max_years >= (decision_year - 1)].index.tolist()
    
    print(f"\nTotal QBs in dataset: {df['player_id'].nunique()}")
    print(f"QBs with sufficient data (≥{decision_year-1} years): {len(eligible_qbs)}")
    
    # Track results
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, qb_id in enumerate(eligible_qbs, 1):
        qb_data = df[df['player_id'] == qb_id]
        qb_name = qb_data.iloc[0]['player_name']
        
        print(f"\n[{idx}/{len(eligible_qbs)}] {qb_name} ({qb_id})")
        
        success = run_single_qb_comp(
            target_id=qb_id,
            target_name=qb_name,
            decision_year=decision_year,
            n_comps=n_comps
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Eligible QBs: {len(eligible_qbs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: comp_analysis_output/[QB_ID]/")
    print(f"  - Each QB has their own folder")
    print(f"  - Files: comp_summary, yards_comps, anya_comps, trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='QB Trajectory Comp Analysis Tool'
    )
    parser.add_argument(
        '-p', '--projection',
        action='store_true',
        help='Run in projection mode to analyze hypothetical future performance'
    )
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Run comp analysis for ALL QBs in batch mode'
    )
    parser.add_argument(
        '-d', '--decision-year',
        type=int,
        default=4,
        help='Decision year for analysis (default: 4)'
    )
    parser.add_argument(
        '-n', '--n-comps',
        type=int,
        default=10,
        help='Number of comps per metric (default: 10)'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        run_all_qbs_comp(
            decision_year=args.decision_year,
            n_comps=args.n_comps
        )
    elif args.projection:
        run_projection_mode()
    else:
        run_interactive_comp_analysis()