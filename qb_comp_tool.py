"""
QB Trajectory Comp Analysis Tool - UPDATED VERSION
Fixed to handle insufficient data + added -all flag for batch processing
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse

# Import the FIXED comparison function
from QB_research import find_most_similar_qbs


def load_qb_id_lookup():
    """
    Creates a lookup dictionary from player names to player IDs.
    
    Returns:
        dict: {player_name_lower: player_id}
    """
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
        
        name_col = None
        id_col = None
        
        if 'player_name' in df.columns and 'player_id' in df.columns:
            name_col = 'player_name'
            id_col = 'player_id'
        elif 'name' in df.columns and 'player_id' in df.columns:
            name_col = 'name'
            id_col = 'player_id'
        
        if name_col and id_col:
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
            display_name = ' '.join(word.capitalize() for word in name_lower.split())
            matches.append((display_name, player_id))
    
    return matches


def export_comp_analysis_for_tableau(
    target_qb_id,
    target_qb_name,
    yards_comps,
    anya_comps,
    decision_year,
    output_dir='comp_analysis_output'
):
    """
    Exports comp analysis results in Tableau-friendly format.
    
    Creates CSV files with all data needed for visualization:
    1. comp_summary.csv
    2. yards_comps.csv
    3. anya_comps.csv
    4. trajectories.csv
    """
    print(f"\nExporting data for Tableau...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load full trajectory data
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df['years_since_draft'] = df['season'] - df['draft_year']
    df = df[df['years_since_draft'] < decision_year]
    
    # Summary
    summary_data = []
    
    if yards_comps is not None:
        summary_data.append({
            'target_qb_id': target_qb_id,
            'target_qb_name': target_qb_name,
            'decision_year': decision_year,
            'metric': 'total_yards_adj',
            'n_comps': len(yards_comps),
            'payment_rate_pct': yards_comps['got_paid'].mean() * 100,
            'comps_paid': yards_comps['got_paid'].sum(),
            'comps_not_paid': (~yards_comps['got_paid']).sum()
        })
    
    if anya_comps is not None:
        summary_data.append({
            'target_qb_id': target_qb_id,
            'target_qb_name': target_qb_name,
            'decision_year': decision_year,
            'metric': 'Pass_ANY/A_adj',
            'n_comps': len(anya_comps),
            'payment_rate_pct': anya_comps['got_paid'].mean() * 100,
            'comps_paid': anya_comps['got_paid'].sum(),
            'comps_not_paid': (~anya_comps['got_paid']).sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f'{target_qb_id}_comp_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Detailed comps
    if yards_comps is not None:
        yards_file = os.path.join(output_dir, f'{target_qb_id}_yards_comps.csv')
        yards_comps.to_csv(yards_file, index=False)
    
    if anya_comps is not None:
        anya_file = os.path.join(output_dir, f'{target_qb_id}_anya_comps.csv')
        anya_comps.to_csv(anya_file, index=False)
    
    # Trajectories
    qb_ids = [target_qb_id]
    if yards_comps is not None:
        qb_ids.extend(yards_comps['player_id'].tolist())
    if anya_comps is not None:
        qb_ids.extend(anya_comps['player_id'].tolist())
    qb_ids = list(set(qb_ids))
    
    # Create comp rank lookups
    yards_rank_lookup = {}
    anya_rank_lookup = {}
    
    if yards_comps is not None:
        for idx, row in yards_comps.reset_index(drop=True).iterrows():
            yards_rank_lookup[row['player_id']] = idx + 1
    
    if anya_comps is not None:
        for idx, row in anya_comps.reset_index(drop=True).iterrows():
            anya_rank_lookup[row['player_id']] = idx + 1
    
    trajectory_rows = []
    for qb_id in qb_ids:
        qb_data = df[df['player_id'] == qb_id].sort_values('years_since_draft')
        if len(qb_data) == 0:
            continue
        
        qb_name = qb_data.iloc[0]['player_name']
        draft_year = qb_data.iloc[0]['draft_year']
        got_paid = qb_data.iloc[0]['got_paid']
        qb_type = 'target' if qb_id == target_qb_id else 'comp'
        
        # Determine comp type and rank
        is_yards_comp = qb_id in yards_rank_lookup
        is_anya_comp = qb_id in anya_rank_lookup
        yards_rank = yards_rank_lookup.get(qb_id, np.nan)
        anya_rank = anya_rank_lookup.get(qb_id, np.nan)
        
        for _, row in qb_data.iterrows():
            trajectory_rows.append({
                'target_qb_id': target_qb_id,
                'target_qb_name': target_qb_name,
                'qb_id': qb_id,
                'qb_name': qb_name,
                'qb_type': qb_type,
                'yards_comp': is_yards_comp,
                'anya_comp': is_anya_comp,
                'yards_comp_rank': yards_rank,
                'anya_comp_rank': anya_rank,
                'decision_year': decision_year,
                'years_since_draft': int(row['years_since_draft']),
                'season': int(row['season']),
                'draft_year': draft_year,
                'total_yards_adj': pd.to_numeric(row['total_yards_adj'], errors='coerce'),
                'Pass_ANY/A_adj': pd.to_numeric(row['Pass_ANY/A_adj'], errors='coerce'),
                'got_paid': got_paid,
                'payment_year': row.get('payment_year', np.nan)
            })
    
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_file = os.path.join(output_dir, f'{target_qb_id}_trajectories.csv')
    trajectory_df.to_csv(trajectory_file, index=False)
    
    print(f"✓ Exported to: {output_dir}/")


def run_single_qb_comp(target_id, target_name, decision_year, n_comps, output_dir):
    """
    Run comp analysis for a single QB.
    Returns True if successful, False otherwise.
    """
    print("\n" + "="*80)
    print(f"RUNNING COMP ANALYSIS: {target_name}")
    print("="*80)
    
    # Run for yards
    print("\n" + "-"*70)
    print("METRIC: Total Yards (Adjusted)")
    print("-"*70)
    yards_comps = find_most_similar_qbs(
        target_qb_id=target_id,
        decision_year=decision_year,
        metric='total_yards_adj',
        n_comps=n_comps
    )
    
    # Run for ANY/A
    print("\n" + "-"*70)
    print("METRIC: ANY/A (Adjusted)")
    print("-"*70)
    anya_comps = find_most_similar_qbs(
        target_qb_id=target_id,
        decision_year=decision_year,
        metric='Pass_ANY/A_adj',
        n_comps=n_comps
    )
    
    # Check if we got results
    if yards_comps is None and anya_comps is None:
        print(f"\n✗ No comps found for {target_name}")
        return False
    
    # Export results
    if output_dir:
        export_comp_analysis_for_tableau(
            target_qb_id=target_id,
            target_qb_name=target_name,
            yards_comps=yards_comps,
            anya_comps=anya_comps,
            decision_year=decision_year,
            output_dir=output_dir
        )
    
    return True


def run_all_qbs_comp(decision_year=4, n_comps=10, output_dir='all_qb_comps'):
    """
    Run comp analysis for ALL QBs in the dataset.
    """
    print("\n" + "="*80)
    print("BATCH COMP ANALYSIS - ALL QBs")
    print("="*80)
    print(f"Decision Year: {decision_year}")
    print(f"Comps per metric: {n_comps}")
    
    # Load QB list
    data_file = 'qb_seasons_payment_labeled_era_adjusted.csv'
    if not os.path.exists(data_file):
        print(f"\n✗ ERROR: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    all_qb_ids = df['player_id'].unique()
    
    print(f"\nProcessing {len(all_qb_ids)} QBs...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track results
    successful = 0
    failed = 0
    
    for idx, qb_id in enumerate(all_qb_ids, 1):
        qb_data = df[df['player_id'] == qb_id]
        qb_name = qb_data.iloc[0]['player_name']
        
        print(f"\n[{idx}/{len(all_qb_ids)}] {qb_name} ({qb_id})")
        
        try:
            qb_output_dir = os.path.join(output_dir, qb_id)
            success = run_single_qb_comp(
                target_id=qb_id,
                target_name=qb_name,
                decision_year=decision_year,
                n_comps=n_comps,
                output_dir=qb_output_dir
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total: {len(all_qb_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir}/")


def run_interactive_comp_analysis():
    """
    Interactive mode - prompts user for QB name.
    """
    print("\n" + "="*80)
    print("QB TRAJECTORY COMP ANALYSIS - INTERACTIVE MODE")
    print("="*80)
    
    # Check required files
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("\n✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        print("Run prepare_qb_payment_data() and create_era_adjusted_payment_data() first")
        return
    
    # Load QB lookup
    print("\nLoading QB database...")
    lookup = load_qb_id_lookup()
    
    if not lookup:
        print("✗ ERROR: Could not create QB lookup table")
        return
    
    print(f"✓ Loaded {len(lookup)} QBs")
    
    # Get QB name
    print("\n" + "="*80)
    print("ENTER QB NAME")
    print("="*80)
    search_term = input("\nQB Name: ").strip()
    
    if not search_term:
        print("No name entered. Exiting.")
        return
    
    # Search for matches
    matches = search_qb_by_name(search_term, lookup)
    
    if not matches:
        print(f"\n✗ No QBs found matching '{search_term}'")
        return
    
    # Handle multiple matches
    if len(matches) > 1:
        print(f"\n✓ Found {len(matches)} QBs:")
        for idx, (name, player_id) in enumerate(matches, 1):
            print(f"  {idx}. {name} ({player_id})")
        
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
    print("  3 = After Year 2 (5th year option)")
    print("  4 = After Year 3 (typical extension)")
    print("  5 = After Year 4 (late extension)")
    print("  6 = After Year 5 (2nd contract)")
    
    decision_year_input = input("\nDecision Year (default=4): ").strip()
    decision_year = int(decision_year_input) if decision_year_input else 4
    
    # Get number of comps
    n_comps_input = input("\nNumber of comps (default=5): ").strip()
    n_comps = int(n_comps_input) if n_comps_input else 5
    
    # Run analysis
    output_dir = f'comp_analysis_output/{target_id}'
    run_single_qb_comp(target_id, target_name, decision_year, n_comps, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QB Comparison Tool')
    parser.add_argument('-all', '--all', action='store_true',
                       help='Run comparison for all QBs in database')
    parser.add_argument('--decision-year', type=int, default=4,
                       help='Decision year for analysis (default: 4)')
    parser.add_argument('--n-comps', type=int, default=5,
                       help='Number of comps to find (default: 5)')
    parser.add_argument('--output-dir', type=str, default='all_qb_comps',
                       help='Output directory for -all mode (default: all_qb_comps)')
    
    args = parser.parse_args()
    
    if args.all:
        # Batch mode
        run_all_qbs_comp(
            decision_year=args.decision_year,
            n_comps=args.n_comps,
            output_dir=args.output_dir
        )
    else:
        # Interactive mode
        run_interactive_comp_analysis()