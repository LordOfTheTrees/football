"""
QB Trajectory Comp Analysis Tool
Interactive script to find similar QBs based on trajectory matching
"""

import pandas as pd
import numpy as np
import os
import sys

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
    """
    print("\n" + "="*80)
    print("EXPORTING DATA FOR TABLEAU")
    print("="*80)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
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
            'comps_not_paid': (~yards_comps['got_paid']).sum()
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
            'comps_not_paid': (~anya_comps['got_paid']).sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f'{target_qb_id}_comp_summary.csv')
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
        
        yards_file = os.path.join(output_dir, f'{target_qb_id}_yards_comps.csv')
        yards_detailed.to_csv(yards_file, index=False)
        print(f"✓ Saved yards comps: {yards_file}")
    
    if anya_comps is not None:
        anya_detailed = anya_comps.copy()
        anya_detailed['target_qb_id'] = target_qb_id
        anya_detailed['target_qb_name'] = target_qb_name
        anya_detailed['decision_year'] = decision_year
        anya_detailed['metric'] = 'Pass_ANY/A_adj'
        anya_detailed['comp_rank'] = range(1, len(anya_detailed) + 1)
        
        anya_file = os.path.join(output_dir, f'{target_qb_id}_anya_comps.csv')
        anya_detailed.to_csv(anya_file, index=False)
        print(f"✓ Saved ANY/A comps: {anya_file}")
    
    # =========================================================================
    # FILE 4: Year-by-year trajectories
    # =========================================================================
    # Get list of all QBs (target + all comps)
    qb_ids = [target_qb_id]
    
    if yards_comps is not None:
        qb_ids.extend(yards_comps['player_id'].tolist())
    
    if anya_comps is not None:
        qb_ids.extend(anya_comps['player_id'].tolist())
    
    qb_ids = list(set(qb_ids))  # Remove duplicates
    
    # Extract trajectory data for all QBs
    trajectory_rows = []
    
    for qb_id in qb_ids:
        qb_data = df[df['player_id'] == qb_id].sort_values('years_since_draft')
        
        if len(qb_data) == 0:
            continue
        
        qb_name = qb_data.iloc[0]['player_name']
        draft_year = qb_data.iloc[0]['draft_year']
        got_paid = qb_data.iloc[0]['got_paid']
        
        # Determine QB type
        if qb_id == target_qb_id:
            qb_type = 'target'
        else:
            qb_type = 'comp'
        
        # Add comp rank if applicable
        yards_rank = None
        anya_rank = None
        
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
                'Pass_TD_adj': pd.to_numeric(row['Pass_TD_adj'], errors='coerce'),
                'Rush_Rushing_Succ%_adj': pd.to_numeric(row['Rush_Rushing_Succ%_adj'], errors='coerce'),
                'got_paid': got_paid,
                'payment_year': row.get('payment_year', np.nan)
            })
    
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_file = os.path.join(output_dir, f'{target_qb_id}_trajectories.csv')
    trajectory_df.to_csv(trajectory_file, index=False)
    print(f"✓ Saved trajectories: {trajectory_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nFiles created in: {output_dir}/")
    print(f"  1. {target_qb_id}_comp_summary.csv - High-level summary")
    print(f"  2. {target_qb_id}_yards_comps.csv - Yards comp rankings")
    print(f"  3. {target_qb_id}_anya_comps.csv - ANY/A comp rankings")
    print(f"  4. {target_qb_id}_trajectories.csv - Year-by-year data for visualization")
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
        output_dir='comp_analysis_output'
    )
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    run_interactive_comp_analysis()