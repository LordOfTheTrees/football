"""
QB trajectory matching and comparison functions.

This module contains functions for finding similar QBs based on performance trajectories.
"""

import pandas as pd
import numpy as np
import os

from qb_research.utils.data_loading import load_csv_safe


def find_most_similar_qbs(
    target_qb_id,
    decision_year=4,
    metric='total_yards_adj',
    n_comps=5,
    year_weights=None
):
    """
    Finds QBs with most similar trajectories - FIXED to handle insufficient data.
    
    Key fix: Uses available data instead of requiring full decision_year.
    
    Args:
        target_qb_id: QB to find comps for
        decision_year: Desired evaluation year (will use less if needed)
        metric: 'total_yards_adj' or 'Pass_ANY/A_adj'
        n_comps: Number of similar QBs to return
        year_weights: Optional weights {0: weight, 1: weight, ...}
    
    Returns:
        DataFrame: Top N similar QBs with similarity scores and outcomes
        None: If no data available for target QB
    """
    # Load era-adjusted data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        return None
    
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Calculate years since draft
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    df = df.dropna(subset=['season', 'draft_year'])
    df['years_since_draft'] = df['season'] - df['draft_year']
    
    # Get target QB's trajectory
    target_data = df[df['player_id'] == target_qb_id].sort_values('years_since_draft')
    
    if len(target_data) == 0:
        print(f"✗ ERROR: No data found for QB ID: {target_qb_id}")
        return None
    
    # Convert metric to numeric
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    target_data = target_data.copy()
    target_data[metric] = pd.to_numeric(target_data[metric], errors='coerce')
    
    # Get target trajectory - use ALL available years up to decision_year
    target_trajectory_full = target_data[metric].dropna().values
    target_name = target_data.iloc[0]['player_name']
    
    # FIX: Determine actual years available (correctly)
    target_years_available = len(target_trajectory_full)
    
    if target_years_available == 0:
        print(f"✗ ERROR: No valid data for {target_name}")
        return None
    
    # Use minimum of (available years, decision_year)
    comparison_years = min(target_years_available, decision_year)
    target_trajectory = target_trajectory_full[:comparison_years]
    
    print(f"\nTarget QB: {target_name} ({target_qb_id})")
    print(f"Available data: {target_years_available} year(s)")
    print(f"Using: {comparison_years} year(s) for comparison")
    
    # Truncate ALL QB data to the comparison window
    df_truncated = df[df['years_since_draft'] < comparison_years].copy()
    
    # Set up year weights
    if year_weights is None:
        year_weights = {i: 1.0/comparison_years for i in range(comparison_years)}
    else:
        # Adapt provided weights to available years
        adapted_weights = {}
        total_weight = 0
        for year in range(comparison_years):
            adapted_weights[year] = year_weights.get(year, 1.0/comparison_years)
            total_weight += adapted_weights[year]
        year_weights = {k: v/total_weight for k, v in adapted_weights.items()}
    
    # Calculate similarity to all other QBs
    similarities = []
    
    for player_id in df['player_id'].unique():
        if player_id == target_qb_id:
            continue
        
        # Skip QBs still in decision window (less than 4 years since draft)
        player_full_data = df[df['player_id'] == player_id]
        if len(player_full_data) > 0 and player_full_data['years_since_draft'].max() < 4:
            continue
        
        player_data = df_truncated[df_truncated['player_id'] == player_id].sort_values('years_since_draft')
        
        if len(player_data) == 0:
            continue
        
        player_trajectory = player_data[metric].dropna().values
        
        # Need at least the comparison window
        if len(player_trajectory) < comparison_years:
            continue
        
        # Use only the comparison window
        player_trajectory = player_trajectory[:comparison_years]
        
        # Calculate weighted Euclidean distance
        distance = 0
        for year in range(comparison_years):
            weight = year_weights.get(year, 1.0/comparison_years)
            distance += weight * (target_trajectory[year] - player_trajectory[year])**2
        
        distance = np.sqrt(distance)
        
        # Get outcome info
        got_paid = player_data.iloc[0]['got_paid']
        payment_year = player_data.iloc[0].get('payment_year', np.nan)
        draft_year = player_data.iloc[0].get('draft_year', np.nan)
        
        # Get the most recent year value
        year_value = player_trajectory[-1]
        
        similarities.append({
            'player_id': player_id,
            'player_name': player_data.iloc[0]['player_name'],
            'similarity_score': distance,
            f'year_{comparison_years-1}_{metric}': year_value,
            'got_paid': got_paid,
            'payment_year': payment_year,
            'draft_year': draft_year
        })
    
    # Sort by similarity
    results_df = pd.DataFrame(similarities).sort_values('similarity_score')
    
    if len(results_df) == 0:
        print(f"✗ ERROR: No comparable QBs found")
        return None
    
    # Return top N
    return results_df.head(n_comps)


def find_comps_both_metrics(
    target_qb_id,
    decision_year=4,
    n_comps=5,
    year_weights_yards=None,
    year_weights_anya=None
):
    """
    Convenience function to run comp analysis for BOTH metrics.
    
    Args:
        target_qb_id: QB to find comps for
        decision_year: Which year to evaluate at (4 = after Year 3)
        n_comps: Number of comps to return per metric
        year_weights_yards: Weights for total_yards_adj
        year_weights_anya: Weights for Pass_ANY/A_adj
    
    Returns:
        dict: {'yards_comps': DataFrame, 'anya_comps': DataFrame}
    """
    print("\n" + "🏈"*35)
    print(f"COMPREHENSIVE COMP ANALYSIS")
    print("🏈"*35)
    
    # Get QB name for display
    df = pd.read_csv('qb_seasons_payment_labeled_era_adjusted.csv')
    qb_data = df[df['player_id'] == target_qb_id]
    if len(qb_data) > 0:
        qb_name = qb_data.iloc[0]['player_name']
        print(f"Target QB: {qb_name} ({target_qb_id})")
        print(f"Decision Year: {decision_year}")
    
    # Run for yards
    yards_comps = find_most_similar_qbs(
        target_qb_id=target_qb_id,
        decision_year=decision_year,
        metric='total_yards_adj',
        n_comps=n_comps,
        year_weights=year_weights_yards
    )
    
    print("\n")
    
    # Run for ANY/A
    anya_comps = find_most_similar_qbs(
        target_qb_id=target_qb_id,
        decision_year=decision_year,
        metric='Pass_ANY/A_adj',
        n_comps=n_comps,
        year_weights=year_weights_anya
    )
    
    return {
        'yards_comps': yards_comps,
        'anya_comps': anya_comps
    }


def batch_comp_analysis(qb_list, decision_year=4, n_comps=5, use_year_weights=True):
    """
    Run comp analysis for multiple QBs at once.
    
    Args:
        qb_list: List of (qb_id, qb_name) tuples
        decision_year: Which year to evaluate at
        n_comps: Number of comps per metric
        use_year_weights: If True, load weights from year_weighting_regression results
    
    Returns:
        dict: {qb_name: {'yards_comps': df, 'anya_comps': df}}
    """
    print("\n" + "="*80)
    print("BATCH COMP ANALYSIS")
    print("="*80)
    
    # Load year weights if requested
    yards_dict = None
    anya_dict = None
    
    if use_year_weights:
        print("\nLoading year weights from previous regression results...")
        
        # Try to load from year_weighting_regression output files
        yards_file = 'year_weights_total_yards_adj.csv'
        anya_file = 'year_weights_Pass_ANY_A_adj.csv'
        
        if os.path.exists(yards_file):
            yards_df = pd.read_csv(yards_file)
            yards_df = yards_df[yards_df['Decision_Year'] == decision_year]
            if len(yards_df) > 0:
                yards_dict = {}
                for _, row in yards_df.iterrows():
                    year_str = row['Performance_Year']
                    year_num = int(year_str.split()[-1])
                    yards_dict[year_num] = row['Weight_%'] / 100.0
                print(f"✓ Loaded total_yards weights from {yards_file}")
            else:
                print(f"⚠️  No weights for decision year {decision_year} in {yards_file}")
        else:
            print(f"⚠️  {yards_file} not found, using uniform weights for yards")
        
        if os.path.exists(anya_file):
            anya_df = pd.read_csv(anya_file)
            anya_df = anya_df[anya_df['Decision_Year'] == decision_year]
            if len(anya_df) > 0:
                anya_dict = {}
                for _, row in anya_df.iterrows():
                    year_str = row['Performance_Year']
                    year_num = int(year_str.split()[-1])
                    anya_dict[year_num] = row['Weight_%'] / 100.0
                print(f"✓ Loaded ANY/A weights from {anya_file}")
            else:
                print(f"⚠️  No weights for decision year {decision_year} in {anya_file}")
        else:
            print(f"⚠️  {anya_file} not found, using uniform weights for ANY/A")
    
    # Run analysis for each QB
    all_results = {}
    
    for qb_id, qb_name in qb_list:
        print(f"\n\n{'#'*80}")
        print(f"ANALYZING: {qb_name} (ID: {qb_id})")
        print('#'*80)
        
        results = find_comps_both_metrics(
            target_qb_id=qb_id,
            decision_year=decision_year,
            n_comps=n_comps,
            year_weights_yards=yards_dict,
            year_weights_anya=anya_dict
        )
        
        all_results[qb_name] = results
    
    return all_results

