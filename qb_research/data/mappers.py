"""
Data mapping functions for QB research package.

This module provides functions to map contracts to player IDs, create
payment year mappings, pick number mappings, and filter contracts.
"""

import os
import pandas as pd
from qb_research.utils.data_loading import load_csv_safe, validate_columns
from qb_research.utils.name_matching import normalize_player_name
from qb_research.data.loaders import load_contract_data


def filter_contracts_to_first_round_qbs(mapped_contracts_df, first_round_qbs_file='first_round_qbs_with_picks.csv'):
    """
    Filters the full contract mapping to only first-round QBs.
    
    Args:
        mapped_contracts_df: Full contract data with player_ids
        first_round_qbs_file: CSV with first-round QB list
    
    Returns:
        DataFrame: Contracts filtered to only first-round QBs
    """
    print("\n" + "="*80)
    print("FILTERING TO FIRST-ROUND QBs")
    print("="*80)
    
    # Load first-round QB list
    first_round = load_csv_safe(first_round_qbs_file, "first-round QBs")
    if first_round is None:
        return None
    
    if 'player_id' not in first_round.columns:
        print("✗ ERROR: first-round QB file missing player_id column")
        return None
    
    first_round_ids = set(first_round['player_id'].unique())
    print(f"First-round QBs: {len(first_round_ids)} players")
    
    # Filter
    filtered = mapped_contracts_df[mapped_contracts_df['player_id'].isin(first_round_ids)].copy()
    
    print(f"\nFiltered contracts:")
    print(f"  Before: {len(mapped_contracts_df)} contracts")
    print(f"  After: {len(filtered)} contracts ({len(filtered)/len(mapped_contracts_df)*100:.1f}%)")
    print(f"  Unique first-round QBs with contracts: {filtered['player_id'].nunique()}")
    
    return filtered


def map_contract_to_player_ids(contracts_df, player_ids_df, cache_file='cache/contract_player_id_mapping.csv', force_refresh=False):
    """
    Maps contract data to player IDs from the full player ID list.
    Uses exact string matching after normalization.
    
    Args:
        contracts_df: Contract data with 'Player' column
        player_ids_df: Player data with 'player_name' and 'player_id' columns
        cache_file: Where to cache the mapping
        force_refresh: Force recreation of mapping
    
    Returns:
        DataFrame: Contract data with added 'player_id' and 'match_type' columns
    """
    print("\n" + "="*80)
    print("MAPPING CONTRACTS TO PLAYER IDs")
    print("="*80)
    
    # Check cache first
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached mapping from: {cache_file}")
        return load_csv_safe(cache_file)
    
    # Normalize names in both datasets
    contracts_df['Player_normalized'] = contracts_df['Player'].apply(normalize_player_name)
    player_ids_df['player_name_normalized'] = player_ids_df['player_name'].apply(normalize_player_name)
    
    # Create lookup dictionary for fast matching
    name_to_id = dict(zip(player_ids_df['player_name_normalized'], player_ids_df['player_id']))
    name_to_original = dict(zip(player_ids_df['player_name_normalized'], player_ids_df['player_name']))
    
    print(f"\nAttempting to match {len(contracts_df)} contracts to {len(player_ids_df)} players...")
    
    # Perform exact matching
    contracts_df['player_id'] = contracts_df['Player_normalized'].map(name_to_id)
    contracts_df['matched_name'] = contracts_df['Player_normalized'].map(name_to_original)
    contracts_df['match_type'] = contracts_df['player_id'].notna().map({True: 'exact', False: 'no_match'})
    
    # Print matching statistics
    print("\n" + "="*80)
    print("MATCHING RESULTS")
    print("="*80)
    
    matched = contracts_df['player_id'].notna().sum()
    total = len(contracts_df)
    
    print(f"\nMatched: {matched}/{total} ({matched/total*100:.1f}%)")
    print(f"  Exact matches: {matched}")
    print(f"  No match: {total - matched}")
    
    # Show unmatched contracts
    unmatched = contracts_df[contracts_df['player_id'].isna()]
    if len(unmatched) > 0:
        print(f"\nUnmatched contracts ({len(unmatched)}):")
        for _, row in unmatched.head(10).iterrows():
            print(f"  - {row['Player']} ({row['Team']}, {row['Year']})")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    
    # Drop temporary normalized column
    result_df = contracts_df.drop(columns=['Player_normalized'])
    
    # Cache the results
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    result_df.to_csv(cache_file, index=False)
    print(f"\nMapping cached to: {cache_file}")
    
    return result_df


def create_contract_player_mapping(force_refresh=False):
    """
    Main pipeline function to create contract-to-player-ID mapping.
    Uses the FULL player_ids.csv list (not just first-round QBs).
    Uses caching to avoid recomputation.
    
    Args:
        force_refresh (bool): Force recreation of all mappings
    
    Returns:
        DataFrame: Contracts with player_id mapping using only players who have had regular season football games (fantasy list).
    """
    print("\n" + "="*80)
    print("CONTRACT-TO-PLAYER MAPPING PIPELINE")
    print("="*80)
    
    # Create cache directory
    os.makedirs('cache', exist_ok=True)
    
    # Load contract data
    contracts = load_contract_data()
    if contracts is None:
        return None
    
    # Load FULL player ID list (not just first-round QBs)
    print("\n" + "="*80)
    print("LOADING FULL PLAYER ID LIST")
    print("="*80)
    
    player_ids = load_csv_safe("player_ids.csv", "player IDs")
    if player_ids is None:
        print("\n✗ ERROR: player_ids.csv not found")
        print("This file should contain all NFL players with their IDs")
        return None
    
    # Validate player_ids structure
    required_cols = ['player_name', 'player_id']
    if not validate_columns(player_ids, required_cols, "player_ids"):
        return None
    
    print(f"Loaded {len(player_ids)} total players")
    print(f"Unique players: {player_ids['player_id'].nunique()}")
    
    # Show sample
    print("\nSample player IDs:")
    print(player_ids.head(3)[['player_name', 'player_id']].to_string(index=False))
    
    # Perform mapping (with caching)
    mapped_contracts = map_contract_to_player_ids(
        contracts, 
        player_ids, 
        cache_file='cache/contract_player_id_mapping.csv',
        force_refresh=force_refresh
    )
    
    # Note: Validation can be done separately via validate_contract_mapping()
    # which is still in QB_research.py (will be moved to validators.py later)
    # For now, we'll skip validation in this function to avoid circular imports
    
    return mapped_contracts


def create_payment_year_mapping(contract_df):
    """
    Creates mapping of player_id -> payment_year.
    
    Strategy: For each QB, find FIRST contract with draft team 
    that occurs after Year 1 but before Year 7 (Years 2-6 inclusive).
    
    Args:
        contract_df (DataFrame): Contract data with player_id, Year, and Team columns
    
    Returns:
        dict: {player_id: payment_year} for QBs who got 2nd contract with draft team
    """
    print("\n" + "="*80)
    print("CREATING PAYMENT YEAR MAPPING (DRAFT TEAM, YEARS 2-6)")
    print("="*80)
    
    # Validate required columns
    required_cols = ['player_id', 'Year', 'Team']
    if not validate_columns(contract_df, required_cols, "contract data"):
        return {}
    
    # Filter to only contracts with player_id
    paid_contracts = contract_df[contract_df['player_id'].notna()].copy()
    
    print(f"Total contracts: {len(contract_df)}")
    print(f"Contracts with player_id: {len(paid_contracts)}")
    
    if len(paid_contracts) == 0:
        print("⚠️  WARNING: No contracts have player_id mapped!")
        return {}
    
    # Convert Year to numeric
    paid_contracts['Year'] = pd.to_numeric(paid_contracts['Year'], errors='coerce')
    paid_contracts = paid_contracts.dropna(subset=['Year'])
    paid_contracts['Year'] = paid_contracts['Year'].astype(int)
    
    # Load draft years and draft teams
    if os.path.exists('all_seasons_df.csv'):
        qb_seasons = load_csv_safe('all_seasons_df.csv')
        qb_seasons['draft_year'] = pd.to_numeric(qb_seasons['draft_year'], errors='coerce')
        draft_info = qb_seasons.groupby('player_id').agg({
            'draft_year': 'first',
            'draft_team': 'first'
        }).to_dict('index')
        print(f"✓ Loaded draft years and teams for {len(draft_info)} players")
    else:
        print("✗ ERROR: Cannot load draft years from all_seasons_df.csv")
        return {}
    
    # Add draft year and draft team to contracts
    paid_contracts['draft_year'] = paid_contracts['player_id'].map(lambda x: draft_info.get(x, {}).get('draft_year'))
    paid_contracts['draft_team'] = paid_contracts['player_id'].map(lambda x: draft_info.get(x, {}).get('draft_team'))
    paid_contracts = paid_contracts.dropna(subset=['draft_year', 'draft_team'])
    
    print(f"Contracts with valid year, draft_year, and draft_team: {len(paid_contracts)}")
    
    # Normalize team names
    team_mapping = {
        'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
        'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
        'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GNB',
        'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Chiefs': 'KAN',
        'Chargers': 'LAC', 'Rams': 'LAR', 'Raiders': 'LVR', 'Dolphins': 'MIA',
        'Vikings': 'MIN', 'Saints': 'NOR', 'Patriots': 'NWE', 'Giants': 'NYG',
        'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', 'Seahawks': 'SEA',
        '49ers': 'SFO', 'Buccaneers': 'TAM', 'Titans': 'TEN', 'Commanders': 'WAS',
        'Football Team': 'WAS', 'Redskins': 'WAS'
    }
    
    # First, extract signing team from multi-team strings (e.g., "PHI/WAS/IND" -> "PHI")
    paid_contracts['Team_signing'] = paid_contracts['Team'].apply(
        lambda x: x.split('/')[0].strip() if pd.notna(x) and isinstance(x, str) else x
    )

    # Then normalize using the mapping
    paid_contracts['team_normalized'] = paid_contracts['Team_signing'].map(team_mapping).fillna(paid_contracts['Team_signing'])

    # For each player, find FIRST contract with draft team in Years 2-6
    payment_mapping = {}
    excluded_other_team = 0
    excluded_outside_window = 0
    
    # Handle draft-day trades
    draft_day_trade_corrections = {
        'MannEl00': 'NYG',  # Eli Manning: drafted by SDG, traded to NYG
        'RivePh00': 'SDG',  # back half of the swap
    }

    for player_id, corrected_team in draft_day_trade_corrections.items():
        if player_id in paid_contracts['player_id'].values:
            paid_contracts.loc[paid_contracts['player_id'] == player_id, 'draft_team'] = corrected_team
            print(f"  ✓ Corrected draft_team for player {player_id} → {corrected_team}")

    for player_id, group in paid_contracts.groupby('player_id'):
        draft_year = group.iloc[0]['draft_year']
        draft_team = group.iloc[0]['draft_team']
        
        # Filter to contracts with draft team in Years 2-6 (after Year 1, before Year 7)
        eligible_contracts = group[
            (group['Year'] > draft_year + 1) &  # After Year 1
            (group['Year'] < draft_year + 7) &  # Before Year 7
            (group['team_normalized'] == draft_team)
        ].sort_values('Year')
        
        if len(eligible_contracts) > 0:
            # Take the FIRST eligible contract
            payment_year = eligible_contracts.iloc[0]['Year']
            payment_mapping[player_id] = int(payment_year)
        else:
            # Track why excluded
            other_team = group[
                (group['Year'] > draft_year + 1) & 
                (group['Year'] < draft_year + 7) &
                (group['team_normalized'] != draft_team)
            ]
            
            if len(other_team) > 0:
                excluded_other_team += 1
            else:
                excluded_outside_window += 1
    
    print(f"\nPayment mapping created for {len(payment_mapping)} QBs")
    print(f"Excluded (signed with other team): {excluded_other_team}")
    print(f"Excluded (outside Years 2-6 window): {excluded_outside_window}")
    
    return payment_mapping


def create_pick_number_mapping(pick_numbers_file='first_round_qbs_with_picks.csv'):
    """
    Creates a lightweight mapping of player_id -> pick_number.
    
    Args:
        pick_numbers_file (str): Path to CSV with pick numbers
    
    Returns:
        dict: {player_id: pick_number} for all first round QBs
    """
    print("\n" + "="*80)
    print("CREATING PICK NUMBER MAPPING")
    print("="*80)
    
    if not os.path.exists(pick_numbers_file):
        print(f"✗ ERROR: File not found: {pick_numbers_file}")
        return {}
    
    df = load_csv_safe(pick_numbers_file)
    print(f"Loaded {len(df)} QBs with pick numbers")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle column name variations
    if 'player_id' not in df.columns:
        if 'name' in df.columns:
            # Need to map name -> player_id first
            print("⚠️  'player_id' not found, need to map names to IDs...")
            
            # Load player_ids.csv to get the mapping
            player_ids = load_csv_safe("player_ids.csv", "player IDs")
            if player_ids is None:
                print("✗ ERROR: Cannot create pick mapping without player_ids.csv")
                return {}
            
            # Create name -> player_id mapping
            player_ids['player_name_normalized'] = player_ids['player_name'].apply(normalize_player_name)
            df['name_normalized'] = df['name'].apply(normalize_player_name)
            
            # Merge to get player_ids
            df = df.merge(
                player_ids[['player_name_normalized', 'player_id']].drop_duplicates(),
                left_on='name_normalized',
                right_on='player_name_normalized',
                how='left'
            )
            
            missing = df['player_id'].isna().sum()
            if missing > 0:
                print(f"⚠️  Warning: {missing} QBs couldn't be matched to player_id (the player may not have one)")
                print("Missing QBs:")
                for name in df[df['player_id'].isna()]['name'].head(10):
                    print(f"  - {name}")
            
            df = df[df['player_id'].notna()]  # Keep only matched
            print(f"✓ Matched {len(df)} QBs to player_ids")
    
    # Create mapping
    pick_mapping = dict(zip(df['player_id'], df['pick_number']))
    
    print(f"Created pick number mapping for {len(pick_mapping)} QBs")
    
    # Show some examples
    if pick_mapping:
        print("\nExample mappings:")
        for i, (player_id, pick) in enumerate(list(pick_mapping.items())[:5]):
            # Try to get player name
            if 'name' in df.columns:
                player_name = df[df['player_id'] == player_id].iloc[0]['name']
            else:
                player_name = "Unknown"
            print(f"  {player_name} ({player_id}): Pick #{pick}")
    
    return pick_mapping

