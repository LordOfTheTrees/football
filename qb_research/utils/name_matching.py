"""
Player name normalization and matching utilities.

This module provides functions for normalizing player names and debugging
name matching issues between different data sources.
"""

import pandas as pd


def normalize_player_name(name):
    """
    Normalizes player names for exact matching.
    Handles common variations (Jr., III, periods, extra spaces, initials, etc.)
    
    Args:
        name (str): Player name
    
    Returns:
        str: Normalized name
    """
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove periods (handles J.J. -> JJ, T.J. -> TJ, etc.)
    name = name.replace('.', '')
    
    # Handle specific initial patterns that might need spaces
    # "JJ McCarthy" should match "J.J. McCarthy" or "J J McCarthy"
    name = name.replace('  ', ' ')  # Collapse double spaces
    
    # Remove suffixes
    suffixes = [' Jr', ' Sr', ' III', ' II', ' IV', ' V']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Lowercase for case-insensitive matching
    name = name.lower()
    
    return name


def debug_name_matching(player_name, contracts_df, player_ids_df):
    """
    Debug why a specific player isn't matching.
    
    Args:
        player_name (str): Name to search for (e.g., "Jalen Hurts")
        contracts_df: Contract dataframe
        player_ids_df: Player IDs dataframe
    """
    print("\n" + "="*80)
    print(f"DEBUGGING: {player_name}")
    print("="*80)
    
    # Check contracts
    contract_matches = contracts_df[contracts_df['Player'].str.contains(player_name, case=False, na=False)]
    print(f"\nIn contracts ({len(contract_matches)} matches):")
    if len(contract_matches) > 0:
        for _, row in contract_matches.iterrows():
            print(f"  Raw: '{row['Player']}'")
            print(f"  Normalized: '{normalize_player_name(row['Player'])}'")
    else:
        print("  Not found")
    
    # Check player_ids
    player_id_matches = player_ids_df[player_ids_df['player_name'].str.contains(player_name, case=False, na=False)]
    print(f"\nIn player_ids ({len(player_id_matches)} matches):")
    if len(player_id_matches) > 0:
        for _, row in player_id_matches.iterrows():
            print(f"  Raw: '{row['player_name']}'")
            print(f"  Normalized: '{normalize_player_name(row['player_name'])}'")
            print(f"  ID: {row['player_id']}")
    else:
        print("  Not found")
    
    # Check if normalized versions match
    if len(contract_matches) > 0 and len(player_id_matches) > 0:
        contract_norm = normalize_player_name(contract_matches.iloc[0]['Player'])
        player_id_norm = normalize_player_name(player_id_matches.iloc[0]['player_name'])
        
        print(f"\nNormalized comparison:")
        print(f"  Contract: '{contract_norm}'")
        print(f"  Player ID: '{player_id_norm}'")
        print(f"  Match: {contract_norm == player_id_norm}")
