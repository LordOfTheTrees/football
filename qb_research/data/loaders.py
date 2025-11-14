"""
Data loading functions for QB research package.

This module provides functions to load various QB-related data files
including contracts, player IDs, and train/test splits.
"""

import os
import pandas as pd
from IPython.display import display
from qb_research.utils.data_loading import load_csv_safe


def load_contract_data():
    """
    Loads and cleans the QB contract data.
    
    Returns:
        DataFrame: Cleaned contract data with standardized column names
    """
    print("\n" + "="*80)
    print("LOADING CONTRACT DATA")
    print("="*80)
    
    try:
        contracts = load_csv_safe("QB_contract_data.csv")
        print(f"Loaded {len(contracts)} contract records")
        
        # Standardize column names
        contracts.columns = contracts.columns.str.strip()
        
        # Show sample
        print("\nSample contract data:")
        display(contracts.head(3)[['Player', 'Team', 'Year', 'Value', 'APY', 'Guaranteed']])
        
        # Convert year to int
        contracts['Year'] = pd.to_numeric(contracts['Year'], errors='coerce')
        contracts = contracts.dropna(subset=['Year'])
        contracts['Year'] = contracts['Year'].astype(int)
        
        # Convert financial columns to numeric
        for col in ['Value', 'APY', 'Guaranteed']:
            if col in contracts.columns:
                contracts[col] = pd.to_numeric(contracts[col], errors='coerce')
        
        print(f"\nContract years range: {contracts['Year'].min()}-{contracts['Year'].max()}")
        print(f"Unique players: {contracts['Player'].nunique()}")
        
        return contracts
        
    except FileNotFoundError:
        print("ERROR: QB_contract_data.csv not found")
        return None


def load_first_round_qbs_with_ids():
    """
    Loads the first round QB data with player IDs.
    Tries multiple possible file locations and names.
    
    Returns:
        DataFrame: First round QBs with player_name, player_id, draft_year, draft_team
    """
    print("\n" + "="*80)
    print("LOADING FIRST ROUND QB DATA WITH PLAYER IDs")
    print("="*80)
    
    # Try different possible files
    possible_files = [
        "1st_rd_qb_ids.csv",
        "first_round_qbs.csv",
        "QB_Data/1st_rd_qb_ids.csv"
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"\nFound file: {filepath}")
            qbs = load_csv_safe(filepath)
            print(f"Loaded {len(qbs)} records")
            print(f"Columns: {qbs.columns.tolist()}")
            
            # Show sample
            print("\nSample data:")
            display(qbs.head(3))
            
            # Check for player_id
            if 'player_id' in qbs.columns:
                print("\n✓ player_id column found")
                
                # Standardize column names if needed
                if 'name' in qbs.columns and 'player_name' not in qbs.columns:
                    qbs = qbs.rename(columns={'name': 'player_name'})
                
                return qbs
            else:
                print(f"⚠️  player_id not found in {filepath}, trying next file...")
    
    print("\n✗ ERROR: Could not find file with player_id column")
    print("\nYou need to:")
    print("  1. Make sure 1st_rd_qb_ids.csv exists (created by PFR_Tools.update_qb_ids())")
    print("  2. Or run: PFR.update_qb_ids() to create it")
    
    return None


def load_train_test_split():
    """
    Loads previously created train/test split files.
    
    Returns:
        tuple: (train_df, test_df)
    """
    try:
        train_df = load_csv_safe("all_seasons_df_train.csv")
        test_df = load_csv_safe("all_seasons_df_test.csv")
        
        print(f"Loaded train set: {len(train_df)} records")
        print(f"Loaded test set: {len(test_df)} records")
        
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Train/test files not found: {e}")
        print("Please run create_train_test_split() first")
        return None, None

