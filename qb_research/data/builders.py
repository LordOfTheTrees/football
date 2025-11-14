"""
Data builder functions for creating core data files.

This module contains functions for:
- Creating all_seasons_df.csv from QB_Data files
- Creating player_ids.csv from QB data
- Creating train/test splits
"""

import pandas as pd
import numpy as np
import os
import glob

from qb_research.utils.data_loading import load_csv_safe


def create_all_seasons_from_existing_qb_files():
    """Create all_seasons_df.csv from existing QB_Data/*.csv files (no web pulling)"""
    
    # Check if QB_Data directory exists
    if not os.path.exists('QB_Data'):
        print("✗ QB_Data directory not found")
        return False
    
    # Find all QB CSV files
    qb_files = glob.glob('QB_Data/*.csv')
    if not qb_files:
        print("✗ No QB files found in QB_Data/ directory")
        return False
    
    print(f"Found {len(qb_files)} QB files to process")
    
    all_seasons = []
    
    for file_path in qb_files:
        try:
            df = pd.read_csv(file_path)
            
            # Extract player_id from filename (e.g., 'AlleJo02.csv' -> 'AlleJo02')
            player_id = os.path.basename(file_path).replace('.csv', '')
            
            # Add player_id if not present
            if 'player_id' not in df.columns:
                df['player_id'] = player_id
            
            all_seasons.append(df)
            print(f"  ✓ Loaded {len(df)} seasons for {player_id}")
            
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")
    
    if not all_seasons:
        return False
    
    # Combine all QB data
    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_df.to_csv('all_seasons_df.csv', index=False)
    print(f"✓ Created all_seasons_df.csv with {len(combined_df)} total seasons")
    
    # Create best_seasons_df.csv (top season per player by total_yards)
    if 'total_yards' in combined_df.columns:
        # Convert to numeric and handle NaN
        combined_df['total_yards'] = pd.to_numeric(combined_df['total_yards'], errors='coerce')
        combined_df = combined_df.dropna(subset=['total_yards'])
        
        if len(combined_df) > 0:
            best_seasons = combined_df.loc[combined_df.groupby('player_id')['total_yards'].idxmax()]
            best_seasons.to_csv('best_seasons_df.csv', index=False)
            print(f"✓ Created best_seasons_df.csv with {len(best_seasons)} best seasons")
    
    return True


def create_player_ids_from_qb_data():
    """Create player_ids.csv from QB data (no web pulling)"""
    
    if not os.path.exists('all_seasons_df.csv'):
        return False
    
    df = pd.read_csv('all_seasons_df.csv')
    
    # Extract unique player info
    player_info = df.groupby('player_id').agg({
        'player_name': 'first',
        'season': 'min'  # Use first season as 'year'
    }).reset_index()
    
    player_info = player_info.rename(columns={'season': 'year'})
    player_info.to_csv('player_ids.csv', index=False)
    print(f"✓ Created player_ids.csv with {len(player_info)} players")
    
    return True


def create_train_test_split(test_size=0.2, random_state=42, split_by='random'):
    """
    Creates train and test splits from all_seasons_df.csv
    
    Args:
        test_size (float): Proportion of data for test set (default 0.2 = 20%)
        random_state (int): Random seed for reproducibility
        split_by (str): How to split the data:
            - 'random': Random split across all seasons
            - 'temporal': Split by year (older years = train, recent years = test)
            - 'player': Split by player (some players in train, others in test)
    
    Returns:
        tuple: (train_df, test_df)
    """
    print("\n" + "="*80)
    print(f"CREATING TRAIN/TEST SPLIT (split_by='{split_by}')")
    print("="*80)
    
    # Load the data
    try:
        all_seasons = load_csv_safe("all_seasons_df.csv")
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        return None, None
    
    print(f"\nTotal records: {len(all_seasons)}")
    
    # Clean the data
    all_seasons = all_seasons[all_seasons['Team'].str.len() == 3]
    all_seasons = all_seasons[~all_seasons['Team'].str.contains('Did not', na=False)]
    
    print(f"After cleaning: {len(all_seasons)}")
    
    if split_by == 'random':
        # Random shuffle and split
        np.random.seed(random_state)
        shuffled = all_seasons.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        split_idx = int(len(shuffled) * (1 - test_size))
        train_df = shuffled[:split_idx]
        test_df = shuffled[split_idx:]
        
        print(f"\nRandom split:")
        print(f"  Train: {len(train_df)} records ({len(train_df)/len(all_seasons)*100:.1f}%)")
        print(f"  Test: {len(test_df)} records ({len(test_df)/len(all_seasons)*100:.1f}%)")
        
    elif split_by == 'temporal':
        # Split by year - use most recent years for testing
        all_seasons['season'] = pd.to_numeric(all_seasons['season'], errors='coerce')
        all_seasons = all_seasons.dropna(subset=['season'])
        all_seasons = all_seasons.sort_values('season')
        
        split_idx = int(len(all_seasons) * (1 - test_size))
        train_df = all_seasons.iloc[:split_idx]
        test_df = all_seasons.iloc[split_idx:]
        
        train_years = train_df['season'].unique()
        test_years = test_df['season'].unique()
        
        print(f"\nTemporal split:")
        print(f"  Train: {len(train_df)} records from {int(train_years.min())}-{int(train_years.max())}")
        print(f"  Test: {len(test_df)} records from {int(test_years.min())}-{int(test_years.max())}")
        
    elif split_by == 'player':
        # Split by player - some players entirely in train, others in test
        unique_players = all_seasons['player_id'].unique()
        np.random.seed(random_state)
        shuffled_players = np.random.permutation(unique_players)
        
        split_idx = int(len(shuffled_players) * (1 - test_size))
        train_players = shuffled_players[:split_idx]
        test_players = shuffled_players[split_idx:]
        
        train_df = all_seasons[all_seasons['player_id'].isin(train_players)]
        test_df = all_seasons[all_seasons['player_id'].isin(test_players)]
        
        print(f"\nPlayer-based split:")
        print(f"  Train: {len(train_df)} records from {len(train_players)} players")
        print(f"  Test: {len(test_df)} records from {len(test_players)} players")
        
    else:
        print(f"Unknown split_by method: {split_by}")
        return None, None
    
    # Save to CSV
    train_df.to_csv("all_seasons_df_train.csv", index=False)
    test_df.to_csv("all_seasons_df_test.csv", index=False)
    
    print("\nSaved:")
    print("  - all_seasons_df_train.csv")
    print("  - all_seasons_df_test.csv")
    
    return train_df, test_df

