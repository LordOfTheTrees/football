"""
Injury projection functions for QB research.

This module contains functions for:
- Projecting per-game averages to full 16/17 game seasons
- Creating injury-projected versions of era-adjusted counting stats
- Exporting comparison DataFrames for analysis
"""

import pandas as pd
import numpy as np
import os


def get_season_length(season_year):
    """
    Returns the full season length for a given year.
    
    Args:
        season_year (int): The season year
        
    Returns:
        int: 17 for 2021+, 16 for earlier seasons
    """
    if season_year >= 2021: # Starting in 2021, the NFL expanded to 17 games
        return 17
    else:
        return 16


def calculate_per_game_averages(qb_season_data, stats_to_project, games_started_col='GS'):
    """
    Calculates per-game averages from era-adjusted season totals and games started.
    
    Args:
        qb_season_data (DataFrame or Series): QB season data with era-adjusted stats
        stats_to_project (list): List of stat column names to calculate averages for
        games_started_col (str): Column name for games started (default: 'GS')
        
    Returns:
        dict: Dictionary mapping stat names to per-game averages
    """
    per_game_avgs = {}
    
    # Handle both DataFrame row and Series
    if isinstance(qb_season_data, pd.DataFrame):
        if len(qb_season_data) != 1:
            raise ValueError("calculate_per_game_averages expects a single row DataFrame or Series")
        row = qb_season_data.iloc[0]
    else:
        row = qb_season_data
    
    # Get games started
    gs = pd.to_numeric(row.get(games_started_col), errors='coerce')
    if pd.isna(gs) or gs <= 0:
        return per_game_avgs  # Can't calculate if no games started
    
    # Calculate per-game average for each stat
    for stat in stats_to_project:
        if stat not in row.index:
            continue
            
        stat_value = pd.to_numeric(row[stat], errors='coerce')
        if pd.isna(stat_value) or stat_value <= 0:
            continue
            
        per_game_avg = stat_value / gs
        per_game_avgs[stat] = per_game_avg
    
    return per_game_avgs


def project_to_full_season(per_game_avg, season_year, games_started):
    """
    Projects per-game average to full season length.
    
    Args:
        per_game_avg (float): Per-game average value
        season_year (int): Season year (determines if 16 or 17 games)
        games_started (int): Number of games started
        
    Returns:
        float: Projected value for full season, or original if already full season
    """
    season_length = get_season_length(season_year)
    
    # If already played full season, no projection needed
    if games_started >= season_length:
        return per_game_avg * games_started  # Return original total
    
    # Project to full season
    return per_game_avg * season_length


def fill_missing_seasons(qb_df, stats_to_fill=None, games_started_col='GS', 
                        season_col='season', player_id_col='player_id'):
    """
    Fills in missing seasons (GS=NaN) for players who had complete season skips.
    
    For missing seasons:
    - If season before and after exist: Fill with average of those two seasons
    - If no return (no season after): Set all stats to 0 and mark as final season
    
    Args:
        qb_df: DataFrame with QB seasons
        stats_to_fill: List of era-adjusted stat columns to fill (default: auto-detect _adj columns)
        games_started_col: Column name for games started (default: 'GS')
        season_col: Column name for season year (default: 'season')
        player_id_col: Column name for player ID (default: 'player_id')
        
    Returns:
        DataFrame with missing seasons filled in, plus 'is_final_season' column
    """
    print("\n" + "="*80)
    print("FILLING MISSING SEASONS")
    print("="*80)
    print("Handling complete season skips (injury/conduct policy violations)")
    
    df = qb_df.copy()
    
    # Ensure required columns are numeric
    df[season_col] = pd.to_numeric(df[season_col], errors='coerce')
    df[games_started_col] = pd.to_numeric(df[games_started_col], errors='coerce')
    
    # Auto-detect era-adjusted stats if not specified
    if stats_to_fill is None:
        stats_to_fill = [col for col in df.columns if col.endswith('_adj') and not col.endswith('_adj_proj')]
        # Filter to counting stats (exclude rate stats like Pass_ANY/A_adj, Rush_Rushing_Succ%_adj)
        counting_stats = [col for col in stats_to_fill if any(x in col for x in ['total_yards', 'Pass_TD', 'Pass_Yds', 'Rush_Yds', 'Rush_TD'])]
        if counting_stats:
            stats_to_fill = counting_stats
    
    print(f"\nStats to fill: {stats_to_fill}")
    
    # Initialize is_final_season column
    if 'is_final_season' not in df.columns:
        df['is_final_season'] = False
    
    # Identify rows with missing GS
    missing_gs_mask = df[games_started_col].isna()
    missing_seasons = df[missing_gs_mask].copy()
    
    if len(missing_seasons) == 0:
        print("\nNo missing seasons found (all players have GS values)")
        return df
    
    print(f"\nFound {len(missing_seasons)} rows with missing GS")
    
    # Group by player and process each player's missing seasons
    filled_count = 0
    zero_filled_count = 0
    
    for player_id in missing_seasons[player_id_col].unique():
        player_data = df[df[player_id_col] == player_id].copy()
        player_data = player_data.sort_values(season_col).reset_index(drop=True)
        
        # Find missing seasons for this player
        player_missing = player_data[player_data[games_started_col].isna()].copy()
        
        if len(player_missing) == 0:
            continue
        
        # Process each missing season
        for _, missing_row in player_missing.iterrows():
            missing_season = missing_row[season_col]
            
            # Find position in sorted player_data
            missing_positions = player_data[player_data[season_col] == missing_season].index.tolist()
            if not missing_positions:
                continue
            
            missing_pos = missing_positions[0]  # Position in sorted player_data (0-indexed)
            
            # Find previous and next seasons (with valid GS)
            prev_season = None
            next_season = None
            
            # Look backwards
            for i in range(missing_pos - 1, -1, -1):
                if pd.notna(player_data.iloc[i][games_started_col]):
                    prev_season = player_data.iloc[i]
                    break
            
            # Look forwards
            for i in range(missing_pos + 1, len(player_data)):
                if pd.notna(player_data.iloc[i][games_started_col]):
                    next_season = player_data.iloc[i]
                    break
            
            # Find the actual row index in the main dataframe
            main_df_idx = df[(df[player_id_col] == player_id) & (df[season_col] == missing_season)].index
            
            if len(main_df_idx) == 0:
                continue
            
            main_idx = main_df_idx[0]
            
            # Apply filling logic
            if prev_season is not None and next_season is not None:
                # Case 1: Both before and after exist - average ALL columns from both seasons
                # Get all numeric columns that exist in both seasons
                common_cols = [col for col in prev_season.index if col in next_season.index]
                
                for col in common_cols:
                    # Skip columns that shouldn't be averaged
                    if col in [player_id_col, 'player_name', 'is_final_season', season_col]:
                        continue
                    
                    # Get original dtype to preserve it
                    orig_dtype = df[col].dtype
                    
                    # Handle boolean columns - use the value from next season (or prev if next doesn't exist)
                    if pd.api.types.is_bool_dtype(orig_dtype):
                        if pd.notna(next_season[col]):
                            df.at[main_idx, col] = bool(next_season[col])
                        elif pd.notna(prev_season[col]):
                            df.at[main_idx, col] = bool(prev_season[col])
                        else:
                            df.at[main_idx, col] = False
                        continue
                    
                    # Handle string/object columns - use next season's value
                    if pd.api.types.is_object_dtype(orig_dtype) and not pd.api.types.is_numeric_dtype(orig_dtype):
                        if pd.notna(next_season[col]):
                            df.at[main_idx, col] = next_season[col]
                        elif pd.notna(prev_season[col]):
                            df.at[main_idx, col] = prev_season[col]
                        continue
                    
                    # Handle numeric columns
                    prev_val = pd.to_numeric(prev_season[col], errors='coerce')
                    next_val = pd.to_numeric(next_season[col], errors='coerce')
                    
                    if pd.notna(prev_val) and pd.notna(next_val):
                        avg_val = (prev_val + next_val) / 2.0
                        # Round to integer for counting stats and GS
                        if col in stats_to_fill or col == games_started_col:
                            df.at[main_idx, col] = int(round(avg_val)) if pd.notna(avg_val) else 0
                        # For other numeric columns, preserve float if original was float
                        elif pd.api.types.is_integer_dtype(orig_dtype):
                            df.at[main_idx, col] = int(round(avg_val)) if pd.notna(avg_val) else 0
                        else:
                            df.at[main_idx, col] = float(avg_val) if pd.notna(avg_val) else 0.0
                    elif pd.notna(prev_val):
                        # If only prev exists, use it (preserve type)
                        if pd.api.types.is_integer_dtype(orig_dtype):
                            df.at[main_idx, col] = int(prev_val) if pd.notna(prev_val) else 0
                        else:
                            df.at[main_idx, col] = prev_val
                    elif pd.notna(next_val):
                        # If only next exists, use it (preserve type)
                        if pd.api.types.is_integer_dtype(orig_dtype):
                            df.at[main_idx, col] = int(next_val) if pd.notna(next_val) else 0
                        else:
                            df.at[main_idx, col] = next_val
                    # If both are NaN, set to 0 for counting stats, leave NaN for others
                    elif col in stats_to_fill or col == games_started_col:
                        df.at[main_idx, col] = 0
                
                # Ensure GS is set (should already be set above, but double-check)
                if pd.isna(df.at[main_idx, games_started_col]) or df.at[main_idx, games_started_col] == 0:
                    prev_gs = pd.to_numeric(prev_season[games_started_col], errors='coerce')
                    next_gs = pd.to_numeric(next_season[games_started_col], errors='coerce')
                    if pd.notna(prev_gs) and pd.notna(next_gs):
                        avg_gs = (prev_gs + next_gs) / 2.0
                        df.at[main_idx, games_started_col] = int(round(avg_gs)) if avg_gs > 0 else 0
                
                df.at[main_idx, 'is_final_season'] = False
                filled_count += 1
                
            elif prev_season is not None:
                # Case 2: Only before exists (no return) - set to 0, mark as final
                for stat in stats_to_fill:
                    df.at[main_idx, stat] = 0
                
                df.at[main_idx, games_started_col] = 0
                df.at[main_idx, 'is_final_season'] = True
                zero_filled_count += 1
                
            elif next_season is not None:
                # Case 3: Only after exists (edge case) - copy all values from next season
                for col in next_season.index:
                    # Skip columns that shouldn't be copied
                    if col in [player_id_col, 'player_name', 'is_final_season', season_col]:
                        continue
                    
                    if col in df.columns:
                        next_val = next_season[col]
                        # Round to integer for counting stats
                        if col in stats_to_fill or col == games_started_col:
                            next_val_num = pd.to_numeric(next_val, errors='coerce')
                            df.at[main_idx, col] = int(round(next_val_num)) if pd.notna(next_val_num) else 0
                        else:
                            df.at[main_idx, col] = next_val
                
                df.at[main_idx, 'is_final_season'] = False
                filled_count += 1
                
            else:
                # Case 4: Neither exists - set to 0
                for stat in stats_to_fill:
                    df.at[main_idx, stat] = 0
                
                df.at[main_idx, games_started_col] = 0
                df.at[main_idx, 'is_final_season'] = True
                zero_filled_count += 1
    
    print(f"\nFilled {filled_count} missing seasons with averages or next-season values")
    print(f"Filled {zero_filled_count} missing seasons with zeros (no return)")
    print(f"Total filled: {filled_count + zero_filled_count}")
    
    final_seasons = df['is_final_season'].sum()
    if final_seasons > 0:
        print(f"Marked {final_seasons} seasons as final (player never returned)")
    
    return df


def apply_injury_projection(qb_df, stats_to_project=None, games_started_col='GS', season_col='season', player_id_col='player_id'):
    """
    Applies injury projection to era-adjusted counting stats.
    
    Projects per-game averages to full 16/17 game seasons for QBs who didn't play full seasons.
    Works on era-adjusted stats (columns ending in '_adj') and preserves those adjustments.
    
    Logic:
    - First seasons (before first GS>0): Use era-adjusted value as-is (no projection)
    - After first start, if GS < season_length: Project per-game average to full season
    - If GS goes back to 0 (relief role): Use era-adjusted value as-is (no projection)
    
    Args:
        qb_df (DataFrame): QB data with era-adjusted stats (columns ending in '_adj')
        stats_to_project (list, optional): List of era-adjusted stat columns to project.
            If None, uses common counting stats: ['total_yards_adj', 'Pass_TD_adj', 'Pass_Yds_adj', 'Rush_Yds_adj']
        games_started_col (str): Column name for games started (default: 'GS')
        season_col (str): Column name for season year (default: 'season')
        player_id_col (str): Column name for player ID (default: 'player_id')
        
    Returns:
        DataFrame: Original df with new projected columns (suffix '_proj')
    """
    print("\n" + "="*80)
    print("APPLYING INJURY PROJECTION")
    print("="*80)
    print("Projecting per-game averages to full 16/17 game seasons")
    print("(Preserves existing era adjustments)")
    
    df = qb_df.copy()
    
    # Default stats to project if not specified
    if stats_to_project is None:
        stats_to_project = ['total_yards_adj', 'Pass_TD_adj', 'Pass_Yds_adj', 'Rush_Yds_adj']
    
    # Filter to only stats that exist in the dataframe
    available_stats = [stat for stat in stats_to_project if stat in df.columns]
    if not available_stats:
        print("⚠️  No matching stats found in dataframe")
        return df
    
    print(f"\nStats to project: {available_stats}")
    
    # Ensure required columns are numeric
    df[season_col] = pd.to_numeric(df[season_col], errors='coerce')
    df[games_started_col] = pd.to_numeric(df[games_started_col], errors='coerce')
    
    # Group by player to track first GS>0 season
    df = df.sort_values([player_id_col, season_col]).reset_index(drop=True)
    
    # Find first season with GS>0 for each player
    first_start_seasons = {}
    for player_id in df[player_id_col].unique():
        player_data = df[df[player_id_col] == player_id].copy()
        player_data = player_data.sort_values(season_col)
        
        # Find first season where GS > 0
        first_start = player_data[player_data[games_started_col] > 0]
        if len(first_start) > 0:
            first_start_seasons[player_id] = int(first_start.iloc[0][season_col])
        else:
            first_start_seasons[player_id] = None  # Never started
    
    # Process each stat
    for stat in available_stats:
        proj_col = f"{stat}_proj"
        
        # Ensure stat is numeric
        df[stat] = pd.to_numeric(df[stat], errors='coerce')
        
        # Initialize projected column
        df[proj_col] = np.nan
        
        # Process each row
        for idx, row in df.iterrows():
            era_adj_value = row[stat]
            gs = row[games_started_col]
            season_year = row[season_col]
            player_id = row[player_id_col]
            
            # Skip if missing data
            if pd.isna(era_adj_value) or pd.isna(gs) or pd.isna(season_year):
                continue
            
            season_length = get_season_length(int(season_year))
            first_start_season = first_start_seasons.get(player_id)
            
            # Determine projection logic based on player's career stage
            if gs <= 0:
                # GS = 0: Use era-adjusted value as-is (no projection)
                # This covers: initial seasons before starting, and relief/backup roles
                df.at[idx, proj_col] = int(round(era_adj_value)) if pd.notna(era_adj_value) else np.nan
            elif first_start_season is None:
                # Player never started - use era-adjusted as-is
                df.at[idx, proj_col] = int(round(era_adj_value)) if pd.notna(era_adj_value) else np.nan
            elif int(season_year) < first_start_season:
                # Before first start season - use era-adjusted as-is (initial seasons)
                df.at[idx, proj_col] = int(round(era_adj_value)) if pd.notna(era_adj_value) else np.nan
            elif gs >= season_length:
                # Already played full season - use era-adjusted value (no projection needed)
                df.at[idx, proj_col] = int(round(era_adj_value)) if pd.notna(era_adj_value) else np.nan
            else:
                # After first start, GS > 0 but < season_length - apply injury projection
                per_game_avg = era_adj_value / gs
                proj_value = per_game_avg * season_length
                df.at[idx, proj_col] = int(round(proj_value))
        
        # Show results
        non_null = df[proj_col].notna().sum()
        
        # Count by category
        projected_count = 0
        initial_seasons_count = 0
        full_seasons_count = 0
        relief_seasons_count = 0
        
        for idx, row in df[df[proj_col].notna()].iterrows():
            gs = row[games_started_col]
            season_year = int(row[season_col])
            player_id = row[player_id_col]
            first_start = first_start_seasons.get(player_id)
            season_length = get_season_length(season_year)
            
            if gs <= 0:
                if first_start is None or season_year < first_start:
                    initial_seasons_count += 1
                else:
                    relief_seasons_count += 1
            elif gs >= season_length:
                full_seasons_count += 1
            elif first_start is not None and season_year >= first_start:
                projected_count += 1
        
        print(f"✓ {stat} → {proj_col}: {non_null} values")
        print(f"  - Projected (injury-adjusted): {projected_count} seasons (after first start, GS < season_length)")
        print(f"  - Initial seasons (GS=0, before first start): {initial_seasons_count} seasons")
        print(f"  - Full seasons (GS >= season_length): {full_seasons_count} seasons")
        print(f"  - Relief roles (GS=0, after first start): {relief_seasons_count} seasons")
        
        # Show example
        sample = df[df[proj_col].notna()].head(3)
        if len(sample) > 0:
            for _, s in sample.iterrows():
                gs_val = int(s[games_started_col])
                season_val = int(s[season_col])
                season_len = get_season_length(season_val)
                status = "projected" if gs_val < season_len else "unchanged"
                print(f"  Example ({season_val}, GS={gs_val}): "
                      f"Era-adj={int(s[stat])}, Proj={int(s[proj_col])} ({status})")
                break
    
    # Verify both _adj and _proj columns exist
    adj_cols = [col for col in df.columns if col.endswith('_adj') and not col.endswith('_adj_proj')]
    proj_cols = [col for col in df.columns if col.endswith('_proj')]
    
    print(f"\n✓ Injury projection complete")
    print(f"  Era-adjusted columns preserved: {len(adj_cols)} ({', '.join(adj_cols[:3])}...)")
    print(f"  Projected columns created: {len(proj_cols)} ({', '.join(proj_cols[:3])}...)")
    print(f"  Both column sets available for analysis")
    
    return df


def export_injury_projection_comparison(qb_df, stats_to_compare=None, output_file='injury_projection_comparison.csv', 
                                        games_started_col='GS', season_col='season'):
    """
    Exports a side-by-side comparison DataFrame of era-adjusted vs projected stats.
    
    Args:
        qb_df (DataFrame): QB data with both era-adjusted ('_adj') and projected ('_proj') columns
        stats_to_compare (list, optional): List of stat base names to compare (e.g., ['total_yards', 'Pass_TD']).
            If None, auto-detects from '_proj' columns
        output_file (str): Output CSV filename
        games_started_col (str): Column name for games started
        season_col (str): Column name for season year
        
    Returns:
        DataFrame: Comparison DataFrame with original, projected, difference, and percent change columns
    """
    print("\n" + "="*80)
    print("EXPORTING INJURY PROJECTION COMPARISON")
    print("="*80)
    
    df = qb_df.copy()
    
    # Auto-detect stats to compare if not specified
    if stats_to_compare is None:
        proj_cols = [col for col in df.columns if col.endswith('_proj')]
        stats_to_compare = [col.replace('_adj_proj', '').replace('_proj', '') for col in proj_cols]
        # Remove duplicates while preserving order
        stats_to_compare = list(dict.fromkeys(stats_to_compare))
    
    # Filter to stats that have both _adj and _proj columns
    valid_stats = []
    for stat_base in stats_to_compare:
        adj_col = f"{stat_base}_adj"
        proj_col = f"{stat_base}_adj_proj"
        
        if adj_col in df.columns and proj_col in df.columns:
            valid_stats.append(stat_base)
        else:
            print(f"⚠️  Skipping {stat_base}: missing {adj_col} or {proj_col}")
    
    if not valid_stats:
        print("✗ ERROR: No valid stats to compare")
        return None
    
    print(f"\nComparing {len(valid_stats)} stats: {valid_stats}")
    
    # Start building comparison DataFrame
    comparison_cols = ['player_id', 'player_name', season_col, games_started_col]
    
    # Add season_length column
    df['season_length'] = df[season_col].apply(lambda x: get_season_length(int(x)) if pd.notna(x) else np.nan)
    comparison_cols.append('season_length')
    
    # Add was_projected flag
    df['was_projected'] = df[games_started_col] < df['season_length']
    comparison_cols.append('was_projected')
    
    # Add comparison columns for each stat
    for stat_base in valid_stats:
        adj_col = f"{stat_base}_adj"
        proj_col = f"{stat_base}_adj_proj"
        
        # Original (era-adjusted)
        comparison_cols.append(adj_col)
        
        # Projected
        comparison_cols.append(proj_col)
        
        # Difference
        diff_col = f"{stat_base}_adj_difference"
        df[diff_col] = df[proj_col] - df[adj_col]
        comparison_cols.append(diff_col)
        
        # Percent change
        pct_col = f"{stat_base}_adj_pct_change"
        df[pct_col] = np.where(
            df[adj_col] != 0,
            (df[diff_col] / df[adj_col]) * 100,
            np.nan
        )
        comparison_cols.append(pct_col)
    
    # Create comparison DataFrame
    comparison_df = df[comparison_cols].copy()
    
    # Sort by player_id, then season
    if 'player_id' in comparison_df.columns and season_col in comparison_df.columns:
        comparison_df = comparison_df.sort_values(['player_id', season_col]).reset_index(drop=True)
    
    # Summary statistics
    total_seasons = len(comparison_df)
    projected_seasons = comparison_df['was_projected'].sum() if 'was_projected' in comparison_df.columns else 0
    unchanged_seasons = total_seasons - projected_seasons
    
    print(f"\nSummary:")
    print(f"  Total seasons: {total_seasons}")
    print(f"  Projected: {projected_seasons} ({projected_seasons/total_seasons*100:.1f}%)")
    print(f"  Unchanged: {unchanged_seasons} ({unchanged_seasons/total_seasons*100:.1f}%)")
    
    # Save to CSV
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison to: {output_file}")
    print(f"  Columns: {len(comparison_df.columns)}")
    print(f"  Rows: {len(comparison_df)}")
    
    return comparison_df
