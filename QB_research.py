import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import PFR_Tools as PFR
import random
import os
from IPython.display import display
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY
import traceback
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def load_csv_safe(filepath, description="file"):
    """
    Safely load a CSV file with consistent error handling.
    
    Args:
        filepath (str): Path to CSV file
        description (str): Human-readable description for error messages
    
    Returns:
        DataFrame or None: Loaded dataframe, or None if file not found/invalid
    """
    if not os.path.exists(filepath):
        print(f"✗ ERROR: {description} not found at: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} records from {description}")
        return df
    except Exception as e:
        print(f"✗ ERROR: Failed to read {description}: {e}")
        return None

def validate_columns(df, required_cols, df_name="dataframe"):
    """
    Validates that required columns exist in a dataframe.
    
    Args:
        df (DataFrame): Dataframe to validate
        required_cols (list): List of required column names
        df_name (str): Human-readable name for error messages
    
    Returns:
        bool: True if all columns exist, False otherwise
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"✗ ERROR: Missing required columns in {df_name}:")
        for col in missing:
            print(f"  - {col}")
        print(f"\nAvailable columns: {df.columns.tolist()}")
        return False
    
    return True

def validate_payment_years(df, draft_year_col='draft_year', payment_year_col='payment_year', 
                           years_to_payment_col='years_to_payment', max_years=12):
    """
    Validates that payment years make logical sense.
    
    Args:
        df (DataFrame): Dataframe with payment data
        draft_year_col (str): Column name for draft year
        payment_year_col (str): Column name for payment year
        years_to_payment_col (str): Column name for years to payment
        max_years (int): Maximum reasonable years between draft and payment
    
    Returns:
        DataFrame: Filtered dataframe with invalid records removed
    """
    print("\n" + "="*80)
    print("VALIDATING PAYMENT YEARS")
    print("="*80)
    
    initial_count = len(df)
    paid_df = df[df[payment_year_col].notna()].copy()
    
    if len(paid_df) == 0:
        print("No payment data to validate")
        return df
    
    print(f"\nValidating {len(paid_df)} seasons with payment data...")
    
    # Check 1: Payment year should be after draft year
    invalid_order = paid_df[paid_df[payment_year_col] < paid_df[draft_year_col]]
    if len(invalid_order) > 0:
        print(f"\n⚠️  Found {len(invalid_order)} records where payment year < draft year:")
        for idx, row in invalid_order.head(5).iterrows():
            print(f"  - Player {row.get('player_name', row.get('player_id', 'Unknown'))}: "
                  f"Drafted {row[draft_year_col]}, Paid {row[payment_year_col]}")
        if len(invalid_order) > 5:
            print(f"  ... and {len(invalid_order) - 5} more")
    
    # Check 2: Payment shouldn't be too many years after draft (rookie deals are 4-5 years)
    too_late = paid_df[paid_df[years_to_payment_col].abs() > max_years]
    if len(too_late) > 0:
        print(f"\n⚠️  Found {len(too_late)} records where payment is >{max_years} years from draft:")
        for idx, row in too_late.head(5).iterrows():
            print(f"  - Player {row.get('player_name', row.get('player_id', 'Unknown'))}: "
                  f"Years to payment = {row[years_to_payment_col]}")
        if len(too_late) > 5:
            print(f"  ... and {len(too_late) - 5} more")
    
    # Check 3: Payment year shouldn't be in the future
    current_year = 2025  # Update this as needed
    future_payments = paid_df[paid_df[payment_year_col] > current_year]
    if len(future_payments) > 0:
        print(f"\n⚠️  Found {len(future_payments)} records with future payment years:")
        for idx, row in future_payments.head(5).iterrows():
            print(f"  - Player {row.get('player_name', row.get('player_id', 'Unknown'))}: "
                  f"Payment year = {row[payment_year_col]}")
    
    # Summary
    total_issues = len(invalid_order) + len(too_late) + len(future_payments)
    
    if total_issues == 0:
        print("\n✓ All payment years look valid")
    else:
        print(f"\n⚠️  Total validation issues: {total_issues}")
        print("These records will be kept but may need investigation")
    
    return df  # Return original df - just warning, not filtering

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

def bestqbseasons():
    try:
        if os.path.exists("best_seasons_df.csv"):
            print("starting best QB seasons pull")
            df_QB_best_seasons = load_csv_safe("best_seasons_df.csv")
            #print( df_QB_best_seasons.shape, "columns: ", df_QB_best_seasons.columns)
            df_QB_best_seasons.sort_values(by=['total_yards'], ascending=[False], inplace=True)
            display(df_QB_best_seasons.head(5)[['player_name', 'draft_team', 'season', 'total_yards', 'AdvPass_Air Yards_IAY/PA', 'Pass_Cmp%', 'Pass_Rate', 'Pass_Yds', 'Rush_Rushing_Yds', 'Rush_Rushing_TD']])
    except Exception as e:
        print(f"Error running the best QB seasons pull: {e}")

def best_season_averages():
    try:
        if os.path.exists("season_averages.csv"):
            print(f"\nstarting best season averages pull")
            df_season_averages = load_csv_safe("season_averages.csv")
            #print( df_season_averages.shape, "columns: ", df_season_averages.columns)
            #Year,Teams,PF,Total_Yards,Plays,Y/P,TO,FL,1stD,Pass_Cmp,Pass_Att,Pass_Yds,Pass_TD,Int,NY/A,Pass_1stD,Rush_Att,Rush_Yds,Rush_TD,Rush_Y/A,Rush_1stD,Penalties,Pen_Yds,1stPy,Drives,Drive_Sc%,Drive_TO%,Drive_Plays,Drive_Yds,Drive_Pts
            df_season_averages.sort_values(by=['Total_Yards'], ascending=[False], inplace=True)
            display(df_season_averages.head(5)[['Year', 'Y/P', 'Total_Yards', 'Pass_Yds', 'NY/A', 'Rush_Y/A']])
    except Exception as e:
        print(f"Error running the season averages pull: {e}")

def most_expensive_qb_contracts():
    try:
        if os.path.exists("QB_contract_data.csv"):
            print(f"\nstarting most expensive QB contract data pull (apy as % of cap at signing)")
            df_QB_contract_data = load_csv_safe("QB_contract_data.csv")
            #print( df_QB_contract_data.shape, "columns: ", df_QB_contract_data.columns)
            #Player,Team,Year,Years,,Value,APY,Guaranteed,,APY as % Of,,Inflated,Inflated,Inflated
            #df_QB_contract_data = df_QB_contract_data[df_QB_contract_data['Year'] == '2024'] # take only the 2024 year
            df_QB_contract_data.sort_values(by=['APY as % Of'], ascending=[False], inplace=True)
            display(df_QB_contract_data.head(5)[['Player', 'Team', 'Year', 'APY as % Of', 'Value', 'Guaranteed']])
            print("\nlooking for Deshaun Watson contract details")
            display((df_QB_contract_data[df_QB_contract_data['Player']=="Deshaun Watson"])[['Player', 'Team', 'Year', 'APY as % Of', 'Value', 'Guaranteed']])
    except Exception as e:
        print(f"Error running the most expensive QB contracts pull: {e}")

def best_season_records():
    try:
        if os.path.exists("season_records.csv"):
            print(f"\nstarting best season records pull")
            df_season_records = load_csv_safe("season_records.csv")
            #print( df_season_records.shape, "columns: ", df_season_records.columns)
            #Rk,Season,Team,W,G,W,L,T,W-L%,Pts,PtsO,PtDif
            df_season_records.sort_values(by=['W-L%'], ascending=[False], inplace=True)
            display(df_season_records.head(5)[['Season', 'Team', 'PtDif', 'W-L%']])
    except Exception as e:
        print(f"Error running the season records pull: {e}")

def analyze_qb_stat_correlations_with_wins():
    """
    Analyzes which QB performance statistics are most highly correlated with
    team win-loss percentage in the same season, including statistical significance.
    
    Returns:
        DataFrame: Sorted correlations between QB stats and team W-L% with p-values
    """
    from scipy import stats
    import numpy as np
    
    print("\n=== Analyzing QB Stats vs Team Win Percentage ===")
    
    # Load the data
    qb_seasons = load_csv_safe("all_seasons_df.csv", "QB seasons data")
    season_records = load_csv_safe("season_records.csv", "season records")

    if qb_seasons is None or season_records is None:
        return None
    
    # Validate required columns
    required_qb_cols = ['season', 'Team', 'GS', 'player_id']
    required_season_cols = ['Season', 'Team', 'W', 'W-L%']

    if not validate_columns(qb_seasons, required_qb_cols, "QB seasons"):
        return None
    if not validate_columns(season_records, required_season_cols, "season records"):
        return None
    
    # Clean QB data - remove invalid team entries
    qb_seasons = qb_seasons[qb_seasons['Team'].str.len() == 3]  # Only keep 3-letter codes
    qb_seasons = qb_seasons[~qb_seasons['Team'].str.contains('Did not', na=False)]
    
    # Convert season to int for merging
    qb_seasons['season'] = pd.to_numeric(qb_seasons['season'], errors='coerce')
    qb_seasons = qb_seasons.dropna(subset=['season'])
    qb_seasons['season'] = qb_seasons['season'].astype(int)
    
    # Convert Season column in records to int
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Check team code overlap between datasets
    print("\nChecking team code compatibility...")  
    
    print(f"QB seasons: {len(qb_seasons)} records")
    print(f"Season records: {len(season_records)} records")
    print(f"\nUnique QB teams ({len(qb_seasons['Team'].unique())}): {sorted(qb_seasons['Team'].unique())}")
    print(f"\nUnique season record teams ({len(season_records['Team'].unique())}): {sorted(season_records['Team'].unique())}")
    
    # Find which teams overlap
    qb_teams = set(qb_seasons['Team'].unique())
    record_teams = set(season_records['Team'].unique())
    overlapping = qb_teams.intersection(record_teams)
    qb_only = qb_teams - record_teams
    record_only = record_teams - qb_teams
    
    print(f"\nTeams in both datasets ({len(overlapping)}): {sorted(overlapping)}")
    print(f"\nTeams only in QB data ({len(qb_only)}): {sorted(qb_only)}")
    print(f"\nTeams only in season records ({len(record_only)}): {sorted(record_only)}")
    
    # Merge QB data with season records
    # Note: Only teams with matching codes will merge successfully
    # Based on diagnostics above, most teams should match
    merged_df = pd.merge(
        qb_seasons,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"\nSuccessfully merged {len(merged_df)} QB season-team records")
    
    if merged_df.empty or len(merged_df) < 50:
        print("\n✗ ERROR: Insufficient matching records after merge")
        print(f"Only {len(merged_df)} records matched between datasets")
        print("This likely means team codes don't align between QB seasons and season records")
        print("\nTo fix:")
        print("  1. Check the team code diagnostics above")
        print("  2. Create a mapping dict if codes differ (e.g., 'RAV' → 'BAL')")
        print("  3. Apply mapping before merge")
        return None
    
    # Filter to QBs who actually played significant snaps (at least 8 games started)
    merged_df['GS_numeric'] = pd.to_numeric(merged_df['GS'], errors='coerce')
    merged_df = merged_df[merged_df['GS_numeric'] >= 8]
    print(f"Filtered to {len(merged_df)} QB seasons with 8+ games started")
    
    if merged_df.empty:
        print("No records after filtering for games started")
        return None
    
    # Select QB performance columns (numeric only)
    qb_stat_cols = [col for col in merged_df.columns if 
                    col.startswith('Pass_') or 
                    col.startswith('Rush_') or 
                    col.startswith('AdvPass_')]
    
    # Also include total_yards if available
    if 'total_yards' in merged_df.columns:
        qb_stat_cols.append('total_yards')
    
    # Convert W-L% to numeric if it's not already
    merged_df['W-L%_numeric'] = pd.to_numeric(merged_df['W-L%'], errors='coerce')
    
    # Check if we have valid win% data
    valid_winpct = merged_df['W-L%_numeric'].notna().sum()
    print(f"\nRecords with valid W-L%: {valid_winpct}/{len(merged_df)}")
    
    if valid_winpct < 30:
        print("Not enough valid win% data to calculate correlations")
        print("\nSample of W-L% column:")
        print(merged_df[['Team', 'season', 'W-L%', 'W-L%_numeric']].head(10))
        return None
    
    print(f"\nAnalyzing {len(qb_stat_cols)} QB statistics against team W-L%")
    
    # Calculate correlations with significance tests
    correlations = {}
    
    for col in qb_stat_cols:
        try:
            # Convert both columns to numeric
            stat_values = pd.to_numeric(merged_df[col], errors='coerce')
            win_pct = merged_df['W-L%_numeric']
            
            # Drop NaN values
            valid_data = pd.DataFrame({'stat': stat_values, 'win_pct': win_pct}).dropna()
            
            if len(valid_data) >= 30:  # Need sufficient data points
                # Calculate Pearson correlation and p-value
                corr, p_value = stats.pearsonr(valid_data['stat'], valid_data['win_pct'])
                
                # Calculate confidence interval (95%)
                # Using Fisher z-transformation
                n = len(valid_data)
                stderr = 1.0 / np.sqrt(n - 3)
                z = 0.5 * (np.log(1 + corr) - np.log(1 - corr)) if abs(corr) < 0.9999 else 0
                z_crit = 1.96  # 95% confidence
                lo_z = z - z_crit * stderr
                hi_z = z + z_crit * stderr
                lo = (np.exp(2 * lo_z) - 1) / (np.exp(2 * lo_z) + 1)
                hi = (np.exp(2 * hi_z) - 1) / (np.exp(2 * hi_z) + 1)
                
                correlations[col] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'highly_significant': p_value < 0.01,
                    'very_highly_significant': p_value < 0.001,
                    'n_samples': len(valid_data),
                    'ci_lower': lo,
                    'ci_upper': hi
                }
        except Exception as e:
            # Skip columns that can't be converted or calculated
            print(f"Skipped {col}: {e}")
            continue
    
    if not correlations:
        print("No correlations could be calculated. Check data quality.")
        return None
    
    # Create DataFrame from correlations
    corr_df = pd.DataFrame.from_dict(correlations, orient='index')
    
    # Add significance level indicator
    def sig_level(row):
        if row['very_highly_significant']:
            return '***'
        elif row['highly_significant']:
            return '**'
        elif row['significant']:
            return '*'
        else:
            return ''
    
    corr_df['sig_level'] = corr_df.apply(sig_level, axis=1)
    
    # Sort by absolute correlation value
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"\nCalculated correlations for {len(corr_df)} QB statistics")
    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print("="*80)
    print("Top 20 Most Correlated Stats (by absolute value):")
    print("="*80)
    display(corr_df[['correlation', 'p_value', 'sig_level', 'n_samples', 'ci_lower', 'ci_upper']].head(20))
    
    # Filter to only significant correlations
    sig_corr = corr_df[corr_df['significant']]
    print(f"\n{len(sig_corr)} out of {len(corr_df)} correlations are statistically significant (p<0.05)")
    
    print("\n" + "="*80)
    print("Top 10 Significant Positive Correlations:")
    print("="*80)
    positive_corr = corr_df[(corr_df['correlation'] > 0) & (corr_df['significant'])].head(10)
    display(positive_corr[['correlation', 'p_value', 'sig_level', 'n_samples']])
    
    print("\n" + "="*80)
    print("Top 10 Significant Negative Correlations:")
    print("="*80)
    negative_corr = corr_df[(corr_df['correlation'] < 0) & (corr_df['significant'])].head(10)
    display(negative_corr[['correlation', 'p_value', 'sig_level', 'n_samples']])
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Total stats analyzed: {len(corr_df)}")
    print(f"Significant (p<0.05): {len(corr_df[corr_df['significant']])}")
    print(f"Highly significant (p<0.01): {len(corr_df[corr_df['highly_significant']])}")
    print(f"Very highly significant (p<0.001): {len(corr_df[corr_df['very_highly_significant']])}")
    print(f"\nStrongest correlation: {corr_df.iloc[0].name}")
    print(f"  r = {corr_df.iloc[0]['correlation']:.3f}, p = {corr_df.iloc[0]['p_value']:.6f}")
    
    # Save results
    corr_df.to_csv('qb_stats_win_correlations.csv')
    print("\nSaved full results to qb_stats_win_correlations.csv")
    
    return corr_df

def pca_factor_analysis_qb_stats():
    """
    Performs Principal Component Analysis (PCA) on QB statistics to identify
    underlying factors that explain QB performance.
    
    Outputs:
    1. Scree plot showing eigenvalues
    2. Cumulative variance explained
    3. Component loadings (which stats load on which factors)
    4. Recommended number of components
    
    Returns:
        dict: PCA results including model, loadings, and variance explained
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print("PRINCIPAL COMPONENT ANALYSIS: QB Stats Factor Structure")
    print("="*80)
    
    qb_seasons = load_csv_safe("all_seasons_df.csv", "QB seasons data")
    season_records = load_csv_safe("season_records.csv", "season records")

    if qb_seasons is None or season_records is None:
        return None

    required_qb_cols = ['season', 'Team', 'GS']
    required_season_cols = ['Season', 'Team', 'W']

    if not validate_columns(qb_seasons, required_qb_cols, "QB seasons"):
        return None
    if not validate_columns(season_records, required_season_cols, "season records"):
        return None

    # Clean QB data
    qb_seasons = qb_seasons[qb_seasons['Team'].str.len() == 3]
    qb_seasons = qb_seasons[~qb_seasons['Team'].str.contains('Did not', na=False)]
    
    # Convert season to int
    qb_seasons['season'] = pd.to_numeric(qb_seasons['season'], errors='coerce')
    qb_seasons = qb_seasons.dropna(subset=['season'])
    qb_seasons['season'] = qb_seasons['season'].astype(int)
    
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Merge datasets
    merged_df = pd.merge(
        qb_seasons,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"Merged {len(merged_df)} QB season-team records")
    
    # Filter to significant playing time
    merged_df['GS_numeric'] = pd.to_numeric(merged_df['GS'], errors='coerce')
    merged_df = merged_df[merged_df['GS_numeric'] >= 8]
    print(f"Filtered to {len(merged_df)} QB seasons with 8+ games started")
    
    # Get wins for later correlation
    merged_df['Wins'] = pd.to_numeric(merged_df['W'], errors='coerce')
    
    # Select QB performance columns
    qb_stat_cols = [col for col in merged_df.columns if 
                    col.startswith('Pass_') or 
                    col.startswith('Rush_') or 
                    col.startswith('AdvPass_')]
    
    if 'total_yards' in merged_df.columns:
        qb_stat_cols.append('total_yards')
    
    print(f"\nStarting with {len(qb_stat_cols)} QB statistics")
    
    # Check data availability for each stat
    print("\nAnalyzing data availability...")
    stat_availability = {}
    for col in qb_stat_cols:
        non_null = pd.to_numeric(merged_df[col], errors='coerce').notna().sum()
        stat_availability[col] = non_null
    
    # Filter to stats with good availability (at least 200 non-null values)
    min_samples = 200
    available_stats = [col for col, count in stat_availability.items() if count >= min_samples]
    
    print(f"Stats with ≥{min_samples} samples: {len(available_stats)}")
    
    # Create feature matrix
    feature_data = merged_df[available_stats].copy()
    
    # Convert all to numeric
    for col in available_stats:
        feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
    
    # Drop rows with any missing values
    feature_data = feature_data.dropna()
    
    print(f"Complete cases: {len(feature_data)}")
    
    if len(feature_data) < 50:
        print("Insufficient complete cases for PCA")
        return None
    
    # Prepare data
    X = feature_data.values
    feature_names = feature_data.columns.tolist()
    
    print(f"\nPerforming PCA on {len(feature_names)} stats with {len(feature_data)} samples")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Get eigenvalues (variance explained by each component)
    eigenvalues = pca_full.explained_variance_
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # =========================================================================
    # PART 1: EIGENVALUE ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("EIGENVALUE ANALYSIS")
    print("="*80)
    
    # Create eigenvalue table
    eigen_df = pd.DataFrame({
        'Component': range(1, len(eigenvalues) + 1),
        'Eigenvalue': eigenvalues,
        'Variance_Explained_%': explained_variance_ratio * 100,
        'Cumulative_Variance_%': cumulative_variance * 100
    })
    
    print("\nFirst 20 Components:")
    display(eigen_df.head(20))
    
    # Kaiser criterion (eigenvalue > 1)
    kaiser_components = sum(eigenvalues > 1)
    print(f"\nKaiser Criterion (eigenvalue > 1): {kaiser_components} components")
    
    # Find components explaining 80% of variance
    components_80 = sum(cumulative_variance < 0.80) + 1
    print(f"Components explaining 80% variance: {components_80}")
    
    # Find components explaining 90% of variance
    components_90 = sum(cumulative_variance < 0.90) + 1
    print(f"Components explaining 90% variance: {components_90}")
    
    # =========================================================================
    # PART 2: SCREE PLOT
    # =========================================================================
    print("\n" + "="*80)
    print("SCREE PLOT")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Eigenvalues (Scree Plot)
    components_to_show = min(20, len(eigenvalues))
    axes[0].plot(range(1, components_to_show + 1), 
                 eigenvalues[:components_to_show], 
                 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (λ=1)')
    axes[0].set_xlabel('Component Number', fontsize=12)
    axes[0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0].set_title('Scree Plot: Eigenvalues by Component', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Cumulative Variance Explained
    axes[1].plot(range(1, components_to_show + 1), 
                 cumulative_variance[:components_to_show] * 100, 
                 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=80, color='orange', linestyle='--', label='80% Variance')
    axes[1].axhline(y=90, color='red', linestyle='--', label='90% Variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=300, bbox_inches='tight')
    print("\nScree plot saved to: pca_scree_plot.png")
    plt.show()
    
    # =========================================================================
    # PART 3: COMPONENT LOADINGS
    # =========================================================================
    print("\n" + "="*80)
    print("COMPONENT LOADINGS (First 5 Components)")
    print("="*80)
    
    # Get loadings for first N components
    n_components_to_analyze = min(5, kaiser_components, len(eigenvalues))
    
    pca_limited = PCA(n_components=n_components_to_analyze)
    pca_limited.fit(X_scaled)
    
    # Create loadings matrix
    loadings = pca_limited.components_.T * np.sqrt(pca_limited.explained_variance_)
    
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(n_components_to_analyze)],
        index=feature_names
    )
    
    # For each component, show top positive and negative loadings
    for i in range(n_components_to_analyze):
        pc_name = f'PC{i+1}'
        print(f"\n{pc_name} (Explains {explained_variance_ratio[i]*100:.2f}% of variance):")
        print(f"Eigenvalue: {eigenvalues[i]:.3f}")
        print("\nTop 10 Positive Loadings:")
        top_positive = loadings_df[pc_name].nlargest(10)
        for stat, loading in top_positive.items():
            print(f"  {stat}: {loading:.3f}")
        
        print("\nTop 10 Negative Loadings:")
        top_negative = loadings_df[pc_name].nsmallest(10)
        for stat, loading in top_negative.items():
            print(f"  {stat}: {loading:.3f}")
    
    # =========================================================================
    # PART 4: CORRELATION WITH WINS
    # =========================================================================
    print("\n" + "="*80)
    print("COMPONENT SCORES vs TEAM WINS")
    print("="*80)
    
    # Transform data to get component scores
    component_scores = pca_limited.transform(X_scaled)
    
    # Add wins data
    wins_data = merged_df.loc[feature_data.index, 'Wins'].values
    
    # Calculate correlations
    from scipy.stats import pearsonr
    
    component_win_correlations = []
    for i in range(n_components_to_analyze):
        # Remove NaN values
        valid_mask = ~np.isnan(wins_data)
        if valid_mask.sum() > 0:
            corr, p_val = pearsonr(component_scores[valid_mask, i], wins_data[valid_mask])
            component_win_correlations.append({
                'Component': f'PC{i+1}',
                'Correlation_with_Wins': corr,
                'P_Value': p_val,
                'Variance_Explained_%': explained_variance_ratio[i] * 100,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
    
    corr_wins_df = pd.DataFrame(component_win_correlations)
    print("\nComponent Correlations with Team Wins:")
    display(corr_wins_df)
    
    # =========================================================================
    # PART 5: INTERPRETATION AND RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("="*80)
    
    # Interpret first component
    pc1_loadings = loadings_df['PC1'].abs().sort_values(ascending=False).head(10)
    print("\nPC1 (Primary Factor) appears to capture:")
    print("Top contributing stats:")
    for stat in pc1_loadings.index:
        print(f"  - {stat}")
    
    # Find which component correlates best with wins
    best_win_predictor = corr_wins_df.loc[corr_wins_df['Correlation_with_Wins'].abs().idxmax()]
    print(f"\n{best_win_predictor['Component']} has strongest correlation with wins:")
    print(f"  r = {best_win_predictor['Correlation_with_Wins']:.3f}")
    print(f"  p = {best_win_predictor['P_Value']:.6f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print(f"\nSuggested number of components to retain: {kaiser_components}")
    print(f"These {kaiser_components} components explain {cumulative_variance[kaiser_components-1]*100:.1f}% of total variance")
    
    # Save results
    results = {
        'pca_model': pca_limited,
        'scaler': scaler,
        'eigenvalues': eigenvalues,
        'variance_explained': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'loadings': loadings_df,
        'component_scores': component_scores,
        'feature_names': feature_names,
        'kaiser_components': kaiser_components,
        'component_win_correlations': corr_wins_df
    }
    
    # Save to CSV
    eigen_df.to_csv('pca_eigenvalues.csv', index=False)
    loadings_df.to_csv('pca_loadings.csv')
    corr_wins_df.to_csv('pca_component_win_correlations.csv', index=False)
    
    print("\nSaved results to:")
    print("  - pca_eigenvalues.csv")
    print("  - pca_loadings.csv")
    print("  - pca_component_win_correlations.csv")
    print("  - pca_scree_plot.png")
    
    return results

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

def regression_with_pc1_factors(train_df=None, test_df=None):
    """
    Performs regression analysis using the top factors from PC1 (primary component).
    PC1 represents overall passing efficiency and volume - the core QB performance metrics.
    
    Trains on training set and evaluates on test set.
    
    Args:
        train_df: Training data (if None, loads from file)
        test_df: Test data (if None, loads from file)
    
    Returns:
        dict: Regression results and variable importance
    """
    print("\n" + "="*80)
    print("MULTIVARIATE REGRESSION: PC1 Factors → Team Wins")
    print("="*80)
    
    # Load train/test data if not provided
    if train_df is None or test_df is None:
        print("\nLoading train/test split from files...")
        train_df, test_df = load_train_test_split()
        if train_df is None or test_df is None:
            print("Error: Could not load train/test data")
            return None
    
    # Load season records for both train and test
    try:
        season_records = load_csv_safe("season_records.csv")
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        return None
    
    # Convert Season column to int
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Process training data
    print("\n" + "="*80)
    print("PROCESSING TRAINING DATA")
    print("="*80)
    
    train_df['season'] = pd.to_numeric(train_df['season'], errors='coerce')
    train_df = train_df.dropna(subset=['season'])
    train_df['season'] = train_df['season'].astype(int)
    
    # Merge with season records
    train_merged = pd.merge(
        train_df,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"Training set: {len(train_merged)} QB season-team records")
    
    if not validate_columns(train_merged, ['GS', 'W'], "merged training data"):
        return None

    # Filter to significant playing time
    train_merged['GS_numeric'] = pd.to_numeric(train_merged['GS'], errors='coerce')
    train_merged = train_merged[train_merged['GS_numeric'] >= 8]
    print(f"After filtering (8+ GS): {len(train_merged)} records")
    
    # Get wins as target
    train_merged['Wins'] = pd.to_numeric(train_merged['W'], errors='coerce')
    train_merged = train_merged.dropna(subset=['Wins'])
    
    # Process test data
    print("\n" + "="*80)
    print("PROCESSING TEST DATA")
    print("="*80)
    
    test_df['season'] = pd.to_numeric(test_df['season'], errors='coerce')
    test_df = test_df.dropna(subset=['season'])
    test_df['season'] = test_df['season'].astype(int)
    
    test_merged = pd.merge(
        test_df,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"Test set: {len(test_merged)} QB season-team records")
    
    test_merged['GS_numeric'] = pd.to_numeric(test_merged['GS'], errors='coerce')
    test_merged = test_merged[test_merged['GS_numeric'] >= 8]
    print(f"After filtering (8+ GS): {len(test_merged)} records")
    
    test_merged['Wins'] = pd.to_numeric(test_merged['W'], errors='coerce')
    test_merged = test_merged.dropna(subset=['Wins'])
    
    # Define PC1 chosen factors
    pc1_factors = [
        'total_yards',      # Instead of Pass_Yds - captures dual-threat
        'Pass_TD',          # Point scoring metric
        'Pass_ANY/A',       # Best passing efficiency metric (accounts for TDs and INTs)
        'Pass_Succ%',       # Situational success rate
        'Rush_Rushing_Succ%', # QB rushing success rate
    ]
    
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    
    print(f"\nUsing {len(pc1_factors)} PC1 factors:")
    for factor in pc1_factors:
        print(f"  - {factor}")
    
    # Check which factors are available
    available_factors = [f for f in pc1_factors if f in train_merged.columns]
    missing_factors = [f for f in pc1_factors if f not in train_merged.columns]
    
    if missing_factors:
        print(f"\nWarning: {len(missing_factors)} factors not found:")
        for factor in missing_factors:
            print(f"  - {factor}")
    
    print(f"\nProceeding with {len(available_factors)} available factors")
    
    # Build training dataset ONCE - before multicollinearity check
    train_features = train_merged[available_factors + ['Wins']].copy()
    for col in available_factors:
        train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
    
    initial_train = len(train_features)
    train_features = train_features.dropna()
    final_train = len(train_features)
    
    print(f"\nTraining set complete cases: {final_train}/{initial_train} ({final_train/initial_train*100:.1f}%)")
    
    # Build test dataset ONCE - before multicollinearity check
    test_features = test_merged[available_factors + ['Wins']].copy()
    for col in available_factors:
        test_features[col] = pd.to_numeric(test_features[col], errors='coerce')
    
    initial_test = len(test_features)
    test_features = test_features.dropna()
    final_test = len(test_features)
    
    print(f"Test set complete cases: {final_test}/{initial_test} ({final_test/initial_test*100:.1f}%)")
    
    if final_train < 50 or final_test < 10:
        print("Insufficient data for regression")
        return None
    
    # =========================================================================
    # MULTICOLLINEARITY CHECK (on the already-built training data)
    # =========================================================================
    print("\n" + "="*80)
    print("MULTICOLLINEARITY CHECK")
    print("="*80)
    
    correlation_matrix = train_features[available_factors].corr()
    
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
        for col1, col2, corr in high_corr_pairs:
            print(f"  {col1} <-> {col2}: r={corr:.3f}")
        
        print("\nRemoving redundant variables...")
        cols_to_remove = set()
        for col1, col2, _ in high_corr_pairs:
            # Always remove the second variable in each pair
            if col2 not in cols_to_remove:
                cols_to_remove.add(col2)
        
        # Update available_factors list
        available_factors = [col for col in available_factors if col not in cols_to_remove]
        
        print(f"\nReduced to {len(available_factors)} factors:")
        for factor in available_factors:
            print(f"  - {factor}")
        
        # Rebuild ONLY the feature matrices (not the full merged datasets)
        train_features = train_features[available_factors + ['Wins']].copy()
        test_features = test_features[available_factors + ['Wins']].copy()
        
        print(f"\nAfter removing correlated features:")
        print(f"  Training set: {len(train_features)} complete cases")
        print(f"  Test set: {len(test_features)} complete cases")
    else:
        print("No high multicollinearity detected (all |r| < 0.9)")



    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    X_train = train_features[available_factors].values
    y_train = train_features['Wins'].values
    
    X_test = test_features[available_factors].values
    y_test = test_features['Wins'].values
    
    # Fit model WITHOUT standardization (for interpretable coefficients)
    model_unstd = LinearRegression()
    model_unstd.fit(X_train, y_train)
    
    # Also fit standardized version (for variable importance comparison)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_std = LinearRegression()
    model_std.fit(X_train_scaled, y_train)
    
    print(f"\nModel trained on {len(X_train)} samples with {len(available_factors)} features")
    
    # =========================================================================
    # EVALUATE ON TRAINING SET
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING SET PERFORMANCE")
    print("="*80)
    
    y_train_pred = model_unstd.predict(X_train)
    train_residuals = y_train - y_train_pred
    
    ss_res_train = np.sum(train_residuals**2)
    ss_tot_train = np.sum((y_train - np.mean(y_train))**2)
    r2_train = 1 - (ss_res_train / ss_tot_train)
    
    n_train = len(train_features)
    p = len(available_factors)
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    
    rmse_train = np.sqrt(ss_res_train / n_train)
    mae_train = np.mean(np.abs(train_residuals))
    
    print(f"\n  R² = {r2_train:.4f}")
    print(f"  Adjusted R² = {adj_r2_train:.4f}")
    print(f"  RMSE = {rmse_train:.3f} wins")
    print(f"  MAE = {mae_train:.3f} wins")
    
    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE (HELD-OUT DATA)")
    print("="*80)
    
    y_test_pred = model_unstd.predict(X_test)
    test_residuals = y_test - y_test_pred
    
    ss_res_test = np.sum(test_residuals**2)
    ss_tot_test = np.sum((y_test - np.mean(y_test))**2)
    r2_test = 1 - (ss_res_test / ss_tot_test)
    
    rmse_test = np.sqrt(ss_res_test / len(y_test))
    mae_test = np.mean(np.abs(test_residuals))
    
    print(f"\n  R² = {r2_test:.4f}")
    print(f"  RMSE = {rmse_test:.3f} wins")
    print(f"  MAE = {mae_test:.3f} wins")
    
    r2_drop = r2_train - r2_test
    print(f"\n  R² drop (train→test) = {r2_drop:.4f}")
    
    if r2_drop > 0.1:
        print("  ⚠️  Warning: Significant performance drop suggests overfitting")
    elif r2_drop < 0:
        print("  ✓ Model generalizes well (test R² higher than train)")
    else:
        print("  ✓ Model generalizes well (minimal performance drop)")
    
    # =========================================================================
    # VARIABLE IMPORTANCE
    # =========================================================================
    print("\n" + "="*80)
    print("VARIABLE IMPORTANCE")
    print("="*80)
    
    # Create importance dataframe with BOTH versions
    importance_df = pd.DataFrame({
        'Variable': available_factors,
        'Coefficient': model_unstd.coef_,  # Raw, interpretable
        'Std_Coefficient': model_std.coef_,  # Standardized for comparison
        'Abs_Std_Coefficient': np.abs(model_std.coef_)
    })
    
    # Calculate partial R² on training data
    partial_r2_list = []
    for i, var in enumerate(available_factors):
        X_without = np.delete(X_train, i, axis=1)
        model_without = LinearRegression()
        model_without.fit(X_without, y_train)
        
        y_pred_without = model_without.predict(X_without)
        ss_res_without = np.sum((y_train - y_pred_without)**2)
        r2_without = 1 - (ss_res_without / ss_tot_train)
        
        partial_r2 = r2_train - r2_without
        partial_r2_list.append(partial_r2)
    
    importance_df['Partial_R2'] = partial_r2_list
    importance_df['Partial_R2_%'] = importance_df['Partial_R2'] * 100
    
    importance_df = importance_df.sort_values('Abs_Std_Coefficient', ascending=False)
    
    print("\nVariable Importance:")
    print("  Coefficient = Raw coefficient (interpretable, e.g., +0.05 means 1 unit increase → +0.05 wins)")
    print("  Std_Coefficient = Standardized (for comparing importance across different scales)")
    print("  Partial_R2_% = Unique variance explained by this variable\n")
    display(importance_df[['Variable', 'Coefficient', 'Std_Coefficient', 'Partial_R2_%']])
    
    # =========================================================================
    # MODEL EQUATION (INTERPRETABLE)
    # =========================================================================
    print("\n" + "="*80)
    print("PREDICTION EQUATION (Raw Coefficients - Human Readable)")
    print("="*80)
    
    print(f"\nPredicted Wins = {model_unstd.intercept_:.3f}")
    for var, coef in zip(available_factors, model_unstd.coef_):
        sign = "+" if coef >= 0 else ""
        print(f"  {sign} {coef:.6f} × {var}")
    
    print("\nInterpretation: Each coefficient shows the change in wins for a 1-unit")
    print("increase in that stat (holding all other variables constant)")
    print("\nExample:")
    example_var = available_factors[0]
    example_coef = model_unstd.coef_[0]
    if example_coef > 0:
        print(f"  If {example_var} increases by 100, team wins increase by ~{example_coef*100:.2f}")
    else:
        print(f"  If {example_var} increases by 100, team wins decrease by ~{abs(example_coef)*100:.2f}")
    
    # =========================================================================
    # RESIDUAL ANALYSIS ON TEST SET
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SET RESIDUAL ANALYSIS")
    print("="*80)
    
    test_features_copy = test_features.copy()
    test_features_copy['Predicted_Wins'] = y_test_pred
    test_features_copy['Residuals'] = test_residuals
    test_features_copy['Abs_Residual'] = np.abs(test_residuals)
    
    print("\nTop 5 Over-predicted (predicted more wins than actual):")
    over_pred = test_features_copy.nlargest(5, 'Residuals')[['Wins', 'Predicted_Wins', 'Residuals']]
    display(over_pred)
    
    print("\nTop 5 Under-predicted (predicted fewer wins than actual):")
    under_pred = test_features_copy.nsmallest(5, 'Residuals')[['Wins', 'Predicted_Wins', 'Residuals']]
    display(under_pred)
    
    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    top_3_vars = importance_df.head(3)['Variable'].tolist()
    
    print("\nTop 3 Most Important Variables:")
    for i, var in enumerate(top_3_vars, 1):
        coef = importance_df[importance_df['Variable']==var]['Coefficient'].values[0]
        std_coef = importance_df[importance_df['Variable']==var]['Std_Coefficient'].values[0]
        partial_r2 = importance_df[importance_df['Variable']==var]['Partial_R2_%'].values[0]
        print(f"  {i}. {var}")
        print(f"     Raw β = {coef:.6f} (interpretable)")
        print(f"     Std β = {std_coef:.3f} (for comparison)")
        print(f"     Partial R² = {partial_r2:.2f}%")
    
    print("\n" + "="*80)
    print("FOR TRAJECTORY VISUALIZATION:")
    print("="*80)
    
    best_single_metric = top_3_vars[0]
    print(f"\nRecommended Y-axis metric: {best_single_metric}")
    print(f"  - Highest importance in multivariate model")
    print(f"  - Model R² on test set: {r2_test:.4f}")
    
    print("\nAlternative: Use composite score from top 3 metrics:")
    for var in top_3_vars:
        print(f"  - {var}")
    
    # Save results
    results = {
        'model': model_unstd,  # Use unstandardized model
        'model_standardized': model_std,  # Keep standardized for reference
        'scaler': scaler,
        'features': available_factors,
        'r2_train': r2_train,
        'adj_r2_train': adj_r2_train,
        'rmse_train': rmse_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'variable_importance': importance_df,
        'test_predictions': test_features_copy[['Wins', 'Predicted_Wins', 'Residuals']],
        'best_metric': best_single_metric
    }
    
    # Save to CSV
    importance_df.to_csv('pc1_regression_variable_importance.csv', index=False)
    test_features_copy[['Wins', 'Predicted_Wins', 'Residuals']].to_csv('pc1_regression_test_predictions.csv')
    
    print("\nSaved results to:")
    print("  - pc1_regression_variable_importance.csv")
    print("  - pc1_regression_test_predictions.csv")
    
    return results

def ridge_regression_with_cv(train_df=None, test_df=None, alpha_range=None, use_extended_features=True):
    """
    Performs Ridge regression with 5-fold cross-validation.
    
    Args:
        train_df: Training data
        test_df: Test data
        alpha_range: List of alpha values to test
        use_extended_features: If True, uses top PC1 factors; if False, uses original 5
    
    Returns:
        dict: Results including best model, coefficients, and CV scores
    """
    # .
   
    print("\n" + "="*80)
    print("RIDGE REGRESSION WITH 5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Load train/test data if not provided
    if train_df is None or test_df is None:
        print("\nLoading train/test split from files...")
        train_df, test_df = load_train_test_split()
        if train_df is None or test_df is None:
            print("Error: Could not load train/test data")
            return None
    
    # Load season records
    season_records = load_csv_safe("season_records.csv", "season records")
    if season_records is None:
        return None
    
    required_cols = ['Season', 'Team', 'W', 'W-L%']
    if not validate_columns(season_records, required_cols, "season records"):
        return None
    
    # Convert Season column to int
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Process training data
    print("\n" + "="*80)
    print("PROCESSING TRAINING DATA")
    print("="*80)
    
    train_df['season'] = pd.to_numeric(train_df['season'], errors='coerce')
    train_df = train_df.dropna(subset=['season'])
    train_df['season'] = train_df['season'].astype(int)
    
    train_merged = pd.merge(
        train_df,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"Training set: {len(train_merged)} QB season-team records")
    
    train_merged['GS_numeric'] = pd.to_numeric(train_merged['GS'], errors='coerce')
    train_merged = train_merged[train_merged['GS_numeric'] >= 8]
    print(f"After filtering (8+ GS): {len(train_merged)} records")
    
    train_merged['Wins'] = pd.to_numeric(train_merged['W'], errors='coerce')
    train_merged = train_merged.dropna(subset=['Wins'])
    
    # Process test data
    print("\n" + "="*80)
    print("PROCESSING TEST DATA")
    print("="*80)
    
    test_df['season'] = pd.to_numeric(test_df['season'], errors='coerce')
    test_df = test_df.dropna(subset=['season'])
    test_df['season'] = test_df['season'].astype(int)
    
    test_merged = pd.merge(
        test_df,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"Test set: {len(test_merged)} QB season-team records")
    
    test_merged['GS_numeric'] = pd.to_numeric(test_merged['GS'], errors='coerce')
    test_merged = test_merged[test_merged['GS_numeric'] >= 8]
    print(f"After filtering (8+ GS): {len(test_merged)} records")
    
    test_merged['Wins'] = pd.to_numeric(test_merged['W'], errors='coerce')
    test_merged = test_merged.dropna(subset=['Wins'])
    
    if use_extended_features:
        print("\nUsing EXTENDED feature set (top PC1 loadings)")
        performance_factors = [
            # Core efficiency
            'Pass_TD',              
            'Pass_ANY/A',           
            'Rush_Rushing_Succ%',   
            'Pass_Int%',            
            'Pass_Sk%',
            
            # Volume
            'total_yards',          
            
            # Clutch
            'Pass_4QC',             
            'Pass_GWD',             
            
            # Dual-threat
            'Rush_Rushing_TD',      
            'Rush_Rushing_Yds',     
        ]
    else:
        print("\nUsing ORIGINAL feature set (manual selection)")
        performance_factors = [
            'Pass_TD',              # Scoring ability
            'Pass_ANY/A',           # Best efficiency metric
            'Rush_Rushing_Succ%',   # Dual-threat efficiency
            'Pass_Int%',            # Turnover avoidance
            'Pass_Sk%',             # Pressure handling
        ]
    
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    
    print(f"\nUsing {len(performance_factors)} performance factors:")
    for factor in performance_factors:
        print(f"  - {factor}")
    
    # Check which factors are available
    available_factors = [f for f in performance_factors if f in train_merged.columns]
    missing_factors = [f for f in performance_factors if f not in train_merged.columns]
    
    if missing_factors:
        print(f"\nWarning: {len(missing_factors)} factors not found:")
        for factor in missing_factors:
            print(f"  - {factor}")
    
    print(f"\nProceeding with {len(available_factors)} available factors")
    
    # Build datasets
    train_features = train_merged[available_factors + ['Wins']].copy()
    for col in available_factors:
        train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
    train_features = train_features.dropna()
    
    test_features = test_merged[available_factors + ['Wins']].copy()
    for col in available_factors:
        test_features[col] = pd.to_numeric(test_features[col], errors='coerce')
    test_features = test_features.dropna()
    
    print(f"\nTraining set complete cases: {len(train_features)}")
    print(f"Test set complete cases: {len(test_features)}")
    
    if len(train_features) < 50 or len(test_features) < 10:
        print("Insufficient data for regression")
        return None
    
    # Prepare data
    X_train = train_features[available_factors].values
    y_train = train_features['Wins'].values
    
    X_test = test_features[available_factors].values
    y_test = test_features['Wins'].values
    
    # Standardize features (CRITICAL for Ridge regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # =========================================================================
    # RIDGE REGRESSION WITH CROSS-VALIDATION
    # =========================================================================
    print("\n" + "="*80)
    print("CROSS-VALIDATION TO SELECT OPTIMAL ALPHA")
    print("="*80)
    
    # Define alpha range to test
    if alpha_range is None:
        alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    print(f"\nTesting {len(alpha_range)} alpha values with 5-fold CV:")
    print(f"Alpha range: {alpha_range}")
    
    # Use RidgeCV to automatically find best alpha
    ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring='r2')
    ridge_cv.fit(X_train_scaled, y_train)
    
    best_alpha = ridge_cv.alpha_
    print(f"\nBest alpha (regularization strength): {best_alpha}")
    
    # =========================================================================
    # CROSS-VALIDATION SCORES
    # =========================================================================
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION PERFORMANCE")
    print("="*80)
    
    # Fit final model with best alpha
    ridge_model = Ridge(alpha=best_alpha)
    
    # Get CV scores
    cv_r2_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_rmse_scores = -cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, 
                                       scoring='neg_root_mean_squared_error')
    
    print(f"\nCross-Validation R² Scores (5 folds):")
    for i, score in enumerate(cv_r2_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"\nMean CV R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    
    print(f"\nCross-Validation RMSE Scores (5 folds):")
    for i, score in enumerate(cv_rmse_scores, 1):
        print(f"  Fold {i}: {score:.3f} wins")
    print(f"\nMean CV RMSE: {cv_rmse_scores.mean():.3f} (+/- {cv_rmse_scores.std() * 2:.3f}) wins")
    
    # =========================================================================
    # TRAIN FINAL MODEL
    # =========================================================================
    ridge_model.fit(X_train_scaled, y_train)
    
    # Training performance
    y_train_pred = ridge_model.predict(X_train_scaled)
    train_r2 = ridge_model.score(X_train_scaled, y_train)
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
    
    print("\n" + "="*80)
    print("TRAINING SET PERFORMANCE")
    print("="*80)
    print(f"\n  R² = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.3f} wins")
    
    # Test performance
    y_test_pred = ridge_model.predict(X_test_scaled)
    test_r2 = ridge_model.score(X_test_scaled, y_test)
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
    
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE (HELD-OUT DATA)")
    print("="*80)
    print(f"\n  R² = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.3f} wins")
    
    r2_drop = train_r2 - test_r2
    print(f"\n  R² drop (train→test) = {r2_drop:.4f}")
    
    if r2_drop > 0.1:
        print("  ⚠️  Warning: Significant performance drop suggests overfitting")
    else:
        print("  ✓ Model generalizes well")
    
    # =========================================================================
    # VARIABLE IMPORTANCE (STANDARDIZED COEFFICIENTS)
    # =========================================================================
    print("\n" + "="*80)
    print("VARIABLE IMPORTANCE (Ridge Regression)")
    print("="*80)
    
    # Ridge coefficients are already on standardized scale
    importance_df = pd.DataFrame({
        'Variable': available_factors,
        'Ridge_Coefficient': ridge_model.coef_,
        'Abs_Coefficient': np.abs(ridge_model.coef_)
    })
    
    importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nRidge Regression Coefficients (standardized):")
    print("  Interpretation: Effect on wins for 1 std dev change in predictor")
    print("  (controlling for other variables)")
    display(importance_df)
    
    # =========================================================================
    # COMPARISON WITH OLS RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON: Ridge vs OLS")
    print("="*80)
    
    # Load OLS results if available
    if os.path.exists('pc1_regression_variable_importance.csv'):
        ols_importance = load_csv_safe('pc1_regression_variable_importance.csv')
        
        comparison_df = importance_df.merge(
            ols_importance[['Variable', 'Std_Coefficient']], 
            on='Variable',
            how='left'
        )
        comparison_df = comparison_df.rename(columns={'Std_Coefficient': 'OLS_Std_Coefficient'})
        comparison_df['Difference'] = comparison_df['Ridge_Coefficient'] - comparison_df['OLS_Std_Coefficient']
        comparison_df['Shrinkage_%'] = (1 - np.abs(comparison_df['Ridge_Coefficient']) / 
                                         np.abs(comparison_df['OLS_Std_Coefficient'])) * 100
        
        print("\nCoefficient Comparison:")
        print("(Ridge applies shrinkage - coefficients pulled toward zero)")
        display(comparison_df[['Variable', 'OLS_Std_Coefficient', 'Ridge_Coefficient', 'Shrinkage_%']])
        
        print("\nKey Insights:")
        print("  - Variables with large shrinkage may be less stable/reliable")
        print("  - Variables that maintain magnitude are more robust predictors")
        print(f"  - Mean shrinkage: {comparison_df['Shrinkage_%'].mean():.1f}%")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'model': ridge_model,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'features': available_factors,
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'variable_importance': importance_df
    }
    
    importance_df.to_csv('ridge_regression_variable_importance.csv', index=False)
    print("\n✓ Results saved to: ridge_regression_variable_importance.csv")
    
    return results

def load_or_create_cache(cache_file, creation_function, *args, force_refresh=False, **kwargs):
    """
    Generic caching function to avoid recomputing expensive operations.
    
    Args:
        cache_file (str): Path to cache file (CSV)
        creation_function (callable): Function to call if cache doesn't exist
        *args: Positional arguments for creation_function
        force_refresh (bool): If True, ignore cache and recreate
        **kwargs: Keyword arguments for creation_function
    
    Returns:
        DataFrame: Cached or newly created data
    """
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached data from: {cache_file}")
        return load_csv_safe(cache_file)
    else:
        print(f"Cache not found or refresh forced. Creating new data...")
        df = creation_function(*args, **kwargs)
        if df is not None and not df.empty:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            df.to_csv(cache_file, index=False)
            print(f"Data cached to: {cache_file}")
        return df

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

def validate_contract_mapping(mapped_contracts_df):
    """
    Validates the contract-to-player-ID mapping and generates a report.
    
    Note: This includes ALL players who got contracts, not just first-round QBs.
    You can filter to first-round QBs later if needed.
    
    Args:
        mapped_contracts_df: DataFrame with player_id mapping
    
    Returns:
        dict: Validation report with statistics and issues
    """
    print("\n" + "="*80)
    print("VALIDATING CONTRACT MAPPING")
    print("="*80)
    
    report = {
        'total_contracts': len(mapped_contracts_df),
        'mapped_contracts': mapped_contracts_df['player_id'].notna().sum(),
        'unmapped_contracts': mapped_contracts_df['player_id'].isna().sum(),
        'unique_players': mapped_contracts_df['player_id'].nunique(),
        'issues': []
    }
    
    # Filter to only mapped contracts for validation
    mapped_only = mapped_contracts_df[mapped_contracts_df['player_id'].notna()].copy()
    
    if len(mapped_only) == 0:
        print("⚠️  Warning: No contracts mapped to player IDs")
        return report
    
    # Check for duplicate contracts (same player, year)
    duplicates = mapped_only.groupby(['player_id', 'Year']).size()
    duplicates = duplicates[duplicates > 1]
    
    if len(duplicates) > 0:
        report['issues'].append(f"Found {len(duplicates)} player-years with multiple contracts")
        print(f"⚠️  Warning: {len(duplicates)} player-years have multiple contracts")
        print("   (This might be extensions + restructures in same year)")
        
        # Show examples
        for (player_id, year), count in duplicates.head(3).items():
            examples = mapped_only[(mapped_only['player_id'] == player_id) & (mapped_only['Year'] == year)]
            player_name = examples.iloc[0]['matched_name']
            print(f"   Example: {player_name} in {year} has {count} contracts")
    
    print(f"\n✓ Validation complete")
    print(f"  Total contracts: {report['total_contracts']}")
    print(f"  Mapped: {report['mapped_contracts']} ({report['mapped_contracts']/report['total_contracts']*100:.1f}%)")
    print(f"  Unique players: {report['unique_players']}")
    
    if len(report['issues']) == 0:
        print("  No issues found")
    
    return report

def test_name_mapping():
    """
    Quick test of the name normalization functionality.
    """
    print("\n" + "="*80)
    print("TESTING NAME NORMALIZATION")
    print("="*80)
    
    # Test normalization
    test_names = [
        ("Patrick Mahomes II", "patrick mahomes"),
        ("Baker Mayfield Jr.", "baker mayfield"),
        ("Joe Burrow", "joe burrow"),
        ("LAMAR JACKSON", "lamar jackson"),
        ("Josh Allen", "josh allen"),
        ("Tom   Brady", "tom brady"),  # Extra spaces
        ("Peyton Manning Jr", "peyton manning"),
    ]
    
    print("\nName normalization test:")
    all_pass = True
    for original, expected in test_names:
        normalized = normalize_player_name(original)
        passed = normalized == expected
        status = "✓" if passed else "✗"
        print(f"  {status} '{original}' → '{normalized}' (expected: '{expected}')")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ All normalization tests passed!")
    else:
        print("\n✗ Some normalization tests failed")
    
    return all_pass

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
    
    # Validate
    report = validate_contract_mapping(mapped_contracts)
    
    # Save validation report
    with open('cache/contract_mapping_report.txt', 'w') as f:
        f.write("CONTRACT MAPPING VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    
    print("\n✓ Pipeline complete!")
    print(f"  Results saved to: cache/contract_player_id_mapping.csv")
    print(f"  Report saved to: cache/contract_mapping_report.txt")
    
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
    
    paid_contracts['team_normalized'] = paid_contracts['Team'].map(team_mapping).fillna(paid_contracts['Team'])
    
    # For each player, find FIRST contract with draft team in Years 2-6
    payment_mapping = {}
    excluded_other_team = 0
    excluded_outside_window = 0
    
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
    
    print(f"\nCreated payment mapping for {len(payment_mapping)} players")
    print(f"Excluded {excluded_other_team} players who signed with different team")
    print(f"Excluded {excluded_outside_window} players with no contracts in Years 2-6 window")
    
    # Show examples
    if payment_mapping:
        print("\nExample mappings (Years 2-6 with draft team):")
        examples = sorted(payment_mapping.items(), key=lambda x: x[1], reverse=True)[:10]
        for player_id, year in examples:
            player_rows = paid_contracts[paid_contracts['player_id'] == player_id]
            if len(player_rows) > 0:
                player_name = player_rows.iloc[0].get('Player', player_rows.iloc[0].get('matched_name', 'Unknown'))
                draft_year = int(player_rows.iloc[0]['draft_year'])
                draft_team = player_rows.iloc[0]['draft_team']
                years_later = year - draft_year
                print(f"  {player_name} ({draft_team}): Drafted {draft_year}, Paid (Year {years_later}) = {year}")
    
    return payment_mapping

def validate_payment_data(df):
    """Validates the payment-labeled dataset."""
    print("\n" + "="*80)
    print("PAYMENT DATA VALIDATION REPORT")
    print("="*80)
    
    report = {
        'total_seasons': len(df),
        'unique_players': df['player_id'].nunique(),
        'qbs_who_got_paid': len(df[df['got_paid'] == True]['player_id'].unique()),
        'qbs_who_did_not_get_paid': len(df[df['got_paid'] == False]['player_id'].unique()),
        'issues': []
    }
    
    print(f"\n📊 BASIC STATISTICS")
    print(f"   Total seasons: {report['total_seasons']:,}")
    print(f"   Unique players: {report['unique_players']}")
    print(f"   QBs who got paid: {report['qbs_who_got_paid']}")
    print(f"   QBs who did not get paid: {report['qbs_who_did_not_get_paid']}")
    
    paid_seasons = df[df['got_paid'] == True]
    if len(paid_seasons) > 0:
        print(f"\n💰 PAYMENT DISTRIBUTION")
        print(f"   Seasons from paid QBs: {len(paid_seasons):,}")
        print(f"   Payment years range: {int(paid_seasons['payment_year'].min())}-{int(paid_seasons['payment_year'].max())}")
        
        print(f"\n📅 YEARS RELATIVE TO PAYMENT")
        year_dist = paid_seasons['years_to_payment'].value_counts().sort_index()
        for year, count in year_dist.items():
            if pd.notna(year):
                print(f"   Y{int(year):+d}: {count:>4} seasons")
    
    print(f"\n🔍 LOOKBACK FEATURES")
    lag_cols = [col for col in df.columns if '_lag' in col]
    if lag_cols:
        for col in lag_cols[:6]:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"   {col}: {non_null:>5} ({pct:>5.1f}%)")
        if len(lag_cols) > 6:
            print(f"   ... and {len(lag_cols)-6} more")
    else:
        report['issues'].append("No lag features found!")
    
    print(f"\n👥 SAMPLE QBs WHO GOT PAID")
    paid_qbs = df[df['got_paid'] == True].groupby('player_id').agg({
        'player_name': 'first',
        'payment_year': 'first',
        'draft_year': 'first',
        'pick_number': 'first'
    }).dropna(subset=['draft_year']).sort_values('payment_year', ascending=False).head(10)
    
    for idx, row in paid_qbs.iterrows():
        years_to_pay = int(row['payment_year'] - row['draft_year'])
        print(f"   {row['player_name']}: Drafted {int(row['draft_year'])} (#{int(row['pick_number'])}), Paid {int(row['payment_year'])} ({years_to_pay} yrs)")
    
    print(f"\n👥 SAMPLE QBs WHO DID NOT GET PAID")
    unpaid_qbs = df[df['got_paid'] == False].groupby('player_id').agg({
        'player_name': 'first',
        'draft_year': 'first',
        'pick_number': 'first',
        'season': 'max'
    }).dropna(subset=['draft_year']).sort_values('draft_year', ascending=False).head(10)
    
    for idx, row in unpaid_qbs.iterrows():
        print(f"   {row['player_name']}: Drafted {int(row['draft_year'])} (#{int(row['pick_number'])}), Last season {int(row['season'])}")
    
    return report

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

def label_seasons_relative_to_payment(qb_seasons_df, payment_mapping):
    """
    Labels each QB season with years relative to their payment year.
    
    For QBs who got paid in year Y:
    - Y-4 = 4 years before payment
    - Y-3 = 3 years before payment
    - Y-2 = 2 years before payment
    - Y-1 = 1 year before payment (most recent year before decision)
    - Y = payment year
    - Y+1 = 1 year after payment
    - Y+2 = 2 years after payment
    etc.
    
    For QBs who never got paid: years_to_payment = NaN
    
    Args:
        qb_seasons_df (DataFrame): All QB seasons data
        payment_mapping (dict): {player_id: payment_year}
    
    Returns:
        DataFrame: Original df with new columns:
            - payment_year: The year they got paid (NaN if never paid)
            - years_to_payment: Relative year (negative before, 0 at, positive after)
            - got_paid: Boolean indicator
    """
    print("\n" + "="*80)
    print("LABELING SEASONS RELATIVE TO PAYMENT")
    print("="*80)
    
    required_cols = ['season', 'player_id']
    if not validate_columns(qb_seasons_df, required_cols, "QB seasons"):
        return None

    df = qb_seasons_df.copy()
    
    # Ensure season is integer
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(subset=['season'])
    df['season'] = df['season'].astype(int)
    
    # Add payment year from mapping
    df['payment_year'] = df['player_id'].map(payment_mapping)
    
    # Filter out invalid payment years (0 or negative)
    invalid_payments = df[(df['payment_year'].notna()) & (df['payment_year'] <= 0)]
    if len(invalid_payments) > 0:
        print(f"⚠️  Filtering out {len(invalid_payments)} records with invalid payment_year <= 0")
        df.loc[invalid_payments.index, 'payment_year'] = np.nan
    
    # Calculate years relative to payment
    # Negative = before payment, 0 = payment year, positive = after payment
    df['years_to_payment'] = df['season'] - df['payment_year']
    
    # Add boolean indicator
    df['got_paid'] = df['payment_year'].notna()
    
    df = validate_payment_years(df, 
                           draft_year_col='draft_year', 
                           payment_year_col='payment_year',
                           years_to_payment_col='years_to_payment',
                           max_years=12)

    # Stats
    paid_qbs = df['got_paid'].sum() / len(df) * 100
    unique_paid = df[df['got_paid']]['player_id'].nunique()
    unique_total = df['player_id'].nunique()
    
    print(f"\nTotal seasons: {len(df)}")
    print(f"Seasons from QBs who got paid: {df['got_paid'].sum()} ({paid_qbs:.1f}%)")
    print(f"QBs who got paid: {unique_paid}/{unique_total}")
    
    # Show distribution of years_to_payment
    print("\nDistribution of years relative to payment:")
    paid_df = df[df['got_paid']].copy()
    if len(paid_df) > 0:
        for year_rel in sorted(paid_df['years_to_payment'].dropna().unique()):
            count = (paid_df['years_to_payment'] == year_rel).sum()
            print(f"  Y{int(year_rel):+d}: {count} seasons")
    
    # Show examples
    print("\nExample labeled seasons:")
    sample_qb = df[df['got_paid']].iloc[0]['player_id']
    sample_data = df[df['player_id'] == sample_qb][['season', 'player_name', 'payment_year', 'years_to_payment', 'got_paid']].head(10)
    print(sample_data.to_string(index=False))
    
    return df

def create_lookback_performance_features(labeled_df, metrics=['Pass_ANY/A', 'Pass_Rate', 'total_yards']):
    """
    Creates lookback features for discrete-time modeling.
    
    For each season at time t, creates columns for performance in previous years:
    - metric_lag1: Performance 1 year ago (t-1)
    - metric_lag2: Performance 2 years ago (t-2)
    - metric_lag3: Performance 3 years ago (t-3)
    - metric_lag4: Performance 4 years ago (t-4)
    
    This enables modeling payment decisions based on recent performance history.
    
    Args:
        labeled_df (DataFrame): QB seasons with years_to_payment labels
        metrics (list): Performance metrics to create lags for
    
    Returns:
        DataFrame: Original df with lag features added
    """
    print("\n" + "="*80)
    print("CREATING LOOKBACK PERFORMANCE FEATURES")
    print("="*80)
    
    required_cols = ['player_id', 'season']
    if not validate_columns(labeled_df, required_cols, "labeled seasons"):
        return None
    
    # Sort by player and season
    df = labeled_df.sort_values(['player_id', 'season']).reset_index(drop=True)

    # For each metric, create lag features
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️  Warning: {metric} not found in dataframe, skipping")
            continue
        
        print(f"\nCreating lags for {metric}...")
        
        # Convert to numeric
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        # Create lags (1-4 years back)
        for lag in range(1, 5):
            lag_col = f"{metric}_lag{lag}"
            df[lag_col] = df.groupby('player_id')[metric].shift(lag)
            
            # Count non-null values
            non_null = df[lag_col].notna().sum()
            print(f"  {lag_col}: {non_null} non-null values")
    
    print(f"\nTotal columns after adding lags: {len(df.columns)}")
    
    # Show example of lookback features
    print("\nExample lookback features (showing Pass_ANY/A):")
    sample_qb = df[df['got_paid']].iloc[0]['player_id']
    sample_data = df[df['player_id'] == sample_qb][
        ['season', 'player_name', 'Pass_ANY/A', 'Pass_ANY/A_lag1', 'Pass_ANY/A_lag2', 'Pass_ANY/A_lag3']
    ].head(10)
    print(sample_data.to_string(index=False))
    
    return df

def create_decision_point_dataset(labeled_df, decision_year_relative=0, metrics=['Pass_ANY/A']):
    """
    Creates a dataset for modeling payment decisions at a specific decision point.
    
    For example, decision_year_relative=-1 means we're modeling whether the QB got paid
    based on their performance up to 1 year before payment (Y-1, Y-2, Y-3, Y-4).
    
    This is useful for discrete-time logistic regression at each decision point:
    - Year 3 decisions (5th year option): decision_year_relative=-2
    - Year 4 decisions (extension): decision_year_relative=-1
    - Year 5 decisions (2nd contract): decision_year_relative=0
    
    Args:
        labeled_df (DataFrame): QB seasons with lookback features
        decision_year_relative (int): Which year to evaluate at (relative to payment)
        metrics (list): Which performance metrics to include
    
    Returns:
        DataFrame: Dataset ready for modeling payment probability
    """
    print("\n" + "="*80)
    print(f"CREATING DECISION POINT DATASET (Y{decision_year_relative:+d})")
    print("="*80)
    
    df = labeled_df.copy()
    
    # Filter to the specific decision year
    decision_df = df[df['years_to_payment'] == decision_year_relative].copy()
    
    print(f"Seasons at Y{decision_year_relative:+d}: {len(decision_df)}")
    
    # Select relevant columns
    feature_cols = ['player_id', 'player_name', 'season', 'draft_year', 'got_paid']
    
    # Add the performance metrics and their lags
    for metric in metrics:
        if metric in decision_df.columns:
            feature_cols.append(metric)
        
        # Add lags
        for lag in range(1, 5):
            lag_col = f"{metric}_lag{lag}"
            if lag_col in decision_df.columns:
                feature_cols.append(lag_col)
    
    # Add pick number if available
    if 'pick_number' in decision_df.columns:
        feature_cols.append('pick_number')
    
    decision_df = decision_df[feature_cols].copy()
    
    # Remove rows with missing lag data (can't evaluate without history)
    initial_count = len(decision_df)
    decision_df = decision_df.dropna(subset=[col for col in feature_cols if '_lag' in col], how='all')
    final_count = len(decision_df)
    
    print(f"After removing seasons with insufficient history: {final_count} ({initial_count - final_count} dropped)")
    
    # Show payment statistics
    if 'got_paid' in decision_df.columns:
        paid_count = decision_df['got_paid'].sum()
        paid_pct = paid_count / len(decision_df) * 100 if len(decision_df) > 0 else 0
        print(f"\nPayment outcome distribution:")
        print(f"  Got paid: {paid_count} ({paid_pct:.1f}%)")
        print(f"  Not paid: {len(decision_df) - paid_count} ({100-paid_pct:.1f}%)")
    
    # Show sample
    print("\nSample decision point data:")
    sample_cols = ['player_name', 'season', 'got_paid'] + [col for col in feature_cols if 'Pass_ANY/A' in col][:4]
    print(decision_df[sample_cols].head(10).to_string(index=False))
    
    return decision_df

def prepare_qb_payment_data(
    qb_seasons_file='all_seasons_df.csv',
    pick_numbers_file='first_round_qbs_with_picks.csv',
    save_output=True,
    output_file='qb_seasons_payment_labeled.csv'
):
    """
    Master function that runs the full data preparation pipeline.
    
    Steps:
    1. Load QB seasons data
    2. Load and map contract data (payment years) using existing cached mapping
    3. Load and map pick numbers
    4. Label seasons relative to payment
    5. Create lookback performance features
    
    Args:
        qb_seasons_file (str): Path to QB seasons CSV
        pick_numbers_file (str): Path to pick numbers CSV
        save_output (bool): Whether to save the prepared dataset
        output_file (str): Where to save the output
    
    Returns:
        DataFrame: Fully prepared dataset ready for modeling
    """
    print("\n" + "="*80)
    print("MASTER DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Step 1: Load QB seasons
    print("\n[1/5] Loading QB seasons data...")
    if not os.path.exists(qb_seasons_file):
        print(f"✗ ERROR: {qb_seasons_file} not found")
        return None

    qb_seasons = load_csv_safe(qb_seasons_file, "QB seasons data")

    if qb_seasons is None:
        return None

    required_cols = ['season', 'player_id', 'Team']
    if not validate_columns(qb_seasons, required_cols, "QB seasons"):
        return None

    print(f"✓ Loaded {len(qb_seasons)} QB seasons")
    
    # Step 2: Create payment mapping from existing contract mapping
    print("\n[2/5] Creating payment year mapping...")

    # Use the cached contract mapping, create if missing
    if os.path.exists('cache/contract_player_id_mapping.csv'):
        contracts = load_csv_safe('cache/contract_player_id_mapping.csv')
        print(f"✓ Loaded {len(contracts)} contracts from cache")
    else:
        print(f"⚠️  cache/contract_player_id_mapping.csv not found")
        print("   Auto-creating contract mapping now...")
        
        # Auto-create the mapping
        mapped_contracts = create_contract_player_mapping(force_refresh=False)

        if mapped_contracts is None:
            print("✗ ERROR: Failed to create contract mapping")
            print("   Make sure QB_contract_data.csv and player_ids.csv exist")
            return None
        
        contracts = mapped_contracts
        print(f"✓ Created mapping for {len(contracts)} contracts")
    
    payment_mapping = create_payment_year_mapping(contracts)
    
    # Step 3: Create pick number mapping
    print("\n[3/5] Creating pick number mapping...")
    pick_mapping = create_pick_number_mapping(pick_numbers_file)
    
    # Add pick numbers to QB seasons
    qb_seasons['pick_number'] = qb_seasons['player_id'].map(pick_mapping)
    print(f"✓ Added pick numbers to {qb_seasons['pick_number'].notna().sum()} seasons")
    
    # Step 4: Label seasons relative to payment
    print("\n[4/5] Labeling seasons relative to payment...")
    labeled_df = label_seasons_relative_to_payment(qb_seasons, payment_mapping)
    
    if labeled_df is None:
        print("✗ ERROR: Failed to label seasons")
        return None
    
    # Step 5: Create lookback features
    print("\n[5/5] Creating lookback performance features...")
    metrics_to_lag = ['Pass_ANY/A', 'Pass_Rate', 'Pass_TD', 'Pass_Yds', 'total_yards', 'Pass_Succ%']
    final_df = create_lookback_performance_features(labeled_df, metrics=metrics_to_lag)
    
    if final_df is None:
        print("✗ ERROR: Failed to create lookback features")
        return None
    
    # Save if requested
    if save_output:
        final_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved prepared dataset to: {output_file}")
        print(f"  Total rows: {len(final_df)}")
        print(f"  Total columns: {len(final_df.columns)}")
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    
    return final_df

def validate_payment_data(df):
    """Validates the payment-labeled dataset."""
    print("\n" + "="*80)
    print("PAYMENT DATA VALIDATION REPORT")
    print("="*80)
    
    report = {
        'total_seasons': len(df),
        'unique_players': df['player_id'].nunique(),
        'qbs_who_got_paid': len(df[df['got_paid'] == True]['player_id'].unique()),
        'qbs_who_did_not_get_paid': len(df[df['got_paid'] == False]['player_id'].unique()),
        'issues': []
    }
    
    print(f"\n📊 BASIC STATISTICS")
    print(f"   Total seasons: {report['total_seasons']:,}")
    print(f"   Unique players: {report['unique_players']}")
    print(f"   QBs who got paid: {report['qbs_who_got_paid']}")
    print(f"   QBs who did not get paid: {report['qbs_who_did_not_get_paid']}")
    
    paid_seasons = df[df['got_paid'] == True]
    if len(paid_seasons) > 0:
        print(f"\n💰 PAYMENT DISTRIBUTION")
        print(f"   Seasons from paid QBs: {len(paid_seasons):,}")
        print(f"   Payment years range: {int(paid_seasons['payment_year'].min())}-{int(paid_seasons['payment_year'].max())}")
        
        print(f"\n📅 YEARS RELATIVE TO PAYMENT")
        year_dist = paid_seasons['years_to_payment'].value_counts().sort_index()
        for year, count in year_dist.items():
            if pd.notna(year):
                print(f"   Y{int(year):+d}: {count:>4} seasons")
    
    print(f"\n🔍 LOOKBACK FEATURES")
    lag_cols = [col for col in df.columns if '_lag' in col]
    if lag_cols:
        for col in lag_cols[:6]:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"   {col}: {non_null:>5} ({pct:>5.1f}%)")
        if len(lag_cols) > 6:
            print(f"   ... and {len(lag_cols)-6} more")
    else:
        report['issues'].append("No lag features found!")
    
    print(f"\n👥 SAMPLE QBs WHO GOT PAID")
    paid_qbs = df[df['got_paid'] == True].groupby('player_id').agg({
        'player_name': 'first',
        'payment_year': 'first',
        'draft_year': 'first',
        'pick_number': 'first'
    }).sort_values('payment_year', ascending=False).head(10)
    
    for idx, row in paid_qbs.iterrows():
        years_to_pay = int(row['payment_year'] - row['draft_year'])
        print(f"   {row['player_name']}: Drafted {int(row['draft_year'])} (#{int(row['pick_number'])}), Paid {int(row['payment_year'])} ({years_to_pay} yrs)")
    
    print(f"\n👥 SAMPLE QBs WHO DID NOT GET PAID")
    unpaid_qbs = df[df['got_paid'] == False].groupby('player_id').agg({
        'player_name': 'first',
        'draft_year': 'first',
        'pick_number': 'first',
        'season': 'max'
    }).sort_values('draft_year', ascending=False).head(10)
    
    for idx, row in unpaid_qbs.iterrows():
        print(f"   {row['player_name']}: Drafted {int(row['draft_year'])} (#{int(row['pick_number'])}), Last season {int(row['season'])}")
    
    return report

def ridge_regression_payment_prediction(alpha_range=None, exclude_recent_drafts=True):
    """
    Ridge regression to identify what predicts getting paid.
    
    Uses AVERAGED performance metrics across Years 1-3:
    - QB metrics: total_yards, Pass_TD, Pass_ANY/A, Rush_Rushing_Succ%
    - Team metrics: W (Wins), Pts (Points For)
    
    Args:
        alpha_range: List of alpha values to test
        exclude_recent_drafts: If True, exclude QBs drafted after 2020 (haven't reached contract window)
    
    Returns:
        dict: Results including most important predictor
    """
    print("\n" + "="*80)
    print("RIDGE REGRESSION: WHAT PREDICTS GETTING PAID?")
    print("Using AVERAGED performance (Years 1-3)")
    print("="*80)
    
    # Load prepared payment data
    if not os.path.exists('qb_seasons_payment_labeled.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled.csv not found")
        return None
    
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        print("Run create_era_adjusted_payment_data() first")
        return None

    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with payment labels")
    
    # CRITICAL: Remove QBs who haven't reached contract window yet
    if exclude_recent_drafts:
        print("\n" + "="*80)
        print("EXCLUDING RECENT DRAFT CLASSES")
        print("="*80)
        
        current_year = 2025
        cutoff_year = 2020  # QBs drafted 2020 or earlier have had chance to get paid
        
        payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
        
        before_filter = len(payment_df['player_id'].unique())
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        after_filter = len(payment_df['player_id'].unique())
        
        print(f"Excluding QBs drafted after {cutoff_year}")
        print(f"  Before: {before_filter} unique QBs")
        print(f"  After: {after_filter} unique QBs")
        print(f"  Removed: {before_filter - after_filter} QBs (2021-2024 classes)")
    
    # Load season records for team metrics
    season_records = load_csv_safe("season_records.csv", "season records")
    if season_records is None:
        return None
    
    # Prepare season records
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Merge payment data with team metrics
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
    payment_df = payment_df.dropna(subset=['season'])
    payment_df['season'] = payment_df['season'].astype(int)
    
    merged_df = pd.merge(
        payment_df,
        season_records[['Season', 'Team', 'W-L%', 'Pts']],
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"✓ Merged with team metrics: {len(merged_df)} seasons")
    
    # ADD THIS NEW SECTION HERE:
    print("\n" + "="*80)
    print("ADJUSTING TEAM METRICS")
    print("="*80)

    # Load adjustment factors
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        
        # Apply Pts adjustment
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors)
        print(f"✓ Created Pts_adj: {merged_df['Pts_adj'].notna().sum()} values")
    else:
        print("⚠️ No adjustment factors found, using raw Pts")
        merged_df['Pts_adj'] = merged_df['Pts']

    # Define reduced metric set
    qb_metrics = [
        'total_yards_adj',
        'Pass_TD_adj',
        'Pass_ANY/A_adj',
        'Rush_Rushing_Succ%_adj'
    ]
    
    team_metrics = [
        'W-L%',      # Changed from 'W'
        'Pts_adj'
    ]
    
    all_metrics = qb_metrics + team_metrics
    
    # Check which QB metrics already have lags
    print("\n" + "="*80)
    print("CHECKING EXISTING LAGS")
    print("="*80)
    
    missing_qb_metrics = []
    
    for metric in qb_metrics:
        has_lags = all(f"{metric}_lag{lag}" in merged_df.columns for lag in [1, 2, 3])
        if has_lags:
            print(f"✓ {metric}: has lag features")
        else:
            print(f"✗ {metric}: missing lag features, will create")
            missing_qb_metrics.append(metric)
    
    # Create missing QB metric lags
    if missing_qb_metrics:
        print("\n" + "="*80)
        print("CREATING MISSING QB METRIC LAGS")
        print("="*80)
        
        merged_df = merged_df.sort_values(['player_id', 'season'])
        
        for metric in missing_qb_metrics:
            merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
            for lag in [1, 2, 3]:
                lag_col = f"{metric}_lag{lag}"
                merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
                non_null = merged_df[lag_col].notna().sum()
                print(f"  {lag_col}: {non_null} non-null values")
    
    # Create team metric lags
    print("\n" + "="*80)
    print("CREATING TEAM METRIC LAGS")
    print("="*80)
    
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in team_metrics:
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            lag_col = f"{metric}_lag{lag}"
            merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
            non_null = merged_df[lag_col].notna().sum()
            print(f"  {lag_col}: {non_null} non-null values")
    
    # Create AVERAGED features across lags
    print("\n" + "="*80)
    print("CREATING AVERAGED FEATURES (lag1, lag2, lag3)")
    print("="*80)
    
    features = []
    
    for metric in all_metrics:
        avg_col = f"{metric}_avg"
        merged_df[avg_col] = merged_df[[f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]].mean(axis=1)
        features.append(avg_col)
        non_null = merged_df[avg_col].notna().sum()
        print(f"  {avg_col}: {non_null} non-null values")
    
    print(f"\n✓ Total features: {len(features)} (reduced from {len(all_metrics) * 3})")
    
    # Prepare features and target
    X = merged_df[features + ['got_paid', 'player_id']].copy()
    
    # Drop rows with missing values
    X = X.dropna(subset=features)
    print(f"\nComplete cases: {len(X)}")
    
    # Check payment distribution among eligible QBs
    unique_qbs = X.groupby('player_id')['got_paid'].first()
    paid_qbs = unique_qbs.sum()
    total_qbs = len(unique_qbs)
    
    print(f"\n" + "="*80)
    print("ELIGIBLE QB PAYMENT DISTRIBUTION")
    print("="*80)
    print(f"Total eligible QBs (drafted ≤{cutoff_year}): {total_qbs}")
    print(f"  Got paid: {paid_qbs} ({paid_qbs/total_qbs*100:.1f}%)")
    print(f"  Not paid: {total_qbs - paid_qbs} ({(total_qbs - paid_qbs)/total_qbs*100:.1f}%)")
    
    if len(X) < 30:
        print("\n✗ ERROR: Insufficient complete cases for modeling")
        return None
    
    y = X['got_paid'].astype(int)
    X = X[features]
    
    # Payment distribution in dataset
    paid_count = y.sum()
    paid_pct = paid_count / len(y) * 100
    print(f"\nPayment outcome in dataset (season-level):")
    print(f"  Got paid: {paid_count} ({paid_pct:.1f}%)")
    print(f"  Not paid: {len(y) - paid_count} ({100-paid_pct:.1f}%)")
    
    # Sample size check
    observations_per_feature = len(X) / len(features)
    print(f"\nSample size ratio:")
    print(f"  Observations per feature: {observations_per_feature:.1f}")
    if observations_per_feature < 10:
        print(f"  ⚠️  Warning: Low ratio (recommend 10-20+)")
    else:
        print(f"  ✓ Good ratio for modeling")
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} paid)")
    print(f"  Test: {len(X_test)} samples ({y_test.sum()} paid)")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge regression with CV
    print("\n" + "="*80)
    print("RIDGE REGRESSION WITH 5-FOLD CV")
    print("="*80)
    
    if alpha_range is None:
        alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring='r2')
    ridge_cv.fit(X_train_scaled, y_train)
    
    best_alpha = ridge_cv.alpha_
    print(f"\n✓ Best alpha: {best_alpha}")
    
    # Train final model
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train_scaled, y_train)
    
    # CV scores
    cv_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\nCross-validation R² scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Training performance
    y_train_pred = ridge_model.predict(X_train_scaled)
    train_r2 = ridge_model.score(X_train_scaled, y_train)
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
    
    print(f"\n" + "="*80)
    print("TRAINING PERFORMANCE")
    print("="*80)
    print(f"  R² = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.4f}")
    
    # Test performance
    y_test_pred = ridge_model.predict(X_test_scaled)
    test_r2 = ridge_model.score(X_test_scaled, y_test)
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
    
    print(f"\n" + "="*80)
    print("TEST PERFORMANCE")
    print("="*80)
    print(f"  R² = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.4f}")
    
    # Variable importance
    print(f"\n" + "="*80)
    print("VARIABLE IMPORTANCE")
    print("="*80)
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': ridge_model.coef_,
        'Abs_Coefficient': np.abs(ridge_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nAll features ranked by importance:")
    display(importance_df)
    
    # Identify most important metric overall
    print(f"\n" + "="*80)
    print("MOST IMPORTANT PREDICTOR")
    print("="*80)
    
    top_feature = importance_df.iloc[0]['Feature']
    top_coef = importance_df.iloc[0]['Coefficient']
    
    # Extract base metric name (remove _avg)
    base_metric = top_feature.replace('_avg', '')
    
    print(f"\n🏆 WINNER: {base_metric}")
    print(f"   Feature: {top_feature}")
    print(f"   Coefficient: {top_coef:.4f}")
    
    # Separate QB vs Team
    print(f"\n" + "="*80)
    print("QB METRICS vs TEAM METRICS")
    print("="*80)
    
    qb_features = importance_df[importance_df['Feature'].str.contains('|'.join([m.replace('%', '') for m in qb_metrics]))]
    team_features = importance_df[importance_df['Feature'].str.contains('|'.join(team_metrics))]
    
    print(f"\nQB metric features:")
    display(qb_features)
    
    print(f"\nTeam metric features:")
    display(team_features)
    
    # Save results
    results = {
        'model': ridge_model,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'features': features,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'variable_importance': importance_df,
        'top_predictor': base_metric,
        'top_feature': top_feature,
        'eligible_qbs': total_qbs,
        'paid_qbs': paid_qbs
    }
    
    importance_df.to_csv('payment_prediction_importance.csv', index=False)
    print(f"\n✓ Results saved to: payment_prediction_importance.csv")
    
    return results

def year_weighting_regression(metric='total_yards_adj', max_decision_year=6):
    """
    Determines how each prior year is weighted in payment decisions.
    
    For each decision year (3-6), runs regression:
        got_paid ~ metric_year0 + metric_year1 + metric_year2 + ... + metric_year(N-1)
    
    Shows which years matter most for the payment decision.
    
    Args:
        metric (str): Metric to analyze ('total_yards' or 'W')
        max_decision_year (int): Latest decision year to model (default 6)
    
    Returns:
        dict: Weighting results for each decision year
    """
    print("\n" + "="*80)
    print(f"YEAR-BY-YEAR WEIGHTING ANALYSIS: {metric}")
    print("="*80)
    
    # Load prepared payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        print("Run create_era_adjusted_payment_data() first")
        return None

    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with era-adjusted payment labels")
    
    # If analyzing team metric (W), need to merge with season records
    if metric == 'W-L%':
        season_records = load_csv_safe("season_records.csv", "season records")
        if season_records is None:
            return None
        
        season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
        season_records = season_records.dropna(subset=['Season'])
        season_records['Season'] = season_records['Season'].astype(int)
        
        payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
        payment_df = payment_df.dropna(subset=['season'])
        payment_df['season'] = payment_df['season'].astype(int)
        
        payment_df = pd.merge(
            payment_df,
            season_records[['Season', 'Team', 'W-L%']],
            left_on=['season', 'Team'],
            right_on=['Season', 'Team'],
            how='inner'
        )
    
    # Filter to eligible QBs (drafted ≤2020)
    cutoff_year = 2020
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
    
    # Calculate years since draft
    payment_df['years_since_draft'] = payment_df['season'] - payment_df['draft_year']
    
    print(f"Analyzing metric: {metric}")
    print(f"QBs in dataset: {payment_df['player_id'].nunique()}")
    
    # Results storage
    all_results = {}
    
    # For each decision year (3-6)
    for decision_year in range(3, max_decision_year + 1):
        print("\n" + "="*80)
        print(f"DECISION YEAR {decision_year} (payment in Year {decision_year})")
        print(f"Using performance from Years 0 to {decision_year - 1}")
        print("="*80)
        
        # Get performance for each year 0 to (decision_year - 1)
        year_features = []
        
        # Pivot data to get one row per player with columns for each year
        player_data = []
        
        for player_id, group in payment_df.groupby('player_id'):
            group = group.sort_values('years_since_draft')
            
            record = {
                'player_id': player_id,
                'got_paid': group['got_paid'].iloc[0]
            }
            
            # Get metric value for each year 0 to (decision_year - 1)
            for year in range(decision_year):
                year_data = group[group['years_since_draft'] == year]
                if len(year_data) > 0:
                    record[f'{metric}_year{year}'] = pd.to_numeric(year_data[metric].iloc[0], errors='coerce')
                else:
                    record[f'{metric}_year{year}'] = np.nan
                
                year_features.append(f'{metric}_year{year}')
        
            player_data.append(record)
        
        year_features = sorted(list(set(year_features)))  # Remove duplicates
        df = pd.DataFrame(player_data)
        
        print(f"Players with data: {len(df)}")
        
        # Drop rows with missing values
        df = df.dropna(subset=year_features)
        
        print(f"Complete cases: {len(df)}")
        
        if len(df) < 20:
            print(f"⚠️  Insufficient data for Year {decision_year} decision")
            continue
        
        # Payment distribution
        paid = df['got_paid'].sum()
        print(f"Payment distribution: {paid} paid, {len(df) - paid} not paid")
        
        # Prepare X and y
        X = df[year_features]
        y = df['got_paid'].astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ridge regression
        ridge_model = Ridge(alpha=10.0)  # Use moderate regularization
        ridge_model.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'Year': [f'Year {i}' for i in range(decision_year)],
            'Feature': year_features,
            'Coefficient': ridge_model.coef_,
            'Abs_Coefficient': np.abs(ridge_model.coef_)
        }).sort_values('Year')
        
        # Normalize coefficients to sum to 1 (show as weights)
        total_abs = coefficients['Abs_Coefficient'].sum()
        coefficients['Weight_%'] = (coefficients['Abs_Coefficient'] / total_abs * 100)
        
        print(f"\nYear-by-year coefficients:")
        display(coefficients[['Year', 'Coefficient', 'Weight_%']])
        
        # Model performance
        r2 = ridge_model.score(X_scaled, y)
        print(f"\nModel R²: {r2:.4f}")
        
        # Store results
        all_results[f'year_{decision_year}'] = {
            'decision_year': decision_year,
            'coefficients': coefficients,
            'model': ridge_model,
            'scaler': scaler,
            'features': year_features,
            'r2': r2,
            'n_samples': len(df)
        }
    
    # Summary across all decision years
    print("\n" + "="*80)
    print(f"SUMMARY: YEAR WEIGHTS ACROSS DECISION YEARS ({metric})")
    print("="*80)
    
    summary_data = []
    for decision_year in range(3, max_decision_year + 1):
        key = f'year_{decision_year}'
        if key in all_results:
            result = all_results[key]
            coefs = result['coefficients']
            
            for _, row in coefs.iterrows():
                summary_data.append({
                    'Decision_Year': decision_year,
                    'Performance_Year': row['Year'],
                    'Weight_%': row['Weight_%'],
                    'Coefficient': row['Coefficient']
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) > 0:
        # Pivot to show weights matrix
        pivot_weights = summary_df.pivot(index='Performance_Year', 
                                         columns='Decision_Year', 
                                         values='Weight_%')
        
        print("\nWeight Matrix (% importance of each year):")
        print("Rows = Performance year, Columns = Decision year")
        display(pivot_weights)
        
        # Average weight across all decision years for each performance year
        print("\n" + "="*80)
        print("AVERAGE WEIGHTS ACROSS ALL DECISION YEARS")
        print("="*80)
        
        avg_weights = summary_df.groupby('Performance_Year')['Weight_%'].mean().sort_index()
        print("\nAverage importance of each performance year:")
        for year, weight in avg_weights.items():
            print(f"  {year}: {weight:.1f}%")
        
        # Save results
        summary_df.to_csv(f'year_weights_{metric}.csv', index=False)
        pivot_weights.to_csv(f'year_weights_matrix_{metric}.csv')
        print(f"\n✓ Results saved to year_weights_{metric}.csv and year_weights_matrix_{metric}.csv")
    
    return all_results

def calculate_era_adjustment_factors(reference_year=2024):
    """
    Calculates linear inflation adjustment factors for offensive stats.
    
    Fits linear trend: league_avg_stat = β0 + β1 * year
    Then creates adjustment factor: factor = reference_year_avg / year_avg
    
    Args:
        reference_year (int): Year to adjust all stats to (default 2024)
    
    Returns:
        dict: Adjustment factors by year and stat
    """
    print("\n" + "="*80)
    print(f"CALCULATING ERA ADJUSTMENT FACTORS (baseline: {reference_year})")
    print("="*80)
    
    # Load league averages by year
    season_avg = load_csv_safe("season_averages.csv", "season averages")
    if season_avg is None:
        return None
    
    # Convert Year to numeric
    season_avg['Year'] = pd.to_numeric(season_avg['Year'], errors='coerce')
    season_avg = season_avg.dropna(subset=['Year'])
    season_avg = season_avg.sort_values('Year')
    
    print(f"League averages available: {int(season_avg['Year'].min())}-{int(season_avg['Year'].max())}")
    
    # Stats to adjust (map internal names to season_averages.csv column names)
    stats_mapping = {
        'total_yards': 'Total_Yards',
        'Pass_TD': 'Pass_TD',
        'Pass_Yds': 'Pass_Yds',
        'Pass_ANY/A': 'NY/A',  # NEW - use Net Yards per Attempt as proxy
        'Rush_Rushing_Succ%': None,  # Can't calculate from season_averages, handle separately
        'W-L%': None,  # NEW - Calculate from season_records
        'Pts': 'PF'
    }
        
    adjustment_factors = {}
    
    for stat_name, csv_col in stats_mapping.items():
        if csv_col is None:
            # Wins don't need adjustment
            print(f"\n{stat_name}: No adjustment needed (fixed scale)")
            adjustment_factors[stat_name] = {year: 1.0 for year in season_avg['Year']}
            continue
        
        if csv_col not in season_avg.columns:
            print(f"\n⚠️  Warning: {csv_col} not found in season_averages.csv, skipping {stat_name}")
            continue
        
        print(f"\n{stat_name} (using {csv_col}):")
        
        # Get year and stat values
        year_data = season_avg[['Year', csv_col]].copy()
        year_data[csv_col] = pd.to_numeric(year_data[csv_col], errors='coerce')
        year_data = year_data.dropna()
        
        years = year_data['Year'].values
        values = year_data[csv_col].values
        
        # Fit linear regression: stat = β0 + β1 * year
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(years.reshape(-1, 1), values)
        
        # Get predicted values
        predicted = model.predict(years.reshape(-1, 1))
        
        # Calculate reference year average
        reference_avg = model.predict([[reference_year]])[0]
        
        # Calculate adjustment factors for each year
        year_factors = {}
        for year, pred_val in zip(years, predicted):
            # factor = reference_year_avg / year_avg
            # Multiply old stats by this to get 2024-equivalent
            factor = reference_avg / pred_val
            year_factors[int(year)] = factor
        
        adjustment_factors[stat_name] = year_factors
        
        # Show trend
        print(f"  Linear trend: {csv_col} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * year")
        print(f"  R² = {model.score(years.reshape(-1, 1), values):.4f}")
        print(f"  {reference_year} predicted average: {reference_avg:.2f}")
        
        # Show sample adjustments
        print(f"\n  Sample adjustment factors (multiply stat by factor):")
        sample_years = [2005, 2010, 2015, 2020, 2024]
        for year in sample_years:
            if year in year_factors:
                pred_val = model.predict([[year]])[0]
                print(f"    {year}: {year_factors[year]:.4f} (avg was {pred_val:.1f}, now {pred_val * year_factors[year]:.1f})")
    
    print("\nCalculating adjustment for Rush_Rushing_Succ% from QB data...")
    if os.path.exists('qb_seasons_payment_labeled.csv'):
        qb_df = pd.read_csv('qb_seasons_payment_labeled.csv')
        qb_df['season'] = pd.to_numeric(qb_df['season'], errors='coerce')
        qb_df['Rush_Rushing_Succ%'] = pd.to_numeric(qb_df['Rush_Rushing_Succ%'], errors='coerce')
        
        # Calculate yearly averages
        yearly_avg = qb_df.groupby('season')['Rush_Rushing_Succ%'].mean().reset_index()
        yearly_avg = yearly_avg.dropna()
        
        if len(yearly_avg) > 10:  # Need enough data
            years = yearly_avg['season'].values
            values = yearly_avg['Rush_Rushing_Succ%'].values
            
            # Fit linear trend
            model = LinearRegression()
            model.fit(years.reshape(-1, 1), values)
            
            reference_avg = model.predict([[reference_year]])[0]
            predicted = model.predict(years.reshape(-1, 1))
            
            year_factors = {}
            for year, pred_val in zip(years, predicted):
                year_factors[int(year)] = reference_avg / pred_val
            
            adjustment_factors['Rush_Rushing_Succ%'] = year_factors
            
            print(f"  Rush_Rushing_Succ%: R² = {model.score(years.reshape(-1, 1), values):.4f}")
            print(f"  {reference_year} predicted average: {reference_avg:.2f}%")

    print("\nCalculating adjustment for W-L% from season records...")
    if os.path.exists('season_records.csv'):
        season_rec = pd.read_csv('season_records.csv')
        season_rec['Season'] = pd.to_numeric(season_rec['Season'], errors='coerce')
        season_rec['W-L%'] = pd.to_numeric(season_rec['W-L%'], errors='coerce')
        
        # Calculate yearly averages
        yearly_avg = season_rec.groupby('Season')['W-L%'].mean().reset_index()
        yearly_avg = yearly_avg.dropna()
        
        if len(yearly_avg) > 10:
            years = yearly_avg['Season'].values
            values = yearly_avg['W-L%'].values
            
            model = LinearRegression()
            model.fit(years.reshape(-1, 1), values)
            
            reference_avg = model.predict([[reference_year]])[0]
            predicted = model.predict(years.reshape(-1, 1))
            
            year_factors = {}
            for year, pred_val in zip(years, predicted):
                year_factors[int(year)] = reference_avg / pred_val
            
            adjustment_factors['W-L%'] = year_factors
            
            print(f"  W-L%: R² = {model.score(years.reshape(-1, 1), values):.4f}")
            print(f"  {reference_year} predicted average: {reference_avg:.4f}")

    # Save adjustment factors
    print("\n" + "="*80)
    print("SAVING ADJUSTMENT FACTORS")
    print("="*80)
    
    # Convert to DataFrame for easy saving
    factor_rows = []
    for stat_name, year_dict in adjustment_factors.items():
        for year, factor in year_dict.items():
            factor_rows.append({
                'stat': stat_name,
                'year': year,
                'adjustment_factor': factor
            })
    
    factor_df = pd.DataFrame(factor_rows)
    factor_df.to_csv('era_adjustment_factors.csv', index=False)
    print("✓ Saved to: era_adjustment_factors.csv")
    
    return adjustment_factors

def apply_era_adjustments(df, adjustment_factors, stats_to_adjust=['total_yards', 'Pass_TD', 'Pass_ANY/A', 'Rush_Rushing_Succ%']):
    """
    Applies era adjustment factors to a dataframe with QB/team stats.
    
    Creates new columns with '_adj' suffix containing adjusted values.
    
    Args:
        df (DataFrame): Data with 'season' column and stats to adjust
        adjustment_factors (dict): Output from calculate_era_adjustment_factors()
        stats_to_adjust (list): Which stats to adjust
    
    Returns:
        DataFrame: Original df with new adjusted columns
    """
    print("\n" + "="*80)
    print("APPLYING ERA ADJUSTMENTS")
    print("="*80)
    
    df = df.copy()
    
    # Ensure season is int
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(subset=['season'])
    df['season'] = df['season'].astype(int)
    
    for stat in stats_to_adjust:
        if stat not in adjustment_factors:
            print(f"⚠️  No adjustment factors for {stat}, skipping")
            continue
        
        if stat not in df.columns:
            print(f"⚠️  {stat} not in dataframe, skipping")
            continue
        
        # Create adjusted column
        adj_col = f"{stat}_adj"
        
        # Map year to factor
        df['_temp_factor'] = df['season'].map(adjustment_factors[stat])
        
        # Apply adjustment: adjusted = original * factor
        df[stat] = pd.to_numeric(df[stat], errors='coerce')
        df[adj_col] = df[stat] * df['_temp_factor']
        
        # Clean up
        df = df.drop(columns=['_temp_factor'])
        
        # Show results
        non_null = df[adj_col].notna().sum()
        print(f"✓ {stat} → {adj_col}: {non_null} values adjusted")
        
        # Show example
        sample = df[df[stat].notna()].head(3)
        if len(sample) > 0:
            print(f"  Example: {sample.iloc[0]['season']:.0f} - "
                  f"Original: {sample.iloc[0][stat]:.1f}, "
                  f"Adjusted: {sample.iloc[0][adj_col]:.1f}")
    
    return df

def create_era_adjusted_payment_data(force_refresh=False):
    """
    Creates era-adjusted version of payment data.
    
    Args:
        force_refresh (bool): Recalculate adjustment factors
    
    Returns:
        DataFrame: Payment data with era-adjusted stats
    """
    print("\n" + "="*80)
    print("CREATING ERA-ADJUSTED PAYMENT DATA")
    print("="*80)
    
    # Calculate or load adjustment factors
    if force_refresh or not os.path.exists('era_adjustment_factors.csv'):
        adjustment_factors = calculate_era_adjustment_factors(reference_year=2024)
    else:
        print("Loading existing adjustment factors from era_adjustment_factors.csv")
        factor_df = load_csv_safe('era_adjustment_factors.csv')
        
        # Convert back to dict format
        adjustment_factors = {}
        for stat in factor_df['stat'].unique():
            stat_data = factor_df[factor_df['stat'] == stat]
            adjustment_factors[stat] = dict(zip(stat_data['year'], stat_data['adjustment_factor']))
        
        print(f"✓ Loaded adjustment factors for {len(adjustment_factors)} stats")
    
    # Load payment data
    if not os.path.exists('qb_seasons_payment_labeled.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled.csv not found")
        print("Run prepare_qb_payment_data() first")
        return None

    # Try to load era-adjusted version first, fall back to regular
    if os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
        print(f"✓ Using ERA-ADJUSTED data")
    else:
        payment_df = load_csv_safe('qb_seasons_payment_labeled.csv')
        print(f"⚠️  Using non-adjusted data (run create_era_adjusted_payment_data() first)")

    print(f"✓ Loaded {len(payment_df)} seasons")
    
    # Apply adjustments
    stats_to_adjust = [
        'total_yards', 
        'Pass_TD', 
        'Pass_ANY/A',
        'Rush_Rushing_Succ%'
    ]
    adjusted_df = apply_era_adjustments(payment_df, adjustment_factors, stats_to_adjust)
    
    # Save
    output_file = 'qb_seasons_payment_labeled_era_adjusted.csv'
    adjusted_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved era-adjusted data to: {output_file}")
    
    return adjusted_df

def create_payment_probability_surface(
    metric='total_yards_adj', 
    years_range=(0, 6),
    value_bins=50,
    min_qbs_per_cell=3
):
    """
    Creates a 2D grid of payment probabilities based on actual performance values.
    
    For each combination of (years_since_draft, metric_value), calculates:
    P(getting paid) = count(got_paid) / count(total QBs in that cell)
    
    Args:
        metric (str): Performance metric to analyze
        years_range (tuple): Min and max years since draft to include
        value_bins (int): Number of bins to divide metric values into
        min_qbs_per_cell (int): Minimum QBs required in cell to calculate probability
    
    Returns:
        DataFrame: Grid with years_since_draft, metric_bin_min, metric_bin_max, 
                   pct_got_paid, n_qbs, n_paid
    """
    print("\n" + "="*80)
    print(f"CREATING PAYMENT PROBABILITY SURFACE: {metric}")
    print("="*80)
    
    # Load era-adjusted payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: Run create_era_adjusted_payment_data() first")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Filter to eligible QBs (drafted ≤2020)
    cutoff_year = 2020
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
    
    # Calculate years since draft
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
    payment_df['years_since_draft'] = payment_df['season'] - payment_df['draft_year']
    
    # Filter to years range
    min_year, max_year = years_range
    payment_df = payment_df[
        (payment_df['years_since_draft'] >= min_year) & 
        (payment_df['years_since_draft'] <= max_year)
    ]
    
    # Convert metric to numeric
    payment_df[metric] = pd.to_numeric(payment_df[metric], errors='coerce')
    payment_df = payment_df.dropna(subset=[metric, 'years_since_draft', 'got_paid'])
    
    print(f"Total QB seasons: {len(payment_df)}")
    print(f"Unique QBs: {payment_df['player_id'].nunique()}")
    print(f"Years range: {min_year}-{max_year}")
    print(f"Metric range: {payment_df[metric].min():.1f} to {payment_df[metric].max():.1f}")
    
    # Create metric value bins based on overall distribution
    payment_df['metric_bin'] = pd.cut(
        payment_df[metric], 
        bins=value_bins,
        include_lowest=True
    )
    
    # Group by (years_since_draft, metric_bin) and calculate payment rate
    grouped = payment_df.groupby(['years_since_draft', 'metric_bin'], observed=True).agg({
        'got_paid': ['sum', 'count']
    }).reset_index()
    
    grouped.columns = ['years_since_draft', 'metric_bin', 'n_paid', 'n_qbs']
    
    # Calculate payment percentage
    grouped['pct_got_paid'] = (grouped['n_paid'] / grouped['n_qbs'] * 100)
    
    # Filter out cells with too few QBs
    grouped = grouped[grouped['n_qbs'] >= min_qbs_per_cell]
    
    # Extract bin boundaries - do this BEFORE calculating mid
    min_vals = []
    max_vals = []
    for interval in grouped['metric_bin']:
        min_vals.append(float(interval.left))
        max_vals.append(float(interval.right))
    
    grouped['metric_bin_min'] = min_vals
    grouped['metric_bin_max'] = max_vals
    grouped['metric_bin_mid'] = (grouped['metric_bin_min'] + grouped['metric_bin_max']) / 2
    
    # Drop the interval column
    grouped = grouped.drop(columns=['metric_bin'])
    
    print(f"\nCreated {len(grouped)} grid cells")
    print(f"Cells per year:")
    for year in sorted(grouped['years_since_draft'].unique()):
        count = len(grouped[grouped['years_since_draft'] == year])
        print(f"  Year {int(year)}: {count} cells")
    
    # Show payment rate distribution
    print(f"\nPayment probability distribution:")
    print(f"  Min: {grouped['pct_got_paid'].min():.1f}%")
    print(f"  25th: {grouped['pct_got_paid'].quantile(0.25):.1f}%")
    print(f"  Median: {grouped['pct_got_paid'].median():.1f}%")
    print(f"  75th: {grouped['pct_got_paid'].quantile(0.75):.1f}%")
    print(f"  Max: {grouped['pct_got_paid'].max():.1f}%")
    
    # Save to CSV (sanitize metric name for filesystem)
    safe_metric_name = metric.replace('/', '_').replace('%', 'pct')
    output_file = f'payment_probability_surface_{safe_metric_name}.csv'
    grouped.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    return grouped

def create_simple_knn_payment_surface(
    metric='total_yards_adj',
    decision_year=4,
    k_values=[5, 10, 15, 20],
    year_weights=None
):
    """
    Simple KNN with proper year weighting.
    Year weights are universal (same for all metrics) since they represent
    importance of each year in payment decisions.
    """
    print("\n" + "="*80)
    print(f"SIMPLE KNN PAYMENT SURFACE: {metric}, Decision Year {decision_year}")
    print("="*80)
    
    # Load era-adjusted payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: Run create_era_adjusted_payment_data() first")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Filter to eligible QBs (drafted ≤2020)
    cutoff_year = 2020
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
    
    # Calculate years since draft
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce')
    payment_df['years_since_draft'] = payment_df['season'] - payment_df['draft_year']
    
    # CRITICAL: Only use years 0 through (decision_year - 1)
    max_year = decision_year - 1
    payment_df = payment_df[payment_df['years_since_draft'] <= max_year]
    
    print(f"Decision Year {decision_year}: Using performance from Years 0-{max_year}")
    print(f"Total QB seasons: {len(payment_df)}")
    print(f"Unique QBs: {payment_df['player_id'].nunique()}")
    
    # Load year weights if not provided
    # NOTE: Weights are universal, not metric-specific
    if year_weights is None:
        # Try to load from either metric's weights file (they should be the same)
        weights_file = 'year_weights_total_yards_adj.csv'  # Use yards as canonical source
        
        if not os.path.exists(weights_file):
            weights_file = 'year_weights_Pass_ANY_A_adj.csv'  # Fallback
        
        if os.path.exists(weights_file):
            weights_df = load_csv_safe(weights_file)
            weights_df = weights_df[weights_df['Decision_Year'] == decision_year]
            
            if len(weights_df) == 0:
                print(f"⚠️  No weights found for decision year {decision_year}, using uniform weights")
                year_weights = {i: 1.0 / decision_year for i in range(decision_year)}
            else:
                year_weights = {}
                for _, row in weights_df.iterrows():
                    year_str = row['Performance_Year']
                    year_num = int(year_str.split()[-1])
                    # Convert percentage to decimal
                    year_weights[year_num] = row['Weight_%'] / 100.0
                
                print(f"✓ Loaded universal year weights from {weights_file}")
        else:
            print(f"⚠️  No weights file found, using uniform weights")
            year_weights = {i: 1.0 / decision_year for i in range(decision_year)}
    
    # Verify weights cover all years and normalize
    print(f"\nYear importance weights for Decision Year {decision_year}:")
    total_weight = 0
    for year in range(decision_year):
        weight = year_weights.get(year, 1.0 / decision_year)
        year_weights[year] = weight  # Fill in any missing years
        total_weight += weight
    
    # Normalize to sum to 1.0
    if abs(total_weight - 1.0) > 0.01:
        print(f"  Normalizing weights (sum was {total_weight:.3f})")
        year_weights = {k: v/total_weight for k, v in year_weights.items()}
        total_weight = 1.0
    
    for year in range(decision_year):
        print(f"  Year {year}: {year_weights[year]:.3f} ({year_weights[year]*100:.1f}%)")
    print(f"  Total: {sum(year_weights.values()):.3f}")
    
    # Prepare data
    payment_df[metric] = pd.to_numeric(payment_df[metric], errors='coerce')
    payment_df = payment_df.dropna(subset=[metric, 'years_since_draft', 'got_paid'])
    
    print(f"\nValid observations: {len(payment_df)}")
    print(f"  Got paid: {payment_df['got_paid'].sum()}")
    print(f"  Not paid: {(~payment_df['got_paid']).sum()}")
    
    # Create feature matrix: [years_since_draft, metric_value]
    X = payment_df[['years_since_draft', metric]].values
    y = payment_df['got_paid'].values.astype(int)
    
    print(f"\nMetric range by year:")
    for year in range(decision_year):
        year_data = payment_df[payment_df['years_since_draft'] == year][metric]
        if len(year_data) > 0:
            print(f"  Year {year}: {year_data.min():.0f} to {year_data.max():.0f} ({len(year_data)} QBs)")
    
    # Standardize for distance calculation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply year weights to scale importance
    # For each observation, scale its year dimension by that year's importance weight
    X_scaled_weighted = X_scaled.copy()
    
    # Create mapping of observation index to its year weight
    observation_weights = []
    for i in range(len(X_scaled)):
        year = int(payment_df.iloc[i]['years_since_draft'])
        weight = year_weights.get(year, 1.0)
        # Scale the year dimension (first dimension) by sqrt(weight)
        # This makes differences in important years contribute more to distance
        scale_factor = np.sqrt(weight * decision_year)  # Multiply by decision_year to amplify effect
        X_scaled_weighted[i, 0] *= scale_factor
        observation_weights.append(scale_factor)
    
    print(f"\nYear dimension scaling factors (applied to distance calculation):")
    for year in range(decision_year):
        weight = year_weights[year]
        scale_factor = np.sqrt(weight * decision_year)
        print(f"  Year {year}: {scale_factor:.3f} (from weight {weight:.3f})")
    
    # Create fine prediction grid
    year_grid = np.linspace(0, max_year, 40)
    metric_grid = np.linspace(X[:, 1].min() * 0.9, X[:, 1].max() * 1.1, 100)
    
    print(f"\nCreating prediction grid:")
    print(f"  Years: {len(year_grid)} points from 0.0 to {max_year}")
    print(f"  Metric: {len(metric_grid)} points from {metric_grid.min():.0f} to {metric_grid.max():.0f}")
    print(f"  Total grid points: {len(year_grid) * len(metric_grid)}")
    
    # Create all combinations
    grid_points = []
    for year in year_grid:
        for value in metric_grid:
            grid_points.append([year, value])
    
    X_grid = np.array(grid_points)
    X_grid_scaled = scaler.transform(X_grid)
    
    # Apply same year weighting to grid points
    X_grid_scaled_weighted = X_grid_scaled.copy()
    for i in range(len(X_grid_scaled_weighted)):
        year_float = X_grid[i, 0]
        year_idx = int(round(year_float))
        year_idx = max(0, min(year_idx, decision_year - 1))
        weight = year_weights.get(year_idx, 1.0)
        scale_factor = np.sqrt(weight * decision_year)
        X_grid_scaled_weighted[i, 0] *= scale_factor
    
    # Run KNN for each K value
    results = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"Running KNN with K={k}")
        print(f"{'='*80}")
        
        predictions = []
        
        for i, grid_point_weighted in enumerate(X_grid_scaled_weighted):
            if i % 500 == 0:
                print(f"  Processing grid point {i}/{len(X_grid_scaled_weighted)}...", end='\r')
            
            # Calculate weighted Euclidean distance
            distances = np.sqrt(np.sum((X_scaled_weighted - grid_point_weighted)**2, axis=1))
            
            # Find K nearest neighbors
            k_nearest_idx = np.argsort(distances)[:k]
            k_nearest_labels = y[k_nearest_idx]
            
            # Payment probability = proportion that got paid
            prob = k_nearest_labels.mean()
            predictions.append(prob * 100)
        
        print(f"  Processing grid point {len(X_grid_scaled_weighted)}/{len(X_grid_scaled_weighted)}... Done!")
        
        predictions = np.array(predictions)
        
        # Create results dataframe
        results_df = pd.DataFrame(X_grid, columns=['year', metric])
        results_df['payment_probability'] = predictions
        results_df['decision_year'] = decision_year
        results_df['k_value'] = k
        results_df['metric_name'] = metric
        
        print(f"\n  Prediction range: {predictions.min():.1f}% to {predictions.max():.1f}%")
        print(f"  Mean probability: {predictions.mean():.1f}%")
        
        # DEBUG: Show prediction distribution by year
        print(f"\n  Prediction statistics by year:")
        for check_year in range(decision_year):
            year_mask = np.abs(X_grid[:, 0] - check_year) < 0.1
            if year_mask.sum() > 0:
                year_preds = predictions[year_mask]
                print(f"    Year {check_year}: min={year_preds.min():.1f}%, "
                      f"mean={year_preds.mean():.1f}%, max={year_preds.max():.1f}%")
        
        results[k] = {
            'predictions_df': results_df,
            'predictions': predictions,
            'scaler': scaler,
            'year_weights': year_weights
        }
        
        # Save to CSV
        safe_metric_name = metric.replace('/', '_').replace('%', 'pct')
        output_file = f'simple_knn_decision{decision_year}_{safe_metric_name}_k{k}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
    
    # Compare K values
    print(f"\n{'='*80}")
    print(f"COMPARISON ACROSS K VALUES")
    print(f"{'='*80}")
    
    comparison_df = pd.DataFrame([
        {
            'K': k,
            'Min_Prob_%': results[k]['predictions'].min(),
            'Mean_Prob_%': results[k]['predictions'].mean(),
            'Max_Prob_%': results[k]['predictions'].max(),
            'Std_Prob_%': results[k]['predictions'].std()
        }
        for k in k_values
    ])
    
    display(comparison_df)
    
    return results

def run_all_simple_knn_surfaces(
    metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
    decision_years=[3, 4, 5, 6],
    k_values=[5, 10, 15, 20]
):
    """
    Master function to generate all simple KNN surfaces.
    
    Args:
        metrics (list): Performance metrics to analyze
        decision_years (list): Which decision years to model
        k_values (list): K values to test
    
    Returns:
        dict: All results organized by metric and decision year
    """
    print("\n" + "="*80)
    print("GENERATING ALL SIMPLE KNN PAYMENT SURFACES")
    print("="*80)
    
    all_results = {}
    
    for metric in metrics:
        print(f"\n\n{'#'*80}")
        print(f"METRIC: {metric}")
        print(f"{'#'*80}")
        
        all_results[metric] = {}
        
        for decision_year in decision_years:
            results = create_simple_knn_payment_surface(
                metric=metric,
                decision_year=decision_year,
                k_values=k_values
            )
            
            if results is not None:
                all_results[metric][decision_year] = results
    
    # Create summary report
    print("\n\n" + "="*80)
    print("GENERATION COMPLETE - FILES CREATED")
    print("="*80)
    
    for metric in metrics:
        safe_metric_name = metric.replace('/', '_').replace('%', 'pct')
        print(f"\n{metric}:")
        for decision_year in decision_years:
            for k in k_values:
                filename = f'simple_knn_decision{decision_year}_{safe_metric_name}_k{k}.csv'
                print(f"  {filename}")
    
    return all_results

def export_individual_qb_trajectories(
    metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
    qb_list=None,
    years_range=(0, 6),
    include_recent_drafts=True,
    recent_draft_cutoff=2021
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
    trajectories_df.to_csv(output_file, index=False)
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
    summary_df.to_csv(output_file, index=False)
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
        metrics=['total_yards_adj'],
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

def export_cohort_summary_stats(
    metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
    years_range=(0, 6),
    include_recent_drafts=True  # ADD THIS PARAMETER
):
    """
    Exports summary statistics for the entire first-round QB cohort by year.
    
    UPDATED: Now includes recent draft picks for complete visualization.
    
    Args:
        metrics (list): Performance metrics to summarize
        years_range (tuple): Min and max years since draft to include
        include_recent_drafts (bool): Whether to include recent drafts (NEW PARAMETER)
    
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
    
    # UPDATED: Handle draft year filtering based on new parameter
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
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    return summary_df


if __name__ == "__main__":
    # Uncomment these if you need to regenerate the data
    # PFR.get_player_ids()
    # PFR.update_qb_ids()
    # PFR.pull_updated_QB_data()
    # Initialize the client with your API key
    #client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    #response = client.messages.create(
    #model="claude-sonnet-4-20250514",  # Specify the model
    #max_tokens=1000,  # Maximum tokens in the response
    #temperature=0.7,  # Controls randomness (0-1)
    #system="You are a helpful assistant",  # Optional system prompt
    #messages=[
    #   {
    #        "role": "user",
    #        "content": "who is the best quarterback in the NFL based on sustained peak combined total yards per season?"
    #    }
    #]
    #)

    # Access the response
    #print(response.content[0].text)
    
    #uncomment these to do the regression analysis again
    #train_df, test_df = create_train_test_split(test_size=0.2, split_by='temporal')
    #regression_with_pc1_factors(train_df, test_df)

    #train_df, test_df = load_train_test_split()
    
    # Test QB name normalization
    # test_name_mapping()
    
    # Create mapping (first time - will process and cache)
    #mapped_contracts = create_contract_player_mapping()
    #mapped_contracts = create_contract_player_mapping(force_refresh=True)   

    # Debug specific player
    #contracts = load_csv_safe("QB_contract_data.csv")
    #player_ids = load_csv_safe("player_ids.csv")
    #debug_name_matching("Ben Roethlisberger", contracts, player_ids)
    #mapped_contracts = create_contract_player_mapping(force_refresh=True)
    #now check the number of QBs in contract_player_id_mapping.csv after year 2000, without a match
        
    # Test the pipeline step by step
    
    # Step 1: Load contracts
    contracts = load_csv_safe('cache/contract_player_id_mapping.csv')
    print(f"✓ Loaded {len(contracts)} contracts from cache")
    print(f"Loaded contracts: {len(contracts)}")
    print(f"Columns: {contracts.columns.tolist()}")
    print(f"\nContracts with player_id: {contracts['player_id'].notna().sum()}")
    
    # Step 2: Test payment mapping
    payment_mapping = create_payment_year_mapping(contracts)
    print(f"\nPayment mapping created: {len(payment_mapping)} players")
    
    print("Starting payment data preparation pipeline...")
    prepared_df = prepare_qb_payment_data()
    
    # Run the complete export with recent drafts
    generate_complete_tableau_exports()
    
    # Verify recent QBs are included
    check_recent_qb_inclusion()

    if prepared_df is not None:
        '''report = validate_payment_data(prepared_df)
        #plot_sample_trajectories(prepared_df)
        
        # Create era-adjusted version
        print("\n\n" + "="*80)
        print("ERA ADJUSTMENT")
        print("="*80)
        adjusted_df = create_era_adjusted_payment_data(force_refresh=True)

        if adjusted_df is None:
            print("✗ ERROR: Failed to create era-adjusted data")
            exit(1)'''

        '''# REGRESSION 1: What predicts getting paid?
        print("\n\n" + "="*80)
        print("REGRESSION 1: IDENTIFYING MOST IMPORTANT PREDICTOR (ERA-ADJUSTED)")
        print("="*80)
        
        payment_results = ridge_regression_payment_prediction()
        
        if payment_results:
            print(f"\n\n🏆 TOP PREDICTOR: {payment_results['top_predictor']}")
            
            # REGRESSION 2: Year-by-year weighting
            print("\n\n" + "="*80)
            print("REGRESSION 2: YEAR-BY-YEAR WEIGHTING ANALYSIS")
            print("="*80)
            
            yards_weights = year_weighting_regression(metric='total_yards_adj', max_decision_year=6)
            wins_weights = year_weighting_regression(metric='W-L%', max_decision_year=6)
            '''
        '''results = run_trajectory_visualization_pipeline(
            metric='total_yards_adj',
            n_sample_qbs=3,
            force_refresh=False  # Change to True to regenerate all files
        )'''
        '''results = run_trajectory_visualization_pipeline(
            metric='Pass_ANY/A_adj',
            n_sample_qbs=3,
            force_refresh=False
        )'''

    '''print("\n" + "="*80)
    print("GENERATING TABLEAU EXPORT FILES")
    print("="*80)
    
    # Define metrics to analyze
    primary_metrics = ['total_yards_adj', 'Pass_ANY/A_adj']
    years_to_include = (0, 6)  # Y0 through Y6
    
    # 1. Create payment probability surface (KNN)
    print("\n\n[1/3] Creating payment probability surfaces...")
    all_knn_results = run_all_simple_knn_surfaces(
        metrics=['total_yards_adj', 'Pass_ANY/A_adj'],
        decision_years=[3, 4, 5, 6],
        k_values=[5, 10, 15, 20]
    )'''    
    '''# 2. Export individual QB trajectories (overlay lines)
    print("\n\n[2/3] Exporting QB trajectories...")
    trajectories = export_individual_qb_trajectories(
        metrics=primary_metrics,
        qb_list=None,  # None = all QBs, or pass list of player_ids
        years_range=years_to_include
    )'''
    
    '''# 3. Export cohort summary stats (reference lines/ranges)
    print("\n\n[3/3] Exporting cohort summary statistics...")
    summary = export_cohort_summary_stats(
        metrics=primary_metrics,
        years_range=years_to_include
    )'''