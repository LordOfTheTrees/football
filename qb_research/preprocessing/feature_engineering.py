"""
Feature engineering and data preprocessing functions for QB payment analysis.

This module contains functions for:
- Labeling seasons relative to payment
- Creating lookback performance features
- Creating decision point datasets
- Preparing complete payment datasets
- Validating payment data
"""

import pandas as pd
import numpy as np
import os

from qb_research.utils.data_loading import (
    load_csv_safe,
    validate_columns,
    validate_payment_years
)
from qb_research.data.mappers import (
    create_contract_player_mapping,
    create_payment_year_mapping,
    create_pick_number_mapping
)


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

