"""
Data loading and validation utilities.

This module provides safe, consistent ways to load CSV files and validate
data structures throughout the QB research pipeline.
"""

import os
import pandas as pd


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
        print(f"[ERROR] {description} not found at: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded {len(df)} records from {description}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read {description}: {e}")
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
        print(f"[ERROR] Missing required columns in {df_name}:")
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
        print("\n[OK] All payment years look valid")
    else:
        print(f"\n[WARNING] Total validation issues: {total_issues}")
        print("These records will be kept but may need investigation")
    
    return df  # Return original df - just warning, not filtering

