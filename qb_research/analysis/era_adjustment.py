"""
Era adjustment functions for QB research.

This module contains functions for:
- Calculating era adjustment factors
- Applying era adjustments to dataframes
- Creating era-adjusted payment datasets
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

from qb_research.utils.data_loading import load_csv_safe


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
        'Pass_ANY/A': 'NY/A',  # Use Net Yards per Attempt as proxy
        'Pass_Rate': None,  # Calculate from QB data (passer rating)
        'Pass_Cmp%': None,  # Calculate from QB data (completion percentage)
        'Pass_QBR': None,  # Calculate from QB data (QBR)
        'Rush_Rushing_Succ%': None,  # Can't calculate from season_averages, handle separately
        'W-L%': None,  # Calculate from season_records
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

    # Calculate adjustments for traditional QB metrics from QB data
    print("\nCalculating adjustments for traditional QB metrics from QB data...")
    traditional_metrics = ['Pass_Rate', 'Pass_Cmp%', 'Pass_QBR']
    
    for metric in traditional_metrics:
        if os.path.exists('qb_seasons_payment_labeled.csv'):
            qb_df = pd.read_csv('qb_seasons_payment_labeled.csv')
            qb_df['season'] = pd.to_numeric(qb_df['season'], errors='coerce')
            qb_df['GS'] = pd.to_numeric(qb_df['GS'], errors='coerce')
            qb_df[metric] = pd.to_numeric(qb_df[metric], errors='coerce')
            
            # Filter to starters only (8+ games)
            qb_filtered = qb_df[qb_df['GS'] >= 8].copy()
            
            # Calculate yearly averages
            yearly_avg = qb_filtered.groupby('season')[metric].mean().reset_index()
            yearly_avg = yearly_avg.dropna()
            
            if len(yearly_avg) > 10:  # Need enough data
                years = yearly_avg['season'].values
                values = yearly_avg[metric].values
                
                # Fit linear trend
                model = LinearRegression()
                model.fit(years.reshape(-1, 1), values)
                
                reference_avg = model.predict([[reference_year]])[0]
                predicted = model.predict(years.reshape(-1, 1))
                
                year_factors = {}
                for year, pred_val in zip(years, predicted):
                    year_factors[int(year)] = reference_avg / pred_val
                
                adjustment_factors[metric] = year_factors
                
                print(f"  {metric}: R² = {model.score(years.reshape(-1, 1), values):.4f}")
                print(f"  {reference_year} predicted average: {reference_avg:.2f}")

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


def apply_era_adjustments(df, adjustment_factors, stats_to_adjust=['total_yards', 'Pass_TD', 'Pass_ANY/A', 'Rush_Rushing_Succ%', 'Pass_Rate', 'Pass_Cmp%', 'Pass_QBR']):
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

    payment_df = load_csv_safe('qb_seasons_payment_labeled.csv')
    print(f"✓ Loaded fresh payment_labeled.csv")
    print(f"✓ Loaded {len(payment_df)} seasons")
    
    # Apply adjustments
    stats_to_adjust = [
        'total_yards', 
        'Pass_TD', 
        'Pass_ANY/A',
        'Rush_Rushing_Succ%',
        'Pass_Rate',      # Add passer rating
        'Pass_Cmp%',      # Add completion percentage
        'Pass_QBR'        # Add QBR
    ]
    adjusted_df = apply_era_adjustments(payment_df, adjustment_factors, stats_to_adjust)
    
    # Save
    output_file = 'qb_seasons_payment_labeled_era_adjusted.csv'
    adjusted_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved era-adjusted data to: {output_file}")
    
    return adjusted_df

