"""
Surface and KNN-based payment probability models.

This module contains functions for creating 2D probability surfaces
that visualize payment probabilities across different performance metrics
and years since draft.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from IPython.display import display

from qb_research.utils.data_loading import load_csv_safe


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
        # Ensure KNN_surfaces directory exists
        os.makedirs('KNN_surfaces', exist_ok=True)
        output_file = f'KNN_surfaces/simple_knn_decision{decision_year}_{safe_metric_name}_k{k}.csv'
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

