"""
Year weighting regression for payment decision analysis.

This module contains functions for determining how each prior year is weighted
in payment decisions using statistical significance testing.
"""

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from qb_research.utils.data_loading import load_csv_safe


def year_weighting_regression(metric='total_yards_adj', max_decision_year=6, n_bootstrap=1000):
    """
    FIXED VERSION: Properly defines all variables and handles bootstrap results.
    
    Determines how each prior year is weighted in payment decisions WITH STATISTICAL SIGNIFICANCE TESTING.
    """
    print("\n" + "="*80)
    print(f"FIXED YEAR WEIGHTING WITH P-VALUES: {metric}")
    print("="*80)
    
    # Load prepared payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        return None

    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with era-adjusted payment labels")
    
    # Handle team metrics if needed
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
    print(f"Using {n_bootstrap} bootstrap samples for significance testing")
    
    # Results storage
    all_results = {}
    
    # For each decision year (3-6)
    for decision_year in range(3, max_decision_year + 1):
        print("\n" + "="*80)
        print(f"DECISION YEAR {decision_year} WITH P-VALUES")
        print(f"Using performance from Years 0 to {decision_year - 1}")
        print("="*80)
        
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
            
            player_data.append(record)
        
        year_features = [f'{metric}_year{year}' for year in range(decision_year)]
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
        
        # Bootstrap significance testing
        print(f"\n🔬 BOOTSTRAP SIGNIFICANCE TESTING ({n_bootstrap} samples)...")
        
        bootstrap_coefs = []
        bootstrap_r2s = []
        
        for i in range(n_bootstrap):
            if i % 200 == 0:
                print(f"  Bootstrap sample {i}/{n_bootstrap}...")
            
            # Sample with replacement
            indices = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
            X_boot = X_scaled[indices]
            y_boot = y.iloc[indices].values
            
            # Fit model on bootstrap sample
            model_boot = LogisticRegression(penalty='l2', C=10.0, max_iter=1000, random_state=42)
            model_boot.fit(X_boot, y_boot)
            
            # Store coefficients and pseudo-R²
            bootstrap_coefs.append(model_boot.coef_[0])  # Extract 1D array
            
            # Calculate pseudo-R² (McFadden's R²)
            y_pred_prob = model_boot.predict_proba(X_boot)[:, 1]
            y_pred_prob = np.clip(y_pred_prob, 1e-15, 1-1e-15)  # Avoid log(0)
            
            log_likelihood = np.sum(y_boot * np.log(y_pred_prob) + (1 - y_boot) * np.log(1 - y_pred_prob))
            null_prob = np.mean(y_boot)
            null_prob = np.clip(null_prob, 1e-15, 1-1e-15)
            null_log_likelihood = len(y_boot) * (null_prob * np.log(null_prob) + (1 - null_prob) * np.log(1 - null_prob))
            
            pseudo_r2 = 1 - (log_likelihood / null_log_likelihood) if null_log_likelihood != 0 else 0
            bootstrap_r2s.append(pseudo_r2)
        
        bootstrap_coefs = np.array(bootstrap_coefs)
        bootstrap_r2s = np.array(bootstrap_r2s)
        
        print(f"✓ Bootstrap complete!")
        
        # Statistical significance results
        print(f"\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE RESULTS")
        print("="*60)
        
        # Original model for comparison
        original_model = LogisticRegression(penalty='l2', C=10.0, max_iter=1000, random_state=42)
        original_model.fit(X_scaled, y)
        
        significance_results = []
        
        for i, feature in enumerate(year_features):
            year_num = int(feature.split('year')[-1])
            coef_dist = bootstrap_coefs[:, i]
            original_coef = original_model.coef_[0][i]
            
            # Calculate confidence intervals (95%)
            ci_lower = np.percentile(coef_dist, 2.5)
            ci_upper = np.percentile(coef_dist, 97.5)
            
            # Statistical significance test: CI doesn't include zero
            is_significant = not (ci_lower <= 0 <= ci_upper)
            
            # Calculate p-value approximation (two-tailed)
            if original_coef > 0:
                p_value_approx = 2 * np.mean(coef_dist <= 0)
            else:
                p_value_approx = 2 * np.mean(coef_dist >= 0)
            p_value_approx = min(p_value_approx, 1.0)
            
            # Standard error
            se = np.std(coef_dist)
            
            # T-statistic approximation
            t_stat = original_coef / se if se > 0 else 0
            
            significance_results.append({
                'Decision_Year': decision_year,
                'Performance_Year': f'Year {year_num}',
                'Year_Number': year_num,
                'Feature': feature,
                'Coefficient': original_coef,
                'Std_Error': se,
                'T_Statistic': t_stat,
                'P_Value_Bootstrap': p_value_approx,
                'CI_Lower_95': ci_lower,
                'CI_Upper_95': ci_upper,
                'Significant_95': is_significant,
                'Coefficient_Mean_Bootstrap': np.mean(coef_dist),
                'Coefficient_Std_Bootstrap': np.std(coef_dist)
            })
        
        significance_df = pd.DataFrame(significance_results)
        significance_df = significance_df.sort_values('Year_Number')
        
        print(f"\nYear-by-year coefficients WITH P-VALUES:")
        print("(Bootstrap method with 95% confidence intervals)")
        
        for _, row in significance_df.iterrows():
            sig_marker = "***" if row['P_Value_Bootstrap'] < 0.001 else "**" if row['P_Value_Bootstrap'] < 0.01 else "*" if row['P_Value_Bootstrap'] < 0.05 else ""
            print(f"  {row['Performance_Year']}: coef={row['Coefficient']:7.4f}, p={row['P_Value_Bootstrap']:6.4f} {sig_marker}")
        
        # Calculate normalized weights
        total_abs = significance_df['Coefficient'].abs().sum()
        if total_abs > 0:
            significance_df['Weight_%'] = (significance_df['Coefficient'].abs() / total_abs * 100)
        else:
            significance_df['Weight_%'] = 0
        
        # FIXED: Model performance with proper variable definitions
        # Calculate original model performance
        y_pred_prob = original_model.predict_proba(X_scaled)[:, 1]
        y_pred_prob = np.clip(y_pred_prob, 1e-15, 1-1e-15)
        
        log_likelihood = np.sum(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
        null_prob = np.mean(y)
        null_prob = np.clip(null_prob, 1e-15, 1-1e-15)
        null_log_likelihood = len(y) * (null_prob * np.log(null_prob) + (1 - null_prob) * np.log(1 - null_prob))
        
        pseudo_r2 = 1 - (log_likelihood / null_log_likelihood) if null_log_likelihood != 0 else 0
        
        # FIXED: Properly define bootstrap statistics
        r2_mean_bootstrap = np.mean(bootstrap_r2s)
        r2_std_bootstrap = np.std(bootstrap_r2s)
        
        print(f"\nModel Performance:")
        print(f"  Original Pseudo-R²: {pseudo_r2:.4f}")
        print(f"  Bootstrap Pseudo-R² mean: {r2_mean_bootstrap:.4f} ± {r2_std_bootstrap:.4f}")
        
        # Identify significant years
        significant_years = significance_df[significance_df['Significant_95']]
        
        print(f"\n🔍 STATISTICALLY SIGNIFICANT YEARS ({len(significant_years)} of {len(significance_df)}):")
        for _, row in significant_years.iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"  ✓ {row['Performance_Year']}: {direction} payment probability (p = {row['P_Value_Bootstrap']:.4f})")
        
        if len(significant_years) == 0:
            print("  ⚠️  No years are statistically significant at p < 0.05")
        
        # Store results (FIXED: use properly defined variables)
        all_results[f'year_{decision_year}'] = {
            'decision_year': decision_year,
            'significance_results': significance_df,
            'coefficients': significance_df,  # For backward compatibility with extract_year_weights
            'model': original_model,
            'scaler': scaler,
            'features': year_features,
            'pseudo_r2': pseudo_r2,
            'r2_bootstrap_mean': r2_mean_bootstrap,  # FIXED: properly defined
            'r2_bootstrap_std': r2_std_bootstrap,    # FIXED: properly defined
            'n_samples': len(df),
            'bootstrap_coefficients': bootstrap_coefs,
            'bootstrap_r2s': bootstrap_r2s,
            'significant_years': significant_years
        }
    
    # Save results and create summary (same as original)
    if all_results:
        safe_metric_name = metric.replace('/', '_').replace('%', 'pct')
        
        # Combine all significance results
        all_significance_data = []
        for decision_year in range(3, max_decision_year + 1):
            key = f'year_{decision_year}'
            if key in all_results:
                result = all_results[key]
                sig_df = result['significance_results']
                
                for _, row in sig_df.iterrows():
                    all_significance_data.append({
                        'Decision_Year': decision_year,
                        'Performance_Year': row['Performance_Year'],
                        'Year_Number': row['Year_Number'],
                        'Coefficient': row['Coefficient'],
                        'P_Value_Bootstrap': row['P_Value_Bootstrap'],
                        'Significant_95': row['Significant_95'],
                        'Weight_%': row['Weight_%'],
                        'T_Statistic': row['T_Statistic'],
                        'CI_Lower_95': row['CI_Lower_95'],
                        'CI_Upper_95': row['CI_Upper_95']
                    })
        
        all_significance_df = pd.DataFrame(all_significance_data)
        
        if len(all_significance_df) > 0:
            # Save detailed results
            all_significance_df.to_csv(f'year_weights_significance_{safe_metric_name}.csv', index=False)
            print(f"\n✓ Detailed significance results saved to: year_weights_significance_{safe_metric_name}.csv")
            
            # Create and save summary matrices
            significance_matrix = all_significance_df.pivot(
                index='Performance_Year', 
                columns='Decision_Year', 
                values='Significant_95'
            )
            
            pvalue_matrix = all_significance_df.pivot(
                index='Performance_Year', 
                columns='Decision_Year', 
                values='P_Value_Bootstrap'
            )
            
            weight_matrix = all_significance_df.pivot(
                index='Performance_Year', 
                columns='Decision_Year', 
                values='Weight_%'
            )
            
            significance_matrix.to_csv(f'year_significance_matrix_{safe_metric_name}.csv')
            pvalue_matrix.to_csv(f'year_pvalue_matrix_{safe_metric_name}.csv')
            weight_matrix.to_csv(f'year_weight_matrix_{safe_metric_name}.csv')
            
            print(f"✓ Summary matrices saved:")
            print(f"  - year_significance_matrix_{safe_metric_name}.csv")
            print(f"  - year_pvalue_matrix_{safe_metric_name}.csv") 
            print(f"  - year_weight_matrix_{safe_metric_name}.csv")
            
            # Also save the weights file for backward compatibility
            weight_summary = all_significance_df[['Decision_Year', 'Performance_Year', 'Year_Number', 'Weight_%']].copy()
            weight_summary.to_csv(f'year_weights_{safe_metric_name}.csv', index=False)
            print(f"  - year_weights_{safe_metric_name}.csv")
    
    return all_results


def extract_year_weights_from_regression_results(results, decision_year):
    """
    Helper function to extract year weights from year_weighting_regression() output.
    
    Args:
        results: Output from year_weighting_regression()
        decision_year: Which decision year to extract weights for
    
    Returns:
        dict: {0: weight0, 1: weight1, ...} normalized to sum to 1.0
    """
    key = f'year_{decision_year}'
    
    if key not in results:
        print(f"⚠️  Warning: Decision year {decision_year} not found in results")
        return None
    
    coef_df = results[key]['coefficients']
    
    # Extract weights - they're already as percentages
    weights_dict = {}
    for _, row in coef_df.iterrows():
        year_num = int(row['Year_Number'])
        weight_pct = row['Weight_%']
        weights_dict[year_num] = weight_pct / 100.0  # Convert to decimal
    
    # Verify they sum to 1.0
    total = sum(weights_dict.values())
    print(f"\nExtracted weights for Decision Year {decision_year}:")
    for year in sorted(weights_dict.keys()):
        print(f"  Year {year}: {weights_dict[year]:.3f} ({weights_dict[year]*100:.1f}%)")
    print(f"  Total: {total:.3f}")
    
    return weights_dict

