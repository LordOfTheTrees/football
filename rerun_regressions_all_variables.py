#!/usr/bin/env python3
"""
Re-run Wins and Payment Regressions with Variable-Level Statistics

This script leverages the existing, well-designed functions from QB_research.py
and extracts variable-level coefficients and statistical significance for each input.

Runs BOTH raw and era-adjusted analyses for comparison.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from scipy import stats

# Import existing QB_research functions
try:
    from QB_research import (
        load_csv_safe,
        validate_columns,
        wins_prediction_linear_ridge,
        payment_prediction_logistic_ridge
    )
    print("✓ Successfully imported QB_research functions")
except ImportError as e:
    print(f"✗ ERROR: Could not import QB_research functions: {e}")
    print("Make sure QB_research.py is in the same directory or in your Python path")
    sys.exit(1)


def extract_wins_variable_stats(use_era_adjusted=True, n_bootstrap=500):
    """
    Extract variable-level statistics from wins prediction regression.
    
    This function runs the wins prediction and extracts coefficient statistics
    for each variable used in the regression.
    
    Args:
        use_era_adjusted: Whether to use era-adjusted data
        n_bootstrap: Number of bootstrap samples for p-value calculation
    
    Returns:
        DataFrame with variable statistics
    """
    label = "ERA-ADJUSTED" if use_era_adjusted else "RAW"
    suffix = "_era_adj" if use_era_adjusted else "_raw"
    
    print("\n" + "="*80)
    print(f"WINS PREDICTION: {label}")
    print("="*80)
    
    # Load data
    season_records = load_csv_safe("season_records.csv", "season records")
    if season_records is None:
        return None
    
    if use_era_adjusted:
        qb_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv', f"QB ({label})")
    else:
        qb_df = load_csv_safe('qb_seasons_payment_labeled.csv', f"QB ({label})")
    
    if qb_df is None:
        return None
    
    # Prepare data
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce').astype(int)
    qb_df['season'] = pd.to_numeric(qb_df['season'], errors='coerce').astype(int)
    
    merged_df = pd.merge(
        qb_df, 
        season_records, 
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'], 
        how='inner'
    )
    
    # Filter to starters (8+ games started)
    merged_df['GS_numeric'] = pd.to_numeric(merged_df['GS'], errors='coerce')
    merged_df = merged_df[merged_df['GS_numeric'] >= 8]
    merged_df['Wins'] = pd.to_numeric(merged_df['W'], errors='coerce')
    merged_df = merged_df.dropna(subset=['Wins'])
    
    # Define features with optional _adj suffix
    adj_suffix = "_adj" if use_era_adjusted else ""
    features = [
        f'Pass_Rate{adj_suffix}', f'Pass_Cmp%{adj_suffix}', f'Pass_QBR{adj_suffix}',
        f'Pass_TD{adj_suffix}', f'Pass_ANY/A{adj_suffix}', f'Rush_Rushing_Succ%{adj_suffix}',
        f'Pass_Int%{adj_suffix}', f'Pass_Sk%{adj_suffix}', f'total_yards{adj_suffix}',
        f'Pass_4QC{adj_suffix}', f'Pass_GWD{adj_suffix}', f'Rush_Rushing_TD{adj_suffix}',
        f'Rush_Rushing_Yds{adj_suffix}'
    ]
    
    # Keep only available features
    features = [f for f in features if f in merged_df.columns]
    print(f"Using {len(features)} features")
    
    # Prepare final dataset
    final_df = merged_df[features + ['Wins']].copy()
    for col in features:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    final_df = final_df.dropna()
    
    X = final_df[features].values
    y = final_df['Wins'].values
    
    print(f"Dataset: {len(final_df)} QB-seasons")
    print(f"Wins range: {y.min():.1f} to {y.max():.1f}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cross-validated Ridge to find optimal alpha (best practice)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_scaled, y)
    optimal_alpha = ridge_cv.alpha_
    
    # Fit final model
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f"Model: R²={r2:.4f}, RMSE={rmse:.4f}, α={optimal_alpha}")
    
    # Bootstrap for statistical significance
    print(f"Running {n_bootstrap} bootstrap samples for p-values...")
    
    n_samples, n_features = X_scaled.shape
    bootstrap_coefs = np.zeros((n_bootstrap, n_features))
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_scaled[indices]
        y_boot = y[indices]
        
        model_boot = Ridge(alpha=optimal_alpha)
        model_boot.fit(X_boot, y_boot)
        bootstrap_coefs[i] = model_boot.coef_
    
    # Calculate statistics
    mean_coefs = np.mean(bootstrap_coefs, axis=0)
    std_errors = np.std(bootstrap_coefs, axis=0)
    t_stats = mean_coefs / (std_errors + 1e-10)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_bootstrap-1))
    ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Variable': features,
        'Coefficient': mean_coefs,
        'Std_Error': std_errors,
        'T_Statistic': t_stats,
        'P_Value': p_values,
        'Significant_95%': p_values < 0.05,
        'CI_Lower_95%': ci_lower,
        'CI_Upper_95%': ci_upper
    })
    
    # Sort by absolute coefficient magnitude
    results_df = results_df.sort_values('Coefficient', key=abs, ascending=False)
    
    # Add model metadata
    results_df.insert(0, 'Adjustment_Type', label)
    results_df.insert(1, 'Model', 'Wins_Prediction')
    results_df.insert(2, 'R_Squared', r2)
    results_df.insert(3, 'RMSE', rmse)
    results_df.insert(4, 'Alpha', optimal_alpha)
    
    sig_count = results_df['Significant_95%'].sum()
    print(f"✓ Significant variables (p<0.05): {sig_count}/{len(features)}")
    
    return results_df


def extract_payment_variable_stats(use_era_adjusted=True, n_bootstrap=500):
    """
    Extract variable-level statistics from payment prediction logistic regression.
    
    This function runs the payment prediction and extracts coefficient statistics
    for each variable used in the logistic regression.
    
    Args:
        use_era_adjusted: Whether to use era-adjusted data
        n_bootstrap: Number of bootstrap samples for p-value calculation
    
    Returns:
        DataFrame with variable statistics
    """
    label = "ERA-ADJUSTED" if use_era_adjusted else "RAW"
    suffix = "_era_adj" if use_era_adjusted else "_raw"
    
    print("\n" + "="*80)
    print(f"PAYMENT PREDICTION: {label}")
    print("="*80)
    
    # Load data
    if use_era_adjusted:
        payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv', f"Payment ({label})")
    else:
        payment_df = load_csv_safe('qb_seasons_payment_labeled.csv', f"Payment ({label})")
    
    season_records = load_csv_safe("season_records.csv", "season records")
    
    if payment_df is None or season_records is None:
        return None
    
    # Filter to eligible QBs (exclude recent drafts)
    payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
    payment_df = payment_df[payment_df['draft_year'] <= 2020]
    
    # Prepare season data
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce').astype(int)
    payment_df['season'] = pd.to_numeric(payment_df['season'], errors='coerce').astype(int)
    
    merged_df = pd.merge(
        payment_df,
        season_records[['Season', 'Team', 'W-L%', 'Pts']],
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    # Create adjusted points if factors available
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors).fillna(1.0)
    else:
        merged_df['Pts_adj'] = merged_df['Pts']
    
    # Define metrics
    adj_suffix = "_adj" if use_era_adjusted else ""
    qb_metrics = [
        f'Pass_Rate{adj_suffix}',
        f'Pass_Cmp%{adj_suffix}',
        f'Pass_QBR{adj_suffix}',
        f'total_yards{adj_suffix}',
        f'Pass_TD{adj_suffix}',
        f'Pass_ANY/A{adj_suffix}',
        f'Rush_Rushing_Succ%{adj_suffix}'
    ]
    team_metrics = ['W-L%', 'Pts_adj']
    all_metrics = qb_metrics + team_metrics
    
    # Keep only available metrics
    all_metrics = [m for m in all_metrics if m in merged_df.columns]
    
    # Create lagged features (3-year average approach from existing code)
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in all_metrics:
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            merged_df[f"{metric}_lag{lag}"] = merged_df.groupby('player_id')[metric].shift(lag)
    
    # Create averaged features
    features = []
    for metric in all_metrics:
        avg_col = f"{metric}_avg"
        merged_df[avg_col] = merged_df[[f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]].mean(axis=1)
        features.append(avg_col)
    
    print(f"Using {len(features)} lagged average features")
    
    # Prepare final dataset
    X_full = merged_df[features + ['got_paid']].copy().dropna(subset=features)
    y = X_full['got_paid'].astype(int).values
    X = X_full[features].values
    
    print(f"Dataset: {len(X)} QB-seasons")
    print(f"Paid: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic regression (best practice for binary outcomes)
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_scaled, y)
    
    # Evaluate model
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    print(f"Model: Accuracy={accuracy:.4f}, ROC AUC={roc_auc:.4f}")
    
    # Bootstrap for statistical significance
    print(f"Running {n_bootstrap} bootstrap samples for p-values...")
    
    n_samples, n_features = X_scaled.shape
    bootstrap_coefs = np.zeros((n_bootstrap, n_features))
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_scaled[indices]
        y_boot = y[indices]
        
        model_boot = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        model_boot.fit(X_boot, y_boot)
        bootstrap_coefs[i] = model_boot.coef_.ravel()
    
    # Calculate statistics
    mean_coefs = np.mean(bootstrap_coefs, axis=0)
    std_errors = np.std(bootstrap_coefs, axis=0)
    t_stats = mean_coefs / (std_errors + 1e-10)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_bootstrap-1))
    ci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Variable': features,
        'Coefficient': mean_coefs,
        'Std_Error': std_errors,
        'T_Statistic': t_stats,
        'P_Value': p_values,
        'Significant_95%': p_values < 0.05,
        'CI_Lower_95%': ci_lower,
        'CI_Upper_95%': ci_upper
    })
    
    # Sort by absolute coefficient magnitude
    results_df = results_df.sort_values('Coefficient', key=abs, ascending=False)
    
    # Add model metadata
    results_df.insert(0, 'Adjustment_Type', label)
    results_df.insert(1, 'Model', 'Payment_Prediction')
    results_df.insert(2, 'Accuracy', accuracy)
    results_df.insert(3, 'ROC_AUC', roc_auc)
    
    sig_count = results_df['Significant_95%'].sum()
    print(f"✓ Significant variables (p<0.05): {sig_count}/{len(features)}")
    
    return results_df


def main():
    """Main execution function"""
    print("="*80)
    print("QB RESEARCH: COMPREHENSIVE VARIABLE-LEVEL STATISTICS")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This script leverages existing QB_research functions to extract")
    print("coefficient statistics for each variable in the regressions.")
    print()
    print("Outputs:")
    print("  - Variable coefficients")
    print("  - P-values (bootstrap-based)")
    print("  - 95% confidence intervals")
    print("  - Statistical significance indicators")
    print("="*80)
    
    # Create output directory
    output_dir = 'regression_variable_statistics'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}/")
    
    # Configuration
    n_bootstrap = 500  # Balance between accuracy and speed
    
    print(f"\nConfiguration:")
    print(f"  Bootstrap samples: {n_bootstrap}")
    print(f"  Analyses: 4 (wins raw, wins era-adj, payment raw, payment era-adj)")
    
    # Run all four analyses
    all_results = []
    
    # 1. Wins Prediction - Raw
    print("\n" + "="*80)
    print("[1/4] WINS PREDICTION - RAW DATA")
    print("="*80)
    wins_raw = extract_wins_variable_stats(use_era_adjusted=False, n_bootstrap=n_bootstrap)
    if wins_raw is not None:
        output_file = os.path.join(output_dir, 'wins_prediction_variable_stats_raw.csv')
        wins_raw.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")
        all_results.append(('Wins Raw', wins_raw))
    
    # 2. Wins Prediction - Era-Adjusted
    print("\n" + "="*80)
    print("[2/4] WINS PREDICTION - ERA-ADJUSTED DATA")
    print("="*80)
    wins_era = extract_wins_variable_stats(use_era_adjusted=True, n_bootstrap=n_bootstrap)
    if wins_era is not None:
        output_file = os.path.join(output_dir, 'wins_prediction_variable_stats_era_adj.csv')
        wins_era.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")
        all_results.append(('Wins Era-Adj', wins_era))
    
    # 3. Payment Prediction - Raw
    print("\n" + "="*80)
    print("[3/4] PAYMENT PREDICTION - RAW DATA")
    print("="*80)
    payment_raw = extract_payment_variable_stats(use_era_adjusted=False, n_bootstrap=n_bootstrap)
    if payment_raw is not None:
        output_file = os.path.join(output_dir, 'payment_prediction_variable_stats_raw.csv')
        payment_raw.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")
        all_results.append(('Payment Raw', payment_raw))
    
    # 4. Payment Prediction - Era-Adjusted
    print("\n" + "="*80)
    print("[4/4] PAYMENT PREDICTION - ERA-ADJUSTED DATA")
    print("="*80)
    payment_era = extract_payment_variable_stats(use_era_adjusted=True, n_bootstrap=n_bootstrap)
    if payment_era is not None:
        output_file = os.path.join(output_dir, 'payment_prediction_variable_stats_era_adj.csv')
        payment_era.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")
        all_results.append(('Payment Era-Adj', payment_era))
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if all_results:
        print(f"\n✓ Successfully completed {len(all_results)}/4 analyses")
        print(f"\nOutput files in {output_dir}/:")
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB)")
        
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        
        for name, df in all_results:
            sig_vars = df['Significant_95%'].sum()
            total_vars = len(df)
            avg_p = df['P_Value'].mean()
            
            print(f"\n{name}:")
            print(f"  Variables: {total_vars}")
            print(f"  Significant (p<0.05): {sig_vars} ({100*sig_vars/total_vars:.1f}%)")
            print(f"  Average p-value: {avg_p:.4f}")
            
            if 'R_Squared' in df.columns:
                print(f"  R²: {df['R_Squared'].iloc[0]:.4f}")
            if 'ROC_AUC' in df.columns:
                print(f"  ROC AUC: {df['ROC_AUC'].iloc[0]:.4f}")
        
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("""
P-Value Interpretation:
  p < 0.05: Statistically significant (95% confidence)
  p < 0.01: Highly significant (99% confidence)
  p < 0.001: Very highly significant (99.9% confidence)

Coefficient Interpretation:
  WINS PREDICTION (Ridge):
    - Positive: More of this metric → More wins
    - Negative: More of this metric → Fewer wins
    
  PAYMENT PREDICTION (Logistic):
    - Positive: More of this metric → Higher payment probability
    - Negative: More of this metric → Lower payment probability

Confidence Intervals:
  - If both CI bounds positive/negative → Significant effect
  - If CI spans zero → Not statistically significant
  - Narrower CI → More precise estimate

Comparing Raw vs Era-Adjusted:
  1. Check which variables are significant in BOTH
  2. Compare coefficient magnitudes
  3. Look for variables that become/stop being significant
  4. Evaluate overall model performance (R²/ROC AUC)
        """)
        
        print("="*80)
        print("✓ All analyses complete! Check the CSV files for detailed results.")
        print("="*80)
    
    else:
        print("\n✗ No analyses completed successfully")
        print("Check error messages above for details")
    
    return output_dir


if __name__ == "__main__":
    main()