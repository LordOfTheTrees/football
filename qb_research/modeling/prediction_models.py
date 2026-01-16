"""
Prediction models for QB research.

This module contains functions for:
- Wins prediction using ridge regression
- Payment prediction using logistic regression
- Enhanced payment prediction with confusion matrix analysis
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from IPython.display import display

from qb_research.utils.data_loading import load_csv_safe


def ridge_regression_payment_prediction(alpha_range=None, exclude_recent_drafts=True):
    """
    FIXED VERSION: Uses LogisticRegression for binary classification instead of Ridge.
    
    Predicts QB payment decisions using logistic regression with proper classification metrics.
    """
    print("\n" + "="*80)
    print("FIXED PAYMENT PREDICTION: LOGISTIC REGRESSION")
    print("Using AVERAGED performance (Years 1-3)")
    print("="*80)
    
    # Load prepared payment data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        return None

    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with payment labels")
    
    # Filter to eligible QBs (drafted ≤2020)
    if exclude_recent_drafts:
        cutoff_year = 2020
        payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
        
        before_filter = len(payment_df['player_id'].unique())
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        after_filter = len(payment_df['player_id'].unique())
        
        print(f"Excluding QBs drafted after {cutoff_year}")
        print(f"  Before: {before_filter} unique QBs")
        print(f"  After: {after_filter} unique QBs")
    
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
    
    # Create adjusted Pts
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors)
        print(f"✓ Created Pts_adj: {merged_df['Pts_adj'].notna().sum()} values")
    else:
        merged_df['Pts_adj'] = merged_df['Pts']

    # Define metrics
    qb_metrics = [
        'total_yards_adj',
        'Pass_TD_adj', 
        'Pass_ANY/A_adj',
        'Rush_Rushing_Succ%_adj'
    ]
    
    team_metrics = [
        'W-L%',
        'Pts_adj'
    ]
    
    all_metrics = qb_metrics + team_metrics
    
    # Create lags and averages
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in all_metrics:
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            lag_col = f"{metric}_lag{lag}"
            merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
    
    # Create averaged features
    features = []
    for metric in all_metrics:
        avg_col = f"{metric}_avg"
        merged_df[avg_col] = merged_df[[f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]].mean(axis=1)
        features.append(avg_col)
    
    # Prepare features and target
    X = merged_df[features + ['got_paid', 'player_id']].copy()
    X = X.dropna(subset=features)
    
    print(f"\nComplete cases: {len(X)}")
    
    if len(X) < 30:
        print("\n✗ ERROR: Insufficient complete cases for modeling")
        return None
    
    y = X['got_paid'].astype(int)
    X_features = X[features]
    
    # Payment distribution
    paid_count = y.sum()
    paid_pct = paid_count / len(y) * 100
    print(f"\nPayment outcome distribution:")
    print(f"  Got paid: {paid_count} ({paid_pct:.1f}%)")
    print(f"  Not paid: {len(y) - paid_count} ({100-paid_pct:.1f}%)")
    
    # Train/test split (80/20)
    split_idx = int(len(X_features) * 0.8)
    X_train = X_features.iloc[:split_idx]
    X_test = X_features.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} paid)")
    print(f"  Test: {len(X_test)} samples ({y_test.sum()} paid)")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # FIXED: Use LogisticRegression for binary classification
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION WITH L2 REGULARIZATION")
    print("="*80)
    
    if alpha_range is None:
        # C = 1/alpha, so larger C = less regularization
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    else:
        C_values = [1.0/alpha for alpha in alpha_range]
    
    # Cross-validation to find best C
    best_score = -np.inf
    best_C = 1.0
    
    for C in C_values:
        model = LogisticRegression(penalty='l2', C=C, max_iter=1000, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        mean_score = scores.mean()
        
        print(f"  C={C:8.3f}: CV AUC = {mean_score:.4f} ± {scores.std()*2:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    print(f"\n✓ Best C: {best_C} (CV AUC: {best_score:.4f})")
    
    # Train final model
    final_model = LogisticRegression(penalty='l2', C=best_C, max_iter=1000, random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    # Training performance
    y_train_prob = final_model.predict_proba(X_train_scaled)[:, 1]
    y_train_pred = final_model.predict(X_train_scaled)
    
    train_auc = roc_auc_score(y_train, y_train_prob)
    train_acc = np.mean(y_train == y_train_pred)
    
    print(f"\n" + "="*80)
    print("TRAINING PERFORMANCE")
    print("="*80)
    print(f"  AUC = {train_auc:.4f}")
    print(f"  Accuracy = {train_acc:.4f}")
    
    # Test performance  
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = final_model.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_acc = np.mean(y_test == y_test_pred)
    
    print(f"\n" + "="*80)
    print("TEST PERFORMANCE")
    print("="*80)
    print(f"  AUC = {test_auc:.4f}")
    print(f"  Accuracy = {test_acc:.4f}")
    
    # FIXED: Variable importance with proper coefficient handling
    print(f"\n" + "="*80)
    print("VARIABLE IMPORTANCE")
    print("="*80)
    
    # Get coefficients (1D array for logistic regression)
    coefficients = final_model.coef_[0]  # Extract 1D array from 2D
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nAll features ranked by importance:")
    for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {idx}. {row['Feature']}: {direction} payment probability (coef={row['Coefficient']:.4f})")
    
    # Confusion matrix at default threshold (0.5)
    print(f"\n" + "="*80)
    print("CONFUSION MATRIX (Threshold = 0.5)")
    print("="*80)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTest Set Confusion Matrix:")
    print(f"                Predicted")
    print(f"              No Pay  Pay")
    print(f"Actual No Pay    {tn:3d}  {fp:3d}")
    print(f"Actual Pay       {fn:3d}  {tp:3d}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {precision:.3f} ({tp}/{tp + fp} predicted pays were correct)")
    print(f"  Recall:    {recall:.3f} ({tp}/{tp + fn} actual pays were caught)")
    print(f"  F1-Score:  {f1:.3f}")
    
    # Summary
    top_3_vars = importance_df.head(3)['Feature'].tolist()
    best_predictor = importance_df.iloc[0]['Feature']
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTop 3 Most Important Predictors:")
    for i, var in enumerate(top_3_vars, 1):
        coef = importance_df[importance_df['Feature']==var]['Coefficient'].values[0]
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {i}. {var}: {direction} payment probability")
    
    print(f"\n🏆 BEST PREDICTOR: {best_predictor.replace('_avg', '')}")
    print(f"   Model Performance: AUC = {test_auc:.3f}")
    
    # Save results
    results = {
        'model': final_model,
        'scaler': scaler,
        'best_C': best_C,
        'best_alpha': 1.0 / best_C,  # Convert C to alpha for compatibility
        'features': features,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'variable_importance': importance_df,
        'confusion_matrix': cm,
        'classification_metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'best_predictor': best_predictor.replace('_avg', '')
    }
    
    # Save CSV files
    os.makedirs('payment_prediction_min_variables', exist_ok=True)
    importance_df.to_csv('payment_prediction_min_variables/payment_prediction_logistic_importance.csv', index=False)
    
    # Save predictions
    test_results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted_Probability': y_test_prob,
        'Predicted_Class': y_test_pred
    })
    test_results_df.to_csv('payment_prediction_min_variables/payment_prediction_logistic_test_results.csv', index=False)
    
    print(f"\n✓ Results saved to:")
    print(f"  - payment_prediction_logistic_importance.csv")
    print(f"  - payment_prediction_logistic_test_results.csv")
    
    return results


def wins_prediction_linear_ridge(alpha_range=None, exclude_recent_drafts=True, n_bootstrap=1000, qb_metrics=['total_yards_adj', 'Pass_TD_adj', 'Pass_ANY/A_adj', 'Rush_Rushing_Succ%_adj']):
    """
    Enhanced ridge regression for wins prediction with bootstrap-based statistical significance testing.
    
    Args:
        alpha_range: Alpha values for ridge regression
        exclude_recent_drafts: Whether to exclude recent draft classes
        n_bootstrap: Number of bootstrap samples for significance testing
        qb_metrics: List of QB performance metrics to use
    
    Returns:
        dict: Results including significance tests and confidence intervals
    """
    print("\n" + "="*80)
    print("WINS PREDICTION WITH RIDGE REGRESSION")
    print("Using AVERAGED performance (Years 1-3)")
    print("="*80)
    
    # Load prepared payment data (same source as payment prediction)
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        return None

    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with payment labels")
    
    # Filter to eligible QBs (drafted ≤2020)
    if exclude_recent_drafts:
        cutoff_year = 2020
        payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
        
        before_filter = len(payment_df['player_id'].unique())
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        after_filter = len(payment_df['player_id'].unique())
        
        print(f"Excluding QBs drafted after {cutoff_year}")
        print(f"  Before: {before_filter} unique QBs")
        print(f"  After: {after_filter} unique QBs")
    
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
        season_records[['Season', 'Team', 'W-L%', 'Pts', 'W']],
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"✓ Merged with team metrics: {len(merged_df)} seasons")
    
    # Create adjusted Pts
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors)
        print(f"✓ Created Pts_adj: {merged_df['Pts_adj'].notna().sum()} values")
    else:
        merged_df['Pts_adj'] = merged_df['Pts']

    # Define metrics 
    team_metrics = []
    
    all_metrics = qb_metrics + team_metrics
    
    # Create lags and averages (same as payment prediction)
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in all_metrics:
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            lag_col = f"{metric}_lag{lag}"
            merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
    
    # Create averaged features
    features = []
    for metric in all_metrics:
        avg_col = f"{metric}_avg"
        merged_df[avg_col] = merged_df[[f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]].mean(axis=1)
        features.append(avg_col)
    
    # Prepare features and target (using Wins instead of got_paid)
    X = merged_df[features + ['W', 'player_id']].copy()
    X = X.dropna(subset=features)
    
    print(f"\nComplete cases: {len(X)}")
    
    if len(X) < 30:
        print("\n✗ ERROR: Insufficient complete cases for modeling")
        return None
    
    # Convert wins to numeric
    X['W'] = pd.to_numeric(X['W'], errors='coerce')
    X = X.dropna(subset=['W'])
    
    y = X['W']  # Target is wins, not payment
    X_features = X[features]
    
    # Wins distribution
    print(f"\nWins distribution:")
    print(f"  Min: {y.min()} wins")
    print(f"  Mean: {y.mean():.1f} wins")
    print(f"  Max: {y.max()} wins")
    
    # Train/test split (80/20)
    split_idx = int(len(X_features) * 0.8)
    X_train = X_features.iloc[:split_idx]
    X_test = X_features.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} samples (mean wins: {y_train.mean():.1f})")
    print(f"  Test: {len(X_test)} samples (mean wins: {y_test.mean():.1f})")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge regression with cross-validation
    print("\n" + "="*80)
    print("RIDGE REGRESSION WITH CROSS-VALIDATION")
    print("="*80)
    
    if alpha_range is None:
        alpha_range = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Cross-validation to find best alpha
    best_score = -np.inf
    best_alpha = 1.0
    
    for alpha in alpha_range:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        mean_score = scores.mean()
        
        print(f"  Alpha={alpha:8.1f}: CV R² = {mean_score:.4f} ± {scores.std()*2:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"\n✓ Best Alpha: {best_alpha} (CV R²: {best_score:.4f})")
    
    # Train final model
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train_scaled, y_train)
    
    # Training performance
    y_train_pred = final_model.predict(X_train_scaled)
    train_r2 = final_model.score(X_train_scaled, y_train)
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
    
    print(f"\n" + "="*80)
    print("TRAINING PERFORMANCE")
    print("="*80)
    print(f"  R² = {train_r2:.4f}")
    print(f"  RMSE = {train_rmse:.3f} wins")
    
    # Test performance  
    y_test_pred = final_model.predict(X_test_scaled)
    test_r2 = final_model.score(X_test_scaled, y_test)
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
    
    print(f"\n" + "="*80)
    print("TEST PERFORMANCE")
    print("="*80)
    print(f"  R² = {test_r2:.4f}")
    print(f"  RMSE = {test_rmse:.3f} wins")
    
    # Bootstrap significance testing
    print(f"\n" + "="*80)
    print("BOOTSTRAP STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    print(f"Running {n_bootstrap} bootstrap samples for significance testing...")
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Features: {len(features)}")
    print(f"Optimal alpha: {best_alpha}")
    
    # Bootstrap resampling
    bootstrap_coefs = []
    bootstrap_r2s = []
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"  Bootstrap sample {i}/{n_bootstrap}...")
        
        # Sample with replacement
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_boot = X_train_scaled[indices]
        y_boot = y_train.iloc[indices].values
        
        # Fit model on bootstrap sample
        model_boot = Ridge(alpha=best_alpha)
        model_boot.fit(X_boot, y_boot)
        
        # Store coefficients and R²
        bootstrap_coefs.append(model_boot.coef_)
        r2_boot = model_boot.score(X_boot, y_boot)
        bootstrap_r2s.append(r2_boot)
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    bootstrap_r2s = np.array(bootstrap_r2s)
    
    print(f"\nBootstrap complete!")
    
    # Calculate significance statistics
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("="*80)
    
    # Original model for comparison
    original_model = Ridge(alpha=best_alpha)
    original_model.fit(X_train_scaled, y_train)
    
    significance_results = []
    
    for i, feature in enumerate(features):
        coef_dist = bootstrap_coefs[:, i]
        original_coef = original_model.coef_[i]
        
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
    significance_df = significance_df.sort_values('T_Statistic', key=abs, ascending=False)
    
    print("\nStatistical Significance Test Results:")
    print("(Bootstrap method with 95% confidence intervals)")
    print("\nInterpretation:")
    print("  - Significant_95 = True: 95% CI doesn't include zero (statistically significant)")
    print("  - P_Value_Bootstrap: Approximate two-tailed p-value")
    print("  - T_Statistic: Coefficient / Standard Error")
    
    display(significance_df[['Feature', 'Coefficient', 'Std_Error', 'T_Statistic', 
                            'P_Value_Bootstrap', 'CI_Lower_95', 'CI_Upper_95', 'Significant_95']])
    
    # Model stability analysis
    print(f"\n" + "="*80)
    print("MODEL STABILITY ANALYSIS")
    print("="*80)
    
    print(f"Bootstrap R² statistics:")
    print(f"  Mean R²: {np.mean(bootstrap_r2s):.4f}")
    print(f"  Std R²: {np.std(bootstrap_r2s):.4f}")
    print(f"  95% CI R²: [{np.percentile(bootstrap_r2s, 2.5):.4f}, {np.percentile(bootstrap_r2s, 97.5):.4f}]")
    print(f"  Original R²: {original_model.score(X_train_scaled, y_train):.4f}")
    
    # Identify most stable predictors
    print(f"\n🔍 ALL PREDICTORS WITH P-VALUES ({len(significance_df)} total):")
    print("YOU INTERPRET - no arbitrary significance thresholds applied")
    for _, row in significance_df.iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {row['Feature']}: {direction} wins (p = {row['P_Value_Bootstrap']:.4f})")

    # Keep significant_predictors for the results dict but don't filter the display
    significant_predictors = significance_df[significance_df['Significant_95']]  # Keep for compatibility
    
    # Variable importance
    print(f"\n" + "="*80)
    print("VARIABLE IMPORTANCE")
    print("="*80)
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': final_model.coef_,
        'Abs_Coefficient': np.abs(final_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nAll features ranked by importance:")
    for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {idx}. {row['Feature']}: {direction} wins (coef={row['Coefficient']:.4f})")
    
    # Save results
    significance_df.to_csv('wins_prediction_ridge_significance_tests.csv', index=False)
    importance_df.to_csv('wins_prediction_ridge_importance.csv', index=False)
    
    # Create results dict
    results = {
        'model': final_model,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'features': features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'significance_tests': significance_df,
        'bootstrap_coefficients': bootstrap_coefs,
        'bootstrap_r2s': bootstrap_r2s,
        'significant_predictors': significant_predictors,
        'variable_importance': importance_df,
        'model_stability': {
            'r2_mean': np.mean(bootstrap_r2s),
            'r2_std': np.std(bootstrap_r2s),
            'r2_ci_lower': np.percentile(bootstrap_r2s, 2.5),
            'r2_ci_upper': np.percentile(bootstrap_r2s, 97.5)
        }
    }
    
    print(f"\n✓ Results saved to:")
    print(f"  - wins_prediction_ridge_significance_tests.csv")
    print(f"  - wins_prediction_ridge_importance.csv")
    
    return results


def payment_prediction_logistic_ridge(alpha_range=None, exclude_recent_drafts=True, probability_thresholds=None, qb_metrics = ['total_yards_adj', 'Pass_TD_adj', 'Pass_ANY/A_adj', 'Rush_Rushing_Succ%_adj'], n_bootstrap=1000):
    """
    Enhanced payment prediction with comprehensive confusion matrix analysis, classification metrics, and bootstrap p-values.
    
    Args:
        alpha_range: Alpha values for ridge regression
        exclude_recent_drafts: Whether to exclude recent draft classes
        probability_thresholds: List of probability thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7])
        n_bootstrap: Number of bootstrap samples for p-value calculation
    
    Returns:
        dict: Results including confusion matrices, ROC curves, optimal thresholds, and p-values
    """
    print("\n" + "="*80)
    print("PAYMENT PREDICTION WITH CONFUSION MATRIX ANALYSIS AND P-VALUES")
    print("="*80)
    
    # Run the base payment prediction first
    base_results = ridge_regression_payment_prediction(alpha_range, exclude_recent_drafts)
    
    if base_results is None:
        return None
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX AND CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # We need to rebuild the data and model to get predictions
    # (This duplicates some logic from the original function, but ensures we get the exact same data)
    
    # Load data (same as original function)
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    
    # Filter to eligible QBs
    if exclude_recent_drafts:
        cutoff_year = 2020
        payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
    
    # Load and merge season records (same as original)
    season_records = load_csv_safe("season_records.csv")
    if season_records is None:
        return None
    
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
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
    
    # Create adjusted Pts (same as original)
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors)
    else:
        merged_df['Pts_adj'] = merged_df['Pts']
    
    # Create features (same as original)
    team_metrics = ['W-L%', 'Pts_adj']
    all_metrics = qb_metrics + team_metrics
    
    # Create lags and averages (simplified version)
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in all_metrics:
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            lag_col = f"{metric}_lag{lag}"
            merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
    
    # Create averaged features
    features = []
    for metric in all_metrics:
        avg_col = f"{metric}_avg"
        merged_df[avg_col] = merged_df[[f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]].mean(axis=1)
        features.append(avg_col)
    
    # Prepare final dataset
    X_full = merged_df[features + ['got_paid', 'player_id']].copy()
    X_full = X_full.dropna(subset=features)
    
    y_full = X_full['got_paid'].astype(int)
    X_full_features = X_full[features]
    
    # Train/test split (80/20)
    split_idx = int(len(X_full_features) * 0.8)
    X_train = X_full_features.iloc[:split_idx]
    X_test = X_full_features.iloc[split_idx:]
    y_train = y_full.iloc[:split_idx]
    y_test = y_full.iloc[split_idx:]
    
    print(f"Dataset for confusion matrix analysis:")
    print(f"  Total samples: {len(X_full_features)}")
    print(f"  Train: {len(X_train)} (paid: {y_train.sum()})")
    print(f"  Test: {len(X_test)} (paid: {y_test.sum()})")
    
    # Standardize and train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use the best alpha from base results (convert C to alpha)
    best_C = base_results.get('best_C', 1.0)
    best_alpha = 1.0 / best_C if best_C > 0 else 100.0
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)
    
    # Get probability predictions
    y_train_prob = model.predict(X_train_scaled)
    y_test_prob = model.predict(X_test_scaled)
    
    # Clip probabilities to [0, 1] range
    y_train_prob = np.clip(y_train_prob, 0, 1)
    y_test_prob = np.clip(y_test_prob, 0, 1)
    
    print(f"\nPredicted probability range:")
    print(f"  Train: {y_train_prob.min():.3f} to {y_train_prob.max():.3f}")
    print(f"  Test: {y_test_prob.min():.3f} to {y_test_prob.max():.3f}")
    
    # Test multiple probability thresholds
    if probability_thresholds is None:
        probability_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS AT DIFFERENT THRESHOLDS")
    print("="*80)
    
    threshold_results = []
    
    for threshold in probability_thresholds:
        print(f"\n📊 THRESHOLD = {threshold}")
        print("-" * 40)
        
        # Convert probabilities to binary predictions
        y_train_pred = (y_train_prob >= threshold).astype(int)
        y_test_pred = (y_test_prob >= threshold).astype(int)
        
        # Training set confusion matrix
        cm_train = confusion_matrix(y_train, y_train_pred)
        print(f"\nTraining Set Confusion Matrix:")
        print(f"                Predicted")
        print(f"              No Pay  Pay")
        print(f"Actual No Pay    {cm_train[0,0]:3d}  {cm_train[0,1]:3d}")
        print(f"Actual Pay       {cm_train[1,0]:3d}  {cm_train[1,1]:3d}")
        
        # Test set confusion matrix  
        cm_test = confusion_matrix(y_test, y_test_pred)
        print(f"\nTest Set Confusion Matrix:")
        print(f"                Predicted")
        print(f"              No Pay  Pay")
        print(f"Actual No Pay    {cm_test[0,0]:3d}  {cm_test[0,1]:3d}")
        print(f"Actual Pay       {cm_test[1,0]:3d}  {cm_test[1,1]:3d}")
        
        # Calculate metrics for test set
        tn, fp, fn, tp = cm_test.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nTest Set Classification Metrics:")
        print(f"  Accuracy:    {accuracy:.3f} ({tp + tn}/{len(y_test)} correct)")
        print(f"  Precision:   {precision:.3f} ({tp}/{tp + fp} predicted pays were correct)")
        print(f"  Recall:      {recall:.3f} ({tp}/{tp + fn} actual pays were caught)")
        print(f"  Specificity: {specificity:.3f} ({tn}/{tn + fp} actual no-pays were caught)")
        print(f"  F1-Score:    {f1_score:.3f}")
        
        # Store results
        threshold_results.append({
            'threshold': threshold,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_specificity': specificity,
            'test_f1_score': f1_score,
            'test_tp': tp,
            'test_fp': fp,
            'test_tn': tn,
            'test_fn': fn,
            'confusion_matrix_test': cm_test,
            'confusion_matrix_train': cm_train
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    # Find optimal threshold
    print(f"\n" + "="*80)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*80)
    
    print("\nPerformance Summary Across Thresholds:")
    display(threshold_df[['threshold', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']])
    
    # Find best threshold by F1 score
    best_f1_idx = threshold_df['test_f1_score'].idxmax()
    best_threshold = threshold_df.iloc[best_f1_idx]
    
    print(f"\n🎯 OPTIMAL THRESHOLD (by F1-score): {best_threshold['threshold']}")
    print(f"   F1-Score: {best_threshold['test_f1_score']:.3f}")
    print(f"   Accuracy: {best_threshold['test_accuracy']:.3f}")
    print(f"   Precision: {best_threshold['test_precision']:.3f}")
    print(f"   Recall: {best_threshold['test_recall']:.3f}")
    
    # ROC Analysis
    try:
        auc_score = roc_auc_score(y_test, y_test_prob)
        print(f"\n📈 ROC AUC Score: {auc_score:.3f}")
        
        if auc_score > 0.7:
            print("   → Good discriminative ability")
        elif auc_score > 0.6:
            print("   → Moderate discriminative ability")  
        else:
            print("   → Limited discriminative ability")
            
    except Exception as e:
        print(f"\n⚠️  Could not calculate ROC AUC: {e}")
        auc_score = None
    
    # Feature importance in classification context
    print(f"\n" + "="*80)
    print("FEATURE IMPORTANCE FOR CLASSIFICATION")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nTop predictors of getting paid:")
    for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {i+1}. {row['Feature']}: {direction} payment probability")
    
    # Bootstrap significance testing for p-values
    print(f"\n" + "="*80)
    print("BOOTSTRAP P-VALUE CALCULATION")
    print("="*80)
    
    print(f"Running {n_bootstrap} bootstrap samples for p-value calculation...")
    
    bootstrap_coefs = []
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"  Bootstrap sample {i}/{n_bootstrap}...")
        
        # Sample with replacement
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_boot = X_train_scaled[indices]
        y_boot = y_train.iloc[indices].values
        
        # Fit model on bootstrap sample
        model_boot = LogisticRegression(penalty='l2', C=best_C, max_iter=1000, random_state=42)
        model_boot.fit(X_boot, y_boot)
        
        # Store coefficients
        bootstrap_coefs.append(model_boot.coef_[0])  # Extract 1D array
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    print(f"Bootstrap complete!")
    
    # Calculate p-values for each feature
    print(f"\nCalculating p-values for each feature...")
    
    # Fit logistic regression version of the model for comparison
    logistic_model = LogisticRegression(penalty='l2', C=best_C, max_iter=1000, random_state=42)
    logistic_model.fit(X_train_scaled, y_train)
    
    p_value_results = []
    
    for i, feature in enumerate(features):
        coef_dist = bootstrap_coefs[:, i]
        original_coef = logistic_model.coef_[0][i]
        
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
        
        # Z-statistic approximation (for logistic regression)
        z_stat = original_coef / se if se > 0 else 0
        
        p_value_results.append({
            'Feature': feature,
            'Coefficient': original_coef,
            'Std_Error': se,
            'Z_Statistic': z_stat,
            'P_Value_Bootstrap': p_value_approx,
            'CI_Lower_95': ci_lower,
            'CI_Upper_95': ci_upper,
            'Significant_95': is_significant,
            'Coefficient_Mean_Bootstrap': np.mean(coef_dist),
            'Coefficient_Std_Bootstrap': np.std(coef_dist)
        })
    
    p_values_df = pd.DataFrame(p_value_results)
    p_values_df = p_values_df.sort_values('Z_Statistic', key=abs, ascending=False)
    
    print("\nBootstrap P-Values for Payment Prediction:")
    print("(Bootstrap method with 95% confidence intervals)")
    display(p_values_df[['Feature', 'Coefficient', 'P_Value_Bootstrap', 'Significant_95', 'CI_Lower_95', 'CI_Upper_95']])
    
    print(f"\n🔍 FEATURE SIGNIFICANCE WITH P-VALUES:")
    for _, row in p_values_df.iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        sig_marker = "***" if row['P_Value_Bootstrap'] < 0.001 else "**" if row['P_Value_Bootstrap'] < 0.01 else "*" if row['P_Value_Bootstrap'] < 0.05 else ""
        print(f"  {row['Feature']}: {direction} payment probability (p = {row['P_Value_Bootstrap']:.4f}) {sig_marker}")
    
    # Save results with p-values
    os.makedirs('payment_prediction_min_variables', exist_ok=True)
    threshold_df.to_csv('payment_prediction_min_variables/payment_prediction_confusion_matrix_analysis.csv', index=False)
    feature_importance.to_csv('payment_prediction_min_variables/payment_prediction_classification_importance.csv', index=False)
    p_values_df.to_csv('payment_prediction_min_variables/payment_prediction_pvalues_bootstrap.csv', index=False)
    
    print(f"\n✓ Results saved to:")
    print(f"  - payment_prediction_confusion_matrix_analysis.csv")
    print(f"  - payment_prediction_classification_importance.csv")
    print(f"  - payment_prediction_pvalues_bootstrap.csv (NEW: includes raw p-values)")
    
    # Combine with base results
    enhanced_results = base_results.copy()
    enhanced_results.update({
        'confusion_matrix_analysis': threshold_df,
        'optimal_threshold': best_threshold['threshold'],
        'p_values_analysis': p_values_df,
        'bootstrap_coefficients': bootstrap_coefs,
        'optimal_metrics': {
            'f1_score': best_threshold['test_f1_score'],
            'accuracy': best_threshold['test_accuracy'],
            'precision': best_threshold['test_precision'],
            'recall': best_threshold['test_recall']
        },
        'roc_auc': auc_score,
        'classification_feature_importance': feature_importance,
        'test_probabilities': y_test_prob,
        'test_actual': y_test.values
    })
    
    print(f"\n✓ Confusion matrix analysis saved to: payment_prediction_confusion_matrix_analysis.csv")
    print(f"✓ Classification feature importance saved to: payment_prediction_classification_importance.csv")
    
    return enhanced_results


def _prepare_and_train_payment_model(
    payment_df,
    season_records,
    use_projection=False,
    alpha_range=None,
    random_seed=42,
    dataset_name="Dataset"
):
    """
    Prepares dataset and trains logistic regression model for payment prediction.
    
    Shared helper function for payment prediction models.
    
    Args:
        payment_df: QB payment data with era-adjusted (and optionally projected) stats
        season_records: Season records DataFrame for team metrics
        use_projection: If True, use projected stats (_proj columns), else use era-adjusted (_adj)
        alpha_range: Alpha values for regularization (converted to C)
        random_seed: Random seed for train/test split
        dataset_name: Name for logging/output
        
    Returns:
        dict: Model results with metrics, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MODEL: {dataset_name}")
    print('='*80)
    
    # Merge with team metrics
    payment_df = payment_df.copy()
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
    
    # Create adjusted Pts
    if os.path.exists('era_adjustment_factors.csv'):
        factor_df = pd.read_csv('era_adjustment_factors.csv')
        pts_factors = factor_df[factor_df['stat'] == 'Pts'].set_index('year')['adjustment_factor'].to_dict()
        merged_df['Pts_adj'] = merged_df['Pts'] * merged_df['season'].map(pts_factors)
    else:
        merged_df['Pts_adj'] = merged_df['Pts']
    
    # Define metrics - use projected versions if use_projection=True
    if use_projection:
        qb_metrics = [
            'total_yards_adj_proj',
            'Pass_TD_adj_proj',
            'Pass_ANY/A_adj',  # Rate stat, no projection
            'Rush_Rushing_Succ%_adj'  # Rate stat, no projection
        ]
    else:
        qb_metrics = [
            'total_yards_adj',
            'Pass_TD_adj',
            'Pass_ANY/A_adj',
            'Rush_Rushing_Succ%_adj'
        ]
    
    team_metrics = ['W-L%', 'Pts_adj']
    all_metrics = qb_metrics + team_metrics
    
    # Create lags and averages
    merged_df = merged_df.sort_values(['player_id', 'season'])
    
    for metric in all_metrics:
        if metric not in merged_df.columns:
            continue
        merged_df[metric] = pd.to_numeric(merged_df[metric], errors='coerce')
        for lag in [1, 2, 3]:
            lag_col = f"{metric}_lag{lag}"
            merged_df[lag_col] = merged_df.groupby('player_id')[metric].shift(lag)
    
    # Create averaged features
    features = []
    for metric in all_metrics:
        if metric not in merged_df.columns:
            continue
        avg_col = f"{metric}_avg"
        lag_cols = [f"{metric}_lag1", f"{metric}_lag2", f"{metric}_lag3"]
        if all(col in merged_df.columns for col in lag_cols):
            # Average available lags (skipna=True), but if all are NaN (first season), use current value
            merged_df[avg_col] = merged_df[lag_cols].mean(axis=1, skipna=True)
            # For first seasons (all lags NaN), use the current season's value
            first_season_mask = merged_df[lag_cols].isna().all(axis=1)
            merged_df.loc[first_season_mask, avg_col] = merged_df.loc[first_season_mask, metric]
            features.append(avg_col)
    
    # Prepare features and target
    X = merged_df[features + ['got_paid', 'player_id', 'player_name', 'season', 'GS']].copy()
    
    # Debug: Show filtering steps
    print(f"\nFiltering analysis:")
    print(f"  Initial merged rows: {len(merged_df)}")
    
    # Check which features are missing
    missing_by_feature = {}
    for feat in features:
        missing_count = X[feat].isna().sum()
        if missing_count > 0:
            missing_by_feature[feat] = missing_count
    
    if missing_by_feature:
        print(f"  Missing values by feature:")
        for feat, count in sorted(missing_by_feature.items(), key=lambda x: x[1], reverse=True):
            print(f"    {feat}: {count} missing")
    
    X_before_dropna = len(X)
    
    # Identify which rows will be dropped and why
    rows_with_missing = X[X[features].isna().any(axis=1)]
    if len(rows_with_missing) > 0:
        print(f"\n  Rows that will be dropped: {len(rows_with_missing)}")
        
        # If using projection, specifically check projected columns
        if use_projection:
            proj_cols_in_features = [f for f in features if '_proj' in f]
            if proj_cols_in_features:
                for proj_feat in proj_cols_in_features:
                    missing_proj_rows = rows_with_missing[rows_with_missing[proj_feat].isna()]
                    if len(missing_proj_rows) > 0:
                        print(f"\n    Rows missing {proj_feat}: {len(missing_proj_rows)}")
                        # Show details
                        for idx, row in missing_proj_rows.head(10).iterrows():
                            gs_val = f"{row['GS']:.0f}" if pd.notna(row['GS']) else "NaN"
                            # Check if era-adjusted version exists
                            era_adj_col = proj_feat.replace('_proj', '')
                            era_adj_val = "exists" if era_adj_col in row.index and pd.notna(row[era_adj_col]) else "missing"
                            print(f"      {row.get('player_name', 'Unknown')}: Season={row['season']:.0f}, GS={gs_val}, {era_adj_col}={era_adj_val}")
        
        # Check for other missing features (non-projected)
        non_proj_features = [f for f in features if '_proj' not in f]
        for feat in non_proj_features:
            missing_non_proj = rows_with_missing[rows_with_missing[feat].isna()]
            if len(missing_non_proj) > 0:
                print(f"\n    Rows missing {feat}: {len(missing_non_proj)}")
                for idx, row in missing_non_proj.head(5).iterrows():
                    gs_val = f"{row['GS']:.0f}" if pd.notna(row['GS']) else "NaN"
                    print(f"      {row.get('player_name', 'Unknown')}: Season={row['season']:.0f}, GS={gs_val}")
    
    X = X.dropna(subset=features)
    X_after_dropna = len(X)
    rows_dropped = X_before_dropna - X_after_dropna
    
    print(f"\n  Summary:")
    print(f"    Rows before dropna: {X_before_dropna}")
    print(f"    Rows after dropna: {X_after_dropna}")
    print(f"    Rows dropped: {rows_dropped}")
    
    # Remove debug columns before proceeding
    X = X.drop(columns=['player_name', 'season', 'GS'], errors='ignore')
    
    if len(X) < 30:
        print(f"✗ ERROR: Insufficient complete cases ({len(X)})")
        return None
    
    y = X['got_paid'].astype(int)
    X_features = X[features]
    
    # Train/test split (80/20) with fixed random seed
    # Use same split method for both datasets to ensure fair comparison
    np.random.seed(random_seed)
    indices = np.arange(len(X_features))
    np.random.shuffle(indices)
    split_idx = int(len(X_features) * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X_features.iloc[train_indices]
    X_test = X_features.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    print(f"Complete cases: {len(X)}")
    print(f"Train: {len(X_train)} ({y_train.sum()} paid)")
    print(f"Test: {len(X_test)} ({y_test.sum()} paid)")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation to find best C
    if alpha_range is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    else:
        C_values = [1.0/alpha for alpha in alpha_range]
    
    best_score = -np.inf
    best_C = 1.0
    
    for C in C_values:
        model = LogisticRegression(penalty='l2', C=C, max_iter=1000, random_state=random_seed)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
    
    print(f"Best C: {best_C} (CV AUC: {best_score:.4f})")
    
    # Train final model
    final_model = LogisticRegression(penalty='l2', C=best_C, max_iter=1000, random_state=random_seed)
    final_model.fit(X_train_scaled, y_train)
    
    # Test performance
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = final_model.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_acc = np.mean(y_test == y_test_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTest Performance:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1,
        'best_C': best_C,
        'cv_auc': best_score,
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': features,
        'model': final_model,
        'scaler': scaler
    }


def compare_injury_projection_predictiveness(alpha_range=None, exclude_recent_drafts=True, random_seed=42):
    """
    Compares predictive power of era-adjusted vs injury-projected counting stats for QB contracts.
    
    Runs logistic regression models on both datasets and compares F1 score, accuracy, and other metrics
    to determine which approach is more predictive of QB free-agent contracts.
    
    Args:
        alpha_range: Alpha values for ridge regression (converted to C for logistic regression)
        exclude_recent_drafts: Whether to exclude QBs drafted after 2020
        random_seed: Random seed for train/test split (ensures identical splits)
        
    Returns:
        dict: Comparison results with metrics for both approaches
    """
    print("\n" + "="*80)
    print("INJURY PROJECTION PREDICTIVENESS COMPARISON")
    print("="*80)
    print("Comparing era-adjusted vs injury-projected counting stats")
    print("="*80)
    
    # Load era-adjusted data
    if not os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
        print("✗ ERROR: qb_seasons_payment_labeled_era_adjusted.csv not found")
        print("Run create_era_adjusted_payment_data() first")
        return None
    
    payment_df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
    print(f"✓ Loaded {len(payment_df)} seasons with era-adjusted and projected stats")
    
    # Verify projected columns exist
    proj_cols = [col for col in payment_df.columns if col.endswith('_proj')]
    if not proj_cols:
        print("⚠️  WARNING: No projected columns found in dataset")
        print("   Projected columns should already be in the master DataFrame")
        print("   Run create_era_adjusted_payment_data() to regenerate with projections")
    else:
        print(f"✓ Found {len(proj_cols)} projected columns: {proj_cols[:3]}...")
    
    # Filter to eligible QBs
    if exclude_recent_drafts:
        cutoff_year = 2020
        payment_df['draft_year'] = pd.to_numeric(payment_df['draft_year'], errors='coerce')
        payment_df = payment_df[payment_df['draft_year'] <= cutoff_year]
        print(f"✓ Filtered to QBs drafted ≤{cutoff_year}")
    
    # Load season records
    season_records = load_csv_safe("season_records.csv", "season records")
    if season_records is None:
        return None
    
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Run models on both datasets using shared helper function
    # Both use the same DataFrame - just different columns (_adj vs _adj_proj)
    baseline_results = _prepare_and_train_payment_model(
        payment_df.copy(),
        season_records,
        use_projection=False,
        alpha_range=alpha_range,
        random_seed=random_seed,
        dataset_name="BASELINE (Era-Adjusted)"
    )
    projected_results = _prepare_and_train_payment_model(
        payment_df.copy(),  # Same DataFrame, just uses different columns
        season_records,
        use_projection=True,
        alpha_range=alpha_range,
        random_seed=random_seed,
        dataset_name="PROJECTED (Era-Adjusted + Injury Projection)"
    )
    
    if baseline_results is None or projected_results is None:
        print("\n✗ ERROR: Failed to run one or both models")
        return None
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = {
        'Metric': ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC', 'Best C', 'CV AUC'],
        'Baseline (Era-Adjusted)': [
            baseline_results['test_f1_score'],
            baseline_results['test_accuracy'],
            baseline_results['test_precision'],
            baseline_results['test_recall'],
            baseline_results['test_auc'],
            baseline_results['best_C'],
            baseline_results['cv_auc']
        ],
        'Projected (Era-Adjusted + Projection)': [
            projected_results['test_f1_score'],
            projected_results['test_accuracy'],
            projected_results['test_precision'],
            projected_results['test_recall'],
            projected_results['test_auc'],
            projected_results['best_C'],
            projected_results['cv_auc']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Difference'] = comparison_df['Projected (Era-Adjusted + Projection)'] - comparison_df['Baseline (Era-Adjusted)']
    comparison_df['Improvement'] = comparison_df['Difference'] > 0
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Determine winner
    f1_improvement = projected_results['test_f1_score'] - baseline_results['test_f1_score']
    acc_improvement = projected_results['test_accuracy'] - baseline_results['test_accuracy']
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("="*80)
    
    if f1_improvement > 0:
        print(f"✓ INJURY PROJECTION IS MORE PREDICTIVE")
        print(f"  F1-Score improvement: +{f1_improvement:.4f}")
        print(f"  Accuracy improvement: {acc_improvement:+.4f}")
    elif f1_improvement < 0:
        print(f"✗ ERA-ADJUSTED (BASELINE) IS MORE PREDICTIVE")
        print(f"  F1-Score difference: {f1_improvement:.4f}")
        print(f"  Accuracy difference: {acc_improvement:+.4f}")
    else:
        print(f"≈ NO SIGNIFICANT DIFFERENCE")
        print(f"  F1-Score difference: {f1_improvement:.4f}")
    
    # Save results
    comparison_df.to_csv('injury_projection_predictiveness_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to: injury_projection_predictiveness_comparison.csv")
    
    return {
        'comparison': comparison_df,
        'baseline_results': baseline_results,
        'projected_results': projected_results,
        'winner': 'projected' if f1_improvement > 0 else ('baseline' if f1_improvement < 0 else 'tie'),
        'f1_improvement': f1_improvement,
        'accuracy_improvement': acc_improvement
    }

