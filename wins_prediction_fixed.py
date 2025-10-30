def wins_prediction_linear_ridge(train_df=None, test_df=None, alpha_range=None, exclude_recent_drafts=True, n_bootstrap=1000, qb_metrics=['total_yards_adj', 'Pass_TD_adj', 'Pass_ANY/A_adj', 'Rush_Rushing_Succ%_adj']):
    """
    Enhanced ridge regression for wins prediction with bootstrap-based statistical significance testing.
    
    Args:
        train_df: Training data (if provided, uses your specific split)
        test_df: Test data (if provided, uses your specific split)
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
    
    # Load season records (needed for both paths)
    season_records = load_csv_safe("season_records.csv", "season records")
    if season_records is None:
        return None
    
    season_records['Season'] = pd.to_numeric(season_records['Season'], errors='coerce')
    season_records = season_records.dropna(subset=['Season'])
    season_records['Season'] = season_records['Season'].astype(int)
    
    # Use provided train/test data or load from files
    if train_df is not None and test_df is not None:
        print("✓ Using provided train/test split")
        
        # Process train data
        X_train, y_train = process_data_for_wins(train_df, season_records, qb_metrics)
        X_test, y_test = process_data_for_wins(test_df, season_records, qb_metrics)
        
        if X_train is None or X_test is None:
            return None
            
        features = X_train.columns.tolist()
        
        print(f"\nUsing provided train/test split:")
        print(f"  Train: {len(X_train)} samples (mean wins: {y_train.mean():.1f})")
        print(f"  Test: {len(X_test)} samples (mean wins: {y_test.mean():.1f})")
        
    else:
        print("✓ Loading data and creating internal split")
        
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

        # Define metrics (same structure as payment prediction)
        team_metrics = [
            'W-L%',
            'Pts_adj'
        ]
        
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
        
        print(f"\nInternal train/test split:")
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

def process_data_for_wins(df, season_records, qb_metrics):
    """Helper function to process train/test data for wins prediction"""
    # Merge with season records
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df.dropna(subset=['season'])
    df['season'] = df['season'].astype(int)
    
    merged_df = pd.merge(
        df,
        season_records[['Season', 'Team', 'W-L%', 'Pts', 'W']],
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
    
    # Define metrics
    team_metrics = ['W-L%', 'Pts_adj']
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
    X = merged_df[features].dropna()
    y = pd.to_numeric(merged_df.loc[X.index, 'W'], errors='coerce').dropna()
    X = X.loc[y.index]
    
    if len(X) < 10:
        print(f"⚠️ Warning: Only {len(X)} complete cases in this split")
        return None, None
    
    return X, y