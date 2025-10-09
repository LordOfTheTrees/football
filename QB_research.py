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

def bestqbseasons():
    try:
        if os.path.exists("best_seasons_df.csv"):
            print("starting best QB seasons pull")
            df_QB_best_seasons = pd.read_csv("best_seasons_df.csv")
            #print( df_QB_best_seasons.shape, "columns: ", df_QB_best_seasons.columns)
            df_QB_best_seasons.sort_values(by=['total_yards'], ascending=[False], inplace=True)
            display(df_QB_best_seasons.head(5)[['player_name', 'draft_team', 'season', 'total_yards', 'AdvPass_Air Yards_IAY/PA', 'Pass_Cmp%', 'Pass_Rate', 'Pass_Yds', 'Rush_Rushing_Yds', 'Rush_Rushing_TD']])
    except Exception as e:
        print(f"Error running the best QB seasons pull: {e}")

def best_season_averages():
    try:
        if os.path.exists("season_averages.csv"):
            print(f"\nstarting best season averages pull")
            df_season_averages = pd.read_csv("season_averages.csv")
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
            df_QB_contract_data = pd.read_csv("QB_contract_data.csv")
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
            df_season_records = pd.read_csv("season_records.csv")
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
    try:
        qb_seasons = pd.read_csv("all_seasons_df.csv")
        season_records = pd.read_csv("season_records.csv")
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
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
    
    # Create comprehensive team mapping (PFR uses different codes in different places)
    team_mapping = {
        # Map season_records codes TO QB data codes
        'NWE': 'NWE', 'NYJ': 'NYJ', 'MIA': 'MIA', 'BUF': 'BUF',
        'PIT': 'PIT', 'BAL': 'RAV', 'CLE': 'CLE', 'CIN': 'CIN',
        'IND': 'CLT', 'HOU': 'HTX', 'TEN': 'OTI', 'JAX': 'JAX',
        'KAN': 'KAN', 'DEN': 'DEN', 'LAC': 'SDG', 'OAK': 'RAI', 'LVR': 'RAI',
        'PHI': 'PHI', 'DAL': 'DAL', 'NYG': 'NYG', 'WAS': 'WAS',
        'GNB': 'GNB', 'MIN': 'MIN', 'CHI': 'CHI', 'DET': 'DET',
        'TAM': 'TAM', 'NOR': 'NOR', 'ATL': 'ATL', 'CAR': 'CAR',
        'SEA': 'SEA', 'SFO': 'SFO', 'LAR': 'RAM', 'ARI': 'CRD',
        'STL': 'STL', 'RAM': 'RAM',
    }
    
    # Actually, let's check what team codes each dataset is using
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
    
    # Build mapping based on what we see
    # For now, let's try matching on the overlapping teams first
    
    # Merge QB data with season records on overlapping teams
    merged_df = pd.merge(
        qb_seasons,
        season_records,
        left_on=['season', 'Team'],
        right_on=['Season', 'Team'],
        how='inner'
    )
    
    print(f"\nSuccessfully merged {len(merged_df)} QB season-team records")
    
    if merged_df.empty or len(merged_df) < 50:
        print("\nInsufficient matching records. Will need to create team mapping.")
        print("Please review the team codes above and update the mapping.")
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
    
    # Load the data
    try:
        qb_seasons = pd.read_csv("all_seasons_df.csv")
        season_records = pd.read_csv("season_records.csv")
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
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
        all_seasons = pd.read_csv("all_seasons_df.csv")
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
        train_df = pd.read_csv("all_seasons_df_train.csv")
        test_df = pd.read_csv("all_seasons_df_test.csv")
        
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
        season_records = pd.read_csv("season_records.csv")
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
    
    # Define PC1 top factors
    pc1_factors = [
        'Pass_Yds',
        'total_yards',
        'Pass_TD',
        'Pass_1D',
        'Pass_AV',
        'Pass_ANY/A',
        'Pass_Rate',
        'Pass_Succ%',
        'Pass_AY/A',
        'Pass_NY/A'
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
    
    # Build training dataset
    train_features = train_merged[available_factors + ['Wins']].copy()
    for col in available_factors:
        train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
    
    initial_train = len(train_features)
    train_features = train_features.dropna()
    final_train = len(train_features)
    
    print(f"\nTraining set complete cases: {final_train}/{initial_train} ({final_train/initial_train*100:.1f}%)")
    
    # Build test dataset
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
    # MULTICOLLINEARITY CHECK (on training data)
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
            if col2 not in cols_to_remove:
                cols_to_remove.add(col2)
        
        available_factors = [col for col in available_factors if col not in cols_to_remove]
        print(f"\nReduced to {len(available_factors)} factors:")
        for factor in available_factors:
            print(f"  - {factor}")
        
        # Rebuild datasets with reduced features
        train_features = train_merged[available_factors + ['Wins']].copy()
        for col in available_factors:
            train_features[col] = pd.to_numeric(train_features[col], errors='coerce')
        train_features = train_features.dropna()
        
        test_features = test_merged[available_factors + ['Wins']].copy()
        for col in available_factors:
            test_features[col] = pd.to_numeric(test_features[col], errors='coerce')
        test_features = test_features.dropna()
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
    train_df, test_df = create_train_test_split(test_size=0.2, split_by='temporal')
    regression_with_pc1_factors(train_df, test_df)