#!/usr/bin/env python3
"""
QB Variable Scatter Plot Generator

Generates scatter plots for every numeric QB variable where:
- X-axis: Years since draft (Y-0, Y-1, Y-2, etc.)
- Y-axis: Value of the variable (in original units)
- Each point: One QB season
- All QBs on same chart for comparison

Outputs all plots to scatter_plots/ folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def setup_output_directory():
    """Create scatter_plots directory if it doesn't exist"""
    output_dir = Path('scatter_plots')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def build_from_qb_data_folder():
    """Build combined dataset from individual QB files in QB_Data/"""
    import glob
    
    print("\nAttempting to build from QB_Data/ folder...")
    
    if not os.path.exists('QB_Data'):
        return None
    
    qb_files = glob.glob('QB_Data/*.csv')
    if not qb_files:
        return None
    
    print(f"Found {len(qb_files)} QB files")
    
    all_seasons = []
    for file_path in qb_files:
        try:
            df = pd.read_csv(file_path)
            player_id = os.path.basename(file_path).replace('.csv', '')
            
            if 'player_id' not in df.columns:
                df['player_id'] = player_id
            
            all_seasons.append(df)
        except Exception as e:
            print(f"  Warning: Skipped {file_path}: {e}")
    
    if not all_seasons:
        return None
    
    combined_df = pd.concat(all_seasons, ignore_index=True)
    print(f"Combined {len(combined_df)} total QB seasons")
    return combined_df


def load_qb_data():
    """Load QB season data with proper handling of different file locations"""
    
    possible_files = [
        'qb_seasons_payment_labeled_era_adjusted.csv',
        'all_seasons_df.csv',
        'QB_Data/all_seasons_df.csv',
        'qb_trajectories_for_tableau.csv'
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} QB seasons")
            return df
    
    # Try building from QB_Data folder
    df = build_from_qb_data_folder()
    if df is not None:
        return df
    
    raise FileNotFoundError(
        "Could not find QB season data. Tried:\n" + 
        "\n".join(f"  - {f}" for f in possible_files) +
        "\n  - Building from QB_Data/ folder"
    )


def calculate_years_since_draft(df):
    """
    Calculate years since draft for each season.
    
    Handles both pre-calculated and raw data.
    """
    
    if 'years_since_draft' in df.columns:
        print("Using existing 'years_since_draft' column")
        return df
    
    # Calculate from season and draft_year
    if 'season' in df.columns and 'draft_year' in df.columns:
        print("Calculating years_since_draft from season and draft_year")
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
        df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
        df['years_since_draft'] = df['season'] - df['draft_year']
        return df
    
    raise ValueError(
        "Cannot calculate years since draft. Need either 'years_since_draft' "
        "or both 'season' and 'draft_year' columns"
    )


def identify_plottable_variables(df):
    """
    Identify numeric variables suitable for plotting.
    
    Excludes:
    - ID/categorical columns
    - Columns with too many missing values
    - Year/season identifiers
    - Lag variables (we want current season only)
    """
    
    # Exclude these patterns
    exclude_patterns = [
        'player_id', 'player_name', 'draft_year', 'draft_team',
        'season', 'years_since_draft', 'Team', 'Lg', 'Pos',
        '_lag', 'Unnamed', 'Awards', 'QBrec', 'payment_year',
        'years_to_payment', 'got_paid', 'pick_number'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plottable_vars = []
    
    for col in numeric_cols:
        # Skip if matches exclude pattern
        if any(pattern in col for pattern in exclude_patterns):
            continue
        
        # Skip if too many missing values (>50%)
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > 0.5:
            print(f"Skipping {col}: {missing_pct*100:.1f}% missing")
            continue
        
        # Skip if all zeros or constant
        if df[col].std() == 0:
            print(f"Skipping {col}: no variation")
            continue
        
        plottable_vars.append(col)
    
    return plottable_vars


def create_scatter_plot(df, variable, output_dir):
    """
    Create scatter plot for a single variable.
    
    Args:
        df: DataFrame with QB data
        variable: Column name to plot
        output_dir: Directory to save plot
    """
    
    # Filter to valid data
    plot_data = df[['years_since_draft', variable]].dropna()
    
    if len(plot_data) == 0:
        print(f"  Skipping {variable}: no valid data")
        return False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create scatter plot
    ax.scatter(
        plot_data['years_since_draft'],
        plot_data[variable],
        alpha=0.4,
        s=30,
        c='steelblue',
        edgecolors='none'
    )
    
    # Add trend line
    try:
        z = np.polyfit(plot_data['years_since_draft'], plot_data[variable], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(
            plot_data['years_since_draft'].min(),
            plot_data['years_since_draft'].max(),
            100
        )
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
    except:
        pass  # Skip trend line if it fails
    
    # Formatting
    ax.set_xlabel('Years Since Draft', fontsize=12, fontweight='bold')
    ax.set_ylabel(variable, fontsize=12, fontweight='bold')
    ax.set_title(f'{variable} by Career Year (All QBs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add stats text box
    stats_text = f'N = {len(plot_data):,} seasons\n'
    stats_text += f'Mean = {plot_data[variable].mean():.2f}\n'
    stats_text += f'Median = {plot_data[variable].median():.2f}\n'
    stats_text += f'Std = {plot_data[variable].std():.2f}'
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save with clean filename
    clean_var_name = variable.replace('/', '_').replace('%', 'pct').replace(' ', '_')
    output_file = output_dir / f'{clean_var_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    """Main execution function"""
    
    print("="*80)
    print("QB VARIABLE SCATTER PLOT GENERATOR")
    print("="*80)
    
    # Setup
    output_dir = setup_output_directory()
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Load data
    print("\n" + "-"*80)
    df = load_qb_data()
    
    # Calculate years since draft
    print("\n" + "-"*80)
    df = calculate_years_since_draft(df)
    print(f"Years since draft range: {df['years_since_draft'].min():.0f} to {df['years_since_draft'].max():.0f}")
    
    # Identify variables to plot
    print("\n" + "-"*80)
    print("Identifying plottable variables...")
    variables = identify_plottable_variables(df)
    print(f"\nFound {len(variables)} variables to plot")
    
    # Generate plots
    print("\n" + "-"*80)
    print("Generating scatter plots...")
    print("-"*80)
    
    success_count = 0
    fail_count = 0
    
    for i, var in enumerate(variables, 1):
        print(f"[{i}/{len(variables)}] {var}...", end=' ')
        
        try:
            if create_scatter_plot(df, var, output_dir):
                success_count += 1
                print("✓")
            else:
                fail_count += 1
                print("✗")
        except Exception as e:
            fail_count += 1
            print(f"✗ Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SCATTER PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"Successfully created: {success_count} plots")
    print(f"Failed: {fail_count} plots")
    print(f"\nAll plots saved to: {output_dir.absolute()}/")
    print("="*80)


if __name__ == "__main__":
    main()