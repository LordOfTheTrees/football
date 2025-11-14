"""
Debug and utility functions for QB research.

This module contains functions for:
- Fixing individual QB data files
- Standardizing QB column names
- Debugging specific QBs through the pipeline
"""

import os
import pandas as pd

from qb_research.utils.data_loading import load_csv_safe


def fix_individual_qb_files():
    """
    Adds missing draft_year and draft_team columns to individual QB files.
    """
    print("="*80)
    print("FIXING INDIVIDUAL QB FILES - ADDING MISSING METADATA")
    print("="*80)
    
    # Define the missing metadata for each QB
    qb_metadata = {
        'AlleJo02': {
            'draft_year': 2018,
            'draft_team': 'BUF',
            'pick_number': 7
        },
        'FielJu00': {
            'draft_year': 2021, 
            'draft_team': 'CHI',
            'pick_number': 11
        }
        # Add more QBs here as needed:
        # 'BurrJo01': {
        #     'draft_year': 2020,
        #     'draft_team': 'CIN', 
        #     'pick_number': 1
        # }
    }
    
    qb_data_dir = 'QB_Data'
    if not os.path.exists(qb_data_dir):
        print(f"✗ ERROR: {qb_data_dir} directory not found")
        return
    
    fixed_count = 0
    
    for player_id, metadata in qb_metadata.items():
        file_path = f"{qb_data_dir}/{player_id}.csv"
        
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} not found, skipping")
            continue
        
        print(f"\n📝 Fixing {player_id}...")
        
        # Load the QB file
        df = pd.read_csv(file_path)
        print(f"   Loaded {len(df)} records")
        
        # Check if columns already exist
        missing_cols = []
        for col in ['draft_year', 'draft_team', 'pick_number']:
            if col not in df.columns:
                missing_cols.append(col)
        
        if not missing_cols:
            print(f"   ✓ All metadata columns already exist")
            continue
        
        print(f"   Adding missing columns: {missing_cols}")
        
        # Add the missing metadata columns
        for col, value in metadata.items():
            if col in missing_cols:
                df[col] = value
                print(f"     + {col} = {value}")
        
        # Save the updated file
        backup_path = f"{file_path}.backup"
        if not os.path.exists(backup_path):
            df_original = pd.read_csv(file_path)
            df_original.to_csv(backup_path, index=False)
            print(f"   💾 Created backup: {backup_path}")
        
        df.to_csv(file_path, index=False)
        print(f"   ✅ Updated {file_path}")
        fixed_count += 1
    
    print(f"\n" + "="*80)
    print(f"SUMMARY: Fixed {fixed_count} QB files")
    print("="*80)
    
    if fixed_count > 0:
        print("✅ Individual QB files now have required metadata columns")
        print("🔄 Run the full pipeline rebuild script next")
    else:
        print("ℹ️  No files needed fixing")


def standardize_qb_columns(df):
    """Standardizes column names across QB datasets"""
    
    # Handle numbered unnamed columns
    df.columns = [col.replace('Rush_Unnamed: 32_level_0_Fmb', 'Rush_Fumbles') 
                  for col in df.columns]
    df.columns = [col.replace('Rush_Unnamed: 33_level_0_Awards', 'Rush_Awards') 
                  for col in df.columns]
    
    # Ensure required columns exist
    required_cols = ['draft_year', 'draft_team', 'pick_number', 'player_name', 'player_id']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            
    return df


def debug_specific_qb(qb_name, data_stage='all'):
    """
    Debug a specific QB through the pipeline.
    
    Args:
        qb_name (str): QB name to search for (case-insensitive, partial match)
        data_stage (str): Which stage to check - 'contracts', 'payment', 'prepared', 'era', 'export', 'all'
    """
    print("\n" + "="*80)
    print(f"🔍 DEBUGGING: {qb_name}")
    print("="*80)
    
    stages_to_check = ['contracts', 'payment', 'prepared', 'era', 'export'] if data_stage == 'all' else [data_stage]
    
    for stage in stages_to_check:
        print(f"\n--- Stage: {stage.upper()} ---")
        
        if stage == 'contracts':
            # Check mapped contracts
            if os.path.exists('cache/contract_player_id_mapping.csv'):
                df = load_csv_safe('cache/contract_player_id_mapping.csv')
                if df is not None:
                    qb_data = df[df['Player'].str.contains(qb_name, case=False, na=False)]
                    if len(qb_data) > 0:
                        print(f"  ✅ Found {len(qb_data)} contracts")
                        for _, row in qb_data.iterrows():
                            print(f"    {row['Year']}: {row['Player']} → player_id: {row.get('player_id', 'NO_ID')}, Team: {row['Team']}")
                    else:
                        print(f"  ❌ Not found in contracts")
            else:
                print(f"  ⚠️  Contract mapping file not found")
        
        elif stage == 'payment':
            # Check payment mapping (this is a dict, need to reconstruct)
            print(f"  ℹ️  Payment mapping is in-memory dict - run full pipeline to check")
        
        elif stage == 'prepared':
            # Check prepared payment data
            if os.path.exists('qb_seasons_payment_labeled.csv'):
                df = load_csv_safe('qb_seasons_payment_labeled.csv')
                if df is not None:
                    qb_data = df[df['player_name'].str.contains(qb_name, case=False, na=False)]
                    if len(qb_data) > 0:
                        print(f"  ✅ Found {len(qb_data)} seasons")
                        sample = qb_data.iloc[0]
                        print(f"    player_id: {sample['player_id']}")
                        print(f"    draft_year: {sample['draft_year']}")
                        print(f"    got_paid: {sample['got_paid']}")
                        print(f"    payment_year: {sample['payment_year']}")
                    else:
                        print(f"  ❌ Not found in prepared data")
            else:
                print(f"  ⚠️  Prepared data file not found")
        
        elif stage == 'era':
            # Check era-adjusted data
            if os.path.exists('qb_seasons_payment_labeled_era_adjusted.csv'):
                df = load_csv_safe('qb_seasons_payment_labeled_era_adjusted.csv')
                if df is not None:
                    qb_data = df[df['player_name'].str.contains(qb_name, case=False, na=False)]
                    if len(qb_data) > 0:
                        print(f"  ✅ Found {len(qb_data)} seasons")
                        print(f"    got_paid: {qb_data.iloc[0]['got_paid']}")
                        print(f"    payment_year: {qb_data.iloc[0]['payment_year']}")
                    else:
                        print(f"  ❌ Not found in era-adjusted data")
            else:
                print(f"  ⚠️  Era-adjusted data file not found")
        
        elif stage == 'export':
            # Check Tableau export
            if os.path.exists('qb_trajectories_for_tableau.csv'):
                df = load_csv_safe('qb_trajectories_for_tableau.csv')
                if df is not None:
                    qb_data = df[df['player_name'].str.contains(qb_name, case=False, na=False)]
                    if len(qb_data) > 0:
                        print(f"  ✅ Found {len(qb_data)} seasons in export")
                        print(f"    Years range: {int(qb_data['years_since_draft'].min())} to {int(qb_data['years_since_draft'].max())}")
                    else:
                        print(f"  ❌ Not found in export")
            else:
                print(f"  ⚠️  Export file not found")
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

