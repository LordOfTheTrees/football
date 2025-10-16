import pandas as pd
import os
import glob
from datetime import datetime

def rebuild_all_seasons_df():
    """
    Rebuilds all_seasons_df.csv from individual QB files in QB_Data folder.
    
    This is the critical step that was missing Josh Allen and Justin Fields.
    """
    print("="*80)
    print("REBUILDING all_seasons_df.csv FROM INDIVIDUAL QB FILES")
    print("="*80)
    
    start_time = datetime.now()
    
    # Step 1: Find all QB CSV files
    qb_data_dir = 'QB_Data'
    if not os.path.exists(qb_data_dir):
        print(f"✗ ERROR: {qb_data_dir} directory not found")
        print("Make sure QB_Data folder exists with individual QB CSV files")
        return False
    
    qb_files = glob.glob(f"{qb_data_dir}/*.csv")
    print(f"Found {len(qb_files)} QB files in {qb_data_dir}/")
    
    if len(qb_files) == 0:
        print(f"✗ ERROR: No CSV files found in {qb_data_dir}/")
        return False
    
    # Step 2: Load and validate each QB file
    all_seasons_data = []
    successful_files = 0
    failed_files = []
    missing_metadata_files = []
    
    print(f"\nProcessing individual QB files...")
    
    for file_path in qb_files:
        filename = os.path.basename(file_path)
        
        # Skip hidden files
        if filename.startswith('.'):
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_cols = ['player_name', 'player_id', 'season']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                failed_files.append((filename, f"Missing columns: {missing_cols}"))
                continue
            
            # Check for draft metadata (recommended but not required)
            metadata_cols = ['draft_year', 'draft_team']
            missing_metadata = [col for col in metadata_cols if col not in df.columns]
            
            if missing_metadata:
                missing_metadata_files.append((filename, missing_metadata))
            
            # Add the data
            all_seasons_data.append(df)
            successful_files += 1
            
            print(f"  ✓ {filename}: {len(df)} seasons")
            
            # Show sample for key QBs
            if 'AlleJo02' in filename:
                print(f"    → Josh Allen: {df['season'].min()}-{df['season'].max()}")
                if 'draft_year' in df.columns:
                    print(f"      Draft info: {df['draft_year'].iloc[0]} {df['draft_team'].iloc[0]}")
                    
            elif 'FielJu00' in filename:
                print(f"    → Justin Fields: {df['season'].min()}-{df['season'].max()}")
                if 'draft_year' in df.columns:
                    print(f"      Draft info: {df['draft_year'].iloc[0]} {df['draft_team'].iloc[0]}")
            
        except Exception as e:
            failed_files.append((filename, str(e)))
    
    # Step 3: Report on file processing
    print(f"\n" + "="*60)
    print("FILE PROCESSING SUMMARY")
    print("="*60)
    
    print(f"✅ Successful: {successful_files} files")
    print(f"❌ Failed: {len(failed_files)} files")
    print(f"⚠️  Missing metadata: {len(missing_metadata_files)} files")
    
    if failed_files:
        print(f"\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    if missing_metadata_files:
        print(f"\nFiles missing draft metadata (still included):")
        for filename, missing in missing_metadata_files:
            print(f"  - {filename}: missing {missing}")
    
    if successful_files == 0:
        print("✗ ERROR: No valid QB files processed")
        return False
    
    # Step 4: Combine all QB data
    print(f"\n" + "="*60)
    print("COMBINING QB DATA")
    print("="*60)
    
    combined_df = pd.concat(all_seasons_data, ignore_index=True)
    
    print(f"Combined dataset:")
    print(f"  Total records: {len(combined_df):,}")
    print(f"  Unique QBs: {combined_df['player_id'].nunique()}")
    print(f"  Season range: {combined_df['season'].min()}-{combined_df['season'].max()}")
    
    # Step 5: Verify key QBs are included
    print(f"\n" + "="*60)
    print("VERIFYING KEY QBs")
    print("="*60)
    
    key_qbs = {
        'AlleJo02': 'Josh Allen',
        'FielJu00': 'Justin Fields'
    }
    
    for player_id, name in key_qbs.items():
        qb_data = combined_df[combined_df['player_id'] == player_id]
        if len(qb_data) > 0:
            seasons = f"{qb_data['season'].min()}-{qb_data['season'].max()}"
            print(f"  ✅ {name} ({player_id}): {len(qb_data)} seasons ({seasons})")
        else:
            print(f"  ❌ {name} ({player_id}): NOT FOUND")
    
    # Step 6: Save the combined file
    output_file = 'all_seasons_df.csv'
    
    # Backup existing file if it exists
    if os.path.exists(output_file):
        backup_file = f"{output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(output_file, backup_file)
        print(f"\n💾 Backed up existing file to: {backup_file}")
    
    combined_df.to_csv(output_file, index=False)
    
    # Step 7: Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print("REBUILD COMPLETE")
    print("="*80)
    
    print(f"✅ Created: {output_file}")
    print(f"📊 Records: {len(combined_df):,}")
    print(f"👥 QBs: {combined_df['player_id'].nunique()}")
    print(f"⏱️  Duration: {duration}")
    
    file_size = os.path.getsize(output_file) / (1024*1024)
    print(f"💾 File size: {file_size:.1f} MB")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Run prepare_qb_payment_data() to add payment labels")
    print(f"  2. Run create_era_adjusted_payment_data() for era adjustments")
    print(f"  3. Run export_individual_qb_trajectories() for Tableau")
    
    return True

def quick_check_missing_qbs():
    """
    Quick check to see if the key missing QBs are now included.
    """
    print("="*60)
    print("QUICK CHECK: KEY QBs IN all_seasons_df.csv")
    print("="*60)
    
    if not os.path.exists('all_seasons_df.csv'):
        print("❌ all_seasons_df.csv not found - run rebuild_all_seasons_df() first")
        return
    
    df = pd.read_csv('all_seasons_df.csv')
    
    key_qbs = {
        'AlleJo02': 'Josh Allen',
        'FielJu00': 'Justin Fields',
        'BurrJo01': 'Joe Burrow',
        'HerbJu00': 'Justin Herbert',
        'LawrTr00': 'Trevor Lawrence'
    }
    
    for player_id, name in key_qbs.items():
        qb_data = df[df['player_id'] == player_id]
        if len(qb_data) > 0:
            seasons = f"{qb_data['season'].min()}-{qb_data['season'].max()}"
            has_draft = 'draft_year' in qb_data.columns and qb_data['draft_year'].notna().any()
            draft_info = ""
            if has_draft:
                draft_year = qb_data['draft_year'].iloc[0]
                draft_team = qb_data['draft_team'].iloc[0] if 'draft_team' in qb_data.columns else 'Unknown'
                draft_info = f" (Drafted {int(draft_year)} {draft_team})"
            
            print(f"  ✅ {name}: {len(qb_data)} seasons ({seasons}){draft_info}")
        else:
            print(f"  ❌ {name}: NOT FOUND")

if __name__ == "__main__":
    print("🚀 QB DATA PIPELINE: REBUILD all_seasons_df.csv")
    print("="*80)
    
    success = rebuild_all_seasons_df()
    
    if success:
        print("\n🎉 SUCCESS: all_seasons_df.csv rebuilt successfully!")
        print("\nRunning quick verification...")
        quick_check_missing_qbs()
    else:
        print("\n💥 FAILED: Could not rebuild all_seasons_df.csv")
        print("Check the error messages above and fix the issues")
