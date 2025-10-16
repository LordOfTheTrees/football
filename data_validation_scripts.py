import pandas as pd
import os

def check_qb_coverage():
    """
    Checks if the QB trajectory CSV contains all QBs from the QB_Data folder.
    FIXED: Now handles missing draft_year and draft_team values properly.
    """
    print("\n" + "="*80)
    print("CHECKING QB COVERAGE IN TRAJECTORY DATA")
    print("="*80)
    
    # 1. Load trajectory export data
    trajectory_file = 'qb_trajectories_for_tableau.csv'
    if not os.path.exists(trajectory_file):
        print(f"✗ ERROR: {trajectory_file} not found")
        print("   Run export_individual_qb_trajectories() first")
        return None
    
    trajectory_df = pd.read_csv(trajectory_file)
    print(f"✓ Loaded trajectory data: {len(trajectory_df)} records")
    
    # 2. Load complete QB dataset
    complete_qb_file = 'all_seasons_df.csv'
    if not os.path.exists(complete_qb_file):
        complete_qb_file = 'QB_Data/all_seasons_df.csv'
    
    if not os.path.exists(complete_qb_file):
        print(f"✗ ERROR: all_seasons_df.csv not found in current directory or QB_Data/")
        return None
    
    complete_df = pd.read_csv(complete_qb_file)
    print(f"✓ Loaded complete QB data: {len(complete_df)} records")
    
    # 3. Compare QB coverage
    print(f"\n" + "="*80)
    print("QB COVERAGE COMPARISON")
    print("="*80)
    
    # Get unique QBs from each dataset
    trajectory_qbs = set(trajectory_df['player_id'].unique())
    complete_qbs = set(complete_df['player_id'].unique())
    
    print(f"\nUnique QBs in trajectory data: {len(trajectory_qbs)}")
    print(f"Unique QBs in complete dataset: {len(complete_qbs)}")
    
    # Find missing QBs
    missing_qbs = complete_qbs - trajectory_qbs
    extra_qbs = trajectory_qbs - complete_qbs
    
    print(f"\n📊 COVERAGE ANALYSIS:")
    print(f"   QBs in both datasets: {len(trajectory_qbs & complete_qbs)}")
    print(f"   Missing from trajectory: {len(missing_qbs)}")
    print(f"   Extra in trajectory: {len(extra_qbs)}")
    
    coverage_pct = len(trajectory_qbs & complete_qbs) / len(complete_qbs) * 100
    print(f"   Coverage percentage: {coverage_pct:.1f}%")
    
    # 4. Show missing QBs (if any)
    if missing_qbs:
        print(f"\n❌ MISSING QBs ({len(missing_qbs)}):")
        
        # Get details for missing QBs
        missing_details = complete_df[complete_df['player_id'].isin(missing_qbs)].groupby('player_id').agg({
            'player_name': 'first',
            'draft_year': 'first',
            'draft_team': 'first',
            'season': ['min', 'max', 'count']
        })
        
        missing_details.columns = ['player_name', 'draft_year', 'draft_team', 'first_season', 'last_season', 'total_seasons']
        missing_details = missing_details.sort_values('draft_year', ascending=False, na_position='last')
        
        for player_id, row in missing_details.head(20).iterrows():
            # Handle missing values safely
            draft_year = 'Unknown' if pd.isna(row['draft_year']) else str(int(row['draft_year']))
            draft_team = 'Unknown' if pd.isna(row['draft_team']) else str(row['draft_team'])
            
            print(f"   {row['player_name']}: Drafted {draft_year} ({draft_team}), "
                  f"{int(row['first_season'])}-{int(row['last_season'])} ({int(row['total_seasons'])} seasons)")
        
        if len(missing_details) > 20:
            print(f"   ... and {len(missing_details) - 20} more")
            
        # Show breakdown of missing QBs by reason
        print(f"\n🔍 MISSING QB ANALYSIS:")
        missing_with_draft = missing_details[missing_details['draft_year'].notna()]
        missing_no_draft = missing_details[missing_details['draft_year'].isna()]
        
        print(f"   Missing QBs with known draft year: {len(missing_with_draft)}")
        print(f"   Missing QBs with unknown draft year: {len(missing_no_draft)}")
        
        if len(missing_no_draft) > 0:
            print(f"   QBs without draft info (likely undrafted/practice squad):")
            for player_id, row in missing_no_draft.head(5).iterrows():
                print(f"     - {row['player_name']}: {int(row['first_season'])}-{int(row['last_season'])}")
    else:
        print(f"\n✅ ALL QBs from complete dataset are included in trajectory data!")
    
    # 5. Check draft year coverage (only for QBs with known draft years)
    print(f"\n" + "="*80)
    print("DRAFT YEAR COVERAGE (First-Round QBs Only)")
    print("="*80)
    
    # Filter to QBs with known draft years
    complete_with_draft = complete_df[complete_df['draft_year'].notna()]
    trajectory_with_draft = trajectory_df[trajectory_df['draft_year'].notna()]
    
    complete_years = complete_with_draft.groupby('draft_year')['player_id'].nunique().sort_index()
    trajectory_years = trajectory_with_draft.groupby('draft_year')['player_id'].nunique().sort_index()
    
    print(f"\nFirst-round QBs by draft year:")
    print(f"{'Year':<6} {'Complete':<10} {'Trajectory':<12} {'Missing':<8} {'%':<6}")
    print("-" * 45)
    
    for year in sorted(complete_years.index):
        complete_count = complete_years.get(year, 0)
        trajectory_count = trajectory_years.get(year, 0)
        missing_count = complete_count - trajectory_count
        pct = trajectory_count / complete_count * 100 if complete_count > 0 else 0
        
        status = "✓" if missing_count == 0 else "✗"
        print(f"{int(year):<6} {complete_count:<10} {trajectory_count:<12} {missing_count:<8} {pct:<5.1f}% {status}")
    
    # 6. Recent QBs check (2021+)
    print(f"\n" + "="*80)
    print("RECENT QBs CHECK (2021+)")
    print("="*80)
    
    # Filter out QBs with missing draft_year before checking recent QBs
    recent_complete = complete_df[
        (complete_df['draft_year'] >= 2021) & 
        (complete_df['draft_year'].notna())
    ]['player_id'].unique()
    
    recent_trajectory = trajectory_df[
        (trajectory_df['draft_year'] >= 2021) & 
        (trajectory_df['draft_year'].notna())
    ]['player_id'].unique()
    
    print(f"Recent QBs (2021+) in complete data: {len(recent_complete)}")
    print(f"Recent QBs (2021+) in trajectory data: {len(recent_trajectory)}")
    
    missing_recent = set(recent_complete) - set(recent_trajectory)
    if missing_recent:
        print(f"\n❌ Missing recent QBs:")
        recent_missing_details = complete_df[complete_df['player_id'].isin(missing_recent)].groupby('player_id').agg({
            'player_name': 'first',
            'draft_year': 'first',
            'draft_team': 'first'
        }).sort_values('draft_year', ascending=False)
        
        for player_id, row in recent_missing_details.iterrows():
            # Handle missing values safely
            draft_year = 'Unknown' if pd.isna(row['draft_year']) else str(int(row['draft_year']))
            draft_team = 'Unknown' if pd.isna(row['draft_team']) else str(row['draft_team'])
            
            print(f"   {row['player_name']}: {draft_year} {draft_team}")
    else:
        print(f"✅ All recent QBs included!")
    
    # 7. Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if len(missing_qbs) == 0:
        print("🎉 PERFECT: All QBs from complete dataset are in trajectory data")
    elif len(missing_qbs) < 5:
        print(f"✅ GOOD: Only {len(missing_qbs)} QBs missing ({coverage_pct:.1f}% coverage)")
    else:
        print(f"⚠️  ISSUES: {len(missing_qbs)} QBs missing ({coverage_pct:.1f}% coverage)")
        print("   Check the missing QBs list above")
    
    if len(missing_recent) == 0:
        print("✅ All recent draft picks (2021+) included")
    else:
        print(f"❌ {len(missing_recent)} recent draft picks missing")
    
    return {
        'trajectory_qbs': len(trajectory_qbs),
        'complete_qbs': len(complete_qbs),
        'missing_qbs': missing_qbs,
        'coverage_pct': coverage_pct,
        'missing_recent': missing_recent
    }

if __name__ == "__main__":
    
    results = check_qb_coverage()
