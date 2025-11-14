"""
Data validation functions for QB research.

This module contains functions for:
- Validating contract-to-player-ID mappings
- Testing name normalization functionality
"""

import pandas as pd

from qb_research.utils.name_matching import normalize_player_name


def validate_contract_mapping(mapped_contracts_df):
    """
    Validates the contract-to-player-ID mapping and generates a report.
    
    Note: This includes ALL players who got contracts, not just first-round QBs.
    You can filter to first-round QBs later if needed.
    
    Args:
        mapped_contracts_df: DataFrame with player_id mapping
    
    Returns:
        dict: Validation report with statistics and issues
    """
    print("\n" + "="*80)
    print("VALIDATING CONTRACT MAPPING")
    print("="*80)
    
    report = {
        'total_contracts': len(mapped_contracts_df),
        'mapped_contracts': mapped_contracts_df['player_id'].notna().sum(),
        'unmapped_contracts': mapped_contracts_df['player_id'].isna().sum(),
        'unique_players': mapped_contracts_df['player_id'].nunique(),
        'issues': []
    }
    
    # Filter to only mapped contracts for validation
    mapped_only = mapped_contracts_df[mapped_contracts_df['player_id'].notna()].copy()
    
    if len(mapped_only) == 0:
        print("⚠️  Warning: No contracts mapped to player IDs")
        return report
    
    # Check for duplicate contracts (same player, year)
    duplicates = mapped_only.groupby(['player_id', 'Year']).size()
    duplicates = duplicates[duplicates > 1]
    
    if len(duplicates) > 0:
        report['issues'].append(f"Found {len(duplicates)} player-years with multiple contracts")
        print(f"⚠️  Warning: {len(duplicates)} player-years have multiple contracts")
        print("   (This might be extensions + restructures in same year)")
        
        # Show examples
        for (player_id, year), count in duplicates.head(3).items():
            examples = mapped_only[(mapped_only['player_id'] == player_id) & (mapped_only['Year'] == year)]
            player_name = examples.iloc[0]['matched_name']
            print(f"   Example: {player_name} in {year} has {count} contracts")
    
    print(f"\n✓ Validation complete")
    print(f"  Total contracts: {report['total_contracts']}")
    print(f"  Mapped: {report['mapped_contracts']} ({report['mapped_contracts']/report['total_contracts']*100:.1f}%)")
    print(f"  Unique players: {report['unique_players']}")
    
    if len(report['issues']) == 0:
        print("  No issues found")
    
    return report


def test_name_mapping():
    """
    Quick test of the name normalization functionality.
    """
    print("\n" + "="*80)
    print("TESTING NAME NORMALIZATION")
    print("="*80)
    
    # Test normalization
    test_names = [
        ("Patrick Mahomes II", "patrick mahomes"),
        ("Baker Mayfield Jr.", "baker mayfield"),
        ("Joe Burrow", "joe burrow"),
        ("LAMAR JACKSON", "lamar jackson"),
        ("Josh Allen", "josh allen"),
        ("Tom   Brady", "tom brady"),  # Extra spaces
        ("Peyton Manning Jr", "peyton manning"),
    ]
    
    print("\nName normalization test:")
    all_pass = True
    for original, expected in test_names:
        normalized = normalize_player_name(original)
        passed = normalized == expected
        status = "✓" if passed else "✗"
        print(f"  {status} '{original}' → '{normalized}' (expected: '{expected}')")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ All name normalization tests passed!")
    else:
        print("\n✗ Some name normalization tests failed")
    
    return all_pass

