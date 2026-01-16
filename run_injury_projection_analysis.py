"""
Script to run injury projection predictiveness comparison analysis.

This script compares era-adjusted vs injury-projected counting stats
to determine which is more predictive of QB free-agent contracts.
"""

import sys
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from qb_research.modeling.prediction_models import compare_injury_projection_predictiveness

if __name__ == "__main__":
    print("="*80)
    print("INJURY PROJECTION PREDICTIVENESS ANALYSIS")
    print("="*80)
    print("\nThis will compare:")
    print("  - BASELINE: Era-adjusted stats (_adj columns)")
    print("  - PROJECTED: Era-adjusted + injury-projected stats (_adj_proj columns)")
    print("\nUsing F1-score and accuracy as primary metrics.")
    print("="*80)
    
    # Run the comparison
    results = compare_injury_projection_predictiveness(
        alpha_range=[0.01, 0.1, 1.0, 10.0, 100.0],
        exclude_recent_drafts=True,
        random_seed=42
    )
    
    if results:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: injury_projection_predictiveness_comparison.csv")
        print(f"Winner: {results['winner']}")
        print(f"F1 improvement: {results['f1_improvement']:+.4f}")
        print(f"Accuracy improvement: {results['accuracy_improvement']:+.4f}")
    else:
        print("\n✗ Analysis failed. Check error messages above.")
