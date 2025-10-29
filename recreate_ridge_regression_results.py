#!/usr/bin/env python3
"""
QB Research: Ridge Regression Results Recreation Script

This script re-runs the existing ridge regression functions from QB_research.py
and saves their terminal outputs and results to properly documented files.
"""

import sys
import os
from datetime import datetime
from contextlib import redirect_stdout
import io

# Import the QB research functions
try:
    from QB_research import *
except ImportError:
    print("Error: QB_research.py not found. Make sure it's in the same directory.")
    sys.exit(1)

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ridge_regression_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    return results_dir

def capture_function_output(func, *args, **kwargs):
    """Capture both the function output and any return values"""
    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error running function: {e}")
            result = None
    
    output = f.getvalue()
    return output, result

def save_text_output(content, filepath, title):
    """Save text content to a file with header"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(content)
    print(f"Saved: {filepath}")

def main():
    """Main function to run all ridge regression analyses"""
    print("="*80)
    print("QB RESEARCH: RECREATING RIDGE REGRESSION RESULTS")
    print("="*80)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # 1. Ridge Regression for Wins Prediction (with extended features)
    print("\n[1/4] Running Ridge Regression for Wins Prediction (Extended Features)...")
    output1, result1 = capture_function_output(
        ridge_regression_with_cv,
        train_df=None,  # Will load from files
        test_df=None,   # Will load from files
        alpha_range=None,  # Use default
        use_extended_features=True
    )
    
    save_text_output(
        output1, 
        os.path.join(results_dir, "wins_prediction_ridge_extended.txt"),
        "Ridge Regression: Wins Prediction (Extended Features)"
    )
    
    # 2. Ridge Regression for Wins Prediction (Original features)
    print("\n[2/4] Running Ridge Regression for Wins Prediction (Original Features)...")
    output2, result2 = capture_function_output(
        ridge_regression_with_cv,
        train_df=None,
        test_df=None,
        alpha_range=None,
        use_extended_features=False
    )
    
    save_text_output(
        output2,
        os.path.join(results_dir, "wins_prediction_ridge_original.txt"),
        "Ridge Regression: Wins Prediction (Original Features)"
    )
    
    # 3. Ridge Regression for Payment Probability
    print("\n[3/4] Running Ridge Regression for Payment Probability...")
    output3, result3 = capture_function_output(
        ridge_regression_payment_prediction,
        alpha_range=None,
        exclude_recent_drafts=True
    )
    
    save_text_output(
        output3,
        os.path.join(results_dir, "payment_probability_ridge.txt"),
        "Ridge Regression: Payment Probability Prediction"
    )
    
    # 4. Year Weighting Analysis (for context)
    print("\n[4/4] Running Year Weighting Analysis...")
    
    # Run for total_yards_adj
    output4a, result4a = capture_function_output(
        year_weighting_regression,
        metric='total_yards_adj',
        max_decision_year=6
    )
    
    save_text_output(
        output4a,
        os.path.join(results_dir, "year_weighting_total_yards.txt"),
        "Year Weighting Analysis: Total Yards (Era Adjusted)"
    )
    
    # Run for Pass_ANY/A_adj
    output4b, result4b = capture_function_output(
        year_weighting_regression,
        metric='Pass_ANY/A_adj',
        max_decision_year=6
    )
    
    save_text_output(
        output4b,
        os.path.join(results_dir, "year_weighting_any_a.txt"),
        "Year Weighting Analysis: Pass ANY/A (Era Adjusted)"
    )
    
    # 5. Create summary report
    print("\n[5/5] Creating Summary Report...")
    
    summary_content = f"""RIDGE REGRESSION ANALYSES SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

ANALYSES COMPLETED:

1. WINS PREDICTION (Extended Features)
   - File: wins_prediction_ridge_extended.txt
   - Features: 10 performance factors (top PC1 loadings)
   - Output: R², RMSE, variable importance, test performance

2. WINS PREDICTION (Original Features) 
   - File: wins_prediction_ridge_original.txt
   - Features: 5 core performance factors (manual selection)
   - Output: R², RMSE, variable importance, test performance

3. PAYMENT PROBABILITY PREDICTION
   - File: payment_probability_ridge.txt
   - Features: Averaged performance (Years 1-3)
   - Output: R², classification metrics, variable importance
   - Note: Uses eligible QBs (drafted ≤2020)

4. YEAR WEIGHTING ANALYSIS
   - Files: year_weighting_total_yards.txt, year_weighting_any_a.txt
   - Analysis: How each prior year is weighted in payment decisions
   - Output: Year-by-year coefficients and importance weights

KEY FILES TO CHECK:
- Variable importance CSVs are automatically saved by the functions
- Look for files like: ridge_regression_variable_importance.csv
- Payment prediction importance: payment_prediction_importance.csv
- Year weights: year_weights_*.csv

NOTES:
- All functions use proper train/test splits for unbiased evaluation
- Ridge regression includes cross-validation for alpha selection
- Statistical significance testing is built into the functions
- Payment probability analysis includes confusion matrix-like metrics
"""
    
    with open(os.path.join(results_dir, "SUMMARY_REPORT.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"\nSummary report saved: {os.path.join(results_dir, 'SUMMARY_REPORT.txt')}")
    
    # List all generated files
    print(f"\n{'='*80}")
    print("ALL GENERATED FILES:")
    print(f"{'='*80}")
    
    for filename in sorted(os.listdir(results_dir)):
        filepath = os.path.join(results_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename} ({size_kb:.1f} KB)")
    
    print(f"\n{'='*80}")
    print("RECREATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    print("Check the SUMMARY_REPORT.txt for an overview of all analyses.")
    
    return results_dir

if __name__ == "__main__":
    main()