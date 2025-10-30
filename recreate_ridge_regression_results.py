#!/usr/bin/env python3
"""
QB Research: Ridge Regression Results Recreation Script (FIXED FOR P-VALUES)

This script runs the ENHANCED ridge regression functions from QB_research.py
that include bootstrap-based statistical significance testing with p-values.
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

def check_enhanced_functions():
    """Check if the enhanced functions with p-values are available"""
    try:
        ridge_regression_with_significance_testing
        payment_prediction_with_confusion_matrix
        print("✓ Enhanced functions with p-values detected!")
        return True
    except NameError:
        print("⚠️  Enhanced functions not found in QB_research.py")
        print("\nYou need to add these functions to QB_research.py:")
        print("  - ridge_regression_with_significance_testing()")
        print("  - payment_prediction_with_confusion_matrix()")
        print("\nThese functions provide bootstrap-based p-values and statistical significance.")
        return False

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ridge_regression_results_with_pvalues_{timestamp}"
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
            import traceback
            traceback.print_exc()
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
    """Main function to run ridge regression analyses WITH P-VALUES"""
    print("="*80)
    print("QB RESEARCH: RECREATING RIDGE REGRESSION RESULTS WITH P-VALUES")
    print("="*80)
    
    # Check if enhanced functions are available
    if not check_enhanced_functions():
        print("\n" + "="*80)
        print("FALLBACK: RUNNING BASIC FUNCTIONS (NO P-VALUES)")
        print("="*80)
        print("Will run basic ridge regression without statistical significance testing.")
        input("Press Enter to continue or Ctrl+C to exit...")
        use_enhanced = False
    else:
        use_enhanced = True
    
    # Create results directory
    results_dir = create_results_directory()
    
    if use_enhanced:
        # === ENHANCED FUNCTIONS WITH P-VALUES ===
        
        # 1. Enhanced Wins Prediction with Statistical Significance (Extended Features)
        print("\n[1/4] Running Enhanced Wins Prediction with P-Values (Extended Features)...")
        print("  - Using 1000 bootstrap samples for significance testing")
        print("  - This may take 2-3 minutes...")
        
        output1, result1 = capture_function_output(
            ridge_regression_with_significance_testing,
            train_df=None,
            test_df=None,
            use_extended_features=True,
            n_bootstrap=1000
        )
        
        save_text_output(
            output1,
            os.path.join(results_dir, "wins_prediction_with_pvalues_extended.txt"),
            "Enhanced Wins Prediction: Statistical Significance Testing (Extended Features)"
        )
        
        # 2. Enhanced Wins Prediction with Statistical Significance (Original Features)
        print("\n[2/4] Running Enhanced Wins Prediction with P-Values (Original Features)...")
        
        output2, result2 = capture_function_output(
            ridge_regression_with_significance_testing,
            train_df=None,
            test_df=None,
            use_extended_features=False,
            n_bootstrap=1000
        )
        
        save_text_output(
            output2,
            os.path.join(results_dir, "wins_prediction_with_pvalues_original.txt"),
            "Enhanced Wins Prediction: Statistical Significance Testing (Original Features)"
        )
        
        # 3. Enhanced Payment Prediction with Confusion Matrix
        print("\n[3/4] Running Enhanced Payment Prediction with Confusion Matrix...")
        
        output3, result3 = capture_function_output(
            payment_prediction_with_confusion_matrix,
            alpha_range=None,
            exclude_recent_drafts=True,
            probability_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]
        )
        
        save_text_output(
            output3,
            os.path.join(results_dir, "payment_prediction_with_confusion_matrix.txt"),
            "Enhanced Payment Prediction: Confusion Matrix & Classification Analysis"
        )
        
    else:
        # === FALLBACK: BASIC FUNCTIONS (NO P-VALUES) ===
        
        # 1. Basic Ridge Regression for Wins Prediction (Extended Features)
        print("\n[1/4] Running Basic Ridge Regression (Extended Features)...")
        output1, result1 = capture_function_output(
            ridge_regression_with_cv,
            train_df=None,
            test_df=None,
            alpha_range=None,
            use_extended_features=True
        )
        
        save_text_output(
            output1,
            os.path.join(results_dir, "wins_prediction_ridge_extended_basic.txt"),
            "Basic Ridge Regression: Wins Prediction (Extended Features)"
        )
        
        # 2. Basic Ridge Regression for Wins Prediction (Original Features)
        print("\n[2/4] Running Basic Ridge Regression (Original Features)...")
        output2, result2 = capture_function_output(
            ridge_regression_with_cv,
            train_df=None,
            test_df=None,
            alpha_range=None,
            use_extended_features=False
        )
        
        save_text_output(
            output2,
            os.path.join(results_dir, "wins_prediction_ridge_original_basic.txt"),
            "Basic Ridge Regression: Wins Prediction (Original Features)"
        )
        
        # 3. Basic Payment Prediction
        print("\n[3/4] Running Basic Payment Prediction...")
        output3, result3 = capture_function_output(
            ridge_regression_payment_prediction,
            alpha_range=None,
            exclude_recent_drafts=True
        )
        
        save_text_output(
            output3,
            os.path.join(results_dir, "payment_prediction_ridge_basic.txt"),
            "Basic Ridge Regression: Payment Prediction"
        )
    
    # 4. Year Weighting Analysis (Works with both versions)
    print("\n[4/4] Running Year Weighting Analysis...")
    
    # Run for total_yards_adj
    print("  - Analyzing total_yards_adj...")
    output4a, result4a = capture_function_output(
        year_weighting_regression_with_significance,
        metric='total_yards_adj',
        max_decision_year=6
    )
    
    save_text_output(
        output4a,
        os.path.join(results_dir, "year_weighting_total_yards.txt"),
        "Year Weighting Analysis: Total Yards (Era Adjusted)"
    )
    
    # Run for Pass_ANY/A_adj
    print("  - Analyzing Pass_ANY/A_adj...")
    output4b, result4b = capture_function_output(
        year_weighting_regression_with_significance,
        metric='Pass_ANY/A_adj',
        max_decision_year=6
    )
    
    save_text_output(
        output4b,
        os.path.join(results_dir, "year_weighting_any_a.txt"),
        "Year Weighting Analysis: Pass ANY/A (Era Adjusted)"
    )
    
    # 5. Create comprehensive summary report
    print("\n[5/5] Creating Summary Report...")
    
    if use_enhanced:
        summary_content = f"""RIDGE REGRESSION WITH P-VALUES: COMPREHENSIVE RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

🎯 STATISTICAL SIGNIFICANCE TESTING COMPLETED!

This analysis provides rigorous statistical testing with p-values for all variables.

FILES GENERATED:
{"="*80}

📊 WINS PREDICTION WITH P-VALUES:
- wins_prediction_with_pvalues_extended.txt
- wins_prediction_with_pvalues_original.txt
- ridge_regression_significance_tests.csv ⭐ (KEY FILE: P-VALUES HERE)

📊 PAYMENT PREDICTION WITH CLASSIFICATION METRICS:
- payment_prediction_with_confusion_matrix.txt
- payment_prediction_confusion_matrix_analysis.csv
- payment_prediction_classification_importance.csv

📊 YEAR WEIGHTING ANALYSIS:
- year_weighting_total_yards.txt
- year_weighting_any_a.txt
- year_weights_*.csv

KEY STATISTICAL OUTPUTS:
{"="*80}

🔍 P-VALUES AND SIGNIFICANCE:
Check ridge_regression_significance_tests.csv for:
- P_Value_Bootstrap: Statistical p-value for each variable
- Significant_95: True/False for statistical significance (p < 0.05)
- CI_Lower_95, CI_Upper_95: 95% confidence intervals
- T_Statistic: Coefficient divided by standard error

🔍 INTERPRETATION GUIDE:
- P_Value_Bootstrap < 0.05: Statistically significant predictor
- P_Value_Bootstrap < 0.01: Highly significant predictor  
- P_Value_Bootstrap < 0.001: Very highly significant predictor
- Significant_95 = True: 95% confidence interval doesn't include zero

🔍 CONFUSION MATRIX METRICS:
Check payment_prediction_confusion_matrix_analysis.csv for:
- test_accuracy: Overall prediction accuracy
- test_precision: Of predicted "gets paid", how many were correct
- test_recall: Of actual "gets paid", how many were caught
- test_f1_score: Balanced metric (harmonic mean of precision/recall)

RESEARCH QUESTIONS ANSWERED:
{"="*80}

❓ Which QB metrics significantly predict team wins?
✅ See Significant_95 = True in ridge_regression_significance_tests.csv

❓ What are the p-values for each predictor?
✅ See P_Value_Bootstrap column in ridge_regression_significance_tests.csv

❓ How reliable are the coefficient estimates?
✅ See confidence intervals (CI_Lower_95, CI_Upper_95)

❓ What's the effect size of each significant predictor?
✅ See Coefficient column (effect per 1 standard deviation change)

❓ How accurately can we predict QB payments?
✅ See ROC AUC and confusion matrix metrics

❓ What's the optimal decision threshold?
✅ See optimal_threshold in confusion matrix analysis

NEXT STEPS:
{"="*80}

1. 🔍 Open ridge_regression_significance_tests.csv in Excel/spreadsheet
2. 📊 Sort by P_Value_Bootstrap (ascending) to see most significant predictors
3. 📈 Focus on variables with Significant_95 = True for reliable insights
4. 🎯 Use optimal threshold from confusion matrix for binary decisions
5. 📝 Report effect sizes (coefficients) with confidence intervals
"""
    else:
        summary_content = f"""BASIC RIDGE REGRESSION RESULTS (NO P-VALUES)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

⚠️  BASIC ANALYSIS COMPLETED (NO STATISTICAL SIGNIFICANCE TESTING)

To get p-values and statistical significance testing, you need to add the enhanced 
functions to your QB_research.py file.

FILES GENERATED:
{"="*80}

📊 BASIC WINS PREDICTION:
- wins_prediction_ridge_extended_basic.txt
- wins_prediction_ridge_original_basic.txt

📊 BASIC PAYMENT PREDICTION:
- payment_prediction_ridge_basic.txt

📊 YEAR WEIGHTING ANALYSIS:
- year_weighting_total_yards.txt
- year_weighting_any_a.txt

TO GET P-VALUES:
{"="*80}

You need to add these functions to QB_research.py:
1. ridge_regression_with_significance_testing()
2. payment_prediction_with_confusion_matrix()

These functions provide:
- Bootstrap-based p-values
- 95% confidence intervals
- Statistical significance tests
- Confusion matrix analysis
- ROC curves and AUC scores

The enhanced functions are available in the additional_qb_research_functions.py file.
Copy them into your QB_research.py file and run this script again.
"""
    
    with open(os.path.join(results_dir, "ANALYSIS_SUMMARY.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"\nAnalysis summary saved: {os.path.join(results_dir, 'ANALYSIS_SUMMARY.txt')}")
    
    # List all generated files
    print(f"\n{'='*80}")
    print("ALL GENERATED FILES:")
    print(f"{'='*80}")
    
    for filename in sorted(os.listdir(results_dir)):
        filepath = os.path.join(results_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        
        # Highlight key files
        if 'significance_tests.csv' in filename:
            print(f"  ⭐ {filename} ({size_kb:.1f} KB) ← P-VALUES HERE!")
        elif 'confusion_matrix' in filename and filename.endswith('.csv'):
            print(f"  📊 {filename} ({size_kb:.1f} KB) ← CLASSIFICATION METRICS")
        else:
            print(f"     {filename} ({size_kb:.1f} KB)")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    
    if use_enhanced:
        print("\n🎉 SUCCESS: Statistical significance testing completed!")
        print("\n🔍 Key files to check for p-values:")
        print("   - ridge_regression_significance_tests.csv")
        print("   - payment_prediction_confusion_matrix_analysis.csv")
        print("\n📊 Your research now has proper statistical rigor!")
    else:
        print("\n⚠️  To get p-values, add the enhanced functions to QB_research.py")
    
    return results_dir

if __name__ == "__main__":
    main()