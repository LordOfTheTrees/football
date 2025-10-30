#!/usr/bin/env python3
"""
Enhanced QB Research: Ridge Regression with Statistical Significance & Confusion Matrices

This script runs enhanced versions of the ridge regression analyses that include:
1. Bootstrap-based statistical significance testing for wins prediction
2. Comprehensive confusion matrix analysis for payment prediction

SETUP: First copy the functions from additional_qb_research_functions.py into your QB_research.py file.
"""

import sys
import os
from datetime import datetime
from contextlib import redirect_stdout
import io

# Import the QB research functions (including the new enhanced ones)
try:
    from QB_research import *
    # Try to import the new functions to verify they were added
    try:
        wins_prediction_linear_ridge
        payment_prediction_logistic_ridge
        print("✓ Enhanced functions detected!")
    except NameError:
        print("⚠️  Enhanced functions not found. Please add the functions from additional_qb_research_functions.py to QB_research.py")
        print("   Then run this script again.")
        sys.exit(1)
        
except ImportError:
    print("Error: QB_research.py not found. Make sure it's in the same directory.")
    sys.exit(1)

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"enhanced_ridge_regression_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    return results_dir

def capture_function_output(func, *args, **kwargs):
    """Capture both the function output and any return values"""
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
    """Main function to run enhanced ridge regression analyses"""
    print("="*80)
    print("ENHANCED QB RESEARCH: STATISTICAL SIGNIFICANCE & CONFUSION MATRICES")
    print("="*80)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # 1. Enhanced Wins Prediction with Statistical Significance (Extended Features)
    print("\n[1/4] Running Enhanced Wins Prediction with Statistical Significance (Extended Features)...")
    output1, result1 = capture_function_output(
        wins_prediction_linear_ridge,
        train_df=None,
        test_df=None,
        use_extended_features=True,
        n_bootstrap=1000  # Can reduce to 500 if too slow
    )
    
    save_text_output(
        output1,
        os.path.join(results_dir, "wins_prediction_with_significance_extended.txt"),
        "Enhanced Wins Prediction: Statistical Significance Testing (Extended Features)"
    )
    
    # 2. Enhanced Wins Prediction with Statistical Significance (Original Features)
    print("\n[2/4] Running Enhanced Wins Prediction with Statistical Significance (Original Features)...")
    output2, result2 = capture_function_output(
        wins_prediction_linear_ridge,
        train_df=None,
        test_df=None,
        use_extended_features=False,
        n_bootstrap=1000
    )
    
    save_text_output(
        output2,
        os.path.join(results_dir, "wins_prediction_with_significance_original.txt"),
        "Enhanced Wins Prediction: Statistical Significance Testing (Original Features)"
    )
    
    # 3. Enhanced Payment Prediction with Confusion Matrix Analysis
    print("\n[3/4] Running Enhanced Payment Prediction with Confusion Matrix Analysis...")
    output3, result3 = capture_function_output(
        wins_prediction_linear_ridge,
        alpha_range=None,
        exclude_recent_drafts=True,
        probability_thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    )
    
    save_text_output(
        output3,
        os.path.join(results_dir, "payment_prediction_with_confusion_matrix.txt"),
        "Enhanced Payment Prediction: Confusion Matrix & Classification Analysis"
    )
    
    # 4. Comparative Analysis Summary
    print("\n[4/4] Generating Comparative Analysis Summary...")
    
    summary_content = f"""ENHANCED RIDGE REGRESSION ANALYSES: STATISTICAL SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

This analysis provides statistical rigor to the QB research findings through:

1. BOOTSTRAP STATISTICAL SIGNIFICANCE TESTING
   └── Determines which variables are statistically significant predictors of wins
   └── Provides confidence intervals and p-values for each coefficient
   └── Tests model stability across 1000 bootstrap samples

2. COMPREHENSIVE CONFUSION MATRIX ANALYSIS  
   └── Evaluates payment prediction accuracy at multiple probability thresholds
   └── Provides precision, recall, F1-score, and ROC AUC metrics
   └── Identifies optimal decision threshold for classification

KEY FILES GENERATED:
{"="*80}

WINS PREDICTION ANALYSIS:
- wins_prediction_with_significance_extended.txt
- wins_prediction_with_significance_original.txt  
- ridge_regression_significance_tests.csv (significance test results)

PAYMENT PREDICTION ANALYSIS:
- payment_prediction_with_confusion_matrix.txt
- payment_prediction_confusion_matrix_analysis.csv (threshold analysis)
- payment_prediction_classification_importance.csv (feature importance)

RESEARCH QUESTIONS ANSWERED:
{"="*80}

🎯 WINS PREDICTION:
   Q: Which QB performance metrics significantly predict team wins?
   A: See significance test results - look for Significant_95 = True

   Q: How reliable are the coefficient estimates?
   A: Check confidence intervals and bootstrap stability metrics

   Q: What's the effect size of each significant predictor?
   A: Coefficient values show impact per 1 standard deviation change

🎯 PAYMENT PREDICTION:
   Q: How accurately can we predict which QBs get paid?
   A: See ROC AUC score and confusion matrix at optimal threshold

   Q: What's the optimal probability threshold for decisions?
   A: Check optimal threshold analysis (maximizes F1-score)

   Q: What are the trade-offs between precision and recall?
   A: Compare metrics across different probability thresholds

   Q: Which performance metrics best predict getting paid?
   A: See classification feature importance rankings

STATISTICAL INTERPRETATION GUIDE:
{"="*80}

SIGNIFICANCE TESTING (Wins Prediction):
- Significant_95 = True: Variable is statistically significant (95% confidence)
- P_Value_Bootstrap < 0.05: Strong evidence of real effect
- T_Statistic: Coefficient divided by standard error (larger = more significant)
- CI bounds: If both positive or both negative, effect is significant

CONFUSION MATRIX (Payment Prediction):
- True Positives (TP): Correctly predicted "gets paid"
- False Positives (FP): Incorrectly predicted "gets paid" 
- True Negatives (TN): Correctly predicted "doesn't get paid"
- False Negatives (FN): Incorrectly predicted "doesn't get paid"

- Precision = TP/(TP+FP): Of predicted pays, how many were correct?
- Recall = TP/(TP+FN): Of actual pays, how many did we catch?
- F1-Score: Harmonic mean of precision and recall (balanced metric)
- ROC AUC: Overall discriminative ability (0.5 = random, 1.0 = perfect)

BUSINESS IMPLICATIONS:
{"="*80}

FOR WINS PREDICTION:
- Use only statistically significant variables for reliable predictions
- Focus on variables with largest T-statistics for maximum impact
- Consider confidence intervals when making personnel decisions

FOR PAYMENT PREDICTION:  
- Use optimal threshold for binary classification decisions
- Higher precision = fewer "false alarms" (wrongly predicting payment)
- Higher recall = catch more QBs who actually get paid
- Choose threshold based on cost of false positives vs false negatives

NEXT STEPS:
{"="*80}

1. Review significance test results to identify most reliable predictors
2. Examine confusion matrix to understand prediction accuracy limitations  
3. Consider ensemble methods if single model performance is insufficient
4. Validate findings on additional data or different time periods
5. Implement decision framework using optimal probability thresholds

{"="*80}
END OF SUMMARY
{"="*80}
"""
    
    with open(os.path.join(results_dir, "ENHANCED_ANALYSIS_SUMMARY.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"\nSummary report saved: {os.path.join(results_dir, 'ENHANCED_ANALYSIS_SUMMARY.txt')}")
    
    # List all generated files
    print(f"\n{'='*80}")
    print("ALL GENERATED FILES:")
    print(f"{'='*80}")
    
    for filename in sorted(os.listdir(results_dir)):
        filepath = os.path.join(results_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename} ({size_kb:.1f} KB)")
    
    print(f"\n{'='*80}")
    print("ENHANCED ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    print("")
    print("🔍 Key files to examine:")
    print("  - ENHANCED_ANALYSIS_SUMMARY.txt (interpretation guide)")
    print("  - ridge_regression_significance_tests.csv (which variables are significant)")
    print("  - payment_prediction_confusion_matrix_analysis.csv (classification accuracy)")
    print("")
    print("📊 Your research now has proper statistical rigor!")
    
    return results_dir

if __name__ == "__main__":
    main()
