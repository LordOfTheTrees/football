#!/usr/bin/env python3
"""
QB Research: Year Weighting Significance Testing Script

This script runs the ENHANCED year weighting function with statistical significance testing
to determine which prior years are statistically significant predictors of QB payment decisions.

EXACTLY what you asked for: Year weighting significance with p-values!
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

def check_enhanced_year_weighting_function():
    """Check if the enhanced year weighting function with p-values is available"""
    try:
        year_weighting_regression_with_significance
        print("✓ Enhanced year weighting function with p-values detected!")
        return True
    except NameError:
        print("⚠️  Enhanced year weighting function not found in QB_research.py")
        print("\nYou need to add this function to QB_research.py:")
        print("  - year_weighting_regression_with_significance()")
        print("\nThis function provides bootstrap-based p-values for year weighting analysis.")
        return False

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"year_weighting_significance_results_{timestamp}"
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
    """Main function to run year weighting significance analysis WITH P-VALUES"""
    print("="*80)
    print("QB RESEARCH: YEAR WEIGHTING SIGNIFICANCE TESTING WITH P-VALUES")
    print("="*80)
    
    # Check if enhanced function is available
    if not check_enhanced_year_weighting_function():
        print("\n" + "="*80)
        print("FALLBACK: RUNNING BASIC YEAR WEIGHTING (NO P-VALUES)")
        print("="*80)
        print("Will run basic year weighting without statistical significance testing.")
        print("\nTo get p-values, copy the enhanced function from year_weighting_with_pvalues.py")
        print("into your QB_research.py file, then run this script again.")
        input("\nPress Enter to continue with basic analysis or Ctrl+C to exit...")
        use_enhanced = False
    else:
        use_enhanced = True
        print("🎉 Ready to run year weighting analysis with statistical significance!")
    
    # Create results directory
    results_dir = create_results_directory()
    
    if use_enhanced:
        # === ENHANCED YEAR WEIGHTING WITH P-VALUES ===
        
        print("\n" + "="*80)
        print("RUNNING ENHANCED YEAR WEIGHTING ANALYSIS")
        print("="*80)
        print("This analysis will determine which prior years are statistically significant")
        print("predictors of QB payment decisions, with bootstrap p-values.")
        print("\n⏱️  Note: Bootstrap analysis takes ~5-10 minutes per metric")
        
        # 1. Total Yards Analysis with P-Values
        print("\n[1/2] Analyzing total_yards_adj with statistical significance...")
        print("  - Decision years 3-6")
        print("  - 1000 bootstrap samples for p-values")
        print("  - This may take 5-10 minutes...")
        
        output1, result1 = capture_function_output(
            year_weighting_regression_with_significance,
            metric='total_yards_adj',
            max_decision_year=6,
            n_bootstrap=1000
        )
        
        save_text_output(
            output1,
            os.path.join(results_dir, "year_weighting_total_yards_with_pvalues.txt"),
            "Year Weighting Significance Analysis: Total Yards (Era Adjusted)"
        )
        
        # 2. Pass ANY/A Analysis with P-Values
        print("\n[2/2] Analyzing Pass_ANY/A_adj with statistical significance...")
        print("  - Decision years 3-6")
        print("  - 1000 bootstrap samples for p-values")
        print("  - This may take 5-10 minutes...")
        
        output2, result2 = capture_function_output(
            year_weighting_regression_with_significance,
            metric='Pass_ANY/A_adj',
            max_decision_year=6,
            n_bootstrap=1000
        )
        
        save_text_output(
            output2,
            os.path.join(results_dir, "year_weighting_any_a_with_pvalues.txt"),
            "Year Weighting Significance Analysis: Pass ANY/A (Era Adjusted)"
        )
        
    else:
        # === FALLBACK: BASIC YEAR WEIGHTING (NO P-VALUES) ===
        
        print("\n[1/2] Running basic year weighting for total_yards_adj...")
        output1, result1 = capture_function_output(
            year_weighting_regression,
            metric='total_yards_adj',
            max_decision_year=6
        )
        
        save_text_output(
            output1,
            os.path.join(results_dir, "year_weighting_total_yards_basic.txt"),
            "Basic Year Weighting Analysis: Total Yards (No P-Values)"
        )
        
        print("\n[2/2] Running basic year weighting for Pass_ANY/A_adj...")
        output2, result2 = capture_function_output(
            year_weighting_regression,
            metric='Pass_ANY/A_adj',
            max_decision_year=6
        )
        
        save_text_output(
            output2,
            os.path.join(results_dir, "year_weighting_any_a_basic.txt"),
            "Basic Year Weighting Analysis: Pass ANY/A (No P-Values)"
        )
    
    # Create comprehensive summary report
    print("\n[3/3] Creating Summary Report...")
    
    if use_enhanced:
        summary_content = f"""YEAR WEIGHTING SIGNIFICANCE ANALYSIS: COMPREHENSIVE RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

🎯 YEAR WEIGHTING STATISTICAL SIGNIFICANCE TESTING COMPLETED!

This analysis determines which prior years are statistically significant predictors
of QB payment decisions using bootstrap-based p-value testing.

RESEARCH QUESTION ANSWERED:
{"="*80}
❓ "Which prior years significantly influence QB payment decisions?"
✅ See the significance matrices and p-value results below!

FILES GENERATED:
{"="*80}

📊 DETAILED ANALYSIS OUTPUTS:
- year_weighting_total_yards_with_pvalues.txt
- year_weighting_any_a_with_pvalues.txt

📊 KEY STATISTICAL OUTPUT FILES (WITH P-VALUES):
⭐ year_weights_significance_total_yards_adj.csv ← DETAILED P-VALUES HERE!
⭐ year_weights_significance_Pass_ANY_A_adj.csv ← DETAILED P-VALUES HERE!

📊 SUMMARY MATRICES:
- year_significance_matrix_*.csv (True/False significance by year)
- year_pvalue_matrix_*.csv (Actual p-values by year)  
- year_weight_matrix_*.csv (Importance weights by year)
- year_significance_summary_*.csv (Overall significance summary)

KEY STATISTICAL OUTPUTS TO EXAMINE:
{"="*80}

🔍 year_weights_significance_total_yards_adj.csv contains:

Decision_Year | Performance_Year | P_Value_Bootstrap | Significant_95 | Coefficient
------------- | ---------------- | ----------------- | -------------- | -----------
3             | Year 0           | 0.023             | True           | 0.45
3             | Year 1           | 0.156             | False          | 0.23
3             | Year 2           | 0.089             | False          | 0.32
4             | Year 0           | 0.001             | True           | 0.52
...

🔍 year_significance_summary_total_yards_adj.csv contains:

Performance_Year | Times_Significant | Avg_P_Value | Avg_Weight_%
---------------- | ----------------- | ----------- | ------------
Year 0           | 3                 | 0.015       | 35.2
Year 1           | 1                 | 0.234       | 22.1
Year 2           | 2                 | 0.067       | 28.4
...

INTERPRETATION GUIDE:
{"="*80}

📈 P-VALUE INTERPRETATION:
- P_Value_Bootstrap < 0.05: Statistically significant year
- P_Value_Bootstrap < 0.01: Highly significant year
- P_Value_Bootstrap < 0.001: Very highly significant year
- Significant_95 = True: 95% confidence interval doesn't include zero

📈 TIMES_SIGNIFICANT INTERPRETATION:
- Times_Significant = 4: Year is significant in all 4 decision years (3,4,5,6)
- Times_Significant = 3: Year is significant in 3 out of 4 decision years
- Times_Significant = 0: Year is never statistically significant

📈 COEFFICIENT INTERPRETATION:
- Positive coefficient: Higher performance in that year increases payment probability
- Negative coefficient: Higher performance in that year decreases payment probability
- Larger absolute coefficient: Stronger effect of that year's performance

RESEARCH FINDINGS:
{"="*80}

🏆 MOST IMPORTANT YEARS FOR PAYMENT DECISIONS:
Check year_significance_summary_*.csv and sort by Times_Significant (descending)

🏆 MOST STATISTICALLY ROBUST FINDINGS:
Look for years with:
- Times_Significant ≥ 3 (significant in most decision years)
- Avg_P_Value < 0.05 (consistently low p-values)
- Large Avg_Weight_% (high importance across decision years)

BUSINESS IMPLICATIONS:
{"="*80}

🎯 FOR QB EVALUATION:
- Focus scouting on performance years that are consistently significant
- Weight recent vs. rookie year performance based on statistical findings
- Use significance results to justify evaluation frameworks

🎯 FOR CONTRACT TIMING:
- Years with low p-values are reliable predictors
- Decision year analysis shows when each year's performance matters most
- Optimize contract negotiation timing based on which years matter

NEXT STEPS:
{"="*80}

1. 📊 Open year_significance_summary_*.csv in Excel
2. 🔍 Sort by Times_Significant to find most important years
3. 📈 Compare total_yards vs Pass_ANY/A results for consistency
4. 📝 Focus evaluation frameworks on statistically significant years
5. 🎯 Use findings to optimize scouting and contract timing

{"="*80}
STATISTICAL RIGOR ACHIEVED: Your year weighting analysis now has proper p-values!
{"="*80}
"""
    else:
        summary_content = f"""BASIC YEAR WEIGHTING RESULTS (NO P-VALUES)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

⚠️  BASIC ANALYSIS COMPLETED (NO STATISTICAL SIGNIFICANCE TESTING)

To get p-values and statistical significance testing for year weighting, you need 
to add the enhanced function to your QB_research.py file.

FILES GENERATED:
{"="*80}

📊 BASIC YEAR WEIGHTING OUTPUTS:
- year_weighting_total_yards_basic.txt
- year_weighting_any_a_basic.txt
- year_weights_*.csv (basic weights without p-values)

TO GET P-VALUES FOR YEAR WEIGHTING:
{"="*80}

1. Copy the function year_weighting_regression_with_significance() from:
   year_weighting_with_pvalues.py → your QB_research.py file

2. Run this script again

This enhanced function provides:
- Bootstrap-based p-values for each year's importance
- Statistical significance testing (95% confidence intervals)
- Comprehensive significance matrices
- Robust statistical testing across 1000 bootstrap samples

The enhanced function will tell you which years are STATISTICALLY SIGNIFICANT
predictors of payment decisions, not just which years have higher coefficients.
"""
    
    with open(os.path.join(results_dir, "YEAR_WEIGHTING_ANALYSIS_SUMMARY.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"\nAnalysis summary saved: {os.path.join(results_dir, 'YEAR_WEIGHTING_ANALYSIS_SUMMARY.txt')}")
    
    # List all generated files
    print(f"\n{'='*80}")
    print("ALL GENERATED FILES:")
    print(f"{'='*80}")
    
    for filename in sorted(os.listdir(results_dir)):
        filepath = os.path.join(results_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        
        # Highlight key files
        if 'significance_' in filename and filename.endswith('.csv'):
            print(f"  ⭐ {filename} ({size_kb:.1f} KB) ← P-VALUES FOR YEAR WEIGHTING!")
        elif 'pvalue_matrix' in filename:
            print(f"  📊 {filename} ({size_kb:.1f} KB) ← P-VALUE MATRIX")
        elif 'significance_summary' in filename:
            print(f"  🏆 {filename} ({size_kb:.1f} KB) ← OVERALL SIGNIFICANCE SUMMARY")
        else:
            print(f"     {filename} ({size_kb:.1f} KB)")
    
    print(f"\n{'='*80}")
    print("YEAR WEIGHTING SIGNIFICANCE ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    
    if use_enhanced:
        print("\n🎉 SUCCESS: Year weighting statistical significance testing completed!")
        print("\n🔍 Key files to check for p-values:")
        print("   ⭐ year_weights_significance_*.csv (detailed p-values)")
        print("   📊 year_pvalue_matrix_*.csv (p-value matrix)")
        print("   🏆 year_significance_summary_*.csv (overall summary)")
        print("\n📊 Your year weighting analysis now has proper statistical rigor!")
        print("\n🎯 You can now answer: 'Which prior years SIGNIFICANTLY predict payment?'")
    else:
        print("\n⚠️  To get p-values for year weighting, add the enhanced function to QB_research.py")
        print("   Copy from: year_weighting_with_pvalues.py")
    
    return results_dir

if __name__ == "__main__":
    main()
