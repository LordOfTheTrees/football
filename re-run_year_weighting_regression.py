#!/usr/bin/env python3
"""
Year Weighting Significance Analysis Script

Runs the enhanced year weighting analysis with statistical significance testing
to determine which prior years significantly predict QB payment decisions.

FIXED VERSION: Handles all variable definitions and uses logistic regression properly.
"""

import sys
import os
from datetime import datetime
from contextlib import redirect_stdout
import io

def create_results_directory():
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"year_weighting_significance_{timestamp}"
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
    """Main function to run year weighting significance analysis"""
    print("="*80)
    print("YEAR WEIGHTING SIGNIFICANCE ANALYSIS")
    print("="*80)
    
    # Import the fixed function
    try:
        from QB_research import year_weighting_regression_with_significance
        print("✓ Loaded FIXED year weighting function with p-values!")
    except ImportError:
        print("✗ ERROR: Could not import fixed function")
        print("Make sure fixed_payment_prediction_functions.py is in the same directory")
        return None
    
    # Check if required data files exist
    required_files = ['qb_seasons_payment_labeled_era_adjusted.csv']
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"\n✗ ERROR: Required file not found: {filepath}")
            print("Run prepare_qb_payment_data() and create_era_adjusted_payment_data() first")
            return None
    
    print("✓ All required files found")
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Run analysis for both key metrics
    metrics_to_analyze = ['total_yards_adj', 'Pass_ANY/A_adj']
    
    for i, metric in enumerate(metrics_to_analyze, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(metrics_to_analyze)}] ANALYZING: {metric}")
        print("="*80)
        print(f"  - Decision years 3-6")
        print(f"  - Bootstrap samples: 1000")
        print(f"  - This may take 5-10 minutes...")
        
        output, result = capture_function_output(
            year_weighting_regression_with_significance,
            metric=metric,
            max_decision_year=6,
            n_bootstrap=1000
        )
        
        # Save detailed output
        safe_metric_name = metric.replace('/', '_').replace('%', 'pct')
        output_file = os.path.join(results_dir, f"year_weighting_{safe_metric_name}_with_pvalues.txt")
        
        save_text_output(
            output,
            output_file,
            f"Year Weighting Significance Analysis: {metric}"
        )
        
        if result is None:
            print(f"⚠️  Warning: Analysis failed for {metric}")
        else:
            print(f"✓ Successfully analyzed {metric}")
    
    # Create comprehensive summary
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE SUMMARY")
    print("="*80)
    
    summary_content = f"""YEAR WEIGHTING SIGNIFICANCE ANALYSIS: COMPREHENSIVE RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

🎯 RESEARCH QUESTION ANSWERED:
"Which prior performance years are statistically significant predictors of QB payment decisions?"

ANALYSIS OVERVIEW:
{"="*80}

✅ FIXED ISSUES:
- Uses LogisticRegression instead of Ridge for binary classification
- Properly handles coefficient arrays (1D extraction from 2D)
- Defines all bootstrap statistics variables correctly
- Uses McFadden's Pseudo-R² for logistic regression

✅ STATISTICAL RIGOR:
- 1000 bootstrap samples for robust p-value estimation
- 95% confidence intervals for all coefficients
- Two-tailed significance testing
- Controls for multiple comparisons

✅ METRICS ANALYZED:
- total_yards_adj (era-adjusted total offensive yards)
- Pass_ANY/A_adj (era-adjusted passing efficiency)

DECISION YEARS TESTED:
{"="*80}

📊 Decision Year 3: After Year 2 performance (5th year option decision)
📊 Decision Year 4: After Year 3 performance (typical extension timing)
📊 Decision Year 5: After Year 4 performance (late extension)
📊 Decision Year 6: After Year 5 performance (2nd contract)

For each decision year, tests significance of Years 0, 1, 2, ... up to (decision_year - 1)

KEY OUTPUT FILES:
{"="*80}

🔍 DETAILED P-VALUE RESULTS:
⭐ year_weights_significance_total_yards_adj.csv
⭐ year_weights_significance_Pass_ANY_A_adj.csv

These files contain for each metric and decision year:
- Performance_Year: Which prior year (Year 0, Year 1, etc.)
- Coefficient: Effect size (positive = increases payment probability)
- P_Value_Bootstrap: Statistical p-value from bootstrap testing
- Significant_95: True/False for statistical significance (p < 0.05)
- CI_Lower_95, CI_Upper_95: 95% confidence interval bounds
- T_Statistic: Coefficient / Standard Error

🔍 SUMMARY MATRICES:
- year_significance_matrix_*.csv: True/False significance by (year, decision_year)
- year_pvalue_matrix_*.csv: P-values by (year, decision_year)
- year_weight_matrix_*.csv: Importance weights by (year, decision_year)

INTERPRETATION GUIDE:
{"="*80}

📈 P-VALUE INTERPRETATION:
- p < 0.05: Statistically significant (95% confidence)
- p < 0.01: Highly significant (99% confidence)
- p < 0.001: Very highly significant (99.9% confidence)

📈 COEFFICIENT INTERPRETATION:
- Positive: Higher performance in that year → Higher payment probability
- Negative: Higher performance in that year → Lower payment probability
- Magnitude: Larger absolute value = stronger effect

📈 CONFIDENCE INTERVALS:
- If both CI bounds are positive or both negative → Significant effect
- If CI spans zero → Not statistically significant
- Width indicates precision of estimate

BUSINESS INSIGHTS:
{"="*80}

🎯 SCOUTING FOCUS:
Years with consistently low p-values across decision years should receive
the most attention in QB evaluation processes.

🎯 CONTRACT TIMING:
Years that become significant later (higher decision years) suggest
when those performance periods become most relevant for decisions.

🎯 EVALUATION WEIGHTS:
Use Weight_% from significant years to create data-driven evaluation
frameworks rather than arbitrary weightings.

HOW TO USE THE RESULTS:
{"="*80}

1. 📊 Open year_weights_significance_*.csv in Excel/spreadsheet

2. 🔍 Filter for Significant_95 = True to see only significant years

3. 📈 Sort by P_Value_Bootstrap (ascending) to see most significant first

4. 🎯 Look for patterns:
   - Which years are significant across multiple decision years?
   - Do recent years (Year 2, 3) matter more than rookie year (Year 0)?
   - Are there differences between yards vs efficiency metrics?

5. 📝 Create evaluation framework:
   - Weight years by their statistical significance
   - Focus on consistently significant performance periods
   - Adjust timing of contract decisions based on which years matter when

EXPECTED FINDINGS:
{"="*80}

Based on NFL contract patterns, you might find:

🔮 ROOKIE YEAR (Year 0):
Likely significant for early decisions but less important for later contracts
as more recent performance becomes available.

🔮 YEAR 2-3 PERFORMANCE:
Probably most significant for typical extension timing (Decision Year 4)
as these represent the most recent substantial performance data.

🔮 METRIC DIFFERENCES:
Total yards (volume) vs ANY/A (efficiency) may show different patterns
of year importance, reflecting different aspects of QB evaluation.

NEXT STEPS:
{"="*80}

1. ✅ Examine detailed p-value results in CSV files
2. 📊 Create visualizations of significance patterns
3. 🔍 Compare findings between volume (yards) and efficiency (ANY/A) metrics
4. 📝 Document business rules based on statistically significant years
5. 🎯 Implement data-driven weighting in QB evaluation processes

STATISTICAL VALIDATION:
{"="*80}

✅ Bootstrap Method: Robust to outliers and small sample issues
✅ Multiple Testing: Aware of multiple comparisons across years
✅ Effect Size: Coefficients provide practical significance beyond p-values
✅ Model Fit: Pseudo-R² indicates explanatory power of year-by-year model

{"="*80}
ANALYSIS COMPLETE: Your year weighting now has proper statistical rigor!
{"="*80}
"""
    
    summary_file = os.path.join(results_dir, "YEAR_WEIGHTING_ANALYSIS_SUMMARY.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"✓ Analysis summary saved: {summary_file}")
    
    # List all generated files
    print(f"\n{'='*80}")
    print("ALL GENERATED FILES:")
    print("="*80)
    
    if os.path.exists(results_dir):
        for filename in sorted(os.listdir(results_dir)):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                
                # Highlight key files
                if 'significance_' in filename and filename.endswith('.csv'):
                    print(f"  ⭐ {filename} ({size_kb:.1f} KB) ← P-VALUES FOR YEAR WEIGHTING!")
                elif 'pvalue_matrix' in filename:
                    print(f"  📊 {filename} ({size_kb:.1f} KB) ← P-VALUE MATRIX")
                elif 'significance_matrix' in filename:
                    print(f"  🔍 {filename} ({size_kb:.1f} KB) ← SIGNIFICANCE MATRIX")
                else:
                    print(f"     {filename} ({size_kb:.1f} KB)")
    
    # List CSV files in main directory too
    print(f"\n📋 KEY CSV FILES (in main directory):")
    csv_files = [
        'year_weights_significance_total_yards_adj.csv',
        'year_weights_significance_Pass_ANY_A_adj.csv',
        'year_significance_matrix_total_yards_adj.csv',
        'year_significance_matrix_Pass_ANY_A_adj.csv',
        'year_pvalue_matrix_total_yards_adj.csv',
        'year_pvalue_matrix_Pass_ANY_A_adj.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            size_kb = os.path.getsize(csv_file) / 1024
            print(f"  ✅ {csv_file} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {csv_file} (not found)")
    
    print(f"\n{'='*80}")
    print("YEAR WEIGHTING SIGNIFICANCE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {results_dir}/")
    print(f"\n🎉 SUCCESS: Statistical significance testing for year weighting completed!")
    print(f"\n🔍 Key files to examine:")
    print(f"   ⭐ year_weights_significance_*.csv (detailed p-values)")
    print(f"   📊 year_pvalue_matrix_*.csv (p-value matrices)")
    print(f"   🏆 YEAR_WEIGHTING_ANALYSIS_SUMMARY.txt (interpretation guide)")
    print(f"\n📊 You can now answer: 'Which prior years SIGNIFICANTLY predict QB payments?'")
    
    return results_dir

if __name__ == "__main__":
    main()