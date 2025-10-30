# QB Performance Analysis: Pre- vs Post-Era Adjustment Comparison

---

## 📊 Model Performance Summary

### Wins Prediction Model Performance

| Metric | Raw Data | Era-Adjusted | Change |
|--------|----------|--------------|--------|
| R² | 0.4931 | 0.4244 | -0.0687 |
| RMSE | 2.1261 | 2.2656 | +0.1395 |
| Variables | 13 | 7 | -6 |
| Significant (p<0.05) | 10 | 5 | -5 |

### Payment Prediction Model Performance

| Metric | Raw Data | Era-Adjusted | Change |
|--------|----------|--------------|--------|
| Accuracy | 0.7500 | 0.7062 | -0.0438 |
| ROC AUC | 0.8012 | 0.7944 | -0.0068 |
| Variables | 9 | 9 | 0 |
| Significant (p<0.05) | 3 | 3 | +0 |

---

## 🏈 Wins Prediction: Variable Comparison

| Variable | Raw Coef | p-value | Sig? | Era-Adj Coef | p-value | Sig? | Impact |
|----------|----------|---------|------|--------------|---------|------|--------|
| Pass_QBR | +0.619 | 0.0000 | ✓ | +0.779 | 0.0000 | ✓ | 🟢 Robust |
| Pass_GWD | +0.433 | 0.0000 | ✓ | N/A | N/A | ✓ | N/A |
| Rush_Rushing_Succ% | -0.380 | 0.0001 | ✓ | -0.260 | 0.0082 | ✓ | 🟢 Robust |
| Pass_ANY/A | +0.362 | 0.0000 | ✓ | +0.406 | 0.0000 | ✓ | 🟢 Robust |
| Pass_TD | +0.353 | 0.0000 | ✓ | +0.385 | 0.0000 | ✓ | 🟢 Robust |
| Pass_Rate | +0.297 | 0.0000 | ✓ | +0.297 | 0.0000 | ✓ | 🟢 Robust |
| Rush_Rushing_Yds | +0.207 | 0.0177 | ✓ | N/A | N/A | ✓ | N/A |
| Pass_Sk% | -0.201 | 0.0365 | ✓ | N/A | N/A | ✓ | N/A |
| Rush_Rushing_TD | +0.197 | 0.0268 | ✓ | N/A | N/A | ✓ | N/A |
| Pass_4QC | +0.175 | 0.0269 | ✓ | N/A | N/A | ✓ | N/A |
| Pass_Int% | -0.104 | 0.2856 | ✗ | N/A | N/A | ✓ | N/A |
| total_yards | -0.094 | 0.3150 | ✗ | +0.101 | 0.3240 | ✗ | ⚪ Not sig |
| Pass_Cmp% | -0.075 | 0.3893 | ✗ | -0.143 | 0.1306 | ✗ | ⚪ Not sig |

---

## 💰 Payment Prediction: Variable Comparison

| Variable | Raw Coef | p-value | Sig? | Era-Adj Coef | p-value | Sig? | Impact |
|----------|----------|---------|------|--------------|---------|------|--------|
| W-L%_avg | +0.829 | 0.0000 | ✓ | +0.663 | 0.0001 | ✓ | 🟢 Robust |
| Rush_Rushing_Succ%_avg | +0.747 | 0.0000 | ✓ | +0.712 | 0.0000 | ✓ | 🟢 Robust |
| total_yards_avg | +0.719 | 0.0211 | ✓ | +0.781 | 0.0114 | ✓ | 🟢 Robust |
| Pass_TD_avg | -0.313 | 0.3617 | ✗ | -0.344 | 0.2843 | ✗ | ⚪ Not sig |
| Pts_avg | -0.267 | 0.2409 | ✗ | -0.036 | 0.8556 | ✗ | ⚪ Not sig |
| Pass_Cmp%_avg | +0.263 | 0.2548 | ✗ | +0.305 | 0.2027 | ✗ | ⚪ Not sig |
| Pass_QBR_avg | +0.208 | 0.3962 | ✗ | +0.211 | 0.3528 | ✗ | ⚪ Not sig |
| Pass_Rate_avg | +0.168 | 0.6340 | ✗ | -0.153 | 0.6829 | ✗ | ⚪ Not sig |
| Pass_ANY/A_avg | +0.056 | 0.8564 | ✗ | +0.314 | 0.4308 | ✗ | ⚪ Not sig |

---

## 🔍 Key Findings

### Legend
- 🟢 **Robust**: Significant in both raw and era-adjusted models
- 🔵 **Era reveals**: Only significant after era adjustment
- 🟡 **Era hides**: Only significant in raw data
- ⚪ **Not sig**: Not significant in either model

### Wins Prediction Insights
- **Robust predictors** (5): Pass_ANY/A, Pass_QBR, Pass_Rate, Pass_TD, Rush_Rushing_Succ%
- **Era adjustment reveals** (0): None
- **Era adjustment hides** (0): None

### Payment Prediction Insights
- **Robust predictors** (3): Rush_Rushing_Succ%_avg, W-L%_avg, total_yards_avg
- **Era adjustment reveals** (0): None
- **Era adjustment hides** (0): None

---

## 📈 Statistical Impact Analysis

### Wins Prediction: Coefficient Changes

Variables ranked by absolute coefficient change after era adjustment:

| Variable | Raw Coef | Era Coef | Abs Change | % Change | Maintains Sig? |
|----------|----------|----------|------------|----------|----------------|
| total_yards | -0.094 | +0.101 | +0.195 | +207.2% | ✗ |
| Pass_QBR | +0.619 | +0.779 | +0.160 | +25.8% | ✓ |
| Rush_Rushing_Succ% | -0.380 | -0.260 | +0.119 | +31.5% | ✓ |
| Pass_Cmp% | -0.075 | -0.143 | -0.067 | -89.4% | ✗ |
| Pass_ANY/A | +0.362 | +0.406 | +0.044 | +12.2% | ✓ |
| Pass_TD | +0.353 | +0.385 | +0.032 | +9.1% | ✓ |
| Pass_Rate | +0.297 | +0.297 | -0.000 | -0.1% | ✓ |

### Payment Prediction: Coefficient Changes

Variables ranked by absolute coefficient change after era adjustment:

| Variable | Raw Coef | Era Coef | Abs Change | % Change | Maintains Sig? |
|----------|----------|----------|------------|----------|----------------|
| Pass_Rate_avg | +0.168 | -0.153 | -0.321 | -191.5% | ✗ |
| Pass_ANY/A_avg | +0.056 | +0.314 | +0.258 | +461.7% | ✗ |
| Pts_avg | -0.267 | -0.036 | +0.231 | +86.6% | ✗ |
| W-L%_avg | +0.829 | +0.663 | -0.166 | -20.0% | ✓ |
| total_yards_avg | +0.719 | +0.781 | +0.062 | +8.6% | ✓ |
| Pass_Cmp%_avg | +0.263 | +0.305 | +0.042 | +16.0% | ✗ |
| Rush_Rushing_Succ%_avg | +0.747 | +0.712 | -0.035 | -4.7% | ✓ |
| Pass_TD_avg | -0.313 | -0.344 | -0.030 | -9.7% | ✗ |
| Pass_QBR_avg | +0.208 | +0.211 | +0.003 | +1.6% | ✗ |

---

## 🎯 Executive Summary

### Wins Prediction Model

**Model Fit**: Raw data (R²=0.493) slightly outperforms era-adjusted (R²=0.424)

**Key Takeaways**:

1. **5 robust predictors** maintain significance across both models:
   - Pass_QBR
   - Pass_ANY/A
   - Pass_TD
   - Pass_Rate
   - Rush_Rushing_Succ%

2. **6 variables** not available in era-adjusted model (likely due to data availability)

### Payment Prediction Model

**Model Fit**: Raw data (AUC=0.801) slightly outperforms era-adjusted (AUC=0.794)

**Key Takeaways**:

1. **3 robust predictors** maintain significance across both models:
   - total_yards_avg
   - Rush_Rushing_Succ%_avg
   - W-L%_avg

2. **Largest coefficient shifts** after era adjustment:
   - Pass_ANY/A_avg: increased by 0.258 (+461.7%)
   - Pts_avg: increased by 0.231 (+86.6%)
   - total_yards_avg: increased by 0.062 (+8.6%)

---

## 💡 Recommendations

### For Model Selection

1. **Wins Prediction**: Use **raw data model** for better predictive accuracy
   - Higher R² (0.493 vs 0.424)
   - Lower RMSE (2.13 vs 2.27)
   - More variables available (13 vs 7)

2. **Payment Prediction**: Use **raw data model** for better classification
   - Higher accuracy (0.750 vs 0.706)
   - Higher ROC AUC (0.801 vs 0.794)

### For Variable Selection

**Focus on robust predictors that maintain significance:**

**Wins Prediction:**
- `Pass_QBR` (positive impact)
- `Pass_ANY/A` (positive impact)
- `Pass_TD` (positive impact)
- `Pass_Rate` (positive impact)
- `Rush_Rushing_Succ%` (negative impact)

**Payment Prediction:**
- `total_yards_avg` (increases payment probability)
- `Rush_Rushing_Succ%_avg` (increases payment probability)
- `W-L%_avg` (increases payment probability)

---

## 📝 Methodology Notes

- **Significance Testing**: Bootstrap-based p-values (500 iterations)
- **Significance Threshold**: p < 0.05 (95% confidence)
- **Coefficient Interpretation**:
  - Wins: Effect on win total per 1 SD increase in predictor
  - Payment: Log-odds effect on payment probability (logistic regression)
- **Era Adjustment**: Controls for league-wide statistical inflation over time