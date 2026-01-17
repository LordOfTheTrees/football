# Football Quarterback Research

A comprehensive Python package for analyzing NFL quarterback performance, contracts, and career trajectories. This repository provides tools for data collection, statistical analysis, era adjustment, injury projection, and predictive modeling of quarterback free-agent contracts.

## Overview

This project analyzes quarterback performance data with a focus on:
- **Era Adjustment**: Normalizing statistics across different NFL eras (accounting for rule changes, offensive trends, etc.)
- **Injury Projection**: Projecting per-game averages to full 16/17-game seasons for quarterbacks who missed games due to injury
- **Contract Prediction**: Using machine learning models to predict which quarterbacks will receive free-agent contracts
- **Trajectory Analysis**: Comparing quarterback career trajectories and finding similar players

## Key Features

### Data Adjustments

- **Era Adjustment** (`qb_research/adjustments/era_adjustment.py`): Adjusts statistics for inflation and rule changes since 2000
  - Creates `_adj` suffix columns (e.g., `total_yards_adj`, `Pass_TD_adj`)
  - Accounts for changes in passing efficiency, scoring, and game pace

- **Injury Projection** (`qb_research/adjustments/injury_projection.py`): Projects partial seasons to full seasons
  - Creates `_adj_proj` suffix columns (e.g., `total_yards_adj_proj`, `Pass_TD_adj_proj`)
  - Handles initial seasons (GS=0 before first start), relief roles (GS=0 after starting), and missing seasons
  - Fills complete season skips by averaging surrounding seasons

### Predictive Modeling

- **Payment Prediction** (`qb_research/modeling/prediction_models.py`): Logistic regression models to predict free-agent contracts
  - Compares era-adjusted vs. injury-projected statistics
  - Uses lag features, averaged features, and cross-validation
  - Results show injury-projected stats are more predictive (F1: +0.0109, Accuracy: +0.0097)

### Data Structure

The repository uses a modular package structure:

```
qb_research/
├── adjustments/      # Era adjustment and injury projection
├── analysis/         # Statistical analysis functions
├── comparisons/      # Trajectory matching and year weighting
├── data/            # Data loading, building, mapping, validation
├── exports/         # Tableau export functions
├── modeling/        # Prediction and surface models
├── preprocessing/   # Feature engineering
└── utils/           # Utilities (caching, data loading, name matching)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd football
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn scipy matplotlib beautifulsoup4 requests
```

3. Set up configuration:
```bash
cp config.example.py config.py
# Edit config.py with your API keys if needed
```

## Usage

### Regenerating Data with Adjustments

To create the master dataset with both era-adjusted and injury-projected statistics:

```bash
python regenerate_data_with_projections.py
```

This creates `qb_seasons_payment_labeled_era_adjusted.csv` with:
- `_adj` columns: Era-adjusted statistics
- `_adj_proj` columns: Era-adjusted + injury-projected statistics

### Running Injury Projection Analysis

To compare the predictiveness of era-adjusted vs. injury-projected stats:

```bash
python run_injury_projection_analysis.py
```

This outputs:
- Model performance metrics (F1-score, accuracy, AUC-ROC)
- Comparison CSV: `injury_projection_predictiveness_comparison.csv`

### Rebuilding All Seasons Pipeline

To rebuild the entire dataset from scratch:

```bash
python rebuild_all_seasons_pipeline.py
```

### Using the Package

```python
from qb_research.adjustments import create_era_adjusted_payment_data
from qb_research.modeling.prediction_models import compare_injury_projection_predictiveness

# Load data with adjustments
df = create_era_adjusted_payment_data()

# Run comparison analysis
results = compare_injury_projection_predictiveness()
```

## Data Files

### Input Files (Required)
- `qb_seasons_payment_labeled.csv`: Main dataset with QB seasons and contract labels
- `era_adjustment_factors.csv`: Pre-calculated era adjustment factors
- `season_averages.csv`: League-wide per-game averages by season (used for era adjustment)
- Individual QB files in `QB_Data/` directory (if using scraping pipeline)

### Output Files (Generated)
- `qb_seasons_payment_labeled_era_adjusted.csv`: Master dataset with all adjustments
- `injury_projection_predictiveness_comparison.csv`: Model comparison results
- `all_seasons_df.csv`: Consolidated seasons data
- `player_ids.csv`: Player ID mappings

## Data Sources

All data is scraped from [Pro-Football-Reference.com](https://www.pro-football-reference.com/):

### Season Averages (Per-Game)
- **Source URL**: `https://www.pro-football-reference.com/years/NFL/`
- **Purpose**: League-wide per-game averages used for era adjustment calculations
- **Data File**: `season_averages.csv`
- **Table Location**: "Team Offense League Averages Per Team Game" table
- **Update Frequency**: After each NFL season ends (typically February)

### Draft Data
- **Source URL Format**: `https://www.pro-football-reference.com/years/{YEAR}/draft.htm`
- **Example**: `https://www.pro-football-reference.com/years/2025/draft.htm`
- **Purpose**: Draft order and player information for determining draft position
- **Data File**: `draft_data/draft_class_{YEAR}.csv`
- **Update Frequency**: After each NFL draft (typically late April/early May)

### Player Statistics
- **Source URL Format**: `https://www.pro-football-reference.com/players/{FIRST_LETTER}/{PLAYER_ID}.htm`
- **Purpose**: Individual quarterback season-by-season statistics
- **Data Files**: Individual CSV files in `QB_Data/` directory
- **Update Frequency**: After each NFL season ends, or as needed for active players

## Key Concepts

### Era Adjustment
Statistics are adjusted to account for:
- Rule changes (e.g., pass interference, roughing the passer)
- Offensive trends (increased passing, scoring inflation)
- League-wide efficiency improvements

### Injury Projection Logic
1. **Initial seasons** (GS=0 before first start): Use era-adjusted values as-is
2. **After first start, partial seasons** (GS < season_length): Project per-game average to full season
3. **Full seasons** (GS >= season_length): Use era-adjusted values as-is
4. **Relief roles** (GS=0 after first start): Use era-adjusted values as-is
5. **Missing seasons** (GS=NaN): Fill by averaging surrounding seasons, or set to 0 if no return

### Nomenclature
- `_adj`: Era-adjusted statistics
- `_adj_proj`: Era-adjusted + injury-projected statistics
- `_lag1`, `_lag2`, `_lag3`: Lag features (previous 1-3 seasons)
- `_avg`: Averaged lag features

## Project Structure

### Main Scripts
- `regenerate_data_with_projections.py`: Regenerates master dataset
- `run_injury_projection_analysis.py`: Runs predictiveness comparison
- `rebuild_all_seasons_pipeline.py`: Full data rebuild pipeline
- `QB_research.py`: Legacy wrapper (maintains backwards compatibility)

### Package Modules
- **adjustments**: Era adjustment and injury projection
- **modeling**: Logistic regression, ridge regression, surface models
- **data**: Data loading, building, mapping, validation
- **preprocessing**: Feature engineering (lag features, averaged features)
- **comparisons**: Trajectory matching, year weighting
- **exports**: Tableau export utilities
- **utils**: Caching, data loading, name matching, debugging

## Results

Based on analysis of 65 first-round quarterbacks (2000-2020), the research has identified several critical patterns:

#### Era Adjustment & Predictive Metrics
- **Offensive inflation**: +6.84% in total yards, +17% in passing touchdowns from 2000-2024
- **Primary predictive metrics**: Total Yards (era-adjusted) and ANY/A (Adjusted Net Yards per Attempt)
- **Team factors**: Wins and team-dependent factors receive nearly equal weighting to individual performance metrics

#### Temporal Bias in Contract Decisions
- **Extreme recency bias**: Year 3 performance receives **59.8% weight** in Year 4 contract decisions (vs. 25% equal-weight baseline)
- **Critical performance thresholds**: 
  - ~**4,200 era-adjusted yards** and **6.5 ANY/A** represent inflection points where contract probability exceeds 60%
  - QBs failing to reach **6.0+ ANY/A by Year 3** almost never receive extensions from their drafting team

#### Quarterback Career Archetypes
Four distinct trajectory patterns identified:
1. **Elite Early Developers**: Explosive Year 1-2 breakthrough with sustained excellence (e.g., Mahomes, Burrow, Herbert)
2. **Steady Reliable Improvers**: Incremental growth with consistent production (e.g., Tannehill, Flacco, Ryan)
3. **Late Bloomers**: Slow start with dramatic Year 2+ emergence (e.g., Rodgers, Love)
4. **Early Peak**: Strong initial performance followed by plateau/decline (e.g., Mac Jones, Blake Bortles)

#### Leading Indicators of Success
- **Efficiency improvements**: QBs improving ANY/A by **+1.5 or more** between Years 0-2 signal developmental upside
- **Year 2 performance gap**: Paid QBs average 3,639 yards and 6.68 ANY/A in Year 2, vs. 2,453 yards and 5.31 ANY/A for unpaid QBs
- **Growth trajectory**: Paid QBs show average improvement of +1,016 yards from Year 0 to Year 2, vs. +214 yards for unpaid QBs


### Injury Projection Analysis

The injury projection analysis demonstrates that **injury-projected statistics are more predictive** of free-agent contracts:

- **F1-Score**: +0.0109 improvement (0.7748 → 0.7857)
- **Accuracy**: +0.0097 improvement (0.7573 → 0.7670)
- **AUC-ROC**: +0.0276 improvement (0.7702 → 0.7978)

This suggests that accounting for missed games due to injury provides a more accurate picture of quarterback performance when predicting contract outcomes.

## Contributing

This is a personal research project. For questions or suggestions, please open an issue.

## License

See `LICENSE` file for details.

## Notes

- The project uses Windows console encoding fixes for Unicode output
- Data files (CSV) are gitignored - regenerate them using the provided scripts
- The `backwards_compatibility/` folder contains migration notes from the old structure
- Debug utilities are available in `qb_research/utils/debug_utils.py`
