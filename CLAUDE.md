# CLAUDE.md - AI Assistant Guide for NFL QB Research Repository

**Last Updated**: 2026-01-10
**Project**: NFL Quarterback Contract Prediction and Trajectory Analysis System

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Key Data Flows and Pipelines](#key-data-flows-and-pipelines)
4. [Coding Conventions and Standards](#coding-conventions-and-standards)
5. [Development Workflows](#development-workflows)
6. [Testing and Validation](#testing-and-validation)
7. [Common Tasks Guide](#common-tasks-guide)
8. [Important Files Reference](#important-files-reference)
9. [Data Schemas and Structures](#data-schemas-and-structures)
10. [Dependencies and External Services](#dependencies-and-external-services)
11. [Troubleshooting Guide](#troubleshooting-guide)

---

## Project Overview

### Purpose

This is a **QB contract prediction and trajectory analysis system** designed to:

- Analyze first-round QB draft performance trajectories across their careers
- Predict QB contract decisions (when/if QBs receive big contract extensions)
- Compare similar QBs based on performance metrics and predict future development
- Identify patterns between performance metrics and contract outcomes
- Adjust for era inflation in statistics across different decades of NFL play
- Enable interactive analysis through comparison tools and Tableau exports

### Target Audience

- NFL teams evaluating QB contracts
- Analysts studying QB salary cap impact
- Researchers examining draft capital value

### Project Status

- **Recent Major Refactoring**: The codebase was recently refactored from a monolithic `QB_research.py` file into a modular package structure under `qb_research/`
- **Backwards Compatibility**: `QB_research.py` now acts as a thin wrapper that imports and re-exports all functions, maintaining compatibility with existing scripts
- **Active Development**: See `to_work_on.txt` for planned features and investigations

---

## Repository Structure

```
football/
├── QB_research.py                      # Main entry point (wrapper/re-exports from qb_research package)
├── PFR_Tools.py                        # Pro Football Reference web scraping utilities
├── config.example.py                   # Configuration template (copy to config.py)
├── data_structure_schema               # Documentation of CSV data structures
├── to_work_on.txt                      # TODO list and future work
│
├── qb_research/                        # Modular package (NEW STRUCTURE)
│   ├── __init__.py
│   ├── utils/                          # Low-level utilities
│   │   ├── data_loading.py            # Safe CSV loading, validation
│   │   ├── name_matching.py           # Player name normalization
│   │   ├── caching.py                 # File caching utilities
│   │   ├── debug_utils.py             # Debugging helpers
│   │   └── exploratory.py             # Exploratory data analysis
│   ├── data/                           # Data builders, loaders, mappers
│   │   ├── builders.py                # Build datasets from source files
│   │   ├── loaders.py                 # Load processed datasets
│   │   ├── mappers.py                 # Map between data sources
│   │   └── validators.py              # Data quality validation
│   ├── preprocessing/                  # Feature engineering
│   │   └── feature_engineering.py     # Create features for modeling
│   ├── analysis/                       # Statistical analysis
│   │   ├── statistical_analysis.py    # Correlations, PCA, regression
│   │   └── era_adjustment.py          # Cross-era normalization
│   ├── modeling/                       # ML models
│   │   ├── prediction_models.py       # Logistic/Ridge regression
│   │   └── surface_models.py          # KNN probability surfaces
│   ├── comparisons/                    # Trajectory matching
│   │   ├── trajectory_matching.py     # Find similar QBs
│   │   └── year_weighting.py          # Year importance analysis
│   └── exports/                        # Output generation
│       └── tableau_exports.py         # Tableau-friendly exports
│
├── backwards_compatibility/            # Old code for reference (see README)
│
├── rebuild_all_seasons_pipeline.py     # Rebuild all_seasons_df from QB_Data/
├── qb_comp_tool.py                     # Interactive QB comparison tool
├── data_validation_scripts.py          # Data quality validation
├── plot_qb_variables_by_year.py        # Visualization script
├── standings_scraper.py                # NFL standings scraper
├── wins_prediction_fixed.py            # Wins prediction modeling
├── human_like_requester.py             # Respectful web scraping helper
│
└── .gitignore                          # Excludes CSV files, cache, outputs
```

### Key Directories (gitignored - generated at runtime)

- `QB_Data/` - Individual QB CSV files (one per player)
- `cache/` - Cached mappings and processed data
- `comp_analysis_output/` - QB comparison outputs
- `KNN_surfaces/` - 2D probability surface data
- `payment_heat_map_probabilities/` - Payment prediction heatmaps
- Various output directories for analysis results

---

## Key Data Flows and Pipelines

### High-Level Data Pipeline

```
1. WEB SCRAPING (PFR_Tools.py)
   ↓ Individual QB files
   QB_Data/*.csv

2. DATASET BUILDING (rebuild_all_seasons_pipeline.py)
   ↓ Consolidate all QB seasons
   all_seasons_df.csv

3. ID MAPPING (qb_research/data/builders.py)
   ↓ Create player ID lookup
   player_ids.csv, first_round_qbs.csv, first_round_qbs_with_picks.csv

4. CONTRACT INTEGRATION (qb_research/data/mappers.py)
   ↓ Map contracts to player IDs
   QB_contract_data.csv → contract_player_mapping.csv

5. PAYMENT LABELING (qb_research/preprocessing/feature_engineering.py)
   ↓ Label seasons relative to payment
   qb_seasons_payment_labeled.csv

6. ERA ADJUSTMENT (qb_research/analysis/era_adjustment.py)
   ↓ Normalize stats to 2024 baseline
   era_adjustment_factors.csv
   qb_seasons_payment_labeled_era_adjusted.csv

7. FEATURE ENGINEERING (qb_research/preprocessing/feature_engineering.py)
   ↓ Add lag features (1-4 years prior performance)
   Final modeling dataset

8. MODELING & ANALYSIS
   ├── Payment prediction (qb_research/modeling/prediction_models.py)
   ├── Trajectory matching (qb_research/comparisons/trajectory_matching.py)
   ├── Surface models (qb_research/modeling/surface_models.py)
   └── Tableau exports (qb_research/exports/tableau_exports.py)
```

### Critical Data Transformations

#### 1. Payment Year Labeling

Seasons are labeled relative to when QB received their big contract:

- `years_to_payment`: Signed integer (-4, -3, -2, -1, 0, 1, 2, ...)
  - Negative = years before payment
  - 0 = payment year
  - Positive = years after payment
- `got_paid`: Boolean flag
- `payment_year`: Year of contract signing

#### 2. Era Adjustment

Statistics are normalized to 2024 baseline using linear regression:

```python
# For each stat (e.g., total_yards):
league_avg_by_year = fit_linear_trend(year, league_avg_stat)
adjustment_factor = 2024_predicted_avg / year_avg
stat_adj = stat_raw * adjustment_factor
```

Applied to: `total_yards`, `Pass_TD`, `Pass_Yds`, `Pass_ANY/A`, `Pts`, `Rush_Rushing_Succ%`

#### 3. Lag Features

Prior performance metrics (1-4 years back):

- `Pass_ANY/A_lag1` = Last year's ANY/A
- `Pass_ANY/A_lag2` = Two years ago ANY/A
- `total_yards_lag3` = Three years ago total yards
- etc.

Enables models to see performance trends leading to payment decisions.

---

## Coding Conventions and Standards

### Naming Conventions

#### Player Identifiers

- **player_id**: 8-character Pro Football Reference ID (e.g., `MahoPa00`, `AlleJo02`)
- **player_name**: Full name string (e.g., "Patrick Mahomes", "Josh Allen")

#### Metric Suffixes

- `_adj`: Era-adjusted version (e.g., `total_yards_adj`)
- `_lagN`: Value from N years ago (e.g., `Pass_Rate_lag1`, `Pass_TD_lag4`)
- No suffix: Raw value from that season

#### Column Prefixes (from Pro Football Reference structure)

- `Pass_*`: Passing statistics
- `Rush_*`: Rushing statistics
- `AdvPass_*`: Advanced passing statistics
- No prefix: General season info (season, Age, Team, etc.)

### Code Style

#### Import Organization

```python
# Standard library
import pandas as pd
import numpy as np
import os
import time

# Third-party
from sklearn.linear_model import LogisticRegression
from scipy import stats

# Local/package
import PFR_Tools as PFR
from qb_research.utils.data_loading import load_csv_safe
from qb_research.data.builders import create_all_seasons_from_existing_qb_files
```

#### Function Documentation

Use docstrings with clear parameter and return descriptions:

```python
def example_function(player_id, metric='Pass_ANY/A'):
    """
    Brief description of what function does.

    Args:
        player_id (str): 8-character PFR player ID
        metric (str): Performance metric to analyze (default: 'Pass_ANY/A')

    Returns:
        pd.DataFrame: Processed data with columns [...]
        or None if processing fails
    """
```

#### Error Handling

- Use graceful degradation with detailed logging
- Print diagnostic information liberally (this is a research codebase)
- Return `None` on critical failures
- Show sample data after transformations for validation

```python
try:
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows from {file_path}")
except FileNotFoundError:
    print(f"✗ ERROR: File not found: {file_path}")
    return None
```

### Data Validation Patterns

Always validate input data:

```python
from qb_research.utils.data_loading import load_csv_safe, validate_columns

df = load_csv_safe('data.csv')
if df is None:
    return None

required_cols = ['player_id', 'season', 'Pass_ANY/A']
if not validate_columns(df, required_cols):
    print(f"Missing required columns: {set(required_cols) - set(df.columns)}")
    return None
```

### Module Organization Best Practices

- **utils/**: Pure functions with no side effects, reusable across modules
- **data/**: Functions that read/write CSV files
- **preprocessing/**: Functions that transform data for modeling
- **modeling/**: Functions that fit/predict with ML models
- **analysis/**: Functions that generate insights/statistics
- **comparisons/**: Functions that compare QBs
- **exports/**: Functions that generate output formats (Tableau, CSV, plots)

---

## Development Workflows

### Setting Up Development Environment

1. **Clone repository**:
   ```bash
   git clone <repo_url>
   cd football
   ```

2. **Create config file**:
   ```bash
   cp config.example.py config.py
   # Edit config.py with your API keys if needed
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib requests beautifulsoup4
   ```

4. **Verify data exists** (if starting fresh, see "Rebuilding Data Pipeline" below):
   - Check for `QB_Data/*.csv` (individual QB files)
   - Run `rebuild_all_seasons_pipeline.py` to generate master datasets

### Adding New Features

#### Adding a New Metric to Analysis

1. **Update data pipeline** if metric doesn't exist yet:
   - Check `data_structure_schema` to see if metric is in existing CSVs
   - If not, update scraping in `PFR_Tools.py` to capture it

2. **Add era adjustment** (if needed):
   - Edit `qb_research/analysis/era_adjustment.py`
   - Add metric to `calculate_era_adjustment_factors()` function
   - Add to `apply_era_adjustments()` function

3. **Add lag features** (if needed):
   - Edit `qb_research/preprocessing/feature_engineering.py`
   - Add metric to `create_lookback_performance_features()` function

4. **Update models**:
   - Edit `qb_research/modeling/prediction_models.py` to include new metric in feature list

5. **Update exports**:
   - Edit `qb_research/exports/tableau_exports.py` to include in Tableau export

#### Adding a New Analysis Function

1. Create function in appropriate module (e.g., `qb_research/analysis/statistical_analysis.py`)
2. Add import to `qb_research/analysis/__init__.py`
3. Add re-export to `QB_research.py` (for backwards compatibility)
4. Create a standalone script if interactive (like `qb_comp_tool.py`)
5. Test with sample data

### Making Changes to Package Structure

**CRITICAL**: When adding/moving functions in `qb_research/` package:

1. Update the module's `__init__.py` to export the function
2. Update `QB_research.py` to import and re-export (backwards compatibility)
3. Update any standalone scripts that import the function

Example:

```python
# qb_research/analysis/new_module.py
def new_analysis_function():
    pass

# qb_research/analysis/__init__.py
from .new_module import new_analysis_function

# QB_research.py
from qb_research.analysis import new_analysis_function
```

### Git Workflow

#### Committing Changes

Follow existing conventions:

- **Clear commit messages**: Describe what changed and why
- **Logical commits**: Group related changes together
- **Update to_work_on.txt**: Remove completed items, add new discoveries

#### Branch Strategy

- Work on designated branch: `claude/claude-md-mk8slcwcswmf1wr2-7xp9Q`
- Always develop on this branch (don't create new branches without permission)
- Push to origin when changes are complete: `git push -u origin <branch-name>`

### Rebuilding Data Pipeline

If you need to rebuild the entire data pipeline from scratch:

```bash
# 1. Ensure QB_Data/*.csv files exist (scraped from PFR)
python PFR_Tools.py  # Run scraping if needed

# 2. Rebuild master season file
python rebuild_all_seasons_pipeline.py

# 3. Run QB_research.py functions to rebuild derived datasets
python -c "from QB_research import create_era_adjusted_payment_data; create_era_adjusted_payment_data()"

# 4. Validate data quality
python data_validation_scripts.py
```

---

## Testing and Validation

### Data Validation Functions

Located in `data_validation_scripts.py`:

- `check_qb_coverage()`: Validates all QBs from all_seasons_df.csv are in trajectory export
- `validate_contract_mapping()`: Checks if contracts properly mapped to player IDs
- `test_name_mapping()`: Unit tests on player name normalization

### Validation During Processing

All major pipeline steps include validation:

```python
# In qb_research/preprocessing/feature_engineering.py
validate_payment_data(df)  # Checks payment_year >= draft_year, flags outliers

# In qb_research/utils/data_loading.py
validate_columns(df, required_columns)  # Ensures required columns exist
validate_payment_years(df)  # Checks payment year logic
```

### Interactive Validation Tools

- **qb_comp_tool.py**: Interactive trajectory comparison for manual verification
- **plot_qb_variables_by_year.py**: Visual scatter plots to spot anomalies
- **data_validation_scripts.py**: Run comprehensive validation suite

### Quality Assurance Checklist

Before finalizing major changes:

1. [ ] Run `python rebuild_all_seasons_pipeline.py` successfully
2. [ ] Run `python data_validation_scripts.py` - all checks pass
3. [ ] Spot-check 3-5 random QBs in generated CSVs
4. [ ] Verify era adjustments are reasonable (check adjustment factors)
5. [ ] Test interactive tools still work (qb_comp_tool.py)
6. [ ] Verify backwards compatibility (existing scripts run without changes)

---

## Common Tasks Guide

### Task: Find Similar QBs to a Current Player

```python
from QB_research import find_comps_both_metrics

# Find QBs with similar trajectory to Josh Allen through Year 3
similar_qbs = find_comps_both_metrics(
    player_id='AlleJo02',
    years_to_compare=3,
    n_comps=10,
    metric1='total_yards_adj',
    metric2='Pass_ANY/A_adj'
)

print(similar_qbs)
```

Or use interactive tool:

```bash
python qb_comp_tool.py
# Follow prompts to enter QB name and comparison parameters
```

### Task: Predict Payment Probability for a QB

```python
from QB_research import ridge_regression_payment_prediction

# Train model on historical data and predict
model_results = ridge_regression_payment_prediction(
    decision_year=4,  # Predict at year 4 decision point
    metrics=['Pass_ANY/A_adj', 'total_yards_adj'],
    include_team_metrics=True
)

# Get prediction for specific QB
# (Extract QB's year 4 stats and apply model)
```

### Task: Analyze Era Adjustment Impact

```python
from QB_research import calculate_era_adjustment_factors, apply_era_adjustments

# Calculate adjustment factors
factors = calculate_era_adjustment_factors('season_averages.csv')
print(factors[factors['stat'] == 'total_yards'])

# Apply to data
df_adjusted = apply_era_adjustments(df_raw, factors)

# Compare before/after
print(df_raw[['player_name', 'season', 'total_yards']].head())
print(df_adjusted[['player_name', 'season', 'total_yards_adj']].head())
```

### Task: Generate Tableau Export

```python
from QB_research import create_qb_trajectories_for_tableau

# Create Tableau-friendly long-format data
tableau_df = create_qb_trajectories_for_tableau(
    payment_labeled_df='qb_seasons_payment_labeled_era_adjusted.csv',
    qb_ids_df='first_round_qbs_with_picks.csv'
)

tableau_df.to_csv('qb_trajectories_for_tableau.csv', index=False)
print(f"Exported {len(tableau_df)} rows for Tableau")
```

### Task: Update Data for Current Season

1. **Scrape latest season data**:
   ```bash
   python PFR_Tools.py  # Update scraping logic for current year
   ```

2. **Rebuild pipeline**:
   ```bash
   python rebuild_all_seasons_pipeline.py
   ```

3. **Update contract data** (manual):
   - Edit `QB_contract_data.csv` with new contracts signed
   - Ensure format matches existing rows

4. **Regenerate era adjustments** (baseline shifts with new year):
   ```python
   from QB_research import create_era_adjusted_payment_data
   create_era_adjusted_payment_data(force_refresh=True)
   ```

### Task: Debug Name Matching Issues

```python
from qb_research.utils.name_matching import debug_name_matching, normalize_player_name

# Check how a name is normalized
print(normalize_player_name("Patrick Mahomes II"))
# Output: patrick mahomes

# Debug why a QB isn't matching between datasets
debug_name_matching(
    qb_name="Josh Allen",
    all_seasons_df=all_seasons,
    contract_df=contracts
)
# Prints normalized forms and potential matches
```

### Task: Add a New QB to Analysis (Mid-Season Update)

1. **Scrape individual QB file**:
   ```bash
   # Manually run PFR scraping for specific player
   # Save to QB_Data/PlayerID.csv
   ```

2. **Rebuild all_seasons_df**:
   ```bash
   python rebuild_all_seasons_pipeline.py
   ```

3. **Update first-round QB list** (if applicable):
   - Edit `first_round_qbs_with_picks.csv` to include new QB
   - Add draft year, team, pick number

4. **Regenerate payment labels**:
   ```python
   from QB_research import prepare_qb_payment_data
   df_labeled = prepare_qb_payment_data(force_refresh=True)
   ```

---

## Important Files Reference

### Core Data Files (Generated)

| File | Description | Generated By |
|------|-------------|--------------|
| `all_seasons_df.csv` | All QB seasons consolidated | `rebuild_all_seasons_pipeline.py` |
| `best_seasons_df.csv` | Best season per QB (by total_yards) | `qb_research/data/builders.py` |
| `player_ids.csv` | QB name to player_id mapping | `qb_research/data/builders.py` |
| `first_round_qbs.csv` | List of first-round QBs | `qb_research/data/loaders.py` |
| `first_round_qbs_with_picks.csv` | First-round QBs with pick numbers | Manual + `qb_research/data/mappers.py` |
| `QB_contract_data.csv` | QB contract terms | Manual data entry |
| `season_averages.csv` | League-wide statistics by year | Manual/scraped |
| `season_records.csv` | Team records by season | `standings_scraper.py` |
| `qb_seasons_payment_labeled.csv` | QB seasons with payment labels | `qb_research/preprocessing/feature_engineering.py` |
| `qb_seasons_payment_labeled_era_adjusted.csv` | Payment-labeled + era-adjusted | `qb_research/analysis/era_adjustment.py` |
| `era_adjustment_factors.csv` | Adjustment factors by stat/year | `qb_research/analysis/era_adjustment.py` |
| `qb_trajectories_for_tableau.csv` | Long-format for Tableau | `qb_research/exports/tableau_exports.py` |

### Core Python Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `QB_research.py` | Main entry point, re-exports all functions | All public API functions |
| `PFR_Tools.py` | Web scraping Pro Football Reference | `scrape_qb_stats()`, `extract_table_from_page()` |
| `qb_research/utils/data_loading.py` | Safe data loading utilities | `load_csv_safe()`, `validate_columns()` |
| `qb_research/utils/name_matching.py` | Player name normalization | `normalize_player_name()`, `debug_name_matching()` |
| `qb_research/data/builders.py` | Build datasets from source | `create_all_seasons_from_existing_qb_files()` |
| `qb_research/data/loaders.py` | Load processed datasets | `load_train_test_split()`, `load_contract_data()` |
| `qb_research/data/mappers.py` | Map between data sources | `map_contract_to_player_ids()` |
| `qb_research/preprocessing/feature_engineering.py` | Feature creation | `label_seasons_relative_to_payment()`, `create_lookback_performance_features()` |
| `qb_research/analysis/era_adjustment.py` | Cross-era normalization | `calculate_era_adjustment_factors()`, `apply_era_adjustments()` |
| `qb_research/modeling/prediction_models.py` | ML models | `ridge_regression_payment_prediction()` |
| `qb_research/modeling/surface_models.py` | KNN probability surfaces | `create_2d_payment_probability_surface()` |
| `qb_research/comparisons/trajectory_matching.py` | Find similar QBs | `find_comps_both_metrics()`, `find_most_similar_qbs()` |

### Standalone Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `rebuild_all_seasons_pipeline.py` | Rebuild master QB season file | `python rebuild_all_seasons_pipeline.py` |
| `qb_comp_tool.py` | Interactive QB comparison | `python qb_comp_tool.py` |
| `data_validation_scripts.py` | Data quality checks | `python data_validation_scripts.py` |
| `plot_qb_variables_by_year.py` | Visual scatter plots | `python plot_qb_variables_by_year.py` |
| `standings_scraper.py` | Scrape NFL standings | `python standings_scraper.py` |
| `wins_prediction_fixed.py` | Wins prediction modeling | `python wins_prediction_fixed.py` |

---

## Data Schemas and Structures

See `data_structure_schema` file for comprehensive examples. Key schemas:

### all_seasons_df.csv (QB Season Stats)

**Core Columns**:
- `player_name`: Full name (string)
- `player_id`: 8-char PFR ID (string)
- `season`: Year (int)
- `Age`: Player age (int)
- `Team`: 3-letter team code (string)
- `G`, `GS`: Games, Games Started (int)

**Passing Stats** (prefix `Pass_`):
- `Pass_Cmp`, `Pass_Att`, `Pass_Cmp%`: Completions, Attempts, Percentage
- `Pass_Yds`, `Pass_TD`, `Pass_Int`: Yards, Touchdowns, Interceptions
- `Pass_Y/A`, `Pass_AY/A`, `Pass_ANY/A`: Yards per Attempt variations
- `Pass_Rate`, `Pass_QBR`: Passer Rating, QBR
- `Pass_Sk`, `Pass_Yds.1`, `Pass_Sk%`: Sacks, Sack Yards, Sack %

**Rushing Stats** (prefix `Rush_Rushing_`):
- `Rush_Rushing_Att`, `Rush_Rushing_Yds`, `Rush_Rushing_TD`: Attempts, Yards, Touchdowns
- `Rush_Rushing_Y/A`, `Rush_Rushing_Y/G`: Yards per Attempt, per Game
- `Rush_Rushing_Succ%`: Success rate

**Advanced Passing** (prefix `AdvPass_`):
- `AdvPass_Air Yards_IAY`: Intended Air Yards
- `AdvPass_Air Yards_CAY`: Completed Air Yards
- `AdvPass_Air Yards_YAC`: Yards After Catch
- `AdvPass_Accuracy_Drops`, `AdvPass_Accuracy_BadTh`: Drops, Bad Throws
- `AdvPass_Pressure_Prss%`: Pressure Percentage

**Derived**:
- `total_yards`: Pass_Yds + Rush_Rushing_Yds
- `draft_year`, `draft_team`: Draft metadata

### QB_contract_data.csv

**Columns**:
- `Player`: Full name
- `Team`: Team abbreviation
- `Year`: Contract signing year
- `Years`: Contract length
- `Value`: Total contract value ($)
- `APY`: Average per year ($)
- `Guaranteed`: Guaranteed money ($)
- `APY as % Of`: APY as % of salary cap

### first_round_qbs_with_picks.csv

**Columns**:
- `player_name`: Full name
- `draft_year`: Year drafted
- `draft_team`: Team that drafted
- `pick_number`: Overall pick number (1-32)

### qb_seasons_payment_labeled_era_adjusted.csv

All columns from `all_seasons_df.csv` plus:

**Payment Labels**:
- `got_paid`: Boolean
- `payment_year`: Year of contract
- `years_to_payment`: Signed int (-4 to +10)
- `pick_number`: Draft pick

**Lag Features** (1-4 years prior):
- `Pass_ANY/A_lag1`, `Pass_ANY/A_lag2`, `Pass_ANY/A_lag3`, `Pass_ANY/A_lag4`
- `Pass_Rate_lag1`, ..., `Pass_Rate_lag4`
- `Pass_TD_lag1`, ..., `Pass_TD_lag4`
- `Pass_Yds_lag1`, ..., `Pass_Yds_lag4`
- `total_yards_lag1`, ..., `total_yards_lag4`

**Era-Adjusted Metrics** (suffix `_adj`):
- `total_yards_adj`
- `Pass_TD_adj`
- `Pass_ANY/A_adj`
- `Rush_Rushing_Succ%_adj`

### Team Code Mapping

**35 Unique Team Codes**:
- `'ARI'` = Arizona Cardinals
- `'ATL'` = Atlanta Falcons
- `'BAL'` = Baltimore Ravens
- `'BUF'` = Buffalo Bills
- `'CAR'` = Carolina Panthers
- `'CHI'` = Chicago Bears
- `'CIN'` = Cincinnati Bengals
- `'CLE'` = Cleveland Browns
- `'DAL'` = Dallas Cowboys
- `'DEN'` = Denver Broncos
- `'DET'` = Detroit Lions
- `'GNB'` = Green Bay Packers
- `'HOU'` = Houston Texans
- `'IND'` = Indianapolis Colts
- `'JAX'` = Jacksonville Jaguars
- `'KAN'` = Kansas City Chiefs
- `'LAC'` = LA Chargers (formerly SDG)
- `'LAR'` = LA Rams (formerly STL)
- `'LVR'` = Las Vegas Raiders (formerly OAK)
- `'MIA'` = Miami Dolphins
- `'MIN'` = Minnesota Vikings
- `'NOR'` = New Orleans Saints
- `'NWE'` = New England Patriots
- `'NYG'` = New York Giants
- `'NYJ'` = New York Jets
- `'OAK'` = Oakland Raiders (now LVR)
- `'PHI'` = Philadelphia Eagles
- `'PIT'` = Pittsburgh Steelers
- `'SDG'` = San Diego Chargers (now LAC)
- `'SEA'` = Seattle Seahawks
- `'SFO'` = San Francisco 49ers
- `'STL'` = St. Louis Rams (now LAR)
- `'TAM'` = Tampa Bay Buccaneers
- `'TEN'` = Tennessee Titans
- `'WAS'` = Washington Commanders/Football Team
- `'2TM'` = Played for 2 teams in same season (rare)

---

## Dependencies and External Services

### Python Packages Required

**Core Data Science**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations

**Machine Learning**:
- `scikit-learn` - Models, scaling, PCA, cross-validation
- `scipy` - Statistical tests, distributions

**Web Scraping**:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing

**Visualization**:
- `matplotlib` - Plotting and visualizations

**Utilities**:
- `argparse` - Command-line parsing
- Standard library: `os`, `time`, `random`, `datetime`, `glob`, `io`, `traceback`

### External Data Sources

| Source | Purpose | Access Method |
|--------|---------|---------------|
| **Pro Football Reference** | QB stats, draft info | Web scraping (`PFR_Tools.py`) |
| **NFL.com / ESPN** | Season standings, team stats | Web scraping (`standings_scraper.py`) |
| **Manual Entry** | Contract data | CSV file (`QB_contract_data.csv`) |

### Rate Limiting and Respectful Scraping

The `HumanLikeRequester` class (`human_like_requester.py`) implements:

- **Random delays**: 3-7 seconds between requests (default)
- **Long delays**: 10-20 seconds (10% of requests), 30-60 seconds (1% of requests)
- **Rotating user agents**: Simulates different browsers
- **Session persistence**: Maintains cookies across requests
- **Referer headers**: Realistic browsing behavior

**Best Practices**:
- Always use `requester.get(url)` instead of direct `requests.get()`
- Respect robots.txt
- Cache results to minimize repeated requests
- Run scraping during off-peak hours
- Monitor for rate limit errors (HTTP 429)

### Configuration

**config.py** (create from `config.example.py`):
- `API_KEY`: Reserved for future API integrations
- `DATABASE_URL`: Reserved for future database connections
- `ANTHROPIC_API_KEY`: For AI-assisted analysis (currently imported but not actively used)

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "Player not found" or Name Matching Failures

**Symptoms**: QB present in one dataset but not matching in another

**Diagnosis**:
```python
from qb_research.utils.name_matching import debug_name_matching
debug_name_matching("Josh Allen", all_seasons_df, contract_df)
```

**Common Causes**:
1. Name suffixes (Jr, II, III) causing mismatches
2. Extra spaces or capitalization differences
3. Nickname vs. legal name (e.g., "Pat" vs "Patrick")
4. Typos in manual data entry

**Solutions**:
- Check normalized form: `normalize_player_name("Patrick Mahomes II")` → `"patrick mahomes"`
- Update name in CSV to match Pro Football Reference exactly
- Add custom mapping in `qb_research/utils/name_matching.py` if persistent

#### Issue: "Missing required columns" Error

**Symptoms**: `validate_columns()` fails, script crashes

**Diagnosis**:
```python
print(df.columns.tolist())
# Compare to expected columns in error message
```

**Common Causes**:
1. CSV file from old version of pipeline
2. Manual CSV edit removed columns
3. Column name typo in code

**Solutions**:
- Regenerate CSV from pipeline: `python rebuild_all_seasons_pipeline.py`
- Check column names match `data_structure_schema` exactly
- Update validation function if intentionally changed schema

#### Issue: Era Adjustments Seem Wrong

**Symptoms**: Adjusted values are extreme or inverted

**Diagnosis**:
```python
from QB_research import calculate_era_adjustment_factors
factors = calculate_era_adjustment_factors('season_averages.csv')
print(factors[factors['stat'] == 'total_yards'])
# Check R² and adjustment factors by year
```

**Common Causes**:
1. Missing years in season_averages.csv
2. Outlier year skewing linear fit
3. Stat not actually trending over time (R² < 0.5)

**Solutions**:
- Update season_averages.csv with missing years
- Visual inspection: `python plot_qb_variables_by_year.py`
- Consider removing stat from era adjustment if R² low

#### Issue: QB Missing from Trajectory Export

**Symptoms**: QB in all_seasons_df.csv but not in qb_trajectories_for_tableau.csv

**Diagnosis**:
```bash
python data_validation_scripts.py
# Look for missing QB reports
```

**Common Causes**:
1. QB not in first_round_qbs.csv (not a first-round pick)
2. Missing draft_year or draft_team metadata
3. Filtered out due to insufficient seasons

**Solutions**:
- Add QB to first_round_qbs_with_picks.csv if they are first-round
- Update QB_Data/{player_id}.csv with draft metadata
- Check filter criteria in `create_qb_trajectories_for_tableau()`

#### Issue: Model Predictions Seem Random

**Symptoms**: Payment predictions don't align with intuition

**Diagnosis**:
```python
# Check model performance
model_results = ridge_regression_payment_prediction(decision_year=4)
print(f"AUC: {model_results['auc']}")
print(f"Accuracy: {model_results['accuracy']}")
# Check feature importances
print(model_results['coefficients'])
```

**Common Causes**:
1. Insufficient training data (too few QBs)
2. Class imbalance (few QBs got paid vs. not paid)
3. Features not predictive for this decision year
4. Overfitting on small dataset

**Solutions**:
- Increase decision_year window (e.g., years 3-5 pooled)
- Try different features or feature combinations
- Use cross-validation to assess generalization
- Consider simpler model (fewer features)

#### Issue: Web Scraping Fails with 403/429 Errors

**Symptoms**: HTTP 403 Forbidden or 429 Too Many Requests

**Diagnosis**:
```python
# Check last request time
from human_like_requester import HumanLikeRequester
requester = HumanLikeRequester()
print(requester.request_history)
```

**Common Causes**:
1. Too many requests too quickly
2. User agent blocked
3. IP temporarily banned

**Solutions**:
- Increase delay in `HumanLikeRequester` (e.g., 10-15 seconds)
- Wait 1-2 hours before retrying
- Use VPN or different network
- Check if Pro Football Reference updated their blocking rules

#### Issue: Performance is Slow

**Symptoms**: Scripts take very long to run

**Diagnosis**:
- Profile code with `time.time()` around sections
- Check file sizes: `ls -lh *.csv`

**Common Causes**:
1. Large CSV files with unnecessary columns
2. Repeated file I/O (loading same file multiple times)
3. Inefficient pandas operations (iterrows, apply on large data)
4. No caching of intermediate results

**Solutions**:
- Use `qb_research/utils/caching.py` to cache expensive operations
- Load CSV once, pass DataFrame as parameter
- Use vectorized pandas operations instead of loops
- Filter data early in pipeline to reduce size

---

## Additional Resources

### Git History

Recent major changes documented in commit history:

- **c05c640**: Remove unnecessary backup scripts and outdated refactoring documentation
- **0bbed64**: Added backwards_compatibility/README.md
- **2e70916**: Remove temporary refactoring files and clean up .gitignore
- **3a75514**: Large refactoring - moved functions to qb_research package

### Future Work

See `to_work_on.txt` for planned features:

- Automated scripts to run based on current year
- Injury removal and performance extrapolation
- All-rounds QB analysis (not just first round)
- Draft slot average trajectories
- Biggest model vs. reality differences

### Getting Help

When asking for help (from humans or AI assistants):

1. **Provide context**: Which script were you running? What were you trying to do?
2. **Show error messages**: Full traceback, not just last line
3. **Share sample data**: First few rows of problematic DataFrame
4. **Describe expected vs. actual**: What did you expect to happen?
5. **Show what you tried**: What debugging steps have you already taken?

### Contributing

When contributing improvements:

1. **Test thoroughly**: Run validation scripts, spot-check outputs
2. **Maintain backwards compatibility**: Don't break existing scripts
3. **Document changes**: Update this CLAUDE.md if you change architecture
4. **Follow conventions**: Match existing code style and patterns
5. **Update to_work_on.txt**: Cross off completed items, add new discoveries

---

## Quick Reference Commands

```bash
# Rebuild entire data pipeline
python rebuild_all_seasons_pipeline.py

# Validate data quality
python data_validation_scripts.py

# Interactive QB comparison
python qb_comp_tool.py

# Visualize metrics by year
python plot_qb_variables_by_year.py

# Update season standings
python standings_scraper.py

# Run wins prediction model
python wins_prediction_fixed.py

# Regenerate era-adjusted data
python -c "from QB_research import create_era_adjusted_payment_data; create_era_adjusted_payment_data(force_refresh=True)"

# Export for Tableau
python -c "from QB_research import create_qb_trajectories_for_tableau; df = create_qb_trajectories_for_tableau(); df.to_csv('qb_trajectories_for_tableau.csv', index=False)"
```

---

**End of CLAUDE.md**

*This document should be updated whenever significant architectural changes are made to the repository. Last update: 2026-01-10*
