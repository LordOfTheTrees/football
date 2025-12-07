# Backwards Compatibility Code

This folder contains old function definitions that were moved to the new `qb_research` package structure.

## What Happened

All functions from `QB_research.py` have been refactored into a modular package structure under `qb_research/`. The old function definitions that were kept for backwards compatibility have been removed from `QB_research.py`.

## Current Structure

All functionality is now available through the `qb_research` package:

- `qb_research.utils.data_loading` - Data loading utilities
- `qb_research.utils.name_matching` - Name matching utilities
- `qb_research.analysis.statistical_analysis` - Statistical analysis functions
- `qb_research.analysis.era_adjustment` - Era adjustment functions
- `qb_research.data.builders` - Data building functions
- `qb_research.data.loaders` - Data loading functions
- `qb_research.data.mappers` - Data mapping functions
- `qb_research.data.validators` - Data validation functions
- `qb_research.preprocessing.feature_engineering` - Feature engineering
- `qb_research.modeling.prediction_models` - Prediction models
- `qb_research.modeling.surface_models` - Surface models
- `qb_research.exports.tableau_exports` - Tableau export functions
- `qb_research.comparisons` - Comparison and trajectory matching
- `qb_research.utils.debug_utils` - Debug utilities

## Migration

`QB_research.py` now acts as a thin wrapper that imports and re-exports all functions from the new package structure, maintaining backwards compatibility for existing scripts.

If you need the old function definitions for reference, they can be found in the git history before this refactoring.

