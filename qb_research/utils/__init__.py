"""
Utility functions for QB research.

This module contains:
- Data loading and validation
- Name matching and normalization
- Caching utilities
- Debug and utility functions
"""

from .data_loading import (
    load_csv_safe,
    validate_columns,
    validate_payment_years
)

from .name_matching import (
    normalize_player_name,
    debug_name_matching
)

from .caching import (
    load_or_create_cache
)

from .debug_utils import (
    fix_individual_qb_files,
    standardize_qb_columns,
    debug_specific_qb
)

from .exploratory import (
    bestqbseasons,
    best_season_averages,
    most_expensive_qb_contracts,
    best_season_records
)
