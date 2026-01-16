"""
Adjustments module for QB research.

This module contains functions for:
- Era adjustment (inflation adjustments since 2000)
- Injury projection (projecting per-game averages to full seasons)
"""

from qb_research.adjustments.era_adjustment import (
    calculate_era_adjustment_factors,
    apply_era_adjustments,
    create_era_adjusted_payment_data
)

from qb_research.adjustments.injury_projection import (
    get_season_length,
    calculate_per_game_averages,
    project_to_full_season,
    fill_missing_seasons,
    apply_injury_projection,
    export_injury_projection_comparison
)

__all__ = [
    # Era adjustment
    'calculate_era_adjustment_factors',
    'apply_era_adjustments',
    'create_era_adjusted_payment_data',
    # Injury projection
    'get_season_length',
    'calculate_per_game_averages',
    'project_to_full_season',
    'fill_missing_seasons',
    'apply_injury_projection',
    'export_injury_projection_comparison',
]
