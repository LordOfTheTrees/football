"""
QB comparison and trajectory matching functions.

This module contains functions for:
- Finding similar QBs based on trajectory matching
- Year weighting regression for payment decisions
- Batch comparison analysis
"""

from .trajectory_matching import (
    find_most_similar_qbs,
    find_comps_both_metrics,
    batch_comp_analysis
)

from .year_weighting import (
    year_weighting_regression,
    extract_year_weights_from_regression_results
)

