"""
Year utility functions for QB research.

This module provides centralized logic for determining years used throughout
the pipeline, making it easy to update annually without hardcoded values.
"""

from datetime import datetime


def get_current_year():
    """
    Returns the current calendar year.
    
    Returns:
        int: Current year (e.g., 2026)
    """
    return datetime.now().year


def get_current_season_year():
    """
    Returns the most recent completed NFL season year.
    
    NFL seasons end in January/February, so the most recent completed season
    is typically the previous calendar year.
    
    Returns:
        int: Most recent completed season year (e.g., 2025 if current year is 2026)
    """
    return datetime.now().year - 1


def get_reference_year(override=None):
    """
    Returns the reference year for era adjustments.
    
    This is typically the most recent completed season, as we want to adjust
    all historical stats to match the most recent season's baseline.
    
    Args:
        override (int, optional): Override the default reference year
    
    Returns:
        int: Reference year for era adjustments
    """
    if override is not None:
        return override
    return get_current_season_year()


def get_recent_draft_cutoff(override=None):
    """
    Returns the cutoff year for "recent" draft picks.
    
    Typically set to 2 years before current year to capture recent draft classes.
    
    Args:
        override (int, optional): Override the default cutoff year
    
    Returns:
        int: Cutoff year for recent drafts
    """
    if override is not None:
        return override
    return datetime.now().year - 2
