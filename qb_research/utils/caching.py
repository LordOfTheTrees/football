"""
Caching utilities for QB research package.

This module provides functions for caching expensive operations to avoid
recomputing them on subsequent runs.
"""

import os
from qb_research.utils.data_loading import load_csv_safe


def load_or_create_cache(cache_file, creation_function, *args, force_refresh=False, **kwargs):
    """
    Generic caching function to avoid recomputing expensive operations.
    
    Args:
        cache_file (str): Path to cache file (CSV)
        creation_function (callable): Function to call if cache doesn't exist
        *args: Positional arguments for creation_function
        force_refresh (bool): If True, ignore cache and recreate
        **kwargs: Keyword arguments for creation_function
    
    Returns:
        DataFrame: Cached or newly created data
    """
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached data from: {cache_file}")
        return load_csv_safe(cache_file)
    else:
        print(f"Cache not found or refresh forced. Creating new data...")
        df = creation_function(*args, **kwargs)
        if df is not None and not df.empty:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            df.to_csv(cache_file, index=False)
            print(f"Data cached to: {cache_file}")
        return df

