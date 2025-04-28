import numpy as np
import pandas as pd
from numba import jit, int64

@jit(nopython=True)
def calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Numba-optimized function to calculate continental mode hour
    
    Args:
        continents: Array of continent codes (int64)
        days_since_epoch: Array of dates as days since epoch (int64)
        hours: Array of hours (0-23) (int64)
        clicks: Array of click counts (int64)
        current_indices: Indices of current records (int64)
        lookback_days: Array of lookback periods (int64[:])
        
    Returns:
        Array of shape (n, len(lookback_days)) containing mode hours
    """
    n = len(current_indices)
    n_periods = len(lookback_days)
    results = np.full((n, n_periods), -1, dtype=int64)  # Initialize with -1 (no activity)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_continent = continents[current_idx]
        current_day = days_since_epoch[current_idx]
        
        for period_idx in range(n_periods):
            days = lookback_days[period_idx]
            min_day = current_day - days
            hour_counts = np.zeros(24, dtype=int64)
            
            j = current_idx - 1
            while j >= 0 and continents[j] == current_continent and days_since_epoch[j] >= min_day:
                if days_since_epoch[j] < current_day:  # Exclude current date
                    hour = hours[j]
                    click_count = clicks[j]
                    hour_counts[hour] += click_count
                j -= 1
            
            if np.sum(hour_counts) > 0:
                results[i, period_idx] = np.argmax(hour_counts)
                
    return results

def calculate_continent_features_N_lookback(data, continent_col, date_col, clicks_col, hour_col, lookback_days_list):
    """
    Calculate continental activity mode hours for multiple lookback periods
    
    Args:
        data: Input DataFrame
        continent_col: Continent code column name
        date_col: Date column name
        clicks_col: Clicks count column name
        hour_col: Hour column name (0-23)
        lookback_days_list: List of lookback periods in days
        
    Returns:
        DataFrame with original columns plus mode hour features
    """
    # Ensure proper datetime type
    if not np.issubdtype(data[date_col].dtype, np.datetime64):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort data by continent and date (critical for algorithm)
    data = data.sort_values([continent_col, date_col]).copy()
    
    # Create days since epoch for fast comparison
    min_date = data[date_col].min()
    data['days_since_epoch'] = (data[date_col] - min_date).dt.days
    
    # Prepare numpy arrays with explicit types for Numba
    continents = data[continent_col].values.astype(int64)
    days_since_epoch = data['days_since_epoch'].values.astype(int64)
    hours = data[hour_col].values.astype(int64)
    clicks = data[clicks_col].values.astype(int64)
    current_indices = np.arange(len(data), dtype=int64)
    lookback_days_arr = np.array(lookback_days_list, dtype=int64)
    
    # Calculate features
    mode_hours = calculate_continental_activity_mode(
        continents, days_since_epoch, hours, clicks, 
        current_indices, lookback_days_arr
    )
    
    # Add features to dataframe
    feature_cols = []
    for idx, days in enumerate(lookback_days_list):
        col_name = f'continent_mode_hour_{days}d'
        data[col_name] = mode_hours[:, idx]
        feature_cols.append(col_name)
    
    # Clean up and return
    data.drop('days_since_epoch', axis=1, inplace=True)
    data.drop_duplicates(subset=[continent_col, date_col], keep='first', inplace=True)
    
    return data[[continent_col, date_col] + feature_cols]
