import numpy as np
import pandas as pd
from numba import jit, int64, prange

@jit(nopython=True, parallel=True)
def calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Optimized Numba function to calculate continental mode hours.
    Uses parallel processing and memory-efficient operations.
    """
    n = len(current_indices)
    n_periods = len(lookback_days)
    results = np.full((n, n_periods), -1, dtype=int64)
    
    # Parallelize the outer loop
    for i in prange(n):
        current_idx = current_indices[i]
        current_continent = continents[current_idx]
        current_day = days_since_epoch[current_idx]
        
        # Pre-calculate all min days
        min_days = current_day - lookback_days
        
        for period_idx in range(n_periods):
            hour_counts = np.zeros(24, dtype=int64)
            j = current_idx - 1
            
            while j >= 0 and continents[j] == current_continent and days_since_epoch[j] >= min_days[period_idx]:
                if days_since_epoch[j] < current_day:
                    hour = hours[j]
                    hour_counts[hour] += clicks[j]
                j -= 1
            
            if hour_counts.sum() > 0:
                results[i, period_idx] = hour_counts.argmax()
                
    return results

def calculate_continent_features_optimized(data, continent_col, date_col, clicks_col, hour_col, lookback_days_list):
    """
    Optimized version with memory-efficient operations and parallel processing.
    """
    # Convert to datetime if needed
    if not isinstance(data[date_col].dtype, np.dtype('datetime64[ns]')):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort and deduplicate in one operation
    data = data.sort_values([continent_col, date_col])\
              .drop_duplicates([continent_col, date_col], keep='first')\
              .copy()
    
    # Calculate days since epoch using numpy directly for speed
    min_date = data[date_col].values.min()
    data['days_since_epoch'] = (data[date_col].view('int64') - min_date.view('int64')) // (86400 * 1e9)
    
    # Prepare numpy arrays - use views where possible
    continents = data[continent_col].values.astype(int64)
    days_since_epoch = data['days_since_epoch'].values.astype(int64)
    hours = data[hour_col].values.astype(int64)
    clicks = data[clicks_col].values.astype(int64)
    current_indices = np.arange(len(data), dtype=int64)
    lookback_days_arr = np.array(lookback_days_list, dtype=int64)
    
    # Calculate features in parallel
    mode_hours = calculate_continental_activity_mode(
        continents, days_since_epoch, hours, clicks, 
        current_indices, lookback_days_arr
    )
    
    # Efficient column assignment
    feature_data = {
        f'continent_mode_hour_{days}d': mode_hours[:, i]
        for i, days in enumerate(lookback_days_list)
    }
    
    # Use assign for efficient DataFrame construction
    result_df = data.assign(**feature_data)\
                   .drop('days_since_epoch', axis=1)
    
    return result_df[[continent_col, date_col] + list(feature_data.keys())]
