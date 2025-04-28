import numpy as np
import pandas as pd
from numba import jit, int64, prange

@jit(nopython=True, parallel=True)
def calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Optimized Numba function to calculate continental mode hours with:
    - Parallel processing
    - Memory-efficient operations
    - Preserved data integrity
    """
    n = len(current_indices)
    n_periods = len(lookback_days)
    results = np.full((n, n_periods), -1, dtype=int64)  # -1 indicates no activity
    
    # Parallel processing across records
    for i in prange(n):
        current_idx = current_indices[i]
        current_continent = continents[current_idx]
        current_day = days_since_epoch[current_idx]
        
        # Vectorized min day calculation
        min_days = current_day - lookback_days
        
        for period_idx in range(n_periods):
            hour_counts = np.zeros(24, dtype=int64)
            total_clicks = 0
            j = current_idx - 1
            
            # Scan backward through history
            while j >= 0 and continents[j] == current_continent and days_since_epoch[j] >= min_days[period_idx]:
                if days_since_epoch[j] < current_day:  # Exclude current date
                    hour = hours[j]
                    click_count = clicks[j]
                    hour_counts[hour] += click_count
                    total_clicks += click_count
                j -= 1
            
            # Only calculate mode if there was activity
            if total_clicks > 0:
                results[i, period_idx] = np.argmax(hour_counts)
                
    return results

def calculate_continent_features_optimized(data, continent_col, date_col, clicks_col, hour_col, lookback_days_list):
    """
    Optimized feature calculation that:
    - Preserves all original data for accurate feature calculation
    - Uses parallel processing
    - Minimizes memory usage
    """
    # Convert to datetime using fastest method
    if not isinstance(data[date_col].dtype, np.dtype('datetime64[ns]')):
        data[date_col] = pd.to_datetime(data[date_col], cache=True)
    
    # Critical: Sort without removing duplicates to preserve data integrity
    data = data.sort_values([continent_col, date_col]).copy()
    
    # Fast days calculation using numpy datetime64 operations
    min_date = data[date_col].values.min()
    data['days_since_epoch'] = (data[date_col].view('int64') - min_date.view('int64')) // (86400 * 1e9)
    
    # Memory-efficient array preparation
    continents = data[continent_col].values.astype(int64)
    days_since_epoch = data['days_since_epoch'].values.astype(int64)
    hours = data[hour_col].values.astype(int64)
    clicks = data[clicks_col].values.astype(int64)
    current_indices = np.arange(len(data), dtype=int64)
    lookback_days_arr = np.array(lookback_days_list, dtype=int64)
    
    # Parallel feature calculation
    mode_hours = calculate_continental_activity_mode(
        continents, days_since_epoch, hours, clicks, 
        current_indices, lookback_days_arr
    )
    
    # Efficient column assignment without modifying original data
    for i, days in enumerate(lookback_days_list):
        data[f'continent_mode_hour_{days}d'] = mode_hours[:, i]
    
    # Only remove temporary column at the end
    data.drop('days_since_epoch', axis=1, inplace=True)
    
    return data
