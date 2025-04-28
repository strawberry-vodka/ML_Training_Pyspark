import numpy as np
import pandas as pd
from numba import jit, int64

@jit(nopython=True)
def calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Ultra-optimized Numba function for continental mode hour calculation.
    Uses manual argmax and loop optimizations for maximum performance.
    """
    n = len(current_indices)
    n_periods = len(lookback_days)
    results = np.full((n, n_periods), -1, dtype=int64)
    
    # Pre-compute lookback days as array
    lookback_days_arr = np.asarray(lookback_days, dtype=int64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_continent = continents[current_idx]
        current_day = days_since_epoch[current_idx]
        
        # Vectorized min day calculation
        min_days = current_day - lookback_days_arr
        
        for period_idx in range(n_periods):
            hour_counts = np.zeros(24, dtype=int64)
            j = current_idx - 1
            
            # Fast backward scan with early termination
            while j >= 0 and continents[j] == current_continent:
                day_diff = days_since_epoch[j]
                if day_diff < min_days[period_idx]:
                    break
                if day_diff < current_day:  # Exclude current date
                    hour = hours[j]
                    hour_counts[hour] += clicks[j]
                j -= 1
            
            # Manual argmax with early exit for better performance
            if hour_counts.sum() > 0:
                max_val = hour_counts[0]
                max_idx = 0
                for k in range(1, 24):
                    if hour_counts[k] > max_val:
                        max_val = hour_counts[k]
                        max_idx = k
                        # Early exit if we find the theoretical maximum
                        if max_val == clicks.max():
                            break
                results[i, period_idx] = max_idx
                
    return results

def calculate_continent_features_ultra_optimized(data, continent_col, date_col, clicks_col, hour_col, lookback_days_list):
    """
    Most optimized version with:
    - Memory-efficient operations
    - Pre-sorting optimization
    - Minimal pandas overhead
    """
    # Fast datetime conversion
    if not isinstance(data[date_col].dtype, np.dtype('datetime64[ns]')):
        data[date_col] = pd.to_datetime(data[date_col], cache=True)
    
    # Sort once (critical for algorithm)
    sort_cols = [continent_col, date_col]
    if not data.index.is_monotonic_increasing:
        data = data.sort_values(sort_cols)
    
    # Convert to numpy arrays first for faster operations
    continents = data[continent_col].values.astype(int64)
    dates = data[date_col].values.astype('datetime64[D]')
    hours = data[hour_col].values.astype(int64)
    clicks = data[clicks_col].values.astype(int64)
    
    # Fast days calculation using numpy
    min_date = dates.min()
    days_since_epoch = (dates - min_date).astype(int64)
    current_indices = np.arange(len(data), dtype=int64)
    
    # Calculate features
    mode_hours = calculate_continental_activity_mode(
        continents, days_since_epoch, hours, clicks, 
        current_indices, np.array(lookback_days_list, dtype=int64)
    )
    
    # Efficient column assignment
    for i, days in enumerate(lookback_days_list):
        data[f'continent_mode_hour_{days}d'] = mode_hours[:, i]
    
    return data
