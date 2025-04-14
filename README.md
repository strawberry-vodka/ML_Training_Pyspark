import numpy as np
import pandas as pd
from numba import jit, int64, float64

# Time period mapping
PERIOD_MAP = {
    'Night': 0,
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3
}

@jit(nopython=True)
def calculate_time_period_features(user_ids, days_since_epoch, hours, time_periods, clicks, current_indices, lookback_days=90):
    """
    Numba-optimized function to calculate time period features.
    Now expects time_periods as integers (0-3) based on the mapping.
    """
    n = len(current_indices)
    n_periods = 4
    results = np.zeros((n, n_periods * 2), dtype=np.float64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_user = user_ids[current_idx]
        current_day = days_since_epoch[current_idx]
        min_day = current_day - lookback_days
        
        period_clicks = np.zeros(n_periods, dtype=np.float64)
        period_hour_counts = np.zeros((n_periods, 24), dtype=np.int64)
        total_clicks = 0.0
        
        j = current_idx - 1
        while j >= 0 and user_ids[j] == current_user and days_since_epoch[j] >= min_day:
            if days_since_epoch[j] < current_day:
                period = time_periods[j]
                hour = hours[j]
                click_count = clicks[j]
                
                period_clicks[period] += float64(click_count)
                period_hour_counts[period, hour] += click_count
                total_clicks += float64(click_count)
            j -= 1
        
        for p_idx in range(n_periods):
            if period_clicks[p_idx] > 0:
                mode_hour = np.argmax(period_hour_counts[p_idx])
                ratio = period_clicks[p_idx] / total_clicks if total_clicks > 0 else 0.0
                
                results[i, p_idx*2] = float64(mode_hour)
                results[i, p_idx*2+1] = ratio
                
    return results

def calculate_user_time_features(data, date_col, user_id, clicks_col, hour_col, timeperiod_col, lookback_days):
    """
    Main function with proper string time period to numeric conversion.
    """
    # Ensure proper data types
    if not np.issubdtype(data[date_col].dtype, np.datetime64):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Convert string time periods to numeric codes
    data['period_code'] = data[timeperiod_col].map(PERIOD_MAP)
    
    # Sort data by user and date
    data = data.sort_values([user_id, date_col]).copy()
    
    # Calculate days since min date
    min_date = data[date_col].min()
    data['days_since_min'] = (data[date_col] - min_date).dt.days
    
    # Prepare numpy arrays with explicit types for Numba
    user_ids = data[user_id].values.astype(np.int64)
    days_since_epoch = data['days_since_min'].values.astype(np.int64)
    hours = data[hour_col].values.astype(np.int64)
    timeperiods = data['period_code'].values.astype(np.int64)  # Use numeric codes
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    # Calculate features
    features = calculate_time_period_features(
        user_ids, days_since_epoch, hours, timeperiods, 
        clicks, current_indices, lookback_days
    )
    
    # Add features to dataframe
    feature_columns = []
    for period_name, period_code in PERIOD_MAP.items():
        data[f'{period_name}_mode_hour'] = features[:, period_code*2]
        data[f'{period_name}_clicks_ratio'] = features[:, period_code*2+1]
        feature_columns.extend([f'{period_name}_mode_hour', f'{period_name}_clicks_ratio'])
    
    return data[list(data.columns) + feature_columns]
