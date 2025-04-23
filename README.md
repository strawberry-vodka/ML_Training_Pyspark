import numpy as np
import pandas as pd
from numba import jit, int64, float64

@jit(nopython=True)
def calculate_user_activity_mode(user_ids, days_since_epoch, hours, weekdays, clicks, current_indices, lookback_days):
    """
    Numba-optimized function to calculate mode hour
    
    Args:
        user_ids: Array of user IDs (int64)
        days_since_epoch: Array of dates as days since epoch (int64)
        hours: Array of hours (0-23) (int64)
        weekdays: Array of weekdays (0-6 where 5-6 are weekend) (int64)
        clicks: Array of click counts (int64)
        current_indices: Indices of current records (int64)
        
    Returns:
        Array of features (n x 5) containing:
        - Overall mode hour
        - Weekday mode hour
        - Weekend mode hour
        - Weekday clicks
        - Weekend clicks
    """
    n = len(current_indices)
    results = np.zeros((n, 5), dtype=np.float64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_user = user_ids[current_idx]
        current_day = days_since_epoch[current_idx]
        min_day = current_day - lookback_days
        
        hour_counts = np.zeros(24, dtype=np.int64)
        weekday_count = np.zeros(24, dtype=np.int64)
        weekend_count = np.zeros(24, dtype=np.int64)
        weekday_clicks = 0.0
        weekend_clicks = 0.0
        
        # Look backward through history
        j = current_idx - 1
        while j >= 0 and user_ids[j] == current_user and days_since_epoch[j] >= min_day:
            if days_since_epoch[j] < current_day:  # Exclude current date
                hour = hours[j]
                click_count = clicks[j]
                is_weekend = weekdays[j] >= 5  # Saturday (5) or Sunday (6)
                
                hour_counts[hour] += click_count
                if is_weekend:
                    weekend_count[hour] += click_count
                    weekend_clicks += float64(click_count)
                else:
                    weekday_count[hour] += click_count
                    weekday_clicks += float64(click_count)
            j -= 1
            
        # Store results
        results[i, 0] = np.argmax(hour_counts) if np.sum(hour_counts) > 0 else -1
        results[i, 1] = np.argmax(weekday_count) if np.sum(weekday_count) > 0 else -1
        results[i, 2] = np.argmax(weekend_count) if np.sum(weekend_count) > 0 else -1
        results[i, 3] = weekday_clicks
        results[i, 4] = weekend_clicks
    
    return results

def calculate_user_activity_features(data, user_col, date_col, clicks_col, hour_col, weekday_col, lookback_days):
    """
    Main function to calculate user activity features.
    
    Args:
        data: Input DataFrame
        user_col: User ID column name
        date_col: Date column name
        clicks_col: Clicks count column name
        hour_col: Hour column name
        weekday_col: Weekday column name (0-6)
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with original columns plus calculated features
    """
    # Ensure proper datetime type
    if not np.issubdtype(data[date_col].dtype, np.datetime64):
        data[date_col] = pd.to_datetime(data[date_col])
    
    # Sort data by user and date (critical for algorithm)
    data = data.sort_values([user_col, date_col]).copy()
    
    # Create days since epoch for fast comparison
    min_date = data[date_col].min()
    data['days_since_epoch'] = (data[date_col] - min_date).dt.days
    
    # Prepare numpy arrays with explicit types for Numba
    user_ids = data[user_col].values.astype(np.int64)
    days_since_epoch = data['days_since_epoch'].values.astype(np.int64)
    hours = data[hour_col].values.astype(np.int64)
    weekdays = data[weekday_col].values.astype(np.int64)
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    # Calculate features
    features = calculate_user_activity_mode(
        user_ids, days_since_epoch, hours, weekdays, clicks, current_indices, lookback_days
    )
    
    # Add features to dataframe
    data['mode_hour'] = features[:, 0]
    data['weekday_mode_hour'] = features[:, 1]
    data['weekend_mode_hour'] = features[:, 2]
    data['weekday_clicks'] = features[:, 3]
    data['weekend_clicks'] = features[:, 4]
    
    # Calculate ratio safely
    weekday_nonzero = data['weekday_clicks'] > 0
    data['weekend_to_weekday_ratio'] = np.where(
        weekday_nonzero,
        data['weekend_clicks'] / data['weekday_clicks'],
        0.0
    )
    
    return data[['audienceid', 'click_date', 'mode_hour', 'weekday_mode_hour', 
                'weekend_mode_hour', 'weekday_clicks', 'weekend_clicks', 
                'weekend_to_weekday_ratio']]
