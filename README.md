import numpy as np
import pandas as pd
from numba import jit, int64
from datetime import datetime

@jit(nopython=True)
def calculate_last_active_mode_hour(user_ids, dates, hours, clicks, current_indices):
    """
    Numba-optimized function to calculate mode hour from last active day.
    
    Args:
        user_ids: Array of user IDs (int64)
        dates: Array of dates as days since epoch (int64)
        hours: Array of hours (0-23) (int64)
        clicks: Array of click counts (int64)
        current_indices: Indices of current records (int64)
        
    Returns:
        Tuple of (mode_hours, last_active_hours) arrays
    """
    n = len(current_indices)
    mode_hours = np.full(n, -1, dtype=int64)  # Initialize with -1 (no activity)
    last_active_hours = np.full(n, -1, dtype=int64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_user = user_ids[current_idx]
        current_date = dates[current_idx]
        
        # Initialize tracking for last active date
        last_active_date = -1
        hour_counts = np.zeros(24, dtype=int64)
        last_hour = -1
        
        # Look backward through history
        j = current_idx - 1
        while j >= 0 and user_ids[j] == current_user:
            if dates[j] < current_date:  # Exclude current date
                if last_active_date == -1:
                    # First older date found - initialize
                    last_active_date = dates[j]
                    hour_counts[hours[j]] += clicks[j]
                    last_hour = hours[j]
                elif dates[j] == last_active_date:
                    # Same last active date - accumulate
                    hour_counts[hours[j]] += clicks[j]
                    last_hour = hours[j]
                else:
                    # Found older date - break loop
                    break
            j -= 1
        
        # Store results if we found activity
        if last_active_date != -1:
            mode_hours[i] = np.argmax(hour_counts)
            last_active_hours[i] = last_hour
    
    return mode_hours, last_active_hours

def calculate_user_last_active_features(data, user_col, date_col, clicks_col, hour_col):
    """
    Main function to calculate last active day features.
    
    Args:
        data: Input DataFrame
        user_col: User ID column name
        date_col: Date column name
        clicks_col: Clicks count column name
        hour_col: Hour column name
        
    Returns:
        DataFrame with original columns plus:
        - mode_hour_last_active: Most frequent hour on last active day
        - last_active_hour: Last hour of activity on last active day
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
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    # Calculate features
    mode_hours, last_active_hours = calculate_last_active_mode_hour(
        user_ids, days_since_epoch, hours, clicks, current_indices
    )
    
    # Add features to dataframe
    data['mode_hour_last_active'] = mode_hours
    data['last_active_hour'] = last_active_hours
    
    # Clean up temporary column
    data.drop('days_since_epoch', axis=1, inplace=True)
    
    return data

# Example usage
if __name__ == "__main__":
    # Create sample data
    n_samples = 10000
    dates = pd.date_range('2023-01-01', periods=30)
    data = pd.DataFrame({
        'user_id': np.random.randint(1, 100, n_samples),
        'date': np.random.choice(dates, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'clicks': np.random.randint(1, 10, n_samples)
    })
    
    # Calculate features
    result_df = calculate_user_last_active_features(
        data=data,
        user_col='user_id',
        date_col='date',
        clicks_col='clicks',
        hour_col='hour'
    )
    
    print(result_df.head())
    
    return result_df#data[[user_col, date_col, output_col]]
