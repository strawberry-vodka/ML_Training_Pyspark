from numba import jit, int64, float64

@jit(nopython=True)
def calculate_recent_user_activity_mode(user_ids, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Numba-optimized function to calculate mode hour
    
    Args:
        user_ids: Array of user IDs (int64)
        days_since_epoch: Array of dates as days since epoch (int64)
        hours: Array of hours (0-23) (int64)
        clicks: Array of click counts (int64)
        current_indices: Indices of current records (int64)
        
    Returns:
        Array of features (n x 5) containing:
        - Overall mode hour
    """
    n = len(current_indices)
    results = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_user = user_ids[current_idx]
        current_day = days_since_epoch[current_idx]
        min_day = current_day - lookback_days
        
        hour_counts = np.zeros(24, dtype=np.int64)
        # Look backward through history
        j = current_idx - 1
        while j >= 0 and user_ids[j] == current_user and days_since_epoch[j] >= min_day:
            if days_since_epoch[j] < current_day:  # Exclude current date
                hour = hours[j]
                click_count = clicks[j]
                hour_counts[hour] += click_count
            j -= 1            
            
        # Store results
        results[i, 0] = np.argmax(hour_counts) if np.sum(hour_counts) > 0 else -1
    return results

def calculate_user_activity_features_N_lookback(data, user_col, date_col, clicks_col, hour_col, lookback_days):
    """
    Main function to calculate user activity features.
    
    Args:
        data: Input DataFrame
        user_col: User ID column name
        date_col: Date column name
        clicks_col: Clicks count column name
        hour_col: Hour column name
        lookback_days: List of Number of days to look back
        
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
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    # Calculate features
    for days in lookback_days:
        data[f'mode_hour_{days}_days'] = calculate_recent_user_activity_mode(
            user_ids, days_since_epoch, hours, clicks, current_indices, days)
        feature_names.extend(f'mode_hour_{days}')
    data.drop('days_since_epoch', axis=1, inplace=True)
    
    data.drop_duplicates(subset = [user_col,date_col], keep='first', inplace=True)
    
    return data[list([user_col,date_col]) + feature_names]
