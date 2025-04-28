from numba import jit

@jit(nopython=True)
def calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices, lookback_days):
    """
    Numba-optimized function to calculate mode hour
    
    Args:
        continents: Array of continents (int64)
        days_since_epoch: Array of dates as days since epoch (int64)
        hours: Array of hours (0-23) (int64)
        clicks: Array of click counts (int64)
        current_indices: Indices of current records (int64)
        
    Returns:
        Array of features (n) containing:
        - Overall mode hour from N lookback days
    """
    n = len(current_indices)
    results = np.zeros((n, len(lookback_days)), dtype=np.float64)
    
    for i in range(n):
        current_idx = current_indices[i]
        current_continent = continents[current_idx]
        current_day = days_since_epoch[current_idx]
        
        # Look backward through history
        for idx, days in enumerate(lookback_days):
            hour_counts = np.zeros(24, dtype=np.int64)
            min_day = current_day - days
            j = current_idx - 1
            while j >= 0 and continents[j] == current_continent and days_since_epoch[j] >= min_day:
                if days_since_epoch[j] < current_day:  # Exclude current date
                    hour = hours[j]
                    click_count = clicks[j]
                    hour_counts[hour] += click_count
                j -= 1            
            
            results[i,idx] = np.argmax(hour_counts) if np.sum(hour_counts) > 0 else -1
            
    return results

def calculate_continent_features_N_lookback(data, continent_enc_col, date_col, clicks_col, hour_col, lookback_days_list):
    """
    Main function to calculate user activity features.
    
    Args:
        data: Input DataFrame
        continent_enc_col: User ID column name
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
    data = data.sort_values([continent_enc_col, date_col]).copy()
    data.drop_duplicates(subset = [continent_enc_col,date_col], keep='first', inplace=True)
    
    # Create days since epoch for fast comparison
    min_date = data[date_col].min()
    data['days_since_epoch'] = (data[date_col] - min_date).dt.days
    
    # Prepare numpy arrays with explicit types for Numba
    continents = data[continent_enc_col].values.astype(np.int64)
    days_since_epoch = data['days_since_epoch'].values.astype(np.int64)
    hours = data[hour_col].values.astype(np.int64)
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    # Calculate features
    result_df = calculate_continental_activity_mode(continents, days_since_epoch, hours, clicks, current_indices,
                                                    lookback_days_list)
    feature_names = []
    for i in range(len(lookback_days_list)):
        val = lookback_days_list[i]
        data[f"continent_mode_hour_{val}_days"] = result_df[:,i]
        feature_names.append(f'continent_mode_hour_{val}_days')
    
    data.drop('days_since_epoch', axis=1, inplace=True)
    
    return data[list([continent_enc_col,date_col]) + feature_names]
