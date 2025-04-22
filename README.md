from numba import jit, int64
from datetime import timedelta

@jit(nopython=True)
def calculate_last_active_mode_hour(user_ids, dates, hours, clicks, current_indices):
    """
    Numba-optimized function to calculate mode hour from last active day.
    
    Args:
        user_ids: Array of user IDs
        dates: Array of dates as days since epoch
        hours: Array of hours (0-23)
        clicks: Array of click counts
        current_indices: Indices of current records
        
    Returns:
        Array of mode hours for each record
    """
    results = np.zeros(len(current_indices), dtype=int64)
    results_1 = np.zeros(len(current_indices), dtype=int64)
    
    
    for i in range(len(current_indices)):
        current_idx = current_indices[i]
        current_user = user_ids[current_idx]
        current_date = dates[current_idx]
        
        # Initialize tracking for last active date
        last_active_date = -1
        hour_counts = np.zeros(24, dtype=int64)
        
        # Look backward through history
        j = current_idx - 1
        while j >= 0 and user_ids[j] == current_user:
            if dates[j] < current_date:  # Exclude current date
                if last_active_date == -1:
                    # First older date found - initialize
                    last_active_date = dates[j]
                    hour_counts[hours[j]] += clicks[j]
                elif dates[j] == last_active_date:
                    # Same last active date - accumulate
                    hour_counts[hours[j]] += clicks[j]
                else:
                    # Found older date - break loop
                    break
            j -= 1
        
        # Calculate mode hour if we found any activity
        if last_active_date != -1:
            results[i] = np.argmax(hour_counts)
            results_1[i] = np.max(np.nonzero(hour_counts))
            
        else:
            results[i] = -1  # No previous activity
            results_1[i] = np.max(np.nonzero(hour_counts))
    
    return results, results_1

def calculate_user_last_active_features(data, user_col, date_col, clicks_col, hour_col, output_col):
    """
    Main function to calculate last active day mode hour features.
    
    Args:
        df: Input DataFrame
        user_col: User ID column name
        date_col: Date column name
        clicks_col: Clicks count column name
        hour_col: Hour column name
        
    Returns:
        DataFrame with original columns plus mode_hour_last_active
    """
#     if not np.issubdtype(data[date_col].dtype, np.datetime64):
#         data[date_col] = pd.to_datetime(data[date_col])
    
    data = data.sort_values([user_col, date_col]).copy()
    
    # Create days since epoch for faster comparison
    min_date = data[date_col].min()
    data['days_since_epoch'] = (data[date_col] - min_date).dt.days
    
    # Prepare numpy arrays for Numba
    user_ids = data[user_col].values.astype(np.int64)
    days_since_epoch = data['days_since_epoch'].values.astype(np.int64)
    hours = data[hour_col].values.astype(np.int64)
    clicks = data[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(data), dtype=np.int64)
    
    result_df = calculate_last_active_mode_hour(
        user_ids, days_since_epoch, hours, clicks, current_indices
    )
    
    # Add to dataframe
#     (pd.concat([data,pd.DataFrame(result_df,columns=['Last_Active_Day_Mode','L')
    
#     # Clean up temporary column
#     data.drop('days_since_epoch', axis=1, inplace=True)
    
    return result_df#data[[user_col, date_col, output_col]]
