import numpy as np
import pandas as pd
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
        else:
            results[i] = -1  # No previous activity
    
    return results

def calculate_user_last_active_features(df, user_col='user_id', 
                                     date_col='date', 
                                     clicks_col='clicks',
                                     hour_col='hour'):
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
    # Convert to datetime if needed
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by user and date (critical for performance)
    df = df.sort_values([user_col, date_col]).copy()
    
    # Create days since epoch for faster comparison
    min_date = df[date_col].min()
    df['days_since_epoch'] = (df[date_col] - min_date).dt.days
    
    # Prepare numpy arrays for Numba
    user_ids = df[user_col].values.astype(np.int64)
    days_since_epoch = df['days_since_epoch'].values.astype(np.int64)
    hours = df[hour_col].values.astype(np.int64)
    clicks = df[clicks_col].values.astype(np.int64)
    current_indices = np.arange(len(df), dtype=np.int64)
    
    # Calculate features with Numba
    mode_hours = calculate_last_active_mode_hour(
        user_ids, days_since_epoch, hours, clicks, current_indices
    )
    
    # Add to dataframe
    df['mode_hour_last_active'] = mode_hours
    
    # Clean up temporary column
    df.drop('days_since_epoch', axis=1, inplace=True)
    
    return df

# Example usage:
if __name__ == "__main__":
    # Create sample data (11M records in real usage)
    n_samples = 100000  # Smaller sample for demonstration
    dates = pd.date_range('2023-01-01', periods=30)
    data = {
        'user_id': np.random.randint(1, 10000, n_samples),
        'date': np.random.choice(dates, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'clicks': np.random.randint(1, 10, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Calculate features
    result_df = calculate_user_last_active_features(
        df, 
        user_col='user_id',
        date_col='date',
        clicks_col='clicks',
        hour_col='hour'
    )
    
    print(result_df.head())
