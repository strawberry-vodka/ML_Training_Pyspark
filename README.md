import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_optimized_test_set(historical_df, upcoming_start_date=None):
    """
    Optimized function to create test set for 1M+ users.

    Parameters:
    - historical_df (pd.DataFrame): Must contain 'user_id', 'date', and feature columns.
    - upcoming_start_date (str or None): 'YYYY-MM-DD'. If None, uses today's date.

    Returns:
    - pd.DataFrame: Test set with 7 days of upcoming data per user.
    """

    # Step 1: Parse date column
    historical_df['date'] = pd.to_datetime(historical_df['date'])

    # Step 2: Get last known feature row per user (fastest using groupby().idxmax())
    last_idx = historical_df.groupby('user_id')['date'].idxmax()
    user_features_df = historical_df.loc[last_idx].drop(columns='date').reset_index(drop=True)

    # Step 3: Create upcoming date range
    if upcoming_start_date is None:
        upcoming_start_date = datetime.today().date()
    else:
        upcoming_start_date = pd.to_datetime(upcoming_start_date).date()

    upcoming_dates = pd.date_range(start=upcoming_start_date, periods=7)

    # Step 4: Efficiently broadcast user_ids to dates using repeat + tile
    user_ids = user_features_df['user_id'].values
    n_users = len(user_ids)
    n_days = len(upcoming_dates)

    # Repeat users and tile dates to create cartesian product efficiently
    test_user_ids = np.repeat(user_ids, n_days)
    test_dates = np.tile(upcoming_dates, n_users)

    test_df = pd.DataFrame({
        'user_id': test_user_ids,
        'date': test_dates
    })

    # Step 5: Merge features
    final_test_df = test_df.merge(user_features_df, on='user_id', how='left')

    return final_test_df
