from pyspark.sql import SparkSession
from clickhouse_driver import Client
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("ClickHouse_Data_Fetch").getOrCreate()

# Function to split a date range into weekly chunks
def split_date_range(start_date, end_date):
    """
    Splits a date range into weekly chunks.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=7)  # Weekly intervals
    
    date_ranges = []
    while start < end:
        next_start = start + delta
        date_ranges.append((start.strftime("%Y-%m-%d"), min(next_start, end).strftime("%Y-%m-%d")))
        start = next_start
    
    return date_ranges

# Function to fetch data in batches
def fetch_data_batch(date_range):
    """
    Fetches data from ClickHouse for a specific date range.
    This function runs on each worker node.
    """
    start_date, end_date = date_range
    try:
        # Create a new ClickHouse client connection
        client = Client(host='your_clickhouse_host', port=9000, user='your_username', password='your_password')
        
        # Query data for the specified date range
        query = f"SELECT * FROM your_table WHERE date >= '{start_date}' AND date < '{end_date}'"
        logger.info(f"Executing query: {query}")
        
        # Fetch data as a Pandas DataFrame
        df = client.query_df(query)
        
        # Log the number of rows fetched
        logger.info(f"Fetched {len(df)} rows for range {start_date} to {end_date}")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for range {start_date} to {end_date}: {e}")
        return None

# Function to save monthly Parquet files
def save_monthly_parquet(monthly_data):
    """
    Saves monthly DataFrames as Parquet files.
    """
    for month, df in monthly_data.items():
        output_path = f"dbfs:/path_to_save_data/{month}.parquet"
        logger.info(f"Saving data for {month} to {output_path}")
        df.to_parquet(output_path)

# Define the overall date range
start_date = "2023-01-01"
end_date = "2023-12-31"

# Split the date range into weekly chunks
date_ranges = split_date_range(start_date, end_date)

# Parallelize the date ranges into an RDD
batches_rdd = spark.sparkContext.parallelize(date_ranges, numSlices=100)  # 100 partitions

# Fetch data in parallel
fetched_data_rdd = batches_rdd.map(fetch_data_batch)

# Collect the fetched data (Pandas DataFrames) to the driver
fetched_data = fetched_data_rdd.collect()

# Combine weekly DataFrames into monthly DataFrames
monthly_data = {}
for df in fetched_data:
    if df is not None:
        # Extract the month from the date column
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        
        # Group by month and concatenate DataFrames
        for month, group in df.groupby('month'):
            month_str = month.strftime("%Y-%m")
            if month_str in monthly_data:
                monthly_data[month_str] = pd.concat([monthly_data[month_str], group])
            else:
                monthly_data[month_str] = group

# Save the combined data as monthly Parquet files
save_monthly_parquet(monthly_data)
