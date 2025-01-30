# ML_Training_Pyspark

from clickhouse_driver import Client

# Connect to ClickHouse
client = Client(host='your_clickhouse_host', port=9000, user='your_username', password='your_password')

# Query data
query = "SELECT * FROM your_table WHERE date >= '2023-01-01'"
result = client.execute(query)

def fetch_data_chunk(start_date, end_date):
    client = Client(host='your_clickhouse_host', port=9000, user='your_username', password='your_password')
    query = f"SELECT * FROM your_table WHERE date >= '{start_date}' AND date < '{end_date}'"
    return client.execute(query)

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ClickHouse_Data_Fetch").getOrCreate()

# Define date ranges for parallel fetching
date_ranges = [
    ("2023-01-01", "2023-02-01"),
    ("2023-02-01", "2023-03-01"),
    ("2023-03-01", "2023-04-01"),
    # Add more ranges as needed
]

# Parallelize data fetching
data_chunks = spark.sparkContext.parallelize(date_ranges).map(lambda x: fetch_data_chunk(x[0], x[1]))


# Collect data chunks and convert to a PySpark DataFrame
data = data_chunks.collect()
pdf = spark.createDataFrame(data)

# Save to DBFS
pdf.write.parquet("dbfs:/path_to_save_data/")

# Save to Azure Blob Storage
pdf.write.parquet("wasbs://your_container@your_storage_account.blob.core.windows.net/path_to_save_data/")

from pyspark.sql import SparkSession
from clickhouse_driver import Client

# Initialize Spark session
spark = SparkSession.builder.appName("ClickHouse_Data_Fetch").getOrCreate()

# Function to fetch data in chunks
def fetch_data_chunk(start_date, end_date):
    client = Client(host='your_clickhouse_host', port=9000, user='your_username', password='your_password')
    query = f"SELECT * FROM your_table WHERE date >= '{start_date}' AND date < '{end_date}'"
    return client.execute(query)

# Define date ranges for parallel fetching
date_ranges = [
    ("2023-01-01", "2023-02-01"),
    ("2023-02-01", "2023-03-01"),
    ("2023-03-01", "2023-04-01"),
    # Add more ranges as needed
]

# Parallelize data fetching
data_chunks = spark.sparkContext.parallelize(date_ranges).map(lambda x: fetch_data_chunk(x[0], x[1]))

# Collect data chunks and convert to a PySpark DataFrame
data = data_chunks.collect()
pdf = spark.createDataFrame(data)

# Save to DBFS
pdf.write.parquet("dbfs:/path_to_save_data/")

# Save to Azure Blob Storage
pdf.write.parquet("wasbs://your_container@your_storage_account.blob.core.windows.net/path_to_save_data/")
