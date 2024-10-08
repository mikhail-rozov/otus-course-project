from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, hour, count, row_number
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType
from pyspark.sql.window import Window


BUCKET_NAME = "mrozov-mlops"
BUCKET_PATH = BUCKET_NAME + "/" + "course_project/data/"
CSV_FILE_NAME = "train.csv"
PARQUET_FILE_NAME = "train.parquet"
PARQUET_FILE_NAME_TRAIN = "train_dataset.parquet"
N_PARTITIONS = 12


def create_time_features(df):
    df = df.withColumn('day_of_week', dayofweek('click_time')) \
           .withColumn('hour', hour('click_time'))
           
    return df

def create_grouped_features(df):
    # Group by 'ip' and calculate the count of occurrences
    ip_count = df.groupBy('ip').agg(count("*").alias('ip_count'))
    
    # Group by 'ip', 'day_of_week', 'hour' and calculate the count
    ip_day_hour = df.groupBy('ip', 'day_of_week', 'hour').agg(count("*").alias('ip_day_hour'))
    
    # Group by 'ip', 'hour', 'channel' and calculate the count
    ip_hour_channel = df.groupBy('ip', 'hour', 'channel').agg(count("*").alias('ip_hour_channel'))
    
    # Group by 'ip', 'hour', 'os' and calculate the count of 'channel'
    ip_hour_os = df.groupBy('ip', 'hour', 'os').agg(count('channel').alias('ip_hour_os'))
    
    # Group by 'ip', 'hour', 'app' and calculate the count of 'channel'
    ip_hour_app = df.groupBy('ip', 'hour', 'app').agg(count('channel').alias('ip_hour_app'))
    
    # Group by 'ip', 'hour', 'device' and calculate the count of 'channel'
    ip_hour_device = df.groupBy('ip', 'hour', 'device').agg(count('channel').alias('ip_hour_device'))
    
    # Make joins of the new features
    df = df.join(ip_count, on='ip', how='left') \
           .join(ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left') \
           .join(ip_hour_channel, on=['ip', 'hour', 'channel'], how='left') \
           .join(ip_hour_os, on=['ip', 'hour', 'os'], how='left') \
           .join(ip_hour_app, on=['ip', 'hour', 'app'], how='left') \
           .join(ip_hour_device, on=['ip', 'hour', 'device'], how='left')

    return df

def get_sample(df, time_col, 
                 previous_sample_size=500_000,
                 sample_size=200_000):

    # Create a window specification to order by the time column
    window_spec = Window.orderBy(time_col)
    
    # Add a row number to each row based on the order of the time column
    df = df.withColumn('row_num', row_number().over(window_spec))
    
    # Split the dataset based on row number
    sample_df = df.filter(col('row_num') <= previous_sample_size + sample_size)\
                  .drop('row_num')
    
    return sample_df

def main():
    schema = StructType([
        StructField("ip", IntegerType(), True),
        StructField("app", IntegerType(), True),
        StructField("device", IntegerType(), True),
        StructField("os", IntegerType(), True),
        StructField("channel", IntegerType(), True),
        StructField("click_time", TimestampType(), True),
        StructField("attributed_time", TimestampType(), True),
        StructField("is_attributed", IntegerType(), True),
    ])

    spark = (
        SparkSession
        .builder
        .appName("Otus-course-project")
        .getOrCreate()
    )

    df = spark.read.csv("s3a://" + BUCKET_PATH + CSV_FILE_NAME, header=True, schema=schema)
    df = df.select("ip", "app", "device", "os", "channel", "click_time", "is_attributed")

    df.repartition(N_PARTITIONS)\
      .write.mode("overwrite")\
      .parquet("s3a://" + BUCKET_PATH + PARQUET_FILE_NAME)
    
    df = spark.read.parquet("s3a://" + BUCKET_PATH + PARQUET_FILE_NAME)

    df = get_sample(df, "click_time")

    df = create_time_features(df)
    df = create_grouped_features(df)

    df.repartition(N_PARTITIONS)\
      .write.mode("overwrite")\
      .parquet("s3a://" + BUCKET_PATH + PARQUET_FILE_NAME_TRAIN)
    
    spark.stop()

if __name__ == "__main__":
    main()
