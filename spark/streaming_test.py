from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)

spark = (
    SparkSession.builder
    .appName("EVBatteryKafkaStreaming")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("time", StringType(), True),
    StructField("value_id", IntegerType(), True),
    StructField("value", DoubleType(), True)
])

stream_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "battery-telemetry")
    .option("startingOffsets", "earliest")
    .load()
)

telemetry = (
    stream_df
    .select(
        F.from_json(
            F.col("value").cast("string"),
            schema
        ).alias("data")
    )
    .select("data.*")
)

telemetry = telemetry.withColumn(
    "event_time",
    F.to_timestamp("time")
)

query = (
    telemetry
    .select(
        "vehicle_id",
        "event_time",
        "value_id",
        "value"
    )
    .writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", "false")
    .option("numRows", 20)
    .start()
)

print("Spark Structured Streaming started...")
print("Reading from Kafka topic: battery-telemetry")

query.awaitTermination()
