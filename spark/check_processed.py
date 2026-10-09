from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("CheckProcessedData")
    .master("local[2]")
    .getOrCreate()
)

df = spark.read.parquet(
    "/home/sashank_karn/EV_BigData_CA3/dataset/processed"
)

print("Total processed rows:", df.count())

df.printSchema()

df.show(20, truncate=False)

print("Temperature availability:")
df.groupBy("temperature_available").count().show()

spark.stop()
