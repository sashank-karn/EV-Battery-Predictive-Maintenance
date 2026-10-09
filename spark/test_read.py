from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EVBatteryParquetTest")
    .master("local[*]")
    .getOrCreate()
)

df = spark.read.parquet(
    "/home/sashank_karn/EV_BigData_CA3/dataset/cup1_test.parquet"
)

df.printSchema()

print("Total rows:", df.count())

df.show(20, truncate=False)

spark.stop()
