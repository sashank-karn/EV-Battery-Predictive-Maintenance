from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("EVBatterySignalInspection")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

df = spark.read.parquet(
    "/home/sashank_karn/EV_BigData_CA3/dataset/cup1_test.parquet"
)

selected_ids = [900, 1200, 1290, 1293, 1294, 1295]

df = df.filter(F.col("value_id").isin(selected_ids))

print("=== SIGNAL COUNTS ===")

df.groupBy("value_id").count().orderBy("value_id").show()

print("=== SIGNAL STATISTICS ===")

df.groupBy("value_id").agg(
    F.min("value").alias("min_value"),
    F.max("value").alias("max_value"),
    F.avg("value").alias("avg_value"),
    F.stddev("value").alias("std_value")
).orderBy("value_id").show()

print("=== SIGNAL DEFINITIONS ===")

definitions = {
    900: "SOC",
    1200: "HV Battery Voltage",
    1290: "Depth of Discharge",
    1293: "Cell Voltage Max",
    1294: "Cell Voltage Min",
    1295: "Cell Voltage Delta"
}

for value_id, name in definitions.items():
    print(value_id, "->", name)

spark.stop()
