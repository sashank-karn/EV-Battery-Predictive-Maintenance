from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("EVBatteryVoltageAnalysis")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

df = spark.read.parquet(
    "/home/sashank_karn/EV_BigData_CA3/dataset/cup1_test.parquet"
)

voltage = (
    df.filter(F.col("value_id") == 1200)
      .select("time", "value")
      .withColumnRenamed("value", "battery_voltage")
)

print("=== TOTAL VOLTAGE RECORDS ===")
print(voltage.count())

print("=== TIME RANGE ===")
voltage.select(
    F.min("time").alias("start_time"),
    F.max("time").alias("end_time")
).show(truncate=False)

print("=== VOLTAGE PERCENTILES ===")

voltage.select(
    F.expr("percentile_approx(battery_voltage, 0.01)").alias("p01"),
    F.expr("percentile_approx(battery_voltage, 0.05)").alias("p05"),
    F.expr("percentile_approx(battery_voltage, 0.25)").alias("p25"),
    F.expr("percentile_approx(battery_voltage, 0.50)").alias("median"),
    F.expr("percentile_approx(battery_voltage, 0.75)").alias("p75"),
    F.expr("percentile_approx(battery_voltage, 0.95)").alias("p95"),
    F.expr("percentile_approx(battery_voltage, 0.99)").alias("p99")
).show()

print("=== EXTREME VALUES ===")

voltage.filter(
    (F.col("battery_voltage") < 300) |
    (F.col("battery_voltage") > 500)
).select(
    "time",
    "battery_voltage"
).orderBy("time").show(20, truncate=False)

print("=== ZERO VOLTAGE COUNT ===")

print(
    voltage.filter(F.col("battery_voltage") <= 0).count()
)

print("=== VALID VOLTAGE COUNT ===")

print(
    voltage.filter(
        (F.col("battery_voltage") >= 300) &
        (F.col("battery_voltage") <= 500)
    ).count()
)

spark.stop()
