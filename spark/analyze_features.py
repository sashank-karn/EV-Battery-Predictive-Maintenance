from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("EVBatteryFeatureAnalysis")
    .master("local[2]")
    .getOrCreate()
)

df = spark.read.parquet(
    "/home/sashank_karn/EV_BigData_CA3/dataset/processed"
)

print("=== BASIC STATISTICS ===")

df.select(
    "vehicle_speed",
    "ambient_air_temp",
    "hv_aux_power",
    "hv_soc",
    "hv_battery_voltage",
    "battery_temp_avg",
    "battery_temp_delta"
).describe().show()

print("=== NULL COUNTS ===")

df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()

print("=== SOC DISTRIBUTION ===")

df.select(
    F.min("hv_soc").alias("min_soc"),
    F.max("hv_soc").alias("max_soc"),
    F.avg("hv_soc").alias("avg_soc")
).show()

print("=== BATTERY VOLTAGE DISTRIBUTION ===")

df.select(
    F.min("hv_battery_voltage").alias("min_voltage"),
    F.max("hv_battery_voltage").alias("max_voltage"),
    F.avg("hv_battery_voltage").alias("avg_voltage")
).show()

print("=== VEHICLE COUNT ===")

df.select("vehicle_id").distinct().show()

spark.stop()
