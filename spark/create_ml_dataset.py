from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("EVBatteryMLDataset")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/normalized_sources"
OUTPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"

df = spark.read.parquet(INPUT)

selected_ids = [4, 15, 56, 900, 1200]

df = df.filter(F.col("value_id").isin(selected_ids))

aggregated = (
    df.groupBy(
        "vehicle_id",
        F.window("time", "10 seconds").alias("time_window")
    )
    .agg(
        F.avg(
            F.when(F.col("value_id") == 4, F.col("value"))
        ).alias("vehicle_speed"),

        F.avg(
            F.when(F.col("value_id") == 15, F.col("value"))
        ).alias("ambient_air_temp"),

        F.avg(
            F.when(F.col("value_id") == 56, F.col("value"))
        ).alias("hv_aux_power"),

        F.avg(
            F.when(F.col("value_id") == 900, F.col("value"))
        ).alias("hv_soc"),

        F.avg(
            F.when(F.col("value_id") == 1200, F.col("value"))
        ).alias("hv_battery_voltage")
    )
)

data = (
    aggregated
    .select(
        "vehicle_id",
        F.col("time_window.start").alias("time"),
        "vehicle_speed",
        "ambient_air_temp",
        "hv_aux_power",
        "hv_soc",
        "hv_battery_voltage"
    )
)

data = data.filter(
    (F.col("hv_battery_voltage") >= 300) &
    (F.col("hv_battery_voltage") <= 500)
)

data = data.filter(
    F.col("hv_soc").isNotNull()
)

data = data.withColumn(
    "voltage_per_soc",
    F.when(
        F.col("hv_soc") > 0,
        F.col("hv_battery_voltage") / F.col("hv_soc")
    )
)

window_spec = (
    Window
    .partitionBy("vehicle_id")
    .orderBy("time")
)

data = (
    data
    .withColumn(
        "previous_voltage",
        F.lag("hv_battery_voltage", 1).over(window_spec)
    )
    .withColumn(
        "previous_soc",
        F.lag("hv_soc", 1).over(window_spec)
    )
)

data = (
    data
    .withColumn(
        "voltage_change",
        F.col("hv_battery_voltage") -
        F.col("previous_voltage")
    )
    .withColumn(
        "soc_change",
        F.col("hv_soc") -
        F.col("previous_soc")
    )
)

future_window = (
    Window
    .partitionBy("vehicle_id")
    .orderBy("time")
    .rowsBetween(1, 6)
)

data = data.withColumn(
    "future_max_voltage_change",
    F.max(
        F.abs(F.col("voltage_change"))
    ).over(future_window)
)

data = data.withColumn(
    "maintenance_risk",
    F.when(
        F.col("future_max_voltage_change") > 20,
        1
    ).otherwise(0)
)

data = data.filter(
    F.col("previous_voltage").isNotNull() &
    F.col("previous_soc").isNotNull() &
    F.col("future_max_voltage_change").isNotNull()
)

data = data.dropDuplicates(
    ["vehicle_id", "time"]
)

print("=== ML DATASET ===")

print("Total rows:", data.count())

print("=== CLASS DISTRIBUTION ===")

data.groupBy("maintenance_risk").count().orderBy(
    "maintenance_risk"
).show()

print("=== SAMPLE DATA ===")

data.select(
    "vehicle_id",
    "time",
    "hv_soc",
    "hv_battery_voltage",
    "vehicle_speed",
    "ambient_air_temp",
    "hv_aux_power",
    "voltage_change",
    "soc_change",
    "future_max_voltage_change",
    "maintenance_risk"
).show(20, truncate=False)

print("=== FINAL SCHEMA ===")

data.printSchema()

data.write.mode("overwrite").parquet(OUTPUT)

print("ML dataset written to:", OUTPUT)

spark.stop()
