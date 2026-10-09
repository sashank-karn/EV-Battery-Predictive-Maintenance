from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("EVBatteryPreprocessing")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/normalized_sources"
OUTPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/processed"

selected_ids = [4, 15, 56, 900, 1200, 1208, 1209]

df = spark.read.parquet(INPUT)

df = df.filter(
    F.col("value_id").isin(selected_ids)
)

aggregated = (
    df.groupBy(
        "vehicle_id",
        F.window("time", "10 seconds").alias("time_window")
    )
    .agg(
        F.avg(F.when(F.col("value_id") == 4, F.col("value"))).alias("vehicle_speed"),
        F.avg(F.when(F.col("value_id") == 15, F.col("value"))).alias("ambient_air_temp"),
        F.avg(F.when(F.col("value_id") == 56, F.col("value"))).alias("hv_aux_power"),
        F.avg(F.when(F.col("value_id") == 900, F.col("value"))).alias("hv_soc"),
        F.avg(F.when(F.col("value_id") == 1200, F.col("value"))).alias("hv_battery_voltage"),
        F.avg(F.when(F.col("value_id") == 1208, F.col("value"))).alias("hv_temp_min"),
        F.avg(F.when(F.col("value_id") == 1209, F.col("value"))).alias("hv_temp_max")
    )
)

result = (
    aggregated
    .select(
        "vehicle_id",
        F.col("time_window.start").alias("time"),
        F.col("time_window.end").alias("window_end"),
        "vehicle_speed",
        "ambient_air_temp",
        "hv_aux_power",
        "hv_soc",
        "hv_battery_voltage",
        "hv_temp_min",
        "hv_temp_max"
    )
    .withColumn(
        "battery_temp_avg",
        (F.col("hv_temp_min") + F.col("hv_temp_max")) / 2
    )
    .withColumn(
        "battery_temp_delta",
        F.col("hv_temp_max") - F.col("hv_temp_min")
    )
    .withColumn(
        "temperature_available",
        F.when(
            F.col("hv_temp_min").isNotNull() &
            F.col("hv_temp_max").isNotNull(),
            1
        ).otherwise(0)
    )
    .filter(
        F.col("hv_soc").isNotNull() &
        F.col("hv_battery_voltage").isNotNull()
    )
)

result.write.mode("overwrite").parquet(OUTPUT)

print("Preprocessing completed successfully.")
print("Output:", OUTPUT)

result.printSchema()

spark.stop()
