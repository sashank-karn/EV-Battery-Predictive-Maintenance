
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType
)
from pyspark.ml import PipelineModel

spark = (
    SparkSession.builder
    .appName("EVBatteryStreamingML")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

MODEL_PATH = "/home/sashank_karn/EV_BigData_CA3/ml/final_model"
CHECKPOINT_PATH = "/home/sashank_karn/EV_BigData_CA3/checkpoints/streaming_ml"

model = PipelineModel.load(MODEL_PATH)

schema = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("time", StringType(), True),
    StructField("value_id", IntegerType(), True),
    StructField("value", DoubleType(), True)
])

feature_schema = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("time", TimestampType(), True),
    StructField("vehicle_speed", DoubleType(), True),
    StructField("ambient_air_temp", DoubleType(), True),
    StructField("hv_aux_power", DoubleType(), True),
    StructField("hv_soc", DoubleType(), True),
    StructField("hv_battery_voltage", DoubleType(), True),
    StructField("voltage_per_soc", DoubleType(), True),
    StructField("previous_voltage", DoubleType(), True),
    StructField("previous_soc", DoubleType(), True),
    StructField("voltage_change", DoubleType(), True),
    StructField("soc_change", DoubleType(), True)
])

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "battery-telemetry")
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

telemetry = (
    kafka_df
    .select(
        F.from_json(
            F.col("value").cast("string"),
            schema
        ).alias("data")
    )
    .select("data.*")
    .withColumn("event_time", F.to_timestamp("time"))
    .filter(
        F.col("value_id").isin(4, 15, 56, 900, 1200)
        & F.col("event_time").isNotNull()
        & F.col("value").isNotNull()
    )
)

windowed = (
    telemetry
    .withWatermark("event_time", "30 seconds")
    .groupBy(
        "vehicle_id",
        F.window("event_time", "10 seconds")
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
    .select(
        "vehicle_id",
        F.col("window.start").alias("time"),
        "vehicle_speed",
        "ambient_air_temp",
        "hv_aux_power",
        "hv_soc",
        "hv_battery_voltage"
    )
)

last_state = {}

def process_batch(batch_df, batch_id):
    global last_state

    print("\n" + "=" * 90)
    print("MICROBATCH:", batch_id)
    print("=" * 90)

    valid_batch = batch_df.filter(
        F.col("hv_soc").isNotNull()
        & F.col("hv_battery_voltage").isNotNull()
        & (F.col("hv_soc") > 0)
        & (F.col("hv_battery_voltage") >= 300)
        & (F.col("hv_battery_voltage") <= 500)
    )

    if valid_batch.isEmpty():
        print("No valid completed windows in this batch.")
        return

    rows = (
        valid_batch
        .orderBy("vehicle_id", "time")
        .collect()
    )

    feature_rows = []

    for row in rows:
        vehicle_id = row["vehicle_id"]
        current_voltage = float(row["hv_battery_voltage"])
        current_soc = float(row["hv_soc"])

        previous = last_state.get(vehicle_id)

        if previous is None:
            previous_voltage = current_voltage
            previous_soc = current_soc
            voltage_change = 0.0
            soc_change = 0.0
        else:
            previous_voltage = previous["voltage"]
            previous_soc = previous["soc"]

            voltage_change = current_voltage - previous_voltage
            soc_change = current_soc - previous_soc

        voltage_per_soc = current_voltage / current_soc

        feature_rows.append({
            "vehicle_id": vehicle_id,
            "time": row["time"],
            "vehicle_speed": row["vehicle_speed"],
            "ambient_air_temp": row["ambient_air_temp"],
            "hv_aux_power": row["hv_aux_power"],
            "hv_soc": current_soc,
            "hv_battery_voltage": current_voltage,
            "voltage_per_soc": voltage_per_soc,
            "previous_voltage": previous_voltage,
            "previous_soc": previous_soc,
            "voltage_change": voltage_change,
            "soc_change": soc_change
        })

        last_state[vehicle_id] = {
            "voltage": current_voltage,
            "soc": current_soc,
            "time": row["time"]
        }

    if not feature_rows:
        print("No feature rows generated.")
        return

    feature_df = spark.createDataFrame(
        feature_rows,
        schema=feature_schema
    )

    feature_df = feature_df.fillna(
        0.0,
        subset=[
            "vehicle_speed",
            "ambient_air_temp",
            "hv_aux_power",
            "voltage_per_soc",
            "previous_voltage",
            "previous_soc",
            "voltage_change",
            "soc_change"
        ]
    )

    predictions = model.transform(feature_df)

    result = (
        predictions
        .select(
            "vehicle_id",
            "time",
            "hv_soc",
            "hv_battery_voltage",
            "voltage_change",
            "soc_change",
            "prediction",
            "probability"
        )
        .orderBy("vehicle_id", "time")
    )

    result.show(20, truncate=False)

    print("Rows processed:", result.count())

query = (
    windowed
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("update")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(processingTime="5 seconds")
    .start()
)

print("Streaming ML inference started...")
print("Kafka topic: battery-telemetry")
print("Model:", MODEL_PATH)

query.awaitTermination()
