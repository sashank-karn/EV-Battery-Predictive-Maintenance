from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel

spark = (
    SparkSession.builder
    .appName("EVBatteryTestThresholdEvaluation")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

base = "/home/sashank_karn/EV_BigData_CA3"

features = [
    "vehicle_speed",
    "ambient_air_temp",
    "hv_aux_power",
    "hv_soc",
    "hv_battery_voltage",
    "voltage_per_soc",
    "previous_voltage",
    "previous_soc",
    "voltage_change",
    "soc_change"
]

data = spark.read.parquet(f"{base}/dataset/ml_dataset")
data = data.select(
    "vehicle_id", "time", "maintenance_risk", *features
).dropna()

total = data.count()

data = data.withColumn(
    "row_id",
    F.row_number().over(Window.orderBy("time", "vehicle_id"))
)

train_end = int(total * 0.60)
validation_end = int(total * 0.80)

test = data.filter(
    F.col("row_id") > validation_end
).drop("row_id")

model = PipelineModel.load(f"{base}/ml/final_model")

predictions = model.transform(test).select(
    "maintenance_risk", "probability"
)

probability_one = F.udf(
    lambda v: float(v[1]) if v is not None and len(v) > 1 else 0.0,
    "double"
)

predictions = predictions.withColumn(
    "risk_probability", probability_one("probability")
).cache()

print(f"Test rows: {predictions.count()}")
print("\nTest-set threshold comparison")
print("Threshold | Precision | Recall | F1 | TP | FP | FN")

for threshold in [0.50, 0.60, 0.70, 0.80]:
    evaluated = predictions.withColumn(
        "predicted_risk",
        (F.col("risk_probability") >= threshold).cast("int")
    )

    counts = {
        (int(r["maintenance_risk"]), int(r["predicted_risk"])): r["count"]
        for r in evaluated.groupBy(
            "maintenance_risk", "predicted_risk"
        ).count().collect()
    }

    tp = counts.get((1, 1), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(
        f"{threshold:9.2f} | {precision:9.4f} | {recall:6.4f} | "
        f"{f1:6.4f} | {tp:4d} | {fp:5d} | {fn:4d}"
    )

predictions.unpersist()
spark.stop()
