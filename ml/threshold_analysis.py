from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

spark = (
    SparkSession.builder
    .appName("EVBatteryThresholdAnalysis")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"
MODEL = "/home/sashank_karn/EV_BigData_CA3/ml/model"

df = spark.read.parquet(INPUT)

feature_columns = [
    "hv_soc",
    "hv_battery_voltage",
    "vehicle_speed",
    "ambient_air_temp",
    "hv_aux_power",
    "voltage_per_soc",
    "previous_voltage",
    "previous_soc",
    "voltage_change",
    "soc_change"
]

df = df.select(
    "time",
    "maintenance_risk",
    *feature_columns
).orderBy("time")

split_time = df.select(
    F.expr("percentile_approx(time, 0.8)").alias("split_time")
).collect()[0]["split_time"]

test = df.filter(F.col("time") > split_time)

model = PipelineModel.load(MODEL)

predictions = model.transform(test)

predictions = predictions.withColumn(
    "risk_probability",
    vector_to_array("probability")[1]
)

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

print()
print("=" * 95)
print("THRESHOLD ANALYSIS")
print("=" * 95)

print(
    f"{'Threshold':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'TN':<8}"
    f"{'Precision':<14}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)

for threshold in thresholds:

    scored = predictions.withColumn(
        "threshold_prediction",
        F.when(
            F.col("risk_probability") >= threshold,
            1
        ).otherwise(0)
    )

    counts = (
        scored
        .groupBy("maintenance_risk", "threshold_prediction")
        .count()
        .collect()
    )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in counts:

        actual = row["maintenance_risk"]
        predicted = row["threshold_prediction"]
        count = row["count"]

        if actual == 1 and predicted == 1:
            tp = count
        elif actual == 0 and predicted == 0:
            tn = count
        elif actual == 0 and predicted == 1:
            fp = count
        elif actual == 1 and predicted == 0:
            fn = count

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(
        f"{threshold:<12.2f}"
        f"{tp:<8}"
        f"{fp:<8}"
        f"{fn:<8}"
        f"{tn:<8}"
        f"{precision:<14.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fpr:<12.4f}"
    )

print()
print("Threshold analysis completed.")

spark.stop()
