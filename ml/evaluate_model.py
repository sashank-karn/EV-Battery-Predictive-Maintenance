from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("EVBatteryModelEvaluation")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"
MODEL_PATH = "/home/sashank_karn/EV_BigData_CA3/ml/model"

df = spark.read.parquet(INPUT)

df = df.select(
    "time",
    "maintenance_risk",
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
).orderBy("time")

split_time = df.select(
    F.expr("percentile_approx(time, 0.8)").alias("split_time")
).collect()[0]["split_time"]

test = df.filter(F.col("time") > split_time)

model = PipelineModel.load(MODEL_PATH)

predictions = model.transform(test)

tp = predictions.filter(
    (F.col("maintenance_risk") == 1) &
    (F.col("prediction") == 1)
).count()

fp = predictions.filter(
    (F.col("maintenance_risk") == 0) &
    (F.col("prediction") == 1)
).count()

fn = predictions.filter(
    (F.col("maintenance_risk") == 1) &
    (F.col("prediction") == 0)
).count()

tn = predictions.filter(
    (F.col("maintenance_risk") == 0) &
    (F.col("prediction") == 0)
).count()

risk_precision = tp / (tp + fp) if (tp + fp) else 0
risk_recall = tp / (tp + fn) if (tp + fn) else 0

risk_f1 = (
    2 * risk_precision * risk_recall /
    (risk_precision + risk_recall)
    if (risk_precision + risk_recall)
    else 0
)

roc_auc = BinaryClassificationEvaluator(
    labelCol="maintenance_risk",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(predictions)

pr_auc = BinaryClassificationEvaluator(
    labelCol="maintenance_risk",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
).evaluate(predictions)

print("\n==============================")
print("MINORITY CLASS EVALUATION")
print("==============================")

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print("\nRisk Precision:", risk_precision)
print("Risk Recall   :", risk_recall)
print("Risk F1 Score :", risk_f1)

print("\nROC-AUC:", roc_auc)
print("PR-AUC :", pr_auc)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

rf_model = model.stages[-1]

feature_names = [
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

for name, importance in sorted(
    zip(feature_names, rf_model.featureImportances),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {importance:.6f}")

print("\nEvaluation completed.")

spark.stop()
