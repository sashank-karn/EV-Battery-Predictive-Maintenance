from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

spark = (
    SparkSession.builder
    .appName("EVBatteryPredictiveMaintenance")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"

df = spark.read.parquet(INPUT)

print("\nTotal rows:", df.count())

df = df.orderBy("time", "vehicle_id")

print("\nClass distribution:")
df.groupBy("maintenance_risk").count().orderBy("maintenance_risk").show()

feature_cols = [
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

df = df.select(
    "vehicle_id",
    "time",
    "maintenance_risk",
    *feature_cols
).dropna()

total = df.count()

train_end = int(total * 0.60)
validation_end = int(total * 0.80)

windowed = df.withColumn(
    "row_id",
    F.row_number().over(
        __import__("pyspark").sql.Window.orderBy("time", "vehicle_id")
    )
)

train = windowed.filter(F.col("row_id") <= train_end).drop("row_id")
validation = windowed.filter(
    (F.col("row_id") > train_end) &
    (F.col("row_id") <= validation_end)
).drop("row_id")
test = windowed.filter(F.col("row_id") > validation_end).drop("row_id")

print("\nDataset split:")
print("Training:", train.count())
print("Validation:", validation.count())
print("Test:", test.count())

print("\nTraining class distribution:")
train.groupBy("maintenance_risk").count().orderBy("maintenance_risk").show()

print("\nValidation class distribution:")
validation.groupBy("maintenance_risk").count().orderBy("maintenance_risk").show()

print("\nTest class distribution:")
test.groupBy("maintenance_risk").count().orderBy("maintenance_risk").show()

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol="maintenance_risk",
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42,
    weightCol="classWeight"
)

train_counts = {
    row["maintenance_risk"]: row["count"]
    for row in train.groupBy("maintenance_risk").count().collect()
}

normal_count = train_counts.get(0, 1)
risk_count = train_counts.get(1, 1)

weight_normal = 1.0
weight_risk = normal_count / risk_count

train = train.withColumn(
    "classWeight",
    F.when(
        F.col("maintenance_risk") == 1,
        F.lit(weight_risk)
    ).otherwise(F.lit(weight_normal))
)

validation = validation.withColumn("classWeight", F.lit(1.0))
test = test.withColumn("classWeight", F.lit(1.0))

def print_risk_metrics(predictions, dataset_name):
    metrics = MulticlassMetrics(
        predictions.select("prediction", "maintenance_risk")
        .rdd
        .map(lambda row: (float(row["prediction"]), float(row["maintenance_risk"])))
    )
    print(f"\\n{dataset_name} risk-class metrics:")
    print(f"Risk precision: {metrics.precision(1.0):.4f}")
    print(f"Risk recall: {metrics.recall(1.0):.4f}")
    print(f"Risk F1: {metrics.fMeasure(1.0):.4f}")


pipeline = Pipeline(stages=[assembler, rf])

print("\nTraining Random Forest...")

model = pipeline.fit(train)

print("Training completed.")

validation_predictions = model.transform(validation)
print_risk_metrics(validation_predictions, "Validation")

print("\nValidation metrics:")

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="f1"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="weightedRecall"
)

auc_eval = BinaryClassificationEvaluator(
    labelCol="maintenance_risk",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

print("Accuracy:", accuracy_eval.evaluate(validation_predictions))
print("F1:", f1_eval.evaluate(validation_predictions))
print("Weighted Precision:", precision_eval.evaluate(validation_predictions))
print("Weighted Recall:", recall_eval.evaluate(validation_predictions))
print("ROC-AUC:", auc_eval.evaluate(validation_predictions))

print("\nValidation confusion matrix:")
validation_predictions.groupBy(
    "maintenance_risk",
    "prediction"
).count().orderBy(
    "maintenance_risk",
    "prediction"
).show()

test_predictions = model.transform(test)
print_risk_metrics(test_predictions, "Test")

print("\nTest metrics:")

test_accuracy = accuracy_eval.evaluate(test_predictions)
test_f1 = f1_eval.evaluate(test_predictions)
test_precision = precision_eval.evaluate(test_predictions)
test_recall = recall_eval.evaluate(test_predictions)
test_auc = auc_eval.evaluate(test_predictions)

print("Accuracy:", test_accuracy)
print("F1:", test_f1)
print("Weighted Precision:", test_precision)
print("Weighted Recall:", test_recall)
print("ROC-AUC:", test_auc)

print("\nTest confusion matrix:")
test_predictions.groupBy(
    "maintenance_risk",
    "prediction"
).count().orderBy(
    "maintenance_risk",
    "prediction"
).show()

rf_model = model.stages[-1]

print("\nFeature Importance:")

importance = list(zip(feature_cols, rf_model.featureImportances.toArray()))

for feature, score in sorted(
    importance,
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {score:.6f}")

model_path = "/home/sashank_karn/EV_BigData_CA3/ml/final_model"

model.write().overwrite().save(model_path)

print("\nFinal model saved to:", model_path)

spark.stop()
