from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = (
    SparkSession.builder
    .appName("EVBatteryPredictiveMaintenance")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"
MODEL_OUTPUT = "/home/sashank_karn/EV_BigData_CA3/ml/model"

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
)

df = df.orderBy("time")

split_time = df.select(
    F.expr("percentile_approx(time, 0.8)").alias("split_time")
).collect()[0]["split_time"]

train = df.filter(F.col("time") <= split_time)
test = df.filter(F.col("time") > split_time)

print("=== DATA SPLIT ===")
print("Training rows:", train.count())
print("Testing rows:", test.count())
print("Split time:", split_time)

print("=== TRAIN CLASS DISTRIBUTION ===")
train.groupBy("maintenance_risk").count().orderBy(
    "maintenance_risk"
).show()

print("=== TEST CLASS DISTRIBUTION ===")
test.groupBy("maintenance_risk").count().orderBy(
    "maintenance_risk"
).show()

class_counts = {
    row["maintenance_risk"]: row["count"]
    for row in train.groupBy("maintenance_risk").count().collect()
}

negative_count = class_counts.get(0, 1)
positive_count = class_counts.get(1, 1)

positive_weight = negative_count / positive_count

train = train.withColumn(
    "class_weight",
    F.when(
        F.col("maintenance_risk") == 1,
        F.lit(positive_weight)
    ).otherwise(F.lit(1.0))
)

imputer = Imputer(
    inputCols=feature_columns,
    outputCols=[f"{c}_imputed" for c in feature_columns],
    strategy="median"
)

imputed_features = [f"{c}_imputed" for c in feature_columns]

assembler = VectorAssembler(
    inputCols=imputed_features,
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol="maintenance_risk",
    featuresCol="features",
    weightCol="class_weight",
    numTrees=100,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(
    stages=[
        imputer,
        assembler,
        rf
    ]
)

print("=== TRAINING MODEL ===")

model = pipeline.fit(train)

print("Model training completed.")

predictions = model.transform(test)

print("=== PREDICTIONS ===")

predictions.select(
    "time",
    "maintenance_risk",
    "prediction",
    "probability"
).show(20, truncate=False)

accuracy = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="maintenance_risk",
    predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

print("=== MODEL METRICS ===")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("=== CONFUSION MATRIX ===")

predictions.groupBy(
    "maintenance_risk",
    "prediction"
).count().orderBy(
    "maintenance_risk",
    "prediction"
).show()

model.write().overwrite().save(MODEL_OUTPUT)

print("Model saved to:", MODEL_OUTPUT)

spark.stop()
