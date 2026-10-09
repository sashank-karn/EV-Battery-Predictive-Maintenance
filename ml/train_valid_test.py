from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

spark = (
    SparkSession.builder
    .appName("EVBatteryTrainValidationTest")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/ml_dataset"
MODEL_OUTPUT = "/home/sashank_karn/EV_BigData_CA3/ml/final_model"

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

df = spark.read.parquet(INPUT)

df = df.select(
    "time",
    "maintenance_risk",
    *feature_columns
).orderBy("time")

total_rows = df.count()

train_cutoff = df.select(
    F.expr("percentile_approx(time, 0.6)").alias("cutoff")
).collect()[0]["cutoff"]

validation_cutoff = df.select(
    F.expr("percentile_approx(time, 0.8)").alias("cutoff")
).collect()[0]["cutoff"]

train = df.filter(
    F.col("time") <= train_cutoff
)

validation = df.filter(
    (F.col("time") > train_cutoff) &
    (F.col("time") <= validation_cutoff)
)

test = df.filter(
    F.col("time") > validation_cutoff
)

print()
print("=" * 60)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 60)

print("Total rows:", total_rows)
print("Training rows:", train.count())
print("Validation rows:", validation.count())
print("Testing rows:", test.count())

print("Training cutoff:", train_cutoff)
print("Validation cutoff:", validation_cutoff)

print()
print("TRAINING CLASS DISTRIBUTION")
train.groupBy("maintenance_risk").count().orderBy(
    "maintenance_risk"
).show()

print("VALIDATION CLASS DISTRIBUTION")
validation.groupBy("maintenance_risk").count().orderBy(
    "maintenance_risk"
).show()

print("TEST CLASS DISTRIBUTION")
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

print()
print("Positive class weight:", positive_weight)

train = train.withColumn(
    "class_weight",
    F.when(
        F.col("maintenance_risk") == 1,
        F.lit(positive_weight)
    ).otherwise(F.lit(1.0))
)

imputer = Imputer(
    inputCols=feature_columns,
    outputCols=[
        f"{c}_imputed"
        for c in feature_columns
    ],
    strategy="median"
)

imputed_features = [
    f"{c}_imputed"
    for c in feature_columns
]

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

print()
print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = pipeline.fit(train)

print("Model training completed.")

validation_predictions = model.transform(validation)

validation_predictions = validation_predictions.withColumn(
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
print("=" * 100)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 100)

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

best_threshold = 0.50
best_f1 = -1.0

for threshold in thresholds:

    scored = validation_predictions.withColumn(
        "threshold_prediction",
        F.when(
            F.col("risk_probability") >= threshold,
            1
        ).otherwise(0)
    )

    counts = (
        scored
        .groupBy(
            "maintenance_risk",
            "threshold_prediction"
        )
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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

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

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print()
print("Selected threshold:", best_threshold)
print("Validation F1:", best_f1)

print()
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_predictions = model.transform(test)

test_predictions = test_predictions.withColumn(
    "risk_probability",
    vector_to_array("probability")[1]
)

test_predictions = test_predictions.withColumn(
    "threshold_prediction",
    F.when(
        F.col("risk_probability") >= best_threshold,
        1
    ).otherwise(0)
)

counts = (
    test_predictions
    .groupBy(
        "maintenance_risk",
        "threshold_prediction"
    )
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

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

if precision + recall > 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0

accuracy = (
    (tp + tn)
    / (tp + tn + fp + fn)
)

fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

print("Selected threshold:", best_threshold)

print()
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print()
print("Risk Precision:", precision)
print("Risk Recall   :", recall)
print("Risk F1 Score :", f1)
print("Accuracy      :", accuracy)
print("False Positive Rate:", fpr)

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

rf_model = model.stages[-1]

feature_importance = rf_model.featureImportances

for name, importance in sorted(
    zip(feature_columns, feature_importance),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"{name}: {importance:.6f}"
    )

model.write().overwrite().save(MODEL_OUTPUT)

print()
print("Final model saved to:")
print(MODEL_OUTPUT)

spark.stop()
