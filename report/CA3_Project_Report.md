# Predictive Maintenance of EV Battery Systems Using Apache Kafka, Spark Streaming and MLlib

## Big Data Analytics Project Report

**Project type:** Big Data Analytics case study and implementation  
**Primary technologies:** Apache Kafka, Apache Spark Structured Streaming, Spark MLlib, Parquet, Python and Streamlit

## Abstract

Electric vehicles produce telemetry that can be used to inspect operating conditions and investigate abnormal battery behaviour. This project builds a data-processing and experimental predictive-maintenance pipeline around electric-vehicle telemetry stored in Parquet files. The workflow normalizes timestamps, filters selected signals, aggregates observations into 10-second windows, engineers voltage-related features, and trains a Random Forest classifier with Spark MLlib. Kafka and Spark Structured Streaming demonstrate a streaming inference path, while a Streamlit dashboard supports historical exploration and model evaluation.

The source data was processed into 853,128 telemetry windows and 853,106 machine-learning rows. The experimental risk label is based on a future voltage-change rule rather than verified battery failures. The positive class represents approximately 0.54% of the machine-learning dataset. At a probability threshold of 0.70 selected using validation results, the held-out test precision was 8.90%, recall was 8.49%, and F1-score was 0.0869. These results show that the current classifier is a baseline for further investigation, not a reliable operational battery-failure detector.

## 1. Introduction

Battery monitoring is important to electric-vehicle operation because battery measurements vary with driving conditions, state of charge, temperature and electrical load. Big data tools can help transform large volumes of telemetry into structured time windows and derived features that can be inspected or used in predictive modelling.

This project demonstrates a practical workflow spanning batch processing, streaming infrastructure, machine learning and visualization. It also documents the limitations of using a proxy label when verified fault or maintenance records are unavailable.

## 2. Problem Statement

EV telemetry is distributed across large Parquet files and includes multiple signal identifiers and timestamps. Directly analysing these raw records is inconvenient, and timestamp inconsistencies can affect downstream processing. The project therefore aims to normalize the source data, aggregate selected battery and vehicle signals, investigate future voltage-change patterns, and demonstrate a streaming architecture that can score incoming telemetry.

The model target is an experimental proxy for abnormal voltage behaviour. It is not a ground-truth battery failure label.

## 3. Objectives

1. Normalize source telemetry and correct the identified CUP1 timestamp anomaly.
2. Filter selected vehicle and battery signals and aggregate readings into 10-second windows.
3. Engineer voltage-change and operating-condition features.
4. Create a future-voltage-change proxy label for supervised experiments.
5. Train a Random Forest classifier with chronological training, validation and test partitions.
6. Demonstrate telemetry publishing through Kafka and inference using Spark Structured Streaming.
7. Provide a Streamlit dashboard for historical telemetry and held-out model evaluation.
8. Report class imbalance and classification errors transparently.

## 4. Dataset and Data Preparation

The project uses the electric-vehicle-uds-dataset, including CUP and ID Parquet files. The source schema contains vehicle identifier, timestamp, signal identifier and numeric signal value. The files contain approximately 98.1 million raw records in total.

### 4.1 Timestamp normalization

The original CUP1 data contained timestamps with an anomalous future-year offset. A corrected CUP1 source was created before normalization. The normalization script standardizes timestamp precision and writes normalized Parquet sources so Spark can read the data consistently.

### 4.2 Selected signals

The main processing and modelling steps use selected signals corresponding to vehicle speed, ambient air temperature, auxiliary high-voltage power, state of charge and high-voltage battery voltage. The historical preprocessing workflow also derives minimum and maximum battery temperature features when those temperature signals are available.

### 4.3 Processing scale

| Source | Normalized rows |
|---|---:|
| CUP1 | 8,659,492 |
| CUP2 | 3,743,488 |
| CUP3 | 986,864 |
| CUP4 | 4,943,471 |
| CUP5 | 3,305,981 |
| ID1 | 40,401,700 |
| ID2 | 36,020,218 |
| **Total** | **98,061,214** |

The telemetry preprocessing stage produced **853,128** 10-second windows. The machine-learning dataset contains **853,106** rows after feature creation and removal of rows that lack the required previous or future observations.

## 5. System Architecture

The implementation contains the following logical stages:

1. **Source data:** EV telemetry stored in Parquet files.
2. **Normalization:** timestamp correction and normalized Parquet output.
3. **Batch processing:** Spark filters selected signal IDs and aggregates measurements into 10-second windows.
4. **Feature engineering:** derive voltage-per-SOC, previous voltage and SOC, voltage change and SOC change.
5. **Label creation:** inspect the future voltage-change horizon and assign a proxy risk label.
6. **Model training:** train a Spark MLlib Random Forest classifier and evaluate it on chronological splits.
7. **Streaming demonstration:** a Python producer publishes telemetry records to Kafka; Spark Structured Streaming consumes records and performs batch inference.
8. **Dashboard:** Streamlit and Plotly display historical telemetry and model evaluation metrics.

### 5.1 Technology roles

- **Apache Spark:** distributed Parquet processing, time-window aggregation and model training.
- **Apache Kafka:** telemetry transport using the `battery-telemetry` topic.
- **Spark Structured Streaming:** consumes Kafka records and runs inference over micro-batches.
- **Spark MLlib:** Random Forest classification and model evaluation.
- **Parquet/PyArrow:** columnar storage and inspection of generated datasets.
- **Streamlit/Plotly:** interactive historical telemetry and evaluation dashboard.

## 6. Methodology

### 6.1 Ten-second aggregation

Selected telemetry records are grouped by vehicle and 10-second time windows. Signal values are pivoted into model-friendly columns. The preprocessing output includes speed, ambient temperature, auxiliary power, state of charge, battery voltage and battery-temperature summaries where available.

### 6.2 Feature engineering

The machine-learning workflow derives the following features:

- Vehicle speed
- Ambient air temperature
- Auxiliary high-voltage power
- State of charge (SOC)
- High-voltage battery voltage
- Voltage divided by SOC, with safeguards required for invalid or zero SOC
- Previous observed voltage and SOC
- Voltage change and SOC change

### 6.3 Proxy target

A window is labelled as risk when the maximum absolute voltage change over a future observation horizon exceeds **20 V**. This threshold is project-defined for experimentation. It is not validated against manufacturer specifications, diagnostic fault codes, or verified maintenance outcomes. The label should therefore be described as a future-voltage-change proxy, not an actual battery-failure label.

### 6.4 Chronological split and classifier

Rows are partitioned chronologically into training, validation and test sets to reduce the risk of evaluating on future records during model development. A Random Forest classifier with 100 trees and maximum depth 8 was trained using Spark MLlib. The risk class is rare, so class weighting and threshold analysis were used to investigate the effect of the decision threshold.

The threshold of 0.70 was selected from validation threshold analysis. The held-out test set was then evaluated at that threshold.

## 7. Experimental Results

### 7.1 Label distribution

| Class | Rows |
|---|---:|
| Normal proxy label (0) | 848,515 |
| Risk proxy label (1) | 4,591 |
| **Total** | **853,106** |

The risk class accounts for approximately **0.538%** of rows. This severe class imbalance means accuracy alone would be misleading.

### 7.2 Held-out test results at threshold 0.70

| Metric | Result |
|---|---:|
| Precision | 0.0890 (8.90%) |
| Recall | 0.0849 (8.49%) |
| F1-score | 0.0869 |
| True positives | 68 |
| False positives | 696 |
| False negatives | 733 |

At this threshold, the classifier identified 68 positive proxy-label cases, produced 696 false alarms, and missed 733 positive cases. Precision and recall are both low. The experiment therefore does not establish that the model can reliably predict battery failures.

### 7.3 Interpretation

The results are limited by the rarity of the proxy positive class and by the proxy label itself. Voltage changes may reflect operating conditions or data characteristics rather than a fault. Further work should compare multiple thresholds using validation data, report precision-recall curves and class-specific metrics, and evaluate on verified fault or maintenance records if available. Any threshold chosen for deployment would require independent validation and domain review.

## 8. Kafka and Spark Streaming Demonstration

The streaming demonstration uses a Python producer to publish JSON telemetry records to the Kafka topic `battery-telemetry`. Spark Structured Streaming reads from Kafka, filters the selected signals, aggregates records into event-time windows and prepares model features for inference. Predictions are printed from micro-batches.

This implementation demonstrates the integration path, but the current inference stream is not a production service: predictions are not continuously persisted, and the streaming state and feature assembly require further engineering for robust deployment. The dashboard currently presents historical data and held-out evaluation results rather than live inference output.

## 9. Dashboard

The Streamlit dashboard supports exploration of processed historical telemetry, including battery voltage, SOC, speed, ambient temperature, available battery-temperature summaries, recent records and a CSV export. Its model evaluation section displays test metrics across decision thresholds and allows the user to inspect the precision-recall trade-off.

The dashboard should be presented as a historical analytics and evaluation interface, not as a live vehicle monitoring system.

## 10. Limitations

1. The target is a project-defined voltage-change proxy, not verified battery degradation, failure or maintenance data.
2. The 20 V threshold is experimental and is not a manufacturer-certified safety threshold.
3. The positive class is approximately 0.54% of the dataset, making evaluation sensitive to class imbalance.
4. The reported precision, recall and F1-score at threshold 0.70 are low and indicate weak risk-class detection.
5. The current streaming inference output is printed to the console rather than stored in a durable prediction database.
6. The dashboard shows historical telemetry and evaluation results, not live streaming predictions.
7. The source data contains timestamp anomalies that require careful normalization and validation.
8. Results should not be used for real-world battery safety, maintenance or warranty decisions.

## 11. Future Work

- Obtain verified diagnostic, fault or maintenance labels and validate the target with domain experts.
- Compare the Random Forest baseline with calibrated models and suitable anomaly-detection methods.
- Use precision-recall curves, PR-AUC, class-specific confusion matrices and threshold selection based only on validation data.
- Investigate vehicle-level or time-based holdouts to test generalization to unseen vehicles and periods.
- Persist streaming predictions and monitoring metadata in a durable storage layer.
- Add live prediction status and stream-health indicators to the dashboard.
- Add automated tests for timestamp normalization, feature generation, label construction and streaming schema.
- Document data provenance and dataset licensing before redistribution.

## 12. Conclusion

This project demonstrates a complete educational workflow combining large-scale Parquet processing, Spark time-window aggregation, Kafka telemetry transport, Spark MLlib classification and a Streamlit analytics dashboard. It processes approximately 98.1 million source records into 853,128 telemetry windows and 853,106 machine-learning rows. The experiment also reveals a key limitation: the current proxy-label classifier performs poorly on the rare risk class. The main outcome is therefore a reproducible big-data engineering demonstration and baseline for further research, rather than a validated predictive-maintenance product.

## References

1. Apache Spark Documentation. https://spark.apache.org/docs/latest/
2. Apache Kafka Documentation. https://kafka.apache.org/documentation/
3. Streamlit Documentation. https://docs.streamlit.io/
4. Plotly Python Documentation. https://plotly.com/python/
5. EV telemetry source dataset: electric-vehicle-uds-dataset used in the project. Consult the original dataset repository and its licence/provenance details before formal publication or redistribution.
