# Predictive Maintenance of EV Battery Systems

A Big Data Analytics project using Apache Kafka, Apache Spark Structured Streaming, and Spark MLlib to process electric vehicle telemetry and investigate abnormal battery-voltage patterns.

## Objectives

- Process large-scale EV telemetry stored in Parquet format.
- Normalize timestamps and aggregate readings into 10-second windows.
- Stream telemetry through Kafka.
- Train and evaluate a Random Forest classifier using Spark MLlib.
- Explore historical battery telemetry and model evaluation in a Streamlit dashboard.

## Technology Stack

- Apache Kafka 4.2.2 (KRaft mode)
- Apache Spark 4.2.0
- Spark Structured Streaming and MLlib
- Python, PyArrow and Parquet
- Streamlit and Plotly

## Architecture

1. **Data source:** EV telemetry from the electric-vehicle-uds-dataset.
2. **Normalization:** Standardize timestamps and create normalized Parquet sources.
3. **Processing:** Filter selected signals and aggregate readings into 10-second windows.
4. **Feature engineering:** Generate voltage-related features and an experimental future-voltage-change label.
5. **Model training:** Train a Random Forest classifier using chronological training, validation and test splits.
6. **Streaming inference:** Publish telemetry to Kafka and process incoming records with Spark Structured Streaming.
7. **Visualization:** Explore historical telemetry and held-out model evaluation metrics in Streamlit.

## Repository Structure

- `spark/` - preprocessing, feature engineering, model training and streaming scripts
- `kafka/` - telemetry producer and consumer test
- `dashboard/` - Streamlit dashboard
- `report/` - [project report](report/CA3_Project_Report.md) and evaluation results
- `screenshots/` - project evidence and demonstration screenshots
- `dataset/` - local source data and generated Parquet files, excluded from Git
- `ml/` - locally generated model artifacts, excluded from Git
- `checkpoints/` - streaming checkpoint data, excluded from Git
- `requirements.txt` - Python dependencies for dashboard and data utilities

## Prerequisites

Install and configure Java, Apache Spark 4.2.0, Apache Kafka 4.2.2 in KRaft mode, Python 3, and the source EV telemetry dataset separately. The data and trained model are not committed to this repository.

The Spark Kafka connector used by the streaming job is:

`org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0`

## Python Environment

Run from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Batch Pipeline

Place the source dataset in the local location expected by the scripts. Run these commands in order:

```bash
python spark/normalize_sources.py
spark-submit spark/preprocessing.py
spark-submit spark/create_ml_dataset.py
spark-submit spark/train_valid_test.py
```

The training step creates the local model artifact required for streaming inference. Generated Parquet files and model artifacts are excluded from Git.

## Kafka Streaming

Start Kafka in KRaft mode and ensure the `battery-telemetry` topic exists. In one terminal, publish sample telemetry:

```bash
source .venv/bin/activate
python kafka/producer.py
```

In a separate terminal, run Spark streaming inference:

```bash
spark-submit spark/streaming_ml.py
```

The producer reads a locally prepared CUP1 Parquet file. The streaming query consumes Kafka messages and prints batch inference output to the console. This is a demonstration pipeline, not a continuously persisted prediction service.

## Dashboard

After generating the processed datasets and evaluation results, run:

```bash
source .venv/bin/activate
python -m streamlit run dashboard/app.py
```

The dashboard displays historical telemetry and model evaluation results, not live streaming predictions.

## Model Evaluation

The experimental label marks a window as risk when the maximum absolute voltage change over a future observation horizon exceeds 20 V. This is a project-defined proxy, not a confirmed battery failure label or a manufacturer-certified safety threshold.

The risk class is rare: 4,591 of 853,106 windows (approximately 0.54%). At the threshold of 0.70 selected using validation results, held-out test metrics were:

| Metric | Result |
|---|---:|
| Precision | 8.90% |
| Recall | 8.49% |
| F1-score | 0.0869 |
| True positives | 68 |
| False positives | 696 |
| False negatives | 733 |

The baseline has limited ability to identify the proxy risk class and produces many false alarms. These results must not be interpreted as a production-ready battery-failure predictor.

## Reproducibility

- Prepare the source dataset separately.
- Run normalization and processing before training.
- Generate the model before running streaming inference.
- Start Kafka before running the producer and streaming job.
- Keep generated datasets, model artifacts and checkpoints local; they are excluded from Git.

## Academic Project

Developed as a Big Data Analytics case study on EV telemetry processing and experimental predictive maintenance.
