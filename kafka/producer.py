import json
import time
import pyarrow.parquet as pq
from kafka import KafkaProducer

INPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/cup1_fixed.parquet"
TOPIC = "battery-telemetry"

MAX_MESSAGES = 1000
BATCH_SIZE = 1000

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

parquet_file = pq.ParquetFile(INPUT)

print("Total records:", parquet_file.metadata.num_rows)
print("Starting Kafka producer...")

count = 0

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["vehicle_id", "time", "value_id", "value"]
):
    rows = batch.to_pylist()

    for row in rows:

        message = {
            "vehicle_id": row["vehicle_id"],
            "time": row["time"].isoformat(),
            "value_id": int(row["value_id"]),
            "value": float(row["value"])
        }

        producer.send(TOPIC, value=message)

        count += 1

        if count % 100 == 0:
            producer.flush()
            print("Messages sent:", count)

        if count >= MAX_MESSAGES:
            producer.flush()
            producer.close()

            print("Test completed.")
            print("Total messages sent:", count)

            exit()

        time.sleep(0.01)

producer.flush()
producer.close()

print("Kafka producer completed.")
print("Total messages sent:", count)
