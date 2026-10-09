from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "battery-telemetry",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="battery-test-consumer",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Waiting for battery telemetry...")

for message in consumer:
    print(message.value)
