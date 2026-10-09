import pyarrow.parquet as pq
import pyarrow as pa
from datetime import timedelta
import os

INPUT = "electric-vehicle-uds-dataset/data/uds_data/CUP1.parquet"
OUTPUT = "dataset/cup1_fixed.parquet"

os.makedirs("dataset", exist_ok=True)

pf = pq.ParquetFile(INPUT)

writer = None
total_rows = 0
corrected_rows = 0

LIMIT_NS = 1704067200000000000

for batch in pf.iter_batches(batch_size=100000):
    table = pa.Table.from_batches([batch])

    time_array = table["time"]
    raw_ns = time_array.cast(pa.int64())

    corrected = []

    for value in raw_ns.to_pylist():
        if value > LIMIT_NS:
            from datetime import datetime, timezone

            dt = datetime.fromtimestamp(
                value / 1_000_000_000,
                tz=timezone.utc
            )

            dt = dt.replace(year=dt.year - 64)
            corrected.append(dt.replace(tzinfo=None))

            corrected_rows += 1
        else:
            corrected.append(
                datetime.fromtimestamp(
                    value / 1_000_000_000
                )
            )

    corrected_time = pa.array(
        corrected,
        type=pa.timestamp("us")
    )

    time_index = table.schema.get_field_index("time")

    table = table.set_column(
        time_index,
        "time",
        corrected_time
    )

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    total_rows += table.num_rows

    print(
        f"Processed: {total_rows:,} | "
        f"Corrected: {corrected_rows:,}"
    )

if writer:
    writer.close()

print()
print("Timestamp correction completed.")
print("Total rows:", total_rows)
print("Corrected timestamps:", corrected_rows)
print("Output:", OUTPUT)
