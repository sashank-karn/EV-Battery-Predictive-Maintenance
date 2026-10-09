
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path.home() / "EV_BigData_CA3"
SOURCE = ROOT / "electric-vehicle-uds-dataset" / "data" / "uds_data"
OUTPUT = ROOT / "dataset" / "normalized_sources"

OUTPUT.mkdir(parents=True, exist_ok=True)

sources = {
    "CUP1": ROOT / "dataset" / "cup1_fixed.parquet",
    "CUP2": SOURCE / "CUP2.parquet",
    "CUP3": SOURCE / "CUP3.parquet",
    "CUP4": SOURCE / "CUP4.parquet",
    "CUP5": SOURCE / "CUP5.parquet",
    "ID1": SOURCE / "ID1.parquet",
    "ID2": SOURCE / "ID2.parquet",
}

for vehicle, source_path in sources.items():
    output_path = OUTPUT / f"{vehicle}.parquet"
    print(f"Processing {vehicle}: {source_path.name}", flush=True)

    parquet = pq.ParquetFile(source_path)
    writer = None
    rows = 0

    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            table = pa.Table.from_batches([batch])
            time_index = table.schema.get_field_index("time")

            time_array = table.column(time_index).cast(
                pa.timestamp("us")
            )

            table = table.set_column(
                time_index,
                pa.field("time", pa.timestamp("us")),
                time_array
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)
            rows += table.num_rows

        print(
            f"Completed {vehicle}: {rows:,} rows -> {output_path}",
            flush=True
        )
    finally:
        if writer is not None:
            writer.close()

print("All seven source files normalized.")
