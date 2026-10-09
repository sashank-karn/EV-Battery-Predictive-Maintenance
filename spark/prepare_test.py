import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

INPUT = "data/uds_data/CUP1.parquet"
OUTPUT = "/home/sashank_karn/EV_BigData_CA3/dataset/cup1_test.parquet"

selected_ids = [4, 15, 56, 900, 1200, 1208, 1209]

pf = pq.ParquetFile(INPUT)

writer = None
total_rows = 0

for batch in pf.iter_batches(
    columns=["vehicle_id", "time", "value_id", "value"],
    batch_size=500000
):
    table = pa.Table.from_batches([batch])

    mask = pc.is_in(
        table["value_id"],
        value_set=pa.array(selected_ids)
    )

    table = table.filter(mask)

    if table.num_rows == 0:
        continue

    time_ms = pc.cast(
        table["time"],
        pa.timestamp("ms")
    )

    table = table.set_column(
        table.schema.get_field_index("time"),
        "time",
        time_ms
    )

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)
    total_rows += table.num_rows

if writer:
    writer.close()

print(f"Rows written: {total_rows}")
print(f"Output: {OUTPUT}")
