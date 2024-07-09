# Standard
import concurrent.futures
import glob
import os

# Third Party
from tqdm import tqdm
import pyarrow as pa


def find_arrow_files(directory):
    pattern = os.path.join(directory, "**", "*.arrow")
    arrow_files = glob.glob(pattern, recursive=True)
    return arrow_files


def process_file(file, directory, dest):
    dest_path = os.path.join(dest, file[len(directory) + 1 :])
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        return f"Skipped {dest_path} (already exists)"
    with open(file, "rb") as f:
        with pa.OSFile(dest_path, "wb") as wf:
            with pa.ipc.new_stream(
                wf, pa.schema([("tokens", pa.list_(pa.int64()))])
            ) as writer:
                reader = pa.ipc.open_file(f)
                batches = [
                    reader.get_batch(i) for i in range(reader.num_record_batches)
                ]
                for batch_idx, record_batch in enumerate(batches):
                    record_batch = pa.RecordBatch.from_pydict(
                        {"tokens": [record_batch.to_pydict()["tokens"]]}
                    )
                    writer.write_batch(record_batch)
    return f"Written at: {dest_path}"


directory = "/data/data/spanish-gov-tokenized/llama3/arrow/lang=es"
arrow_files = find_arrow_files(directory)

dest = "/data/data/fixed-spanish-gov-tokenized"

print(arrow_files)

# Parallel processing using ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(
                process_file,
                arrow_files,
                [directory] * len(arrow_files),
                [dest] * len(arrow_files),
            ),
            total=len(arrow_files),
        )
    )

for result in results:
    print(result)
