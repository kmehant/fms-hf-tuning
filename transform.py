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


def process_file(file, directory, dest, pbar=None):
    dest_path = os.path.join(dest, file[len(directory) + 1 :])
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        if pbar is not None:
            pbar.update(1)
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
    if pbar is not None:
        pbar.update(1)
    return f"Written at: {dest_path}"


directory = "/data/data/spanish-gov-tokenized/llama3/arrow/lang=es"
arrow_files = find_arrow_files(directory)

dest = "/data/data/fixed-spanish-gov-tokenized"


# Parallel processing using ProcessPoolExecutor with 2 CPUs
with tqdm(total=len(arrow_files)) as pbar:
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_file, file, directory, dest, pbar)
            for file in arrow_files
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
