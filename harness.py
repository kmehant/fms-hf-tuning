# Standard
from itertools import product
import os
import subprocess

# Third Party
from tqdm import tqdm

nproc_per_node_p = [1, 2, 3, 4]
model_name_or_path_p = ["codellama/CodeLlama-7b-hf"]
flash_attn_p = ["true", "false"]
max_seq_length_p = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
torch_dtype_p = ["bfloat16"]
per_device_bs_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
checkpointing_p = ["true", "false"]

combinations = list(
    product(
        nproc_per_node_p,
        model_name_or_path_p,
        flash_attn_p,
        max_seq_length_p,
        torch_dtype_p,
        per_device_bs_p,
        checkpointing_p,
    )
)

data_yaml = """
train_datasets:
  - path: /workspace/fms-hf-tuning/tests/data/twitter_complaints_input_output.jsonl
    prob: 1
seq_length: {max_seq_len}
add_bos_token: false
add_eos_token: false
"""

fsdp_flags = ' --fsdp "hybrid_shard auto_wrap" --fsdp_config /workspace/fms-hf-tuning/config.json '

torchrun_cmd = "torchrun --nnodes=1 --node_rank=0 --nproc_per_node={nproc_per_node} --rdzv_id=101 --rdzv_endpoint=0.0.0.0:8888 ./tuning/sft_trainer.py --model_name_or_path {model_name_or_path} --output_dir ./train_output --max_steps 5 --learning_rate 2e-5 --torch_dtype {torch_dtype} --logging_strategy steps --logging_steps 1 --save_strategy no --per_device_train_batch_size {per_device_bs} --max_seq_length {max_seq_length} --use_flash_attn {flash_attn} --packing true --include_tokens_per_second true --data_config_path ./data.yaml --dataset_text_field input --gradient_checkpointing {checkpointing}"

scanner_logs = "./10_sep_logs.jsonl"
bench_logs = "./bench_10_sep_logs.log"

for combo in tqdm(combinations, total=len(combinations)):
    dy = data_yaml.format(max_seq_len=combo[3])
    if os.path.exists("./data.yaml"):
        os.remove("./data.yaml")
    with open("./data.yaml", "w") as f:
        f.write(dy)
    ff = "" if combo[0] == 1 else fsdp_flags
    tc = torchrun_cmd.format(
        nproc_per_node=combo[0],
        model_name_or_path=combo[1],
        torch_dtype=combo[4],
        per_device_bs=combo[5],
        max_seq_length=combo[3],
        flash_attn=combo[2],
        checkpointing=combo[6],
    )
    logs = str(combo) + "\n"
    try:
        result = subprocess.run(
            tc, shell=True, check=True, text=True, capture_output=True
        )
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
        logs = (
            logs + "\nstdout\n" + str(result.stdout) + "\nstderr\n" + str(result.stderr)
        )
    except subprocess.CalledProcessError as e:
        logs = logs + f"failed running the command : {e}"
    output_c = ""
    if os.path.exists("./output.json"):
        with open("output.json", "r") as f:
            output_c = "\n".join(f.readlines())
            output_c += "\n"
        with open(scanner_logs, "a") as f:
            f.write(output_c)
        os.remove("output.json")
    with open(bench_logs, "a") as f:
        f.write(str(logs))
