# Standard
from itertools import product
import os
import subprocess

# Third Party
from tqdm import tqdm

# model_name_or_path_p = ["codellama/CodeLlama-7b-hf"]
num_hidden_layers_p = [1, 2, 32]

# llama 7B 32
num_attention_heads_p = [32]

# defaults for llama 7B
# head_dim = number of heads // hidden size should be less than 256
hidden_size_p = [4096]

intermediate_size_p = [11008]

model_combos = list(
    product(
        num_hidden_layers_p,
        num_attention_heads_p,
        hidden_size_p,
        intermediate_size_p,
    )
)

# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

models_path = "/dev/shm/models"

print("preparing models")
mpaths = []
for mc in tqdm(model_combos, total=len(model_combos)):
    mpath = f"{models_path}/layers_{mc[0]}_attnh_{mc[1]}_hs_{mc[2]}_is_{mc[3]}"
    if not os.path.exists(mpath):
        config = LlamaConfig(
            num_hidden_layers=mc[0],
            num_attention_heads=mc[1],
            hidden_size=mc[2],
            intermediate_size=mc[3],
        )
        model = AutoModelForCausalLM.from_config(config)
        model.save_pretrained(mpath)
    mpaths.append(mpath)
    tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    tok.save_pretrained(mpath)


# mpaths

nproc_per_node_p = [1]

flash_attn_p = ["true", "false"]

max_seq_length_p = [64, 128, 256, 8192]

torch_dtype_p = ["bfloat16"]

per_device_bs_p = [1, 2, 3, 4, 8]

checkpointing_p = ["true", "false"]

combinations = list(
    product(
        nproc_per_node_p,
        mpaths,
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

scanner_logs = "./scanner_11_sep_logs.jsonl"
bench_logs = "./bench_11_sep_logs.log"

if os.path.exists(scanner_logs):
    os.remove(scanner_logs)

if os.path.exists(bench_logs):
    os.remove(bench_logs)

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
    # add fsdp flags
    tc = tc + " " + ff
    logs = "Command: \n" + tc + "\n" + str(combo) + "\n"
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
            output_c = "Command: \n" + tc + "\n" + str(combo) + "\n" + output_c + "\n"
        with open(scanner_logs, "a") as f:
            f.write(output_c)
        os.remove("output.json")

    with open(bench_logs, "a") as f:
        f.write(str(logs))
