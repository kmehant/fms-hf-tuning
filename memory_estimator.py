# Third Party
from torch import optim
from torch.distributed._tools.memory_tracker import MemoryTracker
from transformers import AutoModelForCausalLM
import torch

modelclass = AutoModelForCausalLM


def train_step(model, optimizer, inp):
    print(inp)
    out = model(**inp)
    print(out)
    loss = out.loss
    # loss = F.cross_entropy(out.logits, inp["labels"])
    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _init_model_and_args(
    bsz,
    model_type="/dev/shm/downloaded_models/TinyLlama-1.1B-Chat-v1.0",
):
    # dev = torch.cuda.current_device()
    # print(dev)
    # print(torch.device(dev))
    device = "cuda"
    with torch.device("meta"):
        model = modelclass.from_pretrained(model_type, torch_dtype=torch.bfloat16)
    model.to_empty(device=device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    print(model.config.vocab_size)
    rand_tensor = torch.randint(0, model.config.vocab_size, (bsz, 128), device=device)
    rand_tensor_labels = torch.randint(
        0, model.config.vocab_size, (bsz, 128), device=device
    )
    print(rand_tensor.device)
    inp = {"input_ids": rand_tensor, "labels": rand_tensor_labels}
    return (model, optimizer, inp)


mt = MemoryTracker()

dev = torch.device("cuda")
device = "cuda"
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.reset_peak_memory_stats()
args = _init_model_and_args(1)
mt.start_monitor(args[0])
try:
    train_step(*args)
except Exception as e:
    print("failed")
    print(e)
    pass
mt.stop()
print(mt.summary())
mt.show_traces()
