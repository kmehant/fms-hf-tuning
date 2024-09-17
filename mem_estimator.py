# Third Party
from torch import optim
from torch.distributed._tools.mem_tracker import MemTracker
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
    model_type="/dev/shm/py",
):
    # dev = torch.cuda.current_device()
    # print(dev)
    # print(torch.device(dev))
    device = "cuda"
    with torch.device("meta"):
        model = modelclass.from_pretrained(model_type)
    model.to_empty(device=device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    print(model.config.vocab_size)
    rand_tensor = torch.randint(0, model.config.vocab_size, (bsz, 2048), device=device)
    rand_tensor_labels = torch.randint(
        0, model.config.vocab_size, (bsz, 2048), device=device
    )
    print(rand_tensor.device)
    inp = {"input_ids": rand_tensor, "labels": rand_tensor_labels}
    return (model, optimizer, inp)


dev = torch.device("cuda")
device = "cuda"
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.reset_peak_memory_stats()
args = _init_model_and_args(2)
mem_tracker = MemTracker()
mem_tracker.track_external(args[0], args[1])
with mem_tracker as mt:
    train_step(args)

tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
cuda_max = torch.cuda.max_memory_allocated(dev)
accuracy = tracker_max / cuda_max
print(cuda_max)
print(tracker_max)
print(accuracy)
print(mt.get_tracker_snapshot())
