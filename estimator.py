# Third Party
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from transformers import AutoModel, AutoModelForCausalLM
import torch
import torch.nn.functional as F

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


def _runtime_estimate(
    estimate_mode,
    func,
    args,
) -> float:
    func(*args)
    runtime_estimator = RuntimeEstimator()
    with runtime_estimator(estimate_mode_type=estimate_mode):
        func(*args)
    return runtime_estimator.total_runtime


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


# https://github.com/pytorch/pytorch/issues/128394
with FakeTensorMode(allow_non_fake_inputs=True):
    args = _init_model_and_args(2)
    roofline_estimate = _runtime_estimate("operator-level-cost-model", train_step, args)
    print(roofline_estimate)
