# First Party
from transformers import TrainerCallback, TrainerState


class ODMCallback(TrainerCallback):
    def __init__(self, eval_iter=10):
        self.eval_iter = 10

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        print("in callback", state.dataset)
        state.dataset.update_mixer(state.inputs, state.loss.item(), None)
        with open("./rl_proj.jsonl", "w+") as f:
            f.write(str(state.dataset.rl_agent._probabilities) + "\n")
