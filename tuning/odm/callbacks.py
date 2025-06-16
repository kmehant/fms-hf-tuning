# First Party
from transformers import TrainerCallback, TrainerState


class ODMCallback(TrainerCallback):
    def __init__(self, eval_iter=10):
        self.eval_iter = 10

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        state.dataset.training_step_callback(self.state.inputs, self.state.loss.item())
        with open("./rl_proj.jsonl", "w") as f:
            f.write(state.dataset.rl_agent._probabilities + "\n")
