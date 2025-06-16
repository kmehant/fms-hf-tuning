# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
import os

# Third Party
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

# Local
from tuning.odm.mixers import RLAgent


from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.inputs = [tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length") for text in texts]
        self.texts = [text for text, ip in zip(self.texts, self.inputs) if ip['input_ids'][0][-1] == tokenizer.pad_token_id]
        self.inputs = [ip for ip in self.inputs if ip['input_ids'][0][-1] == tokenizer.pad_token_id]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        input_ids = input['input_ids'].squeeze(0)
        attention_mask = input['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # <== mask pad tokens

        ###### Learn on last token only
        labels = torch.full_like(input_ids, -100)  # Initialize all labels as -100

        # Find the last non-padding token
        last_token_index = attention_mask.nonzero(as_tuple=True)[0][-1]
        labels[last_token_index] = input_ids[last_token_index]  # Only train on last token
        ####################

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': self.texts[idx]
        }



class OnlineDataset(IterableDataset):
    def __init__(self, datasets, model, batch_size, global_batch_size=None, **kwargs):
        super().__init__()
        self.datasets = datasets
        self.model = model
        self.batch_size = batch_size
        self.global_batch_size = (
            global_batch_size if global_batch_size is not None else batch_size
        )
        self.dataset_iters = [iter(dataset) for dataset in datasets]
        self.current_domain = None
        self.ready_for_iteration = (
            True  # Flag to indicate if the dataset is ready for iteration
        )
        self.iteration = 0
        self.domain_logs = []

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        length = len(self)
        for i in range(length // self.batch_size):
            if not self.ready_for_iteration:
                # TODO: Training step callback not needed after each iteration, should be able to handle gradient accumulation and multi-gpu logic
                #  eg: iter called k = (global bs / bs) times then training_step_callback called once with list of training signals of size k
                #  Note: __iter__ should call __get_next_domain__ k times and training_step_callback should call __take_training_signals__ k times
                #  __get_next_domain__ and __take_training_signals__ should not worry about any of the gradient accumulation or multi gpu logic
                raise RuntimeError(
                    "Training step callback not called before yielding next batch. Ensure to call training_step_callback after each training step."
                )
            self.current_domain = self.__get_next_domain__()
            self.domain_logs.append(self.current_domain)
            for j in range(self.batch_size):
                if j == self.batch_size - 1:
                    self.ready_for_iteration = False
                try:
                    sample = next(self.dataset_iters[self.current_domain])
                    yield sample
                except StopIteration:
                    # If the dataset is exhausted, reset the iterator
                    self.dataset_iters[self.current_domain] = iter(
                        self.datasets[self.current_domain]
                    )
                    sample = next(self.dataset_iters[self.current_domain])
                    yield sample

    def training_step_callback(self, batch, loss):
        """
        This method should be called after each training step to process the batch and loss.
        :param batch: The current batch of data (from data loader).
        :param loss: The loss or reward signal to process.
        :return:
        """
        if self.ready_for_iteration:
            raise RuntimeError(
                "Ensure a new batch is yielded before calling training_step_callback."
            )
        self.ready_for_iteration = True
        assert self.current_domain is not None
        batch["metadata"] = {"domain_index": self.current_domain}
        self.__take_training_signals__(batch, loss)
        self.iteration += 1

    def __get_next_domain__(self):
        """
        This method should be implemented by subclasses to determine the next domain index.
        :return:
            int: The index of the next domain to sample from.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __take_training_signals__(self, batch, loss):
        """
        This method should be implemented by subclasses to handle training signals.
        :param batch: The current batch of data (with metadata).
        :param loss: The loss or reward signal to process.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class UniformDataMixing(OnlineDataset):
    def __init__(self, datasets, model, batch_size, global_batch_size=None, **kwargs):
        super().__init__(datasets, model, batch_size, global_batch_size, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)])

    def __get_next_domain__(self):
        return self.rl_agent.sample()

    def __take_training_signals__(self, batch, loss):
        pass


class StaticDataMixing(OnlineDataset):
    def __init__(
        self, datasets, model, batch_size, weights, global_batch_size=None, **kwargs
    ):
        super().__init__(datasets, model, batch_size, global_batch_size, **kwargs)
        self.num_domains = len(self.datasets)
        assert len(weights) == self.num_domains
        self.rl_agent = RLAgent(weights)

    def __get_next_domain__(self):
        return self.rl_agent.sample()

    def __take_training_signals__(self, batch, loss):
        pass


class OnlineDataMixing(OnlineDataset):
    def __init__(
        self, datasets, model, batch_size, alpha=1.0, global_batch_size=None, **kwargs
    ):
        super().__init__(datasets, model, batch_size, global_batch_size, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)], alpha=alpha)

    def __get_next_domain__(self):
        if self.iteration < self.num_domains:
            return self.iteration
        return self.rl_agent.sample()

    def __take_training_signals__(self, batch, loss):
        domain_index = batch["metadata"]["domain_index"]
        ### Prevents overflow while calculating the exponential of the reward
        reward = np.clip(loss, 0, 5)
        if self.iteration >= self.num_domains:
            self.rl_agent.update(domain_index, reward=reward)


class PrevLossDiff(OnlineDataset):
    def __init__(self, datasets, model, batch_size, global_batch_size=None, **kwargs):
        super().__init__(datasets, model, batch_size, global_batch_size, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)])
        self.prev_loss = [None for _ in range(self.num_domains)]

    def __get_next_domain__(self):
        if self.iteration < self.num_domains:
            return self.iteration
        return self.rl_agent.sample()

    def __take_training_signals__(self, batch, loss):
        domain_index = batch["metadata"]["domain_index"]
        ### Prevents overflow while calculating the exponential of the reward
        if self.iteration >= self.num_domains:
            loss = np.clip(loss, 0, 5)
            if self.prev_loss[domain_index] is not None:
                diff = self.prev_loss[domain_index] - loss
                self.rl_agent.update(domain_index, reward=diff)
            self.prev_loss[domain_index] = loss


class CurrentLossDiff(OnlineDataset):
    def __init__(self, datasets, model, batch_size, global_batch_size=None, **kwargs):
        super().__init__(datasets, model, batch_size, global_batch_size, **kwargs)
        self.num_domains = len(self.datasets)
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)])
        self.rewards = [[] for _ in range(self.num_domains)]
        self.iters = [[] for _ in range(self.num_domains)]

    def __get_next_domain__(self):
        if self.iteration < self.num_domains:
            return self.iteration
        return self.rl_agent.sample()

    def __take_training_signals__(self, batch, loss):
        domain_index = batch["metadata"]["domain_index"]
        if self.iteration >= self.num_domains:
            self.model.eval()
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            new_loss = outputs.loss.item()
            diff = loss - new_loss
            ### Prevents overflow while calculating the exponential of the reward
            diff = np.clip(diff, -5, 5)
            print(
                f"Iteration: {self.iteration}, Domain: {domain_index}, New Loss: {new_loss}, Previous Loss: {loss}, Difference: {diff}"
            )
            # log the reward and iteration for the domain
            self.rewards[domain_index].append(diff)
            self.iters[domain_index].append(self.iteration)
            self.rl_agent.update(domain_index, reward=diff)
            self.model.train()

        if self.iteration % 300 == 0:
            print("Plotting rewards over iterations for each domain...")
            plt.figure(figsize=(10, 5))
            for i in range(self.num_domains):
                plt.plot(self.iters[i], self.rewards[i], label=f"Domain {i}")
            plt.xlabel("Iteration")
            plt.ylabel("Reward")
            plt.title(
                "Rewards Over Iterations for Each Domain (Current Loss Difference)"
            )
            plt.legend()
            os.makedirs("CurrentLossDiff", exist_ok=True)
            plt.savefig(f"CurrentLossDiff/rewards_over_iterations_{self.iteration}.png")


if __name__ == "__main__":
    # Third Party
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # First Party
    from transformers import AutoTokenizer

    # Local
    from tuning.odm.dataset import SimpleTextDataset

    model_name = "meta-llama/Llama-3.2-3B"
    texts1 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] * 5
    texts2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * 5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None  # Placeholder for model, not used in this example
    tokenizer.pad_token = tokenizer.eos_token
    BS = 4
    dataset = PrevLossDiff(
        [SimpleTextDataset(texts1, tokenizer), SimpleTextDataset(texts2, tokenizer)],
        model,
        batch_size=BS,
    )
    dataloader = DataLoader(dataset, batch_size=BS)

    p1, p2 = [], []

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        text = batch["text"]
        if i < 30:
            reward = 5 if text[0] in texts1 else 0
        else:
            reward = 5 if text[0] in texts2 else 0
        print(
            "Sample: {}\tReward: {}\tProbabilities{}".format(
                text[0], reward, dataset.rl_agent._probabilities
            )
        )
        p1.append(dataset.rl_agent._probabilities[0])
        p2.append(dataset.rl_agent._probabilities[1])
        dataset.training_step_callback(batch, reward)

    plt.plot(p1, label="Domain 1")
    plt.plot(p2, label="Domain 2")
    plt.xlabel("Batch Index")
    plt.ylabel("Sampling Probability")
    plt.title("Sampling Probabilities Over Batches")
    plt.legend()
    plt.show()
