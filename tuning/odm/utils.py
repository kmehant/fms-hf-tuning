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
from collections import defaultdict
import datetime
import os

# Third Party
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# First Party
from transformers import AutoTokenizer

# Local
from tuning.odm.dataset import (
    CurrentLossDiff,
    OnlineDataMixing,
    PrevLossDiff,
    SimpleTextDataset,
    UniformDataMixing,
)
from tuning.odm.evaluators import evaluate


def get_pa_train_dataset(language, n=10):
    def make_prompt(s1, s2):
        return f"""Question: Are the following two sentences paraphrases of each other?

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    Answer with 'Yes' or 'No' only.

    Answer: """

    dataset = load_dataset("xtreme", f"PAWS-X.{language}")["train"]
    prompts = []
    answers = []
    for item in dataset.select(range(n)):
        prompt = make_prompt(item["sentence1"], item["sentence2"])
        prompts.append(prompt)
        if int(item["label"]) == 1:
            answers.append("Yes")
        else:
            answers.append("No")
    return prompts, answers


def get_nli_train_dataset(language, n=10):
    def make_prompt(s1, s2):
        return f"""Question: Are the following two sentences neutral, contradiction or entailment?

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    Answer with 'neutral', 'contradiction' or 'entailment' only.

    Answer: """

    dataset = load_dataset("xtreme", f"XNLI")["validation"]
    dataset = dataset.filter(lambda item: item["language"] == language)

    prompts = []
    answers = []
    for item in dataset.select(range(n)):
        prompt = make_prompt(item["sentence1"], item["sentence2"])
        prompts.append(prompt)
        answers.append(item["gold_label"])
    return prompts, answers


def get_odm_dataset(
    model,
    tokenizer,
    languages,
    tasks,
    method="UniformDataMixing",
    num_samples=10,
    batch_size=4,
    alpha=1.0,
):
    # def plot():
    #     print(f"Plotting to {method}/{now}...")
    #     # Sampling probability plot
    #     for tlp in task_lang:
    #         plt.plot(p[tlp], label=tlp)
    #     plt.xlabel('Batch Index')
    #     plt.ylabel('Sampling Probability')
    #     plt.title(f'Sampling Probabilities Over Batches ({method})')
    #     plt.legend()
    #     plt.savefig(f"{method}/probabilities_{now}.png")
    #     plt.close()

    #     # Loss plot
    #     plt.title("Loss")
    #     plt.plot(losses)
    #     plt.xlabel('Batch Index')
    #     plt.ylabel('Loss')
    #     plt.savefig(f"{method}/losses_{now}.png")
    #     plt.close()

    #     # Learning rate plot
    #     plt.title("Learning Rate")
    #     plt.plot(optimizer_lrs)
    #     plt.xlabel('Batch Index')
    #     plt.ylabel('Learning Rate')
    #     plt.savefig(f"{method}/lr_{now}.png")

    #     # Domain logs plot
    #     domain_logs = np.array(dataset.domain_logs)
    #     one_hot = np.zeros((len(task_lang), len(domain_logs)), dtype=int)
    #     one_hot[domain_logs, np.arange(len(domain_logs))] = 1
    #     cumulative_counts = np.cumsum(one_hot, axis=1)
    #     plt.title("Domain Sampling Counts")
    #     for i, tlp in enumerate(task_lang):
    #         plt.plot(cumulative_counts[i], label=tlp)
    #     plt.xlabel('Batch Index')
    #     plt.ylabel('Domain Sampling Counts')
    #     plt.legend()
    #     plt.savefig(f"{method}/sampling_counts_{now}.png")
    #     plt.close()

    texts = []
    tasks = set(tasks)
    for task in tasks:
        if task == "pa":
            for language in languages:
                prompts, answers = get_pa_train_dataset(language, n=num_samples)
                text = [p + a for p, a in zip(prompts, answers)]
                texts.append(text)
        elif task == "nli":
            for language in languages:
                prompts, answers = get_nli_train_dataset(language, n=num_samples)
                text = [p + a for p, a in zip(prompts, answers)]
                texts.append(text)
        else:
            raise ValueError(f"Unknown task {task}")

    task_lang = [f"{task}_{lang}" for task in tasks for lang in languages]

    all_ds = [SimpleTextDataset(text, tokenizer) for text in texts]
    print("Size of datasets:")
    for tlp, ds in zip(task_lang, all_ds):
        print(f"{tlp}: {len(ds)}")

    dataset = UniformDataMixing(all_ds, model, batch_size=batch_size)
    if method == "PrevLossDiff":
        dataset = PrevLossDiff(all_ds, model, batch_size=batch_size)
    elif method == "OnlineDataMixing":
        dataset = OnlineDataMixing(all_ds, model, batch_size=batch_size, alpha=alpha)
    elif method == "CurrentLossDiff":
        dataset = CurrentLossDiff(all_ds, model, batch_size=batch_size)

    return dataset

    # move this to trainer
    # optimizer = AdamW(model.parameters(), lr=lr)
    # model.train()

    # total_iters = min(iterations, len(dataloader)) if iterations is not None else len(dataloader)

    # # Linear decay from lr to 0.1 * lr
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda step: max(0.1, 1 - 0.9 * (step / total_iters)))
    # optimizer_lrs = []
    # p = defaultdict(list)
    # losses = []

    # os.makedirs(method, exist_ok=True)
    # now_time = datetime.datetime.now()
    # now = now_time.strftime("%Y-%m-%d_%H-%M-%S")

    # for i, batch in enumerate(dataloader):
    #     input_ids = batch['input_ids'].to(model.device)
    #     attention_mask = batch['attention_mask'].to(model.device)
    #     labels = batch['labels'].to(model.device)
    #     text = batch['text']
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     loss = outputs.loss
    #     losses.append(loss.item())
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     optimizer_lrs.append(optimizer.param_groups[0]['lr'])
    #     scheduler.step()
    #     if i % every == 0:
    #         print(f"------{method}------")
    #         print("Iteration: {}/{}\tLoss: {}\nProbabilities: {}\n".format(i, total_iters, loss,
    #                                                                    dataset.rl_agent._probabilities))
    #         if i > 0 and intermediate_eval:
    #             evaluate(model, tokenizer, tasks, languages, num_samples=num_inter_samples)
    #         plot()

    #         # Time taken and ETA
    #         elapsed = (datetime.datetime.now() - now_time).total_seconds()
    #         eta_seconds = elapsed / (i + 1) * (total_iters - i - 1)
    #         eta_minutes = eta_seconds / 60
    #         print(f"\nTime taken: {elapsed/60:.1f} min(s), ETA: {eta_minutes:.1f} min(s)")
    #         print('---------------------------')
    #         model.train()
    #     for tlp, prob in zip(task_lang, dataset.rl_agent._probabilities):
    #         p[tlp].append(prob)
    #     dataset.training_step_callback(batch, loss.item())
    #     if i == total_iters:
    #         break

    # plot()
