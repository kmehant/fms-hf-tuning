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


# Third Party
from datasets import load_dataset

# Local
from tuning.odm.dataset import OnlineDataMixing, SimpleTextDataset, UniformDataMixing


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
    alpha=1.0,
):

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

    dataset = UniformDataMixing(all_ds, model)
    if method == "OnlineDataMixing":
        dataset = OnlineDataMixing(all_ds, model, alpha=alpha)

    return dataset
