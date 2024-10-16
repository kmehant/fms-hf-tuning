# Copyright The IBM Tuning Team
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
from dataclasses import dataclass, field
from typing import Optional, Union

# Third Party
import torch
import transformers

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_OPTIMIZER = "adamw_torch"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use Flash attention v2 from transformers, default is True"},
    )
    torch_dtype: Optional[Union[torch.dtype, str]] = torch.bfloat16


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data in JSONL format."}
    )
    response_template: str = field(
        default=None,
        metadata={"help": "Response template, separator to train on completions only"},
    )
    dataset_text_field: str = field(
        default=None, metadata={"help": "Training dataset text field"}
    )
    validation_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data in JSONL format."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default=DEFAULT_OPTIMIZER)
    model_max_length: int = field(
        default=DEFAULT_CONTEXT_LENGTH,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded \
            (and possibly truncated)."
        },
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Packing to be enabled in SFT Trainer, default is False"},
    )
