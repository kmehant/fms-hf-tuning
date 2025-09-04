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
from dataclasses import dataclass

# Local
from .utils import ensure_nested_dataclasses_initialized, parsable_dataclass

is_recover_safetensors_from_dcp_available = True
try:
    # Third Party
    from fms_acceleration_moe.utils import recover_safetensors_from_dcp
except ImportError:
    is_recover_safetensors_from_dcp_available = False


@parsable_dataclass
@dataclass
class AC:
    level: int = None


@dataclass
class ACConfig:

    ac: AC = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)
