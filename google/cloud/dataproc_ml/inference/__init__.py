# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A subpackage for model inference handlers."""

from .gen_ai_model_handler import GenAiModelHandler
from .pytorch_model_handler import PyTorchModelHandler
from .tensorflow_model_handler import TensorFlowModelHandler

__all__ = ("GenAiModelHandler", "PyTorchModelHandler", "TensorFlowModelHandler")
