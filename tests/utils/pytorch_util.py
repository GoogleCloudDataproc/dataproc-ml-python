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

import io

import torch


import torchvision.transforms as transforms
from PIL import Image


def save_pytorch_model_state_dict(model: torch.nn.Module) -> bytes:
    """Saves a PyTorch model's state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.getvalue()


def save_pytorch_model_full_object(model: torch.nn.Module) -> bytes:
    """Saves a full PyTorch model object to bytes."""
    buffer = io.BytesIO()
    # Security note: This saves the entire model object using pickle.
    # It's less portable and has security implications if loading from untrusted sources.
    torch.save(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def preprocess_real_image_data(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses real image bytes for a model like ResNet.
    Input: Raw image bytes (e.g., from a JPG file).
    Output: A PyTorch tensor suitable for a pre-trained ResNet model.
    """
    if not isinstance(image_bytes, bytes):
        raise TypeError(f"Expected image bytes, got {type(image_bytes)}")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    _resnet_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = _resnet_transform(image)
    return input_tensor
