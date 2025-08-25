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

"""A module for handling PyTorch model inference on Spark DataFrames."""

from io import BytesIO
from typing import Type, Optional, Callable

import numpy as np
import pandas as pd
import torch

from google.cloud.dataproc.ml.inference.base_model_handler import BaseModelHandler
from google.cloud.dataproc.ml.inference.base_model_handler import Model
from google.cloud.dataproc.ml._utils._gcs_utils import (
    download_gcs_blob_to_buffer,
)


class PyTorchModel(Model):
    """A concrete implementation of the Model interface for PyTorch models."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str],
        model_class: Optional[Type[torch.nn.Module]] = None,
        model_args: Optional[tuple] = None,
        model_kwargs: Optional[dict] = None,
    ):
        """Initializes the PyTorchModel.

        Args:
            model_path: The GCS path to the saved PyTorch model (e.g.,
                "gs://my-bucket/model.pt").
            device: (Optional) The device to load the model on ("cpu" or
                "cuda").
            model_class: (Optional) The Python class of the PyTorch model when
                we need to load from statedict
            model_args: (Optional) A tuple of positional arguments to pass to
                `model_class` constructor.
            model_kwargs: (Optional) A dictionary of keyword arguments to pass
                to `model_class` constructor.
        """
        self._model_path = model_path
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._model_class = model_class
        self._model_args = model_args if model_args is not None else ()
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._underlying_model = self._load_model_from_gcs()
        # Set model to evaluation mode
        self._underlying_model.eval()

    def _state_dict_model_load(self, model_weights: BytesIO):
        """Loads a model's state_dict after performing upfront validations."""

        if not callable(self._model_class):
            raise TypeError(
                "model_class must be a PyTorch model class, but got "
                f"{type(self._model_class)}."
            )

        model_instance = self._model_class(
            *self._model_args, **self._model_kwargs
        )

        if not isinstance(model_instance, torch.nn.Module):
            raise TypeError(
                f"The provided callable '{self._model_class_name()}' did not "
                "return a torch.nn.Module instance. Instead, it returned type: "
                f"{type(model_instance)}."
            )

        try:
            state_dict = torch.load(
                model_weights, map_location=self._device, weights_only=True
            )

            if not isinstance(state_dict, dict):
                raise TypeError(
                    "Expected a state_dict (dict) for architecture "
                    f"'{self._model_class_name()}', but the loaded file was of "
                    f"type {type(state_dict)}. Full model load is only  "
                    "attempted when a model architecture is NOT provided."
                )

            model_instance.load_state_dict(state_dict)
            return model_instance

        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load state_dict from {self._model_path} into the "
                f"provided '{self._model_class_name()}' architecture: {e}"
            )

    def _model_class_name(self):
        model_class_name = getattr(self._model_class, "__name__", "unknown")
        return model_class_name

    def _full_model_load(self, model: BytesIO):
        """Loads a full PyTorch model object from a file-like object."""

        try:
            model_instance = torch.load(
                model, map_location=self._device, weights_only=False
            )
        except Exception as e:
            # This catches errors during the load process (e.g., pickle
            # errors, corrupted file).
            raise RuntimeError(
                f"Failed to load the PyTorch model object {self._model_path}. "
                f"The file may be corrupted or not a valid PyTorch model. "
                f"Original error: {e}"
            )

        if not isinstance(model_instance, torch.nn.Module):
            raise TypeError(
                f"The file at {self._model_path} was loaded successfully, but "
                "it is not a torch.nn.Module instance. Instead, it is of "
                f"type: {type(model_instance)}."
            )
        return model_instance

    def _load_model_from_gcs(self):
        """Loads the PyTorch model from GCS"""
        model_data_buffer = download_gcs_blob_to_buffer(self._model_path)
        if self._model_class:
            return self._state_dict_model_load(model_data_buffer)
        else:
            return self._full_model_load(model_data_buffer)

    def call(self, batch: pd.Series) -> pd.Series:
        """
        Processes a batch of inputs for the PyTorch model.
        """
        try:
            np_array = np.stack(batch.values)
            batch_tensor = torch.from_numpy(np_array).to(
                dtype=torch.float32, device=self._device
            )

            if batch_tensor.dim() == 1:
                batch_tensor = batch_tensor.unsqueeze(1)
        except Exception as e:
            raise ValueError(
                f"Error converting batch to PyTorch tensors: {e}. Ensure "
                "preprocessed data in the Series is consistently shaped and "
                "of a uniform type."
            )

        with torch.inference_mode():
            predictions = self._underlying_model(batch_tensor)

        return pd.Series(predictions.cpu().tolist(), index=batch.index)


class PyTorchModelHandler(BaseModelHandler):
    """A handler for running inference with PyTorch models on Spark DataFrames.

    This handler supports two primary modes of operation:
    1.  Loading a full, saved PyTorch model object.
    2.  Loading a model from a `state_dict` (weights file), which requires
        providing the model's class architecture separately.

    Required Configuration:
        - `.model_path(str)`: The GCS path to the model file (`.pt`, `.pth`).
        - `.input_col(str)`: The column in the DataFrame to use as input.
        - `.set_model_architecture(...)`: This is **required** when loading from
          a state dict, but should **not** be used when loading a full model.

    Optional Configuration:
        - `.output_col(str)`: The name for the prediction output column
          (defaults to "predictions").
        - `.device(str)`: The device to run on (e.g., "cpu", "cuda"). Defaults
          to "cuda" if available.
        - `.pre_processor(callable)`: A function to preprocess input data.
        - `.set_return_type(DataType)`: The Spark `DataType` of the output.

    Example:
        Load a full model saved in GCS:

        >>> result_df = (
        ...     PyTorchModelHandler()
        ...     .model_path("gs://test-bucket/test-model.pt")
        ...     .input_col("input_col")
        ...     .output_col("prediction")
        ...     .transform(input_df)
        ... )

        Load a model from a state dictionary saved in GCS:

        >>> from torchvision import models
        >>> model_class = models.resnet18
        >>> model_kwargs = {"weights": None}
        >>> result_df = (
        ...     PyTorchModelHandler()
        ...     .model_path("gs://my-bucket/resnet18_statedict.pt")
        ...     .set_model_architecture(model_class, **model_kwargs)
        ...     .input_col("features")
        ...     .output_col("predictions")
        ...     .transform(input_df)
        ... )
    """

    def __init__(self):
        super().__init__()
        self._model_path = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_class: Optional[Type[torch.nn.Module]] = None
        self._model_args: Optional[tuple] = None
        self._model_kwargs: Optional[dict] = None

    def model_path(self, path: str) -> "PyTorchModelHandler":
        """Sets the GCS path to the saved PyTorch model.

        Args:
            path: The GCS path to the model file (e.g., "gs://bucket/model.pt").

        Returns:
            The handler instance for method chaining.

        Raises:
            ValueError: If the path does not start with "gs://".
        """
        if not path.startswith("gs://"):
            raise ValueError("Model path must start with 'gs://'")
        self._model_path = path
        return self

    def device(self, device: str) -> "PyTorchModelHandler":
        """Sets the device to load the PyTorch model on.

        Args:
            device: The device to use, either "cpu" or "cuda".

        Returns:
            The handler instance for method chaining.

        Raises:
            ValueError: If the device is not "cpu" or "cuda".
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        self._device = device
        return self

    def set_model_architecture(
        self, model_callable: Callable[..., torch.nn.Module], *args, **kwargs
    ) -> "PyTorchModelHandler":
        """Sets the PyTorch model's architecture using a class or a factory
        function.

        This is required if you are loading a model saved as a state_dict.

        Args:
            model_callable: The model's class (e.g., `MyImageClassifier`) or a
                factory function (e.g., `torchvision.models.resnet18`) that
                returns a model instance.
            *args: Positional arguments for the model's constructor or factory
                function.
            **kwargs: Keyword arguments for the model's constructor or factory
                function.

        Returns:
            The handler instance for method chaining.
        """
        self._model_class = model_callable
        self._model_args = args
        self._model_kwargs = kwargs
        return self

    def _load_model(self) -> Model:
        """Loads the PyTorchModel instance on each Spark executor."""
        if not self._model_path:
            raise ValueError("Model path must be set using .model_path().")
        return PyTorchModel(
            model_path=self._model_path,
            device=self._device,
            model_class=self._model_class,
            model_args=self._model_args,
            model_kwargs=self._model_kwargs,
        )
