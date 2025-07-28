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

from typing import Type, Optional, Callable

import pandas as pd
import torch
from google.cloud.exceptions import NotFound

from google.cloud import storage
from google.cloud.dataproc.ml.inference.base_model_handler import BaseModelHandler
from google.cloud.dataproc.ml.inference.base_model_handler import Model
from google.cloud.dataproc.ml.utils.gcs_utils import validate_and_parse_gcs_path


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
            model_path: The GCS path to the saved PyTorch model (e.g., "gs://my-bucket/model.pt").
            device: (Optional) The device to load the model on ("cpu" or "cuda").
            model_class: (Optional) The Python class of the PyTorch model when we need to load from statedict
            model_args: (Optional) A tuple of positional arguments to pass to `model_class` constructor.
            model_kwargs: (Optional) A dictionary of keyword arguments to pass to `model_class` constructor.
        """
        self._model_path = model_path
        self._device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model_class = model_class
        self._model_args = model_args if model_args is not None else ()
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._underlying_model = self._load_model_from_gcs()
        # Set model to evaluation mode
        self._underlying_model.eval()

    def _state_dict_model_load(self, model_weights):
        """Loads a model's state_dict after performing upfront validations."""

        if not callable(self._model_class):
            raise TypeError(
                f"model_class must be a PyTorch model class, but got {type(self._model_class)}."
            )

        model_instance = self._model_class(
            *self._model_args, **self._model_kwargs
        )

        if not isinstance(model_instance, torch.nn.Module):
            model_class_name = getattr(self._model_class, "__name__", "unknown")
            raise TypeError(
                f"The provided callable '{model_class_name}' did not return a "
                f"torch.nn.Module instance. Instead, it returned type: {type(model_instance)}."
            )

        try:
            state_dict = torch.load(
                model_weights, map_location=self._device, weights_only=True
            )

            if not isinstance(state_dict, dict):
                model_class_name = getattr(
                    self._model_class, "__name__", "unknown"
                )
                raise TypeError(
                    f"Expected a state_dict (dict) for architecture '{model_class_name}', "
                    f"but the loaded file was of type {type(state_dict)}. "
                    "Full model load is only attempted when a model architecture is NOT provided."
                )

            model_instance.load_state_dict(state_dict)
            return model_instance

        except RuntimeError as e:
            model_class_name = getattr(self._model_class, "__name__", "unknown")
            raise RuntimeError(
                f"Failed to load state_dict from {self._model_path} into the "
                f"provided '{model_class_name}' architecture: {e}"
            )

    def _full_model_load(self, model):
        """Loads a full PyTorch model object from a file-like object."""

        try:
            model_instance = torch.load(
                model, map_location=self._device, weights_only=False
            )
        except Exception as e:
            # This catches errors during the load process (e.g., pickle errors, corrupted file).
            raise RuntimeError(
                f"Failed to load the PyTorch model object from {self._model_path}. "
                f"The file may be corrupted or not a valid PyTorch model. Original error: {e}"
            )

        if not isinstance(model_instance, torch.nn.Module):
            raise TypeError(
                f"The file at {self._model_path} was loaded successfully, but it is not a "
                f"torch.nn.Module instance. Instead, it is of type: {type(model_instance)}."
            )
        return model_instance

    def _download_gcs_blob_to_buffer(self, bucket_name, blob_name):
        """Downloads a GCS blob into an in-memory BytesIO buffer."""
        model_data_buffer = io.BytesIO()
        try:
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(blob_name)
            blob.download_to_file(model_data_buffer)
            model_data_buffer.seek(0)
            return model_data_buffer
        except NotFound:
            raise FileNotFoundError(
                f"Model file not found at GCS path: {self._model_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from GCS at {self._model_path}. "
                f"Check permissions/path. Original error: {e}"
            )

    def _load_model_from_gcs(self):
        """Loads the PyTorch model from GCS with verbose logging for debugging."""

        bucket_name, blob_name = validate_and_parse_gcs_path(self._model_path)

        model_data_buffer = self._download_gcs_blob_to_buffer(
            bucket_name, blob_name
        )

        if self._model_class:
            return self._state_dict_model_load(model_data_buffer)
        else:
            return self._full_model_load(model_data_buffer)

    def call(self, batch: pd.Series) -> pd.Series:
        """
        Processes a batch of inputs for the PyTorch model.
        Assumes the 'batch' pandas Series contains preprocessed data that can be directly
        converted into PyTorch tensors.
        """
        try:
            batch_tensors = [
                torch.tensor(item, dtype=torch.float32)
                for item in batch.tolist()
            ]
            batch_tensor = torch.stack(batch_tensors).to(self._device)
        except Exception as e:
            raise ValueError(
                f"Error converting batch to PyTorch tensors: {e}. "
                "Ensure preprocessed data in the Series is consistently shaped and numerical."
            )
        # disables gradient calculation as we aren't training during inference, this is faster and memory efficient
        with torch.no_grad():
            predictions = self._underlying_model(batch_tensor)

        return pd.Series(predictions.cpu().tolist(), index=batch.index)


class PyTorchModelHandler(BaseModelHandler):
    """
     A handler for running inference with PyTorch models on Spark DataFrames.
     Example:

    Example usage:
    Load Full model saved in gcs:
    result_df = PyTorchModelHandler()
             .model_path("gs://test-bucket/test-model.pt")
             .device("cpu") #optional
             .input_col("input_col")
             .output_col("prediction")
             .pre_processor(preprocess_function) #optional
             .set_return_type(ArrayType(FloatType()))
             .transform(input_df)

     Load state dict saved in gcs:
     1. Define the model's class and constructor arguments
       eg: For ResNet-18, passing weights=None initializes an empty model.
     model_class = models.resnet18
     model_kwargs = {"weights": None} # Or weights=False in older versions

     # 2. Configure the handler
     result_df = (
         PyTorchModelHandler()
         .model_path("gs://my-bucket/resnet18_statedict.pt")
         .device("cpu") #optional
         .set_model_architecture(model_class, **model_kwargs)
         .input_col("features")
         .output_col("predictions")
         .pre_processor(preprocess_function) #optional
         .set_return_type(ArrayType(FloatType()))
         .transform(input_df)
     )
    """

    def __init__(self):
        super().__init__()
        self._model_path = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_class: Optional[Type[torch.nn.Module]] = None
        self._model_args: Optional[tuple] = None
        self._model_kwargs: Optional[dict] = None

    def model_path(self, path: str) -> "PyTorchModelHandler":
        """Sets the GCS path to the saved PyTorch model."""

        if not path.startswith("gs://"):
            raise ValueError("Model path must start with 'gs://'")
        self._model_path = path
        return self

    def device(self, device: str) -> "PyTorchModelHandler":
        """Sets the device to load the PyTorch model on."""
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        self._device = device
        return self

    def set_model_architecture(
        self, model_callable: Callable[..., torch.nn.Module], *args, **kwargs
    ) -> "PyTorchModelHandler":
        """
        Sets the PyTorch model's architecture using a class or a factory function.

        This is required if you are loading a model saved as a state_dict.

        Args:
            model_callable: The model's class (e.g., `MyImageClassifier`) or a factory
                            function (e.g., `torchvision.models.resnet18`) that
                            returns a model instance.
            *args: Positional arguments for the model's constructor or factory function.
            **kwargs: Keyword arguments for the model's constructor or factory function.
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
