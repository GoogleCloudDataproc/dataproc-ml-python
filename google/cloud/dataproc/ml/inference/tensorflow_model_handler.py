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

"""A module for handling TensorFlow model inference on Spark DataFrames."""

import logging
from typing import Optional, Type, Union

import pandas as pd
import tensorflow as tf

from pyspark.sql.types import ArrayType, FloatType

from google.api_core import exceptions as gcp_exceptions
from google.cloud.dataproc.ml.inference.base_model_handler import (
    BaseModelHandler,
    Model,
)

logging.basicConfig(level=logging.INFO)


class TensorFlowModel(Model):
    """
    A concrete implementation of the Model interface for TensorFlow models.
    """

    def __init__(
        self,
        model_path: str,
        model_class: Optional[Type[tf.keras.Model]] = None,
        model_args: Optional[tuple] = None,
        model_kwargs: Optional[dict] = None,
    ):
        """Initializes the TensorFlowModel.

        Args:
            model_path: The path to the saved model artifact.
            model_class: (Optional) The Python class of the Keras model. This
                is required for loading weights-only TF Checkpoints.
            model_args: (Optional) Positional arguments for the model's
                constructor.
            model_kwargs: (Optional) Keyword arguments for the model's
                constructor.
        """
        self._model_path = model_path
        self._model_class = model_class
        self._model_args = model_args if model_args is not None else ()
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._underlying_model = self._load_model_from_gcs()

    def _load_weights_from_checkpoint(self) -> tf.keras.Model:
        """Loads a model from a local checkpoint directory."""
        if not callable(self._model_class):
            raise TypeError(
                "The provided model_class must be a subclass of "
                f"tf.keras.Model, but got {type(self._model_class)}."
            )

        model_instance = self._model_class(
            *self._model_args, **self._model_kwargs
        )

        model_instance.load_weights(self._model_path)
        return model_instance

    def _load_full_model(self) -> tf.Module:
        """Loads a full SavedModel from gcs path."""
        return tf.saved_model.load(self._model_path)

    def _load_model_from_gcs(self) -> Union[tf.Module, tf.keras.Model]:
        """Orchestrates loading a TensorFlow model directly from GCS."""
        logging.info("Loading model from GCS path: %s", self._model_path)
        is_weights_file = self._model_path.endswith(
            (".h5", ".hdf5", ".weights", ".ckpt")
        )
        try:
            if is_weights_file:
                if not self._model_class:
                    raise ValueError(
                        f"Model path '{self._model_path}' appears to be a "
                        "weights file, but the model architecture was not "
                        "provided. "
                        "Please use .set_model_architecture() to specify it."
                    )
                return self._load_weights_from_checkpoint()
            else:
                # No model class, so we load a full SavedModel.
                return self._load_full_model()

        except (gcp_exceptions.NotFound, gcp_exceptions.PermissionDenied) as e:
            raise RuntimeError(
                "A Google Cloud Storage error occurred while loading the model "
                f"from {self._model_path}. Please check the GCS path and "
                f"that the executor has read permissions. Original error: {e}"
            )
        except (IOError, tf.errors.OpError, ValueError) as e:
            # Catching ValueError for cases like 'no checkpoint found'.
            raise RuntimeError(
                f"Failed to load the TensorFlow model from {self._model_path}. "
                "Ensure the artifact is a valid and uncorrupted SavedModel or "
                f"TF Checkpoint. Original error: {e}"
            )

    def call(self, batch: pd.Series) -> pd.Series:
        """Processes a batch of inputs using the loaded TensorFlow model."""
        try:
            # Convert to list first to handle potential mixed types in Series
            # gracefully before stacking.
            batch_list = batch.to_numpy()
            # Explicitly cast to float32 as models often expect this dtype
            input_tensor = tf.cast(tf.stack(batch_list), tf.float32)
        except (
            tf.errors.InvalidArgumentError,
            tf.errors.UnimplementedError,
            ValueError,
            TypeError,
        ) as e:
            raise ValueError(
                f"Error converting batch to TensorFlow tensor: {e}. "
                f"Ensure input data is numerical and consistently shaped."
            )
        try:
            # --- Prediction Logic ---
            if isinstance(self._underlying_model, tf.keras.Model):
                predictions = self._underlying_model(
                    input_tensor, training=False
                )
            else:
                input_name = next(
                    iter(
                        self._underlying_model.signatures["serving_default"]
                        .structured_input_signature[1]
                        .keys()
                    )
                )
                predictions = self._underlying_model.signatures[
                    "serving_default"
                ](**{input_name: input_tensor})
        except tf.errors.InvalidArgumentError as e:

            raise ValueError(
                "The input data's shape or dtype does not match the model's "
                f"expected input. Original error: {e}"
            )

        if isinstance(predictions, dict):
            # FAIL if the model returns more than one output.
            if len(predictions) > 1:
                raise ValueError(
                    "Model returned multiple outputs: "
                    f"{list(predictions.keys())}. This handler expects a "
                    "model with a single output."
                )
            # If there's exactly one output, extract it.
            output_tensor = next(iter(predictions.values()))
        else:
            # Handle single, non-dict tensor output.
            output_tensor = predictions

        return pd.Series(output_tensor.numpy().tolist(), index=batch.index)


class TensorFlowModelHandler(BaseModelHandler):
    """A handler for running inference with Tensorflow models on
    Spark DataFrames.

    This handler supports two primary modes of operation:
    1.  Loading a full TensorFlow SavedModel.
    2.  Loading a model from a weights file (e.g., H5 or checkpoint), which
        requires providing the model's Keras class architecture separately.

    Currently, only models returning a single output are supported. The handler
    casts the input to a tensor with dtype `tf.float32`.

    Required Configuration:
        - `.model_path(str)`: The GCS path to the model directory or file.
        - `.input_col(str)`: The column in the DataFrame to use as input.
        - `.set_model_architecture(...)`: This is **required** when loading from
          a weights file, but should **not** be used when loading a SavedModel.

    Optional Configuration:
        - `.output_col(str)`: The name for the prediction output column
          (defaults to "predictions").
        - `.pre_processor(callable)`: A function to preprocess input data.
        - `.set_return_type(DataType)`: The Spark `DataType` of the output
          (defaults to `ArrayType(FloatType())`).

    Example:
        Load a full model in SavedModel format:

        >>> result_df = (
        ...     TensorFlowModelHandler()
        ...     .model_path("gs://test-bucket/test-model-saved-dir")
        ...     .input_col("input_col")
        ...     .transform(input_df)
        ... )

        Load a model from a checkpoint file:

        >>> result_df = (
        ...     TensorFlowModelHandler()
        ...     .model_path("gs://test-bucket/test-model-checkpoint.h5")
        ...     .set_model_architecture(model_class, **model_kwargs)
        ...     .input_col("input_col")
        ...     .transform(input_df)
        ... )
    """

    def __init__(self):
        super().__init__()
        self._model_path: Optional[str] = None
        self._model_class: Optional[Type[tf.keras.Model]] = None
        self._model_args: Optional[tuple] = None
        self._model_kwargs: Optional[dict] = None
        self._return_type = ArrayType(FloatType())

    def model_path(self, path: str) -> "TensorFlowModelHandler":
        """Sets the GCS path to the saved TensorFlow model artifact.

        Args:
            path: The GCS path to the model directory or file.

        Returns:
            The handler instance for method chaining.

        Raises:
            ValueError: If the path is not a string or does not start with
                "gs://".
        """

        if not isinstance(path, str) or not path.startswith("gs://"):
            raise ValueError("Model path must start with 'gs://'")

        self._model_path = path

        return self

    def set_model_architecture(
        self, model_class: Type[tf.keras.Model], *args, **kwargs
    ) -> "TensorFlowModelHandler":
        """Sets the TensorFlow Keras model's architecture using its class.

        This is required if you are loading a model from a weights file (e.g., a
        checkpoint).

        Args:
            model_class: The model's class (e.g.,
                `tf.keras.applications.ResNet50`).
            *args: Positional arguments for the model's constructor.
            **kwargs: Keyword arguments for the model's constructor.

        Returns:
            The handler instance for method chaining.

        Raises:
            TypeError: If `model_class` is not a callable.
        """
        if not callable(model_class):
            raise TypeError(
                "model_class must be a callable that returns a "
                "tf.keras.Model instance."
            )

        self._model_class = model_class
        self._model_args = args
        self._model_kwargs = kwargs

        return self

    def _load_model(self) -> Model:
        """Factory method to create the TensorFlowModel instance."""
        if not self._model_path:
            raise ValueError("Model path must be set using .model_path().")

        return TensorFlowModel(
            model_path=self._model_path,
            model_class=self._model_class,
            model_args=self._model_args,
            model_kwargs=self._model_kwargs,
        )
