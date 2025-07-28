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

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import tensorflow as tf
from google.api_core import exceptions as gcp_exceptions

# Assume the handler code is in a file accessible by this path
TENSORFLOW_HANDLER_PATH = (
    "google.cloud.dataproc.ml.inference.tensorflow_model_handler"
)

from google.cloud.dataproc.ml.inference.tensorflow_model_handler import TensorFlowModel, TensorFlowModelHandler


class TestTensorFlowModelUnit(unittest.TestCase):
    """Unit tests for the worker-side TensorFlowModel class using mocks."""

    @patch(f"{TENSORFLOW_HANDLER_PATH}.tf.saved_model.load")
    def test_load_non_existent_path_raises_runtime_error(self, mock_tf_load):
        """
        Tests that a GCS error during loading is caught and re-raised as a RuntimeError.
        """

        # Simulate an error from TensorFlow's GCS integration (e.g., path not found).
        mock_tf_load.side_effect = gcp_exceptions.NotFound("GCS path not found")

        expected_regex = (
            "A Google Cloud Storage error occurred while loading the model from .*. "
            "Please check the GCS path and that the executor has read permissions. "
            "Original error: .*"
        )

        with self.assertRaisesRegex(RuntimeError, expected_regex):
            TensorFlowModel(model_path="gs://fake-bucket/non-existent-path/")

    @patch(f"{TENSORFLOW_HANDLER_PATH}.tf.saved_model.load")
    def test_load_corrupted_model_raises_runtime_error(self, mock_tf_load):
        """
        Tests that a TensorFlow error from a corrupted file is re-raised as a RuntimeError.
        """

        # Simulate a TensorFlow error (e.g., corrupted file).
        mock_tf_load.side_effect = tf.errors.OpError(
            None, None, "Corrupted file", None
        )

        expected_regex = (
            "Failed to load the TensorFlow model from .*. "
            "Ensure the artifact is a valid and uncorrupted SavedModel or TF Checkpoint. "
            "Original error: .*"
        )

        with self.assertRaisesRegex(RuntimeError, expected_regex):
            TensorFlowModel(model_path="gs://fake-bucket/corrupted-model/")

    def test_call_with_multiple_outputs_raises_error(self):
        """
        Tests that the handler fails if the model returns a dictionary with multiple outputs.
        """
        # 1. Create a mock model that returns a dict with two items.
        mock_prediction_dict = {
            "output_1": tf.constant([1.0]),
            "output_2": tf.constant([2.0]),
        }
        mock_signature = MagicMock(return_value=mock_prediction_dict)
        mock_loaded_model = MagicMock()
        mock_loaded_model.signatures = {"serving_default": mock_signature}
        mock_loaded_model.signatures[
            "serving_default"
        ].structured_input_signature[1].keys.return_value = iter(
            ["input_tensor"]
        )

        # 2. Patch the model loading to return our mock model.
        with patch.object(
            TensorFlowModel,
            "_load_model_from_gcs",
            return_value=mock_loaded_model,
        ):
            model = TensorFlowModel(model_path="gs://fake-bucket/fake-model")

            # 3. Assert that a ValueError is raised with a clear message.
            with self.assertRaisesRegex(
                ValueError,
                "Model returned multiple outputs: \\['output_1', 'output_2'\\].*expects a model with a single output",
            ):
                # A Series where each element is a NumPy array.
                model.call(pd.Series([np.array([1.0, 2.0])]))

    def test_call_with_single_output_dict_succeeds(self):
        """
        Tests that the handler correctly processes a model that returns a single-item dictionary.
        """
        # 1. Mock a model returning a single-item dict.
        expected_output = np.array([[0.5], [0.8]])
        mock_prediction_dict = {"my_only_output": tf.constant(expected_output)}
        mock_signature = MagicMock(return_value=mock_prediction_dict)
        mock_loaded_model = MagicMock()
        mock_loaded_model.signatures = {"serving_default": mock_signature}
        mock_loaded_model.signatures[
            "serving_default"
        ].structured_input_signature[1].keys.return_value = iter(
            ["input_tensor"]
        )

        # 2. Patch loading and call the model.
        with patch.object(
            TensorFlowModel,
            "_load_model_from_gcs",
            return_value=mock_loaded_model,
        ):
            model = TensorFlowModel(model_path="gs://fake-bucket/fake-model")
            # Create a dummy series with the correct index for comparison.
            input_series = pd.Series(
                [np.array([1.0]), np.array([2.0])], index=[10, 20]
            )
            result = model.call(input_series)

            # 3. Assert the result is the unpacked value from the dictionary.
            self.assertIsInstance(result, pd.Series)
            # Compare list representations to avoid issues with NumPy array comparison.
            self.assertEqual(result.tolist(), expected_output.tolist())
            self.assertTrue(result.index.equals(input_series.index))

    def test_call_with_invalid_data_type(self):
        with patch.object(
            TensorFlowModel, "_load_model_from_gcs", return_value=MagicMock()
        ):
            model = TensorFlowModel(model_path="gs://fake-bucket/fake-model")
            with self.assertRaisesRegex(
                ValueError, "Error converting batch to TensorFlow tensor"
            ):
                model.call(pd.Series(["a", "b", "c"]))

    def test_call_with_shape_mismatch(self):
        # Mock a Keras model this time to test the other branch of the call method.
        mock_underlying_model = MagicMock(spec=tf.keras.Model)
        mock_underlying_model.side_effect = tf.errors.InvalidArgumentError(
            None, None, "Shape mismatch"
        )
        with patch.object(
            TensorFlowModel,
            "_load_model_from_gcs",
            return_value=mock_underlying_model,
        ):
            model = TensorFlowModel(model_path="gs://fake-bucket/fake-model")
            with self.assertRaisesRegex(
                ValueError,
                "The input data's shape or dtype does not match the model's expected input",
            ):
                model.call(pd.Series([1.0, 2.0]))


class TestTensorFlowModelHandlerUnit(unittest.TestCase):
    """Unit tests for the user-facing TensorFlowModelHandler's configuration."""

    def setUp(self):
        """Create a fresh TensorFlowModelHandler for each test."""
        self.handler = TensorFlowModelHandler()

    def test_builder_methods_chaining_and_value_setting(self):
        gcs_path = "gs://test-bucket/model"
        model_class = tf.keras.applications.MobileNetV2
        model_args = (128,)
        model_kwargs = {"alpha": 1.0}

        chained_handler = self.handler.model_path(
            gcs_path
        ).set_model_architecture(model_class, *model_args, **model_kwargs)

        self.assertIs(chained_handler, self.handler)

        self.assertEqual(self.handler._model_path, gcs_path)
        self.assertEqual(self.handler._model_class, model_class)
        self.assertEqual(self.handler._model_args, model_args)
        self.assertEqual(self.handler._model_kwargs, model_kwargs)

    def test_invalid_model_path_scheme(self):
        invalid_path = "/local/path/to/model"

        with self.assertRaisesRegex(
            ValueError, "Model path must start with 'gs://'"
        ):
            self.handler.model_path(invalid_path)

    def test_load_model_without_path_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Model path must be set using .model_path()."
        ):
            self.handler._load_model()

    @patch(f"{TENSORFLOW_HANDLER_PATH}.TensorFlowModel")
    def test_load_model_instantiates_tensorflow_model_correctly(
        self, mock_tf_model
    ):
        gcs_path = "gs://test-bucket/model"
        model_class = tf.keras.applications.MobileNetV2

        self.handler.model_path(gcs_path).set_model_architecture(model_class)
        self.handler._load_model()

        mock_tf_model.assert_called_once_with(
            model_path=gcs_path,
            model_class=model_class,
            model_args=(),
            model_kwargs={},
        )


if __name__ == "__main__":
    unittest.main()
