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

"""Unit tests for VertexEndpointHandler."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from pyspark.sql.types import ArrayType, DoubleType, StringType

from google.cloud.dataproc_ml.inference.vertex_endpoint_handler import (
    VertexEndpoint,
    VertexEndpointHandler,
)

# The path for patching must be where the object is *looked up*.
ENDPOINT_HANDLER_PATH = (
    "google.cloud.dataproc_ml.inference.vertex_endpoint_handler"
)


class TestVertexEndpoint(unittest.TestCase):
    """Tests for the VertexEndpoint class."""

    @patch(f"{ENDPOINT_HANDLER_PATH}.aiplatform")
    def test_init(self, mock_aiplatform):
        """Tests that the VertexEndpoint initializes the API client."""
        endpoint_name = "my-endpoint"
        project = "my-project"
        location = "us-central1"

        VertexEndpoint(
            endpoint=endpoint_name, project=project, location=location
        )

        mock_aiplatform.init.assert_called_once_with(
            project=project, location=location
        )
        mock_aiplatform.Endpoint.assert_called_once_with(
            endpoint_name=endpoint_name
        )

    @patch(f"{ENDPOINT_HANDLER_PATH}.aiplatform")
    def test_call_sends_batched_requests(self, mock_aiplatform):
        """Tests that the call method sends requests in batches."""
        # 1. Setup mock endpoint and a side_effect to generate dynamic responses
        mock_endpoint_client = mock_aiplatform.Endpoint.return_value

        def mock_predict_side_effect(instances, **kwargs):
            mock_result = MagicMock()
            # Return a number of predictions matching the number of instances
            mock_result.predictions = [[0.1, 0.9]] * len(instances)
            return mock_result

        mock_endpoint_client.predict.side_effect = mock_predict_side_effect

        # 2. Instantiate the model and create a test batch
        predict_params = {"param1": "value1"}
        model = VertexEndpoint(
            endpoint="test-endpoint",
            batch_size=2,
            predict_parameters=predict_params,
            use_dedicated_endpoint=True,
        )

        input_batch = pd.Series([[1], [2], [3]], index=[10, 20, 30])

        # 3. Call the method to be tested
        output_series = model.call(input_batch)

        # 4. Assertions
        self.assertEqual(mock_endpoint_client.predict.call_count, 2)
        mock_endpoint_client.predict.assert_any_call(
            instances=[[1], [2]],
            parameters=predict_params,
            use_dedicated_endpoint=True,
        )
        mock_endpoint_client.predict.assert_any_call(
            instances=[[3]],
            parameters=predict_params,
            use_dedicated_endpoint=True,
        )

        # The side_effect returns [[0.1, 0.9]] for each instance.
        expected_output = pd.Series(
            [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]], index=[10, 20, 30]
        )
        pd.testing.assert_series_equal(output_series, expected_output)

    @patch(f"{ENDPOINT_HANDLER_PATH}.aiplatform")
    def test_call_raises_error_on_prediction_mismatch(self, mock_aiplatform):
        """Tests that call() raises an error if prediction count mismatches."""
        # 1. Setup mock endpoint to return fewer predictions than instances
        mock_endpoint_client = mock_aiplatform.Endpoint.return_value
        mock_prediction_result = MagicMock()
        mock_prediction_result.predictions = [[0.1, 0.9]]  # Only 1 prediction
        mock_endpoint_client.predict.return_value = mock_prediction_result

        # 2. Instantiate the model and create a test batch
        model = VertexEndpoint(endpoint="test-endpoint", batch_size=2)
        input_batch = pd.Series([[1], [2]])  # 2 instances

        # 3. Call the method and assert it raises an AssertionError
        with self.assertRaisesRegex(AssertionError, "Mismatch between number"):
            model.call(input_batch)


class TestVertexEndpointHandler(unittest.TestCase):
    """Tests for the VertexEndpointHandler class."""

    def setUp(self):
        """Set up a new handler for each test."""
        self.handler = VertexEndpointHandler(endpoint="test-endpoint")

    def test_initialization_defaults(self):
        """Test that the handler initializes with correct default values."""
        self.assertEqual(self.handler.endpoint, "test-endpoint")
        self.assertIsNone(self.handler._project)
        self.assertIsNone(self.handler._location)
        self.assertIsNone(self.handler._predict_parameters)
        self.assertEqual(self.handler._batch_size, 10)
        self.assertFalse(self.handler._use_dedicated_endpoint)
        self.assertIsInstance(self.handler._return_type, ArrayType)
        self.assertIsInstance(self.handler._return_type.elementType, DoubleType)

    def test_builder_methods_chaining(self):
        """Test that builder methods correctly set values and allow chaining."""
        project = "test-project"
        location = "us-central1"
        params = {"key": "value"}
        return_type = StringType()

        chained_handler = (
            self.handler.project(project)
            .location(location)
            .predict_parameters(params)
            .batch_size(50)
            .use_dedicated_endpoint(True)
            .set_return_type(return_type)
        )

        self.assertIs(chained_handler, self.handler)
        self.assertEqual(self.handler._project, project)
        self.assertEqual(self.handler._location, location)
        self.assertEqual(self.handler._predict_parameters, params)
        self.assertEqual(self.handler._batch_size, 50)
        self.assertTrue(self.handler._use_dedicated_endpoint)
        self.assertEqual(self.handler._return_type, return_type)

    @patch(f"{ENDPOINT_HANDLER_PATH}.VertexEndpoint")
    def test_load_model_success(self, mock_vertex_endpoint):
        """Test the successful loading of a model."""
        project = "my-project"
        location = "us-east1"
        params = {"a": 1}

        self.handler.project(project).location(location).predict_parameters(
            params
        ).batch_size(20).use_dedicated_endpoint(True)

        loaded_model = self.handler._load_model()

        mock_vertex_endpoint.assert_called_once_with(
            self.handler.endpoint,
            project=project,
            location=location,
            predict_parameters=params,
            batch_size=20,
            use_dedicated_endpoint=True,
        )
        self.assertEqual(loaded_model, mock_vertex_endpoint.return_value)
