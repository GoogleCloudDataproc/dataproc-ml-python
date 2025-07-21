import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from pyspark.sql.types import StringType

from google.cloud.dataproc.ml.inference.gen_ai_model_handler import (
    GenAiModelHandler,
    GeminiModel,
    ModelProvider,
)

# The path for patching must be where the object is *looked up*, not where it's defined.
GEN_AI_HANDLER_PATH = "google.cloud.dataproc.ml.inference.gen_ai_model_handler"


class TestGeminiModel(unittest.TestCase):
    """Tests for the GeminiModel class."""

    @patch(f"{GEN_AI_HANDLER_PATH}.GenerativeModel")
    def test_call_sends_individual_prompts(self, mock_generative_model):
        """Test the call method sends a request for each prompt individually."""
        # 1. Setup mock model and its response
        mock_model_instance = mock_generative_model.return_value

        # Mock individual response objects that the API would return
        response1, response2 = MagicMock(), MagicMock()
        response1.text = "response1"
        response2.text = "response2"

        # The mock will return a different response on each call
        mock_model_instance.generate_content.side_effect = [
            response1,
            response2,
        ]

        # 2. Instantiate the model and create a test batch
        gemini_model = GeminiModel("test-model")
        input_batch = pd.Series(["p1", "p2"], index=[10, 20])

        # 3. Call the method to be tested
        output_series = gemini_model.call(input_batch)

        self.assertEqual(mock_model_instance.generate_content.call_count, 2)
        mock_model_instance.generate_content.assert_any_call("p1")
        mock_model_instance.generate_content.assert_any_call("p2")

        expected_output = pd.Series(["response1", "response2"], index=[10, 20])
        pd.testing.assert_series_equal(output_series, expected_output)


class TestGenAiModelHandler(unittest.TestCase):
    """Tests for the GenAiModelHandler class."""

    def setUp(self):
        """Set up a new handler for each test."""
        self.handler = GenAiModelHandler()

    def test_initialization_defaults(self):
        """Test that the handler initializes with correct default values."""
        self.assertIsNone(self.handler._project)
        self.assertIsNone(self.handler._location)
        self.assertEqual(self.handler._model, "gemini-2.5-flash")
        self.assertEqual(self.handler._provider, ModelProvider.GOOGLE)
        self.assertIsInstance(self.handler._return_type, StringType)

    def test_builder_methods_chaining(self):
        """Test that builder methods correctly set values and allow chaining."""
        project = "test-project"
        location = "us-central1"
        model_name = "gemini-pro"

        chained_handler = (
            self.handler.project(project).location(location).model(model_name)
        )

        self.assertIs(chained_handler, self.handler)
        self.assertEqual(self.handler._project, project)
        self.assertEqual(self.handler._location, location)
        self.assertEqual(self.handler._model, model_name)

    @patch(f"{GEN_AI_HANDLER_PATH}.aiplatform")
    @patch(f"{GEN_AI_HANDLER_PATH}.GeminiModel")
    def test_load_model_success(self, mock_gemini_model, mock_aiplatform):
        """Test the successful loading of a model."""
        project = "my-project"
        location = "us-east1"
        model_name = "gemini-1.5-flash"

        self.handler.project(project).location(location).model(model_name)
        loaded_model = self.handler._load_model()

        mock_aiplatform.init.assert_called_once_with(
            project=project, location=location
        )
        mock_gemini_model.assert_called_once_with(model_name)
        self.assertEqual(loaded_model, mock_gemini_model.return_value)

    def test_load_model_missing_project_raises_error(self):
        """Test that _load_model raises ValueError if project is not set."""
        self.handler.location("us-central1")
        with self.assertRaisesRegex(ValueError, "Project must be set"):
            self.handler._load_model()

    def test_load_model_missing_location_raises_error(self):
        """Test that _load_model raises ValueError if location is not set."""
        self.handler.project("my-project")
        with self.assertRaisesRegex(ValueError, "Location must be set"):
            self.handler._load_model()


if __name__ == "__main__":
    unittest.main()
