import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd
import tenacity
from google.api_core import exceptions
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
        mock_model_instance.generate_content_async = AsyncMock()

        # Mock individual response objects that the API would return
        response1, response2 = MagicMock(), MagicMock()
        response1.text = "response1"
        response2.text = "response2"

        # The mock will return a different response on each call
        mock_model_instance.generate_content_async.side_effect = [
            response1,
            response2,
        ]

        # 2. Instantiate the model and create a test batch
        no_retry = tenacity.retry(stop=tenacity.stop_after_attempt(1))
        gemini_model = GeminiModel(
            "test-model", retry_strategy=no_retry, max_concurrent_requests=2
        )
        input_batch = pd.Series(["p1", "p2"], index=[10, 20])

        # 3. Call the method to be tested
        output_series = gemini_model.call(input_batch)

        self.assertEqual(
            mock_model_instance.generate_content_async.call_count, 2
        )
        mock_model_instance.generate_content_async.assert_any_call("p1")
        mock_model_instance.generate_content_async.assert_any_call("p2")

        expected_output = pd.Series(["response1", "response2"], index=[10, 20])
        pd.testing.assert_series_equal(output_series, expected_output)

    @patch(f"{GEN_AI_HANDLER_PATH}.GenerativeModel")
    def test_call_with_retry_on_api_error(self, mock_generative_model):
        """Test that the model retries on a retryable API error and eventually succeeds."""
        # 1. Setup mock model to simulate failure then success
        mock_model_instance = mock_generative_model.return_value
        successful_response = MagicMock()
        successful_response.text = "success"
        mock_model_instance.generate_content_async = AsyncMock(
            side_effect=[
                exceptions.ResourceExhausted("Rate limit exceeded"),
                exceptions.ServiceUnavailable("Server busy"),
                successful_response,
            ]
        )

        # 2. Create a retry strategy for the test
        retry_strategy = tenacity.retry(
            retry=tenacity.retry_if_exception_type(
                (exceptions.ResourceExhausted, exceptions.ServiceUnavailable)
            ),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_none(),
        )

        # 3. Instantiate the model and create a test batch
        gemini_model = GeminiModel(
            "test-model",
            retry_strategy=retry_strategy,
            max_concurrent_requests=1,
        )
        input_batch = pd.Series(["prompt1"], index=[0])

        # 4. Call the method
        output_series = gemini_model.call(input_batch)

        # 5. Assertions
        # It should be called 3 times: 2 failures, 1 success
        self.assertEqual(
            mock_model_instance.generate_content_async.call_count, 3
        )
        mock_model_instance.generate_content_async.assert_called_with("prompt1")

        expected_output = pd.Series(["success"], index=[0])
        pd.testing.assert_series_equal(output_series, expected_output)

    @patch(f"{GEN_AI_HANDLER_PATH}.GenerativeModel")
    def test_call_fails_after_exhausting_retries(self, mock_generative_model):
        """Test that the call fails if the API error persists after all retries."""
        # 1. Setup mock model to always fail
        mock_model_instance = mock_generative_model.return_value
        mock_model_instance.generate_content_async = AsyncMock(
            side_effect=exceptions.ResourceExhausted("Rate limit exceeded")
        )

        # 2. Create a retry strategy that will be exhausted
        retry_strategy = tenacity.retry(
            retry=tenacity.retry_if_exception_type(
                exceptions.ResourceExhausted
            ),
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_none(),
            reraise=True,
        )

        # 3. Instantiate the model
        gemini_model = GeminiModel(
            "test-model",
            retry_strategy=retry_strategy,
            max_concurrent_requests=1,
        )
        input_batch = pd.Series(["prompt1"])

        # 4. Call the method and assert it raises the final exception
        with self.assertRaises(exceptions.ResourceExhausted):
            gemini_model.call(input_batch)

        # 5. Assert it was called 3 times
        self.assertEqual(
            mock_model_instance.generate_content_async.call_count, 3
        )

    @patch(f"{GEN_AI_HANDLER_PATH}.GenerativeModel")
    def test_max_concurrent_requests_is_respected(self, mock_generative_model):
        """Test that the semaphore correctly limits the number of concurrent API calls."""
        # 1. Setup
        max_concurrent_requests = 3
        total_prompts = 10

        mock_model_instance = mock_generative_model.return_value

        # Helper class to track the number of "in-flight" calls.
        class ConcurrencyTracker:

            def __init__(self):
                self.active_calls = 0
                self.max_concurrent_calls = 0
                self._lock = asyncio.Lock()

            async def mock_api_call(self, prompt: str):
                async with self._lock:
                    self.active_calls += 1
                    self.max_concurrent_calls = max(
                        self.max_concurrent_calls, self.active_calls
                    )

                # Simulate I/O wait. It allows the asyncio event loop to switch to other tasks.
                await asyncio.sleep(0.01)

                async with self._lock:
                    self.active_calls -= 1

                response = MagicMock()
                response.text = f"response for {prompt}"
                return response

        tracker = ConcurrencyTracker()
        mock_model_instance.generate_content_async.side_effect = (
            tracker.mock_api_call
        )

        # 2. Instantiate the model with a concurrency limit and no retries.
        no_retry = tenacity.retry(stop=tenacity.stop_after_attempt(1))
        gemini_model = GeminiModel(
            "test-model",
            retry_strategy=no_retry,
            max_concurrent_requests=max_concurrent_requests,
        )
        input_batch = pd.Series([f"prompt_{i}" for i in range(total_prompts)])

        # 3. Call the method and assert the concurrency high-water mark.
        gemini_model.call(input_batch)
        self.assertEqual(tracker.max_concurrent_calls, max_concurrent_requests)


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

    def test_prompt_with_single_placeholder_succeeds(self):
        """Test that prompt() correctly configures the handler with a valid template."""
        template = "What is the capital of {country}?"
        self.handler.prompt(template)

        self.assertEqual(self.handler._input_col, "country")
        self.assertIsNotNone(self.handler._pre_processor)

        # Test the pre-processor function
        processed_prompt = self.handler._pre_processor("France")
        self.assertEqual(processed_prompt, "What is the capital of France?")

    def test_prompt_with_no_placeholders_raises_error(self):
        """Test that prompt() raises ValueError for a template with no placeholders."""
        template = "This is a static prompt."
        with self.assertRaisesRegex(
            ValueError,
            r"The prompt template must contain exactly one placeholder column, but found 0: \[\]."
            r" Input to prompt should be dynamic based on each row.",
        ):
            self.handler.prompt(template)

    def test_prompt_with_multiple_placeholders_raises_error(self):
        """Test that prompt() raises ValueError for a template with multiple placeholders."""
        template = "What is the population of {city} in {country}?"
        with self.assertRaisesRegex(
            ValueError,
            r"The prompt template must contain exactly one placeholder column, but found 2: \['city', 'country'\]."
            r" To use multiple columns in the prompt, first combine them into a new derived column using dataframe APIs.",
        ):
            self.handler.prompt(template)

    @patch(f"{GEN_AI_HANDLER_PATH}.aiplatform")
    @patch(f"{GEN_AI_HANDLER_PATH}.GeminiModel")
    def test_load_model_success(self, mock_gemini_model, mock_aiplatform):
        """Test the successful loading of a model."""
        project = "my-project"
        location = "us-east1"
        model_name = "gemini-1.5-flash"
        max_requests = 10

        self.handler.project(project).location(location).model(
            model_name
        ).max_concurrent_requests(max_requests)
        loaded_model = self.handler._load_model()

        mock_aiplatform.init.assert_called_once_with(
            project=project, location=location
        )
        mock_gemini_model.assert_called_once_with(
            model_name,
            retry_strategy=self.handler._retry_strategy,
            max_concurrent_requests=max_requests,
        )
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
