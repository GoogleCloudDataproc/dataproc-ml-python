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

"""A module for handling Generative AI model inference on Spark DataFrames."""

import asyncio
import logging
import string
from enum import Enum
from typing import List, Optional

import pandas as pd
import tenacity
from google.api_core import exceptions
from google.cloud import aiplatform
from google.cloud.dataproc.ml.inference.base_model_handler import (
    BaseModelHandler,
    Model,
)
from pyspark.sql.types import StringType
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enumeration for supported model providers."""

    GOOGLE = "google"


class GeminiModel(Model):
    """A concrete implementation of the Model interface for Vertex AI
    Gemini models."""

    def __init__(
        self,
        model_name: str,
        retry_strategy: tenacity.BaseRetrying,
        max_concurrent_requests: int,
        generation_config: GenerationConfig = None,
    ):
        """Initializes the GeminiModel.

        Args:
            model_name: The name of the Gemini model to use (e.g.,
                "gemini-2.5-flash").
            retry_strategy: The tenacity retry decorator to use for API calls.
            max_concurrent_requests: The maximum number of concurrent API
                requests.
            generation_config: The generation configuration for the model.
        """
        self._underlying_model = GenerativeModel(model_name)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._retry_strategy = retry_strategy
        self._generation_config = generation_config

    async def _infer_individual_prompt_async(self, prompt: str):
        # Note: Locking before making retryable calls is important,
        # so we actually wait in this "thread" instead of making requests for
        # other prompts. This will try to control overwhelming the gemini API.
        async with self._semaphore:
            # Wrap the core API call with the retry decorator.
            retryable_call = self._retry_strategy(
                self._underlying_model.generate_content_async
            )
            return await retryable_call(
                prompt, generation_config=self._generation_config
            )

    async def _process_batch_async(self, prompts: List[str]):
        """Processes a batch of prompts with retries and concurrency control."""
        tasks = [
            self._infer_individual_prompt_async(prompt) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def call(self, batch: pd.Series) -> pd.Series:
        """
        Overrides the base method to send prompts to the Gemini API.
        If any API call fails after retries or a prompt is blocked, an
        exception will be raised, allowing Spark to handle the task failure and
        retry the entire task.
        """
        logger.info("Processing batch of size %s", batch.size)
        if not batch.empty:
            logger.debug("Sample prompt: '%s'", batch.iloc[0])

        responses = asyncio.run(self._process_batch_async(batch.tolist()))

        assert len(responses) == len(batch), (
            f"Mismatch between number of prompts ({len(batch)}) and responses "
            f"({len(responses)}). This indicates a potential API issue."
        )
        return pd.Series(
            [response.text for response in responses], index=batch.index
        )


class GenAiModelHandler(BaseModelHandler):
    """A handler for running inference with Gemini models on Spark DataFrames.

    This class extends `BaseModelHandler` to provide a convenient way to apply
    Google's Gemini generative models to data in a distributed manner using
    Spark. It uses a builder pattern for configuration.

    Required Configuration:
        - `.project(str)`: Your Google Cloud project ID.
        - `.location(str)`: The GCP region for the Vertex AI API call.
        - `.prompt(str)`: A prompt template with one or more placeholders for
          input column/s (e.g., "Compare these two texts {col1} & {col2}").

    Optional Configuration:
        - `.model(str)`: The model to use (defaults to "gemini-2.5-flash").
        - `.output_col(str)`: The name of the prediction output column
          (defaults to "predictions").
        - Other methods like `generation_config`,`max_concurrent_requests`, etc.

    Example:
        >>> result_df = (
        ...     GenAiModelHandler()
        ...     .project("my-gcp-project")
        ...     .location("us-central1")
        ...     .model("gemini-2.5-flash")
        ...     .prompt("What is the capital of {city} in single word?")
        ...     .output_col("predictions")
        ...     .generation_config(GenerationConfig(temperature=0.2))
        ...     .transform(df)
        ... )
    """

    _RETRYABLE_GOOGLE_API_EXCEPTIONS = (
        exceptions.ResourceExhausted,  # 429
        exceptions.ServiceUnavailable,  # 503
        exceptions.InternalServerError,  # 500
        exceptions.GatewayTimeout,  # 504
    )

    _DEFAULT_RETRY_STRATEGIES = {
        ModelProvider.GOOGLE: tenacity.retry(
            retry=tenacity.retry_if_exception_type(
                exception_types=_RETRYABLE_GOOGLE_API_EXCEPTIONS
            ),
            wait=tenacity.wait_random_exponential(multiplier=10, min=5, max=60),
            stop=tenacity.stop_after_attempt(5),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            # Re-raising to propagate the underlying exception to the user
            reraise=True,
        )
    }

    def __init__(self):
        super().__init__()
        self._project = None
        self._location = None
        self._model = "gemini-2.5-flash"
        self._provider = ModelProvider.GOOGLE
        self._return_type = StringType()
        self._max_concurrent_requests = 5
        self._retry_strategy = self._DEFAULT_RETRY_STRATEGIES.get(
            self._provider
        )
        self._generation_config: Optional[GenerationConfig] = None

    def model(
        self,
        model: str = "gemini-2.5-flash",
        provider: ModelProvider = ModelProvider.GOOGLE,
    ) -> "GenAiModelHandler":
        """Sets the Gemini model to be used for inference.

        Args:
            model: The name of the model. Defaults to "gemini-2.5-flash".
            provider: Provider of the model.
            Currently only `ModelProvider.GOOGLE` is supported.

        Returns:
            The handler instance for method chaining.
        """
        self._model = model
        self._provider = provider

        # Update the retry strategy to the default for the selected provider.
        self._retry_strategy = self._DEFAULT_RETRY_STRATEGIES.get(
            self._provider
        )
        if self._retry_strategy is None:
            logger.warning(
                "No default retry strategy found for provider '%s'. "
                "Retries will be disabled unless a strategy is set manually.",
                provider.name,
            )
        return self

    def project(self, project: str) -> "GenAiModelHandler":
        """Sets the Google Cloud project for the Vertex AI API call.

        Args:
            project: The GCP project ID.

        Returns:
            The handler instance for method chaining.
        """
        self._project = project
        return self

    def location(self, location: str) -> "GenAiModelHandler":
        """Sets the Google Cloud location (region) for the Vertex AI API call.

        Args:
            location: The GCP location (e.g., "us-central1").

        Returns:
            The handler instance for method chaining.
        """
        self._location = location
        return self

    def prompt(self, prompt_template: str) -> "GenAiModelHandler":
        """Configures the handler using a string template for the prompt.

        This method parses a string template (e.g., "Summarize this:
        {text_column}") to automatically identify the input column(s) of the
        dataframe and create the necessary vectorized pre-processor. It supports
        one or more placeholders.

        Args:
            prompt_template: A string with named placeholders, like
                "Capital of {city} in {country}?".

        Returns:
            The handler instance for method chaining.

        Raises:
            ValueError: If the prompt template contains no placeholders.
        """
        # Get unique placeholders from the input string while preserving order.
        param_names = list(
            dict.fromkeys(
                fname
                for _, fname, _, _ in string.Formatter().parse(prompt_template)
                if fname is not None
            )
        )

        if not param_names:
            raise ValueError(
                "The prompt template must contain at least one placeholder "
                "column, but none were found. Example: 'Summarize: {text_col}'"
            )

        self.input_cols(*param_names)

        def prompt_pre_processor(*series_args: pd.Series) -> pd.Series:
            """Applies prompt template to series in a vectorized way."""

            # Initialize with an empty Series that has the correct index.
            result = pd.Series("", index=series_args[0].index, dtype=object)

            series_map = {
                name: col_series.astype(str)
                for name, col_series in zip(param_names, series_args)
            }

            for literal, field, _, _ in string.Formatter().parse(
                prompt_template
            ):
                if literal:
                    result += literal
                if field:
                    result += series_map[field]
            return result

        self.pre_processor(prompt_pre_processor)
        return self

    def generation_config(
        self, generation_config: GenerationConfig
    ) -> "GenAiModelHandler":
        """Sets the generation config for the model.

        Args:
            generation_config: The `vertexai.generative_models.GenerationConfig`
                object for the model.

        Returns:
            The handler instance for method chaining.
        """
        self._generation_config = generation_config
        return self

    def max_concurrent_requests(self, n: int) -> "GenAiModelHandler":
        """Sets the maximum number of concurrent requests to the model API by
        each Python process.

        Defaults to 5.

        Args:
            n: The maximum number of concurrent requests.

        Returns:
            The handler instance for method chaining.
        """
        self._max_concurrent_requests = n
        return self

    def retry_strategy(
        self, retry_strategy: tenacity.BaseRetrying
    ) -> "GenAiModelHandler":
        """Sets a custom tenacity retry strategy for API calls.

        This will override the default strategy for the selected provider.

        Args:
            retry_strategy: A tenacity retry object (e.g.,
                tenacity.retry(stop=tenacity.stop_after_attempt(3))).

        Returns:
            The handler instance for method chaining.
        """
        self._retry_strategy = retry_strategy
        return self

    def _load_model(self) -> Model:
        """Loads the GeminiModel instance on each Spark executor."""
        if not self._project:
            raise ValueError(
                "Project must be set using .project(project_name)."
            )
        if not self._location:
            raise ValueError(
                "Location must be set using .location(location_name)."
            )
        if self._provider is ModelProvider.GOOGLE:
            aiplatform.init(project=self._project, location=self._location)
            logger.debug("Creating GenerativeModel client for calls to Gemini")
            return GeminiModel(
                model_name=self._model,
                retry_strategy=self._retry_strategy,
                max_concurrent_requests=self._max_concurrent_requests,
                generation_config=self._generation_config,
            )
        else:
            raise NotImplementedError(
                f"Provider '{self._provider.name}' is not supported."
            )
