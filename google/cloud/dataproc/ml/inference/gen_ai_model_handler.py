from enum import Enum

from pyspark.sql.types import StringType
from vertexai.generative_models import GenerativeModel

import pandas as pd
from google.cloud import aiplatform

from google.cloud.dataproc.ml.inference.base_model_handler import BaseModelHandler
from google.cloud.dataproc.ml.inference.base_model_handler import Model


class ModelProvider(Enum):
    """Enumeration for supported model providers."""

    GOOGLE = "google"


class GeminiModel(Model):
    """A concrete implementation of the Model interface for Vertex AI Gemini models."""

    def __init__(self, model_name: str, api_batch_size: int):
        """Initializes the GeminiModel.

        Args:
            model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash").
            api_batch_size: The number of prompts to send in a single API call.
        """
        self._underlying_model = GenerativeModel(model_name)
        self._api_batch_size = api_batch_size

    def call(self, batch: pd.Series) -> pd.Series:
        """
        Overrides the base method to send prompts to the Gemini API in a single batch.
        This is significantly more efficient than making individual calls and handles
        API limits by chunking large batches. If any API call fails or a prompt
        is blocked, an exception will be raised, allowing Spark to handle the
        task failure and retry.
        """
        prompts = batch.tolist()

        all_predictions = []
        for i in range(0, len(prompts), self._api_batch_size):
            chunk = prompts[i : i + self._api_batch_size]
            # Let exceptions from the API call or blocked candidates propagate up.
            response = self._underlying_model.generate_content(chunk)
            predictions_chunk = [
                candidate.text for candidate in response.candidates
            ]
            all_predictions.extend(predictions_chunk)

        # This will raise an error if the number of predictions does not match
        # the number of prompts, which is the desired behavior for data integrity.
        return pd.Series(all_predictions, index=batch.index)


class GenAiModelHandler(BaseModelHandler):
    """A handler for running inference with Gemini models on Spark DataFrames.

    This class extends `BaseModelHandler` to provide a convenient way to apply
    Google's Gemini generative models to data in a distributed manner using Spark.
    It uses a builder pattern for configuration.

    Example usage:
        handler = GenAiModelHandler()
        result_df = handler.project("my-gcp-project") \\
                           .location("us-central1") \\
                           .model("gemini-1.5-flash") \\
                           .input_col("prompts") \\
                           .output_col("predictions") \\
                           .transform(df)
    """

    def __init__(self):
        super().__init__()
        self._project = None
        self._location = None
        self._model = "gemini-2.5-flash"
        self._api_batch_size = 250
        self._provider = ModelProvider.GOOGLE
        self._return_type = StringType()

    # TODO: Support other parameters like endpoint
    def model(
        self,
        model: str = "gemini-2.5-flash",
        provider: ModelProvider = ModelProvider.GOOGLE,
    ) -> "GenAiModelHandler":
        """Sets the Gemini model to be used for inference.

        Args:
            model: The name of the model (e.g., "gemini-1.5-flash").
            provider: Provider of the model. Currently only `ModelProvider.GOOGLE`
                is supported.

        Returns:
            The handler instance for method chaining.
        """
        self._model = model
        self._provider = provider
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

    def set_api_batch_size(self, size: int) -> "GenAiModelHandler":
        """Sets the batch size for API calls to the Gemini model.

        Args:
            size: The number of prompts to send in a single API request.

        Returns:
            The handler instance for method chaining.
        """
        self._api_batch_size = size
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
            return GeminiModel(self._model, api_batch_size=self._api_batch_size)
        else:
            raise NotImplementedError(
                f"Provider '{self._provider.name}' is not supported."
            )
