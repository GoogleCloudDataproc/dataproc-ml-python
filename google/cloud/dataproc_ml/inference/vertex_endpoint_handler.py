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

"""A module for handling model inference on Spark DataFrames using a
Vertex AI Endpoint."""
import logging
from typing import Dict, List, Optional

import pandas as pd
from pyspark.sql.types import ArrayType, DoubleType

from google.cloud import aiplatform
from google.cloud.dataproc_ml.inference.base_model_handler import (
    BaseModelHandler,
    Model,
)

logger = logging.getLogger(__name__)


class VertexEndpoint(Model):
    """A concrete implementation of the Model interface for a
    Vertex AI Endpoint."""

    def __init__(
        self,
        endpoint: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        predict_parameters: Optional[Dict] = None,
        batch_size: Optional[int] = None,
        use_dedicated_endpoint: bool = False,
    ):
        """Initializes the VertexEndpoint.

        Args:
            endpoint: The name of the Vertex AI Endpoint.
            project: The GCP project ID.
            location: The GCP location.
            predict_parameters: Parameters for the prediction call.
            batch_size: The number of instances to include in each prediction
                request. Defaults to 10.
            use_dedicated_endpoint: Whether to use the dedicated endpoint for
                prediction. Defaults to False.
        """
        aiplatform.init(project=project, location=location)
        self.endpoint_client = aiplatform.Endpoint(endpoint_name=endpoint)
        self.predict_parameters = predict_parameters
        self.batch_size = batch_size
        self.use_dedicated_endpoint = use_dedicated_endpoint

    def call(self, batch: pd.Series) -> pd.Series:
        """Overrides the base method to send instances to the
        Vertex AI Endpoint."""

        # Convert the pandas Series to a list of instances.
        instances: List = batch.tolist()

        all_predictions = []

        for i in range(0, len(instances), self.batch_size):
            batch_instances = instances[i : i + self.batch_size]
            prediction_result = self.endpoint_client.predict(
                instances=batch_instances,
                parameters=self.predict_parameters,
                use_dedicated_endpoint=self.use_dedicated_endpoint,
            )
            all_predictions.extend(prediction_result.predictions)

        assert len(all_predictions) == len(instances), (
            f"Mismatch between number of instances ({len(instances)}) and "
            f"predictions ({len(all_predictions)}). Potential API issue."
        )

        return pd.Series(all_predictions, index=batch.index)


class VertexEndpointHandler(BaseModelHandler):
    """A handler for running inference with a deployed model on a
    Vertex AI Endpoint."""

    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint
        self._project = None
        self._location = None
        self._predict_parameters = None
        self._batch_size = 10
        self._use_dedicated_endpoint = False
        self.set_return_type(ArrayType(DoubleType()))

    def project(self, project: str) -> "VertexEndpointHandler":
        """Sets the Google Cloud project for the Vertex AI API call."""
        self._project = project
        return self

    def location(self, location: str) -> "VertexEndpointHandler":
        """Sets the Google Cloud location (region) for Vertex AI API call."""
        self._location = location
        return self

    def predict_parameters(self, parameters: Dict) -> "VertexEndpointHandler":
        """Sets the parameters for the prediction call."""
        self._predict_parameters = parameters
        return self

    def batch_size(self, batch_size: int) -> "VertexEndpointHandler":
        """Sets the number of instances to send in each prediction request.

        Defaults to 10 if not set.
        """
        self._batch_size = batch_size
        return self

    def use_dedicated_endpoint(
        self, use_dedicated_endpoint: bool
    ) -> "VertexEndpointHandler":
        """Sets whether to use the dedicated endpoint for prediction."""
        self._use_dedicated_endpoint = use_dedicated_endpoint
        return self

    def _load_model(self) -> Model:
        """Loads the VertexEndpoint instance on each Spark executor."""
        return VertexEndpoint(
            self.endpoint,
            project=self._project,
            location=self._location,
            predict_parameters=self._predict_parameters,
            batch_size=self._batch_size,
            use_dedicated_endpoint=self._use_dedicated_endpoint,
        )
