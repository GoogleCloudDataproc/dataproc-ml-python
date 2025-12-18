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

"""Integration test for VertexEndpointHandler."""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

from google.cloud.dataproc_ml.inference import VertexEndpointHandler


def create_prompt(cities: pd.Series, countries: pd.Series) -> pd.Series:
    """A pre-processor that wraps each text input in a dictionary with
    a 'prompt' key."""
    prompt_series = "Describe" + cities + " in " + countries
    return prompt_series.apply(lambda x: {"prompt": x, "max_tokens": 256})


def test_vertex_endpoint_handler():
    """Tests the VertexEndpointHandler with a live endpoint."""
    spark = SparkSession.builder.appName(
        "VertexEndpointHandlerTest"
    ).getOrCreate()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # TODO: Replace with endpoint creation during test run which shouldn't
    #  take more than 20 mins
    endpoint_name = "1121351227238514688"

    # Create a sample DataFrame with feature vectors
    data = [
        ("Delhi", "India"),
        ("Beijing", "China"),
    ]
    df = spark.createDataFrame(data, ["city", "country"])

    # Configure and apply the handler
    handler = (
        VertexEndpointHandler(endpoint=endpoint_name)
        .input_cols("city", "country")
        .output_col("predictions")
        .use_dedicated_endpoint(True)
        .pre_processor(create_prompt)
        .set_return_type(StringType())
        .project(project)
        .location(location)
    )

    result_df = handler.transform(df)
    results = result_df.collect()

    assert len(results) == 2
    assert "predictions" in result_df.columns
    assert len(results[0]["predictions"]) > 0  # Check for non-empty prediction

    spark.stop()
