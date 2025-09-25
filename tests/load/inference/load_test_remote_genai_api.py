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

import math
import os
import unittest

from pyspark.sql import SparkSession

from google.cloud.dataproc_ml.inference import GenAiModelHandler

# --- Constants for configuration ---
# Total number of prompts to send to the Gemini API.
NUM_PROMPTS = 100
# The number of prompts to process in each Spark partition.
# This helps control the parallelism of the API calls.
ROWS_PER_PARTITION = 50


class GenAITestSuite(unittest.TestCase):
    """
    A load test suite for the GenAiModelHandler.

    This class is designed to test the performance and reliability of the
    GenAiModelHandler by sending a configurable number of prompts to the
    Gemini API in a distributed Spark environment.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initializes the SparkSession with the BigQuery connector.
        This is executed once for the entire test class.
        """
        cls.spark = (
            SparkSession.builder.config(
                "spark.jars.packages",
                "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.0",
            )
            .appName("GenAI Load Test")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        """
        Stops the SparkSession after all tests are complete.
        """
        cls.spark.stop()

    def test_gemini_api_calls(self):
        """
        Reads some data from public BQ tables and makes calls to Gemini in parallel
        """
        gen_ai_handler = (
            GenAiModelHandler()
            .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
            .location(os.getenv("GOOGLE_CLOUD_REGION"))
            .input_cols("word")
            .output_col("explanation")
            .pre_processor(
                lambda word: "Explain the word in plain english: " + word
            )
        )
        words_df = (
            self.spark.read.format("bigquery")
            .option("table", "bigquery-public-data:samples.shakespeare")
            .load()
            .limit(NUM_PROMPTS)
        )

        num_partitions = math.ceil(NUM_PROMPTS / ROWS_PER_PARTITION)
        repartitioned_df = words_df.repartition(num_partitions)

        result_df = gen_ai_handler.transform(repartitioned_df)

        self.assertEqual(result_df.count(), NUM_PROMPTS)

        print("GenAI model output:")
        result_df.show(NUM_PROMPTS, truncate=False)
