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

import os
import unittest
import json
import pandas as pd

from pyspark.errors.exceptions.captured import PythonException
from pyspark.sql import SparkSession, DataFrame
from google.cloud.dataproc_ml.inference import GenAiModelHandler
from vertexai.generative_models import GenerationConfig


class TestGenAiModelHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.df = cls.spark.createDataFrame(
            [
                ("Bengaluru", "India"),
                ("London", "UK"),
                ("San Francisco", "USA"),
                ("Paris", "France"),
                ("Tokyo", "Japan"),
                ("Sydney", "Australia"),
                ("New York", "USA"),
            ],
            ["city", "country"],
        ).repartition(  # Repartitioning to have multiple rows in same udf batch
            3
        )

        cls.expected = cls.spark.createDataFrame(
            [
                ("Bengaluru", "BLR"),
                ("London", "LHR"),
                ("San Francisco", "SFO"),
                ("Paris", "CDG"),
                ("Tokyo", "HND"),
                ("Sydney", "SYD"),
                ("New York", "JFK"),
            ],
            ["city", "predictions"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_pre_processor(self):
        def get_prompt(city: pd.Series) -> pd.Series:
            return (
                "What is the airport code of largest airport in "
                + city
                + "? Answer in single word."
            )

        gen_ai_handler = (
            GenAiModelHandler()
            .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
            .location(os.getenv("GOOGLE_CLOUD_REGION"))
            .input_cols("city")
            .pre_processor(get_prompt)
        )

        df = gen_ai_handler.transform(self.df)
        self._assert_dataframe_equals(
            df.select("city", "predictions"), self.expected
        )

    def test_prompt_template(self):
        gen_ai_handler = GenAiModelHandler().prompt(
            "What is the airport code of largest airport in {city}? "
            "Answer in single word."
        )

        df = gen_ai_handler.transform(self.df)
        self._assert_dataframe_equals(
            df.select("city", "predictions"), self.expected
        )

    def test_prompt_template_multiple_cols(self):
        gen_ai_handler = GenAiModelHandler().prompt(
            "We are visiting {city}. What is the airport code of the "
            "largest airport in {city}, {country}? Answer in a single word."
        )

        df = gen_ai_handler.transform(self.df)
        self._assert_dataframe_equals(
            df.select("city", "predictions"), self.expected
        )

    def test_generation_config(self):
        gen_ai_handler = (
            GenAiModelHandler()
            .prompt(
                "What is the airport code of largest airport in {city}? "
                "Answer in single word."
            )
            .generation_config(
                GenerationConfig(candidate_count=2, temperature=0.7)
            )
        )

        # response.text property fails for multiple candidates
        with self.assertRaises(PythonException) as e:
            gen_ai_handler.transform(self.df).collect()
        self.assertIn("The response has multiple candidates.", str(e.exception))

    def test_json_output_with_schema(self):
        """Tests that model can return JSON output conforming to a schema."""
        customer_requests_df = self.spark.createDataFrame(
            [
                (
                    "xyz@gmail.com",
                    "I need the 'AcousticPro Guitar', model G-123, "
                    "willing to pay up to $499.99, do you have it?",
                ),
                (
                    "abc@gmail.com",
                    "I am looking for a monitor, MSI XYZ, is it available "
                    "for anything under $99?",
                ),
            ],
            ["email", "request"],
        )
        # Define the schema for a product
        product_schema = {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The name of the product.",
                },
                "item_id": {
                    "type": "string",
                    "description": "A unique identifier for the product, SKU.",
                },
                "price": {
                    "type": "number",
                    "description": "The price of the product.",
                },
            },
            "required": ["product_name", "price"],
        }
        gen_ai_handler = (
            GenAiModelHandler()
            .prompt(
                "Please extract information for the following item: {request}"
            )
            .generation_config(
                GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=product_schema,
                )
            )
        )

        results = gen_ai_handler.transform(customer_requests_df).collect()

        self.assertEqual(len(results), 2)
        for row in results:
            prediction_json = json.loads(row["predictions"])
            self.assertIn("product_name", prediction_json)
            self.assertIn("price", prediction_json)
            self.assertIsInstance(prediction_json["product_name"], str)
            self.assertIsInstance(prediction_json["price"], (int, float))
            if "item_id" in prediction_json:
                self.assertIsInstance(prediction_json["item_id"], str)

    def test_unsupported_model_name_raises_exception(self):
        gen_ai_handler = (
            GenAiModelHandler().model("gemini-xyz-4").input_cols("city")
        )

        with self.assertRaises(PythonException) as e:
            gen_ai_handler.transform(self.df).collect()  # collect to force eval
        self.assertIn("NotFound: 404", str(e.exception))
        self.assertIn("gemini-xyz-4", str(e.exception))

    def _assert_dataframe_equals(
        self, actual_df: DataFrame, expected_df: DataFrame
    ):
        self.assertEqual(
            actual_df.schema, expected_df.schema, "Schemas are not equal"
        )
        self.assertTrue(
            actual_df.exceptAll(expected_df).isEmpty(),
            "Actual DataFrame has rows not in Expected",
        )
        self.assertTrue(
            expected_df.exceptAll(actual_df).isEmpty(),
            "Expected DataFrame has rows not in Actual",
        )


if __name__ == "__main__":
    unittest.main()
