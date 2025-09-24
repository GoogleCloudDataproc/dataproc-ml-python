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

"""Unit tests for BaseModelHandler."""

import unittest

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType

from google.cloud.dataproc_ml.inference.base_model_handler import (
    BaseModelHandler,
    Model,
)


class PrefixModel(Model):
    """A simple mock model that applies a prefix to the input."""

    def __init__(self, prefix="processed:"):
        self.prefix = prefix

    def call(self, batch: pd.Series) -> pd.Series:
        return self.prefix + batch.astype(str)


class TestModelHandler(BaseModelHandler):
    """A concrete implementation of BaseModelHandler for testing."""

    def __init__(self, model_to_load: Model):
        super().__init__()
        self._model_to_load = model_to_load

    def _load_model(self) -> Model:
        return self._model_to_load


class BaseModelHandlerTest(unittest.TestCase):
    """Test suite for BaseModelHandler."""

    @classmethod
    def setUpClass(cls):
        """Sets up the SparkSession and a sample DataFrame for tests."""
        cls.spark = (
            SparkSession.builder.appName("BaseModelHandlerTests")
            .master("local[*]")
            .getOrCreate()
        )
        cls.df = cls.spark.createDataFrame(
            [
                ("hello", "world", 1),
                ("foo", "bar", 2),
                ("quick", "brown", 3),
                ("fox", "jumps", 4),
            ],
            ["col_a", "col_b", "col_c"],
        ).repartition(2)

    @classmethod
    def tearDownClass(cls):
        """Stops the SparkSession."""
        cls.spark.stop()

    def _assert_dataframe_equals(self, df1: DataFrame, df2: DataFrame):
        """Asserts that two DataFrames are equal, ignoring row order."""
        self.assertEqual(df1.schema, df2.schema)
        self.assertTrue(df1.exceptAll(df2).isEmpty())
        self.assertTrue(df2.exceptAll(df1).isEmpty())

    def test_transform_single_input_col(self):
        """Tests transform with a single input column and no preprocessor."""
        handler = TestModelHandler(PrefixModel())
        result_df = (
            handler.input_cols("col_a")
            .output_col("prediction")
            .set_return_type(StringType())
            .transform(self.df)
        )

        expected_data = [
            ("hello", "world", 1, "processed:hello"),
            ("foo", "bar", 2, "processed:foo"),
            ("quick", "brown", 3, "processed:quick"),
            ("fox", "jumps", 4, "processed:fox"),
        ]
        expected_df = self.spark.createDataFrame(
            expected_data, self.df.columns + ["prediction"]
        )

        self._assert_dataframe_equals(result_df, expected_df)

    def test_transform_multiple_inputs_with_preprocessor(self):
        """Tests transform with multiple input columns and a preprocessor."""
        handler = TestModelHandler(PrefixModel())

        def concat_preprocessor(
            col_a: pd.Series, col_b: pd.Series
        ) -> pd.Series:
            return col_a + "-" + col_b

        result_df = (
            handler.input_cols("col_a", "col_b")
            .output_col("prediction")
            .pre_processor(concat_preprocessor)
            .set_return_type(StringType())
            .transform(self.df)
        )

        expected_data = [
            ("hello", "world", 1, "processed:hello-world"),
            ("foo", "bar", 2, "processed:foo-bar"),
            ("quick", "brown", 3, "processed:quick-brown"),
            ("fox", "jumps", 4, "processed:fox-jumps"),
        ]
        expected_df = self.spark.createDataFrame(
            expected_data, self.df.columns + ["prediction"]
        )

        self._assert_dataframe_equals(result_df, expected_df)

    def test_transform_raises_error_for_multiple_inputs_no_preprocessor(self):
        """Tests ValueError for multiple inputs without a preprocessor."""
        handler = TestModelHandler(PrefixModel())
        with self.assertRaisesRegex(
            ValueError,
            "A pre_processor must be provided when using multiple input "
            "columns to combine them into a single series for the model.",
        ):
            handler.input_cols("col_a", "col_b").transform(self.df)


if __name__ == "__main__":
    unittest.main()
