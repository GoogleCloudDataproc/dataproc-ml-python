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

"""Defines the base classes for handling model inference on Spark DataFrames."""

from abc import ABC
from typing import Iterator, Callable, Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DataType, StructType


class Model(ABC):
    """An abstract interface for a model to be used within a BaseModelHandler.

    This class defines the contract for models, requiring them to implement
    method for performing batch prediction.
    """

    def call(self, batch: pd.Series) -> pd.Series:
        """Applies the model to the given input batch.

        Args:
            batch: A pandas Series containing the batch of inputs to process.

        Returns:
            A pandas Series containing the prediction results.
        """
        raise NotImplementedError


class BaseModelHandler(ABC):
    """An abstract base class for applying a model to a Spark DataFrame.

    This handler uses the high-performance Pandas UDF (iterator of series)
    pattern to apply a model to each partition of a DataFrame. It is
    designed to be configured using a builder pattern.

    Subclasses must implement the `_load_model` method, which is responsible
    for loading the model instance on each Spark executor.

    Example:
        >>> class MyModelHandler(BaseModelHandler):
        ...     def _load_model(self):
        ...         return MyModel()
        ...
        >>> handler = MyModelHandler()
        >>> result_df = (
        ...     handler.input_col("features")
        ...     .output_col("predictions")
        ...     .pre_processor(my_pre_processor)
        ...     .transform(df)
        ... )
    """

    def __init__(self):
        self._input_col = None
        self._output_col: str = "predictions"
        self._return_type: DataType = StructType()
        self._pre_processor: Callable[[Any], Any] = None

    def _load_model(self) -> Model:
        """Loads the model instance.

        This method is called once per Spark task (on the executor) to
        initialize the model. Subclasses must implement this method.

        Returns:
            An instance of a class that inherits from `Model`.
        """
        raise NotImplementedError

    def input_col(self, input_col: str) -> "BaseModelHandler":
        """Sets the name of the input column from the DataFrame.

        Args:
            input_col: The name of the column to be used as input for the model.

        Returns:
            The handler instance for method chaining.
        """
        self._input_col = input_col
        return self

    def output_col(self, output_col: str) -> "BaseModelHandler":
        """Sets the name of the output column to be created.

        Args:
            output_col: The name for the new column that will store
                predictions. Defaults to "predictions".

        Returns:
            The handler instance for method chaining.
        """
        self._output_col = output_col
        return self

    def set_return_type(self, return_type: DataType) -> "BaseModelHandler":
        """Sets the Spark DataType of the output column.

        Defaults to StringType if not specified.

        Args:
            return_type: The Spark DataType of the prediction column (e.g.,
                FloatType(), IntegerType()).

        Returns:
            The handler instance for method chaining.
        """
        self._return_type = return_type
        return self

    def pre_processor(
        self, pre_processor: Callable[[Any], Any]
    ) -> "BaseModelHandler":
        """Sets the preprocessing function to be applied to the input column.

        Args:
            pre_processor: A function that takes a single value from the
                input column and returns a processed value.

        Returns:
            The handler instance for method chaining.
        """
        self._pre_processor = pre_processor
        return self

    def _create_predict_udf(self):
        """Creates a Pandas UDF for model inference.

        This internal method constructs the UDF that Spark will distribute.
        The UDF handles loading the model and applying it to batches of data.

        Returns:
            A configured Pandas UDF.
        """

        @pandas_udf(returnType=self._return_type)
        def _apply_predict_model_internal(
            series_iter: Iterator[pd.Series],
        ) -> Iterator[pd.Series]:
            model = self._load_model()
            for series in series_iter:
                if self._pre_processor:
                    series = series.apply(self._pre_processor)
                yield model.call(series)

        return _apply_predict_model_internal

    def transform(self, df: DataFrame) -> DataFrame:
        """Transforms a DataFrame by applying the model.

        This is the main function that runs the model and appends its
        predictions as a new column to the input Dataframe.

        Args:
            df: The input Spark DataFrame.

        Returns:
            A new DataFrame with the prediction column added.

        Raises:
            ValueError: If the input or output column is not set.
        """
        if not self._input_col:
            raise ValueError("Input column must be set using .input_col().")
        if not self._output_col:
            raise ValueError("Output column must be set using .output_col().")

        return df.withColumn(
            self._output_col,
            self._create_predict_udf()(col(self._input_col)),
        )
