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
import logging
from abc import ABC
from typing import Any, Callable, Iterator, Tuple

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DataType, StructType

logger = logging.getLogger(__name__)


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
        ...     handler.input_cols("features")
        ...     .output_col("predictions")
        ...     .pre_processor(my_pre_processor)
        ...     .transform(df)
        ... )
    """

    def __init__(self):
        self._input_cols: Tuple[str, ...] = None
        self._output_col: str = "predictions"
        self._return_type: DataType = StructType()
        self._pre_processor: Callable[..., pd.Series] = None

    def _load_model(self) -> Model:
        """Loads the model instance.

        This method is called once per Spark task (on the executor) to
        initialize the model. Subclasses must implement this method.

        Returns:
            An instance of a class that inherits from `Model`.
        """
        raise NotImplementedError

    def input_cols(self, *input_cols: str) -> "BaseModelHandler":
        """Sets the names of the input columns from the DataFrame.

        Args:
            *input_cols: The names of the columns to be used as input for the
                model, passed as multiple string arguments (e.g.,
                `.input_cols('col1', 'col2')`).

        Returns:
            The handler instance for method chaining.

        Raises:
            ValueError: If no input columns are provided.
        """
        if not input_cols:
            raise ValueError("At least one input column must be provided.")
        self._input_cols = input_cols
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
        self, pre_processor: Callable[..., pd.Series]
    ) -> "BaseModelHandler":
        """Sets the vectorized preprocessing function.

        The function is applied to the input columns for each batch,
        return of which is then applied to the model.
        It should accept one or more pandas Series as
        input (matching the number of input columns) in same order as
        `input_cols` and return a single pandas Series.

        Example for single input:
            def my_preprocessor(col1: pd.Series) -> pd.Series:
                return col1 * 2

        Example for multiple inputs:
            def my_preprocessor(col1: pd.Series, col2: pd.Series) -> pd.Series:
                return col1 + " delimeter " + col2

        Args:
            pre_processor: A vectorized function that takes one or more pandas
                Series and returns a single pandas Series.

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

        def _apply_predict_model_internal(
            series_iter: Iterator[Any],
        ) -> Iterator[pd.Series]:
            """The internal UDF logic that runs on Spark executors."""
            model = self._load_model()
            for batch_input in series_iter:
                # PySpark's Pandas UDF behavior:
                # - For a single input column, it yields the pd.Series directly.
                # - For multiple input columns, it yields a tuple of pd.Series.
                # We normalize this to always be a tuple for preprocessor.
                if isinstance(batch_input, pd.Series):
                    series_tuple = (batch_input,)
                else:
                    series_tuple = batch_input

                if self._pre_processor:
                    # The pre_processor combines columns.
                    processed_series = self._pre_processor(*series_tuple)
                else:
                    # If no pre_processor, we can only handle a single column.
                    processed_series = series_tuple[0]
                yield model.call(processed_series)

        # Set the correct type hint for the UDF based on the number of input
        # columns, which is required by PySpark's Pandas UDF implementation.
        if len(self._input_cols) > 1:
            _apply_predict_model_internal.__annotations__["series_iter"] = (
                Iterator[Tuple[pd.Series, ...]]
            )
        else:
            _apply_predict_model_internal.__annotations__["series_iter"] = (
                Iterator[pd.Series]
            )

        return pandas_udf(
            _apply_predict_model_internal, returnType=self._return_type
        )

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
        if not self._input_cols:
            raise ValueError("Input column(s) must be set using .input_cols().")
        if not self._output_col:
            raise ValueError("Output column must be set using .output_col().")

        if len(self._input_cols) > 1 and not self._pre_processor:
            raise ValueError(
                "A pre_processor must be provided when using multiple input "
                "columns to combine them into a single series for the model "
                "input."
            )

        return df.withColumn(
            self._output_col,
            self._create_predict_udf()(*[col(c) for c in self._input_cols]),
        )
