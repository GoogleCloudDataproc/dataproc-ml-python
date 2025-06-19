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
        """
        If single_call fails for any row, the entire Spark task will fail.

        Args:
            batch: A pandas Series containing the batch of inputs to process.

        Returns:
            A pandas Series containing the prediction results.
        """
        raise NotImplementedError


class BaseModelHandler(ABC):

    def __init__(self):
        """An abstract base class for applying a model to a Spark DataFrame.

        This handler uses the high-performance Pandas UDF (iterator of series)
        pattern to apply a model to each partition of a DataFrame. It is designed
        to be configured using a builder pattern.

        Subclasses must implement the `_load_model` method, which is responsible
        for loading the model instance on each Spark executor.

        Example usage:
            class MyModelHandler(BaseModelHandler):
                def _load_model(self):
                    return MyModel()

            handler = MyModelHandler()
            result_df = handler.input_col("features") \\
                               .output_col("predictions") \\
                               .pre_processor(my_pre_processor) \\
                               .transform(df)
        """
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
            output_col: The name for the new column that will store the predictions.

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

    # TODO: Consider applying the pre-processor on the entire row instead of single col
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

        This is the main entry point for the handler. It adds a new column
        to the input DataFrame containing the model's predictions.

        Args:
            df: The input Spark DataFrame.

        Returns:
            A new DataFrame with the prediction column added.
        """
        return df.withColumn(
            self._output_col,
            self._create_predict_udf()(col(self._input_col)),
        )
