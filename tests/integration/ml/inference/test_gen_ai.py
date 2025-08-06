import os
import unittest

from pyspark.errors.exceptions.captured import PythonException
from pyspark.sql import SparkSession, DataFrame
from google.cloud.dataproc.ml.inference import GenAiModelHandler


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
        ).repartition(
            3
        )  # Repartitioning to have multiple rows in same pandas batch

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
        pre_processor = (
            lambda city: "What is the airport code of largest airport in "
            + city
            + "? Answer in single word."
        )
        gen_ai_handler = (
            GenAiModelHandler()
            .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
            .location(os.getenv("GOOGLE_CLOUD_REGION"))
            .input_col("city")
            .pre_processor(pre_processor)
        )

        df = gen_ai_handler.transform(self.df)
        self._assert_dataframe_equals(
            self.expected, df.select("city", "predictions")
        )

    def test_prompt_template(self):
        gen_ai_handler = (
            GenAiModelHandler()
            .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
            .location(os.getenv("GOOGLE_CLOUD_REGION"))
            .prompt(
                "What is the airport code of largest airport in {city}? Answer in single word."
            )
        )

        df = gen_ai_handler.transform(self.df)
        self._assert_dataframe_equals(
            self.expected, df.select("city", "predictions")
        )

    def test_unsupported_model_name_raises_exception(self):
        gen_ai_handler = (
            GenAiModelHandler()
            .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
            .location(os.getenv("GOOGLE_CLOUD_REGION"))
            .model("gemini-xyz-4")
            .input_col("city")
        )

        df = self.spark.createDataFrame(
            [("Bengaluru", "India")], ["city", "country"]
        )
        with self.assertRaises(PythonException) as e:
            gen_ai_handler.transform(df).collect()  # collect to force eval
        self.assertIn("NotFound: 404", str(e.exception))
        self.assertIn("gemini-xyz-4", str(e.exception))

    @staticmethod
    def _assert_dataframe_equals(df1: DataFrame, df2: DataFrame):
        assert df1.schema == df2.schema
        assert df1.exceptAll(df2).isEmpty()
        assert df2.exceptAll(df1).isEmpty()


if __name__ == "__main__":
    unittest.main()
