import os
import unittest

from pyspark.sql import SparkSession, DataFrame
from google.cloud.dataproc.ml.inference import GenAiModelHandler


class TestGenAiModelHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_gen_ai(self):
        pre_processor = lambda city: "Answer in single word. What is the airport code of largest airport in " + city
        gen_ai_handler = (GenAiModelHandler()
                          .project(os.getenv("GOOGLE_CLOUD_PROJECT"))
                          .location(os.getenv("GOOGLE_CLOUD_REGION"))
                          .input_col("city")
                          .output_col("predicted")
                          .pre_processor(pre_processor))
        df = self.spark.createDataFrame(
            [("Bengaluru", "India"), ("London", "UK")], ["city", "country"]
        )
        df = gen_ai_handler.transform(df)

        expected = self.spark.createDataFrame(
            [("BLR",), ("LHR",)], ["predicted"]
        )
        self._assert_dataframe_equals(expected, df.select("predicted"))

    @staticmethod
    def _assert_dataframe_equals(df1: DataFrame, df2: DataFrame):
        assert df1.schema == df2.schema
        assert df1.exceptAll(df2).isEmpty()
        assert df2.exceptAll(df1).isEmpty()


if __name__ == '__main__':
    unittest.main()
