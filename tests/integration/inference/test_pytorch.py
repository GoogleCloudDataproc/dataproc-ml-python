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

import logging
import os
import unittest
import uuid

import pandas as pd
import numpy as np
import torch
import torchvision.models as models
from pyspark.errors.exceptions.captured import PythonException
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from torchvision.models import resnet18

from google.cloud import storage, exceptions as gcloud_exceptions
from google.cloud.dataproc_ml.inference import PyTorchModelHandler
from tests.utils.gcs_util import download_image_from_gcs
from tests.utils.pytorch_util import (
    preprocess_real_image_data as scalar_preprocess_real_image_data,
    save_pytorch_model_full_object,
    save_pytorch_model_state_dict,
)


def vectorized_preprocess_real_image_data(
    image_bytes_series: pd.Series,
) -> pd.Series:
    """Vectorized preprocessor for ResNet."""
    return image_bytes_series.apply(scalar_preprocess_real_image_data)


class TestPyTorchModelHandler(unittest.TestCase):
    TEST_GCS_BUCKET = os.getenv("TEST_GCS_BUCKET")
    SAMPLE_PUBLIC_IMAGE_GCS_PATH = (
        "gs://cloud-samples-data/vision/label/wakeupcat.jpg"
    )

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.sample_real_image_bytes = download_image_from_gcs(
            cls.SAMPLE_PUBLIC_IMAGE_GCS_PATH
        )
        cls.actual_model_resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        cls.actual_model_resnet18.eval()

        cls.imagenet_categories = models.ResNet18_Weights.DEFAULT.meta[
            "categories"
        ]

        with torch.no_grad():
            input_tensor = scalar_preprocess_real_image_data(
                cls.sample_real_image_bytes
            ).unsqueeze(
                0
            )  # Add batch dimension
            output_tensor = cls.actual_model_resnet18(input_tensor)
            predicted_idx = torch.argmax(output_tensor, dim=1).item()

            cls.expected_label = cls.imagenet_categories[predicted_idx]

    @classmethod
    def tearDownClass(cls):
        """Stop the shared SparkSession after all tests are done."""
        if hasattr(cls, "spark"):
            cls.spark.stop()

    def setUp(self):
        """
        Runs BEFORE EACH test. Creates a fresh GCS prefix and uploads
        clean copies of the models for this specific test to ensure isolation.
        """
        self.pytorch_handler = PyTorchModelHandler()
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.bucket(self.TEST_GCS_BUCKET)

        # Each test gets a unique GCS folder to prevent interference
        self.test_gcs_prefix = (
            f"dataproc-ml-inference-pytorch-test/{uuid.uuid4()}"
        )

        full_model_bytes = save_pytorch_model_full_object(
            self.actual_model_resnet18
        )
        self.gcs_full_model_path = (
            f"gs://{self.TEST_GCS_BUCKET}/"
            f"{self.test_gcs_prefix}/full_resnet18.pt"
        )
        self.gcs_bucket.blob(
            f"{self.test_gcs_prefix}/full_resnet18.pt"
        ).upload_from_string(full_model_bytes)

        state_dict_bytes = save_pytorch_model_state_dict(
            self.actual_model_resnet18
        )
        self.gcs_statedict_path = (
            f"gs://{self.TEST_GCS_BUCKET}/"
            f"{self.test_gcs_prefix}/resnet18_state_dict.pt"
        )
        self.gcs_bucket.blob(
            f"{self.test_gcs_prefix}/resnet18_state_dict.pt"
        ).upload_from_string(state_dict_bytes)

    def tearDown(self):
        """
        Runs AFTER EACH test. Deletes the GCS artifacts created for
        the test that just finished to ensure a clean slate for the next one.
        """
        blobs = self.gcs_bucket.list_blobs(prefix=self.test_gcs_prefix)
        for blob in blobs:
            try:
                blob.delete()
            except gcloud_exceptions.GoogleCloudError as e:
                logging.warning("Failed to clean up GCS artifacts: %s", e)

    def test_pytorch_full_model_object_inference(self):

        pytorch_handler = (
            self.pytorch_handler.model_path(self.gcs_full_model_path)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions_full_model_object")
            .pre_processor(vectorized_preprocess_real_image_data)
            .set_return_type(ArrayType(FloatType()))
        )

        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,) for _ in range(3)],
            ["image_bytes"],
        )

        result_df = pytorch_handler.transform(df)

        self.assertIn("predictions_full_model_object", result_df.columns)
        self.assertEqual(
            result_df.schema["predictions_full_model_object"].dataType,
            ArrayType(FloatType()),
        )
        self.assertEqual(result_df.count(), 3, "Expected 3 prediction rows.")

        all_actual_predictions_raw = result_df.select(
            "predictions_full_model_object"
        ).collect()
        # Assert label prediction for the first row (since all are same image)
        first_prediction_list = all_actual_predictions_raw[0][0]
        predicted_class_idx = np.array(first_prediction_list).argmax()
        actual_label = self.imagenet_categories[predicted_class_idx]
        self.assertEqual(
            actual_label,
            self.expected_label,
            "Predicted label mismatch for full model object.",
        )

    def test_pytorch_statedict_inference(self):
        """
        Tests inference using a model loaded from a state_dict.
        """
        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,) for _ in range(3)],
            ["image_bytes"],
        )

        pytorch_handler = (
            self.pytorch_handler.model_path(self.gcs_statedict_path)
            .device("cpu")
            # Use weights=None to initialize an empty model structure.
            .set_model_architecture(resnet18, weights=None)
            .input_cols("image_bytes")
            .output_col("predictions_statedict")
            .pre_processor(vectorized_preprocess_real_image_data)
            .set_return_type(ArrayType(FloatType()))
        )

        result_df = pytorch_handler.transform(df)

        self.assertIn("predictions_statedict", result_df.columns)
        self.assertEqual(
            result_df.schema["predictions_statedict"].dataType,
            ArrayType(FloatType()),
        )
        self.assertEqual(result_df.count(), 3, "Expected 3 prediction rows.")

        all_actual_predictions_raw = result_df.select(
            "predictions_statedict"
        ).collect()

        # It's the same model architecture, the prediction logic is identical
        first_prediction_list = all_actual_predictions_raw[0][0]
        predicted_class_idx = np.array(first_prediction_list).argmax()
        actual_label = self.imagenet_categories[predicted_class_idx]
        self.assertEqual(
            actual_label,
            self.expected_label,
            "Predicted label mismatch for state_dict load.",
        )

    def test_state_dict_no_model_architecture(self):
        """Tests loading a state_dict fails without model_architecture."""
        state_dict_path = self.gcs_statedict_path

        handler = (
            self.pytorch_handler.model_path(state_dict_path)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )
        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,)], ["image_bytes"]
        )

        with self.assertRaisesRegex(
            PythonException,
            "TypeError: The file at .* was loaded successfully, but it is "
            "not a torch.nn.Module instance.*",
        ):
            handler.transform(df).collect()

    def test_invalid_model_path_missing_components(self):
        """Tests error for incomplete GCS paths during model loading."""
        # Scenario 1: Path with only "gs://"
        path_only_scheme = "gs://"
        pytorch_handler = (
            self.pytorch_handler.model_path(path_only_scheme)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )

        df = SparkSession.builder.getOrCreate().createDataFrame(
            [(self.sample_real_image_bytes,)], ["image_bytes"]
        )

        with self.assertRaisesRegex(
            PythonException,
            "ValueError: Invalid GCS path: .*?Bucket name is missing",
        ):
            pytorch_handler.transform(df).collect()

        # Scenario 2: Path with bucket but no object
        path_no_object = f"gs://{self.TEST_GCS_BUCKET}/"
        handler_no_object = (
            self.pytorch_handler.model_path(path_no_object)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )

        with self.assertRaisesRegex(
            PythonException,
            "ValueError: Invalid GCS path: .* Object name is missing.",
        ):
            handler_no_object.transform(df).collect()

    def test_non_existent_gcs_model(self):
        """Tests error is raised when the GCS model path does not exist."""
        non_existent_path = f"gs://{self.TEST_GCS_BUCKET}/non_existent_model.pt"
        pytorch_handler = (
            self.pytorch_handler.model_path(non_existent_path)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )
        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,)], ["image_bytes"]
        )

        with self.assertRaisesRegex(
            PythonException,
            "FileNotFoundError: File not found at GCS path.*",
        ):
            pytorch_handler.transform(df).collect()

    def test_missing_model_path(self):
        """Tests that an error is raised if model_path is not set."""
        pytorch_handler = (
            self.pytorch_handler.device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )

        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,)], ["image_bytes"]
        )

        with self.assertRaisesRegex(
            PythonException, "ValueError: Model path must be set"
        ):
            pytorch_handler.transform(df).collect()

    def test_full_model_load_from_non_model_file(self):
        """Tests that loading a full model fails for a non-model file."""
        non_model_path = self.SAMPLE_PUBLIC_IMAGE_GCS_PATH

        pytorch_handler = (
            self.pytorch_handler.model_path(non_model_path)
            .device("cpu")
            .input_cols("image_bytes")
            .output_col("predictions")
            .set_return_type(ArrayType(FloatType()))
        )

        df = self.spark.createDataFrame(
            [(self.sample_real_image_bytes,)], ["image_bytes"]
        )

        with self.assertRaisesRegex(
            PythonException,
            r"RuntimeError: Failed to load .* The file may be corrupted .*"
            r"invalid load key, '\\xff'.*",
        ):
            pytorch_handler.transform(df).collect()


if __name__ == "__main__":
    unittest.main()
