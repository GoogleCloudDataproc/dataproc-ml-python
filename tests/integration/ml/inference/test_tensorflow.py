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
import shutil
import unittest
import uuid

import pandas as pd
import numpy as np
import tensorflow as tf
from pyspark.errors.exceptions.captured import PythonException
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType

from google.cloud import storage, exceptions as gcloud_exceptions
from google.cloud.dataproc.ml.inference import TensorFlowModelHandler
from tests.utils.gcs_util import (
    download_image_from_gcs,
    upload_directory_to_gcs,
)
from tests.utils.tensorflow_util import (
    preprocess_for_mobilenetv2 as scalar_preprocess_for_mobilenetv2,
    save_model_as_savedmodel,
    save_model_as_checkpoint,
)


def vectorized_preprocess_for_mobilenetv2(
    image_bytes_series: pd.Series,
) -> pd.Series:
    """Vectorized preprocessor for MobileNetV2."""
    return image_bytes_series.apply(scalar_preprocess_for_mobilenetv2)


class TestTensorFlowModelHandlerIntegration(unittest.TestCase):

    TEST_GCS_BUCKET = os.getenv("TEST_GCS_BUCKET")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    SAMPLE_PUBLIC_IMAGE_GCS_PATH = (
        "gs://cloud-samples-data/vision/label/wakeupcat.jpg"
    )

    @classmethod
    def setUpClass(cls):
        """Set up resources shared across all tests
        (Spark, model, expected answer)."""

        cls.gcs_client = storage.Client(project=cls.GOOGLE_CLOUD_PROJECT)

        cls.sample_image_bytes = download_image_from_gcs(
            cls.SAMPLE_PUBLIC_IMAGE_GCS_PATH
        )

        cls.actual_model_mobilenet = tf.keras.applications.MobileNetV2(
            weights="imagenet"
        )

        imagenet_labels_path = tf.keras.utils.get_file(
            "ImageNetLabels.txt",
            "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",  # pylint: disable=line-too-long
        )
        with open(imagenet_labels_path, "r", encoding="utf-8") as f:
            cls.imagenet_categories = f.read().splitlines()

        input_tensor = scalar_preprocess_for_mobilenetv2(cls.sample_image_bytes)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        output_tensor = cls.actual_model_mobilenet(input_tensor)
        predicted_idx = tf.argmax(output_tensor, axis=1).numpy().item()
        cls.expected_label = cls.imagenet_categories[predicted_idx]

    def tearDown(self):
        """Runs AFTER EACH test. Stops the SparkSession."""
        if self.spark:
            self.spark.stop()

    def setUp(self):
        self.spark = (
            SparkSession.builder.appName(
                f"TF_Handler_Integration_Test_{self.id()}"
            )
            .master("local[*]")
            .getOrCreate()
        )

        self.handler = TensorFlowModelHandler()
        self.gcs_bucket = self.gcs_client.bucket(self.TEST_GCS_BUCKET)
        self.test_gcs_prefix = f"dataproc-ml-tf-test/{uuid.uuid4()}"

        # To guarantee that resources are deleted even if setUp fails.
        self.addCleanup(self._cleanup_gcs_artifacts)

    def _cleanup_gcs_artifacts(self):
        """Helper function to delete GCS objects for a test."""
        try:
            blobs = self.gcs_bucket.list_blobs(prefix=self.test_gcs_prefix)
            for blob in blobs:
                blob.delete()
        except gcloud_exceptions.GoogleCloudError as e:
            logging.warning("Failed to clean up GCS artifacts: %s", e)

    def test_e2e_full_model_load_savedmodel(self):
        """Tests inference with a full SavedModel."""
        # --- Test-specific setup ---
        saved_model_path = save_model_as_savedmodel(self.actual_model_mobilenet)
        # Use the robust cleanup for local files too
        self.addCleanup(shutil.rmtree, os.path.dirname(saved_model_path))
        gcs_saved_model_path = upload_directory_to_gcs(
            self.gcs_bucket,
            saved_model_path,
            f"{self.test_gcs_prefix}/saved_model",
        )
        # --- End of setup ---

        handler = (
            self.handler.model_path(gcs_saved_model_path)
            .input_cols("image_bytes")
            .output_col("predictions")
            .pre_processor(vectorized_preprocess_for_mobilenetv2)
            .set_return_type(ArrayType(FloatType()))
        )

        df = self.spark.createDataFrame(
            [(self.sample_image_bytes,)], ["image_bytes"]
        )

        result_df = handler.transform(df)

        predictions = result_df.select("predictions").first()
        predicted_idx = np.array(predictions).argmax()
        actual_label = self.imagenet_categories[predicted_idx]
        self.assertEqual(actual_label, self.expected_label)

    def test_e2e_weights_only_load_checkpoint(self):
        """Tests inference using a model loaded from a TF Checkpoint."""

        # --- Test-specific setup ---
        local_weights_filepath = save_model_as_checkpoint(
            self.actual_model_mobilenet
        )
        local_weights_dir = os.path.dirname(local_weights_filepath)

        self.addCleanup(
            shutil.rmtree,
            local_weights_dir,  # Correctly delete the directory we created.
        )

        checkpoint_gcs_dir = upload_directory_to_gcs(
            self.gcs_bucket,
            local_weights_dir,  # Upload the directory.
            f"{self.test_gcs_prefix}/checkpoint",
        )

        gcs_weights_path = os.path.join(
            checkpoint_gcs_dir, os.path.basename(local_weights_filepath)
        )

        handler = (
            self.handler.model_path(gcs_weights_path)
            .set_model_architecture(
                tf.keras.applications.MobileNetV2, weights=None
            )
            .input_cols("image_bytes")
            .output_col("predictions")
            .pre_processor(vectorized_preprocess_for_mobilenetv2)
            .set_return_type(ArrayType(FloatType()))
        )

        df = self.spark.createDataFrame(
            [(self.sample_image_bytes,)], ["image_bytes"]
        )

        result_df = handler.transform(df)

        predictions = result_df.select("predictions").first()
        predicted_idx = np.array(predictions).argmax()
        actual_label = self.imagenet_categories[predicted_idx]

        self.assertEqual(actual_label, self.expected_label)

    def test_non_existent_gcs_path_fails(self):
        """Tests that an error is raised when the GCS path does not exist."""
        non_existent_path = (
            f"gs://{self.TEST_GCS_BUCKET}/this-path-does-not-exist/"
        )
        handler = self.handler.model_path(non_existent_path).input_cols(
            "image_bytes"
        )
        df = self.spark.createDataFrame(
            [(self.sample_image_bytes,)], ["image_bytes"]
        )

        expected_regex = (
            "RuntimeError: Failed to load the TensorFlow model from .*"
        )

        with self.assertRaisesRegex(PythonException, expected_regex):
            handler.transform(df).collect()

    def test_weights_load_without_architecture_fails(self):
        """Tests loading a checkpoint fails without a model architecture."""

        local_weights_filepath = save_model_as_checkpoint(
            self.actual_model_mobilenet
        )
        local_weights_dir = os.path.dirname(local_weights_filepath)

        self.addCleanup(shutil.rmtree, local_weights_dir)

        checkpoint_gcs_dir = upload_directory_to_gcs(
            self.gcs_bucket,
            local_weights_dir,
            f"{self.test_gcs_prefix}/checkpoint",
        )

        gcs_weights_path = os.path.join(
            checkpoint_gcs_dir, os.path.basename(local_weights_filepath)
        )

        handler = self.handler.model_path(gcs_weights_path).input_cols(
            "image_bytes"
        )

        df = self.spark.createDataFrame(
            [(self.sample_image_bytes,)], ["image_bytes"]
        )

        expected_regex = (
            "Model path .* appears to be a weights file, "
            "but the model architecture was not provided"
        )
        with self.assertRaisesRegex(PythonException, expected_regex):
            handler.transform(df).collect()


if __name__ == "__main__":
    unittest.main()
