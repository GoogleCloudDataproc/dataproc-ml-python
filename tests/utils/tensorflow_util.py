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
import tempfile
import tensorflow as tf
from google.cloud import storage
import urllib.parse


def save_model_as_savedmodel(model: tf.keras.Model) -> str:
    """Saves a Keras model in the SavedModel format to a temporary local directory."""
    temp_dir = tempfile.mkdtemp()
    saved_model_local_path = os.path.join(temp_dir, "test_saved_model")
    model.export(saved_model_local_path)
    return saved_model_local_path


def save_model_as_checkpoint(model: tf.keras.Model) -> str:
    """Saves a Keras model's weights as a TF Checkpoint to a temporary local directory."""
    temp_dir = tempfile.mkdtemp()
    weights_filepath = os.path.join(temp_dir, "model.weights.h5")
    model.save_weights(weights_filepath)
    return weights_filepath


def preprocess_for_mobilenetv2(image_bytes: bytes) -> tf.Tensor:
    """Decodes and preprocesses an image for MobileNetV2."""
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image
