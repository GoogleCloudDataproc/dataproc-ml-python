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

import unittest

from google.cloud.dataproc.ml.inference import PyTorchModelHandler


class TestPyTorchModelHandler(unittest.TestCase):

    def setUp(self):
        """Create a fresh PyTorchModelHandler for each test."""
        self.pytorch_handler = PyTorchModelHandler()

    def test_invalid_model_path_scheme(self):
        """Tests model_path() raises ValueError for unsupported scheme."""
        invalid_path = "http://example.com/model.pt"

        # This error is raised by the builder method, before Spark is involved.
        with self.assertRaisesRegex(
            ValueError, "Model path must start with 'gs://'"
        ):
            self.pytorch_handler.model_path(invalid_path)

    def test_invalid_device_string(self):
        """Tests an error is raised for an invalid device string."""
        invalid_device = "tpu"

        with self.assertRaisesRegex(
            ValueError, "Device must be 'cpu' or 'cuda'."
        ):
            self.pytorch_handler.model_path(
                "gs://sample-gcs-dir/model.pt"
            ).device(invalid_device)


if __name__ == "__main__":
    unittest.main()
