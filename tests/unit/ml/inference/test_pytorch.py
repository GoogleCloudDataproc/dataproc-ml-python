import unittest

from google.cloud.dataproc.ml.inference import PyTorchModelHandler


class TestPyTorchModelHandler(unittest.TestCase):

    def setUp(self):
        """Create a fresh PyTorchModelHandler and sample DataFrame for each test."""
        self.pytorch_handler = PyTorchModelHandler()

    def test_invalid_model_path_scheme(self):
        """Tests that model_path() builder method raises ValueError for unsupported scheme."""
        invalid_path = "http://example.com/model.pt"

        # This error is raised directly by the builder method itself, before Spark is involved.
        with self.assertRaisesRegex(
            ValueError, "Model path must start with 'gs://'"
        ):
            self.pytorch_handler.model_path(invalid_path)

    def test_invalid_device_string(self):
        """Tests that an error is raised for an invalid device string directly at the builder step."""
        invalid_device = "tpu"

        with self.assertRaisesRegex(
            ValueError, "Device must be 'cpu' or 'cuda'."
        ):
            (
                self.pytorch_handler.model_path(
                    "gs://sample-gcs-dir/model.pt"
                ).device(invalid_device)
            )


if __name__ == "__main__":
    unittest.main()
