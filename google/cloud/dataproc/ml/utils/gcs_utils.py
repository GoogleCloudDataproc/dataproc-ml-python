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

"""A module for Google Cloud Storage (GCS) utility functions."""

import urllib.parse
from io import BytesIO

from google.api_core.exceptions import NotFound
from google.cloud import storage


def _validate_and_parse_gcs_path(model_path: str):
    """Validates the GCS path format and extracts bucket and blob names.

    Args:
        model_path: The GCS path string (e.g., "gs://bucket/object").

    Returns:
        A tuple containing the bucket name and the blob name.

    Raises:
        ValueError: If the path format is invalid (e.g., wrong scheme, missing
            bucket or object name).
    """
    parsed_path = urllib.parse.urlparse(model_path)
    if parsed_path.scheme != "gs":
        raise ValueError(
            f"Unsupported path scheme: {parsed_path.scheme}. Must be 'gs://'."
        )

    bucket_name = parsed_path.netloc
    blob_name = parsed_path.path.lstrip("/")

    if not bucket_name:
        raise ValueError(
            f"Invalid GCS path: '{model_path}'. Bucket name is missing."
        )
    if not blob_name:
        raise ValueError(
            f"Invalid GCS path: '{model_path}'. Object name is missing."
        )

    return bucket_name, blob_name


def _download_gcs_blob_to_buffer(model_path: str) -> BytesIO:
    """Downloads a GCS blob into an in-memory BytesIO buffer."""
    bucket_name, blob_name = _validate_and_parse_gcs_path(model_path)
    data_buffer = BytesIO()
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_file(data_buffer)
        data_buffer.seek(0)
        return data_buffer
    except NotFound:
        raise FileNotFoundError(f"File not found at GCS path: {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download file from GCS at {model_path}. "
            f"Check permissions/path. Original error: {e}"
        )
