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

from google.cloud import storage
import urllib.parse
from google.cloud.storage import Bucket
import os


def download_image_from_gcs(image_path: str) -> bytes:
    """Downloads image bytes from GCS."""
    parsed_path = urllib.parse.urlparse(image_path)
    if parsed_path.scheme != "gs":
        raise ValueError(
            f"Unsupported GCS path scheme: {parsed_path.scheme}. "
            "Must be 'gs://'."
        )

    bucket_name = parsed_path.netloc
    blob_name = parsed_path.path.lstrip("/")

    if not bucket_name or not blob_name:
        raise ValueError(
            f"Invalid GCS path: '{image_path}'. Must be gs://bucket/object."
        )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    image_bytes = blob.download_as_bytes()

    return image_bytes


def upload_directory_to_gcs(
    bucket: Bucket, local_path: str, gcs_prefix: str
) -> str:
    """Helper to upload a local directory to a GCS bucket prefix."""
    for root, _, files in os.walk(local_path):
        for filename in files:
            local_file = os.path.join(root, filename)
            blob_name = os.path.join(
                gcs_prefix, os.path.relpath(local_file, local_path)
            )
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
    return f"gs://{bucket.name}/{gcs_prefix}/"
