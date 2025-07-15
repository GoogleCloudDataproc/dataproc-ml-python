import io

import torch
from google.cloud import storage
import urllib.parse

import torchvision.transforms as transforms
from PIL import Image


def save_pytorch_model_state_dict(model: torch.nn.Module) -> bytes:
    """Saves a PyTorch model's state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.getvalue()


def save_pytorch_model_full_object(model: torch.nn.Module) -> bytes:
    """Saves a full PyTorch model object to bytes."""
    buffer = io.BytesIO()
    # Security note: This saves the entire model object using pickle.
    # It's less portable and has security implications if loading from untrusted sources.
    torch.save(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def download_image_from_gcs(image_path: str) -> bytes:
    """Downloads image bytes from GCS."""
    parsed_path = urllib.parse.urlparse(image_path)
    if parsed_path.scheme != "gs":
        raise ValueError(
            f"Unsupported GCS path scheme: {parsed_path.scheme}. Must be 'gs://'."
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


def preprocess_real_image_data(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses real image bytes for a model like ResNet.
    Input: Raw image bytes (e.g., from a JPG file).
    Output: A PyTorch tensor suitable for a pre-trained ResNet model.
    """
    if not isinstance(image_bytes, bytes):
        raise TypeError(f"Expected image bytes, got {type(image_bytes)}")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    _resnet_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = _resnet_transform(image)
    return input_tensor
