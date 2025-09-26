<div align="center">
  <img src="https://cloud.google.com/images/social-icon-google-cloud-1200-630.png" width="120" alt="Google Cloud logo">
  <h1>Dataproc ML</h1>
</div>

[![PyPI version](https://img.shields.io/pypi/v/dataproc-ml)](https://pypi.org/project/dataproc-ml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Public Preview Disclaimer**
>
> Interfaces and functionality are subject to change. It is not recommended for production-critical applications without thorough testing and understanding of the potential risks.

`Dataproc ML` is a Python library that simplifies distributed ML inference on Google Cloud Dataproc. It provides high-level handlers to run PyTorch and Vertex AI Gemini models at scale using Apache Spark, without the complexity of manual model distribution and batch processing.

## Installation

You can install the library using pip:

```bash
pip install dataproc-ml
```

## Usage Examples

Here are a couple of examples demonstrating how to use the handlers for distributed inference on a Spark DataFrame.

### Generative AI (Gemini) Model Inference

> **Note:** Using the `GenAiModelHandler` involves making API calls to 
> Vertex AI, which will incur costs. Please review the [Vertex AI Generative 
> AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).

Use Google's Gemini models to perform generative tasks on your data.
This example uses a prompt template to ask for the capital of countries listed in a Spark DataFrame.

```python
from pyspark.sql import SparkSession
from google.cloud.dataproc_ml.inference import GenAiModelHandler

spark = SparkSession.builder.getOrCreate()

# Create a sample DataFrame
data = [("USA",), ("France",), ("Japan",)]
input_df = spark.createDataFrame(data, ["country"])

# The handler will automatically use the 'country' column
result_df = (
    GenAiModelHandler()
    .prompt("What is the capital of {country}?")
    .output_col("capital_city")
    .transform(input_df)
)

result_df.show()
# +-------+----------------+
# |country|capital_city    |
# +-------+----------------+
# |USA    |Washington, D.C.|
# |France |Paris           |
# |Japan  |Tokyo           |
# +-------+----------------+
```

### PyTorch Model Inference

Run distributed inference using a pre-trained PyTorch model stored in Google Cloud Storage.
This example assumes you have a Spark DataFrame `input_df` with a column named `features` containing image tensors or other numerical data.

```python
from pyspark.sql import SparkSession
from google.cloud.dataproc_ml.inference import PyTorchModelHandler

spark = SparkSession.builder.getOrCreate()

data = [([0.1, 0.2, 0.3],), ([0.4, 0.5, 0.6],), ([0.7, 0.8, 0.9],)]
input_df = spark.createDataFrame(data, ["features"])

# Path to your saved PyTorch model in GCS
model_gcs_path = "gs://your-bucket/path/to/model.pt"

# Apply the model for inference
result_df = (
    PyTorchModelHandler()
    .model_path(model_gcs_path)
    .input_cols("features")
    .transform(input_df)
)

result_df.show()
# +------------------+--------------------+
# |          features|         predictions|
# +------------------+--------------------+
# |[0.1, 0.2, 0.3]   |[0.543, 0.457]      |
# |[0.4, 0.5, 0.6]   |[0.621, 0.379]      |
# |[0.7, 0.8, 0.9]   |[0.789, 0.211]      |
# +------------------+--------------------+
```

## Documentation

For more detailed information on the available handlers and their configurations,
please refer to our official [documentation](https://dataproc-ml.readthedocs.io/).

## Contributing

Contributions are welcome! Please see contributing.md for details on how to 
set up your development environment, run linters/tests, etc.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
