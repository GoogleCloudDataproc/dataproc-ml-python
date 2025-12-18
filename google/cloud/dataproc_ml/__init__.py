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

"""A python library to ease MLOps for Dataproc customers"""

try:
    import pyspark
except ImportError:
    raise ImportError(
        "PySpark is not installed. The `dataproc-ml` library requires a Spark "
        "environment.\n"
        "Please install one of the following packages:\n"
        "1. For standard Spark: pip install pyspark\n"
        "2. For Spark Connect:  pip install pyspark-client"
    ) from None
