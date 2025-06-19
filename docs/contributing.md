# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

### Setup
1. It is recommended to do development in a separate virtual environment

    ```shell
    python3.11 -m venv <your env>
    ```

2. Install all the build, dev and test dependencies

    ```shell
    pip install ".[dev, test]"
    ```

3. Configure your IDE to the same venv interpreter

### Code Reviews
1. Do run the linter and make sure all the tests pass before raising the PR
    ```shell
    pyink .
    pytest .
    ```
2. Add the output of the test run in the PR description

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.
