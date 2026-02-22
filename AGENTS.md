# Instructions for Agents

This document outlines the development workflow and standards for the `repeng` repository.

## Environment Setup

This project uses `uv` for dependency management.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Testing

Tests are located in `tests.py` and potentially other files matching `test_*.py`.

1.  **Run tests**:
    ```bash
    uv run pytest
    ```

2.  **Running slow tests**:
    Some tests are marked as slow. By default, you might skip them or run them explicitly depending on your needs. Check `pyproject.toml` for markers.
    To run all tests including slow ones:
    ```bash
    uv run pytest -m "slow or not slow"
    ```
    (Or simply `uv run pytest` if no default exclusion is configured in `pyproject.toml`, but check the config.)

## Linting and Formatting

This project uses `ruff` for linting and formatting.

1.  **Check linting**:
    ```bash
    uv run ruff check .
    ```

2.  **Format code**:
    ```bash
    uv run ruff format .
    ```

## Project Structure

-   `repeng/control.py`: Contains `ControlModel` and `ControlModule` classes which wrap Hugging Face models to apply control vectors.
-   `repeng/extract.py`: Contains `ControlVector` class and `read_representations` function for training vectors.
-   `repeng/saes.py`: Contains utilities for loading and using Sparse Autoencoders (SAEs).
-   `repeng/tests.py`: Contains the test suite.

## Coding Standards

-   Follow existing code style.
-   Ensure all new features have corresponding tests.
-   Write clear docstrings for public classes and methods.
-   When modifying `ControlModel`, be aware that it mutates the underlying model. Ensure `unwrap` works correctly.
