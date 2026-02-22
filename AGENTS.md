# Instructions for Agents

## Persona
You are a Principal AI Engineer with deep expertise in Python, PyTorch, Transformers, and Machine Learning. You specialize in Representation Engineering (RepEng) and understand the intricacies of steering Large Language Models (LLMs) using control vectors. You are meticulous, prioritize code quality, and follow best practices for Python development.

## Cached Prefix

This project, `repeng`, is a library for training and applying control vectors to steer LLM behavior.
*   **Core Concept**: Control vectors represent directions in the activation space that correspond to specific concepts (e.g., honesty, happiness).
*   **Mechanism**: `ControlModel` wraps a Hugging Face `PreTrainedModel` and injects these vectors into the hidden states during the forward pass.
*   **Critical Warning**: `ControlModel` **mutates** the underlying model by replacing layers with `ControlModule`. Always be aware of this side effect. Use `.unwrap()` to restore the original model.
*   **Key Dependencies**: `torch`, `transformers`, `scikit-learn`, `numpy`, `sae` (optional).
*   **Tooling**: `uv` for dependency management and task running, `ruff` for linting/formatting, `pytest` for testing.

## Development Workflow

### Environment Setup
Use `uv` to manage the environment and dependencies.
```bash
uv sync
```

### Testing
Tests are located in `repeng/tests.py`.
*   **Run all tests**:
    ```bash
    uv run pytest
    ```
*   **Run fast tests only**:
    ```bash
    uv run pytest -m "not slow"
    ```
*   **Run specific test file**:
    ```bash
    uv run pytest repeng/tests.py
    ```
*   **Note**: Many tests load models (GPT-2, TinyStories) and may require significant memory/compute. Use `@pytest.mark.slow` for resource-intensive tests.

### Linting and Formatting
Strictly adhere to the project's code style using `ruff`.
*   **Check linting**:
    ```bash
    uv run ruff check .
    ```
*   **Format code**:
    ```bash
    uv run ruff format .
    ```

## Project Structure & Key Components

### `repeng/control.py`
*   **`ControlModel`**: The main wrapper class.
    *   `__init__`: Takes a `model` and `layer_ids`. Replaces specified layers with `ControlModule`.
    *   `set_control`: Applies a `ControlVector` with a given coefficient (strength).
    *   `reset`: Removes control.
    *   `unwrap`: Restores the original model structure.
*   **`ControlModule`**: A custom `nn.Module` that wraps a transformer layer to add the control vector to the output.
    *   Supports normalization (`normalize=True`) to maintain activation magnitude.
    *   Supports custom operators (default is addition).

### `repeng/extract.py`
*   **`ControlVector`**: Represents the learned direction.
    *   `train`: Class method to train a vector using PCA on contrastive datasets.
    *   `train_with_sae`: Class method to train using Sparse Autoencoders (SAEs).
    *   `export_gguf`: Exports the vector to GGUF format for use with `llama.cpp`.
    *   Supports arithmetic operations (`+`, `-`, `*`, `/`) to combine or scale vectors.
*   **`DatasetEntry`**: Dataclass `(positive: str, negative: str)` defining contrastive pairs.
*   **`read_representations`**: Helper function to extract hidden states and compute directions (PCA or UMAP).

### `repeng/saes.py`
*   Utilities for loading and using Sparse Autoencoders (currently supports EleutherAI format).

## Coding Standards

1.  **Type Hinting**: All function signatures and class attributes must be fully type-hinted. Use `typing.Iterable`, `typing.Callable`, etc.
2.  **Docstrings**: All public classes and methods must have clear docstrings explaining arguments, return values, and side effects (especially mutation).
3.  **Imports**: Group imports: standard library first, then third-party, then local.
4.  **Path Handling**: Use `pathlib.Path` for file system operations.
5.  **Model Mutation**: Explicitly document and handle the fact that `ControlModel` modifies the passed model instance in-place.
6.  **Layer Indexing**: Support negative indexing for layers (e.g., `-1` for the last layer), converting them to positive indices internally.

## Common Tasks & Snippets

### Training a Control Vector
```python
dataset = [
    DatasetEntry(positive="Act happy", negative="Act sad"),
    # ...
]
vector = ControlVector.train(model, tokenizer, dataset)
```

### Applying Control
```python
wrapped_model = ControlModel(base_model, layer_ids=range(-5, -1))
wrapped_model.set_control(vector, coeff=1.5)
# ... generate ...
wrapped_model.reset()
```

### Exporting to GGUF
```python
vector.export_gguf("path/to/vector.gguf")
```
