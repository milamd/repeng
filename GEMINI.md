# repeng: Gemini Context

## Overview
`repeng` is a Python library for representation engineering (RepEng). It allows training "control vectors" that can be applied to language models (LMs) to steer their behavior (e.g., making them more honest, happy, or trippy). It works by identifying directions in the model's activation space that correspond to specific concepts and then shifting activations in that direction during inference.

## Core Concepts

### ControlVector
- Represents a learned direction in the model's activation space.
- Stores a dictionary mapping layer indices to direction vectors (numpy arrays).
- Can be trained using PCA on contrastive datasets (`ControlVector.train`) or using Sparse Autoencoders (`ControlVector.train_with_sae`).
- Supports arithmetic operations (addition, subtraction, scaling).
- Can be exported to GGUF format for use with `llama.cpp`.

### ControlModel
- A wrapper around a Hugging Face `PreTrainedModel`.
- Intercepts the forward pass to inject control vectors into the hidden states.
- **Mutates the underlying model** (replaces layers with `ControlModule`).
- Use `model.set_control(vector, strength)` to apply a vector.
- Use `model.reset()` or `model.set_control(None)` to remove control.
- `model.unwrap()` restores the original model structure.

### DatasetEntry
- A dataclass `DatasetEntry(positive: str, negative: str)`.
- Represents a pair of contrasting prompts (e.g., "Act happy", "Act sad") used to define the concept for the control vector.

### SAE (Sparse Autoencoder)
- Represents a loaded Sparse Autoencoder.
- Used in `ControlVector.train_with_sae` to potentially find cleaner directions by leveraging the sparse features of the SAE.
- Loaded via `repeng.saes.from_eleuther`.

## Key Modules

- **`repeng.control`**: Contains `ControlModel` and `ControlModule`. Handles the runtime application of control vectors.
- **`repeng.extract`**: Contains `ControlVector` and `read_representations`. Handles the training logic (PCA, etc.) and dataset processing.
- **`repeng.saes`**: Utilities for loading SAEs (currently supports EleutherAI's format).

## Usage Patterns

### Standard Training
```python
from repeng import ControlVector, DatasetEntry

dataset = [
    DatasetEntry(positive="Act happy", negative="Act sad"),
    # ... more entries
]
vector = ControlVector.train(model, tokenizer, dataset)
```

### SAE Training
```python
from repeng import ControlVector
from repeng.saes import from_eleuther

sae = from_eleuther("EleutherAI/sae-gpt2-small-hook-z-32k")
vector = ControlVector.train_with_sae(model, tokenizer, sae, dataset)
```

### Inference
```python
from repeng import ControlModel

model = ControlModel(base_model, layer_ids=list(range(-1, -10, -1)))
model.set_control(vector, coeff=1.5)
output = model.generate(**inputs)
model.reset() # Important to reset after use if reusing the model instance
```

### GGUF Export
```python
vector.export_gguf("vector.gguf")
```
