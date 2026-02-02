# 🤝 Contributing Guide - Digit Recognition

Terima kasih atas ketertarikan Anda untuk berkontribusi pada proyek Digit Recognition! Dokumen ini berisi panduan untuk berkontribusi dengan efektif.

---

## 📋 Daftar Isi

1. [Code of Conduct](#1-code-of-conduct)
2. [Getting Started](#2-getting-started)
3. [Development Workflow](#3-development-workflow)
4. [Coding Standards](#4-coding-standards)
5. [Testing Guidelines](#5-testing-guidelines)
6. [Pull Request Process](#6-pull-request-process)
7. [Issue Guidelines](#7-issue-guidelines)

---

## 1. Code of Conduct

### 1.1 Prinsip Dasar

Kami berkomitmen untuk menciptakan lingkungan yang:

- **Inklusif** - Semua kontributor diterima tanpa memandang latar belakang
- **Respectful** - Hormati pendapat dan perspektif berbeda
- **Collaborative** - Bekerja sama untuk kemajuan proyek
- **Constructive** - Berikan kritik yang membangun

### 1.2 Perilaku yang Diharapkan

✅ **Do:**

- Gunakan bahasa yang sopan dan inklusif
- Terima feedback dengan positif
- Fokus pada apa yang terbaik untuk proyek
- Tunjukkan empati kepada sesama kontributor

❌ **Don't:**

- Menggunakan bahasa yang menyinggung
- Trolling atau komentar merendahkan
- Serangan personal atau politik
- Harassment dalam bentuk apapun

---

## 2. Getting Started

### 2.1 Prerequisites

```bash
# Pastikan terinstall:
- Python 3.10+
- Git
- pip atau conda
```

### 2.2 Fork & Clone

```bash
# 1. Fork repository di GitHub

# 2. Clone fork Anda
git clone https://github.com/YOUR_USERNAME/digit-recognition.git
cd digit-recognition

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/digit-recognition.git

# 4. Verify remotes
git remote -v
# origin    https://github.com/YOUR_USERNAME/digit-recognition.git (fetch)
# origin    https://github.com/YOUR_USERNAME/digit-recognition.git (push)
# upstream  https://github.com/ORIGINAL_OWNER/digit-recognition.git (fetch)
# upstream  https://github.com/ORIGINAL_OWNER/digit-recognition.git (push)
```

### 2.3 Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
python verify_setup.py
pytest
```

### 2.4 Project Structure Overview

```
digit-recognition/
├── core/               # Neural network core
│   ├── layers.py       # Layer implementations
│   ├── activations.py  # Activation functions
│   ├── loss.py         # Loss functions
│   ├── optimizers.py   # Optimizers
│   ├── network.py      # Network class
│   └── trainer.py      # Training logic
├── data/               # Data handling
│   └── mnist_loader.py
├── gui/                # GUI components
│   ├── main_window.py
│   └── canvas.py
├── preprocessing/      # Image preprocessing
│   └── pipeline.py
├── tests/              # Test files
│   ├── unit/
│   ├── integration/
│   └── conftest.py
└── docs/               # Documentation
```

---

## 3. Development Workflow

### 3.1 Branching Strategy

```
main            ─────────────────────────────────────────────►
                     │                   │
                     │                   │
feature/xxx    ──────┴───────────────────┴──►
                     │
                     │
bugfix/yyy     ──────┴───────────────────────►
```

**Branch Naming Convention:**

- `feature/` - Fitur baru (e.g., `feature/add-dropout-layer`)
- `bugfix/` - Perbaikan bug (e.g., `bugfix/fix-softmax-overflow`)
- `docs/` - Update dokumentasi (e.g., `docs/update-readme`)
- `refactor/` - Refactoring kode (e.g., `refactor/optimize-forward-pass`)
- `test/` - Penambahan test (e.g., `test/add-integration-tests`)

### 3.2 Making Changes

```bash
# 1. Sync dengan upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Buat branch baru
git checkout -b feature/your-feature-name

# 3. Buat perubahan
# ... edit files ...

# 4. Stage changes
git add -A

# 5. Commit (ikuti commit message convention)
git commit -m "feat: add dropout layer implementation"

# 6. Push ke fork
git push origin feature/your-feature-name

# 7. Buka Pull Request di GitHub
```

### 3.3 Commit Message Convention

Kami menggunakan [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | Fitur baru |
| `fix` | Bug fix |
| `docs` | Dokumentasi |
| `style` | Formatting (tidak mengubah logic) |
| `refactor` | Refactoring |
| `test` | Penambahan test |
| `chore` | Maintenance tasks |
| `perf` | Performance improvement |

**Examples:**

```bash
# Feature
git commit -m "feat(layers): add dropout layer with configurable rate"

# Bug fix
git commit -m "fix(softmax): prevent numerical overflow for large inputs"

# Documentation
git commit -m "docs: update installation instructions for Windows"

# Test
git commit -m "test(optimizers): add unit tests for Adam optimizer"
```

---

## 4. Coding Standards

### 4.1 Python Style Guide

Kami mengikuti [PEP 8](https://pep8.org/) dengan beberapa tambahan:

```python
# ✅ Good
class DenseLayer:
    """
    Fully connected neural network layer.

    Parameters
    ----------
    input_size : int
        Number of input features
    output_size : int
        Number of output features
    initializer : str, optional
        Weight initialization method ('he' or 'xavier')

    Attributes
    ----------
    weights : np.ndarray
        Weight matrix of shape (input_size, output_size)
    bias : np.ndarray
        Bias vector of shape (1, output_size)

    Examples
    --------
    >>> layer = DenseLayer(784, 128, initializer='he')
    >>> output = layer.forward(np.random.randn(32, 784))
    >>> output.shape
    (32, 128)
    """

    def __init__(self, input_size: int, output_size: int,
                 initializer: str = 'he') -> None:
        self.input_size = input_size
        self.output_size = output_size
        self._initialize_weights(initializer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, input_size)

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, output_size)
        """
        self._input_cache = x
        return x @ self.weights + self.bias
```

### 4.2 Formatting Rules

```python
# Line length: max 88 characters (Black default)

# Imports: organized by category
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from core.layers import DenseLayer
from core.activations import ReLU

# Function/method signatures: parameters on separate lines if too long
def train(
    network: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, List[float]]:
    ...
```

### 4.3 Type Hints

Gunakan type hints untuk semua public functions:

```python
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

# Simple types
def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    ...

# Collections
def get_batch_indices(n_samples: int, batch_size: int) -> List[np.ndarray]:
    ...

# Optional parameters
def load_model(path: str, device: Optional[str] = None) -> NeuralNetwork:
    ...

# Complex types
def train(
    data: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Union[int, float, str]],
) -> Dict[str, List[float]]:
    ...
```

### 4.4 Docstrings

Gunakan NumPy style docstrings:

```python
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray,
                       epsilon: float = 1e-10) -> float:
    """
    Calculate cross-entropy loss for multi-class classification.

    Cross-entropy measures the difference between two probability
    distributions. For classification, it compares predicted probabilities
    with one-hot encoded true labels.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded true labels of shape (batch_size, n_classes)
    y_pred : np.ndarray
        Predicted probabilities of shape (batch_size, n_classes)
        Values should be in range (0, 1) and sum to 1 per row
    epsilon : float, optional
        Small value for numerical stability (default: 1e-10)

    Returns
    -------
    float
        Average cross-entropy loss over the batch

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes

    Examples
    --------
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])
    >>> y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    >>> loss = cross_entropy_loss(y_true, y_pred)
    >>> loss < 0.5
    True

    Notes
    -----
    The formula used is:

    .. math::
        L = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{C} y_{ij} \\log(\\hat{y}_{ij})

    where N is batch size, C is number of classes.

    See Also
    --------
    mse_loss : Mean squared error loss
    binary_cross_entropy : Binary classification loss
    """
    ...
```

---

## 5. Testing Guidelines

### 5.1 Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_loss.py
│   └── test_optimizers.py
├── integration/
│   ├── test_training.py
│   └── test_prediction.py
└── e2e/
    └── test_full_workflow.py
```

### 5.2 Writing Tests

```python
# tests/unit/test_layers.py

import pytest
import numpy as np
from core.layers import DenseLayer

class TestDenseLayer:
    """Tests for DenseLayer class"""

    @pytest.fixture
    def layer(self):
        """Create a standard layer for testing"""
        np.random.seed(42)
        return DenseLayer(4, 3)

    def test_initialization_shapes(self, layer):
        """Test that weights and bias have correct shapes"""
        assert layer.weights.shape == (4, 3)
        assert layer.bias.shape == (1, 3)

    def test_forward_shape(self, layer):
        """Test forward pass output shape"""
        x = np.random.randn(10, 4)
        output = layer.forward(x)
        assert output.shape == (10, 3)

    def test_forward_computation(self):
        """Test forward pass mathematical correctness"""
        layer = DenseLayer(2, 2)
        layer.weights = np.array([[1, 0], [0, 1]], dtype=float)
        layer.bias = np.zeros((1, 2))

        x = np.array([[2, 3]])
        output = layer.forward(x)

        expected = np.array([[2, 3]])
        np.testing.assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_different_batch_sizes(self, layer, batch_size):
        """Test layer works with various batch sizes"""
        x = np.random.randn(batch_size, 4)
        output = layer.forward(x)
        assert output.shape == (batch_size, 3)

    def test_backward_gradient_shape(self, layer):
        """Test backward pass returns correct gradient shape"""
        x = np.random.randn(10, 4)
        layer.forward(x)

        grad = np.random.randn(10, 3)
        grad_input = layer.backward(grad)

        assert grad_input.shape == (10, 4)
        assert layer.dW.shape == (4, 3)
        assert layer.db.shape == (1, 3)
```

### 5.3 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/unit/test_layers.py

# Run specific test
pytest tests/unit/test_layers.py::TestDenseLayer::test_forward_shape

# Run with coverage
pytest --cov=core --cov-report=html

# Run only fast tests (skip slow markers)
pytest -m "not slow"
```

### 5.4 Test Requirements

Sebelum membuat PR, pastikan:

- [ ] Semua test lama tetap pass
- [ ] Test baru ditambahkan untuk fitur/fix baru
- [ ] Coverage tidak menurun (target ≥80%)
- [ ] Tidak ada test yang di-skip tanpa alasan

---

## 6. Pull Request Process

### 6.1 Before Creating PR

Checklist sebelum membuat PR:

- [ ] Kode mengikuti coding standards
- [ ] Semua tests pass (`pytest`)
- [ ] Linter pass (`flake8`)
- [ ] Format benar (`black --check .`)
- [ ] Type hints ditambahkan
- [ ] Docstrings lengkap
- [ ] CHANGELOG diupdate (jika applicable)

### 6.2 PR Template

```markdown
## Description

[Describe your changes]

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issue

Fixes #(issue number)

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Code commented where necessary
- [ ] Documentation updated
- [ ] No new warnings

## Screenshots (if applicable)

[Add screenshots for UI changes]
```

### 6.3 PR Review Process

```
┌─────────────────┐
│  Create PR      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  CI Checks      │────►│  Review Request │
│  (tests, lint)  │     │  (2 reviewers)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    ┌────┴────┐           ┌─────────────────┐
    │ Failed? │──Yes─────►│  Fix Issues     │
    └────┬────┘           └─────────────────┘
         │No                     │
         ▼                       │
┌─────────────────┐             │
│  Approved?      │◄────────────┘
└────────┬────────┘
         │Yes
         ▼
┌─────────────────┐
│  Merge to main  │
└─────────────────┘
```

### 6.4 Code Review Guidelines

**Untuk Reviewers:**

- Review dalam 2 hari kerja
- Berikan feedback yang konstruktif
- Approve jika memenuhi standar
- Request changes dengan penjelasan jelas

**Untuk Contributors:**

- Respond ke feedback dengan cepat
- Diskusikan jika tidak setuju
- Update PR sesuai feedback

---

## 7. Issue Guidelines

### 7.1 Bug Reports

Template untuk bug report:

```markdown
## Bug Description

[Clear description of the bug]

## Steps to Reproduce

1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior

[What you expected to happen]

## Actual Behavior

[What actually happened]

## Environment

- OS: [e.g., Windows 11]
- Python: [e.g., 3.11.0]
- Package versions: [pip list]

## Screenshots/Logs

[If applicable]

## Additional Context

[Any other relevant information]
```

### 7.2 Feature Requests

Template untuk feature request:

```markdown
## Feature Description

[Clear description of the feature]

## Use Case

[Why this feature is needed]

## Proposed Solution

[How you think it should work]

## Alternatives Considered

[Other solutions you've considered]

## Additional Context

[Any other relevant information]
```

### 7.3 Labels

| Label              | Description                |
| ------------------ | -------------------------- |
| `bug`              | Something isn't working    |
| `feature`          | New feature request        |
| `docs`             | Documentation improvements |
| `good first issue` | Good for newcomers         |
| `help wanted`      | Extra attention needed     |
| `priority: high`   | High priority issue        |
| `priority: low`    | Low priority issue         |

---

## 🙏 Thank You!

Terima kasih telah meluangkan waktu untuk berkontribusi pada proyek Digit Recognition. Setiap kontribusi, sekecil apapun, sangat berarti!

Jika ada pertanyaan, jangan ragu untuk:

- Membuka issue dengan label `question`
- Menghubungi maintainers

Happy coding! 🚀
