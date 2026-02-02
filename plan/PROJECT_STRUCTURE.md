# 📁 Project Structure - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. Overview

Dokumen ini menjelaskan struktur folder dan file untuk project Digit Recognition. Struktur dirancang dengan prinsip **separation of concerns**, **modularity**, dan **scalability**.

---

## 2. Root Directory Structure

```
digit-recognition/
│
├── 📁 src/                    # Source code utama
│   ├── 📁 core/               # Neural network core
│   ├── 📁 gui/                # GUI components
│   ├── 📁 preprocessing/      # Image preprocessing
│   ├── 📁 training/           # Training utilities
│   ├── 📁 utils/              # Utility functions
│   └── __init__.py
│
├── 📁 data/                   # Dataset storage
│   ├── 📁 mnist/              # MNIST dataset
│   ├── 📁 custom/             # Custom images
│   └── 📁 augmented/          # Augmented data
│
├── 📁 models/                 # Saved model weights
│   ├── default.npz            # Default trained model
│   └── checkpoints/           # Training checkpoints
│
├── 📁 config/                 # Configuration files
│   ├── default.json           # Default settings
│   ├── training.json          # Training hyperparameters
│   └── gui.json               # GUI settings
│
├── 📁 tests/                  # Test files
│   ├── 📁 unit/               # Unit tests
│   ├── 📁 integration/        # Integration tests
│   └── conftest.py            # Pytest fixtures
│
├── 📁 docs/                   # Documentation
│   ├── 📁 api/                # API documentation
│   ├── 📁 guides/             # User guides
│   └── 📁 images/             # Documentation images
│
├── 📁 scripts/                # Utility scripts
│   ├── download_mnist.py      # Download MNIST dataset
│   ├── train_model.py         # Training script
│   └── evaluate_model.py      # Evaluation script
│
├── 📁 assets/                 # Static assets
│   ├── 📁 icons/              # Application icons
│   └── 📁 fonts/              # Custom fonts
│
├── main.py                    # Main entry point (GUI)
├── train.py                   # Training entry point
├── predict.py                 # CLI prediction tool
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package setup
├── pyproject.toml             # Project configuration
├── .gitignore                 # Git ignore rules
├── LICENSE                    # License file
└── README.md                  # Project readme
```

---

## 3. Source Code Structure (`src/`)

### 3.1 Core Module (`src/core/`)

Neural network implementation dari scratch.

```
src/core/
├── __init__.py                # Module exports
├── network.py                 # NeuralNetwork class
├── layers.py                  # Layer implementations
│   ├── Layer (ABC)            # Abstract base class
│   ├── DenseLayer             # Fully connected layer
│   ├── ActivationLayer        # Activation wrapper
│   └── DropoutLayer           # Dropout regularization
├── activations.py             # Activation functions
│   ├── sigmoid()
│   ├── relu()
│   ├── leaky_relu()
│   ├── softmax()
│   └── derivatives
├── losses.py                  # Loss functions
│   ├── CrossEntropyLoss
│   ├── MSELoss
│   └── BinaryCrossEntropy
├── optimizers.py              # Optimization algorithms
│   ├── SGD
│   ├── SGDMomentum
│   ├── Adam
│   └── RMSprop
├── initializers.py            # Weight initialization
│   ├── xavier_init()
│   ├── he_init()
│   └── random_init()
├── regularizers.py            # Regularization
│   ├── L1Regularizer
│   ├── L2Regularizer
│   └── ElasticNet
└── metrics.py                 # Evaluation metrics
    ├── accuracy()
    ├── precision()
    ├── recall()
    ├── f1_score()
    └── confusion_matrix()
```

**Key Files:**

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `network.py` | Main network container | `NeuralNetwork`, `NetworkBuilder` |
| `layers.py` | Layer implementations | `DenseLayer`, `Layer` (ABC) |
| `activations.py` | Activation functions | `relu()`, `softmax()`, `sigmoid()` |
| `losses.py` | Loss computation | `CrossEntropyLoss`, `MSELoss` |
| `optimizers.py` | Weight optimization | `Adam`, `SGD`, `SGDMomentum` |

### 3.2 Preprocessing Module (`src/preprocessing/`)

Image preprocessing pipeline.

```
src/preprocessing/
├── __init__.py
├── pipeline.py                # Main preprocessing pipeline
│   └── PreprocessingPipeline
├── transforms.py              # Image transformations
│   ├── resize()
│   ├── normalize()
│   ├── center_digit()
│   ├── invert_colors()
│   └── flatten()
├── augmentation.py            # Data augmentation
│   ├── rotate()
│   ├── scale()
│   ├── translate()
│   ├── add_noise()
│   └── elastic_distortion()
└── canvas_capture.py          # Canvas to image conversion
    └── capture_canvas()
```

**Pipeline Flow:**

```
Raw Image
    │
    ▼
┌─────────────────┐
│ Convert to      │
│ Grayscale       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Resize to       │
│ 28x28 pixels    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Invert Colors   │
│ (if needed)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Center Digit    │
│ (center of mass)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize       │
│ [0, 1] range    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Flatten to      │
│ 784 vector      │
└────────┬────────┘
         │
         ▼
    Processed Vector (784,)
```

### 3.3 GUI Module (`src/gui/`)

Graphical user interface components.

```
src/gui/
├── __init__.py
├── app.py                     # Main application window
│   └── DigitRecognitionApp
├── views/
│   ├── __init__.py
│   ├── main_view.py           # Main window layout
│   ├── canvas_view.py         # Drawing canvas
│   ├── training_view.py       # Training dashboard
│   ├── results_view.py        # Results display
│   └── settings_view.py       # Settings panel
├── components/
│   ├── __init__.py
│   ├── drawing_canvas.py      # Canvas widget
│   ├── probability_bar.py     # Probability visualization
│   ├── progress_chart.py      # Training chart
│   ├── history_list.py        # Prediction history
│   └── toolbar.py             # Tool buttons
├── dialogs/
│   ├── __init__.py
│   ├── file_dialog.py         # File selection
│   ├── settings_dialog.py     # Settings configuration
│   └── about_dialog.py        # About information
├── styles/
│   ├── __init__.py
│   ├── theme.py               # Color themes
│   ├── colors.py              # Color definitions
│   └── fonts.py               # Font settings
└── utils/
    ├── __init__.py
    ├── threading.py           # Async operations
    └── events.py              # Event handling
```

**Window Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Digit Recognition                              [─] [□] [×]  │
├─────────────────────────────────────────────────────────────────┤
│  [File]  [Edit]  [Model]  [Help]                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │                      │  │  Prediction: 7                 │   │
│  │                      │  │  Confidence: 98.5%             │   │
│  │                      │  │                                │   │
│  │    Drawing Canvas    │  │  ┌─────────────────────────┐   │   │
│  │       280x280        │  │  │ 0 ██░░░░░░░░░░░░░ 2.1% │   │    │
│  │                      │  │  │ 1 ███░░░░░░░░░░░░ 5.3% │   │    │
│  │                      │  │  │ 2 █░░░░░░░░░░░░░░ 1.2% │   │    │
│  │                      │  │  │ 3 ░░░░░░░░░░░░░░░ 0.4% │   │    │
│  └──────────────────────┘  │  │ 4 ░░░░░░░░░░░░░░░ 0.2% │   │    │
│                            │  │ 5 ░░░░░░░░░░░░░░░ 0.1% │   │    │
│  [Clear] [Undo] [Upload]   │  │ 6 █░░░░░░░░░░░░░░ 0.8% │   │    │
│                            │  │ 7 ████████████████98.5%│   │    │
│                            │  │ 8 ░░░░░░░░░░░░░░░ 0.3% │   │    │
│                            │  │ 9 █░░░░░░░░░░░░░░ 1.1% │   │    │
│                            │  └─────────────────────────┘   │   │
│                            └─────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Status: Ready │ Model: default.npz │ Accuracy: 97.2%          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Training Module (`src/training/`)

Training utilities and data handling.

```
src/training/
├── __init__.py
├── trainer.py                 # Main trainer class
│   └── Trainer
├── data_loader.py             # Dataset loading
│   ├── MNISTLoader
│   ├── DataLoader
│   └── BatchGenerator
├── callbacks.py               # Training callbacks
│   ├── Callback (ABC)
│   ├── EarlyStopping
│   ├── ModelCheckpoint
│   ├── LearningRateScheduler
│   └── ProgressCallback
├── history.py                 # Training history
│   └── TrainingHistory
└── evaluator.py               # Model evaluation
    └── Evaluator
```

**Trainer Class Interface:**

```python
class Trainer:
    def __init__(self, network, optimizer, loss_fn):
        ...
    
    def fit(self, X_train, y_train, epochs, batch_size,
            validation_data=None, callbacks=None) -> History:
        ...
    
    def evaluate(self, X_test, y_test) -> dict:
        ...
    
    def predict(self, X) -> np.ndarray:
        ...
```

### 3.5 Utils Module (`src/utils/`)

Common utility functions.

```
src/utils/
├── __init__.py
├── config.py                  # Configuration management
│   ├── Config
│   ├── load_config()
│   └── save_config()
├── logger.py                  # Logging utilities
│   ├── setup_logger()
│   └── get_logger()
├── file_io.py                 # File operations
│   ├── save_model()
│   ├── load_model()
│   ├── save_image()
│   └── load_image()
├── math_utils.py              # Math helpers
│   ├── one_hot_encode()
│   ├── shuffle_data()
│   └── train_test_split()
├── image_utils.py             # Image helpers
│   ├── array_to_image()
│   ├── image_to_array()
│   └── display_image()
└── validators.py              # Input validation
    ├── validate_image()
    ├── validate_model()
    └── validate_config()
```

---

## 4. Data Directory Structure (`data/`)

```
data/
├── mnist/
│   ├── train-images-idx3-ubyte.gz   # Training images (60,000)
│   ├── train-labels-idx1-ubyte.gz   # Training labels
│   ├── t10k-images-idx3-ubyte.gz    # Test images (10,000)
│   └── t10k-labels-idx1-ubyte.gz    # Test labels
│
├── custom/                          # User-uploaded images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
└── augmented/                       # Augmented training data
    ├── rotated/
    ├── scaled/
    └── noisy/
```

---

## 5. Models Directory Structure (`models/`)

```
models/
├── default.npz                # Pre-trained default model
├── custom/                    # User-trained models
│   ├── model_20260201_v1.npz
│   └── model_20260201_v2.npz
│
└── checkpoints/               # Training checkpoints
    ├── epoch_001.npz
    ├── epoch_005.npz
    ├── epoch_010.npz
    └── best_model.npz
```

**Model File Format (.npz):**

```python
# Structure of saved model
{
    'architecture': [784, 128, 64, 10],
    'activations': ['relu', 'relu', 'softmax'],
    'weights_0': np.ndarray,  # Layer 0 weights
    'bias_0': np.ndarray,     # Layer 0 bias
    'weights_1': np.ndarray,  # Layer 1 weights
    'bias_1': np.ndarray,     # Layer 1 bias
    'weights_2': np.ndarray,  # Layer 2 weights
    'bias_2': np.ndarray,     # Layer 2 bias
    'metadata': {
        'created': '2026-02-01',
        'accuracy': 0.972,
        'epochs_trained': 20
    }
}
```

---

## 6. Configuration Directory (`config/`)

```
config/
├── default.json               # Default application settings
├── training.json              # Training hyperparameters
├── gui.json                   # GUI settings
└── logging.json               # Logging configuration
```

### 6.1 default.json

```json
{
    "app": {
        "name": "Digit Recognition",
        "version": "1.0.0",
        "debug": false
    },
    "model": {
        "default_weights": "models/default.npz",
        "architecture": [784, 128, 64, 10],
        "activations": ["relu", "relu", "softmax"]
    },
    "preprocessing": {
        "image_size": [28, 28],
        "normalize": true,
        "center_digit": true,
        "invert_colors": "auto"
    }
}
```

### 6.2 training.json

```json
{
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "optimizer": "adam"
    },
    "adam": {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8
    },
    "sgd": {
        "momentum": 0.9
    },
    "regularization": {
        "l2_lambda": 0.0001,
        "dropout_rate": 0.5
    },
    "early_stopping": {
        "patience": 5,
        "min_delta": 0.001
    },
    "checkpointing": {
        "save_best": true,
        "save_frequency": 5
    }
}
```

### 6.3 gui.json

```json
{
    "window": {
        "width": 1200,
        "height": 800,
        "min_width": 800,
        "min_height": 600,
        "resizable": true
    },
    "canvas": {
        "width": 280,
        "height": 280,
        "brush_size": 20,
        "brush_color": "#FFFFFF",
        "background_color": "#000000"
    },
    "theme": {
        "mode": "dark",
        "primary_color": "#3B82F6",
        "secondary_color": "#10B981",
        "background": "#1F2937",
        "surface": "#374151",
        "text": "#F9FAFB"
    },
    "fonts": {
        "family": "Segoe UI",
        "size_normal": 12,
        "size_large": 16,
        "size_title": 24
    }
}
```

---

## 7. Tests Directory Structure (`tests/`)

```
tests/
├── __init__.py
├── conftest.py                # Pytest fixtures
│
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── core/
│   │   ├── test_network.py
│   │   ├── test_layers.py
│   │   ├── test_activations.py
│   │   ├── test_losses.py
│   │   └── test_optimizers.py
│   ├── preprocessing/
│   │   ├── test_pipeline.py
│   │   ├── test_transforms.py
│   │   └── test_augmentation.py
│   └── utils/
│       ├── test_config.py
│       └── test_file_io.py
│
├── integration/               # Integration tests
│   ├── __init__.py
│   ├── test_training_flow.py
│   ├── test_prediction_flow.py
│   └── test_gui_integration.py
│
├── performance/               # Performance tests
│   ├── __init__.py
│   ├── test_inference_speed.py
│   ├── test_training_speed.py
│   └── test_memory_usage.py
│
└── fixtures/                  # Test data
    ├── sample_images/
    │   ├── digit_0.png
    │   ├── digit_1.png
    │   └── ...
    └── sample_models/
        └── test_model.npz
```

---

## 8. Scripts Directory (`scripts/`)

```
scripts/
├── download_mnist.py          # Download MNIST dataset
├── train_model.py             # Command-line training
├── evaluate_model.py          # Evaluate model accuracy
├── convert_model.py           # Convert model formats
├── visualize_weights.py       # Visualize learned weights
├── benchmark.py               # Run benchmarks
└── generate_docs.py           # Generate API documentation
```

**Example: download_mnist.py**

```python
#!/usr/bin/env python
"""Download MNIST dataset"""

import argparse
from src.training.data_loader import MNISTLoader

def main():
    parser = argparse.ArgumentParser(description='Download MNIST dataset')
    parser.add_argument('--output', '-o', default='data/mnist',
                        help='Output directory')
    args = parser.parse_args()
    
    loader = MNISTLoader(data_dir=args.output)
    loader.download()
    print(f"MNIST dataset downloaded to {args.output}")

if __name__ == '__main__':
    main()
```

---

## 9. Entry Points

### 9.1 main.py (GUI Application)

```python
#!/usr/bin/env python
"""
Digit Recognition - GUI Application
Main entry point for the graphical user interface
"""

import sys
from src.gui.app import DigitRecognitionApp
from src.utils.config import load_config
from src.utils.logger import setup_logger

def main():
    # Setup logging
    setup_logger()
    
    # Load configuration
    config = load_config('config/default.json')
    
    # Create and run application
    app = DigitRecognitionApp(config)
    app.run()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

### 9.2 train.py (Training Script)

```python
#!/usr/bin/env python
"""
Digit Recognition - Training Script
Command-line interface for training models
"""

import argparse
from src.core.network import NetworkBuilder
from src.core.losses import CrossEntropyLoss
from src.core.optimizers import Adam
from src.training.trainer import Trainer
from src.training.data_loader import MNISTLoader

def main():
    parser = argparse.ArgumentParser(description='Train digit recognition model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output', '-o', default='models/trained.npz')
    args = parser.parse_args()
    
    # Load data
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()
    
    # Create network
    network = NetworkBuilder() \
        .input(784) \
        .dense(128, activation='relu') \
        .dense(64, activation='relu') \
        .dense(10, activation='softmax') \
        .build()
    
    # Train
    trainer = Trainer(
        network=network,
        optimizer=Adam(learning_rate=args.learning_rate),
        loss_fn=CrossEntropyLoss()
    )
    
    history = trainer.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test)
    )
    
    # Save model
    trainer.save(args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
```

### 9.3 predict.py (CLI Prediction)

```python
#!/usr/bin/env python
"""
Digit Recognition - CLI Prediction Tool
Predict digits from image files
"""

import argparse
from src.core.network import NeuralNetwork
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.file_io import load_model, load_image

def main():
    parser = argparse.ArgumentParser(description='Predict digit from image')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', '-m', default='models/default.npz',
                        help='Path to model weights')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show probability distribution')
    args = parser.parse_args()
    
    # Load model
    network = load_model(args.model)
    
    # Load and preprocess image
    pipeline = PreprocessingPipeline()
    image = load_image(args.image)
    processed = pipeline.process(image)
    
    # Predict
    probabilities = network.forward(processed.reshape(1, -1))
    prediction = probabilities.argmax()
    confidence = probabilities.max() * 100
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    
    if args.verbose:
        print("\nProbability Distribution:")
        for i, prob in enumerate(probabilities[0]):
            bar = '█' * int(prob * 20)
            print(f"  {i}: {bar:20s} {prob*100:5.2f}%")

if __name__ == '__main__':
    main()
```

---

## 10. File Naming Conventions

### 10.1 Python Files

| Type | Convention | Example |
|------|------------|---------|
| Modules | lowercase_snake | `neural_network.py` |
| Classes | PascalCase | `class NeuralNetwork` |
| Functions | lowercase_snake | `def forward_pass()` |
| Constants | UPPERCASE_SNAKE | `LEARNING_RATE = 0.01` |
| Private | _leading_underscore | `def _compute_gradient()` |

### 10.2 Other Files

| Type | Convention | Example |
|------|------------|---------|
| Config files | lowercase | `config.json` |
| Model files | descriptive_date | `model_20260201.npz` |
| Test files | test_module | `test_network.py` |
| Documentation | UPPERCASE | `README.md` |

---

## 11. Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
from PIL import Image

# Local imports
from src.core.network import NeuralNetwork
from src.core.layers import DenseLayer
from src.utils.config import load_config
```

**Import Order:**
1. Standard library
2. Third-party packages
3. Local modules
4. Separate groups with blank line

---

## 12. Dependencies

### 12.1 requirements.txt (Production)

```
numpy>=1.24.0
Pillow>=9.0.0
customtkinter>=5.0.0
matplotlib>=3.7.0
```

### 12.2 requirements-dev.txt (Development)

```
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Code quality
black>=23.0.0
pylint>=2.17.0
mypy>=1.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
```

---

**Document Status**: ✅ Complete  
**Related Documents**: 
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md)
