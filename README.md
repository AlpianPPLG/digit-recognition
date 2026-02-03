# 🔢 AI Digit Recognition

A handwritten digit recognition system built from scratch using pure NumPy - no TensorFlow or PyTorch required!

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-green.svg)
![Tests](https://img.shields.io/badge/Tests-236%20passed-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97.97%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <img src="docs/demo.gif" alt="Demo" width="600">
</p>

## ✨ Features

- 🧠 **Neural Network from Scratch** - No deep learning frameworks, just NumPy
- 🎨 **Modern GUI** - CustomTkinter-based interface with dark/light mode
- ✏️ **Drawing Canvas** - Draw digits with mouse or touch
- 📊 **Real-time Predictions** - See confidence scores for all digits
- 💾 **Model Persistence** - Save/load trained models
- 🎯 **97.97% Accuracy** - Trained on MNIST dataset

## 🏗️ Architecture

```
Input (784) → Dense (128, ReLU) → Dense (64, ReLU) → Dense (10, Softmax)
```

| Layer     | Parameters                 |
| --------- | -------------------------- |
| Input     | 784 neurons (28×28 pixels) |
| Hidden 1  | 128 neurons + ReLU         |
| Hidden 2  | 64 neurons + ReLU          |
| Output    | 10 neurons + Softmax       |
| **Total** | **109,386 parameters**     |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AiDigit.git
cd AiDigit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Launch GUI application
python run.py

# Or using module
python -m src.main
```

### Train Your Own Model

```bash
# Full training (50 epochs)
python scripts/train_mnist.py

# Quick training (10 epochs)
python scripts/train_mnist.py --quick

# Custom training
python scripts/train_mnist.py --epochs 30 --batch-size 64 --lr 0.001
```

## 📁 Project Structure

```
AiDigit/
├── src/
│   ├── core/                    # Neural network core
│   │   ├── network.py           # NeuralNetwork & NetworkBuilder
│   │   ├── layers.py            # Dense, Activation, Dropout layers
│   │   ├── activations.py       # ReLU, Sigmoid, Softmax
│   │   ├── losses.py            # CrossEntropy, MSE losses
│   │   ├── optimizers.py        # SGD, Adam, RMSprop
│   │   ├── initializers.py      # Xavier, He initialization
│   │   ├── metrics.py           # Accuracy, Precision, F1
│   │   └── serialization.py     # Model save/load
│   │
│   ├── preprocessing/           # Data preprocessing
│   │   ├── mnist_loader.py      # MNIST dataset loader
│   │   └── image_preprocessor.py # Image processing & augmentation
│   │
│   ├── training/                # Training utilities
│   │   └── trainer.py           # Trainer, EarlyStopping, LRScheduler
│   │
│   ├── gui/                     # User interface
│   │   ├── main_window.py       # Main application window
│   │   ├── drawing_canvas.py    # Drawing canvas widget
│   │   └── app.py               # Application integration
│   │
│   ├── utils/                   # Utilities
│   │   └── math_utils.py        # Mathematical operations
│   │
│   └── main.py                  # Main entry point
│
├── tests/                       # Test suite
│   └── unit/                    # Unit tests
│       ├── test_layers.py
│       ├── test_network.py
│       ├── test_optimizers.py
│       ├── test_metrics.py
│       ├── test_losses.py
│       ├── test_math_utils.py
│       ├── test_preprocessing.py
│       ├── test_trainer.py
│       └── test_serialization.py
│
├── scripts/                     # Utility scripts
│   └── train_mnist.py           # MNIST training script
│
├── models/                      # Saved models
│   └── digit_recognition_model.npz
│
├── data/                        # Dataset storage
│   └── mnist/
│
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_network.py -v
```

**Test Results:** 236 tests passing ✅

## 📊 Training Results

| Metric              | Value                   |
| ------------------- | ----------------------- |
| Training Accuracy   | 99.63%                  |
| Validation Accuracy | 97.88%                  |
| **Test Accuracy**   | **97.97%**              |
| Training Time       | ~25 seconds (10 epochs) |

### Training Progress

```
Epoch   1/10 - loss: 0.3063 - acc: 0.9104 - val_acc: 0.9607
Epoch   2/10 - loss: 0.1158 - acc: 0.9659 - val_acc: 0.9722
Epoch   3/10 - loss: 0.0792 - acc: 0.9762 - val_acc: 0.9695
Epoch   4/10 - loss: 0.0600 - acc: 0.9814 - val_acc: 0.9762
Epoch   5/10 - loss: 0.0474 - acc: 0.9856 - val_acc: 0.9763
...
Final Test Accuracy: 97.97% ✓
```

## 🔧 Configuration

### Command Line Options

```bash
python -m src.main [OPTIONS]

Options:
  --train          Train the model before launching GUI
  --epochs N       Number of training epochs (default: 30)
  --cli            Run in command-line mode (no GUI)
  --test           Run tests instead of launching app
  --version        Show version number
```

### Appearance Settings

The GUI supports Dark, Light, and System appearance modes. Change it from the sidebar menu.

## 🛠️ Built With

| Library       | Version | Purpose                                |
| ------------- | ------- | -------------------------------------- |
| NumPy         | 1.26.4  | Matrix operations, neural network math |
| Pillow        | 12.1.0  | Image processing                       |
| CustomTkinter | 5.2.2   | Modern GUI framework                   |
| Matplotlib    | 3.10.8  | Visualization                          |
| pytest        | 9.0.2   | Testing framework                      |

## 📖 How It Works

### 1. Neural Network Forward Pass

```python
# Input: 784-dimensional flattened image
x = image.flatten()  # Shape: (784,)

# Hidden Layer 1: Linear + ReLU
z1 = x @ W1 + b1      # Shape: (128,)
a1 = relu(z1)

# Hidden Layer 2: Linear + ReLU
z2 = a1 @ W2 + b2     # Shape: (64,)
a2 = relu(z2)

# Output Layer: Linear + Softmax
z3 = a2 @ W3 + b3     # Shape: (10,)
output = softmax(z3)  # Probabilities for digits 0-9
```

### 2. Training with Backpropagation

```python
# Forward pass
predictions = network.forward(X_batch)

# Compute loss
loss = cross_entropy(predictions, y_batch)

# Backward pass (gradient computation)
gradients = network.backward(y_batch)

# Update weights
optimizer.step(network.layers, gradients)
```

### 3. Image Preprocessing

```python
# 1. Convert to grayscale
# 2. Find bounding box of digit
# 3. Center and resize to 28x28
# 4. Normalize to [0, 1]
# 5. Flatten to 784 dimensions
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Yann LeCun
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Tom Schimansky
- Neural Network concepts inspired by [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/AiDigit](https://github.com/yourusername/AiDigit)

---

<p align="center">
  Made with ❤️ and pure NumPy
</p>
