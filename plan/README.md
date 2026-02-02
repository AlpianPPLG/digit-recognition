# 🔢 Digit Recognition - Mathematical AI Engine

> Sistem pengenalan angka tulisan tangan berbasis matematis dengan akurasi tinggi menggunakan Python

[![Status](https://img.shields.io/badge/status-planning-yellow)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematical_Foundation-013243)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Tentang Project

**Digit Recognition** adalah sistem pengenalan angka tulisan tangan (0-9) yang dibangun dari fondasi matematis murni. Project ini mengimplementasikan algoritma machine learning **dari nol** tanpa bergantung pada framework high-level seperti TensorFlow atau PyTorch, dengan tujuan pemahaman mendalam tentang matematika di balik AI.

Sistem mendukung multiple input method:

- ✏️ **Canvas GUI** - Menggambar angka langsung di canvas interaktif
- 🖼️ **Image Upload** - Upload gambar angka untuk dikenali
- 📷 **Webcam Capture** - Tangkap angka dari kamera real-time
- 📁 **Batch Processing** - Proses multiple images sekaligus

### 🎯 Status Project

**Current Phase**: 📝 Planning Complete - Ready for Development  
**Last Updated**: 1 Feb 2026

### 📚 Complete Planning Documentation

Semua dokumentasi planning telah dibuat dan siap untuk development. Lihat **[PLANNING_SUMMARY.md](PLANNING_SUMMARY.md)** untuk overview lengkap.

## ✨ Fitur Utama

### 🧮 Mathematical Foundation

- **Neural Network from Scratch** - Implementasi forward/backward propagation manual
- **Activation Functions** - Sigmoid, ReLU, Softmax dengan derivatif
- **Gradient Descent Optimization** - SGD, Mini-batch, Adam optimizer
- **Loss Functions** - Cross-entropy, MSE dengan mathematical derivation
- **Regularization** - L1/L2, Dropout untuk mencegah overfitting

### 🎨 User Interface

- **Interactive Canvas** - Draw digits dengan mouse/stylus
- **Real-time Prediction** - Lihat probabilitas setiap digit
- **Confidence Visualization** - Bar chart probabilitas
- **History & Statistics** - Track accuracy dan performance

### 🔬 Advanced Features

- **Model Training UI** - Train model dengan visualisasi progress
- **Hyperparameter Tuning** - Adjust learning rate, epochs, batch size
- **Performance Metrics** - Confusion matrix, precision, recall, F1-score
- **Model Export/Import** - Save dan load trained models

## 🛠️ Tech Stack

### Core Libraries

```
Python 3.10+          - Programming Language
NumPy                 - Mathematical Operations & Matrix Algebra
Pillow (PIL)          - Image Processing
```

### GUI Framework

```
Tkinter              - Native Python GUI (Primary)
CustomTkinter        - Modern UI Components
Matplotlib           - Visualization & Charts
```

### Optional Enhancements

```
OpenCV               - Advanced Image Processing
Pygame               - Alternative Canvas Implementation
```

### Development Tools

```
pytest               - Testing Framework
mypy                 - Static Type Checking
black                - Code Formatter
pylint               - Code Quality
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Canvas  │  │  Image   │  │  Webcam  │  │  Training Panel  │ │
│  │   GUI    │  │  Upload  │  │  Capture │  │                  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │             │             │                 │
        v             v             v                 v
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Resize &   │  │  Grayscale   │  │   Normalization &    │   │
│  │   Centering  │  │  Conversion  │  │   Feature Scaling    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK ENGINE                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Input   │  │  Hidden  │  │  Hidden  │  │      Output      │ │
│  │  Layer   │→ │  Layer 1 │→ │  Layer 2 │→ │      Layer       │ │
│  │  (784)   │  │  (128)   │  │  (64)    │  │      (10)        │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                                 │
│  Mathematical Components:                                       │
│  • Forward Propagation    • Backward Propagation               │
│  • Activation Functions   • Weight Updates                      │
│  • Loss Calculation       • Gradient Computation                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT & RESULTS                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Prediction  │  │  Confidence  │  │   Visualization &    │   │
│  │    Result    │  │    Scores    │  │      Analytics       │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 atau lebih baru
- pip (Python package manager)
- Git

### Installation

1. **Clone repository**

```bash
git clone https://github.com/username/digit-recognition.git
cd digit-recognition
```

2. **Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download MNIST dataset** (otomatis saat pertama run)

```bash
python scripts/download_mnist.py
```

5. **Run application**

```bash
# GUI Application
python main.py

# Training Mode
python train.py

# CLI Mode
python predict.py --image path/to/image.png
```

## 📚 Dokumentasi

Dokumentasi lengkap tersedia di folder ini:

### Planning & Requirements

- [PRD (Product Requirements Document)](PRD.md)
- [PLANNING_SUMMARY](PLANNING_SUMMARY.md)
- [DEVELOPMENT_ROADMAP](DEVELOPMENT_ROADMAP.md)

### Architecture & Design

- [ARCHITECTURE](ARCHITECTURE.md)
- [MATHEMATICAL_FOUNDATION](MATHEMATICAL_FOUNDATION.md)
- [PROJECT_STRUCTURE](PROJECT_STRUCTURE.md)

### Algorithm & Implementation

- [NEURAL_NETWORK_DESIGN](NEURAL_NETWORK_DESIGN.md)
- [PREPROCESSING_PIPELINE](PREPROCESSING_PIPELINE.md)
- [TRAINING_STRATEGY](TRAINING_STRATEGY.md)

### UI/UX & Interface

- [GUI_DESIGN](GUI_DESIGN.md)
- [USER_GUIDE](USER_GUIDE.md)

### Testing & Quality

- [TESTING_STRATEGY](TESTING_STRATEGY.md)
- [PERFORMANCE_BENCHMARKS](PERFORMANCE_BENCHMARKS.md)

### Setup & Deployment

- [SETUP_GUIDE](SETUP_GUIDE.md)
- [CONTRIBUTING](CONTRIBUTING.md)
- [CHANGELOG](CHANGELOG.md)

## 📊 Target Performance

| Metric             | Target  | Notes                |
| ------------------ | ------- | -------------------- |
| **Accuracy**       | ≥ 97%   | Pada MNIST test set  |
| **Inference Time** | < 50ms  | Per single image     |
| **Training Time**  | < 5 min | Full MNIST dataset   |
| **Model Size**     | < 5 MB  | Saved model file     |
| **GUI Response**   | < 100ms | Real-time prediction |

## 🎓 Learning Outcomes

Project ini dirancang untuk pemahaman mendalam tentang:

1. **Linear Algebra** - Matrix operations, dot products, transpose
2. **Calculus** - Derivatives, chain rule, gradient computation
3. **Probability** - Softmax, cross-entropy, probability distributions
4. **Optimization** - Gradient descent variants, learning rate scheduling
5. **Neural Network Theory** - Layers, activations, backpropagation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Kontribusi sangat diterima! Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk guidelines.

---

**Made with 🧮 Mathematics and ❤️ Python**
