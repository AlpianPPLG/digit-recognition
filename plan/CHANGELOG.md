# 📝 Changelog - Digit Recognition

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned

- GPU acceleration support
- Additional layer types (Conv2D, MaxPool)
- Model export to ONNX format
- Web interface version
- Mobile-friendly PWA

---

## [1.0.0] - 2026-03-15

### 🎉 Initial Release

First stable release of Digit Recognition application with neural network from scratch.

### Added

#### Core Neural Network

- **DenseLayer** - Fully connected layer with forward/backward pass
  - He and Xavier weight initialization
  - Gradient computation and caching
  - Configurable input/output sizes

- **Activation Functions**
  - ReLU activation with leaky variant support
  - Sigmoid activation (numerically stable)
  - Softmax activation (numerically stable)
  - Tanh activation

- **Loss Functions**
  - Cross-Entropy Loss for classification
  - Mean Squared Error (MSE) for regression

- **Optimizers**
  - SGD (Stochastic Gradient Descent)
  - SGD with Momentum
  - Adam optimizer with bias correction

#### Training Pipeline

- **Trainer class** with complete training loop
  - Mini-batch processing
  - Epoch-based training
  - Validation evaluation
  - Training history tracking
  - Progress bar display

- **Learning Rate Schedulers**
  - StepLR - decrease LR at intervals
  - ExponentialLR - exponential decay
  - CosineAnnealingLR - cosine annealing
  - ReduceOnPlateau - reduce on validation plateau

- **Callbacks**
  - EarlyStopping - prevent overfitting
  - ModelCheckpoint - save best model
  - Training history logging

#### Data Handling

- **MNIST Loader**
  - Automatic download from official source
  - IDX file format parsing
  - Normalization (0-1 range)
  - One-hot encoding for labels
  - Train/validation split support

- **Preprocessing Pipeline**
  - Canvas image processing (280x280 → 28x28)
  - External image file support
  - Grayscale conversion
  - Center of mass centering
  - MNIST-style normalization

#### GUI Application

- **Main Window**
  - Modern dark/light theme support
  - Responsive layout
  - Keyboard shortcuts
  - Status bar with system info

- **Drawing Canvas**
  - 280x280 drawing area
  - Configurable brush size
  - Clear button
  - Real-time preview
  - Smooth anti-aliased drawing

- **Prediction Display**
  - Large predicted digit display
  - Confidence percentage
  - Probability bar chart for all digits
  - Top-3 predictions list
  - Real-time updates

- **Training Dashboard**
  - Parameter configuration panel
  - Start/Stop/Pause controls
  - Progress bar with ETA
  - Live loss/accuracy charts
  - Training history export

- **File Operations**
  - Load external image files (PNG, JPG)
  - Save/load trained models
  - Export predictions
  - Export training history

#### Model Architecture

- Default architecture: 784 → 128 → 64 → 10
- Total parameters: 109,386
- Activation: ReLU (hidden), Softmax (output)
- Loss: Cross-Entropy
- Optimizer: Adam (lr=0.001)

#### Performance

- Test accuracy: **97.3%** on MNIST
- Single inference: **~15ms**
- Training time: **~4 minutes** (20 epochs)
- Memory usage: **<300MB** during training

#### Documentation

- Complete API documentation
- User guide with screenshots
- Installation guide (Windows, macOS, Linux)
- Mathematical foundation document
- Architecture documentation

### Technical Details

- **Python**: 3.10+
- **Dependencies**:
  - NumPy ≥1.24.0
  - Pillow ≥9.0.0
  - Matplotlib ≥3.6.0
  - CustomTkinter ≥5.0.0 (optional)

---

## [0.9.0] - 2026-03-01 (Beta)

### Added

- Complete GUI implementation
- All core neural network components
- MNIST data loader
- Basic preprocessing pipeline

### Changed

- Improved numerical stability in softmax
- Optimized matrix operations

### Fixed

- Memory leak in training loop
- GUI freezing during training (moved to thread)

---

## [0.8.0] - 2026-02-22 (Alpha)

### Added

- Training pipeline with Trainer class
- All optimizers (SGD, Momentum, Adam)
- Learning rate schedulers
- Early stopping callback
- Model checkpoint saving

### Changed

- Refactored layer architecture
- Improved gradient computation

---

## [0.7.0] - 2026-02-15 (Alpha)

### Added

- DenseLayer implementation
- ReLU, Sigmoid, Softmax activations
- Cross-Entropy loss function
- Network class with forward/backward
- NetworkBuilder for easy construction

### Fixed

- Gradient computation in dense layer
- Numerical instability in cross-entropy

---

## [0.6.0] - 2026-02-08 (Alpha)

### Added

- Basic GUI with drawing canvas
- Canvas to numpy conversion
- Simple prediction display

### Known Issues

- GUI freezes during prediction (to be fixed)
- No training interface yet

---

## [0.5.0] - 2026-02-01 (Alpha)

### Added

- Project structure
- Mathematical utilities
- Base classes for layers, activations, loss
- MNIST downloader
- Basic unit tests

---

## Version History Summary

| Version | Date       | Highlights                 |
| ------- | ---------- | -------------------------- |
| 1.0.0   | 2026-03-15 | 🎉 Initial stable release  |
| 0.9.0   | 2026-03-01 | Beta: Complete GUI         |
| 0.8.0   | 2026-02-22 | Alpha: Training pipeline   |
| 0.7.0   | 2026-02-15 | Alpha: Core neural network |
| 0.6.0   | 2026-02-08 | Alpha: Basic GUI           |
| 0.5.0   | 2026-02-01 | Alpha: Project foundation  |

---

## Migration Guides

### Migrating from 0.x to 1.0

Tidak ada breaking changes. Upgrade langsung:

```bash
pip install --upgrade digit-recognition
```

---

## Future Roadmap

### v1.1.0 (Planned)

- [ ] Batch normalization layer
- [ ] Dropout layer
- [ ] Data augmentation options
- [ ] Model comparison tool

### v1.2.0 (Planned)

- [ ] Convolutional layers (Conv2D)
- [ ] Pooling layers (MaxPool, AvgPool)
- [ ] Support for Fashion-MNIST

### v2.0.0 (Future)

- [ ] GPU acceleration (CuPy backend)
- [ ] Web interface
- [ ] Model export (ONNX, TensorFlow Lite)

---

## Contributors

Thanks to all contributors who helped make this project possible!

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

---

## Links

- [Documentation](docs/INDEX.md)
- [GitHub Repository](https://github.com/username/digit-recognition)
- [Issue Tracker](https://github.com/username/digit-recognition/issues)
- [Releases](https://github.com/username/digit-recognition/releases)
