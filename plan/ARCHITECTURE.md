# 🏗️ System Architecture - Digit Recognition

**Version**: 1.0  
**Date**: 1 Feb 2026  
**Status**: Planning

---

## 1. High-Level Architecture

Aplikasi ini menggunakan arsitektur **Layered Modular** dengan pemisahan jelas antara concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Canvas    │  │   Image     │  │   Webcam    │  │    Training         │ │
│  │   Module    │  │   Upload    │  │   Capture   │  │    Dashboard        │ │
│  │             │  │   Module    │  │   Module    │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼─────────────────────┼───────────┘
          │                │                │                     │
          v                v                v                     v
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Controller / Coordinator                       │   │
│  │   • Input Routing                                                     │   │
│  │   • State Management                                                  │   │
│  │   • Event Handling                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                     │
          v                v                v                     v
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DOMAIN LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Preprocessing│  │   Neural     │  │   Training   │  │  Evaluation    │   │
│  │   Engine     │  │   Network    │  │   Engine     │  │    Engine      │   │
│  │              │  │   Core       │  │              │  │                │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                     │
          v                v                v                     v
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │    Image     │  │    Model     │  │    Data      │  │    Config      │   │
│  │    I/O       │  │ Persistence  │  │   Loader     │  │   Manager      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Presentation Layer

#### Canvas Module
```python
class CanvasModule:
    """Interactive drawing canvas untuk digit input"""
    
    Components:
    - DrawingCanvas: 280x280 pixel drawing area
    - ToolBar: Clear, Undo, Brush size controls
    - PredictionDisplay: Show result dan confidence
    - ProbabilityBars: Visual bar chart untuk 10 classes
```

#### Image Upload Module
```python
class ImageUploadModule:
    """Handle image file input"""
    
    Components:
    - FileSelector: Drag-drop atau browse
    - ImagePreview: Show uploaded image
    - ProcessButton: Trigger prediction
```

#### Training Dashboard
```python
class TrainingDashboard:
    """Interface untuk model training"""
    
    Components:
    - HyperparameterPanel: Learning rate, epochs, batch size
    - ProgressDisplay: Current epoch, loss, accuracy
    - LiveChart: Real-time loss/accuracy graph
    - ControlButtons: Start, stop, pause, save
```

### 2.2 Application Layer

#### Controller / Coordinator
```python
class AppController:
    """Central coordinator untuk application flow"""
    
    Responsibilities:
    - Route input dari presentation ke domain
    - Manage application state
    - Coordinate async operations
    - Handle events dan callbacks
    
    Methods:
    - predict(image_data) -> Prediction
    - train(config) -> TrainingResult
    - save_model(path) -> bool
    - load_model(path) -> bool
```

### 2.3 Domain Layer

#### Preprocessing Engine
```python
class PreprocessingEngine:
    """Image preprocessing pipeline"""
    
    Pipeline:
    1. Convert to grayscale
    2. Resize to 28x28
    3. Invert colors (if needed)
    4. Center digit
    5. Normalize to [0, 1]
    6. Flatten to 784 vector
```

#### Neural Network Core
```python
class NeuralNetwork:
    """Core neural network implementation"""
    
    Components:
    - Layer: Abstract base class
    - DenseLayer: Fully connected layer
    - ActivationLayer: Apply activation function
    - Network: Container untuk layers
    
    Methods:
    - forward(x) -> output
    - backward(gradient) -> None
    - update_weights(learning_rate) -> None
```

#### Training Engine
```python
class TrainingEngine:
    """Manage training process"""
    
    Features:
    - Batch processing
    - Epoch management
    - Loss computation
    - Gradient updates
    - Checkpointing
    
    Methods:
    - train(data, epochs, batch_size) -> History
    - evaluate(test_data) -> Metrics
```

#### Evaluation Engine
```python
class EvaluationEngine:
    """Model evaluation dan metrics"""
    
    Metrics:
    - Accuracy
    - Precision per class
    - Recall per class
    - F1-score
    - Confusion matrix
```

### 2.4 Infrastructure Layer

#### Image I/O
```python
class ImageIO:
    """Handle image file operations"""
    
    Supported formats: PNG, JPG, BMP, GIF
    
    Methods:
    - load_image(path) -> ndarray
    - save_image(array, path) -> bool
    - capture_canvas(canvas) -> ndarray
```

#### Model Persistence
```python
class ModelPersistence:
    """Save and load model weights"""
    
    Format: NumPy .npz atau custom JSON
    
    Methods:
    - save_model(network, path) -> bool
    - load_model(path) -> Network
    - export_weights(network) -> dict
```

#### Data Loader
```python
class DataLoader:
    """Load dan manage datasets"""
    
    Features:
    - MNIST download dan parsing
    - Train/test split
    - Batch generation
    - Data shuffling
    
    Methods:
    - load_mnist() -> (X_train, y_train, X_test, y_test)
    - get_batches(data, batch_size) -> generator
```

---

## 3. Data Flow

### 3.1 Prediction Flow

```
┌──────────────┐
│ User draws   │
│ on canvas    │
└──────┬───────┘
       │
       v
┌──────────────┐
│ Capture      │
│ canvas image │
└──────┬───────┘
       │
       v
┌──────────────────────────────────────────────────────┐
│                  PREPROCESSING                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Grayscale│→ │  Resize  │→ │  Center  │           │
│  │          │  │  28x28   │  │          │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│                                    │                 │
│  ┌──────────┐  ┌──────────┐       │                 │
│  │ Flatten  │← │ Normalize│← ─────┘                 │
│  │  (784)   │  │  [0,1]   │                         │
│  └────┬─────┘  └──────────┘                         │
└───────┼──────────────────────────────────────────────┘
        │
        v
┌──────────────────────────────────────────────────────┐
│                 NEURAL NETWORK                        │
│                                                       │
│   Input (784) → Hidden1 (128) → Hidden2 (64) → Output (10)  │
│                  ReLU            ReLU           Softmax      │
│                                                       │
└───────┬──────────────────────────────────────────────┘
        │
        v
┌──────────────┐
│ Probabilities│
│ [0.01, 0.02, │
│  0.95, ...]  │
└──────┬───────┘
       │
       v
┌──────────────┐
│ Display:     │
│ "Predicted:2"│
│ Conf: 95%    │
└──────────────┘
```

### 3.2 Training Flow

```
┌──────────────┐
│ Load MNIST   │
│ Dataset      │
└──────┬───────┘
       │
       v
┌──────────────┐
│ Initialize   │
│ Network      │
│ Weights      │
└──────┬───────┘
       │
       v
┌──────────────────────────────────────────────────────┐
│              TRAINING LOOP (per epoch)               │
│                                                       │
│   for each batch:                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │ Forward  │→ │ Compute  │→ │ Backward │          │
│   │  Pass    │  │  Loss    │  │  Pass    │          │
│   └──────────┘  └──────────┘  └──────────┘          │
│                                     │                │
│                      ┌──────────────┘                │
│                      v                               │
│               ┌──────────┐                           │
│               │  Update  │                           │
│               │  Weights │                           │
│               └──────────┘                           │
│                                                       │
└───────┬──────────────────────────────────────────────┘
        │
        v
┌──────────────┐
│ Evaluate on  │
│ test set     │
└──────┬───────┘
       │
       v
┌──────────────┐
│ Save model   │
│ if improved  │
└──────────────┘
```

---

## 4. Module Dependencies

```
                    ┌─────────────────┐
                    │     main.py     │
                    │   (Entry Point) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              v              v              v
    ┌─────────────────┐ ┌─────────┐ ┌─────────────────┐
    │   gui/         │ │ core/   │ │    utils/       │
    │   __init__     │ │ __init__│ │    __init__     │
    └────────┬────────┘ └────┬────┘ └────────┬────────┘
             │               │               │
    ┌────────┴────────┐     │      ┌────────┴────────┐
    │                 │     │      │                 │
    v                 v     │      v                 v
┌────────┐     ┌────────┐  │  ┌────────┐     ┌────────┐
│ canvas │     │training│  │  │ image  │     │ config │
│ _view  │     │ _view  │  │  │ _utils │     │        │
└────────┘     └────────┘  │  └────────┘     └────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         v                 v                 v
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ network │      │ layers  │      │ trainer │
    │         │ ────>│         │      │         │
    └─────────┘      └─────────┘      └─────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          v
                    ┌─────────┐
                    │  math   │
                    │ (numpy) │
                    └─────────┘
```

### Dependency Rules
1. **Presentation** depends on **Application** dan **Domain**
2. **Application** depends on **Domain** dan **Infrastructure**
3. **Domain** depends only on **Infrastructure** (for data)
4. **Infrastructure** has no internal dependencies
5. All layers depend on **Utils** (cross-cutting concerns)

---

## 5. Technology Decisions

### 5.1 Core Libraries

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| **Python** | 3.10+ | Runtime | Modern features, type hints |
| **NumPy** | 1.24+ | Matrix ops | Fast vectorized operations |
| **Pillow** | 9.0+ | Image proc | Simple image manipulation |

### 5.2 GUI Framework

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Tkinter** | Built-in, cross-platform | Basic styling | ✅ Primary |
| **CustomTkinter** | Modern look | Extra dependency | ✅ Enhancement |
| **PyQt** | Feature-rich | License, size | ❌ Rejected |
| **Pygame** | Good for graphics | Not suited for forms | ❌ Rejected |

**Decision**: Tkinter + CustomTkinter untuk modern appearance dengan minimal dependencies.

### 5.3 Visualization

| Library | Purpose | Usage |
|---------|---------|-------|
| **Matplotlib** | Training charts | Loss/accuracy plots |
| **Embedded Canvas** | Weight visualization | Display learned features |

### 5.4 Data Format

| Format | Purpose | Specification |
|--------|---------|---------------|
| **NumPy .npz** | Model weights | Compressed array storage |
| **JSON** | Configuration | Human-readable config |
| **PNG** | Image export | Lossless image format |

---

## 6. Design Patterns Used

### 6.1 Creational Patterns

#### Factory Pattern
```python
class LayerFactory:
    @staticmethod
    def create(layer_type: str, **kwargs) -> Layer:
        if layer_type == "dense":
            return DenseLayer(**kwargs)
        elif layer_type == "activation":
            return ActivationLayer(**kwargs)
```

#### Builder Pattern
```python
class NetworkBuilder:
    def __init__(self):
        self.layers = []
    
    def add_dense(self, units, activation=None):
        self.layers.append(DenseLayer(units))
        if activation:
            self.layers.append(ActivationLayer(activation))
        return self
    
    def build(self) -> Network:
        return Network(self.layers)
```

### 6.2 Structural Patterns

#### Composite Pattern
```python
class Network:
    """Composite of layers"""
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

#### Facade Pattern
```python
class DigitRecognizer:
    """Simplified interface to complex subsystem"""
    def __init__(self):
        self.preprocessor = PreprocessingEngine()
        self.network = Network.load("model.npz")
    
    def predict(self, image) -> int:
        processed = self.preprocessor.process(image)
        probabilities = self.network.forward(processed)
        return np.argmax(probabilities)
```

### 6.3 Behavioral Patterns

#### Strategy Pattern
```python
class Optimizer(ABC):
    @abstractmethod
    def update(self, weights, gradients):
        pass

class SGD(Optimizer):
    def update(self, weights, gradients):
        return weights - self.lr * gradients

class Adam(Optimizer):
    def update(self, weights, gradients):
        # Adam algorithm
        pass
```

#### Observer Pattern
```python
class TrainingObserver(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch, loss, accuracy):
        pass

class ChartUpdater(TrainingObserver):
    def on_epoch_end(self, epoch, loss, accuracy):
        self.update_chart(epoch, loss, accuracy)
```

---

## 7. Error Handling Strategy

### 7.1 Exception Hierarchy

```python
class DigitRecognitionError(Exception):
    """Base exception for application"""
    pass

class ModelError(DigitRecognitionError):
    """Model-related errors"""
    pass

class PreprocessingError(DigitRecognitionError):
    """Image preprocessing errors"""
    pass

class DataError(DigitRecognitionError):
    """Data loading/handling errors"""
    pass

class GUIError(DigitRecognitionError):
    """GUI-related errors"""
    pass
```

### 7.2 Error Handling Approach
1. **Validation First**: Validate inputs before processing
2. **Fail Fast**: Raise exceptions early for invalid states
3. **Graceful Degradation**: Continue with defaults when possible
4. **User Feedback**: Always inform user of errors clearly
5. **Logging**: Log all errors for debugging

---

## 8. Configuration Management

### 8.1 Configuration Files

```
config/
├── default.json         # Default settings
├── user.json           # User customizations
└── models/
    └── model_config.json   # Model architecture
```

### 8.2 Configuration Structure

```json
{
  "app": {
    "title": "Digit Recognition",
    "window_size": [1200, 800],
    "theme": "dark"
  },
  "model": {
    "architecture": [784, 128, 64, 10],
    "activations": ["relu", "relu", "softmax"],
    "weights_file": "models/default.npz"
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
    "optimizer": "adam"
  },
  "preprocessing": {
    "image_size": [28, 28],
    "normalize": true,
    "center_digit": true
  }
}
```

---

## 9. Performance Considerations

### 9.1 Optimization Strategies

| Area | Strategy | Expected Impact |
|------|----------|-----------------|
| Matrix Operations | Use NumPy vectorization | 10-100x faster |
| Batch Processing | Process images in batches | Better GPU/CPU utilization |
| Lazy Loading | Load data on demand | Reduced memory footprint |
| Caching | Cache preprocessed data | Faster repeated operations |

### 9.2 Memory Management
- Release large arrays after use
- Use generators for data loading
- Limit history size
- Compress saved models

### 9.3 GUI Responsiveness
- Run training in separate thread
- Use async callbacks for updates
- Debounce canvas updates
- Progressive rendering for charts

---

## 10. Security Considerations

### 10.1 Input Validation
- Validate image file formats
- Check file sizes before loading
- Sanitize file paths
- Validate numeric inputs

### 10.2 Model Safety
- Verify model file integrity
- Use checksums for downloads
- Validate loaded weights shape

---

## 11. Future Architecture Extensions

### 11.1 Planned Enhancements
```
┌─────────────────────────────────────────────────────────────┐
│                     FUTURE ADDITIONS                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │     CNN      │  │     GPU      │  │       Web        │   │
│  │   Layers     │  │  Acceleration│  │    Interface     │   │
│  │              │  │  (CuPy)      │  │    (Flask)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Modular Extension Points
- Custom layer types via plugin system
- Alternative optimizers
- Different dataset loaders
- GUI themes and layouts

---

**Document Status**: ✅ Complete  
**Related Documents**: 
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- [NEURAL_NETWORK_DESIGN.md](NEURAL_NETWORK_DESIGN.md)
- [MATHEMATICAL_FOUNDATION.md](MATHEMATICAL_FOUNDATION.md)
