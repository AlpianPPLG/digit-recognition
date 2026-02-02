# 🧠 Neural Network Design - Digit Recognition

**Version**: 1.0
**Date**: 1 Feb 2026
**Status**: Planning

---

## 1. Network Architecture Overview

### 1.1 Architecture Decision

Untuk digit recognition (MNIST), kita menggunakan **Fully Connected Feed-Forward Neural Network** (Multi-Layer Perceptron / MLP).

**Rationale:**
- MNIST images cukup kecil (28x28 = 784 pixels)
- MLP cukup powerful untuk achieve 97%+ accuracy
- Lebih mudah diimplementasikan dari scratch
- Educational value: memahami fundamentals sebelum CNN

### 1.2 Final Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NETWORK ARCHITECTURE                              │
│                                                                          │
│   INPUT        HIDDEN 1        HIDDEN 2        OUTPUT                   │
│   Layer          Layer           Layer          Layer                    │
│                                                                          │
│  ┌─────┐       ┌─────┐        ┌─────┐        ┌────┐                   │
│  │     │       │     │        │     │        │     │                   │
│  │ 784 │ ───►  │ 128 │ ────►  │  64 │ ────►  │  10 │                   │
│  │     │       │     │        │     │        │     │                   │
│  │     │       │     │        │     │        │     │                   │
│  └─────┘       └─────┘        └─────┘        └─────┘                   │
│   28×28         ReLU           ReLU          Softmax                    │
│  flatten                                                                 │
│                                                                          │
│  Params:        100,480       8,256          650                        │
│  (W + b)       (784×128+128) (128×64+64)   (64×10+10)                  │
│                                                                          │
│  Total Parameters: 109,386                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Architecture Specifications

| Layer | Type | Input Size | Output Size | Activation | Parameters |
|-------|------|------------|-------------|------------|------------|
| Input | Flatten | 28×28 | 784 | None | 0 |
| Hidden 1 | Dense | 784 | 128 | ReLU | 100,480 |
| Hidden 2 | Dense | 128 | 64 | ReLU | 8,256 |
| Output | Dense | 64 | 10 | Softmax | 650 |
| **Total** | | | | | **109,386** |

---

## 2. Layer Design

### 2.1 Base Layer Abstract Class

```python
from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """Abstract base class untuk semua layers"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass computation"""
        pass
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass computation"""
        pass
    
    @property
    def parameters(self) -> dict:
        """Return trainable parameters"""
        return {}
    
    @property
    def gradients(self) -> dict:
        """Return parameter gradients"""
        return {}
```

### 2.2 Dense Layer Implementation

```python
class DenseLayer(Layer):
    """
    Fully connected layer: y = Wx + b
    
    Attributes:
        weights: Weight matrix (input_size, output_size)
        bias: Bias vector (1, output_size)
        input_cache: Cached input for backward pass
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 initializer: str = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights
        if initializer == 'he':
            std = np.sqrt(2.0 / input_size)
        elif initializer == 'xavier':
            std = np.sqrt(2.0 / (input_size + output_size))
        else:
            std = 0.01
        
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros((1, output_size))
        
        # Cache for backward pass
        self.input_cache = None
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = Wx + b
        
        Args:
            x: Input of shape (batch_size, input_size)
        
        Returns:
            z: Output of shape (batch_size, output_size)
        """
        self.input_cache = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients
        
        Args:
            grad: Gradient from next layer (batch_size, output_size)
        
        Returns:
            grad_input: Gradient to pass to previous layer
        """
        batch_size = grad.shape[0]
        
        # Gradient w.r.t weights: dL/dW = X^T @ dL/dz
        self.dW = np.dot(self.input_cache.T, grad) / batch_size
        
        # Gradient w.r.t bias: dL/db = sum(dL/dz)
        self.db = np.sum(grad, axis=0, keepdims=True) / batch_size
        
        # Gradient to pass to previous layer: dL/dX = dL/dz @ W^T
        return np.dot(grad, self.weights.T)
    
    @property
    def parameters(self) -> dict:
        return {'weights': self.weights, 'bias': self.bias}
    
    @property
    def gradients(self) -> dict:
        return {'weights': self.dW, 'bias': self.db}
```

### 2.3 Activation Layers

```python
class ReLULayer(Layer):
    """ReLU activation: f(x) = max(0, x)"""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0)
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask


class SigmoidLayer(Layer):
    """Sigmoid activation: f(x) = 1 / (1 + e^-x)"""
    
    def __init__(self):
        self.output_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output_cache = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output_cache
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.output_cache * (1 - self.output_cache)


class SoftmaxLayer(Layer):
    """Softmax activation for output layer"""
    
    def __init__(self):
        self.output_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output_cache = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output_cache
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # When used with cross-entropy loss, gradient simplifies
        # This is handled in loss function
        return grad
```

---

## 3. Network Implementation

### 3.1 Sequential Network Class

```python
class NeuralNetwork:
    """
    Sequential neural network container
    
    Supports:
    - Adding layers
    - Forward propagation
    - Backward propagation
    - Training and evaluation
    """
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
    
    def add(self, layer: Layer):
        """Add layer to network"""
        self.layers.append(layer)
        return self  # Enable chaining
    
    def set_loss(self, loss_fn):
        """Set loss function"""
        self.loss_fn = loss_fn
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers
        
        Args:
            x: Input data (batch_size, input_features)
        
        Returns:
            output: Network output (batch_size, num_classes)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward pass through all layers
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
        """
        # Compute loss gradient
        grad = self.loss_fn.gradient(y_true, y_pred)
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_parameters(self) -> list:
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            if layer.parameters:
                params.append(layer.parameters)
        return params
    
    def get_gradients(self) -> list:
        """Get all gradients"""
        grads = []
        for layer in self.layers:
            if layer.gradients:
                grads.append(layer.gradients)
        return grads
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def summary(self):
        """Print network summary"""
        print("=" * 60)
        print("Network Summary")
        print("=" * 60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            params = layer.parameters
            num_params = sum(p.size for p in params.values()) if params else 0
            total_params += num_params
            print(f"Layer {i}: {name:20s} | Parameters: {num_params:,}")
        
        print("=" * 60)
        print(f"Total Parameters: {total_params:,}")
        print("=" * 60)
```

### 3.2 Network Builder

```python
class NetworkBuilder:
    """
    Fluent builder for creating neural networks
    
    Usage:
        network = NetworkBuilder() \
            .input(784) \
            .dense(128, activation='relu') \
            .dense(64, activation='relu') \
            .dense(10, activation='softmax') \
            .build()
    """
    
    def __init__(self):
        self.network = NeuralNetwork()
        self.prev_size = None
    
    def input(self, size: int):
        """Set input size"""
        self.prev_size = size
        return self
    
    def dense(self, units: int, activation: str = None):
        """Add dense layer with optional activation"""
        if self.prev_size is None:
            raise ValueError("Must call input() first")
        
        self.network.add(DenseLayer(self.prev_size, units))
        
        if activation:
            if activation == 'relu':
                self.network.add(ReLULayer())
            elif activation == 'sigmoid':
                self.network.add(SigmoidLayer())
            elif activation == 'softmax':
                self.network.add(SoftmaxLayer())
        
        self.prev_size = units
        return self
    
    def build(self) -> NeuralNetwork:
        """Build and return the network"""
        return self.network
```

---

## 4. Loss Functions

### 4.1 Cross-Entropy Loss

```python
class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification
    
    Loss = -sum(y_true * log(y_pred))
    """
    
    def __init__(self):
        self.epsilon = 1e-15  # Prevent log(0)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            y_true: One-hot encoded labels (batch_size, num_classes)
            y_pred: Predicted probabilities (batch_size, num_classes)
        
        Returns:
            loss: Scalar loss value
        """
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute cross-entropy
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t predictions
        
        For softmax + cross-entropy, gradient simplifies to: y_pred - y_true
        """
        return y_pred - y_true
```

### 4.2 Mean Squared Error Loss

```python
class MSELoss:
    """
    Mean Squared Error loss
    
    Loss = mean((y_true - y_pred)^2)
    """
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]
```

---

## 5. Optimizers

### 5.1 SGD Optimizer

```python
class SGD:
    """
    Stochastic Gradient Descent optimizer
    
    w = w - lr * gradient
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
    
    def update(self, params: dict, grads: dict):
        """Update parameters using gradients"""
        for key in params:
            if grads[key] is not None:
                params[key] -= self.lr * grads[key]
```

### 5.2 SGD with Momentum

```python
class SGDMomentum:
    """
    SGD with Momentum
    
    v = momentum * v - lr * gradient
    w = w + v
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params: dict, grads: dict, layer_id: int):
        """Update parameters with momentum"""
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                key: np.zeros_like(params[key]) for key in params
            }
        
        for key in params:
            if grads[key] is not None:
                v = self.velocities[layer_id][key]
                v = self.momentum * v - self.lr * grads[key]
                self.velocities[layer_id][key] = v
                params[key] += v
```

### 5.3 Adam Optimizer

```python
class Adam:
    """
    Adam optimizer
    
    Combines momentum and adaptive learning rates
    """
    
    def __init__(self, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params: dict, grads: dict, layer_id: int):
        """Update parameters using Adam"""
        self.t += 1
        
        if layer_id not in self.m:
            self.m[layer_id] = {
                key: np.zeros_like(params[key]) for key in params
            }
            self.v[layer_id] = {
                key: np.zeros_like(params[key]) for key in params
            }
        
        for key in params:
            if grads[key] is None:
                continue
            
            g = grads[key]
            
            # Update biased first moment estimate
            self.m[layer_id][key] = (
                self.beta1 * self.m[layer_id][key] + 
                (1 - self.beta1) * g
            )
            
            # Update biased second moment estimate
            self.v[layer_id][key] = (
                self.beta2 * self.v[layer_id][key] + 
                (1 - self.beta2) * g**2
            )
            
            # Compute bias-corrected estimates
            m_hat = self.m[layer_id][key] / (1 - self.beta1**self.t)
            v_hat = self.v[layer_id][key] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

---

## 6. Weight Initialization Strategies

### 6.1 Initialization Methods

```python
class Initializer:
    """Weight initialization methods"""
    
    @staticmethod
    def zeros(shape: tuple) -> np.ndarray:
        """Initialize with zeros"""
        return np.zeros(shape)
    
    @staticmethod
    def ones(shape: tuple) -> np.ndarray:
        """Initialize with ones"""
        return np.ones(shape)
    
    @staticmethod
    def random_normal(shape: tuple, mean: float = 0, std: float = 0.01) -> np.ndarray:
        """Initialize with random normal distribution"""
        return np.random.randn(*shape) * std + mean
    
    @staticmethod
    def xavier(shape: tuple) -> np.ndarray:
        """
        Xavier/Glorot initialization
        Good for tanh and sigmoid activations
        """
        fan_in, fan_out = shape[0], shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std
    
    @staticmethod
    def he(shape: tuple) -> np.ndarray:
        """
        He initialization
        Good for ReLU activations
        """
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std
    
    @staticmethod
    def lecun(shape: tuple) -> np.ndarray:
        """
        LeCun initialization
        """
        fan_in = shape[0]
        std = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * std
```

### 6.2 Initialization Comparison

| Method | Formula | Best For |
|--------|---------|----------|
| Xavier | $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$ | Sigmoid, Tanh |
| He | $\sigma = \sqrt{\frac{2}{n_{in}}}$ | ReLU |
| LeCun | $\sigma = \sqrt{\frac{1}{n_{in}}}$ | SELU |

---

## 7. Regularization

### 7.1 L2 Regularization

```python
class L2Regularizer:
    """
    L2 regularization (weight decay)
    
    Adds lambda * ||W||^2 to loss
    """
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def loss(self, weights: np.ndarray) -> float:
        """Compute regularization loss"""
        return self.lambda_ * np.sum(weights ** 2)
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute regularization gradient"""
        return 2 * self.lambda_ * weights
```

### 7.2 Dropout Layer

```python
class DropoutLayer(Layer):
    """
    Dropout regularization layer
    
    Randomly sets activations to 0 during training
    """
    
    def __init__(self, rate: float = 0.5):
        self.rate = rate  # Probability of dropping
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x
        
        # Create dropout mask
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
        
        # Scale by 1/(1-rate) to maintain expected value
        return x * self.mask / (1 - self.rate)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training:
            return grad
        
        return grad * self.mask / (1 - self.rate)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
```

---

## 8. Complete Network Example

### 8.1 Building the Network

```python
def create_digit_recognition_network():
    """
    Create neural network for MNIST digit recognition
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden 1: 128 neurons, ReLU
    - Hidden 2: 64 neurons, ReLU
    - Output: 10 neurons, Softmax
    """
    
    network = NetworkBuilder() \
        .input(784) \
        .dense(128, activation='relu') \
        .dense(64, activation='relu') \
        .dense(10, activation='softmax') \
        .build()
    
    network.set_loss(CrossEntropyLoss())
    
    return network


# Alternative: Manual construction
def create_network_manual():
    network = NeuralNetwork()
    
    # Layer 1: Dense + ReLU
    network.add(DenseLayer(784, 128, initializer='he'))
    network.add(ReLULayer())
    
    # Layer 2: Dense + ReLU
    network.add(DenseLayer(128, 64, initializer='he'))
    network.add(ReLULayer())
    
    # Output layer: Dense + Softmax
    network.add(DenseLayer(64, 10, initializer='xavier'))
    network.add(SoftmaxLayer())
    
    network.set_loss(CrossEntropyLoss())
    
    return network
```

### 8.2 Network Usage

```python
# Create network
network = create_digit_recognition_network()

# Print summary
network.summary()

# Create optimizer
optimizer = Adam(learning_rate=0.001)

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        # Forward pass
        predictions = network.forward(batch_x)
        
        # Compute loss
        loss = network.loss_fn(batch_y, predictions)
        
        # Backward pass
        network.backward(batch_y, predictions)
        
        # Update weights
        for i, layer in enumerate(network.layers):
            if layer.parameters:
                optimizer.update(
                    layer.parameters,
                    layer.gradients,
                    layer_id=i
                )

# Inference
test_predictions = network.predict(test_images)
```

---

## 9. Hyperparameter Choices

### 9.1 Recommended Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Learning Rate | 0.001 | Standard for Adam |
| Batch Size | 32 | Balance speed/stability |
| Epochs | 20 | Sufficient for convergence |
| Hidden Units | [128, 64] | Empirically good for MNIST |
| Activation | ReLU | Fast, no vanishing gradient |
| Optimizer | Adam | Adaptive, fast convergence |
| Weight Init | He | Optimal for ReLU |
| Loss | Cross-Entropy | Standard for classification |

### 9.2 Hyperparameter Tuning Guide

```python
# Hyperparameter search space
hyperparameters = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64, 128],
    'hidden_units': [[64, 32], [128, 64], [256, 128]],
    'dropout_rate': [0.0, 0.2, 0.5],
    'optimizer': ['sgd', 'momentum', 'adam']
}
```

---

## 10. Alternative Architectures

### 10.1 Deeper Network

```python
# Deeper architecture (3 hidden layers)
network = NetworkBuilder() \
    .input(784) \
    .dense(256, activation='relu') \
    .dense(128, activation='relu') \
    .dense(64, activation='relu') \
    .dense(10, activation='softmax') \
    .build()
```

### 10.2 With Dropout

```python
# Architecture with dropout for regularization
network = NeuralNetwork()
network.add(DenseLayer(784, 256))
network.add(ReLULayer())
network.add(DropoutLayer(0.5))
network.add(DenseLayer(256, 128))
network.add(ReLULayer())
network.add(DropoutLayer(0.5))
network.add(DenseLayer(128, 10))
network.add(SoftmaxLayer())
```

### 10.3 Architecture Comparison

| Architecture | Parameters | Expected Accuracy |
|--------------|------------|-------------------|
| Simple (128-64) | ~109K | 97-98% |
| Deeper (256-128-64) | ~235K | 98%+ |
| With Dropout | ~235K | 98%+ (better generalization) |

---

**Document Status**: ✅ Complete  
**Related Documents**: 
- [MATHEMATICAL_FOUNDATION.md](MATHEMATICAL_FOUNDATION.md)
- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
