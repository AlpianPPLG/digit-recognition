"""
Layers - Neural Network Layer Implementations
=============================================

This module contains layer implementations for the neural network:
- Layer: Abstract base class
- DenseLayer: Fully connected layer
- ActivationLayer: Activation function wrapper
- DropoutLayer: Dropout regularization
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    
    All layer implementations must inherit from this class and
    implement the forward and backward methods.
    """
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass computation.
        
        Args:
            x: Input array
            
        Returns:
            Output array
        """
        pass
    
    @abstractmethod
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass computation.
        
        Args:
            gradient: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        pass
    
    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {}
    
    @property
    def gradients(self) -> Dict[str, np.ndarray]:
        """Return parameter gradients."""
        return {}


class DenseLayer(Layer):
    """
    Fully Connected (Dense) Layer.
    
    Computes: output = input @ weights + bias
    
    Attributes:
        input_size: Number of input features
        output_size: Number of output features
        weights: Weight matrix of shape (input_size, output_size)
        bias: Bias vector of shape (1, output_size)
        
    Example:
        >>> layer = DenseLayer(784, 128)
        >>> output = layer.forward(x)  # x shape: (batch, 784)
        >>> # output shape: (batch, 128)
    """
    
    def __init__(self, input_size: int, output_size: int,
                 initializer: str = 'he', use_bias: bool = True):
        """
        Initialize dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            initializer: Weight initialization method ('he', 'xavier', 'random')
            use_bias: Whether to use bias term
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights based on initializer
        if initializer == 'he':
            # He initialization (good for ReLU)
            std = np.sqrt(2.0 / input_size)
        elif initializer == 'xavier':
            # Xavier/Glorot initialization (good for tanh/sigmoid)
            std = np.sqrt(2.0 / (input_size + output_size))
        else:
            # Simple random initialization
            std = 0.01
        
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros((1, output_size)) if use_bias else None
        
        # Cache for backward pass
        self._input_cache: Optional[np.ndarray] = None
        
        # Gradients
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: z = x @ W + b
        
        Args:
            x: Input of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        self._input_cache = x
        output = np.dot(x, self.weights)
        
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients.
        
        Math:
            dL/dW = X.T @ dL/dZ
            dL/db = sum(dL/dZ, axis=0)
            dL/dX = dL/dZ @ W.T
        
        Args:
            gradient: Gradient from next layer, shape (batch_size, output_size)
            
        Returns:
            Gradient to pass to previous layer, shape (batch_size, input_size)
        """
        batch_size = gradient.shape[0]
        
        # Gradient w.r.t. weights: dL/dW = X.T @ gradient
        self.dW = np.dot(self._input_cache.T, gradient) / batch_size
        
        # Gradient w.r.t. bias: dL/db = mean of gradient along batch
        if self.use_bias:
            self.db = np.mean(gradient, axis=0, keepdims=True)
        
        # Gradient w.r.t. input: dL/dX = gradient @ W.T
        grad_input = np.dot(gradient, self.weights.T)
        
        return grad_input
    
    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        params = {'weights': self.weights}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    
    @property
    def gradients(self) -> Dict[str, np.ndarray]:
        """Return parameter gradients."""
        grads = {'weights': self.dW}
        if self.use_bias:
            grads['bias'] = self.db
        return grads
    
    def __repr__(self) -> str:
        return f"DenseLayer({self.input_size}, {self.output_size})"


class ActivationLayer(Layer):
    """
    Activation function layer.
    
    Wraps activation functions as a layer for modular network construction.
    
    Supported activations:
        - 'relu': ReLU activation
        - 'leaky_relu': Leaky ReLU
        - 'sigmoid': Sigmoid activation
        - 'tanh': Hyperbolic tangent
        - 'softmax': Softmax (for output layer)
        
    Example:
        >>> layer = ActivationLayer('relu')
        >>> output = layer.forward(x)
    """
    
    def __init__(self, activation: str = 'relu', **kwargs):
        """
        Initialize activation layer.
        
        Args:
            activation: Name of activation function
            **kwargs: Additional arguments for activation (e.g., alpha for leaky_relu)
        """
        self.activation_name = activation.lower()
        self.kwargs = kwargs
        self._input_cache: Optional[np.ndarray] = None
        self._output_cache: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated output
        """
        self._input_cache = x
        
        if self.activation_name == 'relu':
            output = self._relu(x)
        elif self.activation_name == 'leaky_relu':
            alpha = self.kwargs.get('alpha', 0.01)
            output = self._leaky_relu(x, alpha)
        elif self.activation_name == 'sigmoid':
            output = self._sigmoid(x)
        elif self.activation_name == 'tanh':
            output = self._tanh(x)
        elif self.activation_name == 'softmax':
            output = self._softmax(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
        
        self._output_cache = output
        return output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Compute gradient through activation.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient multiplied by activation derivative
        """
        if self.activation_name == 'relu':
            return gradient * self._relu_derivative(self._input_cache)
        elif self.activation_name == 'leaky_relu':
            alpha = self.kwargs.get('alpha', 0.01)
            return gradient * self._leaky_relu_derivative(self._input_cache, alpha)
        elif self.activation_name == 'sigmoid':
            return gradient * self._sigmoid_derivative(self._output_cache)
        elif self.activation_name == 'tanh':
            return gradient * self._tanh_derivative(self._output_cache)
        elif self.activation_name == 'softmax':
            # For softmax with cross-entropy, gradient is already computed
            return gradient
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    # Activation functions
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative: 1 if x > 0, else 0"""
        return (x > 0).astype(np.float64)
    
    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU: x if x > 0, else alpha * x"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def _leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU derivative: 1 if x > 0, else alpha"""
        return np.where(x > 0, 1.0, alpha)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid: 1 / (1 + exp(-x))"""
        # Clip for numerical stability
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def _sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
        """Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))"""
        return sigmoid_output * (1.0 - sigmoid_output)
    
    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent"""
        return np.tanh(x)
    
    @staticmethod
    def _tanh_derivative(tanh_output: np.ndarray) -> np.ndarray:
        """Tanh derivative: 1 - tanh(x)^2"""
        return 1.0 - tanh_output ** 2
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax: exp(x) / sum(exp(x))
        
        Numerically stable implementation using max subtraction.
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def __repr__(self) -> str:
        return f"ActivationLayer('{self.activation_name}')"


class DropoutLayer(Layer):
    """
    Dropout regularization layer.
    
    Randomly sets a fraction of inputs to zero during training
    to prevent overfitting.
    
    Attributes:
        rate: Fraction of inputs to drop (0 to 1)
        training: Whether in training mode
        
    Example:
        >>> dropout = DropoutLayer(rate=0.5)
        >>> dropout.training = True
        >>> output = dropout.forward(x)  # 50% of values zeroed
    """
    
    def __init__(self, rate: float = 0.5):
        """
        Initialize dropout layer.
        
        Args:
            rate: Dropout rate (probability of dropping a unit)
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        
        self.rate = rate
        self.training = True
        self._mask: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout during training.
        
        During training: randomly zero out units and scale by 1/(1-rate)
        During inference: pass through unchanged (inverted dropout)
        
        Args:
            x: Input array
            
        Returns:
            Output with dropout applied (training) or unchanged (inference)
        """
        if self.training and self.rate > 0:
            # Create dropout mask
            self._mask = (np.random.rand(*x.shape) > self.rate).astype(np.float64)
            # Apply mask and scale (inverted dropout)
            return (x * self._mask) / (1.0 - self.rate)
        else:
            return x
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.
        
        Args:
            gradient: Gradient from next layer
            
        Returns:
            Gradient with same dropout mask applied
        """
        if self.training and self.rate > 0:
            return (gradient * self._mask) / (1.0 - self.rate)
        else:
            return gradient
    
    def __repr__(self) -> str:
        return f"DropoutLayer(rate={self.rate})"
